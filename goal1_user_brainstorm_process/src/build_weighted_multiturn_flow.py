from __future__ import annotations

import re
import os
from pathlib import Path

import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
ROOT = BASE / "13_multiturn_flow_and_workflow"
DATA = ROOT / "01_analysis_data"
RESULTS = ROOT / "02_results"
for folder in (DATA, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

AUDIT = BASE / "12_wide_vs_local_search_analysis" / "07_strict_validation" / "strict_validation_sample_160_calibrated.csv"
POPULATION = BASE / "12_wide_vs_local_search_analysis" / "07_strict_validation" / "strict_candidate_rule_based_summary.csv"
SOURCE = BASE / "08_exploration_turn_analysis" / "01_analysis_data" / "turn_level_exploration_analysis_dataset.csv"

OPTION_REF_RE = re.compile(
    r"\b(?:option|idea|choice|suggestion|approach|concept|name|title|version)\s*"
    r"(?:#|number|no\.?)?\s*[-:]?\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
    r"|#\s*\d+",
    re.I,
)
LOCAL_RE = re.compile(
    r"\b(?:i like|i prefer|choose|pick|select|go with|let'?s use|expand|develop|elaborate|"
    r"refine|revise|rewrite|improve|build on|based on|combine|merge|polish|adapt|modify|"
    r"variation of|more like|similar to|that one|this one|the above)\b",
    re.I,
)
WIDE_RE = re.compile(
    r"\b(?:more|another|other|different|alternative|alternatives|options|ideas|suggestions|"
    r"approaches|directions|examples|names|titles|try again|come up with|brainstorm|list more|"
    r"keep going|what else)\b",
    re.I,
)

STATE_ORDER = [
    "local_search",
    "wide_search",
    "creative_continuation_other",
    "non_creative_shift",
    "conversation_end",
]


def classify_later_turn(text: str, predicted_label: str, previous_state: str) -> str:
    text = str(text or "")
    label = str(predicted_label or "")
    option_ref = bool(OPTION_REF_RE.search(text))
    local_signal = bool(LOCAL_RE.search(text))
    wide_signal = bool(WIDE_RE.search(text))

    if option_ref or local_signal or label == "refinement_or_selection":
        return "local_search"
    if label == "strict_brainstorming":
        return "wide_search"
    if wide_signal and label in {"strict_brainstorming", "single_creative_generation"}:
        return "wide_search"
    if label == "non_creative_task":
        return "non_creative_shift"
    if label == "single_creative_generation":
        # After an established wide/local episode, a single directed creative
        # request is treated as narrowing/developing one direction unless the
        # user explicitly asks for additional alternatives.
        if previous_state in {"wide_search", "local_search"}:
            return "local_search"
        return "creative_continuation_other"
    return "non_creative_shift"


def main() -> None:
    audit = pd.read_csv(AUDIT)
    audit = audit[audit["human_is_true_multiple_options"].eq("yes")].copy()

    population = pd.read_csv(POPULATION).set_index("rule_based_transition")["n"]
    audit_counts = pd.read_csv(AUDIT)["wide_local_transition_label"].value_counts()
    audit["population_weight"] = audit["wide_local_transition_label"].map(population / audit_counts)

    source = pd.read_csv(
        SOURCE,
        usecols=["conversation_hash", "turn_index", "turn_text", "multiclass_prediction_label"],
    ).sort_values(["conversation_hash", "turn_index"])
    grouped = {key: g.reset_index(drop=True) for key, g in source.groupby("conversation_hash", sort=False)}

    rows = []
    for row in audit.itertuples(index=False):
        conv = grouped.get(row.conversation_hash)
        if conv is None:
            continue
        positions = conv.index[conv["turn_index"].eq(row.turn_index)].tolist()
        if not positions:
            continue
        pos = positions[0]
        states = [row.human_transition_label]
        texts = [str(row.next_user_turn_text_short)]
        labels = [str(row.next_user_label)]

        # t+1 is manually coded. t+2 and t+3 use the same explicit rubric.
        for offset in (2, 3):
            idx = pos + offset
            if idx >= len(conv):
                states.append("conversation_end")
                texts.append("")
                labels.append("")
            else:
                later = conv.iloc[idx]
                text = str(later["turn_text"])
                label = str(later["multiclass_prediction_label"])
                states.append(classify_later_turn(text, label, states[-1]))
                texts.append(text)
                labels.append(label)

        rows.append(
            {
                "audit_id": row.audit_id,
                "conversation_hash": row.conversation_hash,
                "origin_turn_index": row.turn_index,
                "population_weight": row.population_weight,
                "state_t1": states[0],
                "state_t2": states[1],
                "state_t3": states[2],
                "text_t1": texts[0],
                "text_t2": texts[1],
                "text_t3": texts[2],
                "model_label_t1": labels[0],
                "model_label_t2": labels[1],
                "model_label_t3": labels[2],
            }
        )

    flow = pd.DataFrame(rows)
    flow.to_csv(DATA / "human_calibrated_weighted_multiturn_flow.csv", index=False, encoding="utf-8-sig")

    transition_rows = []
    for stage_from, stage_to in (("start", "state_t1"), ("state_t1", "state_t2"), ("state_t2", "state_t3")):
        if stage_from == "start":
            temp = flow.assign(start="genuine_multiple_option_response")
        else:
            temp = flow
        summary = (
            temp.groupby([stage_from, stage_to], dropna=False)["population_weight"]
            .sum()
            .reset_index(name="estimated_n")
        )
        summary.insert(0, "transition", f"{stage_from}_to_{stage_to}")
        transition_rows.append(summary.rename(columns={stage_from: "source", stage_to: "target"}))
    transitions = pd.concat(transition_rows, ignore_index=True)
    transitions.to_csv(RESULTS / "weighted_multiturn_transitions.csv", index=False, encoding="utf-8-sig")

    stage_rows = []
    for stage in ("state_t1", "state_t2", "state_t3"):
        s = flow.groupby(stage)["population_weight"].sum().reset_index(name="estimated_n")
        s.insert(0, "stage", stage)
        s = s.rename(columns={stage: "state"})
        s["share"] = s["estimated_n"] / s["estimated_n"].sum()
        stage_rows.append(s)
    stage_summary = pd.concat(stage_rows, ignore_index=True)
    stage_summary.to_csv(RESULTS / "weighted_multiturn_stage_summary.csv", index=False, encoding="utf-8-sig")

    total_weight = flow["population_weight"].sum()
    print(f"manual_valid_starts={len(flow)}")
    print(f"population_weighted_start_n={total_weight:.3f}")
    print(stage_summary.to_string(index=False))


if __name__ == "__main__":
    main()
