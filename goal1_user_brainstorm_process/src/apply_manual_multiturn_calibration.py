import os
from pathlib import Path

import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
ROOT = Path(
    os.environ.get(
        "GENAI_GOAL1_ROOT",
        str(BASE / "15_goal1_brainstorm_process_model"),
    )
)
DATA = ROOT / "01_analysis_data"
RESULTS = ROOT / "02_results"
for folder in (DATA, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

SOURCE = (
    BASE
    / "13_multiturn_flow_and_workflow"
    / "01_analysis_data"
    / "human_calibrated_weighted_multiturn_flow.csv"
)

# Every t+2 and t+3 state was read manually using the full user-turn text.
# t+1 was already manually coded in the original 160-case audit.
HUMAN_LABELS = {
    2: ("creative_continuation_other", "non_creative_shift"),
    4: ("local_search", "local_search"),
    8: ("wide_search", "conversation_end"),
    14: ("conversation_end", "conversation_end"),
    16: ("wide_search", "non_creative_shift"),
    17: ("local_search", "local_search"),
    18: ("conversation_end", "conversation_end"),
    19: ("wide_search", "conversation_end"),
    20: ("non_creative_shift", "non_creative_shift"),
    21: ("wide_search", "wide_search"),
    23: ("creative_continuation_other", "creative_continuation_other"),
    26: ("local_search", "conversation_end"),
    27: ("creative_continuation_other", "creative_continuation_other"),
    31: ("wide_search", "conversation_end"),
    35: ("local_search", "creative_continuation_other"),
    38: ("non_creative_shift", "conversation_end"),
    43: ("creative_continuation_other", "creative_continuation_other"),
    45: ("local_search", "local_search"),
    48: ("conversation_end", "conversation_end"),
    50: ("local_search", "creative_continuation_other"),
    52: ("wide_search", "local_search"),
    53: ("conversation_end", "conversation_end"),
    54: ("creative_continuation_other", "non_creative_shift"),
    55: ("wide_search", "wide_search"),
    56: ("non_creative_shift", "non_creative_shift"),
    58: ("wide_search", "wide_search"),
    59: ("wide_search", "conversation_end"),
    60: ("wide_search", "conversation_end"),
    61: ("non_creative_shift", "conversation_end"),
    62: ("conversation_end", "conversation_end"),
    63: ("wide_search", "wide_search"),
    64: ("wide_search", "wide_search"),
    68: ("local_search", "conversation_end"),
    69: ("conversation_end", "conversation_end"),
    74: ("wide_search", "local_search"),
    77: ("creative_continuation_other", "conversation_end"),
    78: ("creative_continuation_other", "local_search"),
    79: ("local_search", "conversation_end"),
    80: ("local_search", "local_search"),
    83: ("local_search", "local_search"),
    87: ("creative_continuation_other", "creative_continuation_other"),
    88: ("wide_search", "local_search"),
    89: ("wide_search", "local_search"),
    91: ("local_search", "wide_search"),
    93: ("creative_continuation_other", "conversation_end"),
    94: ("wide_search", "creative_continuation_other"),
    95: ("wide_search", "local_search"),
    100: ("conversation_end", "conversation_end"),
    102: ("creative_continuation_other", "creative_continuation_other"),
    107: ("non_creative_shift", "non_creative_shift"),
    110: ("local_search", "local_search"),
    112: ("conversation_end", "conversation_end"),
    114: ("local_search", "local_search"),
    115: ("wide_search", "conversation_end"),
    118: ("local_search", "conversation_end"),
    120: ("conversation_end", "conversation_end"),
    121: ("creative_continuation_other", "creative_continuation_other"),
    122: ("local_search", "conversation_end"),
    127: ("local_search", "wide_search"),
    129: ("conversation_end", "conversation_end"),
    131: ("local_search", "conversation_end"),
    133: ("conversation_end", "conversation_end"),
    135: ("non_creative_shift", "non_creative_shift"),
    137: ("conversation_end", "conversation_end"),
    138: ("creative_continuation_other", "conversation_end"),
    143: ("conversation_end", "conversation_end"),
    145: ("creative_continuation_other", "creative_continuation_other"),
    151: ("local_search", "wide_search"),
    154: ("creative_continuation_other", "conversation_end"),
    158: ("conversation_end", "conversation_end"),
}


def rationale(rule_label: str, human_label: str) -> str:
    if rule_label == human_label:
        return "Rule label confirmed after semantic review."
    explanations = {
        "wide_search": "User requests additional or differently constrained alternatives.",
        "local_search": "User selects, constrains, critiques, or develops a specific direction.",
        "creative_continuation_other": "User starts another creative subtask without reacting to the prior option set.",
        "non_creative_shift": "User leaves creative search for factual, procedural, or unrelated work.",
        "conversation_end": "No later user turn is present.",
    }
    return explanations[human_label]


def main():
    flow = pd.read_csv(SOURCE)
    assert set(flow["audit_id"]) == set(HUMAN_LABELS)

    audit_rows = []
    for row in flow.itertuples(index=False):
        labels = HUMAN_LABELS[int(row.audit_id)]
        for stage, human_label in zip((2, 3), labels):
            rule_label = getattr(row, f"state_t{stage}")
            audit_rows.append(
                {
                    "audit_id": int(row.audit_id),
                    "conversation_hash": row.conversation_hash,
                    "origin_turn_index": int(row.origin_turn_index),
                    "reviewed_stage": f"t{stage}",
                    "turn_text": getattr(row, f"text_t{stage}"),
                    "model_turn_label": getattr(row, f"model_label_t{stage}"),
                    "rule_state_label": rule_label,
                    "human_state_label": human_label,
                    "agreement": rule_label == human_label,
                    "human_confidence": "high",
                    "human_rationale": rationale(rule_label, human_label),
                    "population_weight": row.population_weight,
                }
            )
        flow.loc[flow["audit_id"].eq(row.audit_id), "human_state_t2"] = labels[0]
        flow.loc[flow["audit_id"].eq(row.audit_id), "human_state_t3"] = labels[1]

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(DATA / "manual_multiturn_calibration_140_turns_private.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(DATA / "human_calibrated_multiturn_flow_final.csv", index=False, encoding="utf-8-sig")

    agreement = (
        audit.groupby("reviewed_stage")["agreement"]
        .agg(reviewed="size", agreements="sum", agreement_rate="mean")
        .reset_index()
    )
    agreement.to_csv(RESULTS / "later_turn_rule_agreement.csv", index=False)

    transition_rows = []
    stage_pairs = [
        ("state_t1", "human_state_t2", "t1_to_t2"),
        ("human_state_t2", "human_state_t3", "t2_to_t3"),
    ]
    for source, target, name in stage_pairs:
        summary = (
            flow.groupby([source, target])["population_weight"]
            .sum()
            .reset_index(name="estimated_n")
            .rename(columns={source: "source_state", target: "target_state"})
        )
        summary.insert(0, "transition", name)
        source_total = summary.groupby("source_state")["estimated_n"].transform("sum")
        summary["conditional_share"] = summary["estimated_n"] / source_total
        transition_rows.append(summary)
    transitions = pd.concat(transition_rows, ignore_index=True)
    transitions.to_csv(RESULTS / "human_calibrated_transition_matrix.csv", index=False)

    stage_rows = []
    for stage in ("state_t1", "human_state_t2", "human_state_t3"):
        summary = flow.groupby(stage)["population_weight"].sum().reset_index(name="estimated_n")
        summary["share"] = summary["estimated_n"] / summary["estimated_n"].sum()
        summary.insert(0, "stage", stage)
        summary = summary.rename(columns={stage: "state"})
        stage_rows.append(summary)
    pd.concat(stage_rows, ignore_index=True).to_csv(
        RESULTS / "human_calibrated_stage_distribution.csv", index=False
    )

    print(agreement.to_string(index=False))
    print(transitions.to_string(index=False))


if __name__ == "__main__":
    main()
