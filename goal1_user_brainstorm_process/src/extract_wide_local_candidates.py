from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
ROOT = BASE / "12_wide_vs_local_search_analysis"
DATA = ROOT / "01_analysis_data"
RESULTS = ROOT / "02_results"
SAMPLES = ROOT / "03_qualitative_samples"
REPORTS = ROOT / "04_reports"

SOURCE = BASE / "08_exploration_turn_analysis" / "01_analysis_data" / "turn_level_exploration_analysis_dataset.csv"

for folder in [DATA, RESULTS, SAMPLES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)


OPTION_REF_RE = re.compile(
    r"\b(option|idea|choice|suggestion|approach|concept|name|title|version)\s*(#|number|no\.?)?\s*[-:]?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
    r"|\b#\s*\d+\b",
    re.I,
)

LOCAL_RE = re.compile(
    r"\b(i like|i prefer|choose|pick|select|use|go with|let'?s use|expand|develop|elaborate|refine|revise|rewrite|improve|make it|turn (it|this|that)|build on|based on|combine|merge|shorten|lengthen|polish|adapt|modify|variation of|more like|similar to|that one|this one|the above|from above)\b",
    re.I,
)

WIDE_RE = re.compile(
    r"\b(more|another|other|different|alternative|alternatives|options|ideas|suggestions|approaches|directions|examples|names|titles|try again|give me \d+ more|come up with|brainstorm|list more|keep going|what else|else can)\b",
    re.I,
)

OPTION_REQUEST_RE = re.compile(
    r"\b(options?|ideas?|alternatives?|suggestions?|approaches?|directions?|examples?|names?|titles?)\b",
    re.I,
)


def compact_text(value: object, limit: int = 900) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def classify_next_turn(next_text: str, next_label: str) -> tuple[str, str, bool, bool, bool]:
    text = next_text or ""
    label = next_label or ""
    option_ref = bool(OPTION_REF_RE.search(text))
    local_signal = bool(LOCAL_RE.search(text))
    wide_signal = bool(WIDE_RE.search(text) and OPTION_REQUEST_RE.search(text))

    if label == "non_creative_task":
        if option_ref or local_signal:
            return "local_search", "next turn references/selects/refines an AI-provided option", option_ref, local_signal, wide_signal
        return "non_creative_shift", "next turn is predicted non-creative", option_ref, local_signal, wide_signal

    if label == "refinement_or_selection":
        return "local_search", "next turn is refinement/selection", option_ref, local_signal, wide_signal

    if option_ref or local_signal:
        return "local_search", "next turn references/selects/refines a specific option", option_ref, local_signal, wide_signal

    if wide_signal:
        return "wide_search", "next turn asks for more/different options or alternatives", option_ref, local_signal, wide_signal

    if label in {"strict_brainstorming", "single_creative_generation"}:
        return "creative_continuation_other", "next turn continues creative exploration without a clear wide/local signal", option_ref, local_signal, wide_signal

    return "other_shift", "no clear wide/local rule matched", option_ref, local_signal, wide_signal


def build_dataset() -> pd.DataFrame:
    usecols = [
        "conversation_hash",
        "turn_index",
        "turn_text",
        "multiclass_prediction_label",
        "assistant_response",
        "assistant_word_count",
        "assistant_numbered_or_bulleted_items",
        "assistant_has_numbered_or_bulleted_list",
        "assistant_has_multiple_option_cue",
        "assistant_option_term_count",
        "assistant_has_followup_question",
        "next_label",
        "has_next_user_turn",
        "Q1.MainTask",
        "Q1.MainDomain",
        "Q2",
    ]
    df = pd.read_csv(SOURCE, usecols=usecols)
    df = df.sort_values(["conversation_hash", "turn_index"]).reset_index(drop=True)

    df["next_user_turn_text"] = df.groupby("conversation_hash")["turn_text"].shift(-1)
    df["next_turn_index"] = df.groupby("conversation_hash")["turn_index"].shift(-1)
    df["next_user_label"] = df.groupby("conversation_hash")["multiclass_prediction_label"].shift(-1)

    candidates = df[
        (df["has_next_user_turn"].astype(bool))
        & (df["assistant_numbered_or_bulleted_items"].fillna(0) >= 2)
    ].copy()

    classified = candidates.apply(
        lambda row: classify_next_turn(
            compact_text(row["next_user_turn_text"], limit=2000),
            str(row["next_user_label"]),
        ),
        axis=1,
        result_type="expand",
    )
    classified.columns = [
        "wide_local_transition_label",
        "wide_local_transition_reason",
        "next_mentions_option_reference",
        "next_has_local_refinement_signal",
        "next_has_wide_search_signal",
    ]
    candidates = pd.concat([candidates, classified], axis=1)
    candidates["current_user_turn_text_short"] = candidates["turn_text"].map(lambda x: compact_text(x, 900))
    candidates["assistant_response_short"] = candidates["assistant_response"].map(lambda x: compact_text(x, 1200))
    candidates["next_user_turn_text_short"] = candidates["next_user_turn_text"].map(lambda x: compact_text(x, 900))

    keep = [
        "conversation_hash",
        "turn_index",
        "next_turn_index",
        "multiclass_prediction_label",
        "next_user_label",
        "assistant_numbered_or_bulleted_items",
        "assistant_word_count",
        "assistant_has_numbered_or_bulleted_list",
        "assistant_has_multiple_option_cue",
        "assistant_option_term_count",
        "assistant_has_followup_question",
        "wide_local_transition_label",
        "wide_local_transition_reason",
        "next_mentions_option_reference",
        "next_has_local_refinement_signal",
        "next_has_wide_search_signal",
        "current_user_turn_text_short",
        "assistant_response_short",
        "next_user_turn_text_short",
        "Q1.MainTask",
        "Q1.MainDomain",
        "Q2",
    ]
    out = candidates[keep].copy()
    out.to_csv(DATA / "wide_vs_local_candidate_turns.csv", index=False, encoding="utf-8-sig")
    return out


def summarize(data: pd.DataFrame) -> dict:
    total = len(data)
    label_counts = data["wide_local_transition_label"].value_counts().rename_axis("transition_label").reset_index(name="n")
    label_counts["share"] = label_counts["n"] / total
    label_counts.to_csv(RESULTS / "wide_vs_local_transition_summary.csv", index=False, encoding="utf-8-sig")

    by_current = (
        data.groupby(["multiclass_prediction_label", "wide_local_transition_label"])
        .size()
        .rename("n")
        .reset_index()
    )
    by_current["share_within_current_label"] = by_current["n"] / by_current.groupby("multiclass_prediction_label")["n"].transform("sum")
    by_current.to_csv(RESULTS / "wide_vs_local_by_current_label.csv", index=False, encoding="utf-8-sig")

    by_option_bin = data.copy()
    by_option_bin["option_count_bin"] = pd.cut(
        by_option_bin["assistant_numbered_or_bulleted_items"],
        bins=[1, 2, 3, 5, 10, np.inf],
        labels=["2", "3", "4-5", "6-10", "11+"],
        right=True,
    )
    option_summary = (
        by_option_bin.groupby(["option_count_bin", "wide_local_transition_label"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    option_summary["share_within_option_bin"] = option_summary["n"] / option_summary.groupby("option_count_bin")["n"].transform("sum")
    option_summary.to_csv(RESULTS / "wide_vs_local_by_option_count_bin.csv", index=False, encoding="utf-8-sig")

    local_rate = float((data["wide_local_transition_label"] == "local_search").mean())
    wide_rate = float((data["wide_local_transition_label"] == "wide_search").mean())
    shift_rate = float((data["wide_local_transition_label"] == "non_creative_shift").mean())
    anchoring_rate = float(data["next_mentions_option_reference"].mean())

    summary = {
        "candidate_turns_with_2plus_ai_option_items_and_next_user_turn": int(total),
        "local_search_rate": local_rate,
        "wide_search_rate": wide_rate,
        "non_creative_shift_rate": shift_rate,
        "explicit_option_reference_rate": anchoring_rate,
        "transition_counts": dict(zip(label_counts["transition_label"], label_counts["n"].astype(int))),
    }
    (RESULTS / "wide_vs_local_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sample_examples(data: pd.DataFrame) -> None:
    samples = []
    rng = 42
    for label, n in {
        "local_search": 35,
        "wide_search": 25,
        "non_creative_shift": 20,
        "creative_continuation_other": 15,
        "other_shift": 5,
    }.items():
        sub = data[data["wide_local_transition_label"].eq(label)]
        if len(sub) == 0:
            continue
        samples.append(sub.sample(min(n, len(sub)), random_state=rng))
        rng += 1
    sample = pd.concat(samples, ignore_index=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    sample.to_csv(SAMPLES / "wide_vs_local_qualitative_sample_100.csv", index=False, encoding="utf-8-sig")


def write_report(summary: dict) -> None:
    text = f"""# Wide vs Local Search First-Pass Analysis

## Goal

Analyze what happens after the AI provides multiple option-like items. The first-pass unit is a user turn whose following AI response contains at least 2 numbered/bulleted items and has a next user turn.

## Rule-Based Labels

- `wide_search`: next user turn asks for more/different/general ideas, options, alternatives, or directions.
- `local_search`: next user turn selects, references, expands, modifies, or refines a specific AI-provided option.
- `non_creative_shift`: next user turn shifts to a predicted non-creative task without clear local option reference.
- `creative_continuation_other`: next user turn remains creative but does not clearly match wide/local rules.
- `other_shift`: no clear rule matched.

## Key Results

- Candidate turns: {summary['candidate_turns_with_2plus_ai_option_items_and_next_user_turn']:,}
- Local search rate: {summary['local_search_rate']:.2%}
- Wide search rate: {summary['wide_search_rate']:.2%}
- Non-creative shift rate: {summary['non_creative_shift_rate']:.2%}
- Explicit option-reference / anchoring indicator rate: {summary['explicit_option_reference_rate']:.2%}

## Outputs

- `01_analysis_data/wide_vs_local_candidate_turns.csv`
- `02_results/wide_vs_local_transition_summary.csv`
- `02_results/wide_vs_local_by_current_label.csv`
- `02_results/wide_vs_local_by_option_count_bin.csv`
- `03_qualitative_samples/wide_vs_local_qualitative_sample_100.csv`

## Caveat

This is a transparent rule-based first pass. It is meant for exploration and qualitative review before training or validating a stronger model.
"""
    (REPORTS / "wide_vs_local_first_pass_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    data = build_dataset()
    summary = summarize(data)
    sample_examples(data)
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
