import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
PRED_FILE = BASE / "07_multiclass_creativity_classifier" / "03_full_dataset_predictions" / "merged_full_expanded_turn_level_multiclass_predictions.csv"
FULL_CONV_FILE = BASE / "04_large_dataset_prediction_analysis" / "classification model prediction" / "merged_full_expanded.csv"

OUT = BASE / "08_exploration_turn_analysis"
DATA_OUT = OUT / "01_analysis_data"
RESULT_OUT = OUT / "02_results"
REPORT_OUT = OUT / "03_reports"
SAMPLE_OUT = OUT / "04_samples"
for folder in [DATA_OUT, RESULT_OUT, REPORT_OUT, SAMPLE_OUT]:
    folder.mkdir(parents=True, exist_ok=True)


EXPLORATION_LABELS = {"strict_brainstorming", "single_creative_generation"}
REFINEMENT_LABELS = {"refinement_or_selection"}
CREATIVE_LABELS = {"strict_brainstorming", "single_creative_generation", "refinement_or_selection"}


def parse_conversation(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []
    return []


def extract_assistant_responses():
    usecols = ["conversation_hash", "conversation"]
    rows = []
    for chunk in pd.read_csv(FULL_CONV_FILE, usecols=usecols, chunksize=5000):
        for conv_hash, conv_text in zip(chunk["conversation_hash"], chunk["conversation"]):
            messages = parse_conversation(conv_text)
            user_idx = 0
            last_user_idx = None
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    user_idx += 1
                    last_user_idx = user_idx
                elif role == "assistant" and last_user_idx is not None:
                    rows.append(
                        {
                            "conversation_hash": conv_hash,
                            "turn_index": last_user_idx,
                            "assistant_response": str(content or ""),
                        }
                    )
                    last_user_idx = None
    resp = pd.DataFrame(rows)
    if resp.empty:
        return resp
    # If a conversation format has multiple assistant messages after a user turn,
    # keep the first as the immediate response.
    resp = resp.drop_duplicates(["conversation_hash", "turn_index"], keep="first")
    return resp


def count_numbered_items(text):
    if not text:
        return 0
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    numbered = sum(bool(re.match(r"^(\d+[\).]|[-*]\s+)", ln)) for ln in lines)
    return numbered


def response_features(text):
    text = str(text or "")
    low = text.lower()
    words = re.findall(r"\b\w+\b", low)
    numbered_items = count_numbered_items(text)
    option_terms = [
        "option",
        "idea",
        "alternative",
        "suggestion",
        "variation",
        "approach",
        "ways",
        "possibility",
        "choose",
        "select",
    ]
    followup_patterns = [
        r"\?",
        r"\bwould you like\b",
        r"\bdo you want\b",
        r"\bwhich\b",
        r"\bwhat kind\b",
        r"\bif you want\b",
        r"\blet me know\b",
        r"你想|要不要|需要我|如果你想|哪一个|哪个",
    ]
    multiple_cues = [
        r"\bhere are\b",
        r"\bseveral\b",
        r"\ba few\b",
        r"\bmultiple\b",
        r"\boptions\b",
        r"\bideas\b",
        r"\balternatives\b",
        r"\bvariations\b",
        r"以下是|几个|一些|多个|选项|想法|方案|建议",
    ]
    return {
        "assistant_word_count": len(words),
        "assistant_char_count": len(text),
        "assistant_numbered_or_bulleted_items": numbered_items,
        "assistant_has_numbered_or_bulleted_list": numbered_items >= 2,
        "assistant_has_multiple_option_cue": any(re.search(p, low, flags=re.I) for p in multiple_cues),
        "assistant_option_term_count": sum(low.count(term) for term in option_terms),
        "assistant_has_followup_question": any(re.search(p, text, flags=re.I) for p in followup_patterns),
        "assistant_question_mark_count": text.count("?") + text.count("？"),
    }


def build_turn_analysis_dataset():
    pred_cols = [
        "source_row_id",
        "conversation_hash",
        "declared_turns",
        "parsed_user_turns",
        "turn_index",
        "turn_text",
        "prob_brainstorm",
        "prediction_label",
        "category",
        "Q1.MainTask",
        "Q1.MainDomain",
        "Q2",
        "multiclass_prediction_label",
        "broad_creativity_related_v2",
        "strict_brainstorming_v2",
    ]
    pred = pd.read_csv(PRED_FILE, usecols=pred_cols)
    pred = pred.sort_values(["conversation_hash", "turn_index"]).reset_index(drop=True)

    resp_path = DATA_OUT / "assistant_responses_by_user_turn.csv"
    if resp_path.exists():
        resp = pd.read_csv(resp_path)
    else:
        resp = extract_assistant_responses()
        resp.to_csv(resp_path, index=False, encoding="utf-8-sig")

    df = pred.merge(resp, on=["conversation_hash", "turn_index"], how="left")
    df["assistant_response"] = df["assistant_response"].fillna("")
    feature_df = pd.DataFrame([response_features(x) for x in df["assistant_response"]])
    df = pd.concat([df, feature_df], axis=1)

    df["is_exploration_turn"] = df["multiclass_prediction_label"].isin(EXPLORATION_LABELS)
    df["is_strict_brainstorming_turn"] = df["multiclass_prediction_label"].eq("strict_brainstorming")
    df["is_single_generation_turn"] = df["multiclass_prediction_label"].eq("single_creative_generation")
    df["is_refinement_turn"] = df["multiclass_prediction_label"].isin(REFINEMENT_LABELS)
    df["is_creativity_related_turn"] = df["multiclass_prediction_label"].isin(CREATIVE_LABELS)
    df["next_label"] = df.groupby("conversation_hash")["multiclass_prediction_label"].shift(-1)
    df["next_is_exploration"] = df.groupby("conversation_hash")["is_exploration_turn"].shift(-1).fillna(False).astype(bool)
    df["next_is_refinement"] = df.groupby("conversation_hash")["is_refinement_turn"].shift(-1).fillna(False).astype(bool)
    df["next_is_creativity_related"] = df.groupby("conversation_hash")["is_creativity_related_turn"].shift(-1).fillna(False).astype(bool)
    df["has_next_user_turn"] = df.groupby("conversation_hash")["turn_index"].shift(-1).notna()
    df.to_csv(DATA_OUT / "turn_level_exploration_analysis_dataset.csv", index=False, encoding="utf-8-sig")
    return df


def assign_exploration_runs(df):
    df = df.sort_values(["conversation_hash", "turn_index"]).copy()
    run_rows = []
    turn_run_ids = []
    for conv_hash, g in df.groupby("conversation_hash", sort=False):
        current_run_id = None
        current_run_turns = []
        run_counter = 0
        for idx, row in g.iterrows():
            if row["is_exploration_turn"]:
                prev_idx = current_run_turns[-1] if current_run_turns else None
                if current_run_id is None:
                    run_counter += 1
                    current_run_id = f"{conv_hash}_explore_{run_counter}"
                    current_run_turns = []
                current_run_turns.append(idx)
                turn_run_ids.append((idx, current_run_id))
            else:
                if current_run_id is not None:
                    run_rows.append((current_run_id, current_run_turns))
                    current_run_id = None
                    current_run_turns = []
                turn_run_ids.append((idx, ""))
        if current_run_id is not None:
            run_rows.append((current_run_id, current_run_turns))

    run_id_map = dict(turn_run_ids)
    df["exploration_run_id"] = df.index.map(run_id_map).fillna("")

    summaries = []
    for run_id, indices in run_rows:
        rg = df.loc[indices].sort_values("turn_index")
        conv_hash = rg["conversation_hash"].iloc[0]
        full_conv = df[df["conversation_hash"].eq(conv_hash)].sort_values("turn_index")
        start_turn = int(rg["turn_index"].min())
        end_turn = int(rg["turn_index"].max())
        after = full_conv[full_conv["turn_index"].gt(end_turn)]
        next_label = after["multiclass_prediction_label"].iloc[0] if len(after) else ""
        summaries.append(
            {
                "conversation_hash": conv_hash,
                "exploration_run_id": run_id,
                "run_start_turn": start_turn,
                "run_end_turn": end_turn,
                "exploration_run_length": int(len(rg)),
                "strict_brainstorm_turns_in_run": int(rg["is_strict_brainstorming_turn"].sum()),
                "single_generation_turns_in_run": int(rg["is_single_generation_turn"].sum()),
                "mean_assistant_word_count_during_run": float(rg["assistant_word_count"].mean()),
                "mean_assistant_list_items_during_run": float(rg["assistant_numbered_or_bulleted_items"].mean()),
                "any_assistant_list_during_run": bool(rg["assistant_has_numbered_or_bulleted_list"].any()),
                "any_assistant_multiple_option_cue_during_run": bool(rg["assistant_has_multiple_option_cue"].any()),
                "any_assistant_followup_question_during_run": bool(rg["assistant_has_followup_question"].any()),
                "next_label_after_run": next_label,
                "ends_by_refinement": next_label == "refinement_or_selection",
                "ends_by_non_creative": next_label == "non_creative_task",
                "ends_by_conversation_end": next_label == "",
            }
        )
    runs = pd.DataFrame(summaries)
    df.to_csv(DATA_OUT / "turn_level_exploration_analysis_dataset.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(DATA_OUT / "exploration_runs.csv", index=False, encoding="utf-8-sig")
    return df, runs


def summarize_stats(df, runs):
    overall = {
        "n_conversations": int(df["conversation_hash"].nunique()),
        "n_user_turns": int(len(df)),
        "n_exploration_turns": int(df["is_exploration_turn"].sum()),
        "exploration_turn_rate": float(df["is_exploration_turn"].mean()),
        "n_refinement_turns": int(df["is_refinement_turn"].sum()),
        "refinement_turn_rate": float(df["is_refinement_turn"].mean()),
        "n_exploration_runs": int(len(runs)),
        "mean_exploration_run_length": float(runs["exploration_run_length"].mean()),
        "median_exploration_run_length": float(runs["exploration_run_length"].median()),
        "pct_runs_length_1": float((runs["exploration_run_length"] == 1).mean()),
        "pct_runs_length_ge_2": float((runs["exploration_run_length"] >= 2).mean()),
        "pct_runs_length_ge_3": float((runs["exploration_run_length"] >= 3).mean()),
        "pct_runs_end_by_refinement": float(runs["ends_by_refinement"].mean()),
        "pct_runs_end_by_non_creative": float(runs["ends_by_non_creative"].mean()),
        "pct_runs_end_by_conversation_end": float(runs["ends_by_conversation_end"].mean()),
    }
    (RESULT_OUT / "exploration_overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    bins = pd.cut(
        runs["exploration_run_length"],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=["1", "2", "3", "4-5", "6-10", "11+"],
    )
    run_bins = runs.groupby(bins, observed=False).size().rename("n_runs").reset_index().rename(columns={"exploration_run_length": "run_length_bin"})
    run_bins["pct_runs"] = run_bins["n_runs"] / len(runs)
    run_bins.to_csv(RESULT_OUT / "exploration_run_length_distribution.csv", index=False, encoding="utf-8-sig")

    end_counts = runs["next_label_after_run"].replace("", "conversation_end").value_counts().rename_axis("next_label_after_run").reset_index(name="n_runs")
    end_counts["pct_runs"] = end_counts["n_runs"] / len(runs)
    end_counts.to_csv(RESULT_OUT / "exploration_run_end_transition_distribution.csv", index=False, encoding="utf-8-sig")

    # AI response features: for exploration turns that have a next user turn, compare stay vs leave exploration.
    exp = df[df["is_exploration_turn"] & df["has_next_user_turn"]].copy()
    feature_cols = [
        "assistant_word_count",
        "assistant_char_count",
        "assistant_numbered_or_bulleted_items",
        "assistant_has_numbered_or_bulleted_list",
        "assistant_has_multiple_option_cue",
        "assistant_option_term_count",
        "assistant_has_followup_question",
        "assistant_question_mark_count",
    ]
    rows = []
    for col in feature_cols:
        if exp[col].dtype == bool:
            by = exp.groupby("next_is_exploration")[col].mean()
        else:
            by = exp.groupby("next_is_exploration")[col].mean()
        rows.append(
            {
                "feature": col,
                "mean_if_next_leaves_exploration": float(by.get(False, np.nan)),
                "mean_if_next_stays_exploration": float(by.get(True, np.nan)),
                "difference_stay_minus_leave": float(by.get(True, np.nan) - by.get(False, np.nan)),
            }
        )
    feature_compare = pd.DataFrame(rows)
    feature_compare.to_csv(RESULT_OUT / "ai_response_features_stay_vs_leave_exploration.csv", index=False, encoding="utf-8-sig")

    grouped = exp.groupby(["assistant_has_numbered_or_bulleted_list", "assistant_has_multiple_option_cue", "assistant_has_followup_question"]).agg(
        n_turns=("turn_index", "count"),
        next_exploration_rate=("next_is_exploration", "mean"),
        next_refinement_rate=("next_is_refinement", "mean"),
    ).reset_index()
    grouped.to_csv(RESULT_OUT / "ai_response_feature_combo_next_turn_rates.csv", index=False, encoding="utf-8-sig")

    conv_summary = df.groupby("conversation_hash").agg(
        n_turns=("turn_index", "count"),
        exploration_turns=("is_exploration_turn", "sum"),
        strict_brainstorming_turns=("is_strict_brainstorming_turn", "sum"),
        single_generation_turns=("is_single_generation_turn", "sum"),
        refinement_turns=("is_refinement_turn", "sum"),
        first_label=("multiclass_prediction_label", "first"),
    ).reset_index()
    conv_summary["has_exploration"] = conv_summary["exploration_turns"] > 0
    conv_summary["has_refinement"] = conv_summary["refinement_turns"] > 0
    conv_summary["exploration_turn_share"] = conv_summary["exploration_turns"] / conv_summary["n_turns"]
    conv_summary.to_csv(RESULT_OUT / "conversation_exploration_summary.csv", index=False, encoding="utf-8-sig")
    return overall, feature_compare


def write_samples(df, runs):
    long_runs = runs.sort_values(["exploration_run_length"], ascending=False).head(100)
    long_runs.to_csv(SAMPLE_OUT / "sample_long_exploration_runs.csv", index=False, encoding="utf-8-sig")

    exp = df[df["is_exploration_turn"] & df["has_next_user_turn"]].copy()
    stay = exp[exp["next_is_exploration"]].sample(n=min(100, int(exp["next_is_exploration"].sum())), random_state=42)
    leave = exp[~exp["next_is_exploration"]].sample(n=min(100, int((~exp["next_is_exploration"]).sum())), random_state=42)
    cols = [
        "conversation_hash",
        "turn_index",
        "turn_text",
        "assistant_response",
        "multiclass_prediction_label",
        "next_label",
        "assistant_word_count",
        "assistant_numbered_or_bulleted_items",
        "assistant_has_multiple_option_cue",
        "assistant_has_followup_question",
    ]
    pd.concat([stay.assign(sample_type="next_stays_exploration"), leave.assign(sample_type="next_leaves_exploration")], ignore_index=True)[["sample_type"] + cols].to_csv(
        SAMPLE_OUT / "sample_ai_response_stay_vs_leave_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_report(overall, feature_compare):
    feature_rows = "\n".join(
        f"- {r.feature}: stay mean {r.mean_if_next_stays_exploration:.3f}, leave mean {r.mean_if_next_leaves_exploration:.3f}, diff {r.difference_stay_minus_leave:.3f}"
        for r in feature_compare.itertuples()
    )
    report = f"""# Exploration Turn Analysis

This analysis uses the current predicted multi-class labels, because validated teacher labels are not available yet.

## Definitions

- Exploration turns: `strict_brainstorming` + `single_creative_generation`
- Refinement turns: `refinement_or_selection`
- Non-exploration: `non_creative_task`

The goal is to estimate how long users stay in exploration before moving to refinement/non-creative work/end of conversation, and whether assistant response features are associated with staying in exploration.

## Overall Results

- Conversations: {overall['n_conversations']:,}
- User turns: {overall['n_user_turns']:,}
- Exploration turns: {overall['n_exploration_turns']:,} ({overall['exploration_turn_rate']*100:.2f}%)
- Refinement turns: {overall['n_refinement_turns']:,} ({overall['refinement_turn_rate']*100:.2f}%)
- Exploration runs: {overall['n_exploration_runs']:,}
- Mean exploration run length: {overall['mean_exploration_run_length']:.2f} turns
- Median exploration run length: {overall['median_exploration_run_length']:.0f} turn
- Runs with length 1: {overall['pct_runs_length_1']*100:.2f}%
- Runs with length >= 2: {overall['pct_runs_length_ge_2']*100:.2f}%
- Runs with length >= 3: {overall['pct_runs_length_ge_3']*100:.2f}%

## How Exploration Runs End

- End by refinement/selection: {overall['pct_runs_end_by_refinement']*100:.2f}%
- End by non-creative task: {overall['pct_runs_end_by_non_creative']*100:.2f}%
- End by conversation end: {overall['pct_runs_end_by_conversation_end']*100:.2f}%

## Assistant Response Features And Staying In Exploration

For exploration turns that have a next user turn, I compared assistant response features when the next user turn stays in exploration vs leaves exploration.

{feature_rows}

## Interpretation

This is descriptive rather than causal. It can show which assistant-response patterns are associated with longer exploration, but not prove that the response caused the user to stay.

The next research step is to turn this into a more formal model, for example predicting whether the next user turn stays in exploration from assistant response features, conversation context, and current turn type.

## Output Files

- `01_analysis_data/turn_level_exploration_analysis_dataset.csv`
- `01_analysis_data/exploration_runs.csv`
- `02_results/exploration_overall_summary.json`
- `02_results/exploration_run_length_distribution.csv`
- `02_results/exploration_run_end_transition_distribution.csv`
- `02_results/ai_response_features_stay_vs_leave_exploration.csv`
- `02_results/ai_response_feature_combo_next_turn_rates.csv`
- `02_results/conversation_exploration_summary.csv`
- `04_samples/sample_long_exploration_runs.csv`
- `04_samples/sample_ai_response_stay_vs_leave_cases.csv`
"""
    (REPORT_OUT / "exploration_turn_analysis_report.md").write_text(report, encoding="utf-8")

    slack = f"""老师好，我开始做您之前提到的 exploration turn analysis 了。因为目前还没有新的人工复核 label，我先用现有 multi-class classifier 的预测 label 来做。

我把 `strict_brainstorming` 和 `single_creative_generation` 定义为 exploration turn，把 `refinement_or_selection` 定义为进入修改/选择阶段。初步结果是：exploration turns 占全部 user turns 的 {overall['exploration_turn_rate']*100:.2f}%，一共有 {overall['n_exploration_runs']:,} 段 exploration run。平均每段 exploration 持续 {overall['mean_exploration_run_length']:.2f} 个 user turns，中位数是 {overall['median_exploration_run_length']:.0f} 个 turn；其中 {overall['pct_runs_length_1']*100:.2f}% 的 exploration run 只有 1 个 turn，{overall['pct_runs_length_ge_2']*100:.2f}% 会持续至少 2 个 turns。

我也初步看了 exploration run 后面怎么结束：{overall['pct_runs_end_by_refinement']*100:.2f}% 会进入 refinement/selection，{overall['pct_runs_end_by_non_creative']*100:.2f}% 转到 non-creative task，{overall['pct_runs_end_by_conversation_end']*100:.2f}% 直接在 conversation 结束。接下来可以进一步看 AI response 的特征，比如是否给多个 options、是否 numbered list、是否问 follow-up question，会不会让用户下一轮继续 stay in exploration。"""
    (REPORT_OUT / "slack_update_exploration_analysis.md").write_text(slack, encoding="utf-8")


def main():
    df = build_turn_analysis_dataset()
    df, runs = assign_exploration_runs(df)
    overall, feature_compare = summarize_stats(df, runs)
    write_samples(df, runs)
    write_report(overall, feature_compare)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
