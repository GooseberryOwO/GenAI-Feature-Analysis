import os
from pathlib import Path

import joblib
import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
INPUT = BASE / "04_large_dataset_prediction_analysis" / "classification model prediction" / "merged_full_expanded_turn_level_predictions.csv"
OUT = BASE / "07_multiclass_creativity_classifier"
PRED_OUT = OUT / "03_full_dataset_predictions"
MODEL_PATH = OUT / "01_model" / "creativity_multiclass_rubric_classifier.joblib"
PRED_OUT.mkdir(parents=True, exist_ok=True)


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    usecols = [
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
    ]
    df = pd.read_csv(INPUT, usecols=usecols)
    text = df["turn_text"].fillna("").astype(str)
    pred = model.predict(text)

    output = df.copy()
    output["multiclass_prediction_label"] = pred
    output["broad_creativity_related_v2"] = output["multiclass_prediction_label"].isin(
        ["strict_brainstorming", "single_creative_generation", "refinement_or_selection"]
    )
    output["strict_brainstorming_v2"] = output["multiclass_prediction_label"].eq("strict_brainstorming")

    out_path = PRED_OUT / "merged_full_expanded_turn_level_multiclass_predictions.csv"
    output.to_csv(out_path, index=False, encoding="utf-8-sig")

    label_counts = output["multiclass_prediction_label"].value_counts().rename_axis("label").reset_index(name="turns")
    label_counts["turn_pct"] = label_counts["turns"] / len(output)
    label_counts.to_csv(PRED_OUT / "full_data_multiclass_turn_label_counts.csv", index=False, encoding="utf-8-sig")

    conv = output.groupby("conversation_hash").agg(
        n_turns=("turn_index", "count"),
        broad_creativity_turns=("broad_creativity_related_v2", "sum"),
        strict_brainstorm_turns=("strict_brainstorming_v2", "sum"),
        first_multiclass_label=("multiclass_prediction_label", "first"),
    ).reset_index()
    conv["has_broad_creativity_v2"] = conv["broad_creativity_turns"] > 0
    conv["has_strict_brainstorm_v2"] = conv["strict_brainstorm_turns"] > 0
    conv["broad_creativity_turn_share"] = conv["broad_creativity_turns"] / conv["n_turns"]
    conv["strict_brainstorm_turn_share"] = conv["strict_brainstorm_turns"] / conv["n_turns"]
    conv.to_csv(PRED_OUT / "full_data_multiclass_conversation_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "turn_rows": int(len(output)),
        "conversation_rows": int(len(conv)),
        "turn_label_counts": label_counts.to_dict(orient="records"),
        "turn_broad_creativity_related_rate": float(output["broad_creativity_related_v2"].mean()),
        "turn_strict_brainstorming_rate": float(output["strict_brainstorming_v2"].mean()),
        "conversation_broad_creativity_related_rate": float(conv["has_broad_creativity_v2"].mean()),
        "conversation_strict_brainstorming_rate": float(conv["has_strict_brainstorm_v2"].mean()),
        "prediction_file": str(out_path),
    }
    pd.Series(summary).to_json(PRED_OUT / "full_data_multiclass_prediction_summary.json", force_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
