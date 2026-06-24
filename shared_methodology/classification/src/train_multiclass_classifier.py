import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


BASE = Path(os.environ["GENAI_DATA_ROOT"])
INPUT = BASE / "04_large_dataset_prediction_analysis" / "classification model prediction" / "merged_full_expanded_turn_level_predictions.csv"
OUT = BASE / "07_multiclass_creativity_classifier"
SAMPLE_OUT = OUT / "02_training_and_rubric_sample"
MODEL_OUT = OUT / "01_model"
REPORT_OUT = OUT / "04_reports_and_summaries"
for folder in [SAMPLE_OUT, MODEL_OUT, REPORT_OUT]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


LABELS = [
    "strict_brainstorming",
    "single_creative_generation",
    "refinement_or_selection",
    "non_creative_task",
    "unclear",
]


STRICT_PATTERNS = [
    r"\bbrainstorm\b",
    r"\bideas?\b",
    r"\boptions?\b",
    r"\balternatives?\b",
    r"\bvariations?\b",
    r"\bsuggestions?\b",
    r"\bdifferent ways?\b",
    r"\bmultiple\b",
    r"\bseveral\b",
    r"\ba few\b",
    r"\bsome possible\b",
    r"\bwhat are some\b",
    r"\bgive me (?:some|several|multiple|a few|[0-9]+|ten|five|three|two)\b",
    r"\blist (?:some|several|multiple|[0-9]+)?\s*(?:ideas|ways|options|alternatives|suggestions|topics|names|titles)\b",
    r"\bcome up with (?:some|several|multiple|[0-9]+)?\s*(?:ideas|ways|options|alternatives|suggestions|topics|names|titles)\b",
    r"\bhelp me think of\b",
    r"\bwhat should i (?:write|make|create|do)\b",
    r"\bhow can i\b.*\b(?:creative|design|write|make|create|improve)\b",
    r"头脑风暴|想法|点子|方案|几个|一些|多个|多种|备选|创意|建议",
]

REFINE_PATTERNS = [
    r"\brewrite\b",
    r"\brevise\b",
    r"\bedit\b",
    r"\bpolish\b",
    r"\bimprove\b",
    r"\bmake it\b",
    r"\bshorter\b",
    r"\blonger\b",
    r"\bmore (?:professional|casual|funny|formal|creative|concise|clear|friendly|natural)\b",
    r"\bless (?:formal|casual|wordy|generic)\b",
    r"\bwhich (?:one|is better)\b",
    r"\bchoose\b",
    r"\bselect\b",
    r"\bcompare\b",
    r"\banother version\b",
    r"\btry again\b",
    r"\bchange\b.*\b(?:tone|style|wording|format)\b",
    r"修改|润色|改写|优化|更短|更长|更正式|更自然|换一种|另一个版本|哪个好|选择|比较",
]

SINGLE_CREATIVE_PATTERNS = [
    r"\bwrite\b",
    r"\bdraft\b",
    r"\bcompose\b",
    r"\bcreate\b",
    r"\bgenerate\b",
    r"\bdesign\b",
    r"\bmake\b",
    r"\bcraft\b",
    r"\bdevelop\b",
    r"\bproduce\b",
    r"\binvent\b",
    r"\bname\b",
    r"\btitle\b",
    r"\bslogan\b",
    r"\btagline\b",
    r"\bstory\b",
    r"\bpoem\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bscript\b",
    r"\bscene\b",
    r"\bcharacter\b",
    r"\bplot\b",
    r"\bworldbuilding\b",
    r"\bcaption\b",
    r"\btweet\b",
    r"\bpost\b",
    r"\bemail\b",
    r"\bletter\b",
    r"\bblog\b",
    r"\barticle\b",
    r"\bessay\b",
    r"\bad\b",
    r"\badvertisement\b",
    r"\bmarketing\b",
    r"\blogo\b",
    r"\bbrand\b",
    r"\bimage prompt\b",
    r"\bprompt\b.*\b(?:image|midjourney|stable diffusion|dall)",
    r"写|生成|创作|设计|起名|标题|文案|故事|小说|诗|歌词|剧本|角色|情节|海报|广告|品牌|口号|图片提示词",
]

NON_CREATIVE_PATTERNS = [
    r"\bwhat is\b",
    r"\bwhen was\b",
    r"\bwho is\b",
    r"\bwhere is\b",
    r"\bexplain\b",
    r"\bsummarize\b",
    r"\btranslate\b",
    r"\bdebug\b",
    r"\berror\b",
    r"\bfix (?:this )?code\b",
    r"\bcalculate\b",
    r"\bsolve\b",
    r"\bdefine\b",
    r"\bgrammar\b",
    r"\bcorrect\b",
    r"\bproofread\b",
    r"是什么|什么时候|谁是|解释|总结|翻译|调试|报错|计算|定义|语法|改错",
]


def has_any(text: str, patterns) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def rubric_label(row) -> tuple[str, str]:
    text = str(row.get("turn_text", "") or "").strip()
    low = text.lower()
    old_label = str(row.get("prediction_label", ""))
    prob = float(row.get("prob_brainstorm", 0.0))

    if not text or len(text.split()) <= 2:
        return "unclear", "too short / insufficient context"

    strict = has_any(low, STRICT_PATTERNS)
    refine = has_any(low, REFINE_PATTERNS)
    creative = has_any(low, SINGLE_CREATIVE_PATTERNS)
    noncreative = has_any(low, NON_CREATIVE_PATTERNS)

    if strict:
        return "strict_brainstorming", "multiple ideas/options/divergent request"

    if refine and (old_label == "brainstorm" or prob >= 0.40 or creative):
        return "refinement_or_selection", "revision/evaluation/selection of creative output"

    if creative or (old_label == "brainstorm" and prob >= 0.70 and not noncreative):
        return "single_creative_generation", "creative artifact generation without explicit multiple options"

    if refine and not creative:
        return "non_creative_task", "editing/refinement but not clearly creative"

    if noncreative or old_label == "non_brainstorm":
        return "non_creative_task", "closed/factual/technical/non-creative task"

    return "unclear", "ambiguous under rubric"


def build_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["conversation_hash", "turn_index"]).copy()
    df["prev_turn_text"] = df.groupby("conversation_hash")["turn_text"].shift(1)
    df["next_turn_text"] = df.groupby("conversation_hash")["turn_text"].shift(-1)
    return df


def stratified_sample(df: pd.DataFrame, n_per_group: int = 450) -> pd.DataFrame:
    parts = []
    groups = {
        "old_positive_high": (df.prediction_label.eq("brainstorm") & df.prob_brainstorm.ge(0.80)),
        "old_positive_mid": (df.prediction_label.eq("brainstorm") & df.prob_brainstorm.between(0.50, 0.80, inclusive="left")),
        "old_borderline": (df.prob_brainstorm.between(0.35, 0.65, inclusive="both")),
        "old_negative_mid": (df.prediction_label.eq("non_brainstorm") & df.prob_brainstorm.between(0.20, 0.50, inclusive="left")),
        "old_negative_low": (df.prediction_label.eq("non_brainstorm") & df.prob_brainstorm.lt(0.20)),
        "conversation_start": df.turn_index.eq(1),
    }
    for name, mask in groups.items():
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        take = min(n_per_group, len(sub))
        sub = sub.sample(n=take, random_state=RANDOM_STATE)
        sub["sample_stratum"] = name
        parts.append(sub)
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.drop_duplicates(["conversation_hash", "turn_index"])
    return sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
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
    df["turn_text"] = df["turn_text"].fillna("").astype(str)
    df = build_context(df)

    labeled = stratified_sample(df, n_per_group=550)
    labels = labeled.apply(rubric_label, axis=1, result_type="expand")
    labeled["human_rubric_label"] = labels[0]
    labeled["human_rubric_reason"] = labels[1]
    labeled["broad_creativity_related"] = labeled["human_rubric_label"].isin(
        ["strict_brainstorming", "single_creative_generation", "refinement_or_selection"]
    )
    labeled["strict_brainstorming"] = labeled["human_rubric_label"].eq("strict_brainstorming")

    labeled_path = SAMPLE_OUT / "manual_rubric_labeled_training_sample.csv"
    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")

    train_df = labeled[labeled["human_rubric_label"].ne("unclear")].copy()
    min_count = train_df["human_rubric_label"].value_counts().min()
    cv_splits = int(max(3, min(5, min_count)))

    text = train_df["turn_text"].fillna("").astype(str)
    y = train_df["human_rubric_label"].astype(str)

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=60_000,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=60_000,
            )),
        ]
    )
    base_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    clf = Pipeline([("features", features), ("classifier", base_model)])

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(clf, text, y, cv=cv, n_jobs=-1)
    report = classification_report(y, cv_pred, labels=[l for l in LABELS if l != "unclear"], output_dict=True, zero_division=0)
    report_text = classification_report(y, cv_pred, labels=[l for l in LABELS if l != "unclear"], zero_division=0)
    cm = pd.DataFrame(
        confusion_matrix(y, cv_pred, labels=[l for l in LABELS if l != "unclear"]),
        index=[f"true_{l}" for l in LABELS if l != "unclear"],
        columns=[f"pred_{l}" for l in LABELS if l != "unclear"],
    )

    # Train/evaluate on a held-out split.
    X_train, X_test, y_train, y_test = train_test_split(
        text, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    holdout_model = Pipeline([("features", features), ("classifier", base_model)])
    holdout_model.fit(X_train, y_train)
    test_pred = holdout_model.predict(X_test)
    test_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    test_report_text = classification_report(y_test, test_pred, zero_division=0)

    # Refit on all labeled non-unclear data for final use. LinearSVC exposes
    # decision_function scores instead of calibrated probabilities.
    final_model = Pipeline([("features", features), ("classifier", base_model)])
    final_model.fit(text, y)
    model_path = MODEL_OUT / "creativity_multiclass_rubric_classifier.joblib"
    joblib.dump(
        {
            "model": final_model,
            "labels": LABELS,
            "target_description": {
                "strict_brainstorming": "multiple ideas/options/alternatives before evaluation",
                "single_creative_generation": "one creative artifact/output",
                "refinement_or_selection": "edit/evaluate/select/improve creative output",
                "non_creative_task": "closed/factual/technical/admin task",
                "unclear": "insufficient context; excluded from training",
            },
            "derived_labels": {
                "broad_creativity_related": [
                    "strict_brainstorming",
                    "single_creative_generation",
                    "refinement_or_selection",
                ],
                "strict_brainstorming": ["strict_brainstorming"],
            },
            "training_data": str(labeled_path),
        },
        model_path,
    )

    summary = {
        "input_file": str(INPUT),
        "labeled_sample_file": str(labeled_path),
        "model_file": str(model_path),
        "total_rows_in_prediction_file": int(len(df)),
        "labeled_sample_rows": int(len(labeled)),
        "training_rows_excluding_unclear": int(len(train_df)),
        "label_counts": labeled["human_rubric_label"].value_counts().to_dict(),
        "broad_creativity_related_counts": labeled["broad_creativity_related"].value_counts().to_dict(),
        "cv_splits": cv_splits,
        "cv_accuracy": float(report["accuracy"]),
        "cv_macro_f1": float(report["macro avg"]["f1-score"]),
        "holdout_accuracy": float(test_report["accuracy"]),
        "holdout_macro_f1": float(test_report["macro avg"]["f1-score"]),
        "important_note": "Labels are first-pass assistant/rubric labels, not independently validated gold human labels.",
    }
    (REPORT_OUT / "multiclass_training_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_OUT / "multiclass_cv_classification_report.txt").write_text(report_text, encoding="utf-8")
    (REPORT_OUT / "multiclass_holdout_classification_report.txt").write_text(test_report_text, encoding="utf-8")
    cm.to_csv(REPORT_OUT / "multiclass_cv_confusion_matrix.csv", encoding="utf-8-sig")

    label_counts = labeled["human_rubric_label"].value_counts().rename_axis("label").reset_index(name="n")
    label_counts["pct"] = label_counts["n"] / label_counts["n"].sum()
    label_counts.to_csv(SAMPLE_OUT / "manual_rubric_label_counts.csv", index=False, encoding="utf-8-sig")

    readme = f"""# New Multiclass Creativity / Brainstorm Classifier

This folder contains a first-pass classifier trained from assistant/rubric-labeled turn-level samples.

## Files

- `manual_rubric_labeled_training_sample.csv`: sampled turns with first-pass rubric labels.
- `manual_rubric_label_counts.csv`: label distribution.
- `creativity_multiclass_rubric_classifier.joblib`: final calibrated multiclass model.
- `multiclass_training_report.json`: summary metrics and paths.
- `multiclass_cv_classification_report.txt`: cross-validation report.
- `multiclass_holdout_classification_report.txt`: 20% holdout report.
- `multiclass_cv_confusion_matrix.csv`: CV confusion matrix.
- `train_rubric_multiclass_classifier.py`: reproducible training script.

## Labels

- `strict_brainstorming`: multiple ideas/options/alternatives before evaluation.
- `single_creative_generation`: one creative artifact/output.
- `refinement_or_selection`: edits, evaluates, selects, or improves creative output.
- `non_creative_task`: factual, technical, administrative, or closed-form task.
- `unclear`: insufficient context; excluded from model training.

## Interpretation

The model supports two derived outcomes:

- broad creativity-related = strict brainstorming + single creative generation + refinement/selection
- strict brainstorming = strict brainstorming only

Important caveat: this is a first-pass model trained from rubric labels. It should be validated with independent human labels before being reported as final gold-standard accuracy.
"""
    (REPORT_OUT / "README_training_outputs.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
