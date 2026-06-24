import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


BASE = Path(os.environ["GENAI_DATA_ROOT"])
INPUT = BASE / "08_exploration_turn_analysis" / "01_analysis_data" / "turn_level_exploration_analysis_dataset.csv"
OUT = BASE / "09_exploration_continuation_prediction"
DATA_OUT = OUT / "01_model_data"
MODEL_OUT = OUT / "02_models"
RESULT_OUT = OUT / "03_results"
REPORT_OUT = OUT / "04_reports"
SIM_OUT = OUT / "05_simulation_outputs"
for folder in [DATA_OUT, MODEL_OUT, RESULT_OUT, REPORT_OUT, SIM_OUT]:
    folder.mkdir(parents=True, exist_ok=True)


USECOLS = [
    "conversation_hash",
    "declared_turns",
    "parsed_user_turns",
    "turn_index",
    "turn_text",
    "prob_brainstorm",
    "multiclass_prediction_label",
    "Q1.MainDomain",
    "assistant_word_count",
    "assistant_char_count",
    "assistant_numbered_or_bulleted_items",
    "assistant_has_numbered_or_bulleted_list",
    "assistant_has_multiple_option_cue",
    "assistant_option_term_count",
    "assistant_has_followup_question",
    "assistant_question_mark_count",
    "is_exploration_turn",
    "is_strict_brainstorming_turn",
    "is_single_generation_turn",
    "is_refinement_turn",
    "next_label",
    "next_is_exploration",
    "next_is_refinement",
    "has_next_user_turn",
]


NUMERIC_FEATURES = [
    "log_assistant_word_count",
    "log_assistant_char_count",
    "log_assistant_numbered_or_bulleted_items",
    "assistant_has_numbered_or_bulleted_list",
    "assistant_has_multiple_option_cue",
    "log_assistant_option_term_count",
    "assistant_has_followup_question",
    "log_assistant_question_mark_count",
    "log_user_word_count",
    "log_user_char_count",
    "turn_index",
    "relative_turn_position",
    "prob_brainstorm",
]

CATEGORICAL_FEATURES = [
    "current_turn_type",
    "domain_top",
]


def word_count(text):
    return len(re.findall(r"\b\w+\b", str(text or "")))


def load_and_prepare_base():
    df = pd.read_csv(INPUT, usecols=USECOLS)
    df = df[df["has_next_user_turn"].astype(bool)].copy()
    df["turn_text"] = df["turn_text"].fillna("").astype(str)
    df["user_word_count"] = df["turn_text"].map(word_count)
    df["user_char_count"] = df["turn_text"].str.len()
    df["relative_turn_position"] = df["turn_index"] / df["parsed_user_turns"].replace(0, np.nan)
    df["relative_turn_position"] = df["relative_turn_position"].fillna(0)

    for col in [
        "assistant_has_numbered_or_bulleted_list",
        "assistant_has_multiple_option_cue",
        "assistant_has_followup_question",
    ]:
        df[col] = df[col].astype(bool).astype(int)

    df["log_assistant_word_count"] = np.log1p(df["assistant_word_count"].fillna(0))
    df["log_assistant_char_count"] = np.log1p(df["assistant_char_count"].fillna(0))
    df["log_assistant_numbered_or_bulleted_items"] = np.log1p(df["assistant_numbered_or_bulleted_items"].fillna(0))
    df["log_assistant_option_term_count"] = np.log1p(df["assistant_option_term_count"].fillna(0))
    df["log_assistant_question_mark_count"] = np.log1p(df["assistant_question_mark_count"].fillna(0))
    df["log_user_word_count"] = np.log1p(df["user_word_count"].fillna(0))
    df["log_user_char_count"] = np.log1p(df["user_char_count"].fillna(0))
    df["prob_brainstorm"] = df["prob_brainstorm"].fillna(0)

    df["current_turn_type"] = df["multiclass_prediction_label"].fillna("unknown")
    domain = df["Q1.MainDomain"].fillna("Unknown").astype(str)
    top_domains = set(domain.value_counts().head(20).index)
    df["domain_top"] = domain.where(domain.isin(top_domains), "Other")

    df["target_broad_continue_exploration"] = df["next_is_exploration"].astype(bool).astype(int)
    df["target_strict_continue_exploration"] = df["next_label"].eq("strict_brainstorming").astype(int)

    DATA_OUT.joinpath("continuation_prediction_base_rows.csv").write_text("", encoding="utf-8")
    return df


def build_design_matrix(df, train_columns=None, scaler=None, fit=False):
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES, dummy_na=False)
    X = pd.concat([X_num, X_cat], axis=1)

    if fit:
        train_columns = list(X.columns)
    else:
        for col in train_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[train_columns]

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, train_columns, scaler


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
        prob = 1 / (1 + np.exp(-score))

    metrics = {
        "model": name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else math.nan,
        "average_precision": float(average_precision_score(y_test, prob)) if len(np.unique(y_test)) == 2 else math.nan,
    }
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return model, pred, prob, metrics, report, cm


def train_task(base_df, task_name, subset_mask, target_col):
    df = base_df[subset_mask].copy()
    df.to_csv(DATA_OUT / f"{task_name}_modeling_rows.csv", index=False, encoding="utf-8-sig")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20260606)
    train_idx, test_idx = next(splitter.split(df, groups=df["conversation_hash"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    X_train, columns, scaler = build_design_matrix(train, fit=True)
    X_test, _, _ = build_design_matrix(test, train_columns=columns, scaler=scaler, fit=False)
    y_train = train[target_col].astype(int).to_numpy()
    y_test = test[target_col].astype(int).to_numpy()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=20260606,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=20260606,
        ),
    }

    rows = []
    artifacts = {}
    for model_name, model in models.items():
        fitted, pred, prob, metrics, report, cm = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        rows.append(metrics)
        artifacts[model_name] = {
            "model": fitted,
            "pred": pred,
            "prob": prob,
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm,
        }
        pd.DataFrame(cm, index=["true_leave", "true_continue"], columns=["pred_leave", "pred_continue"]).to_csv(
            RESULT_OUT / f"{task_name}_{model_name}_confusion_matrix.csv",
            encoding="utf-8-sig",
        )
        Path(RESULT_OUT / f"{task_name}_{model_name}_classification_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(RESULT_OUT / f"{task_name}_model_metrics.csv", index=False, encoding="utf-8-sig")
    best_name = metrics_df.iloc[0]["model"]
    best = artifacts[best_name]["model"]

    # Refit best model on all rows for scenario simulation.
    X_all, columns_all, scaler_all = build_design_matrix(df, fit=True)
    y_all = df[target_col].astype(int).to_numpy()
    best_refit = models[best_name]
    best_refit.fit(X_all, y_all)

    bundle = {
        "task_name": task_name,
        "target_col": target_col,
        "best_model_name": best_name,
        "model": best_refit,
        "feature_columns": columns_all,
        "scaler": scaler_all,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "positive_class": "next turn continues exploration",
        "training_rows": int(len(df)),
        "positive_rate": float(df[target_col].mean()),
    }
    joblib.dump(bundle, MODEL_OUT / f"{task_name}_best_model.joblib")

    # Coefficients / importances for interpretability.
    if best_name == "logistic_regression":
        coef = pd.DataFrame({"feature": columns_all, "coefficient": best_refit.coef_[0]})
        coef["abs_coefficient"] = coef["coefficient"].abs()
        coef.sort_values("abs_coefficient", ascending=False).to_csv(
            RESULT_OUT / f"{task_name}_logistic_feature_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif best_name == "random_forest":
        imp = pd.DataFrame({"feature": columns_all, "importance": best_refit.feature_importances_})
        imp.sort_values("importance", ascending=False).to_csv(
            RESULT_OUT / f"{task_name}_random_forest_feature_importances.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return {
        "task_name": task_name,
        "n_rows": int(len(df)),
        "positive_rate": float(df[target_col].mean()),
        "best_model_name": str(best_name),
        "metrics": metrics_df.to_dict(orient="records"),
        "bundle": bundle,
        "train_columns": columns_all,
        "scaler": scaler_all,
    }


def baseline_profile(df, task_name):
    med = {}
    for col in NUMERIC_FEATURES:
        med[col] = float(df[col].median())
    med["current_turn_type"] = "strict_brainstorming" if task_name == "strict_only" else "single_creative_generation"
    med["domain_top"] = "Other"
    return med


def scenario_rows(base_profile):
    scenarios = [
        {
            "scenario": "short_no_options_no_followup",
            "assistant_words": 80,
            "items": 0,
            "list": 0,
            "option_cue": 0,
            "option_terms": 0,
            "followup": 0,
            "questions": 0,
        },
        {
            "scenario": "short_with_followup_question",
            "assistant_words": 80,
            "items": 0,
            "list": 0,
            "option_cue": 0,
            "option_terms": 0,
            "followup": 1,
            "questions": 1,
        },
        {
            "scenario": "medium_multiple_options_list",
            "assistant_words": 250,
            "items": 5,
            "list": 1,
            "option_cue": 1,
            "option_terms": 3,
            "followup": 0,
            "questions": 0,
        },
        {
            "scenario": "medium_options_plus_followup",
            "assistant_words": 250,
            "items": 5,
            "list": 1,
            "option_cue": 1,
            "option_terms": 3,
            "followup": 1,
            "questions": 1,
        },
        {
            "scenario": "long_polished_single_answer",
            "assistant_words": 600,
            "items": 0,
            "list": 0,
            "option_cue": 0,
            "option_terms": 0,
            "followup": 0,
            "questions": 0,
        },
        {
            "scenario": "long_many_options_list",
            "assistant_words": 600,
            "items": 10,
            "list": 1,
            "option_cue": 1,
            "option_terms": 6,
            "followup": 0,
            "questions": 0,
        },
        {
            "scenario": "long_options_plus_followup",
            "assistant_words": 600,
            "items": 10,
            "list": 1,
            "option_cue": 1,
            "option_terms": 6,
            "followup": 1,
            "questions": 2,
        },
    ]
    rows = []
    for s in scenarios:
        row = dict(base_profile)
        words = s["assistant_words"]
        row["log_assistant_word_count"] = math.log1p(words)
        row["log_assistant_char_count"] = math.log1p(words * 7)
        row["log_assistant_numbered_or_bulleted_items"] = math.log1p(s["items"])
        row["assistant_has_numbered_or_bulleted_list"] = s["list"]
        row["assistant_has_multiple_option_cue"] = s["option_cue"]
        row["log_assistant_option_term_count"] = math.log1p(s["option_terms"])
        row["assistant_has_followup_question"] = s["followup"]
        row["log_assistant_question_mark_count"] = math.log1p(s["questions"])
        row["scenario"] = s["scenario"]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_scenarios(task_result, task_df):
    bundle = joblib.load(MODEL_OUT / f"{task_result['task_name']}_best_model.joblib")
    profile = baseline_profile(task_df, task_result["task_name"])
    scenarios = scenario_rows(profile)
    X, _, _ = build_design_matrix(
        scenarios,
        train_columns=bundle["feature_columns"],
        scaler=bundle["scaler"],
        fit=False,
    )
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
        prob = 1 / (1 + np.exp(-score))
    out = scenarios[["scenario"]].copy()
    out["predicted_continue_probability"] = prob
    out["task_name"] = task_result["task_name"]
    out.to_csv(SIM_OUT / f"{task_result['task_name']}_scenario_simulation.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(broad_result, strict_result, broad_sim, strict_sim):
    broad_metrics = pd.DataFrame(broad_result["metrics"]).iloc[0].to_dict()
    strict_metrics = pd.DataFrame(strict_result["metrics"]).iloc[0].to_dict()

    sim = pd.concat([broad_sim, strict_sim], ignore_index=True)
    pivot = sim.pivot(index="scenario", columns="task_name", values="predicted_continue_probability").reset_index()
    pivot.to_csv(SIM_OUT / "scenario_simulation_broad_vs_strict.csv", index=False, encoding="utf-8-sig")

    def metrics_text(m):
        return (
            f"best model `{m['model']}`, ROC-AUC {m['roc_auc']:.3f}, "
            f"average precision {m['average_precision']:.3f}, balanced accuracy {m['balanced_accuracy']:.3f}, "
            f"F1 {m['f1']:.3f}"
        )

    sim_lines = []
    for _, row in pivot.iterrows():
        broad_p = row.get("broad", np.nan)
        strict_p = row.get("strict_only", np.nan)
        sim_lines.append(
            f"- {row['scenario']}: broad {broad_p*100:.1f}%, strict-only {strict_p*100:.1f}%"
        )

    report = f"""# Exploration Continuation Prediction

This experiment predicts whether the user's next turn continues exploration after an exploration turn. It uses current predicted multi-class labels, not teacher-validated labels.

## Targets

Broad continuation model:

- Rows: current turn is broad exploration (`strict_brainstorming` or `single_creative_generation`) and has a next user turn.
- Target: next user turn is also broad exploration.
- Positive rate: {broad_result['positive_rate']*100:.2f}%.

Strict-only continuation model:

- Rows: current turn is `strict_brainstorming` and has a next user turn.
- Target: next user turn is also `strict_brainstorming`.
- Positive rate: {strict_result['positive_rate']*100:.2f}%.

## Features

The models use assistant response features and context features:

- assistant response length
- number of numbered/bulleted items
- whether the response has a list
- whether it contains multiple-option cues
- option-term count
- whether it asks a follow-up question
- number of question marks
- current turn type
- user prompt length
- turn position
- task domain

## Model Results

Broad continuation: {metrics_text(broad_metrics)}

Strict-only continuation: {metrics_text(strict_metrics)}

## Scenario Simulation

Predicted probability that the next user turn continues exploration:

{chr(10).join(sim_lines)}

## Interpretation

This is still predictive/descriptive rather than causal. It suggests which AI response patterns are associated with continued exploration, but it does not prove that changing the AI response would cause the user to continue exploring.

The useful next step is to inspect high-probability and low-probability examples, then potentially fit a more formal model with conversation-level controls.

## Output Files

- `01_model_data/broad_modeling_rows.csv`
- `01_model_data/strict_only_modeling_rows.csv`
- `02_models/broad_best_model.joblib`
- `02_models/strict_only_best_model.joblib`
- `03_results/broad_model_metrics.csv`
- `03_results/strict_only_model_metrics.csv`
- `05_simulation_outputs/scenario_simulation_broad_vs_strict.csv`
"""
    (REPORT_OUT / "exploration_continuation_prediction_report.md").write_text(report, encoding="utf-8")

    slack = f"""老师好，我继续做了下一步 exploration continuation prediction，也就是预测用户在一个 exploration turn 后，下一轮是否继续 exploration。

我做了两个模型：broad continuation 模型把 `strict_brainstorming` 和 `single_creative_generation` 都算 exploration；strict-only 模型只看 `strict_brainstorming` 是否继续。模型 features 包括 AI 回复长度、是否 numbered/bulleted list、是否有 multiple option cues、是否问 follow-up question、question marks、当前 turn type、用户 prompt 长度和 turn position 等。

初步结果：broad continuation 的 positive rate 是 {broad_result['positive_rate']*100:.2f}%，最佳模型 ROC-AUC 是 {broad_metrics['roc_auc']:.3f}；strict-only continuation 的 positive rate 是 {strict_result['positive_rate']*100:.2f}%，最佳模型 ROC-AUC 是 {strict_metrics['roc_auc']:.3f}。

我也做了一个简单 scenario simulation，看不同 AI response pattern 下用户继续 exploration 的预测概率。这个目前仍然是 predictive/descriptive，不是 causal，但可以帮助我们初步判断什么样的 AI response 和 continued exploration 更相关。"""
    (REPORT_OUT / "slack_update_continuation_prediction.md").write_text(slack, encoding="utf-8")


def main():
    base_df = load_and_prepare_base()

    broad_mask = base_df["is_exploration_turn"].astype(bool)
    strict_mask = base_df["is_strict_brainstorming_turn"].astype(bool)

    broad_result = train_task(
        base_df,
        task_name="broad",
        subset_mask=broad_mask,
        target_col="target_broad_continue_exploration",
    )
    strict_result = train_task(
        base_df,
        task_name="strict_only",
        subset_mask=strict_mask,
        target_col="target_strict_continue_exploration",
    )

    broad_df = base_df[broad_mask].copy()
    strict_df = base_df[strict_mask].copy()
    broad_sim = predict_scenarios(broad_result, broad_df)
    strict_sim = predict_scenarios(strict_result, strict_df)

    summary = {
        "broad": {
            "n_rows": broad_result["n_rows"],
            "positive_rate": broad_result["positive_rate"],
            "best_model": broad_result["best_model_name"],
            "best_metrics": broad_result["metrics"][0],
        },
        "strict_only": {
            "n_rows": strict_result["n_rows"],
            "positive_rate": strict_result["positive_rate"],
            "best_model": strict_result["best_model_name"],
            "best_metrics": strict_result["metrics"][0],
        },
    }
    (RESULT_OUT / "continuation_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(broad_result, strict_result, broad_sim, strict_sim)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
