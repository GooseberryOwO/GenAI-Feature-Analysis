from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


BASE = Path(os.environ["GENAI_DATA_ROOT"])
ROOT = Path(
    os.environ.get(
        "GENAI_GOAL1_ROOT",
        str(BASE / "15_goal1_brainstorm_process_model"),
    )
)
DATA = ROOT / "01_analysis_data"
RESULTS = ROOT / "02_results"
FIGURES = ROOT / "03_figures"
REPORTS = ROOT / "04_reports"
PRIVATE = ROOT / "05_private_examples"
for folder in (DATA, RESULTS, FIGURES, REPORTS, PRIVATE):
    folder.mkdir(parents=True, exist_ok=True)

TURN_FILE = (
    BASE
    / "07_multiclass_creativity_classifier"
    / "03_full_dataset_predictions"
    / "merged_full_expanded_turn_level_multiclass_predictions.csv"
)
CALIBRATED_FLOW = DATA / "human_calibrated_multiturn_flow_final.csv"

LABEL_TO_STATE = {
    "strict_brainstorming": "W",
    "single_creative_generation": "G",
    "refinement_or_selection": "L",
    "non_creative_task": "N",
}
STATE_NAMES = {
    "W": "wide/strict brainstorming",
    "G": "single creative generation",
    "L": "local refinement/selection",
    "N": "non-creative",
}
TAXONOMY_NAMES = {
    "single_generation_only": "Single creative generation only",
    "wide_without_local": "Wide exploration without local refinement",
    "local_without_wide": "Local refinement without observed wide exploration",
    "wide_to_local": "Wide to local",
    "local_to_wide": "Local to wide",
    "iterative_reopening": "Iterative reopening",
    "mixed_creative": "Other mixed creative process",
}
GROUP_NAMES = {
    "creative_session_dominant": "Creative-session dominant",
    "brief_creativity_in_broader_task": "Brief creativity within broader tasks",
    "extended_iterative_process": "Extended iterative process",
}
RUN_NAMES = {
    "broad_exploration": "Broad creative exploration",
    "strict_brainstorming": "Strict brainstorming",
    "local_refinement": "Local refinement",
    "all_creative_activity": "All creative activity",
}
LABEL_ORDER = [
    "non_creative_task",
    "single_creative_generation",
    "strict_brainstorming",
    "refinement_or_selection",
]

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def compress(values):
    output = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def run_lengths(values, accepted):
    runs = []
    current = 0
    for value in values:
        if value in accepted:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def process_taxonomy(states):
    creative = [state for state in states if state in {"W", "G", "L"}]
    compressed = compress(creative)
    wl = [state for state in compressed if state in {"W", "L"}]
    if not creative:
        return "non_creative"
    if set(creative) == {"G"}:
        return "single_generation_only"
    if "W" in creative and "L" not in creative:
        return "wide_without_local"
    if "L" in creative and "W" not in creative:
        return "local_without_wide"
    switches = sum(a != b for a, b in zip(wl, wl[1:]))
    if switches >= 2:
        return "iterative_reopening"
    if wl and wl[0] == "W":
        return "wide_to_local"
    if wl and wl[0] == "L":
        return "local_to_wide"
    return "mixed_creative"


def conversation_features(group):
    labels = group["multiclass_prediction_label"].tolist()
    states = [LABEL_TO_STATE[label] for label in labels]
    n = len(states)
    creative_mask = np.array([state != "N" for state in states], dtype=bool)
    creative_positions = np.flatnonzero(creative_mask)
    compressed = compress(states)
    state_switches = sum(a != b for a, b in zip(compressed, compressed[1:]))
    creative_sequence = [s for s in states if s != "N"]
    creative_compressed = compress(creative_sequence)
    creative_switches = sum(a != b for a, b in zip(creative_compressed, creative_compressed[1:]))
    broad_runs = run_lengths(states, {"W", "G"})
    strict_runs = run_lengths(states, {"W"})
    local_runs = run_lengths(states, {"L"})
    creative_runs = run_lengths(states, {"W", "G", "L"})

    def share(state):
        return states.count(state) / n

    return pd.Series(
        {
            "n_turns": n,
            "wide_turns": states.count("W"),
            "generation_turns": states.count("G"),
            "local_turns": states.count("L"),
            "noncreative_turns": states.count("N"),
            "wide_share": share("W"),
            "generation_share": share("G"),
            "local_share": share("L"),
            "noncreative_share": share("N"),
            "creative_share": creative_mask.mean(),
            "state_switches": state_switches,
            "creative_state_switches": creative_switches,
            "creative_run_count": len(creative_runs),
            "longest_creative_run": max(creative_runs, default=0),
            "longest_broad_run": max(broad_runs, default=0),
            "longest_strict_run": max(strict_runs, default=0),
            "longest_local_run": max(local_runs, default=0),
            "first_creative_position": int(creative_positions[0] + 1) if len(creative_positions) else np.nan,
            "first_creative_relative": (
                float((creative_positions[0] + 1) / n) if len(creative_positions) else np.nan
            ),
            "starts_creative": bool(creative_mask[0]),
            "ends_creative": bool(creative_mask[-1]),
            "process_taxonomy": process_taxonomy(states),
            "state_sequence": ">".join(states),
            "compressed_state_sequence": ">".join(compressed),
            "main_domain": group["Q1.MainDomain"].iloc[0],
            "main_task": group["Q1.MainTask"].iloc[0],
        }
    )


def build_conversation_data(turns):
    conv = (
        turns.sort_values(["conversation_hash", "turn_index"])
        .groupby("conversation_hash", sort=False)
        .apply(conversation_features, include_groups=False)
        .reset_index()
    )
    conv["is_creative_conversation"] = conv["creative_share"] > 0
    conv.to_csv(DATA / "conversation_process_features_private.csv", index=False)
    return conv


def label_clusters(cluster_summary):
    labels = {}
    extended_cluster = (
        cluster_summary["n_turns"].rank(pct=True)
        + cluster_summary["creative_state_switches"].rank(pct=True)
        + cluster_summary["creative_run_count"].rank(pct=True)
    ).idxmax()
    labels[extended_cluster] = "extended_iterative_process"
    remaining = cluster_summary.drop(index=extended_cluster)
    creative_session_cluster = remaining["noncreative_share"].idxmin()
    labels[creative_session_cluster] = "creative_session_dominant"
    for cluster in cluster_summary.index:
        labels.setdefault(cluster, "brief_creativity_in_broader_task")
    return labels


def cluster_conversations(conv):
    creative = conv[conv["is_creative_conversation"]].copy()
    features = [
        "wide_share",
        "generation_share",
        "local_share",
        "noncreative_share",
        "creative_state_switches",
        "creative_run_count",
        "longest_creative_run",
        "first_creative_relative",
        "n_turns",
    ]
    x = creative[features].copy()
    x["n_turns"] = np.log1p(x["n_turns"])
    x = x.fillna(1.0)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    validation = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=25, random_state=42)
        labels = model.fit_predict(z)
        validation.append(
            {
                "k": k,
                "silhouette": silhouette_score(z, labels, sample_size=min(10000, len(z)), random_state=42),
                "inertia": model.inertia_,
            }
        )
    pd.DataFrame(validation).to_csv(RESULTS / "cluster_validation.csv", index=False)

    model = KMeans(n_clusters=3, n_init=50, random_state=42)
    reference_labels = model.fit_predict(z)
    creative["cluster_id"] = reference_labels
    stability = []
    for seed in range(10):
        alternate = KMeans(n_clusters=3, n_init=25, random_state=seed).fit_predict(z)
        stability.append(
            {
                "seed": seed,
                "adjusted_rand_index_vs_seed42": adjusted_rand_score(reference_labels, alternate),
            }
        )
    pd.DataFrame(stability).to_csv(RESULTS / "three_group_cluster_stability.csv", index=False)
    summary = creative.groupby("cluster_id")[features].mean()
    summary["conversations"] = creative.groupby("cluster_id").size()
    summary["share"] = summary["conversations"] / len(creative)
    labels = label_clusters(summary)
    creative["user_process_group"] = creative["cluster_id"].map(labels)
    summary["user_process_group"] = summary.index.map(labels)
    summary.reset_index().to_csv(RESULTS / "three_group_cluster_summary.csv", index=False)

    conv = conv.merge(
        creative[["conversation_hash", "cluster_id", "user_process_group"]],
        on="conversation_hash",
        how="left",
    )
    conv.to_csv(DATA / "conversation_process_features_with_clusters_private.csv", index=False)
    return conv, summary.reset_index()


def transition_matrix(turns):
    rows = []
    for _, group in turns.sort_values(["conversation_hash", "turn_index"]).groupby("conversation_hash"):
        labels = group["multiclass_prediction_label"].tolist()
        rows.extend(zip(labels, labels[1:]))
    counts = pd.DataFrame(rows, columns=["source_label", "target_label"]).value_counts().reset_index(name="n")
    counts["conditional_share"] = counts["n"] / counts.groupby("source_label")["n"].transform("sum")
    counts.to_csv(RESULTS / "four_class_transition_matrix.csv", index=False)
    return counts


def summarize_taxonomy(conv):
    creative = conv[conv["is_creative_conversation"]]
    taxonomy = (
        creative.groupby("process_taxonomy")
        .agg(
            conversations=("conversation_hash", "size"),
            mean_turns=("n_turns", "mean"),
            median_turns=("n_turns", "median"),
            mean_creative_share=("creative_share", "mean"),
            mean_switches=("creative_state_switches", "mean"),
        )
        .reset_index()
    )
    taxonomy["share"] = taxonomy["conversations"] / taxonomy["conversations"].sum()
    se = np.sqrt(taxonomy["share"] * (1 - taxonomy["share"]) / taxonomy["conversations"].sum())
    taxonomy["ci95_low"] = (taxonomy["share"] - 1.96 * se).clip(lower=0)
    taxonomy["ci95_high"] = (taxonomy["share"] + 1.96 * se).clip(upper=1)
    taxonomy = taxonomy.sort_values("conversations", ascending=False)
    taxonomy.to_csv(RESULTS / "process_taxonomy_summary.csv", index=False)
    return taxonomy


def summarize_runs(turns):
    records = []
    state_sets = {
        "broad_exploration": {"W", "G"},
        "strict_brainstorming": {"W"},
        "local_refinement": {"L"},
        "all_creative_activity": {"W", "G", "L"},
    }
    for conv_hash, group in turns.sort_values(["conversation_hash", "turn_index"]).groupby("conversation_hash"):
        states = [LABEL_TO_STATE[label] for label in group["multiclass_prediction_label"]]
        for run_type, accepted in state_sets.items():
            for run_number, length in enumerate(run_lengths(states, accepted), start=1):
                records.append(
                    {
                        "conversation_hash": conv_hash,
                        "run_type": run_type,
                        "run_number": run_number,
                        "run_length": length,
                    }
                )
    runs = pd.DataFrame(records)
    runs.to_csv(DATA / "creative_run_lengths_private.csv", index=False)
    summary = (
        runs.groupby("run_type")["run_length"]
        .agg(runs="size", mean="mean", median="median", std="std", max="max")
        .reset_index()
    )
    for threshold in (1, 2, 3, 5):
        values = runs.groupby("run_type")["run_length"].apply(lambda s: (s >= threshold).mean())
        summary[f"share_ge_{threshold}"] = summary["run_type"].map(values)
    summary.to_csv(RESULTS / "run_length_summary.csv", index=False)
    distribution = (
        runs.assign(run_length_bin=runs["run_length"].clip(upper=8))
        .groupby(["run_type", "run_length_bin"])
        .size()
        .reset_index(name="n")
    )
    distribution["share"] = distribution["n"] / distribution.groupby("run_type")["n"].transform("sum")
    distribution.to_csv(RESULTS / "run_length_distribution.csv", index=False)
    return summary, distribution


def calibrated_wide_local_duration():
    flow = pd.read_csv(CALIBRATED_FLOW)
    records = []
    for row in flow.itertuples(index=False):
        sequence = [row.state_t1, row.human_state_t2, row.human_state_t3]
        active = [state for state in sequence if state in {"wide_search", "local_search"}]
        initial = sequence[0]
        initial_run = 1
        for state in sequence[1:]:
            if state == initial:
                initial_run += 1
            else:
                break
        first_local = next((idx + 1 for idx, state in enumerate(sequence) if state == "local_search"), np.nan)
        first_wide = next((idx + 1 for idx, state in enumerate(sequence) if state == "wide_search"), np.nan)
        records.append(
            {
                "audit_id": row.audit_id,
                "population_weight": row.population_weight,
                "initial_state": initial,
                "initial_run_observed": initial_run,
                "active_turns_observed": len(active),
                "first_local_position": first_local,
                "first_wide_position": first_wide,
            }
        )
    durations = pd.DataFrame(records)
    durations.to_csv(DATA / "calibrated_wide_local_duration_private.csv", index=False)

    summary_rows = []
    for state, group in durations.groupby("initial_state"):
        weights = group["population_weight"].to_numpy()
        summary_rows.append(
            {
                "initial_state": state,
                "estimated_n": weights.sum(),
                "weighted_mean_initial_run_observed": np.average(group["initial_run_observed"], weights=weights),
                "weighted_mean_active_turns_observed": np.average(group["active_turns_observed"], weights=weights),
                "share_persist_to_t2": np.average(group["initial_run_observed"] >= 2, weights=weights),
                "share_persist_to_t3": np.average(group["initial_run_observed"] >= 3, weights=weights),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "calibrated_wide_local_duration_summary.csv", index=False)
    return summary


def cramers_v(table):
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return chi2, p, np.sqrt(phi2 / max(1, min(k - 1, r - 1)))


def topic_analysis(conv):
    creative = conv[conv["is_creative_conversation"]].copy()
    creative["main_domain"] = creative["main_domain"].fillna("Unknown").astype(str)
    top_domains = creative["main_domain"].value_counts().head(15).index
    scoped = creative[creative["main_domain"].isin(top_domains)]

    topic = (
        scoped.groupby("main_domain")
        .agg(
            conversations=("conversation_hash", "size"),
            mean_turns=("n_turns", "mean"),
            strict_conversation_rate=("wide_turns", lambda s: (s > 0).mean()),
            mean_wide_share=("wide_share", "mean"),
            mean_local_share=("local_share", "mean"),
            mean_generation_share=("generation_share", "mean"),
            mean_switches=("creative_state_switches", "mean"),
        )
        .reset_index()
        .sort_values("conversations", ascending=False)
    )
    topic.to_csv(RESULTS / "topic_process_summary_top15.csv", index=False)

    cluster_table = pd.crosstab(scoped["main_domain"], scoped["user_process_group"])
    cluster_share = cluster_table.div(cluster_table.sum(axis=1), axis=0).reset_index()
    cluster_share.to_csv(RESULTS / "topic_three_group_distribution_top15.csv", index=False)

    taxonomy_table = pd.crosstab(scoped["main_domain"], scoped["process_taxonomy"])
    chi_c, p_c, v_c = cramers_v(cluster_table)
    chi_t, p_t, v_t = cramers_v(taxonomy_table)
    pd.DataFrame(
        [
            {"relationship": "topic_x_three_group", "chi2": chi_c, "p_value": p_c, "cramers_v": v_c},
            {"relationship": "topic_x_process_taxonomy", "chi2": chi_t, "p_value": p_t, "cramers_v": v_t},
        ]
    ).to_csv(RESULTS / "topic_association_tests.csv", index=False)

    lower = creative["main_domain"].str.lower()
    conditions = [
        lower.str.contains("creative writing|fiction|screenwriting|story|poetry", regex=True),
        lower.str.contains("translation|language|linguistic", regex=True),
        lower.str.contains("software|program|web development|machine learning|data science|coding", regex=True),
        lower.str.contains("marketing|brand|advertis|social media|copywriting", regex=True),
        lower.str.contains("education|academic|teaching|learning|research", regex=True),
        lower.str.contains("business|management|finance|career|human resources", regex=True),
        lower.str.contains("game|entertainment|film|television|music", regex=True),
        lower.str.contains("design|visual|art|image|video", regex=True),
    ]
    choices = [
        "Creative writing",
        "Translation and language",
        "Software and technology",
        "Marketing and branding",
        "Education and research",
        "Business and management",
        "Games and entertainment",
        "Visual design and media",
    ]
    creative["topic_family"] = np.select(conditions, choices, default="Other")
    family = (
        creative.groupby("topic_family")
        .agg(
            conversations=("conversation_hash", "size"),
            mean_turns=("n_turns", "mean"),
            strict_conversation_rate=("wide_turns", lambda s: (s > 0).mean()),
            mean_wide_share=("wide_share", "mean"),
            mean_local_share=("local_share", "mean"),
            mean_switches=("creative_state_switches", "mean"),
        )
        .reset_index()
        .sort_values("conversations", ascending=False)
    )
    family.to_csv(RESULTS / "topic_family_process_summary.csv", index=False)
    family_cluster = pd.crosstab(creative["topic_family"], creative["user_process_group"])
    family_cluster_share = family_cluster.div(family_cluster.sum(axis=1), axis=0).reset_index()
    family_cluster_share.to_csv(RESULTS / "topic_family_three_group_distribution.csv", index=False)
    return family, family_cluster_share


def private_examples(turns, conv):
    eligible = conv[conv["is_creative_conversation"]].copy()
    selected = pd.concat(
        [
            group.sample(min(5, len(group)), random_state=42)
            for _, group in eligible.groupby("process_taxonomy", sort=True)
        ],
        ignore_index=True,
    )
    chosen = set(selected["conversation_hash"])
    sample_turns = turns[turns["conversation_hash"].isin(chosen)].copy()
    sample_turns = sample_turns.merge(
        selected[["conversation_hash", "process_taxonomy", "user_process_group"]],
        on="conversation_hash",
        how="left",
    )
    sample_turns.to_csv(PRIVATE / "qualitative_examples_private.csv", index=False, encoding="utf-8-sig")


def make_figures(taxonomy, clusters, run_summary, transitions, topic, cluster_share):
    colors = ["#2F6B9E", "#E5832A", "#459D8B", "#7B68A6", "#8B949E", "#C45A52"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    t = taxonomy.sort_values("share")
    labels = t["process_taxonomy"].map(TAXONOMY_NAMES).fillna(t["process_taxonomy"])
    ax.barh(labels, t["share"] * 100, color=colors[: len(t)])
    ax.set_xlabel("Creative conversations (%)")
    ax.set_title("Goal 1: Brainstorming process taxonomy", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "process_taxonomy_distribution.png", bbox_inches="tight")
    plt.close(fig)

    profile_cols = ["wide_share", "generation_share", "local_share", "noncreative_share"]
    profile = clusters.set_index("user_process_group")[profile_cols]
    profile.index = [GROUP_NAMES.get(name, name) for name in profile.index]
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    profile.plot(kind="bar", ax=ax, color=["#2F6B9E", "#E5832A", "#459D8B", "#8B949E"])
    ax.set_ylabel("Mean share of user turns")
    ax.set_xlabel("")
    ax.set_title("Three user-process groups", weight="bold")
    ax.legend(["Wide/strict", "Single generation", "Local refinement", "Non-creative"], frameon=False)
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "three_group_cluster_profiles.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    r = run_summary.sort_values("mean")
    run_labels = r["run_type"].map(RUN_NAMES).fillna(r["run_type"])
    ax.barh(run_labels, r["mean"], color=["#8B949E", "#E5832A", "#2F6B9E", "#459D8B"])
    for idx, value in enumerate(r["mean"]):
        ax.text(value + 0.02, idx, f"{value:.2f}", va="center")
    ax.set_xlabel("Mean consecutive user turns")
    ax.set_title("Creative-search run length", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "creative_run_length_summary.png", bbox_inches="tight")
    plt.close(fig)

    matrix = transitions.pivot(index="source_label", columns="target_label", values="conditional_share").fillna(0)
    matrix = matrix.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=180)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.to_numpy().max())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center")
    ax.set_xticks(range(4), ["Non-creative", "Single generation", "Strict brainstorm", "Refinement"])
    ax.set_yticks(range(4), ["Non-creative", "Single generation", "Strict brainstorm", "Refinement"])
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Next-turn transition matrix", weight="bold")
    fig.colorbar(image, ax=ax, label="Conditional probability")
    fig.tight_layout()
    fig.savefig(FIGURES / "four_class_transition_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    heat = cluster_share.set_index("topic_family")
    heat = heat.rename(columns=GROUP_NAMES)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    image = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Topic-family differences in the three user-process groups", weight="bold")
    fig.colorbar(image, ax=ax, label="Share within topic")
    fig.tight_layout()
    fig.savefig(FIGURES / "topic_three_group_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def write_public_examples():
    text = """# Qualitative process patterns

These examples are paraphrased composites. Raw conversation text is retained
only in the private local audit file.

## Single-generation dominant

The user requests one creative artifact, accepts or lightly revises it, and
does not enter a sustained option-search process.

## Wide without local

The user repeatedly asks for additional names, titles, translations, captions,
or design alternatives without selecting one direction.

## Wide to local

The user first requests multiple alternatives, then selects one candidate,
adds constraints, asks for a deeper version, or begins implementation.

## Local to wide

The user initially develops one direction but later reopens the search space
because the selected direction is unsatisfactory or new constraints emerge.

## Iterative reopening

The conversation alternates between generating alternatives and refining a
chosen option. This pattern is closest to an iterative design process.

## Local without wide

The user enters with an existing idea and asks the AI to improve, critique,
expand, or operationalize it rather than generating alternatives first.
"""
    (REPORTS / "qualitative_process_patterns_public.md").write_text(text, encoding="utf-8")


def main():
    usecols = [
        "conversation_hash",
        "turn_index",
        "turn_text",
        "multiclass_prediction_label",
        "Q1.MainDomain",
        "Q1.MainTask",
    ]
    turns = pd.read_csv(TURN_FILE, usecols=usecols)
    turns = turns[turns["multiclass_prediction_label"].isin(LABEL_TO_STATE)].copy()
    turns = turns.sort_values(["conversation_hash", "turn_index"])

    conv = build_conversation_data(turns)
    conv, clusters = cluster_conversations(conv)
    taxonomy = summarize_taxonomy(conv)
    transitions = transition_matrix(turns)
    run_summary, _ = summarize_runs(turns)
    calibrated_wide_local_duration()
    topic, cluster_share = topic_analysis(conv)
    private_examples(turns, conv)
    write_public_examples()
    make_figures(taxonomy, clusters, run_summary, transitions, topic, cluster_share)

    print(f"turns={len(turns):,}")
    print(f"conversations={conv['conversation_hash'].nunique():,}")
    print(taxonomy.to_string(index=False))
    print(clusters.to_string(index=False))
    print(run_summary.to_string(index=False))


if __name__ == "__main__":
    main()
