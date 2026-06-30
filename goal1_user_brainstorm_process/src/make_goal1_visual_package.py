from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
RESULTS = REPO_ROOT / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_PRIVATE_ROOT = (
    Path(os.environ.get("GENAI_DATA_ROOT", r"F:\umich\summer project\organized_project_files"))
    / "15_goal1_brainstorm_process_model"
)
PRIVATE_ROOT = Path(os.environ.get("GENAI_GOAL1_PRIVATE_ROOT", DEFAULT_PRIVATE_ROOT))
PRIVATE_DATA = PRIVATE_ROOT / "01_analysis_data"

CONV_FILE = PRIVATE_DATA / "conversation_process_features_with_clusters_private.csv"

COLORS = {
    "blue": "#2F6B9E",
    "orange": "#E5832A",
    "green": "#459D8B",
    "purple": "#7B68A6",
    "red": "#C45A52",
    "gray": "#7D8790",
    "light_bg": "#F2F6F9",
    "grid": "#D6DEE6",
    "ink": "#202933",
}

GROUP_LABELS = {
    "brief_creativity_in_broader_task": "Brief creativity\nwithin broader tasks",
    "creative_session_dominant": "Creative-session\ndominant",
    "extended_iterative_process": "Extended\niterative process",
}

GROUP_COLORS = {
    "brief_creativity_in_broader_task": COLORS["blue"],
    "creative_session_dominant": COLORS["orange"],
    "extended_iterative_process": COLORS["green"],
}

TAXONOMY_LABELS = {
    "single_generation_only": "Single generation\nonly",
    "wide_without_local": "Wide without\nlocal refinement",
    "local_without_wide": "Local without\nwide exploration",
    "wide_to_local": "Wide -> local",
    "local_to_wide": "Local -> wide",
    "iterative_reopening": "Iterative\nreopening",
}

TAXONOMY_COLORS = {
    "single_generation_only": COLORS["blue"],
    "wide_without_local": COLORS["orange"],
    "local_without_wide": COLORS["green"],
    "wide_to_local": COLORS["purple"],
    "local_to_wide": COLORS["red"],
    "iterative_reopening": COLORS["gray"],
}

FEATURES = [
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


def style_ax(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#B8C3CC")
    ax.spines["bottom"].set_color("#B8C3CC")
    ax.tick_params(colors=COLORS["ink"], labelsize=10)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.65)
        ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_ribbon(ax, x0, y0, x1, y1, width0, width1, color, alpha=0.36):
    verts = [
        (x0, y0 + width0 / 2),
        ((x0 + x1) / 2, y0 + width0 / 2),
        ((x0 + x1) / 2, y1 + width1 / 2),
        (x1, y1 + width1 / 2),
        (x1, y1 - width1 / 2),
        ((x0 + x1) / 2, y1 - width1 / 2),
        ((x0 + x1) / 2, y0 - width0 / 2),
        (x0, y0 - width0 / 2),
        (x0, y0 + width0 / 2),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def add_card(ax, xy, wh, title, value, subtitle, color):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white",
            edgecolor="#D8E1E8",
            linewidth=1.2,
        )
    )
    subtitle_wrapped = textwrap.fill(subtitle, width=38)
    ax.text(x + 0.035, y + h - 0.06, title, fontsize=12.5, weight="bold", color=color, va="top")
    ax.text(x + 0.035, y + 0.118, value, fontsize=23, weight="bold", color=COLORS["ink"], va="bottom")
    ax.text(x + 0.035, y + 0.05, subtitle_wrapped, fontsize=8.9, color="#5B6975", va="bottom", linespacing=1.15)


def load_public_results():
    return {
        "taxonomy": pd.read_csv(RESULTS / "process_taxonomy_summary.csv"),
        "run_summary": pd.read_csv(RESULTS / "run_length_summary.csv"),
        "run_dist": pd.read_csv(RESULTS / "run_length_distribution.csv"),
        "cluster_summary": pd.read_csv(RESULTS / "three_group_cluster_summary.csv"),
        "topic_summary": pd.read_csv(RESULTS / "topic_family_process_summary.csv"),
        "manual_ci": pd.read_csv(RESULTS / "manual_calibrated_t1_to_t2_with_ci.csv"),
    }


def make_pca_cluster_projection(conv: pd.DataFrame) -> None:
    creative = conv[conv["is_creative_conversation"]].copy()
    x = creative[FEATURES].fillna(creative[FEATURES].median(numeric_only=True))
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    creative["pc1"] = coords[:, 0]
    creative["pc2"] = coords[:, 1]

    # Public aggregate output only: no conversation ids.
    centroid = (
        creative.groupby("user_process_group")
        .agg(
            conversations=("user_process_group", "size"),
            pc1_mean=("pc1", "mean"),
            pc2_mean=("pc2", "mean"),
            pc1_sd=("pc1", "std"),
            pc2_sd=("pc2", "std"),
            mean_turns=("n_turns", "mean"),
            mean_creative_share=("creative_share", "mean"),
            mean_switches=("creative_state_switches", "mean"),
        )
        .reset_index()
    )
    centroid["pc1_explained_variance"] = pca.explained_variance_ratio_[0]
    centroid["pc2_explained_variance"] = pca.explained_variance_ratio_[1]
    centroid.to_csv(RESULTS / "pca_cluster_projection_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 7.2), facecolor=COLORS["light_bg"])
    ax.set_facecolor("white")
    for group, g in creative.groupby("user_process_group"):
        sample = g.sample(min(len(g), 4500), random_state=42)
        color = GROUP_COLORS[group]
        ax.scatter(
            sample["pc1"],
            sample["pc2"],
            s=8,
            alpha=0.12,
            color=color,
            edgecolors="none",
            label=GROUP_LABELS[group].replace("\n", " "),
        )
        center = centroid[centroid["user_process_group"] == group].iloc[0]
        ax.scatter(center["pc1_mean"], center["pc2_mean"], s=170, color=color, edgecolor="white", linewidth=2.2, zorder=5)
        ell = Ellipse(
            (center["pc1_mean"], center["pc2_mean"]),
            width=2.2 * center["pc1_sd"],
            height=2.2 * center["pc2_sd"],
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=1.5,
        )
        ax.add_patch(ell)
    label_positions = {
        "brief_creativity_in_broader_task": (-1.65, 2.6, "left"),
        "creative_session_dominant": (-1.35, -3.1, "left"),
        "extended_iterative_process": (3.7, 2.6, "left"),
    }
    for group, (x_text, y_text, ha) in label_positions.items():
        center = centroid[centroid["user_process_group"] == group].iloc[0]
        ax.annotate(
            GROUP_LABELS[group],
            xy=(center["pc1_mean"], center["pc2_mean"]),
            xytext=(x_text, y_text),
            ha=ha,
            va="center",
            fontsize=10.6,
            weight="bold",
            color=GROUP_COLORS[group],
            arrowprops={"arrowstyle": "-", "color": GROUP_COLORS[group], "linewidth": 1.2, "alpha": 0.7},
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#D8E1E8", "lw": 0.9, "alpha": 0.92},
        )
    style_ax(ax, grid_axis=None)
    ax.axhline(0, color="#CBD5DD", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#CBD5DD", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=11)
    xlim = creative["pc1"].quantile([0.005, 0.995]).to_numpy()
    ylim = creative["pc2"].quantile([0.005, 0.995]).to_numpy()
    ax.set_xlim(xlim[0] - 0.5, xlim[1] + 0.8)
    ax.set_ylim(ylim[0] - 0.5, ylim[1] + 0.8)
    fig.subplots_adjust(top=0.82)
    fig.suptitle("Three user-process groups form interpretable regions", x=0.08, y=0.965, ha="left", fontsize=20, weight="bold", color=COLORS["ink"])
    fig.text(
        0.08,
        0.91,
        "PCA projection of conversation-level process features; points are creative conversations, ellipses show approximate group spread.",
        fontsize=11.5,
        color="#5B6975",
    )
    save(fig, "pca_user_process_clusters.png")


def make_process_taxonomy_flow(taxonomy: pd.DataFrame) -> None:
    taxonomy = taxonomy[taxonomy["process_taxonomy"].isin(TAXONOMY_LABELS)].copy()
    taxonomy = taxonomy.sort_values("share", ascending=False)
    total = taxonomy["conversations"].sum()
    fig, ax = plt.subplots(figsize=(13, 8.4), facecolor=COLORS["light_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "Most creative conversations stop before iterative search", fontsize=22, weight="bold", color=COLORS["ink"])
    ax.text(
        0.02,
        0.89,
        "Conversation-level taxonomy among 36,018 creative conversations.",
        fontsize=12,
        color="#5B6975",
    )
    source_x, target_x = 0.12, 0.62
    source_y, source_h = 0.48, 0.56
    ax.add_patch(Rectangle((source_x - 0.018, source_y - source_h / 2), 0.036, source_h, color="#AAB5BE"))
    ax.text(source_x - 0.07, source_y, "Creative\nconversations\n100%", ha="center", va="center", fontsize=13, weight="bold")
    max_share = taxonomy["share"].max()
    y_positions = np.linspace(0.78, 0.18, len(taxonomy))
    for (_, row), y in zip(taxonomy.iterrows(), y_positions):
        key = row["process_taxonomy"]
        share = row["share"]
        color = TAXONOMY_COLORS[key]
        width = 0.04 + 0.20 * (share / max_share)
        draw_ribbon(ax, source_x + 0.018, source_y, target_x - 0.03, y, width * 0.55, width, color)
        ax.add_patch(Rectangle((target_x - 0.012, y - width / 2), 0.024, width, color=color))
        label_x = 0.70
        ax.text(
            label_x,
            y + 0.022,
            TAXONOMY_LABELS[key].replace("\n", " "),
            fontsize=11.8,
            weight="bold",
            color=COLORS["ink"],
            va="bottom",
        )
        ax.text(
            label_x,
            y - 0.022,
            f"{int(row['conversations']):,} conversations | {share * 100:.1f}%",
            fontsize=10.6,
            color="#5B6975",
            va="top",
        )
    ax.text(0.02, 0.07, "Finding: explicit wide-to-local and reopening patterns are rare, but they are much longer conversations.", fontsize=12, color="#5B6975")
    save(fig, "process_taxonomy_flow_story.png")


def make_run_length_survival(run_dist: pd.DataFrame) -> None:
    labels = {
        "all_creative_activity": "All creative activity",
        "broad_exploration": "Broad exploration",
        "strict_brainstorming": "Strict brainstorming",
        "local_refinement": "Local refinement",
    }
    colors = {
        "all_creative_activity": COLORS["gray"],
        "broad_exploration": COLORS["blue"],
        "strict_brainstorming": COLORS["orange"],
        "local_refinement": COLORS["green"],
    }
    rows = []
    for run_type, g in run_dist.groupby("run_type"):
        expanded = []
        for _, row in g.iterrows():
            label = str(row["run_length_bin"])
            if label.endswith("+"):
                length = int(label[:-1])
            else:
                length = int(label)
            expanded.append((length, int(row["n"])))
        total = sum(n for _, n in expanded)
        for threshold in range(1, 8):
            ge = sum(n for length, n in expanded if length >= threshold)
            rows.append({"run_type": run_type, "threshold": threshold, "share": ge / total if total else 0})
    surv = pd.DataFrame(rows)
    surv.to_csv(RESULTS / "run_length_survival_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor=COLORS["light_bg"])
    for run_type in ["all_creative_activity", "broad_exploration", "strict_brainstorming", "local_refinement"]:
        g = surv[surv["run_type"] == run_type]
        ax.plot(g["threshold"], g["share"] * 100, marker="o", linewidth=2.6, color=colors[run_type], label=labels[run_type])
    style_ax(ax, grid_axis="y")
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Creative activity usually ends after one turn", x=0.08, y=0.965, ha="left", fontsize=19, weight="bold", color=COLORS["ink"])
    fig.text(0.08, 0.905, "Survival-style view: share of creative runs lasting at least N user turns.", fontsize=11.5, color="#5B6975")
    ax.set_xlabel("Run lasts at least N turns", fontsize=11)
    ax.set_ylabel("Share of runs (%)", fontsize=11)
    ax.set_xticks(range(1, 8))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "creative_run_length_survival.png")


def make_topic_process_bubble(topic: pd.DataFrame) -> None:
    topic = topic.copy()
    topic["label"] = topic["topic_family"]
    fig, ax = plt.subplots(figsize=(11.5, 7.2), facecolor=COLORS["light_bg"])
    sizes = 200 + 1900 * (topic["conversations"] / topic["conversations"].max())
    sc = ax.scatter(
        topic["strict_conversation_rate"] * 100,
        topic["mean_switches"],
        s=sizes,
        c=topic["mean_turns"],
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=1.6,
        alpha=0.92,
    )
    label_offsets = {
        "Education and research": (-0.9, 0.018, "right"),
        "Translation and language": (-0.45, 0.018, "right"),
        "Business and management": (0.20, 0.018, "left"),
        "Marketing and branding": (0.25, 0.016, "left"),
    }
    for _, row in topic.iterrows():
        x = row["strict_conversation_rate"] * 100
        y = row["mean_switches"]
        dx, dy, ha = label_offsets.get(row["label"], (0.25, 0.015, "left"))
        ax.text(x + dx, y + dy, row["label"], fontsize=9.0, color=COLORS["ink"], ha=ha)
    style_ax(ax, grid_axis="both")
    fig.subplots_adjust(top=0.80, right=0.86)
    fig.suptitle("Topic families differ in brainstorming intensity and switching", x=0.08, y=0.965, ha="left", fontsize=19, weight="bold", color=COLORS["ink"])
    fig.text(0.08, 0.905, "Bubble size = number of creative conversations; color = mean conversation turns.", fontsize=11.5, color="#5B6975")
    ax.set_xlim(topic["strict_conversation_rate"].min() * 100 - 1.0, topic["strict_conversation_rate"].max() * 100 + 2.0)
    ax.set_xlabel("Conversations with at least one strict brainstorming turn (%)", fontsize=11)
    ax.set_ylabel("Mean creative-state switches", fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mean turns", fontsize=10)
    save(fig, "topic_process_bubble_map.png")


def make_manual_transition_story(manual_ci: pd.DataFrame) -> None:
    states = ["wide_search", "local_search"]
    target_order = ["wide_search", "local_search", "creative_continuation_other", "non_creative_shift", "conversation_end"]
    target_labels = {
        "wide_search": "Wide",
        "local_search": "Local",
        "creative_continuation_other": "Other creative",
        "non_creative_shift": "Non-creative",
        "conversation_end": "End",
    }
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["gray"], COLORS["red"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True, facecolor=COLORS["light_bg"])
    for ax, source in zip(axes, states):
        g = manual_ci[manual_ci["source_state"] == source].set_index("target_state").reindex(target_order).fillna(0)
        y = np.arange(len(g))[::-1]
        ax.barh(y, g["estimate"] * 100, color=colors, alpha=0.92)
        for yi, (_, row) in zip(y, g.iterrows()):
            low = row["ci95_low"] * 100
            high = row["ci95_high"] * 100
            est = row["estimate"] * 100
            ax.hlines(yi, low, high, color="#333333", linewidth=1.2)
            ax.text(est + 1.0, yi, f"{est:.1f}%", va="center", fontsize=10.5)
        ax.set_yticks(y, [target_labels[t] for t in g.index])
        ax.set_xlim(0, 60)
        ax.set_title("After initial " + ("wide search" if source == "wide_search" else "local search"), fontsize=13.5, weight="bold")
        ax.set_xlabel("Estimated next-turn share (%)")
        style_ax(ax, grid_axis="x")
    axes[0].set_ylabel("Next user move")
    fig.subplots_adjust(top=0.78, wspace=0.20)
    fig.suptitle("Users do not always continue broad option search", x=0.06, y=0.965, ha="left", fontsize=19, weight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.905, "Human-calibrated genuine multiple-option paths; horizontal lines show bootstrap 95% CI.", fontsize=11.5, color="#5B6975")
    save(fig, "manual_transition_story_ci.png")


def make_finding_summary_board(data: dict) -> None:
    taxonomy = data["taxonomy"]
    run_summary = data["run_summary"]
    manual_ci = data["manual_ci"]
    cluster_summary = data["cluster_summary"]
    single = taxonomy.loc[taxonomy["process_taxonomy"] == "single_generation_only", "share"].iloc[0]
    wide_no_local = taxonomy.loc[taxonomy["process_taxonomy"] == "wide_without_local", "share"].iloc[0]
    all_mean = run_summary.loc[run_summary["run_type"] == "all_creative_activity", "mean"].iloc[0]
    wide_to_local = manual_ci[(manual_ci["source_state"] == "wide_search") & (manual_ci["target_state"] == "local_search")]["estimate"].iloc[0]
    local_to_local = manual_ci[(manual_ci["source_state"] == "local_search") & (manual_ci["target_state"] == "local_search")]["estimate"].iloc[0]
    extended = cluster_summary.loc[cluster_summary["user_process_group"] == "extended_iterative_process", "share"].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 7.2), facecolor=COLORS["light_bg"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.91, "Goal 1 visual summary", fontsize=24, weight="bold", color=COLORS["ink"])
    ax.text(0.04, 0.855, "How do people use GenAI to conduct brainstorming activity?", fontsize=13, color="#5B6975")
    add_card(ax, (0.04, 0.55), (0.28, 0.25), "Creative runs", f"{all_mean:.2f} turns", "Most creative states end quickly; median run length is 1 turn.", COLORS["blue"])
    add_card(ax, (0.36, 0.55), (0.28, 0.25), "Single generation", f"{single * 100:.1f}%", "Most creative conversations request one output rather than sustained search.", COLORS["orange"])
    add_card(ax, (0.68, 0.55), (0.28, 0.25), "Wide without local", f"{wide_no_local * 100:.1f}%", "Many users ask for alternatives but do not visibly narrow to one option.", COLORS["green"])
    add_card(ax, (0.04, 0.24), (0.28, 0.25), "Wide -> local", f"{wide_to_local * 100:.1f}%", "Human-calibrated next-turn transition after wide search.", COLORS["purple"])
    add_card(ax, (0.36, 0.24), (0.28, 0.25), "Local persists", f"{local_to_local * 100:.1f}%", "Once users narrow to local refinement, they often stay local.", COLORS["red"])
    add_card(ax, (0.68, 0.24), (0.28, 0.25), "Extended process", f"{extended * 100:.1f}%", "A smaller but important group shows longer iterative creative work.", COLORS["gray"])
    ax.text(0.04, 0.09, "Takeaway: the common pattern is short creative use; richer wide/local iteration is rarer but structurally distinct.", fontsize=13.2, weight="bold", color=COLORS["ink"])
    save(fig, "goal1_finding_summary_board.png")


def main() -> None:
    if not CONV_FILE.exists():
        raise FileNotFoundError(
            f"Missing private conversation features: {CONV_FILE}. "
            "Run goal1 process analysis locally before creating the visual package."
        )
    conv = pd.read_csv(CONV_FILE)
    data = load_public_results()
    make_pca_cluster_projection(conv)
    make_process_taxonomy_flow(data["taxonomy"])
    make_run_length_survival(data["run_dist"])
    make_topic_process_bubble(data["topic_summary"])
    make_manual_transition_story(data["manual_ci"])
    make_finding_summary_board(data)
    print("Created Goal 1 visual package:")
    for name in [
        "goal1_finding_summary_board.png",
        "process_taxonomy_flow_story.png",
        "creative_run_length_survival.png",
        "manual_transition_story_ci.png",
        "pca_user_process_clusters.png",
        "topic_process_bubble_map.png",
    ]:
        print(FIGURES / name)


if __name__ == "__main__":
    main()
