from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
FIGURES = ROOT / "03_figures"
for folder in (RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

FLOW = DATA / "human_calibrated_multiturn_flow_final.csv"
STATE_NAMES = {
    "wide_search": "Wide search",
    "local_search": "Local search",
    "creative_continuation_other": "Other creative continuation",
    "non_creative_shift": "Non-creative shift",
    "conversation_end": "Conversation ends",
}
STATE_ORDER = [
    "wide_search",
    "local_search",
    "creative_continuation_other",
    "non_creative_shift",
    "conversation_end",
]
COLORS = {
    "wide_search": "#2F6B9E",
    "local_search": "#E5832A",
    "creative_continuation_other": "#7B68A6",
    "non_creative_shift": "#459D8B",
    "conversation_end": "#8B949E",
}

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def weighted_share(frame: pd.DataFrame, source: str, target: str) -> float:
    scoped = frame[frame["state_t1"].eq(source)]
    weights = scoped["population_weight"].to_numpy()
    return float(np.average(scoped["human_state_t2"].eq(target), weights=weights))


def stratified_bootstrap(flow: pd.DataFrame, repetitions: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260624)
    sources = ["wide_search", "local_search"]
    estimates = []
    bootstrap_values = {}

    for source in sources:
        for target in STATE_ORDER:
            key = (source, target)
            estimate = weighted_share(flow, source, target)
            values = []
            group = flow[flow["state_t1"].eq(source)].reset_index(drop=True)
            for _ in range(repetitions):
                sample = group.iloc[rng.integers(0, len(group), len(group))]
                values.append(weighted_share(sample, source, target))
            bootstrap_values[key] = np.asarray(values)
            estimates.append(
                {
                    "source_state": source,
                    "target_state": target,
                    "estimate": estimate,
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "manual_cases": len(group),
                    "bootstrap_repetitions": repetitions,
                }
            )

    output = pd.DataFrame(estimates)
    output.to_csv(RESULTS / "manual_calibrated_t1_to_t2_with_ci.csv", index=False)
    return output


def plot_transition_estimates(estimates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=190, sharey=True)
    for ax, source in zip(axes, ["wide_search", "local_search"]):
        scoped = estimates[estimates["source_state"].eq(source)].copy()
        scoped["order"] = scoped["target_state"].map({name: i for i, name in enumerate(STATE_ORDER)})
        scoped = scoped.sort_values("order")
        x = np.arange(len(scoped))
        y = scoped["estimate"].to_numpy() * 100
        low = scoped["ci95_low"].to_numpy() * 100
        high = scoped["ci95_high"].to_numpy() * 100
        colors = [COLORS[state] for state in scoped["target_state"]]
        ax.bar(x, y, color=colors, width=0.68)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="none",
            ecolor="#26323F",
            capsize=4,
            linewidth=1.1,
        )
        for xpos, value in zip(x, y):
            ax.text(xpos, value + 1.4, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(
            x,
            [STATE_NAMES[state].replace(" ", "\n") for state in scoped["target_state"]],
            fontsize=8.5,
        )
        ax.set_title(f"Next move after initial {STATE_NAMES[source].lower()}", weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Population-weighted share (%)")
    fig.suptitle(
        "Human-calibrated next-turn behavior after genuine multiple-option responses",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Bars show weighted estimates; error bars show stratified bootstrap 95% confidence intervals.",
        ha="center",
        color="#5F6B78",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIGURES / "manual_calibrated_next_turn_transitions.png", bbox_inches="tight")
    plt.close(fig)


def plot_rule_agreement() -> None:
    agreement = pd.read_csv(RESULTS / "later_turn_rule_agreement.csv")
    labels = agreement["reviewed_stage"].map({"t2": "Next user turn", "t3": "Following user turn"})
    values = agreement["agreement_rate"] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=190)
    bars = ax.bar(labels, values, color=["#2F6B9E", "#459D8B"], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Agreement with human semantic coding (%)")
    ax.set_title("Rule labels require semantic calibration", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "rule_vs_human_agreement.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flow = pd.read_csv(FLOW)
    estimates = stratified_bootstrap(flow)
    plot_transition_estimates(estimates)
    plot_rule_agreement()
    print(estimates.to_string(index=False))


if __name__ == "__main__":
    main()
