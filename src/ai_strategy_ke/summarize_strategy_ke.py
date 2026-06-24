import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get("GENAI_ANALYSIS_ROOT", Path.cwd()))
RESULTS = ROOT / "02_results"
FIGURES = ROOT / "03_figures"
REPORTS = ROOT / "04_reports"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

FILES = {
    "Broad creativity": RESULTS / "broad_creativity" / "dtr_strategy_effects_broad_creativity.csv",
    "Strict brainstorming": RESULTS / "strict_brainstorming" / "dtr_strategy_effects_strict_brainstorming.csv",
}

LABELS = {
    "motif_late_synthesis": "后期综合整理",
    "max_synthesis_late": "曾在后期进行综合",
    "max_assistant_question": "曾主动提问",
    "motif_clarify_then_answer": "先澄清再回答",
    "max_context_uptake": "吸收用户上下文",
    "max_contrastive_explanation": "对比不同方案/概念",
    "max_selection_criteria": "提供选择标准",
    "max_convergent_synthesis": "收敛并综合方案",
    "max_helpful_clarification": "有帮助的澄清",
    "mean_helpful_clarification": "持续进行有帮助的澄清",
    "max_explanation": "提供解释",
    "max_example": "提供例子",
    "max_support": "提供支持性回应",
    "max_option_generation": "生成多个选项",
    "max_divergent_ideas": "提出差异化想法",
    "max_knowledge_bridge": "解释可迁移知识",
    "max_constraint_aware_brainstorming": "遵守约束的头脑风暴",
    "mean_direct_answer": "持续直接给答案",
    "mean_procedure": "持续给操作步骤",
    "mean_option_generation": "持续生成选项",
    "mean_divergent_ideas": "持续发散想法",
}

FOCAL = [
    "motif_late_synthesis",
    "max_assistant_question",
    "motif_clarify_then_answer",
    "max_context_uptake",
    "max_contrastive_explanation",
    "max_selection_criteria",
    "max_convergent_synthesis",
    "max_option_generation",
    "max_divergent_ideas",
    "max_knowledge_bridge",
    "max_constraint_aware_brainstorming",
    "mean_direct_answer",
    "mean_procedure",
    "mean_option_generation",
    "mean_divergent_ideas",
]


def load_results():
    frames = []
    for sample, path in FILES.items():
        df = pd.read_csv(path)
        df = df[(df["target"] == "KE") & (df["feature"].isin(FOCAL))].copy()
        df["sample"] = sample
        df["中文策略"] = df["feature"].map(LABELS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_plot(df):
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False

    selected = [
        "motif_late_synthesis",
        "max_assistant_question",
        "motif_clarify_then_answer",
        "max_context_uptake",
        "max_contrastive_explanation",
        "max_selection_criteria",
        "max_convergent_synthesis",
        "max_option_generation",
        "max_divergent_ideas",
        "max_knowledge_bridge",
        "max_constraint_aware_brainstorming",
    ]
    plot_df = df[df["feature"].isin(selected)].copy()
    pivot = plot_df.pivot(index="feature", columns="sample", values="dtr_adjusted_corr")
    pivot["order"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("order")
    labels = [LABELS[x] for x in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    fig.patch.set_facecolor("#F4F7FA")
    ax.set_facecolor("#F4F7FA")
    y = range(len(pivot))
    height = 0.36
    ax.barh(
        [v - height / 2 for v in y],
        pivot["Broad creativity"],
        height=height,
        color="#2F6B9E",
        label="Broad creativity conversations",
    )
    ax.barh(
        [v + height / 2 for v in y],
        pivot["Strict brainstorming"],
        height=height,
        color="#E5832A",
        label="Strict brainstorming conversations",
    )
    ax.axvline(0, color="#4B5560", linewidth=0.8)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("与 KE 的 residualized adjusted correlation")
    fig.suptitle(
        "哪些 AI 回答策略与更高的 Knowledge Expansion 关系最大？",
        fontsize=18,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.125,
        0.925,
        "控制 conversation length、topic、searchability 与用户状态；数值用于排序关联，不代表 causal effect",
        fontsize=10.5,
        color="#596776",
    )
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIGURES / "AI_strategy与KE关联_broad_vs_strict.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def make_tables(df):
    columns = [
        "sample",
        "feature",
        "中文策略",
        "phase",
        "dtr_adjusted_corr",
        "split_a_corr",
        "split_b_corr",
        "stable_sign",
        "n",
    ]
    detailed = df[columns].sort_values(["sample", "dtr_adjusted_corr"], ascending=[True, False])
    detailed.to_csv(RESULTS / "creativity_strategy_KE_comparison.csv", index=False)

    positive = (
        detailed[
            detailed["phase"].isin(["ever", "motif"])
            & detailed["stable_sign"].eq(1)
            & detailed["dtr_adjusted_corr"].gt(0)
        ]
        .sort_values(["sample", "dtr_adjusted_corr"], ascending=[True, False])
        .groupby("sample", as_index=False)
        .head(10)
    )
    positive.to_csv(RESULTS / "top_interpretable_strategy_KE_associations.csv", index=False)
    return detailed, positive


def make_report(df, positive, figure):
    broad = df[df["sample"] == "Broad creativity"].set_index("feature")
    strict = df[df["sample"] == "Strict brainstorming"].set_index("feature")

    lines = [
        "# AI回答策略与Knowledge Expansion：Creativity子样本分析",
        "",
        "## 做了什么",
        "",
        "- 将 Yixin 的 `multi-turn-analysis` 输出与我们的四分类 creativity labels 按 `conversation_hash` 一对一合并。",
        "- 56,942 个 conversations 全部成功匹配；其中 broad creativity 36,018 个，strict brainstorming 12,566 个。",
        "- 分别在两组样本中重新运行原仓库的 DTR-style residualized association screen。",
        "- Outcome 为 conversation-level `KE = ke_overall_ke_mean`。",
        "- 模型控制 conversation length、topic buckets、searchability、用户问题状态及其他 state features，并加入同一策略的后期 summaries。",
        "",
        "## 主要结果",
        "",
        "在 strict brainstorming conversations 中，与 KE 正向关系最强的可解释策略包括：",
        "",
    ]
    for row in positive[positive["sample"] == "Strict brainstorming"].itertuples():
        lines.append(f"- {row.中文策略} (`{row.feature}`): adjusted r = {row.dtr_adjusted_corr:+.3f}")

    lines.extend(
        [
            "",
            "最稳定的整体模式不是“单纯提供更多 options”，而是 **发散之后进行组织、澄清和收敛**：",
            "",
            f"- 后期 synthesis 与 KE 的关系最强：strict r={strict.loc['motif_late_synthesis','dtr_adjusted_corr']:+.3f}，broad r={broad.loc['motif_late_synthesis','dtr_adjusted_corr']:+.3f}。",
            f"- context uptake 在 strict 样本中更重要：strict r={strict.loc['max_context_uptake','dtr_adjusted_corr']:+.3f}，broad r={broad.loc['max_context_uptake','dtr_adjusted_corr']:+.3f}。",
            f"- selection criteria 与 convergent synthesis 均为正：strict r={strict.loc['max_selection_criteria','dtr_adjusted_corr']:+.3f} 和 {strict.loc['max_convergent_synthesis','dtr_adjusted_corr']:+.3f}。",
            f"- option generation 和 divergent ideas 在“至少出现过一次”时为正：strict r={strict.loc['max_option_generation','dtr_adjusted_corr']:+.3f} 和 {strict.loc['max_divergent_ideas','dtr_adjusted_corr']:+.3f}。",
            f"- 但持续高比例使用 option generation/divergent ideas 时为负：strict mean r={strict.loc['mean_option_generation','dtr_adjusted_corr']:+.3f} 和 {strict.loc['mean_divergent_ideas','dtr_adjusted_corr']:+.3f}。",
            "",
            "这支持一个值得继续验证的 process hypothesis：AI 可以先扩大 option space，但更高的 user knowledge diversity 可能来自随后对方案进行比较、解释、组织和收敛，而不是在每一轮持续增加列表。",
            "",
            "## Latent strategy补充",
            "",
            "- Strict 样本中最强 latent dimensions 为 `max_z06`（r=+0.095）和 `max_z11`（r=+0.091）。",
            "- 在选定的 z16 profile 中，z06 更接近 compact direct answer/context uptake，而 z11 更接近 direct answer + example + contrastive explanation。",
            "- Latent labels 是 post-hoc interpretation，应作为补充证据，主要结论仍优先基于可解释 action features。",
            "",
            "## 解释限制",
            "",
            "- `dtr_adjusted_corr` 是 residualized standardized association，不是 randomized treatment effect。",
            "- `first_*` 表示策略首次出现的相对位置，不是策略强度，因此没有作为主要结论。",
            "- `max_*` 表示对话中至少有一轮较明显出现该策略；`mean_*` 表示整个对话中的平均使用程度，两者符号不同可能反映 dosage/timing。",
            "- 当前 creativity labels 属于 first-pass rubric labels，后续应在人工 gold labels 上复核。",
            "",
            f"主要图：`{figure}`",
        ]
    )
    out = REPORTS / "AI_strategy_KE_creativity_analysis_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    df = load_results()
    detailed, positive = make_tables(df)
    figure = make_plot(df)
    report = make_report(detailed, positive, figure)
    print(RESULTS / "creativity_strategy_KE_comparison.csv")
    print(RESULTS / "top_interpretable_strategy_KE_associations.csv")
    print(figure)
    print(report)


if __name__ == "__main__":
    main()
