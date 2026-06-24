import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(os.environ.get("GENAI_ANALYSIS_ROOT", Path.cwd()))
FIGURES = ROOT / "03_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

INK = "#161A1D"
MUTED = "#5C6874"
UNIT_BLUE = "#276496"
UNIT_GREEN = "#3F8E7D"
UNIT_ORANGE = "#D97822"


def add_box(ax, y, height, unit, title, lines, unit_color):
    x, width = 0.12, 0.76
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor="white", edgecolor="black", linewidth=1.55,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018, y + height - 0.012, unit,
        ha="left", va="top", fontsize=9.2, fontweight="bold", color=unit_color,
    )
    ax.text(
        0.5, y + height * 0.54, title,
        ha="center", va="center", fontsize=12.8, fontweight="bold", color=INK,
    )
    ax.text(
        0.5, y + height * 0.20, lines,
        ha="center", va="center", fontsize=10.0, color=INK, linespacing=1.22,
    )


def add_arrow(ax, y_top, y_bottom, label):
    ax.add_patch(FancyArrowPatch(
        (0.5, y_top), (0.5, y_bottom),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.35, color="black",
    ))
    ax.text(
        0.515, (y_top + y_bottom) / 2, label,
        ha="left", va="center", fontsize=8.8, color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    )


def main():
    fig, ax = plt.subplots(figsize=(10.5, 16), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.982, "头脑风暴分析样本与分析单位构建流程",
        ha="center", va="top", fontsize=22, fontweight="bold", color=INK,
    )
    ax.text(
        0.5, 0.951,
        "先在轮次层面分类，再汇总识别创意相关对话，最后回到轮次层面分析搜索路径",
        ha="center", va="top", fontsize=11.3, color=MUTED,
    )

    boxes = [
        (0.856, 0.073, "分析单位：对话（conversation）", "原始 Human-AI 对话数据",
         "56,942 个对话；共包含 262,718 个用户轮次", UNIT_GREEN),
        (0.733, 0.094, "分析单位：用户轮次（turn）", "四分类创意活动模型",
         "非创意 172,054｜单一创意生成 64,194｜严格头脑风暴 20,698｜修改/选择 5,772\n"
         "5-fold CV accuracy = 83.2%；holdout accuracy = 81.5%", UNIT_BLUE),
        (0.606, 0.098, "分析单位：对话（由 turn 标签汇总）", "识别创意相关对话",
         "若一个对话至少包含 1 个创意类 turn，则定义为 creative conversation\n"
         "广义创意相关：36,018 个对话（63.25%）；含严格头脑风暴：12,566 个（22.07%）", UNIT_GREEN),
        (0.472, 0.105, "分析单位：创意相关对话中的当前 turn", "抽取可观察后续行为的候选轮次",
         "当前用户 turn 属于广义创意类；AI 回答含至少 2 个编号/项目符号条目；\n"
         "并且存在下一用户 turn；候选 n = 27,803", UNIT_BLUE),
        (0.357, 0.086, "分析单位：当前 AI 回答", "严格筛选真实多方案回答",
         "要求 AI 明确提供多个 ideas/options/alternatives；排除步骤、事实清单和计算列表\n"
         "规则筛选 n = 7,564", UNIT_ORANGE),
        (0.231, 0.097, "分析单位：当前 AI 回答（分层人工复核）", "人工校准多方案回答",
         "人工复核 n = 160；确认真实多方案 n = 70；总体加权 precision = 40.4%\n"
         "估计真实多方案回答约 n = 3,059", UNIT_ORANGE),
        (0.091, 0.109, "分析单位：相邻 turn 转换及后续路径", "对下一用户行为进行分类并继续追踪",
         "下一轮：局部搜索 45.7%｜广泛搜索 38.1%｜退出路径 16.2%\n"
         "后续第2轮：广泛 29.7%｜局部 8.5%｜退出 61.8%；第3轮：广泛 24.4%｜局部 7.0%｜退出 68.6%", UNIT_BLUE),
    ]
    for args in boxes:
        add_box(ax, *args)

    arrows = [
        (0.856, 0.827, "拆分每个对话中的用户轮次"),
        (0.733, 0.704, "将 turn 预测标签汇总到 conversation"),
        (0.606, 0.577, "仅在 creative conversations 内选择候选 turn"),
        (0.472, 0.443, "筛除并非真实备选方案的列表"),
        (0.357, 0.328, "用分层人工样本校正规则误差"),
        (0.231, 0.200, "以真实多方案回答为起点观察下一轮"),
    ]
    for args in arrows:
        add_arrow(ax, *args)

    ax.text(
        0.5, 0.037,
        "关键区别：四分类模型预测的是每个 user turn；creative conversation 是由同一对话内的 turn 标签汇总定义，\n"
        "并不是另一个 conversation-level classifier。wide/local 则是对筛选后的相邻 turns 进行路径编码。",
        ha="center", va="center", fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF3F7", edgecolor="#AEB9C4"),
    )

    png = FIGURES / "分析单位与样本构建_workflow_中文.png"
    pdf = FIGURES / "分析单位与样本构建_workflow_中文.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
