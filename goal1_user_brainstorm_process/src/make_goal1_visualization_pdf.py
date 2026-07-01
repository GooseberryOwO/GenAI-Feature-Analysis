from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "goal1_visualization.pdf"

FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")


PAGES = [
    {
        "title": "1. Goal 1 总览：用户如何使用 GenAI 进行 brainstorming",
        "image": "goal1_finding_summary_board.png",
        "bullets": [
            "这张图把 Goal 1 的主要结果压缩到一页，用来快速说明整体故事。",
            "核心发现是：常见 creative use 比较短，single generation 和 wide without local 是主要模式。",
            "真正 wide → local 或 extended iterative process 比例较低，但更能代表深入的 brainstorming / design-like interaction。",
        ],
    },
    {
        "title": "2. Creative conversation 的 process taxonomy",
        "image": "process_taxonomy_flow_story.png",
        "bullets": [
            "这张 flow-style 图展示 creative conversations 如何分布到不同 process types。",
            "大多数 conversation 停留在 single generation only 或 wide without local refinement。",
            "wide → local、local → wide、iterative reopening 都比较少见，但这些类型平均 turns 更长。",
        ],
    },
    {
        "title": "3. Creative activity 通常持续多久",
        "image": "creative_run_length_survival.png",
        "bullets": [
            "这张 survival-style 图展示不同 creative states 能持续至少 N 个 user turns 的比例。",
            "所有 creative runs 在第 1 到第 2 turn 之间下降很快，说明 creative activity 通常比较短。",
            "broad exploration 比 strict brainstorming 和 local refinement 更容易持续多轮。",
        ],
    },
    {
        "title": "4. AI 给出多个 genuine options 后，用户下一步怎么走",
        "image": "manual_transition_story_ci.png",
        "bullets": [
            "这张图基于人工校准后的 genuine multiple-option paths，横线表示 bootstrap 95% CI。",
            "initial wide search 后，用户不一定直接进入 local refinement，也常继续 other creative continuation 或结束。",
            "initial local search 后，用户更容易继续 local，说明一旦收窄到某个 option，后续更容易围绕它深化。",
        ],
    },
    {
        "title": "5. 三类 user-process groups 的 PCA projection",
        "image": "pca_user_process_clusters.png",
        "bullets": [
            "这张图回应老师建议的 PCA projection：每个点是一条 creative conversation，不同颜色代表三类 user-process groups。",
            "三类过程在低维空间中呈现可解释的相对位置：brief creativity、creative-session dominant、extended iterative process。",
            "extended iterative process 分布更宽，说明这类 conversation 更长、更复杂，也有更多 creative-state switching。",
        ],
    },
    {
        "title": "6. Topic family 与 brainstorming intensity / switching",
        "image": "topic_process_bubble_map.png",
        "bullets": [
            "这张 bubble chart 展示不同 topic family 的 process 差异。",
            "x 轴是至少包含一个 strict brainstorming turn 的 conversation 比例，y 轴是 mean creative-state switches。",
            "bubble size 表示 creative conversation 数量，颜色表示 mean turns；topic 有影响，但不是唯一解释因素。",
        ],
    },
]


def add_text(ax, x: float, y: float, text: str, size: float, weight: str = "normal", color: str = "#202933") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        color=color,
        fontproperties=BOLD if weight == "bold" else FONT,
        linespacing=1.35,
    )


def make_page(pdf: PdfPages, page: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#F2F6F9")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    add_text(ax, 0.055, 0.94, page["title"], 20, weight="bold")
    y = 0.875
    for bullet in page["bullets"]:
        add_text(ax, 0.075, y, "• " + bullet, 11.7, color="#42515F")
        y -= 0.048

    image_path = FIGURES / page["image"]
    image = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.055, 0.055, 0.89, 0.62])
    img_ax.imshow(image)
    img_ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        for page in PAGES:
            make_page(pdf, page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
