from __future__ import annotations

import textwrap
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
        "points": [
            "本页概括 Goal 1 的主要结果，展示 creative use 的常见长度、主要过程类型，以及 wide/local 转换的大致比例。",
            "Creative runs 平均 1.86 turns，中位数为 1 turn，说明大多数 creative activity 并不会持续很多轮。",
            "Single generation 和 wide without local 是最常见的过程；wide → local 与 extended process 占比较低，但更接近持续探索和深化的 brainstorming process。",
        ],
    },
    {
        "title": "2. Creative conversations 的过程类型分布",
        "image": "process_taxonomy_flow_story.png",
        "points": [
            "本页展示 creative conversations 在不同 process taxonomy 中的分布。",
            "Single generation only：用户主要要求一个 creative output，没有明显进入多方案探索或后续深化。",
            "Wide without local refinement：用户要求多个 alternatives，但没有明显选择其中一个继续深化。Local without wide exploration：用户直接围绕已有方向修改、选择或细化。",
            "Wide → local / local → wide / iterative reopening 表示用户在发散搜索和局部深化之间发生转换；这些模式较少见，但通常 conversation 更长。",
        ],
    },
    {
        "title": "3. Creative activity 的连续长度",
        "image": "creative_run_length_survival.png",
        "points": [
            "本页展示不同 creative states 能持续至少 N 个 user turns 的比例。",
            "从第 1 到第 2 turn 的下降最明显，说明多数 creative runs 很快结束。",
            "Broad exploration 相比 strict brainstorming 和 local refinement 更容易持续多轮，但整体持续长度仍然偏短。",
        ],
    },
    {
        "title": "4. AI 给出多个 genuine options 后的下一步用户行为",
        "image": "manual_transition_story_ci.png",
        "points": [
            "本页基于人工校准后的 genuine multiple-option paths，展示用户下一轮如何继续；横线表示 bootstrap 95% CI。",
            "Wide 表示继续要求更多或更广的 alternatives；Local 表示围绕某一个 option 选择、比较、修改或深化。",
            "Initial wide search 后，用户并不总是直接进入 local refinement，也可能继续其他 creative subtask 或结束。Initial local search 后，用户更容易继续 local。",
        ],
    },
    {
        "title": "5. 三类 user-process groups 的 PCA projection",
        "image": "pca_user_process_clusters.png",
        "points": [
            "本页用 conversation-level process features 做 PCA projection，每个点代表一条 creative conversation，颜色代表三类 user-process groups。",
            "Brief creativity within broader tasks：creative activity 只是 broader task 中的一小段。Creative-session dominant：conversation 主要围绕 creative output 展开。",
            "Extended iterative process：conversation 更长、creative runs 更多、state switching 更多，代表更复杂的反复探索过程。",
        ],
    },
    {
        "title": "6. Topic family 与 brainstorming intensity / switching",
        "image": "topic_process_bubble_map.png",
        "points": [
            "本页展示不同 topic family 与 brainstorming intensity 和 creative-state switching 的关系。",
            "x 轴表示至少包含一个 strict brainstorming turn 的 conversation 比例；y 轴表示平均 creative-state switches。",
            "Bubble size 表示 creative conversation 数量，颜色表示平均 turns。Topic family 与过程差异有关，但不能单独解释用户如何 brainstorming。",
        ],
    },
]


def draw_text(ax, x: float, y: float, text: str, size: float, bold: bool = False, color: str = "#202933") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        color=color,
        fontproperties=BOLD if bold else FONT,
        linespacing=1.28,
    )


def draw_wrapped_point(ax, x: float, y: float, text: str, width: int = 82) -> float:
    wrapped = textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
    draw_text(ax, x, y, "• " + wrapped.replace("\n", "\n  "), 10.6, color="#42515F")
    line_count = wrapped.count("\n") + 1
    return y - 0.037 * line_count - 0.014


def make_page(pdf: PdfPages, page: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#F2F6F9")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    draw_text(ax, 0.055, 0.945, page["title"], 18.5, bold=True)
    y = 0.885
    for point in page["points"]:
        y = draw_wrapped_point(ax, 0.072, y, point)

    image_top = min(0.64, y - 0.025)
    image_bottom = 0.045
    image_height = max(0.50, image_top - image_bottom)

    image = mpimg.imread(FIGURES / page["image"])
    img_ax = fig.add_axes([0.055, image_bottom, 0.89, image_height])
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
