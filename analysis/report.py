#!/usr/bin/env python3
"""Generate report figures from the recorded experiment summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath
import numpy as np

FINAL_DIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = FINAL_DIR / "results" / "experiment_summary.json"
METRICS_DIR = FINAL_DIR / "results" / "cloud_metrics"
SPARSE_TASK_ID = 523939  # abl-sparse-cold30
FIGURES_DIR = FINAL_DIR / "report" / "figures"
PREVIEW_DIR = Path("/tmp/final_fig_preview")

BLUE = "#285D8F"
ORANGE = "#AD5C31"
GREEN = "#35786C"
PURPLE = "#79618F"
CHARCOAL = "#26323D"
GREY = "#697580"
GRID = "#DCE2E7"
BLUE_BG = "#EEF4FA"
GREEN_BG = "#EEF6F3"
FIGURE_WIDTH = 6.7


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Sarasa Gothic SC", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": GREY,
            "axes.linewidth": 0.65,
            "axes.labelsize": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.2,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "hatch.linewidth": 0.45,
            "figure.constrained_layout.h_pad": 0.08,
            "figure.constrained_layout.w_pad": 0.05,
        }
    )


def save_both(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = PREVIEW_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=200)
    plt.close(fig)
    print(f"Saved {pdf_path} and preview {png_path}")


def finish_axis(ax: plt.Axes, title: str, grid_axis: str = "y") -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    facecolor: str = "white",
    edgecolor: str = "#B3BEC8",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
            fontsize=9.2, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center",
            fontsize=8.6, linespacing=1.25, zorder=3)


def draw_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    color: str = GREY,
    dashed: bool = False,
) -> None:
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=9,
            color=color, linewidth=0.9,
            linestyle=(0, (3, 2)) if dashed else "-",
            joinstyle="round", zorder=1,
        )
    )


def plot_system_arch() -> None:
    fig, (ax_net, ax_mask) = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH, 5.0),
        gridspec_kw={"height_ratios": [1.18, 1]}, layout="constrained",
    )
    for ax in (ax_net, ax_mask):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    ax_net.set_title("(a) 实体编码与循环策略–价值网络", loc="left", pad=8)
    ax_mask.set_title("(b) 采样与更新共享条件动作掩码", loc="left", pad=8)

    draw_box(ax_net, 0.015, 0.45, 0.145, 0.25, "实体观测", "英雄 · 小兵\n防御设施")
    draw_box(ax_net, 0.205, 0.45, 0.15, 0.25, "实体编码", "共享编码器")
    draw_box(ax_net, 0.405, 0.45, 0.13, 0.25, "特征融合", "拼接 + 投影")
    draw_box(ax_net, 0.59, 0.45, 0.14, 0.25, "LSTM", "时序记忆", BLUE_BG, BLUE)
    draw_box(ax_net, 0.825, 0.72, 0.16, 0.18, "动作分支", "5 个分类头")
    draw_box(ax_net, 0.825, 0.47, 0.16, 0.18, "状态价值", r"$V_\phi$")
    draw_box(ax_net, 0.825, 0.18, 0.16, 0.18, "目标分支", "点积匹配")

    for start, end in ((0.16, 0.205), (0.355, 0.405), (0.535, 0.59)):
        draw_arrow(ax_net, [(start, 0.575), (end, 0.575)])
    draw_arrow(ax_net, [(0.73, 0.64), (0.775, 0.64), (0.775, 0.81), (0.825, 0.81)])
    draw_arrow(ax_net, [(0.73, 0.56), (0.825, 0.56)])
    draw_arrow(ax_net, [(0.66, 0.45), (0.66, 0.27), (0.825, 0.27)])
    ax_net.text(0.742, 0.30, "查询", ha="center", va="bottom", fontsize=8.5)
    draw_arrow(ax_net, [(0.28, 0.45), (0.28, 0.08), (0.905, 0.08), (0.905, 0.18)])
    ax_net.text(0.48, 0.10, "目标嵌入", ha="center", va="bottom", fontsize=8.5)
    draw_arrow(ax_net, [(0.625, 0.70), (0.625, 0.88), (0.70, 0.88), (0.70, 0.70)], BLUE)
    ax_net.text(0.6625, 0.91, r"$(h,c)$", ha="center", va="bottom", fontsize=9, color=BLUE)

    draw_box(ax_mask, 0.015, 0.56, 0.16, 0.23, "环境输入", "观测 · 合法动作")
    draw_box(ax_mask, 0.23, 0.56, 0.18, 0.23, "战术约束", r"条件掩码 $m_t$", GREEN_BG, GREEN)
    draw_box(ax_mask, 0.50, 0.56, 0.21, 0.23, "掩码策略", r"$\pi_{\mathrm{old}}(a_t\mid z_t,m_t)$")
    draw_box(ax_mask, 0.79, 0.56, 0.19, 0.23, "动作采样", "与环境交互")
    draw_box(ax_mask, 0.50, 0.055, 0.48, 0.225, "训练样本", "观测 · 动作 · 旧概率 · 掩码")
    draw_box(ax_mask, 0.23, 0.055, 0.23, 0.225, "PPO 更新", "新 / 旧策略比率", GREEN_BG, GREEN)
    for start, end in ((0.175, 0.23), (0.41, 0.50), (0.71, 0.79)):
        draw_arrow(ax_mask, [(start, 0.675), (end, 0.675)])
    ax_mask.text(0.605, 0.94, "网络输出", ha="center", va="center", fontsize=9)
    draw_arrow(ax_mask, [(0.605, 0.895), (0.605, 0.79)])
    draw_arrow(ax_mask, [(0.885, 0.56), (0.885, 0.28)])
    draw_arrow(ax_mask, [(0.32, 0.56), (0.32, 0.405), (0.66, 0.405), (0.66, 0.28)], GREEN, dashed=True)
    ax_mask.text(0.50, 0.425, "随样本保存", ha="center", va="bottom", fontsize=8.5, color=GREEN)
    draw_arrow(ax_mask, [(0.50, 0.1675), (0.46, 0.1675)], GREEN)

    save_both(fig, "system_arch")


ABLATION_RUNS = (  # (task id, legend label, color, line style)
    (472732, "基线", GREY, ":"),
    (523716, "仅奖励", ORANGE, "-"),
    (SPARSE_TASK_ID, "仅掩码", PURPLE, "-"),
    (523898, "完整", GREEN, "-"),
    (474124, "完整（旧规则）", BLUE, "--"),
)


def load_run(task_id: int) -> dict:
    return json.loads((METRICS_DIR / f"{task_id}.json").read_text(encoding="utf-8"))


def series_minutes(run: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Monitor series as (minutes since the first monitor point, value)."""
    pts = np.array(run["series"][key], dtype=float)
    t0 = np.array(run["series"]["train_global_step"], dtype=float)[0, 0]
    return (pts[:, 0] - t0) / 60.0, pts[:, 1]


def plot_cloud_ablation() -> None:
    runs = [(load_run(tid), label, color, ls) for tid, label, color, ls in ABLATION_RUNS]
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.6), layout="constrained")
    panels = (
        (axes[0, 0], "win_rate_common_ai", "(a) 对规则对手胜率", "胜率", (-0.04, 1.04)),
        (axes[0, 1], "enemy_tower_hp_common_ai", "(b) 敌方外塔余血", "外塔生命值 / HP", (-400, 12600)),
        (axes[1, 0], "self_tower_hp_common_ai", "(c) 己方外塔余血", "外塔生命值 / HP", (-400, 12600)),
        (axes[1, 1], "hurt_to_hero_common_ai", "(d) 对敌方英雄伤害", "伤害（监控值）", None),
    )
    for ax, key, title, ylabel, ylim in panels:
        for run, label, color, ls in runs:
            x, y = series_minutes(run, key)
            # 基线置顶，避免被与它重合的“仅奖励”曲线盖住
            ax.plot(x, y, color=color, linestyle=ls, label=label,
                    zorder=3 if label == "基线" else 2)
        ax.set(xlim=(-1, 34), xlabel="训练时长 / min", ylabel=ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        finish_axis(ax, title)
    axes[0, 1].legend(loc="center right", labelspacing=0.3)
    save_both(fig, "cloud_ablation")


def plot_head_to_head(data: dict) -> None:
    ev = data["official_eval_842359"]
    matches = ev["matches"]
    fig = plt.figure(figsize=(FIGURE_WIDTH, 7.1), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2])
    ax_dmg = fig.add_subplot(gs[0, 0])
    ax_econ = fig.add_subplot(gs[1, 0], sharex=ax_dmg)
    ax_ratio = fig.add_subplot(gs[2, 0])

    x = np.arange(len(matches))
    w = 0.34
    dmg_a = [m["a"]["damage_to_heroes"] / 1000.0 for m in matches]
    dmg_b = [m["b"]["damage_to_heroes"] / 1000.0 for m in matches]
    eco_a = [m["a"]["economy"] for m in matches]
    eco_b = [m["b"]["economy"] for m in matches]
    ax_dmg.bar(x - w / 2, dmg_a, width=w, color=BLUE, label="A：续训终点")
    ax_dmg.bar(x + w / 2, dmg_b, width=w, color="white", edgecolor=ORANGE,
               linewidth=0.8, hatch="///", label="B：续训起点")
    for idx, m in enumerate(matches):
        winner = "A 胜" if m["result"] == "A wins" else "B 胜"
        ax_dmg.text(idx, max(dmg_a[idx], dmg_b[idx]) + 1.5,
                    f"{winner}\n{m['a']['kills']}/{m['a']['deaths']} : {m['b']['kills']}/{m['b']['deaths']}",
                    ha="center", va="bottom", fontsize=8.5)
    ax_dmg.set(ylim=(0, 62), ylabel="英雄伤害 / 千点")
    ax_dmg.legend(loc="upper right", ncol=2)
    finish_axis(ax_dmg, "(a) 逐局英雄伤害与胜负")
    plt.setp(ax_dmg.get_xticklabels(), visible=False)

    ax_econ.bar(x - w / 2, eco_a, width=w, color=BLUE)
    ax_econ.bar(x + w / 2, eco_b, width=w, color="white", edgecolor=ORANGE, linewidth=0.8, hatch="///")
    for idx, (ea, eb) in enumerate(zip(eco_a, eco_b)):
        ax_econ.text(idx - w / 2, ea + 150, f"{ea}", ha="center", va="bottom", fontsize=8.5)
        ax_econ.text(idx + w / 2, eb + 150, f"{eb}", ha="center", va="bottom", fontsize=8.5)
    x_labels = [
        f"第 {idx + 1} 局\n{'A红' if m['a_side'] == 'Red' else 'A蓝'} · {m['frames']}帧"
        for idx, m in enumerate(matches)
    ]
    ax_econ.set_xticks(x, x_labels)
    ax_econ.set(ylim=(0, 10800), ylabel="经济 / 金币", xlabel="对局序号、A 的阵营与局长")
    finish_axis(ax_econ, "(b) 逐局经济")

    avg_a = ev["average_data"]["camp_a"]
    avg_b = ev["average_data"]["camp_b"]
    metrics = [
        ("经济", avg_a["economy"] / avg_b["economy"], f"{avg_a['economy']:.0f} / {avg_b['economy']:.0f}"),
        ("经验", avg_a["experience"] / avg_b["experience"], f"{avg_a['experience']:.0f} / {avg_b['experience']:.0f}"),
        ("死亡（反向）", avg_b["deaths"] / avg_a["deaths"], f"{avg_a['deaths']:.2f} / {avg_b['deaths']:.2f}"),
        ("击杀", avg_a["kills"] / avg_b["kills"], f"{avg_a['kills']:.2f} / {avg_b['kills']:.2f}"),
        ("胜率", 4.0 / 2.0, "66.7% / 33.3%"),
        ("KDA", avg_a["kda"] / avg_b["kda"], f"{avg_a['kda']:.2f} / {avg_b['kda']:.2f}"),
    ]
    y = np.arange(len(metrics))
    vals = [m[1] for m in metrics]
    bars = ax_ratio.barh(y, vals, height=0.54, color=BLUE)
    ax_ratio.axvline(1.0, color=GREY, linestyle="--", linewidth=0.9)
    for rect, val, (_, _, desc) in zip(bars, vals, metrics):
        ax_ratio.text(val + 0.05, rect.get_y() + rect.get_height() / 2,
                      f"{val:.2f}×   ({desc})", va="center", ha="left", fontsize=8.5)
    ax_ratio.set_yticks(y, [m[0] for m in metrics])
    ax_ratio.set(xlim=(0, 3.20), xlabel="A 相对 B 的倍数（虚线 = 1）")
    finish_axis(ax_ratio, "(c) 六局场均指标比较", grid_axis="x")
    save_both(fig, "head_to_head")


def main() -> None:
    configure_style()
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    plot_system_arch()
    plot_cloud_ablation()
    plot_head_to_head(data)


if __name__ == "__main__":
    main()
