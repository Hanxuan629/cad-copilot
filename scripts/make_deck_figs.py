#!/usr/bin/env python3
"""Generate the three single-panel figures for the report deck.

Each figure makes ONE claim (report-deck house style). English text only (matplotlib
CJK renders as tofu). Reads results/*.jsonl produced by the pipeline.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))   # for `from common import ...`
RES = ROOT / "results"
FIG = RES / "figs"
FIG.mkdir(parents=True, exist_ok=True)
# match dataset for the task-setup example (built locally to /tmp, or in data/)
MATCH_JSONL = "/tmp/mm.jsonl" if Path("/tmp/mm.jsonl").exists() else str(ROOT / "data/match_cad2tgt.jsonl")

# light theme: figures sit in the deck's white raster frame, so use a light bg
DARK = "#ffffff"     # figure background (white, matches .fig-frame)
INK = "#1f2328"      # main text / axes
ACCENT = "#1f6feb"   # perception (blue)
WARN = "#e0913b"     # geometry (orange)
GOOD = "#2da44e"     # good/green
MUTE = "#8b949e"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def fig_stages():
    """Stage 1+2: perception jumps with LoRA, geometry only drops after symbolic refine."""
    stages = ["base\n(stock VLM)", "+ LoRA", "+ symbolic\nrefine"]
    type_acc = [0.002, 0.388, 0.401]      # perception
    chamfer = [2.90, 2.94, 2.48]          # geometry (lower better)
    x = np.arange(len(stages))
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor(DARK); ax1.set_facecolor(DARK)
    b1 = ax1.bar(x - 0.2, type_acc, 0.4, color=ACCENT, label="type accuracy (perception) ↑")
    ax1.set_ylabel("type accuracy (↑ better)")
    ax1.set_ylim(0, 0.5)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, chamfer, 0.4, color=WARN, label="chamfer (geometry) ↓")
    ax2.set_ylabel("chamfer, accepted pairs (↓ better)")
    ax2.set_ylim(0, 3.4)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=INK); ax2.yaxis.label.set_color(INK)
    for s in ("left", "bottom", "right"):
        ax1.spines[s].set_color(MUTE); ax2.spines[s].set_color(MUTE)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(colors=INK); ax1.yaxis.label.set_color(INK)
    ax1.set_xticks(x); ax1.set_xticklabels(stages, color=INK)
    for xi, v in zip(x - 0.2, type_acc):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", color=ACCENT, fontsize=10, weight="bold")
    for xi, v in zip(x + 0.2, chamfer):
        ax2.text(xi, v + 0.05, f"{v:.2f}", ha="center", color=WARN, fontsize=10, weight="bold")
    # annotate the orthogonal moves
    ax1.annotate("perception jumps\n(LoRA)", xy=(1 - 0.2, 0.388), xytext=(0.15, 0.44),
                 color=ACCENT, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax2.annotate("geometry drops\nonly after refine", xy=(2 + 0.2, 2.48), xytext=(1.35, 3.1),
                 color=WARN, fontsize=9, arrowprops=dict(arrowstyle="->", color=WARN))
    # title omitted: the slide kicker states the claim
    lines = [b1, b2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left",
               facecolor=DARK, edgecolor=MUTE, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "stages.png", facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("wrote stages.png")


def fig_matching():
    """Task 2: VLM near-chance at line matching; training-free geometry is perfect."""
    methods = ["geometric\n(= GT reference)", "VLM\ntext prompt", "VLM\nvisual prompt"]
    acc = [1.000, 0.147, 0.112]
    colors = [MUTE, WARN, WARN]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    bars = ax.bar(methods, acc, color=colors, width=0.6)
    _style(ax)
    ax.set_ylabel("top-1 matching accuracy (↑ better)")
    ax.set_ylim(0, 1.1)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", color=INK, fontsize=12, weight="bold")
    ax.axhline(1.0, color=MUTE, ls=":", lw=1, alpha=0.5)
    # title omitted: the slide kicker states the claim
    fig.tight_layout()
    fig.savefig(FIG / "matching.png", facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("wrote matching.png")


def fig_embedding():
    """Task 3: geometric AE embedding tracks geometric similarity; text embedding doesn't."""
    d = json.loads((RES / "embed_compare.json").read_text())
    methods = ["geometric\nautoencoder", "Qwen3-Embedding\n(CAD-code text)", "random\n(control)"]
    keys = ["geometric-AE", "qwen-text", "random"]
    rho = [d[k]["spearman"] for k in keys]
    colors = [GOOD, WARN, MUTE]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    bars = ax.bar(methods, rho, color=colors, width=0.6)
    _style(ax)
    ax.set_ylabel("Spearman ρ  vs geometric distance (↑ better)")
    ax.set_ylim(0, 0.5)
    for b, v in zip(bars, rho):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                ha="center", color=INK, fontsize=12, weight="bold")
    ax.set_title("Text embedding misses geometric similarity; a small geo-AE captures it",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "embedding.png", facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("wrote embedding.png")


def fig_task_setup():
    """Task 2 setup: what line matching IS — pair each drawn curve to a target curve."""
    from common import sample_curve
    # a real, clean trial: 7 CAD curves, all correctly matched (type + direction consistent)
    trial_id = "571f3ec8772f237b58d18794_0_81"
    rows = [json.loads(l) for l in open(MATCH_JSONL)]
    r = next(x for x in rows if x["target_id"] == trial_id)
    cad, tgt, gm = r["cad_curves"], r["target_curves"], r["gt_match"]
    palette = ["#1f6feb", "#e0913b", "#2da44e", "#a371f7", "#e5534b", "#1098ad",
               "#d29922", "#57606a"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(DARK)
    # color each target by its own index; color each CAD curve by the target it maps to
    def draw(ax, curves, colors, title, off):
        ax.set_facecolor(DARK)
        for i, c in enumerate(curves):
            pts = sample_curve(c, 40)
            ax.plot(pts[:, 0] + off, pts[:, 1], color=colors[i], lw=2.2)
        ax.set_xlim(-24 + off, 24 + off); ax.set_ylim(-24, 24)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, color=INK, fontsize=12, weight="bold")
    tgt_colors = [palette[j % len(palette)] for j in range(len(tgt))]
    cad_colors = [palette[gm[i] % len(palette)] if gm[i] != -1 else MUTE
                  for i in range(len(cad))]
    draw(axL, cad, cad_colors, "drawn CAD curves", 0)
    draw(axR, tgt, tgt_colors, "target curves", 0)
    fig.text(0.5, 0.06,
             "match each drawn curve  →  its target   (same colour = a matched pair)",
             ha="center", color=INK, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG / "task_setup.png", facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("wrote task_setup.png")


if __name__ == "__main__":
    fig_task_setup()
    fig_stages()
    fig_matching()
    fig_embedding()
    print("all figures ->", FIG)
