"""Rebuild publication figures from the supplied verified measurements.

Requires matplotlib and numpy. Run: python plot_figures.py
The source JSON contains only the two reported text conditions.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data/evaluation_extract.json").read_text())
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
MODES = ("without_moe", "with_moe")
LABELS = ("Without MoE", "Cross-encoder + MoE")
COLORS = ("#178779", "#CF675B")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "figure.facecolor": "white",
})

def row(mode, k, ds="all", modality="all"):
    return next(r for r in DATA["summary"] if r["mode"] == mode and r["k"] == k
                and r["target_dataset"] == ds and r["modality"] == modality)

def save(fig, name):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

def style(ax, maximum=100):
    ax.set_ylim(0, maximum)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E8ED", linewidth=0.7)
    ax.tick_params(axis="both", length=0)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
for ax, k in zip(axes, (3, 6)):
    x = np.arange(2)
    for i, mode in enumerate(MODES):
        bars = ax.bar(x + (i - 0.5) * 0.34,
                      [100 * row(mode, k)[metric] for metric in ("exact_match", "answer_f1")],
                      width=0.32, color=COLORS[i], label=LABELS[i])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, ["Exact Match", "Answer F1"])
    ax.set_title(f"Top {k} evidence passages", pad=10)
    style(ax)
axes[0].set_ylabel("Score (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.subplots_adjust(top=0.78, bottom=0.13, wspace=0.19)
save(fig, "01_answer_quality")

fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.1))
for ax, metric, title in zip(axes, ("hit_rate", "mrr", "ndcg"), ("Evidence Hit", "MRR", "nDCG")):
    x = np.arange(2)
    scale = 100 if metric == "hit_rate" else 1
    for i, mode in enumerate(MODES):
        bars = ax.bar(x + (i - 0.5) * 0.36,
                      [scale * row(mode, k)[metric] for k in (3, 6)],
                      width=0.33, color=COLORS[i], label=LABELS[i])
        # Vertical labels fit inside the narrow bars without overlap.
        ax.bar_label(bars, fmt="%.0f" if metric == "hit_rate" else "%.4f",
                     label_type="center", fontsize=8, rotation=90, color="white")
    ax.set_xticks(x, ["Top 3", "Top 6"])
    ax.set_title(title, pad=15)
    style(ax, 100 if metric == "hit_rate" else 1)
    ax.margins(x=0.18)
axes[0].set_ylabel("Hit (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.09))
fig.subplots_adjust(top=0.72, bottom=0.12, wspace=0.38)
save(fig, "02_evidence_ranking")

fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4), gridspec_kw={"width_ratios": [1.9, 1]})
datasets = ("DOSA", "SCALE", "DRISHTIKON", "DIWALI", "Indian Cultural Dataset")
for ax, targets in ((axes[0], datasets), (axes[1], ("text_only", "image_associated"))):
    y = np.arange(len(targets))
    for i, mode in enumerate(MODES):
        vals = [100 * row(mode, 3, ds=t)["answer_f1"] if ax is axes[0]
                else 100 * row(mode, 3, modality=t)["answer_f1"] for t in targets]
        bars = ax.barh(y + (i - 0.5) * 0.36, vals, height=0.33,
                       color=COLORS[i], label=LABELS[i])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
    ax.set_yticks(y, ["DOSA", "SCALE", "DRISHTIKON", "DIWALI", "ICD"] if ax is axes[0]
                  else ["Text-only\n(n = 80)", "Image-associated\n(n = 20)"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Answer F1 (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E8ED", linewidth=0.7)
    ax.tick_params(length=0)
axes[0].set_title("Target dataset (n = 20 each)", pad=12)
axes[1].set_title("Target source modality", pad=12)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.13, right=0.97, wspace=0.72)
save(fig, "03_subgroup_quality")
print(f"Saved three figures in PDF, PNG and SVG to {FIGURES}")
