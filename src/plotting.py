"""Figure generators in the presentation colour palette.

Every plot writes to a PNG file (300 DPI) and returns the matplotlib Figure
so the caller can either display or further customise it. The plot style is
applied through :func:`config.apply_matplotlib_style` and the colour palette
is taken from :data:`config.PALETTE`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import PALETTE


def _save(fig: plt.Figure, output_path: Optional[Path]) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def plot_reliability(
    summary: pd.DataFrame, output_path: Optional[Path] = None
) -> plt.Figure:
    """Bar chart of median relative difference per feature with 95% CI."""
    df = summary.sort_values("median_rel_diff")
    x = np.arange(len(df))
    medians = df["median_rel_diff"].values
    yerr = np.vstack([medians - df["ci_low"].values, df["ci_high"].values - medians])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, medians, color=PALETTE["navy"], width=0.65, zorder=2)
    ax.errorbar(
        x, medians, yerr=yerr, fmt="none",
        ecolor=PALETTE["accent"], elinewidth=1.4, capsize=3, zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Median relative difference")
    ax.set_title("Split-half reliability with 95% bootstrap CI")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Iterable[str],
    title: str,
    output_path: Optional[Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Heatmap of a confusion matrix in the navy palette."""
    labels = list(labels)
    if normalize:
        with np.errstate(invalid="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = "{:.2f}"
        vmax = 1.0
    else:
        cm_disp = cm.astype(float)
        fmt = "{:d}"
        vmax = float(cm.max())

    cmap = LinearSegmentedColormap.from_list(
        "navy_white", ["#FFFFFF", PALETTE["accent"], PALETTE["navy"]]
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm_disp, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = vmax / 2.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            value = cm_disp[i, j]
            text = fmt.format(int(cm[i, j]) if not normalize else value)
            color = "white" if value > threshold else PALETTE["text"]
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_feature_importances(
    importances: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top features."""
    df = importances.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1.2))
    ax.barh(df["feature"], df[value_col], color=PALETTE["accent"], zorder=2)
    ax.set_xlabel(value_col.replace("_", " ").capitalize())
    ax.set_title(title)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Configuration sensitivity
# ---------------------------------------------------------------------------

def plot_cfg_sensitivity(
    table: pd.DataFrame, output_path: Optional[Path] = None
) -> plt.Figure:
    """Two-panel bar chart: balanced accuracy and RT detection rate per config."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    x = np.arange(len(table))
    width = 0.65

    axes[0].bar(x, table["balanced_accuracy"], color=PALETTE["navy"], width=width, zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(table["config"], rotation=45, ha="right")
    axes[0].set_ylim(
        max(0.0, table["balanced_accuracy"].min() - 0.05),
        min(1.0, table["balanced_accuracy"].max() + 0.05),
    )
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Balanced accuracy under perturbed configs")
    axes[0].yaxis.grid(True)
    axes[0].set_axisbelow(True)
    for xi, v in zip(x, table["balanced_accuracy"]):
        axes[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=PALETTE["navy"])

    axes[1].bar(x, table["rt_found_rate_median"] * 100, color=PALETTE["accent"], width=width, zorder=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(table["config"], rotation=45, ha="right")
    axes[1].set_ylim(
        max(0.0, table["rt_found_rate_median"].min() * 100 - 2),
        100,
    )
    axes[1].set_ylabel("RT detection rate (%)")
    axes[1].set_title("Median RT detection rate per recording")
    axes[1].yaxis.grid(True)
    axes[1].set_axisbelow(True)
    for xi, v in zip(x, table["rt_found_rate_median"] * 100):
        axes[1].text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=PALETTE["navy"])

    fig.tight_layout()
    _save(fig, output_path)
    return fig
