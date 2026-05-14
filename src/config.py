"""Pipeline configuration and visual style.

All numeric thresholds that govern the processing pipeline are defined here.
Modifying a parameter in this module is the only supported way to change
pipeline behaviour. The default values match those reported in the paper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Default processing configuration
# ---------------------------------------------------------------------------

CFG_DEFAULT: Dict[str, Any] = {
    # Reaction-time detection (gaze-based)
    "thr_px_ratio": 0.05,                # fraction of frame width if known
    "thr_px_fallback": 60,               # fallback absolute threshold (px)
    "hold_ms_reaction": 40,              # sustained deviation duration (ms)
    "search_window_reaction": (0.08, 0.70),  # seconds post-onset

    # Target-acquisition detection
    "hold_ms_target": 80,                # sustained in-sector duration (ms)
    "search_window_target": (0.0, 1.5),  # seconds post-onset

    # Short-latency saccade label
    "express_thr_ms": 120,

    # Large-jump proxy
    "jump_thr_px_ratio": 0.03,
    "jump_thr_px_fallback": 40,

    # Recording-level quality control
    "qc_worn_min": 0.80,
    "qc_gaze_rate_min_hz": 50,
    "qc_imu_gyro_rms_max": 50.0,
}


def make_cfg(thr_factor: float = 1.0, hold_factor: float = 1.0) -> Dict[str, Any]:
    """Return a configuration with RT threshold and hold time scaled.

    Parameters
    ----------
    thr_factor
        Multiplicative factor applied to ``thr_px_ratio``.
    hold_factor
        Multiplicative factor applied to ``hold_ms_reaction``.

    Returns
    -------
    dict
        A new configuration dictionary. The default values are returned when
        both factors equal 1.
    """
    cfg = deepcopy(CFG_DEFAULT)
    cfg["thr_px_ratio"] = CFG_DEFAULT["thr_px_ratio"] * thr_factor
    cfg["hold_ms_reaction"] = CFG_DEFAULT["hold_ms_reaction"] * hold_factor
    return cfg


# Configurations used in the ablation study reported in the paper.
SENSITIVITY_CONFIGS = [
    ("CFG_default",      make_cfg(1.0, 1.0)),
    ("CFG_x0.2",         make_cfg(0.2, 0.2)),
    ("CFG_x0.8",         make_cfg(0.8, 0.8)),
    ("CFG_x1.2",         make_cfg(1.2, 1.2)),
    ("CFG_x2.0",         make_cfg(2.0, 2.0)),
    ("CFG_x3.0",         make_cfg(3.0, 3.0)),
    ("CFG_thr0.2_hold1", make_cfg(0.2, 1.0)),
    ("CFG_thr0.2_hold3", make_cfg(0.2, 3.0)),
]


# ---------------------------------------------------------------------------
# Visual style (matches the conference presentation palette)
# ---------------------------------------------------------------------------

PALETTE = {
    "navy":      "#003D7A",
    "navy_dark": "#00264D",
    "accent":    "#00A8E1",
    "success":   "#10A37F",
    "warn":      "#E67E22",
    "danger":    "#C0392B",
    "muted":     "#667085",
    "rule":      "#D9DEE6",
    "bg_light":  "#F7F9FC",
    "text":      "#1C2A3A",
}


def apply_matplotlib_style() -> None:
    """Configure matplotlib defaults to match the presentation style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     PALETTE["muted"],
        "axes.labelcolor":    PALETTE["text"],
        "axes.titlecolor":    PALETTE["navy"],
        "axes.titleweight":   "bold",
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        PALETTE["text"],
        "ytick.color":        PALETTE["text"],
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "grid.color":         PALETTE["rule"],
        "grid.linewidth":     0.6,
        "grid.alpha":         0.8,
        "legend.frameon":     False,
        "legend.fontsize":    10,
        "font.family":        "DejaVu Sans",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })
