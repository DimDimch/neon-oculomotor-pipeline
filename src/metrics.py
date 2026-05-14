"""Trial-level metric extraction from gaze and saccade streams."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import CFG_DEFAULT
from .io_utils import get_frame_width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaze_valid_mask(gaze: pd.DataFrame) -> pd.Series:
    """Boolean mask excluding samples with ``worn != 1`` or active blink id."""
    mask = pd.Series(True, index=gaze.index)
    if "worn" in gaze.columns:
        mask &= (gaze["worn"] == 1)
    if "blink id" in gaze.columns:
        mask &= gaze["blink id"].isna()
    return mask


def sustained_condition_time(
    ts: np.ndarray, cond: np.ndarray, min_hold_s: float
) -> Optional[float]:
    """Return the first timestamp at which ``cond`` stays True for ``min_hold_s`` seconds."""
    n = len(ts)
    if n == 0:
        return None
    i = 0
    while i < n:
        if not cond[i]:
            i += 1
            continue
        j = i
        while j < n and cond[j]:
            j += 1
        if ts[j - 1] - ts[i] >= min_hold_s:
            return float(ts[i])
        i = j
    return None


def compute_thr_px(rec: dict, cfg: dict = CFG_DEFAULT) -> int:
    """Effective deviation threshold in pixels, derived from the frame width."""
    width = get_frame_width(rec.get("scene_camera", {}))
    if width is None:
        return int(cfg["thr_px_fallback"])
    return int(max(cfg["thr_px_fallback"], cfg["thr_px_ratio"] * width))


def compute_jump_thr_px(rec: dict, cfg: dict = CFG_DEFAULT) -> int:
    """Effective large-jump threshold in pixels."""
    width = get_frame_width(rec.get("scene_camera", {}))
    if width is None:
        return int(cfg["jump_thr_px_fallback"])
    return int(max(cfg["jump_thr_px_fallback"], cfg["jump_thr_px_ratio"] * width))


# ---------------------------------------------------------------------------
# Gaze-based reaction time and target acquisition
# ---------------------------------------------------------------------------

def gaze_based_rt(
    rec: dict, onset_s: float, cfg: dict = CFG_DEFAULT
) -> Dict[str, Any]:
    """Estimate the reaction-time and response direction from gaze samples.

    The baseline horizontal position is the median of valid gaze samples in
    ``[onset - 0.20, onset - 0.05]`` seconds. The reaction time is the first
    moment after onset at which ``|gaze_x - baseline_x|`` exceeds the
    per-recording deviation threshold and stays above it for at least
    ``hold_ms_reaction`` ms.
    """
    gaze = rec["gaze"]
    thr_px = compute_thr_px(rec, cfg)

    base = gaze[
        (gaze["t_s"] >= onset_s - 0.20) & (gaze["t_s"] <= onset_s - 0.05)
    ]
    base = base[gaze_valid_mask(base)]
    if len(base) < 5:
        return _rt_failure(thr_px)

    baseline_x = float(base["gaze x [px]"].median())
    baseline_y = float(base["gaze y [px]"].median())

    w0, w1 = cfg["search_window_reaction"]
    seg = gaze[(gaze["t_s"] >= onset_s + w0) & (gaze["t_s"] <= onset_s + w1)]
    seg = seg[gaze_valid_mask(seg)]
    if len(seg) < 10:
        return _rt_failure(thr_px, baseline_x=baseline_x, baseline_y=baseline_y)

    ts = seg["t_s"].values
    dx = seg["gaze x [px]"].values - baseline_x
    cond = np.abs(dx) >= thr_px

    rt_s = sustained_condition_time(ts, cond, cfg["hold_ms_reaction"] / 1000.0)
    if rt_s is None:
        return {
            "qc_trial_valid": 1, "baseline_x": baseline_x, "baseline_y": baseline_y,
            "rt_s": None, "rt_found": 0, "rt_gaze_ms": np.nan,
            "response_dir": None, "thr_px": thr_px,
        }

    rt_gaze_ms = float((rt_s - onset_s) * 1000.0)

    resp = gaze[(gaze["t_s"] >= rt_s) & (gaze["t_s"] <= rt_s + 0.12)]
    resp = resp[gaze_valid_mask(resp)]
    if len(resp) < 5:
        response_dir = None
    else:
        dx_med = float(resp["gaze x [px]"].median() - baseline_x)
        response_dir = "RIGHT" if dx_med > 0 else "LEFT"

    return {
        "qc_trial_valid": 1, "baseline_x": baseline_x, "baseline_y": baseline_y,
        "rt_s": rt_s, "rt_found": 1, "rt_gaze_ms": rt_gaze_ms,
        "response_dir": response_dir, "thr_px": thr_px,
    }


def _rt_failure(thr_px: int, baseline_x=np.nan, baseline_y=np.nan) -> Dict[str, Any]:
    return {
        "qc_trial_valid": 0, "baseline_x": baseline_x, "baseline_y": baseline_y,
        "rt_s": None, "rt_found": 0, "rt_gaze_ms": np.nan,
        "response_dir": None, "thr_px": thr_px,
    }


def target_acquired_time(
    rec: dict,
    onset_s: float,
    target_side: str,
    baseline_x: float,
    cfg: dict = CFG_DEFAULT,
) -> Dict[str, Any]:
    """Detect when the gaze enters and stays in the target sector."""
    gaze = rec["gaze"]
    thr_px = compute_thr_px(rec, cfg)

    w0, w1 = cfg["search_window_target"]
    seg = gaze[(gaze["t_s"] >= onset_s + w0) & (gaze["t_s"] <= onset_s + w1)]
    seg = seg[gaze_valid_mask(seg)]
    if (
        len(seg) < 10
        or target_side not in ("LEFT", "RIGHT")
        or np.isnan(baseline_x)
    ):
        return {"target_reached": 0, "t_target": None, "time_to_target_ms": np.nan}

    x = seg["gaze x [px]"].values
    ts = seg["t_s"].values
    if target_side == "LEFT":
        cond = x <= (baseline_x - thr_px)
    else:
        cond = x >= (baseline_x + thr_px)

    t_target = sustained_condition_time(ts, cond, cfg["hold_ms_target"] / 1000.0)
    if t_target is None:
        return {"target_reached": 0, "t_target": None, "time_to_target_ms": np.nan}
    return {
        "target_reached": 1,
        "t_target": t_target,
        "time_to_target_ms": float((t_target - onset_s) * 1000.0),
    }


def rt_saccade_reference(
    rec: dict, onset_s: float, cfg: dict = CFG_DEFAULT
) -> Dict[str, Any]:
    """Reference reaction time obtained from the saccades.csv stream.

    Used as a quality check against the gaze-based RT only; never as the
    primary metric.
    """
    sacc = rec["saccades"]
    if sacc is None or sacc.empty or "t_start_s" not in sacc.columns:
        return {"rt_saccade_found": 0, "rt_saccade_ms": np.nan}
    w0, w1 = cfg["search_window_reaction"]
    candidates = sacc[
        (sacc["t_start_s"] >= onset_s + w0) & (sacc["t_start_s"] <= onset_s + w1)
    ].sort_values("t_start_s")
    if candidates.empty:
        return {"rt_saccade_found": 0, "rt_saccade_ms": np.nan}
    t = float(candidates.iloc[0]["t_start_s"])
    return {"rt_saccade_found": 1, "rt_saccade_ms": float((t - onset_s) * 1000.0)}


# ---------------------------------------------------------------------------
# Fixation and jump summaries
# ---------------------------------------------------------------------------

def trial_window_from_parsed(
    parsed_block: pd.DataFrame, trial_id: int, onset_s: float
) -> tuple[float, float]:
    """Return the time window of a trial used for fixation aggregation."""
    t0, t1 = onset_s - 0.5, onset_s + 1.5
    sub = parsed_block[parsed_block["trial_id"] == trial_id]
    s1 = sub[(sub["step"] == 1) & (sub["edge"] == "start")]["t_s"]
    e2 = sub[(sub["step"] == 2) & (sub["edge"] == "end")]["t_s"]
    if not s1.empty:
        t0 = float(s1.min())
    if not e2.empty:
        t1 = float(e2.max())
    return t0, t1


def fixation_trial_stats(rec: dict, t0: float, t1: float) -> Dict[str, float]:
    """Mean and median fixation duration within the trial window (in ms)."""
    fix = rec["fixations"]
    nan_result = {
        "fix_dur_ms_mean_trial":   np.nan,
        "fix_dur_ms_median_trial": np.nan,
    }
    if fix is None or fix.empty or "duration [ms]" not in fix.columns:
        return nan_result
    seg = fix[(fix["t_end_s"] >= t0) & (fix["t_start_s"] <= t1)]
    if seg.empty:
        return nan_result
    return {
        "fix_dur_ms_mean_trial":   float(seg["duration [ms]"].mean()),
        "fix_dur_ms_median_trial": float(seg["duration [ms]"].median()),
    }


def count_large_jumps(
    gaze: pd.DataFrame, t0: float, t1: float, jump_thr_px: float
) -> int:
    """Count consecutive-sample horizontal jumps exceeding ``jump_thr_px``."""
    seg = gaze[(gaze["t_s"] >= t0) & (gaze["t_s"] <= t1)]
    seg = seg[gaze_valid_mask(seg)]
    if len(seg) < 3:
        return 0
    dx = np.diff(seg["gaze x [px]"].values)
    return int(np.sum(np.abs(dx) > jump_thr_px))
