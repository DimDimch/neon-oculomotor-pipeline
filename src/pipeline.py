"""Block-level metric computation and full pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CFG_DEFAULT
from .events import BLOCKS, infer_block_windows, parse_trial_events
from .io_utils import load_recording, normalize_time
from .metrics import (
    compute_jump_thr_px,
    count_large_jumps,
    fixation_trial_stats,
    gaze_based_rt,
    gaze_valid_mask,
    rt_saccade_reference,
    target_acquired_time,
    trial_window_from_parsed,
)


# ---------------------------------------------------------------------------
# Per-trial metric computation, dispatched by block type
# ---------------------------------------------------------------------------

def _saccadic_trial(
    rec: dict,
    parsed_events: pd.DataFrame,
    trial: pd.DataFrame,
    trial_id: int,
    block: str,
    cfg: dict,
    base_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute metrics for PREDICTION, GAP and OVERLAP trials."""
    onset_candidates = trial[
        (trial["step"] == 2)
        & (trial["edge"] == "start")
        & (trial["side"].isin(["LEFT", "RIGHT"]))
    ][["t_s", "side"]]
    if onset_candidates.empty:
        return base_row

    onset_s = float(onset_candidates["t_s"].min())
    target_side = onset_candidates.sort_values("t_s").iloc[0]["side"]
    base_row.update({"onset_s": onset_s, "target_side": target_side})

    rt = gaze_based_rt(rec, onset_s, cfg)
    base_row.update({
        "qc_trial_valid": int(rt["qc_trial_valid"]),
        "rt_found":      int(rt["rt_found"]),
        "rt_gaze_ms":    rt["rt_gaze_ms"],
        "response_dir":  rt["response_dir"],
    })
    base_row.update(rt_saccade_reference(rec, onset_s, cfg))

    if base_row["rt_found"] == 1 and base_row["response_dir"] in ("LEFT", "RIGHT"):
        base_row["direction_error"] = int(base_row["response_dir"] != target_side)
    else:
        base_row["direction_error"] = np.nan

    base_row["express_like"] = int(
        base_row["rt_found"] == 1
        and base_row["rt_gaze_ms"] < cfg["express_thr_ms"]
    )

    ta = target_acquired_time(rec, onset_s, target_side, rt["baseline_x"], cfg)
    base_row["target_reached"] = int(ta["target_reached"])
    base_row["time_to_target_ms"] = ta["time_to_target_ms"]

    # Post-response horizontal displacement
    if rt["rt_s"] is not None:
        gaze = rec["gaze"]
        post = gaze[
            (gaze["t_s"] >= rt["rt_s"] + 0.15)
            & (gaze["t_s"] <= rt["rt_s"] + 0.35)
        ]
        post = post[gaze_valid_mask(post)]
        post_dx = (
            float(post["gaze x [px]"].median() - rt["baseline_x"])
            if len(post) >= 5 else np.nan
        )
    else:
        post_dx = np.nan
    base_row["post_response_dx_px"] = post_dx

    if not np.isnan(post_dx) and target_side in ("LEFT", "RIGHT"):
        base_row["accuracy_sector"] = int(
            (post_dx > 0 and target_side == "RIGHT")
            or (post_dx < 0 and target_side == "LEFT")
        )
    else:
        base_row["accuracy_sector"] = np.nan

    # Fixation aggregates within the trial window
    t0, t1 = trial_window_from_parsed(parsed_events, trial_id, onset_s)
    base_row.update(fixation_trial_stats(rec, t0, t1))

    # Large horizontal jumps before the target is reached
    jump_thr_px = compute_jump_thr_px(rec, cfg)
    end_t = (
        ta["t_target"]
        if ta["t_target"] is not None
        else onset_s + cfg["search_window_target"][1]
    )
    n_jumps = count_large_jumps(rec["gaze"], onset_s, end_t, jump_thr_px)
    base_row["n_large_jumps_to_target"] = n_jumps
    base_row["n_steps_to_target"] = n_jumps

    # Trial duration and time-to-target share
    step2_end = trial[(trial["step"] == 2) & (trial["edge"] == "end")]["t_s"]
    trial_dur_ms = (
        float((step2_end.max() - onset_s) * 1000.0)
        if not step2_end.empty
        else 2000.0
    )
    base_row["trial_duration_ms"] = trial_dur_ms
    if (
        base_row["target_reached"] == 1
        and not np.isnan(base_row["time_to_target_ms"])
        and trial_dur_ms > 0
    ):
        base_row["time_to_target_share"] = float(
            base_row["time_to_target_ms"] / trial_dur_ms
        )
    else:
        base_row["time_to_target_share"] = np.nan

    return base_row


def _decision_trial(
    rec: dict,
    parsed_events: pd.DataFrame,
    trial: pd.DataFrame,
    trial_id: int,
    cfg: dict,
    base_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute metrics for DECISION trials."""
    onset = trial[(trial["step"] == 2) & (trial["edge"] == "start")]["t_s"]
    if onset.empty:
        return base_row
    onset_s = float(onset.min())
    base_row["onset_s"] = onset_s

    correct = trial[
        (trial["step"] == 3)
        & (trial["correctness"] == "CORRECT")
        & (trial["edge"] == "start")
    ]["side"]
    correct_side = correct.iloc[0] if not correct.empty else None

    rt = gaze_based_rt(rec, onset_s, cfg)
    base_row.update({
        "qc_trial_valid":      int(rt["qc_trial_valid"]),
        "decision_rt_gaze_ms": rt["rt_gaze_ms"],
        "choice_side":         rt["response_dir"],
        "correct_side":        correct_side,
    })
    if base_row["choice_side"] in ("LEFT", "RIGHT") and correct_side in ("LEFT", "RIGHT"):
        base_row["decision_correct"] = int(base_row["choice_side"] == correct_side)
    else:
        base_row["decision_correct"] = np.nan

    t0, t1 = trial_window_from_parsed(parsed_events, trial_id, onset_s)
    base_row.update(fixation_trial_stats(rec, t0, t1))
    return base_row


def _antisaccade_trial(
    rec: dict,
    parsed_events: pd.DataFrame,
    trial: pd.DataFrame,
    trial_id: int,
    cfg: dict,
    base_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute metrics for ANTISACCADE trials."""
    cue = trial[(trial["step"] == 2) & (trial["edge"] == "start")]["color"]
    if cue.empty:
        return base_row
    onset_s = float(trial[(trial["step"] == 2) & (trial["edge"] == "start")]["t_s"].min())
    base_row.update({"onset_s": onset_s, "cue_color": cue.iloc[0]})

    rt = gaze_based_rt(rec, onset_s, cfg)
    base_row.update({
        "qc_trial_valid": int(rt["qc_trial_valid"]),
        "rt_found":       int(rt["rt_found"]),
        "rt_gaze_ms":     rt["rt_gaze_ms"],
        "response_dir":   rt["response_dir"],
    })

    t0, t1 = trial_window_from_parsed(parsed_events, trial_id, onset_s)
    base_row.update(fixation_trial_stats(rec, t0, t1))
    return base_row


def compute_trial_metrics_for_block(
    rec: dict,
    parsed_events: pd.DataFrame,
    block: str,
    cfg: dict = CFG_DEFAULT,
) -> pd.DataFrame:
    """Compute trial-level metrics for all trials of a single block."""
    pe = parsed_events[parsed_events["block"] == block]
    if pe.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for trial_id in sorted(pe["trial_id"].unique()):
        trial = pe[pe["trial_id"] == trial_id]
        base_row: Dict[str, Any] = {
            "recording_id": rec["recording_id"],
            "block": block,
            "trial_id": int(trial_id),
            "onset_s": np.nan,
            "target_side": None,
            "qc_trial_valid": 0,
        }
        if block in ("PREDICTION", "GAP", "OVERLAP"):
            rows.append(_saccadic_trial(rec, parsed_events, trial, trial_id, block, cfg, base_row))
        elif block == "DECISION":
            rows.append(_decision_trial(rec, parsed_events, trial, trial_id, cfg, base_row))
        elif block == "ANTISACCADE":
            rows.append(_antisaccade_trial(rec, parsed_events, trial, trial_id, cfg, base_row))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full-recording and full-dataset processing
# ---------------------------------------------------------------------------

def process_one_recording(
    recording_folder: Path, cfg: dict = CFG_DEFAULT
) -> pd.DataFrame:
    """Return the trial-metrics DataFrame for a single recording."""
    rec = normalize_time(load_recording(recording_folder))
    block_windows = infer_block_windows(rec["events"])
    parsed_events = parse_trial_events(rec["events"], block_windows)

    pieces = []
    for block in BLOCKS:
        tm = compute_trial_metrics_for_block(rec, parsed_events, block, cfg)
        if not tm.empty:
            pieces.append(tm)
    return (
        pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    )


def build_trial_table(
    dataset_index: pd.DataFrame,
    cfg: dict = CFG_DEFAULT,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Process every recording in ``dataset_index`` and concatenate the results.

    When ``output_path`` is provided the resulting table is also written there
    as CSV.
    """
    pieces = []
    for _, row in dataset_index.iterrows():
        folder = Path(row["recording_folder"])
        tm = process_one_recording(folder, cfg)
        if tm.empty:
            continue
        tm = tm.copy()
        tm["zip_name"]   = row["zip_name"]
        tm["zip_stem"]   = row["zip_stem"]
        tm["label"]      = row["label"]
        tm["label_text"] = row["label_text"]
        pieces.append(tm)

    trial_metrics = (
        pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trial_metrics.to_csv(output_path, index=False)
    return trial_metrics
