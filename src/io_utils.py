"""Recording discovery, ZIP extraction and time normalisation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Subject labelling from ZIP filenames
# ---------------------------------------------------------------------------

HEALTHY_TOKENS = ("hc", "healthy", "control", "ctl", "norm", "normal")
PATIENT_TOKENS = ("pat", "patient", "ad", "alz", "alzheimer", "mci", "dement")


def infer_label_from_zipname(zip_path: Path) -> Tuple[int | None, str]:
    """Infer the participant label from a ZIP file stem.

    Returns
    -------
    (label, label_text)
        ``label`` is ``0`` for healthy controls, ``1`` for patients, or ``None``
        when the stem matches no recognised token. ``label_text`` is one of
        ``"healthy"``, ``"patient"`` or ``"unknown"``.
    """
    name = zip_path.stem.lower()
    if any(tok in name for tok in PATIENT_TOKENS):
        return 1, "patient"
    if any(tok in name for tok in HEALTHY_TOKENS):
        return 0, "healthy"
    return None, "unknown"


def build_zip_index(
    dataset_dir: Path,
    manual_labels: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """Enumerate ZIP recordings in ``dataset_dir`` and assign labels.

    Parameters
    ----------
    dataset_dir
        Directory containing one ``*.zip`` file per recording.
    manual_labels
        Optional override mapping ``{zip_filename: label}`` for ZIPs whose
        stem does not match the recognised tokens.
    """
    dataset_dir = Path(dataset_dir)
    zips = sorted(dataset_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files found in {dataset_dir}")

    rows = []
    for zp in zips:
        label, label_text = infer_label_from_zipname(zp)
        rows.append({
            "zip_path": str(zp),
            "zip_name": zp.name,
            "zip_stem": zp.stem,
            "label": label,
            "label_text": label_text,
        })

    df = pd.DataFrame(rows)
    if manual_labels:
        for fname, value in manual_labels.items():
            mask = df["zip_name"] == fname
            df.loc[mask, "label"] = value
            df.loc[mask, "label_text"] = "patient" if value == 1 else "healthy"
    return df


# ---------------------------------------------------------------------------
# Extraction (cached) and recording discovery
# ---------------------------------------------------------------------------

def unzip_cached(zip_path: Path, work_dir: Path) -> Path:
    """Extract ``zip_path`` into ``work_dir/<stem>/``; skip if already present."""
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    out_dir = work_dir / zip_path.stem
    if out_dir.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def find_recording_folders(root: Path) -> List[Path]:
    """Find Pupil Labs recording folders containing both info.json and gaze.csv."""
    root = Path(root)
    folders = set()
    for info_path in root.rglob("info.json"):
        folder = info_path.parent
        if (folder / "gaze.csv").exists():
            folders.add(folder.resolve())
    return sorted(folders)


def build_dataset_index(
    zip_index: pd.DataFrame,
    work_dir: Path,
) -> pd.DataFrame:
    """Return one row per recording, joined with its subject-level metadata."""
    records = []
    for _, row in zip_index.iterrows():
        extracted = unzip_cached(Path(row["zip_path"]), Path(work_dir))
        for rec_folder in find_recording_folders(extracted):
            records.append({
                "zip_name": row["zip_name"],
                "zip_stem": row["zip_stem"],
                "label": row["label"],
                "label_text": row["label_text"],
                "recording_folder": str(rec_folder),
                "recording_id": rec_folder.name,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Loading and time normalisation
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_recording(folder: Path) -> dict:
    """Load all streams of a Pupil Labs recording into a dictionary of DataFrames."""
    folder = Path(folder)
    return {
        "folder":           folder,
        "recording_id":     folder.name,
        "info":             _read_json(folder / "info.json"),
        "scene_camera":     _read_json(folder / "scene_camera.json"),
        "gaze":             _read_csv(folder / "gaze.csv"),
        "fixations":        _read_csv(folder / "fixations.csv"),
        "saccades":         _read_csv(folder / "saccades.csv"),
        "blinks":           _read_csv(folder / "blinks.csv"),
        "events":           _read_csv(folder / "events.csv"),
        "imu":              _read_csv(folder / "imu.csv"),
        "world_timestamps": _read_csv(folder / "world_timestamps.csv"),
    }


def _add_time_s(df: pd.DataFrame, ts_col: str, t0_ns: int, out_col: str) -> pd.DataFrame:
    if df is None or df.empty or ts_col not in df.columns:
        return df
    df = df.copy()
    df[out_col] = (df[ts_col].astype(np.int64) - int(t0_ns)) / 1e9
    return df


def normalize_time(rec: dict) -> dict:
    """Convert all UTC nanosecond timestamps to seconds relative to recording start.

    ``info.json::start_time`` is used as the anchor. When that field is missing,
    the minimum timestamp across the gaze and event streams is used instead.
    """
    t0_ns = rec["info"].get("start_time")
    if t0_ns is None:
        candidates = []
        if not rec["gaze"].empty:
            candidates.append(rec["gaze"]["timestamp [ns]"].min())
        if not rec["events"].empty:
            candidates.append(rec["events"]["timestamp [ns]"].min())
        if not candidates:
            raise ValueError(
                f"Cannot determine t0_ns for recording {rec['recording_id']}"
            )
        t0_ns = int(min(candidates))
    rec["t0_ns"] = int(t0_ns)

    rec["gaze"] = _add_time_s(rec["gaze"], "timestamp [ns]", t0_ns, "t_s")
    rec["events"] = _add_time_s(rec["events"], "timestamp [ns]", t0_ns, "t_s")
    rec["imu"] = _add_time_s(rec["imu"], "timestamp [ns]", t0_ns, "t_s")
    rec["world_timestamps"] = _add_time_s(
        rec["world_timestamps"], "timestamp [ns]", t0_ns, "t_s"
    )
    for key in ("fixations", "saccades", "blinks"):
        rec[key] = _add_time_s(rec[key], "start timestamp [ns]", t0_ns, "t_start_s")
        rec[key] = _add_time_s(rec[key], "end timestamp [ns]",   t0_ns, "t_end_s")
    return rec


def get_frame_width(scene_camera: dict) -> int | None:
    """Extract the scene-camera frame width (px) from ``scene_camera.json``.

    Pupil Labs has changed this layout over releases, so several locations
    are probed. Returns ``None`` when the width cannot be found.
    """
    if not scene_camera:
        return None
    res = scene_camera.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) == 2:
        return int(res[0])
    if "width" in scene_camera:
        return int(scene_camera["width"])
    cam = scene_camera.get("camera")
    if isinstance(cam, dict):
        res = cam.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            return int(res[0])
    return None
