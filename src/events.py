"""Block-window inference and trial-event parsing."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


BLOCKS = ("PREDICTION", "GAP", "OVERLAP", "DECISION", "ANTISACCADE")

SIDE_TOKENS = {"LEFT", "RIGHT", "CENTER", "UP", "DOWN"}
COLOR_TOKENS = {"RED", "GREEN", "BLUE", "YELLOW", "WHITE", "BLACK"}

_EVENT_PATTERN = re.compile(
    r"(?P<trial>\d+)\.(?P<step>\d+)_(?P<tag>.*)_(?P<edge>start|end)$"
)


def infer_block_windows(events: pd.DataFrame) -> pd.DataFrame:
    """Locate the time windows of each task block from the event stream.

    For every block ``B`` the conference stimulus protocol emits four markers
    forming three contiguous parts:

    ``intro``  : ``START_<B>_BLOCK_START`` to ``START_<B>_BLOCK_END``
    ``trials`` : ``START_<B>_BLOCK_END``   to ``END_<B>_BLOCK_START``
    ``outro``  : ``END_<B>_BLOCK_START``   to ``END_<B>_BLOCK_END``

    The calibration block uses a simpler ``CALIB_BLOCK_START`` / ``..._END``
    pair.

    Returns
    -------
    DataFrame
        Columns ``block``, ``part``, ``t_start``, ``t_end``.
    """
    ev = events.sort_values("timestamp [ns]")
    rows = []

    if (ev["name"] == "CALIB_BLOCK_START").any() and (ev["name"] == "CALIB_BLOCK_END").any():
        rows.append({
            "block":   "CALIB",
            "part":    "trials",
            "t_start": ev.loc[ev["name"] == "CALIB_BLOCK_START", "t_s"].min(),
            "t_end":   ev.loc[ev["name"] == "CALIB_BLOCK_END",   "t_s"].min(),
        })

    for b in BLOCKS:
        s1 = f"START_{b}_BLOCK_START"
        s2 = f"START_{b}_BLOCK_END"
        e1 = f"END_{b}_BLOCK_START"
        e2 = f"END_{b}_BLOCK_END"
        required = (s1, s2, e1, e2)
        if not all((ev["name"] == r).any() for r in required):
            continue
        intro_start  = ev.loc[ev["name"] == s1, "t_s"].min()
        intro_end    = ev.loc[ev["name"] == s2, "t_s"].min()
        trials_end   = ev.loc[ev["name"] == e1, "t_s"].min()
        outro_end    = ev.loc[ev["name"] == e2, "t_s"].min()
        rows.extend([
            {"block": b, "part": "intro",  "t_start": intro_start, "t_end": intro_end},
            {"block": b, "part": "trials", "t_start": intro_end,   "t_end": trials_end},
            {"block": b, "part": "outro",  "t_start": trials_end,  "t_end": outro_end},
        ])
    return (
        pd.DataFrame(rows)
        .sort_values(["t_start", "block", "part"])
        .reset_index(drop=True)
    )


def parse_event_name(name: str) -> Optional[dict]:
    """Decompose an event label of the form ``<trial>.<step>_<tag>_<edge>``.

    The ``tag`` may carry side, colour, correctness and a numeric parameter
    encoded by underscores. Returns ``None`` when the label does not match.
    """
    m = _EVENT_PATTERN.match(str(name))
    if not m:
        return None
    d = m.groupdict()
    d["trial_id"] = int(d.pop("trial"))
    d["step"] = int(d["step"])

    parts = d["tag"].split("_")
    d["side"]  = parts[0] if parts and parts[0] in SIDE_TOKENS else None
    d["color"] = parts[0] if parts and parts[0] in COLOR_TOKENS else None

    if "CORRECT" in parts:
        d["correctness"] = "CORRECT"
    elif "INCORRECT" in parts:
        d["correctness"] = "INCORRECT"
    else:
        d["correctness"] = None

    d["param_num"] = None
    for token in parts[1:]:
        if token.isdigit():
            d["param_num"] = int(token)
            break
    return d


def parse_trial_events(
    events: pd.DataFrame, block_windows: pd.DataFrame
) -> pd.DataFrame:
    """Return a long-format table of parsed trial events restricted to ``trials`` windows."""
    out = []
    trial_windows = block_windows[block_windows["part"] == "trials"]
    for _, bw in trial_windows.iterrows():
        segment = events[
            (events["t_s"] >= bw["t_start"]) & (events["t_s"] <= bw["t_end"])
        ].sort_values("timestamp [ns]")
        for _, ev in segment.iterrows():
            parsed = parse_event_name(ev["name"])
            if parsed is None:
                continue
            parsed["block"] = bw["block"]
            parsed["t_s"] = float(ev["t_s"])
            out.append(parsed)

    if not out:
        return pd.DataFrame()

    cols = [
        "block", "trial_id", "step", "tag", "edge",
        "side", "color", "correctness", "param_num", "t_s",
    ]
    return (
        pd.DataFrame(out)[cols]
        .sort_values(["block", "trial_id", "step", "t_s"])
        .reset_index(drop=True)
    )
