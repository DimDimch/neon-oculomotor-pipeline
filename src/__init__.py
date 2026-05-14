"""Reproducible processing pipeline for wearable eye-tracking recordings.

Modules
-------
config
    Default pipeline configuration, sensitivity-ablation variants and the
    plotting palette used across the repository.
io_utils
    ZIP enumeration, cached extraction, recording loading and time
    normalisation.
events
    Block-window inference and trial-event parsing.
metrics
    Gaze-based reaction-time and target-acquisition detection plus fixation
    and large-jump summaries.
pipeline
    Per-block metric dispatch and full-dataset processing.
analyses
    Reliability, classification and configuration-sensitivity studies.
plotting
    Figure generators in the conference-presentation style.
"""

from . import (
    analyses,
    config,
    events,
    io_utils,
    metrics,
    pipeline,
    plotting,
)

__all__ = [
    "analyses",
    "config",
    "events",
    "io_utils",
    "metrics",
    "pipeline",
    "plotting",
]
