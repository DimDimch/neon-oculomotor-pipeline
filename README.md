# Reproducible Processing and Validation of Wearable Eye-Tracking Data

Companion code for the USBEREIT 2026 paper *Reproducible Processing and
Validation of Wearable Eye-Tracking Data for Cognitive Task Analysis*
(Serdyukov, Fedorov, Safonova; ITMO University, 2026).

This repository contains a reproducible pipeline that converts raw Pupil Labs
Neon recordings (200 Hz gaze plus IMU and event streams) into analysis-ready
trial-level metrics, together with the three validation studies reported in
the paper: split-half reliability, task-type classification, and
configuration sensitivity.

## Repository layout

```
.
├── notebooks/
│   └── usbereit2026_analysis.ipynb   # single entry point, reproduces all figures
├── src/
│   ├── config.py                     # default CFG, ablation variants, palette
│   ├── io_utils.py                   # ZIP indexing, extraction, loading
│   ├── events.py                     # block windows and trial-event parsing
│   ├── metrics.py                    # gaze-based RT, target acquisition, fixations
│   ├── pipeline.py                   # per-block dispatch, full-dataset orchestration
│   ├── analyses.py                   # reliability, classification, sensitivity
│   └── plotting.py                   # figure generators in the paper palette
├── figures/                          # generated PNGs (example outputs included)
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

## Installation

The code targets Python 3.10 or newer.

```bash
git clone https://gitlab.com/<your-namespace>/neon-oculomotor-usbereit2026.git
cd neon-oculomotor-usbereit2026

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

The pipeline expects one ZIP archive per recording. Each archive should
contain a standard Pupil Labs Neon export (`gaze.csv`, `events.csv`,
`fixations.csv`, `saccades.csv`, `blinks.csv`, `imu.csv`, `info.json`,
`scene_camera.json`). Place the ZIPs in a single directory, by default
`neon_zips/`:

```
neon_zips/
├── HC_001.zip
├── HC_002.zip
├── PAT_001.zip
└── ...
```

Subject labels (healthy / patient) are inferred from filename tokens:

| Token (case-insensitive) | Label |
|---|---|
| `hc`, `healthy`, `control`, `ctl`, `norm`, `normal` | healthy |
| `pat`, `patient`, `ad`, `alz`, `alzheimer`, `mci`, `dement` | patient |

For ZIPs whose stem matches neither set, pass an explicit mapping via the
`MANUAL_LABELS` variable in the notebook.

## How to run

Open and execute the notebook:

```bash
jupyter notebook notebooks/usbereit2026_analysis.ipynb
```

The notebook produces, in order:

1. `neon_outputs/trial_metrics.csv` — per-trial metrics for all recordings.
2. `tables/reliability_summary.csv` — median relative difference + 95% CI per metric.
3. `tables/task_classification_summary.csv` — balanced accuracy and macro F1 per model.
4. `tables/top_features_logreg.csv`, `tables/top_features_rf.csv` — most informative features.
5. `tables/cfg_sensitivity.csv` — balanced accuracy and RT detection rate per configuration.
6. `figures/*.png` — paper-quality figures (300 DPI) used in the slides and manuscript.

Each cell is independent within its section, so individual analyses can be
re-run without restarting the kernel.

## Pipeline summary

The pipeline has four stages:

1. **Ingest.** Standard Pupil Labs Neon exports are loaded from disk.
2. **Time normalisation.** UTC nanosecond timestamps are converted to a
   shared relative axis anchored to `info.json::start_time`. Events serve
   as the synchronisation backbone for the downstream stages.
3. **Quality control.** Effective gaze sampling rate, worn ratio, and a
   gyroscope-based head-motion proxy (RMS of available gyroscope channels)
   are computed and thresholded into recording-level QC flags.
4. **Trial reconstruction and feature extraction.** Block boundaries are
   inferred from `START_<B>_BLOCK_*` / `END_<B>_BLOCK_*` markers, and trial
   events are parsed using the `<trial>.<step>_<tag>_<edge>` naming scheme.
   For each trial the pipeline extracts gaze-based reaction time, inferred
   response direction, direction-error indicators, target-acquisition time,
   fixation summaries, and inter-sample jump counts.

Each behaviour-bearing decision (deviation threshold, hold time, search
window, QC cut-off) lives in the configuration dictionary in
`src/config.py`. The default values reproduce the numbers reported in the
paper.

## Validation studies

### 1. Reliability (split-half + bootstrap)

For every `(recording_id, block)` pair in the PREDICTION / GAP / OVERLAP
blocks, trials are sorted by index and split into odd and even halves. The
per-half means of each numeric metric are computed and the relative
difference is recorded. A bootstrap procedure (B = 1000) over the
per-pair relative differences yields the median and 95% CI shown in
`figures/fig_reliability.png`.

### 2. Task-type classification

A scaled logistic regression and a 300-tree random forest are trained on
the full set of numeric trial-level features with the five task labels as
the target. Cross-validation uses GroupKFold (k = 4) with `recording_id`
as the group, ensuring that all trials from one participant stay in the
same fold. Missing numeric values are filled with column means; missing
categorical values are filled with a placeholder and label-encoded.

The reported accuracies should be read as an upper bound on the purely
physiological component of discriminability: the most informative features
include `trial_duration_ms` and `onset_s`, which are determined by the
stimulus program rather than oculomotor behaviour. A run restricted to
biologically grounded features is planned as future work.

### 3. Configuration sensitivity

The full pipeline is re-run under eight configurations that scale the
RT-detection threshold (`thr_px_ratio`) and the RT hold time
(`hold_ms_reaction`) by different factors:

| Configuration | Threshold factor | Hold-time factor |
|---|---|---|
| `CFG_default` | 1.0 | 1.0 |
| `CFG_x0.2` | 0.2 | 0.2 |
| `CFG_x0.8` | 0.8 | 0.8 |
| `CFG_x1.2` | 1.2 | 1.2 |
| `CFG_x2.0` | 2.0 | 2.0 |
| `CFG_x3.0` | 3.0 | 3.0 |
| `CFG_thr0.2_hold1` | 0.2 | 1.0 |
| `CFG_thr0.2_hold3` | 0.2 | 3.0 |

For each configuration the full trial table is rebuilt, a logistic
regression is evaluated with the same cross-validation scheme, and the
median per-recording RT detection rate is recorded.

## Reproducibility notes

* Random seeds are fixed for the bootstrap (`seed=42`) and for the random
  forest (`random_state=42`).
* Group-aware cross-validation removes participant leakage between folds.
* All processing decisions are stored in a single configuration dictionary;
  the same dictionary controls the sensitivity ablation.
* Outputs are deterministic given identical inputs and configuration.

## Limitations

The original evaluation is based on eight clinical recordings (six healthy
controls and two patients). No group-level statistical or diagnostic
claims should be drawn from the published numbers, and external replication
on larger cohorts is needed before translational use.

## Citation

If you build on this work, please cite the conference paper:

```bibtex
@inproceedings{serdyukov2026reproducible,
  author    = {Serdyukov, Dmitrii and Fedorov, Dmitriy and Safonova, Liudmila},
  title     = {Reproducible Processing and Validation of Wearable
               Eye-Tracking Data for Cognitive Task Analysis},
  booktitle = {Proc. IEEE Ural-Siberian Conf. on Biomedical Engineering,
               Radioelectronics and Information Technology (USBEREIT)},
  year      = {2026},
  address   = {Yekaterinburg, Russia},
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Contact

Dmitrii Serdyukov, Faculty of Applied Informatics, ITMO University,
Saint Petersburg, Russia. Issues and pull requests are welcome on the
project repository; for direct correspondence, email
[dvserdiukov@itmo.ru](mailto:dvserdiukov@itmo.ru).
