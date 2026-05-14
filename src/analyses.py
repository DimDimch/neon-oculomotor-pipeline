"""Validation analyses: reliability, classification and configuration sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CFG_DEFAULT, SENSITIVITY_CONFIGS
from .pipeline import build_trial_table


# ---------------------------------------------------------------------------
# Common preprocessing
# ---------------------------------------------------------------------------

_CATEGORICAL_COLUMNS = (
    "target_side", "response_dir", "choice_side", "correct_side",
    "cue_color", "recording_id", "zip_name", "zip_stem",
)

_NON_FEATURE_COLUMNS = (
    "trial_id", "block", "label_text", "recording_id",
    "target_side", "response_dir", "choice_side", "correct_side",
    "cue_color", "zip_name", "zip_stem",
)


def preprocess_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and label-encode categorical columns.

    Numeric columns are filled with column means. Categorical columns are
    filled with the literal ``"missing"`` and additionally label-encoded into
    a parallel ``*_num`` column so they can be passed to estimators as
    integers.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].mean())

    for col in _CATEGORICAL_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].fillna("missing")
        out[f"{col}_num"] = LabelEncoder().fit_transform(out[col])
    return out


def feature_matrix(trial_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split ``trial_metrics`` into ``(X, y, feature_names)`` for sklearn."""
    drop = [c for c in _NON_FEATURE_COLUMNS if c in trial_metrics.columns]
    X = trial_metrics.drop(columns=drop)
    y = trial_metrics["block"]
    return X, y, list(X.columns)


# ---------------------------------------------------------------------------
# Reliability analysis (split-half + bootstrap)
# ---------------------------------------------------------------------------

RELIABILITY_BLOCKS = ("PREDICTION", "GAP", "OVERLAP")


def split_half_relative_differences(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-recording, per-block, per-metric relative difference.

    Trials are split by odd / even ``trial_id`` within each
    ``(recording_id, block)`` pair, and the relative difference of the means
    is returned.
    """
    df = trial_metrics[trial_metrics["block"].isin(RELIABILITY_BLOCKS)]
    metric_cols = df.select_dtypes(include="number").columns.difference(["trial_id"])

    rows = []
    for (rec_id, block), pair in df.groupby(["recording_id", "block"]):
        pair = pair.sort_values("trial_id")
        odd  = pair[pair["trial_id"] % 2 == 1]
        even = pair[pair["trial_id"] % 2 == 0]
        for metric in metric_cols:
            m_all = pair[metric].mean()
            rel = abs(odd[metric].mean() - even[metric].mean()) / (abs(m_all) + 1e-6)
            rows.append({
                "recording_id": rec_id, "block": block,
                "metric": metric, "rel_diff": rel,
            })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_iter: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """95% bootstrap confidence interval around the median of ``values``."""
    rng = np.random.default_rng(seed)
    medians = np.empty(n_iter)
    n = len(values)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def reliability_summary(
    pair_results: pd.DataFrame, n_iter: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Per-metric median relative difference with bootstrap 95% CI."""
    rows = []
    for metric, sub in pair_results.groupby("metric"):
        values = sub["rel_diff"].values
        ci_low, ci_high = bootstrap_ci(values, n_iter=n_iter, seed=seed)
        rows.append({
            "feature": metric,
            "median_rel_diff": float(np.median(values)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_groups": len(values),
        })
    return pd.DataFrame(rows).sort_values("median_rel_diff").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task-type classification
# ---------------------------------------------------------------------------

def _make_classifiers() -> Dict[str, object]:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    }


def cross_validate_classifiers(
    trial_metrics: pd.DataFrame,
    n_splits: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Run GroupKFold cross-validation with recordings as groups.

    Returns
    -------
    summary : DataFrame
        Mean and std of balanced accuracy and macro-F1 per model.
    details : dict
        Per-model dict with stacked ``y_true``, ``y_pred`` and class labels,
        suitable for confusion-matrix plotting.
    """
    X, y, _ = feature_matrix(trial_metrics)
    groups = trial_metrics["recording_id_num"]
    gkf = GroupKFold(n_splits=n_splits)

    summary_rows = []
    details: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in _make_classifiers().items():
        bal_accs: List[float] = []
        f1s: List[float] = []
        y_true_all: List = []
        y_pred_all: List = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            bal_accs.append(balanced_accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="macro"))
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        summary_rows.append({
            "model": name,
            "balanced_accuracy_mean": float(np.mean(bal_accs)),
            "balanced_accuracy_std":  float(np.std(bal_accs)),
            "macro_f1_mean":          float(np.mean(f1s)),
            "macro_f1_std":           float(np.std(f1s)),
        })
        details[name] = {
            "y_true": np.array(y_true_all),
            "y_pred": np.array(y_pred_all),
            "labels": sorted(set(y_true_all)),
            "confusion": confusion_matrix(
                y_true_all, y_pred_all, labels=sorted(set(y_true_all))
            ),
        }
    return pd.DataFrame(summary_rows), details


def top_features(
    trial_metrics: pd.DataFrame, k: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``k`` most informative features for each classifier."""
    X, y, feature_names = feature_matrix(trial_metrics)

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    logreg.fit(X, y)
    abs_coef = np.abs(logreg.named_steps["clf"].coef_).mean(axis=0)
    logreg_top = (
        pd.DataFrame({"feature": feature_names, "abs_coef": abs_coef})
        .sort_values("abs_coef", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)
    rf_top = (
        pd.DataFrame({
            "feature": feature_names,
            "feature_importance": rf.feature_importances_,
        })
        .sort_values("feature_importance", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
    return logreg_top, rf_top


# ---------------------------------------------------------------------------
# Configuration sensitivity ablation
# ---------------------------------------------------------------------------

def configuration_sensitivity(
    dataset_index: pd.DataFrame,
    configs: list[tuple[str, dict]] = SENSITIVITY_CONFIGS,
    n_splits: int = 4,
) -> pd.DataFrame:
    """Re-run the pipeline under each config and report classifier performance.

    For every named config the full trial table is rebuilt, then a logistic
    regression is evaluated with GroupKFold using recordings as groups. The
    median per-recording RT detection rate is also recorded.
    """
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])

    for name, cfg in configs:
        trial_metrics = build_trial_table(dataset_index, cfg=cfg)
        trial_metrics = preprocess_fillna(trial_metrics)

        rt_found_rate_median = float(
            trial_metrics["rt_found"]
            .groupby(trial_metrics["recording_id"])
            .mean()
            .median()
        )

        X, y, _ = feature_matrix(trial_metrics)
        groups = trial_metrics["recording_id_num"]

        bal_accs: List[float] = []
        f1s: List[float] = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=42)),
            ])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            bal_accs.append(balanced_accuracy_score(y.iloc[test_idx], y_pred))
            f1s.append(f1_score(y.iloc[test_idx], y_pred, average="macro"))

        rows.append({
            "config": name,
            "balanced_accuracy":    float(np.mean(bal_accs)),
            "macro_f1":             float(np.mean(f1s)),
            "rt_found_rate_median": rt_found_rate_median,
        })
    return pd.DataFrame(rows)
