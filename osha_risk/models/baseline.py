"""Baseline LightGBM model: P(any recordable injury next year).

This module is deliberately the SIMPLEST useful model: gradient-boosted trees
on the panel features we already built. We treat NAICS as a native categorical
feature (LightGBM handles this without one-hot encoding).

Validation methodology:
  Temporal holdout. Train on year <= train_through; validate on
  year == train_through + 1; test on year == train_through + 2.
  Random splits would leak information across time and inflate metrics.

Calibration:
  Raw GBT outputs are decent rankers but often miscalibrated. We fit an
  isotonic regression on the validation set to map raw scores -> calibrated
  probabilities. The test set gets the calibrated scores.

What we report:
  - AUC (rank quality)
  - Brier score (calibration + sharpness)
  - Calibration table by decile (predicted vs observed rate)
  - Lift at top decile (the metric brokers actually care about:
    "if I focus on the top 10% of my book, what fraction of next year's
    incidents do I catch?")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Features used by the model. Categorical columns are flagged at train time.
NUMERIC_FEATURES: tuple[str, ...] = (
    # Self-reported injury features (from ITA panel)
    "employees",
    "hours_worked",
    "n_recordable",
    "n_dafw",
    "n_djtr",
    "n_other",
    "tcr",
    "dart_rate",
    "tcr_lag1",
    "tcr_lag2",
    "dart_lag1",
    "n_recordable_lag1",
    "employees_lag1",
    "naics4_tcr_median",
    "naics4_dart_median",
    "naics4_n_estabs",
    "tcr_vs_naics4_median",
)
# Enforcement features from DOL inspection/violation data (lagged 1 year)
# These are joined onto the panel by features.attach_enforcement_to_panel.
ENFORCEMENT_FEATURES: tuple[str, ...] = (
    "enf_n_inspections_lag1",
    "enf_n_inspections_unprog_lag1",
    "enf_n_citations_lag1",
    "enf_n_serious_citations_lag1",
    "enf_n_willful_citations_lag1",
    "enf_total_penalty_lag1",
    "enf_max_severity_lag1",
    "enf_had_fatality_lag1",
    "enf_penalty_lag1to2",
    "enf_penalty_per_employee_lag1",
    "enf_inspected_lag1",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "naics_2",
    "naics_4",
    "size_code",
    "state",
)
ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + ENFORCEMENT_FEATURES + CATEGORICAL_FEATURES

TARGET: str = "target_any_recordable_next"


@dataclass
class TrainResult:
    """Container for a trained model and its evaluation artifacts."""

    model: lgb.Booster
    calibrator: IsotonicRegression | None
    feature_names: list[str]
    categorical_features: list[str]
    train_through: int
    val_year: int
    test_year: int
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration_table: list[dict] = field(default_factory=list)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)


def _fetch_split(
    con: duckdb.DuckDBPyConnection,
    year_filter_sql: str,
    panel_table: str = "ita_panel",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Pull features, target, and estab_key from one panel slice.

    Returns:
        features dict {name: array}
        target array
        estab_key array (for QA / report joining)
    """
    # Tolerate panels that don't have enforcement columns (e.g. ita_panel built
    # before enforcement.attach_enforcement_to_panel was run). Missing cols
    # are SELECTed as NULL so the model sees them as missing data, which
    # LightGBM handles natively.
    available_cols = {
        r[1]
        for r in con.execute(f"PRAGMA table_info('{panel_table}')").fetchall()
    }
    feature_select_parts: list[str] = []
    for col in ALL_FEATURES:
        if col in available_cols:
            feature_select_parts.append(col)
        else:
            feature_select_parts.append(f"CAST(NULL AS DOUBLE) AS {col}")
    feature_cols = ",\n  ".join(feature_select_parts)

    sql = f"""
        SELECT
          estab_key,
          {feature_cols},
          {TARGET} AS y
        FROM {panel_table}
        WHERE {TARGET} IS NOT NULL
          AND ({year_filter_sql})
    """
    res = con.execute(sql)
    rows = res.fetchnumpy()

    features: dict[str, np.ndarray] = {}
    for col in ALL_FEATURES:
        arr = rows[col]
        if col in CATEGORICAL_FEATURES:
            # LightGBM categorical features must be integer-coded.
            arr = _to_categorical_codes(arr)
        else:
            arr = np.asarray(arr, dtype=np.float64)
        features[col] = arr

    y = np.asarray(rows["y"], dtype=np.int32)
    estab_key = np.asarray(rows["estab_key"])
    return features, y, estab_key


def _to_categorical_codes(arr: np.ndarray) -> np.ndarray:
    """Map an object/str array to integer codes. Missing -> -1.

    LightGBM expects integer-coded categoricals. We do NOT persist the
    code mapping across train/val/test; instead we build a single combined
    mapping by hashing values modulo a large prime, which is stable across
    splits without needing to track a vocabulary file.
    """
    arr = np.asarray(arr)
    out = np.empty(len(arr), dtype=np.int32)
    for i, v in enumerate(arr):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            out[i] = -1
        else:
            # Stable integer hash mod a prime. Collisions are rare and harmless
            # for tree models (they'll just learn to split at the collision boundary).
            out[i] = hash(str(v)) % 100_003
    return out


def _features_to_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Stack feature dict into an (n_samples, n_features) float matrix."""
    cols = [features[name].astype(np.float64) for name in ALL_FEATURES]
    return np.column_stack(cols)


def train_baseline(
    con: duckdb.DuckDBPyConnection,
    *,
    train_through: int,
    val_year: int | None = None,
    test_year: int | None = None,
    panel_table: str = "ita_panel",
    lgb_params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    calibrate: bool = True,
) -> TrainResult:
    """Train the baseline model with temporal holdout.

    Args:
        train_through: include panel years <= this in training
        val_year: validation year (default: train_through + 1)
        test_year: test year (default: train_through + 2)
        calibrate: fit isotonic calibration on validation set
    """
    val_year = val_year if val_year is not None else train_through + 1
    test_year = test_year if test_year is not None else train_through + 2

    train_features, y_train, _ = _fetch_split(
        con, f"year <= {train_through}", panel_table
    )
    val_features, y_val, _ = _fetch_split(
        con, f"year = {val_year}", panel_table
    )
    test_features, y_test, _ = _fetch_split(
        con, f"year = {test_year}", panel_table
    )

    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError(
            f"Empty split: train n={len(y_train)}, val n={len(y_val)}. "
            f"Check year boundaries against panel coverage."
        )

    X_train = _features_to_matrix(train_features)
    X_val = _features_to_matrix(val_features)
    X_test = _features_to_matrix(test_features) if len(y_test) else None

    # Identify categorical column indices for LightGBM
    cat_idx = [i for i, name in enumerate(ALL_FEATURES) if name in CATEGORICAL_FEATURES]

    default_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    params = {**default_params, **(lgb_params or {})}

    train_ds = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_idx, feature_name=list(ALL_FEATURES)
    )
    val_ds = lgb.Dataset(
        X_val, label=y_val, reference=train_ds, categorical_feature=cat_idx,
    )

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Raw predictions on val and test
    raw_val = booster.predict(X_val)
    raw_test = booster.predict(X_test) if X_test is not None else None

    # Calibration
    calibrator: IsotonicRegression | None = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_val, y_val)
        cal_val = calibrator.transform(raw_val)
        cal_test = calibrator.transform(raw_test) if raw_test is not None else None
    else:
        cal_val = raw_val
        cal_test = raw_test

    metrics = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test) if X_test is not None else 0,
        "train_base_rate": float(y_train.mean()),
        "val_base_rate": float(y_val.mean()),
        "val_auc_raw": float(roc_auc_score(y_val, raw_val)),
        "val_brier_raw": float(brier_score_loss(y_val, raw_val)),
        "val_logloss_raw": float(log_loss(y_val, raw_val)),
        "val_auc_calibrated": float(roc_auc_score(y_val, cal_val)),
        "val_brier_calibrated": float(brier_score_loss(y_val, cal_val)),
        "best_iteration": int(booster.best_iteration),
    }
    if X_test is not None and len(y_test) > 0:
        metrics["test_base_rate"] = float(y_test.mean())
        metrics["test_auc"] = float(roc_auc_score(y_test, cal_test))
        metrics["test_brier"] = float(brier_score_loss(y_test, cal_test))
        metrics["test_logloss"] = float(log_loss(y_test, cal_test))
        metrics["test_lift_top_decile"] = lift_at_top_k(y_test, cal_test, k=0.10)

    cal_table = calibration_table(y_val, cal_val)

    importance_pairs = sorted(
        zip(ALL_FEATURES, booster.feature_importance(importance_type="gain"), strict=True),
        key=lambda x: -x[1],
    )

    return TrainResult(
        model=booster,
        calibrator=calibrator,
        feature_names=list(ALL_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        train_through=train_through,
        val_year=val_year,
        test_year=test_year,
        metrics=metrics,
        calibration_table=cal_table,
        feature_importance=[(n, float(v)) for n, v in importance_pairs],
    )


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Decile calibration table: predicted vs observed rate per bin.

    Bins are by predicted-probability quantile (equal-count). For each bin:
      - mean predicted probability
      - observed rate
      - absolute calibration error (|pred - obs|)
      - count
    """
    if len(y_true) == 0:
        return []
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    bin_edges = np.linspace(0, len(y_true), n_bins + 1, dtype=int)

    out: list[dict] = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if hi == lo:
            continue
        bin_pred = y_pred_sorted[lo:hi]
        bin_true = y_true_sorted[lo:hi]
        out.append(
            {
                "bin": b + 1,
                "n": int(hi - lo),
                "mean_predicted": float(bin_pred.mean()),
                "observed_rate": float(bin_true.mean()),
                "calibration_error": float(abs(bin_pred.mean() - bin_true.mean())),
            }
        )
    return out


def lift_at_top_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.10) -> float:
    """Lift over base rate when targeting the top-k fraction by predicted score.

    lift = (positive rate in top-k) / (overall positive rate)

    For brokers: a lift of 3.0 at k=0.10 means "if you focus on the top 10% of
    your book, you'll find positives at 3x the average rate."
    """
    if len(y_true) == 0:
        return float("nan")
    n_top = max(1, int(len(y_true) * k))
    top_idx = np.argsort(-y_pred)[:n_top]
    top_rate = float(y_true[top_idx].mean())
    base_rate = float(y_true.mean())
    if base_rate == 0:
        return float("nan")
    return top_rate / base_rate


def save_model(result: TrainResult, dest_dir: Path) -> None:
    """Persist model + metadata to a directory."""
    import json

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(dest_dir / "model.txt"))

    meta = {
        "feature_names": result.feature_names,
        "categorical_features": result.categorical_features,
        "train_through": result.train_through,
        "val_year": result.val_year,
        "test_year": result.test_year,
        "metrics": result.metrics,
        "calibration_table": result.calibration_table,
        "feature_importance": result.feature_importance,
    }
    with open(dest_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if result.calibrator is not None:
        import pickle
        with open(dest_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(result.calibrator, f)


def load_model(model_dir: Path) -> TrainResult:
    """Reconstruct a TrainResult from artifacts written by save_model.

    This is what the API server calls at startup: train once (offline,
    scheduled), save artifacts, then load-and-serve without retraining.
    The returned TrainResult has a real Booster and calibrator; it's safe to
    pass directly into report.risk_report.
    """
    import json
    import pickle

    model_dir = Path(model_dir)
    if not (model_dir / "model.txt").exists():
        raise FileNotFoundError(
            f"No model.txt found in {model_dir}. Train and save a model first "
            f"with models.baseline.train_baseline() + save_model()."
        )

    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    calibrator = None
    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    return TrainResult(
        model=booster,
        calibrator=calibrator,
        feature_names=meta.get("feature_names", list(ALL_FEATURES)),
        categorical_features=meta.get("categorical_features", list(CATEGORICAL_FEATURES)),
        train_through=meta.get("train_through", -1),
        val_year=meta.get("val_year", -1),
        test_year=meta.get("test_year", -1),
        metrics=meta.get("metrics", {}),
        calibration_table=meta.get("calibration_table", []),
        feature_importance=[tuple(x) for x in meta.get("feature_importance", [])],
    )
