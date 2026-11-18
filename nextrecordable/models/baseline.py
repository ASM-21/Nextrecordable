"""Models over the establishment-year panel, and the metrics that judge them.

Two LightGBM models, trained together with a temporal holdout (train on
years <= T, early-stop and calibrate on T+1, report on T+2):

1. Binary: P(any recordable case next year), isotonic-calibrated.
   Mostly a measure of exposure: a 2,000-person plant almost always logs a
   case, a 30-person shop often doesn't. It answers "will something happen
   here?", which is the loss-control question.

2. Rate: expected next-year cases per 200,000 hours (expected TCR).
   Poisson objective whose offset is log(hours) + log(experience-rated
   rate), where the experience-rated rate is the classic credibility blend
   of the establishment's own last two years with its industry's rate. Size
   enters as exposure rather than as risk, and the trees learn multiplicative
   corrections to the actuarial estimate. Early stopping on the validation
   year decides how much correction the data supports. That alone was not
   enough on real data: corrections that helped on the validation year hurt
   on the next one, so the holdout year now decides which rate is served
   (`TrainResult.rate_served`), and both are always reported. This
   is the size-adjusted ranking (a 40-person shop and a 2,000-person plant
   compare fairly), the underwriting question, and what reports rank by.

Metrics are chosen so they can't flatter the models:
  * Every model is compared with a baseline a skeptic would propose: the
    binary model against a size + NAICS model, the rate model against the
    industry rate and against a credibility-weighted blend of the
    establishment's own history and its industry (classic experience rating).
  * Top-decile lift is reported next to its ceiling, 1 / base rate. When 75%
    of establishments log a case, lift can't exceed 1.33 however good the
    model is, so capture rate (share of next year's cases in the top 10%)
    is the headline for the binary model.
  * The rate model is judged on realized next-year rates using next-year
    hours, by Poisson deviance skill against both baselines and by the pooled
    rate of its top decile relative to everyone.

Categorical features use a vocabulary fitted on the training years and saved
with the model. (Codes used to come from Python's hash(), which is salted per
process: a server restart changed scores for about half of all
establishments.) Unknown categories encode to -1, which LightGBM treats as
missing. Artifacts are LightGBM text files plus JSON; nothing is pickled.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..features import RATE_DENOMINATOR

NUMERIC_FEATURES: tuple[str, ...] = (
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
    "naics4_pooled_tcr",
    "tcr_vs_naics4_median",
)
ENFORCEMENT_FEATURES: tuple[str, ...] = (
    "enf_inspections_2y",
    "enf_unprogrammed_2y",
    "enf_complaint_referral_2y",
    "enf_fatcat_2y",
    "enf_citations_2y",
    "enf_serious_citations_2y",
    "enf_willful_repeat_2y",
    "enf_initial_penalty_2y",
    "enf_max_severity_2y",
    "enf_penalty_per_100_employees_2y",
    "enf_years_since_inspection",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("naics_2", "naics_4", "size_code", "state")
ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + ENFORCEMENT_FEATURES + CATEGORICAL_FEATURES
SIZE_BASELINE_FEATURES: tuple[str, ...] = ("employees", "hours_worked", "naics_2", "naics_4", "size_code")

TARGET = "target_any_recordable_next"
COUNT_TARGET = "target_n_recordable_next"
CREDIBILITY_HOURS = 200_000.0  # 100 full-time worker-years of weight on the industry rate

FEATURE_LABELS: dict[str, str] = {
    "employees": "Average employees",
    "hours_worked": "Hours worked",
    "n_recordable": "Recordable cases this year",
    "n_dafw": "Days-away cases this year",
    "n_djtr": "Restricted/transfer cases this year",
    "n_other": "Other recordable cases this year",
    "tcr": "Total case rate this year",
    "dart_rate": "DART rate this year",
    "tcr_lag1": "Total case rate last year",
    "tcr_lag2": "Total case rate two years ago",
    "dart_lag1": "DART rate last year",
    "n_recordable_lag1": "Recordable cases last year",
    "employees_lag1": "Employees last year",
    "naics4_tcr_median": "Industry median case rate",
    "naics4_dart_median": "Industry median DART rate",
    "naics4_n_estabs": "Industry peer count",
    "naics4_pooled_tcr": "Industry pooled case rate",
    "tcr_vs_naics4_median": "Case rate vs industry median",
    "enf_inspections_2y": "OSHA inspections (2 yrs)",
    "enf_unprogrammed_2y": "Complaint/accident-driven inspections (2 yrs)",
    "enf_complaint_referral_2y": "Complaint or referral inspections (2 yrs)",
    "enf_fatcat_2y": "Fatality/catastrophe inspections (2 yrs)",
    "enf_citations_2y": "Citations (2 yrs)",
    "enf_serious_citations_2y": "Serious, willful or repeat citations (2 yrs)",
    "enf_willful_repeat_2y": "Willful or repeat citations (2 yrs)",
    "enf_initial_penalty_2y": "Initial penalties (2 yrs)",
    "enf_max_severity_2y": "Most severe citation (2 yrs)",
    "enf_penalty_per_100_employees_2y": "Penalties per 100 employees (2 yrs)",
    "enf_years_since_inspection": "Years since last inspection",
    "naics_2": "Industry sector",
    "naics_4": "Industry group",
    "size_code": "Size class",
    "state": "State",
}


# ---------------------------------------------------------------- encoding

@dataclass
class Vocab:
    """Category -> consecutive int codes, fitted on training rows only."""

    mapping: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> Vocab:
        mapping = {}
        for col in columns:
            values = sorted({str(v) for v in frame[col].dropna().tolist() if str(v) != ""})
            mapping[col] = {v: i for i, v in enumerate(values)}
        return cls(mapping)

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, codes in self.mapping.items():
            if col in out.columns:
                out[col] = [codes.get(str(v), -1) if v is not None and v == v else -1 for v in out[col]]
        return out

    def to_json(self) -> dict:
        return self.mapping

    @classmethod
    def from_json(cls, data: dict) -> Vocab:
        return cls({c: {str(k): int(v) for k, v in m.items()} for c, m in data.items()})


@dataclass
class IsotonicMap:
    """Isotonic calibration stored as breakpoints (no pickle). transform == sklearn predict."""

    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, raw: np.ndarray, target: np.ndarray) -> IsotonicMap:
        # Clipped to [0.005, 0.995]: an all-positive top bin in one validation
        # year is not evidence of certainty, and "100%" misleads a reader.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(raw, target)
        return cls([float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_])

    def transform(self, raw) -> np.ndarray:
        return np.interp(np.asarray(raw, dtype=float), self.x, self.y)


# ------------------------------------------------------------------ data

EVAL_COLUMNS: tuple[str, ...] = (
    "estab_key", "year", TARGET, COUNT_TARGET, "n_dart_next", "hours_next",
    "hours_lag1", "naics4_pooled_tcr",
)


def fetch_frame(con: duckdb.DuckDBPyConnection, panel_table: str, where_sql: str) -> pd.DataFrame:
    """Feature + evaluation columns for panel rows matching `where_sql`.

    Columns missing from the panel (e.g. enforcement features on an
    un-enforced panel) come back as NULL, which LightGBM treats as missing.
    """
    available = {r[1] for r in con.execute(f"PRAGMA table_info('{panel_table}')").fetchall()}
    cols = []
    for c in dict.fromkeys(EVAL_COLUMNS + ALL_FEATURES):
        if c in available:
            cols.append(f'"{c}"')
        elif c in CATEGORICAL_FEATURES:
            cols.append(f"CAST(NULL AS VARCHAR) AS {c}")
        else:
            cols.append(f"CAST(NULL AS DOUBLE) AS {c}")
    return con.execute(f"SELECT {', '.join(cols)} FROM {panel_table} WHERE {where_sql}").df()


def _matrix(frame: pd.DataFrame, vocab: Vocab, features: tuple[str, ...]) -> np.ndarray:
    enc = vocab.encode(frame[list(features)])
    return enc.astype(float).to_numpy()


def _cat_idx(features: tuple[str, ...]) -> list[int]:
    return [i for i, f in enumerate(features) if f in CATEGORICAL_FEATURES]


def _default_params(objective: str) -> dict:
    return {
        "objective": objective,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "cat_smooth": 10,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
    }


# ---------------------------------------------------------------- metrics

def capture_rate(score: np.ndarray, weight: np.ndarray, k: float = 0.10) -> float:
    """Share of total `weight` (e.g. next-year cases) held by the top-k fraction by `score`."""
    weight = np.asarray(weight, dtype=float)
    total = weight.sum()
    if len(score) == 0 or total <= 0:
        return float("nan")
    n_top = max(1, round(len(score) * k))
    top = np.argsort(-np.asarray(score), kind="stable")[:n_top]
    return float(weight[top].sum() / total)


def lift_at_top_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.10) -> float:
    """Positive rate in the top-k fraction divided by the overall positive rate."""
    if len(y_true) == 0:
        return float("nan")
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    n_top = max(1, round(len(y_true) * k))
    top = np.argsort(-np.asarray(y_pred), kind="stable")[:n_top]
    return float(np.mean(np.asarray(y_true)[top]) / base)


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-count bins by prediction: mean predicted vs observed rate."""
    if len(y_true) == 0:
        return []
    order = np.argsort(y_pred, kind="stable")
    yt, yp = np.asarray(y_true)[order], np.asarray(y_pred)[order]
    edges = np.linspace(0, len(yt), n_bins + 1, dtype=int)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            out.append({
                "bin": b + 1,
                "n": int(hi - lo),
                "mean_predicted": float(yp[lo:hi].mean()),
                "observed_rate": float(yt[lo:hi].mean()),
                "calibration_error": float(abs(yp[lo:hi].mean() - yt[lo:hi].mean())),
            })
    return out


def expected_calibration_error(table: list[dict]) -> float:
    """Count-weighted mean |predicted - observed| over calibration_table() bins."""
    n = sum(b["n"] for b in table)
    return float(sum(b["n"] * b["calibration_error"] for b in table) / n) if n else float("nan")


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def binary_metrics(frame: pd.DataFrame, p_model: np.ndarray, p_baseline: np.ndarray | None) -> dict:
    y = frame[TARGET].to_numpy(dtype=int)
    cases = frame[COUNT_TARGET].fillna(0).to_numpy(dtype=float)
    dart = frame["n_dart_next"].fillna(0).to_numpy(dtype=float)
    base = float(y.mean()) if len(y) else float("nan")
    out = {
        "n": len(y),
        "base_rate": base,
        "auc": float(roc_auc_score(y, p_model)) if 0 < base < 1 else float("nan"),
        "brier": float(brier_score_loss(y, p_model)),
        # Brier score of always predicting the base rate, and skill against it.
        "brier_reference": base * (1 - base),
        "brier_skill": (1 - float(brier_score_loss(y, p_model)) / (base * (1 - base))) if 0 < base < 1 else float("nan"),
        "logloss": float(log_loss(y, np.clip(p_model, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "lift_top10": lift_at_top_k(y, p_model),
        "lift_ceiling": (1.0 / base) if base > 0 else float("nan"),
        "capture_cases_top10": capture_rate(p_model, cases),
        "capture_dart_top10": capture_rate(p_model, dart),
    }
    if p_baseline is not None:
        out["baseline_auc"] = float(roc_auc_score(y, p_baseline)) if 0 < base < 1 else float("nan")
        out["baseline_capture_cases_top10"] = capture_rate(p_baseline, cases)
        out["baseline_capture_dart_top10"] = capture_rate(p_baseline, dart)
    return out


def baseline_rates(frame: pd.DataFrame, fallback_rate_per_hour: float) -> dict[str, np.ndarray]:
    """Per-hour rates from the two baselines a skeptic would propose."""
    ind = (frame["naics4_pooled_tcr"].astype(float) / RATE_DENOMINATOR).to_numpy()
    ind = np.where(np.isfinite(ind) & (ind > 0), ind, fallback_rate_per_hour)
    # A year counts only when its hours are usable: cases without a denominator
    # would be added to the numerator alone. (A real filer with 1,923 keyed-in
    # cases and rejected hours got an "experience-rated" rate of 1,900.)
    own = (frame["hours_worked"].fillna(0) > 0).to_numpy()
    lag = (frame["n_recordable_lag1"].notna() & (frame["hours_lag1"].fillna(0) > 0)).to_numpy()
    cases = (np.where(own, frame["n_recordable"].fillna(0).to_numpy(float), 0.0)
             + np.where(lag, frame["n_recordable_lag1"].fillna(0).to_numpy(float), 0.0))
    hours = (np.where(own, frame["hours_worked"].fillna(0).to_numpy(float), 0.0)
             + np.where(lag, frame["hours_lag1"].fillna(0).to_numpy(float), 0.0))
    cred = (cases + CREDIBILITY_HOURS * ind) / (hours + CREDIBILITY_HOURS)
    return {"industry": ind, "credibility": cred}


def credibility_rate(frame: pd.DataFrame, fallback_rate_per_hour: float) -> np.ndarray:
    """Experience-rated cases per hour: own two years blended with the industry rate."""
    return baseline_rates(frame, fallback_rate_per_hour)["credibility"]


def rate_metrics(frame: pd.DataFrame, rate_model: np.ndarray, fallback_rate_per_hour: float) -> dict:
    """Judge per-hour rate forecasts on realized next-year cases over next-year hours."""
    mask = ((frame["hours_next"].fillna(0) > 0).to_numpy() & frame[COUNT_TARGET].notna().to_numpy()
            & (frame["hours_worked"].fillna(0) > 0).to_numpy())
    f = frame[mask]
    if len(f) == 0:
        return {"n": 0}
    y = f[COUNT_TARGET].to_numpy(float)
    h = f["hours_next"].to_numpy(float)
    preds = {"model": np.asarray(rate_model)[mask], **{
        k: v for k, v in baseline_rates(f, fallback_rate_per_hour).items()
    }}
    realized_tcr = y * RATE_DENOMINATOR / h
    pooled_all = y.sum() / h.sum()
    out: dict[str, Any] = {"n": len(f), "realized_pooled_tcr": float(pooled_all * RATE_DENOMINATOR)}
    for name, rate in preds.items():
        mu = rate * h
        n_top = max(1, round(len(f) * 0.10))
        top = np.argsort(-rate, kind="stable")[:n_top]
        out[name] = {
            "poisson_deviance": poisson_deviance(y, mu),
            "spearman_vs_realized_tcr": float(spearmanr(rate, realized_tcr).statistic),
            "top10_pooled_rate_lift": float((y[top].sum() / h[top].sum()) / pooled_all) if pooled_all > 0 else float("nan"),
            "predicted_over_actual_cases": float(mu.sum() / y.sum()) if y.sum() > 0 else float("nan"),
        }
    # Triage view: rank by expected CASES (rate x this year's hours) and ask
    # what share of next year's cases the top 10% hold. "Largest sites first"
    # is the size-only answer to beat.
    cur_hours = f["hours_worked"].fillna(0).to_numpy(float)
    for name, rate in preds.items():
        out[name]["capture_cases_top10"] = capture_rate(rate * cur_hours, y)
    out["largest_sites_capture_cases_top10"] = capture_rate(cur_hours, y)
    for b in ("industry", "credibility"):
        out[f"deviance_skill_vs_{b}"] = 1.0 - out["model"]["poisson_deviance"] / out[b]["poisson_deviance"]
    return out


# ------------------------------------------------------------------ model

@dataclass
class TrainResult:
    """Trained models plus everything needed to reproduce their scores."""

    binary: lgb.Booster
    calibrator: IsotonicMap | None
    rate: lgb.Booster | None
    vocab: Vocab
    train_through: int
    val_year: int
    test_year: int
    panel_table: str = "ita_panel"
    # Pooled training-year cases per hour; the industry fallback inside the
    # credibility estimate when a row has no NAICS rate.
    fallback_rate_per_hour: float = 0.0
    # Which rate reports and scores use: "model" (boosted corrections) or
    # "credibility" (the experience-rated estimate alone). Decided on the
    # holdout year at training time: the boosted model is served only if it
    # beats the estimate it starts from. On the first real run it did not.
    rate_served: str = "model"
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURES))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration_table: list[dict] = field(default_factory=list)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    rate_feature_importance: list[tuple[str, float]] = field(default_factory=list)
    # Serializes Booster.predict calls from server threads.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def model(self) -> lgb.Booster:
        """The binary model (older call sites use `.model`)."""
        return self.binary

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return _matrix(frame, self.vocab, tuple(self.feature_names))

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """p_raw, p (calibrated), rate_per_hour, and the experience-rated baseline it adjusts."""
        X = self.matrix(frame)
        with self._lock:
            p_raw = self.binary.predict(X)
            raw_rate = self.rate.predict(X, raw_score=True) if self.rate is not None else None
        p = self.calibrator.transform(p_raw) if self.calibrator else p_raw
        cred = credibility_rate(frame, self.fallback_rate_per_hour)
        boosted = cred * np.exp(raw_rate) if raw_rate is not None else np.full(len(frame), np.nan)
        rate = boosted if self.rate_served == "model" else cred
        return {"p_raw": p_raw, "p": p, "rate_per_hour": rate, "baseline_rate_per_hour": cred,
                "boosted_rate_per_hour": boosted}

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Per-feature contributions: log-rate (rate model) or log-odds (binary)."""
        X = self.matrix(frame)
        with self._lock:
            if self.rate is not None and self.rate_served == "model":
                return self.rate.predict(X, pred_contrib=True)[:, :-1], "log_rate"
            return self.binary.predict(X, pred_contrib=True)[:, :-1], "log_odds"


def _fit(params, X, y, Xv, yv, cat_idx, names, rounds, stop, init=None, init_v=None) -> lgb.Booster:
    train_ds = lgb.Dataset(X, label=y, init_score=init, categorical_feature=cat_idx,
                           feature_name=names, free_raw_data=False)
    val_ds = lgb.Dataset(Xv, label=yv, init_score=init_v, reference=train_ds, categorical_feature=cat_idx)
    return lgb.train(
        params, train_ds, num_boost_round=rounds, valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(period=0)],
    )


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
    rate_model: bool = True,
    baselines: bool = True,
) -> TrainResult:
    """Train binary (+ rate) models with a temporal holdout and compute metrics."""
    val_year = val_year if val_year is not None else train_through + 1
    test_year = test_year if test_year is not None else train_through + 2
    where = f"{TARGET} IS NOT NULL"
    train = fetch_frame(con, panel_table, f"{where} AND year <= {int(train_through)}")
    val = fetch_frame(con, panel_table, f"{where} AND year = {int(val_year)}")
    test = fetch_frame(con, panel_table, f"{where} AND year = {int(test_year)}")
    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            f"Empty split: train n={len(train)}, val n={len(val)}. "
            "Check year boundaries against the panel's years."
        )

    vocab = Vocab.fit(train, CATEGORICAL_FEATURES)
    feats = ALL_FEATURES
    cat_idx = _cat_idx(feats)
    X, Xv = _matrix(train, vocab, feats), _matrix(val, vocab, feats)
    y, yv = train[TARGET].to_numpy(int), val[TARGET].to_numpy(int)

    bparams = {**_default_params("binary"), "metric": ["binary_logloss"], **(lgb_params or {})}
    binary = _fit(bparams, X, y, Xv, yv, cat_idx, list(feats), num_boost_round, early_stopping_rounds)
    raw_v = binary.predict(Xv)
    calibrator = IsotonicMap.fit(raw_v, yv) if calibrate else None

    def cal(raw):
        return calibrator.transform(raw) if calibrator else raw

    rate = None
    m_tr = (train["hours_worked"].fillna(0) > 0).to_numpy()
    m_va = (val["hours_worked"].fillna(0) > 0).to_numpy()
    fallback = float(
        train[COUNT_TARGET].to_numpy(float)[m_tr].sum()
        / max(train["hours_worked"].to_numpy(float)[m_tr].sum(), 1.0)
    )
    if rate_model and m_tr.sum() > 0 and m_va.sum() > 0 and fallback > 0:
        rparams = {**_default_params("poisson"), "metric": ["poisson"], **(lgb_params or {})}
        rparams["objective"] = "poisson"
        cred_tr = credibility_rate(train, fallback)[m_tr]
        cred_va = credibility_rate(val, fallback)[m_va]
        rate = _fit(
            rparams, X[m_tr], train[COUNT_TARGET].to_numpy(float)[m_tr],
            Xv[m_va], val[COUNT_TARGET].to_numpy(float)[m_va], cat_idx, list(feats),
            num_boost_round, early_stopping_rounds,
            init=np.log(train["hours_worked"].to_numpy(float)[m_tr]) + np.log(cred_tr),
            init_v=np.log(val["hours_worked"].to_numpy(float)[m_va]) + np.log(cred_va),
        )

    result = TrainResult(
        binary=binary, calibrator=calibrator, rate=rate, vocab=vocab,
        train_through=int(train_through), val_year=int(val_year), test_year=int(test_year),
        panel_table=panel_table, fallback_rate_per_hour=fallback,
    )

    metrics: dict[str, Any] = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_base_rate": float(y.mean()), "val_base_rate": float(yv.mean()),
        "binary_best_iteration": int(binary.best_iteration),
        "rate_best_iteration": int(rate.best_iteration) if rate is not None else None,
        "val_auc_raw": float(roc_auc_score(yv, raw_v)) if 0 < yv.mean() < 1 else float("nan"),
        "val_brier_raw": float(brier_score_loss(yv, raw_v)),
        "val_brier_calibrated": float(brier_score_loss(yv, cal(raw_v))),
    }
    result.calibration_table = calibration_table(yv, cal(raw_v))

    size_model = None
    if baselines:
        size_feats = SIZE_BASELINE_FEATURES
        size_model = _fit(
            bparams, _matrix(train, vocab, size_feats), y, _matrix(val, vocab, size_feats), yv,
            _cat_idx(size_feats), list(size_feats), num_boost_round, early_stopping_rounds,
        )

    if len(test):
        preds = result.predict(test)
        p_base = size_model.predict(_matrix(test, vocab, SIZE_BASELINE_FEATURES)) if size_model else None
        metrics["test_binary"] = binary_metrics(test, preds["p"], p_base)
        # Reliability on the holdout year itself (the calibrator was fitted on the year before).
        yt = test[TARGET].to_numpy(int)
        metrics["test_calibration"] = calibration_table(yt, preds["p"])
        metrics["test_ece"] = expected_calibration_error(metrics["test_calibration"])
        # The same establishment appears in training years and in the holdout
        # year. Scoring only establishments the model never saw shows how much
        # of the headline is recognition rather than prediction.
        known = set(train["estab_key"]) | set(val["estab_key"])
        seen = test["estab_key"].isin(known).to_numpy()
        for label, mask in (("seen", seen), ("unseen", ~seen)):
            if mask.sum() >= 50 and 0 < yt[mask].mean() < 1:
                metrics[f"test_binary_{label}_establishments"] = binary_metrics(
                    test[mask], preds["p"][mask], p_base[mask] if p_base is not None else None)
        if rate is not None:
            metrics["test_rate"] = rate_metrics(test, preds["boosted_rate_per_hour"], fallback)
            skill = metrics["test_rate"].get("deviance_skill_vs_credibility")
            if skill is not None and not skill > 0:
                result.rate_served = "credibility"
        metrics["rate_served"] = result.rate_served
        # Flat aliases kept for older readers of meta.json.
        metrics["test_auc"] = metrics["test_binary"]["auc"]
        metrics["test_brier"] = metrics["test_binary"]["brier"]
        metrics["test_base_rate"] = metrics["test_binary"]["base_rate"]
    result.metrics = metrics

    result.feature_importance = sorted(
        zip(feats, (float(g) for g in binary.feature_importance(importance_type="gain")), strict=True),
        key=lambda kv: -kv[1],
    )
    if rate is not None:
        result.rate_feature_importance = sorted(
            zip(feats, (float(g) for g in rate.feature_importance(importance_type="gain")), strict=True),
            key=lambda kv: -kv[1],
        )
    return result


# ---------------------------------------------------------------- scoring

RISK_BANDS: tuple[tuple[float, str], ...] = (
    (0.20, "much lower than peers"),
    (0.40, "lower than peers"),
    (0.60, "similar to peers"),
    (0.80, "higher than peers"),
    (1.01, "much higher than peers"),
)
MIN_PEERS = 20


def risk_band(pctile: float | None) -> str | None:
    if pctile is None or pctile != pctile:
        return None
    for upper, label in RISK_BANDS:
        if pctile < upper:
            return label
    return RISK_BANDS[-1][1]


def materialize_scores(
    con: duckdb.DuckDBPyConnection,
    result: TrainResult,
    *,
    panel_table: str | None = None,
    output: str = "estab_scores",
) -> int:
    """Score every establishment's latest filing into `output` (one row per estab_key).

    The API reads scores from this table, so single lookups, batch uploads and
    PDF reports always agree. Peer percentiles compare expected TCR within the
    NAICS-4 group, falling back to NAICS-2 and then to everyone when a group
    has fewer than MIN_PEERS establishments.
    """
    panel = panel_table or result.panel_table
    frame = fetch_frame(
        con, panel,
        "TRUE QUALIFY ROW_NUMBER() OVER (PARTITION BY estab_key ORDER BY year DESC) = 1",
    )
    preds = result.predict(frame)

    def nullable(values) -> pd.arrays.FloatingArray:
        # NaN -> NULL. DuckDB keeps a pandas NaN as NaN, which COUNT() counts and
        # cume_dist() ranks; a nullable Float64 array becomes a real NULL.
        return pd.array(np.asarray(values, dtype=float), dtype="Float64")

    scores = pd.DataFrame({
        "estab_key": frame["estab_key"],
        "score_year": frame["year"].astype(int),
        "p_any_next": nullable(preds["p"]),
        "expected_tcr_next": nullable(preds["rate_per_hour"] * RATE_DENOMINATOR),
        "expected_cases_next": nullable(preds["rate_per_hour"] * frame["hours_worked"].astype(float)),
        "experience_rated_tcr_next": nullable(preds["baseline_rate_per_hour"] * RATE_DENOMINATOR),
        "naics_4": frame["naics_4"].fillna("").astype(str),
        "naics_2": frame["naics_2"].fillna("").astype(str),
    })
    con.register("_scores_df", scores)
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        WITH g AS (
            SELECT *,
                   COUNT(expected_tcr_next) OVER (PARTITION BY naics_4) AS n4,
                   COUNT(expected_tcr_next) OVER (PARTITION BY naics_2) AS n2,
                   CASE WHEN expected_tcr_next IS NOT NULL THEN
                       cume_dist() OVER (PARTITION BY naics_4, expected_tcr_next IS NULL ORDER BY expected_tcr_next) END AS c4,
                   CASE WHEN expected_tcr_next IS NOT NULL THEN
                       cume_dist() OVER (PARTITION BY naics_2, expected_tcr_next IS NULL ORDER BY expected_tcr_next) END AS c2,
                   CASE WHEN expected_tcr_next IS NOT NULL THEN
                       cume_dist() OVER (PARTITION BY expected_tcr_next IS NULL ORDER BY expected_tcr_next) END AS c_all
            FROM _scores_df
        )
        SELECT estab_key, score_year, p_any_next, expected_tcr_next, expected_cases_next,
               experience_rated_tcr_next,
               CASE WHEN n4 >= {MIN_PEERS} AND naics_4 <> '' THEN c4
                    WHEN n2 >= {MIN_PEERS} AND naics_2 <> '' THEN c2 ELSE c_all END AS peer_pctile,
               CASE WHEN n4 >= {MIN_PEERS} AND naics_4 <> '' THEN 'NAICS ' || naics_4
                    WHEN n2 >= {MIN_PEERS} AND naics_2 <> '' THEN 'NAICS ' || naics_2
                    ELSE 'all industries' END AS peer_group,
               CASE WHEN n4 >= {MIN_PEERS} AND naics_4 <> '' THEN n4
                    WHEN n2 >= {MIN_PEERS} AND naics_2 <> '' THEN n2
                    ELSE (SELECT COUNT(expected_tcr_next) FROM _scores_df) END AS peer_count,
               c_all AS overall_pctile
        FROM g
        """
    )
    con.unregister("_scores_df")
    bands = con.execute(f"SELECT estab_key, peer_pctile FROM {output}").fetchall()
    band_df = pd.DataFrame({"estab_key": [b[0] for b in bands],
                            "risk_band": [risk_band(b[1]) for b in bands]})
    con.register("_bands_df", band_df)
    con.execute(f"ALTER TABLE {output} ADD COLUMN risk_band VARCHAR")
    con.execute(f"UPDATE {output} SET risk_band = b.risk_band FROM _bands_df b WHERE {output}.estab_key = b.estab_key")
    con.unregister("_bands_df")
    return con.execute(f"SELECT COUNT(*) FROM {output}").fetchone()[0]


# ------------------------------------------------------------- persistence

def save_model(result: TrainResult, dest_dir: Path) -> None:
    """Write LightGBM text models + meta.json (vocab, calibration, metrics)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result.binary.save_model(str(dest / "binary.txt"))
    if result.rate is not None:
        result.rate.save_model(str(dest / "rate.txt"))
    elif (dest / "rate.txt").exists():
        (dest / "rate.txt").unlink()
    meta = {
        "format": 2,
        "feature_names": result.feature_names,
        "categorical_features": result.categorical_features,
        "vocab": result.vocab.to_json(),
        "calibrator": {"x": result.calibrator.x, "y": result.calibrator.y} if result.calibrator else None,
        "train_through": result.train_through,
        "val_year": result.val_year,
        "test_year": result.test_year,
        "panel_table": result.panel_table,
        "fallback_rate_per_hour": result.fallback_rate_per_hour,
        "rate_served": result.rate_served,
        "rate_offset": "log(hours) + log(credibility rate)",
        "metrics": result.metrics,
        "calibration_table": result.calibration_table,
        "feature_importance": result.feature_importance,
        "rate_feature_importance": result.rate_feature_importance,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


def load_model(model_dir: Path) -> TrainResult:
    """Rebuild a TrainResult from save_model() output. Scores match the training process exactly."""
    src = Path(model_dir)
    if not (src / "binary.txt").exists():
        raise FileNotFoundError(f"No binary.txt in {src}. Train first (cli train).")
    meta = json.loads((src / "meta.json").read_text())
    cal = meta.get("calibrator")
    return TrainResult(
        binary=lgb.Booster(model_file=str(src / "binary.txt")),
        calibrator=IsotonicMap(cal["x"], cal["y"]) if cal else None,
        rate=lgb.Booster(model_file=str(src / "rate.txt")) if (src / "rate.txt").exists() else None,
        vocab=Vocab.from_json(meta["vocab"]),
        train_through=meta["train_through"],
        val_year=meta["val_year"],
        test_year=meta["test_year"],
        panel_table=meta.get("panel_table", "ita_panel"),
        fallback_rate_per_hour=meta.get("fallback_rate_per_hour", 0.0),
        rate_served=meta.get("rate_served", "model"),
        feature_names=meta["feature_names"],
        categorical_features=meta["categorical_features"],
        metrics=meta.get("metrics", {}),
        calibration_table=meta.get("calibration_table", []),
        feature_importance=[tuple(x) for x in meta.get("feature_importance", [])],
        rate_feature_importance=[tuple(x) for x in meta.get("rate_feature_importance", [])],
    )
