"""Attempts to break the headline metrics. Writes reports/falsification.json.

FINDINGS.md quotes these numbers; tests/test_docs.py checks that it quotes
them correctly. Each check asks one question a skeptic would ask:

  size_bands       Is the binary AUC just "big sites log cases"? AUC inside
                   employee-count bands, model vs the size + NAICS baseline,
                   plus "rank by hours worked" and "had a case this year".
  identity         Does the "unseen establishment" result survive when an
                   establishment counts as seen if its OSHA establishment_id
                   was seen under a different name or address?
  other_years      Is the holdout year special? Retrain with each earlier
                   year as the test year.
  denominators     How much do tiny sites drive the rate results and the
                   SST percentile positions?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .models.baseline import (
    COUNT_TARGET,
    SIZE_BASELINE_FEATURES,
    TARGET,
    TrainResult,
    _cat_idx,
    _default_params,
    _fit,
    _matrix,
    fetch_frame,
    rate_metrics,
    train_baseline,
)

EMPLOYEE_BANDS: tuple[tuple[int, int | None], ...] = ((1, 19), (20, 49), (50, 99), (100, 249), (250, 999), (1000, None))
HOURS_BANDS: tuple[tuple[int, int | None], ...] = ((1, 40_000), (40_000, 200_000), (200_000, 1_000_000), (1_000_000, None))


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(y) < 50 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def _band_label(lo: int, hi: int | None) -> str:
    return f"{lo:,}+" if hi is None else f"{lo:,}-{hi:,}"


def _size_baseline(con, result: TrainResult, panel: str) -> Any:
    where = f"{TARGET} IS NOT NULL"
    train = fetch_frame(con, panel, f"{where} AND year <= {result.train_through}")
    val = fetch_frame(con, panel, f"{where} AND year = {result.val_year}")
    params = {**_default_params("binary"), "metric": ["binary_logloss"]}
    feats = SIZE_BASELINE_FEATURES
    return _fit(params, _matrix(train, result.vocab, feats), train[TARGET].to_numpy(int),
                _matrix(val, result.vocab, feats), val[TARGET].to_numpy(int), _cat_idx(feats), list(feats), 300, 30)


def size_bands(con, result: TrainResult, test: pd.DataFrame, p_model: np.ndarray, p_base: np.ndarray) -> dict:
    y = test[TARGET].to_numpy(int)
    emp = test["employees"].to_numpy(float)
    out: dict[str, Any] = {
        "overall": {
            "model": _auc(y, p_model), "size_naics_baseline": _auc(y, p_base),
            "hours_worked_only": _auc(y, test["hours_worked"].fillna(0).to_numpy(float)),
            "had_case_this_year": _auc(y, (test["n_recordable"].fillna(0) > 0).to_numpy(float)),
            "cases_this_year": _auc(y, test["n_recordable"].fillna(0).to_numpy(float)),
        },
        "bands": [],
    }
    for lo, hi in EMPLOYEE_BANDS:
        mask = (emp >= lo) & (emp <= hi) if hi is not None else emp >= lo
        out["bands"].append({
            "employees": _band_label(lo, hi), "n": int(mask.sum()),
            "base_rate": float(y[mask].mean()) if mask.sum() else None,
            "model": _auc(y[mask], p_model[mask]), "size_naics_baseline": _auc(y[mask], p_base[mask]),
            "had_case_this_year": _auc(y[mask], (test["n_recordable"].fillna(0).to_numpy(float) > 0)[mask]),
        })
    return out


def identity(con, result: TrainResult, panel: str, test: pd.DataFrame, p_model: np.ndarray, p_base: np.ndarray) -> dict:
    """Seen/unseen split by estab_key, then again treating a shared establishment_id as 'seen'."""
    seen_keys = {r[0] for r in con.execute(
        f"SELECT DISTINCT estab_key FROM {panel} WHERE year <= {result.val_year} AND {TARGET} IS NOT NULL").fetchall()}
    seen_ids = {r[0] for r in con.execute(
        f"SELECT DISTINCT establishment_id FROM {panel} WHERE year <= {result.val_year} AND {TARGET} IS NOT NULL "
        "AND establishment_id IS NOT NULL").fetchall()}
    ids = dict(con.execute(
        f"SELECT estab_key, establishment_id FROM {panel} WHERE year = {result.test_year}").fetchall())
    key_seen = test["estab_key"].isin(seen_keys).to_numpy()
    id_seen = np.array([ids.get(k) in seen_ids for k in test["estab_key"]])
    y = test[TARGET].to_numpy(int)
    out = {"n_test": len(test), "n_with_establishment_id": int(sum(ids.get(k) is not None for k in test["estab_key"]))}
    for label, mask in (("key_seen", key_seen), ("key_unseen", ~key_seen),
                        ("key_unseen_but_id_seen", ~key_seen & id_seen),
                        ("unseen_by_key_and_id", ~key_seen & ~id_seen)):
        out[label] = {"n": int(mask.sum()), "base_rate": float(y[mask].mean()) if mask.sum() else None,
                      "model": _auc(y[mask], p_model[mask]), "size_naics_baseline": _auc(y[mask], p_base[mask])}
    return out


def other_years(con, result: TrainResult, panel: str, rounds: int = 300) -> list[dict]:
    """Retrain with each available earlier test year (needs train, val and test years)."""
    years = [r[0] for r in con.execute(
        f"SELECT DISTINCT year FROM {panel} WHERE {TARGET} IS NOT NULL ORDER BY 1").fetchall()]
    out = []
    for test_year in years[2:]:
        r = train_baseline(con, train_through=test_year - 2, val_year=test_year - 1, test_year=test_year,
                           panel_table=panel, num_boost_round=rounds)
        tb, tr = r.metrics.get("test_binary", {}), r.metrics.get("test_rate", {})
        out.append({"train_through": test_year - 2, "val_year": test_year - 1, "test_year": test_year,
                    "n_test": tb.get("n"), "auc": tb.get("auc"), "baseline_auc": tb.get("baseline_auc"),
                    "brier_skill": tb.get("brier_skill"), "capture_cases_top10": tb.get("capture_cases_top10"),
                    "rate_skill_vs_credibility": tr.get("deviance_skill_vs_credibility"),
                    "rate_skill_vs_industry": tr.get("deviance_skill_vs_industry")})
    return out


def denominators(con, result: TrainResult, test: pd.DataFrame, rates: np.ndarray, fallback: float) -> dict:
    out: dict[str, Any] = {"rate_by_hours_band": []}
    h = test["hours_worked"].fillna(0).to_numpy(float)
    for lo, hi in HOURS_BANDS:
        mask = (h >= lo) & (h < hi) if hi is not None else h >= lo
        rm = rate_metrics(test[mask], rates[mask], fallback)
        out["rate_by_hours_band"].append({
            "hours_worked": _band_label(lo, hi), "approx_fte": _band_label(lo // 2000, None if hi is None else hi // 2000),
            "n": rm.get("n"),
            "share_of_next_year_cases": float(test[mask][COUNT_TARGET].fillna(0).sum() / max(test[COUNT_TARGET].fillna(0).sum(), 1)),
            "spearman_model": rm.get("model", {}).get("spearman_vs_realized_tcr"),
            "spearman_credibility": rm.get("credibility", {}).get("spearman_vs_realized_tcr"),
            "skill_vs_credibility": rm.get("deviance_skill_vs_credibility"),
            "skill_vs_industry": rm.get("deviance_skill_vs_industry"),
        })
    if _exists(con, "sst_position"):
        y3 = 2023
        rows = con.execute(
            f"""WITH c AS (
                    SELECT s.estab_key, s.sst_group, s.group_share_below, p.n_dart, p.hours_worked
                    FROM sst_position s JOIN ita_panel p ON p.estab_key = s.estab_key AND p.year = {y3}
                    WHERE s.in_comparison)
                SELECT sst_group, COUNT(*) FILTER (WHERE group_share_below >= 0.9),
                       COUNT(*) FILTER (WHERE group_share_below >= 0.9 AND n_dart <= 2),
                       COUNT(*) FILTER (WHERE group_share_below >= 0.9 AND hours_worked < 100000)
                FROM c GROUP BY 1 ORDER BY 1"""
        ).fetchall()
        out["sst_top_decile"] = [{"sst_group": g, "n_top_decile": int(n), "with_two_or_fewer_dart_cases": int(k),
                                  "share_two_or_fewer": (k / n) if n else None,
                                  "under_50_fte": int(s), "share_under_50_fte": (s / n) if n else None}
                                 for g, n, k, s in rows]
        # How many DART cases would it take to cross from the median to the 90th percentile, for a 50-FTE site?
        stats = con.execute("SELECT sst_group, p50, p90 FROM sst_group_stats ORDER BY 1").fetchall()
        out["cases_median_to_p90_at_50_fte"] = {g: (p90 - p50) * 100_000 / 200_000 for g, p50, p90 in stats}
    return out


def _exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0])


def run(con: duckdb.DuckDBPyConnection, result: TrainResult, *, out_path: Path | None = None,
        retrain_years: bool = True, rounds: int = 300) -> dict[str, Any]:
    panel = result.panel_table
    test = fetch_frame(con, panel, f"{TARGET} IS NOT NULL AND year = {result.test_year}")
    preds = result.predict(test)
    base = _size_baseline(con, result, panel)
    p_base = base.predict(_matrix(test, result.vocab, SIZE_BASELINE_FEATURES))
    report = {
        "split": {"train_through": result.train_through, "val_year": result.val_year, "test_year": result.test_year},
        "size_bands": size_bands(con, result, test, preds["p"], p_base),
        "identity": identity(con, result, panel, test, preds["p"], p_base),
        "denominators": denominators(con, result, test, preds["boosted_rate_per_hour"], result.fallback_rate_per_hour),
        "other_years": other_years(con, result, panel, rounds) if retrain_years else [],
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=float))
    return report
