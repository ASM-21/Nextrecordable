"""Per-establishment risk report generator.

Produces the JSON-shaped object the web app and PDF report will render.
Inputs: a trained model + a panel row (or estab_key). Outputs: a structured
report with risk score, history, drivers, and benchmarks.

Design principles:
  - Honest about uncertainty: if we have < 2 years of history for an
    establishment, we say so. Never produce a confident score on a single
    year of data.
  - Drivers, not black box: top contributing features for each prediction
    (via SHAP-like local contributions from LightGBM).
  - Caveats inline: underreporting, coverage gaps, soft-key risk all
    surfaced in the report payload rather than hidden in a methodology doc.
"""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np

from .models.baseline import ALL_FEATURES, CATEGORICAL_FEATURES, TrainResult, _to_categorical_codes


def _latest_panel_row(
    con: duckdb.DuckDBPyConnection, estab_key: str, panel_table: str = "ita_panel"
) -> dict | None:
    """Fetch the most recent panel row for an establishment as a dict."""
    sql = f"""
        SELECT *
        FROM {panel_table}
        WHERE estab_key = ?
        ORDER BY year DESC
        LIMIT 1
    """
    res = con.execute(sql, [estab_key])
    cols = [d[0] for d in res.description]
    row = res.fetchone()
    if row is None:
        return None
    return dict(zip(cols, row, strict=True))


def _history(
    con: duckdb.DuckDBPyConnection, estab_key: str, panel_table: str = "ita_panel"
) -> list[dict]:
    """All years of panel data for one establishment, oldest first."""
    sql = f"""
        SELECT year, employees, hours_worked, n_recordable,
               n_dafw, n_djtr, tcr, dart_rate,
               naics4_tcr_median
        FROM {panel_table}
        WHERE estab_key = ?
        ORDER BY year ASC
    """
    res = con.execute(sql, [estab_key])
    cols = [d[0] for d in res.description]
    return [dict(zip(cols, r, strict=True)) for r in res.fetchall()]


def _row_to_feature_matrix(row: dict) -> np.ndarray:
    """Convert a single panel row dict into a (1, n_features) matrix."""
    vec = []
    for name in ALL_FEATURES:
        v = row.get(name)
        if name in CATEGORICAL_FEATURES:
            # Reuse the same int-coding the trainer used
            v = _to_categorical_codes(np.array([v]))[0]
        else:
            v = float(v) if v is not None else np.nan
        vec.append(v)
    return np.array([vec], dtype=np.float64)


def _percentile_in_naics(
    con: duckdb.DuckDBPyConnection,
    naics_4: str,
    year: int,
    tcr: float | None,
    panel_table: str = "ita_panel",
) -> float | None:
    """What fraction of NAICS-4 peers in `year` have TCR <= this one?

    Higher percentile means this establishment is RISKIER than that fraction
    of peers. Returns None if peer set is too small (<5) or TCR is missing.
    """
    if tcr is None or naics_4 is None:
        return None
    res = con.execute(
        f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN tcr <= ? THEN 1 ELSE 0 END) AS n_leq
        FROM {panel_table}
        WHERE naics_4 = ? AND year = ? AND tcr IS NOT NULL
        """,
        [tcr, naics_4, year],
    ).fetchone()
    n, n_leq = res
    if n is None or n < 5:
        return None
    return float(n_leq) / float(n)


def _local_drivers(
    booster, X: np.ndarray, top_k: int = 5
) -> list[dict]:
    """Top-k features contributing to this prediction (signed).

    LightGBM `predict(pred_contrib=True)` returns per-feature contributions
    in log-odds space, plus a bias term as the last column.
    """
    contribs = booster.predict(X, pred_contrib=True)[0]
    # Last entry is the bias / expected value
    feat_contribs = contribs[:-1]
    # Sort by absolute magnitude
    order = np.argsort(-np.abs(feat_contribs))[:top_k]
    out: list[dict] = []
    for idx in order:
        out.append(
            {
                "feature": ALL_FEATURES[idx],
                "contribution_logodds": float(feat_contribs[idx]),
                "direction": "increases_risk" if feat_contribs[idx] > 0 else "decreases_risk",
            }
        )
    return out


# Standard caveats attached to every report
STANDARD_CAVEATS: list[str] = [
    "OSHA ITA data is known to undercount recordable injuries. Multiple studies "
    "estimate 30 to 70 percent underreporting across industries. Risk scores "
    "reflect predicted reported risk, not necessarily true risk.",
    "Only establishments meeting OSHA reporting thresholds appear in this data. "
    "Establishments with fewer than 20 employees in most industries are not covered.",
    "Establishment identity across years is matched on a hash of normalized "
    "name, state, and ZIP. Same-establishment merges are imperfect; some "
    "establishments may appear under multiple keys.",
]


def risk_report(
    con: duckdb.DuckDBPyConnection,
    estab_key: str,
    train_result: TrainResult,
    *,
    panel_table: str = "ita_panel",
) -> dict[str, Any]:
    """Generate a structured risk report for one establishment.

    Returns a dict shaped for direct serialization to JSON (for the API) or
    rendering as HTML/PDF.
    """
    row = _latest_panel_row(con, estab_key, panel_table)
    if row is None:
        return {
            "estab_key": estab_key,
            "error": "no_data",
            "message": "No panel data found for this establishment key.",
        }

    history = _history(con, estab_key, panel_table)

    # Predict
    X = _row_to_feature_matrix(row)
    raw_score = float(train_result.model.predict(X)[0])
    if train_result.calibrator is not None:
        risk_score = float(train_result.calibrator.transform([raw_score])[0])
    else:
        risk_score = raw_score

    pct = _percentile_in_naics(
        con,
        row.get("naics_4"),
        row.get("year"),
        row.get("tcr"),
        panel_table=panel_table,
    )

    drivers = _local_drivers(train_result.model, X, top_k=5)

    # Confidence note based on history depth
    n_years = len(history)
    if n_years < 2:
        confidence = "low"
        confidence_note = (
            f"Only {n_years} year of history available. Score is heavily "
            "dependent on industry baseline rather than this establishment's pattern."
        )
    elif n_years < 4:
        confidence = "moderate"
        confidence_note = f"{n_years} years of history. Trend may be unstable."
    else:
        confidence = "high"
        confidence_note = f"{n_years} years of history available."

    return {
        "estab_key": estab_key,
        "establishment_name": row.get("establishment_name"),
        "state": row.get("state"),
        "zip5": row.get("zip5"),
        "naics_code": row.get("naics_code"),
        "naics_4": row.get("naics_4"),
        "latest_year": row.get("year"),
        "employees": row.get("employees"),
        "current_year_metrics": {
            "n_recordable": row.get("n_recordable"),
            "tcr": row.get("tcr"),
            "dart_rate": row.get("dart_rate"),
            "naics4_tcr_median": row.get("naics4_tcr_median"),
            "tcr_vs_naics4_median": row.get("tcr_vs_naics4_median"),
        },
        "industry_percentile": pct,
        "risk_score_next_year": risk_score,
        "risk_score_raw": raw_score,
        "risk_drivers": drivers,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "history": history,
        "model_metadata": {
            "train_through": train_result.train_through,
            "val_year": train_result.val_year,
            "test_year": train_result.test_year,
            "calibrated": train_result.calibrator is not None,
            "val_auc": train_result.metrics.get("val_auc_calibrated"),
            "test_auc": train_result.metrics.get("test_auc"),
        },
        "caveats": STANDARD_CAVEATS,
    }
