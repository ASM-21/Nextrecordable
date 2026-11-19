"""Per-establishment reports and vectorized batch scoring.

Scores come from the `estab_scores` table written at training time
(models.baseline.materialize_scores), so a single lookup, a batch upload and
a PDF always show the same numbers for the same establishment. The model is
only consulted for drivers, the per-feature contributions behind a score.
"""

from __future__ import annotations

import math
import re
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .models.baseline import FEATURE_LABELS, TrainResult, fetch_frame

_KEY_RE = re.compile(r"[0-9a-f]{32}")

STANDARD_CAVEATS: list[str] = [
    "OSHA injury logs undercount recordable cases; published estimates of the "
    "share missed run from roughly 30 to 70 percent, depending on industry and "
    "method. Scores predict reported cases, which also reflect how carefully an "
    "employer records them.",
    "Only establishments required to file with OSHA's Injury Tracking "
    "Application appear here; most workplaces under 20 employees do not file.",
    "Expected case rate is size-adjusted (cases per 200,000 hours). The chance "
    "of any recordable case rises with headcount, so compare establishments of "
    "different sizes on the rate, not the chance.",
    "Establishments are matched across years by name, state and ZIP. Name "
    "changes split one site into two records; two sites with one name in one "
    "ZIP are combined.",
    "Enforcement figures count OSHA inspections on record. Zero means no "
    "inspection was found, not that the site was inspected and found clean.",
]


def _clean(value: Any) -> Any:
    """JSON-safe scalar: numpy -> Python, NaN/inf -> None."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _table_exists(con, name: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0])


def score_keys(
    con,
    keys: list[str],
    *,
    panel_table: str = "ita_panel",
    scores_table: str = "estab_scores",
) -> pd.DataFrame:
    """Score fields + display fields for many establishments in three queries."""
    frame = pd.DataFrame({"estab_key": pd.Series(list(dict.fromkeys(keys)), dtype="object")})
    con.register("_score_keys_df", frame)
    try:
        out = con.execute(
            f"""
            WITH latest AS (
                SELECT * FROM {panel_table}
                WHERE estab_key IN (SELECT estab_key FROM _score_keys_df)
                QUALIFY ROW_NUMBER() OVER (PARTITION BY estab_key ORDER BY year DESC) = 1
            ),
            years AS (
                SELECT estab_key, COUNT(*) AS n_years FROM {panel_table}
                WHERE estab_key IN (SELECT estab_key FROM _score_keys_df) GROUP BY 1
            )
            SELECT p.estab_key, p.establishment_name, p.street_address, p.city, p.state, p.zip5,
                   p.naics_code, p.year AS latest_year, p.employees, p.n_recordable,
                   p.tcr AS current_tcr, p.dart_rate AS current_dart_rate,
                   p.naics4_tcr_median, y.n_years,
                   s.p_any_next, s.expected_tcr_next, s.expected_cases_next, s.experience_rated_tcr_next,
                   s.peer_pctile, s.peer_group, s.peer_count, s.overall_pctile, s.risk_band
            FROM latest p
            JOIN years y USING (estab_key)
            LEFT JOIN {scores_table} s USING (estab_key)
            """
        ).df()
    finally:
        con.unregister("_score_keys_df")
    return out


def confidence_for(n_years: int) -> tuple[str, str]:
    if n_years < 2:
        return "low", (
            f"{n_years} year of filings, so the estimate leans on the industry rate "
            "more than on this establishment's own record"
        )
    if n_years < 4:
        return "moderate", f"{n_years} years of filings"
    return "high", f"{n_years} years of filings"


def _drivers(result: TrainResult, frame: pd.DataFrame, top_k: int = 5) -> list[dict]:
    contrib, space = result.contributions(frame)
    row = contrib[0]
    order = np.argsort(-np.abs(row), kind="stable")[:top_k]
    out = []
    for i in order:
        name = result.feature_names[i]
        c = float(row[i])
        item = {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": _clean(frame.iloc[0].get(name)),
            "contribution": c,
            "space": space,
            "direction": "increases_risk" if c > 0 else "decreases_risk",
        }
        if space == "log_rate":
            item["rate_multiplier"] = float(math.exp(c))
        out.append(item)
    return out


def risk_report(
    con,
    estab_key: str,
    result: TrainResult,
    *,
    panel_table: str | None = None,
    scores_table: str = "estab_scores",
) -> dict[str, Any]:
    """Structured report for one establishment (JSON-serializable)."""
    panel = panel_table or result.panel_table
    if not _KEY_RE.fullmatch(estab_key or ""):
        return {"estab_key": estab_key, "error": "no_data",
                "message": "Not a valid establishment key."}
    latest = fetch_frame(
        con, panel,
        f"estab_key = '{estab_key}' "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY estab_key ORDER BY year DESC) = 1",
    )
    if latest.empty:
        return {"estab_key": estab_key, "error": "no_data",
                "message": "No filings found for this establishment."}

    info = con.execute(
        f"""SELECT establishment_name, street_address, city, state, zip5, naics_code,
                   year, employees, hours_worked, n_recordable, n_dart, tcr, dart_rate,
                   naics4_tcr_median, naics4_pooled_tcr, tcr_vs_naics4_median, naics4_n_estabs
            FROM {panel} WHERE estab_key = ? ORDER BY year DESC LIMIT 1""",
        [estab_key],
    ).fetchone()
    cols = ("establishment_name", "street_address", "city", "state", "zip5", "naics_code",
            "latest_year", "employees", "hours_worked", "n_recordable", "n_dart", "tcr", "dart_rate",
            "naics4_tcr_median", "naics4_pooled_tcr", "tcr_vs_naics4_median", "naics4_n_estabs")
    cur = dict(zip(cols, (_clean(v) for v in info), strict=True))

    history = [
        {k: _clean(v) for k, v in zip(
            ("year", "employees", "hours_worked", "n_recordable", "n_dart", "tcr", "dart_rate",
             "naics4_tcr_median"), r, strict=True)}
        for r in con.execute(
            f"""SELECT year, employees, hours_worked, n_recordable, n_dart, tcr, dart_rate,
                       naics4_tcr_median
                FROM {panel} WHERE estab_key = ? ORDER BY year""",
            [estab_key],
        ).fetchall()
    ]

    if _table_exists(con, scores_table):
        srow = con.execute(
            f"""SELECT p_any_next, expected_tcr_next, expected_cases_next, experience_rated_tcr_next,
                       peer_pctile, peer_group, peer_count, overall_pctile, risk_band
                FROM {scores_table} WHERE estab_key = ?""",
            [estab_key],
        ).fetchone()
    else:
        srow = None
    if srow is None:  # not materialized yet: score directly, without peer context
        preds = result.predict(latest)
        rate = float(preds["rate_per_hour"][0])
        srow = (float(preds["p"][0]), rate * 200_000 if math.isfinite(rate) else None, None,
                float(preds["baseline_rate_per_hour"][0]) * 200_000, None, None, None, None, None)
    score_cols = ("p_any_next", "expected_tcr_next", "expected_cases_next", "experience_rated_tcr_next",
                  "peer_pctile", "peer_group", "peer_count", "overall_pctile", "risk_band")
    scores = {k: _clean(v) for k, v in zip(score_cols, srow, strict=True)}

    enforcement = None
    enf_cols = [c for c in latest.columns if c.startswith("enf_")]
    if enf_cols and latest[enf_cols].notna().any(axis=None):
        enforcement = {c: _clean(latest.iloc[0][c]) for c in enf_cols}

    level, note = confidence_for(len(history))
    tb = result.metrics.get("test_binary", {})
    tr = result.metrics.get("test_rate", {})
    return {
        "estab_key": estab_key,
        **{k: cur[k] for k in ("establishment_name", "street_address", "city", "state",
                               "zip5", "naics_code", "latest_year", "employees")},
        "current_year_metrics": {k: cur[k] for k in (
            "hours_worked", "n_recordable", "n_dart", "tcr", "dart_rate", "naics4_tcr_median",
            "naics4_pooled_tcr", "tcr_vs_naics4_median", "naics4_n_estabs")},
        "forecast_year": (cur["latest_year"] or 0) + 1,
        **scores,
        # Older clients read these two names.
        "risk_score_next_year": scores["p_any_next"],
        "industry_percentile": scores["peer_pctile"],
        # Drivers explain the served rate. When the experience-rated estimate is
        # served there is nothing to explain beyond the formula, and the binary
        # model's log-odds must not be shown as if they moved the rate.
        "risk_drivers": _drivers(result, latest) if result.rate_served == "model" and result.rate is not None else [],
        "rate_served": result.rate_served,
        "drivers_note": (None if result.rate_served == "model" else
                         "The boosted rate model did not beat the experience-rated estimate on the holdout "
                         "year, so the estimate shown is the experience-rated one and has no model drivers."),
        "enforcement": enforcement,
        "confidence": level,
        "confidence_note": note,
        "history": history,
        "model_metadata": {
            "train_through": result.train_through,
            "val_year": result.val_year,
            "test_year": result.test_year,
            "test_auc": _clean(tb.get("auc")),
            "test_capture_cases_top10": _clean(tb.get("capture_cases_top10")),
            "test_rate_deviance_skill_vs_credibility": _clean(tr.get("deviance_skill_vs_credibility")),
        },
        "caveats": STANDARD_CAVEATS,
    }


def ensure_scores(con: duckdb.DuckDBPyConnection, result: TrainResult, *,
                  panel_table: str | None = None, scores_table: str = "estab_scores") -> None:
    """Materialize scores if the table is missing (tests and first boot)."""
    if not _table_exists(con, scores_table):
        from .models.baseline import materialize_scores

        materialize_scores(con, result, panel_table=panel_table, output=scores_table)
