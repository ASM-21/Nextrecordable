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


# ---------------------------------------------------------------------- PDF

def render_pdf(report: dict[str, Any], *, compress: bool = True) -> bytes:
    """One- to two-page PDF of a risk_report() (the broker's leave-behind). Requires reportlab."""
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    def esc(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def num(v: Any, d: int = 1) -> str:
        return "n/a" if v is None else f"{v:,.{d}f}"

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=9.6, spaceAfter=1,
                           textColor=colors.HexColor("#444444"))
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
    big = ParagraphStyle("big", parent=body, fontSize=26, leading=30, fontName="Helvetica-Bold")
    warn = ParagraphStyle("w", parent=body, fontName="Helvetica-Bold", textColor=colors.HexColor("#8a1f11"),
                          borderColor=colors.HexColor("#8a1f11"), borderWidth=0.8, borderPadding=6,
                          spaceBefore=6, spaceAfter=10)
    notice = report.get("data_notice")
    story: list[Any] = [Paragraph("Injury risk report", h1)]
    if notice:
        story.append(Paragraph(esc(notice), warn))
    if report.get("error"):
        story.append(Paragraph(esc(report.get("message", "No data.")), body))
    else:
        addr = ", ".join(str(report[k]) for k in ("street_address", "city", "state", "zip5") if report.get(k))
        story.append(Paragraph(f"<b>{esc(report.get('establishment_name') or '')}</b><br/>{esc(addr)}<br/>"
                               f"NAICS {esc(report.get('naics_code') or 'n/a')} &middot; latest filing "
                               f"{esc(report.get('latest_year'))}", body))
        story.append(Spacer(1, 8))
        story.append(Paragraph(num(report.get("expected_tcr_next")), big))
        story.append(Paragraph(f"expected recordable cases per 100 full-time workers in {esc(report.get('forecast_year'))} "
                               "(per 200,000 hours worked)", body))
        pct = report.get("peer_pctile")
        if pct is not None:
            story.append(Paragraph(f"{esc((report.get('risk_band') or '').capitalize())}: "
                                   f"{round(100 * pct)}th percentile among {num(report.get('peer_count'), 0)} "
                                   f"establishments in {esc(report.get('peer_group'))}.", body))
        cur = report.get("current_year_metrics") or {}
        enf = report.get("enforcement")
        if enf is None:
            enf_txt = "enforcement data not loaded"
        elif (enf.get("enf_inspections_2y") or 0) > 0:
            enf_txt = (f"{num(enf.get('enf_inspections_2y'), 0)} inspections, {num(enf.get('enf_citations_2y'), 0)} "
                       f"citations, ${num(enf.get('enf_initial_penalty_2y'), 0)} initial penalties")
        else:
            enf_txt = "none on record"
        facts = [
            ["Chance of at least one recordable case", "n/a" if report.get("p_any_next") is None
             else f"{round(100 * report['p_any_next'])}%"],
            ["Experience-rated starting point", num(report.get("experience_rated_tcr_next"))],
            [f"Case rate in {report.get('latest_year')}", num(cur.get("tcr"))],
            ["Industry median case rate", num(cur.get("naics4_tcr_median"))],
            ["OSHA inspections, last two years", enf_txt],
            ["Confidence", f"{report.get('confidence', '')}: {report.get('confidence_note', '')}"],
        ]
        t = Table(facts, hAlign="LEFT", colWidths=[2.6 * inch, 4.2 * inch])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story += [Spacer(1, 6), t]
        drivers = report.get("risk_drivers") or []
        if drivers:
            story.append(Paragraph("What moves the estimate off its starting point", h2))
            for dr in drivers:
                mult = f"x{dr['rate_multiplier']:.2f}" if dr.get("rate_multiplier") else ""
                story.append(Paragraph(f"{esc(mult)} {esc(dr['label'])}", body))
        elif report.get("drivers_note"):
            story.append(Paragraph(esc(report["drivers_note"]), small))
        story.append(Paragraph("Filed with OSHA", h2))
        rows = [["Year", "Employees", "Hours worked", "Recordable", "DART", "Case rate", "DART rate", "Industry median"]]
        for h in reversed(report.get("history") or []):
            rows.append([str(h["year"]), num(h.get("employees"), 0),
                         "not usable" if h.get("hours_worked") is None else num(h["hours_worked"], 0),
                         num(h.get("n_recordable"), 0), num(h.get("n_dart"), 0), num(h.get("tcr")),
                         num(h.get("dart_rate")), num(h.get("naics4_tcr_median"))])
        ht = Table(rows, hAlign="LEFT")
        ht.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                                ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        story.append(ht)
        mm = report.get("model_metadata") or {}
        story.append(Paragraph("How good is this estimate", h2))
        story.append(Paragraph(
            f"Models trained on filings through {esc(mm.get('train_through'))} and tested on {esc(mm.get('test_year'))} "
            f"filings predicting the following year. On that test year, ranking by the chance of any case reached an AUC "
            f"of {num(mm.get('test_auc'), 3)}. Full results and the checks that tried to break them are published with "
            "the methodology.", small))
        story.append(Paragraph("Read before relying on this", h2))
        for c in report.get("caveats") or []:
            story.append(Paragraph("&bull; " + esc(c), small))
        story.append(Paragraph("Independent analysis of public OSHA and Department of Labor data. Not affiliated with, "
                               "endorsed by, or produced on behalf of OSHA or the U.S. Department of Labor. Not legal "
                               "advice. Do not republish scores about named establishments without legal review.", small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.75 * inch, 0.5 * inch, "nextrecordable injury risk report: public data, reported cases only.")
        if notice:
            canvas.setFillColor(colors.HexColor("#8a1f11"))
            canvas.drawString(0.75 * inch, 0.36 * inch, notice[:150])
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.8 * inch, title="Injury risk report", author="nextrecordable",
                      pageCompression=1 if compress else 0).build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
