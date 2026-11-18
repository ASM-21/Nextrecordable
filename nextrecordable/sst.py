"""SST exposure: where an establishment sits against OSHA's published targeting criteria.

OSHA's Site-Specific Targeting directive (CPL 02-01-067, effective May 20,
2025) builds its inspection lists from the same Form 300A data this project
loads. The directive is encoded, with verbatim quotes and its source, in
`data/sst_cpl_02_01_067.json`. This module compares one establishment's
filings with what that file says, and with other filers.

What it will not do, by design:

  * It never predicts an inspection. Three of the four lists are random
    samples, the inspection lists are visible only to OSHA and State Plans,
    area offices delete and defer entries, and the whole previous program
    produced 652 inspections in twenty months. `STATEMENT` is printed on
    every report.
  * It never states a DART cutoff for the high-rate list. The directive says
    two cutoffs exist (manufacturing, non-manufacturing) and publishes
    neither, so the report gives percentile position inside each group.
  * It uses no Form 300/301 case data, so no case narrative or any other
    detail about an individual worker can reach a report.

The one number the directive does pin down, "twice the private sector
national average in CY 2022", is not printed in the directive either. The
data file carries the BLS figure, its source, and a note that reading the
sentence as "CY 2022 DART rate >= 2 x 1.7" is this project's interpretation.
"""

from __future__ import annotations

import io
import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

from . import provenance

STATEMENT = (
    "This report describes position against published criteria and never predicts "
    "whether OSHA will inspect."
)
DIRECTIVE_FILE = Path(__file__).parent / "data" / "sst_cpl_02_01_067.json"
MIN_PEERS = 20
MANUFACTURING_PREFIXES = ("31", "32", "33")
GROUP_LABELS = {
    "manufacturing": "manufacturing (NAICS 31-33)",
    "non_manufacturing": "non-manufacturing, excluding construction",
    "construction": "construction (outside the SST program)",
    "unknown": "unknown industry",
}
# ITA establishment_type: 1 = not a government entity, 2 = state government, 3 = local government.
GOVERNMENT_TYPES = ("2", "3")
_KEY_RE = re.compile(r"[0-9a-f]{32}")

CAVEATS: list[str] = [
    "Rates are computed from the establishment's own Form 300A filings as published by OSHA: "
    "(days-away cases + restricted or transferred cases) x 200,000 / hours worked. An error in the "
    "filing is an error here.",
    "OSHA does not publish the DART cutoffs it uses for the high-rate list, and three of the four "
    "lists are random samples. Position near a criterion says nothing about whether an establishment "
    "is on a list.",
    "Peer comparisons use establishments that filed for the same year with 20 or more employees. "
    "Establishments that did not file are not in the comparison.",
    "Establishments are followed across years by name, state and ZIP. A name change can make one "
    "site look like a new filer and a missing filer.",
    "OSHA deletes establishments from its lists for reasons this report cannot see in full: a "
    "comprehensive inspection in the previous 36 months, public-sector status, VPP or SHARP "
    "participation, or having gone out of business.",
]


@lru_cache(maxsize=1)
def load_directive() -> dict[str, Any]:
    return json.loads(DIRECTIVE_FILE.read_text())


def _exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0])


def _columns(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def group_sql(naics: str) -> str:
    mfg = ", ".join(f"'{p}'" for p in MANUFACTURING_PREFIXES)
    return (
        f"CASE WHEN coalesce({naics}, '') = '' THEN 'unknown' "
        f"WHEN substr({naics}, 1, 2) = '23' THEN 'construction' "
        f"WHEN substr({naics}, 1, 2) IN ({mfg}) THEN 'manufacturing' ELSE 'non_manufacturing' END"
    )


def build_sst_table(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_table: str = "ita_panel",
    output: str = "sst_position",
) -> int:
    """One row per establishment that filed in any SST data year. Returns row count.

    Columns: DART rate and filing flags for each data year, SST group, scope
    flags, and the share of in-scope establishments with a lower selection-year
    DART rate (within the SST group, and within the NAICS-4 or NAICS-2 peers).
    """
    d = load_directive()
    y1, y2, y3 = d["data_years"]
    min_emp = int(d["scope"]["min_employees"])
    threshold = float(d["criteria"]["upward_trending"]["derived_threshold"])
    cols = _columns(con, panel_table)
    est_type = "establishment_type" if "establishment_type" in cols else "CAST(NULL AS VARCHAR)"
    gov = ", ".join(f"'{t}'" for t in GOVERNMENT_TYPES)

    def pick(col: str, year: int, alias: str) -> str:
        return f"max(CASE WHEN year = {year} THEN {col} END) AS {alias}_{year}"

    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        WITH wide AS (
            SELECT
                estab_key,
                {pick('dart_rate', y1, 'dart')}, {pick('dart_rate', y2, 'dart')}, {pick('dart_rate', y3, 'dart')},
                {pick('employees', y2, 'employees')}, {pick('employees', y3, 'employees')},
                bool_or(year = {y1}) AS filed_{y1},
                bool_or(year = {y2}) AS filed_{y2},
                bool_or(year = {y3}) AS filed_{y3},
                arg_max(naics_code, year) AS naics_code,
                arg_max({est_type}, year) AS establishment_type,
                max(year) AS latest_year
            FROM {panel_table}
            WHERE year IN ({y1}, {y2}, {y3})
            GROUP BY estab_key
        ),
        flagged AS (
            SELECT *,
                {group_sql('naics_code')} AS sst_group,
                substr(coalesce(naics_code, ''), 1, 4) AS naics_4,
                substr(coalesce(naics_code, ''), 1, 2) AS naics_2,
                coalesce(establishment_type IN ({gov}), false) AS is_government
            FROM wide
        ),
        scoped AS (
            SELECT *,
                (filed_{y3} AND employees_{y3} >= {min_emp} AND sst_group IN ('manufacturing', 'non_manufacturing')
                 AND dart_{y3} IS NOT NULL AND NOT is_government) AS in_comparison
            FROM flagged
        ),
        ranked AS (
            SELECT estab_key,
                (rank() OVER (PARTITION BY sst_group ORDER BY dart_{y3}) - 1) * 1.0
                    / COUNT(*) OVER (PARTITION BY sst_group) AS group_share_below,
                COUNT(*) OVER (PARTITION BY sst_group) AS group_n,
                (rank() OVER (PARTITION BY naics_4 ORDER BY dart_{y3}) - 1) * 1.0
                    / COUNT(*) OVER (PARTITION BY naics_4) AS n4_share_below,
                COUNT(*) OVER (PARTITION BY naics_4) AS n4_n,
                (rank() OVER (PARTITION BY naics_2 ORDER BY dart_{y3}) - 1) * 1.0
                    / COUNT(*) OVER (PARTITION BY naics_2) AS n2_share_below,
                COUNT(*) OVER (PARTITION BY naics_2) AS n2_n
            FROM scoped WHERE in_comparison
        )
        SELECT
            s.*,
            r.group_share_below, r.group_n,
            CASE WHEN r.n4_n >= {MIN_PEERS} THEN r.n4_share_below ELSE r.n2_share_below END AS industry_share_below,
            CASE WHEN r.n4_n >= {MIN_PEERS} THEN r.n4_n ELSE r.n2_n END AS industry_n,
            CASE WHEN r.n4_n >= {MIN_PEERS} THEN 'NAICS ' || s.naics_4 ELSE 'NAICS ' || s.naics_2 END AS industry_group,
            (s.dart_{y2} >= {threshold}) AS upward_rate_condition,
            (s.dart_{y1} < s.dart_{y2} AND s.dart_{y2} < s.dart_{y3}) AS upward_trend_condition,
            (NOT s.filed_{y3} AND s.filed_{y2} AND s.employees_{y2} >= {min_emp}
             AND s.sst_group IN ('manufacturing', 'non_manufacturing')) AS no_filing_found_selection_year
        FROM scoped s
        LEFT JOIN ranked r USING (estab_key)
        """
    )
    build_group_stats(con, table=output)
    return con.execute(f"SELECT COUNT(*) FROM {output}").fetchone()[0]


def build_group_stats(con, *, table: str = "sst_position", output: str = "sst_group_stats") -> None:
    """Distribution of selection-year DART among compared establishments, per SST group."""
    y3 = load_directive()["selection_year"]
    y1, y2, _ = load_directive()["data_years"]
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        SELECT sst_group,
               COUNT(*) AS n,
               avg(CASE WHEN dart_{y3} = 0 THEN 1.0 ELSE 0.0 END) AS share_zero,
               quantile_cont(dart_{y3}, 0.50) AS p50,
               quantile_cont(dart_{y3}, 0.75) AS p75,
               quantile_cont(dart_{y3}, 0.90) AS p90,
               quantile_cont(dart_{y3}, 0.95) AS p95,
               COUNT(*) FILTER (WHERE upward_rate_condition AND upward_trend_condition) AS n_upward_both,
               COUNT(*) FILTER (WHERE filed_{y1} AND filed_{y2}) AS n_with_three_years
        FROM {table}
        WHERE in_comparison
        GROUP BY sst_group
        """
    )


def group_stats(con, *, table: str = "sst_group_stats") -> dict[str, dict[str, Any]]:
    if not _exists(con, table):
        return {}
    cols = ("sst_group", "n", "share_zero", "p50", "p75", "p90", "p95", "n_upward_both", "n_with_three_years")
    return {r[0]: dict(zip(cols[1:], r[1:], strict=True))
            for r in con.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()}


def program_summary(con, *, table: str = "sst_position") -> dict[str, Any]:
    """Counts for METRICS.md: how many establishments sit where."""
    if not _exists(con, table):
        return {}
    d = load_directive()
    y3 = d["selection_year"]
    row = con.execute(
        f"""SELECT COUNT(*), COUNT(*) FILTER (WHERE filed_{y3}), COUNT(*) FILTER (WHERE in_comparison),
                   COUNT(*) FILTER (WHERE in_comparison AND upward_rate_condition AND upward_trend_condition),
                   COUNT(*) FILTER (WHERE no_filing_found_selection_year)
            FROM {table}"""
    ).fetchone()
    keys = ("establishments_in_data_years", "filed_selection_year", "in_comparison",
            "meet_both_upward_conditions", "no_filing_found_selection_year")
    return {**dict(zip(keys, row, strict=True)), "groups": group_stats(con),
            "selection_year": y3, "selection_year_snapshot": provenance.snapshot_for(con, y3)}


def _pct(share: float | None) -> str:
    return "n/a" if share is None else f"{100 * share:.0f}%"


def _rate(v: float | None) -> str:
    return "not available" if v is None else f"{v:.2f}"


def _last_comprehensive_inspection(con, estab_key: str) -> dict[str, Any] | None:
    """Most recent comprehensive-scope inspection linked to this establishment, if tables exist."""
    if not (_exists(con, "inspection_linked") and _exists(con, "osha_inspection")):
        return None
    row = con.execute(
        """SELECT max(substr(i.open_date, 1, 10)), COUNT(*)
           FROM inspection_linked l JOIN osha_inspection i USING (activity_nr)
           WHERE l.estab_key = ? AND upper(left(coalesce(i.insp_scope, ''), 1)) = 'A'""",
        [estab_key],
    ).fetchone()
    total = con.execute(
        "SELECT COUNT(*), max(substr(open_date, 1, 10)) FROM inspection_linked WHERE estab_key = ?", [estab_key]
    ).fetchone()
    return {"last_comprehensive_open_date": row[0], "n_comprehensive_on_record": int(row[1]),
            "n_inspections_on_record": int(total[0]), "last_inspection_open_date": total[1]}


def _months_between(earlier: str, later: date) -> int | None:
    try:
        d0 = date.fromisoformat(earlier[:10])
    except (TypeError, ValueError):
        return None
    return (later.year - d0.year) * 12 + (later.month - d0.month) - (1 if later.day < d0.day else 0)


def sst_exposure(
    con: duckdb.DuckDBPyConnection,
    estab_key: str,
    *,
    panel_table: str = "ita_panel",
    table: str = "sst_position",
    as_of: date | None = None,
) -> dict[str, Any]:
    """Structured SST exposure report for one establishment (JSON-serializable)."""
    d = load_directive()
    crit = d["criteria"]
    y1, y2, y3 = d["data_years"]
    as_of = as_of or date.today()
    base: dict[str, Any] = {
        "report": "sst_exposure",
        "estab_key": estab_key,
        "statement": STATEMENT,
        "data_notice": provenance.notice(con),
        "as_of": as_of.isoformat(),
    }
    if not _KEY_RE.fullmatch(estab_key or ""):
        return {**base, "error": "no_data", "message": "Not a valid establishment key."}
    if not _exists(con, table):
        build_sst_table(con, panel_table=panel_table, output=table)

    ident = con.execute(
        f"""SELECT establishment_name, street_address, city, state, zip5, naics_code, year, employees
            FROM {panel_table} WHERE estab_key = ? ORDER BY year DESC LIMIT 1""",
        [estab_key],
    ).fetchone()
    if ident is None:
        return {**base, "error": "no_data", "message": "No filings found for this establishment."}
    ident_cols = ("establishment_name", "street_address", "city", "state", "zip5", "naics_code",
                  "latest_year", "employees")
    identity = dict(zip(ident_cols, ident, strict=True))

    pcols = _columns(con, panel_table)
    ok_expr = "exposure_ok AND counts_ok" if {"exposure_ok", "counts_ok"} <= pcols else "hours_worked > 0"
    history = []
    for r in con.execute(
        f"""SELECT year, employees, hours_worked, n_dart, dart_rate, n_recordable, tcr,
                   naics4_dart_median, coalesce({ok_expr}, false)
            FROM {panel_table} WHERE estab_key = ? ORDER BY year""",
        [estab_key],
    ).fetchall():
        history.append({
            "year": int(r[0]), "employees": r[1], "hours_worked": r[2], "n_dart": r[3], "dart_rate": r[4],
            "n_recordable": r[5], "tcr": r[6], "industry_median_dart": r[7], "usable": bool(r[8]),
            "sst_data_year": int(r[0]) in (y1, y2, y3),
        })

    res = con.execute(f"SELECT * FROM {table} WHERE estab_key = ?", [estab_key])
    row_cols = [c[0] for c in res.description]
    row = res.fetchone()
    report: dict[str, Any] = {
        **base, **identity,
        "directive": {
            k: d[k] for k in ("directive", "title", "signature_date", "effective_date", "terminates",
                              "expires_on_or_about", "data_years", "selection_year")
        } | {"source": d["source"], "successor": d["successor"], "program_size": d["program_size"]},
        "dart_history": history,
        "caveats": CAVEATS,
        "data_vintage": {
            "selection_year_file_snapshot": provenance.snapshot_for(con, y3),
            "note": "A filing made after the snapshot date is not in this data.",
        },
    }
    if row is None:
        report.update({
            "in_scope": False,
            "scope_note": f"No filing for CY {y1}, {y2} or {y3} was found, so the establishment cannot "
                          "be placed against criteria that use those years.",
            "criteria": [], "peers": None, "non_filer": None, "deletion_context": None,
        })
        return report
    s = dict(zip(row_cols, row, strict=True))
    grp = s["sst_group"]
    stats = group_stats(con).get(grp)

    reasons = []
    if grp == "construction":
        reasons.append("Construction (NAICS 23) is outside the SST program.")
    if grp == "unknown":
        reasons.append("No NAICS code on the filing, so the industry group is unknown.")
    if not s[f"filed_{y3}"]:
        reasons.append(f"No CY {y3} filing was found in this data.")
    elif s[f"employees_{y3}"] is None or s[f"employees_{y3}"] < d["scope"]["min_employees"]:
        reasons.append(f"The CY {y3} filing reports fewer than {d['scope']['min_employees']} employees.")
    if s["is_government"]:
        reasons.append("The filing marks the establishment as state or local government; "
                       "OSHA deletes public-sector employers from SST lists.")
    if s[f"filed_{y3}"] and s[f"dart_{y3}"] is None:
        reasons.append(f"The CY {y3} filing's hours or case counts are not usable, so no rate can be computed.")
    in_scope = not reasons
    report["in_scope"] = in_scope
    report["scope_note"] = (
        d["scope"]["summary"] if in_scope else " ".join(reasons)
    )
    report["sst_group"] = grp
    report["sst_group_label"] = GROUP_LABELS[grp]

    darts = {y: s[f"dart_{y}"] for y in (y1, y2, y3)}
    report["sst_years"] = {str(y): {"filed": bool(s[f"filed_{y}"]), "dart_rate": darts[y]} for y in (y1, y2, y3)}

    peers = None
    if s["in_comparison"]:
        peers = {
            "year": y3,
            "group": {"label": GROUP_LABELS[grp], "n": int(s["group_n"]),
                      "share_with_lower_rate": s["group_share_below"],
                      **({k: stats[k] for k in ("p50", "p75", "p90", "p95", "share_zero")} if stats else {})},
            "industry": {"label": s["industry_group"], "n": int(s["industry_n"]),
                         "share_with_lower_rate": s["industry_share_below"]},
        }
    report["peers"] = peers

    # ---- criteria -------------------------------------------------------
    criteria = []
    hr = crit["high_rate"]
    if s["in_comparison"]:
        pos = (f"CY {y3} DART rate {_rate(darts[y3])}. {_pct(s['group_share_below'])} of the "
               f"{int(s['group_n']):,} compared {GROUP_LABELS[grp]} establishments reported a lower rate. "
               "OSHA sets a separate cutoff for this group and does not publish it.")
    else:
        pos = f"Not placed: {report['scope_note']}"
    criteria.append({"id": "high_rate", "label": hr["label"], "quote": hr["quote"],
                     "numeric_cutoff_published": False, "status": "no_published_cutoff" if s["in_comparison"] else "not_evaluable",
                     "position": pos, "share_with_lower_rate": s["group_share_below"]})

    ut = crit["upward_trending"]
    thr = ut["derived_threshold"]
    if None in darts.values():
        missing = [str(y) for y, v in darts.items() if v is None]
        ustatus, upos = "not_evaluable", (
            f"Needs a usable DART rate for each of CY {y1}, {y2} and {y3}; missing {', '.join(missing)}.")
    else:
        rate_ok, trend_ok = bool(s["upward_rate_condition"]), bool(s["upward_trend_condition"])
        ustatus = ("meets_both_conditions_as_read" if rate_ok and trend_ok else
                   "meets_rate_condition_only" if rate_ok else
                   "meets_trend_condition_only" if trend_ok else "meets_neither_condition")
        upos = (
            f"DART rates {_rate(darts[y1])} ({y1}), {_rate(darts[y2])} ({y2}), {_rate(darts[y3])} ({y3}). "
            f"CY {y2} rate is {'at or above' if rate_ok else 'below'} {thr:.1f} (twice the BLS private-sector "
            f"DART rate of {ut['private_sector_dart_rate']:.1f} for {ut['reference_year']}); the rate "
            f"{'rose' if trend_ok else 'did not rise'} in each year from {y1} to {y3}."
        )
    criteria.append({"id": "upward_trending", "label": ut["label"], "quote": ut["quote"],
                     "numeric_cutoff_published": False, "status": ustatus, "position": upos,
                     "interpretation_note": ut["cutoff_note"], "derived_threshold": thr})

    lr = crit["low_rate"]
    if s["in_comparison"]:
        lpos = (f"CY {y3} DART rate {_rate(darts[y3])}; {_pct(stats['share_zero'] if stats else None)} of compared "
                f"{GROUP_LABELS[grp]} establishments reported a rate of zero. The directive does not define "
                "\"low\", and selection within the group is random.")
        lstatus = "no_published_cutoff"
    else:
        lpos, lstatus = f"Not placed: {report['scope_note']}", "not_evaluable"
    criteria.append({"id": "low_rate", "label": lr["label"], "quote": lr["quote"],
                     "numeric_cutoff_published": False, "status": lstatus, "position": lpos})

    nr = crit["non_responders"]
    snap = provenance.snapshot_for(con, y3)
    snap_txt = f" (file snapshot: {snap})" if snap else ""
    if s[f"filed_{y3}"]:
        nstatus, npos = "filing_found", f"A CY {y3} Form 300A filing is in this data{snap_txt}."
    elif s["no_filing_found_selection_year"]:
        nstatus = "no_filing_found"
        npos = (f"No CY {y3} filing was found{snap_txt}, and the CY {y2} filing reported "
                f"{s[f'employees_{y2}']:,.0f} employees. This is not a finding that a required filing was "
                "missed: the establishment may have filed after the snapshot, under a different name or "
                "address, closed, or fallen outside the filing requirement.")
    else:
        nstatus = "not_evaluable"
        npos = (f"No CY {y3} filing was found{snap_txt}, and there is no CY {y2} filing with 20 or more "
                "employees to suggest one was due.")
    criteria.append({"id": "non_responders", "label": nr["label"], "quote": nr["quote"],
                     "numeric_cutoff_published": False, "status": nstatus, "position": npos})
    report["criteria"] = criteria
    report["non_filer"] = {"status": nstatus, "selection_year": y3, "snapshot": snap}

    insp = _last_comprehensive_inspection(con, estab_key)
    months = _months_between(insp["last_comprehensive_open_date"], as_of) if insp and insp["last_comprehensive_open_date"] else None
    report["deletion_context"] = {
        "rule": d["deletions"]["quote"],
        "other_deletions": d["deletions"]["other"],
        "inspections": insp,
        "months_since_last_comprehensive": months,
        "within_36_months": (months is not None and months < d["deletions"]["previous_inspection_months"]),
        "is_government": bool(s["is_government"]),
        "note": ("Inspection records are matched by name and address and may be incomplete; no record "
                 "found is not proof that none exists." if insp is not None else
                 "Enforcement records are not loaded, so prior inspections are not shown."),
    }
    return report


# ---------------------------------------------------------------------- PDF

def render_pdf(report: dict[str, Any], *, compress: bool = True) -> bytes:
    """A two-page PDF of an sst_exposure() report. `compress=False` keeps text searchable for tests."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    def esc(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
    quote = ParagraphStyle("quote", parent=small, leftIndent=10, fontName="Helvetica-Oblique")
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4)
    box = ParagraphStyle("box", parent=body, fontName="Helvetica-Bold", borderColor=colors.black,
                         borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=10)
    warn = ParagraphStyle("warn", parent=box, textColor=colors.HexColor("#8a1f11"),
                          borderColor=colors.HexColor("#8a1f11"))

    d = report.get("directive") or load_directive()
    story: list[Any] = [Paragraph("SST exposure report", h1)]
    if report.get("data_notice"):
        story.append(Paragraph(esc(report["data_notice"]), warn))
    story.append(Paragraph(esc(STATEMENT), box))

    if report.get("error"):
        story.append(Paragraph(esc(report.get("message", "No data.")), body))
    else:
        addr = ", ".join(str(report[k]) for k in ("street_address", "city", "state", "zip5") if report.get(k))
        story.append(Paragraph(f"<b>{esc(report.get('establishment_name') or '')}</b><br/>{esc(addr)}<br/>"
                               f"NAICS {esc(report.get('naics_code') or 'n/a')} &middot; "
                               f"{esc(report.get('sst_group_label') or '')}", body))
        story.append(Paragraph(
            f"Measured against OSHA Instruction {esc(d['directive'])}, {esc(d['title'])}, effective "
            f"{esc(d['effective_date'])}, which uses Form 300A data for calendar years "
            f"{', '.join(str(y) for y in d['data_years'])}. Report date {esc(report.get('as_of', ''))}.", small))
        story.append(Paragraph(f"<b>Scope.</b> {esc(report.get('scope_note', ''))}", body))

        story.append(Paragraph("DART rate by year", h2))
        rows = [["Year", "Employees", "Hours worked", "DART cases", "DART rate", "Industry median", "Used by SST"]]
        for h in report.get("dart_history", []):
            rows.append([
                str(h["year"]),
                "" if h["employees"] is None else f"{h['employees']:,.0f}",
                "not usable" if h["hours_worked"] is None else f"{h['hours_worked']:,.0f}",
                "" if h["n_dart"] is None else f"{h['n_dart']:,.0f}",
                "n/a" if h["dart_rate"] is None else f"{h['dart_rate']:.2f}",
                "n/a" if h["industry_median_dart"] is None else f"{h['industry_median_dart']:.2f}",
                "yes" if h["sst_data_year"] else "",
            ])
        t = Table(rows, hAlign="LEFT", colWidths=[0.6 * inch, 0.9 * inch, 1.1 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch, 0.9 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("ALIGN", (1, 0), (-2, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f3f3")]),
        ]))
        story.append(t)
        story.append(Paragraph("DART rate = (days-away cases + restricted or transferred cases) x 200,000 / "
                               "hours worked. Industry median is the same-year median among filers in the "
                               "same four-digit NAICS group.", small))

        peers = report.get("peers")
        if peers:
            story.append(Paragraph(f"Position among filers, CY {peers['year']}", h2))
            g, i = peers["group"], peers["industry"]
            txt = (f"{_pct(g['share_with_lower_rate'])} of {g['n']:,} compared {esc(g['label'])} establishments "
                   f"reported a lower DART rate. Within {esc(i['label'])}: {_pct(i['share_with_lower_rate'])} of {i['n']:,}.")
            if g.get("p50") is not None:
                txt += (f" In the group, the median rate is {g['p50']:.2f}, the 75th percentile {g['p75']:.2f}, "
                        f"the 90th {g['p90']:.2f} and the 95th {g['p95']:.2f}.")
            story.append(Paragraph(txt, body))

        story.append(Paragraph("The four SST selection lists", h2))
        for c in report.get("criteria", []):
            story.append(Paragraph(f"<b>{esc(c['label'])}.</b> {esc(c['position'])}", body))
            story.append(Paragraph(f"Directive text: \"{esc(c['quote'])}\"", quote))
            if c.get("interpretation_note"):
                story.append(Paragraph(esc(c["interpretation_note"]), small))
            story.append(Spacer(1, 4))

        dc = report.get("deletion_context")
        if dc:
            story.append(Paragraph("Why a listed establishment may still be removed", h2))
            insp = dc.get("inspections")
            if insp is None:
                line = dc["note"]
            elif insp["last_comprehensive_open_date"]:
                line = (f"Most recent comprehensive inspection on record opened {insp['last_comprehensive_open_date']} "
                        f"({dc['months_since_last_comprehensive']} months before this report). {dc['note']}")
            else:
                line = (f"No comprehensive inspection on record ({insp['n_inspections_on_record']} inspections of "
                        f"any scope). {dc['note']}")
            story.append(Paragraph(esc(line), body))
            story.append(Paragraph(f"Directive text: \"{esc(dc['rule'])}\" Also deleted: "
                                   f"{esc('; '.join(dc['other_deletions']))}.", quote))

        story.append(Paragraph("Read this before acting on it", h2))
        ps = d.get("program_size") or {}
        if ps:
            story.append(Paragraph(f"Scale: {esc(ps['quote'])} ({esc(ps['note'])})", small))
        for c in report.get("caveats", []):
            story.append(Paragraph("&bull; " + esc(c), small))
        dv = report.get("data_vintage") or {}
        if dv.get("selection_year_file_snapshot"):
            story.append(Paragraph(f"&bull; CY {d['selection_year']} file snapshot: "
                                   f"{esc(dv['selection_year_file_snapshot'])}. {esc(dv['note'])}", small))
        src = d.get("source", {})
        story.append(Paragraph(f"Directive: {esc(src.get('official_url', ''))}. Text read from "
                               f"{esc(src.get('text_read_from', ''))} on {esc(src.get('retrieved', ''))}. "
                               f"Successor directive announced as of {esc(d.get('successor', {}).get('checked', ''))}: "
                               f"{'yes' if d.get('successor', {}).get('announced') else 'none found'}.", small))
        story.append(Paragraph("Independent analysis of public data. Not affiliated with, endorsed by, or "
                               "produced on behalf of OSHA or the U.S. Department of Labor. Not legal advice.", small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.75 * inch, 0.5 * inch, STATEMENT)
        if report.get("data_notice"):
            canvas.setFillColor(colors.HexColor("#8a1f11"))
            canvas.drawString(0.75 * inch, 0.36 * inch, report["data_notice"][:150])
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                      title="SST exposure report", author="nextrecordable",
                      pageCompression=1 if compress else 0).build(
        story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
