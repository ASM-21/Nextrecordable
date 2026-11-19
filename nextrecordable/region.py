"""Aggregate SST positions for a region: counts only, no establishment named.

For conversations with a manufacturing extension center or an industry
group: how many plants in the region file with OSHA, how many sit high in
their SST group, how many meet the upward-trend conditions as this product
reads them. Counts under SUPPRESS_BELOW are shown as "<5" so no single plant
can be singled out from a table.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from . import provenance, sst

SUPPRESS_BELOW = 5


def suppress(n: int | None) -> int | str | None:
    if n is None:
        return None
    return f"<{SUPPRESS_BELOW}" if 0 < n < SUPPRESS_BELOW else int(n)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def region_summary(con: duckdb.DuckDBPyConnection, *, name: str, state: str, zip3: list[str],
                   naics2: list[str], panel_table: str = "ita_panel", as_of: date | None = None) -> dict[str, Any]:
    d = sst.load_directive()
    y1, y2, y3 = d["data_years"]
    as_of = as_of or date.today()
    zips = ", ".join(f"'{z}'" for z in zip3 if re.fullmatch(r"\d{3}", z))
    sectors = ", ".join(f"'{n}'" for n in naics2 if re.fullmatch(r"\d{2}", n))
    if not zips or not sectors or not re.fullmatch(r"[A-Z]{2}", state):
        raise ValueError("state must be two letters, zip3 three digits, naics2 two digits")
    keys = [r[0] for r in con.execute(
        f"""SELECT DISTINCT estab_key FROM {panel_table}
            WHERE year IN ({y1}, {y2}, {y3}) AND state = ? AND substr(zip5, 1, 3) IN ({zips})
              AND substr(coalesce(naics_code, ''), 1, 2) IN ({sectors})""", [state]).fetchall()]
    rows = sst.sst_batch(con, keys, panel_table=panel_table)
    comp = [r for r in rows if r["in_comparison"]]

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    filed = {y: con.execute(
        f"""SELECT COUNT(DISTINCT estab_key) FROM {panel_table} WHERE year = {y} AND state = ?
            AND substr(zip5, 1, 3) IN ({zips}) AND substr(coalesce(naics_code, ''), 1, 2) IN ({sectors})""",
        [state]).fetchone()[0] for y in (y1, y2, y3)}
    darts = sorted(r["dart_2023"] for r in comp if r["dart_2023"] is not None)
    median = darts[len(darts) // 2] if darts else None
    groups = sst.group_stats(con)
    recent = 0
    for r in rows:
        lc = r.get("last_comprehensive_inspection")
        if lc:
            months = (as_of.year - int(lc[:4])) * 12 + (as_of.month - int(lc[5:7]))
            recent += months < d["deletions"]["previous_inspection_months"]
    out = {
        "region": name, "definition": {"state": state, "zip3": zip3, "naics2": naics2},
        "as_of": as_of.isoformat(), "directive": d["directive"], "statement": sst.STATEMENT,
        "data_notice": provenance.notice(con),
        "selection_year_snapshot": provenance.snapshot_for(con, y3),
        "establishments_filing": {str(y): suppress(n) for y, n in filed.items()},
        "establishments_in_any_sst_year": suppress(len(rows)),
        "in_comparison": suppress(len(comp)),
        "median_dart_2023_in_comparison": median,
        "national_group_median_dart_2023": {g: s["p50"] for g, s in groups.items()},
        "at_or_above_group_p75": suppress(sum(1 for r in comp if r["share_lower_in_group"] >= 0.75)),
        "at_or_above_group_p90": suppress(sum(1 for r in comp if r["share_lower_in_group"] >= 0.90)),
        "upward_trending": {k: suppress(count(lambda r, k=k: r["upward_trending_status"] == k)) for k in (
            "meets_both_conditions_as_read", "meets_rate_condition_only", "meets_trend_condition_only",
            "meets_neither_condition", "not_evaluable")},
        "selection_year_filing": {k: suppress(count(lambda r, k=k: r["selection_year_filing"] == k))
                                  for k in ("filing_found", "no_filing_found", "not_evaluable")},
        "comprehensive_inspection_within_36_months": suppress(recent),
        "suppression": f"counts from 1 to {SUPPRESS_BELOW - 1} are shown as <{SUPPRESS_BELOW}",
    }
    return out


def render_md(s: dict[str, Any]) -> str:
    def v(x):
        return "n/a" if x is None else (f"{x:,}" if isinstance(x, int) else (f"{x:.2f}" if isinstance(x, float) else str(x)))

    up, fl = s["upward_trending"], s["selection_year_filing"]
    lines = [f"# SST positions: {s['region']}", ""]
    if s["data_notice"]:
        lines += [f"> **{s['data_notice']}**", ""]
    lines += [
        f"> **{s['statement']}**", "",
        f"Establishments in {s['definition']['state']} with ZIP codes starting {', '.join(s['definition']['zip3'])} "
        f"and NAICS sectors {', '.join(s['definition']['naics2'])}, measured against {s['directive']}. "
        f"Counts only; no establishment is named. Generated {s['as_of']} by `nextrecordable region-summary`. "
        f"CY 2023 file snapshot: {s['selection_year_snapshot'] or 'n/a'}.", "",
        "| item | count |", "|---|---|",
        *[f"| filed Form 300A for CY {y} | {v(n)} |" for y, n in s["establishments_filing"].items()],
        f"| compared in CY 2023 (20+ employees, usable rate, not government) | {v(s['in_comparison'])} |",
        f"| at or above the 75th percentile of their national SST group | {v(s['at_or_above_group_p75'])} |",
        f"| at or above the 90th percentile of their national SST group | {v(s['at_or_above_group_p90'])} |",
        f"| meet both upward-trend conditions as this product reads them | {v(up['meets_both_conditions_as_read'])} |",
        f"| meet the CY 2022 rate condition only | {v(up['meets_rate_condition_only'])} |",
        f"| filed CY 2022 with 20+ employees, no CY 2023 filing found | {v(fl['no_filing_found'])} |",
        f"| comprehensive OSHA inspection on record in the last 36 months | {v(s['comprehensive_inspection_within_36_months'])} |",
        "",
        f"Median CY 2023 DART rate among compared establishments here: {v(s['median_dart_2023_in_comparison'])}. "
        "National medians by SST group: "
        + ", ".join(f"{g} {v(m)}" for g, m in sorted(s["national_group_median_dart_2023"].items())) + ".",
        "",
        "OSHA does not publish the DART cutoffs it uses, and three of its four lists are random samples, so "
        "\"at or above the 90th percentile\" is a position, not a list. A missing CY 2023 filing is usually a "
        "rename, closure or move rather than non-compliance. " + s["suppression"].capitalize() + ".",
        "",
    ]
    return "\n".join(lines)


def write(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / slug(summary["region"])
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float))
    base.with_suffix(".md").write_text(render_md(summary))
    return base.with_suffix(".json"), base.with_suffix(".md")
