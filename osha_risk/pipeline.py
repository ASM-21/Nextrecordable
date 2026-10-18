"""End-to-end steps: build tables, train, score, and write a metrics report.

    load (raw CSVs) -> build -> sanity -> train -> serve

`build` and `train` are what `nextrecordable run-all` calls after loading;
tests run the same functions on synthetic files.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from . import __version__
from .enforcement import attach_enforcement_to_panel, build_enforcement_aggregates, link_inspections
from .features import build_keyed_view, build_panel, panel_summary
from .match import build_directory
from .models.baseline import TrainResult, materialize_scores, save_model, train_baseline
from .sanity import run_gates, summarize
from .schema import build_canonical_view, summary_tables


def _exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0])


def build(con: duckdb.DuckDBPyConnection, *, enforcement: bool | None = None) -> dict[str, Any]:
    """Canonical view -> keyed table -> panel -> directory (-> enforcement).

    Enforcement steps run when osha_inspection and osha_violation are loaded,
    unless `enforcement=False`.
    """
    out: dict[str, Any] = {"summary_tables": build_canonical_view(con)}
    out["keyed_rows"] = build_keyed_view(con)
    out["panel_rows"] = build_panel(con)
    out["panel"] = panel_summary(con)
    out["directory_rows"] = build_directory(con, panel_table="ita_panel")
    have_enf = _exists(con, "osha_inspection") and _exists(con, "osha_violation")
    if have_enf and enforcement is not False:
        out["inspection_links"] = link_inspections(con)
        out["enforcement_rows"] = build_enforcement_aggregates(con)
        out["enforced_panel_rows"] = attach_enforcement_to_panel(con)
    elif enforcement:
        raise RuntimeError("Enforcement requested but osha_inspection/osha_violation are not loaded.")
    return out


def default_panel(con) -> str:
    return "ita_panel_enforced" if _exists(con, "ita_panel_enforced") else "ita_panel"


def default_split(con, panel_table: str) -> tuple[int, int, int]:
    """Latest year with known next-year outcomes is the test year."""
    years = [r[0] for r in con.execute(
        f"SELECT DISTINCT year FROM {panel_table} WHERE target_any_recordable_next IS NOT NULL ORDER BY 1"
    ).fetchall()]
    if len(years) < 3:
        raise ValueError(f"Need at least 3 years with next-year outcomes to train; have {years}.")
    test = years[-1]
    return test - 2, test - 1, test


def train(
    con: duckdb.DuckDBPyConnection,
    *,
    model_dir: Path,
    reports_dir: Path | None = None,
    panel_table: str | None = None,
    num_boost_round: int = 500,
    gate_kwargs: dict | None = None,
) -> TrainResult:
    """Train on the default split, save the model, materialize scores, write reports."""
    panel = panel_table or default_panel(con)
    train_through, val_year, test_year = default_split(con, panel)
    result = train_baseline(
        con, train_through=train_through, val_year=val_year, test_year=test_year,
        panel_table=panel, num_boost_round=num_boost_round,
    )
    save_model(result, model_dir)
    materialize_scores(con, result)
    if reports_dir is not None:
        write_reports(con, result, reports_dir, gate_kwargs=gate_kwargs)
    return result


# ------------------------------------------------------------------ reports

def _fmt(v, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:,.{digits}f}" if isinstance(v, float) else f"{v:,}"


def write_reports(con, result: TrainResult, reports_dir: Path, *, gate_kwargs: dict | None = None) -> dict:
    """reports/metrics.json (everything) and reports/METRICS.md (the readable part)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    gates = summarize(run_gates(con, **(gate_kwargs or {})))
    years = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in summary_tables(con)}
    panel = panel_summary(con, result.panel_table)
    links = None
    if _exists(con, "inspection_linked"):
        links = dict(con.execute("SELECT link_method, COUNT(*) FROM inspection_linked GROUP BY 1").fetchall())
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": platform.python_version(),
        "raw_rows_by_table": years,
        "panel_table": result.panel_table,
        "panel": panel,
        "inspection_links": links,
        "split": {"train_through": result.train_through, "val_year": result.val_year,
                  "test_year": result.test_year},
        "metrics": result.metrics,
        "binary_feature_importance_top10": result.feature_importance[:10],
        "rate_feature_importance_top10": result.rate_feature_importance[:10],
        "sanity": gates,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))

    tb = result.metrics.get("test_binary", {})
    tr = result.metrics.get("test_rate", {})
    lines = [
        "# nextrecordable metrics",
        "",
        f"Generated {payload['generated_at']} from `{result.panel_table}`. Trained on filings through "
        f"{result.train_through}, early-stopped and calibrated on {result.val_year}, scored on "
        f"{result.test_year} (predicting {result.test_year + 1} outcomes).",
        "",
        "## Data",
        "",
        "| item | value |",
        "|---|---|",
        *[f"| {t} rows | {_fmt(n)} |" for t, n in years.items()],
        f"| panel establishment-years | {_fmt(panel['total_rows'])} |",
        f"| distinct establishments | {_fmt(panel['distinct_establishments'])} |",
        f"| key-years with duplicate filings (deduplicated) | {_fmt(panel['key_years_with_duplicates'])} |",
    ]
    if links:
        lines.append(f"| inspections by link method | {', '.join(f'{k}: {v:,}' for k, v in sorted(links.items()))} |")
    lines += [
        "",
        "## Binary model: any recordable case next year",
        "",
        "| metric | model | size + NAICS baseline |",
        "|---|---|---|",
        f"| AUC | {_fmt(tb.get('auc'))} | {_fmt(tb.get('baseline_auc'))} |",
        f"| share of next-year cases in top 10% | {_fmt(tb.get('capture_cases_top10'), pct=True)} | "
        f"{_fmt(tb.get('baseline_capture_cases_top10'), pct=True)} |",
        f"| share of next-year DART cases in top 10% | {_fmt(tb.get('capture_dart_top10'), pct=True)} | "
        f"{_fmt(tb.get('baseline_capture_dart_top10'), pct=True)} |",
        f"| Brier score | {_fmt(tb.get('brier'))} | |",
        f"| top-10% lift (ceiling = 1 / base rate) | {_fmt(tb.get('lift_top10'))} "
        f"(ceiling {_fmt(tb.get('lift_ceiling'))}) | |",
        "",
        f"Base rate: {_fmt(tb.get('base_rate'), pct=True)} of establishments logged a case. "
        "This model answers whether any case occurs; to find where next year's cases will be, "
        "rank by expected cases (below).",
        "",
        "## Rate model: expected cases per 200,000 hours next year",
        "",
        "Judged on realized next-year cases over next-year hours.",
        "",
        "| metric | model | credibility baseline | industry rate |",
        "|---|---|---|---|",
    ]
    for label, key, pct in (
        ("Poisson deviance (lower is better)", "poisson_deviance", False),
        ("Spearman vs realized rate", "spearman_vs_realized_tcr", False),
        ("top-10% pooled rate / overall", "top10_pooled_rate_lift", False),
        ("predicted / actual cases", "predicted_over_actual_cases", False),
        ("share of next-year cases in top 10% by expected cases", "capture_cases_top10", True),
    ):
        vals = [tr.get(m, {}).get(key) for m in ("model", "credibility", "industry")]
        lines.append(f"| {label} | " + " | ".join(_fmt(v, pct=pct) for v in vals) + " |")
    lines += [
        "",
        f"Largest sites first (rank by hours worked) puts "
        f"{_fmt(tr.get('largest_sites_capture_cases_top10'), pct=True)} of next-year cases in the top 10%.",
        "",
        f"Deviance skill vs credibility baseline: {_fmt(tr.get('deviance_skill_vs_credibility'), pct=True)}; "
        f"vs industry rate: {_fmt(tr.get('deviance_skill_vs_industry'), pct=True)}.",
        "",
        "## Sanity gates",
        "",
        "| gate | status | value | rule |",
        "|---|---|---|---|",
        *[f"| {g['name']} | {g['status']} | {_fmt(g['value'])} | {g['rule']} |" for g in gates["gates"]],
        "",
    ]
    (reports_dir / "METRICS.md").write_text("\n".join(lines))
    return payload
