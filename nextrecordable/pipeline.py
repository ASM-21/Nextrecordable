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

from . import __version__, provenance, sst
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


def build(con: duckdb.DuckDBPyConnection, *, enforcement: bool | None = None,
          reuse_links: bool = False) -> dict[str, Any]:
    """Canonical view -> keyed table -> panel -> directory (-> enforcement) -> SST positions.

    Enforcement steps run when osha_inspection and osha_violation are loaded,
    unless `enforcement=False`. `reuse_links` keeps an existing
    inspection_linked table: linking 1.8 million inspections takes about ten
    minutes, and the links only change when the establishment keys do.
    """
    out: dict[str, Any] = {"summary_tables": build_canonical_view(con)}
    out["keyed_rows"] = build_keyed_view(con)
    out["panel_rows"] = build_panel(con)
    out["panel"] = panel_summary(con)
    out["directory_rows"] = build_directory(con, panel_table="ita_panel")
    have_enf = _exists(con, "osha_inspection") and _exists(con, "osha_violation")
    if have_enf and enforcement is not False:
        if reuse_links and _exists(con, "inspection_linked"):
            out["inspection_links"] = dict(con.execute(
                "SELECT link_method, COUNT(*) FROM inspection_linked GROUP BY 1").fetchall())
        else:
            out["inspection_links"] = link_inspections(con)
        out["enforcement_rows"] = build_enforcement_aggregates(con)
        out["enforced_panel_rows"] = attach_enforcement_to_panel(con)
    elif enforcement:
        raise RuntimeError("Enforcement requested but osha_inspection/osha_violation are not loaded.")
    out["sst_rows"] = sst.build_sst_table(con, panel_table="ita_panel")
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


SYNTHETIC_BASELINE_FILE = "synthetic_baseline.json"

# Headline numbers compared against the synthetic baseline: (label, path into metrics, is a share).
HEADLINE: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("binary AUC", ("test_binary", "auc"), False),
    ("binary AUC, size + NAICS baseline", ("test_binary", "baseline_auc"), False),
    ("binary AUC, establishments unseen in training", ("test_binary_unseen_establishments", "auc"), False),
    ("Brier score", ("test_binary", "brier"), False),
    ("Brier skill vs base rate", ("test_binary", "brier_skill"), True),
    ("expected calibration error (holdout)", ("test_ece",), False),
    ("share of next-year cases in top 10% (binary)", ("test_binary", "capture_cases_top10"), True),
    ("rate deviance skill vs credibility", ("test_rate", "deviance_skill_vs_credibility"), True),
    ("rate deviance skill vs industry", ("test_rate", "deviance_skill_vs_industry"), True),
)


def dig(d: dict | None, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def link_rates(con, *, panel_table: str = "ita_panel") -> dict[str, Any] | None:
    """Inspection link outcomes, all years and within the panel's years."""
    if not _exists(con, "inspection_linked"):
        return None
    lo, hi = con.execute(f"SELECT min(year), max(year) FROM {panel_table}").fetchone()
    out = {}
    for label, where in (("all_years", "TRUE"), ("panel_years", f"inspection_year BETWEEN {lo} AND {hi}")):
        rows = dict(con.execute(
            f"SELECT link_method, COUNT(*) FROM inspection_linked WHERE {where} GROUP BY 1").fetchall())
        n = sum(rows.values())
        linked = rows.get("exact_key", 0) + rows.get("fuzzy", 0)
        out[label] = {"counts": rows, "n": n, "linked_share": linked / n if n else None,
                      "fuzzy_share_of_linked": rows.get("fuzzy", 0) / linked if linked else None}
    out["panel_years_range"] = [lo, hi]
    return out


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
    prov = provenance.summary(con)
    sst_summary = sst.program_summary(con)
    exposure = None
    if "exposure_ok" in {r[1] for r in con.execute(f"PRAGMA table_info('{result.panel_table}')").fetchall()}:
        exposure = dict(zip(("rows", "exposure_rejected", "counts_rejected"), con.execute(
            f"SELECT COUNT(*), COUNT(*) FILTER (WHERE NOT exposure_ok), COUNT(*) FILTER (WHERE NOT counts_ok) "
            f"FROM {result.panel_table}").fetchone(), strict=True))
    baseline_path = reports_dir / SYNTHETIC_BASELINE_FILE
    synthetic = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": platform.python_version(),
        "data_provenance": prov,
        "raw_rows_by_table": years,
        "panel_table": result.panel_table,
        "panel": panel,
        "exposure_checks": exposure,
        "inspection_links": links,
        "link_rates": link_rates(con, panel_table=result.panel_table),
        "split": {"train_through": result.train_through, "val_year": result.val_year,
                  "test_year": result.test_year},
        "rate_served": result.rate_served,
        "metrics": result.metrics,
        "binary_feature_importance_top10": result.feature_importance[:10],
        "rate_feature_importance_top10": result.rate_feature_importance[:10],
        "sst": sst_summary,
        "sanity": gates,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    (reports_dir / "METRICS.md").write_text(render_metrics_md(payload, synthetic))
    return payload


def render_metrics_md(payload: dict, synthetic: dict | None = None) -> str:
    """METRICS.md from a metrics.json payload. Every number in it comes from the payload."""
    m = payload["metrics"]
    tb, tr = m.get("test_binary", {}), m.get("test_rate", {})
    split = payload["split"]
    prov = payload.get("data_provenance") or {"kind": "undeclared", "notice": provenance.NOTICES["undeclared"]}
    lines: list[str] = ["# nextrecordable metrics", ""]
    if prov.get("notice"):
        lines += [f"> **{prov['notice']}**", ""]
    lines += [
        f"Generated {payload['generated_at']} from `{payload['panel_table']}`. Trained on filings through "
        f"{split['train_through']}, early-stopped and calibrated on {split['val_year']}, scored on "
        f"{split['test_year']} (predicting {split['test_year'] + 1} outcomes). Every number below is "
        "recomputed from `reports/metrics.json` by `pipeline.render_metrics_md`.",
        "",
        "## Data",
        "",
        "| item | value |",
        "|---|---|",
        f"| data provenance | {prov['kind']} |",
        *[f"| {t} rows | {_fmt(n)} |" for t, n in payload["raw_rows_by_table"].items()],
        f"| panel establishment-years | {_fmt(payload['panel']['total_rows'])} |",
        f"| distinct establishments | {_fmt(payload['panel']['distinct_establishments'])} |",
        f"| key-years with duplicate filings (deduplicated) | {_fmt(payload['panel']['key_years_with_duplicates'])} |",
    ]
    ex = payload.get("exposure_checks")
    if ex:
        lines += [
            f"| establishment-years with impossible hours or headcount (kept, no rate) | {_fmt(ex['exposure_rejected'])} |",
            f"| establishment-years with impossible case counts (kept, no rate) | {_fmt(ex['counts_rejected'])} |",
        ]
    if prov.get("files"):
        lines += ["", "Source files:", "", "| file | kind | snapshot | origin |", "|---|---|---|---|"]
        lines += [f"| {f['file']} | {f['kind']} | {f.get('snapshot') or ''} | {f.get('origin') or ''} |"
                  for f in prov["files"]]
    if prov.get("note"):
        lines += ["", prov["note"]]

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
        f"| Brier score (always-base-rate: {_fmt(tb.get('brier_reference'))}) | {_fmt(tb.get('brier'))} | |",
        f"| Brier skill vs base rate | {_fmt(tb.get('brier_skill'), pct=True)} | |",
        f"| expected calibration error, holdout year | {_fmt(m.get('test_ece'))} | |",
        f"| top-10% lift (ceiling = 1 / base rate) | {_fmt(tb.get('lift_top10'))} "
        f"(ceiling {_fmt(tb.get('lift_ceiling'))}) | |",
        "",
        f"Base rate: {_fmt(tb.get('base_rate'), pct=True)} of establishments logged a case.",
        "",
        "### Split by establishment",
        "",
        "Every establishment that filed in a training year and again in the holdout year is in both. "
        "Scoring only establishments the model never saw separates prediction from recognition.",
        "",
        "| holdout rows | n | base rate | AUC | baseline AUC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("seen", "unseen"):
        b = m.get(f"test_binary_{label}_establishments")
        if b:
            lines.append(f"| establishments {label} in training/validation | {_fmt(b['n'])} | "
                         f"{_fmt(b['base_rate'], pct=True)} | {_fmt(b['auc'])} | {_fmt(b.get('baseline_auc'))} | "
                         f"{_fmt(b['brier'])} |")
    cal = m.get("test_calibration") or []
    if cal:
        lines += ["", "### Reliability on the holdout year", "",
                  "Calibrator fitted on the validation year; ten equal-count bins of the holdout year.", "",
                  "| bin | n | mean predicted | observed | gap |", "|---|---|---|---|---|"]
        lines += [f"| {b['bin']} | {_fmt(b['n'])} | {_fmt(b['mean_predicted'])} | {_fmt(b['observed_rate'])} | "
                  f"{_fmt(b['calibration_error'])} |" for b in cal]

    served = payload.get("rate_served", "model")
    lines += [
        "",
        "## Rate model: expected cases per 200,000 hours next year",
        "",
        "Judged on realized next-year cases over next-year hours. "
        + ("**Served rate: the credibility estimate.** The boosted model did not beat it on the holdout "
           "year, so reports and scores use the credibility estimate and show no rate drivers."
           if served == "credibility" else "Served rate: the boosted model."),
        "",
        "| metric | boosted model | credibility baseline | industry rate |",
        "|---|---|---|---|",
    ]
    for label, key, pct in (
        ("Poisson deviance (lower is better)", "poisson_deviance", False),
        ("Spearman vs realized rate", "spearman_vs_realized_tcr", False),
        ("top-10% pooled rate / overall", "top10_pooled_rate_lift", False),
        ("predicted / actual cases", "predicted_over_actual_cases", False),
        ("share of next-year cases in top 10% by expected cases", "capture_cases_top10", True),
    ):
        vals = [tr.get(k, {}).get(key) for k in ("model", "credibility", "industry")]
        lines.append(f"| {label} | " + " | ".join(_fmt(v, pct=pct) for v in vals) + " |")
    lines += [
        "",
        f"Largest sites first (rank by hours worked) puts "
        f"{_fmt(tr.get('largest_sites_capture_cases_top10'), pct=True)} of next-year cases in the top 10%.",
        "",
        f"Deviance skill of the boosted model vs credibility baseline: "
        f"{_fmt(tr.get('deviance_skill_vs_credibility'), pct=True)}; vs industry rate: "
        f"{_fmt(tr.get('deviance_skill_vs_industry'), pct=True)}.",
    ]

    lr = payload.get("link_rates")
    if lr:
        py = lr["panel_years"]
        lines += [
            "",
            "## Matching",
            "",
            f"DOL inspections opened {lr['panel_years_range'][0]} to {lr['panel_years_range'][1]} linked to an ITA "
            f"filer: {_fmt(py['linked_share'], pct=True)} of {_fmt(py['n'])} "
            f"(exact key {_fmt(py['counts'].get('exact_key', 0))}, fuzzy {_fmt(py['counts'].get('fuzzy', 0))}, "
            f"ambiguous and left unlinked {_fmt(py['counts'].get('ambiguous', 0))}). Most inspections are of sites "
            "that never file with ITA (construction, under 20 employees), so this share is low by design. "
            f"Cross-year link: {_fmt(payload['panel']['rows_with_lag'])} establishment-years have last year's filing.",
        ]

    sm = payload.get("sst")
    if sm:
        lines += ["", f"## SST positions (CY {sm['selection_year']} selection year)", "",
                  "Counts of establishments by position against CPL 02-01-067. Positions, not inspection lists: "
                  "OSHA publishes no DART cutoffs and three of four lists are random samples.", "",
                  "| item | value |", "|---|---|",
                  f"| filed for CY {sm['selection_year']} (file snapshot: {sm.get('selection_year_snapshot') or 'n/a'}) | "
                  f"{_fmt(sm['filed_selection_year'])} |",
                  f"| compared (20+ employees, general industry, usable rate, not government) | {_fmt(sm['in_comparison'])} |",
                  f"| meet both upward-trend conditions as this product reads them | {_fmt(sm['meet_both_upward_conditions'])} |",
                  f"| filed CY {sm['selection_year'] - 1} with 20+ employees, no CY {sm['selection_year']} filing found | "
                  f"{_fmt(sm['no_filing_found_selection_year'])} |"]
        for g, st in sorted((sm.get("groups") or {}).items()):
            lines.append(f"| {g}: n, median / p75 / p90 / p95 DART | {_fmt(st['n'])}: {_fmt(st['p50'], digits=2)} / "
                         f"{_fmt(st['p75'], digits=2)} / {_fmt(st['p90'], digits=2)} / {_fmt(st['p95'], digits=2)} |")

    if synthetic:
        sm_ = synthetic.get("metrics", {})
        lines += ["", "## Against the synthetic baseline", "",
                  f"The same code on the synthetic fixtures (`{SYNTHETIC_BASELINE_FILE}`, generated "
                  f"{synthetic.get('generated_at', 'n/a')}). Synthetic data was built to carry signal, so a drop "
                  "here is expected and is the honest number.", "",
                  "| metric | real | synthetic | change |", "|---|---|---|---|"]
        for label, path, pct in HEADLINE:
            a, b = dig(m, path), dig(sm_, path)
            delta = (a - b) if isinstance(a, int | float) and isinstance(b, int | float) else None
            lines.append(f"| {label} | {_fmt(a, pct=pct)} | {_fmt(b, pct=pct)} | "
                         f"{'n/a' if delta is None else (f'{100 * delta:+.1f} pts' if pct else f'{delta:+.3f}')} |")

    gates = payload["sanity"]
    lines += [
        "",
        "## Sanity gates",
        "",
        "| gate | status | value | rule |",
        "|---|---|---|---|",
        *[f"| {g['name']} | {g['status']} | {_fmt(g['value'])} | {g['rule']} |" for g in gates["gates"]],
        "",
    ]
    return "\n".join(lines)
