"""Command-line interface.

    nextrecordable run-all              download -> load -> build -> sanity -> train
    nextrecordable download summary [--year 2024 ...]
    nextrecordable load
    nextrecordable load-enforcement --inspection osha_inspection.csv --violation osha_violation.csv
    nextrecordable build
    nextrecordable sanity
    nextrecordable train
    nextrecordable serve
    nextrecordable explore | query "SQL"

Global options: --db PATH (default data/nextrecordable.duckdb).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import config
from .download import download_ita_case_detail, download_ita_summary
from .load import connect, load_all_summary

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Establishment injury risk from OSHA data.")
download_app = typer.Typer(no_args_is_help=True)
app.add_typer(download_app, name="download", help="Download raw files from OSHA.")
console = Console()
# The panel looks at most ten years back for "years since last inspection".
ENFORCEMENT_SINCE_YEAR = 2006
STATE: dict[str, Path] = {"db": config.DB_PATH}


@app.callback()
def main(db: Annotated[Path, typer.Option("--db", help="DuckDB file")] = config.DB_PATH) -> None:
    STATE["db"] = db


def _con(read_only: bool = False):
    return connect(STATE["db"], read_only=read_only)


@download_app.command("summary")
def download_summary(
    year: Annotated[list[int] | None, typer.Option("--year", "-y", help="Year(s); default all")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
):
    """ITA Form 300A summary files."""
    years = year or sorted(config.ITA_SUMMARY_URLS)
    paths = download_ita_summary(years=years, force=force)
    console.print(f"[green]{len(paths)} files in {config.RAW_DIR / 'ita_summary'}[/green]")


@download_app.command("case-detail")
def download_case_detail(
    year: Annotated[list[int] | None, typer.Option("--year", "-y")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
):
    """ITA Form 300/301 case detail files (2023+)."""
    paths = download_ita_case_detail(years=year or sorted(config.ITA_CASE_DETAIL_URLS), force=force)
    console.print(f"[green]{len(paths)} files[/green]")


@app.command()
def load():
    """Extract downloaded ITA zips and load them (all columns as text)."""
    with _con() as con:
        counts = load_all_summary(con)
    console.print(f"[green]{sum(counts.values()):,} rows across {len(counts)} years -> {STATE['db']}[/green]")


@app.command("load-enforcement")
def load_enforcement(
    inspection: Annotated[Path, typer.Option(help="DOL osha_inspection CSV, or a directory of chunk CSVs")],
    violation: Annotated[Path, typer.Option(help="DOL osha_violation CSV, or a directory of chunk CSVs")],
    since_year: Annotated[int, typer.Option(help="Keep inspections opened in this year or later")] = ENFORCEMENT_SINCE_YEAR,
):
    """Load DOL enforcement CSVs (bulk files from data.dol.gov; see DATA-GAPS.md)."""
    from .enforcement import load_inspection_csv, load_violation_csv

    with _con() as con:
        n_i = load_inspection_csv(con, inspection, since_year=since_year)
        n_v = load_violation_csv(con, violation, inspection_table="osha_inspection")
    console.print(f"[green]{n_i:,} inspections, {n_v:,} violations[/green]")


@app.command()
def build(no_enforcement: Annotated[bool, typer.Option("--no-enforcement")] = False):
    """Canonical view, keys, panel, match directory (and enforcement if loaded)."""
    from .pipeline import build as run_build

    with _con() as con:
        out = run_build(con, enforcement=False if no_enforcement else None)
    for k, v in out.items():
        console.print(f"  {k}: {v}")


def _print_gates(gates) -> bool:
    t = Table("gate", "status", "value", "rule")
    colors = {"pass": "green", "warn": "yellow", "fail": "red", "skip": "dim"}
    for g in gates:
        val = "" if g.value is None else (f"{g.value:,.3f}" if isinstance(g.value, float) else f"{g.value:,}")
        t.add_row(g.name, f"[{colors[g.status]}]{g.status}[/]", val, g.rule)
    console.print(t)
    return not any(g.status == "fail" for g in gates)


@app.command()
def sanity(min_rows_per_year: Annotated[int, typer.Option()] = 1_000):
    """Data gates. Exits 1 if any gate fails."""
    from .sanity import run_gates

    with _con(read_only=True) as con:
        ok = _print_gates(run_gates(con, min_rows_per_year=min_rows_per_year))
    raise typer.Exit(0 if ok else 1)


@app.command()
def train(
    rounds: Annotated[int, typer.Option(help="Max boosting rounds")] = 500,
    panel: Annotated[str | None, typer.Option(help="Panel table (default: enforced if built)")] = None,
):
    """Train, save the model, score every establishment, write reports/."""
    from .pipeline import train as run_train

    with _con() as con:
        result = run_train(con, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR,
                           panel_table=panel, num_boost_round=rounds)
    console.print(f"[green]Model -> {config.MODEL_DIR}; report -> {config.REPORTS_DIR / 'METRICS.md'}[/green]")
    tb = result.metrics.get("test_binary", {})
    console.print(f"  test year {result.test_year}: AUC {tb.get('auc', float('nan')):.3f} "
                  f"(size+NAICS baseline {tb.get('baseline_auc', float('nan')):.3f})")


@app.command("run-all")
def run_all(
    skip_download: Annotated[bool, typer.Option("--skip-download")] = False,
    rounds: Annotated[int, typer.Option()] = 500,
    force: Annotated[bool, typer.Option("--force", help="Train even if a sanity gate fails")] = False,
):
    """Everything: download, load, (DOL files in data/raw/dol/), build, sanity, train."""
    from .enforcement import find_dol_files, load_inspection_csv, load_violation_csv
    from .pipeline import build as run_build
    from .pipeline import train as run_train
    from .sanity import run_gates

    if not skip_download:
        download_ita_summary(years=sorted(config.ITA_SUMMARY_URLS))
    with _con() as con:
        counts = load_all_summary(con)
        console.print(f"Loaded {sum(counts.values()):,} ITA rows across {len(counts)} years")
        dol = config.RAW_DIR / "dol"
        insp, viol = find_dol_files(dol)
        if insp and viol:
            load_inspection_csv(con, insp, since_year=ENFORCEMENT_SINCE_YEAR)
            load_violation_csv(con, viol, inspection_table="osha_inspection")
            console.print(f"Loaded DOL enforcement files from {dol}")
        else:
            console.print(f"[yellow]No DOL files in {dol}; training without enforcement features[/yellow]")
        run_build(con)
        ok = _print_gates(run_gates(con))
        if not ok and not force:
            console.print("[red]A sanity gate failed; fix the data or rerun with --force.[/red]")
            raise typer.Exit(1)
        run_train(con, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR, num_boost_round=rounds)
    console.print(f"[green]Done. Read {config.REPORTS_DIR / 'METRICS.md'}, then `nextrecordable serve`.[/green]")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
):
    """Run the API (one process per database file)."""
    import os

    import uvicorn

    os.environ.setdefault("NEXTRECORDABLE_DB_PATH", str(STATE["db"]))
    uvicorn.run(f"{__package__}.api:build_app_from_env", host=host, port=port, factory=True, workers=1)


@app.command()
def explore():
    """Row counts by year and the largest NAICS groups in the latest year."""
    with _con(read_only=True) as con:
        t = Table("year", "rows", "establishments")
        for y, n, k in con.execute(
            "SELECT year, COUNT(*), COUNT(DISTINCT estab_key) FROM ita_panel GROUP BY 1 ORDER BY 1"
        ).fetchall():
            t.add_row(str(y), f"{n:,}", f"{k:,}")
        console.print(t)
        t2 = Table("naics_4", "establishments", "median TCR")
        for code, n, med in con.execute(
            """SELECT naics_4, COUNT(*), median(tcr) FROM ita_panel
               WHERE year = (SELECT max(year) FROM ita_panel) AND naics_4 <> ''
               GROUP BY 1 ORDER BY 2 DESC LIMIT 15"""
        ).fetchall():
            t2.add_row(code, f"{n:,}", "" if med is None else f"{med:.2f}")
        console.print(t2)


@app.command()
def query(sql: Annotated[str, typer.Argument(help="SQL to run (read-only)")]):
    """Run an ad-hoc read-only query."""
    with _con(read_only=True) as con:
        res = con.execute(sql)
        rows = res.fetchall()
        t = Table(*[d[0] for d in res.description])
        for row in rows[:200]:
            t.add_row(*[str(x) for x in row])
        console.print(t)
        if len(rows) > 200:
            console.print(f"[dim](first 200 of {len(rows):,} rows)[/dim]")


if __name__ == "__main__":
    app()
