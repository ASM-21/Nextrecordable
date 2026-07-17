"""Command-line interface for the osha_risk pipeline.

Usage:
    python -m osha_risk.cli download summary
    python -m osha_risk.cli download summary --year 2024 --year 2023
    python -m osha_risk.cli download case-detail
    python -m osha_risk.cli load
    python -m osha_risk.cli explore
    python -m osha_risk.cli query "SELECT data_year, COUNT(*) FROM ita_summary GROUP BY 1"
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import DB_PATH, ITA_CASE_DETAIL_URLS, ITA_SUMMARY_URLS
from .download import download_ita_case_detail, download_ita_summary
from .load import connect, load_all_summary

app = typer.Typer(no_args_is_help=True, add_completion=False)
download_app = typer.Typer(no_args_is_help=True)
app.add_typer(download_app, name="download", help="Download raw data from OSHA")

console = Console()


# ---------- download subcommands ----------

@download_app.command("summary")
def download_summary(
    year: Annotated[
        list[int] | None,
        typer.Option("--year", "-y", help="Specific year(s). Repeat for multiple. Default: all available."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if file exists")] = False,
):
    """Download ITA Form 300A summary data."""
    years = year or sorted(ITA_SUMMARY_URLS)
    console.print(f"[bold]Downloading summary years:[/bold] {years}")
    paths = download_ita_summary(years=years, force=force)
    console.print(f"[green]Downloaded {len(paths)} files[/green]")


@download_app.command("case-detail")
def download_case_detail(
    year: Annotated[list[int] | None, typer.Option("--year", "-y")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
):
    """Download ITA Form 300/301 case detail data (2023+ only)."""
    years = year or sorted(ITA_CASE_DETAIL_URLS)
    console.print(f"[bold]Downloading case-detail years:[/bold] {years}")
    paths = download_ita_case_detail(years=years, force=force)
    console.print(f"[green]Downloaded {len(paths)} files[/green]")


# ---------- load ----------

@app.command()
def load():
    """Extract downloaded zips and load CSVs into DuckDB."""
    with connect() as con:
        counts = load_all_summary(con)
    console.print(f"\n[green]Loaded {sum(counts.values()):,} total rows across {len(counts)} years[/green]")
    console.print(f"DuckDB at: {DB_PATH}")


# ---------- explore ----------

@app.command()
def explore():
    """First-pass exploration: row counts, schema, top NAICS."""
    if not DB_PATH.exists():
        console.print(f"[red]Database not found at {DB_PATH}. Run `load` first.[/red]")
        raise typer.Exit(1)

    with connect(read_only=True) as con:
        # Per-year row counts
        rows = con.execute(
            """
            SELECT data_year, COUNT(*) AS rows
            FROM ita_summary
            GROUP BY data_year
            ORDER BY data_year
            """
        ).fetchall()

        t = Table(title="ITA Summary: rows per year")
        t.add_column("Year")
        t.add_column("Rows", justify="right")
        for year, n in rows:
            t.add_row(str(year), f"{n:,}")
        console.print(t)

        # Schema of unified view
        schema = con.execute("DESCRIBE ita_summary").fetchall()
        console.print(f"\n[bold]Unified schema: {len(schema)} columns[/bold]")
        for col, dtype, *_ in schema[:30]:
            console.print(f"  {col:<40} {dtype}")
        if len(schema) > 30:
            console.print(f"  ... and {len(schema) - 30} more")

        # Try a few common column names for NAICS to show top industries
        for naics_col in ("naics_code", "naics", "industry_code"):
            try:
                top = con.execute(
                    f"""
                    SELECT {naics_col}, COUNT(*) AS n
                    FROM ita_summary
                    WHERE data_year = (SELECT MAX(data_year) FROM ita_summary)
                      AND {naics_col} IS NOT NULL
                    GROUP BY 1
                    ORDER BY n DESC
                    LIMIT 15
                    """
                ).fetchall()
                t2 = Table(title=f"Top 15 NAICS (latest year, col={naics_col})")
                t2.add_column("NAICS")
                t2.add_column("Establishments", justify="right")
                for code, n in top:
                    t2.add_row(str(code), f"{n:,}")
                console.print(t2)
                break
            except Exception:
                continue


# ---------- ad-hoc query ----------

@app.command()
def query(sql: Annotated[str, typer.Argument(help="SQL string to execute")]):
    """Run an ad-hoc SQL query against the database."""
    with connect(read_only=True) as con:
        result = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        t = Table(*cols)
        for row in result[:200]:
            t.add_row(*[str(x) for x in row])
        console.print(t)
        if len(result) > 200:
            console.print(f"[dim](showing first 200 of {len(result):,} rows)[/dim]")


if __name__ == "__main__":
    app()
