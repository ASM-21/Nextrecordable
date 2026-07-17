"""Extract ITA zips and load CSVs into DuckDB.

OSHA's ITA schema has drifted across years (added/renamed columns, e.g. the
establishment_id field added in recent years, OIICS codes added to case detail
in 2023+). We:

  1. Land each year's CSV into its own per-year table (e.g. `ita_summary_2024`).
  2. Build a unified VIEW `ita_summary` that UNIONs them by-name, filling
     missing columns with NULL.

This keeps year-specific quirks visible while still giving us one queryable
panel for modeling.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb

from .config import DB_PATH, PROCESSED_DIR, RAW_DIR


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a zip archive and return paths to extracted files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        z.extractall(dest_dir)
        return [dest_dir / m for m in members]


def find_csv(directory: Path) -> Path | None:
    """Find the first CSV in a directory (recursive)."""
    csvs = list(directory.rglob("*.csv"))
    if not csvs:
        return None
    # Prefer the largest CSV (sometimes zips contain a small metadata CSV too)
    return max(csvs, key=lambda p: p.stat().st_size)


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def load_summary_csv(
    csv_path: Path,
    year: int,
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str | None = None,
) -> int:
    """Load one year of ITA summary CSV into DuckDB. Returns row count.

    We use DuckDB's native CSV reader (read_csv_auto) which is fast, tolerant
    of dirty data, and infers types reasonably. The `data_year` column is
    appended so downstream code never has to guess which year a row came from.
    """
    table = table_name or f"ita_summary_{year}"

    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""
        CREATE TABLE {table} AS
        SELECT *, CAST({year} AS INTEGER) AS data_year
        FROM read_csv_auto(
            '{csv_path.as_posix()}',
            header=true,
            sample_size=50000,
            ignore_errors=true,
            null_padding=true
        )
        """
    )

    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def build_unified_summary_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create a `ita_summary` view that unions all per-year tables by column name."""
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'ita_summary_%' "
            "ORDER BY table_name"
        ).fetchall()
    ]
    if not tables:
        return

    union_sql = " UNION ALL BY NAME ".join(f"SELECT * FROM {t}" for t in tables)
    con.execute(f"CREATE OR REPLACE VIEW ita_summary AS {union_sql}")


def load_all_summary(con: duckdb.DuckDBPyConnection) -> dict[int, int]:
    """Walk RAW_DIR / ita_summary / *.zip, extract, and load into DuckDB.

    Returns a dict of {year: rows_loaded}.
    """
    summary_dir = RAW_DIR / "ita_summary"
    extract_root = PROCESSED_DIR / "ita_summary"
    counts: dict[int, int] = {}

    for zip_path in sorted(summary_dir.glob("ita_summary_*.zip")):
        try:
            year = int(zip_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            print(f"Skipping unrecognized file: {zip_path.name}")
            continue

        year_dir = extract_root / str(year)
        if not year_dir.exists() or not list(year_dir.rglob("*.csv")):
            print(f"Extracting {zip_path.name} ...")
            extract_zip(zip_path, year_dir)

        csv = find_csv(year_dir)
        if csv is None:
            print(f"  WARNING: no CSV found for year {year}")
            continue

        print(f"Loading {year} from {csv.name} ...")
        n = load_summary_csv(csv, year, con)
        counts[year] = n
        print(f"  -> {n:,} rows")

    build_unified_summary_view(con)
    return counts
