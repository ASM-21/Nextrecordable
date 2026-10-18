"""Extract ITA zips and land CSVs in DuckDB.

Every column is read as VARCHAR. Type inference is the wrong default for
these files: DuckDB samples the first rows, and if no leading-zero ZIP shows
up in the sample the column becomes BIGINT and every CT/MA/NJ ZIP after that
loses its leading zero. Typing happens later, explicitly, in schema.py.

Each file lands in its own per-year table (`ita_summary_<year>`) with a
`file_year` column; schema.py unions them under one canonical schema.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb

from .config import DB_PATH, PROCESSED_DIR, RAW_DIR


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a zip archive and return paths to extracted files.

    Refuses members that would land outside dest_dir (zip-slip).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            target = (dest_dir / member).resolve()
            if root not in target.parents:
                raise ValueError(f"Refusing to extract {member!r} outside {dest_dir}")
            z.extract(member, dest_dir)
            out.append(target)
    return out


def find_csv(directory: Path) -> Path | None:
    """The largest CSV under `directory` (zips sometimes carry a small metadata CSV)."""
    csvs = list(directory.rglob("*.csv"))
    return max(csvs, key=lambda p: p.stat().st_size) if csvs else None


def connect(db_path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB database (default: config.DB_PATH)."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def read_csv_varchar(con: duckdb.DuckDBPyConnection, csv_path: Path, table: str, extra_select: str = "") -> int:
    """Create `table` from a CSV with every column as VARCHAR. Returns row count."""
    path = str(csv_path).replace("'", "''")
    extra = f", {extra_select}" if extra_select else ""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""
        CREATE TABLE {table} AS
        SELECT *{extra}
        FROM read_csv('{path}', header = true, all_varchar = true,
                      null_padding = true, ignore_errors = true)
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def load_summary_csv(
    csv_path: Path,
    year: int,
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str | None = None,
) -> int:
    """Load one ITA 300A summary CSV into `ita_summary_<year>`. Returns row count.

    `year` is the file's nominal calendar year. Rows carry their own
    `year_filing_for`, which schema.py prefers when present.
    """
    table = table_name or f"ita_summary_{year}"
    return read_csv_varchar(con, csv_path, table, extra_select=f"CAST({int(year)} AS INTEGER) AS file_year")


def load_all_summary(con: duckdb.DuckDBPyConnection, raw_dir: Path | None = None) -> dict[int, int]:
    """Extract and load every downloaded ITA summary zip. Returns {year: rows}."""
    summary_dir = (raw_dir or RAW_DIR) / "ita_summary"
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
        counts[year] = load_summary_csv(csv, year, con)
        print(f"  -> {counts[year]:,} rows")
    return counts
