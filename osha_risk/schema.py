"""Resolve OSHA ITA column names to canonical fields.

OSHA's 300A schema has drifted across years. Column names are sometimes
`naics_code` and sometimes `naics`; `establishment_name` is sometimes
`company_name`; etc. The `config.NAICS_COLUMN_CANDIDATES` (and friends)
tuples enumerate the synonyms we've observed.

This module gives us:

  - `resolve_column(con, table, candidates)` -> first matching column name,
    or None if no candidate is present
  - `canonical_select_sql(con, table)` -> a SELECT clause that aliases the
    found columns to canonical names, with NULLs for any missing fields

We use this to build a clean "canonical" view on top of the raw per-year
tables so downstream feature code never has to think about schema drift.
"""

from __future__ import annotations

import duckdb

from .config import (
    ANNUAL_EMPLOYEES_CANDIDATES,
    CITY_COLUMN_CANDIDATES,
    HOURS_CANDIDATES,
    NAICS_COLUMN_CANDIDATES,
    NAME_COLUMN_CANDIDATES,
    SIZE_COLUMN_CANDIDATES,
    STATE_COLUMN_CANDIDATES,
    STREET_COLUMN_CANDIDATES,
    ZIP_COLUMN_CANDIDATES,
)

# canonical_field -> ordered list of candidate source columns
CANONICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "establishment_name": NAME_COLUMN_CANDIDATES,
    "street_address": STREET_COLUMN_CANDIDATES,
    "city": CITY_COLUMN_CANDIDATES,
    "state": STATE_COLUMN_CANDIDATES,
    "zip_code": ZIP_COLUMN_CANDIDATES,
    "naics_code": NAICS_COLUMN_CANDIDATES,
    "size_code": SIZE_COLUMN_CANDIDATES,
    "annual_average_employees": ANNUAL_EMPLOYEES_CANDIDATES,
    "total_hours_worked": HOURS_CANDIDATES,
}

# Count columns we want as canonical NUMERIC fields. These have stable enough names
# across years that we just enumerate the known forms.
INCIDENT_COUNT_FIELDS: dict[str, tuple[str, ...]] = {
    "total_deaths": ("total_deaths",),
    "total_dafw_cases": ("total_dafw_cases", "total_dafw_count"),
    "total_djtr_cases": ("total_djtr_cases", "total_djtr_count"),
    "total_other_cases": ("total_other_cases", "total_other_count"),
    "total_injuries": ("total_injuries",),
    "total_skin_disorders": ("total_skin_disorders",),
    "total_respiratory_conditions": ("total_respiratory_conditions",),
    "total_poisonings": ("total_poisonings",),
    "total_hearing_loss": ("total_hearing_loss",),
    "total_other_illnesses": ("total_other_illnesses",),
}

# Stable bookkeeping columns
PASSTHROUGH_FIELDS: tuple[str, ...] = ("data_year", "establishment_id")


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return the set of column names (lowercased) for a DuckDB table."""
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    return {r[1].lower() for r in rows}


def resolve_column(
    con: duckdb.DuckDBPyConnection, table: str, candidates: tuple[str, ...]
) -> str | None:
    """Return the first candidate present in `table`, or None."""
    cols = table_columns(con, table)
    for c in candidates:
        if c.lower() in cols:
            return c
    return None


def canonical_select_sql(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Build a SELECT clause for `table` that emits canonical column names.

    Missing columns are emitted as `NULL AS canonical_name` so every per-year
    table produces an identical schema in the unified view.
    """
    cols_in_table = table_columns(con, table)
    pieces: list[str] = []

    # Pass-through (data_year, establishment_id if present)
    for f in PASSTHROUGH_FIELDS:
        if f.lower() in cols_in_table:
            pieces.append(f'"{f}" AS {f}')
        else:
            pieces.append(f"NULL AS {f}")

    # String / identifier fields
    for canonical, candidates in CANONICAL_FIELDS.items():
        found = resolve_column(con, table, candidates)
        if found:
            pieces.append(f'CAST("{found}" AS VARCHAR) AS {canonical}')
        else:
            pieces.append(f"CAST(NULL AS VARCHAR) AS {canonical}")

    # Numeric incident counts
    for canonical, candidates in INCIDENT_COUNT_FIELDS.items():
        found = resolve_column(con, table, candidates)
        if found:
            pieces.append(
                f'TRY_CAST("{found}" AS INTEGER) AS {canonical}'
            )
        else:
            pieces.append(f"CAST(NULL AS INTEGER) AS {canonical}")

    return "SELECT " + ",\n  ".join(pieces) + f"\nFROM {table}"


def build_canonical_view(con: duckdb.DuckDBPyConnection, view_name: str = "ita_canonical") -> int:
    """Create a view that unions all per-year tables under a stable canonical schema.

    Returns the number of source tables included.
    """
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'ita_summary_%' "
            "ORDER BY table_name"
        ).fetchall()
    ]
    if not tables:
        raise RuntimeError("No ita_summary_<year> tables found. Run `load` first.")

    selects = [canonical_select_sql(con, t) for t in tables]
    union_sql = "\nUNION ALL\n".join(f"({s})" for s in selects)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {union_sql}")
    return len(tables)
