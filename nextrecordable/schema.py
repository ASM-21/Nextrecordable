"""Map raw ITA 300A tables (all VARCHAR) onto one typed, canonical schema.

Column names drift across release years; config.*_CANDIDATES lists the known
synonyms. Field meanings and codes follow OSHA's summary data dictionary
(osha.gov/sites/default/files/summary_data_dictionary.pdf), including:

  * year_filing_for is the calendar year the injuries occurred. It is used
    as `data_year` when present; the file's nominal year is the fallback.
  * size code 2 (20-249 employees) was split into 21 (20-99) and 22
    (100-249) starting with 2023 data. `size_code` folds 21/22 back into 2 so
    the category means the same thing in every year; `size_code_raw` keeps
    the original.
  * no_injuries_illnesses: 1 = had recordable cases, 2 = did not.

Two things the first real files taught this module (each has a test):

  * Some year files write every number with two decimals, identifiers
    included: establishment_id "439896.00", size "2.00", year "2021.00".
    Left alone, that year's establishments never share an ID with the years
    around it and its size class is a category of its own. ID_TEXT_FIELDS
    drop a trailing ".0", ".00", ...
  * created_timestamp comes in at least four formats across years
    ("1/2/2020 15:14", "1/2/2021 18:48:47", "1/2/2023", "02JAN24:13:03:00").
    None of them casts to TIMESTAMP, so "latest submission wins" was decided
    by string order, where "9/9/2020" beats "10/1/2020". `created_at` is the
    parsed value; the panel orders by it.
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

TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "establishment_id": ("establishment_id",),
    "ein": ("ein",),
    "establishment_name": NAME_COLUMN_CANDIDATES,
    "company_name": ("company_name",),
    "street_address": STREET_COLUMN_CANDIDATES,
    "city": CITY_COLUMN_CANDIDATES,
    "state": STATE_COLUMN_CANDIDATES,
    "zip_code": ZIP_COLUMN_CANDIDATES,
    "naics_code": NAICS_COLUMN_CANDIDATES,
    "size_code_raw": SIZE_COLUMN_CANDIDATES,
    "establishment_type": ("establishment_type",),
    "created_timestamp": ("created_timestamp",),
    "change_reason": ("change_reason",),
}

# Identifier-like text: a trailing ".0"/".00" is an export artifact, not part of the value.
ID_TEXT_FIELDS: frozenset[str] = frozenset({"establishment_id", "ein", "naics_code", "size_code_raw"})

TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%d%b%y:%H:%M:%S", "%d%b%Y:%H:%M:%S",
)

REAL_FIELDS: dict[str, tuple[str, ...]] = {
    "annual_average_employees": ANNUAL_EMPLOYEES_CANDIDATES,
    "total_hours_worked": HOURS_CANDIDATES,
}

INT_FIELDS: dict[str, tuple[str, ...]] = {
    "year_filing_for": ("year_filing_for",),
    "no_injuries_illnesses": ("no_injuries_illnesses",),
    "total_deaths": ("total_deaths",),
    "total_dafw_cases": ("total_dafw_cases", "total_dafw_count"),
    "total_djtr_cases": ("total_djtr_cases", "total_djtr_count"),
    "total_other_cases": ("total_other_cases", "total_other_count"),
    "total_dafw_days": ("total_dafw_days",),
    "total_djtr_days": ("total_djtr_days",),
    "total_injuries": ("total_injuries",),
    "total_skin_disorders": ("total_skin_disorders",),
    "total_respiratory_conditions": ("total_respiratory_conditions",),
    "total_poisonings": ("total_poisonings",),
    "total_hearing_loss": ("total_hearing_loss",),
    "total_other_illnesses": ("total_other_illnesses",),
}

CANONICAL_COLUMNS: tuple[str, ...] = (
    ("file_year", "data_year", "size_code", "created_at", *TEXT_FIELDS, *REAL_FIELDS, *INT_FIELDS)
)


def _text(col: str, canonical: str) -> str:
    expr = f'NULLIF(trim(CAST("{col}" AS VARCHAR)), \'\')'
    if canonical in ID_TEXT_FIELDS:
        expr = f"regexp_replace({expr}, '^([0-9]+)\\.0+$', '\\1')"
    return expr


def timestamp_sql(expr: str) -> str:
    """Parse any of the timestamp spellings seen in ITA files; NULL when none fits."""
    fmts = ", ".join(f"'{f}'" for f in TIMESTAMP_FORMATS)
    return f"try_strptime({expr}, [{fmts}])"


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """{lowercased name: actual name} for a table's columns."""
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {r[1].lower(): r[1] for r in rows}


def resolve_column(
    con: duckdb.DuckDBPyConnection, table: str, candidates: tuple[str, ...]
) -> str | None:
    """The first candidate present in `table` (actual casing), or None."""
    cols = table_columns(con, table)
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _num(col: str, typ: str) -> str:
    # Values arrive as text; "3.0" must still become 3, so go through DOUBLE.
    inner = f"TRY_CAST(trim(CAST(\"{col}\" AS VARCHAR)) AS DOUBLE)"
    return inner if typ == "DOUBLE" else f"CAST(round({inner}) AS {typ})"


def canonical_select_sql(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """SELECT emitting the canonical schema for one raw per-year table."""
    pieces: list[str] = []
    found: dict[str, str | None] = {}
    for canonical, candidates in TEXT_FIELDS.items():
        col = resolve_column(con, table, candidates)
        found[canonical] = col
        pieces.append(
            f"{_text(col, canonical)} AS {canonical}" if col
            else f"CAST(NULL AS VARCHAR) AS {canonical}"
        )
    for canonical, candidates in REAL_FIELDS.items():
        col = resolve_column(con, table, candidates)
        pieces.append(f"{_num(col, 'DOUBLE')} AS {canonical}" if col else f"CAST(NULL AS DOUBLE) AS {canonical}")
    for canonical, candidates in INT_FIELDS.items():
        col = resolve_column(con, table, candidates)
        found[canonical] = col
        pieces.append(f"{_num(col, 'INTEGER')} AS {canonical}" if col else f"CAST(NULL AS INTEGER) AS {canonical}")

    file_year = "file_year" if "file_year" in table_columns(con, table) else "CAST(NULL AS INTEGER)"
    yff = _num(found["year_filing_for"], "INTEGER") if found["year_filing_for"] else "CAST(NULL AS INTEGER)"
    size_raw = _text(found["size_code_raw"], "size_code_raw") if found["size_code_raw"] else "CAST(NULL AS VARCHAR)"
    created = (
        timestamp_sql(_text(found["created_timestamp"], "created_timestamp"))
        if found["created_timestamp"] else "CAST(NULL AS TIMESTAMP)"
    )
    pieces.insert(0, f"CAST({file_year} AS INTEGER) AS file_year")
    pieces.insert(
        1,
        f"CASE WHEN {yff} BETWEEN 2000 AND 2100 THEN {yff} ELSE CAST({file_year} AS INTEGER) END AS data_year",
    )
    pieces.insert(2, f"CASE WHEN {size_raw} IN ('21', '22') THEN '2' ELSE {size_raw} END AS size_code")
    pieces.insert(3, f"{created} AS created_at")
    return "SELECT " + ",\n  ".join(pieces) + f"\nFROM {table}"


def summary_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name SIMILAR TO 'ita_summary_[0-9]{4}' ORDER BY table_name"
        ).fetchall()
    ]


def build_canonical_view(con: duckdb.DuckDBPyConnection, view_name: str = "ita_canonical") -> int:
    """Union every ita_summary_<year> table under the canonical schema. Returns table count."""
    tables = summary_tables(con)
    if not tables:
        raise RuntimeError("No ita_summary_<year> tables found. Run `load` first.")
    union_sql = "\nUNION ALL BY NAME\n".join(f"({canonical_select_sql(con, t)})" for t in tables)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {union_sql}")
    return len(tables)
