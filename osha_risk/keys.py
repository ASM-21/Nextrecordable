"""Establishment keys: one implementation shared by every dataset.

`estab_key` = md5(normalized name | STATE | zip5). It identifies an
establishment across ITA years and is the exact-match path from DOL
inspections to the ITA panel. Normalization runs in Python (normalize.py)
over DISTINCT raw values and the results are joined back in SQL, so SQL never
carries its own copy of the rules.

The key is deliberately coarse: two sites with the same name in the same ZIP
collapse into one key. The panel keeps one row per key and year and reports
how often that happened (see features.build_panel).
"""

from __future__ import annotations

import hashlib

import duckdb
import pandas as pd

from .normalize import (
    name_first_token,
    name_numbers,
    normalize_address,
    normalize_name,
    state_code,
    zip5,
)


def key_sql(name_expr: str, state_expr: str, zip_expr: str) -> str:
    """SQL expression hashing already-normalized name, state and zip5 columns."""
    return f"md5(concat_ws('|', {name_expr}, {state_expr}, {zip_expr}))"


def python_key(name, state, zip_value) -> str:
    """The same key computed in Python (single lookups, tests)."""
    raw = "|".join([normalize_name(name), state_code(state), zip5(zip_value)])
    return hashlib.md5(raw.encode()).hexdigest()


def state_sql(col_expr: str) -> str:
    """SQL twin of normalize.state_code (trim, first two chars, upper).

    Kept trivially small on purpose; tests/test_keys.py checks parity.
    """
    return f"upper(substr(trim(coalesce(CAST({col_expr} AS VARCHAR), '')), 1, 2))"


def _distinct(con: duckdb.DuckDBPyConnection, table: str, col: str) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            f'SELECT DISTINCT CAST("{col}" AS VARCHAR) FROM {table} WHERE "{col}" IS NOT NULL'
        ).fetchall()
    ]


def _temp_map(con: duckdb.DuckDBPyConnection, name: str, columns: dict[str, list[str]]) -> None:
    """Create TEMP table `name` with VARCHAR columns from Python lists (empty-safe)."""
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(
        f"CREATE TEMP TABLE {name} ({', '.join(f'{c} VARCHAR' for c in columns)})"
    )
    n = len(next(iter(columns.values())))
    if n:
        frame = pd.DataFrame({c: pd.Series(v, dtype="object") for c, v in columns.items()})
        con.register(f"{name}_df", frame)
        con.execute(f"INSERT INTO {name} SELECT * FROM {name}_df")
        con.unregister(f"{name}_df")


def build_keyed_table(
    con: duckdb.DuckDBPyConnection,
    *,
    source: str,
    output: str,
    name_col: str,
    state_col: str,
    zip_col: str,
    address_col: str | None = None,
) -> int:
    """Materialize `output` = `source` plus normalized columns and `estab_key`.

    Added columns: name_norm, name_first, name_nums, state_norm, zip5,
    address_norm ('' when no address column) and estab_key. Returns row count.
    """
    raw_names = _distinct(con, source, name_col)
    norms = [normalize_name(n) for n in raw_names]
    _temp_map(
        con,
        "_name_map",
        {
            "raw": raw_names,
            "norm": norms,
            "first_tok": [name_first_token(n) for n in norms],
            "nums": [name_numbers(n) for n in norms],
        },
    )
    raw_zips = _distinct(con, source, zip_col)
    _temp_map(con, "_zip_map", {"raw": raw_zips, "z5": [zip5(z) for z in raw_zips]})

    addr_select = "CAST('' AS VARCHAR) AS address_norm"
    addr_join = ""
    if address_col:
        raw_addrs = _distinct(con, source, address_col)
        _temp_map(
            con,
            "_addr_map",
            {"raw": raw_addrs, "norm": [normalize_address(a) for a in raw_addrs]},
        )
        addr_select = "coalesce(am.norm, '') AS address_norm"
        addr_join = f'LEFT JOIN _addr_map am ON am.raw = CAST(s."{address_col}" AS VARCHAR)'

    st = state_sql(f's."{state_col}"')
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        SELECT
            s.*,
            coalesce(nm.norm, '')      AS name_norm,
            coalesce(nm.first_tok, '') AS name_first,
            coalesce(nm.nums, '')      AS name_nums,
            {st}                       AS state_norm,
            coalesce(zm.z5, '')        AS zip5,
            {addr_select},
            {key_sql("coalesce(nm.norm, '')", st, "coalesce(zm.z5, '')")} AS estab_key
        FROM {source} s
        LEFT JOIN _name_map nm ON nm.raw = CAST(s."{name_col}" AS VARCHAR)
        LEFT JOIN _zip_map zm ON zm.raw = CAST(s."{zip_col}" AS VARCHAR)
        {addr_join}
        """
    )
    for t in ("_name_map", "_zip_map", "_addr_map"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return con.execute(f"SELECT COUNT(*) FROM {output}").fetchone()[0]
