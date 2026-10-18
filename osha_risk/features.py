"""Build the establishment-year modeling panel from the canonical ITA view.

One row per (estab_key, year) with:
  - that year's counts, hours, rates (TCR, DART) and NAICS/size;
  - lags from year-1 and year-2, only when those exact years were filed;
  - NAICS-4 benchmarks for the year (median and pooled rates);
  - next-year outcomes (counts, hours, rates), only when year+1 was filed.

Rates use OSHA's formula: cases * 200,000 / hours worked (per 100 FTE).

Integrity rules, each pinned by a test because each one silently corrupted
features or labels before it existed:

  * One row per key-year. ITA can hold several rows for one key and year
    (an amended submission, or two sites sharing name+ZIP). Without dedupe
    the window functions step WITHIN a year, so the "next year" target can be
    the same year's other filing. Rows sharing an establishment_id keep the
    latest submission; rows with different establishment_ids are distinct
    sites and are summed into the key. `n_raw_rows` and `n_sites` record what
    happened.
  * Adjacent years only. An establishment that skips a filing year must not
    get a two-year-old value as "lag1" or a two-year-ahead count as "next".
"""

from __future__ import annotations

import duckdb

from .keys import build_keyed_table

RATE_DENOMINATOR = 200_000


def build_keyed_view(con: duckdb.DuckDBPyConnection, *, output: str = "ita_keyed") -> int:
    """Materialize ita_canonical plus normalized fields and estab_key."""
    return build_keyed_table(
        con,
        source="ita_canonical",
        output=output,
        name_col="establishment_name",
        state_col="state",
        zip_col="zip_code",
        address_col="street_address",
    )


# Backwards-compatible name used by earlier notebooks and tests.
build_establishment_key_view = build_keyed_view


def build_panel(
    con: duckdb.DuckDBPyConnection,
    table_name: str = "ita_panel",
    *,
    source: str = "ita_keyed",
) -> int:
    """Materialize the panel as a table. Returns its row count."""
    r = RATE_DENOMINATOR
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        WITH raw AS (
            SELECT
                estab_key,
                data_year,
                establishment_id,
                created_timestamp,
                establishment_name, street_address, city, state_norm AS state, zip5,
                name_norm, name_first, name_nums, address_norm,
                naics_code, size_code,
                annual_average_employees AS employees,
                total_hours_worked AS hours_worked,
                coalesce(total_deaths, 0)      AS n_deaths,
                coalesce(total_dafw_cases, 0)  AS n_dafw,
                coalesce(total_djtr_cases, 0)  AS n_djtr,
                coalesce(total_other_cases, 0) AS n_other
            FROM {source}
            WHERE data_year IS NOT NULL AND name_norm <> ''
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY estab_key, data_year, coalesce(establishment_id, '')
                    ORDER BY TRY_CAST(created_timestamp AS TIMESTAMP) DESC NULLS LAST,
                             created_timestamp DESC NULLS LAST,
                             hours_worked DESC NULLS LAST,
                             (n_deaths + n_dafw + n_djtr + n_other) DESC
                ) AS rn,
                COUNT(*) OVER (PARTITION BY estab_key, data_year) AS n_raw_rows,
                bool_or(establishment_id IS NOT NULL)
                    OVER (PARTITION BY estab_key, data_year) AS key_year_has_id
            FROM raw
        ),
        base AS (
            SELECT
                estab_key,
                data_year AS year,
                arg_max(establishment_name, coalesce(hours_worked, 0)) AS establishment_name,
                arg_max(street_address, coalesce(hours_worked, 0))     AS street_address,
                arg_max(city, coalesce(hours_worked, 0))               AS city,
                any_value(state)                                       AS state,
                any_value(zip5)                                        AS zip5,
                any_value(name_norm)                                   AS name_norm,
                any_value(name_first)                                  AS name_first,
                any_value(name_nums)                                   AS name_nums,
                arg_max(address_norm, coalesce(hours_worked, 0))       AS address_norm,
                arg_max(naics_code, coalesce(hours_worked, 0))         AS naics_code,
                arg_max(size_code, coalesce(hours_worked, 0))          AS size_code,
                arg_max(establishment_id, coalesce(hours_worked, 0))   AS establishment_id,
                sum(employees)    AS employees,
                sum(hours_worked) AS hours_worked,
                sum(n_deaths) AS n_deaths,
                sum(n_dafw)   AS n_dafw,
                sum(n_djtr)   AS n_djtr,
                sum(n_other)  AS n_other,
                COUNT(*)        AS n_sites,
                max(n_raw_rows) AS n_raw_rows
            FROM ranked
            -- Latest row per site. A row without an establishment_id is dropped
            -- when the same key-year has ID'd rows (it's a duplicate of one).
            WHERE rn = 1 AND (establishment_id IS NOT NULL OR NOT key_year_has_id)
            GROUP BY estab_key, data_year
        ),
        rates AS (
            SELECT
                *,
                substr(coalesce(naics_code, ''), 1, 4) AS naics_4,
                substr(coalesce(naics_code, ''), 1, 2) AS naics_2,
                n_deaths + n_dafw + n_djtr + n_other AS n_recordable,
                n_dafw + n_djtr AS n_dart,
                CASE WHEN hours_worked > 0
                     THEN (n_deaths + n_dafw + n_djtr + n_other) * {r} / hours_worked END AS tcr,
                CASE WHEN hours_worked > 0
                     THEN (n_dafw + n_djtr) * {r} / hours_worked END AS dart_rate
            FROM base
        ),
        industry AS (
            SELECT
                year,
                naics_4,
                median(tcr)       AS naics4_tcr_median,
                median(dart_rate) AS naics4_dart_median,
                COUNT(*)          AS naics4_n_estabs,
                sum(n_recordable) * {r} / sum(hours_worked) AS naics4_pooled_tcr
            FROM rates
            WHERE naics_4 <> '' AND hours_worked > 0
            GROUP BY year, naics_4
        ),
        windowed AS (
            SELECT
                x.*,
                LAG(year, 1)  OVER w AS _py1,
                LAG(year, 2)  OVER w AS _py2,
                LEAD(year, 1) OVER w AS _ny1,
                LAG(tcr, 1)          OVER w AS _tcr1,
                LAG(tcr, 2)          OVER w AS _tcr2,
                LAG(dart_rate, 1)    OVER w AS _dart1,
                LAG(n_recordable, 1) OVER w AS _nrec1,
                LAG(employees, 1)    OVER w AS _emp1,
                LAG(hours_worked, 1) OVER w AS _hrs1,
                LEAD(n_recordable, 1) OVER w AS _nrec_n,
                LEAD(n_dart, 1)       OVER w AS _ndart_n,
                LEAD(hours_worked, 1) OVER w AS _hrs_n,
                LEAD(tcr, 1)          OVER w AS _tcr_n
            FROM rates x
            WINDOW w AS (PARTITION BY estab_key ORDER BY year)
        ),
        aligned AS (
            SELECT
                * EXCLUDE (_py1, _py2, _ny1, _tcr1, _tcr2, _dart1, _nrec1, _emp1, _hrs1,
                           _nrec_n, _ndart_n, _hrs_n, _tcr_n),
                CASE WHEN _py1 = year - 1 THEN _tcr1  END AS tcr_lag1,
                CASE WHEN _py1 = year - 1 AND _py2 = year - 2 THEN _tcr2 END AS tcr_lag2,
                CASE WHEN _py1 = year - 1 THEN _dart1 END AS dart_lag1,
                CASE WHEN _py1 = year - 1 THEN _nrec1 END AS n_recordable_lag1,
                CASE WHEN _py1 = year - 1 THEN _emp1  END AS employees_lag1,
                CASE WHEN _py1 = year - 1 THEN _hrs1  END AS hours_lag1,
                CASE WHEN _ny1 = year + 1 THEN _nrec_n  END AS n_recordable_next,
                CASE WHEN _ny1 = year + 1 THEN _ndart_n END AS n_dart_next,
                CASE WHEN _ny1 = year + 1 THEN _hrs_n   END AS hours_next,
                CASE WHEN _ny1 = year + 1 THEN _tcr_n   END AS tcr_next
            FROM windowed
        )
        SELECT
            a.*,
            i.naics4_tcr_median,
            i.naics4_dart_median,
            i.naics4_n_estabs,
            i.naics4_pooled_tcr,
            CASE WHEN i.naics4_tcr_median > 0 AND a.tcr IS NOT NULL
                 THEN a.tcr / i.naics4_tcr_median END AS tcr_vs_naics4_median,
            CASE WHEN a.n_recordable_next IS NULL THEN NULL
                 WHEN a.n_recordable_next > 0 THEN 1 ELSE 0 END AS target_any_recordable_next,
            a.n_recordable_next AS target_n_recordable_next,
            a.tcr_next          AS target_tcr_next
        FROM aligned a
        LEFT JOIN industry i ON a.year = i.year AND a.naics_4 = i.naics_4
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def panel_summary(con: duckdb.DuckDBPyConnection, table_name: str = "ita_panel") -> dict:
    """Summary stats for sanity-checking the panel."""
    row = con.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT estab_key),
            COUNT(*) FILTER (WHERE tcr_lag1 IS NOT NULL),
            COUNT(*) FILTER (WHERE target_any_recordable_next IS NOT NULL),
            avg(target_any_recordable_next),
            COUNT(*) FILTER (WHERE n_raw_rows > 1),
            COUNT(*) FILTER (WHERE n_sites > 1)
        FROM {table_name}
        """
    ).fetchone()
    keys = (
        "total_rows", "distinct_establishments", "rows_with_lag", "rows_with_target",
        "target_base_rate", "key_years_with_duplicates", "key_years_with_multiple_sites",
    )
    return dict(zip(keys, row, strict=True))
