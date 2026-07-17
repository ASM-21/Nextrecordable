"""Build the modeling panel from the canonical ITA view.

The panel is establishment-year level. For each (establishment, year_t) row,
we attach:

  - Features from year_t (NAICS, size, current-year rates, etc.)
  - Lagged features from year_t-1 and year_t-2 (rates, deltas)
  - Industry benchmarks (median rate within NAICS-4 cohort, year_t)
  - Target: indicator and count of recordables in year_t+1

This is a "next-year prediction" panel, suitable for temporal holdout
(train on years <= T-1, validate on T, test on T+1).

Key methodological notes:

  * Establishments without an `establishment_id` (older years) are matched
    by normalized (name, address) hash. This is a SOFT identifier; expect
    join leakage of 5 to 15 percent. We document it rather than hide it.

  * Rates are computed per OSHA's formula:
        rate = (cases * 200_000) / hours_worked
    A worker-year is ~2,000 hours, so this normalizes to "per 100 FTE".

  * Establishments reporting zero hours (data error or no-operations year)
    are EXCLUDED from rate computation but kept in the panel with NULL
    rates (the model can learn to handle this).
"""

from __future__ import annotations

import duckdb

# OSHA's standard rate denominator (per 100 FTE-equivalent).
RATE_DENOMINATOR = 200_000


def build_establishment_key_view(
    con: duckdb.DuckDBPyConnection, view_name: str = "ita_keyed"
) -> None:
    """Add a stable cross-year key to ita_canonical.

    DESIGN NOTE: We use the SOFT key (md5 of normalized name + state + zip5) as
    the primary `estab_key`, even when `establishment_id` is available. Reason:
    OSHA only added establishment_id in recent years, so using it as the primary
    key would split the same physical establishment across two different keys
    at the 2022/2023 schema boundary, breaking lag features.

    The hard `establishment_id` is preserved as a separate column for
    quality-assurance: in years where it exists, all rows with the same hard ID
    should share the same soft_key. Mismatches surface false-positive merges
    in the soft hash.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            *,
            -- normalized name: lowercase, strip non-alphanumeric
            regexp_replace(lower(coalesce(establishment_name, '')), '[^a-z0-9 ]', '', 'g')
                AS name_norm,
            -- ZIP5
            substr(regexp_replace(coalesce(zip_code, ''), '[^0-9]', '', 'g'), 1, 5)
                AS zip5,
            -- Soft key: hash of (name_norm, state, zip5). Used as primary
            -- identifier for consistency across schema-drift boundaries.
            md5(concat_ws(
                '|',
                regexp_replace(lower(coalesce(establishment_name, '')), '[^a-z0-9 ]', '', 'g'),
                upper(coalesce(state, '')),
                substr(regexp_replace(coalesce(zip_code, ''), '[^0-9]', '', 'g'), 1, 5)
            )) AS estab_key
        FROM ita_canonical
        """
    )


def build_panel(
    con: duckdb.DuckDBPyConnection,
    table_name: str = "ita_panel",
) -> int:
    """Materialize the establishment-year panel as a TABLE for fast modeling.

    Returns the row count of the resulting panel.
    """
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        WITH base AS (
            SELECT
                estab_key,
                data_year AS year,
                establishment_name,
                state,
                zip5,
                naics_code,
                substr(coalesce(naics_code, ''), 1, 4) AS naics_4,
                substr(coalesce(naics_code, ''), 1, 2) AS naics_2,
                size_code,
                TRY_CAST(annual_average_employees AS DOUBLE) AS employees,
                TRY_CAST(total_hours_worked       AS DOUBLE) AS hours_worked,
                coalesce(total_deaths, 0)         AS n_deaths,
                coalesce(total_dafw_cases, 0)     AS n_dafw,
                coalesce(total_djtr_cases, 0)     AS n_djtr,
                coalesce(total_other_cases, 0)    AS n_other,
                coalesce(total_deaths, 0)
                  + coalesce(total_dafw_cases, 0)
                  + coalesce(total_djtr_cases, 0)
                  + coalesce(total_other_cases, 0) AS n_recordable
            FROM ita_keyed
            WHERE data_year IS NOT NULL
        ),
        rates AS (
            SELECT
                *,
                CASE
                    WHEN hours_worked > 0
                    THEN (n_recordable * {RATE_DENOMINATOR}) / hours_worked
                END AS tcr,
                CASE
                    WHEN hours_worked > 0
                    THEN ((n_dafw + n_djtr) * {RATE_DENOMINATOR}) / hours_worked
                END AS dart_rate
            FROM base
        ),
        industry_benchmarks AS (
            SELECT
                year,
                naics_4,
                median(tcr) AS naics4_tcr_median,
                median(dart_rate) AS naics4_dart_median,
                COUNT(*) AS naics4_n_estabs
            FROM rates
            WHERE naics_4 <> '' AND tcr IS NOT NULL
            GROUP BY year, naics_4
        ),
        with_lags AS (
            SELECT
                r.*,
                LAG(tcr, 1) OVER w        AS tcr_lag1,
                LAG(tcr, 2) OVER w        AS tcr_lag2,
                LAG(dart_rate, 1) OVER w  AS dart_lag1,
                LAG(n_recordable, 1) OVER w AS n_recordable_lag1,
                LAG(employees, 1) OVER w  AS employees_lag1,
                LEAD(n_recordable, 1) OVER w AS n_recordable_next,
                LEAD(tcr, 1) OVER w          AS tcr_next
            FROM rates r
            WINDOW w AS (PARTITION BY estab_key ORDER BY year)
        )
        SELECT
            wl.*,
            ib.naics4_tcr_median,
            ib.naics4_dart_median,
            ib.naics4_n_estabs,
            CASE
                WHEN ib.naics4_tcr_median > 0 AND wl.tcr IS NOT NULL
                THEN wl.tcr / ib.naics4_tcr_median
            END AS tcr_vs_naics4_median,
            -- TARGETS
            CASE WHEN wl.n_recordable_next IS NULL THEN NULL
                 WHEN wl.n_recordable_next > 0 THEN 1 ELSE 0 END
                AS target_any_recordable_next,
            wl.n_recordable_next AS target_n_recordable_next,
            wl.tcr_next          AS target_tcr_next
        FROM with_lags wl
        LEFT JOIN industry_benchmarks ib
          ON wl.year = ib.year AND wl.naics_4 = ib.naics_4
        """
    )

    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def panel_summary(con: duckdb.DuckDBPyConnection, table_name: str = "ita_panel") -> dict:
    """Return summary stats useful for sanity-checking the panel."""
    out: dict = {}
    out["total_rows"] = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    out["distinct_establishments"] = con.execute(
        f"SELECT COUNT(DISTINCT estab_key) FROM {table_name}"
    ).fetchone()[0]
    out["rows_with_lag"] = con.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE tcr_lag1 IS NOT NULL"
    ).fetchone()[0]
    out["rows_with_target"] = con.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE target_any_recordable_next IS NOT NULL"
    ).fetchone()[0]
    out["target_base_rate"] = con.execute(
        f"""SELECT avg(target_any_recordable_next)
            FROM {table_name}
            WHERE target_any_recordable_next IS NOT NULL"""
    ).fetchone()[0]
    return out
