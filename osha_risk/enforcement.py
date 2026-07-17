"""DOL OSHA enforcement data: inspections, violations, penalties.

The DOL publishes OSHA enforcement data as multiple linked tables at
https://enforcedata.dol.gov/views/data_catalogs.php (migrating to
data.dol.gov). The tables most useful for risk modeling are:

  - osha_inspection: one row per inspection. Establishment name and address,
    state, NAICS, open/close dates, inspection type, scope.
  - osha_violation: zero-or-more rows per inspection. Standard cited,
    gravity, penalty amounts, violation severity (Serious / Willful /
    Repeat / Other).
  - osha_accident: zero-or-more rows per inspection. Fatality and
    hospitalization details.

All tables share `activity_nr` as the inspection key.

This module:
  1. Loads CSVs from a local directory (download via DOL portal manually
     or via the URLs in config; URLs change so we keep this loose).
  2. Computes a soft establishment key matching the ITA pipeline:
     md5(name_norm | state | zip5). This is the same join key used in
     features.build_establishment_key_view.
  3. Aggregates inspection and violation activity to establishment-year
     buckets ready to merge into the ITA panel.

Design choice: penalties are NOT inflation-adjusted in v1. OSHA's penalty
ceilings have changed multiple times (2016, 2024). For ranking purposes
this matters less than for absolute comparison, but document it.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# Known severity values in the DOL violation data.
# Maps to a numeric severity score we use for max_gravity-like features.
SEVERITY_SCORE: dict[str, int] = {
    "Willful": 4,
    "Repeat": 3,
    "Serious": 2,
    "Other": 1,
    "Unclassified": 1,
}


# ---------- Loading raw CSVs ----------

# Column name candidates for the raw DOL files. The DOL has shifted some
# capitalization and underscore conventions over the years.
INSPECTION_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "activity_nr": ("activity_nr", "ACTIVITY_NR"),
    "estab_name": ("estab_name", "ESTAB_NAME", "company_name"),
    "site_address": ("site_address", "SITE_ADDRESS", "address"),
    "site_city": ("site_city", "SITE_CITY", "city"),
    "site_state": ("site_state", "SITE_STATE", "state"),
    "site_zip": ("site_zip", "SITE_ZIP", "zip_code"),
    "naics_code": ("naics_code", "NAICS_CODE", "naics"),
    "insp_type": ("insp_type", "INSP_TYPE", "inspection_type"),
    "insp_scope": ("insp_scope", "INSP_SCOPE", "inspection_scope"),
    "open_date": ("open_date", "OPEN_DATE", "opening_date"),
    "close_case_date": ("close_case_date", "CLOSE_CASE_DATE", "close_date"),
}

VIOLATION_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "activity_nr": ("activity_nr", "ACTIVITY_NR"),
    "citation_id": ("citation_id", "CITATION_ID"),
    "standard": ("standard", "STANDARD", "standards_cited"),
    "viol_type": ("viol_type", "VIOL_TYPE", "violation_type"),
    "gravity": ("gravity", "GRAVITY"),
    "initial_penalty": ("initial_penalty", "INITIAL_PENALTY"),
    "current_penalty": ("current_penalty", "CURRENT_PENALTY", "final_penalty"),
}


def _load_csv_with_column_mapping(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    table_name: str,
    column_map: dict[str, tuple[str, ...]],
) -> int:
    """Load a CSV into DuckDB renaming columns to canonical names.

    Columns not in `column_map` are kept as-is so we don't lose information.
    Missing canonical columns are added as NULL.
    """
    # First: load raw to inspect headers
    raw = f"_raw_{table_name}"
    con.execute(f"DROP TABLE IF EXISTS {raw}")
    con.execute(
        f"""
        CREATE TABLE {raw} AS
        SELECT * FROM read_csv_auto(
            '{csv_path.as_posix()}',
            header=true,
            sample_size=50000,
            ignore_errors=true,
            null_padding=true
        )
        """
    )

    # Available columns (lowercased)
    available = {
        r[1].lower(): r[1]
        for r in con.execute(f"PRAGMA table_info('{raw}')").fetchall()
    }

    # Build the SELECT with renames.
    # We CAST everything to VARCHAR by default so downstream code can rely on
    # consistent string typing. Numeric coercion happens at aggregation time
    # via TRY_CAST. (Real-world DOL CSVs come with mixed types; ZIPs especially
    # get inferred as BIGINT when all-numeric.)
    select_parts: list[str] = []
    for canonical, candidates in column_map.items():
        found = next(
            (available[c.lower()] for c in candidates if c.lower() in available),
            None,
        )
        if found:
            select_parts.append(f'CAST("{found}" AS VARCHAR) AS {canonical}')
        else:
            select_parts.append(f"CAST(NULL AS VARCHAR) AS {canonical}")

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"CREATE TABLE {table_name} AS SELECT {', '.join(select_parts)} FROM {raw}"
    )
    con.execute(f"DROP TABLE {raw}")

    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def load_inspection_csv(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    table_name: str = "osha_inspection",
) -> int:
    """Load the DOL osha_inspection CSV into DuckDB with canonical columns."""
    return _load_csv_with_column_mapping(
        con, csv_path, table_name, INSPECTION_COLUMN_MAP
    )


def load_violation_csv(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    table_name: str = "osha_violation",
) -> int:
    """Load the DOL osha_violation CSV into DuckDB with canonical columns."""
    return _load_csv_with_column_mapping(
        con, csv_path, table_name, VIOLATION_COLUMN_MAP
    )


# ---------- Key generation matching the ITA pipeline ----------

def add_soft_key_to_inspections(
    con: duckdb.DuckDBPyConnection,
    inspection_table: str = "osha_inspection",
) -> None:
    """Add a `estab_key` column to inspections using the same hash as ITA.

    Must use IDENTICAL normalization to features.build_establishment_key_view
    or the join will silently produce zero matches.
    """
    # Check whether column already exists
    cols = {
        r[1].lower()
        for r in con.execute(f"PRAGMA table_info('{inspection_table}')").fetchall()
    }
    if "estab_key" in cols:
        return

    con.execute(
        f"""
        ALTER TABLE {inspection_table} ADD COLUMN estab_key VARCHAR
        """
    )
    con.execute(
        f"""
        UPDATE {inspection_table}
        SET estab_key = md5(concat_ws(
            '|',
            regexp_replace(lower(coalesce(CAST(estab_name AS VARCHAR), '')), '[^a-z0-9 ]', '', 'g'),
            upper(coalesce(CAST(site_state AS VARCHAR), '')),
            substr(regexp_replace(coalesce(CAST(site_zip AS VARCHAR), ''), '[^0-9]', '', 'g'), 1, 5)
        ))
        """
    )

    # Also add an inspection year column from open_date
    if "inspection_year" not in cols:
        con.execute(
            f"ALTER TABLE {inspection_table} ADD COLUMN inspection_year INTEGER"
        )
        con.execute(
            f"""
            UPDATE {inspection_table}
            SET inspection_year = CAST(
                substr(CAST(open_date AS VARCHAR), 1, 4) AS INTEGER
            )
            """
        )


# ---------- Aggregating to establishment-year ----------

def build_enforcement_aggregates(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str = "enforcement_estab_year",
    inspection_table: str = "osha_inspection",
    violation_table: str = "osha_violation",
) -> int:
    """Build establishment-year aggregates from inspections + violations.

    Columns produced (per estab_key, year):
      n_inspections                   total inspections
      n_inspections_unprog            unprogrammed (complaint/accident-driven)
      n_citations                     total citations
      n_serious_citations             Serious + Willful + Repeat
      n_willful_citations             Willful only
      total_initial_penalty           sum of initial penalties ($)
      total_current_penalty           sum of final/current penalties ($)
      max_severity_score              max severity numeric (1-4)
      had_fatality_inspection         1 if any inspection flagged as fatality

    These get joined into the ITA panel via estab_key + year.
    """
    # Helper: severity scoring via CASE
    severity_case = " ".join(
        f"WHEN viol_type = '{k}' THEN {v}" for k, v in SEVERITY_SCORE.items()
    )

    # Unprogrammed inspections: type codes like 'Unprog', 'Complaint',
    # 'Accident', 'Referral'. We pattern-match loosely since DOL labels vary.
    unprog_pattern = "(?i)(unprog|complaint|accident|referral|fatcat|fatal)"

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        WITH viol_agg AS (
            SELECT
                activity_nr,
                COUNT(*) AS n_citations,
                SUM(CASE WHEN viol_type IN ('Serious','Willful','Repeat') THEN 1 ELSE 0 END)
                    AS n_serious_citations,
                SUM(CASE WHEN viol_type = 'Willful' THEN 1 ELSE 0 END)
                    AS n_willful_citations,
                SUM(COALESCE(TRY_CAST(initial_penalty AS DOUBLE), 0))
                    AS total_initial_penalty,
                SUM(COALESCE(TRY_CAST(current_penalty AS DOUBLE), 0))
                    AS total_current_penalty,
                MAX(CASE {severity_case} ELSE 0 END) AS max_severity_score
            FROM {violation_table}
            GROUP BY activity_nr
        ),
        inspection_enriched AS (
            SELECT
                i.estab_key,
                i.inspection_year AS year,
                i.activity_nr,
                regexp_matches(COALESCE(i.insp_type, ''), '{unprog_pattern}')
                    AS is_unprogrammed,
                regexp_matches(COALESCE(i.insp_type, ''), '(?i)(fatal|fatcat)')
                    AS is_fatality_insp,
                v.n_citations,
                v.n_serious_citations,
                v.n_willful_citations,
                v.total_initial_penalty,
                v.total_current_penalty,
                v.max_severity_score
            FROM {inspection_table} i
            LEFT JOIN viol_agg v USING (activity_nr)
            WHERE i.estab_key IS NOT NULL
              AND i.inspection_year IS NOT NULL
        )
        SELECT
            estab_key,
            year,
            COUNT(*) AS n_inspections,
            SUM(CASE WHEN is_unprogrammed THEN 1 ELSE 0 END) AS n_inspections_unprog,
            SUM(COALESCE(n_citations, 0)) AS n_citations,
            SUM(COALESCE(n_serious_citations, 0)) AS n_serious_citations,
            SUM(COALESCE(n_willful_citations, 0)) AS n_willful_citations,
            SUM(COALESCE(total_initial_penalty, 0)) AS total_initial_penalty,
            SUM(COALESCE(total_current_penalty, 0)) AS total_current_penalty,
            MAX(COALESCE(max_severity_score, 0)) AS max_severity_score,
            MAX(CASE WHEN is_fatality_insp THEN 1 ELSE 0 END) AS had_fatality_inspection
        FROM inspection_enriched
        GROUP BY estab_key, year
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


# ---------- Panel attachment ----------

def attach_enforcement_to_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_table: str = "ita_panel",
    enforcement_table: str = "enforcement_estab_year",
    output_table: str = "ita_panel_enforced",
) -> int:
    """Join lagged enforcement features onto the ITA panel.

    We attach 1-year-lagged enforcement signal: in panel year t, we get
    enforcement activity from year t-1. This is what the model gets to see
    when "predicting" year t (since current-year inspection data won't be
    available at the time a broker is making a decision).

    Returns row count of the output table.
    """
    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        SELECT
            p.*,
            -- Lag-1 enforcement features (defaulting NULLs to 0 for COUNT/SUM
            -- features since "no record" means "no inspections")
            COALESCE(e1.n_inspections, 0)         AS enf_n_inspections_lag1,
            COALESCE(e1.n_inspections_unprog, 0)  AS enf_n_inspections_unprog_lag1,
            COALESCE(e1.n_citations, 0)           AS enf_n_citations_lag1,
            COALESCE(e1.n_serious_citations, 0)   AS enf_n_serious_citations_lag1,
            COALESCE(e1.n_willful_citations, 0)   AS enf_n_willful_citations_lag1,
            COALESCE(e1.total_current_penalty, 0) AS enf_total_penalty_lag1,
            COALESCE(e1.max_severity_score, 0)    AS enf_max_severity_lag1,
            COALESCE(e1.had_fatality_inspection, 0) AS enf_had_fatality_lag1,
            -- Lag-2 to lag-1 cumulative penalty (capture recent enforcement burst)
            COALESCE(e1.total_current_penalty, 0)
              + COALESCE(e2.total_current_penalty, 0) AS enf_penalty_lag1to2,
            -- Penalty per employee, lag-1
            CASE
                WHEN p.employees_lag1 > 0
                THEN COALESCE(e1.total_current_penalty, 0) / p.employees_lag1
            END AS enf_penalty_per_employee_lag1,
            -- Ever-inspected flag (across observed history)
            CASE WHEN e1.n_inspections IS NOT NULL THEN 1 ELSE 0 END
                AS enf_inspected_lag1
        FROM {panel_table} p
        LEFT JOIN {enforcement_table} e1
            ON e1.estab_key = p.estab_key AND e1.year = p.year - 1
        LEFT JOIN {enforcement_table} e2
            ON e2.estab_key = p.estab_key AND e2.year = p.year - 2
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
