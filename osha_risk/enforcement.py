"""DOL OSHA enforcement data: inspections and citations, linked to ITA establishments.

Source: the DOL enforcement catalog (enforcedata.dol.gov, moving to
data.dol.gov), tables osha_inspection and osha_violation, joined on
activity_nr. Real column names and codes, checked against a mirror of the
DOL files and IRE's OSHA documentation:

  osha_violation.viol_type   S serious, W willful, R repeat, O other,
                             U unclassified (F used here for failure-to-abate)
  osha_violation.delete_flag 'X' = citation deleted or superseded (excluded)
  osha_inspection.insp_type  unprogrammed: A fatality/catastrophe, B complaint,
                             C referral, D monitoring, E variance, F follow-up,
                             G unprogrammed related, J unprogrammed other;
                             programmed: H planned, I related, K other, L other
  site_zip                   often stored as an integer, so leading zeros are
                             lost; normalize.zip5 restores them

The first version matched the words "Serious"/"Willful" and phrases like
"complaint"; on real files every severity and inspection-type feature came
out zero. Codes are now read by their first letter, so both the letter codes
and the spelled-out labels used by some mirrors work for viol_type.

Penalties use initial_penalty, which is fixed when the citation is issued.
current_penalty changes for years afterwards as contests and settlements
resolve, so using it would leak future information into training rows.

Linking: an inspection is assigned to an ITA establishment by exact
estab_key first, then by the fixed-weight matcher (match.py) for the rest.
DOL and ITA spell the same employer differently often enough that exact keys
alone miss most links.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .keys import build_keyed_table
from .load import read_csv_varchar
from .match import link

SEVERITY_BY_CODE: dict[str, int] = {"W": 4, "R": 3, "F": 3, "S": 2, "O": 1, "U": 1}
SERIOUS_CLASS_CODES = ("S", "W", "R", "F")
WILLFUL_REPEAT_CODES = ("W", "R")
UNPROGRAMMED_INSP_CODES = ("A", "B", "C", "D", "E", "F", "G", "J")
FATCAT_INSP_CODES = ("A",)
COMPLAINT_REFERRAL_CODES = ("B", "C")

INSPECTION_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "activity_nr": ("activity_nr",),
    "estab_name": ("estab_name", "company_name"),
    "site_address": ("site_address", "address"),
    "site_city": ("site_city", "city"),
    "site_state": ("site_state", "state"),
    "site_zip": ("site_zip", "zip_code"),
    "naics_code": ("naics_code", "naics"),
    "insp_type": ("insp_type", "inspection_type"),
    "insp_scope": ("insp_scope", "inspection_scope"),
    "open_date": ("open_date", "opening_date"),
    "close_case_date": ("close_case_date", "close_date"),
}

VIOLATION_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "activity_nr": ("activity_nr",),
    "citation_id": ("citation_id",),
    "delete_flag": ("delete_flag",),
    "standard": ("standard", "standards_cited"),
    "viol_type": ("viol_type", "violation_type"),
    "gravity": ("gravity",),
    "issuance_date": ("issuance_date",),
    "initial_penalty": ("initial_penalty",),
    "current_penalty": ("current_penalty", "final_penalty"),
}


def _load_mapped(con: duckdb.DuckDBPyConnection, csv_path: Path, table: str,
                 column_map: dict[str, tuple[str, ...]]) -> int:
    raw = f"_raw_{table}"
    read_csv_varchar(con, Path(csv_path), raw)
    available = {r[1].lower(): r[1] for r in con.execute(f"PRAGMA table_info('{raw}')").fetchall()}
    parts = []
    for canonical, candidates in column_map.items():
        col = next((available[c] for c in candidates if c in available), None)
        parts.append(
            f'NULLIF(trim("{col}"), \'\') AS {canonical}' if col
            else f"CAST(NULL AS VARCHAR) AS {canonical}"
        )
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} AS SELECT {', '.join(parts)} FROM {raw}")
    con.execute(f"DROP TABLE {raw}")
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def load_inspection_csv(con: duckdb.DuckDBPyConnection, csv_path: Path,
                        table_name: str = "osha_inspection") -> int:
    """Load DOL osha_inspection (all text, canonical column names)."""
    return _load_mapped(con, csv_path, table_name, INSPECTION_COLUMN_MAP)


def load_violation_csv(con: duckdb.DuckDBPyConnection, csv_path: Path,
                       table_name: str = "osha_violation") -> int:
    """Load DOL osha_violation (all text, canonical column names)."""
    return _load_mapped(con, csv_path, table_name, VIOLATION_COLUMN_MAP)


def _year_sql(col: str) -> str:
    """Year from 'YYYY-MM-DD', 'YYYYMMDD' or 'MM/DD/YYYY'."""
    return (
        f"CASE WHEN regexp_matches({col}, '^[0-9]{{4}}') THEN TRY_CAST(substr({col}, 1, 4) AS INTEGER) "
        f"WHEN regexp_matches({col}, '[0-9]{{4}}$') THEN TRY_CAST(right({col}, 4) AS INTEGER) END"
    )


def link_inspections(
    con: duckdb.DuckDBPyConnection,
    *,
    inspection_table: str = "osha_inspection",
    directory: str = "estab_directory",
    output: str = "inspection_linked",
) -> dict[str, int]:
    """Assign each inspection an ITA estab_key (exact key, then fuzzy). Returns counts."""
    build_keyed_table(
        con, source=inspection_table, output="_insp_keyed", name_col="estab_name",
        state_col="site_state", zip_col="site_zip", address_col="site_address",
    )
    # Fuzzy-link the distinct establishments that have no exact key in the directory.
    con.execute("DROP TABLE IF EXISTS _insp_unkeyed")
    con.execute(
        f"""
        CREATE TEMP TABLE _insp_unkeyed AS
        SELECT ROW_NUMBER() OVER () AS rec_id, *
        FROM (
            SELECT DISTINCT name_norm, name_first, name_nums, state_norm AS state, zip5,
                   address_norm, '' AS original_name
            FROM _insp_keyed k
            WHERE name_norm <> ''
              AND NOT EXISTS (SELECT 1 FROM {directory} d WHERE d.estab_key = k.estab_key)
        )
        """
    )
    link(con, left="_insp_unkeyed", directory=directory, output="_insp_fuzzy", candidates_output=None)
    con.execute(f"DROP TABLE IF EXISTS {output}")
    con.execute(
        f"""
        CREATE TABLE {output} AS
        SELECT
            k.activity_nr,
            k.insp_type,
            k.open_date,
            {_year_sql('k.open_date')} AS inspection_year,
            CASE
                WHEN d.estab_key IS NOT NULL THEN k.estab_key
                WHEN f.status = 'matched' THEN f.estab_key
            END AS estab_key,
            CASE
                WHEN d.estab_key IS NOT NULL THEN 'exact_key'
                WHEN f.status = 'matched' THEN 'fuzzy'
                WHEN f.status = 'ambiguous' THEN 'ambiguous'
                ELSE 'unlinked'
            END AS link_method
        FROM _insp_keyed k
        LEFT JOIN {directory} d ON d.estab_key = k.estab_key
        LEFT JOIN _insp_unkeyed u
          ON u.name_norm = k.name_norm AND u.state = k.state_norm AND u.zip5 = k.zip5
         AND u.address_norm = k.address_norm AND u.name_nums = k.name_nums
        LEFT JOIN _insp_fuzzy f ON f.rec_id = u.rec_id
        """
    )
    for t in ("_insp_keyed", "_insp_unkeyed", "_insp_fuzzy"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    rows = con.execute(f"SELECT link_method, COUNT(*) FROM {output} GROUP BY 1").fetchall()
    return {m: int(n) for m, n in rows}


def build_enforcement_aggregates(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str = "enforcement_estab_year",
    linked_table: str = "inspection_linked",
    violation_table: str = "osha_violation",
) -> int:
    """Per (estab_key, year) inspection and citation counts. Returns row count."""
    sev = " ".join(f"WHEN '{c}' THEN {s}" for c, s in SEVERITY_BY_CODE.items())

    def in_list(codes: tuple[str, ...]) -> str:
        return "(" + ", ".join(f"'{c}'" for c in codes) + ")"

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        WITH viol AS (
            SELECT activity_nr,
                   upper(left(trim(coalesce(viol_type, '')), 1)) AS code,
                   coalesce(TRY_CAST(initial_penalty AS DOUBLE), 0) AS initial_penalty
            FROM {violation_table}
            WHERE upper(trim(coalesce(delete_flag, ''))) <> 'X'
        ),
        viol_agg AS (
            SELECT activity_nr,
                   COUNT(*) AS n_citations,
                   COUNT(*) FILTER (WHERE code IN {in_list(SERIOUS_CLASS_CODES)}) AS n_serious,
                   COUNT(*) FILTER (WHERE code IN {in_list(WILLFUL_REPEAT_CODES)}) AS n_willful_repeat,
                   sum(initial_penalty) AS initial_penalty,
                   max(CASE code {sev} ELSE 0 END) AS max_severity
            FROM viol GROUP BY activity_nr
        ),
        insp AS (
            SELECT i.estab_key, i.inspection_year AS year,
                   upper(left(trim(coalesce(i.insp_type, '')), 1)) AS itype,
                   v.*
            FROM {linked_table} i
            LEFT JOIN viol_agg v USING (activity_nr)
            WHERE i.estab_key IS NOT NULL AND i.inspection_year IS NOT NULL
        )
        SELECT
            estab_key, year,
            COUNT(*) AS n_inspections,
            COUNT(*) FILTER (WHERE itype IN {in_list(UNPROGRAMMED_INSP_CODES)}) AS n_unprogrammed,
            COUNT(*) FILTER (WHERE itype IN {in_list(COMPLAINT_REFERRAL_CODES)}) AS n_complaint_referral,
            COUNT(*) FILTER (WHERE itype IN {in_list(FATCAT_INSP_CODES)}) AS n_fatcat,
            sum(coalesce(n_citations, 0)) AS n_citations,
            sum(coalesce(n_serious, 0)) AS n_serious_citations,
            sum(coalesce(n_willful_repeat, 0)) AS n_willful_repeat_citations,
            sum(coalesce(initial_penalty, 0)) AS initial_penalty,
            max(coalesce(max_severity, 0)) AS max_severity
        FROM insp
        GROUP BY estab_key, year
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def attach_enforcement_to_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_table: str = "ita_panel",
    enforcement_table: str = "enforcement_estab_year",
    output_table: str = "ita_panel_enforced",
) -> int:
    """Join two-year enforcement windows (years t-1 and t) onto each panel row.

    A panel row for year t predicts year t+1. Its forecast is made after the
    year-t ITA data is published (spring of t+1), when inspections opened in
    year t are already on record, so year t is fair to use. Zero means "no
    inspection on record", not "inspected and clean".
    """
    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        WITH win AS (
            SELECT p.estab_key, p.year,
                   sum(e.n_inspections)              AS n_insp,
                   sum(e.n_unprogrammed)             AS n_unprog,
                   sum(e.n_complaint_referral)       AS n_cr,
                   sum(e.n_fatcat)                   AS n_fatcat,
                   sum(e.n_citations)                AS n_cit,
                   sum(e.n_serious_citations)        AS n_serious,
                   sum(e.n_willful_repeat_citations) AS n_wr,
                   sum(e.initial_penalty)            AS penalty,
                   max(e.max_severity)               AS max_sev
            FROM {panel_table} p
            JOIN {enforcement_table} e
              ON e.estab_key = p.estab_key AND e.year BETWEEN p.year - 1 AND p.year
            GROUP BY p.estab_key, p.year
        ),
        last_seen AS (
            SELECT p.estab_key, p.year, max(e.year) AS last_year
            FROM {panel_table} p
            JOIN {enforcement_table} e ON e.estab_key = p.estab_key AND e.year <= p.year
            GROUP BY p.estab_key, p.year
        )
        SELECT
            p.*,
            coalesce(w.n_insp, 0)    AS enf_inspections_2y,
            coalesce(w.n_unprog, 0)  AS enf_unprogrammed_2y,
            coalesce(w.n_cr, 0)      AS enf_complaint_referral_2y,
            coalesce(w.n_fatcat, 0)  AS enf_fatcat_2y,
            coalesce(w.n_cit, 0)     AS enf_citations_2y,
            coalesce(w.n_serious, 0) AS enf_serious_citations_2y,
            coalesce(w.n_wr, 0)      AS enf_willful_repeat_2y,
            coalesce(w.penalty, 0)   AS enf_initial_penalty_2y,
            coalesce(w.max_sev, 0)   AS enf_max_severity_2y,
            CASE WHEN p.employees > 0 THEN coalesce(w.penalty, 0) * 100 / p.employees END
                                     AS enf_penalty_per_100_employees_2y,
            least(coalesce(p.year - ls.last_year, 10), 10) AS enf_years_since_inspection
        FROM {panel_table} p
        LEFT JOIN win w ON w.estab_key = p.estab_key AND w.year = p.year
        LEFT JOIN last_seen ls ON ls.estab_key = p.estab_key AND ls.year = p.year
        """
    )
    return con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
