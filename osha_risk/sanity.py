"""Data gates that must pass before a model trained on real files is trusted.

Each gate checks one way the real OSHA/DOL files have broken (or could break)
this pipeline, records the measured value, and returns pass / warn / fail.
`nextrecordable sanity` exits non-zero on any fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb

LEADING_ZERO_STATES = ("CT", "MA", "ME", "NH", "NJ", "PR", "RI", "VT")


@dataclass
class Gate:
    name: str
    status: str  # pass | warn | fail | skip
    value: float | int | None
    rule: str
    note: str = ""


def _exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0])


def _grade(value, *, fail_if, warn_if) -> str:
    if value is None:
        return "skip"
    if fail_if(value):
        return "fail"
    if warn_if(value):
        return "warn"
    return "pass"


def run_gates(
    con: duckdb.DuckDBPyConnection,
    *,
    min_years: int = 4,
    min_rows_per_year: int = 1_000,
) -> list[Gate]:
    """Run every gate whose inputs exist. Thresholds default to real-data scale."""
    gates: list[Gate] = []
    if not _exists(con, "ita_keyed") or not _exists(con, "ita_panel"):
        return [Gate("panel_built", "fail", None, "ita_keyed and ita_panel exist", "run `build` first")]

    rows_by_year = dict(con.execute(
        "SELECT data_year, COUNT(*) FROM ita_keyed GROUP BY 1 ORDER BY 1").fetchall())
    gates.append(Gate("years_loaded", _grade(len(rows_by_year), fail_if=lambda v: v < 3,
                                             warn_if=lambda v: v < min_years),
                      len(rows_by_year), f">= {min_years} years (fail < 3)",
                      f"years: {sorted(rows_by_year)}"))
    small = {y: n for y, n in rows_by_year.items() if n < min_rows_per_year}
    gates.append(Gate("rows_per_year", "warn" if small else "pass", min(rows_by_year.values(), default=0),
                      f">= {min_rows_per_year:,} rows in every year", f"small years: {small}" if small else ""))

    valid_zip = con.execute("SELECT avg(CASE WHEN zip5 <> '' THEN 1 ELSE 0 END) FROM ita_keyed").fetchone()[0]
    gates.append(Gate("zip5_valid_share", _grade(valid_zip, fail_if=lambda v: v < 0.90, warn_if=lambda v: v < 0.98),
                      valid_zip, ">= 0.98 (fail < 0.90)"))

    states = "(" + ", ".join(f"'{s}'" for s in LEADING_ZERO_STATES) + ")"
    lz = con.execute(
        f"SELECT avg(CASE WHEN zip5 LIKE '0%' THEN 1 ELSE 0 END), COUNT(*) FROM ita_keyed "
        f"WHERE state_norm IN {states} AND zip5 <> ''"
    ).fetchone()
    gates.append(Gate("leading_zero_zips_kept", _grade(lz[0] if lz[1] else None, fail_if=lambda v: v < 0.80,
                                                       warn_if=lambda v: v < 0.95),
                      lz[0], ">= 0.95 of CT/MA/ME/NH/NJ/PR/RI/VT ZIPs start with 0",
                      "falls when a loader types ZIP as an integer"))

    hpe = con.execute(
        """SELECT median(total_hours_worked / annual_average_employees),
                  avg(CASE WHEN total_hours_worked / annual_average_employees NOT BETWEEN 100 AND 8760
                           THEN 1 ELSE 0 END)
           FROM ita_keyed WHERE annual_average_employees > 0 AND total_hours_worked > 0"""
    ).fetchone()
    gates.append(Gate("hours_per_employee_median", _grade(hpe[0], fail_if=lambda v: not 500 <= v <= 4000,
                                                          warn_if=lambda v: not 1200 <= v <= 2600),
                      hpe[0], "1,200 to 2,600 hours (fail outside 500 to 4,000)"))
    gates.append(Gate("hours_per_employee_outliers", _grade(hpe[1], fail_if=lambda v: v > 0.10,
                                                            warn_if=lambda v: v > 0.02),
                      hpe[1], "<= 2% outside 100 to 8,760 hours per employee"))

    dup = con.execute("SELECT avg(CASE WHEN n_raw_rows > 1 THEN 1 ELSE 0 END) FROM ita_panel").fetchone()[0]
    gates.append(Gate("duplicate_key_years", _grade(dup, fail_if=lambda v: v > 0.15, warn_if=lambda v: v > 0.05),
                      dup, "<= 5% of key-years had more than one filing (fail > 15%)",
                      "deduplicated in the panel; a high share means the key is too coarse"))

    mism = con.execute(
        "SELECT avg(CASE WHEN year_filing_for IS NOT NULL AND year_filing_for <> file_year THEN 1 ELSE 0 END) "
        "FROM ita_keyed"
    ).fetchone()[0]
    gates.append(Gate("year_filing_for_vs_file_year", _grade(mism, fail_if=lambda v: v > 0.5,
                                                             warn_if=lambda v: v > 0.05),
                      mism, "<= 5% of rows report a different year than their file",
                      "data_year follows year_filing_for either way"))

    split = con.execute(
        """WITH ids AS (SELECT data_year, establishment_id, COUNT(DISTINCT estab_key) AS k
                        FROM ita_keyed WHERE establishment_id IS NOT NULL GROUP BY 1, 2)
           SELECT avg(CASE WHEN k > 1 THEN 1 ELSE 0 END), COUNT(*) FROM ids"""
    ).fetchone()
    gates.append(Gate("hard_id_split_across_keys", _grade(split[0] if split[1] else None,
                                                          fail_if=lambda v: v > 0.10, warn_if=lambda v: v > 0.02),
                      split[0], "<= 2% of establishment_ids map to more than one key in a year"))

    sizes = con.execute("SELECT COUNT(*) FROM ita_panel WHERE size_code IN ('21', '22')").fetchone()[0]
    gates.append(Gate("size_code_harmonized", "fail" if sizes else "pass", sizes,
                      "no 21/22 size codes after harmonization"))

    stats = con.execute(
        """SELECT avg(target_any_recordable_next),
                  avg(CASE WHEN tcr_lag1 IS NOT NULL THEN 1 ELSE 0 END)
           FROM ita_panel"""
    ).fetchone()
    gates.append(Gate("target_base_rate", _grade(stats[0], fail_if=lambda v: v <= 0.02 or v >= 0.995,
                                                 warn_if=lambda v: not 0.2 <= v <= 0.95),
                      stats[0], "0.20 to 0.95"))
    gates.append(Gate("cross_year_link_share", _grade(stats[1], fail_if=lambda v: v < 0.10,
                                                      warn_if=lambda v: v < 0.30),
                      stats[1], ">= 30% of rows have last year's filing",
                      "low means establishment keys don't persist across years"))

    if _exists(con, "inspection_linked"):
        linked = con.execute(
            """SELECT avg(CASE WHEN estab_key IS NOT NULL THEN 1 ELSE 0 END)
               FROM inspection_linked
               WHERE inspection_year BETWEEN (SELECT min(year) FROM ita_panel)
                                         AND (SELECT max(year) FROM ita_panel)"""
        ).fetchone()[0]
        gates.append(Gate("inspection_link_share", _grade(linked, fail_if=lambda v: v < 0.02,
                                                          warn_if=lambda v: v < 0.15),
                          linked, ">= 15% of inspections in panel years link to an ITA filer",
                          "most inspections hit small or non-filing sites, so this is never near 100%"))
    return gates


def summarize(gates: list[Gate]) -> dict:
    counts: dict[str, int] = {}
    for g in gates:
        counts[g.status] = counts.get(g.status, 0) + 1
    return {"counts": counts, "ok": counts.get("fail", 0) == 0, "gates": [asdict(g) for g in gates]}
