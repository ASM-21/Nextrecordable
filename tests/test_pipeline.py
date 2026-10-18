"""Data layer: loading, canonical schema, keys, and the panel's integrity rules."""

from __future__ import annotations

import csv

import duckdb
import pytest

from nextrecordable.features import build_keyed_view, build_panel, panel_summary
from nextrecordable.load import load_summary_csv
from nextrecordable.schema import build_canonical_view

from .conftest import ita_row, write_ita_csv


def _panel(con, years_rows: dict[int, list[dict]], tmp_path) -> None:
    for year, rows in years_rows.items():
        load_summary_csv(write_ita_csv(tmp_path / f"ita_{year}.csv", rows), year, con)
    build_canonical_view(con)
    build_keyed_view(con)
    build_panel(con)


def test_leading_zero_zip_survives_type_sniffing(tmp_path):
    """Regression: with 60k ZIPs and no leading zero first, DuckDB typed the column
    BIGINT and stored CT's 06511 as 6511; the establishment could never be found."""
    path = tmp_path / "ita_2022.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["establishment_name", "state", "zip_code", "annual_average_employees", "total_hours_worked"])
        for i in range(60_000):
            w.writerow([f"Plant {i}", "CA", 90000 + i % 9999, 50, 100_000])
        w.writerow(["Elm City Castings Inc", "CT", "06511", 120, 240_000])
    con = duckdb.connect()
    load_summary_csv(path, 2022, con)
    build_canonical_view(con)
    build_keyed_view(con)
    assert con.execute("SELECT typeof(zip_code) FROM ita_summary_2022 LIMIT 1").fetchone()[0] == "VARCHAR"
    assert con.execute(
        "SELECT zip5 FROM ita_keyed WHERE establishment_name LIKE 'Elm City%'"
    ).fetchone()[0] == "06511"


def test_schema_drift_and_size_harmonization(world):
    con = world["con"]
    # legacy files (company_name / naics / estab_zip) and current files land in one schema
    rows = con.execute(
        "SELECT data_year, COUNT(*) FILTER (WHERE naics_code IS NOT NULL AND establishment_name IS NOT NULL "
        "AND zip_code IS NOT NULL), COUNT(*) FROM ita_canonical GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert [r[0] for r in rows] == [2019, 2020, 2021, 2022, 2023, 2024]
    assert all(complete == total for _, complete, total in rows)
    # 21/22 exist in raw 2023+ files but never in the harmonized size_code
    assert con.execute("SELECT COUNT(*) FROM ita_canonical WHERE size_code_raw IN ('21', '22')").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM ita_panel WHERE size_code IN ('21', '22')").fetchone()[0] == 0


def test_year_filing_for_beats_file_year(tmp_path):
    con = duckdb.connect()
    _panel(con, {2023: [ita_row("Late Filer Inc", year_filing_for=2022), ita_row("On Time Inc", year_filing_for=2023)]},
           tmp_path)
    got = dict(con.execute("SELECT establishment_name, year FROM ita_panel").fetchall())
    assert got == {"Late Filer Inc": 2022, "On Time Inc": 2023}


def test_amended_filing_does_not_leak_into_targets(tmp_path):
    """Regression: two rows for one establishment-year made LAG/LEAD step within the
    year, so a 2021 row's 'next year' target was the same year's amended filing."""
    con = duckdb.connect()
    _panel(con, {
        2020: [ita_row("Acme Forge Inc", cases=2)],
        2021: [ita_row("Acme Forge Inc", cases=0, created_timestamp="2022-02-01 10:00:00"),
               ita_row("Acme Forge Inc", cases=9, created_timestamp="2022-06-01 10:00:00")],
        2022: [ita_row("Acme Forge Inc", cases=5)],
    }, tmp_path)
    rows = con.execute(
        "SELECT year, n_recordable, n_recordable_lag1, target_n_recordable_next, n_raw_rows "
        "FROM ita_panel ORDER BY year"
    ).fetchall()
    assert rows == [(2020, 2, None, 9, 1), (2021, 9, 2, 5, 2), (2022, 5, 9, None, 1)]


def test_two_sites_with_one_name_and_zip_are_combined(tmp_path):
    con = duckdb.connect()
    _panel(con, {2023: [ita_row("Acme Forge Inc", cases=2, establishment_id="1", employees=50, hours=100_000),
                        ita_row("Acme Forge Inc", cases=3, establishment_id="2", employees=70, hours=140_000)]},
           tmp_path)
    row = con.execute("SELECT n_sites, n_recordable, employees, hours_worked FROM ita_panel").fetchone()
    assert row == (2, 5, 120.0, 240_000.0)


def test_skipped_year_gives_no_lag_and_no_target(tmp_path):
    """Regression: an establishment missing 2020 got its 2021 count as the 2019
    row's 'next year' target and its 2019 value as the 2021 row's lag."""
    con = duckdb.connect()
    other = ita_row("Other Co", zip_code="61603")
    _panel(con, {
        2019: [ita_row("Gap Works Inc", cases=1), other],
        2020: [other],
        2021: [ita_row("Gap Works Inc", cases=6), other],
        2022: [ita_row("Gap Works Inc", cases=2), other],
    }, tmp_path)
    rows = con.execute(
        "SELECT year, n_recordable_lag1, target_n_recordable_next, tcr_lag2 FROM ita_panel "
        "WHERE establishment_name = 'Gap Works Inc' ORDER BY year"
    ).fetchall()
    assert rows == [(2019, None, None, None), (2021, None, 2, None), (2022, 6, None, None)]


def test_rates_and_benchmarks(world):
    con = world["con"]
    n, hours, tcr, dart, n_dart = con.execute(
        "SELECT n_recordable, hours_worked, tcr, dart_rate, n_dart FROM ita_panel "
        "WHERE hours_worked > 0 AND n_recordable > 0 LIMIT 1"
    ).fetchone()
    assert tcr == pytest.approx(n * 200_000 / hours)
    assert dart == pytest.approx(n_dart * 200_000 / hours)
    bad = con.execute(
        """SELECT COUNT(*) FROM ita_panel p
           JOIN (SELECT year, naics_4, sum(n_recordable) * 200000 / sum(hours_worked) AS pooled
                 FROM ita_panel WHERE hours_worked > 0 AND naics_4 <> '' GROUP BY 1, 2) b USING (year, naics_4)
           WHERE abs(p.naics4_pooled_tcr - b.pooled) > 1e-9"""
    ).fetchone()[0]
    assert bad == 0


def test_lags_point_at_the_previous_calendar_year(world):
    con = world["con"]
    bad = con.execute(
        """SELECT COUNT(*) FROM ita_panel p1
           LEFT JOIN ita_panel p0 ON p0.estab_key = p1.estab_key AND p0.year = p1.year - 1
           WHERE p1.tcr_lag1 IS DISTINCT FROM p0.tcr"""
    ).fetchone()[0]
    assert bad == 0


def test_panel_summary_counts(world):
    s = panel_summary(world["con"])
    assert s["total_rows"] > 0
    assert s["distinct_establishments"] <= 1200
    assert s["key_years_with_duplicates"] > 0  # the fixture writes amended filings
    assert 0.3 < s["target_base_rate"] < 1.0
