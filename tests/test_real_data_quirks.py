"""Regressions from the first run on real OSHA and DOL files (November 2026).

Each test reproduces something the real files did that the synthetic fixtures
had not, and that either stopped the pipeline or quietly bent its numbers.
"""

from __future__ import annotations

import csv

import duckdb
import pytest

from nextrecordable import config
from nextrecordable.enforcement import (
    build_enforcement_aggregates,
    find_dol_files,
    load_inspection_csv,
    load_violation_csv,
)
from nextrecordable.features import build_keyed_view, build_panel
from nextrecordable.load import load_all_summary, load_summary_csv
from nextrecordable.schema import build_canonical_view

from .conftest import ita_row, write_ita_csv


def _panel(con, years_rows: dict[int, list[dict]], tmp_path) -> None:
    for year, rows in years_rows.items():
        load_summary_csv(write_ita_csv(tmp_path / f"ita_{year}.csv", rows), year, con)
    build_canonical_view(con)
    build_keyed_view(con)
    build_panel(con)


def test_line_break_inside_a_quoted_field_loads(tmp_path):
    """The real 2019 file has addresses with line breaks; the load died on line 4,151."""
    rows = [ita_row(f"Plant {i}") for i in range(50)]
    rows[10]["street_address"] = "5 Mill Rd\nBuilding C"
    con = duckdb.connect()
    assert load_summary_csv(write_ita_csv(tmp_path / "ita_2019.csv", rows), 2019, con) == 50
    assert con.execute(
        "SELECT COUNT(*) FROM ita_summary_2019 WHERE street_address LIKE '%Building C'"
    ).fetchone()[0] == 1


def test_two_decimal_export_does_not_change_ids_sizes_or_years(tmp_path):
    """The real CY 2021 file writes establishment_id 439896.00, size 2.00, year 2021.00."""
    con = duckdb.connect()
    _panel(con, {
        2020: [ita_row("Acme Forge Inc", establishment_id="439896", year_filing_for="2020")],
        2021: [ita_row("Acme Forge Inc", establishment_id="439896.00", year_filing_for="2021.00",
                       size="2.00", naics_code="332111.00")],
    }, tmp_path)
    assert con.execute(
        "SELECT year, establishment_id, size_code, naics_code FROM ita_panel ORDER BY year"
    ).fetchall() == [(2020, "439896", "2", "332111"), (2021, "439896", "2", "332111")]


@pytest.mark.parametrize("earlier, later", [
    ("9/9/2020 21:52", "10/1/2020 8:05"),           # 2019 file; string order picks the wrong one
    ("9/9/2021 3:51:00", "10/12/2021 13:57:57"),    # 2020 and 2021 files
    ("9/9/2023", "10/1/2023"),                      # 2022 file, date only
    ("31JAN24:23:59:00", "02FEB24:13:03:00"),       # 2023 file
])
def test_latest_amendment_wins_in_every_timestamp_format(tmp_path, earlier, later):
    con = duckdb.connect()
    _panel(con, {2021: [
        ita_row("Acme Forge Inc", cases=9, establishment_id="7", created_timestamp=later),
        ita_row("Acme Forge Inc", cases=0, establishment_id="7", created_timestamp=earlier),
    ]}, tmp_path)
    assert con.execute("SELECT n_recordable, n_raw_rows FROM ita_panel").fetchall() == [(9, 2)]


def test_impossible_hours_cannot_enter_a_rate(tmp_path):
    """A seven-employee filer reported 16,831,600,000,000 hours; pooled 2019 TCR fell from 4.0 to 0.02."""
    good = [ita_row(f"Plant {i}", cases=4, hours=200_000, employees=100) for i in range(30)]
    bad = ita_row("Typo Works", cases=0, hours=16_831_600_000_000, employees=7)
    tiny = ita_row("Two Hour Shop", cases=1, hours=40, employees=20)       # TCR would be 5,000
    huge = ita_row("Eight Million Clinic", cases=7, hours=14_000_000_000, employees=8_041_667)
    con = duckdb.connect()
    _panel(con, {2022: [*good, bad, tiny, huge]}, tmp_path)
    flagged = con.execute(
        "SELECT establishment_name, hours_worked, tcr, hours_reported FROM ita_panel "
        "WHERE NOT exposure_ok ORDER BY 1"
    ).fetchall()
    assert [r[0] for r in flagged] == ["Eight Million Clinic", "Two Hour Shop", "Typo Works"]
    assert all(r[1] is None and r[2] is None and r[3] is not None for r in flagged)
    pooled, median, n = con.execute(
        "SELECT naics4_pooled_tcr, naics4_tcr_median, naics4_n_estabs FROM ita_panel "
        "WHERE establishment_name = 'Plant 0'"
    ).fetchone()
    assert (pooled, median, n) == (4.0, 4.0, 30)
    assert con.execute("SELECT max(tcr) FROM ita_panel").fetchone()[0] == 4.0


def test_csv_dropped_into_raw_dir_is_loaded(tmp_path, monkeypatch):
    """DATA-GAPS.md tells Andrew to drop ita_summary_<year>.csv into data/raw/ita_summary."""
    raw = tmp_path / "raw"
    write_ita_csv(raw / "ita_summary" / "ita_summary_2022.csv", [ita_row("Acme Forge Inc")])
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    con = duckdb.connect()
    assert load_all_summary(con, raw_dir=raw) == {2022: 1}


def _dol_bulk(tmp_path):
    """Two chunk files per table, upper-case headers, pandas placeholders."""
    insp_dir, viol_dir = tmp_path / "dol" / "inspection", tmp_path / "dol" / "violation"
    insp_dir.mkdir(parents=True)
    viol_dir.mkdir(parents=True)
    head = ["ACTIVITY_NR", "ESTAB_NAME", "SITE_ADDRESS", "SITE_CITY", "SITE_STATE", "SITE_ZIP",
            "NAICS_CODE", "INSP_TYPE", "INSP_SCOPE", "OPEN_DATE", "CLOSE_CASE_DATE"]
    chunks = {
        "2026-11-15_chunk_1.csv": [
            ["1", "OLD FORGE", "1 A ST", "PEORIA", "IL", "61602", "332111", "H", "A",
             "1978-08-18 00:00:00+00:00", "NaT"]],
        "2026-11-15_chunk_2.csv": [
            ["2", "ACME FORGE INC", "5 MILL RD", "PEORIA", "IL", "61602", "332111", "B", "A",
             "2022-03-05 00:00:00+00:00", "NaT"]],
    }
    for name, rows in chunks.items():
        with open(insp_dir / name, "w", newline="") as f:
            csv.writer(f).writerows([head, *rows])
    vhead = ["ACTIVITY_NR", "CITATION_ID", "DELETE_FLAG", "STANDARD", "VIOL_TYPE", "ISSUANCE_DATE",
             "CURRENT_PENALTY", "INITIAL_PENALTY", "GRAVITY"]
    vrows = [
        ["1", "01001", "", "19100212", "S", "1978-09-01 00:00:00+00:00", "", "100.00", "5"],
        ["2", "01001", "", "19100212", "S", "2022-04-01 00:00:00+00:00", "", "4500.00", "10"],
        ["2", "01002", "", "19100147", "O", "2022-04-01 00:00:00+00:00", "", "nan", "nan"],
    ]
    with open(viol_dir / "2026-11-15_chunk_1.csv", "w", newline="") as f:
        csv.writer(f).writerows([vhead, *vrows])
    return tmp_path / "dol"


def test_dol_bulk_layout_chunks_and_missing_tokens(tmp_path):
    """data.dol.gov ships chunked pandas exports: 'nan' penalties made penalty sums NaN."""
    dol = _dol_bulk(tmp_path)
    insp, viol = find_dol_files(dol)
    assert [p.name for p in insp] == ["2026-11-15_chunk_1.csv", "2026-11-15_chunk_2.csv"]
    con = duckdb.connect()
    assert load_inspection_csv(con, insp, since_year=2006) == 1          # 1978 inspection dropped
    assert load_violation_csv(con, viol, inspection_table="osha_inspection") == 2
    assert con.execute("SELECT close_case_date FROM osha_inspection").fetchone()[0] is None
    con.execute("CREATE TABLE inspection_linked AS SELECT activity_nr, insp_type, open_date, "
                "2022 AS inspection_year, 'k' AS estab_key, 'exact_key' AS link_method FROM osha_inspection")
    build_enforcement_aggregates(con)
    assert con.execute(
        "SELECT n_citations, n_serious_citations, initial_penalty FROM enforcement_estab_year"
    ).fetchall() == [(2, 1, 4500.0)]


def test_dol_legacy_flat_files_still_found(tmp_path):
    dol = tmp_path / "dol"
    dol.mkdir()
    (dol / "osha_inspection_20260101.csv").write_text("activity_nr\n1\n")
    (dol / "osha_violation_20260101.csv").write_text("activity_nr\n1\n")
    insp, viol = find_dol_files(dol)
    assert [p.name for p in insp] == ["osha_inspection_20260101.csv"]
    assert [p.name for p in viol] == ["osha_violation_20260101.csv"]
