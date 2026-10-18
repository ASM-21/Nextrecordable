"""DOL enforcement: real code formats, deleted citations, linking, panel windows."""

from __future__ import annotations

import csv

import duckdb
import pytest

from osha_risk.enforcement import (
    attach_enforcement_to_panel,
    build_enforcement_aggregates,
    link_inspections,
    load_inspection_csv,
    load_violation_csv,
)
from osha_risk.features import build_keyed_view, build_panel
from osha_risk.load import load_summary_csv
from osha_risk.match import build_directory
from osha_risk.schema import build_canonical_view

from .conftest import ita_row, write_ita_csv


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def small_world(tmp_path):
    """One ITA filer and two inspections written exactly like the DOL files."""
    con = duckdb.connect()
    for year in (2020, 2021, 2022):
        load_summary_csv(write_ita_csv(tmp_path / f"ita_{year}.csv", [ita_row("Acme Forge Inc")]), year, con)
    build_canonical_view(con)
    build_keyed_view(con)
    build_panel(con)
    build_directory(con)
    insp = _write(tmp_path / "insp.csv",
                  ["activity_nr", "estab_name", "site_address", "site_city", "site_state", "site_zip",
                   "naics_code", "insp_type", "insp_scope", "open_date"],
                  [["1001", "ACME FORGE, INC.", "5 MILL RD", "PEORIA", "IL", "61602", "332111", "B", "B", "2021-04-02"],
                   ["1002", "Acme Forge", "5 Mill Rd", "Peoria", "IL", "61602", "332111", "A", "A", "09/15/2021"],
                   ["1003", "Unrelated Bakery", "9 Elm St", "Peoria", "IL", "61602", "311811", "H", "A", "2021-05-01"]])
    viol = _write(tmp_path / "viol.csv",
                  ["activity_nr", "citation_id", "delete_flag", "standard", "viol_type",
                   "initial_penalty", "current_penalty"],
                  [["1001", "01001", "", "19100212", "S", "9000", "6000"],
                   ["1002", "01001", "", "19100147", "W", "70000", "70000"],
                   ["1002", "01002", "X", "19100147", "S", "15000", "0"],   # deleted
                   ["1003", "01001", "", "19100023", "O", "500", "500"]])
    load_inspection_csv(con, insp)
    load_violation_csv(con, viol)
    return con


def test_real_dol_codes_are_counted(small_world):
    """Regression: the first version matched 'Serious'/'Willful' and phrases like
    'complaint'. On DOL's letter codes every one of these counts was zero."""
    con = small_world
    links = link_inspections(con)
    assert links.get("unlinked") == 1  # the bakery
    build_enforcement_aggregates(con)
    row = con.execute(
        """SELECT n_inspections, n_unprogrammed, n_complaint_referral, n_fatcat, n_citations,
                  n_serious_citations, n_willful_repeat_citations, initial_penalty, max_severity
           FROM enforcement_estab_year"""
    ).fetchall()
    # two inspections (complaint B + fat/cat A), two live citations (deleted one excluded),
    # initial penalties (not current), willful is the most severe
    assert row == [(2, 2, 1, 1, 2, 2, 1, 79000.0, 4)]


def test_window_attaches_years_t_and_t_minus_1(small_world):
    con = small_world
    link_inspections(con)
    build_enforcement_aggregates(con)
    attach_enforcement_to_panel(con)
    got = dict(con.execute("SELECT year, enf_inspections_2y FROM ita_panel_enforced").fetchall())
    assert got == {2020: 0, 2021: 2, 2022: 2}
    since = dict(con.execute("SELECT year, enf_years_since_inspection FROM ita_panel_enforced").fetchall())
    assert since == {2020: 10, 2021: 0, 2022: 1}


def test_synthetic_world_links_dol_spellings(world):
    """DOL spells employers differently (caps, MFG, dropped suffixes, typos) and
    stores ZIPs as integers; most inspections of ITA filers should still link,
    and inspections of non-filers must not."""
    con = world["con"]
    by_method = dict(con.execute("SELECT link_method, COUNT(*) FROM inspection_linked GROUP BY 1").fetchall())
    assert by_method.get("fuzzy", 0) > 0
    nonfiler_linked = con.execute(
        """SELECT COUNT(*) FROM inspection_linked l JOIN osha_inspection i USING (activity_nr)
           WHERE i.estab_name ILIKE 'smallshop%' AND l.estab_key IS NOT NULL"""
    ).fetchone()[0]
    assert nonfiler_linked == 0
    filer_total, filer_linked = con.execute(
        """SELECT COUNT(*), COUNT(l.estab_key) FROM inspection_linked l JOIN osha_inspection i USING (activity_nr)
           WHERE i.estab_name NOT ILIKE 'smallshop%'"""
    ).fetchone()
    assert filer_linked / filer_total > 0.97


def test_enforced_panel_keeps_every_row(world):
    con = world["con"]
    assert con.execute("SELECT COUNT(*) FROM ita_panel").fetchone()[0] == \
        con.execute("SELECT COUNT(*) FROM ita_panel_enforced").fetchone()[0]
