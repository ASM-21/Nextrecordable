"""SST exposure: the directive file, the position logic, and the wording rules."""

from __future__ import annotations

import inspect
from datetime import date

import duckdb
import pytest
from fastapi.testclient import TestClient

from nextrecordable import provenance, sst
from nextrecordable.api import create_app
from nextrecordable.features import build_keyed_view, build_panel
from nextrecordable.keys import python_key
from nextrecordable.load import load_summary_csv
from nextrecordable.schema import build_canonical_view

from .conftest import ita_row, write_ita_csv


def plant(name: str, dart: int, *, employees: int = 100, naics: str = "332111", **extra) -> dict:
    """A filing whose DART rate equals `dart` (200,000 hours)."""
    return ita_row(name, cases=0, hours=200_000, employees=employees, naics_code=naics,
                   total_dafw_cases=dart, **extra)


@pytest.fixture(scope="module")
def sst_con(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sst")
    peers = {y: [plant(f"Peer Plant {i}", i) for i in range(40)] for y in (2021, 2022, 2023)}
    warehouses = {y: [plant(f"Peer Warehouse {i}", i % 5, naics="493110") for i in range(30)]
                  for y in (2021, 2022, 2023)}
    cases = {
        2021: [plant("Rising Forge", 3), plant("Slow Riser", 1), plant("High Flat", 9),
               plant("Falling Works", 9), plant("Gone Quiet Inc", 2, employees=50),
               plant("Tiny Shop", 1, employees=12), plant("County Garage", 4, establishment_type="3"),
               plant("Build Co", 6, naics="236220")],
        2022: [plant("Rising Forge", 4), plant("Slow Riser", 2), plant("High Flat", 9),
               plant("Falling Works", 6), plant("Gone Quiet Inc", 2, employees=50),
               plant("Tiny Shop", 1, employees=12), plant("County Garage", 4, establishment_type="3"),
               plant("Build Co", 6, naics="236220"), plant("Newcomer LLC", 5)],
        2023: [plant("Rising Forge", 5), plant("Slow Riser", 3), plant("High Flat", 9),
               plant("Falling Works", 4), plant("Tiny Shop", 1, employees=12),
               plant("County Garage", 4, establishment_type="3"), plant("Build Co", 6, naics="236220"),
               plant("Newcomer LLC", 6),
               ita_row("Bad Hours Ltd", cases=0, hours=9_000_000_000, employees=100, total_dafw_cases=3)],
    }
    con = duckdb.connect()
    for y in (2021, 2022, 2023):
        load_summary_csv(write_ita_csv(tmp / f"ita_{y}.csv", peers[y] + warehouses[y] + cases[y]), y, con)
    build_canonical_view(con)
    build_keyed_view(con)
    build_panel(con)
    sst.build_sst_table(con)
    yield con
    con.close()


def report(con, name: str) -> dict:
    return sst.sst_exposure(con, python_key(name, "IL", "61602"), as_of=date(2026, 11, 16))


def criterion(rep: dict, cid: str) -> dict:
    return next(c for c in rep["criteria"] if c["id"] == cid)


# ------------------------------------------------------------ directive file

def test_directive_file_pins_what_was_read_from_the_directive():
    d = sst.load_directive()
    assert (d["directive"], d["signature_date"], d["effective_date"]) == ("CPL 02-01-067", "2025-04-08", "2025-05-20")
    assert d["expires_on_or_about"] == "2027-05-20"            # two years from the effective date
    assert d["data_years"] == [2021, 2022, 2023] and d["selection_year"] == 2023
    assert d["scope"]["min_employees"] == 20 and d["scope"]["excluded_naics_prefixes"] == ["23"]
    assert d["deletions"]["previous_inspection_months"] == 36
    assert d["program_size"]["inspections"] == 652
    for key in ("official_url", "text_read_from", "retrieved"):
        assert d["source"][key]
    assert d["successor"]["announced"] is False and d["successor"]["checked"]


def test_no_criterion_claims_a_published_numeric_cutoff():
    """OSHA publishes no DART cutoffs in the directive; the product must never imply one."""
    crit = sst.load_directive()["criteria"]
    assert set(crit) == {"high_rate", "upward_trending", "low_rate", "non_responders"}
    assert all(c["numeric_cutoff_published"] is False and c["cutoff_note"] and c["quote"] for c in crit.values())
    assert "one DART rate for manufacturing" in crit["high_rate"]["quote"]
    assert "twice the private sector national average in CY 2022" in crit["upward_trending"]["quote"]


def test_derived_threshold_is_arithmetic_on_a_sourced_number():
    ut = sst.load_directive()["criteria"]["upward_trending"]
    assert ut["derived_threshold"] == pytest.approx(ut["multiple"] * ut["private_sector_dart_rate"])
    assert ut["private_sector_dart_rate"] == 1.7 and "BLS" in ut["private_sector_dart_source"]
    assert "ours, not OSHA's" in ut["cutoff_note"]


# ----------------------------------------------------------------- positions

def test_upward_trending_states(sst_con):
    assert criterion(report(sst_con, "Rising Forge"), "upward_trending")["status"] == "meets_both_conditions_as_read"
    assert criterion(report(sst_con, "Slow Riser"), "upward_trending")["status"] == "meets_trend_condition_only"
    assert criterion(report(sst_con, "High Flat"), "upward_trending")["status"] == "meets_rate_condition_only"
    assert criterion(report(sst_con, "Falling Works"), "upward_trending")["status"] == "meets_rate_condition_only"
    newcomer = criterion(report(sst_con, "Newcomer LLC"), "upward_trending")
    assert newcomer["status"] == "not_evaluable" and "missing 2021" in newcomer["position"]


def test_percentile_is_share_of_compared_group_with_a_lower_rate(sst_con):
    rep = report(sst_con, "Rising Forge")                     # DART 5.0 in 2023
    group = rep["peers"]["group"]
    # manufacturing comparison set: 40 peers (0..39) + Rising 5, Slow 3, High 9, Falling 4, Newcomer 6
    assert group["n"] == 45
    lower = 5 + 2                                              # peers 0..4, Slow Riser 3, Falling Works 4
    assert group["share_with_lower_rate"] == pytest.approx(lower / 45)
    assert rep["sst_group"] == "manufacturing" and rep["in_scope"] is True
    high = criterion(rep, "high_rate")
    assert high["status"] == "no_published_cutoff" and "does not publish" in high["position"]
    # warehouses are ranked among non-manufacturing only
    wh = sst.sst_exposure(sst_con, python_key("Peer Warehouse 4", "IL", "61602"))
    assert wh["sst_group"] == "non_manufacturing" and wh["peers"]["group"]["n"] == 30


def test_out_of_scope_cases_are_described_not_ranked(sst_con):
    tiny = report(sst_con, "Tiny Shop")
    assert tiny["in_scope"] is False and "fewer than 20 employees" in tiny["scope_note"] and tiny["peers"] is None
    build = report(sst_con, "Build Co")
    assert build["in_scope"] is False and "Construction" in build["scope_note"]
    gov = report(sst_con, "County Garage")
    assert gov["in_scope"] is False and "public-sector" in gov["scope_note"]
    bad = report(sst_con, "Bad Hours Ltd")
    assert bad["in_scope"] is False and "not usable" in bad["scope_note"]
    assert criterion(bad, "high_rate")["status"] == "not_evaluable"


def test_missing_selection_year_filing_is_reported_without_accusation(sst_con):
    rep = report(sst_con, "Gone Quiet Inc")
    nr = criterion(rep, "non_responders")
    assert nr["status"] == "no_filing_found" and rep["non_filer"]["status"] == "no_filing_found"
    assert "not a finding that a required filing was missed" in nr["position"]
    churn = sst.churn(sst_con)
    assert set(churn) == {2022, 2023} and churn[2023] > 0      # Gone Quiet Inc is in the 2023 share
    assert criterion(report(sst_con, "Rising Forge"), "non_responders")["status"] == "filing_found"


def test_unknown_key_and_bad_key(sst_con):
    assert sst.sst_exposure(sst_con, "not-a-key")["error"] == "no_data"
    assert sst.sst_exposure(sst_con, "0" * 32)["error"] == "no_data"


# -------------------------------------------------------------- wording rules

FORBIDDEN = ("will be inspected", "will inspect you", "likely to be inspected", "probability of inspection",
             "chance of inspection", "inspection risk", "on the list", "you are targeted")


def test_statement_is_on_every_report_and_no_report_predicts(sst_con):
    assert sst.STATEMENT == ("This report describes position against published criteria and never predicts "
                             "whether OSHA will inspect.")
    for name in ("Rising Forge", "High Flat", "Gone Quiet Inc", "Tiny Shop", "Build Co"):
        rep = report(sst_con, name)
        assert rep["statement"] == sst.STATEMENT
        text = " ".join(c["position"] for c in rep["criteria"]).lower() + rep["scope_note"].lower()
        assert not [f for f in FORBIDDEN if f in text]
    assert sst.sst_exposure(sst_con, "zz")["statement"] == sst.STATEMENT


def test_pdf_prints_the_statement_and_the_data_notice(sst_con):
    rep = report(sst_con, "Rising Forge")
    pdf = sst.render_pdf(rep, compress=False)
    assert pdf.startswith(b"%PDF") and sst.STATEMENT.encode() in pdf
    assert b"UNDECLARED DATA" in pdf                      # this fixture declared no source
    assert sst.render_pdf(rep).startswith(b"%PDF")        # compressed default also renders
    assert sst.render_pdf({"error": "no_data", "message": "x", "statement": sst.STATEMENT}).startswith(b"%PDF")


def test_report_uses_only_summary_data():
    """Form 301 narratives and other case-level fields must be unreachable from the SST report.

    Checks code, not prose: the module docstring explains the exclusion in words.
    """
    import ast

    tree = ast.parse(inspect.getsource(sst))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code = ast.unparse(tree).lower()
    for word in ("case_detail", "nar_what", "nar_before", "nar_injury", "incident_description",
                 "job_description", "date_of_incident", "soc_code", "case_number"):
        assert word not in code, word


def test_inspection_context_respects_36_months(sst_con):
    key = python_key("Rising Forge", "IL", "61602")
    sst_con.execute("CREATE OR REPLACE TABLE osha_inspection AS SELECT '1' AS activity_nr, 'A' AS insp_scope, "
                    "'2024-06-10 00:00:00+00:00' AS open_date")
    sst_con.execute("CREATE OR REPLACE TABLE inspection_linked AS SELECT '1' AS activity_nr, ? AS estab_key, "
                    "'2024-06-10 00:00:00+00:00' AS open_date", [key])
    try:
        dc = sst.sst_exposure(sst_con, key, as_of=date(2026, 11, 16))["deletion_context"]
        assert dc["months_since_last_comprehensive"] == 29 and dc["within_36_months"] is True
        late = sst.sst_exposure(sst_con, key, as_of=date(2027, 7, 1))["deletion_context"]
        assert late["within_36_months"] is False
    finally:
        sst_con.execute("DROP TABLE inspection_linked")
        sst_con.execute("DROP TABLE osha_inspection")


# ------------------------------------------------------------------------ API

@pytest.fixture(scope="module")
def client(world):
    return TestClient(create_app(world["con"], world["result"]))


def _lookup(world) -> dict:
    name, state, z = world["con"].execute(
        "SELECT p.establishment_name, p.state, p.zip5 FROM ita_panel p JOIN sst_position s USING (estab_key) "
        "WHERE s.in_comparison AND p.year = 2023 ORDER BY p.estab_key LIMIT 1"
    ).fetchone()
    return {"name": name, "state": state, "zip_code": z}


def test_sst_endpoint_and_pdf(world, client):
    body = _lookup(world)
    r = client.post("/sst-exposure", json=body)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["statement"] == sst.STATEMENT and rep["directive"]["directive"] == "CPL 02-01-067"
    assert [c["id"] for c in rep["criteria"]] == ["high_rate", "upward_trending", "low_rate", "non_responders"]
    assert rep["peers"]["group"]["n"] > 0 and rep["match_info"]["match_status"] == "matched"
    pdf = client.post("/sst-exposure-pdf", json=body)
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    miss = client.post("/sst-exposure", json={"name": "Nonexistent Widgets", "state": "IL", "zip_code": "60601"})
    assert miss.status_code == 404


def test_synthetic_runs_say_so_everywhere(world, client):
    """Any output from a synthetic run carries the notice: API, PDF and METRICS.md."""
    body = _lookup(world)
    notice = provenance.NOTICES["synthetic"]
    assert client.get("/health").json()["data_notice"] == notice
    assert client.post("/score", json=body).json()["data_notice"] == notice
    assert client.post("/sst-exposure", json=body).json()["data_notice"] == notice
    rep = sst.sst_exposure(world["con"], world["con"].execute(
        "SELECT estab_key FROM sst_position LIMIT 1").fetchone()[0])
    assert b"SYNTHETIC DATA" in sst.render_pdf(rep, compress=False)
    assert "SYNTHETIC DATA" in (world["root"] / "reports" / "METRICS.md").read_text()
    csv_body = f"name,state,zip\n{body['name']},{body['state']},{body['zip_code']}\n"
    batch = client.post("/score-batch", files={"file": ("b.csv", csv_body, "text/csv")})
    assert batch.json()["summary"]["data_notice"] == notice


def test_undeclared_database_is_never_silently_real():
    con = duckdb.connect()
    assert provenance.summary(con)["kind"] == "undeclared"
    assert provenance.notice(con).startswith("UNDECLARED DATA")
    provenance.declare(con, "official")
    assert provenance.notice(con) is None
