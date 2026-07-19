"""Tests for the FastAPI scoring API.

Uses FastAPI's TestClient against an app built via create_app() with a
synthetic in-memory database and a model trained on synthetic data. No
network access, no real OSHA data, no env vars required.
"""

from __future__ import annotations

import io

import duckdb
import pytest
from fastapi.testclient import TestClient

from osha_risk.api import create_app
from osha_risk.enforcement import (
    add_soft_key_to_inspections,
    attach_enforcement_to_panel,
    build_enforcement_aggregates,
    load_inspection_csv,
    load_violation_csv,
)
from osha_risk.features import build_establishment_key_view, build_panel
from osha_risk.load import load_summary_csv
from osha_risk.models.baseline import train_baseline
from osha_risk.schema import build_canonical_view

from .synthetic import build_establishments, build_synthetic_dataset
from .synthetic_enforcement import build_synthetic_enforcement


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """Build a synthetic DB + trained model, wrap in a TestClient."""
    root = tmp_path_factory.mktemp("api_data")

    csvs = build_synthetic_dataset(root / "ita", n_establishments=800)
    con = duckdb.connect(str(root / "test.duckdb"))
    for year, p in csvs.items():
        load_summary_csv(p, year, con)
    build_canonical_view(con)
    build_establishment_key_view(con)
    build_panel(con)

    insp_csv, viol_csv = build_synthetic_enforcement(root / "enf", n_establishments=800)
    load_inspection_csv(con, insp_csv)
    load_violation_csv(con, viol_csv)
    add_soft_key_to_inspections(con)
    build_enforcement_aggregates(con)
    attach_enforcement_to_panel(con)

    model = train_baseline(
        con, train_through=2021, panel_table="ita_panel_enforced", num_boost_round=100
    )

    app = create_app(con, model, panel_table="ita_panel_enforced")
    client = TestClient(app)

    # Keep establishments handy for building requests
    estabs = build_establishments(n=800, seed=42)

    yield client, con, estabs
    con.close()


def test_health(api_client):
    client, _, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["panel_table"] == "ita_panel_enforced"
    assert "model_test_auc" in body


def test_score_exact_match(api_client):
    """Scoring an establishment with its exact ITA name/address should succeed."""
    client, _, estabs = api_client
    est = estabs[0]

    resp = client.post(
        "/score",
        json={
            "name": est["establishment_name"],
            "street_address": est["street_address"],
            "city": est["city"],
            "state": est["state"],
            "zip_code": est["zip_code"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["establishment_name"] == est["establishment_name"]
    assert 0 <= body["risk_score_next_year"] <= 1
    assert "risk_drivers" in body
    assert "caveats" in body
    assert body["match_info"]["match_method"] in ("softkey_exact", "splink_fuzzy")


def test_score_fuzzy_match(api_client):
    """A messy name variant should still resolve via splink."""
    client, _, estabs = api_client
    est = estabs[10]

    messy_name = (
        est["establishment_name"]
        .replace("Manufacturing Inc", "Mfg, Inc.")
        .replace("Manufacturing, LLC", "MFG LLC")
    )

    resp = client.post(
        "/score",
        json={
            "name": messy_name,
            "street_address": est["street_address"],
            "state": est["state"],
            "zip_code": f"{est['zip_code']}-1234",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["establishment_name"] == est["establishment_name"]


def test_score_no_match_returns_404(api_client):
    client, _, _ = api_client
    resp = client.post(
        "/score",
        json={
            "name": "Totally Fake Nonexistent Company XYZ",
            "state": "ZZ",
            "zip_code": "00000",
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"]["error"] == "no_match"


def test_score_missing_required_field(api_client):
    client, _, _ = api_client
    # Missing 'state', which is required
    resp = client.post("/score", json={"name": "Foo Inc", "zip_code": "12345"})
    assert resp.status_code == 422  # pydantic validation error


def test_score_batch_csv(api_client):
    client, _, estabs = api_client

    csv_lines = ["name,street_address,state,zip_code"]
    for est in estabs[:15]:
        csv_lines.append(
            f'"{est["establishment_name"]}","{est["street_address"]}",'
            f'{est["state"]},{est["zip_code"]}'
        )
    # Add a couple fakes
    csv_lines.append('"Fake Co A",999 Nowhere St,ZZ,00000')
    csv_lines.append('"Fake Co B",888 Nowhere Ave,ZZ,00001')
    csv_content = "\n".join(csv_lines)

    resp = client.post(
        "/score-batch",
        files={"file": ("upload.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["summary"]["n_input_rows"] == 17
    assert body["summary"]["n_matched"] >= 14  # allow 1 edge case
    assert body["summary"]["n_unmatched"] <= 3

    # Results should be sorted descending by risk score
    scores = [r["risk_score_next_year"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)

    # Each result has the expected shape
    if body["results"]:
        r = body["results"][0]
        for key in (
            "input_row_id", "input_name", "estab_key", "establishment_name",
            "risk_score_next_year", "confidence", "current_tcr",
        ):
            assert key in r


def test_score_batch_rejects_non_csv(api_client):
    client, _, _ = api_client
    resp = client.post(
        "/score-batch",
        files={"file": ("upload.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert resp.status_code == 400


def test_score_batch_rejects_missing_columns(api_client):
    client, _, _ = api_client
    csv_content = "company,location\nAcme,Somewhere"
    resp = client.post(
        "/score-batch",
        files={"file": ("upload.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert resp.status_code == 400
    assert "missing required" in resp.json()["detail"].lower()
