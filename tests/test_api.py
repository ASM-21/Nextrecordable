"""API: lookups, batch uploads, limits, and correctness under concurrency."""

from __future__ import annotations

import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from osha_risk import api as api_module
from osha_risk.api import create_app


@pytest.fixture(scope="module")
def client(world):
    return TestClient(create_app(world["con"], world["result"]))


def _csv(lines):
    return {"file": ("book.csv", io.BytesIO("\n".join(lines).encode()), "text/csv")}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_test_year"] == 2023
    assert body["test_auc"] is not None


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<form" in r.text


def test_score_exact(client, world):
    e = world["estabs"][0]
    r = client.post("/score", json={"name": e["establishment_name"], "state": e["state"],
                                    "zip_code": e["zip_code"], "street_address": e["street_address"]})
    assert r.status_code == 200
    body = r.json()
    assert body["establishment_name"] == e["establishment_name"]
    assert 0 < body["p_any_next"] < 1
    assert body["expected_tcr_next"] > 0
    assert 0 < body["peer_pctile"] <= 1
    assert body["risk_band"]
    assert body["risk_drivers"] and all("label" in d for d in body["risk_drivers"])
    assert body["caveats"]
    assert body["match_info"]["match_status"] == "matched"


def test_score_typo_and_lost_zip_zero(client, world):
    e = next(x for x in world["estabs"] if x["zip_code"].startswith("0"))
    name = e["establishment_name"]
    words = name.split()
    i = next(k for k, w in enumerate(words) if len(w) >= 5)
    w = words[i]
    words[i] = w[:2] + w[3] + w[2] + w[4:]  # swap two adjacent letters
    typo = " ".join(words)
    r = client.post("/score", json={"name": typo, "state": e["state"], "zip_code": str(int(e["zip_code"]))})
    assert r.status_code == 200
    assert r.json()["establishment_name"] == name


def test_no_match_returns_404_with_candidates(client, world):
    e = world["estabs"][1]
    r = client.post("/score", json={"name": "Zzyzx Unrelated Holdings", "state": e["state"], "zip_code": e["zip_code"]})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "no_match"
    assert isinstance(detail["candidates"], list)


def test_validation(client):
    assert client.post("/score", json={"name": "Foo Inc", "zip_code": "12345"}).status_code == 422
    assert client.post("/score", json={"name": "", "state": "IL", "zip_code": "12345"}).status_code == 422


def test_concurrent_lookups_return_the_right_establishment(client, world):
    """Regression: with one shared connection and shared match tables, 120
    concurrent lookups returned 18 wrong establishments and 100 errors."""
    sample = world["estabs"][:120]

    def call(e):
        r = client.post("/score", json={"name": e["establishment_name"], "state": e["state"], "zip_code": e["zip_code"]})
        return r.status_code, r.json().get("establishment_name")

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call, sample))
    assert results == [(200, e["establishment_name"]) for e in sample]


def test_batch_with_messy_headers_and_excel_zips(client, world):
    sample = world["estabs"][:60]
    lines = ["Company,Address,ST,Zip"] + [
        f'"{e["establishment_name"]}","{e["street_address"]}",{e["state"]},{int(e["zip_code"])}' for e in sample
    ] + ['"Nobody Here Inc","1 Nowhere Rd",IL,60601']
    r = client.post("/score-batch", files=_csv(lines))
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n_input_rows"] == 61
    assert body["summary"]["n_matched"] == 60
    assert body["unmatched"] == [{"input_row": 61, "input_name": "Nobody Here Inc", "reason": "unmatched"}]
    rates = [x["expected_tcr_next"] for x in body["results"]]
    assert rates == sorted(rates, reverse=True)


def test_batch_and_single_lookup_agree(client, world):
    e = world["estabs"][5]
    single = client.post("/score", json={"name": e["establishment_name"], "state": e["state"],
                                         "zip_code": e["zip_code"]}).json()
    batch = client.post("/score-batch", files=_csv(
        ["name,state,zip_code", f'"{e["establishment_name"]}",{e["state"]},{e["zip_code"]}'])).json()
    row = batch["results"][0]
    for k in ("p_any_next", "expected_tcr_next", "peer_pctile", "risk_band", "estab_key"):
        assert row[k] == single[k]


def test_batch_rejections(client, monkeypatch):
    assert client.post("/score-batch", files={"file": ("x.txt", io.BytesIO(b"a"), "text/plain")}).status_code == 400
    r = client.post("/score-batch", files=_csv(["company,location", "Acme,Somewhere"]))
    assert r.status_code == 400 and "missing required" in r.json()["detail"].lower()
    monkeypatch.setattr(api_module, "MAX_BATCH_ROWS", 3)
    rows = ["name,state,zip_code"] + ["Acme,IL,60601"] * 4
    assert client.post("/score-batch", files=_csv(rows)).status_code == 413
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 50)
    assert client.post("/score-batch", files=_csv(["name,state,zip_code"] + ["Acme Widgets,IL,60601"] * 5)).status_code == 413
