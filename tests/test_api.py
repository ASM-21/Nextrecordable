"""API: lookups, batch uploads, limits, and correctness under concurrency."""

from __future__ import annotations

import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from nextrecordable import api as api_module
from nextrecordable.api import create_app


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


def test_no_drivers_when_the_credibility_rate_is_served(world):
    """Binary log-odds once appeared under 'what moves the rate estimate' when no rate model was served."""
    from dataclasses import replace

    from nextrecordable.report import risk_report

    con, result = world["con"], world["result"]
    key = con.execute("SELECT estab_key FROM ita_panel ORDER BY estab_key LIMIT 1").fetchone()[0]
    plain = risk_report(con, key, replace(result, rate_served="credibility"))
    assert plain["risk_drivers"] == [] and plain["rate_served"] == "credibility" and plain["drivers_note"]
    boosted = risk_report(con, key, replace(result, rate_served="model"))
    assert boosted["risk_drivers"] and all(d["space"] == "log_rate" for d in boosted["risk_drivers"])


def test_lookup_page_wires_sst_and_data_notice():
    from pathlib import Path

    import nextrecordable

    page = (Path(nextrecordable.__file__).parent / "static" / "index.html").read_text()
    for needle in ('fetch("/sst-exposure"', "/sst-exposure-pdf", "/score-pdf", "r.data_notice", "s.statement", "drivers_note",
                   'href="/pricing"', '"X-API-Key"'):
        assert needle in page, needle
    assert "innerHTML" in page and "esc(" in page          # server text is escaped before it is inserted


def test_score_pdf_renders_with_notice_and_caveats(world):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from nextrecordable.report import render_pdf, risk_report

    con, result = world["con"], world["result"]
    name, state, z = con.execute(
        "SELECT establishment_name, state, zip5 FROM ita_panel WHERE year = 2024 ORDER BY estab_key LIMIT 1").fetchone()
    client = TestClient(create_app(con, result))
    r = client.post("/score-pdf", json={"name": name, "state": state, "zip_code": z})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    key = con.execute("SELECT estab_key FROM ita_panel ORDER BY estab_key LIMIT 1").fetchone()[0]
    rep = risk_report(con, key, replace(result, rate_served="credibility"))
    rep["data_notice"] = "SYNTHETIC DATA: test"
    raw = render_pdf(rep, compress=False)
    assert b"SYNTHETIC DATA: test" in raw and b"undercount" in raw and b"no model drivers" in raw
    assert render_pdf({"error": "no_data", "message": "x"}).startswith(b"%PDF")
