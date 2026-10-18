"""Matcher units: input normalization, statuses, ambiguity, suggestions."""

from __future__ import annotations

import duckdb
import pytest

from nextrecordable.keys import python_key
from nextrecordable.match import link, load_left, match_rows, normalize_inputs, suggestions
from nextrecordable.normalize import name_first_token, name_numbers, normalize_address, normalize_name


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(
        """CREATE TABLE estab_directory (estab_key VARCHAR, establishment_name VARCHAR,
           street_address VARCHAR, city VARCHAR, state VARCHAR, zip5 VARCHAR, name_norm VARCHAR,
           name_first VARCHAR, name_nums VARCHAR, address_norm VARCHAR, naics_code VARCHAR,
           latest_year INTEGER)"""
    )
    for name, street, z in [
        ("Acme Forge Inc", "5 Mill Rd", "61602"),
        ("Acme Forge Plant 2", "9 River Rd", "61602"),
        ("Acme Forge Plant 3", "11 River Rd", "61602"),
        ("Summit Castings", "1 Quarry Ln", "61604"),
        ("Summit Castings Co", "77 Harbor Ave", "61605"),
    ]:
        n = normalize_name(name)
        c.execute(
            "INSERT INTO estab_directory VALUES (?, ?, ?, 'Peoria', 'IL', ?, ?, ?, ?, ?, '332111', 2024)",
            [python_key(name, "IL", z), name, street, z, n, name_first_token(n), name_numbers(n),
             normalize_address(street)],
        )
    return c


def test_normalize_inputs_columns():
    f = normalize_inputs([{"name": "ACME FORGE, INC.", "state": "il", "zip_code": 61602}])
    row = f.iloc[0]
    assert (row["rec_id"], row["name_norm"], row["state"], row["zip5"]) == (1, "acme forge", "IL", "61602")


def test_statuses(con):
    res = match_rows(con, [
        {"name": "Acme Forge", "state": "IL", "zip_code": "61602"},              # exact
        {"name": "Acme Forge Plant 4", "state": "IL", "zip_code": "61602"},      # number disagrees
        {"name": "Summit Castings", "state": "IL", "zip_code": "61600"},         # two equal candidates, no ZIP
        {"name": "Nothing Like It", "state": "IL", "zip_code": "61602"},
    ])
    assert res["status"].tolist() == ["matched", "unmatched", "unmatched", "unmatched"]
    assert res.iloc[0]["estab_key"] == python_key("Acme Forge Inc", "IL", "61602")


def test_ambiguous_when_two_candidates_are_equally_good(con):
    # same name in two nearby ZIPs, street given for neither: tie -> ambiguous, not a guess
    con.execute("UPDATE estab_directory SET zip5 = '61604' WHERE establishment_name = 'Summit Castings Co'")
    res = match_rows(con, [{"name": "Summit Castings", "state": "IL", "zip_code": "61604"}])
    assert res.iloc[0]["status"] == "ambiguous"


def test_street_breaks_the_tie(con):
    con.execute("UPDATE estab_directory SET zip5 = '61604' WHERE establishment_name = 'Summit Castings Co'")
    res = match_rows(con, [{"name": "Summit Castings", "state": "IL", "zip_code": "61604",
                            "street_address": "77 Harbor Avenue"}])
    assert res.iloc[0]["status"] == "matched"
    assert res.iloc[0]["estab_key"] == python_key("Summit Castings Co", "IL", "61605")


def test_suggestions_rank_candidates(con):
    frame = normalize_inputs([{"name": "Acme Forg Plant", "state": "IL", "zip_code": "61602"}])
    load_left(con, frame)
    link(con)
    s = suggestions(con, 1)
    assert 1 <= len(s) <= 3
    assert all(x["establishment_name"].startswith("Acme") for x in s)
    probs = [x["match_probability"] for x in s]
    assert probs == sorted(probs, reverse=True)
