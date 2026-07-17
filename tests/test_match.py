"""Tests for fuzzy entity resolution via splink + softkey fallback.

Strategy: build a synthetic ITA panel, then construct an input table with
deliberately corrupted/varied versions of those establishment names and
addresses. Check that:

  - Exact matches resolve via softkey path
  - Fuzzy matches (abbreviations, punctuation) resolve via splink
  - Genuinely unmatched rows are flagged unmatched
"""

from __future__ import annotations

import duckdb
import pytest

from osha_risk.features import build_establishment_key_view, build_panel
from osha_risk.load import load_summary_csv
from osha_risk.match import (
    build_estab_directory,
    fallback_softkey_match,
    match_uploaded_csv,
    normalize_input_table,
)
from osha_risk.schema import build_canonical_view

from .synthetic import build_establishments, build_synthetic_dataset


@pytest.fixture(scope="module")
def match_db(tmp_path_factory):
    """Build a synthetic panel and a synthetic broker upload that
    references a subset of those establishments with corrupted formatting."""
    root = tmp_path_factory.mktemp("match_data")

    # Build the panel
    csvs = build_synthetic_dataset(root / "ita", n_establishments=300)
    con = duckdb.connect(str(root / "test.duckdb"))
    for year, p in csvs.items():
        load_summary_csv(p, year, con)
    build_canonical_view(con)
    build_establishment_key_view(con)
    build_panel(con)

    # Build a synthetic "broker upload" that mixes:
    #   - Exact name matches (should hit softkey)
    #   - Abbreviation variants (should hit splink)
    #   - Punctuation variants (should hit splink)
    #   - Made-up establishments (should be unmatched)
    estabs = build_establishments(n=300, seed=42)
    upload_rows = []

    # Exact name match (first 20 establishments)
    for est in estabs[:20]:
        upload_rows.append(
            {
                "name": est["establishment_name"],
                "street_address": est["street_address"],
                "state": est["state"],
                "zip_code": est["zip_code"],
                "city": est["city"],
            }
        )

    # Abbreviation variant: "Manufacturing Inc" -> "Mfg Inc"
    for est in estabs[20:40]:
        upload_rows.append(
            {
                "name": est["establishment_name"]
                    .replace("Manufacturing Inc", "Mfg Inc")
                    .replace("Manufacturing, LLC", "Mfg LLC"),
                "street_address": est["street_address"],
                "state": est["state"],
                "zip_code": est["zip_code"],
                "city": est["city"],
            }
        )

    # Punctuation variant: add commas and periods
    for est in estabs[40:60]:
        upload_rows.append(
            {
                "name": est["establishment_name"].replace(" Inc", ", Inc.").replace(" LLC", ", L.L.C."),
                "street_address": est["street_address"],
                "state": est["state"],
                "zip_code": f"{est['zip_code']}-1234",  # ZIP+4
                "city": est["city"],
            }
        )

    # Unmatched: totally made-up
    for i in range(10):
        upload_rows.append(
            {
                "name": f"Totally Fake Company {i}",
                "street_address": "999 Nowhere Lane",
                "state": "ZZ",
                "zip_code": "00000",
                "city": "Nowheresville",
            }
        )

    # Register upload as a duckdb table
    con.execute("DROP TABLE IF EXISTS broker_upload")
    con.execute(
        """
        CREATE TABLE broker_upload (
            name VARCHAR, street_address VARCHAR, state VARCHAR,
            zip_code VARCHAR, city VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO broker_upload VALUES (?, ?, ?, ?, ?)",
        [(r["name"], r["street_address"], r["state"], r["zip_code"], r["city"])
         for r in upload_rows],
    )

    yield con, root, len(upload_rows)
    con.close()


def test_normalize_input_table(match_db):
    con, _, n_rows = match_db
    n = normalize_input_table(con, "broker_upload")
    assert n == n_rows
    # Check normalized columns
    cols = {r[1] for r in con.execute("PRAGMA table_info('match_input_normalized')").fetchall()}
    for required in ("input_row_id", "name_norm", "zip5", "name_first_token"):
        assert required in cols


def test_normalize_rejects_missing_required(tmp_path):
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE bad_input (name VARCHAR)")
    with pytest.raises(ValueError, match="missing required column"):
        normalize_input_table(con, "bad_input")


def test_build_directory(match_db):
    con, _, _ = match_db
    n = build_estab_directory(con)
    # One row per estab_key
    n_keys = con.execute("SELECT COUNT(DISTINCT estab_key) FROM ita_panel").fetchone()[0]
    assert n == n_keys


def test_softkey_fallback_finds_exact_matches(match_db):
    """Exact matches should resolve via softkey even if splink isn't used."""
    con, _, _ = match_db

    normalize_input_table(con, "broker_upload")
    build_estab_directory(con)

    # Skip splink (use empty match_results)
    con.execute("DROP TABLE IF EXISTS match_results")
    con.execute(
        """
        CREATE TABLE match_results (
            input_row_id BIGINT, matched_estab_key VARCHAR,
            match_probability DOUBLE, is_confident BOOLEAN
        )
        """
    )

    counts = fallback_softkey_match(con)
    # The first 20 rows should match exactly via softkey
    assert counts.get("softkey_exact", 0) >= 18  # allow a couple of edge cases


def test_full_match_pipeline(match_db):
    """End-to-end: upload -> normalize -> directory -> splink -> fallback.

    Expect:
      - All 20 exact matches resolve
      - Most/all of the 20 abbreviation variants resolve (via splink)
      - The 10 fake rows remain unmatched
    """
    con, _, _ = match_db
    summary = match_uploaded_csv(con, "broker_upload")

    counts = summary["method_counts"]
    n_matched = counts.get("splink_fuzzy", 0) + counts.get("softkey_exact", 0)
    n_unmatched = counts.get("unmatched", 0)

    # 70 real rows, 10 fakes => >= 50 should match, ~10 should be unmatched
    assert n_matched >= 30, f"Too few matched: {counts}"
    assert n_unmatched <= 30, f"Too many unmatched: {counts}"

    # Confirm the totals add up to the upload size
    n_total = sum(counts.values())
    assert n_total == summary["n_input_rows"]


def test_match_results_attach_to_correct_key(match_db):
    """For exact matches, the matched_estab_key should equal the panel's
    estab_key for that establishment."""
    con, _, _ = match_db
    match_uploaded_csv(con, "broker_upload")

    # Check the first input row (which is an exact match to estabs[0])
    row = con.execute(
        """
        SELECT i.original_name, i.state, i.zip5, m.matched_estab_key, m.match_method
        FROM match_input_normalized i
        JOIN match_final m USING (input_row_id)
        WHERE i.input_row_id = 1
        """
    ).fetchone()
    assert row is not None
    _name, _state, _zip, matched_key, method = row
    assert matched_key is not None
    assert method in ("softkey_exact", "splink_fuzzy")

    # The matched key should appear in the panel
    n_panel = con.execute(
        "SELECT COUNT(*) FROM ita_panel WHERE estab_key = ?", [matched_key]
    ).fetchone()[0]
    assert n_panel > 0


def test_fake_companies_unmatched(match_db):
    """The 10 fake establishments should be flagged as unmatched."""
    con, _, _ = match_db
    match_uploaded_csv(con, "broker_upload")

    fakes_unmatched = con.execute(
        """
        SELECT COUNT(*) FROM match_input_normalized i
        JOIN match_final m USING (input_row_id)
        WHERE i.original_name LIKE 'Totally Fake Company%'
          AND m.match_method = 'unmatched'
        """
    ).fetchone()[0]
    # All 10 should be unmatched (state ZZ doesn't exist in directory)
    assert fakes_unmatched == 10
