"""End-to-end pipeline test using synthetic OSHA-shaped CSVs.

This validates that:
  1. Per-year CSVs load into DuckDB even with drifting schemas
  2. The canonical view resolves columns correctly
  3. The panel builds with sensible lag and target columns
  4. Base rates and counts are within sanity bounds
"""

from __future__ import annotations

import duckdb
import pytest

from osha_risk.features import build_establishment_key_view, build_panel, panel_summary
from osha_risk.load import load_summary_csv
from osha_risk.schema import build_canonical_view, resolve_column

from .synthetic import build_synthetic_dataset


@pytest.fixture
def loaded_db(tmp_path):
    """Build synthetic CSVs and load them into a fresh DuckDB."""
    csv_root = tmp_path / "csvs"
    paths = build_synthetic_dataset(csv_root, n_establishments=200)

    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    for year, csv_path in paths.items():
        load_summary_csv(csv_path, year, con)

    yield con
    con.close()


def test_yearly_tables_loaded(loaded_db):
    """Each year creates its own table with the expected row count."""
    for year in (2019, 2020, 2023, 2024):
        n = loaded_db.execute(f"SELECT COUNT(*) FROM ita_summary_{year}").fetchone()[0]
        assert n == 200, f"year {year} expected 200 rows, got {n}"


def test_schema_drift_handled(loaded_db):
    """Legacy years use `naics`, current years use `naics_code`."""
    legacy_col = resolve_column(loaded_db, "ita_summary_2019", ("naics_code", "naics"))
    current_col = resolve_column(loaded_db, "ita_summary_2024", ("naics_code", "naics"))
    assert legacy_col == "naics"
    assert current_col == "naics_code"


def test_canonical_view_unifies_schema(loaded_db):
    """The canonical view exposes naics_code from BOTH legacy and current years."""
    n_tables = build_canonical_view(loaded_db)
    assert n_tables == 6

    # Total rows across all years
    total = loaded_db.execute("SELECT COUNT(*) FROM ita_canonical").fetchone()[0]
    assert total == 6 * 200

    # Every year should have non-null naics_code in the canonical view
    rows = loaded_db.execute(
        """
        SELECT data_year, COUNT(*) FILTER (WHERE naics_code IS NOT NULL) AS n_naics
        FROM ita_canonical
        GROUP BY data_year
        ORDER BY data_year
        """
    ).fetchall()
    for year, n_naics in rows:
        assert n_naics == 200, f"year {year}: expected 200 non-null naics, got {n_naics}"


def test_panel_has_lags_and_targets(loaded_db):
    """Panel should have lag features for years >= year_2, targets for years <= max-1."""
    build_canonical_view(loaded_db)
    build_establishment_key_view(loaded_db)
    n = build_panel(loaded_db)
    assert n == 6 * 200

    summary = panel_summary(loaded_db)
    # Years 2020 to 2024 have a lag1 (5 years * 200 estabs)
    assert summary["rows_with_lag"] == 5 * 200
    # Years 2019 to 2023 have a next-year target (5 * 200)
    assert summary["rows_with_target"] == 5 * 200


def test_base_rate_is_sensible(loaded_db):
    """The fraction of (estab, year) with any recordable next year should be high
    but not 100 percent, since synthetic data has a long zero tail for small plants."""
    build_canonical_view(loaded_db)
    build_establishment_key_view(loaded_db)
    build_panel(loaded_db)
    s = panel_summary(loaded_db)
    base_rate = s["target_base_rate"]
    # Synthetic data is generally injury-positive, so expect high but bounded
    assert 0.3 <= base_rate <= 1.0, f"base rate {base_rate} outside sanity bounds"


def test_tcr_computed_correctly(loaded_db):
    """TCR should equal (n_recordable * 200000) / hours_worked."""
    build_canonical_view(loaded_db)
    build_establishment_key_view(loaded_db)
    build_panel(loaded_db)

    # Pick a row with positive hours and recordables
    row = loaded_db.execute(
        """
        SELECT n_recordable, hours_worked, tcr
        FROM ita_panel
        WHERE hours_worked > 0 AND n_recordable > 0
        LIMIT 1
        """
    ).fetchone()
    assert row is not None
    n_rec, hours, tcr = row
    expected = (n_rec * 200_000) / hours
    assert tcr == pytest.approx(expected, rel=1e-9)


def test_industry_benchmark_present(loaded_db):
    """Each panel row should have a NAICS-4 median TCR (or NULL if no peers)."""
    build_canonical_view(loaded_db)
    build_establishment_key_view(loaded_db)
    build_panel(loaded_db)

    # Most rows in our synthetic data should have a benchmark since NAICS_POOL is small
    n_with_benchmark = loaded_db.execute(
        "SELECT COUNT(*) FROM ita_panel WHERE naics4_tcr_median IS NOT NULL"
    ).fetchone()[0]
    assert n_with_benchmark > 1000  # most of 1200 rows


def test_lag_actually_lags(loaded_db):
    """tcr_lag1 in year_t equals tcr in year_t-1 for the same establishment."""
    build_canonical_view(loaded_db)
    build_establishment_key_view(loaded_db)
    build_panel(loaded_db)

    row = loaded_db.execute(
        """
        SELECT
            p1.estab_key, p1.year, p1.tcr_lag1, p0.tcr
        FROM ita_panel p1
        JOIN ita_panel p0
          ON p0.estab_key = p1.estab_key
         AND p0.year = p1.year - 1
        WHERE p1.tcr_lag1 IS NOT NULL AND p0.tcr IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    assert row is not None
    _, _, lag1, prior_tcr = row
    assert lag1 == pytest.approx(prior_tcr, rel=1e-9)
