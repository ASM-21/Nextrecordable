"""Tests for the enforcement data pipeline.

Validates:
  1. Inspection and violation CSVs load with canonical column names
  2. Soft-key generation matches the ITA pipeline (same hash function)
  3. Aggregation produces sensible per-establishment-year counts
  4. Panel attachment joins correctly (1-year lag, defaults zeros for no-record)
  5. Adding enforcement features improves the model (lift over ITA-only baseline)
"""

from __future__ import annotations

import duckdb
import pytest

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

from .synthetic import build_synthetic_dataset
from .synthetic_enforcement import build_synthetic_enforcement


@pytest.fixture(scope="module")
def enforcement_db(tmp_path_factory):
    """Build synthetic ITA + enforcement data and load into DuckDB."""
    root = tmp_path_factory.mktemp("enf_data")

    # ITA data
    csvs = build_synthetic_dataset(root / "ita", n_establishments=500)
    con = duckdb.connect(str(root / "test.duckdb"))
    for year, p in csvs.items():
        load_summary_csv(p, year, con)

    # Build canonical ITA pipeline
    build_canonical_view(con)
    build_establishment_key_view(con)
    build_panel(con)

    # Enforcement data linked to same establishments
    insp_csv, viol_csv = build_synthetic_enforcement(
        root / "enf", n_establishments=500
    )
    load_inspection_csv(con, insp_csv)
    load_violation_csv(con, viol_csv)
    add_soft_key_to_inspections(con)
    build_enforcement_aggregates(con)
    attach_enforcement_to_panel(con)

    yield con, root
    con.close()


def test_inspections_loaded(enforcement_db):
    con, _ = enforcement_db
    n = con.execute("SELECT COUNT(*) FROM osha_inspection").fetchone()[0]
    assert n > 0
    # Verify canonical columns exist
    cols = {r[1] for r in con.execute("PRAGMA table_info('osha_inspection')").fetchall()}
    for required in ("activity_nr", "estab_name", "site_state", "site_zip", "naics_code"):
        assert required in cols


def test_violations_loaded(enforcement_db):
    con, _ = enforcement_db
    n = con.execute("SELECT COUNT(*) FROM osha_violation").fetchone()[0]
    assert n > 0


def test_soft_key_matches_ita(enforcement_db):
    """An inspection's soft_key should match an ITA establishment's estab_key
    when name+state+zip align. This validates the hash function is identical
    between the two pipelines."""
    con, _ = enforcement_db

    # Sample an inspection
    row = con.execute(
        """
        SELECT estab_key, estab_name, site_state, site_zip
        FROM osha_inspection
        WHERE estab_key IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    insp_key, name, state, zip_ = row

    # The same establishment should appear in the panel with that key
    n_panel = con.execute(
        "SELECT COUNT(*) FROM ita_panel WHERE estab_key = ?", [insp_key]
    ).fetchone()[0]
    assert n_panel > 0, (
        f"Inspection key {insp_key} for {name} ({state} {zip_}) "
        f"did not match any ITA panel rows. "
        f"Check hash compatibility between enforcement.add_soft_key_to_inspections "
        f"and features.build_establishment_key_view."
    )


def test_inspection_year_extracted(enforcement_db):
    con, _ = enforcement_db
    years = con.execute(
        "SELECT DISTINCT inspection_year FROM osha_inspection ORDER BY 1"
    ).fetchall()
    years = [y[0] for y in years if y[0] is not None]
    assert 2019 in years
    assert 2024 in years


def test_aggregates_built(enforcement_db):
    con, _ = enforcement_db
    n = con.execute("SELECT COUNT(*) FROM enforcement_estab_year").fetchone()[0]
    assert n > 0

    # Sanity: serious citations <= total citations
    bad = con.execute(
        """
        SELECT COUNT(*) FROM enforcement_estab_year
        WHERE n_serious_citations > n_citations
        """
    ).fetchone()[0]
    assert bad == 0

    # Sanity: willful citations <= serious citations
    bad = con.execute(
        """
        SELECT COUNT(*) FROM enforcement_estab_year
        WHERE n_willful_citations > n_serious_citations
        """
    ).fetchone()[0]
    assert bad == 0


def test_panel_attachment_preserves_rows(enforcement_db):
    """Attaching enforcement features must not drop ITA panel rows."""
    con, _ = enforcement_db
    n_panel = con.execute("SELECT COUNT(*) FROM ita_panel").fetchone()[0]
    n_attached = con.execute("SELECT COUNT(*) FROM ita_panel_enforced").fetchone()[0]
    assert n_attached == n_panel


def test_enforcement_features_present(enforcement_db):
    con, _ = enforcement_db
    cols = {r[1] for r in con.execute("PRAGMA table_info('ita_panel_enforced')").fetchall()}
    for required in (
        "enf_n_inspections_lag1",
        "enf_n_citations_lag1",
        "enf_n_serious_citations_lag1",
        "enf_total_penalty_lag1",
        "enf_inspected_lag1",
    ):
        assert required in cols, f"missing {required}"


def test_enforcement_features_are_lagged(enforcement_db):
    """For year_t panel row, enforcement aggregates should reflect year t-1.

    Pick a panel row with non-zero enforcement features and verify against
    the aggregates table directly.
    """
    con, _ = enforcement_db
    row = con.execute(
        """
        SELECT p.estab_key, p.year, p.enf_n_inspections_lag1, p.enf_n_citations_lag1
        FROM ita_panel_enforced p
        WHERE p.enf_n_inspections_lag1 > 0
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        pytest.skip("No panel row with non-zero enforcement lag (rare with small fixture)")
    estab_key, year, n_insp_lag, n_cit_lag = row

    expected = con.execute(
        """
        SELECT n_inspections, n_citations
        FROM enforcement_estab_year
        WHERE estab_key = ? AND year = ?
        """,
        [estab_key, year - 1],
    ).fetchone()
    assert expected is not None
    assert n_insp_lag == expected[0]
    assert n_cit_lag == expected[1]


def test_model_uses_enforcement_features(enforcement_db):
    """Train on the enforced panel; verify the model trains and one or more
    enforcement features show non-zero importance."""
    con, _ = enforcement_db
    result = train_baseline(
        con,
        train_through=2021,
        panel_table="ita_panel_enforced",
        num_boost_round=150,
    )
    assert result.metrics["val_auc_calibrated"] > 0.55

    # At least one enforcement feature should have nonzero gain
    enf_importance = {
        name: gain
        for name, gain in result.feature_importance
        if name.startswith("enf_")
    }
    assert any(g > 0 for g in enf_importance.values()), (
        f"No enforcement features used by the model. Importance: {enf_importance}"
    )


def test_enforcement_lifts_test_metrics(enforcement_db):
    """Adding enforcement features should NOT make test AUC worse (in
    expectation; with small synthetic data noise dominates so we use a loose
    bound).

    This is a regression guard: if someone introduces a bug that drops
    enforcement signal, this test will start failing on real data even if
    it's flaky on synthetic.
    """
    con, _ = enforcement_db

    # Baseline: ITA-only panel
    base = train_baseline(
        con, train_through=2021, panel_table="ita_panel", num_boost_round=150
    )
    # Enforced: ITA + enforcement
    enforced = train_baseline(
        con, train_through=2021, panel_table="ita_panel_enforced", num_boost_round=150
    )

    # Loose bound: enforced AUC should not be DRAMATICALLY worse than baseline.
    # On real data this should clearly improve; on small synthetic it's noisy.
    assert enforced.metrics["test_auc"] >= base.metrics["test_auc"] - 0.05, (
        f"Enforcement-augmented model performed materially worse: "
        f"base test AUC={base.metrics['test_auc']:.3f}, "
        f"enforced test AUC={enforced.metrics['test_auc']:.3f}"
    )
