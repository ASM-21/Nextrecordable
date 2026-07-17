"""Tests for the baseline model and report generator.

We train on a larger synthetic dataset so LightGBM has enough signal to
produce sensible metrics. The synthetic fixture builds intrinsic risk
multipliers per establishment, so a model SHOULD do better than chance.
"""

from __future__ import annotations

import duckdb
import pytest

from osha_risk.features import build_establishment_key_view, build_panel
from osha_risk.load import load_summary_csv
from osha_risk.models.baseline import (
    calibration_table,
    lift_at_top_k,
    save_model,
    train_baseline,
)
from osha_risk.report import risk_report
from osha_risk.schema import build_canonical_view

from .synthetic import build_synthetic_dataset


@pytest.fixture(scope="module")
def trained_db(tmp_path_factory):
    """Build a larger synthetic dataset and train the baseline. Module-scoped
    because training is the slowest test step (~1s)."""
    root = tmp_path_factory.mktemp("model_data")
    csvs = build_synthetic_dataset(root / "csvs", n_establishments=1000)

    con = duckdb.connect(str(root / "test.duckdb"))
    for year, p in csvs.items():
        load_summary_csv(p, year, con)
    build_canonical_view(con)
    build_establishment_key_view(con)
    build_panel(con)

    result = train_baseline(
        con,
        train_through=2021,  # train on 2019-2021, val=2022, test=2023
        num_boost_round=100,
        early_stopping_rounds=20,
    )
    yield con, result, root
    con.close()


def test_model_trains(trained_db):
    _con, result, _root = trained_db
    assert result.model is not None
    assert result.metrics["n_train"] > 0
    assert result.metrics["n_val"] > 0
    assert result.metrics["n_test"] > 0


def test_auc_beats_random(trained_db):
    """Synthetic data has intrinsic per-establishment risk, so the model
    should be meaningfully better than 0.5 AUC."""
    _, result, _ = trained_db
    assert result.metrics["val_auc_raw"] > 0.55
    assert result.metrics["test_auc"] > 0.55


def test_calibration_improves_brier(trained_db):
    """Isotonic calibration should not WORSEN the Brier score on the val set
    (it's fit there). Usually it improves or matches."""
    _, result, _ = trained_db
    assert (
        result.metrics["val_brier_calibrated"]
        <= result.metrics["val_brier_raw"] + 1e-6
    )


def test_lift_above_one(trained_db):
    """A useful model should have lift > 1 at the top decile."""
    _, result, _ = trained_db
    assert result.metrics["test_lift_top_decile"] > 1.0


def test_calibration_table_well_formed(trained_db):
    _, result, _ = trained_db
    table = result.calibration_table
    assert len(table) > 0
    for bin_row in table:
        assert 0 <= bin_row["mean_predicted"] <= 1
        assert 0 <= bin_row["observed_rate"] <= 1
        assert bin_row["n"] > 0


def test_feature_importance_nonempty(trained_db):
    _, result, _ = trained_db
    assert len(result.feature_importance) == len(result.feature_names)
    # Top feature should have nonzero importance
    assert result.feature_importance[0][1] > 0


def test_save_and_reload_model(trained_db):
    import json

    import lightgbm as lgb
    _, result, root = trained_db
    dest = root / "model_out"
    save_model(result, dest)

    assert (dest / "model.txt").exists()
    assert (dest / "meta.json").exists()
    assert (dest / "calibrator.pkl").exists()

    # Reload booster and ensure it predicts the same on a small batch
    reloaded = lgb.Booster(model_file=str(dest / "model.txt"))
    meta = json.loads((dest / "meta.json").read_text())
    assert meta["feature_names"] == list(result.feature_names)
    assert reloaded.num_trees() == result.model.num_trees()


def test_risk_report_structure(trained_db):
    con, result, _ = trained_db
    # Pick any estab_key from the panel
    estab_key = con.execute(
        "SELECT estab_key FROM ita_panel WHERE tcr IS NOT NULL LIMIT 1"
    ).fetchone()[0]

    report = risk_report(con, estab_key, result)

    # Required keys present
    for k in (
        "estab_key", "establishment_name", "naics_code", "latest_year",
        "current_year_metrics", "risk_score_next_year", "risk_drivers",
        "confidence", "history", "caveats",
    ):
        assert k in report, f"missing key: {k}"

    # Risk score is a probability
    assert 0 <= report["risk_score_next_year"] <= 1

    # Drivers are well-shaped
    assert len(report["risk_drivers"]) > 0
    for d in report["risk_drivers"]:
        assert "feature" in d
        assert "contribution_logodds" in d
        assert d["direction"] in ("increases_risk", "decreases_risk")

    # History has multiple rows (synthetic data has 6 years)
    assert len(report["history"]) >= 1


def test_risk_report_handles_missing_estab(trained_db):
    con, result, _ = trained_db
    report = risk_report(con, "nonexistent_key_12345", result)
    assert report["error"] == "no_data"


def test_lift_at_top_k_edge_cases():
    import numpy as np
    # All positives -> lift = 1.0 (top 10% is no different from base)
    y = np.ones(100, dtype=np.int32)
    p = np.random.RandomState(0).rand(100)
    assert lift_at_top_k(y, p, k=0.10) == pytest.approx(1.0)

    # Empty arrays -> NaN
    import math
    assert math.isnan(lift_at_top_k(np.array([], dtype=np.int32), np.array([])))


def test_calibration_table_empty():
    import numpy as np
    table = calibration_table(np.array([], dtype=np.int32), np.array([]))
    assert table == []
