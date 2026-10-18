"""Models: reproducible scoring, honest metrics, persisted artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from nextrecordable.models.baseline import (
    CATEGORICAL_FEATURES,
    IsotonicMap,
    Vocab,
    calibration_table,
    capture_rate,
    fetch_frame,
    lift_at_top_k,
    load_model,
    poisson_deviance,
    risk_band,
)


def test_split_and_metrics_are_present(world):
    r = world["result"]
    assert (r.train_through, r.val_year, r.test_year) == (2021, 2022, 2023)
    tb, tr = r.metrics["test_binary"], r.metrics["test_rate"]
    for k in ("auc", "baseline_auc", "capture_cases_top10", "baseline_capture_cases_top10",
              "lift_top10", "lift_ceiling", "brier"):
        assert k in tb
    assert tb["lift_top10"] <= tb["lift_ceiling"] + 1e-9
    for k in ("model", "credibility", "industry", "deviance_skill_vs_credibility",
              "largest_sites_capture_cases_top10"):
        assert k in tr


def test_models_beat_their_baselines_on_synthetic_data(world):
    """Synthetic data has learnable structure; if these fail, training is broken."""
    tb = world["result"].metrics["test_binary"]
    tr = world["result"].metrics["test_rate"]
    assert tb["auc"] > 0.7
    assert tr["deviance_skill_vs_industry"] > 0.3
    assert tr["model"]["spearman_vs_realized_tcr"] > tr["industry"]["spearman_vs_realized_tcr"]
    assert tr["model"]["capture_cases_top10"] > tr["largest_sites_capture_cases_top10"] - 0.05


def test_rate_model_is_calibrated_in_aggregate(world):
    ratio = world["result"].metrics["test_rate"]["model"]["predicted_over_actual_cases"]
    assert 0.9 < ratio < 1.1


def test_scores_identical_in_a_fresh_process(world, tmp_path):
    """Regression: categorical codes came from hash(), salted per process; serving
    from a new process changed the score of ~half of all establishments."""
    con, root = world["con"], world["root"]
    frame = fetch_frame(con, world["result"].panel_table, "year = 2023 ORDER BY estab_key LIMIT 300")
    here = world["result"].predict(frame)
    frame.to_pickle(tmp_path / "frame.pkl")
    script = textwrap.dedent(f"""
        import json, pandas as pd
        from nextrecordable.models.baseline import load_model
        m = load_model(r"{root / 'model'}")
        p = m.predict(pd.read_pickle(r"{tmp_path / 'frame.pkl'}"))
        print(json.dumps({{"p": p["p"].tolist(), "rate": p["rate_per_hour"].tolist()}}))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"})
    there = json.loads(out.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(here["p"], there["p"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(here["rate_per_hour"], there["rate"], rtol=1e-12)


def test_saved_model_reloads_with_same_predictions(world):
    m = load_model(world["root"] / "model")
    frame = fetch_frame(world["con"], m.panel_table, "year = 2022 LIMIT 200")
    a, b = world["result"].predict(frame), m.predict(frame)
    np.testing.assert_allclose(a["p"], b["p"])
    np.testing.assert_allclose(a["rate_per_hour"], b["rate_per_hour"])
    assert not list((world["root"] / "model").glob("*.pkl"))


def test_vocab_unknown_and_missing_encode_as_missing():
    v = Vocab.fit(pd.DataFrame({c: ["a", "b", None] for c in CATEGORICAL_FEATURES}), CATEGORICAL_FEATURES)
    enc = v.encode(pd.DataFrame({c: ["b", "zzz", None] for c in CATEGORICAL_FEATURES}))
    assert enc["state"].tolist() == [1, -1, -1]
    assert Vocab.from_json(json.loads(json.dumps(v.to_json()))).mapping == v.mapping


def test_isotonic_map_matches_sklearn_inside_clip_range():
    rng = np.random.default_rng(0)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw).astype(int)
    m = IsotonicMap.fit(raw, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(raw, y)
    grid = np.linspace(-0.5, 1.5, 101)
    np.testing.assert_allclose(m.transform(grid), iso.predict(grid))
    assert m.transform(grid).max() <= 0.995


def test_metric_helpers():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.arange(10)[::-1].astype(float)
    assert lift_at_top_k(y, s, k=0.1) == pytest.approx(1 / 0.3)
    assert capture_rate(s, np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 5]), k=0.1) == pytest.approx(0.5)
    assert poisson_deviance(np.array([0.0, 2.0]), np.array([0.0 + 1e-12, 2.0])) == pytest.approx(0.0, abs=1e-9)
    assert calibration_table(np.array([]), np.array([])) == []
    assert risk_band(0.95) == "much higher than peers"
    assert risk_band(0.1) == "much lower than peers"
    assert risk_band(None) is None


def test_materialized_scores(world):
    con = world["con"]
    n_keys = con.execute(f"SELECT COUNT(DISTINCT estab_key) FROM {world['result'].panel_table}").fetchone()[0]
    rows = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT estab_key), min(peer_pctile), max(peer_pctile), "
        "min(p_any_next), max(p_any_next), COUNT(*) FILTER (WHERE risk_band IS NULL) FROM estab_scores"
    ).fetchone()
    assert rows[0] == rows[1] == n_keys
    assert 0 < rows[2] <= rows[3] <= 1
    assert 0 < rows[4] <= rows[5] < 1
    assert rows[6] == 0
