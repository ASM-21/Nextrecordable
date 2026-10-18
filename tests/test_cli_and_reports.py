"""End-to-end through the CLI, plus the sanity gates and the metrics report."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from osha_risk.sanity import run_gates, summarize

from .synthetic import build_synthetic_dataset
from .synthetic_enforcement import build_synthetic_enforcement


def test_gates_pass_on_the_synthetic_world(world):
    gates = run_gates(world["con"], min_rows_per_year=100)
    summary = summarize(gates)
    assert summary["ok"], [g for g in summary["gates"] if g["status"] == "fail"]
    names = {g.name for g in gates}
    assert {"leading_zero_zips_kept", "duplicate_key_years", "size_code_harmonized",
            "inspection_link_share"} <= names


def test_gate_catches_stripped_leading_zeros(world, tmp_path):
    import duckdb

    # A keyed table whose ZIPs bypassed normalize.zip5 (as the old loader did).
    con = duckdb.connect()
    con.execute(
        """CREATE TABLE ita_panel AS SELECT * FROM (VALUES (2022, 1, '2', 1, NULL::DOUBLE))
           t(year, n_raw_rows, size_code, target_any_recordable_next, tcr_lag1)"""
    )
    con.execute(
        """CREATE TABLE ita_keyed AS SELECT * FROM (VALUES
             (2022, 'CT', '6511', 100.0, 200000.0, NULL::INTEGER, 2022, NULL::VARCHAR, 'k1'),
             (2022, 'CT', '6512', 100.0, 200000.0, NULL::INTEGER, 2022, NULL::VARCHAR, 'k2'))
           t(data_year, state_norm, zip5, annual_average_employees, total_hours_worked,
             year_filing_for, file_year, establishment_id, estab_key)"""
    )
    by_name = {g.name: g for g in run_gates(con)}
    assert by_name["leading_zero_zips_kept"].status == "fail"


def test_metrics_report_is_written(world):
    md = (world["root"] / "reports" / "METRICS.md").read_text()
    assert "size + NAICS baseline" in md and "credibility baseline" in md and "Sanity gates" in md
    payload = json.loads((world["root"] / "reports" / "metrics.json").read_text())
    assert payload["split"] == {"train_through": 2021, "val_year": 2022, "test_year": 2023}
    assert payload["metrics"]["test_binary"]["n"] > 0


def test_cli_build_sanity_train(tmp_path, monkeypatch):
    from osha_risk import cli, config

    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    db = tmp_path / "cli.duckdb"
    runner = CliRunner()

    import duckdb

    from osha_risk.load import load_summary_csv

    con = duckdb.connect(str(db))
    for year, path in build_synthetic_dataset(tmp_path / "ita", n_establishments=400).items():
        load_summary_csv(path, year, con)
    con.close()
    insp, viol = build_synthetic_enforcement(tmp_path / "dol", n_establishments=400)

    def run(*args):
        result = runner.invoke(cli.app, ["--db", str(db), *args])
        assert result.exit_code == 0, result.output
        return result.output

    run("load-enforcement", "--inspection", str(insp), "--violation", str(viol))
    assert "enforced_panel_rows" in run("build")
    run("sanity", "--min-rows-per-year", "100")
    run("train", "--rounds", "60")
    assert (tmp_path / "model" / "binary.txt").exists()
    assert (tmp_path / "reports" / "METRICS.md").exists()
    assert "estab_key" in run("query", "SELECT estab_key FROM estab_scores LIMIT 1")
