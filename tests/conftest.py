"""Shared fixtures: one synthetic world, built and trained once per session."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from nextrecordable import pipeline
from nextrecordable.enforcement import load_inspection_csv, load_violation_csv
from nextrecordable.load import load_summary_csv

from .synthetic import build_establishments, build_synthetic_dataset
from .synthetic_enforcement import build_synthetic_enforcement

N_ESTABS = 1200


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Synthetic ITA + DOL files, loaded, built and trained (shared, treat as read-only)."""
    root = tmp_path_factory.mktemp("world")
    con = duckdb.connect(str(root / "world.duckdb"))
    for year, path in build_synthetic_dataset(root / "ita", n_establishments=N_ESTABS).items():
        load_summary_csv(path, year, con)
    insp, viol = build_synthetic_enforcement(root / "dol", n_establishments=N_ESTABS)
    load_inspection_csv(con, insp)
    load_violation_csv(con, viol)
    build_info = pipeline.build(con)
    result = pipeline.train(
        con, model_dir=root / "model", reports_dir=root / "reports", num_boost_round=200,
        gate_kwargs={"min_rows_per_year": 100},
    )
    yield {
        "con": con,
        "root": root,
        "result": result,
        "build": build_info,
        "estabs": build_establishments(N_ESTABS),
    }
    con.close()


def write_ita_csv(path: Path, rows: list[dict]) -> Path:
    """Tiny hand-written 300A file for edge-case tests."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def ita_row(name: str, *, state: str = "IL", zip_code: str = "61602", cases: int = 1,
            hours: int = 200_000, employees: int = 100, **extra) -> dict:
    return {
        "establishment_name": name, "street_address": "5 Mill Rd", "city": "Peoria", "state": state,
        "zip_code": zip_code, "naics_code": "332111", "size": "2",
        "annual_average_employees": employees, "total_hours_worked": hours,
        "total_deaths": 0, "total_dafw_cases": 0, "total_djtr_cases": 0, "total_other_cases": cases,
        **extra,
    }
