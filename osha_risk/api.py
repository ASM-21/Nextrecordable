"""FastAPI service: score establishments and CSV batches for injury risk.

Two endpoints:
  POST /score         - single establishment by name + address
  POST /score-batch    - CSV upload, returns a ranked risk list
  GET  /health         - liveness + model metadata

Design:
  `create_app(con, model, panel_table)` is the testable factory. Tests build
  a synthetic DuckDB + train a small model and pass them in directly - no
  network, no env vars, no real OSHA data required to test the API surface.

  For production, set OSHA_RISK_SERVE=1 and run:
      OSHA_RISK_SERVE=1 uvicorn osha_risk.api:app --host 0.0.0.0 --port 8000
  This reads OSHA_RISK_DB_PATH / OSHA_RISK_MODEL_DIR / OSHA_RISK_PANEL_TABLE
  from the environment, connects to the real DuckDB file, loads the saved
  model, and builds the app. The env var guard means importing this module
  during tests (via `from osha_risk.api import create_app`) never requires
  real data on disk.

Known limitation: the entity-resolution pipeline (match.py) creates shared
temp tables on the DuckDB connection (match_input_normalized, match_final,
etc.) and is NOT safe for concurrent requests against the same connection.
All scoring requests are serialized behind a single lock. Fine for a demo or
low-traffic broker tool; would need a connection-per-request or read-replica
setup before handling real concurrent load. See CONTRIBUTING.md.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .match import match_uploaded_csv
from .models.baseline import TrainResult, load_model
from .report import risk_report


class ScoreRequest(BaseModel):
    name: str = Field(..., description="Establishment / company name")
    street_address: str | None = None
    city: str | None = None
    state: str = Field(..., description="Two-letter state code")
    zip_code: str = Field(..., description="ZIP or ZIP+4")


class ScoreBatchSummary(BaseModel):
    n_input_rows: int
    n_matched: int
    n_unmatched: int
    method_counts: dict[str, int]


def create_app(
    con: duckdb.DuckDBPyConnection,
    model: TrainResult,
    *,
    panel_table: str = "ita_panel",
) -> FastAPI:
    """Build a FastAPI app bound to an existing DuckDB connection + trained model.

    This is the seam tests use: pass a synthetic in-memory connection and a
    model trained on synthetic data to get a fully working app with no real
    data or network access required.
    """
    lock = threading.Lock()

    app = FastAPI(
        title="osha-risk API",
        description="Establishment-level workplace injury risk scoring from OSHA public data.",
        version="0.1.0",
    )
    app.state.con = con
    app.state.model = model
    app.state.panel_table = panel_table
    app.state.lock = lock

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "panel_table": panel_table,
            "model_trained_through": model.train_through,
            "model_test_auc": model.metrics.get("test_auc"),
        }

    @app.post("/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        with lock:
            con.execute("DROP TABLE IF EXISTS _single_score_input")
            con.execute(
                """
                CREATE TABLE _single_score_input (
                    name VARCHAR, street_address VARCHAR, city VARCHAR,
                    state VARCHAR, zip_code VARCHAR
                )
                """
            )
            con.execute(
                "INSERT INTO _single_score_input VALUES (?, ?, ?, ?, ?)",
                [req.name, req.street_address, req.city, req.state, req.zip_code],
            )

            match_uploaded_csv(con, "_single_score_input", panel_table=panel_table)

            match_row = con.execute(
                "SELECT matched_estab_key, match_probability, match_method "
                "FROM match_final WHERE input_row_id = 1"
            ).fetchone()

        if match_row is None or match_row[0] is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "no_match",
                    "message": (
                        f"No establishment found matching '{req.name}' in "
                        f"{req.state} {req.zip_code}. It may not be in OSHA's "
                        f"reporting scope (establishments under 20 employees in "
                        f"most industries are not covered), or the name/address "
                        f"may differ too much from OSHA's records."
                    ),
                },
            )

        matched_key, match_prob, match_method = match_row
        report = risk_report(con, matched_key, model, panel_table=panel_table)
        report["match_info"] = {
            "match_method": match_method,
            "match_probability": match_prob,
            "query": req.model_dump(),
        }
        return report

    @app.post("/score-batch")
    async def score_batch(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a .csv")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            with lock:
                con.execute("DROP TABLE IF EXISTS _batch_score_input")
                con.execute(
                    f"""
                    CREATE TABLE _batch_score_input AS
                    SELECT * FROM read_csv_auto('{tmp_path.as_posix()}', header=true)
                    """
                )

                cols = {
                    r[1].lower()
                    for r in con.execute(
                        "PRAGMA table_info('_batch_score_input')"
                    ).fetchall()
                }
                required = {"name", "state", "zip_code"}
                missing = required - cols
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"CSV missing required column(s): {sorted(missing)}. "
                            f"Required: name, state, zip_code. "
                            f"Optional: street_address, city."
                        ),
                    )

                match_summary = match_uploaded_csv(
                    con, "_batch_score_input", panel_table=panel_table
                )

                matched_rows = con.execute(
                    """
                    SELECT i.input_row_id, i.original_name, m.matched_estab_key,
                           m.match_probability, m.match_method
                    FROM match_input_normalized i
                    JOIN match_final m USING (input_row_id)
                    WHERE m.matched_estab_key IS NOT NULL
                    ORDER BY i.input_row_id
                    """
                ).fetchall()

                unmatched_rows = con.execute(
                    """
                    SELECT i.input_row_id, i.original_name
                    FROM match_input_normalized i
                    JOIN match_final m USING (input_row_id)
                    WHERE m.matched_estab_key IS NULL
                    ORDER BY i.input_row_id
                    """
                ).fetchall()

                results: list[dict[str, Any]] = []
                for row_id, orig_name, estab_key, match_prob, match_method in matched_rows:
                    rpt = risk_report(con, estab_key, model, panel_table=panel_table)
                    if rpt.get("error"):
                        continue
                    results.append(
                        {
                            "input_row_id": row_id,
                            "input_name": orig_name,
                            "match_method": match_method,
                            "match_probability": match_prob,
                            "estab_key": estab_key,
                            "establishment_name": rpt["establishment_name"],
                            "risk_score_next_year": rpt["risk_score_next_year"],
                            "confidence": rpt["confidence"],
                            "current_tcr": rpt["current_year_metrics"]["tcr"],
                            "industry_percentile": rpt["industry_percentile"],
                            "top_driver": (
                                rpt["risk_drivers"][0]["feature"]
                                if rpt["risk_drivers"]
                                else None
                            ),
                        }
                    )

                results.sort(key=lambda r: r["risk_score_next_year"] or 0, reverse=True)

                unmatched = [
                    {"input_row_id": row_id, "input_name": name}
                    for row_id, name in unmatched_rows
                ]
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "summary": ScoreBatchSummary(
                n_input_rows=match_summary["n_input_rows"],
                n_matched=len(results),
                n_unmatched=len(unmatched),
                method_counts=match_summary["method_counts"],
            ).model_dump(),
            "results": results,
            "unmatched": unmatched,
        }

    return app


def _build_production_app() -> FastAPI:
    """Read env vars, connect to real DuckDB + model, build the app.

    Used only by the `uvicorn osha_risk.api:app` entrypoint (guarded by
    OSHA_RISK_SERVE=1 below), never by tests.
    """
    from .config import DB_PATH

    db_path = Path(os.environ.get("OSHA_RISK_DB_PATH", str(DB_PATH)))
    model_dir = Path(os.environ.get("OSHA_RISK_MODEL_DIR", "data/model"))
    panel_table = os.environ.get("OSHA_RISK_PANEL_TABLE", "ita_panel_enforced")

    if not db_path.exists():
        raise FileNotFoundError(
            f"No DuckDB database at {db_path}. Run `make download && make load` "
            f"first, or set OSHA_RISK_DB_PATH."
        )

    con = duckdb.connect(str(db_path), read_only=False)

    existing_tables = {
        r[0]
        for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    if panel_table not in existing_tables:
        panel_table = "ita_panel"
        if panel_table not in existing_tables:
            raise FileNotFoundError(
                f"Neither ita_panel_enforced nor ita_panel exist in {db_path}. "
                f"Run the panel-building pipeline first (see CONTRIBUTING.md)."
            )

    model = load_model(model_dir)
    return create_app(con, model, panel_table=panel_table)


# Module-level `app`, built only when explicitly requested via env var. This
# lets `from osha_risk.api import create_app` (what tests do) import this
# module without requiring a real database or trained model on disk.
if os.environ.get("OSHA_RISK_SERVE") == "1":
    app = _build_production_app()
