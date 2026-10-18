"""FastAPI service: injury-risk lookups for single establishments and CSV books.

Endpoints
  GET  /              lookup page (static HTML, no build step)
  GET  /health        liveness + model summary
  POST /score         one establishment by name + state + ZIP (+ street)
  POST /score-batch   CSV upload (a broker's book), ranked by expected case rate

Concurrency: every request runs on its own DuckDB cursor and keeps its
matching rows in TEMP tables, which are private to that cursor. The shipped
version shared one connection and one set of tables across requests; under
16 concurrent clients it answered with the wrong establishment 18 times in
120. tests/test_api.py now runs concurrent lookups and checks every answer.

Run it:
    nextrecordable serve            (or: make serve)
It reads NEXTRECORDABLE_DB_PATH, NEXTRECORDABLE_MODEL_DIR and
NEXTRECORDABLE_PANEL_TABLE; `create_app(con, model)` is the test seam.
"""

from __future__ import annotations

import csv
import io
import os
import threading
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .match import build_directory, match_rows, suggestions
from .models.baseline import TrainResult, load_model
from .report import _clean, _table_exists, ensure_scores, risk_report, score_keys

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5_000

COLUMN_ALIASES: dict[str, str] = {
    "name": "name", "company": "name", "company_name": "name", "business_name": "name",
    "establishment_name": "name", "employer": "name", "insured": "name", "named_insured": "name",
    "state": "state", "st": "state",
    "zip_code": "zip_code", "zip": "zip_code", "zipcode": "zip_code", "postal_code": "zip_code",
    "street_address": "street_address", "address": "street_address", "street": "street_address",
    "address_1": "street_address", "address1": "street_address",
    "city": "city",
}
REQUIRED_COLUMNS = ("name", "state", "zip_code")


class ScoreRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=300, description="Establishment or company name")
    street_address: str | None = Field(None, max_length=300)
    city: str | None = Field(None, max_length=100)
    state: str = Field(..., min_length=2, max_length=30, description="Two-letter state code")
    zip_code: str = Field(..., min_length=3, max_length=12, description="ZIP or ZIP+4")


def parse_upload(data: bytes) -> list[dict[str, str]]:
    """Parse a CSV upload into row dicts keyed by canonical column names."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV has no header row.")
    header = {}
    for raw in reader.fieldnames:
        key = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
        if key in COLUMN_ALIASES and COLUMN_ALIASES[key] not in header.values():
            header[raw] = COLUMN_ALIASES[key]
    missing = [c for c in REQUIRED_COLUMNS if c not in header.values()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(f"CSV missing required column(s): {missing}. Required: name, state, "
                    "zip_code (zip and company also work). Optional: street_address, city."),
        )
    rows = []
    for i, rec in enumerate(reader, start=1):
        if i > MAX_BATCH_ROWS:
            raise HTTPException(status_code=413, detail=f"Too many rows: the limit is {MAX_BATCH_ROWS:,} per upload.")
        rows.append({canon: (rec.get(raw) or "").strip() for raw, canon in header.items()})
    return rows


def create_app(
    con: duckdb.DuckDBPyConnection,
    model: TrainResult,
    *,
    panel_table: str | None = None,
    scores_table: str = "estab_scores",
    directory: str = "estab_directory",
) -> FastAPI:
    """App bound to an open DuckDB connection and a trained model."""
    panel = panel_table or model.panel_table
    if not _table_exists(con, directory):
        build_directory(con, panel_table=panel, output=directory)
    ensure_scores(con, model, panel_table=panel, scores_table=scores_table)

    app = FastAPI(
        title="nextrecordable",
        description="Establishment-level injury risk from OSHA's public injury and enforcement data.",
        version="0.2.0",
    )
    app.state.con = con
    app.state.model = model
    cursor_lock = threading.Lock()

    def cursor():
        with cursor_lock:
            return con.cursor()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        page = Path(__file__).parent / "static" / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="UI not bundled")
        return HTMLResponse(page.read_text())

    @app.get("/health")
    def health() -> dict[str, Any]:
        tb = model.metrics.get("test_binary", {})
        tr = model.metrics.get("test_rate", {})
        return {
            "status": "ok",
            "panel_table": panel,
            "model_trained_through": model.train_through,
            "model_test_year": model.test_year,
            "test_auc": _clean(tb.get("auc")),
            "test_capture_cases_top10": _clean(tb.get("capture_cases_top10")),
            "test_rate_deviance_skill_vs_credibility": _clean(tr.get("deviance_skill_vs_credibility")),
            "model_test_auc": _clean(tb.get("auc")),  # older clients
        }

    @app.post("/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        cur = cursor()
        try:
            m = match_rows(cur, [req.model_dump()], directory=directory).iloc[0]
            if m["status"] != "matched":
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "ambiguous" if m["status"] == "ambiguous" else "no_match",
                        "message": (
                            "More than one establishment matches closely; add a street address."
                            if m["status"] == "ambiguous" else
                            f"No confident match for '{req.name}' in {req.state} {req.zip_code}. It may "
                            "not file with OSHA (most workplaces under 20 employees don't), or the "
                            "name on file may differ."
                        ),
                        "candidates": suggestions(cur, int(m["rec_id"]), directory=directory),
                    },
                )
            report = risk_report(cur, m["estab_key"], model, panel_table=panel, scores_table=scores_table)
            report["match_info"] = {
                "match_status": m["status"],
                "match_probability": _clean(m["match_probability"]),
                "query": req.model_dump(),
            }
            return report
        finally:
            cur.close()

    @app.post("/score-batch")
    def score_batch(file: UploadFile = File(...)) -> dict[str, Any]:
        # Sync on purpose: FastAPI runs it in a worker thread, so a large book
        # doesn't block the event loop while DuckDB works.
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a .csv")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large: the limit is {MAX_UPLOAD_BYTES // 1_048_576} MB.")
        rows = parse_upload(data)
        if not rows:
            raise HTTPException(status_code=400, detail="CSV has no data rows.")
        cur = cursor()
        try:
            matches = match_rows(cur, rows, directory=directory)
            matched = matches[matches["status"] == "matched"]
            scored = score_keys(cur, matched["estab_key"].tolist(), panel_table=panel, scores_table=scores_table)
        finally:
            cur.close()
        by_key = {r["estab_key"]: r for r in scored.to_dict("records")}
        results = []
        for m in matched.to_dict("records"):
            s = by_key.get(m["estab_key"])
            if s is None:
                continue
            results.append({
                "input_row": int(m["rec_id"]),
                "input_name": m["original_name"],
                "match_probability": _clean(m["match_probability"]),
                **{k: _clean(v) for k, v in s.items()},
                "risk_score_next_year": _clean(s.get("p_any_next")),  # older clients
            })
        results.sort(key=lambda r: (r["expected_tcr_next"] is None, -(r["expected_tcr_next"] or 0.0),
                                    -(r["p_any_next"] or 0.0)))
        unmatched = [
            {"input_row": int(m["rec_id"]), "input_name": m["original_name"], "reason": m["status"]}
            for m in matches[matches["status"] != "matched"].to_dict("records")
        ]
        counts = matches["status"].value_counts().to_dict()
        return {
            "summary": {
                "n_input_rows": len(rows),
                "n_matched": len(results),
                "n_unmatched": len(unmatched),
                "status_counts": {k: int(v) for k, v in counts.items()},
                "ranked_by": "expected_tcr_next (size-adjusted expected case rate), then p_any_next",
            },
            "results": results,
            "unmatched": unmatched,
        }

    return app


def build_app_from_env() -> FastAPI:
    """Production factory: open the DB read-write, load the saved model."""
    from .config import DB_PATH, MODEL_DIR

    db_path = Path(os.environ.get("NEXTRECORDABLE_DB_PATH", str(DB_PATH)))
    model_dir = Path(os.environ.get("NEXTRECORDABLE_MODEL_DIR", str(MODEL_DIR)))
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}. Run `nextrecordable run-all` first.")
    con = duckdb.connect(str(db_path))
    model = load_model(model_dir)
    panel = os.environ.get("NEXTRECORDABLE_PANEL_TABLE") or model.panel_table
    return create_app(con, model, panel_table=panel)
