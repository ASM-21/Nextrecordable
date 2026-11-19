"""FastAPI service: injury-risk lookups for single establishments and CSV books.

Endpoints
  GET  /              lookup page (static HTML, no build step)
  GET  /health        liveness + model summary
  POST /score         one establishment by name + state + ZIP (+ street)
  POST /score-batch   CSV upload (a broker's book), ranked by expected case rate
  POST /sst-exposure      position against OSHA's Site-Specific Targeting criteria (JSON)
  POST /sst-exposure-pdf  the same report as a PDF
  POST /sst-batch         CSV upload of many establishments -> SST positions (JSON, or CSV with ?format=csv)

Accounts (when the app is built with an AccountStore, as in production):
  GET  /pricing           tiers from data/tiers.json, Stripe Payment Links from env
  GET  /account           tier and this month's usage for the X-API-Key
  POST /account/recover   email a one-hour link that replaces a lost key
  POST /stripe/webhook    creates accounts and emails keys (billing.py)
Lookups (/score, /sst-exposure) work without a key, limited per client IP per
day; with a key they count against the tier's monthly quota. /score-batch and
/sst-exposure-pdf need a key on a tier that includes them.

Every response carries `data_notice`: null when the loaded files are OSHA's
own, otherwise a sentence saying they are third-party copies, undeclared or
synthetic (see provenance.py).

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
import json
import os
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from . import billing, provenance, sst
from .accounts import Account, AccountStore, QuotaExceeded, load_tiers
from .mailer import Mailer, key_email, mailer_from_env, recovery_email
from .match import build_directory, match_rows, suggestions
from .models.baseline import TrainResult, load_model
from .normalize import state_code, zip5
from .report import _clean, _table_exists, ensure_scores, risk_report, score_keys

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5_000
ANONYMOUS_DAILY_LOOKUPS = 20
RECOVERY_DAILY_REQUESTS = 5
PAYMENT_LINK_ENV = "NEXTRECORDABLE_PAYMENT_LINK_{}"   # e.g. NEXTRECORDABLE_PAYMENT_LINK_PRO

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


class RecoverRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


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
    accounts: AccountStore | None = None,
    mailer: Mailer | None = None,
    env: dict[str, str] | None = None,
) -> FastAPI:
    """App bound to an open DuckDB connection and a trained model.

    Without `accounts` every endpoint is open (tests, local use). With it,
    keys, quotas and the Stripe webhook are enforced.
    """
    env = dict(os.environ if env is None else env)
    panel = panel_table or model.panel_table
    if not _table_exists(con, directory):
        build_directory(con, panel_table=panel, output=directory)
    ensure_scores(con, model, panel_table=panel, scores_table=scores_table)
    if not _table_exists(con, "sst_position"):
        sst.build_sst_table(con, panel_table=panel)
    data_notice = provenance.notice(con)

    app = FastAPI(
        title="nextrecordable",
        description="Establishment-level injury risk from OSHA's public injury and enforcement data.",
        version="0.2.0",
    )
    app.state.con = con
    app.state.model = model
    cursor_lock = threading.Lock()
    anon_lock = threading.Lock()
    anon_counts: dict[tuple[str, str], int] = defaultdict(int)
    mailer = mailer or (mailer_from_env(env) if accounts is not None else None)
    base_url = env.get("NEXTRECORDABLE_BASE_URL", "http://127.0.0.1:8000")

    def client_ip(request: Request) -> str:
        # Behind the bundled Caddy proxy the client address arrives in X-Forwarded-For.
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def authorize(request: Request, key: str | None, *, units: int = 1, need: str | None = None) -> Account | None:
        """Check the key and quota for a request. Returns the account, or None when anonymous."""
        if accounts is None:
            return None
        account = accounts.verify(key) if key else None
        if key and account is None:
            raise HTTPException(status_code=401, detail="Unknown or inactive API key.")
        if account is None:
            if need is not None:
                raise HTTPException(status_code=401, detail=f"An API key is required for {need}. See /pricing.")
            day = datetime.now(UTC).strftime("%Y-%m-%d")
            with anon_lock:
                k = (client_ip(request), day)
                if anon_counts[k] + units > ANONYMOUS_DAILY_LOOKUPS:
                    raise HTTPException(status_code=429, detail=(
                        f"Free lookups without a key are limited to {ANONYMOUS_DAILY_LOOKUPS} a day. See /pricing."))
                anon_counts[k] += units
            return None
        limits = account.limits
        if need == "batch uploads" and units > (limits.get("batch_rows") or 0):
            raise HTTPException(status_code=403, detail=(
                f"The {limits['name']} tier allows batch uploads of up to {limits.get('batch_rows') or 0:,} rows."))
        if need == "SST exposure PDFs" and not limits.get("sst_pdf"):
            raise HTTPException(status_code=403, detail=f"The {limits['name']} tier does not include PDFs.")
        try:
            accounts.meter(account, units)
        except QuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return account

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
            "rate_served": model.rate_served,
            "data_provenance": provenance.summary(con)["kind"],
            "data_notice": data_notice,
            "sst_directive": sst.load_directive()["directive"],
        }

    def resolve(cur, req: ScoreRequest):
        """Matched directory row for a lookup, or a 404 with suggestions."""
        m = match_rows(cur, [req.model_dump()], directory=directory).iloc[0]
        if m["status"] != "matched":
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ambiguous" if m["status"] == "ambiguous" else "no_match",
                    "message": (
                        "More than one establishment matches closely; add a street address."
                        if m["status"] == "ambiguous" else
                        f"No confident match for '{req.name}' in {state_code(req.state)} "
                        f"{zip5(req.zip_code) or req.zip_code}. It may "
                        "not file with OSHA (most workplaces under 20 employees don't), or the "
                        "name on file may differ."
                    ),
                    "candidates": suggestions(cur, int(m["rec_id"]), directory=directory),
                    "data_notice": data_notice,
                },
            )
        return m

    def match_info(m, req: ScoreRequest) -> dict[str, Any]:
        return {"match_status": m["status"], "match_probability": _clean(m["match_probability"]),
                "query": req.model_dump()}

    def clean_tree(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean_tree(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean_tree(v) for v in value]
        return _clean(value)

    @app.post("/sst-exposure")
    def sst_exposure(req: ScoreRequest, request: Request,
                     x_api_key: str | None = Header(None)) -> dict[str, Any]:
        """Position against OSHA's SST selection criteria. Describes; never predicts an inspection."""
        authorize(request, x_api_key)
        cur = cursor()
        try:
            m = resolve(cur, req)
            report = clean_tree(sst.sst_exposure(cur, m["estab_key"], panel_table=panel))
            report["match_info"] = match_info(m, req)
            return report
        finally:
            cur.close()

    @app.post("/sst-exposure-pdf")
    def sst_exposure_pdf(req: ScoreRequest, request: Request,
                         x_api_key: str | None = Header(None)) -> Response:
        authorize(request, x_api_key, need="SST exposure PDFs")
        cur = cursor()
        try:
            m = resolve(cur, req)
            report = clean_tree(sst.sst_exposure(cur, m["estab_key"], panel_table=panel))
        finally:
            cur.close()
        return Response(
            content=sst.render_pdf(report), media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="sst-exposure.pdf"'},
        )

    @app.post("/score")
    def score(req: ScoreRequest, request: Request,
              x_api_key: str | None = Header(None)) -> dict[str, Any]:
        authorize(request, x_api_key)
        cur = cursor()
        try:
            m = resolve(cur, req)
            report = risk_report(cur, m["estab_key"], model, panel_table=panel, scores_table=scores_table)
            report["match_info"] = match_info(m, req)
            report["data_notice"] = data_notice
            return report
        finally:
            cur.close()

    @app.post("/score-batch")
    def score_batch(request: Request, file: UploadFile = File(...),
                    x_api_key: str | None = Header(None)) -> dict[str, Any]:
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
        authorize(request, x_api_key, units=len(rows), need="batch uploads")
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
                "data_notice": data_notice,
            },
            "results": results,
            "unmatched": unmatched,
        }

    @app.post("/sst-batch")
    def sst_batch(request: Request, file: UploadFile = File(...), format: str = "json",
                  x_api_key: str | None = Header(None)) -> Any:
        """SST positions for a client list (an extension center's plants, a broker's book)."""
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a .csv")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large: the limit is {MAX_UPLOAD_BYTES // 1_048_576} MB.")
        rows = parse_upload(data)
        if not rows:
            raise HTTPException(status_code=400, detail="CSV has no data rows.")
        authorize(request, x_api_key, units=len(rows), need="batch uploads")
        cur = cursor()
        try:
            matches = match_rows(cur, rows, directory=directory)
            matched = matches[matches["status"] == "matched"]
            positions = {p["estab_key"]: p for p in sst.sst_batch(cur, matched["estab_key"].tolist(), panel_table=panel)}
        finally:
            cur.close()
        results = []
        for m in matched.to_dict("records"):
            pos = positions.get(m["estab_key"])
            if pos is not None:
                results.append({"input_row": int(m["rec_id"]), "input_name": m["original_name"],
                                "match_probability": _clean(m["match_probability"]),
                                **{k: _clean(v) for k, v in pos.items()}})
        unmatched = [{"input_row": int(m["rec_id"]), "input_name": m["original_name"], "reason": m["status"]}
                     for m in matches[matches["status"] != "matched"].to_dict("records")]
        if format == "csv":
            buf = io.StringIO()
            buf.write(f"# {sst.STATEMENT}\n")
            if data_notice:
                buf.write(f"# {data_notice}\n")
            cols = ["input_row", "input_name", "match_probability", *sst.BATCH_COLUMNS]
            w = csv.DictWriter(buf, fieldnames=[*cols, "match_status"], extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({**r, "match_status": "matched"})
            for u in unmatched:
                w.writerow({"input_row": u["input_row"], "input_name": u["input_name"], "match_status": u["reason"]})
            return Response(content=buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="sst-positions.csv"'})
        return {
            "statement": sst.STATEMENT,
            "data_notice": data_notice,
            "directive": sst.load_directive()["directive"],
            "summary": {"n_input_rows": len(rows), "n_matched": len(results), "n_unmatched": len(unmatched)},
            "results": results,
            "unmatched": unmatched,
        }

    @app.get("/pricing", response_class=HTMLResponse, include_in_schema=False)
    def pricing() -> HTMLResponse:
        return HTMLResponse(render_pricing(env))

    @app.get("/account")
    def account(x_api_key: str | None = Header(None)) -> dict[str, Any]:
        if accounts is None:
            raise HTTPException(status_code=404, detail="Accounts are not enabled on this server.")
        acct = accounts.verify(x_api_key)
        if acct is None:
            raise HTTPException(status_code=401, detail="Unknown or inactive API key.")
        return {"tier": acct.tier, "key_prefix": acct.key_prefix, "lookups_this_month": accounts.usage(acct.id),
                "monthly_lookups": acct.limits.get("monthly_lookups"), "batch_rows": acct.limits.get("batch_rows"),
                "sst_pdf": acct.limits.get("sst_pdf")}

    recover_lock = threading.Lock()
    recover_counts: dict[tuple[str, str], int] = defaultdict(int)

    @app.post("/account/recover")
    def account_recover(req: RecoverRequest, request: Request) -> dict[str, str]:
        """Same answer whether or not the address has an account, so it cannot be used to find customers."""
        if accounts is None or mailer is None:
            raise HTTPException(status_code=404, detail="Accounts are not enabled on this server.")
        with recover_lock:
            k = (client_ip(request), datetime.now(UTC).strftime("%Y-%m-%d"))
            recover_counts[k] += 1
            if recover_counts[k] > RECOVERY_DAILY_REQUESTS:
                raise HTTPException(status_code=429, detail="Too many recovery requests today.")
        for _acct, token in accounts.start_recovery(req.email):
            subject, body = recovery_email(f"{base_url}/account/recover/confirm?token={token}")
            mailer.send(req.email.strip().lower(), subject, body)
        return {"status": "sent_if_registered",
                "message": "If this address has an account, a link to replace its key is on its way."}

    @app.get("/account/recover/confirm", response_class=HTMLResponse, include_in_schema=False)
    def account_recover_confirm(token: str = "") -> HTMLResponse:
        if accounts is None or mailer is None:
            raise HTTPException(status_code=404, detail="Accounts are not enabled on this server.")
        done = accounts.finish_recovery(token)
        if done is None:
            return HTMLResponse("<!doctype html><title>Link expired</title><p>This link has expired or was "
                                "already used. Ask for a new one.</p>", status_code=400)
        acct, key = done
        subject, body = key_email(acct.limits["name"], key, base_url)
        mailer.send(acct.email, subject, body)
        return HTMLResponse("<!doctype html><title>New key sent</title><p>Your old key no longer works. "
                            "A new key has been emailed to the address on the account.</p>")

    @app.post("/stripe/webhook", include_in_schema=False)
    async def stripe_webhook(request: Request) -> dict[str, Any]:
        if accounts is None or mailer is None:
            raise HTTPException(status_code=404, detail="Billing is not enabled on this server.")
        payload = await request.body()
        try:
            event = billing.verify(payload, request.headers.get("stripe-signature"),
                                   env.get("NEXTRECORDABLE_STRIPE_WEBHOOK_SECRET", ""))
        except billing.SignatureError as exc:
            raise HTTPException(status_code=400, detail=f"Bad signature: {exc}") from exc
        link_tiers = json.loads(env.get("NEXTRECORDABLE_STRIPE_LINK_TIERS", "{}"))
        return billing.handle_event(event, accounts, mailer, link_tiers=link_tiers, base_url=base_url)

    return app


def render_pricing(env: dict[str, str]) -> str:
    """Pricing page from data/tiers.json. Buttons use Payment Links from env; none set means no button."""
    def esc(t: Any) -> str:
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    cards = []
    for t in load_tiers().values():
        price = ("Free" if t["price_usd_month"] == 0 else
                 "Contact us" if t["price_usd_month"] is None else f"${t['price_usd_month']:,}/mo")
        quota = ("Custom volume" if t["monthly_lookups"] is None else f"{t['monthly_lookups']:,} lookups a month")
        batch = f"Batch uploads up to {t['batch_rows']:,} rows" if t["batch_rows"] else "No batch uploads"
        pdf = "SST exposure PDFs" if t["sst_pdf"] else "SST position on screen only"
        link = env.get(PAYMENT_LINK_ENV.format(t["id"].upper()))
        if t["id"] == "free":
            cta = '<a class="btn" href="/">Look up an establishment</a>'
        elif link:
            cta = f'<a class="btn" href="{esc(link)}">Subscribe</a>'
        elif t["id"] == "enterprise" and env.get("NEXTRECORDABLE_CONTACT_EMAIL"):
            cta = f'<a class="btn" href="mailto:{esc(env["NEXTRECORDABLE_CONTACT_EMAIL"])}">Contact</a>'
        else:
            cta = '<span class="btn off">Not yet available</span>'
        cards.append(f"""<section class="tier" id="tier-{esc(t['id'])}"><h2>{esc(t['name'])}</h2>
<p class="price">{esc(price)}</p><p>{esc(t['summary'])}</p>
<ul><li>{esc(quota)}</li><li>{esc(batch)}</li><li>{esc(pdf)}</li></ul>{cta}</section>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pricing · nextrecordable</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:24px 16px;background:#fafaf7;color:#1a1a1a}}
main{{max-width:1040px;margin:0 auto}} h1{{margin:0 0 4px}} .lede{{color:#444;max-width:62ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:24px 0}}
.tier{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:18px}}
.price{{font-size:1.6rem;font-weight:700;margin:4px 0}} ul{{padding-left:18px}}
.btn{{display:inline-block;padding:9px 14px;border-radius:8px;background:#1a1a1a;color:#fff;text-decoration:none}}
.btn.off{{background:#bbb}} .fine{{font-size:.85rem;color:#555;max-width:70ch}}
</style></head><body><main>
<h1>Pricing</h1>
<p class="lede">Establishment injury history, peer position and OSHA Site-Specific Targeting position from
OSHA's public injury logs and DOL enforcement records.</p>
<div class="grid">{''.join(cards)}</div>
<p class="fine">{esc(sst.STATEMENT)} Independent analysis of public data; not affiliated with, endorsed by, or
produced on behalf of OSHA or the U.S. Department of Labor. Not legal advice.</p>
</main></body></html>"""


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
    accounts_path = os.environ.get("NEXTRECORDABLE_ACCOUNTS_DB", str(db_path.parent / "accounts.sqlite"))
    return create_app(con, model, panel_table=panel, accounts=AccountStore(accounts_path))
