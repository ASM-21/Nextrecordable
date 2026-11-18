"""Accounts, quotas, Stripe webhook, key email, pricing page."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from nextrecordable import billing
from nextrecordable.accounts import AccountStore, QuotaExceeded, hash_key, load_tiers
from nextrecordable.api import ANONYMOUS_DAILY_LOOKUPS, create_app, render_pricing
from nextrecordable.mailer import DryRunMailer, mailer_from_env

SECRET = "whsec_test_not_a_real_secret"
LINKS = {"plink_pro": "pro", "plink_team": "team"}


def test_tier_prices_and_quotas_are_pinned():
    """The pricing page, the README and the API all read these; change them deliberately."""
    t = load_tiers()
    assert list(t) == ["free", "pro", "team", "enterprise"]
    assert [t[k]["price_usd_month"] for k in t] == [0, 79, 499, None]
    assert [t[k]["monthly_lookups"] for k in t] == [100, 2000, 25000, None]
    assert [t[k]["batch_rows"] for k in t] == [0, 1000, 5000, 5000]


def test_pricing_page_shows_tiers_and_only_configured_links():
    page = render_pricing({"NEXTRECORDABLE_PAYMENT_LINK_PRO": "https://buy.stripe.com/test_abc"})
    for text in ("$79/mo", "$499/mo", "Free", "Contact us", "2,000 lookups a month", "never predicts"):
        assert text in page, text
    assert "https://buy.stripe.com/test_abc" in page
    assert page.count("Not yet available") == 2            # team and enterprise unset


def test_keys_are_hashed_and_shown_once(tmp_path):
    store = AccountStore(tmp_path / "a.sqlite")
    acct, key = store.create("Buyer@Example.com", "pro")
    assert key.startswith("nr_") and acct.email == "buyer@example.com"
    raw = (tmp_path / "a.sqlite").read_bytes()
    assert key.encode() not in raw and hash_key(key).encode() in raw
    assert store.verify(key).id == acct.id and store.verify(key + "x") is None and store.verify(None) is None
    store.deactivate(acct.id)
    assert store.verify(key) is None


def test_quota_is_monthly_and_all_or_nothing(tmp_path):
    store = AccountStore(tmp_path / "a.sqlite")
    acct, _ = store.create("a@example.com", "free")
    nov, dec = datetime(2026, 11, 30, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC)
    assert store.meter(acct, 99, nov) == 99
    with pytest.raises(QuotaExceeded):
        store.meter(acct, 2, nov)
    assert store.usage(acct.id, nov) == 99                   # the refused request counted nothing
    assert store.meter(acct, 100, dec) == 100


def test_signature_verification(tmp_path):
    body = json.dumps({"id": "evt_1", "type": "ping"}).encode()
    assert billing.verify(body, billing.sign(body, SECRET), SECRET)["id"] == "evt_1"
    for header in (None, "t=1,v1=00", billing.sign(body, "whsec_other")):
        with pytest.raises(billing.SignatureError):
            billing.verify(body, header, SECRET)
    old = billing.sign(body, SECRET, timestamp=int(time.time()) - 3600)
    with pytest.raises(billing.SignatureError):
        billing.verify(body, old, SECRET)
    with pytest.raises(billing.SignatureError):
        billing.verify(body, billing.sign(body, SECRET), "")


def checkout(eid: str, link: str, email: str = "buyer@example.com", customer: str = "cus_1") -> dict:
    return {"id": eid, "type": "checkout.session.completed", "data": {"object": {
        "payment_link": link, "customer": customer, "subscription": "sub_1",
        "customer_details": {"email": email}}}}


def test_checkout_creates_account_and_emails_key_once(tmp_path):
    store, mail = AccountStore(tmp_path / "a.sqlite"), DryRunMailer(tmp_path / "outbox")
    res = billing.handle_event(checkout("evt_1", "plink_pro"), store, mail, link_tiers=LINKS, base_url="https://x")
    assert res["status"] == "created" and res["tier"] == "pro" and "key" not in json.dumps(res).lower().replace("key_prefix", "")
    assert len(mail.sent) == 1 and mail.sent[0]["to"] == "buyer@example.com"
    key = next(line.strip() for line in mail.sent[0]["body"].splitlines() if line.strip().startswith("nr_"))
    assert store.verify(key).tier == "pro"
    assert (tmp_path / "outbox").glob("*.eml")
    # Stripe retries: same event id does nothing
    again = billing.handle_event(checkout("evt_1", "plink_pro"), store, mail, link_tiers=LINKS, base_url="https://x")
    assert again["status"] == "duplicate" and len(mail.sent) == 1
    # upgrade keeps the key; cancellation drops to free
    up = billing.handle_event(checkout("evt_2", "plink_team"), store, mail, link_tiers=LINKS, base_url="https://x")
    assert up["status"] == "upgraded" and store.verify(key).tier == "team" and len(mail.sent) == 1
    gone = {"id": "evt_3", "type": "customer.subscription.deleted", "data": {"object": {"customer": "cus_1"}}}
    assert billing.handle_event(gone, store, mail, link_tiers=LINKS, base_url="x")["status"] == "downgraded"
    assert store.verify(key).tier == "free"


def test_unknown_link_is_ignored(tmp_path):
    store, mail = AccountStore(tmp_path / "a.sqlite"), DryRunMailer(tmp_path / "o")
    res = billing.handle_event(checkout("evt_9", "plink_unknown"), store, mail, link_tiers=LINKS, base_url="x")
    assert res["status"] == "ignored" and not mail.sent


def test_mailer_defaults_to_dry_run(tmp_path):
    m = mailer_from_env({"NEXTRECORDABLE_OUTBOX_DIR": str(tmp_path)})
    assert isinstance(m, DryRunMailer)
    with pytest.raises(ValueError):
        mailer_from_env({"NEXTRECORDABLE_MAIL_PROVIDER": "pigeon"})


# ------------------------------------------------------------------ API gating

@pytest.fixture()
def gated(world, tmp_path):
    store = AccountStore(tmp_path / "accounts.sqlite")
    mail = DryRunMailer(tmp_path / "outbox")
    env = {"NEXTRECORDABLE_STRIPE_WEBHOOK_SECRET": SECRET,
           "NEXTRECORDABLE_STRIPE_LINK_TIERS": json.dumps(LINKS)}
    app = create_app(world["con"], world["result"], accounts=store, mailer=mail, env=env)
    name, state, z = world["con"].execute(
        "SELECT establishment_name, state, zip5 FROM ita_panel WHERE year = 2024 ORDER BY estab_key LIMIT 1").fetchone()
    return TestClient(app), store, mail, {"name": name, "state": state, "zip_code": z}


def test_anonymous_lookups_are_capped_per_day(gated):
    client, _, _, body = gated
    codes = [client.post("/score", json=body).status_code for _ in range(ANONYMOUS_DAILY_LOOKUPS + 1)]
    assert codes[:-1] == [200] * ANONYMOUS_DAILY_LOOKUPS and codes[-1] == 429


def test_pdf_and_batch_need_a_key_on_the_right_tier(gated):
    client, store, _, body = gated
    assert client.post("/sst-exposure-pdf", json=body).status_code == 401
    _, free_key = store.create("f@example.com", "free")
    assert client.post("/sst-exposure-pdf", json=body, headers={"X-API-Key": free_key}).status_code == 403
    _, pro_key = store.create("p@example.com", "pro")
    r = client.post("/sst-exposure-pdf", json=body, headers={"X-API-Key": pro_key})
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    csv_body = f"name,state,zip\n{body['name']},{body['state']},{body['zip_code']}\n"
    files = {"file": ("b.csv", csv_body, "text/csv")}
    assert client.post("/score-batch", files=files).status_code == 401
    assert client.post("/score-batch", files=files, headers={"X-API-Key": free_key}).status_code == 403
    assert client.post("/score-batch", files=files, headers={"X-API-Key": pro_key}).status_code == 200
    usage = client.get("/account", headers={"X-API-Key": pro_key}).json()
    assert usage["lookups_this_month"] == 2 and usage["tier"] == "pro"
    assert client.post("/score", json=body, headers={"X-API-Key": "nr_wrong"}).status_code == 401


def test_webhook_end_to_end(gated):
    client, _store, mail, body = gated
    payload = json.dumps(checkout("evt_web", "plink_pro", email="plant@example.com")).encode()
    bad = client.post("/stripe/webhook", content=payload, headers={"Stripe-Signature": "t=1,v1=0"})
    assert bad.status_code == 400 and not mail.sent
    ok = client.post("/stripe/webhook", content=payload, headers={"Stripe-Signature": billing.sign(payload, SECRET)})
    assert ok.status_code == 200 and ok.json()["status"] == "created"
    key = next(line.strip() for line in mail.sent[0]["body"].splitlines() if line.strip().startswith("nr_"))
    assert client.post("/sst-exposure-pdf", json=body, headers={"X-API-Key": key}).status_code == 200
