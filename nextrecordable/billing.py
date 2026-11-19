"""Stripe webhook: verify, then create, upgrade or downgrade accounts.

Payment Links create Checkout Sessions. On `checkout.session.completed` the
session names its payment link (`payment_link`, a plink_ id); the mapping
from link to tier comes from NEXTRECORDABLE_STRIPE_LINK_TIERS, a JSON object
such as {"plink_123": "pro", "plink_456": "team"}. A new account gets a key
by email. `customer.subscription.deleted` drops the customer's accounts to
Free. Events are processed once each (Stripe retries).

Signature check follows Stripe's documented scheme: the Stripe-Signature
header carries `t=<unix time>` and one or more `v1=<hex>`; the expected value
is HMAC-SHA256(endpoint secret, f"{t}.{raw body}"). Requests older than five
minutes are refused.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from .accounts import AccountStore, load_tiers
from .mailer import Mailer, key_email

TOLERANCE_SECONDS = 300


class SignatureError(Exception):
    pass


def sign(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """A valid Stripe-Signature header for `payload` (tests and local replays)."""
    t = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def verify(payload: bytes, header: str | None, secret: str, *, now: float | None = None) -> dict[str, Any]:
    if not secret:
        raise SignatureError("webhook secret not configured")
    if not header:
        raise SignatureError("missing Stripe-Signature header")
    parts: dict[str, list[str]] = {}
    for item in header.split(","):
        k, _, v = item.strip().partition("=")
        parts.setdefault(k, []).append(v)
    try:
        t = int(parts["t"][0])
    except (KeyError, ValueError) as exc:
        raise SignatureError("no timestamp in signature") from exc
    if abs((now or time.time()) - t) > TOLERANCE_SECONDS:
        raise SignatureError("timestamp outside tolerance")
    expected = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, v) for v in parts.get("v1", [])):
        raise SignatureError("signature mismatch")
    return json.loads(payload)


def handle_event(event: dict[str, Any], store: AccountStore, mailer: Mailer, *,
                 link_tiers: dict[str, str], base_url: str) -> dict[str, Any]:
    """Apply one verified event. Returns what happened (never the key)."""
    etype, eid = event.get("type", ""), event.get("id", "")
    if store.seen_event(eid, etype):
        return {"status": "duplicate", "type": etype}
    obj = (event.get("data") or {}).get("object") or {}
    if etype == "checkout.session.completed":
        tier = link_tiers.get(obj.get("payment_link") or "") or (obj.get("metadata") or {}).get("tier")
        email = (obj.get("customer_details") or {}).get("email") or obj.get("customer_email")
        if tier not in load_tiers() or not email:
            return {"status": "ignored", "type": etype, "reason": "unknown payment link or no email"}
        customer = obj.get("customer")
        existing = store.by_stripe_customer(customer) if customer else []
        if existing:
            for a in existing:
                store.set_tier(a.id, tier)
            return {"status": "upgraded", "type": etype, "tier": tier, "accounts": [a.id for a in existing]}
        account, key = store.create(email, tier, stripe_customer=customer,
                                    stripe_subscription=obj.get("subscription"))
        subject, body = key_email(load_tiers()[tier]["name"], key, base_url)
        mailer.send(email, subject, body)
        return {"status": "created", "type": etype, "tier": tier, "account": account.id,
                "key_prefix": account.key_prefix}
    if etype == "customer.subscription.deleted":
        accounts = store.by_stripe_customer(obj.get("customer") or "")
        for a in accounts:
            store.set_tier(a.id, "free")
        return {"status": "downgraded", "type": etype, "accounts": [a.id for a in accounts]}
    return {"status": "ignored", "type": etype}
