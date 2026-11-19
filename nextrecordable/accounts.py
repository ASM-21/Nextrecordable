"""API keys, tiers and monthly metering, in a small SQLite file.

Separate from the DuckDB analytics file on purpose: that file is rebuilt from
scratch on every data refresh, while accounts must survive it. Keys are shown
once, at creation, and stored only as a SHA-256 hash plus an 8-character
prefix for support conversations.

Tiers and quotas live in data/tiers.json; the pricing page and the API read
the same file.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

TIERS_FILE = Path(__file__).parent / "data" / "tiers.json"
KEY_PREFIX = "nr_"
RECOVERY_TTL = timedelta(hours=1)


@lru_cache(maxsize=1)
def load_tiers() -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in json.loads(TIERS_FILE.read_text())["tiers"]}


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def period(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


@dataclass(frozen=True)
class Account:
    id: int
    email: str
    tier: str
    key_prefix: str
    stripe_customer: str | None
    active: bool

    @property
    def limits(self) -> dict[str, Any]:
        return load_tiers()[self.tier]


class QuotaExceeded(Exception):
    pass


class AccountStore:
    """Thread-safe SQLite store. One file per deployment (default data/accounts.sqlite)."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY, email TEXT NOT NULL, tier TEXT NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL, key_prefix TEXT NOT NULL,
                    stripe_customer TEXT, stripe_subscription TEXT,
                    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage (
                    account_id INTEGER NOT NULL, period TEXT NOT NULL, lookups INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_id, period));
                CREATE TABLE IF NOT EXISTS stripe_events (id TEXT PRIMARY KEY, type TEXT, received_at TEXT);
                CREATE TABLE IF NOT EXISTS recovery_tokens (
                    token_hash TEXT PRIMARY KEY, account_id INTEGER NOT NULL, expires_at TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0);
                """
            )

    def _row(self, sql: str, args: tuple) -> tuple | None:
        with self._lock:
            return self._db.execute(sql, args).fetchone()

    def create(self, email: str, tier: str, *, stripe_customer: str | None = None,
               stripe_subscription: str | None = None) -> tuple[Account, str]:
        """New account and its plaintext key (returned once, never stored)."""
        if tier not in load_tiers():
            raise ValueError(f"unknown tier {tier!r}")
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO accounts (email, tier, key_hash, key_prefix, stripe_customer, stripe_subscription, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (email.strip().lower(), tier, hash_key(key), key[: len(KEY_PREFIX) + 8], stripe_customer,
                 stripe_subscription, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            acct_id = cur.lastrowid
        return self.get(acct_id), key

    def get(self, account_id: int) -> Account:
        r = self._row("SELECT id, email, tier, key_prefix, stripe_customer, active FROM accounts WHERE id = ?",
                      (account_id,))
        return Account(r[0], r[1], r[2], r[3], r[4], bool(r[5]))

    def verify(self, key: str | None) -> Account | None:
        if not key or not key.startswith(KEY_PREFIX):
            return None
        r = self._row("SELECT id FROM accounts WHERE key_hash = ? AND active = 1", (hash_key(key),))
        return self.get(r[0]) if r else None

    def by_stripe_customer(self, customer: str) -> list[Account]:
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT id FROM accounts WHERE stripe_customer = ?", (customer,)).fetchall()]
        return [self.get(i) for i in ids]

    def set_tier(self, account_id: int, tier: str) -> None:
        if tier not in load_tiers():
            raise ValueError(f"unknown tier {tier!r}")
        with self._lock:
            self._db.execute("UPDATE accounts SET tier = ? WHERE id = ?", (tier, account_id))

    def deactivate(self, account_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE accounts SET active = 0 WHERE id = ?", (account_id,))

    def usage(self, account_id: int, when: datetime | None = None) -> int:
        r = self._row("SELECT lookups FROM usage WHERE account_id = ? AND period = ?", (account_id, period(when)))
        return int(r[0]) if r else 0

    def meter(self, account: Account, n: int = 1, when: datetime | None = None) -> int:
        """Count `n` lookups against this month's quota. Raises QuotaExceeded, counting nothing."""
        limit = account.limits.get("monthly_lookups")
        p = period(when)
        with self._lock:
            used = self._db.execute("SELECT lookups FROM usage WHERE account_id = ? AND period = ?",
                                    (account.id, p)).fetchone()
            used = int(used[0]) if used else 0
            if limit is not None and used + n > limit:
                raise QuotaExceeded(f"{used:,} of {limit:,} lookups used this month; this request needs {n:,}.")
            self._db.execute(
                "INSERT INTO usage (account_id, period, lookups) VALUES (?, ?, ?) "
                "ON CONFLICT (account_id, period) DO UPDATE SET lookups = lookups + excluded.lookups",
                (account.id, p, n))
        return used + n

    def seen_event(self, event_id: str, event_type: str) -> bool:
        """Record a Stripe event id; True if it was already processed (webhooks retry)."""
        with self._lock:
            try:
                self._db.execute("INSERT INTO stripe_events VALUES (?, ?, ?)",
                                 (event_id, event_type, datetime.now(UTC).isoformat(timespec="seconds")))
                return False
            except sqlite3.IntegrityError:
                return True

    # ---------------------------------------------------------- recovery

    def start_recovery(self, email: str, now: datetime | None = None) -> list[tuple[Account, str]]:
        """One-time tokens for every active account with this email (empty list if none)."""
        now = now or datetime.now(UTC)
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT id FROM accounts WHERE email = ? AND active = 1", (email.strip().lower(),)).fetchall()]
        out = []
        for acct_id in ids:
            token = secrets.token_urlsafe(32)
            with self._lock:
                self._db.execute("INSERT INTO recovery_tokens VALUES (?, ?, ?, 0)",
                                 (hash_key(token), acct_id, (now + RECOVERY_TTL).isoformat()))
            out.append((self.get(acct_id), token))
        return out

    def finish_recovery(self, token: str, now: datetime | None = None) -> tuple[Account, str] | None:
        """Rotate the account's key if the token is valid, unused and unexpired. Returns (account, new key)."""
        now = now or datetime.now(UTC)
        with self._lock:
            r = self._db.execute(
                "SELECT account_id, expires_at, used FROM recovery_tokens WHERE token_hash = ?",
                (hash_key(token or ""),)).fetchone()
            if r is None or r[2] or datetime.fromisoformat(r[1]) < now:
                return None
            key = KEY_PREFIX + secrets.token_urlsafe(32)
            self._db.execute("UPDATE recovery_tokens SET used = 1 WHERE token_hash = ?", (hash_key(token),))
            self._db.execute("UPDATE accounts SET key_hash = ?, key_prefix = ? WHERE id = ?",
                             (hash_key(key), key[: len(KEY_PREFIX) + 8], r[0]))
        return self.get(r[0]), key
