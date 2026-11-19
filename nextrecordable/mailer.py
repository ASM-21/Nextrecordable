"""Outbound email behind one small interface. Dry-run unless configured otherwise.

    NEXTRECORDABLE_MAIL_PROVIDER = dryrun (default) | smtp
    NEXTRECORDABLE_MAIL_FROM     = sender address
    NEXTRECORDABLE_OUTBOX_DIR    = where dry-run messages are written (default data/outbox)
    NEXTRECORDABLE_SMTP_HOST / _PORT / _USER / _PASSWORD   for smtp (STARTTLS)

Any transactional provider with SMTP credentials (Postmark, SES, Resend,
Mailgun) works through `smtp`; a provider-specific HTTP client can be added
as another Mailer without touching callers.
"""

from __future__ import annotations

import os
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol


class Mailer(Protocol):
    def send(self, to: str, subject: str, body: str) -> str:
        """Send one plain-text message; return an identifier (message id or file path)."""
        ...


def _message(sender: str, to: str, subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content(body)
    return msg


class DryRunMailer:
    """Writes each message to an .eml file instead of sending it."""

    def __init__(self, outbox: Path | str, sender: str = "nextrecordable <no-reply@example.invalid>"):
        self.outbox, self.sender = Path(outbox), sender
        self.sent: list[dict[str, str]] = []

    def send(self, to: str, subject: str, body: str) -> str:
        self.outbox.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self.outbox / f"{stamp}.eml"
        path.write_bytes(bytes(_message(self.sender, to, subject, body)))
        self.sent.append({"to": to, "subject": subject, "body": body, "path": str(path)})
        return str(path)


class SMTPMailer:
    def __init__(self, host: str, port: int, user: str | None, password: str | None, sender: str):
        self.host, self.port, self.user, self.password, self.sender = host, port, user, password, sender

    def send(self, to: str, subject: str, body: str) -> str:
        msg = _message(self.sender, to, subject, body)
        with smtplib.SMTP(self.host, self.port, timeout=30) as s:
            s.starttls()
            if self.user:
                s.login(self.user, self.password or "")
            s.send_message(msg)
        return msg.get("Message-ID", "")


def mailer_from_env(env: dict[str, str] | None = None) -> Mailer:
    env = dict(os.environ if env is None else env)
    provider = env.get("NEXTRECORDABLE_MAIL_PROVIDER", "dryrun").lower()
    sender = env.get("NEXTRECORDABLE_MAIL_FROM", "nextrecordable <no-reply@example.invalid>")
    if provider == "dryrun":
        return DryRunMailer(env.get("NEXTRECORDABLE_OUTBOX_DIR", "data/outbox"), sender)
    if provider == "smtp":
        return SMTPMailer(env["NEXTRECORDABLE_SMTP_HOST"], int(env.get("NEXTRECORDABLE_SMTP_PORT", "587")),
                          env.get("NEXTRECORDABLE_SMTP_USER"), env.get("NEXTRECORDABLE_SMTP_PASSWORD"), sender)
    raise ValueError(f"Unknown NEXTRECORDABLE_MAIL_PROVIDER {provider!r}")


def key_email(tier_name: str, key: str, base_url: str) -> tuple[str, str]:
    subject = f"Your nextrecordable {tier_name} API key"
    body = f"""Thanks for subscribing to nextrecordable {tier_name}.

Your API key:

    {key}

This is the only time it is sent. Keep it somewhere safe; if it is lost or
leaked, reply to this email and we will issue a new one.

Use it in the X-API-Key header:

    curl -X POST {base_url}/sst-exposure-pdf -H "X-API-Key: {key}" \\
      -H "Content-Type: application/json" \\
      -d '{{"name": "...", "state": "PA", "zip_code": "18103"}}' -o report.pdf

What the reports are and are not: they describe an establishment's position
in OSHA's public injury data and against published OSHA criteria. They never
predict whether OSHA will inspect.
"""
    return subject, body
