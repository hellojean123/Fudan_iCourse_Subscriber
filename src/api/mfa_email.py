"""Fetch a newly issued Fudan UIS email verification code via IMAP.

This module never prints codes or mailbox credentials. A dedicated IMAP app
password is preferred; SMTP credentials are reused only when both mailbox
addresses refer to the very same account.
"""
from __future__ import annotations

import email
import html
import imaplib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from email import policy
from email.utils import parsedate_to_datetime, parseaddr

from src.runtime import config


class EmailMFAError(RuntimeError):
    """A non-retryable MFA configuration or verification failure."""


def _mail_config() -> tuple[str, int, str, str, tuple[str, ...]]:
    username = (os.getenv("ICOURSE_MFA_IMAP_USERNAME") or
                config.RECEIVER_EMAIL).strip()
    host = os.getenv("ICOURSE_MFA_IMAP_HOST", "").strip()
    if not host and username.lower().endswith("@qq.com"):
        host = "imap.qq.com"
    if not host:
        raise EmailMFAError(
            "Email MFA requires ICOURSE_MFA_IMAP_HOST for the receiving mailbox"
        )
    if not username:
        raise EmailMFAError("Email MFA requires a receiving mailbox username")
    password = os.getenv("ICOURSE_MFA_IMAP_PASSWORD", "")
    if (not password and username.lower() == config.SMTP_EMAIL.strip().lower()
            and host.lower() == "imap.qq.com"):
        password = config.SMTP_PASSWORD
    if not password:
        raise EmailMFAError(
            "Email MFA requires ICOURSE_MFA_IMAP_PASSWORD for this mailbox"
        )
    sender_domains = tuple(
        d.strip().lower().lstrip("@")
        for d in (os.getenv("ICOURSE_MFA_SENDER_DOMAINS") or
                           "fudan.edu.cn").split(",")
        if d.strip()
    )
    if not sender_domains:
        raise EmailMFAError("Email MFA sender domain allowlist is empty")
    return host, int(os.getenv("ICOURSE_MFA_IMAP_PORT") or "993"), username, password, sender_domains


def _is_allowed_sender(msg: email.message.Message,
                       domains: tuple[str, ...]) -> bool:
    address = parseaddr(msg.get("From", ""))[1].lower()
    if "@" not in address:
        return False
    domain = address.rsplit("@", 1)[-1]
    return any(domain == x or domain.endswith("." + x) for x in domains)


def _message_text(msg: email.message.Message) -> str:
    subject = str(msg.get("Subject", ""))
    parts = []
    for part in msg.walk():
        if part.get_content_maintype() != "text":
            continue
        if part.get_content_disposition() == "attachment":
            continue
        try:
            body = part.get_content()
        except (UnicodeError, LookupError):
            continue
        if isinstance(body, str):
            parts.append(body[:12000])
    content = html.unescape(subject + "\n" + "\n".join(parts))
    return re.sub(r"<[^>]+>", " ", content)


def extract_email_code(raw: bytes, *, sender_domains: tuple[str, ...],
                       not_before: datetime) -> str | None:
    """Only use recent, Fudan-sender messages explicitly about verification."""
    msg = email.message_from_bytes(raw, policy=policy.default)
    if not _is_allowed_sender(msg, sender_domains):
        return None
    try:
        sent_at = parsedate_to_datetime(msg["Date"])
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        if sent_at.astimezone(timezone.utc) < not_before:
            return None
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    content = _message_text(msg)
    if not re.search(r"验证码|驗證碼|验证|verification|one.time|otp|code",
                     content, re.IGNORECASE):
        return None
    # Prefer numbers close to a code label; do not confuse dates or IDs.
    match = re.search(
        r"(?:验证码|驗證碼|校验码|verification\s*code|one.time\s*code|"
        r"otp\s*code|code)\s*(?:为|是|:|：|=|\s){0,10}"
        r"(?<!\d)(\d{6})(?!\d)",
        content, re.IGNORECASE
    )
    return match.group(1) if match else None


def wait_for_email_code(*, requested_at: datetime | None = None,
                        timeout_seconds: float = 120,
                        interval_seconds: float = 5,
                        client_factory=None,
                        monotonic=None,
                        sleep=None) -> str:
    """Poll only recent messages. Never mark messages read or log OTPs."""
    host, port, username, password, domains = _mail_config()
    factory = client_factory or imaplib.IMAP4_SSL
    now = monotonic or time.monotonic
    pause = sleep or time.sleep
    requested_at = requested_at or datetime.now(timezone.utc)
    cutoff = requested_at.astimezone(timezone.utc) - timedelta(seconds=45)
    deadline = now() + timeout_seconds
    try:
        client = factory(host, port)
        try:
            client.login(username, password)
            status, _ = client.select("INBOX", readonly=True)
            if status != "OK":
                raise EmailMFAError("Cannot open MFA email inbox")
            while True:
                status, rows = client.uid(
                    "SEARCH", None, "SINCE", cutoff.strftime("%d-%b-%Y")
                )
                if status != "OK":
                    raise EmailMFAError("IMAP search failed")
                ids = (rows[0] or b"").split()[-40:] if rows else []
                for uid in reversed(ids):
                    status, payload = client.uid(
                        "FETCH", uid, "(BODY.PEEK[])"
                    )
                    if status != "OK":
                        continue
                    for part in payload or []:
                        if isinstance(part, tuple) and isinstance(part[1], bytes):
                            code = extract_email_code(
                                part[1], sender_domains=domains,
                                not_before=cutoff
                            )
                            if code is not None:
                                return code
                if now() >= deadline:
                    break
                pause(min(interval_seconds, max(0, deadline - now())))
        finally:
            try:
                client.logout()
            except Exception:
                pass
    except (imaplib.IMAP4.error, OSError) as exc:
        raise EmailMFAError(
            "IMAP login/read failed; verify the receiving mailbox and its IMAP app password"
        ) from exc
    raise EmailMFAError(
        "No fresh email verification code received before timeout; "
        "check whether the UIS portal actually sent an email"
    )
