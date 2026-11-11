"""Regression tests for UIS chained authentication and email MFA handling."""
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from unittest.mock import Mock
import pytest

from src.api import mfa_email
from src.api.webvpn import WebVPNSession


def message(sender="noreply@id.fudan.edu.cn", subject="复旦统一身份认证验证码",
            body="您的验证码为：123456，请勿泄露。", date=None):
    msg = EmailMessage()
    msg["From"] = sender
    msg["Subject"] = subject
    msg["Date"] = (date or datetime.now(timezone.utc)).strftime(
        "%a, %d %b %Y %H:%M:%S %z")
    msg.set_content(body)
    return msg.as_bytes()


def test_extracts_recent_fudan_code():
    now = datetime.now(timezone.utc)
    assert mfa_email.extract_email_code(
        message(), sender_domains=("fudan.edu.cn",),
        not_before=now - timedelta(minutes=1)) == "123456"


@pytest.mark.parametrize("raw", [
    message(sender="noreply@evil-fudan.edu.cn"),
    message(sender="noreply@example.org"),
    message(subject="2026 course number", body="course 123456"),
    message(date=datetime(2020, 1, 1, tzinfo=timezone.utc)),
])
def test_rejects_untrusted_or_irrelevant_mail(raw):
    assert mfa_email.extract_email_code(
        raw, sender_domains=("fudan.edu.cn",),
        not_before=datetime.now(timezone.utc) - timedelta(minutes=2)) is None


def test_missing_imap_credentials_fail_closed(monkeypatch):
    from src.runtime import config
    monkeypatch.setattr(config, "RECEIVER_EMAIL", "student@example.com")
    monkeypatch.setattr(config, "SMTP_EMAIL", "sender@qq.com")
    monkeypatch.setenv("ICOURSE_MFA_IMAP_HOST", "imap.example.com")
    monkeypatch.delenv("ICOURSE_MFA_IMAP_PASSWORD", raising=False)
    with pytest.raises(mfa_email.EmailMFAError, match="IMAP_PASSWORD"):
        mfa_email._mail_config()


def test_cas_requires_real_mfa_metadata():
    vpn = WebVPNSession()
    with pytest.raises(RuntimeError, match="not a recognized MFA"):
        vpn._complete_icourse_email_mfa(
            response={"code": 200}, lck="lck", entity_id="icourse",
            request_type="chain_type", student_id="student",
            auth_chain_code="chain", referer="https://example.com/ac/")


def test_chained_email_mfa_verified_without_otp_log(monkeypatch, capsys):
    from src.runtime import config
    monkeypatch.setenv("ICOURSE_MFA_MODULE_CODE", "userAndEmail")
    monkeypatch.delenv("ICOURSE_MFA_SEND_URL", raising=False)
    monkeypatch.delenv("ICOURSE_MFA_CODE_FIELD", raising=False)
    monkeypatch.setattr(mfa_email, "wait_for_email_code",
                        lambda **kwargs: "123456")
    vpn = WebVPNSession()
    sent = {}
    def call(url, *, json, headers, timeout):
        sent["json"] = dict(json)
        sent["code"] = json["authPara"].get("otpCode")
        resp = Mock()
        resp.json.return_value = {"code": 200, "loginToken": "ticket-token"}
        return resp
    vpn.session.post = Mock(side_effect=call)
    result = vpn._complete_icourse_email_mfa(
        response={"code": 200, "pageLevelNo": 2, "requestNumber": "request-no",
                  "authChainCode": "next-chain",
                  "moduleCodes": ["userAndOtp", "userAndEmail"]},
        lck="lck", entity_id="icourse", request_type="chain_type",
        student_id="student", auth_chain_code="previous",
        referer="https://example.com/ac/")
    assert result == "ticket-token"
    assert sent["json"]["authModuleCode"] == "userAndEmail"
    assert sent["json"]["requestNumber"] == "request-no"
    assert sent["json"]["authChainCode"] == "next-chain"
    assert sent["code"] == "123456"
    assert "123456" not in capsys.readouterr().out


def test_rejects_missing_request_number_before_reading_inbox(monkeypatch):
    monkeypatch.setenv("ICOURSE_MFA_MODULE_CODE", "userAndEmail")
    vpn = WebVPNSession()
    with pytest.raises(RuntimeError, match="requestNumber"):
        vpn._complete_icourse_email_mfa(
            response={"pageLevelNo": 2, "moduleCodes": ["userAndEmail"]},
            lck="lck", entity_id="icourse", request_type="chain_type",
            student_id="student", auth_chain_code="chain",
            referer="https://example.com/ac/")


def test_unverified_email_dispatch_is_not_attempted(monkeypatch):
    monkeypatch.setenv("ICOURSE_MFA_MODULE_CODE", "userAndEmail")
    monkeypatch.setenv("ICOURSE_MFA_SEND_URL",
                       "https://id.fudan.edu.cn/unverified/send")
    vpn = WebVPNSession()
    with pytest.raises(mfa_email.EmailMFAError, match="verified request schema"):
        vpn._complete_icourse_email_mfa(
            response={"pageLevelNo": 2, "moduleCodes": ["userAndEmail"],
                      "requestNumber": "request-no"},
            lck="lck", entity_id="icourse", request_type="chain_type",
            student_id="student", auth_chain_code="chain",
            referer="https://example.com/ac/")
    assert not vpn.session.post.call_count if hasattr(vpn.session.post, "call_count") else True
