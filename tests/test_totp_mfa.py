"""Regression coverage for Fudan UIS TOTP second-stage authentication.

No network calls or real credentials; RFC 6238 vector and simulated UIS
challenge responses cover both direct WebVPN and proxied iCourse flows.
"""
from unittest.mock import Mock
import pytest

import main
from src.api import webvpn
from src.api.webvpn import SecondFactorError, WebVPNSession, totp_code
from src.runtime import config

# RFC 6238 SHA-1 test seed for the 59-second TOTP vector.
RFC6238_BASE32 = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"


def reply(data, *, status=200, url="", html=""):
    result = Mock()
    result.json.return_value = data
    result.raise_for_status.return_value = None
    result.status_code = status
    result.url = url
    result.text = html
    result.history = []
    return result


def password_challenge():
    return {
        "code": 200,
        "second": True,
        "pageLevelNo": 2,
        "moduleCodes": ["qr", "userAndMail", "userAndSms", "userAndOtp"],
        "requestNumber": "request-number-test",
        "authChainCode": "otp-chain",
        "userName": "test-student",
    }


def test_rfc6238_totp_vector():
    assert totp_code(RFC6238_BASE32, timestamp=59) == "287082"
    assert totp_code(RFC6238_BASE32.lower() + "  ", timestamp=59) == "287082"


def test_missing_and_bad_totp_secret_fails_closed():
    with pytest.raises(SecondFactorError, match="UIS_TOTP_SECRET"):
        totp_code("")
    with pytest.raises(SecondFactorError, match="Base32"):
        totp_code("not-a-valid-base32-secret!")


@pytest.mark.parametrize("via_vpn", [False, True])
def test_second_factor_uses_upstream_payload(monkeypatch, capsys, via_vpn):
    monkeypatch.setattr(config, "TOTP_SECRET", RFC6238_BASE32)
    monkeypatch.setattr(webvpn, "totp_code", lambda secret: "234567")
    vpn = WebVPNSession()
    vpn.session.post = Mock(return_value=reply({"code": 200, "loginToken": "token"}))

    result = vpn._authenticate_totp(
        password_challenge(), "student", "lck", "https://icourse.fudan.edu.cn",
        "chain_type", via_vpn=via_vpn
    )
    assert result["loginToken"] == "token"
    args, kw = vpn.session.post.call_args
    assert args[0].endswith("/idp/authn/authExecute")
    assert kw["json"]["authModuleCode"] == "userAndOtp"
    assert kw["json"]["authChainCode"] == "otp-chain"
    assert kw["json"]["requestNumber"] == "request-number-test"
    assert kw["json"]["authPara"] == {
        "loginName": "test-student", "otpCode": "234567"
    }
    assert "234567" not in capsys.readouterr().out


def test_webvpn_password_response_can_continue_to_totp(monkeypatch):
    monkeypatch.setattr(config, "TOTP_SECRET", RFC6238_BASE32)
    monkeypatch.setattr(webvpn, "totp_code", lambda secret: "234567")
    vpn = WebVPNSession()
    vpn.session.post = Mock(side_effect=[
        reply(password_challenge()), reply({"code": 200, "loginToken": "web-token"})
    ])
    assert vpn._auth_execute("student", "encrypted", "lck", "webvpn",
                             "password-chain", "chain_type") == "web-token"
    assert vpn.session.post.call_count == 2


def test_proxied_icourse_password_response_can_continue_to_totp(monkeypatch):
    monkeypatch.setattr(config, "TOTP_SECRET", RFC6238_BASE32)
    monkeypatch.setattr(webvpn, "totp_code", lambda secret: "234567")
    vpn = WebVPNSession()
    monkeypatch.setattr(vpn, "_encrypt_password", lambda *_: "encrypted")
    vpn.session.get = Mock(side_effect=[
        reply({}, status=200),  # WebVPN preflight
        reply({}, url="https://id.fudan.edu.cn/ac?lck=challenge-test"),
        reply({"data": "dummy-public-key"}),
        reply({}, status=200),  # CAS ticket exchange via WebVPN
        reply({"code": 0}, status=200),  # iCourse info API verification
    ])
    vpn.session.post = Mock(side_effect=[
        reply({"data": [{"moduleCode": "userAndPwd", "authChainCode": "pwd-chain"}],
               "requestType": "chain_type"}),
        reply(password_challenge()),
        reply({"code": 200, "loginToken": "icourse-token"}),
        reply({}, html='var locationValue = "https://icourse.fudan.edu.cn/?ticket=example";'),
    ])
    assert vpn.authenticate_icourse("student", "password") is True
    assert vpn.session.post.call_count == 4
    otp_payload = vpn.session.post.call_args_list[2].kwargs["json"]
    assert otp_payload["authModuleCode"] == "userAndOtp"
    assert otp_payload["authPara"]["otpCode"] == "234567"


def test_incomplete_totp_challenge_blocks_verification(monkeypatch):
    monkeypatch.setattr(config, "TOTP_SECRET", RFC6238_BASE32)
    vpn = WebVPNSession()
    with pytest.raises(SecondFactorError, match="authentication context"):
        vpn._authenticate_totp(
            {"moduleCodes": ["userAndOtp"], "requestNumber": ""},
            "student", "lck", "webvpn", "chain_type"
        )


def test_totp_failure_does_not_loop_ten_logins(monkeypatch):
    vpn = Mock()
    vpn.login.return_value = True
    vpn.authenticate_icourse.side_effect = SecondFactorError("missing TOTP")
    ctor = Mock(return_value=vpn)
    monkeypatch.setattr(main, "WebVPNSession", ctor)
    with pytest.raises(SecondFactorError):
        main.login_with_retry(max_attempts=10)
    assert ctor.call_count == 1
    vpn.session.close.assert_called_once()
