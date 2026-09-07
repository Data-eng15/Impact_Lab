"""Security layer (suite 2) - adversarial probes against this application.

Authorised testing of the owner's own deployment. All probes are
non-destructive: no data is deleted, no third party is targeted, and the
rate-limit case uses the cheapest endpoint with a 60 s recovery window.
"""
import base64
import json
import time

import httpx
import jwt as pyjwt
import pytest

from app.auth import LINKEDIN_JWT_ALG, LINKEDIN_JWT_SECRET, create_orcid_token

SHIPPED_DEFAULT_SECRET = "change-me-in-production"
PROTECTED = "/api/analyze"
PAYLOAD = {"uid": "attacker", "email": "a@evil.test", "name": "Attacker", "provider": "orcid"}


def _post(local_api, token=None, path=PROTECTED, body=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.post(f"{local_api}{path}", json=body or {"query": "test"},
                      headers=headers, timeout=30)


@pytest.mark.case("TC-S-001")
def test_default_secret_not_in_use(local_api, record):
    forged = pyjwt.encode(PAYLOAD, SHIPPED_DEFAULT_SECRET, algorithm="HS256")
    r = _post(local_api, forged)
    configured_is_default = LINKEDIN_JWT_SECRET == SHIPPED_DEFAULT_SECRET
    record(f"forged-with-default-secret -> HTTP {r.status_code}; "
           f"configured secret is default={configured_is_default}")
    # The load-bearing assertion is what the SERVER does: an in-process constant
    # only reflects this test's own environment, not the deployment's.
    assert r.status_code in (401, 403), (
        "server accepted a token forged with the shipped default secret - "
        "anyone who reads the source can mint a valid session"
    )
    assert not configured_is_default, (
        "configured JWT secret is the shipped default"
    )


@pytest.mark.case("TC-S-002")
def test_alg_none_rejected(local_api, record):
    hdr = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    pl = base64.urlsafe_b64encode(json.dumps(PAYLOAD).encode()).rstrip(b"=")
    token = f"{hdr.decode()}.{pl.decode()}."
    r = _post(local_api, token)
    record(f"alg=none -> HTTP {r.status_code}")
    assert r.status_code in (401, 403), "unsigned alg=none token was accepted"


@pytest.mark.case("TC-S-003")
def test_signature_stripped_rejected(local_api, auth_token, record):
    stripped = ".".join(auth_token.split(".")[:2]) + "."
    r = _post(local_api, stripped)
    record(f"signature stripped -> HTTP {r.status_code}")
    assert r.status_code in (401, 403), "token with no signature was accepted"


@pytest.mark.case("TC-S-004")
def test_wrong_secret_rejected(local_api, record):
    forged = pyjwt.encode(PAYLOAD, "attacker-chosen-secret", algorithm="HS256")
    r = _post(local_api, forged)
    record(f"attacker-signed -> HTTP {r.status_code}")
    assert r.status_code in (401, 403), "token signed with an unknown secret was accepted"


@pytest.mark.case("TC-S-005")
def test_token_has_expiry(record):
    token = create_orcid_token(uid="u", email="e@x.test", name="N", orcid="0000")
    claims = pyjwt.decode(token, LINKEDIN_JWT_SECRET, algorithms=[LINKEDIN_JWT_ALG])
    record(f"claims={sorted(claims)}")
    assert "exp" in claims, (
        "session tokens carry no exp claim - a leaked token stays valid forever and cannot expire"
    )


@pytest.mark.case("TC-S-006")
def test_expired_token_rejected(local_api, record):
    expired = pyjwt.encode({**PAYLOAD, "exp": int(time.time()) - 3600},
                           LINKEDIN_JWT_SECRET, algorithm=LINKEDIN_JWT_ALG)
    r = _post(local_api, expired)
    record(f"expired token -> HTTP {r.status_code}")
    assert r.status_code in (401, 403), "a token whose exp is in the past was accepted"


@pytest.mark.case("TC-S-007")
def test_history_not_pivotable(local_api, auth_headers, record):
    r = httpx.get(f"{local_api}/api/history", params={"uid": "victim-user", "limit": 5},
                  headers=dict(auth_headers), timeout=30)
    record(f"HTTP {r.status_code}, uid in body={r.json().get('uid') if r.status_code == 200 else 'n/a'}")
    if r.status_code == 200:
        assert r.json().get("uid") != "victim-user", "uid query parameter overrode the token identity (IDOR)"


@pytest.mark.case("TC-S-008")
def test_dataset_no_user_identifiers(local_api, record):
    r = httpx.get(f"{local_api}/api/dataset", timeout=30)
    if r.status_code in (401, 403):
        record(f"HTTP {r.status_code} - endpoint requires auth")
        return
    body = r.text
    leaky = [f for f in ("uid", "email", "orcid") if f'"{f}"' in body.lower()]
    record(f"HTTP {r.status_code}, {len(body)} bytes, identifier fields present={leaky or 'none'}")
    assert not leaky, (
        f"/api/dataset is unauthenticated and exposes per-user identifier field(s): {leaky}"
    )


@pytest.mark.case("TC-S-009")
def test_llm_endpoints_need_auth(local_api, record):
    codes = {}
    for path in ("/api/analyze", "/api/compose", "/api/evaluate", "/api/ref/beta"):
        codes[path] = _post(local_api, None, path=path).status_code
    record(codes)
    open_paths = [p for p, c in codes.items() if c not in (401, 403, 404, 422)]
    assert not open_paths, f"expensive endpoints reachable anonymously: {open_paths}"


@pytest.mark.case("TC-S-010")
def test_path_traversal(local_api, record):
    codes = {}
    for p in ("../../../../etc/passwd", "....//....//etc/passwd", "%2e%2e%2fetc%2fpasswd"):
        r = httpx.get(f"{local_api}/api/search", params={"q": p}, timeout=30)
        codes[p[:18]] = r.status_code
        assert r.status_code != 500, f"traversal payload caused a server error: {p!r}"
        assert "root:x:" not in r.text, f"filesystem content returned for {p!r}"
    record(codes)


@pytest.mark.case("TC-S-011")
def test_no_unescaped_script_reflection(local_api, record):
    payload = "<script>alert(1)</script>"
    r = httpx.get(f"{local_api}/api/search", params={"q": payload}, timeout=30)
    ctype = r.headers.get("content-type", "")
    reflected_html = payload in r.text and "html" in ctype.lower()
    record(f"HTTP {r.status_code}, content-type={ctype}, reflected_in_html={reflected_html}")
    assert not reflected_html, "script payload reflected into an HTML response context"


@pytest.mark.case("TC-S-012")
def test_deeply_nested_json(local_api, auth_headers, record):
    nested: dict = {"a": 1}
    for _ in range(500):
        nested = {"a": nested}
    try:
        r = httpx.post(f"{local_api}{PROTECTED}", json=nested, headers=dict(auth_headers), timeout=45)
        code = r.status_code
    except Exception as exc:
        code = f"client-side {type(exc).__name__}"
    health = httpx.get(f"{local_api}/health", timeout=15).status_code
    record(f"nested -> {code}; health after -> {health}")
    assert health == 200, "service unhealthy after a deeply nested payload"


@pytest.mark.case("TC-S-013")
def test_rate_limiting(local_api, record):
    """Search is limited to 20 req/60s per IP. The bucket drains in 60 s."""
    codes = []
    for _ in range(30):
        codes.append(httpx.get(f"{local_api}/api/search", params={"q": "rate limit probe"},
                               timeout=20).status_code)
        if codes[-1] == 429:
            break
    record(f"{len(codes)} requests, throttled at #{len(codes) if 429 in codes else 'never'}")
    assert 429 in codes, f"no throttling after {len(codes)} rapid requests - abuse control not enforced"


@pytest.mark.case("TC-S-014")
def test_no_stack_traces(local_api, auth_headers, record):
    bodies = []
    bodies.append(httpx.post(f"{local_api}{PROTECTED}", json={"bad": object.__doc__},
                             headers=dict(auth_headers), timeout=30).text)
    bodies.append(httpx.get(f"{local_api}/api/search", params={"q": "\x00\x01"}, timeout=30).text)
    bodies.append(httpx.get(f"{local_api}/api/does-not-exist", timeout=20).text)
    blob = "\n".join(bodies)
    markers = [m for m in ("Traceback (most recent call last)", "site-packages",
                           "M:\\\\REFlect_AI", "/app/services.py") if m in blob]
    record(f"scanned {len(blob)} bytes; leak markers={markers or 'none'}")
    assert not markers, f"internal detail leaked in error responses: {markers}"


@pytest.mark.case("TC-S-015")
def test_cors_not_wildcard_with_credentials(local_api, record):
    r = httpx.request("OPTIONS", f"{local_api}{PROTECTED}", timeout=30, headers={
        "Origin": "https://attacker.example",
        "Access-Control-Request-Method": "POST",
    })
    origin = r.headers.get("access-control-allow-origin", "")
    creds = r.headers.get("access-control-allow-credentials", "")
    record(f"attacker origin -> allow-origin={origin!r}, allow-credentials={creds!r}")
    assert not (origin == "*" and creds.lower() == "true"), \
        "wildcard origin combined with credentials - any site could call the API with a user's cookies"
