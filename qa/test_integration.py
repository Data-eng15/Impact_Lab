"""Integration layer - live backend, real external APIs, real auth."""
import asyncio

import httpx
import pytest

from app.models import PaperMetadata

DEMO_PAPER = "An EEG-EMG correlation-based brain-computer interface"


def _get(url, **kw):
    kw.setdefault("timeout", 30)
    kw.setdefault("headers", {}).setdefault("ngrok-skip-browser-warning", "1")
    return httpx.get(url, **kw)


@pytest.mark.case("TC-I-001")
def test_health_local(local_api, record):
    r = _get(f"{local_api}/health")
    record(f"HTTP {r.status_code} {r.text[:80]}")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@pytest.mark.case("TC-I-002")
def test_health_tunnel(tunnel_api, record):
    try:
        r = _get(f"{tunnel_api}/health")
    except Exception as exc:
        pytest.fail(f"tunnel unreachable - the deployed frontend cannot call the backend: {exc}")
    record(f"HTTP {r.status_code}")
    assert r.status_code == 200


@pytest.mark.case("TC-I-003")
def test_protected_requires_auth(local_api, record):
    r = httpx.post(f"{local_api}/api/analyze", json={"query": DEMO_PAPER}, timeout=30)
    record(f"HTTP {r.status_code}")
    assert r.status_code in (401, 403), "protected endpoint served an anonymous request"


@pytest.mark.case("TC-I-004")
def test_tampered_token_rejected(local_api, auth_token, record):
    bad = auth_token[:-2] + ("aa" if not auth_token.endswith("aa") else "bb")
    r = httpx.post(f"{local_api}/api/analyze", json={"query": DEMO_PAPER},
                   headers={"Authorization": f"Bearer {bad}"}, timeout=30)
    record(f"HTTP {r.status_code}")
    assert r.status_code in (401, 403), "tampered JWT was accepted"


@pytest.mark.case("TC-I-005")
def test_valid_token_accepted(local_api, auth_headers, record):
    r = _get(f"{local_api}/api/profile", headers=dict(auth_headers))
    record(f"HTTP {r.status_code}")
    assert r.status_code not in (401, 403), "a validly-signed token was rejected"


@pytest.mark.case("TC-I-006")
def test_search_returns_results(local_api, record):
    r = _get(f"{local_api}/api/search", params={"q": DEMO_PAPER})
    record(f"HTTP {r.status_code}")
    assert r.status_code == 200
    body = r.json()
    candidates = body.get("candidates") or []
    titles = [c.get("title", "")[:60] for c in candidates[:2]]
    record(f"HTTP 200, {len(candidates)} candidates, top={titles}")
    assert len(candidates) >= 1, "search returned nothing for a known paper"
    assert any(c.get("title") for c in candidates), "candidates carry no titles"


def _run_retrieval():
    from app.services import initial_statuses, retrieval_node
    state = {
        "metadata": PaperMetadata(title=DEMO_PAPER, year=2018),
        "query": DEMO_PAPER, "logs": [], "statuses": initial_statuses(),
        "citation_count": 99, "routing_fields": [],
    }
    return asyncio.run(retrieval_node(state))


@pytest.fixture(scope="module")
def retrieval():
    return _run_retrieval()


@pytest.mark.case("TC-I-007")
def test_all_sources_invoked(retrieval, record):
    yields = retrieval["routing_metadata"].get("source_yield", {})
    record(yields)
    expected = {"semantic_scholar", "openalex_fallback", "openalex_enrichment", "downstream",
                "github", "patents", "pubmed", "clinical_trials", "soft_sciences"}
    assert expected <= set(yields), f"sources missing from the yield report: {expected - set(yields)}"


@pytest.mark.case("TC-I-008")
def test_evidence_is_diverse(retrieval, record):
    kinds = {}
    for e in retrieval["evidence"][:24]:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    record(f"{len(kinds)} distinct kinds in top-24: {kinds}")
    assert len(kinds) >= 4, f"evidence pool collapsed to {len(kinds)} kind(s): {kinds}"


@pytest.mark.case("TC-I-009")
def test_empty_sources_reported(retrieval, record):
    rm = retrieval["routing_metadata"]
    yields = rm.get("source_yield", {})
    empty = rm.get("sources_empty")
    record(f"sources_empty={empty}")
    assert empty is not None, "sources_empty is not reported at all"
    assert set(empty) == {k for k, v in yields.items() if v == 0}


def _fetch(fn, *args):
    async def _run():
        async with httpx.AsyncClient(headers={"User-Agent": "ReflectAI-QA/1.0"}) as c:
            return await fn(c, *args)
    return asyncio.run(_run())


@pytest.mark.case("TC-I-010")
def test_github_fetcher(record):
    from app.services import fetch_github_adoption
    ev, _ = _fetch(fetch_github_adoption, PaperMetadata(title=DEMO_PAPER, year=2018))
    record(f"{len(ev)} items, kinds={sorted({e.kind for e in ev})}")
    assert all(e.kind == "code" for e in ev)


@pytest.mark.case("TC-I-011")
def test_patent_fetcher(record):
    """An empty result is not proof of health here: `all()` is vacuously true on
    an empty list. If nothing came back, probe upstream and distinguish a real
    code fault from Google blocking this host."""
    from urllib.parse import quote

    from app.services import fetch_google_patents

    ev, _ = _fetch(fetch_google_patents, PaperMetadata(title=DEMO_PAPER, year=2018))
    if ev:
        record(f"{len(ev)} items, kinds={sorted({e.kind for e in ev})}")
        assert all(e.kind == "patent" for e in ev)
        return

    q = quote(f'"{DEMO_PAPER[:80]}"', safe="")
    probe = httpx.get(f"https://patents.google.com/xhr/query?url=q%3D{q}", timeout=20)
    record(f"0 items; upstream probe HTTP {probe.status_code}")
    if probe.status_code in (429, 503) or "Sorry" in probe.text[:200]:
        pytest.skip(
            f"Google Patents is blocking this host (HTTP {probe.status_code} anti-bot page). "
            "Scraped XHR endpoint, not a supported API - patent evidence will be empty."
        )
    pytest.fail(f"patent fetcher returned nothing but upstream answered {probe.status_code}")


@pytest.mark.case("TC-I-012")
def test_downstream_institutional(record):
    from app.services import fetch_downstream_impact
    ev, _ = _fetch(fetch_downstream_impact, PaperMetadata(title=DEMO_PAPER, year=2018), 99)
    kinds = sorted({e.kind for e in ev})
    record(f"{len(ev)} items, kinds={kinds}")
    assert ev, "downstream tracing returned no evidence at all"
    allowed = {"downstream", "industry_adoption", "policy_adoption",
               "clinical_adoption", "software_adoption"}
    assert set(kinds) <= allowed, f"unexpected downstream kind: {set(kinds) - allowed}"


@pytest.mark.case("TC-I-013")
def test_metrics_api(local_api, record):
    r = _get(f"{local_api}/api/metrics")
    assert r.status_code == 200
    body = r.json()
    record(f"keys={sorted(body)[:10]}")
    for key in ("current_rps", "cache_hit_rate_pct", "error_rate_pct", "health_status"):
        assert key in body, f"metrics payload missing '{key}'"


@pytest.mark.case("TC-I-014")
def test_dashboard_page(local_api, record):
    r = _get(f"{local_api}/dashboard")
    record(f"HTTP {r.status_code}, {len(r.text)} bytes")
    assert r.status_code == 200
    assert "<" in r.text and len(r.text) > 500


@pytest.mark.case("TC-I-015")
def test_malformed_body(local_api, auth_headers, record):
    r = httpx.post(f"{local_api}/api/analyze", json={"not_a_field": 123},
                   headers=dict(auth_headers), timeout=30)
    record(f"HTTP {r.status_code}")
    assert 400 <= r.status_code < 500, f"expected a validation error, got {r.status_code}"


@pytest.mark.case("TC-I-016")
def test_empty_search(local_api, record):
    r = _get(f"{local_api}/api/search", params={"q": ""})
    record(f"HTTP {r.status_code}")
    assert r.status_code != 500, "empty query produced a server error"


@pytest.mark.case("TC-I-017")
def test_cors_preflight(local_api, site, record):
    r = httpx.request("OPTIONS", f"{local_api}/api/analyze", timeout=30, headers={
        "Origin": site,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization,x-llm-provider",
    })
    allow = r.headers.get("access-control-allow-origin", "")
    record(f"HTTP {r.status_code}, allow-origin={allow!r}")
    assert r.status_code < 400, "CORS preflight rejected - the browser will block the real call"
    assert allow in (site, "*"), f"frontend origin not allowed: {allow!r}"
