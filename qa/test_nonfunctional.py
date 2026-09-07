"""Non-functional layer - performance, load, security and resilience."""
import asyncio
import concurrent.futures as cf
import re
import subprocess
import time

import httpx
import pytest

from app.models import PaperMetadata


@pytest.mark.case("TC-N-001")
def test_health_latency(local_api, record):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = httpx.get(f"{local_api}/health", timeout=15)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    mean = sum(times) / len(times)
    record(f"mean={mean:.1f} ms, max={max(times):.1f} ms over {len(times)} calls")
    assert mean < 1000, f"health endpoint mean latency {mean:.0f} ms exceeds the 1s budget"


@pytest.mark.case("TC-N-002")
def test_page_load_budget(driver, site, record):
    t0 = time.perf_counter()
    driver.get(site)
    driver.execute_script("return document.readyState")
    elapsed = time.perf_counter() - t0
    record(f"{elapsed:.2f} s to loaded")
    assert elapsed < 10, f"landing page took {elapsed:.1f}s - too slow for a live demo"


@pytest.mark.case("TC-N-003")
def test_cache_hit_speed(record):
    from app.main import _cache_get, _cache_put
    cache: dict = {}
    _cache_put(cache, "qa-key", {"payload": "x" * 1000})
    t0 = time.perf_counter()
    for _ in range(1000):
        _cache_get(cache, "qa-key", 3600)
    per_call_ms = (time.perf_counter() - t0) * 1000 / 1000
    record(f"{per_call_ms:.4f} ms per cache hit")
    assert per_call_ms < 10, f"cache hit unexpectedly slow: {per_call_ms:.2f} ms"


@pytest.mark.case("TC-N-004")
def test_concurrent_health(local_api, record):
    def one():
        return httpx.get(f"{local_api}/health", timeout=30).status_code

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=20) as pool:
        codes = list(pool.map(lambda _: one(), range(20)))
    elapsed = time.perf_counter() - t0
    bad = [c for c in codes if c != 200]
    record(f"20 concurrent in {elapsed:.2f}s, non-200={len(bad)}")
    assert not bad, f"{len(bad)} of 20 concurrent requests failed: {set(bad)}"


@pytest.mark.case("TC-N-005")
def test_injection_payloads(local_api, record):
    payloads = [
        "test'; DROP TABLE papers; --",
        '{"$or": [{"x": "1"}]}',
        "<script>alert(1)</script>",
        "../../../../etc/passwd",
    ]
    codes = {}
    for p in payloads:
        r = httpx.get(f"{local_api}/api/search", params={"q": p}, timeout=30)
        codes[p[:24]] = r.status_code
        assert r.status_code != 500, f"payload caused a server error: {p!r}"
    record(codes)


@pytest.mark.case("TC-N-006")
def test_oversized_input(local_api, record):
    big = "a" * 20000
    r = httpx.get(f"{local_api}/api/search", params={"q": big}, timeout=45)
    health = httpx.get(f"{local_api}/health", timeout=15)
    record(f"oversized -> HTTP {r.status_code}; health after -> {health.status_code}")
    assert health.status_code == 200, "service unhealthy after an oversized request"


@pytest.mark.case("TC-N-007")
def test_no_secret_leakage(local_api, record):
    bodies = []
    for path in ("/health", "/api/metrics", "/dashboard"):
        try:
            bodies.append(httpx.get(f"{local_api}{path}", timeout=20).text)
        except Exception:
            pass
    blob = "\n".join(bodies)
    patterns = {
        "groq key": r"gsk_[A-Za-z0-9]{20,}",
        "google key": r"AIza[A-Za-z0-9_\-]{30,}",
        "bearer jwt": r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}",
        "orcid secret": r"\bclient_secret\b\s*[:=]\s*\S+",
    }
    leaks = {name: True for name, pat in patterns.items() if re.search(pat, blob)}
    record(f"scanned {len(blob)} bytes across 3 endpoints; leaks={leaks or 'none'}")
    assert not leaks, f"possible secret exposure in API responses: {list(leaks)}"


@pytest.mark.case("TC-N-008")
def test_graceful_source_degradation(record):
    """The live run already exercises this: Semantic Scholar is rate-limited on
    the anonymous pool, so the pipeline must still deliver evidence without it."""
    from app.services import initial_statuses, retrieval_node

    state = {
        "metadata": PaperMetadata(title="An EEG-EMG correlation-based brain-computer interface",
                                  year=2018),
        "query": "An EEG-EMG correlation-based brain-computer interface",
        "logs": [], "statuses": initial_statuses(),
        "citation_count": 99, "routing_fields": [],
    }
    out = asyncio.run(retrieval_node(state))
    yields = out["routing_metadata"].get("source_yield", {})
    empty = out["routing_metadata"].get("sources_empty", [])
    record(f"{len(out['evidence'])} items despite {len(empty)} empty source(s): {empty}")
    assert out["evidence"], "pipeline returned no evidence at all"
    assert sum(yields.values()) > 0


@pytest.mark.case("TC-N-009")
def test_backend_supervised(record):
    try:
        proc = subprocess.run(
            ["schtasks", "/query", "/tn", "ReflectAI-Backend", "/fo", "LIST"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        pytest.skip(f"schtasks unavailable: {exc}")
    out = proc.stdout
    record(next((ln.strip() for ln in out.splitlines() if "Status" in ln), out[:120]))
    assert "Running" in out, "backend supervisor task is not running - no auto-restart during the demo"
