"""Shared fixtures and result capture for the REFlect AI QA suite.

Each test is tagged with @pytest.mark.case("TC-x-000"). The hook below records
the outcome, duration and failure reason per case id into results.json, which
build_report.py joins with testcases.py to produce the Excel document.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

QA_DIR = Path(__file__).parent
REPO = QA_DIR.parent
sys.path.insert(0, str(REPO / "backend"))

LOCAL_API = os.getenv("QA_LOCAL_API", "http://127.0.0.1:8000")
TUNNEL_API = os.getenv("QA_TUNNEL_API", "https://gravy-cardboard-brigade.ngrok-free.dev")
SITE = os.getenv("QA_SITE", "https://reflect-ai-a6r.pages.dev")

_RESULTS: dict[str, dict] = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "case(id): link this test to a QA catalogue case id")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    marker = item.get_closest_marker("case")
    if not marker:
        return
    case_id = marker.args[0]
    if report.passed:
        status, detail = "PASS", ""
    elif report.skipped:
        status, detail = "BLOCKED", str(report.longrepr)[:600]
    else:
        status, detail = "FAIL", str(report.longrepr)[:1200]
    _RESULTS[case_id] = {
        "status": status,
        "duration_s": round(report.duration, 3),
        "detail": detail,
        "actual": getattr(item, "_qa_actual", ""),
        "test": item.nodeid,
    }


def pytest_sessionfinish(session, exitstatus):
    out = QA_DIR / "results.json"
    out.write_text(json.dumps(_RESULTS, indent=2), encoding="utf-8")
    print(f"\n[qa] recorded {len(_RESULTS)} case results -> {out}")


@pytest.fixture
def record(request):
    """Attach an observed value to the current test, shown as 'Actual result'."""
    def _record(value):
        request.node._qa_actual = str(value)[:500]
    return _record


@pytest.fixture(scope="session")
def local_api():
    return LOCAL_API.rstrip("/")


@pytest.fixture(scope="session")
def tunnel_api():
    return TUNNEL_API.rstrip("/")


@pytest.fixture(scope="session")
def site():
    return SITE.rstrip("/")


@pytest.fixture(scope="session")
def auth_token():
    """Mint a valid JWT in-process. ORCID OAuth is a real third-party flow and
    cannot (and should not) be driven by a browser test, so authenticated
    coverage lives at the API layer instead."""
    from app.auth import create_orcid_token
    return create_orcid_token(uid="qa-user", email="qa@example.com",
                              name="QA Runner", orcid="0000-0002-1825-0097")


@pytest.fixture(scope="session")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="session")
def driver():
    """Headless Edge. Chrome is not installed on the demo VM."""
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options

    opts = Options()
    for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                "--disable-gpu", "--window-size=1440,900"):
        opts.add_argument(arg)
    opts.set_capability("ms:loggingPrefs", {"browser": "ALL"})
    drv = webdriver.Edge(options=opts)
    drv.set_page_load_timeout(60)
    yield drv
    drv.quit()
