"""Functional layer - Selenium (headless Edge) against the deployed site.

Scope note: ORCID sign-in is a real third-party OAuth flow. It is deliberately
not automated here - browser tests cover everything up to the auth boundary,
and the authenticated surface is covered at the API layer in test_integration.
"""
import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

SIGNIN_XPATH = (
    "//*[self::button or self::a]"
    "[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),"
    " 'sign in')]"
)


@pytest.fixture(scope="module")
def landing(driver, site):
    driver.get(site)
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    return driver


@pytest.mark.case("TC-F-001")
def test_landing_title(landing, record):
    record(f"title={landing.title!r}")
    assert "REFlect" in landing.title, f"unexpected page title: {landing.title!r}"


@pytest.mark.case("TC-F-002")
def test_hero_copy_visible(landing, record):
    body = landing.find_element(By.TAG_NAME, "body").text
    record(body[:160].replace("\n", " | "))
    assert "impact" in body.lower(), "hero copy about research impact not rendered"
    assert len(body) > 300, "page body is suspiciously empty"


@pytest.mark.case("TC-F-003")
def test_signin_cta_present(landing, record):
    els = landing.find_elements(By.XPATH, SIGNIN_XPATH)
    visible = [e for e in els if e.is_displayed()]
    record(f"{len(visible)} visible sign-in control(s): {[e.text.strip()[:40] for e in visible[:3]]}")
    assert visible, "no visible sign-in control - the demo has no entry point"
    assert visible[0].is_enabled()


@pytest.mark.case("TC-F-004")
def test_signin_modal_opens(driver, site, record):
    driver.get(site)
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    btns = [e for e in driver.find_elements(By.XPATH, SIGNIN_XPATH) if e.is_displayed()]
    assert btns, "no sign-in control to click"
    driver.execute_script("arguments[0].click();", btns[0])
    try:
        WebDriverWait(driver, 15).until(
            lambda d: "orcid" in d.find_element(By.TAG_NAME, "body").text.lower()
        )
    except TimeoutException:
        pytest.fail("sign-in modal did not present ORCID guidance within 15s")
    record("modal opened with ORCID guidance")


@pytest.mark.case("TC-F-005")
def test_source_strip(landing, record):
    body = landing.find_element(By.TAG_NAME, "body").text
    found = [s for s in ("OpenAlex", "Semantic Scholar", "Crossref", "ORCID", "GitHub")
             if s.lower() in body.lower()]
    record(f"sources shown: {found}")
    assert len(found) >= 3, f"credibility strip thin - only found {found}"


@pytest.mark.case("TC-F-006")
def test_no_js_errors(landing, record):
    severe = [e for e in landing.get_log("browser") if e.get("level") == "SEVERE"]
    msgs = [e.get("message", "")[:120] for e in severe]
    record(f"{len(severe)} SEVERE console entries: {msgs[:3]}")
    assert not severe, f"uncaught JS errors on load: {msgs[:3]}"


@pytest.mark.case("TC-F-007")
def test_ops_dashboard_renders(driver, local_api, record):
    driver.get(f"{local_api}/dashboard")
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    body = driver.find_element(By.TAG_NAME, "body").text
    hits = [k for k in ("RPS", "Latency", "Cache", "Error", "Uptime") if k.lower() in body.lower()]
    record(f"metric labels present: {hits}")
    assert len(hits) >= 3, f"ops dashboard did not render its metric cards: {hits}"


@pytest.mark.case("TC-F-008")
def test_mobile_no_overflow(driver, site, record):
    driver.set_window_size(390, 844)
    try:
        driver.get(site)
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        scroll_w = driver.execute_script("return document.documentElement.scrollWidth")
        client_w = driver.execute_script("return document.documentElement.clientWidth")
        record(f"scrollWidth={scroll_w} clientWidth={client_w}")
        assert scroll_w <= client_w + 8, f"horizontal overflow at mobile width: {scroll_w} > {client_w}"
    finally:
        driver.set_window_size(1440, 900)


@pytest.mark.case("TC-F-009")
def test_unknown_route(driver, site, record):
    driver.get(f"{site}/this-route-does-not-exist-qa")
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    body = driver.find_element(By.TAG_NAME, "body").text.strip()
    record(f"{len(body)} chars rendered: {body[:80]!r}")
    assert len(body) > 20, "unknown route rendered a blank page"
