"""Unit layer - pure logic, no network, no running server. Fast regression guard."""
import pytest

from app.models import EvidenceItem
from app.services import (
    build_relevance_filter,
    dedupe_evidence,
    diversify_evidence,
    meaningful_title_terms,
)
from app.domain_classifier import ALL_SOURCES, plan_routing
from app.hf_synthesis import _deflate, _trim_words


def _pool():
    """Mirrors the real concatenation order: citations flood the front."""
    return (
        [EvidenceItem(title=f"down{i}", kind="downstream", source="OpenAlex") for i in range(5)]
        + [EvidenceItem(title=f"cite{i}", kind="citation", source="Semantic Scholar") for i in range(25)]
        + [EvidenceItem(title="repo", kind="code", source="GitHub")]
        + [EvidenceItem(title="pat", kind="patent", source="Google Patents")]
        + [EvidenceItem(title="trial", kind="clinical", source="ClinicalTrials.gov")]
        + [EvidenceItem(title=f"pol{i}", kind="ssh", source="GOV.UK") for i in range(4)]
    )


@pytest.mark.case("TC-U-001")
def test_diversify_surfaces_all_kinds(record):
    pool = _pool()
    before = {e.kind for e in pool[:24]}
    after = {e.kind for e in diversify_evidence(pool)[:24]}
    record(f"before={sorted(before)} after={sorted(after)}")
    assert {"code", "patent", "clinical", "ssh"} <= after, (
        f"code/patent/clinical/policy still truncated out of the top-24: {sorted(after)}"
    )


@pytest.mark.case("TC-U-002")
def test_diversify_is_lossless(record):
    pool = _pool()
    out = diversify_evidence(pool)
    record(f"in={len(pool)} out={len(out)}")
    assert len(out) == len(pool)
    assert sorted(e.title for e in out) == sorted(e.title for e in pool)


@pytest.mark.case("TC-U-003")
def test_diversify_edge_cases(record):
    assert diversify_evidence([]) == []
    single = [EvidenceItem(title=f"c{i}", kind="citation", source="S2") for i in range(4)]
    out = diversify_evidence(single)
    record(f"empty=ok single_kind_order_preserved={[e.title for e in out]}")
    assert [e.title for e in out] == [e.title for e in single]


@pytest.mark.case("TC-U-004")
def test_relevance_keeps_matching(record):
    f = build_relevance_filter("Deep Residual Learning for Image Recognition")
    got = f("Residual networks behave like ensembles", "")
    record(f"matching item kept = {got}")
    assert got is True


@pytest.mark.case("TC-U-005")
def test_relevance_rejects_unrelated(record):
    f = build_relevance_filter("Deep Residual Learning for Image Recognition")
    got = f("Eighteenth century maritime trade in the Baltic", "")
    record(f"unrelated item kept = {got}")
    assert got is False


@pytest.mark.case("TC-U-006")
def test_relevance_fails_open_without_terms(record):
    f = build_relevance_filter("The and of a")
    record(f"terms={meaningful_title_terms('The and of a')}")
    assert f("literally anything", "") is True


@pytest.mark.case("TC-U-007")
def test_dedupe(record):
    items = [
        EvidenceItem(title="A", url="http://x/1", kind="citation", source="S2"),
        EvidenceItem(title="A dup", url="http://x/1", kind="citation", source="S2"),
        EvidenceItem(title="B", url="http://x/2", kind="citation", source="S2"),
    ]
    out = dedupe_evidence(items)
    record(f"{len(items)} -> {len(out)}")
    assert len(out) == 2


@pytest.mark.case("TC-U-008")
def test_word_cap(record):
    long_text = " ".join(["word"] * 400) + ". Another sentence here."
    out = _trim_words(long_text, hard_max=150)
    n = len(out.split())
    record(f"{len(long_text.split())} words -> {n} words")
    assert n <= 150, f"summary exceeded the 150-word REF cap: {n}"


@pytest.mark.case("TC-U-009")
def test_deflate_puffery(record):
    out = _deflate("This had a profound influence and was transformative for the field.")
    record(out)
    assert "profound influence" not in out.lower()


@pytest.mark.case("TC-U-010")
def test_routing_calls_all_when_ambiguous(record):
    plan = plan_routing(fields=[], title="A study of things", abstract="")
    record(f"reason={plan['reason']} called={len(plan['sources_called'])}/{len(ALL_SOURCES)}")
    assert plan["reason"] == "no_clear_domain"
    assert set(plan["sources_called"]) == set(ALL_SOURCES)


@pytest.mark.case("TC-U-011")
def test_patents_never_skipped(record):
    titles = [
        "Deep learning for image recognition",
        "A randomised controlled trial of a new drug",
        "Social policy and welfare reform in the UK",
    ]
    missing = [t for t in titles
               if "patents" not in plan_routing(fields=[], title=t, abstract="")["sources_called"]]
    record(f"titles checked={len(titles)} missing_patents={missing}")
    assert not missing
