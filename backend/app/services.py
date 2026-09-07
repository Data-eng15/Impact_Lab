from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .models import AgentState, AgentStatus, EvidenceItem, ImpactSection, PaperMetadata, TraceLog
from .rag import index_and_retrieve
from .hf_synthesis import hf_synthesizer, build_references
from .domain_classifier import plan_routing, ALL_SOURCES, OPTIONAL_SOURCES

# Below this many post-dedup evidence items, a routed (narrowed) retrieval pass
# falls back to querying the remaining sources to protect recall.
MIN_ROUTED_EVIDENCE = 5


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._();/:A-Z0-9]+", re.IGNORECASE)
GENERIC_TITLE_TERMS = {
    "deep",
    "learning",
    "machine",
    "analysis",
    "survey",
    "review",
    "introduction",
    "method",
    "methods",
    "model",
    "models",
    "data",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(agent: str, message: str, **data: Any) -> TraceLog:
    return TraceLog(timestamp=now_iso(), agent=agent, message=message, data=data)


def status(name: str, label: str, state: AgentState, detail: str) -> AgentStatus:
    return AgentStatus(name=name, label=label, state=state, detail=detail)


def normalize_query(query: str) -> str:
    return query.strip()


def extract_doi(query: str) -> str | None:
    match = DOI_PATTERN.search(query)
    return match.group(0) if match else None


def meaningful_title_terms(title: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", title.lower())
        if len(term) > 3 and term not in GENERIC_TITLE_TERMS
    ]


def build_relevance_filter(title: str, topics: list[str] | None = None):
    """Reject search hits sharing no distinctive term with the paper.

    Loose free-text searches (policy portals, trial registries, PubMed word
    search) otherwise collect topically-unrelated results — e.g. a CS
    "attention" paper should not collect welfare-policy documents.
    """
    paper_terms = set(meaningful_title_terms(title))
    if topics:
        for t in topics:
            paper_terms.update(
                w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 4
            )

    def _relevant(item_title: str, item_snippet: str = "") -> bool:
        if not paper_terms:
            return True
        hay = f"{item_title} {item_snippet}".lower()
        return any(term in hay for term in paper_terms)

    return _relevant


def authors_from_crossref(work: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for author in work.get("author", [])[:8]:
        given = author.get("given", "")
        family = author.get("family", "")
        name = " ".join(part for part in [given, family] if part).strip()
        if name:
            authors.append(name)
    return authors


def year_from_crossref(work: dict[str, Any]) -> int | None:
    parts = work.get("published-print", {}).get("date-parts") or work.get("published-online", {}).get("date-parts") or work.get("created", {}).get("date-parts")
    try:
        return int(parts[0][0])
    except Exception:
        return None


async def _semantic_scholar_title_search(
    client: httpx.AsyncClient, query: str
) -> tuple[PaperMetadata | None, int]:
    """
    Resolve a title query to the highest-impact paper using Semantic Scholar.

    Strategy: autocomplete (lightweight, rarely rate-limited) → direct paper fetch
    by the returned SS paper ID. Falls back to the /paper/search endpoint if needed.
    """
    ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": ss_key} if ss_key else {}
    fields  = "title,year,authors,externalIds,citationCount,abstract"

    # ── Step 1: autocomplete to get candidate paper IDs ──────────────────────
    ac_ids: list[str] = []
    try:
        ac_resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/autocomplete",
            params={"query": query},
            headers=headers,
            timeout=10,
        )
        if ac_resp.is_success:
            ac_ids = [m["id"] for m in ac_resp.json().get("matches", [])[:5] if m.get("id")]
    except Exception:
        pass

    # ── Step 2: batch-fetch ALL candidate details in ONE request, retried
    #            once on 429. Far more reliable than sequential per-ID calls,
    #            so the canonical high-impact paper is never silently dropped.
    best_meta: PaperMetadata | None = None
    best_cites = 0
    if ac_ids:
        for attempt in range(2):
            try:
                br = await client.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": fields},
                    json={"ids": ac_ids},
                    headers=headers,
                    timeout=15,
                )
                if br.status_code == 429 and attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                if not br.is_success:
                    break
                for p in br.json():
                    if not p:
                        continue
                    doi   = (p.get("externalIds") or {}).get("DOI")
                    arxiv = (p.get("externalIds") or {}).get("ArXiv")
                    # Skip predatory republication-mill DOIs
                    if doi and doi.lower().startswith("10.65215/"):
                        continue
                    cites = p.get("citationCount") or 0
                    if cites > best_cites:
                        best_meta = PaperMetadata(
                            title=p.get("title", "Unknown"),
                            authors=[a.get("name", "") for a in (p.get("authors") or [])[:8]],
                            year=p.get("year"),
                            doi=doi or (f"10.48550/arXiv.{arxiv}" if arxiv else None),
                            abstract=p.get("abstract"),
                        )
                        best_cites = cites
                break
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                break

    if best_meta and best_cites > 0:
        return best_meta, best_cites

    # ── Step 3: fallback to /paper/search (may be rate-limited) ─────────────
    for attempt in range(2):
        try:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 5, "fields": fields},
                headers=headers,
                timeout=12,
            )
            if resp.status_code == 429:
                await asyncio.sleep(3)
                continue
            if not resp.is_success:
                break
            data = resp.json().get("data", [])
            if data:
                best = max(data, key=lambda p: p.get("citationCount") or 0)
                doi   = (best.get("externalIds") or {}).get("DOI")
                arxiv = (best.get("externalIds") or {}).get("ArXiv")
                meta = PaperMetadata(
                    title=best.get("title", "Unknown"),
                    authors=[a.get("name", "") for a in (best.get("authors") or [])[:8]],
                    year=best.get("year"),
                    doi=doi or (f"10.48550/arXiv.{arxiv}" if arxiv else None),
                    abstract=best.get("abstract"),
                )
                return meta, best.get("citationCount") or 0
        except Exception:
            break
    return None, 0


async def _openalex_title_search(
    client: httpx.AsyncClient, query: str
) -> tuple[PaperMetadata | None, int]:
    """Search OpenAlex by title, filtered to exclude very-recent low-citation republications,
    sorted by citation count descending — always surfaces the highest-impact version."""
    try:
        from datetime import datetime
        current_year = datetime.now().year
        fields = "title,doi,publication_year,cited_by_count,authorships,abstract_inverted_index"
        # Fetch top 10 by citations for the title query
        resp = await client.get(
            "https://api.openalex.org/works",
            params={
                "filter": f"title.search:{query}",
                "sort": "cited_by_count:desc",
                "per-page": 10,
                "select": fields,
            },
            timeout=12,
        )
        if not resp.is_success:
            return None, 0
        results = resp.json().get("results", [])
        if not results:
            return None, 0

        # Filter out (a) predatory republication-mill DOIs, and (b) papers
        # published in the last 2 years with <500 citations (likely recent
        # republications of classic papers).
        def _w_doi(w: dict) -> str:
            return (w.get("doi") or "").lower().replace("https://doi.org/", "")
        filtered = [
            w for w in results
            if not _w_doi(w).startswith("10.65215/")
            and not (
                (w.get("publication_year") or 0) >= current_year - 1
                and (w.get("cited_by_count") or 0) < 500
            )
        ]
        candidates = filtered if filtered else results
        best = max(candidates, key=lambda w: w.get("cited_by_count") or 0)

        doi_raw = best.get("doi", "") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (best.get("authorships") or [])[:8]
        ]
        abstract = inverted_abstract_to_text(best.get("abstract_inverted_index"))
        meta = PaperMetadata(
            title=best.get("title", "Unknown"),
            authors=authors,
            year=best.get("publication_year"),
            doi=doi,
            abstract=abstract,
        )
        return meta, best.get("cited_by_count") or 0
    except Exception:
        return None, 0


# Module-level cache: resolved metadata + citation count, keyed by id.
# Smooths over Semantic Scholar's aggressive anonymous rate limits so a paper
# resolved once stays resolved for an hour.
_meta_cache: dict[str, tuple[float, "PaperMetadata", int]] = {}
_META_TTL = 3600


def _meta_cache_get(key: str):
    import time as _t
    hit = _meta_cache.get(key)
    if hit and (_t.time() - hit[0]) < _META_TTL:
        return hit[1], hit[2]
    return None


def _meta_cache_put(key: str, meta: "PaperMetadata", cites: int) -> None:
    import time as _t
    _meta_cache[key] = (_t.time(), meta, cites)


def _ss_paper_to_meta(p: dict) -> tuple[PaperMetadata | None, int]:
    if not p:
        return None, 0
    ext = p.get("externalIds") or {}
    doi = ext.get("DOI") or (f"10.48550/arXiv.{ext['ArXiv']}" if ext.get("ArXiv") else None)
    meta = PaperMetadata(
        title=p.get("title", "Unknown"),
        authors=[a.get("name", "") for a in (p.get("authors") or [])[:8]],
        year=p.get("year"),
        doi=doi,
        abstract=p.get("abstract"),
    )
    return meta, p.get("citationCount") or 0


async def _ss_get_with_retries(client: httpx.AsyncClient, url: str, headers: dict,
                               attempts: int = 4) -> dict | None:
    """GET a Semantic Scholar paper endpoint, retrying on 429 with backoff.
    Anonymous SS access shares a global ~1 req/s pool, so several retries are
    often needed before a request lands in an open window."""
    fields = "title,year,authors,externalIds,citationCount,abstract"
    for attempt in range(attempts):
        try:
            r = await client.get(url, params={"fields": fields}, headers=headers, timeout=12)
            if r.status_code == 429:
                if attempt < attempts - 1:
                    await asyncio.sleep(1.5 + attempt)  # 1.5, 2.5, 3.5s
                    continue
                return None
            if r.is_success:
                return r.json()
            return None
        except Exception:
            if attempt < attempts - 1:
                await asyncio.sleep(1.0)
                continue
            return None
    return None


async def _resolve_by_arxiv_native(client: httpx.AsyncClient, arxiv_id: str) -> PaperMetadata | None:
    """Resolve an arXiv ID via arXiv's OWN API — free, no key, no rate limit.
    This is the reliable primary path for arXiv DOIs; Semantic Scholar is only
    used afterwards for citation-count enrichment."""
    try:
        r = await client.get("https://export.arxiv.org/api/query",
                             params={"id_list": arxiv_id}, timeout=15)
        if not r.is_success:
            return None
        xml = r.text
        entries = xml.split("<entry>")
        if len(entries) < 2:
            return None
        e = entries[1]
        t = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
        title = re.sub(r"\s+", " ", t.group(1)).strip() if t else "Unknown"
        p = re.search(r"<published>(\d{4})", e)
        year = int(p.group(1)) if p else None
        authors = [a.strip() for a in re.findall(r"<name>(.*?)</name>", e)][:8]
        summ = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
        abstract = re.sub(r"\s+", " ", summ.group(1)).strip() if summ else None
        return PaperMetadata(
            title=title, authors=authors, year=year,
            doi=f"10.48550/arXiv.{arxiv_id}", abstract=abstract,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
        )
    except Exception:
        return None


async def _resolve_by_arxiv(client: httpx.AsyncClient, arxiv_id: str) -> PaperMetadata | None:
    """Resolve an arXiv ID: arXiv native API first (reliable), SS fallback (cached)."""
    cache_key = f"arxiv:{arxiv_id.lower()}"
    cached = _meta_cache_get(cache_key)
    if cached:
        return cached[0]

    # 1. arXiv native API — reliable, unthrottled
    meta = await _resolve_by_arxiv_native(client, arxiv_id)
    if meta:
        _meta_cache_put(cache_key, meta, 0)
        return meta

    # 2. Semantic Scholar fallback
    ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": ss_key} if ss_key else {}
    data = await _ss_get_with_retries(
        client, f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}", headers)
    meta, cites = _ss_paper_to_meta(data) if data else (None, 0)
    if meta:
        _meta_cache_put(cache_key, meta, cites)
    return meta


async def _resolve_by_ss_doi(client: httpx.AsyncClient, doi: str) -> PaperMetadata | None:
    """Resolve a DOI to metadata via Semantic Scholar (cached, retried)."""
    cache_key = f"doi:{doi.lower()}"
    cached = _meta_cache_get(cache_key)
    if cached:
        return cached[0]
    ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": ss_key} if ss_key else {}
    data = await _ss_get_with_retries(
        client, f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}", headers)
    meta, cites = _ss_paper_to_meta(data) if data else (None, 0)
    if meta:
        _meta_cache_put(cache_key, meta, cites)
    return meta


async def fetch_crossref(client: httpx.AsyncClient, query: str) -> tuple[PaperMetadata | None, list[TraceLog]]:
    logs = [log("Metadata", "Resolving input with CrossRef")]
    doi = extract_doi(query)
    try:
        if doi:
            # arXiv DOIs (10.48550/arXiv.XXXX) are registered with DataCite,
            # NOT CrossRef — resolve those via Semantic Scholar instead.
            arxiv_m = re.search(r"10\.48550/arxiv\.(.+)$", doi, re.IGNORECASE)
            if arxiv_m:
                arxiv_id = arxiv_m.group(1)
                meta = await _resolve_by_arxiv(client, arxiv_id)
                if meta:
                    logs.append(log("Metadata", "Metadata resolved via arXiv/Semantic Scholar",
                                    title=meta.title, doi=meta.doi))
                    return meta, logs
                # fall through to CrossRef attempt if SS failed

            # Standard DOI lookup — CrossRef is authoritative
            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            response = await client.get(url, timeout=12)
            if response.is_success:
                work = response.json()["message"]
                title = (work.get("title") or ["Unknown title"])[0]
                metadata = PaperMetadata(
                    title=title,
                    authors=authors_from_crossref(work),
                    year=year_from_crossref(work),
                    doi=work.get("DOI"),
                    abstract=clean_abstract(work.get("abstract")),
                    source_url=work.get("URL"),
                )
                logs.append(log("Metadata", "Metadata resolved via DOI", title=metadata.title, doi=metadata.doi))
                return metadata, logs
            # CrossRef miss — last-ditch Semantic Scholar DOI lookup
            meta = await _resolve_by_ss_doi(client, doi)
            if meta:
                logs.append(log("Metadata", "Metadata resolved via Semantic Scholar DOI",
                                title=meta.title, doi=meta.doi))
                return meta, logs
            logs.append(log("Metadata", "DOI not found in CrossRef or Semantic Scholar", doi=doi))
            return None, logs
        else:
            # Title/free-text query.
            # Strategy: try three sources in order, pick whichever resolves to
            # the HIGHEST citation count — that is always the most-impactful paper.
            logs.append(log("Metadata", "Title query — resolving to highest-impact paper"))

            # 1. Semantic Scholar (best citation data, may be rate-limited)
            ss_meta, ss_cites = await _semantic_scholar_title_search(client, query)

            # 2. OpenAlex (free, no key needed, filters out recent republications)
            oa_meta, oa_cites = await _openalex_title_search(client, query)

            # Pick whichever source gave the higher citation count
            if ss_cites > 0 or oa_cites > 0:
                if ss_cites >= oa_cites and ss_meta:
                    chosen, cites, source = ss_meta, ss_cites, "Semantic Scholar"
                else:
                    chosen, cites, source = oa_meta, oa_cites, "OpenAlex"
                logs.append(log("Metadata", f"Resolved via {source} — highest-impact match",
                                title=chosen.title, year=chosen.year,
                                citations=cites, doi=chosen.doi))
                return chosen, logs

            # 3. Last resort: CrossRef top-5, pick by citation count
            url = "https://api.crossref.org/works"
            response = await client.get(url, params={"query.title": query, "rows": 5}, timeout=12)
            response.raise_for_status()
            items = response.json()["message"].get("items", [])
            # Drop predatory republication-mill DOIs
            items = [w for w in items
                     if not (w.get("DOI") or "").lower().startswith("10.65215/")] or items
            if not items:
                logs.append(log("Metadata", "No match found across all sources"))
                return None, logs
            work = max(items, key=lambda w: w.get("is-referenced-by-count", 0))
            title = (work.get("title") or ["Unknown title"])[0]
            metadata = PaperMetadata(
                title=title,
                authors=authors_from_crossref(work),
                year=year_from_crossref(work),
                doi=work.get("DOI"),
                abstract=clean_abstract(work.get("abstract")),
                source_url=work.get("URL"),
            )
            logs.append(log("Metadata", "Resolved via CrossRef fallback",
                            title=metadata.title, doi=metadata.doi))
            return metadata, logs
    except Exception as exc:
        logs.append(log("Metadata", "CrossRef lookup failed", error=str(exc)))
        return None, logs


def clean_abstract(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"<[^>]+>", "", value).strip()


def inverted_abstract_to_text(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))[:900]


async def fetch_semantic_scholar(client: httpx.AsyncClient, metadata: PaperMetadata, original_query: str) -> tuple[int, list[EvidenceItem], list[TraceLog]]:
    logs = [log("Scholar", "Querying Semantic Scholar Academic Graph")]
    paper_id = metadata.doi or extract_doi(original_query) or original_query
    fields = "title,year,authors,citationCount,url,abstract,citations.title,citations.year,citations.authors,citations.url,citations.abstract,citations.citationCount"
    headers = {}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id, safe='')}"
        # Retry once on 429 (rate-limit) with a short back-off
        for attempt in range(2):
            response = await client.get(url, params={"fields": fields}, headers=headers, timeout=20)
            if response.status_code == 429 and attempt == 0:
                logs.append(log("Scholar", "Rate-limited by Semantic Scholar, retrying in 5 s"))
                await asyncio.sleep(5)
                continue
            break
        response.raise_for_status()
        payload = response.json()
        citations = payload.get("citations", [])[:20]
        evidence = [
            EvidenceItem(
                title=item.get("title") or "Untitled citing paper",
                url=item.get("url"),
                year=item.get("year"),
                authors=[author.get("name", "") for author in item.get("authors", [])[:5] if author.get("name")],
                snippet=item.get("abstract"),
                source="Semantic Scholar",
                kind="citation",
                citation_count=item.get("citationCount"),
            )
            for item in citations
            if item.get("title")
        ]
        count = int(payload.get("citationCount") or len(evidence))
        logs.append(log("Scholar", "Citation data retrieved", citation_count=count, evidence_items=len(evidence)))
        return count, evidence, logs
    except Exception as exc:
        logs.append(log("Scholar", "Semantic Scholar lookup failed", error=str(exc)))
        return 0, [], logs


async def fetch_openalex_fallback(client: httpx.AsyncClient, query: str) -> tuple[int, list[EvidenceItem], list[TraceLog]]:
    logs = [log("Fallback", "Searching OpenAlex for additional evidence")]
    try:
        response = await client.get("https://api.openalex.org/works", params={"search": query, "per-page": 8}, timeout=12)
        response.raise_for_status()
        results = response.json().get("results", [])
        evidence = []
        # This is a metadata-resolution search, not a citation search: nothing here
        # is known to cite the source paper, and past result[0] the title match can
        # drift onto unrelated work. Filter for relevance, and label accordingly.
        _relevant = build_relevance_filter(query)
        for item in results:
            display_name = item.get("display_name") or "Untitled work"
            if not _relevant(display_name):
                continue
            evidence.append(
                EvidenceItem(
                    title=display_name,
                    url=item.get("doi") or item.get("id"),
                    year=item.get("publication_year"),
                    authors=[a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:5] if a.get("author", {}).get("display_name")],
                    snippet=inverted_abstract_to_text(item.get("abstract_inverted_index")),
                    source="OpenAlex",
                    kind="metadata_candidate",
                    citation_count=item.get("cited_by_count"),
                    metric_label="Citations",
                    metric_value=f"{item.get('cited_by_count'):,}" if item.get("cited_by_count") is not None else None,
                )
            )
        citation_count = int(results[0].get("cited_by_count") or 0) if results else 0
        logs.append(log("Fallback", "OpenAlex evidence retrieved", evidence_items=len(evidence), citation_count=citation_count))
        return citation_count, evidence, logs
    except Exception as exc:
        logs.append(log("Fallback", "OpenAlex fallback failed", error=str(exc)))
        return 0, [], logs


async def fetch_openalex_enrichment(client: httpx.AsyncClient, metadata: PaperMetadata) -> tuple[list[EvidenceItem], list[str], list[TraceLog]]:
    logs = [log("Content", "Inspecting OpenAlex for topics, funders, and open-access links")]
    if not metadata.doi and not metadata.title:
        logs.append(log("Content", "Skipped OpenAlex enrichment; no DOI or title"))
        return [], [], logs

    try:
        if metadata.doi:
            response = await client.get(f"https://api.openalex.org/works/doi:{quote(metadata.doi, safe='')}", timeout=12)
        else:
            response = await client.get("https://api.openalex.org/works", params={"search": metadata.title, "per-page": 1}, timeout=12)
        response.raise_for_status()
        payload = response.json()
        work = payload.get("results", [payload])[0] if "results" in payload else payload
        evidence: list[EvidenceItem] = []

        for location_name in ["best_oa_location", "primary_location"]:
            location = work.get(location_name) or {}
            landing_url = location.get("landing_page_url")
            pdf_url = location.get("pdf_url")
            if landing_url or pdf_url:
                source = location.get("source", {}) or {}
                evidence.append(
                    EvidenceItem(
                        title=f"{metadata.title} source record",
                        url=pdf_url or landing_url,
                        year=metadata.year,
                        authors=metadata.authors[:5],
                        snippet="OpenAlex located a landing page or full-text route for this work.",
                        source=source.get("display_name") or "OpenAlex",
                        kind="full_text",
                        metric_label="Access",
                        metric_value="PDF" if pdf_url else "Landing page",
                    )
                )

        for funder in (work.get("funders") or [])[:5]:
            name = funder.get("display_name")
            if name:
                evidence.append(
                    EvidenceItem(
                        title=name,
                        url=funder.get("id"),
                        year=metadata.year,
                        snippet="OpenAlex links this work to a funder record.",
                        source="OpenAlex",
                        kind="funding",
                    )
                )

        topics = [
            topic.get("display_name")
            for topic in (work.get("topics") or [])[:6]
            if topic.get("display_name")
        ]
        if work.get("primary_topic", {}).get("display_name"):
            topics.insert(0, work["primary_topic"]["display_name"])
        topics = list(dict.fromkeys(topics))[:6]
        logs.append(log("Content", "OpenAlex enrichment complete", topics=len(topics), evidence_items=len(evidence)))
        return evidence, topics, logs
    except Exception as exc:
        logs.append(log("Content", "OpenAlex enrichment failed", error=str(exc)))
        return [], [], logs


async def fetch_github_adoption(client: httpx.AsyncClient, metadata: PaperMetadata) -> tuple[list[EvidenceItem], list[TraceLog]]:
    logs = [log("Code", "Searching GitHub repositories for adoption signals")]
    title = metadata.title.strip()
    if not title or title == "Unknown title":
        logs.append(log("Code", "Skipped GitHub search; no reliable title"))
        return [], logs

    title_terms = meaningful_title_terms(title)
    if len(title_terms) < 2:
        logs.append(log("Code", "Skipped GitHub search; title is too generic for reliable repository matching", title=title))
        return [], logs

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    quoted_title = f'"{title[:80]}"'
    query = f"{quoted_title} in:name,description,readme"
    try:
        response = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 6},
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        repos = response.json().get("items", [])
        evidence = []
        for repo in repos:
            evidence.append(
                EvidenceItem(
                    title=repo.get("full_name") or repo.get("name") or "GitHub repository",
                    url=repo.get("html_url"),
                    year=None,
                    authors=[repo.get("owner", {}).get("login", "")] if repo.get("owner", {}).get("login") else [],
                    snippet=repo.get("description") or "Repository matched distinctive title terms from the paper.",
                    source="GitHub",
                    kind="code",
                    metric_label="Stars",
                    metric_value=f"{repo.get('stargazers_count', 0):,}",
                )
            )
        logs.append(log("Code", "GitHub adoption search complete", repositories=len(evidence)))
        return evidence, logs
    except Exception as exc:
        logs.append(log("Code", "GitHub adoption search failed", error=str(exc)))
        return [], logs


async def fetch_pubmed(client: httpx.AsyncClient, metadata: PaperMetadata) -> tuple[list[EvidenceItem], list[TraceLog]]:
    """Search PubMed via E-utilities for citing articles and related biomedical literature.
    Free, no API key required, excellent coverage for medical/life-science papers."""
    logs = [log("PubMed", "Searching PubMed E-utilities")]
    title = metadata.title.strip()
    if not title or title == "Unknown title":
        return [], logs
    try:
        # Step 1: search for article IDs matching the title
        search_resp = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": f'"{title}"[Title]', "retmax": 8,
                    "retmode": "json", "tool": "reflect_ai", "email": "research@reflect.ai"},
            timeout=15,
        )
        if not search_resp.is_success:
            logs.append(log("PubMed", "Search failed", status=search_resp.status_code))
            return [], logs

        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        # Only the word-search fallback is loose enough to need relevance filtering;
        # the exact-title and DOI searches are already precise.
        used_word_fallback = False
        if not pmids:
            # Broader fallback: use DOI or first three title words
            term = metadata.doi if metadata.doi else " ".join(title.split()[:5])
            used_word_fallback = not metadata.doi
            search_resp2 = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": term, "retmax": 6,
                        "retmode": "json", "tool": "reflect_ai", "email": "research@reflect.ai"},
                timeout=15,
            )
            if search_resp2.is_success:
                pmids = search_resp2.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            logs.append(log("PubMed", "No PubMed matches found"))
            return [], logs

        # Step 2: fetch summaries for those IDs
        summary_resp = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids[:6]), "retmode": "json",
                    "tool": "reflect_ai", "email": "research@reflect.ai"},
            timeout=15,
        )
        if not summary_resp.is_success:
            return [], logs

        result = summary_resp.json().get("result", {})
        evidence: list[EvidenceItem] = []
        _relevant = build_relevance_filter(title)
        for pmid in pmids[:6]:
            art = result.get(pmid, {})
            if not art or pmid == "uids":
                continue
            art_title = art.get("title", "")
            if used_word_fallback and not _relevant(art_title):
                continue
            pub_date = art.get("pubdate", "")
            year = int(pub_date[:4]) if pub_date[:4].isdigit() else None
            authors_raw = art.get("authors", [])
            authors = [a.get("name", "") for a in authors_raw[:5]]
            source = art.get("source", "PubMed")
            uid_doi = next((e.get("value") for e in art.get("articleids", []) if e.get("idtype") == "doi"), None)
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            evidence.append(EvidenceItem(
                title=art_title,
                url=url,
                year=year,
                authors=authors,
                snippet=f"PubMed record. Journal: {source}.",
                source="PubMed",
                kind="citation",
                metric_label="PMID",
                metric_value=pmid,
            ))

        logs.append(log("PubMed", "PubMed evidence retrieved", items=len(evidence)))
        return evidence, logs
    except Exception as exc:
        logs.append(log("PubMed", "PubMed search failed", error=str(exc)))
        return [], logs


async def fetch_clinical_trials(client: httpx.AsyncClient, metadata: PaperMetadata) -> tuple[list[EvidenceItem], list[TraceLog]]:
    """Search ClinicalTrials.gov for trials that reference this paper's intervention or topic.
    Free, no API key, critical for biomedical impact evidence."""
    logs = [log("ClinicalTrials", "Searching ClinicalTrials.gov")]
    title = metadata.title.strip()
    if not title or title == "Unknown title":
        return [], logs
    # Only meaningful for biomedical/clinical papers — skip if clearly CS/maths
    cs_keywords = {"network", "algorithm", "learning", "graph", "neural", "transformer",
                   "attention", "model", "compiler", "database", "protocol"}
    title_lower = title.lower()
    if sum(1 for kw in cs_keywords if kw in title_lower) >= 2:
        logs.append(log("ClinicalTrials", "Skipped — paper appears non-biomedical"))
        return [], logs
    try:
        search_terms = " ".join(title.split()[:6])
        resp = await client.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": search_terms, "pageSize": 5,
                    "fields": "NCTId,BriefTitle,OverallStatus,StartDate,Condition,InterventionName"},
            timeout=15,
        )
        if not resp.is_success:
            logs.append(log("ClinicalTrials", "API call failed", status=resp.status_code))
            return [], logs

        studies = resp.json().get("studies", [])
        evidence: list[EvidenceItem] = []
        _relevant = build_relevance_filter(title)
        for study in studies[:5]:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            nct_id = id_mod.get("nctId", "")
            trial_title = id_mod.get("briefTitle", nct_id)
            overall_status = status_mod.get("overallStatus", "")
            start_date = status_mod.get("startDateStruct", {}).get("date", "")
            year = int(start_date[:4]) if start_date[:4].isdigit() else None
            cond_mod = proto.get("conditionsModule", {})
            conditions = ", ".join(cond_mod.get("conditions", [])[:3])
            snippet = f"Clinical trial ({overall_status}). Conditions: {conditions}." if conditions else f"Clinical trial ({overall_status})."
            if not _relevant(trial_title, snippet):
                continue
            evidence.append(EvidenceItem(
                title=trial_title,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                year=year,
                authors=[],
                snippet=snippet,
                source="ClinicalTrials.gov",
                kind="clinical",
                metric_label="NCT ID",
                metric_value=nct_id,
            ))

        logs.append(log("ClinicalTrials", "ClinicalTrials.gov evidence retrieved", items=len(evidence)))
        return evidence, logs
    except Exception as exc:
        logs.append(log("ClinicalTrials", "ClinicalTrials.gov search failed", error=str(exc)))
        return [], logs


async def fetch_soft_sciences(client: httpx.AsyncClient, metadata: PaperMetadata, topics: list[str] | None = None) -> tuple[list[EvidenceItem], list[TraceLog]]:
    """Humanities / social-science / policy evidence — closes the soft-sciences
    coverage gap left by GitHub, PubMed, and Semantic Scholar.

    Two free, no-key sources queried concurrently:
      * GOV.UK Search API   -> UK policy documents (kind="policy") — direct REF
                               social-impact evidence (policy influence).
      * OpenAIRE Research Graph -> SSH research outputs and monographs
                               (kind="monograph").
    """
    logs = [log("SoftSci", "Searching policy portals and SSH repositories")]
    title = metadata.title.strip()
    if not title or title == "Unknown title":
        return [], logs

    _relevant = build_relevance_filter(title, topics)

    # Build a topic-aware query: title head + 1-2 high-signal topic terms
    title_terms = meaningful_title_terms(title)
    query_terms = title_terms[:5]
    if topics:
        for t in topics[:2]:
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) > 3 and w not in query_terms:
                    query_terms.append(w)
    query = " ".join(query_terms[:6]) or " ".join(title.split()[:5])

    evidence: list[EvidenceItem] = []

    # ── GOV.UK policy portal ────────────────────────────────────────────────
    async def _govuk() -> list[EvidenceItem]:
        out: list[EvidenceItem] = []
        try:
            r = await client.get(
                "https://www.gov.uk/api/search.json",
                params={"q": query, "count": 4,
                        "fields": "title,link,description,public_timestamp,format,organisations"},
                timeout=12,
            )
            if not r.is_success:
                return out
            for res in r.json().get("results", [])[:6]:
                item_title = res.get("title", "UK policy document")
                desc = (res.get("description") or "")[:200]
                if not _relevant(item_title, desc):
                    continue
                ts = res.get("public_timestamp", "") or ""
                yr = int(ts[:4]) if ts[:4].isdigit() else None
                orgs = res.get("organisations", []) or []
                org_name = orgs[0].get("title", "") if orgs and isinstance(orgs[0], dict) else ""
                link = res.get("link", "")
                full_url = link if link.startswith("http") else f"https://www.gov.uk{link}"
                out.append(EvidenceItem(
                    title=item_title,
                    url=full_url,
                    year=yr,
                    authors=[org_name] if org_name else [],
                    snippet=f"UK Government policy document ({res.get('format','')}). {desc}",
                    source="GOV.UK",
                    kind="policy",
                    metric_label="Dept" if org_name else None,
                    metric_value=org_name[:25] if org_name else None,
                ))
                if len(out) >= 4:
                    break
        except Exception:
            pass
        return out

    # ── OpenAIRE SSH / monographs ───────────────────────────────────────────
    async def _openaire() -> list[EvidenceItem]:
        out: list[EvidenceItem] = []
        try:
            r = await client.get(
                "https://api.openaire.eu/search/publications",
                params={"keywords": query, "format": "json", "size": 4},
                timeout=15,
            )
            if not r.is_success:
                return out
            results = r.json().get("response", {}).get("results", {}).get("result", [])
            if isinstance(results, dict):
                results = [results]
            for res in (results or [])[:6]:
                try:
                    md = (res.get("metadata", {}) or {}).get("oaf:entity", {}).get("oaf:result", {})
                    t = md.get("title")
                    if isinstance(t, list):
                        t = t[0]
                    title_str = (t.get("$") if isinstance(t, dict) else t) or "SSH research output"
                    title_str = title_str if isinstance(title_str, str) else str(title_str)
                    rtype = (md.get("resulttype", {}) or {}).get("@classname", "publication")
                    if not _relevant(title_str, rtype):
                        continue
                    date_str = (md.get("dateofacceptance", {}) or {}).get("$", "") or ""
                    yr = int(date_str[:4]) if date_str[:4].isdigit() else None
                    # Best-effort DOI / landing page
                    pid = md.get("pid")
                    doi_val = None
                    if isinstance(pid, list):
                        for p in pid:
                            if isinstance(p, dict) and p.get("@classid") == "doi":
                                doi_val = p.get("$")
                                break
                    elif isinstance(pid, dict) and pid.get("@classid") == "doi":
                        doi_val = pid.get("$")
                    url_val = f"https://doi.org/{doi_val}" if doi_val else None
                    kind = "monograph" if "book" in rtype.lower() else "ssh"
                    out.append(EvidenceItem(
                        title=title_str,
                        url=url_val,
                        year=yr,
                        authors=[],
                        snippet=f"OpenAIRE {rtype} — humanities/social-science research output.",
                        source="OpenAIRE",
                        kind=kind,
                        metric_label="Type",
                        metric_value=rtype[:20],
                    ))
                    if len(out) >= 4:
                        break
                except Exception:
                    continue
        except Exception:
            pass
        return out

    try:
        govuk_ev, openaire_ev = await asyncio.gather(_govuk(), _openaire())
        evidence = govuk_ev + openaire_ev
        logs.append(log("SoftSci", "Policy & SSH evidence retrieved",
                        policy=len(govuk_ev), ssh=len(openaire_ev)))
    except Exception as exc:
        logs.append(log("SoftSci", "Soft-sciences search failed", error=str(exc)))
    return evidence, logs


async def fetch_google_patents(client: httpx.AsyncClient, metadata: PaperMetadata) -> tuple[list[EvidenceItem], list[TraceLog]]:
    logs = [log("Patents", "Searching Google Patents via XHR API")]
    title = metadata.title.strip()
    if not title or title == "Unknown title":
        logs.append(log("Patents", "Skipped patent search; no reliable title"))
        return [], logs

    query = quote(f'"{title[:80]}"', safe="")
    url = f"https://patents.google.com/xhr/query?url=q%3D{query}"
    try:
        response = await client.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        
        cluster = payload.get("results", {}).get("cluster", [])
        if not cluster:
            logs.append(log("Patents", "No patent matches found"))
            return [], logs

        results = cluster[0].get("result", [])
        evidence = []
        for res in results[:5]:
            patent_info = res.get("patent", {})
            pub_num = patent_info.get("publication_number", "Unknown")
            patent_title = patent_info.get("title", f"Patent {pub_num}")
            patent_title = re.sub(r"<[^>]+>", "", patent_title).strip()
            snippet = re.sub(r"<[^>]+>", "", patent_info.get("snippet", "")).strip()
            assignee = patent_info.get("assignee", "")
            inventor = patent_info.get("inventor", "")
            
            authors = []
            if assignee: authors.append(assignee)
            elif inventor: authors.append(inventor)
            
            year_str = patent_info.get("publication_date", "")
            year = int(year_str[:4]) if len(year_str) >= 4 and year_str[:4].isdigit() else None
            
            evidence.append(
                EvidenceItem(
                    title=f"[{pub_num}] {patent_title}",
                    url=f"https://patents.google.com/patent/{pub_num}/en",
                    year=year,
                    authors=authors,
                    snippet=snippet,
                    source="Google Patents",
                    kind="patent",
                    metric_label="Assignee" if assignee else None,
                    metric_value=assignee[:25] if assignee else None,
                )
            )
        
        logs.append(log("Patents", "Google Patents search complete", records=len(evidence)))
        return evidence, logs
    except Exception as exc:
        logs.append(log("Patents", "Google Patents search failed", error=str(exc)))
        return [], logs


_OPENALEX_BASE = "https://api.openalex.org"

async def fetch_downstream_impact(
    client: httpx.AsyncClient, metadata: PaperMetadata, citation_count: int
) -> tuple[list[EvidenceItem], list[TraceLog]]:
    """Second-order impact tracing (Phase 3).

    Find papers that CITE this work and are themselves high-impact — i.e. they
    out-cite the source paper, or carry their own grant funding. Such citers are
    strong downstream evidence: the paper's influence is shown by the
    significance of what it enabled. Each surfaced adopter (and its funder) can
    then be cited in the impact narrative.
    """
    logs: list[TraceLog] = []
    try:
        work = None
        if metadata.doi:
            r = await client.get(f"{_OPENALEX_BASE}/works/doi:{metadata.doi}", params={"select": "id,cited_by_count"}, timeout=12)
            if r.is_success:
                work = r.json()
        if work is None and metadata.title:
            r = await client.get(f"{_OPENALEX_BASE}/works", params={"search": metadata.title, "per_page": 1, "select": "id,cited_by_count"}, timeout=12)
            if r.is_success:
                results = r.json().get("results") or []
                work = results[0] if results else None
        if not work:
            logs.append(log("Downstream", "Could not resolve paper on OpenAlex for citation tracing"))
            return [], logs

        work_id = (work.get("id") or "").rstrip("/").split("/")[-1]
        x_cites = work.get("cited_by_count") or citation_count or 0
        # No `select` here: OpenAlex won't let `grants` be selected, and we need it.
        resp = await client.get(f"{_OPENALEX_BASE}/works", params={
            "filter": f"cites:{work_id}", "sort": "cited_by_count:desc", "per_page": 12,
        }, timeout=15)
        if not resp.is_success:
            logs.append(log("Downstream", "Citing-works query failed", status=resp.status_code))
            return [], logs

        strong: list[EvidenceItem] = []   # out-cite the source, or carry grant funding
        notable: list[EvidenceItem] = []  # otherwise the most influential citers
        for idx, c in enumerate(resp.json().get("results", [])):
            c_cites = c.get("cited_by_count", 0) or 0
            grants = c.get("grants") or []
            out_cites = c_cites > x_cites and x_cites > 0
            authors = [a.get("author", {}).get("display_name", "") for a in (c.get("authorships") or [])[:3] if a.get("author")]
            grant_note = ""
            if grants:
                g = grants[0]
                award = f" {g['award_id']}" if g.get("award_id") else ""
                grant_note = f" Funded follow-on research ({g.get('funder_display_name', 'a research funder')}{award})."
            # Institution type and venue are already in the (unselected) response — no extra API cost.
            inst_types: set[str] = set()
            inst_by_type: dict[str, str] = {}
            for a in (c.get("authorships") or []):
                for inst in (a.get("institutions") or []):
                    t = inst.get("type")
                    if not t:
                        continue
                    inst_types.add(t)
                    if inst.get("display_name"):
                        inst_by_type.setdefault(t, inst["display_name"])

            adopter = None
            for t, k, lbl, descriptor, uptake in (
                ("company",    "industry_adoption", "Industry adopter",        "a commercial organisation", "industrial uptake"),
                ("government", "policy_adoption",   "Government body",         "a government body",         "policy-relevant uptake"),
                ("healthcare", "clinical_adoption", "Healthcare organisation", "a healthcare organisation", "clinical translation"),
            ):
                if t in inst_types:
                    # Name the institution of the MATCHED type, not merely the first typed one.
                    adopter = (k, lbl, inst_by_type.get(t), descriptor, uptake)
                    break

            source_name = ((c.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            is_software_paper = (
                c.get("type") == "software"
                or "open source software" in source_name.lower()
                or "softwarex" in source_name.lower()
            )

            kind, is_strong = "downstream", bool(out_cites or grants)
            if adopter:
                kind, label, org, descriptor, uptake = adopter
                subject = f"{org}, {descriptor}," if org else descriptor.capitalize()
                snippet = f"{subject} cites this work ({c_cites:,} citations) — evidencing {uptake}."
                value, is_strong = org or f"{c_cites:,} cites", True
            elif is_software_paper:
                kind, label, value = "software_adoption", "Software adoption", f"{c_cites:,} cites"
                snippet = f"A software paper describing a tool built on this work ({c_cites:,} citations) — evidencing practical/code adoption."
                is_strong = True
            elif out_cites:
                snippet = f"Cites this work and is itself cited {c_cites:,} times — exceeding the source paper, a strong downstream-impact signal.{grant_note}"
                label, value = "Out-cites source", f"{c_cites:,} cites"
            elif grants:
                snippet = f"Cites this work ({c_cites:,} citations).{grant_note}"
                label, value = "Funder", grants[0].get("funder_display_name")
            elif idx < 4 and c_cites >= 50:
                snippet = f"A highly-cited work building on this paper ({c_cites:,} citations), evidencing downstream uptake."
                label, value = "Citing work", f"{c_cites:,} cites"
            else:
                continue
            item = EvidenceItem(
                title=c.get("title") or "Untitled citing work",
                url=c.get("doi"),
                year=c.get("publication_year"),
                authors=authors,
                snippet=snippet,
                source="OpenAlex",
                kind=kind,
                citation_count=c_cites,
                metric_label=label,
                metric_value=value,
            )
            (strong if is_strong else notable).append(item)

        evidence = (strong + notable)[:5]
        logs.append(log("Downstream", f"Traced {len(evidence)} downstream adopters ({len(strong)} strong)", source_citations=x_cites))
        return evidence, logs
    except Exception as exc:
        logs.append(log("Downstream", "Downstream impact tracing failed", error=str(exc)))
        return [], logs


def synthesize(
    metadata: PaperMetadata,
    citation_count: int,
    evidence: list[EvidenceItem],
    topics: list[str],
    rag_contexts: list[str],
) -> tuple[str, list[ImpactSection], float, list[str], list[TraceLog], str, str, dict, list[dict]]:
    logs = [log("Synthesis", "Generating evidence-grounded summary")]
    title = metadata.title
    year = metadata.year or "unknown year"
    authors = ", ".join(metadata.authors[:3]) if metadata.authors else "the authors"
    evidence_titles = [item.title for item in evidence[:4]]
    code_count = len([item for item in evidence if item.kind == "code"])
    full_text_count = len([item for item in evidence if item.kind == "full_text"])
    funding_count = len([item for item in evidence if item.kind == "funding"])
    citation_phrase = f"{citation_count:,} citations" if citation_count else "available citation evidence"

    if evidence_titles:
        evidence_sentence = "Citing work includes " + "; ".join(evidence_titles[:3]) + "."
    else:
        evidence_sentence = "The current run found limited downstream evidence, so the summary should be treated as preliminary."

    hf_summary, model_provider, references, hf_logs = hf_synthesizer.generate(metadata, citation_count, evidence, topics, rag_contexts)
    logs.extend(hf_logs)

    patent_ev  = [e for e in evidence if e.kind == "patent"]
    code_ev    = [e for e in evidence if e.kind == "code"]
    topic_str  = ", ".join(topics[:3]) if topics else "multiple research areas"

    adoption = ""
    if patent_ev:
        adoption = (
            f" Notably, the methodology has been cited in {len(patent_ev)} patent filing(s), "
            f"indicating measurable translation into applied technology."
        )
    elif code_ev:
        adoption = (
            f" Open-source implementations were identified on GitHub, "
            f"demonstrating active practical adoption beyond the academic community."
        )
    else:
        adoption = (
            f" The work's influence spans {topic_str}, "
            f"reflecting broad interdisciplinary reach across the research community."
        )

    deterministic_summary = (
        f"{title} ({year}) by {authors} has accumulated **{citation_phrase}**, "
        f"establishing it as a highly cited and influential contribution to its field. "
        f"{evidence_sentence} "
        f"Retrieved evidence confirms downstream engagement through citation networks and metadata records.{adoption}"
    )
    summary = hf_summary or deterministic_summary

    sections = [
        ImpactSection(
            title="Research Influence",
            body=f"The paper is represented in citation databases with {citation_phrase}. The retrieved citing papers provide the first evidence layer for analysing how later authors position the work. The RAG layer retrieved {len(rag_contexts)} vector-memory chunks for synthesis.",
        ),
        ImpactSection(
            title="Applications",
            body=(
                "Likely application areas are inferred from OpenAlex topic metadata and citing-paper evidence. "
                + (f"Detected topics include {', '.join(topics[:5])}." if topics else "No topic labels were available for this paper yet.")
            ),
        ),
        ImpactSection(
            title="Technical Adoption",
            body=(
                f"The GitHub agent found {code_count} repository matches for adoption or implementation signals. "
                "These are search-based leads and should be verified before being counted as confirmed implementations."
            ),
        ),
        ImpactSection(
            title="Access & Funding",
            body=(
                f"The content agent found {full_text_count} source or full-text routes and {funding_count} funder signals. "
                "This helps distinguish citation influence from practical access and funding context."
            ),
        ),
    ]
    slm_faithfulness, judge_logs = hf_synthesizer.evaluate_faithfulness(summary, evidence, rag_contexts)
    logs.extend(judge_logs)

    if slm_faithfulness >= 0.0:
        faithfulness = slm_faithfulness
    else:
        faithfulness = score_faithfulness(summary, evidence, rag_contexts, citation_count, topics)

    # ── Second-pass cross-validation (independent peer-reviewer) ──
    validation_report, val_logs = hf_synthesizer.cross_validate_summary(
        summary, evidence, citation_count, metadata
    )
    logs.extend(val_logs)

    guardrail_status = "passed" if faithfulness >= 0.75 else "review"
    limitations = []
    if not evidence:
        limitations.append("No citation evidence was retrieved from the available public APIs for this query.")
    if model_provider == "deterministic":
        limitations.append("AI narrative unavailable. Add GOOGLE_API_KEY to backend/.env for Gemini-powered summaries.")

    logs.append(log("Guardrail", "Faithfulness guardrail evaluated summary", faithfulness_score=faithfulness, status=guardrail_status))
    logs.append(log("Synthesis", "Summary generated", faithfulness_score=faithfulness, model_provider=model_provider, references=len(references)))
    return summary, sections, faithfulness, limitations, logs, model_provider, guardrail_status, validation_report, references


def score_faithfulness(summary: str, evidence: list[EvidenceItem], rag_contexts: list[str], citation_count: int, topics: list[str]) -> float:
    if not evidence:
        return 0.38
    evidence_text = " ".join([item.title + " " + (item.snippet or "") for item in evidence] + rag_contexts).lower()
    summary_terms = [term for term in re.findall(r"[a-z0-9]+", summary.lower()) if len(term) > 4]
    if not summary_terms:
        return 0.45
    overlap = sum(1 for term in summary_terms if term in evidence_text) / len(summary_terms)
    score = 0.45 + min(0.3, overlap * 0.35) + (0.08 if citation_count else 0) + (0.07 if rag_contexts else 0) + (0.04 if topics else 0)
    return round(min(0.91, score), 2)


class AnalysisState(TypedDict, total=False):
    query: str
    metadata: PaperMetadata
    citation_count: int
    topics: list[str]
    evidence: list[EvidenceItem]
    statuses: list[AgentStatus]
    logs: list[TraceLog]
    summary: str
    sections: list[ImpactSection]
    faithfulness: float
    limitations: list[str]
    rag_contexts: list[str]
    embedding_provider: str
    model_provider: str
    guardrail_status: str
    ref_report: str
    validation_report: dict
    routing_fields: list[str]
    routing_metadata: dict
    references: list[dict]
    candidates: list[dict]


def initial_statuses() -> list[AgentStatus]:
    return [
        status("metadata", "Metadata", AgentState.running, "Resolving paper identity"),
        status("scholar", "Scholar", AgentState.pending, "Waiting for metadata"),
        status("content", "Content", AgentState.pending, "Waiting for metadata"),
        status("code", "Code", AgentState.pending, "Waiting for metadata"),
        status("rag", "RAG", AgentState.pending, "Waiting for evidence"),
        status("impact", "Impact", AgentState.pending, "Waiting for retrieval"),
        status("synthesis", "Synthesis", AgentState.pending, "Waiting for evidence"),
        status("ref", "REF Report", AgentState.pending, "Waiting for synthesis"),
    ]


async def metadata_node(state: AnalysisState) -> AnalysisState:
    normalized = state["query"]
    logs = state["logs"]
    statuses = state["statuses"]
    async with httpx.AsyncClient(headers={"User-Agent": "ResearchImpactSummariser/0.1 (mailto:student@example.com)"}) as client:
        metadata, crossref_logs = await fetch_crossref(client, normalized)
    logs.extend(crossref_logs)
    if metadata is None:
        metadata = PaperMetadata(title=normalized, source_url=None)
        statuses[0] = status("metadata", "Metadata", AgentState.warning, "Used raw input as title")
    else:
        statuses[0] = status("metadata", "Metadata", AgentState.complete, "Paper metadata resolved")
    return {**state, "metadata": metadata, "statuses": statuses, "logs": logs}


async def retrieval_node(state: AnalysisState) -> AnalysisState:
    metadata = state["metadata"]
    normalized = state["query"]
    logs = state["logs"]
    statuses = state["statuses"]
    statuses[1] = status("scholar", "Scholar", AgentState.running, "Retrieving citations")
    statuses[2] = status("content", "Content", AgentState.running, "Finding full text and topics")
    statuses[3] = status("code", "Code", AgentState.running, "Searching repositories")

    # Phase 2 — dynamic routing. Decide which sources to query from the
    # researcher's profile fields + this paper's own classified domain. When no
    # fields are supplied (e.g. anonymous), plan_routing returns all sources, so
    # behaviour is identical to the original full-scan pipeline.
    plan = plan_routing(
        fields=state.get("routing_fields") or [],
        title=metadata.title or "",
        abstract=(getattr(metadata, "abstract", "") or ""),
    )
    active = list(plan["sources_called"])
    logs.append(log(
        "Router",
        f"Domain routing ({plan['reason']}): {len(active)}/{len(ALL_SOURCES)} sources",
        domains=plan["domains"], sources_called=active, sources_skipped=plan["sources_skipped"],
    ))

    async with httpx.AsyncClient(headers={"User-Agent": "ResearchImpactSummariser/0.1 (mailto:student@example.com)"}) as client:
        optional_callables = {
            "github": lambda: fetch_github_adoption(client, metadata),
            "patents": lambda: fetch_google_patents(client, metadata),
            "pubmed": lambda: fetch_pubmed(client, metadata),
            "clinical_trials": lambda: fetch_clinical_trials(client, metadata),
            "soft_sciences": lambda: fetch_soft_sciences(client, metadata),
        }

        async def _no_ev():
            return [], []

        # Core sources always run; optional sources are gated by the routing plan.
        scholar_task    = fetch_semantic_scholar(client, metadata, normalized)
        fallback_task   = fetch_openalex_fallback(client, metadata.title)
        enrichment_task = fetch_openalex_enrichment(client, metadata)
        downstream_task = fetch_downstream_impact(client, metadata, state.get("citation_count", 0))
        github_task     = optional_callables["github"]()          if "github" in active          else _no_ev()
        patent_task     = optional_callables["patents"]()         if "patents" in active         else _no_ev()
        pubmed_task     = optional_callables["pubmed"]()          if "pubmed" in active          else _no_ev()
        clinical_task   = optional_callables["clinical_trials"]() if "clinical_trials" in active else _no_ev()
        softsci_task    = optional_callables["soft_sciences"]()   if "soft_sciences" in active   else _no_ev()
        (
            (citation_count, scholar_evidence, scholar_logs),
            (fallback_count, fallback_evidence, fallback_logs),
            (content_evidence, topics, content_logs),
            (downstream_evidence, downstream_logs),
            (code_evidence, code_logs),
            (patent_evidence, patent_logs),
            (pubmed_evidence, pubmed_logs),
            (clinical_evidence, clinical_logs),
            (softsci_evidence, softsci_logs),
        ) = await asyncio.gather(
            scholar_task, fallback_task, enrichment_task, downstream_task,
            github_task, patent_task, pubmed_task, clinical_task, softsci_task,
        )
        citation_count = citation_count or fallback_count
        source_yield = {
            "semantic_scholar": len(scholar_evidence), "openalex_fallback": len(fallback_evidence),
            "openalex_enrichment": len(content_evidence), "downstream": len(downstream_evidence),
            "github": len(code_evidence), "patents": len(patent_evidence),
            "pubmed": len(pubmed_evidence), "clinical_trials": len(clinical_evidence),
            "soft_sciences": len(softsci_evidence),
        }
        logs.extend(scholar_logs + fallback_logs + content_logs + downstream_logs + code_logs
                    + patent_logs + pubmed_logs + clinical_logs + softsci_logs)
        evidence = dedupe_evidence(
            downstream_evidence + scholar_evidence + fallback_evidence + content_evidence
            + code_evidence + patent_evidence + pubmed_evidence
            + clinical_evidence + softsci_evidence
        )

        # Insufficient-evidence fallback: top up with the sources we skipped.
        fallback_triggered = False
        missing = [s for s in OPTIONAL_SOURCES if s not in active]
        if missing and len(evidence) < MIN_ROUTED_EVIDENCE:
            fallback_triggered = True
            logs.append(log(
                "Router",
                f"Fallback: only {len(evidence)} items (<{MIN_ROUTED_EVIDENCE}); querying {len(missing)} skipped sources",
                sources=missing,
            ))
            extra_results = await asyncio.gather(*[optional_callables[s]() for s in missing])
            extra_ev: list[EvidenceItem] = []
            for src, (ev, lg) in zip(missing, extra_results):
                extra_ev += ev
                source_yield[src] = len(ev)
                logs.extend(lg)
            evidence = dedupe_evidence(evidence + extra_ev)
            active = active + missing

    evidence = diversify_evidence(evidence)
    empty_sources = sorted(k for k, v in source_yield.items() if v == 0)
    if empty_sources:
        logs.append(log(
            "Router", f"{len(empty_sources)} source(s) returned no evidence",
            sources_empty=empty_sources, source_yield=source_yield,
        ))

    skipped = [s for s in ALL_SOURCES if s not in active]
    routing_metadata = {
        "reason": f"{plan['reason']}+fallback" if fallback_triggered else plan["reason"],
        "domains": plan["domains"],
        "profile_domains": plan["profile_domains"],
        "paper_domains": plan["paper_domains"],
        "sources_called": active,
        "sources_skipped": skipped,
        "all_sources": list(ALL_SOURCES),
        "fallback_triggered": fallback_triggered,
        "evidence_count": len(evidence),
        "source_yield": source_yield,
        "sources_empty": empty_sources,
        "saved_calls": len(skipped),
    }
    statuses[1] = status("scholar", "Scholar", AgentState.complete if evidence else AgentState.warning, f"{len(evidence)} evidence items retrieved")
    statuses[2] = status("content", "Content", AgentState.complete if content_evidence or topics else AgentState.warning, f"{len(topics)} topics, {len(content_evidence)} links")
    statuses[3] = status("code", "Code", AgentState.complete if code_evidence else AgentState.warning, f"{len(code_evidence)} repository leads")
    statuses[5] = status("impact", "Impact", AgentState.complete, f"Routed to {len(active)}/{len(ALL_SOURCES)} sources")
    return {**state, "citation_count": citation_count, "evidence": evidence, "topics": topics, "routing_metadata": routing_metadata, "statuses": statuses, "logs": logs}


async def rag_node(state: AnalysisState) -> AnalysisState:
    logs = state["logs"]
    statuses = state["statuses"]
    statuses[4] = status("rag", "RAG", AgentState.running, "Embedding evidence")
    rag_contexts, embedding_provider, rag_logs = await asyncio.to_thread(index_and_retrieve, state["metadata"], state["evidence"], state["topics"])
    logs.extend(rag_logs)
    statuses[4] = status("rag", "RAG", AgentState.complete if rag_contexts else AgentState.warning, f"{len(rag_contexts)} context chunks")
    return {**state, "rag_contexts": rag_contexts, "embedding_provider": embedding_provider, "statuses": statuses, "logs": logs}


async def synthesis_node(state: AnalysisState) -> AnalysisState:
    logs = state["logs"]
    statuses = state["statuses"]
    statuses[6] = status("synthesis", "Synthesis", AgentState.running, "Synthesizing summary")
    summary, sections, faithfulness, limitations, synth_logs, model_provider, guardrail_status, validation_report, references = await asyncio.to_thread(
        synthesize,
        state["metadata"],
        state["citation_count"],
        state["evidence"],
        state["topics"],
        state["rag_contexts"],
    )
    logs.extend(synth_logs)
    statuses[6] = status("synthesis", "Synthesis", AgentState.complete, "Summary ready")
    logs.append(log("Supervisor", "Analysis complete"))
    return {
        **state,
        "summary": summary,
        "sections": sections,
        "faithfulness": faithfulness,
        "limitations": limitations,
        "logs": logs,
        "statuses": statuses,
        "model_provider": model_provider,
        "guardrail_status": guardrail_status,
        "validation_report": validation_report,
        "references": references,
    }


async def ref_node(state: AnalysisState) -> AnalysisState:
    logs = state["logs"]
    statuses = state["statuses"]
    statuses[7] = status("ref", "REF Report", AgentState.running, "Drafting case study")
    
    # Try generating with HF
    hf_report, ref_logs = await asyncio.to_thread(
        hf_synthesizer.generate_ref_report,
        state["metadata"],
        state["evidence"],
        state["summary"],
        state["topics"]
    )
    logs.extend(ref_logs)
    
    # Fallback: reuse the impact summary as the REF paragraph
    if not hf_report:
        hf_report = state["summary"]
        logs.append(log("REF", "Using impact summary as REF paragraph (Gemini unavailable)"))

    statuses[7] = status("ref", "REF Report", AgentState.complete, "Report generated")
    return {**state, "ref_report": hf_report, "statuses": statuses, "logs": logs}


async def draft_node(state: AnalysisState) -> AnalysisState:
    """Human-in-the-loop: draft one evidence-grounded candidate sentence per
    evidence item for the reviewer to curate (replaces direct synthesis)."""
    logs = state["logs"]
    statuses = state["statuses"]
    statuses[6] = status("synthesis", "Drafting", AgentState.running, "Drafting candidate sentences")
    candidates, draft_logs = await asyncio.to_thread(
        hf_synthesizer.draft_sentences,
        state["metadata"], state["citation_count"], state["evidence"],
    )
    logs.extend(draft_logs)
    statuses[6] = status("synthesis", "Drafting", AgentState.complete, f"{len(candidates)} candidate sentences ready for review")
    logs.append(log("Supervisor", "Evidence gathered — awaiting reviewer sentence selection"))
    return {**state, "candidates": candidates, "statuses": statuses, "logs": logs}


def build_graph():
    """Draft pipeline: retrieve evidence, then stop at candidate sentences for
    human curation. Final synthesis happens later via compose_summary()."""
    graph = StateGraph(AnalysisState)
    graph.add_node("metadata", metadata_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("rag", rag_node)
    graph.add_node("draft", draft_node)
    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "retrieval")
    graph.add_edge("retrieval", "rag")
    graph.add_edge("rag", "draft")
    graph.add_edge("draft", END)
    return graph.compile()


analysis_graph = build_graph()


async def analyze_paper(query: str, fields: list[str] | None = None):
    """Run the pipeline up to candidate-sentence drafting (no final summary).
    Returns evidence + candidate sentences plus everything compose_summary needs."""
    normalized = normalize_query(query)
    state = await analysis_graph.ainvoke(
        {
            "query": normalized,
            "statuses": initial_statuses(),
            "logs": [log("Supervisor", "LangGraph analysis started", query=normalized)],
            "citation_count": 0,
            "topics": [],
            "evidence": [],
            "rag_contexts": [],
            "embedding_provider": "not_run",
            "model_provider": "not_run",
            "guardrail_status": "not_run",
            "limitations": [],
            "ref_report": "",
            "validation_report": {},
            "routing_fields": fields or [],
            "routing_metadata": {},
            "references": [],
            "candidates": [],
        }
    )
    return {
        "metadata": state["metadata"],
        "evidence": state["evidence"][:32],
        "candidates": state.get("candidates", []),
        "agent_statuses": state["statuses"],
        "logs": state["logs"],
        "citation_count": state["citation_count"],
        "topics": state["topics"],
        "rag_contexts": state.get("rag_contexts", []),
        "rag_context_count": len(state.get("rag_contexts", [])),
        "routing": state.get("routing_metadata", {}),
    }


def compose_summary(
    metadata: "PaperMetadata",
    citation_count: int,
    topics: list[str],
    selected_evidence: list[EvidenceItem],
    sentences: list[str],
    rag_contexts: list[str],
) -> dict:
    """Stage 2 (after human curation): weave the reviewer-approved sentences
    into the final REF summary, then validate it. Mirrors synthesize() outputs."""
    logs = [log("Synthesis", "Composing summary from reviewer-approved sentences", approved=len(sentences))]
    references = build_references(selected_evidence, exclude_doi=metadata.doi, exclude_title=metadata.title)

    woven, model_provider, compose_logs = hf_synthesizer.compose(metadata, citation_count, sentences, references)
    logs.extend(compose_logs)
    summary = woven or " ".join(sentences) or "No sentences were selected for the summary."

    code_count = len([e for e in selected_evidence if e.kind == "code"])
    patent_count = len([e for e in selected_evidence if e.kind == "patent"])
    downstream_count = len([e for e in selected_evidence if e.kind == "downstream"])
    sections = [
        ImpactSection(title="Reviewer-curated impact",
                      body=f"This narrative was composed from {len(sentences)} reviewer-approved, evidence-grounded sentences, spanning {downstream_count} downstream adopter(s), {patent_count} patent(s), and {code_count} code adoption(s)."),
        ImpactSection(title="Citation reach",
                      body=f"The paper carries {citation_count:,} citations. " + (f"Detected topics include {', '.join(topics[:5])}." if topics else "No topic labels were available.")),
    ]

    from .hf_synthesis import _llm_mode, _provider_override, _groq_key, _gemini_key
    is_local = _llm_mode() in ("local", "local_only")

    # Faithfulness: on the LOCAL path use the deterministic heuristic — consistent,
    # no extra LLM call, and it avoids the small model grading its own output
    # (which fired the gate far too often). Cloud path keeps the LLM judge.
    if is_local:
        faithfulness = score_faithfulness(summary, selected_evidence, rag_contexts, citation_count, topics)
    else:
        slm_faith, judge_logs = hf_synthesizer.evaluate_faithfulness(summary, selected_evidence, rag_contexts)
        logs.extend(judge_logs)
        faithfulness = slm_faith if slm_faith >= 0.0 else score_faithfulness(summary, selected_evidence, rag_contexts, citation_count, topics)

    # ── Quality gate: local-first, cloud fallback ONLY on genuinely poor output.
    # Low threshold → fires rarely (near-empty / off-topic summaries), so ~all
    # traffic stays on-VM and cloud tokens are conserved. "local_only" (benchmark
    # mode) never falls back.
    if (_llm_mode() == "local" and faithfulness < 0.50 and (_groq_key() or _gemini_key())):
        logs.append(log("Guardrail", "Local summary below floor — regenerating with cloud",
                        local_faithfulness=faithfulness, threshold=0.50))
        _tok = _provider_override.set("cloud")
        try:
            woven2, provider2, logs2 = hf_synthesizer.compose(metadata, citation_count, sentences, references)
            logs.extend(logs2)
            if woven2:
                faith2 = score_faithfulness(woven2, selected_evidence, rag_contexts, citation_count, topics)
                if faith2 >= faithfulness:
                    summary, faithfulness = woven2, faith2
                    model_provider = f"local->{provider2}"
                    logs.append(log("Guardrail", "Cloud fallback accepted", faithfulness_score=faith2))
                else:
                    logs.append(log("Guardrail", "Local summary retained", cloud_faithfulness=faith2))
        finally:
            _provider_override.reset(_tok)

    # Local speed trim: the 4-dimension peer review and the separate REF
    # paragraph are extra LLM calls that roughly double local latency; on the
    # local path we skip them (REF tab reuses the composed summary).
    if is_local:
        validation_report: dict = {}
        ref_report = summary
        logs.append(log("Synthesis", "Local engine: peer-review and separate REF paragraph skipped for speed"))
    else:
        validation_report, val_logs = hf_synthesizer.cross_validate_summary(summary, selected_evidence, citation_count, metadata)
        logs.extend(val_logs)

        ref_report, ref_logs = hf_synthesizer.generate_ref_report(metadata, selected_evidence, summary, topics)
        logs.extend(ref_logs)
        if not ref_report:
            ref_report = summary

    guardrail_status = "passed" if faithfulness >= 0.75 else "review"
    limitations = []
    if not selected_evidence:
        limitations.append("No evidence sentences were selected.")
    if model_provider == "deterministic":
        limitations.append("AI narrative unavailable. Add GROQ_API_KEY or GOOGLE_API_KEY to backend/.env.")
    logs.append(log("Guardrail", "Faithfulness guardrail evaluated composed summary", faithfulness_score=faithfulness, status=guardrail_status))

    return {
        "summary": summary,
        "sections": [s.model_dump() if hasattr(s, "model_dump") else s for s in sections],
        "faithfulness_score": faithfulness,
        "citation_count": citation_count,
        "topics": topics,
        "model_provider": model_provider,
        "guardrail_status": guardrail_status,
        "limitations": limitations,
        "ref_report": ref_report,
        "validation_report": validation_report,
        "references": references,
        "logs": logs,
    }


def diversify_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Round-robin evidence by kind, preserving every item.

    Sources are concatenated with Semantic Scholar's citations near the front,
    so a prolific citation list used to fill the downstream 24-item candidate
    cap on its own and starve code, patent, policy and clinical evidence out of
    the reviewer's list entirely. Interleaving guarantees each evidence type a
    slot in the first pass; nothing is dropped, only reordered.
    """
    buckets: dict[str, list[EvidenceItem]] = {}
    for it in items:
        buckets.setdefault(it.kind, []).append(it)
    out: list[EvidenceItem] = []
    while len(out) < len(items):
        progressed = False
        for bucket in buckets.values():
            if bucket:
                out.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return out


def dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[str] = set()
    unique: list[EvidenceItem] = []
    for item in items:
        key = (item.url or item.title).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
