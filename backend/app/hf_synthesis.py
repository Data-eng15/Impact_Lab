"""
Multi-provider LLM synthesiser.

Priority order:
  1. Groq  (GROQ_API_KEY)  → llama-3.3-70b-versatile  — 30 req/min free tier
  2. Gemini (GOOGLE_API_KEY) → gemini-2.5-flash          — 20 req/min free tier

Keeps the same HFSynthesizer class interface so services.py needs no changes.
"""
from __future__ import annotations

import json as _json
import os
import re
import time

import httpx

from .models import EvidenceItem, PaperMetadata, TraceLog
from .services_support import log

# ── Provider endpoints ────────────────────────────────────────────────────────
_GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_last_error: list[str] = []

# Per-request LLM provider override (async-safe via contextvar; propagates into
# asyncio.to_thread). Lets one authenticated request opt into the local SLM
# without changing the global default. None -> fall back to the env setting.
import contextvars as _cv
_provider_override: "_cv.ContextVar" = _cv.ContextVar("llm_provider_override", default=None)


# ── Key helpers ───────────────────────────────────────────────────────────────

def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")

def _gemini_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "")

def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Local SLM (Ollama, OpenAI-compatible) — opt-in via LLM_PROVIDER ────────────
def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

def _local_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")  # base 3B — matches cloud faithfulness

def _llm_mode() -> str:
    """LLM routing mode (default keeps production on the cloud, untouched):
      cloud       -> Groq -> Gemini (default)
      local       -> local SLM first, cloud fallback on failure
      local_only  -> local SLM only (for clean benchmarking).
    A per-request override (contextvar) wins over the env default when set."""
    ov = _provider_override.get()
    if ov:
        return ov
    return os.getenv("LLM_PROVIDER", "cloud").strip().lower()

def _active_provider() -> str:
    """Return which provider is currently configured."""
    if _llm_mode() in ("local", "local_only"):
        return "local"
    if _groq_key():
        return "groq"
    if _gemini_key():
        return "gemini"
    return "none"


# ── Groq call (OpenAI-compatible) ─────────────────────────────────────────────

def _call_groq(system: str, user: str, max_tokens: int = 500) -> str | None:
    key = _groq_key()
    if not key:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    for attempt in range(2):
        try:
            resp = httpx.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429 and attempt == 0:
                # Groq returns Retry-After header in seconds
                wait = float(resp.headers.get("retry-after", 30)) + 2
                _last_error.append(f"Groq 429: rate-limited, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if not resp.is_success:
                try:
                    err = resp.json().get("error", {}).get("message", resp.text[:120])
                except Exception:
                    err = resp.text[:120]
                _last_error.append(f"Groq {resp.status_code}: {err}")
                return None
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            _last_error.append(f"Groq exception: {exc}")
            return None
    return None


# ── Gemini call ───────────────────────────────────────────────────────────────

def _call_gemini(system: str, user: str, max_tokens: int = 500) -> str | None:
    key = _gemini_key()
    if not key:
        return None
    model = _gemini_model()
    url = f"{_GEMINI_BASE}/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.3,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(2):
        try:
            resp = httpx.post(url, params={"key": key}, json=payload, timeout=90)
            if resp.status_code in (429, 503) and attempt == 0:
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                    m = re.search(r"retry in ([\d.]+)s", err_msg)
                    wait = min(float(m.group(1)) + 2, 65) if m else 60
                except Exception:
                    wait = 60
                _last_error.append(f"Gemini {resp.status_code}: rate-limited, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if not resp.is_success:
                try:
                    err = resp.json().get("error", {}).get("message", resp.text[:120])
                except Exception:
                    err = resp.text[:120]
                _last_error.append(f"Gemini {resp.status_code}: {err}")
                return None
            candidates = resp.json().get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0]["content"].get("parts", [])
            text_parts = [p.get("text", "") for p in parts if not p.get("thought", False)]
            if not text_parts:
                return None
            return text_parts[-1].strip()
        except Exception as exc:
            _last_error.append(f"Gemini exception: {exc}")
            return None
    return None


# ── Local SLM call (Ollama /v1/chat/completions, OpenAI-compatible) ──────────

def _call_local(system: str, user: str, max_tokens: int = 500) -> str | None:
    """Local small language model via Ollama — runs entirely on the VM, no
    network or API key. CPU inference is slow, so the timeout is generous."""
    payload = {
        "model": _local_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    try:
        resp = httpx.post(f"{_ollama_url()}/v1/chat/completions", json=payload, timeout=600)
        if not resp.is_success:
            _last_error.append(f"Local {resp.status_code}: {resp.text[:120]}")
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        _last_error.append(f"Local SLM exception: {exc}")
        return None


# ── Unified call — routes to local or cloud based on LLM_PROVIDER ─────────────

def _call(system: str, user: str, max_tokens: int = 500) -> str | None:
    mode = _llm_mode()
    if mode in ("local", "local_only"):
        result = _call_local(system, user, max_tokens)
        if result or mode == "local_only":
            return result
        _last_error.append("Local SLM failed — falling back to cloud")
    # Cloud path (default; unchanged)
    if _groq_key():
        result = _call_groq(system, user, max_tokens)
        if result:
            return result
        # Groq failed — try Gemini if available
        if _gemini_key():
            _last_error.append("Groq failed, falling back to Gemini")
            return _call_gemini(system, user, max_tokens)
        return None
    if _gemini_key():
        return _call_gemini(system, user, max_tokens)
    return None


def _provider_label() -> str:
    if _llm_mode() in ("local", "local_only"):
        return f"local:{_local_model()}"
    if _groq_key():
        return f"groq:{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}"
    if _gemini_key():
        return f"gemini:{_gemini_model()}"
    return "deterministic"


# Reference priority: downstream-impact adopters first, then applied-adoption
# signals, then citations. This ordering decides the [n] numbering.
_REF_PRIORITY = {"downstream": 0, "patent": 1, "code": 2, "funding": 3, "citation": 4, "full_text": 5}

# Structure SKELETON (placeholders, NOT prose) — teaches the shape without any
# wording to copy, avoiding the example-bleed a full prose example causes on both
# small and large models. Each <...> and [n] is filled with the paper's own facts.
_REF_STRUCTURE = (
    "<Surname> et al. (<year>) introduced <one clause: what the paper contributed>. "
    "<one sentence: citation reach or a specific downstream adopter, grounded in an approved fact> [1]. "
    "<one sentence: real-world or technical adoption, grounded in an approved fact> [2]. "
    "<one sentence: a further adoption or downstream pathway> [3], reflecting **<key impact phrase>**. "
    "<one sentence: broader disciplinary or societal reach> [4]. "
    "<closing sentence: overall significance for research impact>, evidencing **<key impact phrase>** "
    "and **<key impact phrase>**."
)

# Generic impact terms that legitimately occur in grounded summaries — used only
# as a bold fallback if the model emits no **bold** at all.
_BOLD_TARGETS = [
    "downstream influence", "downstream", "real-world adoption", "real-world impact",
    "widely adopted", "widely-adopted", "open-source", "practical adoption",
    "broad adoption", "significant impact", "citation reach", "state-of-the-art",
    "foundational", "clinical adoption", "patent",
]

def _ensure_bold(text: str, n: int = 3) -> str:
    """Small models often drop **bold** even when instructed. If fewer than two
    bold phrases are present, deterministically bold up to n high-signal impact
    phrases that actually appear in the text."""
    if not text or text.count("**") >= 4:
        return text
    added = text.count("**") // 2
    # Prefer concrete facts first — numbers with a unit (citations/stars/%/etc.).
    for pat in (r"\d[\d,]*\s*(?:citations|stars|patents|downloads|forks|publications)",
                r"\d[\d,]*%"):
        if added >= n:
            break
        m = re.search(r"(?<!\*)(" + pat + r")(?!\*)", text, re.IGNORECASE)
        if m:
            text = text[:m.start()] + "**" + m.group(1) + "**" + text[m.end():]
            added += 1
    # Then generic impact terms.
    for t in _BOLD_TARGETS:
        if added >= n:
            break
        pat = re.compile(r"(?<!\*)\b(" + re.escape(t) + r")\b(?!\*)", re.IGNORECASE)
        m = pat.search(text)
        if m:
            text = text[:m.start()] + "**" + m.group(1) + "**" + text[m.end():]
            added += 1
    return text

# Deterministic de-puffing: small models sometimes emit banned filler anyway.
# Word-level swaps keep grammar intact while removing the empty phrasing that the
# real REF samples never use.
_DEFLATE = {
    "profound influence": "influence", "significant impact": "impact",
    "transformative potential": "potential", "transformative": "notable",
    "broad applicability": "broad use", "cutting-edge": "recent",
    "tangible change": "change", "real-world scenarios": "practice",
    "underscoring its significance": "showing its reach",
    "underscore the significance": "show the reach",
    "further cementing": "confirming", "exemplifies": "shows", "profound": "clear",
}

def _deflate(text: str) -> str:
    for a, b in _DEFLATE.items():
        text = re.sub(re.escape(a), b, text, flags=re.IGNORECASE)
    return text

def _trim_words(text: str, hard_max: int = 165) -> str:
    """Safety net so the summary can't run away: keep whole sentences up to a
    hard word cap (small models sometimes ignore the length instruction)."""
    if not text or len(text.split()) <= hard_max:
        return text
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    count = 0
    for s in sents:
        w = len(s.split())
        if count + w > hard_max:
            break
        out.append(s); count += w
    if out:
        return " ".join(out)
    # First sentence alone busts the cap (a run-on from a small model): hard-cut
    # it rather than returning the whole thing and defeating the safety net.
    return " ".join(text.split()[:hard_max]).rstrip(" ,;:") + "."



def build_references(evidence: list[EvidenceItem], max_refs: int = 6,
                     exclude_doi: str | None = None, exclude_title: str | None = None) -> list[dict]:
    """Build the numbered reference list the summary cites with [n].

    Prioritises high-impact downstream adopters, then applied-adoption signals,
    then ordinary citations — so the most impact-bearing sources get the low
    numbers the model is told to lead with. The source paper itself is excluded
    (it is referred to as 'Author et al.', not cited as a reference)."""
    xd = (exclude_doi or "").lower().replace("https://doi.org/", "").strip()
    xt = (exclude_title or "").lower().strip()

    def is_self(e: EvidenceItem) -> bool:
        ed = (e.url or "").lower().replace("https://doi.org/", "")
        if xd and xd in ed:
            return True
        if xt and e.title and e.title.lower().strip() == xt:
            return True
        return False

    ranked = sorted(
        (e for e in evidence if not is_self(e)),
        key=lambda e: (_REF_PRIORITY.get(e.kind, 9), -(e.citation_count or 0)),
    )
    refs: list[dict] = []
    for e in ranked[:max_refs]:
        if e.authors:
            who = f"{e.authors[0].split()[-1]} et al." if len(e.authors) > 1 else e.authors[0]
        else:
            who = e.source
        label = f"{who} ({e.year or 'n.d.'}). {e.title[:100]}. {e.source}."
        note = (e.snippet or "")[:90] if e.kind == "downstream" else None
        refs.append({
            "n": len(refs) + 1,
            "label": label,
            "url": e.url,
            "kind": e.kind,
            "note": note,
        })
    return refs


# ── HFSynthesizer (same public interface, new multi-provider internals) ───────

class HFSynthesizer:
    """Multi-provider synthesiser: Groq → Gemini → deterministic fallback."""

    def enabled(self) -> bool:
        return bool(_llm_mode() in ("local", "local_only") or _groq_key() or _gemini_key())

    def generate(
        self,
        metadata: PaperMetadata,
        citation_count: int,
        evidence: list[EvidenceItem],
        topics: list[str],
        rag_contexts: list[str],
    ) -> tuple[str | None, str, list[dict], list[TraceLog]]:
        logs: list[TraceLog] = []
        references = build_references(evidence, exclude_doi=metadata.doi, exclude_title=metadata.title)
        if not self.enabled():
            logs.append(log("LLM", "No LLM key set — using deterministic synthesis. Add GROQ_API_KEY or GOOGLE_API_KEY to backend/.env."))
            return None, "deterministic", references, logs

        # Numbered reference block the model must cite from. Numbers here MUST
        # match the references returned to the UI.
        ref_block = "\n".join(
            f"[{r['n']}] ({r['kind']}) {r['label']}" + (f" — {r['note']}" if r.get("note") else "")
            for r in references
        ) or "No external sources retrieved."

        first_author = (metadata.authors[0].split()[-1] if metadata.authors else "the authors")
        etal = f"{first_author} et al. ({metadata.year or 'n.d.'})"

        system = (
            "You are a senior UK Research Excellence Framework (REF) impact assessor writing "
            "the 'summary of the impact' for a 4*-grade case study. Your prose is dense, "
            "specific, and impact-led: lead with the contribution, then evidence reach through "
            "concrete downstream outcomes — high-impact citing work, funded follow-on research, "
            "patents, and code adoption. Cite every external claim with a bracketed number [n] "
            "drawn ONLY from the numbered references provided. Refer to the paper itself as "
            "'<Author> et al. (year)'. Never invent companies, acquisitions, grants, or statistics "
            "— if an outcome is not in the references, do not state it."
        )

        user = f"""Write the REF 'summary of the impact' for this paper. Aim for ~100 words (90–115 acceptable), ONE flowing paragraph.

Paper (the subject): {metadata.title}
Refer to it as: {etal}
Headline citations: {citation_count:,}

Numbered references — these are the ONLY facts you may state. Cite each by its number [n]:
{ref_block}

HOW TO WRITE IT (grounded, no invention):
1. Open with one sentence on what {etal} introduced and why it mattered to its field.
2. Then write ONE dedicated sentence for EACH reference above (cover as many as you can — that is how you reach length), describing what that reference actually is — its topic/title and its citation figure or funder — and what it shows about downstream uptake, tagged with its [n]. Prefer references that out-cite the source paper or carry grant funding.
3. Close with one sentence on overall significance.

STRICT RULES:
- Length must come from covering MORE references, never from repetition or filler. Do NOT pad with phrases like 'highlighting the overall significance' or restating citation counts you already mentioned.
- Describe only what each reference literally is. Do NOT write 'follow-on collaboration', 'taken up by industry', 'spun out a company', 'presented to', or any outcome not written in a reference. If a reference is just a highly-cited citing paper, say exactly that (e.g. "a widely-cited work on X [n] (12,000 citations)").
- One paragraph, no headings/lists. Use 'et al.' for the paper; put [n] inline next to each supported claim; bold 2–3 key phrases with **double asterisks**.
- If downstream evidence is genuinely thin, state that honestly instead of inventing reach."""

        summary = _call(system, user, max_tokens=420)
        # Length-repair pass: models routinely under-shoot. If short, expand once
        # using ONLY the same references (no new facts) to hit the REF target.
        if summary and len(summary.split()) < 85:
            expanded = _call(
                system,
                "This REF impact summary is too short at "
                f"{len(summary.split())} words. Rewrite it to ~100 words using ONLY the references "
                "below — add no new facts, invent no outcomes. Reach the length by adding ONE new "
                "sentence for each reference you have not yet described (not by repeating points). "
                "Keep the [n] citations, the 'et al.' reference, and 2–3 **bold** phrases.\n\n"
                f"CURRENT SUMMARY:\n{summary}\n\nREFERENCES:\n{ref_block}",
                max_tokens=420,
            )
            if expanded and len(expanded.split()) > len(summary.split()):
                logs.append(log("LLM", f"Expanded summary {len(summary.split())}→{len(expanded.split())} words for REF length target"))
                summary = expanded
        if summary:
            provider = _provider_label()
            logs.append(log("LLM", f"{provider.split(':')[0].title()} generated REF impact narrative", model=provider, words=len(summary.split()), refs=len(references)))
            return summary, provider, references, logs

        err = _last_error[-1] if _last_error else "unknown error"
        logs.append(log("LLM", f"LLM failed ({err[:80]}) — falling back to deterministic synthesis"))
        return None, "deterministic", references, logs

    def draft_sentences(
        self,
        metadata: PaperMetadata,
        citation_count: int,
        evidence: list[EvidenceItem],
    ) -> tuple[list[dict], list[TraceLog]]:
        """Human-in-the-loop step: draft ONE factual, evidence-grounded
        candidate sentence per evidence item for the reviewer to curate. Each
        sentence stays strictly faithful to its single source."""
        logs: list[TraceLog] = []
        items = evidence[:24]
        # Local CPU inference: LLM-draft only the top items (the rest get
        # deterministic sentences) and cap output tokens, so the draft stage
        # stays ~10s instead of ~30s. Cloud drafts everything as before.
        is_local = _llm_mode() in ("local", "local_only")
        llm_items = items[:12] if is_local else items
        draft_max_tokens = 700 if is_local else 1400
        etal = f"{(metadata.authors[0].split()[-1] if metadata.authors else 'the authors')} et al. ({metadata.year or 'n.d.'})"

        def _fallback(e: EvidenceItem) -> str:
            bits = f"{e.kind.replace('_', ' ')} from {e.source}"
            cc = f" ({e.citation_count:,} citations)" if e.citation_count else ""
            return f"{e.title}{cc} — {bits} evidencing downstream engagement with the work."

        parsed: dict[int, str] = {}
        if self.enabled() and llm_items:
            ev_block = "\n".join(
                f"{i+1}. [{e.kind}] {e.title} ({e.source}, {e.year or 'n.d.'})"
                + (f", {e.citation_count:,} citations" if e.citation_count else "")
                + (f" — {(e.snippet or '')[:160]}" if e.snippet else "")
                for i, e in enumerate(llm_items)
            )
            system = (
                "You are a UK REF impact analyst drafting candidate evidence sentences "
                "for human review. Each sentence states ONE fact grounded strictly in its "
                "evidence item. Never invent outcomes, companies, or statistics."
            )
            user = (
                f"Paper: {metadata.title} ({etal}). Headline citations: {citation_count:,}.\n\n"
                "For EACH numbered evidence item below, write ONE concise, factual sentence "
                "(18-32 words) describing what it shows about the paper's real-world or academic "
                "impact. Stay strictly faithful to that item; do not invent. Output exactly one "
                "numbered line per item, reusing the same numbers, and nothing else.\n\n"
                f"Evidence:\n{ev_block}"
            )
            raw = _call(system, user, max_tokens=draft_max_tokens)
            if raw:
                for line in raw.splitlines():
                    m = re.match(r"\s*(\d+)[\.\)]\s*(.+\S)", line)
                    if m:
                        parsed[int(m.group(1))] = m.group(2).strip()
                logs.append(log("Draft", f"Drafted {len(parsed)} candidate sentences", model=_provider_label()))
            else:
                logs.append(log("Draft", "LLM unavailable — using deterministic candidate sentences"))

        candidates: list[dict] = []
        for i, e in enumerate(items):
            candidates.append({
                "id": i,
                "text": parsed.get(i + 1) or _fallback(e),
                "kind": e.kind,
                "source": e.source,
                "title": e.title,
                "url": e.url,
                "year": e.year,
                "citations": e.citation_count,
            })
        return candidates, logs

    def compose(
        self,
        metadata: PaperMetadata,
        citation_count: int,
        sentences: list[str],
        references: list[dict],
    ) -> tuple[str | None, str, list[TraceLog]]:
        """Weave the reviewer-approved sentences into ONE flowing REF impact
        paragraph. Uses only the facts already present in those sentences."""
        logs: list[TraceLog] = []
        if not self.enabled() or not sentences:
            return None, "deterministic", logs
        etal = f"{(metadata.authors[0].split()[-1] if metadata.authors else 'the authors')} et al. ({metadata.year or 'n.d.'})"
        sent_block = "\n".join(f"- {s}" for s in sentences)
        ref_block = "\n".join(f"[{r['n']}] {r['label']}" for r in references) or "None."
        system = (
            "You are a senior UK REF impact assessor. You are given reviewer-APPROVED, "
            "evidence-grounded sentences and must weave them into ONE flowing impact "
            "paragraph using ONLY the facts they contain — add nothing, invent nothing, "
            "drop none of their substance. You follow formatting rules exactly."
        )
        user = (
            "Write a UK Research Excellence Framework (REF) impact paragraph in the CONCRETE, "
            "evidence-led style of a real REF submission.\n\n"
            "=== STYLE (mandatory) ===\n"
            "- EVERY sentence states a specific, verifiable fact: a number, or a NAMED "
            "repository / patent / citing paper / venue / organisation. Lead with what the "
            "paper introduced, then trace concrete downstream impact.\n"
            "- NAME sources explicitly in the prose (e.g. \"the open-source library "
            "facebookresearch/detectron\", \"a patent assigned to <assignee>\", "
            "\"<Author> et al. (<year>)\"). Do not describe evidence generically.\n"
            "- BANNED — never use these empty phrases: \"significant impact\", \"transformative\", "
            "\"profound influence\", \"broad applicability\", \"underscoring its significance\", "
            "\"exemplifies\", \"cutting-edge\", \"tangible change\", \"real-world scenarios\", "
            "\"traceable pathway\", \"operational practice\".\n\n"
            "=== STRICT RULES ===\n"
            "1. MAXIMUM 150 words. One flowing paragraph — no headings, lists or bullets.\n"
            f"2. Refer to the authors as \"{etal}\".\n"
            "3. Use ONLY the approved facts below; add nothing, invent nothing.\n"
            "4. Include at least THREE inline citations as [1], [2], [3] (the reference numbers), "
            "each right after the fact it supports.\n"
            "5. Bold **exactly three** CONCRETE facts (a number, a named adopter, or a specific "
            "outcome) — bold the FACT itself, never a vague phrase.\n\n"
            f"=== APPROVED FACTS (weave in every one) ===\n{sent_block}\n\n"
            f"=== REFERENCE LIST (cite as [n]) ===\n{ref_block}\n\n"
            f"Now write the concrete REF impact paragraph (max 150 words) for {etal}:"
        )
        summary = _call(system, user, max_tokens=440)
        if summary:
            summary = _ensure_bold(_deflate(_trim_words(summary, hard_max=150)))
            provider = _provider_label()
            logs.append(log("LLM", f"{provider.split(':')[0].title()} composed summary from {len(sentences)} approved sentences", model=provider, words=len(summary.split())))
            return summary, provider, logs
        return None, "deterministic", logs

    def generate_ref_report(
        self,
        metadata: PaperMetadata,
        evidence: list[EvidenceItem],
        summary: str,
        topics: list[str],
    ) -> tuple[str | None, list[TraceLog]]:
        logs: list[TraceLog] = []
        if not self.enabled():
            return None, logs

        ev_all = "\n".join(
            f"- [{e.kind.upper()}] {e.title} ({e.source}, {e.year or 'n.d.'})"
            for e in evidence[:15]
        )
        system = (
            "You are a UK Research Excellence Framework (REF) impact analyst. "
            "Write concise, evidence-grounded academic prose. "
            "Use **bold** markdown for 3–4 key impact phrases only."
        )
        user = f"""Write a single REF-style impact paragraph. Target length: 120–130 words.

Paper: {metadata.title}
Authors: {", ".join(metadata.authors[:4]) or "Unknown"}
Year: {metadata.year or "Unknown"}
DOI: {metadata.doi or "N/A"}
Topics: {", ".join(topics[:4]) or "N/A"}
Overview: {summary}

Evidence:
{ev_all or "Limited evidence available."}

CONTENT — five sentences, each 24–26 words:
  1. Core contribution — what the paper introduced to the field, with context.
  2. Citation reach — state count and name 2 specific citing works from evidence.
  3. Real-world adoption — name specific patents and code repos from evidence with detail.
  4. Broader societal or disciplinary impact drawn from evidence.
  5. Overall REF significance — why this paper matters beyond academia.

FORMAT:
- ONE flowing paragraph, no headings, no lists.
- Use **bold** on 3–4 key impact phrases.
- No invented facts — ground every claim in the evidence above."""

        report = _call(system, user, max_tokens=500)
        if report:
            provider = _provider_label()
            logs.append(log("LLM", f"{provider.split(':')[0].title()} generated REF paragraph", model=provider))
            return report, logs
        return None, logs

    def evaluate_faithfulness(
        self,
        summary: str,
        evidence: list[EvidenceItem],
        rag_contexts: list[str],
    ) -> tuple[float, list[TraceLog]]:
        logs: list[TraceLog] = []
        if not self.enabled():
            return -1.0, logs

        ev = "\n".join(f"- {e.title}: {(e.snippet or '')[:150]}" for e in evidence[:10])
        prompt = f"""Rate faithfulness of this summary against the evidence (0–10, integer only).
0 = hallucinated, 10 = every claim supported. Output ONLY the integer.

Summary: {summary[:500]}

Evidence:
{ev}

Rating:"""

        result = _call("You are a strict academic faithfulness judge.", prompt, max_tokens=10)
        if result:
            m = re.search(r"\d+", result)
            if m:
                score = min(10, max(0, int(m.group(0)))) / 10.0
                provider = _provider_label()
                logs.append(log("Judge", f"Faithfulness score: {score}", model=provider))
                return score, logs
        return -1.0, logs

    def cross_validate_summary(
        self,
        summary: str,
        evidence: list[EvidenceItem],
        citation_count: int,
        metadata: "PaperMetadata",
    ) -> tuple[dict, list[TraceLog]]:
        """Independent peer-reviewer pass. Returns scored validation report."""
        logs: list[TraceLog] = []
        if not self.enabled():
            return {}, logs

        ev = "\n".join(
            f"- [{e.kind.upper()}] {e.title} ({e.source}, {e.year or 'n.d.'}): {(e.snippet or '')[:200]}"
            for e in evidence[:12]
        )
        system = (
            "You are an independent academic peer-reviewer assessing AI-generated research impact summaries. "
            "Be critical, precise, and evidence-grounded. Never be generous without justification."
        )
        prompt = f"""Peer-review this 100-word AI-generated research impact summary.

=== PAPER ===
Title: {metadata.title}
Authors: {", ".join(metadata.authors[:5]) or "Unknown"}
Year: {metadata.year or "?"}
Real citation count: {citation_count:,}

=== SUMMARY TO REVIEW ===
{summary}

=== AVAILABLE EVIDENCE ===
{ev or "No evidence available."}

Score each dimension 1–10 and give a one-line reason:
1. Accuracy – are all claims supported by evidence?
2. Completeness – are key impacts covered?
3. Conciseness – is it tight and free of padding?
4. Word count adherence – is it ~100 words?

Then write 2–3 sentences of overall feedback.

Respond in this exact JSON format:
{{
  "accuracy": <1-10>,
  "accuracy_reason": "<one line>",
  "completeness": <1-10>,
  "completeness_reason": "<one line>",
  "conciseness": <1-10>,
  "conciseness_reason": "<one line>",
  "word_count_score": <1-10>,
  "word_count_reason": "<one line>",
  "overall_score": <1-10>,
  "feedback": "<2-3 sentence peer review>"
}}"""

        raw = _call(system, prompt, max_tokens=600)
        if raw:
            try:
                json_match = re.search(r"\{[\s\S]*\}", raw)
                if json_match:
                    report = _json.loads(json_match.group(0))
                    provider = _provider_label()
                    logs.append(log("CrossValidator", "Peer review complete",
                                    overall=report.get("overall_score"), model=provider))
                    return report, logs
            except Exception as exc:
                logs.append(log("CrossValidator", f"JSON parse failed: {exc}", raw=raw[:100]))
        return {}, logs


hf_synthesizer = HFSynthesizer()
