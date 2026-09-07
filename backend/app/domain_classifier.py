"""
Domain classifier for REFlect AI domain-aware routing.

A zero-dependency, multi-label keyword classifier that detects a paper's
research domain(s) from its title and abstract. The detected domains drive
source routing (see DOMAIN_SOURCE_MAP and services.retrieval_node) so that we
only call the evidence APIs relevant to each paper instead of all of them.

Scoring is deterministic, instant and dependency-free, and rests on three
rules:

1. **Word-boundary matching.** Keywords match whole words only, so "industrial"
   no longer counts as a "trial" (medicine) and "general" no longer counts as a
   "gene" (biology).
2. **Weighted keywords.** A diagnostic term ("convolutional") outweighs a leaky
   generic one ("algorithm"), so a single generic word can never route on its
   own.
3. **Longest-match wins.** Each span of text is claimed by exactly one keyword,
   the most specific one. "neural network" scores computer science *instead of*
   scoring biology for "neural"; "clinical trial" does not also score the bare
   "trial".
"""
from __future__ import annotations

import re
from typing import List, Tuple

# ── Keyword weights ──────────────────────────────────────────────────────────
# STRONG terms are diagnostic on their own; NORMAL terms are indicative; WEAK
# terms leak across disciplines ("algorithm" appears in medicine, "market" in
# sociology) and are kept only as corroborating signal.
STRONG = 2.0
NORMAL = 1.0
WEAK = 0.5

# ── Domain taxonomy ──────────────────────────────────────────────────────────
# Multi-word phrases are matched as whole-word sequences, tolerant of
# punctuation and hyphens ("EEG-based brain-computer interface" matches
# "brain computer interface") and of a trailing plural.
DOMAIN_KEYWORDS: dict[str, dict[str, float]] = {
    "computer_science": {
        "neural network": STRONG, "deep learning": STRONG,
        "machine learning": STRONG, "artificial intelligence": STRONG,
        "reinforcement learning": STRONG, "ensemble learning": STRONG,
        "supervised learning": STRONG, "unsupervised learning": STRONG,
        "transfer learning": STRONG, "federated learning": STRONG,
        "large language model": STRONG, "natural language processing": STRONG,
        "computer vision": STRONG, "support vector machine": STRONG,
        "random forest": STRONG, "gradient boosting": STRONG,
        "convolutional": STRONG, "transformer": STRONG,
        "brain computer interface": STRONG, "data mining": STRONG,
        "feature extraction": STRONG, "feature selection": STRONG,
        "classification algorithm": STRONG, "clustering algorithm": STRONG,
        "image recognition": STRONG, "speech recognition": STRONG,
        "generative adversarial": STRONG, "knowledge graph": STRONG,
        "covariate shift": STRONG, "classifier": STRONG,
        "nlp": NORMAL, "llm": NORMAL, "ai": NORMAL, "software": NORMAL,
        "open source": NORMAL, "source code": NORMAL,
        "embedding": NORMAL, "training data": NORMAL, "cybersecurity": NORMAL,
        "encryption": NORMAL, "blockchain": NORMAL, "cloud computing": NORMAL,
        "semantic segmentation": NORMAL, "machine translation": NORMAL,
        "algorithm": WEAK, "dataset": WEAK, "benchmark": WEAK,
        "computational": WEAK, "optimization": WEAK,
    },
    "medicine": {
        "randomized controlled trial": STRONG, "randomised controlled trial": STRONG,
        "clinical trial": STRONG, "chemotherapy": STRONG, "vaccine": STRONG,
        "epidemiology": STRONG, "comorbidity": STRONG, "biomarker": STRONG,
        "oncology": STRONG, "cardiovascular": STRONG, "immunotherapy": STRONG,
        "patient outcome": STRONG, "prognosis": STRONG, "therapeutic": STRONG,
        "clinical": NORMAL, "patient": NORMAL, "disease": NORMAL,
        "therapy": NORMAL, "treatment": NORMAL, "drug": NORMAL,
        "hospital": NORMAL, "diagnosis": NORMAL, "biomedical": NORMAL,
        "cancer": NORMAL, "tumor": NORMAL, "tumour": NORMAL,
        "surgery": NORMAL, "mortality": NORMAL, "morbidity": NORMAL,
        "symptom": NORMAL, "syndrome": NORMAL, "infection": NORMAL,
        "pathology": NORMAL, "nursing": NORMAL, "primary care": NORMAL,
        "randomized": NORMAL, "randomised": NORMAL,
        "trial": WEAK, "health": WEAK, "medical": WEAK, "screening": WEAK,
        "intervention": WEAK,
    },
    "finance": {
        "stock market": STRONG, "financial market": STRONG,
        "capital market": STRONG, "asset pricing": STRONG,
        "monetary policy": STRONG, "cryptocurrency": STRONG, "fintech": STRONG,
        "venture capital": STRONG, "exchange rate": STRONG,
        "hedge fund": STRONG, "credit risk": STRONG,
        "financial": NORMAL, "investment": NORMAL, "stock": NORMAL,
        "economic": NORMAL, "economics": NORMAL, "revenue": NORMAL,
        "portfolio": NORMAL, "trading": NORMAL, "equity": NORMAL,
        "banking": NORMAL, "inflation": NORMAL, "valuation": NORMAL,
        "profitability": NORMAL,
        "market": WEAK, "cost": WEAK, "price": WEAK,
    },
    "law": {
        "constitutional law": STRONG, "criminal law": STRONG,
        "human rights": STRONG, "case law": STRONG, "jurisprudence": STRONG,
        "judicial review": STRONG, "legislation": STRONG,
        "jurisdiction": STRONG, "statutory": STRONG,
        "legal": NORMAL, "court": NORMAL, "judicial": NORMAL,
        "compliance": NORMAL, "governance": NORMAL, "regulatory": NORMAL,
        "litigation": NORMAL, "tribunal": NORMAL, "treaty": NORMAL,
        "policy": WEAK, "regulation": WEAK, "reform": WEAK,
    },
    "engineering": {
        "finite element": STRONG, "control system": STRONG,
        "structural engineering": STRONG, "mechanical engineering": STRONG,
        "civil engineering": STRONG, "signal processing": STRONG,
        "thermodynamics": STRONG, "fluid dynamics": STRONG,
        "aerodynamics": STRONG, "semiconductor": STRONG,
        "photovoltaic": STRONG, "additive manufacturing": STRONG,
        "actuator": STRONG,
        "manufacturing": NORMAL, "mechanical": NORMAL, "electrical": NORMAL,
        "structural": NORMAL, "robotics": NORMAL, "automation": NORMAL,
        "hardware": NORMAL, "sensor": NORMAL, "turbine": NORMAL,
        "fabrication": NORMAL, "circuit": NORMAL, "embedded system": NORMAL,
        "composite material": NORMAL,
        "materials": WEAK, "engineering": WEAK,
    },
    "social_science": {
        "social policy": STRONG, "welfare reform": STRONG,
        "ethnography": STRONG, "sociology": STRONG, "anthropology": STRONG,
        "social capital": STRONG, "public opinion": STRONG,
        "focus group": STRONG, "pedagogy": STRONG, "curriculum": STRONG,
        "qualitative interview": STRONG,
        "social": NORMAL, "psychological": NORMAL, "behavioral": NORMAL,
        "behavioural": NORMAL, "survey": NORMAL, "education": NORMAL,
        "cognitive": NORMAL, "adolescent": NORMAL, "gender": NORMAL,
        "inequality": NORMAL, "wellbeing": NORMAL, "qualitative": NORMAL,
        "community": WEAK, "culture": WEAK, "student": WEAK,
    },
    "biology": {
        "genomics": STRONG, "gene expression": STRONG,
        "molecular biology": STRONG, "microbiology": STRONG,
        "protein structure": STRONG, "dna sequencing": STRONG,
        "crispr": STRONG, "phylogenetic": STRONG, "biodiversity": STRONG,
        "ecosystem": STRONG, "neuroscience": STRONG,
        "electroencephalography": STRONG, "eeg": STRONG,
        "motor imagery": STRONG, "synaptic": STRONG, "neuron": STRONG,
        "protein": NORMAL, "cell": NORMAL, "molecular": NORMAL,
        "gene": NORMAL, "organism": NORMAL, "ecology": NORMAL,
        "evolution": NORMAL, "enzyme": NORMAL, "bacteria": NORMAL,
        "chromosome": NORMAL, "mutation": NORMAL, "brain": NORMAL,
        "neural": NORMAL, "cortex": NORMAL, "physiological": NORMAL, "species": NORMAL,
        "metabolism": NORMAL, "rna": NORMAL,
        "biological": WEAK, "tissue": WEAK,
    },
}

# ── Source routing config ────────────────────────────────────────────────────
# ── Source vocabulary (aligned to the live retrieval_node tasks) ─────────────
# Core sources are ALWAYS queried, regardless of discipline:
#   - Semantic Scholar / OpenAlex carry citation signal for any field;
#   - Google Patents is a primary REF impact pathway (commercial/industrial
#     translation) and is relevant across every discipline — a machine-learning
#     method is just as patentable as a chemical process — so it is never
#     skipped. Domain routing only gates the remaining *optional* sources.
CORE_SOURCES: list[str] = ["semantic_scholar", "openalex_fallback", "openalex_enrichment", "patents"]
OPTIONAL_SOURCES: list[str] = ["github", "pubmed", "clinical_trials", "soft_sciences"]
ALL_SOURCES: list[str] = CORE_SOURCES + OPTIONAL_SOURCES

# Maps each detected domain to the *optional* sources worth calling for it.
# (Core sources — incl. patents — are added automatically.) "general" routes
# to everything.
DOMAIN_SOURCE_MAP: dict[str, list[str]] = {
    "computer_science": ["github"],
    "medicine":         ["pubmed", "clinical_trials"],
    "finance":          ["soft_sciences"],
    "law":              ["soft_sciences"],
    "engineering":      ["github"],
    "social_science":   ["soft_sciences"],
    "biology":          ["pubmed", "clinical_trials"],
    "general":          list(OPTIONAL_SOURCES),
}

# Maps OpenAlex concept/field display names (lower-cased substrings) → our domain
# taxonomy, so routing can be driven by a researcher's *profile* fields.
FIELD_DOMAIN_MAP: dict[str, str] = {
    "computer science": "computer_science", "artificial intelligence": "computer_science",
    "machine learning": "computer_science", "artificial neural network": "computer_science",
    "data science": "computer_science", "information retrieval": "computer_science",
    "medicine": "medicine", "internal medicine": "medicine", "oncology": "medicine",
    "surgery": "medicine", "pathology": "medicine", "immunology": "medicine",
    "psychiatry": "medicine", "pharmacology": "medicine", "cardiology": "medicine",
    "biology": "biology", "genetics": "biology", "biochemistry": "biology",
    "molecular biology": "biology", "microbiology": "biology", "neuroscience": "biology",
    "economics": "finance", "finance": "finance", "financial economics": "finance",
    "law": "law", "political science": "law", "public administration": "law",
    "engineering": "engineering", "mechanical engineering": "engineering",
    "electrical engineering": "engineering", "materials science": "engineering",
    "robotics": "engineering", "control theory": "engineering",
    "sociology": "social_science", "psychology": "social_science",
    "education": "social_science", "social psychology": "social_science",
}


def fields_to_domains(fields: list[str]) -> list[str]:
    """Map a researcher's OpenAlex fields/concepts to our domain taxonomy.

    Returns an ordered, de-duplicated list of domain keys. Empty if no field
    matches (caller should treat that as ``general`` / full fallback)."""
    domains: list[str] = []
    for f in fields or []:
        fl = (f or "").lower().strip()
        for needle, domain in FIELD_DOMAIN_MAP.items():
            if needle in fl and domain not in domains:
                domains.append(domain)
                break
    return domains


def sources_for_domain_keys(domains: list[str]) -> list[str]:
    """Core sources + the union of optional sources for the given domain keys."""
    out: list[str] = list(CORE_SOURCES)
    if not domains:
        return list(ALL_SOURCES)
    for d in domains:
        for src in DOMAIN_SOURCE_MAP.get(d, []):
            if src not in out:
                out.append(src)
    return out

# Confidence thresholds.
MIN_DOMAIN_CONFIDENCE = 0.15   # below this a domain is not reported at all
LOW_CONFIDENCE_CUTOFF = 0.40   # below this a domain does not drive routing

# The title is the strongest domain signal, so its matches count double.
TITLE_WEIGHT = 2.0

# Saturation constant: the weighted score at which a domain reaches full (1.0)
# confidence. Calibrated against the weights above so that a single diagnostic
# term in the title (STRONG x TITLE_WEIGHT = 4.0) lands at 0.67 — confidently
# past the routing cutoff — while two generic WEAK terms reach only 0.33 and
# cannot route on their own.
CONFIDENCE_SATURATION = 6.0

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lower-case and reduce punctuation to single spaces.

    Hyphens and slashes become spaces so that real-world titles like
    "EEG-based brain-computer interface" match the phrase
    "brain computer interface".
    """
    return _PUNCT_RE.sub(" ", (text or "").lower()).strip()


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    """Compile a keyword into a whole-word pattern tolerant of a trailing plural."""
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"(?:es|s)?\b")


# Patterns are ordered most-specific first (more words, then longer text) so
# that the longest match claims a span before any shorter keyword can.
_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (_compile_phrase(phrase), domain, weight)
    for phrase, domain, weight in sorted(
        (
            (phrase, domain, weight)
            for domain, keywords in DOMAIN_KEYWORDS.items()
            for phrase, weight in keywords.items()
        ),
        key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
    )
]


def _score_text(text: str) -> dict[str, float]:
    """Sum keyword weights in ``text``, giving each span to one keyword only.

    Spans already claimed by a more specific keyword are masked out, so
    "neural network" scores computer science without also scoring biology for
    "neural".
    """
    scores: dict[str, float] = {}
    if not text:
        return scores
    claimed = bytearray(len(text))
    for pattern, domain, weight in _KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(claimed[start:end]):
                continue
            claimed[start:end] = b"\x01" * (end - start)
            scores[domain] = scores.get(domain, 0.0) + weight
    return scores


def classify_domain(title: str, abstract: str = "") -> List[Tuple[str, float]]:
    """Classify the research domain(s) of a paper from its title and abstract.

    Args:
        title: Paper title (required).
        abstract: Paper abstract (optional but strongly improves accuracy).

    Returns:
        A list of ``(domain, confidence)`` tuples sorted by confidence
        descending, including only domains scoring above
        :data:`MIN_DOMAIN_CONFIDENCE`. If nothing clears the threshold, returns
        ``[("general", 1.0)]`` which triggers the full-source fallback.

    Example:
        >>> classify_domain("Deep learning for stock market prediction")
        [('computer_science', 0.667), ('finance', 0.667)]
    """
    totals: dict[str, float] = {}
    for domain, score in _score_text(_normalize(title)).items():
        totals[domain] = totals.get(domain, 0.0) + score * TITLE_WEIGHT
    for domain, score in _score_text(_normalize(abstract)).items():
        totals[domain] = totals.get(domain, 0.0) + score

    scored: list[Tuple[str, float]] = []
    for domain, score in totals.items():
        confidence = min(1.0, score / CONFIDENCE_SATURATION)
        if confidence > MIN_DOMAIN_CONFIDENCE:
            scored.append((domain, round(confidence, 3)))

    if not scored:
        return [("general", 1.0)]

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def is_low_confidence(domains: List[Tuple[str, float]]) -> bool:
    """Return True if the top detected domain is below the confidence cutoff.

    A low-confidence result means the classifier is unsure of the domain, so the
    caller should fall back to querying all sources rather than a narrow subset.
    """
    if not domains:
        return True
    top_domain, top_conf = domains[0]
    if top_domain == "general":
        return True
    return top_conf < LOW_CONFIDENCE_CUTOFF


def sources_for_domains(domains: List[Tuple[str, float]]) -> List[str]:
    """Return the de-duplicated union of source whitelists for the given domains.

    Only domains at or above :data:`LOW_CONFIDENCE_CUTOFF` contribute their
    whitelist; weaker secondary domains are ignored to keep routing tight. If no
    domain qualifies, returns the full "general" source set.
    """
    qualifying = [d for d, conf in domains if conf >= LOW_CONFIDENCE_CUTOFF]
    if not qualifying:
        return list(ALL_SOURCES)
    return sources_for_domain_keys(qualifying)


def plan_routing(fields: list[str] | None = None, title: str = "", abstract: str = "") -> dict:
    """Decide which evidence sources to query for a researcher and/or a paper.

    Phase-2 "dynamic gravity": routing is driven primarily by the researcher's
    *profile* fields, with the paper's own classified domain layered on top — so
    a CS professor analysing a paper that was, say, cited in a medical context
    still picks up the relevant sources (the "twist"). Core scholarly sources are
    always included; recall is protected at retrieval time by an
    insufficient-evidence fallback to the full source set.

    Returns a metadata dict describing the decision (visible in the UI).
    """
    profile_domains = fields_to_domains(fields or [])
    paper_scored = classify_domain(title, abstract) if title else []
    paper_domains = [d for d, c in paper_scored if c >= LOW_CONFIDENCE_CUTOFF and d != "general"]

    combined: list[str] = []
    for d in profile_domains + paper_domains:
        if d not in combined:
            combined.append(d)

    if not combined:
        sources_called = list(ALL_SOURCES)
        reason = "no_clear_domain"
    else:
        sources_called = sources_for_domain_keys(combined)
        if profile_domains and paper_domains:
            reason = "profile+paper"
        elif profile_domains:
            reason = "profile"
        else:
            reason = "paper"

    sources_skipped = [s for s in ALL_SOURCES if s not in sources_called]
    return {
        "profile_domains": profile_domains,
        "paper_domains": paper_domains,
        "domains": combined,
        "sources_called": sources_called,
        "sources_skipped": sources_skipped,
        "all_sources": list(ALL_SOURCES),
        "reason": reason,
        "saved_calls": len(sources_skipped),
    }
