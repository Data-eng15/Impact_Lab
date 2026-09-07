import React, { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  BrainCircuit,
  Check,
  ChevronDown,
  Clock,
  Code2,
  Coins,
  Database,
  Download,
  ExternalLink,
  FileText,
  GraduationCap,
  Globe,
  Landmark,
  Loader2,
  LogOut,
  Menu,
  Search,
  ShieldCheck,
  Sparkles,
  Settings,
  User,
  Mail,
  Building2,
  Linkedin,
  TrendingUp,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";

// ─── Types ──────────────────────────────────────────────────────────────────

type AgentState = "pending" | "running" | "complete" | "warning" | "error";

type AgentStatus = {
  name: string;
  label: string;
  state: AgentState;
  detail: string;
};

type TraceLog = {
  timestamp: string;
  agent: string;
  message: string;
  data: Record<string, unknown>;
};

type PaperMetadata = {
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  abstract: string | null;
  source_url: string | null;
};

type EvidenceItem = {
  title: string;
  url: string | null;
  year: number | null;
  authors: string[];
  snippet: string | null;
  source: string;
  kind: "citation" | "code" | "full_text" | "funding" | "patent" | "policy" | "grant" | string;
  citation_count: number | null;
  metric_label: string | null;
  metric_value: string | null;
};

type ImpactSection = {
  title: string;
  body: string;
};

type AccessCheck = {
  allowed: boolean;
  verified: boolean;
  demo_mode: boolean;
  matched_author: string | null;
  score: number;
  paper_authors: string[];
  reason: string;
};

type ValidationReport = {
  accuracy?: number;
  accuracy_reason?: string;
  completeness?: number;
  completeness_reason?: string;
  conciseness?: number;
  conciseness_reason?: string;
  word_count_score?: number;
  word_count_reason?: string;
  overall_score?: number;
  feedback?: string;
};

type AnalyzeResponse = {
  metadata: PaperMetadata;
  summary: string;
  sections: ImpactSection[];
  evidence: EvidenceItem[];
  agent_statuses: AgentStatus[];
  logs: TraceLog[];
  faithfulness_score: number;
  citation_count: number;
  topics: string[];
  model_provider: string;
  rag_context_count: number;
  guardrail_status: string;
  limitations: string[];
  ref_report: string;
  validation_report?: ValidationReport;
  access?: AccessCheck;
  routing?: RoutingMeta;
  references?: Reference[];
};

type Reference = { n: number; label: string; url: string | null; kind: string; note?: string | null };

type Candidate = { id: number; text: string; kind: string; source: string; title: string; url: string | null; year: number | null; citations: number | null };
type DraftResponse = {
  draft_id: string; stage: string; metadata: PaperMetadata;
  evidence: EvidenceItem[]; candidates: Candidate[]; citation_count: number;
  agent_statuses: AgentStatus[]; logs: TraceLog[]; topics: string[];
  routing?: RoutingMeta; access?: AccessCheck;
};

type RoutingMeta = {
  reason: string;
  domains: string[];
  profile_domains: string[];
  paper_domains: string[];
  sources_called: string[];
  sources_skipped: string[];
  all_sources: string[];
  fallback_triggered: boolean;
  evidence_count: number;
  saved_calls: number;
};

const SOURCE_LABELS: Record<string, string> = {
  semantic_scholar: "Semantic Scholar",
  openalex_fallback: "OpenAlex",
  openalex_enrichment: "OpenAlex Enrich",
  github: "GitHub",
  patents: "Google Patents",
  pubmed: "PubMed",
  clinical_trials: "ClinicalTrials",
  soft_sciences: "Policy / UKRI",
};

type SiteStats = {
  total_analyses: number;
  avg_faithfulness: number;
  avg_citations: number;
};

type SearchCandidate = {
  doi: string | null;
  title: string;
  authors: string[];
  year: number | null;
  venue: string;
  type: string;
  url: string | null;
  citation_count?: number;
  is_top_impact?: boolean;
};

type EvalSide = {
  approach: string;
  label?: string;
  summary: string;
  citation_count?: number;
  evidence_count: number;
  sources_used: string[];
  word_count: number;
  faithfulness_score: number;
  elapsed_seconds?: number | null;
  rag_contexts?: number;
};

type EvalComparison = {
  query: string;
  agentic: EvalSide;
  // New three-way shape:
  baseline_a?: EvalSide;        // CrossRef-only (no RAG)
  baseline_b?: EvalSide;        // Naive multi-source RAG
  agentic_faithfulness?: number;
  baseline_a_faithfulness?: number;
  baseline_b_faithfulness?: number;
  // Legacy single-baseline shape (back-compat):
  baseline?: EvalSide;
  verdict: string;
  agentic_full?: AnalyzeResponse;
};

type VerificationFlag = {
  claim: string;
  section: string;
  reason: string;
  severity: "warning" | "critical";
};

type WordCounts = {
  total: number;
  summary: number;
  research: number;
  impact: number;
  total_ok: boolean;
  summary_ok: boolean;
  research_ok: boolean;
  impact_ok: boolean;
};

type BetaRefResult = {
  case_study: string;
  flags: VerificationFlag[];
  word_counts: WordCounts;
  access?: AccessCheck;
  error?: string;
};

// ─── Constants ───────────────────────────────────────────────────────────────

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ngrok's free tier serves an HTML interstitial to browser requests unless this
// header is present. Harmless against non-ngrok backends. Injected on every API call.
const apiFetch = (input: string, init: RequestInit = {}) =>
  fetch(input, { ...init, headers: { "ngrok-skip-browser-warning": "true", ...(init.headers || {}) } });

const SAMPLE_QUERIES = [
  "10.1038/nature14539",
  "Attention Is All You Need",
  "10.1145/3292500.3330701",
];

const DATA_SOURCES = [
  "CrossRef", "Semantic Scholar", "GitHub", "Europe PMC",
  "OpenAlex", "USPTO Patents", "UKRI Gateway",
];

const INITIAL_STATUSES: AgentStatus[] = [
  { name: "supervisor",  label: "Supervisor",  state: "pending", detail: "Ready" },
  { name: "metadata",   label: "Metadata",    state: "pending", detail: "Ready" },
  { name: "scholar",    label: "Scholar",     state: "pending", detail: "Ready" },
  { name: "content",    label: "Content",     state: "pending", detail: "Ready" },
  { name: "code",       label: "Code",        state: "pending", detail: "Ready" },
  { name: "rag",        label: "RAG",         state: "pending", detail: "Ready" },
  { name: "guardrail",  label: "Guardrail",   state: "pending", detail: "Ready" },
  { name: "impact",     label: "Impact",      state: "pending", detail: "Ready" },
  { name: "synthesis",  label: "Synthesis",   state: "pending", detail: "Ready" },
];

const EVIDENCE_FILTERS = [
  { key: "all",        label: "All" },
  { key: "downstream", label: "Downstream impact" },
  { key: "industry_adoption", label: "Industry adoption" },
  { key: "policy_adoption",   label: "Policy adoption" },
  { key: "clinical_adoption", label: "Clinical adoption" },
  { key: "software_adoption", label: "Software adoption" },
  { key: "metadata_candidate", label: "Metadata candidates" },
  { key: "citation",  label: "Citations" },
  { key: "code",      label: "Code" },
  { key: "patent",    label: "Patents" },
  { key: "policy",    label: "Policy" },
  { key: "grant",     label: "Grants" },
  { key: "funding",   label: "Funding" },
  { key: "full_text", label: "Full text" },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Render **bold** markdown spans inside a string as <strong> elements. */
function BoldText({ text }: { text: string }) {
  // Split on **bold** and [n] citation markers, styling each.
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (/^\[\d+\]$/.test(part)) return <sup key={i} className="ref-marker">{part}</sup>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function formatAuthors(authors: string[]): string {
  if (!authors || !authors.length) return "Unknown authors";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

function scoreClass(score: number): string {
  if (score >= 0.85) return "strong";
  if (score >= 0.7) return "moderate";
  return "weak";
}

function scoreLabel(score: number): string {
  if (score >= 0.85) return "Strong";
  if (score >= 0.7) return "Developing";
  return "Needs evidence";
}

function agentIcon(name: string): React.ReactNode {
  const icons: Record<string, React.ReactNode> = {
    supervisor: <BrainCircuit size={14} />,
    metadata:   <Database size={14} />,
    scholar:    <GraduationCap size={14} />,
    content:    <FileText size={14} />,
    code:       <Code2 size={14} />,
    rag:        <BookOpen size={14} />,
    guardrail:  <ShieldCheck size={14} />,
    impact:     <Sparkles size={14} />,
    synthesis:  <Globe size={14} />,
  };
  return icons[name] ?? <Globe size={14} />;
}

function stepClass(state: AgentState): string {
  return `agent-step ${state}`;
}

/** Convert **bold** and *italic* markdown within a line to JSX. */
function inlineMarkdown(line: string, key: number): React.ReactNode {
  // Split on **bold** and *italic* patterns
  const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, j) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={j}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={j}>{part.slice(1, -1)}</em>;
    return part;
  });
}

function renderMarkdown(text: string): React.ReactNode {
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, i) => {
    if (line.startsWith("### ")) return <h3 key={i} className="ref-section-title">{line.replace(/^###\s*/, "")}</h3>;
    if (line.startsWith("## "))  return <h2 key={i} className="ref-section-title">{line.replace(/^##\s*/, "")}</h2>;
    if (line.startsWith("# "))   return <h2 key={i} className="ref-section-title">{line.replace(/^#\s*/, "")}</h2>;
    if (line.startsWith("- ") || line.startsWith("* "))
      return <p key={i} className="ref-list-item">• {inlineMarkdown(line.replace(/^[-*]\s*/, ""), i)}</p>;
    if (/^\d+\.\s/.test(line))
      return <p key={i} className="ref-list-item">{inlineMarkdown(line, i)}</p>;
    if (!line.trim()) return <br key={i} />;
    return <p key={i} className="ref-paragraph">{inlineMarkdown(line, i)}</p>;
  });
}

function applyInlineHtml(line: string): string {
  return line
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g,     "<em>$1</em>");
}

function downloadReport(content: string, filename: string) {
  const htmlContent = `<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head><meta charset='utf-8'><title>REF Impact Case Study</title></head>
<body style='font-family:Calibri,sans-serif;font-size:11pt;line-height:1.6'>
${content.split("\n").map(line => {
  if (line.startsWith("### ")) return `<h3 style='color:#4F46E5'>${applyInlineHtml(line.replace(/^###\s*/,""))}</h3>`;
  if (line.startsWith("## "))  return `<h2>${applyInlineHtml(line.replace(/^##\s*/,""))}</h2>`;
  if (line.startsWith("- ") || line.startsWith("* ")) return `<p style='margin-left:20px'>• ${applyInlineHtml(line.replace(/^[-*]\s*/,""))}</p>`;
  if (/^\d+\.\s/.test(line)) return `<p style='margin-left:20px'>${applyInlineHtml(line)}</p>`;
  if (!line.trim()) return "<br/>";
  return `<p>${applyInlineHtml(line)}</p>`;
}).join("")}
</body></html>`;
  const blob = new Blob([htmlContent], { type: "application/msword" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function withActiveStatuses(statuses: AgentStatus[]): AgentStatus[] {
  if (statuses.some(s => s.state === "running")) return statuses;
  return statuses.map((s, i) => i === 0 ? { ...s, state: "running", detail: "Starting…" } : s);
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function Logo() {
  return (
    <div className="rf-logo">
      <span className="rf-logo-mark" style={{ fontSize: 24 }}>
        R<span className="rf-logo-dot" />
      </span>
      <span className="rf-logo-wordmark" style={{ fontSize: 13 }}>REFlect<span className="rf-logo-ai"> AI</span></span>
    </div>
  );
}

type ProfileForm = {
  first: string; last: string; role: string; affil: string;
  email: string; orcid: string; linkedin: string; scholar: string;
};

function ProfileDrawer({
  open, onClose, seed, onLogout, onSaved, extra,
}: {
  open: boolean; onClose: () => void; seed: ProfileForm;
  onLogout: () => void; onSaved: () => void;
  extra?: { summary?: string; provider?: string; stats?: { label: string; value: string }[]; fields?: string[] };
}) {
  const [form, setForm] = useState<ProfileForm>(seed);
  useEffect(() => { setForm(seed); }, [seed, open]);
  if (!open) return null;
  const set = (k: keyof ProfileForm) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(prev => ({ ...prev, [k]: e.target.value }));
  const initials = `${(form.first[0] ?? "").toUpperCase()}${(form.last[0] ?? "").toUpperCase()}` || "R";
  const field = (label: string, k: keyof ProfileForm, icon?: React.ReactNode, optional?: boolean, mono?: boolean) => (
    <label className="rf-field">
      <span className="rf-field-label">{label}{optional && <span className="rf-field-optional">optional</span>}</span>
      <div className="rf-field-input">
        {icon}
        <input value={form[k]} onChange={set(k)} style={mono ? { fontFamily: "var(--font-mono)" } : undefined} />
      </div>
    </label>
  );
  return (
    <>
      <div onClick={onClose} className="rf-scrim" />
      <div className="rf-drawer">
        <div className="rf-drawer-head">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <User size={18} /><h3 className="rf-drawer-title">Your profile</h3>
          </div>
          <button onClick={onClose} aria-label="Close" className="rf-icon-btn"><X size={18} /></button>
        </div>

        <div className="rf-drawer-body">
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div className="rf-avatar-lg">{initials}</div>
            <div>
              <div className="rf-profile-name">{form.first} {form.last}</div>
              <div className="rf-profile-role">{form.role}</div>
            </div>
          </div>

          {extra?.stats && extra.stats.length > 0 && (
            <div className="rf-stat-grid">
              {extra.stats.map(s => (
                <div key={s.label} className="rf-stat-cell">
                  <div className="rf-stat-value">{s.value}</div>
                  <div className="rf-stat-label">{s.label}</div>
                </div>
              ))}
            </div>
          )}

          {extra?.summary && (
            <div className="rf-profile-summary">
              <div className="rf-profile-summary-head">
                <span className="rf-field-label" style={{ letterSpacing: "0.08em", textTransform: "uppercase", fontSize: 11 }}>AI profile summary</span>
                {extra.provider && <span className="rf-provider-chip">{extra.provider.split(":")[0]}</span>}
              </div>
              <p className="rf-profile-summary-text">{extra.summary}</p>
            </div>
          )}

          {extra?.fields && extra.fields.length > 0 && (
            <div>
              <div className="rf-field-label" style={{ marginBottom: 8 }}>Research fields</div>
              <div className="rf-field-chips">
                {extra.fields.map(f => <span key={f} className="rf-field-chip">{f}</span>)}
              </div>
            </div>
          )}

          <div className="rf-field-grid">
            {field("Name", "first")}
            {field("Surname", "last")}
          </div>
          {field("Role / title", "role")}
          {field("Affiliation", "affil", <Building2 size={16} />)}
          {field("Email", "email", <Mail size={16} />)}
          {field("ORCID iD", "orcid", <ShieldCheck size={16} />, false, true)}
          {field("LinkedIn", "linkedin", <Linkedin size={16} />, true)}
          {field("Google Scholar", "scholar", <GraduationCap size={16} />, true)}
        </div>

        <div className="rf-drawer-foot">
          <button onClick={onLogout} className="rf-signout"><LogOut size={16} /> Sign out</button>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={onSaved}>Save changes</button>
          </div>
        </div>
      </div>
    </>
  );
}

function StatusIcon({ state }: { state: AgentState }) {
  if (state === "complete") return <Check size={14} />;
  if (state === "running") return <Loader2 size={14} className="spin" />;
  if (state === "warning" || state === "error") return <AlertTriangle size={14} />;
  return <span className="step-dot" />;
}

function AgentLogPanel({ statuses }: { statuses: AgentStatus[] }) {
  return (
    <div className="agent-log-panel">
      {statuses.map((s, i) => (
        <div key={s.name} className="agent-step">
          <div className={`agent-nub ${s.state}`}>
            {s.state === "complete" && <Check size={11} />}
            {s.state === "running"  && <span className="running-dot" />}
            {s.state === "error"    && <AlertTriangle size={11} />}
            {(s.state === "pending" || s.state === "warning") && <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor", display: "inline-block" }} />}
          </div>
          <div className="agent-step-content">
            <div className={`agent-step-label ${s.state}`}>{s.label}</div>
            {s.detail && s.detail !== "Ready" && (
              <div className="agent-step-detail">{s.detail}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function AuthorBadge({ access }: { access: AccessCheck }) {
  if (access.verified) {
    return (
      <div className="author-badge verified">
        <ShieldCheck size={13} />
        Author verified: {access.matched_author} (score {access.score})
      </div>
    );
  }
  if (access.demo_mode) {
    return (
      <div className="author-badge demo">
        <AlertTriangle size={13} />
        Demo mode — {access.reason}
      </div>
    );
  }
  return (
    <div className="author-badge blocked">
      <AlertTriangle size={13} />
      Access restricted — {access.reason}
    </div>
  );
}

// ─── Peer-Review Scorecard ───────────────────────────────────────────────────
function PeerReviewCard({ report }: { report: ValidationReport }) {
  const dims: { key: keyof ValidationReport; label: string; reasonKey: keyof ValidationReport }[] = [
    { key: "accuracy",        label: "Accuracy",         reasonKey: "accuracy_reason" },
    { key: "completeness",    label: "Completeness",     reasonKey: "completeness_reason" },
    { key: "conciseness",     label: "Conciseness",      reasonKey: "conciseness_reason" },
    { key: "word_count_score",label: "Word Count",       reasonKey: "word_count_reason" },
  ];

  const scoreColor = (v?: number) => {
    if (v == null) return "#94a3b8";
    if (v >= 8) return "#22c55e";
    if (v >= 6) return "#f59e0b";
    return "#ef4444";
  };

  const hasAny = dims.some(d => report[d.key] != null);
  if (!hasAny) return null;

  return (
    <div className="peer-review-card">
      <div className="peer-review-header">
        <BrainCircuit size={15} />
        <span>AI Peer Review</span>
        {report.overall_score != null && (
          <span className="peer-overall" style={{ color: scoreColor(report.overall_score) }}>
            {report.overall_score}/10
          </span>
        )}
      </div>
      <div className="peer-dims">
        {dims.map(d => {
          const val = report[d.key] as number | undefined;
          const reason = report[d.reasonKey] as string | undefined;
          if (val == null) return null;
          return (
            <div key={d.key} className="peer-dim-row" title={reason ?? ""}>
              <span className="peer-dim-label">{d.label}</span>
              <div className="peer-bar-track">
                <div
                  className="peer-bar-fill"
                  style={{ width: `${val * 10}%`, background: scoreColor(val) }}
                />
              </div>
              <span className="peer-dim-score" style={{ color: scoreColor(val) }}>{val}</span>
            </div>
          );
        })}
      </div>
      {report.feedback && (
        <p className="peer-feedback">"{report.feedback}"</p>
      )}
    </div>
  );
}

function OverviewTab({
  result, loading, openSections, setOpenSections,
}: {
  result: AnalyzeResponse;
  loading: boolean;
  openSections: Record<string, boolean>;
  setOpenSections: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
}) {
  const statuses = loading ? withActiveStatuses(result.agent_statuses ?? INITIAL_STATUSES) : (result.agent_statuses ?? INITIAL_STATUSES);
  const ev = result.evidence ?? [];
  const citCount   = ev.filter(e => e.kind === "citation").length;
  const codeCount  = ev.filter(e => e.kind === "code").length;
  const patCount   = ev.filter(e => e.kind === "patent").length;
  const polCount   = ev.filter(e => e.kind === "policy").length;
  const grantCount = ev.filter(e => e.kind === "grant").length;
  const fundCount  = ev.filter(e => e.kind === "funding").length;
  const topicCount = result.topics?.length ?? 0;
  const total = Math.max(1, citCount + codeCount + patCount + polCount + grantCount + fundCount);

  return (
    <div className="overview-grid">
      {/* Left column */}
      <div className="overview-left">
        <div className="paper-header-block">
          <div className="paper-title-wrap">
            <p className="eyebrow">Impact Summary</p>
            <h2 className="paper-title">{result.metadata.title}</h2>
            <p className="paper-meta">
              {formatAuthors(result.metadata.authors)}
              {result.metadata.year ? ` · ${result.metadata.year}` : ""}
              {result.metadata.doi ? (
                <> · <a href={`https://doi.org/${result.metadata.doi}`} target="_blank" rel="noreferrer" className="doi-link">{result.metadata.doi}</a></>
              ) : null}
            </p>
            {/* Resolved-paper confirmation banner — prevents confusion when
                multiple papers share a title (e.g. republications) */}
            <div className="resolved-banner">
              <Check size={13} className="resolved-icon" />
              <span className="resolved-label">
                Analysing highest-impact match:&nbsp;
                <strong>{result.metadata.title}</strong>
                {result.metadata.year ? ` (${result.metadata.year})` : ""}
                {result.citation_count > 0
                  ? ` · ${result.citation_count.toLocaleString()} citations`
                  : ""}
                {result.metadata.doi ? (
                  <> · <a
                    href={`https://doi.org/${result.metadata.doi}`}
                    target="_blank"
                    rel="noreferrer"
                    className="resolved-doi"
                  >{result.metadata.doi}</a></>
                ) : " · no DOI (arXiv preprint)"}
              </span>
            </div>
          </div>
          <div className={`faith-chip ${scoreClass(result.faithfulness_score)}`}>
            <span className="faith-score">{result.faithfulness_score.toFixed(2)}</span>
            <span className="faith-label">{scoreLabel(result.faithfulness_score)}</span>
          </div>
        </div>

        {result.access && <AuthorBadge access={result.access} />}

        <div className="summary-block">
          <p className="summary-text"><BoldText text={result.summary} /></p>
        </div>

        {result.references && result.references.length > 0 && (
          <div className="ref-list">
            <div className="eyebrow" style={{ marginBottom: 8 }}>References</div>
            <ol className="ref-list-ol">
              {result.references.map(r => (
                <li key={r.n} className="ref-list-li">
                  <span className={`ref-kind-dot ref-kind-${r.kind}`} title={r.kind} />
                  {r.url
                    ? <a href={r.url.startsWith("http") ? r.url : `https://doi.org/${r.url}`} target="_blank" rel="noreferrer">{r.label}</a>
                    : <span>{r.label}</span>}
                  {r.kind === "downstream" && <span className="ref-downstream-tag">downstream impact</span>}
                </li>
              ))}
            </ol>
          </div>
        )}

        {result.validation_report && Object.keys(result.validation_report).length > 0 && (
          <PeerReviewCard report={result.validation_report} />
        )}

        {result.topics && result.topics.length > 0 && (
          <div className="topics-row">
            {result.topics.map(t => <span key={t} className="topic-chip">{t}</span>)}
          </div>
        )}

        <div className="runtime-row">
          <span className="runtime-chip">Model: {result.model_provider}</span>
          <span className="runtime-chip">Guardrail: {result.guardrail_status}</span>
        </div>

        {result.sections && result.sections.length > 0 && (
          <div className="section-list">
            {result.sections.map(section => (
              <div className="fold-section" key={section.title}>
                <button
                  className="fold-trigger"
                  onClick={() => setOpenSections(prev => ({ ...prev, [section.title]: !prev[section.title] }))}
                >
                  <ChevronDown size={16} className={openSections[section.title] ? "chevron open" : "chevron"} />
                  {section.title}
                </button>
                {openSections[section.title] && (
                  <div className="fold-body">{section.body}</div>
                )}
              </div>
            ))}
          </div>
        )}

        {result.limitations && result.limitations.length > 0 && (
          <div className="limitations-panel">
            <div className="limitations-header">
              <AlertTriangle size={16} />
              <span>Current limitations</span>
            </div>
            {result.limitations.map(lim => (
              <div key={lim} className="limitation-item">{lim}</div>
            ))}
          </div>
        )}

        {result.access && !result.access.verified && result.access.demo_mode && (
          <div className="security-notice">
            <ShieldCheck size={14} />
            <span>Running in demo mode. Author identity not verified — full REF document generation requires sign-in with a verified author account.</span>
          </div>
        )}
      </div>

      {/* Right column */}
      <div className="overview-right">
        <div className="metrics-grid">
          <div className="metric-card">
            <GraduationCap size={16} className="metric-icon" />
            <span className="metric-val">{result.citation_count ? result.citation_count.toLocaleString() : "—"}</span>
            <span className="metric-lbl">Citations</span>
          </div>
          <div className="metric-card">
            <FileText size={16} className="metric-icon" />
            <span className="metric-val">{ev.length}</span>
            <span className="metric-lbl">Evidence items</span>
          </div>
          <div className="metric-card">
            <Code2 size={16} className="metric-icon" />
            <span className="metric-val">{codeCount}</span>
            <span className="metric-lbl">Code leads</span>
          </div>
          <div className="metric-card">
            <Database size={16} className="metric-icon" />
            <span className="metric-val">{result.rag_context_count}</span>
            <span className="metric-lbl">RAG chunks</span>
          </div>
          <div className="metric-card">
            <Landmark size={16} className="metric-icon" />
            <span className="metric-val">{patCount}</span>
            <span className="metric-lbl">Patents</span>
          </div>
          <div className="metric-card">
            <Globe size={16} className="metric-icon" />
            <span className="metric-val">{polCount}</span>
            <span className="metric-lbl">Policy refs</span>
          </div>
          <div className="metric-card">
            <Coins size={16} className="metric-icon" />
            <span className="metric-val">{grantCount + fundCount}</span>
            <span className="metric-lbl">Grants/Funding</span>
          </div>
          <div className="metric-card">
            <Sparkles size={16} className="metric-icon" />
            <span className="metric-val">{topicCount}</span>
            <span className="metric-lbl">Topics</span>
          </div>
        </div>

        <div className="ev-dist-section">
          <p className="ev-dist-title">Evidence distribution</p>
          <div className="evidence-dist-bar">
            {citCount > 0  && <div className="dist-seg citation"  style={{ width: `${(citCount/total)*100}%`  }} title={`Citations: ${citCount}`} />}
            {codeCount > 0 && <div className="dist-seg code"      style={{ width: `${(codeCount/total)*100}%` }} title={`Code: ${codeCount}`} />}
            {patCount > 0  && <div className="dist-seg patent"    style={{ width: `${(patCount/total)*100}%`  }} title={`Patents: ${patCount}`} />}
            {polCount > 0  && <div className="dist-seg policy"    style={{ width: `${(polCount/total)*100}%`  }} title={`Policy: ${polCount}`} />}
            {grantCount > 0 && <div className="dist-seg grant"   style={{ width: `${(grantCount/total)*100}%` }} title={`Grants: ${grantCount}`} />}
            {fundCount > 0 && <div className="dist-seg funding"   style={{ width: `${(fundCount/total)*100}%` }} title={`Funding: ${fundCount}`} />}
          </div>
          <div className="dist-legend">
            {citCount > 0   && <span><span className="dist-dot citation"/>Citations ({citCount})</span>}
            {codeCount > 0  && <span><span className="dist-dot code"/>Code ({codeCount})</span>}
            {patCount > 0   && <span><span className="dist-dot patent"/>Patents ({patCount})</span>}
            {polCount > 0   && <span><span className="dist-dot policy"/>Policy ({polCount})</span>}
            {grantCount > 0 && <span><span className="dist-dot grant"/>Grants ({grantCount})</span>}
            {fundCount > 0  && <span><span className="dist-dot funding"/>Funding ({fundCount})</span>}
          </div>
        </div>

        <AgentLogPanel statuses={statuses} />
      </div>
    </div>
  );
}

function EvidenceTab({ evidence, filter, setFilter }: {
  evidence: EvidenceItem[];
  filter: string;
  setFilter: (f: string) => void;
}) {
  const kindCounts = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of evidence) m.set(e.kind, (m.get(e.kind) ?? 0) + 1);
    return m;
  }, [evidence]);

  const filtered = useMemo(() =>
    filter === "all" ? evidence : evidence.filter(e => e.kind === filter),
    [evidence, filter]
  );

  return (
    <div className="evidence-tab">
      <div className="evidence-filter-bar">
        {EVIDENCE_FILTERS.map(f => {
          const count = f.key === "all" ? evidence.length : (kindCounts.get(f.key) ?? 0);
          if (f.key !== "all" && count === 0) return null;
          return (
            <button
              key={f.key}
              className={`filter-pill ${filter === f.key ? "active" : ""}`}
              onClick={() => setFilter(f.key)}
            >
              {f.label} <span className="filter-count">{count}</span>
            </button>
          );
        })}
      </div>

      <div className="evidence-grid">
        {filtered.map((item, i) => (
          <div key={i} className="evidence-card">
            <div className="evidence-card-top">
              <span className={`kind-tag ${item.kind}`}>{item.kind.replace("_", " ")}</span>
              {item.url && (
                <a href={item.url} target="_blank" rel="noreferrer" className="ev-ext-link">
                  <ExternalLink size={13} />
                </a>
              )}
            </div>
            <h4 className="ev-title">{item.title}</h4>
            <p className="ev-meta">
              {item.source}
              {item.authors && item.authors.length > 0 ? ` · ${formatAuthors(item.authors)}` : ""}
              {item.year ? ` · ${item.year}` : ""}
            </p>
            {item.metric_label && item.metric_value && (
              <span className="ev-metric">{item.metric_label}: {item.metric_value}</span>
            )}
            {item.snippet && <em className="ev-snippet">"{item.snippet}"</em>}
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="empty-evidence">
            <FileText size={24} />
            No evidence in this category.
          </div>
        )}
      </div>
    </div>
  );
}

function RefReportTab({ refReport, doi }: { refReport: string; doi: string | null }) {
  const wordCount = refReport.split(/\s+/).filter(Boolean).length;
  return (
    <div className="ref-tab">
      <div className="ref-tab-header">
        <div>
          <h3>REF Impact Paragraph</h3>
          <p className="ref-tab-sub">{wordCount} words · copy-paste ready</p>
        </div>
        <button
          className="btn-primary"
          onClick={() => downloadReport(refReport, `REF_Impact_${(doi || "report").replace(/\//g,"_")}.doc`)}
        >
          <Download size={14} /> Download .doc
        </button>
      </div>
      <div className="ref-body ref-body-single">
        {renderMarkdown(refReport)}
      </div>
    </div>
  );
}

function DebugLogsTab({ logs }: { logs: TraceLog[] }) {
  return (
    <div className="debug-tab">
      <table className="debug-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Agent</th>
            <th>Message</th>
            <th>Data</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log, i) => (
            <tr key={i}>
              <td className="log-time">{log.timestamp}</td>
              <td><span className="log-agent-badge">{log.agent}</span></td>
              <td className="log-message">{log.message}</td>
              <td className="log-data">
                {Object.keys(log.data).length > 0
                  ? <code>{JSON.stringify(log.data, null, 0).slice(0, 80)}</code>
                  : <span className="log-empty">—</span>}
              </td>
            </tr>
          ))}
          {logs.length === 0 && (
            <tr><td colSpan={4} className="log-empty-row">No logs yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function EvalTab({ result, loading, error }: {
  result: EvalComparison | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="eval-loading">
        <Loader2 size={28} className="spin" />
        <p>Running agentic vs. baseline comparison…</p>
        <span>This re-runs the full pipeline alongside a CrossRef-only baseline.</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="eval-error">
        <AlertTriangle size={18} />
        Evaluation failed: {error}
      </div>
    );
  }
  if (!result) {
    return (
      <div className="eval-empty">
        <BrainCircuit size={28} />
        <p>Click "Force Evaluate" to run the agentic vs. baseline comparison.</p>
      </div>
    );
  }

  const ag = result.agentic;
  // Support both the new three-way shape and the legacy single-baseline shape.
  const naive = result.baseline_b ?? null;                         // Naive multi-source RAG
  const cross = result.baseline_a ?? result.baseline ?? null;      // CrossRef-only
  const fmt = (n: number | undefined | null) =>
    typeof n === "number" ? n.toFixed(2) : "—";

  // Columns to render (skip any baseline the backend didn't return)
  const cols: { key: string; label: string; side: EvalSide | null }[] = [
    { key: "ag",    label: ag.label ?? "REFlect AI (semantic RAG)", side: ag },
    { key: "naive", label: naive?.label ?? "Naive multi-source RAG", side: naive },
    { key: "cross", label: cross?.label ?? "CrossRef-only (no RAG)", side: cross },
  ].filter(c => c.side);

  const bestFaith = Math.max(...cols.map(c => c.side!.faithfulness_score ?? 0));

  return (
    <div className="eval-tab">
      <div className="verdict-banner agentic-wins">
        <BrainCircuit size={16} />
        <strong>Verdict:</strong> {result.verdict}
      </div>

      <div className="eval-comparison">
        <table className="eval-table">
          <thead>
            <tr>
              <th>Metric</th>
              {cols.map(c => <th key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Faithfulness score</td>
              {cols.map(c => (
                <td key={c.key}>
                  <span className={`eval-score-pill ${(c.side!.faithfulness_score ?? 0) >= bestFaith ? "winner" : ""}`}>
                    {fmt(c.side!.faithfulness_score)}
                  </span>
                </td>
              ))}
            </tr>
            <tr>
              <td>Evidence items</td>
              {cols.map(c => <td key={c.key}>{c.side!.evidence_count ?? 0}</td>)}
            </tr>
            <tr>
              <td>Sources used</td>
              {cols.map(c => <td key={c.key}>{c.side!.sources_used?.join(", ") || "—"}</td>)}
            </tr>
            <tr>
              <td>Summary length</td>
              {cols.map(c => <td key={c.key}>{c.side!.word_count ?? 0} words</td>)}
            </tr>
            <tr>
              <td>RAG contexts</td>
              {cols.map(c => (
                <td key={c.key}>
                  {c.key === "ag" ? (c.side!.rag_contexts ?? "—")
                    : c.key === "naive" ? "keyword"
                    : "0"}
                </td>
              ))}
            </tr>
            <tr>
              <td>Elapsed</td>
              {cols.map(c => (
                <td key={c.key}>
                  {c.side!.elapsed_seconds != null ? `${c.side!.elapsed_seconds}s` : "—"}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div className="eval-summaries">
        {cols.map(c => (
          <div key={c.key} className={`eval-summary-card${c.key === "ag" ? "" : " baseline"}`}>
            <div className="eval-summary-head">
              {c.key === "ag" ? <BrainCircuit size={14} /> : <Globe size={14} />} {c.label}
            </div>
            <p>{c.side!.summary}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function WordBar({ label, value, min, max, ok }: {
  label: string; value: number; min: number; max: number; ok: boolean;
}) {
  const pct = Math.min(100, Math.max(0, ((value - 0) / (max * 1.3)) * 100));
  const minPct = (min / (max * 1.3)) * 100;
  const maxPct = (max / (max * 1.3)) * 100;
  return (
    <div className="word-bar-row">
      <div className="word-bar-label">
        <span>{label}</span>
        <span className={`word-bar-count ${ok ? "ok" : "warn"}`}>{value} words</span>
      </div>
      <div className="word-bar-track">
        <div className="word-bar-fill" style={{ width: `${pct}%`, background: ok ? "#4F46E5" : "#F59E0B" }} />
        <div className="word-bar-min" style={{ left: `${minPct}%` }} title={`Min: ${min}`} />
        <div className="word-bar-max" style={{ left: `${maxPct}%` }} title={`Max: ${max}`} />
      </div>
      <span className={`word-bar-status ${ok ? "ok" : "warn"}`}>{ok ? "✓" : "!"}</span>
    </div>
  );
}

function renderCaseStudy(text: string): React.ReactNode {
  const UNVERIFIED_RE = /\[UNVERIFIED:\s*([^\]]+)\]/g;
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, i) => {
    if (line.startsWith("### ")) return <h3 key={i} className="ref-section-title">{line.replace(/^###\s*/,"")}</h3>;
    if (line.startsWith("## "))  return <h2 key={i} className="ref-section-title">{line.replace(/^##\s*/,"")}</h2>;
    if (line.startsWith("- ") || line.startsWith("* ")) {
      const parts = splitUnverified(line.replace(/^[-*]\s*/,""), UNVERIFIED_RE);
      return <p key={i} className="ref-list-item">• {parts}</p>;
    }
    if (/^\d+\.\s/.test(line)) {
      const parts = splitUnverified(line, UNVERIFIED_RE);
      return <p key={i} className="ref-list-item">{parts}</p>;
    }
    if (!line.trim()) return <br key={i} />;
    const parts = splitUnverified(line, UNVERIFIED_RE);
    return <p key={i} className="ref-paragraph">{parts}</p>;
  });
}

function splitUnverified(text: string, re: RegExp): React.ReactNode[] {
  const result: React.ReactNode[] = [];
  let last = 0;
  re.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) result.push(text.slice(last, m.index));
    result.push(<span key={m.index} className="unverified-claim" title={m[1]}>[UNVERIFIED: {m[1]}]</span>);
    last = m.index + m[0].length;
  }
  if (last < text.length) result.push(text.slice(last));
  return result;
}

function BetaRefTab({ result, loading, error, onRun }: {
  result: BetaRefResult | null;
  loading: boolean;
  error: string | null;
  onRun: () => void;
}) {
  if (loading) {
    return (
      <div className="beta-loading">
        <Loader2 size={28} className="spin" />
        <p>Generating REF 2029 impact case study…</p>
        <span>Writer agent crafting narrative · Auditor agent checking claims</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="eval-error">
        <AlertTriangle size={18} />
        Beta REF failed: {error}
      </div>
    );
  }
  if (!result) {
    return (
      <div className="eval-empty">
        <Sparkles size={28} />
        <p>Click "Beta REF" to generate a full REF 2029 impact case study with audit flags.</p>
      </div>
    );
  }

  const wc = result.word_counts;
  const criticals = result.flags.filter(f => f.severity === "critical").length;
  const warnings  = result.flags.filter(f => f.severity === "warning").length;

  return (
    <div className="beta-ref-tab">
      <div className="beta-header">
        <div>
          <span className="beta-badge">Beta REF 2029</span>
          <h3>Impact Case Study</h3>
          <p>AI-generated draft · subject to author review before submission</p>
        </div>
        <button
          className="btn-beta"
          onClick={() => result.case_study && downloadReport(result.case_study, "REF_2029_Beta_Case_Study.doc")}
        >
          <Download size={14} /> Export .doc
        </button>
      </div>

      <div className="word-counts-panel">
        <p className="wc-title">Word count compliance</p>
        <WordBar label="Total"               value={wc.total}    min={1800} max={2600} ok={wc.total_ok} />
        <WordBar label="Summary (§1)"        value={wc.summary}  min={80}   max={180}  ok={wc.summary_ok} />
        <WordBar label="Underpinning (§2)"   value={wc.research} min={150}  max={380}  ok={wc.research_ok} />
        <WordBar label="Details of Impact (§4)" value={wc.impact} min={600} max={950}  ok={wc.impact_ok} />
      </div>

      {result.flags.length > 0 && (
        <div className="flags-panel">
          <div className="flags-header">
            <AlertTriangle size={15} />
            <span>Auditor flags</span>
            {criticals > 0 && <span className="flag-count critical">{criticals} critical</span>}
            {warnings > 0  && <span className="flag-count warning">{warnings} warning</span>}
          </div>
          {result.flags.map((flag, i) => (
            <div key={i} className={`flag-item ${flag.severity}`}>
              <div className="flag-meta">
                <span className="flag-section">{flag.section}</span>
                <span className={`flag-sev ${flag.severity}`}>{flag.severity}</span>
              </div>
              <p className="flag-claim">"{flag.claim}"</p>
              <p className="flag-reason">{flag.reason}</p>
            </div>
          ))}
        </div>
      )}

      {result.error && (
        <div className="eval-error">
          <AlertTriangle size={15} /> {result.error}
        </div>
      )}

      <div className="case-study-body">
        {renderCaseStudy(result.case_study)}
      </div>
    </div>
  );
}

function CandidatePicker({ candidates, onPick }: {
  candidates: SearchCandidate[];
  onPick: (doi: string | null, title: string) => void;
}) {
  return (
    <div className="candidate-picker">
      <p className="picker-header">Multiple papers found — select one:</p>
      {candidates.map((c, i) => (
        <button key={i} className="candidate-row" onClick={() => onPick(c.doi, c.title)}>
          <div className="candidate-info">
            <span className="candidate-title">{c.title}</span>
            <span className="candidate-meta">
              {formatAuthors(c.authors)} {c.year ? `· ${c.year}` : ""} {c.venue ? `· ${c.venue}` : ""}
              {c.doi ? <> · <code>{c.doi}</code></> : null}
            </span>
          </div>
          <ArrowRight size={15} className="candidate-arrow" />
        </button>
      ))}
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

type ActiveTab = "overview" | "evidence" | "ref" | "logs" | "eval" | "betaref";

type TopPaper = { title: string; year: number | null; citations: number; venue: string | null; doi: string | null };
type ResearcherProfile = {
  source: string; orcid: string; name: string; first: string; last: string;
  affiliation: string | null; fields: string[]; works_count: number;
  total_citations: number; h_index: number; i10_index: number;
  top_papers: TopPaper[]; profile_summary: string; summary_provider: string;
};

const DEMO_ORCID = import.meta.env.VITE_DEMO_ORCID || "0000-0002-9322-3515";

const PIPELINE_STEPS: { label: string; caption: string; icon: React.ReactNode }[] = [
  { label: "Supervisor", caption: "Orchestrating the multi-agent pipeline", icon: <BrainCircuit size={24} /> },
  { label: "Metadata",   caption: "Resolving the paper across Crossref, OpenAlex & Semantic Scholar", icon: <FileText size={24} /> },
  { label: "Routing",    caption: "Selecting the field-relevant evidence sources", icon: <Database size={24} /> },
  { label: "Citations",  caption: "Tracing forward citations & downstream impact", icon: <TrendingUp size={24} /> },
  { label: "Evidence",   caption: "Gathering patents, code, clinical & policy signals", icon: <Landmark size={24} /> },
  { label: "Adoption",   caption: "Searching GitHub for real-world implementations", icon: <Code2 size={24} /> },
  { label: "RAG",        caption: "Embedding the evidence into the vector store", icon: <Sparkles size={24} /> },
  { label: "Validation", caption: "Scoring faithfulness against the evidence", icon: <ShieldCheck size={24} /> },
  { label: "Synthesis",  caption: "Writing the REF-grade impact narrative", icon: <Search size={24} /> },
];

function AnalysisProgress() {
  const [step, setStep] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const N = PIPELINE_STEPS.length;
  useEffect(() => {
    // Simulated progression: advance through stages, holding on the final one
    // until the real result arrives (this component then unmounts).
    const timers: number[] = [];
    let i = 0;
    const tick = () => {
      i = Math.min(i + 1, N - 1);
      setStep(i);
      if (i < N - 1) timers.push(window.setTimeout(tick, 2100 + Math.random() * 1000));
    };
    timers.push(window.setTimeout(tick, 1300));
    const sec = window.setInterval(() => setElapsed(e => e + 1), 1000);
    return () => { timers.forEach(clearTimeout); clearInterval(sec); };
  }, [N]);
  const cur = PIPELINE_STEPS[step];
  const pct = Math.round(((step + 0.5) / N) * 100);
  return (
    <div className="rf-prog">
      <div className="rf-prog-rail">
        <div className="rf-prog-rail-line"><div className="rf-prog-rail-fill" style={{ width: `${(step / (N - 1)) * 100}%` }} /></div>
        {PIPELINE_STEPS.map((s, i) => (
          <div key={i} className={`rf-prog-dot ${i < step ? "done" : i === step ? "active" : "pending"}`} title={s.label}>
            {i < step ? <Check size={11} strokeWidth={3} /> : <span className="rf-prog-dot-core" />}
          </div>
        ))}
      </div>

      <div className="rf-prog-stage">
        <div className="rf-prog-orb">
          <span className="rf-prog-orb-ring" />
          <span className="rf-prog-orb-ring rf-prog-orb-ring2" />
          <span className="rf-prog-orb-icon">{cur.icon}</span>
        </div>
        <div className="rf-prog-now" key={step}>
          <div className="rf-prog-eyebrow">Stage {step + 1} of {N} · {cur.label}</div>
          <div className="rf-prog-caption">{cur.caption}</div>
        </div>
      </div>

      <div className="rf-prog-bar"><div className="rf-prog-bar-fill" style={{ width: `${pct}%` }} /></div>
      <div className="rf-prog-meta">
        <span className="rf-prog-pct">{pct}%</span>
        <span className="rf-prog-elapsed"><span className="rf-prog-live" /> Analysing impact · {elapsed}s</span>
      </div>
    </div>
  );
}

const CURATE_KIND_LABEL: Record<string, string> = {
  downstream: "Downstream impact", citation: "Citation", patent: "Patent", code: "Code",
  funding: "Funding", policy: "Policy", grant: "Grant", full_text: "Full text",
};

function CurationView({ draft, selected, onToggle, onAll, onClear, onCompose, composing }: {
  draft: DraftResponse; selected: Set<number>;
  onToggle: (id: number) => void; onAll: () => void; onClear: () => void; onCompose: () => void; composing: boolean;
}) {
  const cands = draft.candidates;
  const n = selected.size;
  return (
    <div className="curate">
      <div className="eyebrow">Human-in-the-loop · evidence review</div>
      <h2 className="curate-title">Select the evidence sentences to include</h2>
      <p className="curate-sub">
        REFlect AI gathered the evidence and drafted one grounded sentence per item. Choose the ones you want —
        the final REF summary is composed <em>only</em> from your selection. Nothing is invented beyond these sentences.
      </p>

      <div className="curate-toolbar">
        <div className="curate-count"><strong>{n}</strong> of {cands.length} selected</div>
        <div className="curate-actions">
          <button className="btn btn-secondary" onClick={onAll}>Select all</button>
          <button className="btn btn-secondary" onClick={onClear}>Clear</button>
          <button className="btn btn-primary" disabled={n === 0 || composing} onClick={onCompose}>
            {composing ? <><Loader2 size={14} className="spin" /> Composing…</> : <><Sparkles size={14} /> Generate summary · {n}</>}
          </button>
        </div>
      </div>

      <div className="curate-list">
        {cands.length === 0 && <div className="curate-empty">No evidence sentences were drafted for this paper.</div>}
        {cands.map(c => {
          const on = selected.has(c.id);
          return (
            <button key={c.id} type="button" className={`curate-item${on ? " on" : ""}`} onClick={() => onToggle(c.id)}>
              <span className={`curate-check${on ? " on" : ""}`}>{on && <Check size={13} strokeWidth={3} />}</span>
              <span className="curate-body">
                <span className="curate-text">{c.text}</span>
                <span className="curate-meta">
                  <span className={`curate-kind kind-${c.kind}`}>{CURATE_KIND_LABEL[c.kind] ?? c.kind}</span>
                  <span className="curate-src">{c.source}{c.citations ? ` · ${c.citations.toLocaleString()} citations` : ""}</span>
                  {c.url && (
                    <a href={c.url.startsWith("http") ? c.url : `https://doi.org/${c.url}`} target="_blank" rel="noreferrer"
                       onClick={e => e.stopPropagation()} className="curate-link"><ExternalLink size={12} /></a>
                  )}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RoutingPanel({ routing }: { routing: RoutingMeta }) {
  if (!routing?.sources_called?.length) return null;
  const total = routing.all_sources.length;
  return (
    <div className="rf-routing">
      <div className="rf-routing-head">
        <span className="rf-routing-title"><Database size={14} /> Dynamic source routing</span>
        <span className="rf-routing-count">
          {routing.sources_called.length}/{total} sources queried · {routing.saved_calls} skipped
        </span>
      </div>
      <div className="rf-routing-domains">
        {routing.domains.length
          ? routing.domains.map(d => <span key={d} className="rf-domain-chip">{d.replace(/_/g, " ")}</span>)
          : <span className="rf-domain-chip muted">no clear domain — full scan</span>}
        <span className="rf-routing-reason" title="Why these sources were chosen">routed by: {routing.reason}</span>
      </div>
      <div className="rf-routing-sources">
        {routing.sources_called.map(s => (
          <span key={s} className="rf-src on"><Check size={11} /> {SOURCE_LABELS[s] || s}</span>
        ))}
        {routing.sources_skipped.map(s => (
          <span key={s} className="rf-src off">{SOURCE_LABELS[s] || s}</span>
        ))}
      </div>
      {routing.fallback_triggered && (
        <div className="rf-routing-fallback">
          ↻ Recall fallback: widened to all sources (routed pass returned too little evidence)
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const [query, setQuery]             = useState(SAMPLE_QUERIES[0]);
  const [sidebarOpen, setSidebarOpen] = useState(false);  // mobile drawer
  const [engine, setEngine]           = useState<"cloud" | "local">("local");  // LLM engine — local-first by default
  const [result, setResult]           = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading]         = useState(false);
  const [draft, setDraft]             = useState<DraftResponse | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [composing, setComposing]     = useState(false);
  const [lastQuery, setLastQuery]     = useState<string>("");
  const [error, setError]             = useState<string | null>(null);
  const [activeTab, setActiveTab]     = useState<ActiveTab>("overview");
  const [evidenceFilter, setEvidenceFilter] = useState("all");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    "Research Influence": true,
    "Applications": true,
    "Technical Adoption": true,
    "Access & Funding": true,
  });
  const [stats, setStats]             = useState<SiteStats | null>(null);
  const [history, setHistory]         = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("searchHistory") ?? "[]"); } catch { return []; }
  });
  const [evalResult, setEvalResult]   = useState<EvalComparison | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError]     = useState<string | null>(null);
  const [candidates, setCandidates]   = useState<SearchCandidate[]>([]);
  const [searchWarning, setSearchWarning] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [betaRef, setBetaRef]         = useState<BetaRefResult | null>(null);
  const [betaLoading, setBetaLoading] = useState(false);
  const [betaError, setBetaError]     = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [toast, setToast]             = useState<string | null>(null);
  const [profile, setProfile]         = useState<ResearcherProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);

  // Load stats
  useEffect(() => {
    apiFetch(`${API_URL}/api/stats`).then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  // Load user history from backend if logged in
  useEffect(() => {
    if (!user) return;
    user.getToken().then(token => {
      apiFetch(`${API_URL}/api/history`, { headers: { Authorization: `Bearer ${token}` } })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data?.queries) setHistory(data.queries.slice(0, 10)); })
        .catch(() => {});
    });
  }, [user]);

  // Document title
  useEffect(() => {
    document.title = result?.metadata?.title
      ? `${result.metadata.title} — REFlect AI`
      : "REFlect AI — Research Impact Analyser";
  }, [result]);

  function updateHistory(q: string) {
    const next = [q, ...history.filter(h => h !== q)].slice(0, 10);
    setHistory(next);
    localStorage.setItem("searchHistory", JSON.stringify(next));
  }

  async function getAuthHeader(): Promise<Record<string, string>> {
    if (!user) return {};
    try { return { Authorization: `Bearer ${await user.getToken()}` }; } catch { return {}; }
  }

  async function runAnalyze(resolvedQuery: string) {
    const routingFields = profile?.fields ?? [];
    setLoading(true); setError(null); setResult(null); setDraft(null);
    setBetaRef(null); setBetaError(null); setEvalResult(null); setEvalError(null);
    setLastQuery(resolvedQuery);

    try {
      const headers = { "Content-Type": "application/json", "X-LLM-Provider": engine, ...(await getAuthHeader()) };
      // Stage 1 — gather evidence + draft candidate sentences for curation.
      const resp = await apiFetch(`${API_URL}/api/analyze`, {
        method: "POST", headers, body: JSON.stringify({ query: resolvedQuery, fields: routingFields }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail ?? `API ${resp.status}`);
      }
      const data = await resp.json() as DraftResponse;
      setDraft(data);
      setSelectedIds(new Set(data.candidates.map(c => c.id)));  // default: all selected
      setActiveTab("overview");
      updateHistory(resolvedQuery);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  // Stage 2 — weave the reviewer-approved sentences into the final summary.
  async function composeSummary() {
    if (!draft || selectedIds.size === 0) return;
    setComposing(true); setError(null);
    try {
      const headers = { "Content-Type": "application/json", "X-LLM-Provider": engine, ...(await getAuthHeader()) };
      const resp = await apiFetch(`${API_URL}/api/compose`, {
        method: "POST", headers,
        body: JSON.stringify({ draft_id: draft.draft_id, selected_ids: [...selectedIds] }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail ?? `API ${resp.status}`);
      }
      const data = await resp.json() as AnalyzeResponse;
      setResult(data); setActiveTab("overview");
      if (lastQuery) runEval(lastQuery, true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Compose failed");
    } finally {
      setComposing(false);
    }
  }

  const toggleSentence = (id: number) =>
    setSelectedIds(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  async function analyze(event?: FormEvent, q?: string, retried = false) {
    event?.preventDefault();
    const searchQuery = (q ?? query).trim();
    if (!searchQuery) return;
    setQuery(searchQuery);
    setCandidates([]);
    setSearchWarning(null);

    // Detect if DOI / arXiv → go direct
    const isDirectId = /^10\.\d{4,9}\//.test(searchQuery) || /^\d{4}\.\d{4,5}/.test(searchQuery);
    if (isDirectId) { await runAnalyze(searchQuery); return; }

    // Title → search first
    setSearchLoading(true);
    try {
      const resp = await apiFetch(`${API_URL}/api/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await resp.json();
      const cands: SearchCandidate[] = data.candidates ?? [];
      // Low confidence = Semantic Scholar (the only reliable canonical source)
      // was unavailable. Retry once after a short backoff before showing results.
      if (data.low_confidence && !retried) {
        await new Promise(r => setTimeout(r, 1600));
        setSearchLoading(false);
        return analyze(undefined, searchQuery, true);
      }
      const warn = !!data.low_confidence;
      if (cands.length === 1 && !warn) {
        await runAnalyze(cands[0].doi ?? cands[0].title);
      } else if (cands.length >= 1) {
        // When low-confidence, always show the picker (never auto-run a paper
        // that might be the wrong "canonical" one) and warn the user.
        if (warn) setSearchWarning("The citation index (Semantic Scholar) was briefly unavailable, so the canonical highest-cited paper may be missing from this list. Choose carefully, or re-run the search in a moment.");
        setCandidates(cands);
      } else {
        await runAnalyze(searchQuery);
      }
    } catch {
      await runAnalyze(searchQuery);
    } finally {
      setSearchLoading(false);
    }
  }

  async function runEval(forceQuery?: string, autoMode = false) {
    const q = forceQuery ?? query;
    if (!q) return;
    setEvalLoading(true); setEvalError(null);
    if (!autoMode) setActiveTab("eval");
    try {
      const headers = { "Content-Type": "application/json", "X-LLM-Provider": engine, ...(await getAuthHeader()) };
      const resp = await apiFetch(`${API_URL}/api/evaluate`, {
        method: "POST", headers, body: JSON.stringify({ query: q }),
      });
      if (!resp.ok) throw new Error(`Eval API ${resp.status}`);
      const data = await resp.json() as EvalComparison;
      setEvalResult(data);
    } catch (e) {
      setEvalError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setEvalLoading(false);
    }
  }

  async function runBetaRef() {
    if (!result) return;
    setBetaLoading(true); setBetaError(null); setBetaRef(null); setActiveTab("betaref");
    try {
      const headers = { "Content-Type": "application/json", "X-LLM-Provider": engine, ...(await getAuthHeader()) };
      const body = {
        query: result.metadata.title ?? query,
        title: result.metadata.title,
        authors: result.metadata.authors ?? [],
        year: result.metadata.year,
        doi: result.metadata.doi,
        citation_count: result.citation_count ?? 0,
        summary: result.summary ?? "",
        evidence: (result.evidence ?? []).map(e => ({
          title: e.title, snippet: e.snippet, kind: e.kind,
          source: e.source, url: e.url, year: e.year, authors: e.authors,
        })),
      };
      const resp = await apiFetch(`${API_URL}/api/ref/beta`, {
        method: "POST", headers, body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : `API ${resp.status}`);
      }
      const data = await resp.json() as BetaRefResult;
      setBetaRef(data);
    } catch (e) {
      setBetaError(e instanceof Error ? e.message : "Beta REF failed");
    } finally {
      setBetaLoading(false);
    }
  }

  async function handleLogout() {
    await logout();
    localStorage.removeItem("isAuthenticated");
    navigate("/");
  }

  function handleExportCSV() {
    window.open(`${API_URL}/api/dataset?fmt=csv`, "_blank");
  }

  const hasResult = !!(result || evalResult || evalLoading || betaLoading || betaRef);
  const showStatuses = loading
    ? withActiveStatuses(INITIAL_STATUSES)
    : (result?.agent_statuses ?? INITIAL_STATUSES);

  // ── Personalised profile (ORCID/OpenAlex) ────────────────
  const manual = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem("rf_manual") || "null"); }
    catch { return null; }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setProfileLoading(true);
      let qs: string;
      if (user?.orcid) {
        qs = `orcid=${encodeURIComponent(user.orcid)}`;
      } else if (manual?.name) {
        const parts = [`name=${encodeURIComponent(manual.name)}`];
        if (manual.affiliation) parts.push(`affiliation=${encodeURIComponent(manual.affiliation)}`);
        qs = parts.join("&");
      } else {
        qs = `orcid=${DEMO_ORCID}`;
      }
      try {
        const res = await apiFetch(`${API_URL}/api/profile?${qs}`).then(r => r.json());
        if (!cancelled) setProfile(res?.resolved ? res.profile : null);
      } catch {
        if (!cancelled) setProfile(null);
      } finally {
        if (!cancelled) setProfileLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [user?.orcid, manual]);

  const firstName = profile?.first || user?.displayName?.split(/\s+/)[0] || manual?.first || "Researcher";
  const lastName = profile?.last || user?.displayName?.split(/\s+/).slice(1).join(" ") || manual?.last || "";
  const greetLast = lastName || firstName;
  const profileSeed: ProfileForm = {
    first: firstName,
    last: lastName,
    role: profile?.fields?.[0] ? `Researcher · ${profile.fields[0]}` : "Researcher",
    affil: profile?.affiliation || manual?.affiliation || "—",
    email: user?.email || manual?.email || "—",
    orcid: profile?.orcid || user?.orcid || manual?.orcid || "—",
    linkedin: manual?.linkedin || "—",
    scholar: manual?.scholar || "—",
  };
  const [returning] = useState(() => localStorage.getItem("rf_seen") === "1");
  useEffect(() => { localStorage.setItem("rf_seen", "1"); }, []);

  function saveProfile() {
    setProfileOpen(false);
    setToast("Profile updated");
    setTimeout(() => setToast(null), 2400);
  }

  return (
    <main className="app-shell">
      {/* ── Sidebar ────────────────────────────────────────── */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar${sidebarOpen ? " sidebar-open" : ""}`} aria-label="Workspace navigation">
        <Logo />

        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <TrendingUp size={13} />
            <div className="eyebrow">Your top papers</div>
          </div>
          {profileLoading ? (
            <div className="rf-toppapers-loading">
              <Loader2 size={13} className="spin" /> Building your profile…
            </div>
          ) : (profile?.top_papers?.length ? (
            <div className="rf-toppapers">
              {profile.top_papers.slice(0, 6).map((p, i) => (
                <button
                  key={(p.doi ?? p.title) + i}
                  className="rf-toppaper"
                  onClick={() => analyze(undefined, p.doi || p.title)}
                  title={p.title}
                >
                  <span className="rf-toppaper-rank">{i + 1}</span>
                  <span className="rf-toppaper-body">
                    <span className="rf-toppaper-title">{p.title}</span>
                    <span className="rf-toppaper-meta">
                      {p.citations.toLocaleString()} citations{p.year ? ` · ${p.year}` : ""}
                    </span>
                  </span>
                </button>
              ))}
            </div>
          ) : history.length ? (
            <div className="sidebar-history-list">
              {history.map(item => (
                <button key={item} className="sidebar-history-item" onClick={() => analyze(undefined, item)}>
                  {item}
                </button>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: "var(--fg-3)", lineHeight: "18px", padding: "4px 2px" }}>
              No publications linked to your ORCID iD yet. Search any paper above to analyse it.
            </div>
          ))}
        </div>

        <div className="sidebar-section">
          <div className="eyebrow" style={{ marginBottom: 8 }}>Data sources</div>
          <div className="sidebar-tags-row">
            {DATA_SOURCES.map(s => (
              <span key={s} className="tag tag-neutral">{s}</span>
            ))}
          </div>
        </div>

        {profile && (
          <div className="sidebar-section">
            <div className="eyebrow" style={{ marginBottom: 10 }}>Research impact</div>
            <div className="sidebar-stat">
              <span className="sidebar-stat-label">Publications</span>
              <span className="sidebar-stat-value">{profile.works_count.toLocaleString()}</span>
            </div>
            <div className="sidebar-stat" style={{ marginTop: 6 }}>
              <span className="sidebar-stat-label">Total citations</span>
              <span className="sidebar-stat-value">{profile.total_citations.toLocaleString()}</span>
            </div>
            <div className="sidebar-stat" style={{ marginTop: 6 }}>
              <span className="sidebar-stat-label">h-index</span>
              <span className="sidebar-stat-value">{profile.h_index}</span>
            </div>
            {profile.i10_index ? (
              <div className="sidebar-stat" style={{ marginTop: 6 }}>
                <span className="sidebar-stat-label">i10-index</span>
                <span className="sidebar-stat-value">{profile.i10_index}</span>
              </div>
            ) : null}
          </div>
        )}

        <button className="sidebar-export" onClick={handleExportCSV}>
          <Download size={13} /> Export dataset CSV
        </button>

        <div className="sidebar-audit-note">
          <span style={{ color: "var(--accent)", marginTop: 1 }}><ShieldCheck size={15} /></span>
          <p>Every agent step and API call is logged. Open the debug logs tab for full auditability.</p>
        </div>

        <div className="sidebar-user" style={{ marginTop: "auto" }}>
          <button
            className="sidebar-user-main"
            onClick={() => setProfileOpen(true)}
            title="View and edit your profile"
          >
            <div className="sidebar-avatar">
              {user?.photoURL
                ? <img src={user.photoURL} alt="" style={{ width: "100%", height: "100%", borderRadius: "50%" }} />
                : `${(firstName[0] ?? "R")}${(lastName[0] ?? "")}`.toUpperCase()
              }
            </div>
            <div style={{ flex: 1, minWidth: 0, textAlign: "left" }}>
              <div className="sidebar-user-name">{profile ? `${firstName} ${lastName}`.trim() : (user?.displayName ?? "Researcher")}</div>
              <div className="sidebar-user-email">{profile?.affiliation ?? user?.email ?? "Demo mode"}</div>
            </div>
            <Settings size={15} style={{ color: "var(--fg-3)", flexShrink: 0 }} />
          </button>
          <button className="sidebar-logout" onClick={handleLogout} title="Sign out">
            <LogOut size={15} />
          </button>
        </div>

        <div className="sidebar-credit">
          Designed &amp; developed by <strong>Soham Dharne</strong> (MSc AI)<br />
          under the supervision of <strong>Dr Raza Haider</strong>
        </div>
      </aside>

      <ProfileDrawer
        open={profileOpen}
        onClose={() => setProfileOpen(false)}
        seed={profileSeed}
        onLogout={handleLogout}
        onSaved={saveProfile}
        extra={profile ? {
          summary: profile.profile_summary,
          provider: profile.summary_provider,
          fields: profile.fields,
          stats: [
            { label: "h-index", value: String(profile.h_index) },
            { label: "Citations", value: profile.total_citations.toLocaleString() },
            { label: "Works", value: profile.works_count.toLocaleString() },
            { label: "i10-index", value: String(profile.i10_index) },
          ],
        } : undefined}
      />

      {toast && <div className="rf-toast"><Check size={15} /> {toast}</div>}

      {/* ── Workspace ──────────────────────────────────────── */}
      <section className="workspace">
        {/* Topbar */}
        <header className="topbar">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <div className="topbar-left">
            <div className="eyebrow">Researcher workspace</div>
            <h1 className="topbar-title">Research impact summariser</h1>
          </div>
          <div className="topbar-right">
            <div className="live-indicator">
              <span className="live-dot" />
              Connected to live APIs
            </div>
            <a
              href="https://github.com/sohamdharne/AI_Research_Impact-summariser"
              target="_blank" rel="noreferrer"
              className="btn btn-secondary"
              style={{ fontSize: 13, padding: "7px 12px" }}
            >
              <BookOpen size={13} /> Documentation
            </a>
          </div>
        </header>

        {/* Query panel */}
        <form className="query-panel" onSubmit={analyze}>
          <div className="query-input-row">
            <div className="query-input-wrap">
              <Search size={16} style={{ color: "var(--fg-3)", flexShrink: 0 }} />
              <input
                id="query-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Paper DOI, title, arXiv ID, or BibTeX entry"
              />
            </div>
            <div className="query-actions">
              <button type="submit" className="btn btn-primary" disabled={loading || searchLoading}>
                {loading || searchLoading
                  ? <><Loader2 size={14} className="spin" /> Analysing…</>
                  : <><Search size={14} /> Analyse</>
                }
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={evalLoading || loading}
                onClick={() => runEval(undefined, false)}
                title="Force a fresh agentic vs. baseline evaluation"
              >
                {evalLoading ? <Loader2 size={14} className="spin" /> : <BrainCircuit size={14} />}
                {evalResult ? "Re-evaluate" : "Run Evaluation"}
              </button>
              {/* Beta REF removed – not required */}
            </div>
          </div>

          <div className="engine-toggle">
            <span className="engine-label">Engine</span>
            <button type="button" className={`engine-opt${engine === "local" ? " active" : ""}`} onClick={() => setEngine("local")}>Local · on-VM 3B</button>
            <button type="button" className={`engine-opt${engine === "cloud" ? " active" : ""}`} onClick={() => setEngine("cloud")}>Cloud</button>
            <span className="engine-note">{engine === "local"
              ? "Default — private on-server inference, auto cloud fallback if quality dips (~20–30s)"
              : "Groq → Gemini · fastest (~8s) · uses shared API quota"}</span>
          </div>

          <div className="quick-pills">
            {SAMPLE_QUERIES.map(s => (
              <button type="button" key={s} className="quick-pill" onClick={() => analyze(undefined, s)}>
                {s}
              </button>
            ))}
          </div>
        </form>

        {/* Tab nav */}
        {hasResult && (
          <nav className="tabs-bar" aria-label="Result tabs">
            <button className={`tab-btn ${activeTab === "overview" ? "active" : ""}`} onClick={() => setActiveTab("overview")}>
              Overview
            </button>
            {result && (
              <button className={`tab-btn ${activeTab === "evidence" ? "active" : ""}`} onClick={() => setActiveTab("evidence")}>
                Evidence
                <span className={`tab-count ${activeTab === "evidence" ? "active-count" : ""}`}>{result.evidence.length}</span>
              </button>
            )}
            {result?.ref_report && (
              <button className={`tab-btn ${activeTab === "ref" ? "active" : ""}`} onClick={() => setActiveTab("ref")}>
                REF report
              </button>
            )}
            {result && (
              <button className={`tab-btn ${activeTab === "logs" ? "active" : ""}`} onClick={() => setActiveTab("logs")}>
                Debug logs
                <span className={`tab-count ${activeTab === "logs" ? "active-count" : ""}`}>{result.logs.length}</span>
              </button>
            )}
            <button className={`tab-btn ${activeTab === "eval" ? "active" : ""}`} onClick={() => setActiveTab("eval")}>
              Evaluation
              {evalLoading
                ? <Loader2 size={11} className="spin" style={{ marginLeft: 4 }} />
                : evalResult
                  ? <span className="tab-count active-count">vs</span>
                  : null}
            </button>
              {/* Beta REF tab removed */}
          </nav>
        )}

        {/* Search resolving indicator */}
        {searchLoading && (
          <div style={{ padding: "12px 40px", display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "var(--fg-2)", borderBottom: "1px solid var(--rule)", background: "var(--linen)" }}>
            <Loader2 size={13} className="spin" />
            Searching CrossRef for paper…
          </div>
        )}

        {/* Candidate picker */}
        {candidates.length > 0 && (
          <div className="candidate-picker-overlay" onClick={e => e.target === e.currentTarget && (setCandidates([]), setSearchWarning(null))}>
            <div className="candidate-picker-sheet">
              <div className="candidate-picker-title">Multiple papers found — select one to analyse</div>
              <div className="candidate-picker-sub">{searchWarning ? "Listed by citation count from the available sources." : "Ranked by citation impact — the highest-impact version is highlighted."}</div>
              {searchWarning && (
                <div className="candidate-picker-warning">
                  <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
                  <span>{searchWarning}</span>
                </div>
              )}
              {candidates.map((c, i) => (
                <div
                  key={i}
                  className={`candidate-item${c.is_top_impact ? " candidate-item-top" : ""}`}
                  onClick={() => { setCandidates([]); runAnalyze(c.doi ?? c.title); }}
                >
                  <div className="candidate-row">
                    <div className="candidate-title">{c.title}</div>
                    {c.is_top_impact && (
                      <span className="candidate-badge">★ Highest impact</span>
                    )}
                  </div>
                  <div className="candidate-authors">
                    {formatAuthors(c.authors)}{c.year ? ` · ${c.year}` : ""}{c.venue ? ` · ${c.venue}` : ""}
                  </div>
                  <div className="candidate-meta-row">
                    {typeof c.citation_count === "number" && (
                      <span className={`candidate-cites${c.is_top_impact ? " candidate-cites-top" : ""}`}>
                        {c.citation_count.toLocaleString()} citations
                      </span>
                    )}
                    {c.doi && <span className="candidate-doi">{c.doi}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div style={{ margin: "16px 40px", padding: "12px 16px", background: "var(--rust-100)", color: "var(--rust)", borderRadius: "var(--r-sm)", display: "flex", alignItems: "center", gap: 10, fontSize: 14 }} role="alert">
            <AlertTriangle size={15} />
            {error}
          </div>
        )}

        {/* Tab content */}
        <div className="tab-content">
          {!hasResult && !loading && !draft && (
            <div className="empty-state">
              <div className="rf-greeting-eyebrow">{returning ? "Welcome back" : "Welcome"}</div>
              <h2 className="rf-greeting">
                {returning ? "Welcome back, " : "Welcome, "}
                <span className="rf-greeting-name">Researcher {greetLast}</span>.
              </h2>
              <div className="empty-body">
                Paste a DOI, title, arXiv ID, or BibTeX entry above and REFlect AI will trace its
                downstream influence — citations, code, patents, policy, and funding — then draft
                evidence sentences for you to review before composing the REF-ready narrative.
              </div>
            </div>
          )}

          {loading && !result && <AnalysisProgress />}

          {draft && !result && !loading && (
            <CurationView
              draft={draft} selected={selectedIds} composing={composing}
              onToggle={toggleSentence}
              onAll={() => setSelectedIds(new Set(draft.candidates.map(c => c.id)))}
              onClear={() => setSelectedIds(new Set())}
              onCompose={composeSummary}
            />
          )}

          {result && activeTab === "overview" && result.routing && (
            <RoutingPanel routing={result.routing} />
          )}

          {result && activeTab === "overview" && (
            <OverviewTab
              result={result}
              loading={loading}
              openSections={openSections}
              setOpenSections={setOpenSections}
            />
          )}

          {result && activeTab === "evidence" && (
            <EvidenceTab
              evidence={result.evidence}
              filter={evidenceFilter}
              setFilter={setEvidenceFilter}
            />
          )}

          {result && activeTab === "ref" && result.ref_report && (
            <RefReportTab refReport={result.ref_report} doi={result.metadata.doi} />
          )}

          {result && activeTab === "logs" && (
            <DebugLogsTab logs={result.logs} />
          )}

          {activeTab === "eval" && (
            <EvalTab result={evalResult} loading={evalLoading} error={evalError} />
          )}

          {/* betaref tab removed */}
        </div>
      </section>
    </main>
  );
}
