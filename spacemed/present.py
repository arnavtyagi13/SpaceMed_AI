"""Display shaping for SpaceMed results: evidence dicts -> human-readable structures.

Pure functions only: no Streamlit, no retrieval, no I/O. Everything here is
deterministic over its inputs and safe to unit-test. Scientific behavior lives
in spacemed.engine; this module only decides how results are *shown*.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence display vocabulary (single source of truth for labels/badges).
# Engine tiers are observed/established/inferred; UI shows Hypothesis, never
# "Inferred", so a reader never mistakes a suggestion for a measurement.
# ---------------------------------------------------------------------------
EV_DISPLAY = {"Observed": "Observed", "Established": "Established", "Inferred": "Hypothesis"}
EV_BADGE = {"Observed": "badge-observed", "Established": "badge-established", "Inferred": "badge-hypothesis"}
EV_DOT = {"Observed": "🔵", "Established": "🟢", "Inferred": "🟡"}
EV_ORDER = {"Established": 0, "Observed": 1, "Inferred": 2}

# Display budgets (how much we show, never what we conclude).
MAX_FINDING_GENES = 4
MAX_TRAIL_GENES = 3
MAX_SHORT_GENES = 5
MAX_SHORT_TISSUES = 3
MAX_SHORT_STRESSORS = 2
MAX_PATHWAY_FRAGMENTS = 2
FINDINGS_POOL_MIN = 12
ROLE_MAX_LEN = 110
GENE_LIT_MAX_LEN = 120
REF_CTX_MAX_LEN = 220
MAX_REF_ROWS = 12


def evidence_key(raw: object) -> str:
    """Canonical tier name: Observed, Established, or Inferred.

    Unknown tiers degrade to Inferred (never to a stronger claim) and are
    logged, so a typo like "Observd" cannot silently pass as measured data.
    """
    text = str(raw or "Inferred").strip()
    if text.lower() == "inferred":
        return "Inferred"
    key = text.capitalize()
    if key not in EV_DISPLAY:
        log.warning("unknown evidence tier %r; treating as Inferred", text)
        return "Inferred"
    return key


def evidence_label(raw: object) -> str:
    """Human label for a tier. Inferred is always shown as Hypothesis."""
    return EV_DISPLAY.get(evidence_key(raw), "Hypothesis")


def arrow(direction: object) -> str:
    """Direction glyph: ↑ / ↓ / ''. Text labels elsewhere carry the meaning."""
    d = str(direction or "").strip().lower()
    if d.startswith("up"):
        return "↑"
    if d.startswith("down"):
        return "↓"
    return ""


def arrow_span(direction: object) -> str:
    """Direction glyph wrapped for card styling."""
    a = arrow(direction)
    if a == "↑":
        return "<span class='arrow-up'>↑</span>"
    if a == "↓":
        return "<span class='arrow-down'>↓</span>"
    return ""


def direction_word(direction: object) -> str:
    """Human direction: Increased / Decreased / raw fallback."""
    d = str(direction or "").strip().lower()
    if d.startswith("up"):
        return "Increased"
    if d.startswith("down"):
        return "Decreased"
    return str(direction or "—")


def format_fc(value: object) -> str:
    """Signed log2FC with 2 decimals; em dash when unparseable."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return f"+{f:.2f}" if f >= 0 else f"{f:.2f}"
    except Exception:
        return "—"


def shorten_pathway(pathway: object) -> str:
    """First readable pathway fragment ('KEGG:mmu04310 Wnt signaling pathway' → 'Wnt signaling pathway')."""
    for part in str(pathway or "").split(";"):
        frag = re.sub(r"^\s*(GO:\d+|KEGG:[A-Za-z0-9]+)\s*", "", part).strip()
        if frag:
            return frag
    return ""


def pathway_fragments(hits: list[dict], limit: int = MAX_PATHWAY_FRAGMENTS) -> list[str]:
    """First `limit` distinct readable pathway fragments across hits, in order."""
    frags: list[str] = []
    for h in hits:
        f = shorten_pathway(str(h.get("pathway", "")))
        if f and f.lower() not in {x.lower() for x in frags}:
            frags.append(f)
        if len(frags) >= limit:
            break
    return frags


def relevance_word(rank_idx: int, raw_ev: object) -> str:
    """Human relevance, rank-relative so it means the same on any backend.

    Raw similarity scores stay in Technical details; the main UI never shows them.
    """
    if evidence_key(raw_ev) == "Inferred":
        return "Limited"
    if rank_idx < 3:
        return "High"
    return "Moderate"

def findings_pool_size(top_k: int) -> int:
    """How many ranked rows to fetch for the key-findings cards.

    Distinct genes need a broader net than a top-5 table; one cached call.
    """
    return max(int(top_k or 5), FINDINGS_POOL_MIN)


def apply_filters(rows: list[dict], stressors: list[str], organism: str,
                  tissues: list[str]) -> list[dict]:
    """Narrow ranked rows by sidebar selections. Pure subset — order preserved."""
    out = list(rows)
    if stressors:
        out = [r for r in out if any(s in (r.get("stressors", []) or []) for s in stressors)]
    if organism != "All":
        out = [r for r in out if r.get("organism", "") == organism]
    if tissues:
        out = [r for r in out if str(r.get("tissue", "—")) in tissues]
    return out


@dataclass
class GeneGroup:
    """One distinct gene across the findings pool, with everything cards need."""
    gene: str
    hits: list[dict] = field(default_factory=list)
    top: dict = field(default_factory=dict)
    dominant_direction: str = ""
    mixed_directions: bool = False
    n_agree: int = 0
    n_total: int = 0
    experiments: list[str] = field(default_factory=list)
    full_name: str = ""
    function: str = ""
    role: str = ""
    pathway: str = ""
    evidence_raw: str = "Inferred"

    @property
    def consistency(self) -> str:
        """'3 / 3 experiments' agreement among this gene's records."""
        if self.dominant_direction.strip().lower() not in ("up", "down"):
            return ""
        return f"{self.n_agree} / {self.n_total} experiments"


def safe_similarity(hit: dict) -> float:
    """Similarity as float, 0.0 when missing/unparseable. Engine hits always
    carry floats (see Hit); this guards hand-built or legacy rows."""
    try:
        return float(hit.get("similarity", 0) or 0)
    except Exception:
        return 0.0


def best_hit(hits: list[dict]) -> dict:
    """Single definition of 'best hit per gene': highest similarity.

    Callers rely on pool order being similarity-descending for ties; the max()
    itself is exact. Do not add secondary criteria here without updating
    group_genes, hypothesis_genes, and their tests together.
    """
    return max(hits, key=safe_similarity)


def group_genes(pool: list[dict], annotations: dict[str, tuple[str, str]] | None = None,
                limit: int = MAX_FINDING_GENES) -> list[GeneGroup]:
    """Group pool rows by gene symbol, best-first. No retrieval, no scoring changes."""
    ann = annotations or {}
    best: dict[str, float] = {}
    order: list[str] = []
    for h in pool:
        g = str(h.get("gene", "")).upper()
        if not g:
            continue
        s = safe_similarity(h)
        if g not in best or s > best[g]:
            best[g] = s
        if g not in order:
            order.append(g)
    groups: list[GeneGroup] = []
    for g in sorted(order, key=lambda x: -best.get(x, 0.0))[: max(1, limit)]:
        ghits = [h for h in pool if str(h.get("gene", "")).upper() == g]
        top = best_hit(ghits)
        dirs = [str(h.get("direction", "")) for h in ghits]
        # sorted(set) so direction ties break alphabetically, not by hash seed.
        dom = max(sorted(set(dirs)), key=dirs.count) if dirs else ""
        dir_set = {str(h.get("direction", "")).strip().lower() for h in ghits}
        mixed = len({d for d in dir_set if d in ("up", "down")}) > 1
        n_up = sum(1 for h in ghits if str(h.get("direction", "")).strip().lower() == "up")
        total = len(ghits)
        agree = n_up if dom.strip().lower() == "up" else (total - n_up)
        full_name, func = ann.get(g, ("", ""))
        role = str(top.get("earth_mechanism", "") or shorten_pathway(str(top.get("pathway", "")))
                   or "Spaceflight-responsive gene")
        if len(role) > ROLE_MAX_LEN:
            role = role[:ROLE_MAX_LEN - 3].rstrip() + "…"
        groups.append(GeneGroup(
            gene=g, hits=ghits, top=dict(top), dominant_direction=dom,
            mixed_directions=mixed, n_agree=agree, n_total=total,
            experiments=sorted({str(h.get("experiment", "?")) for h in ghits}),
            full_name=full_name, function=func, role=role,
            pathway=shorten_pathway(str(top.get("pathway", ""))),
            evidence_raw=evidence_key(str(top.get("evidence", "Inferred"))),
        ))
    return groups


@dataclass
class ExperimentGroup:
    """One NASA experiment across the filtered rows, with display-ready fields."""
    experiment_id: str
    hits: list[dict] = field(default_factory=list)
    tissues: list[str] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    stressors: list[str] = field(default_factory=list)
    gene_lines: list[str] = field(default_factory=list)
    best_evidence_raw: str = "Inferred"
    osdr_link: str = ""


def group_experiments(rows: list[dict]) -> list[ExperimentGroup]:
    """Group filtered rows by experiment, preserving first-seen order."""
    order: list[str] = []
    for h in rows:
        e = str(h.get("experiment", "?"))
        if e not in order:
            order.append(e)
    groups: list[ExperimentGroup] = []
    for e in order:
        ehits = [h for h in rows if str(h.get("experiment", "?")) == e]
        first = ehits[0] if ehits else {}
        by_gene = sorted(ehits, key=lambda h: str(h.get("gene", "")))
        groups.append(ExperimentGroup(
            experiment_id=e,
            hits=ehits,
            tissues=sorted({str(h.get("tissue", "—")) for h in ehits}),
            organisms=sorted({str(h.get("organism", "—")) for h in ehits}),
            stressors=sorted({
                str(s).replace("_", " ")
                for h in ehits
                for s in ((h.get("stressors", []) or [h.get("stressor", "—")]))
                if str(s).strip()
            }),
            gene_lines=[
                f"{h.get('gene', '?')} {arrow(str(h.get('direction', '')))} ({format_fc(h.get('log2FC', 0))})"
                for h in by_gene
            ],
            best_evidence_raw=min(
                (evidence_key(str(h.get("evidence", "Inferred"))) for h in ehits),
                key=lambda v: EV_ORDER.get(v, 9),
            ),
            osdr_link=str(first.get("osdr_link", "")),
        ))
    return groups


def collect_references(rows: list[dict], limit: int = MAX_REF_ROWS) -> list[tuple[str, str]]:
    """Deduplicated (pmid-or-'', context) citations across hits. No fabrication:
    entries without a numeric PMID render as plain text upstream."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for h in rows:
        cands: list[str] = [str(h.get("pubmed", "") or "").strip()]
        for key in ("pubmed_refs", "earth_pubmed"):
            cands += [str(x).strip() for x in (h.get(key, []) or [])]
        for c in cands:
            if not c or c.lower() in seen:
                continue
            seen.add(c.lower())
            m = re.search(r"(\d{7,9})", c)
            out.append((m.group(1) if m else "", c[:REF_CTX_MAX_LEN]))
            if len(out) >= limit:
                return out
    return out


def gene_pmids(hits: list[dict], limit: int = 6) -> list[str]:
    """Numeric PMIDs attached to one gene's hits, in order. Empty when none recorded."""
    out: list[str] = []
    for h in hits:
        cands = [str(h.get("pubmed", "") or "").strip()]
        cands += [str(x).strip() for x in ((h.get("pubmed_refs", []) or []) + (h.get("earth_pubmed", []) or []))]
        for c in cands:
            m = re.search(r"(\d{7,9})", c)
            if m and m.group(1) not in out:
                out.append(m.group(1))
            if len(out) >= limit:
                return out
    return out


def summarize_answer(pool: list[dict], table_hits: list[dict] | None = None) -> dict:
    """Evidence-derived short-answer parts. Assembles no claims beyond the rows.

    Gene/tissue/stressor coverage comes from the wider findings `pool`, while the
    disease label and pathway highlights come from the displayed table rows —
    exactly as the UI has always done (both orderings coincide, kept explicit).
    """
    table_hits = table_hits if table_hits is not None else pool
    genes_seen: list[str] = []
    for h in pool:
        g = str(h.get("gene", "")).upper()
        if g and g not in genes_seen:
            genes_seen.append(g)
    top_genes = genes_seen[:MAX_SHORT_GENES]
    gene_dir = {}
    for g in top_genes:
        hg = next((h for h in pool if str(h.get("gene", "")).upper() == g), {})
        gene_dir[g] = str(hg.get("direction", ""))
    exp_ids = sorted({str(r.get("experiment", "?")) for r in pool if r.get("experiment")})
    tissues = [t for t in dict.fromkeys(str(r.get("tissue", "—")) for r in pool)][:MAX_SHORT_TISSUES]
    stressors = [s for s in dict.fromkeys(
        str(s).strip().replace("_", " ") for r in pool for s in (r.get("stressors", []) or []) if str(s).strip()
    )][:MAX_SHORT_STRESSORS]
    first = table_hits[0] if table_hits else {}
    panel_label = str(first.get("earth_disease", first.get("disease", "this disease")) or "this disease")
    frags = pathway_fragments(table_hits, MAX_PATHWAY_FRAGMENTS)
    stress_phrase = "Spaceflight" if not stressors else " and ".join(s.capitalize() for s in stressors)
    gene_phrase = ", ".join(f"{g} {arrow(gene_dir.get(g, ''))}".strip() for g in top_genes)
    pw_phrase = " and ".join(f"*{f}*" for f in frags) if frags else "disease-relevant pathways"
    text = (
        f"{stress_phrase}-related experiments in the current dataset show measurable expression "
        f"changes in genes linked to {panel_label}. The strongest findings involve **{gene_phrase}** — "
        f"observed across {len(exp_ids)} NASA experiments ({', '.join(tissues)}). "
        f"These point to {pw_phrase} relevant to {panel_label} on Earth. "
        "Each finding and its evidence trail follow below."
    )
    return {
        "text": text,
        "gene_phrase": gene_phrase,
        "stress_phrase": stress_phrase,
        "pw_phrase": pw_phrase,
        "tissues": tissues,
        "stressors": stressors,
        "panel_label": panel_label,
        "exp_ids": exp_ids,
        "top_genes": top_genes,
        "gene_dir": gene_dir,
    }


def pick_trail_hits(pool: list[dict], gene_order: list[str],
                    limit: int = MAX_TRAIL_GENES) -> list[dict]:
    """Best hit per top gene for the evidence-trail cards (distinct genes, in order)."""
    out: list[dict] = []
    for g in (gene_order[: max(1, limit)] or []):
        hit = next((h for h in pool if str(h.get("gene", "")).upper() == g), None)
        if isinstance(hit, dict):
            out.append(hit)
    return out


def hypothesis_genes(pool: list[dict], gene_order: list[str]) -> list[tuple[str, str]]:
    """(gene, mechanism) pairs whose best hit is hypothesis-only, in findings order."""
    out: list[tuple[str, str]] = []
    for g in gene_order:
        gh = [h for h in pool if str(h.get("gene", "")).upper() == g]
        if not gh:
            continue
        gtop = best_hit(gh)
        if evidence_key(str(gtop.get("evidence", "Inferred"))) == "Inferred":
            mech = str(gtop.get("earth_mechanism", "") or "").strip()
            if len(mech) > GENE_LIT_MAX_LEN:
                mech = mech[:GENE_LIT_MAX_LEN - 3].rstrip() + "…"
            out.append((g, mech or "shared pathway"))
    return out


def pw_node_label(pathway: object) -> str:
    """Short graph-node label for a pathway string (truncated, never crashes)."""
    frag = shorten_pathway(pathway)
    return (frag[:42] + "…") if len(frag) > 42 else frag


def table_rows(rows: list[dict], research_view: bool = False) -> list[dict]:
    """Full-results table rows. Scores appear only in research view."""
    out = []
    for i, r in enumerate(rows):
        row = {
            "rank": r.get("rank"),
            "gene": r.get("gene"),
            "experiment": r.get("experiment"),
            "mission": r.get("mission"),
            "tissue": r.get("tissue"),
            "direction": r.get("direction"),
            "log2FC": round(float(r.get("log2FC", 0.0) or 0.0), 2),
            "relevance": relevance_word(i, str(r.get("evidence", "Inferred"))),
            "evidence": evidence_label(str(r.get("evidence", "Inferred"))),
            "OSDR link": r.get("osdr_link"),
        }
        if research_view:
            try:
                row["score"] = round(float(r.get("similarity", 0) or 0), 3)
            except Exception:
                row["score"] = ""
        out.append(row)
    return out


def csv_filename(disease: str) -> str:
    """Filesystem-safe export name for a disease query."""
    slug = re.sub(r"[^a-z0-9]+", "_", (disease or "query").lower()).strip("_") or "query"
    return f"spacemed_{slug}.csv"
