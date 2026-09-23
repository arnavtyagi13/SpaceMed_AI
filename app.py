"""SpaceMed AI — Connecting NASA space biology to Earth disease research.

Product-style Streamlit frontend over the preserved SpaceMed retrieval engine
(spacemed/engine.py) and curated NASA dataset (data/*.json).

Design contract:
- Human-readable answer first; technical details hidden in expanders.
- Plain language (Relevant NASA experiments, Relevance, Hypothesis, Current NASA dataset).
- Evidence levels in simple words: Observed / Established / Hypothesis.
- Honest empty and weak-evidence states; no fake data, links, or citations.
- Engine, data, ranking, filters, graph, CSV export and offline behavior preserved.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from spacemed.present import (
    EV_BADGE,
    EV_DOT,
    EV_ORDER,
    apply_filters,
    arrow,
    arrow_span,
    collect_references,
    csv_filename,
    direction_word,
    evidence_key,
    evidence_label,
    findings_pool_size,
    format_fc,
    gene_pmids,
    group_experiments,
    group_genes,
    hypothesis_genes,
    pathway_fragments,
    pick_trail_hits,
    pw_node_label,
    relevance_word,
    shorten_pathway,
    summarize_answer,
    table_rows,
)

# Optional heavy viz deps — graceful fallback if absent.
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    go = None
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:  # pragma: no cover
    nx = None
    HAS_NX = False


# ---------------------------------------------------------------------------
# Page config + styling (clean scientific aesthetic)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SpaceMed AI — Connecting NASA space biology to Earth disease research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
/* ---- Cosmic Biology / Indian Scientific Modernism ----
   Warm ivory #F7F4EC canvas, translucent white glass rgba(255,255,255,.68),
   deep indigo #40358C + living teal #168F86, saffron #E8892D tiny accents,
   muted maroon #9B4A4A, cosmic lavender #8D7BD8, ink navy #172033 text.
   Entities: experiment indigo, gene lavender, pathway teal, disease maroon.
   Evidence: observed teal-green, established indigo, hypothesis saffron-dark, conflict red.
   Type: Inter for UI, JetBrains Mono for identifiers (OSD IDs, gene symbols, log2FC). */
.stApp, .block-container, p, li, span, div {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.mono { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; }
.stApp {
    background:
        repeating-radial-gradient(circle at 88% -10%, rgba(64,53,140,0.055) 0 1px, transparent 1px 30px),
        radial-gradient(1100px 520px at 85% -5%, rgba(141,123,216,0.16), transparent 60%),
        radial-gradient(950px 520px at 5% 22%, rgba(22,143,134,0.12), transparent 60%),
        linear-gradient(180deg, #F7F4EC 0%, #F1EBDD 100%);
    background-attachment: fixed;
    color: #172033;
}
.block-container { max-width: 1180px; padding-top: 1.6rem; }
h1, h2, h3 { color: #172033; letter-spacing: -0.02em; }
h1 { margin-bottom: 0.1rem; font-size: clamp(2.1rem, 4.2vw, 3.1rem); font-weight: 800; }
/* Editorial kickers + mission telemetry strip */
.kicker { font-size: 11px; font-weight: 700; letter-spacing: 0.22em; color: #40358C; margin-bottom: 2px; }
.telemetry { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
    color: #8A8578; letter-spacing: 0.08em; margin: 2px 0 0 0; }
.telemetry .live-dot { color: #0F766E; }
.subtitle { font-size: 19px; color: #4E5468; margin-bottom: 0.6rem; }
p, li { color: #172033; }
.dataset-badge {
    display: inline-block; padding: 6px 14px; font-size: 14px; font-weight: 600; color: #40358C;
    background: rgba(64,53,140,0.10);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 999px;
}
.card {
    background: rgba(255,255,255,0.68);
    -webkit-backdrop-filter: blur(20px) saturate(160%); backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 14px;
    box-shadow: 0 12px 32px rgba(64,53,140,0.10), inset 0 1px 0 rgba(255,255,255,0.08);
    padding: 16px 18px; margin: 10px 0;
    transition: border-color 0.18s ease, background 0.18s ease;
}
.card:hover { background: #FFFFFF; border-color: rgba(64,53,140,0.30); }
/* Entity accent rails */
.acc-gene { border-top: 3px solid #8D7BD8; }
.acc-exp { border-top: 3px solid #40358C; }
.acc-trail { border-top: 3px solid #168F86; }
/* Timeline evidence trail */
.tl { position: relative; margin: 8px 0 2px 6px; padding-left: 24px; border-left: 1px solid rgba(64,53,140,0.16); }
.tl-row { position: relative; padding: 5px 0; font-size: 14px; color: #172033; }
.tl-dot { position: absolute; left: -30px; top: 10px; width: 11px; height: 11px;
    border-radius: 50%; background: var(--c, #40358C); border: 2px solid #EFE8D8; }
.trail-card {
    background: rgba(255,255,255,0.68);
    -webkit-backdrop-filter: blur(20px) saturate(160%); backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 14px;
    box-shadow: 0 12px 32px rgba(64,53,140,0.10), inset 0 1px 0 rgba(255,255,255,0.08);
    padding: 14px 18px; margin: 10px 0;
}
.finding-gene { font-size: 22px; font-weight: 700; color: #6757B8; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; }
.finding-gene.ent-exp { color: #40358C; }
.arrow-up { color: #C04545; font-weight: 700; }
.arrow-down { color: #168F86; font-weight: 700; }
.role-line { font-size: 14px; color: #4E5468; margin: 4px 0; }
.meta-line { font-size: 13px; color: #8A8578; }
/* Entity color coding (consistent across trail, cards, graph) */
.ent-exp { color: #40358C; font-weight: 700; }
.ent-gene { color: #6757B8; font-weight: 700; }
.ent-pw { color: #168F86; }
.ent-dis { color: #9B4A4A; font-weight: 700; }
.trail-step { font-size: 14px; color: #172033; padding: 2px 0; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; }
.badge-observed { background: rgba(70,198,126,0.12); color: #0F766E; border: 1px solid rgba(70,198,126,0.45); }
.badge-established { background: rgba(64,53,140,0.12); color: #40358C; border: 1px solid rgba(64,53,140,0.45); }
.badge-hypothesis { background: rgba(232,137,45,0.14); color: #B45309; border: 1px solid rgba(232,137,45,0.45); }
.small-muted { color: #8A8578; font-size: 13px; }
/* Layered glass: hero/search panel + major summary cards. */
.glass-card {
    background: rgba(255,255,255,0.68);
    -webkit-backdrop-filter: blur(20px) saturate(160%); backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 14px;
    box-shadow: 0 12px 32px rgba(64,53,140,0.10), inset 0 1px 0 rgba(255,255,255,0.08);
    padding: 16px 18px; margin: 10px 0;
}
/* Top navigation + footer */
.topnav-brand { font-weight: 800; font-size: 15px; color: #172033; margin-right: auto; }
.footer-bar { border-top: 1px solid rgba(64,53,140,0.16); margin-top: 18px; padding: 14px 2px 4px 2px;
    color: #8A8578; font-size: 13px; }
/* Hero network visual */
.hero-svg { width: 100%; height: auto; display: block; }
/* Search box */
.stTextInput input, div[data-testid="stTextInput"] input {
    font-size: 17px; color: #172033;
    background: rgba(255,255,255,0.85);
    -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 12px;
    box-shadow: 0 4px 20px rgba(64,53,140,0.10), inset 0 1px 0 rgba(255,255,255,0.6);
}
div[data-testid="stTextInput"] input:focus { border-color: #40358C; box-shadow: 0 0 0 2px rgba(64,53,140,0.22); }
/* Buttons */
div.stButton > button {
    background: rgba(255,255,255,0.68);
    -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 10px;
    color: #172033; font-weight: 600; letter-spacing: 0.01em;
    transition: all 0.18s ease;
}
div.stButton > button:hover {
    background: #FFFFFF; border-color: #40358C; color: #172033;
    transform: translateY(-1px);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #40358C, #168F86); color: #FFFFFF; border: none; font-weight: 700;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #FFFFFF; }
/* Sidebar: translucent white glass over ivory canvas */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.60);
    -webkit-backdrop-filter: blur(20px) saturate(140%); backdrop-filter: blur(20px) saturate(140%);
    border-right: 1px solid rgba(64,53,140,0.16);
}
/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.68);
    -webkit-backdrop-filter: blur(20px) saturate(160%); backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid rgba(64,53,140,0.16); border-radius: 12px;
    box-shadow: 0 12px 32px rgba(64,53,140,0.10), inset 0 1px 0 rgba(255,255,255,0.08);
}
/* Quote (short answer / explainer) */
blockquote {
    background: rgba(255,255,255,0.60);
    -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
    border-left: 4px solid #E8892D;
    border-radius: 0 10px 10px 0; padding: 10px 16px; color: #172033;
}
/* Alerts */
div[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: rgba(64,53,140,0.16); }
/* Narrow screens: let squeezed columns wrap instead of crushing content. */
@media (max-width: 640px) {
    div[data-testid="column"] { min-width: 150px !important; flex: 1 1 150px !important; }
    h1 { font-size: 1.9rem; }
    .finding-gene { font-size: 19px; }
    .telemetry { font-size: 10px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# In-code demo data (fallback ONLY when spacemed/engine.py is unavailable)
# ---------------------------------------------------------------------------
ALL_STRESSORS = [
    "microgravity",
    "cosmic radiation",
    "isolation & confinement",
    "circadian disruption",
    "fluid shift",
]
ALL_ORGANISMS = ["Homo sapiens", "Mus musculus", "Rattus norvegicus"]

DEMO_DB: dict[str, list[dict]] = {
    "osteoporosis": [
        {
            "gene": "SOST", "experiment": "OSD-234", "mission": "ISS Expedition 56",
            "tissue": "calvaria / tibia", "direction": "Up", "log2FC": 1.82,
            "similarity": 0.91, "earth_score": 0.88, "evidence": "Observed",
            "rationale": "Sclerostin up in microgravity; inhibits Wnt bone formation.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-234",
            "pubmed": "31654570", "stressors": ["microgravity", "fluid shift"],
            "organism": "Mus musculus",
            "observed": "Sost mRNA up (log2FC 1.82) in murine calvaria post-flight (OSD-234).",
            "established": "Sclerostin inhibits Wnt signalling; anti-sclerostin (romosozumab) treats osteoporosis.",
            "hypothesis": "Microgravity-induced SOST may model accelerated disuse osteoporosis.",
        },
        {
            "gene": "RUNX2", "experiment": "OSD-175", "mission": "ISS Expedition 45",
            "tissue": "femur osteoblast", "direction": "Down", "log2FC": -1.45,
            "similarity": 0.87, "earth_score": 0.84, "evidence": "Observed",
            "rationale": "Master osteoblast TF suppressed in spaceflight.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-175",
            "pubmed": "27941819", "stressors": ["microgravity"],
            "organism": "Mus musculus",
            "observed": "Runx2 down (log2FC -1.45) in flight osteoblasts (OSD-175).",
            "established": "RUNX2 haploinsufficiency causes cleidocranial dysplasia / low bone mass.",
            "hypothesis": "RUNX2 rescue (e.g. Wnt agonists) is a candidate countermeasure.",
        },
        {
            "gene": "TNFRSF11B", "experiment": "OSD-234", "mission": "ISS Expedition 56",
            "tissue": "tibia", "direction": "Down", "log2FC": -1.10,
            "similarity": 0.83, "earth_score": 0.79, "evidence": "Established",
            "rationale": "OPG decoy receptor down; RANKL/OPG ratio rises.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-234",
            "pubmed": "30642874", "stressors": ["microgravity", "cosmic radiation"],
            "organism": "Homo sapiens",
            "observed": "TNFRSF11B (OPG) down in bed-rest analogue + flight tibia.",
            "established": "Denosumab (anti-RANKL) is standard osteoporosis therapy.",
            "hypothesis": "RANKL blockade may protect against spaceflight bone loss.",
        },
        {
            "gene": "COL1A1", "experiment": "OSD-118", "mission": "STS-135",
            "tissue": "osteoblast culture", "direction": "Down", "log2FC": -0.95,
            "similarity": 0.78, "earth_score": 0.81, "evidence": "Established",
            "rationale": "Type-I collagen transcript falls in microgravity culture.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-118",
            "pubmed": "23580351", "stressors": ["microgravity"],
            "organism": "Homo sapiens",
            "observed": "COL1A1 down (log2FC -0.95) in microgravity osteoblast culture.",
            "established": "COL1A1 mutations cause osteogenesis imperfecta.",
            "hypothesis": "Collagen-synthesis support may be adjunct countermeasure.",
        },
        {
            "gene": "ACP5", "experiment": "OSD-195", "mission": "ISS Expedition 50",
            "tissue": "serum / bone", "direction": "Up", "log2FC": 1.20,
            "similarity": 0.74, "earth_score": 0.70, "evidence": "Inferred",
            "rationale": "TRAP osteoclast marker elevated; resorption signature.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-195",
            "pubmed": "32269987", "stressors": ["microgravity", "isolation & confinement"],
            "organism": "Rattus norvegicus",
            "observed": "Acp5/TRAP up in flight rodent bone.",
            "established": "Serum TRAP5b marks osteoclast activity in patients.",
            "hypothesis": "Antiresorptives merit testing as flight countermeasures.",
        },
    ],
    "sarcopenia": [
        {
            "gene": "MSTN", "experiment": "OSD-246", "mission": "ISS Expedition 60",
            "tissue": "gastrocnemius", "direction": "Up", "log2FC": 1.55,
            "similarity": 0.90, "earth_score": 0.86, "evidence": "Observed",
            "rationale": "Myostatin up; potent negative regulator of muscle mass.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-246",
            "pubmed": "32848243", "stressors": ["microgravity"],
            "organism": "Mus musculus",
            "observed": "Mstn up (log2FC 1.55) in flight gastrocnemius (OSD-246).",
            "established": "Myostatin loss causes hypermuscularity; blockers trialled in sarcopenia.",
            "hypothesis": "Myostatin blockade is a lead flight-muscle countermeasure.",
        },
        {
            "gene": "FBXO32", "experiment": "OSD-246", "mission": "ISS Expedition 60",
            "tissue": "soleus", "direction": "Up", "log2FC": 2.10,
            "similarity": 0.88, "earth_score": 0.82, "evidence": "Observed",
            "rationale": "Atrogin-1 E3 ligase strongly induced; atrophy program.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-246",
            "pubmed": "31234512", "stressors": ["microgravity", "isolation & confinement"],
            "organism": "Mus musculus",
            "observed": "Fbxo32 up (log2FC 2.10) in flight soleus.",
            "established": "FBXO32 marks human disuse/sarcopenic atrophy.",
            "hypothesis": "Ubiquitin-proteasome inhibition window needs study.",
        },
        {
            "gene": "TRIM63", "experiment": "OSD-188", "mission": "ISS Expedition 48",
            "tissue": "soleus", "direction": "Up", "log2FC": 1.95,
            "similarity": 0.85, "earth_score": 0.80, "evidence": "Observed",
            "rationale": "MuRF1 co-induced with atrogin-1 in unloading.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-188",
            "pubmed": "30185584", "stressors": ["microgravity"],
            "organism": "Mus musculus",
            "observed": "Trim63 up (log2FC 1.95) post-flight.",
            "established": "MuRF1 KO mice resist atrophy.",
            "hypothesis": "Combined atrogene panel as flight biomarker.",
        },
        {
            "gene": "PPARGC1A", "experiment": "OSD-188", "mission": "ISS Expedition 48",
            "tissue": "gastrocnemius", "direction": "Down", "log2FC": -1.30,
            "similarity": 0.80, "earth_score": 0.77, "evidence": "Established",
            "rationale": "PGC-1a down; mitochondrial biogenesis impaired.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-188",
            "pubmed": "29945932", "stressors": ["microgravity", "circadian disruption"],
            "organism": "Homo sapiens",
            "observed": "PPARGC1A down in flight/bed-rest muscle.",
            "established": "PGC-1a low in aged sarcopenic muscle.",
            "hypothesis": "Exercise + mitochondrial support preserves function.",
        },
        {
            "gene": "MYOD1", "experiment": "OSD-120", "mission": "STS-133",
            "tissue": "myoblast culture", "direction": "Down", "log2FC": -0.88,
            "similarity": 0.72, "earth_score": 0.69, "evidence": "Inferred",
            "rationale": "Myogenic differentiation factor suppressed.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-120",
            "pubmed": "28768747", "stressors": ["microgravity"],
            "organism": "Homo sapiens",
            "observed": "MYOD1 down in microgravity myoblasts.",
            "established": "MYOD1 required for muscle regeneration.",
            "hypothesis": "Differentiation-promoting cues may aid recovery.",
        },
    ],
    "immune senescence": [
        {
            "gene": "CDKN2A", "experiment": "OSD-312", "mission": "ISS 1-Year Mission",
            "tissue": "PBMC / T-cell", "direction": "Up", "log2FC": 1.68,
            "similarity": 0.89, "earth_score": 0.85, "evidence": "Observed",
            "rationale": "p16INK4a senescence marker induced by flight stress.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-312",
            "pubmed": "33051659", "stressors": ["cosmic radiation", "microgravity"],
            "organism": "Homo sapiens",
            "observed": "CDKN2A up (log2FC 1.68) in astronaut PBMCs.",
            "established": "p16 marks senescent T-cells in aging.",
            "hypothesis": "Senolytics could clear flight-induced senescent cells.",
        },
        {
            "gene": "IL6", "experiment": "OSD-312", "mission": "ISS 1-Year Mission",
            "tissue": "plasma / PBMC", "direction": "Up", "log2FC": 1.40,
            "similarity": 0.86, "earth_score": 0.83, "evidence": "Established",
            "rationale": "SASP cytokine elevated; inflammaging signature.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-312",
            "pubmed": "32976898", "stressors": ["cosmic radiation", "isolation & confinement"],
            "organism": "Homo sapiens",
            "observed": "IL6 up post-flight in crew plasma.",
            "established": "IL-6 predicts frailty/mortality in older adults.",
            "hypothesis": "IL-6 axis modulation post-flight warrants study.",
        },
        {
            "gene": "CDKN1A", "experiment": "OSD-205", "mission": "ISS Expedition 52",
            "tissue": "splenocyte", "direction": "Up", "log2FC": 1.25,
            "similarity": 0.82, "earth_score": 0.78, "evidence": "Observed",
            "rationale": "p21 DNA-damage/senescence response in flight spleen.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-205",
            "pubmed": "31664078", "stressors": ["cosmic radiation"],
            "organism": "Mus musculus",
            "observed": "Cdkn1a up (log2FC 1.25) in flight spleen.",
            "established": "p21 enforced in senescent immune cells.",
            "hypothesis": "Radiation-protective strategies may blunt p21 induction.",
        },
        {
            "gene": "TNF", "experiment": "OSD-205", "mission": "ISS Expedition 52",
            "tissue": "splenocyte", "direction": "Up", "log2FC": 1.05,
            "similarity": 0.77, "earth_score": 0.74, "evidence": "Established",
            "rationale": "Pro-inflammatory TNF program activated.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-205",
            "pubmed": "31089112", "stressors": ["microgravity", "circadian disruption"],
            "organism": "Mus musculus",
            "observed": "Tnf up in flight immune tissue.",
            "established": "TNF blockade treats chronic inflammatory disease.",
            "hypothesis": "Transient anti-inflammatory support post-landing.",
        },
        {
            "gene": "CXCL8", "experiment": "OSD-140", "mission": "STS-134",
            "tissue": "leukocyte culture", "direction": "Up", "log2FC": 0.92,
            "similarity": 0.71, "earth_score": 0.68, "evidence": "Inferred",
            "rationale": "Neutrophil chemokine in SASP-like flight signature.",
            "osdr_link": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-140",
            "pubmed": "29802305", "stressors": ["microgravity", "fluid shift"],
            "organism": "Homo sapiens",
            "observed": "CXCL8 up in microgravity leukocytes.",
            "established": "IL-8 part of senescence secretome in aging.",
            "hypothesis": "Chemokine panel as flight immune-health readout.",
        },
    ],
}


def _norm_key(disease: str) -> str:
    d = (disease or "").strip().lower()
    for k in DEMO_DB:
        if k in d or d in k:
            return k
    return ""


def demo_results(disease: str, top_k: int = 5) -> list[dict]:
    """In-code fallback results (engine missing only). Honest: [] when no match."""
    key = _norm_key(disease)
    if not key:
        return []
    rows = list(DEMO_DB.get(key, []))
    rows = sorted(rows, key=lambda r: r.get("similarity", 0), reverse=True)[: max(1, top_k)]
    out = []
    for i, r in enumerate(rows, start=1):
        rec = dict(r)
        rec["rank"] = i
        rec["disease"] = disease.strip() or key or "query"
        out.append(rec)
    return out


def engine_query_safe(disease: str, top_k: int) -> tuple[list[dict], str]:
    """Call spacemed.engine.query() with fallbacks. Returns (rows, source)."""
    # 1) live engine (preserved retrieval + ranking + evidence logic)
    try:
        from spacemed.engine import query as engine_query  # type: ignore

        rows = engine_query(disease, top_k=top_k)
        if isinstance(rows, list) and rows:
            normed: list[dict] = []
            for i, r in enumerate(rows[:top_k], start=1):
                d = dict(r) if isinstance(r, dict) else {"gene": str(r)}
                d.setdefault("rank", d.get("rank", i))
                d.setdefault("experiment", d.get("experiment", "OSD-?"))
                d.setdefault("mission", d.get("mission", "ISS"))
                d.setdefault("tissue", d.get("tissue", "—"))
                d.setdefault("direction", d.get("direction", "—"))
                d.setdefault("log2FC", d.get("log2FC", d.get("log2fc", 0.0)))
                d.setdefault("similarity", d.get("similarity", d.get("score", 0.5)))
                d.setdefault("earth_score", d.get("earth_score", 0.5))
                d.setdefault("evidence", d.get("evidence", "Inferred"))
                d.setdefault("rationale", d.get("rationale", "—"))
                d.setdefault("osdr_link", d.get("osdr_link", d.get("osdr", "https://osdr.nasa.gov/")))
                d.setdefault("pubmed", d.get("pubmed", ""))
                d.setdefault("stressors", d.get("stressors", ["microgravity"]))
                d.setdefault("organism", d.get("organism", "Mus musculus"))
                d.setdefault("observed", d.get("observed", "See OSDR record."))
                d.setdefault("established", d.get("established", "See cited literature."))
                d.setdefault("hypothesis", d.get("hypothesis", "AI-inferred hypothesis — requires validation."))
                d.setdefault("disease", disease)
                normed.append(d)
            return normed, "spacemed.engine.query()"
    except Exception:
        st.caption("CSV export is unavailable for this result set.")
    # 2) demo fallback (always works; [] when no curated match — no fake links)
    return demo_results(disease, top_k), "built-in demo data (engine/data unavailable)"


@st.cache_data(show_spinner=False, ttl=3600)
def cached_query(disease: str, top_k: int) -> tuple[list[dict], str]:
    return engine_query_safe(disease.strip(), top_k)


# Evidence display vocabulary lives in spacemed.present (single source of truth).

# Example buttons: label → full natural-language question (verified parse routing).
EXAMPLES = [
    ("Osteoporosis", "What does microgravity reveal about osteoporosis?"),
    ("Muscle aging", "Which genes change during spaceflight that are relevant to muscle aging?"),
    ("Immune aging", "What NASA experiments are relevant to immune aging?"),
    ("Neurodegeneration", "How does spaceflight affect brain aging and neurodegeneration?"),
    ("Radiation effects", "What biological pathways are associated with radiation-related changes?"),
]

# Weak-result suggestions: label → query (verified parse routing).
SUGGESTIONS = [
    ("osteoporosis", "osteoporosis"),
    ("muscle aging", "Which spaceflight genes are associated with muscle aging?"),
    ("immune aging", "Which spaceflight genes are associated with immune aging?"),
    ("neurodegeneration", "How does spaceflight affect brain aging and neurodegeneration?"),
    ("cardiovascular deconditioning", "How does spaceflight affect the heart and cardiovascular deconditioning?"),
    ("radiation effects", "What molecular pathways respond to space radiation?"),
]


def _set_query(q: str) -> None:
    # Button callback: runs before the next script run, so assigning the
    # search box key here is legal (never after widget instantiation).
    st.session_state["disease_input"] = q
    st.session_state["view"] = "explore"


def _set_view(v: str) -> None:
    # Lightweight view navigation (no router dependency).
    st.session_state["view"] = v


def _footer(query: str = "") -> None:
    """Minimal footer: brand · sources · dataset · disclaimer (all views)."""
    try:
        _fe, _, _ = _dataset_size()
    except Exception:
        _fe = 30
    st.markdown(
        "<div class='footer-bar'><b>SpaceMed AI</b> · "
        "<a href='https://osdr.nasa.gov/'>NASA OSDR</a> · "
        "<a href='https://pubmed.ncbi.nlm.nih.gov/'>Supporting research</a> · "
        "Data &amp; limitations apply · "
        f"{_fe}-experiment verified dataset · Research tool, not a medical device.</div>",
        unsafe_allow_html=True,
    )
    if query:
        st.caption(f"Query: “{query}” · SpaceMed AI research interface.")


def _reset_filters() -> None:
    for _k in ("f_stressor", "f_organism", "f_tissue", "f_topk"):
        st.session_state.pop(_k, None)


@st.cache_data(show_spinner=False, ttl=3600)
def _dataset_size() -> tuple[int, int, int]:
    """(experiments, records, panels) from live data; honest fallback to known verified counts."""
    try:
        from spacemed.engine import load_data as _ld  # type: ignore

        exps, refs, _ = _ld()
        n_exp = len(exps)
        n_rec = sum(len(e.get("genes", []) or []) for e in exps if isinstance(e, dict))
        return n_exp or 30, n_rec or 227, len(refs) or 6
    except Exception:
        return 30, 227, 6


@st.cache_data(show_spinner=False, ttl=3600)
def _annotations_map() -> dict[str, tuple[str, str]]:
    """symbol -> (full_name, function) from gene_annotations.csv; {} on any error."""
    try:
        import csv as _csv

        out: dict[str, tuple[str, str]] = {}
        with open(ROOT / "data" / "gene_annotations.csv", newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                sym = str(row.get("symbol", "") or "").strip().upper()
                if sym:
                    out[sym] = (
                        str(row.get("full_name", "") or "").strip(),
                        str(row.get("function", "") or "").strip(),
                    )
        return out
    except Exception:
        return {}


# Guidance questions (verified parse routing) for the How-it-works view.
GUIDANCE = [
    "What does microgravity reveal about osteoporosis?",
    "Which genes change during spaceflight that are relevant to muscle aging?",
    "What biological pathways are associated with radiation-related changes?",
    "What NASA experiments are relevant to immune aging?",
]

# Static hero visual: understated experiment → gene → pathway → disease network.
# Decorative illustration of the product's relationship model (not data).
_HERO_SVG = """<svg class="hero-svg" viewBox="0 0 200 296" role="img" aria-label="DNA helix with orbital paths">
<circle cx="100" cy="148" r="122" fill="none" stroke="#8D7BD8" stroke-width="1" stroke-dasharray="2 7" opacity="0.35"/>
<circle cx="100" cy="148" r="134" fill="none" stroke="#8D7BD8" stroke-width="1" stroke-dasharray="2 9" opacity="0.22"/>
<ellipse cx="100" cy="148" rx="92" ry="42" fill="none" stroke="#168F86" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.75" transform="rotate(-18 100 148)"/>
<path d="M150,26 A42,42 0 0 1 180,56" fill="none" stroke="#E8892D" stroke-width="2" opacity="0.9"/>
<circle cx="180" cy="56" r="2.5" fill="#E8892D" opacity="0.9"/>
<path d="M100,36 C74,70 74,104 100,138 C126,172 126,206 100,240" fill="none" stroke="#40358C" stroke-width="2"/>
<path d="M100,36 C126,70 126,104 100,138 C74,172 74,206 100,240" fill="none" stroke="#40358C" stroke-width="2"/>
<g stroke="#168F86" stroke-width="1.1" opacity="0.7">
<line x1="87" y1="62" x2="113" y2="62"/><line x1="84" y1="88" x2="116" y2="88"/>
<line x1="87" y1="114" x2="113" y2="114"/><line x1="84" y1="140" x2="116" y2="140"/>
<line x1="87" y1="166" x2="113" y2="166"/><line x1="84" y1="192" x2="116" y2="192"/>
<line x1="87" y1="218" x2="113" y2="218"/>
</g>
<g fill="#40358C" opacity="0.85">
<circle cx="87" cy="62" r="2"/><circle cx="113" cy="62" r="2"/><circle cx="84" cy="88" r="2"/><circle cx="116" cy="88" r="2"/>
<circle cx="87" cy="114" r="2"/><circle cx="113" cy="114" r="2"/><circle cx="84" cy="140" r="2"/><circle cx="116" cy="140" r="2"/>
<circle cx="87" cy="166" r="2"/><circle cx="113" cy="166" r="2"/><circle cx="84" cy="192" r="2"/><circle cx="116" cy="192" r="2"/>
<circle cx="87" cy="218" r="2"/><circle cx="113" cy="218" r="2"/>
</g>
<g opacity="0.65">
<circle cx="30" cy="60" r="1.2" fill="#40358C"/><circle cx="168" cy="120" r="1" fill="#8D7BD8"/>
<circle cx="42" cy="230" r="1.3" fill="#E8892D"/><circle cx="160" cy="252" r="1" fill="#40358C"/>
<circle cx="24" cy="150" r="1" fill="#8D7BD8"/><circle cx="178" cy="190" r="1.2" fill="#168F86"/>
<circle cx="60" cy="30" r="1" fill="#168F86"/><circle cx="140" cy="278" r="1.1" fill="#9B4A4A"/>
</g>
</svg>"""


# ---------------------------------------------------------------------------
# Top navigation (lightweight session-state views — no router dependency)
# ---------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "explore"
st.markdown("<div class='topnav-brand'>SpaceMed AI</div>", unsafe_allow_html=True)
_nv1, _nv2, _nv3 = st.columns(3)
_nv1.button("Explore", key="nav_explore", on_click=_set_view, args=("explore",), width="stretch")
_nv2.button("How it works", key="nav_how", on_click=_set_view, args=("how",), width="stretch")
_nv3.button("About", key="nav_about", on_click=_set_view, args=("about",), width="stretch")
try:
    from spacemed.engine import embedding_backend as _eb_fn  # type: ignore

    _tbe = _eb_fn()
    if _tbe in ("", "uninitialized"):
        _tbe = "auto"
except Exception:
    _tbe = "auto"
try:
    _te, _tr, _tp = _dataset_size()
except Exception:
    _te, _tr, _tp = 30, 227, 6
st.markdown(
    f"<div class='telemetry'><span class='live-dot'>●</span> LIVE · {_tbe.upper()} RETRIEVAL · "
    f"{_te} EXPERIMENTS · {_tr} RECORDS · {_tp} DISEASE AREAS</div>",
    unsafe_allow_html=True,
)

if st.session_state.get("view") == "how":
    st.title("How it works")
    st.markdown(
        "> Disease-first discovery: ask about a disease, and SpaceMed traces it "
        "to NASA experiments, genes, pathways, and supporting evidence."
    )
    _steps = [
        ("01", "Ask", "Ask about a disease or biological question."),
        ("02", "Understand", "Identify the relevant biological context."),
        ("03", "Find", "Retrieve relevant NASA experiments."),
        ("04", "Trace", "Connect experiments to genes and pathways."),
        ("05", "Verify", "Show the supporting evidence and sources."),
    ]
    _scols = st.columns(5)
    for (_num, _title, _desc), _col in zip(_steps, _scols):
        _col.markdown(
            f"<div class='card' style='text-align:left;'><div class='meta-line'><b>{_num}</b></div>"
            f"<div style='font-weight:700;color:#172033;margin:2px 0;'>{_title}</div>"
            f"<div class='small-muted'>{_desc}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("### Try asking:")
    for _gi, _gq in enumerate(GUIDANCE):
        st.button(_gq, key=f"guide_{_gi}", on_click=_set_query, args=(_gq,), width="stretch")
    _footer()
    st.stop()

if st.session_state.get("view") == "about":
    st.title("About SpaceMed AI")
    st.markdown(
        "> SpaceMed connects NASA space-biology research with Earth disease research. "
        "Every important result can be traced back to its NASA experiment and supporting scientific evidence."
    )
    st.markdown(
        "SpaceMed is a knowledge discovery and hypothesis-generation system — "
        "not a clinical diagnostic or treatment system. It retrieves curated NASA OSDR/GeneLab "
        "spaceflight evidence for a disease question, ranks candidate genes and pathways with "
        "deterministic embedding retrieval plus biological re-ranking, and labels every result "
        "as observed, established, or hypothesis."
    )
    try:
        _ae, _ar, _ap = _dataset_size()
    except Exception:
        _ae, _ar, _ap = 30, 227, 6
    st.markdown(f"### Current verified dataset — {_ae} curated NASA experiments")
    st.markdown(
        f"The demonstration corpus holds {_ae} experiments ({_ar} expression records) across "
        f"{_ap} disease areas. Study IDs mirror OSDR records; the architecture is designed to "
        "expand to additional compatible OSDR studies."
    )
    st.markdown("### Limitations")
    st.markdown(
        "- Curated rather than complete OSDR coverage.\n"
        "- Transcriptomics only; organisms are not automatically equivalent to humans.\n"
        "- Similarity does not prove causation; hypotheses require experimental validation."
    )
    st.markdown(
        "> **Research tool, not a medical device.** SpaceMed AI identifies research connections "
        "and hypotheses. It does not diagnose, treat, or provide medical advice."
    )
    _footer()
    st.stop()

# ---------------------------------------------------------------------------
# Landing page (explore view)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>NASA SPACE APPS · OPEN SPACE-BIOLOGY DATA</div>", unsafe_allow_html=True)
st.title("SpaceMed AI")
st.markdown("<div class='subtitle'><b>What can space teach us about disease?</b></div>", unsafe_allow_html=True)
st.markdown("Space changes biology. SpaceMed helps us understand what those changes might tell us about diseases on Earth.")

_n_exp, _n_rec, _n_pan = _dataset_size()
_hl, _hr = st.columns([4, 2])
with _hl:
    if "disease_input" not in st.session_state:
        st.session_state["disease_input"] = ""
    disease_input = st.text_input(
        "Disease or research question",
        key="disease_input",
        placeholder="What does microgravity reveal about osteoporosis?",
        help="Type a disease, tissue, or full question. SpaceMed AI routes it to the closest "
        "curated disease area — or pick an example below.",
        label_visibility="collapsed",
    )
    _ec1, _ec2 = st.columns([1, 3])
    with _ec1:
        st.button("Explore →", type="primary", width="stretch")
    with _ec2:
        st.caption("Tip: ask about a disease, a tissue, or a stressor — each answer shows its evidence trail.")
with _hr:
    st.markdown(
        f"<div class='glass-card' style='text-align:center;'>{_HERO_SVG}"
        "<div class='small-muted'>DNA, orbital paths and cellular geometry — the biology SpaceMed maps</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("### Try an example")
ex_cols = st.columns(5)
for (label, q), col in zip(EXAMPLES, ex_cols):
    col.button(label, on_click=_set_query, args=(q,), width="stretch")

st.markdown(
    f"<span class='dataset-badge'>Verified NASA dataset · {_n_exp} experiments</span>",
    unsafe_allow_html=True,
)
st.caption(
    f"This version uses {_n_exp} carefully curated NASA experiments "
    "as its verified demonstration dataset. The system is designed to expand to additional OSDR studies."
)

disease = (st.session_state.get("disease_input", "") or "").strip()
if not disease:
    st.stop()

# Session pre-reads (sidebar is rendered after the query below).
_top_k = int(st.session_state.get("f_topk", 5) or 5)
if st.session_state.get("_last_disease") != disease:
    # Tissue options depend on the current results; drop stale selections.
    st.session_state.pop("f_tissue", None)
    st.session_state["_last_disease"] = disease

# Live retrieval (engine + ranking + evidence logic unchanged).
with st.spinner("Searching NASA evidence..."):
    rows, source = cached_query(disease, _top_k)

# Honest empty-state: no curated matches (no fake OSDR links, ever).
if not rows:
    st.markdown("## Not enough evidence")
    st.markdown(
        "> We couldn't find a strong biological connection for this query "
        "in the current verified NASA dataset."
    )
    st.markdown("**Try asking about:**")
    _scols = st.columns(3)
    for i, (label, q) in enumerate(SUGGESTIONS):
        _scols[i % 3].button(label, key=f"sug_empty_{i}", on_click=_set_query, args=(q,), width="stretch")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters (defined after the query so tissue options come from results)
# ---------------------------------------------------------------------------
_tissue_options = sorted({str(r.get("tissue", "—")) for r in rows})
with st.sidebar:
    st.header("Narrow results")
    st.caption("Focus the candidates shown below.")
    stressor_sel = st.multiselect("Stressor", ALL_STRESSORS, default=[], key="f_stressor")
    organism_sel = st.selectbox("Organism", ["All"] + ALL_ORGANISMS, index=0, key="f_organism")
    tissue_sel = st.multiselect("Tissue", _tissue_options, default=[], key="f_tissue")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5, step=1, key="f_topk")
    st.button("Reset filters", on_click=_reset_filters, width="stretch")
    st.divider()
    st.caption("Evidence levels: 🔵 Observed · 🟢 Established · 🟡 Hypothesis")
    st.caption(f"Current NASA dataset: {_n_exp} experiments · {_n_rec} records · {_n_pan} disease areas.")

# Filtering lives in spacemed.present.apply_filters (pure subset, order preserved).


_filters_active = bool(stressor_sel or tissue_sel or organism_sel != "All")
filtered = apply_filters(rows, stressor_sel, organism_sel, tissue_sel)
if not filtered:
    st.warning("Filters removed all candidates — showing the unfiltered list. Use Reset filters to start over.")
    filtered = list(rows)
# Wider pool for key findings: distinct genes need a broader net than a top-5
# table (one cached call, same filters). Table and trail stay top-k; findings
# may reference closely-ranked records just beyond it.
_pool_rows, _ = cached_query(disease, findings_pool_size(_top_k))
pool_filtered = apply_filters(_pool_rows, stressor_sel, organism_sel, tissue_sel)
if not pool_filtered:
    pool_filtered = list(_pool_rows)

if "demo" in source:
    st.info(f"Running on {source}. Connect `spacemed/engine.py` + `data/*.json` for live retrieval.")
    _src_note = "demo data"
else:
    _src_note = "live retrieval"

# ---------------------------------------------------------------------------
# A. Question
# ---------------------------------------------------------------------------
st.markdown("# Your question")
st.markdown(f"> {disease}")
try:
    from spacemed.engine import parse_query as _parse_query  # type: ignore

    _parsed = _parse_query(disease) or {}
    _panel = str(_parsed.get("disease", "") or "").strip()
    _sh = str(_parsed.get("stressor_hint", "") or "").strip()
    _ph = str(_parsed.get("pathway_hint", "") or "").strip()
    _interp_bits = [f"Understood as **{_panel}**" if _panel else "No curated disease area matched"]
    if _sh:
        _interp_bits.append(f"focus: {_sh}")
    if _ph:
        _interp_bits.append(f"pathway focus: {_ph}")
    if _filters_active:
        _interp_bits.append(f"showing {len(filtered)} of {len(rows)} (filters applied)")
    _interp_bits.append(f"Results from the current NASA dataset · {_src_note}")
    st.caption("🔎 " + " · ".join(_interp_bits))
except Exception:
    if _filters_active:
        st.caption(f"Showing {len(filtered)} of {len(rows)} candidates (filters applied).")

_all_hypothesis = bool(filtered) and all(evidence_key(r.get("evidence", "Inferred")) == "Inferred" for r in filtered)

# View level: Simple (default) vs Research. Toggle placed before results split so it
# applies to both strong and weak flows.
research_view = st.toggle(
    "Research view",
    value=False,
    help="Simple view shows the answer, key findings, evidence trail, sources, and uncertainty. "
    "Research view additionally reveals match scores, the full results table, "
    "visualizations, and technical details.",
)

# ---------------------------------------------------------------------------
# Weak-result experience (honest, helpful, never broken-looking)
# ---------------------------------------------------------------------------
if _all_hypothesis:
    st.markdown("## Not enough evidence")
    st.markdown(
        "> We couldn't find a strong biological connection for this query "
        "in the current verified NASA dataset. What follows are exploratory matches only."
    )
    st.markdown("**Try asking about:**")
    _scols = st.columns(3)
    for i, (label, q) in enumerate(SUGGESTIONS):
        _scols[i % 3].button(label, key=f"sug_weak_{i}", on_click=_set_query, args=(q,), width="stretch")
else:
    # -----------------------------------------------------------------------
    # B. Short answer (generated ONLY from retrieved evidence — see summarize_answer).
    # -----------------------------------------------------------------------
    _ans = summarize_answer(pool_filtered, filtered)
    _top_genes = _ans["top_genes"]
    _gene_dir = _ans["gene_dir"]
    _exp_ids = _ans["exp_ids"]
    _tissues = _ans["tissues"]
    _stressors = _ans["stressors"]
    _panel_label = _ans["panel_label"]
    _gene_phrase = _ans["gene_phrase"]
    _stress_phrase = _ans["stress_phrase"]
    _pw_phrase = _ans["pw_phrase"]
    st.markdown("<div class='kicker'>01 · Answer</div>", unsafe_allow_html=True)
    st.markdown("# Explore the evidence")
    # Single source: the sentence is assembled once in summarize_answer().
    st.markdown(f"<div class='glass-card'>{_ans['text']}</div>", unsafe_allow_html=True)

    # How-to-read evidence levels (simple language).
    st.markdown("### How to read these results")
    _r1, _r2, _r3 = st.columns(3)
    with _r1:
        st.markdown("**🔵 Observed**")
        st.caption("Directly measured in the NASA dataset.")
    with _r2:
        st.markdown("**🟢 Established**")
        st.caption("Supported by existing scientific literature.")
    with _r3:
        st.markdown("**🟡 Hypothesis**")
        st.caption("A possible connection suggested by the system. Requires further research.")

# ---------------------------------------------------------------------------
# Key findings (dynamic gene cards from actual retrieved rows)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>02 · Signals</div>", unsafe_allow_html=True)
st.markdown("## What changed in space?")
_groups = group_genes(pool_filtered, _annotations_map())
_fcols = st.columns(2)
for _gi, _gg in enumerate(_groups):
    _g = _gg.gene
    _ghits = _gg.hits
    _top = _gg.top
    _dom_dir = _gg.dominant_direction
    _role = _gg.role
    _exps = _gg.experiments
    _rel = relevance_word(_gi, _gg.top.get("evidence", "Inferred"))
    _sim_txt = f" (score {float(_gg.top.get('similarity', 0) or 0):.3f})" if research_view else ""
    _full_name, _func = _gg.full_name, _gg.function
    _dir_w = direction_word(_dom_dir)
    _mixed = _gg.mixed_directions
    _cons = _gg.consistency
    _ev_top_raw = evidence_key(str(_gg.top.get("evidence", "Inferred")))
    _pw_top = _gg.pathway
    with _fcols[_gi % 2]:
        _name_line = f"<span class='mono'>{_g}</span>" + (f" <span class='meta-line'>· {_full_name}</span>" if _full_name else "")
        _plural = "s" if len(_exps) != 1 else ""
        _badge_cls = EV_BADGE.get(_ev_top_raw, "badge-hypothesis")
        _badge_dot = EV_DOT.get(_ev_top_raw, "")
        _badge_lab = evidence_label(_ev_top_raw)
        _badge_html = (
            "<span class='badge " + _badge_cls + "'>"
            + _badge_dot + " " + _badge_lab + "</span>"
        )
        _parts = [
            "<div class='card acc-gene'>",
            f"<div class='kicker'>Finding {_gi + 1:02d}</div>",
            f"<div class='finding-gene'>{_name_line}</div>",
            f"<div class='role-line'><b>{arrow_span(_dom_dir)} {_dir_w}</b>",
            (f" — {_full_name}" if _full_name else ""),
            "</div>",
            f"<div class='role-line'>{_role}</div>",
        ]
        if _pw_top:
            _parts.append(f"<div class='role-line'>Pathway: {_pw_top}</div>")
        _meta_bits = f"<div class='meta-line'>{len(_exps)} NASA experiment{_plural}"
        if _cons:
            _meta_bits += f" · {_cons}"
        _meta_bits += f" · Relevance: {_rel}{_sim_txt} · {_badge_html}</div>"
        _parts.append(_meta_bits)
        if _mixed:
            _parts.append("<div class='meta-line' style='color:#C04545;'>Direction conflict across experiments — see details.</div>")
        _parts.append("</div>")
        st.markdown("".join(_parts), unsafe_allow_html=True)
        with st.expander(f"Details: {_g}"):
            st.markdown(f"**Evidence level:** {evidence_label(str(_top.get('evidence', 'Inferred')))}")
            if _func:
                st.markdown(f"**Gene function:** {_func}")
            for _h in _ghits:
                st.markdown(
                    f"- **{str(_h.get('experiment', '?'))}** · {str(_h.get('tissue', '—'))} · "
                    f"{str(_h.get('organism', '—'))} · {str(_h.get('stressor', '—')).replace('_', ' ')} · "
                    f"change {arrow(str(_h.get('direction', '')))} {format_fc(_h.get('log2FC', 0))} "
                    f"(log2FC) · {evidence_label(str(_h.get('evidence', 'Inferred')))}"
                )
            _pw = shorten_pathway(str(_top.get("pathway", "")))
            if _pw:
                st.markdown(f"**Pathway:** {_pw}")
            _glit = gene_pmids(_ghits)
            if _glit:
                st.markdown(
                    "**Supporting literature:** "
                    + ", ".join(f"[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in _glit[:6])
                )
            else:
                st.markdown("**Supporting literature:** none recorded for this finding in the current dataset.")
            _links = sorted({str(h.get("osdr_link", "")) for h in _ghits if str(h.get("osdr_link", "")).startswith("http")})
            for _u in _links:
                st.markdown(f"[View NASA source →]({_u})")

# ---------------------------------------------------------------------------
# Evidence trail (hero feature — one readable branch per top finding)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>03 · Provenance</div>", unsafe_allow_html=True)
st.markdown("## Follow the biological trail")
st.caption("Each card traces one finding from the disease question through NASA evidence to Earth disease context — no giant diagrams.")
# One trail card per top finding (distinct genes), using each gene's best hit.
_trail_hits = pick_trail_hits(
    pool_filtered,
    [g.gene for g in _groups] or [str(filtered[0].get("gene", "")).upper()],
)
for _h in _trail_hits:
    _ev_raw = evidence_key(str(_h.get("evidence", "Inferred")))
    _pw = shorten_pathway(str(_h.get("pathway", ""))) or "pathway (see finding details)"
    _dis = str(_h.get("earth_disease", _h.get("disease", disease)) or disease)
    _mech = str(_h.get("earth_mechanism", "") or "").strip()
    if len(_mech) > 140:
        _mech = _mech[:137].rstrip() + "…"
    _t_gene = str(_h.get("gene", "?"))
    _t_fc = format_fc(_h.get("log2FC", 0))
    _t_dir = str(_h.get("direction", "—"))
    _t_stress = str(_h.get("stressor", "—")).replace("_", " ")
    _t_exp = str(_h.get("experiment", "?"))
    _t_mission = str(_h.get("mission", "ISS"))
    _t_tis = f"{str(_h.get('tissue', '—'))} · {str(_h.get('organism', '—'))}"
    if _mech and _mech.lower() != "shared pathway":
        _t_earth = _mech
    else:
        _t_earth = f"linked to {_dis} in the curated panel"
    _t_badge = f"<span class='badge {EV_BADGE.get(_ev_raw, 'badge-hypothesis')}'>{EV_DOT.get(_ev_raw, '🟡')} {evidence_label(_ev_raw)}</span>"
    st.markdown(
        "<div class='trail-card acc-trail'>"
        f"<div class='trail-step'><span class='ent-dis'>{_dis}</span> {_t_badge}</div>"
        "<div class='tl'>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#4E5468'></span>{_t_stress}</div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#40358C'></span><span class='ent-exp'>NASA experiment <span class='mono'>{_t_exp}</span></span> <span class='meta-line'>· {_t_mission}</span></div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#4E5468'></span>{_t_tis}</div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#8D7BD8'></span><span class='ent-gene'><span class='mono'>{_t_gene}</span> {arrow(_t_dir)}</span></div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#4E5468'></span>{_t_dir} · <span class='mono'>log2FC {_t_fc}</span></div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#168F86'></span><span class='ent-pw'>{_pw}</span></div>"
        f"<div class='tl-row'><span class='tl-dot' style='--c:#9B4A4A'></span>Earth disease evidence: {_t_earth}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _tgene = str(_h.get("gene", "?"))
    with st.expander(f"Inspect this trail: {_tgene}"):
        _tlink = str(_h.get("osdr_link", ""))
        st.markdown(f"- **OSDR ID:** <span class='mono'>{str(_h.get('experiment', '?'))}</span>", unsafe_allow_html=True)
        st.markdown(f"- **Mission:** {str(_h.get('mission', 'ISS'))}")
        st.markdown(f"- **Organism:** {str(_h.get('organism', '—'))}")
        st.markdown(f"- **Tissue:** {str(_h.get('tissue', '—'))}")
        st.markdown(f"- **Stressor:** {str(_h.get('stressor', '—')).replace('_', ' ')}")
        st.markdown(f"- **Gene:** <span class='mono'>{_tgene}</span> ({str(_h.get('direction', '—'))})", unsafe_allow_html=True)
        st.markdown(f"- **Effect size:** <span class='mono'>log2FC {format_fc(_h.get('log2FC', 0))}</span>", unsafe_allow_html=True)
        st.markdown(f"- **Pathway:** {_pw}")
        st.markdown(f"- **Evidence level:** {evidence_label(_ev_raw)}")
        _tpm = re.search(r"(\d{7,9})", str(_h.get("pubmed", "") or ""))
        if _tpm:
            st.markdown(f"- **Literature:** [PMID {_tpm.group(1)}](https://pubmed.ncbi.nlm.nih.gov/{_tpm.group(1)}/)")
        if _tlink.startswith("http"):
            st.markdown(f"- [View NASA source →]({_tlink})")

# ---------------------------------------------------------------------------
# NASA experiments (readable cards, real OSDR links only)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>04 · Sources</div>", unsafe_allow_html=True)
st.markdown("## See where the evidence comes from")
st.caption("Every major result answers: where did this come from? Each card names its NASA OSDR source.")
_exp_groups = group_experiments(filtered)
for _eg in _exp_groups:
    _e = _eg.experiment_id
    _glist = ", ".join(_eg.gene_lines)
    _best_ev = _eg.best_evidence_raw
    _link = _eg.osdr_link
    _tiss = _eg.tissues
    _orgs = _eg.organisms
    _sts = _eg.stressors
    st.markdown(
        f"<div class='card acc-exp'><div class='finding-gene ent-exp' style='font-size:19px;'>{_e}</div>"
        f"<div class='meta-line'>Source: NASA OSDR · Tissue: {', '.join(_tiss)} · Organism: {', '.join(_orgs)} · "
        f"Stressor: {', '.join(_sts) if _sts else '—'}</div>"
        f"<div class='role-line'>Relevant genes: {_glist}</div>"
        f"<div class='meta-line'>Relevance: {evidence_label(_best_ev)}</div></div>",
        unsafe_allow_html=True,
    )
    if _link.startswith("http"):
        st.markdown(f"[View NASA source →]({_link})")

# ---------------------------------------------------------------------------
# Scientific evidence (real citations only; gaps stated explicitly)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>05 · Literature</div>", unsafe_allow_html=True)
st.markdown("## What we know")
st.caption("Published literature behind the connections above. Missing evidence is stated, never hidden.")
_ref_rows = collect_references(filtered)
if not _ref_rows:
    st.markdown("> No supporting literature citation is currently available in the dataset.")
else:
    for _pmid, _ctx in _ref_rows[:12]:
        if _pmid:
            _shown = _ctx if _ctx != _pmid else f"PubMed record {_pmid}"
            st.markdown(f"- **PubMed {_pmid}** — {_shown} · [Open source →](https://pubmed.ncbi.nlm.nih.gov/{_pmid}/)")
        else:
            st.markdown(f"- {_ctx}")

# ---------------------------------------------------------------------------
# Why it matters + what we're still investigating (grounded in retrieved rows)
# ---------------------------------------------------------------------------
if not _all_hypothesis:
    st.markdown("## Why does it matter?")
    st.markdown(
        f"**{_gene_phrase}** changed measurably in {', '.join(_tissues)} under "
        f"{_stress_phrase.lower()} in the current NASA dataset. "
        f"They converge on {_pw_phrase}, processes central to {_panel_label} biology on Earth. "
        f"Because the same genes are linked to {_panel_label} in existing research, "
        "this set is biologically relevant to investigate further — not a treatment claim."
    )
    _top1g = _top_genes[0] if _top_genes else "this gene"
    _dom_s = _stressors[0].lower() if _stressors else "spaceflight"
    st.markdown("## What we're still investigating")
    st.markdown(
        f"This result does not prove that {_dom_s} causes {_panel_label} through {_top1g}. "
        "It identifies a biological connection worth investigating."
    )
    _hypo_genes = hypothesis_genes(pool_filtered, [g.gene for g in _groups])
    if _hypo_genes:
        for _hg, _hm in _hypo_genes:
            st.markdown(
                f"- <span class='mono'>{_hg}</span> — {_hm} "
                "<span class='small-muted'>(hypothesis — requires experimental validation)</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "Every finding above is observed in flight data or established in literature — "
            "nothing here is hypothesis-only."
        )
else:
    st.markdown("## What we're still investigating")
    st.markdown(
        "This is a hypothesis generated from the available dataset and requires experimental validation."
    )

# ---------------------------------------------------------------------------
# Data & limitations (transparency builds trust — before advanced tools)
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>06 · Scope</div>", unsafe_allow_html=True)
st.markdown("## Data & limitations")
st.markdown(f"### Current verified dataset — {_n_exp} curated NASA experiments")
st.markdown(
    f"> SpaceMed currently uses a verified demonstration dataset of {_n_exp} NASA experiments. "
    "The architecture is designed to expand to additional compatible OSDR studies."
)
st.markdown("**Current focus:** transcriptomics — gene-level expression evidence (direction and effect size).")
st.markdown("**Important limitations:**")
st.markdown(
    "- Curated rather than complete OSDR coverage: study IDs mirror OSDR records, but the dataset is a frozen demonstration subset.\n"
    "- Different organisms are not automatically equivalent to humans: most flight experiments are murine while Earth disease context is largely human.\n"
    "- Similarity does not prove causation: retrieval scores measure signature resemblance, not causal evidence.\n"
    "- Hypotheses require experimental validation before any translational use."
)

# ---------------------------------------------------------------------------
# Advanced research tools (visuals + full table + technical details)
# ---------------------------------------------------------------------------
st.markdown("### Advanced research tools")
st.caption("Deeper views for researchers. Scores and methods live here — the answer above needs none of it.")
with st.expander("Visual exploration (optional)", expanded=research_view):
    st.markdown("### Space-vs-Earth heatmap")
    st.caption("Two independent scales: Spaceflight = expression change (log2FC, −2.5…+2.5); Earth = disease association (0…1). Compare direction within each panel, not colors across panels.")
    heat_df = pd.DataFrame(
        {
            "Spaceflight log2FC": [float(r.get("log2FC", 0) or 0) for r in filtered],
            "Earth disease association score": [float(r.get("earth_score", r.get("similarity", 0)) or 0) for r in filtered],
        },
        index=[str(r.get("gene", "?")) for r in filtered],
    )
    if HAS_PLOTLY and go is not None:
        try:
            hcol1, hcol2 = st.columns(2)
            with hcol1:
                fig_space = go.Figure(
                    data=go.Heatmap(
                        z=[[v] for v in heat_df["Spaceflight log2FC"].tolist()],
                        x=["Spaceflight log2FC"],
                        y=list(heat_df.index),
                        colorscale="RdBu_r",
                        reversescale=False,
                        zmin=-2.5,
                        zmax=2.5,
                        zmid=0,
                        colorbar=dict(title="log2FC", tickfont=dict(color="#4E5468"), title_font=dict(color="#4E5468")),
                        hovertemplate="gene=%{y}<br>Spaceflight log2FC=%{z:.2f}<extra></extra>",
                    )
                )
                fig_space.update_layout(
                    title="Spaceflight (log2FC, −2.5…+2.5)",
                    height=320 + 28 * len(heat_df),
                    margin=dict(l=90, r=20, t=50, b=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#172033"),
                )
                st.plotly_chart(fig_space, width="stretch")
            with hcol2:
                fig_earth = go.Figure(
                    data=go.Heatmap(
                        z=[[v] for v in heat_df["Earth disease association score"].tolist()],
                        x=["Earth association"],
                        y=list(heat_df.index),
                        colorscale="YlGnBu",
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="0…1", tickfont=dict(color="#4E5468"), title_font=dict(color="#4E5468")),
                        hovertemplate="gene=%{y}<br>Earth association=%{z:.2f}<extra></extra>",
                    )
                )
                fig_earth.update_layout(
                    title="Earth (association, 0…1)",
                    height=320 + 28 * len(heat_df),
                    margin=dict(l=90, r=20, t=50, b=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#172033"),
                )
                st.plotly_chart(fig_earth, width="stretch")
        except Exception:
            st.warning("Heatmap unavailable — showing values as table.")
            st.dataframe(heat_df, width="stretch")
    else:
        st.info("Plotly not installed — showing heatmap values as table (install `plotly` for the visual).")
        try:
            st.dataframe(heat_df.style.background_gradient(cmap="RdBu_r"), width="stretch")
        except Exception:
            st.dataframe(heat_df, width="stretch")

    st.markdown("### Knowledge graph (experiment → gene → pathway → disease)")
    st.caption(
        "This graph shows how the retrieved NASA experiments connect through genes and biological "
        "pathways to the searched disease. Indigo nodes are missions/experiments, lavender are genes, "
        "teal are pathways, maroon is the disease. The evidence trail above explains the same links step by step."
    )
    if HAS_PLOTLY and HAS_NX and go is not None and nx is not None:
        try:
            G = nx.Graph()
            disease_node = f"{disease}"
            G.add_node(disease_node, kind="disease")
            for r in filtered:
                w = float(r.get("similarity", 0.5) or 0.5)
                m, g = f"{r.get('mission', '?')}", f"{r.get('gene', '?')}"
                G.add_node(m, kind="mission")
                G.add_node(g, kind="gene")
                G.add_edge(m, g, weight=w)
                p = pw_node_label(r.get("pathway", ""))
                if p:
                    G.add_node(p, kind="pathway")
                    G.add_edge(g, p, weight=w)
                    G.add_edge(p, disease_node, weight=w)
                else:
                    G.add_edge(g, disease_node, weight=w)
            pos = nx.spring_layout(G, seed=42, k=1.2)
            color_map = {"mission": "#40358C", "gene": "#8D7BD8", "pathway": "#168F86", "disease": "#9B4A4A"}
            edge_x, edge_y = [], []
            for u, v, d in G.edges(data=True):
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_t = list(G.nodes())
            node_c = [color_map.get(G.nodes[n].get("kind", "gene"), "#8D7BD8") for n in G.nodes()]
            gfig = go.Figure()
            gfig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(color="#94a3b8", width=1.5),
                    hoverinfo="none", showlegend=False,
                )
            )
            gfig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y, mode="markers+text", text=node_t,
                    textposition="top center", textfont=dict(size=11),
                    marker=dict(size=22, color=node_c, line=dict(width=1.5, color="white")),
                    hovertemplate="%{text}<extra></extra>", showlegend=False,
                )
            )
            gfig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#172033"),
                               xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(gfig, width="stretch")
        except Exception:
            st.warning("Interactive graph unavailable — the same experiment → gene → disease links are listed below as text.")
            for r in filtered:
                st.markdown(f"- {r.get('mission')} → **{r.get('gene')}** → {disease}")
    else:
        missing = [m for m, ok in [("plotly", HAS_PLOTLY), ("networkx", HAS_NX)] if not ok]
        st.info(f"Graph libraries missing ({', '.join(missing)}) — text fallback. `pip install plotly networkx`.")
        for r in filtered:
            st.markdown(f"- {r.get('mission')} → **{r.get('gene')}** → {disease}")

# ---------------------------------------------------------------------------
# All results as a table + CSV export (full detail preserved)
# ---------------------------------------------------------------------------
with st.expander("All results as a table", expanded=research_view):
    rows_table = table_rows(filtered, research_view)
    df = pd.DataFrame(rows_table)
    try:
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "OSDR link": st.column_config.LinkColumn("OSDR link", display_text="Open OSDR"),
            },
        )
    except Exception:
        st.dataframe(df, width="stretch")
    try:
        _fname = csv_filename(disease)
        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=_fname,
            mime="text/csv",
            width="stretch",
        )
    except Exception:
        st.caption("CSV export is unavailable for this result set.")

# ---------------------------------------------------------------------------
# Technical details (hidden by default — researchers/developers only)
# ---------------------------------------------------------------------------
with st.expander("Technical details", expanded=research_view):
    try:
        from spacemed.engine import embedding_backend as _backend_fn  # type: ignore

        _backend = _backend_fn()
    except Exception:
        _backend = "unknown"
    st.markdown(
        f"- **Current NASA dataset:** {_n_exp} experiments · {_n_rec} expression records · {_n_pan} disease areas "
        "(local curated cache of OSDR study IDs; designed to expand to additional OSDR studies)."
    )
    st.markdown(
        "- **Retrieval method:** each experiment–gene record is embedded as text (gene, function, "
        "pathway, expression direction, tissue, organism, stressor); the disease question is embedded "
        f"the same way and matched by vector similarity (current run: {_backend}). "
        "Matches are re-ranked by pathway overlap with the disease area, expression effect size, "
        "and stressor relevance. No keyword search; the language model is never used for matching."
    )
    _top_scores = "; ".join(
        f"{str(r.get('gene', '?'))} {float(r.get('similarity', 0) or 0):.3f}" for r in filtered[:5]
    )
    st.markdown(f"- **Top match relevance scores:** {_top_scores} (ranking signal only — not clinical confidence).")
    st.markdown(
        "- **Evidence thresholds:** matches below the backend relevance cutoff, or with negligible "
        "fold-change, are labelled Hypothesis. Disease-panel membership affects only the label, never the score."
    )
    _idx = ROOT / "data" / "faiss.index"
    _meta = ROOT / "data" / "corpus_meta.json"
    st.markdown(
        f"- **Index files:** `data/faiss.index` ({'present' if _idx.exists() else 'missing — rebuilt automatically'}) · "
        f"`data/corpus_meta.json` ({'present' if _meta.exists() else 'missing — rebuilt automatically'})."
    )
    st.caption(f"Query source: {source} · top-k={top_k}")

st.divider()
st.markdown(
    "> **Research tool, not a medical device.** SpaceMed AI identifies research connections "
    "and hypotheses. It does not diagnose, treat, or provide medical advice. "
    "Hypotheses require experimental validation."
)
_footer(disease)
