"""SpaceMed AI matching engine.

Reads sibling-agent data files when present, otherwise uses small in-code
fallback sample data so the engine never crashes (hackathon offline-safe).

Embedding: sentence-transformers (all-MiniLM-L6-v2) if installed AND model
loadable offline; otherwise sklearn TF-IDF fallback.
Retrieval: FAISS (IndexFlatIP) if installed; otherwise numpy cosine brute-force.
Index persistence: data/faiss.index (numpy array) + data/corpus_meta.json,
with no hard faiss dependency.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXP_PATH = DATA_DIR / "osdr_experiments.json"
DIS_PATH = DATA_DIR / "earth_disease_refs.json"
ANN_PATH = DATA_DIR / "gene_annotations.csv"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "corpus_meta.json"

EMBEDDING_BACKEND = "uninitialized"  # "sbert" | "tfidf"

# ---------------------------------------------------------------------------
# Retrieval configuration. Domain decisions live here — not inline.
# Changing these changes ranking/labeling behavior; see demo_queries.md.
# ---------------------------------------------------------------------------
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_ENV_OVERRIDE = "SPACEMED_BACKEND"  # =="tfidf" forces the offline path
# Biological re-rank: score = W_COSINE*cos + W_PATHWAY*overlap + W_EFFECT*effect
RERANK_W_COSINE = 0.6
RERANK_W_PATHWAY = 0.25
RERANK_W_EFFECT = 0.15
RERANK_STRESSOR_BONUS = 0.05
EFFECT_NORM_FC = 2.0  # |log2FC| at/above this counts as full effect
# Wide candidate pool keeps top-k a stable prefix across top_k values.
CANDIDATE_POOL_MULT = 5
CANDIDATE_POOL_MIN = 50
# Similarity below which a hit is hypothesis-only (backend-specific score scales).
INFERRED_SIM_SBERT = 0.65
INFERRED_SIM_TFIDF = 0.30
# Fold-change cutoffs for the observed/inferred boundary.
MIN_ABS_FC_OBSERVED = 0.5
MIN_ABS_FC_DETECT = 0.3
# Canonical OSDR study URL (single source of truth; _iter_gene_pairs backfills it).
OSDR_STUDY_URL = "https://osdr.nasa.gov/bio/repo/data/studies"


def osdr_study_url(experiment_id: str) -> str:
    """Canonical link for an OSDR study. Never invent IDs — callers pass through known ones."""
    return f"{OSDR_STUDY_URL}/{experiment_id}"


EvidenceLevel = Literal["observed", "established", "inferred"]
"""Machine-readable evidence tier. Never merge: observed = measured in flight,
established = in this disease's Earth literature panel, inferred = hypothesis-only."""


class ParsedQuery(TypedDict):
    disease: str
    stressor_hint: str
    pathway_hint: str


class Hit(TypedDict, total=False):
    """One ranked (experiment, gene) result. `total=False`: engine always sets
    the core keys below; frontend aliases are guaranteed present on query() output.

    Two intentional dualities, do not "simplify" them:
    - evidence_level (machine tier) vs evidence (display string) vs is_observed /
      is_established (independent flags). A weak in-panel hit is legitimately
      evidence_level="inferred" AND is_established=True: the gene IS in the
      literature, but THIS measurement is too weak to stand on.
    - similarity vs earth_score: identical values; earth_score is a legacy
      frontend alias kept for compat, not a separate Earth-side score.
    """
    gene: str
    experiment_id: str
    mission: str
    organism: str
    tissue: str
    stressor: str
    log2FC: float
    direction: str
    pathway: str
    earth_disease: str
    earth_mechanism: str
    similarity: float
    evidence_level: EvidenceLevel
    osdr_url: str
    rationale_template: str
    rationale: str
    experiment: str
    stressors: list[str]
    evidence: str
    osdr_link: str
    rank: int
    disease: str
    earth_score: float
    pubmed: str
    observed: str
    established: str
    hypothesis: str
    pubmed_refs: list[str]
    earth_pubmed: list[str]
    go_terms: str
    evidence_chain: dict[str, str]
    is_observed: bool
    is_established: bool

# ---------------------------------------------------------------------------
# Fallback sample data (used when data files are missing / unparsable)
# ---------------------------------------------------------------------------
FALLBACK_EXPERIMENTS = [
    {
        "experiment_id": "OSD-234", "mission": "ISS Expedition 56",
        "organism": "Mus musculus", "tissue": "calvaria / tibia",
        "stressor": "microgravity", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-234",
        "genes": [
            {"symbol": "SOST", "log2FC": 1.82, "direction": "Up"},
            {"symbol": "TNFRSF11B", "log2FC": -1.10, "direction": "Down"},
            {"symbol": "COL1A1", "log2FC": -0.95, "direction": "Down"},
        ],
    },
    {
        "experiment_id": "OSD-175", "mission": "ISS Expedition 45",
        "organism": "Mus musculus", "tissue": "femur osteoblast",
        "stressor": "microgravity", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-175",
        "genes": [{"symbol": "RUNX2", "log2FC": -1.45, "direction": "Down"}],
    },
    {
        "experiment_id": "OSD-246", "mission": "ISS Expedition 60",
        "organism": "Mus musculus", "tissue": "gastrocnemius / soleus",
        "stressor": "microgravity", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-246",
        "genes": [
            {"symbol": "MSTN", "log2FC": 1.55, "direction": "Up"},
            {"symbol": "FBXO32", "log2FC": 2.10, "direction": "Up"},
            {"symbol": "TRIM63", "log2FC": 1.95, "direction": "Up"},
        ],
    },
    {
        "experiment_id": "OSD-188", "mission": "ISS Expedition 48",
        "organism": "Homo sapiens", "tissue": "gastrocnemius",
        "stressor": "microgravity", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-188",
        "genes": [
            {"symbol": "PPARGC1A", "log2FC": -1.30, "direction": "Down"},
            {"symbol": "MYOD1", "log2FC": -0.88, "direction": "Down"},
        ],
    },
    {
        "experiment_id": "OSD-312", "mission": "ISS 1-Year Mission",
        "organism": "Homo sapiens", "tissue": "PBMC / T-cell",
        "stressor": "cosmic radiation", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-312",
        "genes": [
            {"symbol": "CDKN2A", "log2FC": 1.68, "direction": "Up"},
            {"symbol": "IL6", "log2FC": 1.40, "direction": "Up"},
        ],
    },
    {
        "experiment_id": "OSD-205", "mission": "ISS Expedition 52",
        "organism": "Mus musculus", "tissue": "splenocyte",
        "stressor": "cosmic radiation", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-205",
        "genes": [
            {"symbol": "CDKN1A", "log2FC": 1.25, "direction": "Up"},
            {"symbol": "TNF", "log2FC": 1.05, "direction": "Up"},
        ],
    },
    {
        "experiment_id": "OSD-195", "mission": "ISS Expedition 50",
        "organism": "Rattus norvegicus", "tissue": "serum / bone",
        "stressor": "microgravity", "osdr_url": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-195",
        "genes": [{"symbol": "ACP5", "log2FC": 1.20, "direction": "Up"}],
    },
]

FALLBACK_DISEASES = [
    {
        "disease": "osteoporosis", "synonyms": ["bone loss", "low bone mass"],
        "description": "Systemic bone loss with reduced bone mineral density and high fracture risk.",
        "genes": ["SOST", "RUNX2", "TNFRSF11B", "COL1A1", "ACP5"],
        "mechanisms": ["Wnt signaling inhibition", "RANKL/OPG imbalance", "osteoblast suppression", "osteoclast activation"],
        "pathways": ["Wnt signaling", "RANK/RANKL/OPG", "collagen formation"],
    },
    {
        "disease": "sarcopenia", "synonyms": ["muscle atrophy", "muscle wasting"],
        "description": "Age-related loss of muscle mass and function with atrophy gene program.",
        "genes": ["MSTN", "FBXO32", "TRIM63", "PPARGC1A", "MYOD1"],
        "mechanisms": ["myostatin signaling", "ubiquitin-proteasome atrophy", "mitochondrial decline"],
        "pathways": ["myostatin/activin", "ubiquitin-proteasome", "PGC-1a mitochondrial biogenesis"],
    },
    {
        "disease": "immune senescence", "synonyms": ["immunosenescence", "inflammaging"],
        "description": "Aging immune dysfunction with senescent T-cells and chronic inflammation.",
        "genes": ["CDKN2A", "IL6", "CDKN1A", "TNF", "CXCL8"],
        "mechanisms": ["cellular senescence", "SASP secretion", "DNA damage response"],
        "pathways": ["p16/p21 senescence", "IL-6/JAK-STAT", "NF-kB inflammation"],
    },
]

FALLBACK_ANNOTATIONS = {
    "SOST": {"full_name": "sclerostin", "function": "inhibits Wnt bone formation", "go": "ossification", "pathway": "Wnt signaling"},
    "RUNX2": {"full_name": "RUNX family transcription factor 2", "function": "master osteoblast differentiation factor", "go": "osteoblast differentiation", "pathway": "osteoblast differentiation"},
    "TNFRSF11B": {"full_name": "osteoprotegerin", "function": "decoy receptor for RANKL", "go": "bone resorption regulation", "pathway": "RANK/RANKL/OPG"},
    "COL1A1": {"full_name": "collagen type I alpha 1", "function": "major bone matrix collagen", "go": "collagen fibril organization", "pathway": "collagen formation"},
    "ACP5": {"full_name": "tartrate-resistant acid phosphatase", "function": "osteoclast activity marker", "go": "bone resorption", "pathway": "osteoclast differentiation"},
    "MSTN": {"full_name": "myostatin", "function": "negative regulator of muscle mass", "go": "muscle development", "pathway": "myostatin/activin"},
    "FBXO32": {"full_name": "atrogin-1", "function": "E3 ubiquitin ligase driving atrophy", "go": "proteolysis", "pathway": "ubiquitin-proteasome"},
    "TRIM63": {"full_name": "MuRF1", "function": "E3 ubiquitin ligase driving atrophy", "go": "proteolysis", "pathway": "ubiquitin-proteasome"},
    "PPARGC1A": {"full_name": "PGC-1alpha", "function": "mitochondrial biogenesis regulator", "go": "mitochondrion organization", "pathway": "PGC-1a mitochondrial biogenesis"},
    "MYOD1": {"full_name": "myogenic differentiation 1", "function": "myogenic differentiation factor", "go": "muscle differentiation", "pathway": "myogenesis"},
    "CDKN2A": {"full_name": "p16INK4a", "function": "senescence cell-cycle inhibitor", "go": "cellular senescence", "pathway": "p16/p21 senescence"},
    "IL6": {"full_name": "interleukin 6", "function": "SASP pro-inflammatory cytokine", "go": "inflammatory response", "pathway": "IL-6/JAK-STAT"},
    "CDKN1A": {"full_name": "p21", "function": "DNA-damage senescence effector", "go": "cell cycle arrest", "pathway": "p16/p21 senescence"},
    "TNF": {"full_name": "tumor necrosis factor", "function": "pro-inflammatory cytokine", "go": "inflammatory response", "pathway": "NF-kB inflammation"},
    "CXCL8": {"full_name": "interleukin 8", "function": "neutrophil chemokine", "go": "chemotaxis", "pathway": "chemokine signaling"},
}

# Module-level caches
_CORPUS: list[dict] | None = None
_EMB = None
_VECTORIZER = None
_SBERT_MODEL = None
_FAISS_INDEX = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _as_list(blob, keys=("experiments", "data", "items", "diseases", "refs")) -> list:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in keys:
            if isinstance(blob.get(k), list):
                return blob[k]
        # single-record dict
        return [blob]
    return []


def _load_experiments() -> list:
    try:
        if EXP_PATH.exists():
            blob = json.loads(EXP_PATH.read_text(encoding="utf-8"))
            rows = _as_list(blob)
            if rows and isinstance(rows[0], dict):
                return rows
    except Exception:
        pass
    return [dict(e) for e in FALLBACK_EXPERIMENTS]


def _load_diseases() -> list:
    try:
        if DIS_PATH.exists():
            blob = json.loads(DIS_PATH.read_text(encoding="utf-8"))
            rows = _as_list(blob)
            if rows:
                return rows
    except Exception:
        pass
    return [dict(d) for d in FALLBACK_DISEASES]


def _load_annotations() -> dict:
    ann: dict = {}
    try:
        if ANN_PATH.exists():
            with open(ANN_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sym = (row.get("symbol") or row.get("SYMBOL") or row.get("gene") or "").strip().upper()
                    if sym:
                        ann[sym] = {
                            "full_name": row.get("full_name", row.get("name", "")),
                            "function": row.get("function", row.get("description", row.get("uniprot_summary", ""))),
                            "go": row.get("go", row.get("go_term", row.get("go_terms", ""))),
                            "pathway": row.get("pathway", row.get("pathways", "")),
                        }
            if ann:
                return ann
    except Exception:
        pass
    return {k: dict(v) for k, v in FALLBACK_ANNOTATIONS.items()}


def load_data():
    """Return (experiments, disease_refs, annotations). Never raises on missing files."""
    return _load_experiments(), _load_diseases(), _load_annotations()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def _iter_gene_pairs(experiments: list) -> list[dict]:
    """Normalize heterogeneous experiment schemas into (experiment, gene) pairs."""
    pairs = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        exp_id = str(exp.get("experiment_id") or exp.get("osdr_id") or exp.get("osd_id") or exp.get("accession")
                      or exp.get("id") or exp.get("experiment") or "OSD-?")
        mission = str(exp.get("mission", "ISS"))
        organism = str(exp.get("organism", exp.get("species", "Mus musculus")))
        tissue = str(exp.get("tissue", exp.get("tissue_type", "-")))
        stressor = exp.get("stressor", exp.get("stressors", "microgravity"))
        if isinstance(stressor, list):
            stressor = ", ".join(str(s) for s in stressor)
        stressor = str(stressor or "microgravity")
        osdr_url = str(exp.get("osdr_url") or exp.get("osdr_link", "")
                       or osdr_study_url(exp_id))
        _pm = exp.get("pubmed_refs", exp.get("pubmed", exp.get("citations", []))) or []
        if isinstance(_pm, str):
            _pm = [_pm]
        pubmed_refs = [str(x) for x in _pm] if isinstance(_pm, list) else [str(_pm)]
        exp_pathways = exp.get("pathways", []) or []
        exp_pathway_str = "; ".join(str(p) for p in exp_pathways) if isinstance(exp_pathways, list) else str(exp_pathways)
        gene_rows = None
        for k in ("genes", "gene_set", "degs", "deg", "differential_genes",
                  "differential_expression", "results"):
            if isinstance(exp.get(k), list):
                gene_rows = exp[k]
                break
        if gene_rows is None:
            # Experiment dict may itself be a single DEG row
            if exp.get("symbol") or exp.get("gene") or exp.get("SYMBOL"):
                gene_rows = [exp]
            else:
                continue
        for g in gene_rows:
            if not isinstance(g, dict):
                g = {"symbol": str(g)}
            symbol = str(g.get("symbol") or g.get("gene") or g.get("SYMBOL") or "").strip().upper()
            if not symbol:
                continue
            try:
                log2fc = float(g.get("log2FC", g.get("log2fc", g.get("log2_fold_change", 0.0))))
            except (TypeError, ValueError):
                log2fc = 0.0
            direction = str(g.get("direction", "") or ("Up" if log2fc >= 0 else "Down"))
            pathway = str(g.get("pathway", g.get("pathways", "") or "") or exp_pathway_str)
            pairs.append({
                "symbol": symbol, "log2FC": log2fc, "direction": direction,
                "pathway": pathway, "experiment_id": exp_id, "mission": mission,
                "organism": organism, "tissue": tissue, "stressor": stressor,
                "osdr_url": osdr_url, "pubmed_refs": list(pubmed_refs),
            })
    return pairs


def build_corpus(experiments=None, annotations=None) -> list[dict]:
    """Build one doc per (experiment, gene) pair. Returns list of doc dicts."""
    global _CORPUS
    if experiments is None or annotations is None:
        exps, _, anns = load_data()
        if experiments is None:
            exps_in = exps
        else:
            exps_in = experiments
        if annotations is None:
            anns_in = anns
        else:
            anns_in = annotations
    else:
        exps_in, anns_in = experiments, annotations
    docs = []
    for p in _iter_gene_pairs(exps_in):
        a = anns_in.get(p["symbol"], {}) if isinstance(anns_in, dict) else {}
        full_name = a.get("full_name", "")
        function = a.get("function", "")
        go = a.get("go", "")
        pathway = p["pathway"] or a.get("pathway", "")
        fc = p["log2FC"]
        text = (
            f"{p['symbol']} {full_name} {function} {go} {pathway} "
            f"{p['direction']}regulated in {p['tissue']} {p['organism']} "
            f"under {p['stressor']} aboard {p['mission']} fold change {fc}"
        )
        docs.append({**p, "pathway": pathway, "go_terms": go,
                     "pubmed_refs": list(p.get("pubmed_refs", []) or []),
                     "text": " ".join(text.split())})
    _CORPUS = docs
    return docs


def get_corpus() -> list[dict]:
    if _CORPUS is None:
        return build_corpus()
    return _CORPUS


# ---------------------------------------------------------------------------
# Embeddings (sbert preferred, sklearn TF-IDF offline fallback)
# ---------------------------------------------------------------------------
def _try_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL
    # Determinism override for demos/judges: SPACEMED_BACKEND=tfidf forces the
    # offline TF-IDF path even when sentence-transformers is installed.
    if os.environ.get(SBERT_ENV_OVERRIDE, "auto").strip().lower() == "tfidf":
        _SBERT_MODEL = None
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        return _SBERT_MODEL
    except Exception as exc:
        log.debug("sBERT unavailable (%s); using TF-IDF fallback", exc)
        _SBERT_MODEL = None
        return None


def embedding_backend() -> str:
    """Current embedding backend ('sbert', 'tfidf', or 'uninitialized')."""
    return EMBEDDING_BACKEND


def inferred_threshold() -> float:
    """Similarity below which a hit is labelled inferred/hypothesis-only.

    Backend-aware: TF-IDF cosine lives ~0.2 (noise) vs ~0.5+ (signal);
    sBERT rescaled cosine lives ~0.5 (noise) vs ~0.8+ (signal).
    """
    return INFERRED_SIM_SBERT if EMBEDDING_BACKEND == "sbert" else INFERRED_SIM_TFIDF


def get_embedding(texts: list[str]):
    """Embed texts -> L2-normalized numpy array. Sets EMBEDDING_BACKEND.

    Tries sentence-transformers first; falls back to sklearn TF-IDF fitted on
    the corpus (works fully offline) or on the input texts if corpus is empty.
    """
    global EMBEDDING_BACKEND, _VECTORIZER
    import numpy as np

    model = _try_sbert()
    if model is not None:
        try:
            emb = model.encode(list(texts), normalize_embeddings=True,
                               show_progress_bar=False)
            EMBEDDING_BACKEND = "sbert"
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            log.debug("sBERT encode failed (%s); falling back to TF-IDF", exc)
            pass  # fall through to TF-IDF
    # TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if _VECTORIZER is None:
        corpus_texts = [d["text"] for d in get_corpus()] or list(texts)
        _VECTORIZER = TfidfVectorizer().fit(corpus_texts)
    mat = _VECTORIZER.transform(list(texts)).toarray().astype(np.float32)
    if mat.shape[1] == 0:
        mat = np.zeros((len(texts), 1), dtype=np.float32)
    # normalize (rows with zero norm stay zero)
    mat = normalize(mat, norm="l2", axis=1)
    mat = np.nan_to_num(mat, nan=0.0).astype(np.float32)
    EMBEDDING_BACKEND = "tfidf"
    return mat


# ---------------------------------------------------------------------------
# Index (FAISS if available, else numpy brute-force) + persistence
# ---------------------------------------------------------------------------
def build_index():
    """Build embeddings for corpus; use FAISS if installed else brute-force."""
    global _EMB, _FAISS_INDEX
    import numpy as np

    docs = get_corpus()
    _EMB = get_embedding([d["text"] for d in docs])
    _FAISS_INDEX = None
    try:
        import faiss  # type: ignore
        idx = faiss.IndexFlatIP(_EMB.shape[1])
        idx.add(np.ascontiguousarray(_EMB.astype(np.float32)))
        _FAISS_INDEX = idx
    except Exception as exc:
        log.debug("FAISS unavailable (%s); using NumPy brute-force search", exc)
        _FAISS_INDEX = None
    return _FAISS_INDEX if _FAISS_INDEX is not None else _EMB


def _cosine_search(q: object, top_k: int):
    import numpy as np

    docs = get_corpus()
    if _EMB is None:
        build_index()
    assert _EMB is not None
    qv = np.asarray(q, dtype=np.float32).reshape(1, -1)
    if qv.shape[1] != _EMB.shape[1]:
        # Backend switch (e.g. stale TF-IDF index + sBERT query):
        # rebuild so corpus and query share one embedding space.
        build_index()
        assert _EMB is not None
        if qv.shape[1] != _EMB.shape[1]:
            raise ValueError(
                "embedding dim mismatch (query vs index); "
                "delete data/faiss.index + data/corpus_meta.json and rebuild"
            )
    sims = (_EMB @ qv[0])
    # embeddings are L2-normalized; TF-IDF sims in [0,1], sbert in [-1,1] -> rescale
    if EMBEDDING_BACKEND == "sbert":
        sims = (sims + 1.0) / 2.0
    sims = np.clip(sims, 0.0, 1.0)
    order = np.argsort(-sims)[: max(1, top_k)]
    if _FAISS_INDEX is not None:
        try:  # prefer FAISS scores when available
            import numpy as _np
            D, I = _FAISS_INDEX.search(_np.ascontiguousarray(qv.astype(_np.float32)), max(1, top_k))
            fsims = _np.clip((D[0] + 1.0) / 2.0 if EMBEDDING_BACKEND == "sbert" else D[0], 0.0, 1.0)
            return [(int(i), float(s)) for i, s in zip(I[0], fsims) if int(i) < len(docs)]
        except Exception as exc:
            log.debug("FAISS search failed (%s); using precomputed NumPy scores", exc)
    return [(int(i), float(sims[i])) for i in order]


def save_index(index_path: str | Path = INDEX_PATH, meta_path: str | Path = META_PATH):
    """Persist embeddings (numpy, exact filename) + corpus meta JSON."""
    import numpy as np

    docs = get_corpus()
    if _EMB is None:
        build_index()
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        np.save(f, np.asarray(_EMB, dtype=np.float32))
    meta = {"backend": EMBEDDING_BACKEND, "docs": docs}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(index_path), str(meta_path)


def load_index(index_path: str | Path = INDEX_PATH, meta_path: str | Path = META_PATH) -> bool:
    """Load persisted index if present. Returns True on success."""
    global _CORPUS, _EMB, _FAISS_INDEX, EMBEDDING_BACKEND
    import numpy as np

    try:
        if not Path(index_path).exists() or not Path(meta_path).exists():
            log.debug("no persisted index at %s; building fresh", index_path)
            return False
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        _CORPUS = meta.get("docs", [])
        EMBEDDING_BACKEND = meta.get("backend", "tfidf")
        # Try FAISS native format first, then numpy
        loaded = None
        try:
            import faiss  # type: ignore
            loaded = faiss.read_index(str(index_path))
            _FAISS_INDEX = loaded
            _EMB = None
            # need dense matrix too for fallback; try numpy read (fails for faiss fmt)
            try:
                _EMB = np.load(str(index_path), allow_pickle=True).astype(np.float32)
            except Exception:
                _EMB = get_embedding([d["text"] for d in _CORPUS])
            return True
        except Exception:
            pass
        _EMB = np.load(str(index_path), allow_pickle=True).astype(np.float32)
        _FAISS_INDEX = None
        try:
            import faiss  # type: ignore
            idx = faiss.IndexFlatIP(_EMB.shape[1])
            idx.add(np.ascontiguousarray(_EMB.astype(np.float32)))
            _FAISS_INDEX = idx
        except Exception:
            _FAISS_INDEX = None
        return True
    except Exception as exc:
        log.warning("persisted index unusable (%s); will rebuild in memory", exc)
        return False


# ---------------------------------------------------------------------------
# Disease query text + evidence classification + rationale
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _find_disease(disease_name: str, disease_refs: list) -> dict | None:
    q = _norm(disease_name)
    best = None
    for d in disease_refs:
        if not isinstance(d, dict):
            continue
        name = _norm(str(d.get("disease") or d.get("name") or ""))
        syns = [ _norm(s) for s in (d.get("synonyms") or []) if isinstance(s, str)]
        if q and (q in name or name in q or any(q in s or s in q for s in syns if s)):
            return d
        if name == q:
            best = d
    return best


def _gene_symbol(g) -> str:
    if isinstance(g, dict):
        return str(g.get("symbol") or g.get("gene") or g.get("SYMBOL") or "").strip().upper()
    return str(g or "").strip().upper()


def _disease_query_text(disease_name: str, ref: dict | None) -> str:
    if not ref:
        return f"{disease_name} disease gene set"
    gene_syms = [_gene_symbol(g) for g in (ref.get("genes", []) or [])]
    gene_syms = [s for s in gene_syms if s]
    gene_mechs = [str(g.get("mechanism", "")) for g in (ref.get("genes", []) or [])
                  if isinstance(g, dict) and g.get("mechanism")]
    parts = [
        str(ref.get("disease") or ref.get("name") or disease_name),
        " ".join(ref.get("synonyms", []) or []),
        str(ref.get("description", "")),
        " ".join(gene_syms),
        " ".join(gene_mechs),
        " ".join(ref.get("mechanisms", []) or []),
        " ".join(ref.get("pathways", []) or []),
    ]
    return " ".join(p for p in parts if p).strip() or disease_name


def _ref_mechanisms(ref: dict) -> list[str]:
    mechs = list(ref.get("mechanisms", []) or [])
    for g in (ref.get("genes", []) or []):
        if isinstance(g, dict) and g.get("mechanism"):
            mechs.append(str(g["mechanism"]))
    return mechs


def _gene_mechanism(ref: dict | None, symbol: str, default: str = "shared pathway") -> str:
    if ref:
        for g in (ref.get("genes", []) or []):
            if isinstance(g, dict) and _gene_symbol(g) == symbol and g.get("mechanism"):
                return str(g["mechanism"])
    return default


def generate_rationale(hit: Hit) -> str:
    """Cautious template rationale (no LLM call). Prefixed 'Evidence summary:'."""
    if str(hit.get("evidence_level", "")).lower() == "inferred":
        closing = ("limited retrieval similarity - hypothesis only, "
                   "insufficient evidence for a biological connection. "
                   "Requires experimental validation.")
    else:
        closing = ("supported by available evidence. "
                   "Candidate for further investigation. Hypothesis - requires experimental validation.")
    return (
        "Evidence summary: {gene} was {direction}regulated (log2FC {log2fc}) in {tissue} "
        "({organism}, {mission}, {stressor}); potential biological connection to {disease} via {mechanism} "
        "[{pathway}] (similarity {sim:.2f}, {evidence}), {closing}".format(
            gene=hit.get("gene", "?"),
            direction=str(hit.get("direction", "—")),
            log2fc=hit.get("log2FC", 0.0),
            tissue=hit.get("tissue", "—"),
            organism=hit.get("organism", "—"),
            mission=hit.get("mission", "ISS"),
            stressor=hit.get("stressor", "microgravity"),
            disease=hit.get("earth_disease", hit.get("disease", "query")),
            mechanism=hit.get("earth_mechanism", "shared pathway"),
            pathway=hit.get("pathway", "—"),
            sim=float(hit.get("similarity", 0.0)),
            evidence=hit.get("evidence_level", "inferred"),
            closing=closing,
        )
    )


def llm_rationale(hit: dict, disease: str = "") -> str:
    """Optional LLM hook: returns LLM text if API key present, else template.

    Supports OpenAI (OPENAI_API_KEY) and Anthropic (ANTHROPIC_API_KEY).
    Offline / no key -> generate_rationale(hit). Never raises.
    """
    template = generate_rationale(hit)
    try:
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI()
                r = client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": f"One-line rationale for {hit.get('gene')} in {disease or hit.get('earth_disease','')}: {template}"}],
                    max_tokens=120,
                )
                return r.choices[0].message.content.strip()
            except Exception:
                return template
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # type: ignore
                client = anthropic.Anthropic()
                r = client.messages.create(
                    model=os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                    max_tokens=120,
                    messages=[{"role": "user", "content": f"One-line rationale for {hit.get('gene')}: {template}"}],
                )
                return r.content[0].text.strip()
            except Exception:
                return template
    except Exception:
        pass
    return template


def _tok_set(s: str) -> set[str]:
    import re
    return set(t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2)


def _ref_token_set(ref: dict | None) -> set[str]:
    toks: set[str] = set()
    if not ref:
        return toks
    for p in (ref.get("pathways", []) or []):
        toks |= _tok_set(str(p))
    for m in _ref_mechanisms(ref):
        toks |= _tok_set(str(m))
    return toks


def _pathway_overlap_bonus(doc_pathway: str, doc_go: str, ref: dict | None) -> float:
    if not ref:
        return 0.0
    ref_toks = _ref_token_set(ref)
    if not ref_toks:
        return 0.0
    doc_toks = _tok_set(str(doc_pathway or "")) | _tok_set(str(doc_go or ""))
    return 1.0 if (doc_toks & ref_toks) else 0.0


def _extract_first_pmid(exp_refs: list, earth_refs: list) -> str:
    import re
    for lst in (exp_refs, earth_refs):
        for item in lst or []:
            m = re.search(r"PMID\s*:?\s*(\d{4,9})", str(item), re.IGNORECASE)
            if m:
                return m.group(1)
            s = str(item).strip()
            if re.fullmatch(r"\d{4,9}", s):
                return s
    return ""


def _format_earth_pubmed(ref: dict | None) -> list[str]:
    out: list[str] = []
    if not ref:
        return out
    for item in (ref.get("pubmed", []) or []):
        if isinstance(item, dict):
            pmid = str(item.get("pmid", "") or "").strip()
            title = str(item.get("title", "") or item.get("citation", "") or "").strip()
            if pmid and title:
                out.append(f"{title} (PMID {pmid})")
            elif pmid:
                out.append(f"PMID {pmid}")
            elif title:
                out.append(title)
            elif item.get("citation"):
                out.append(str(item["citation"]))
        else:
            out.append(str(item))
    return out


def _rerank(candidates: list[tuple[int, float]], ref: dict | None,
            stressor_hint: str = "") -> list[tuple[int, float]]:
    """Biological re-rank: W_COSINE*cosine + W_PATHWAY*overlap + W_EFFECT*effect.

    Gene-in-ref sets the established *flag* only (no score hack).
    Stressor-hint tie-break: +RERANK_STRESSOR_BONUS if hint substring in doc stressor.
    Deterministic: tie-break by (experiment_id, gene).
    """
    docs = get_corpus()
    hint = (stressor_hint or "").strip().lower()
    scored: list[tuple[float, str, str, int]] = []
    for idx, cos in candidates:
        d = docs[idx] if 0 <= idx < len(docs) else {}
        pw_bonus = _pathway_overlap_bonus(str(d.get("pathway", "")),
                                          str(d.get("go_terms", "")), ref)
        try:
            fc = abs(float(d.get("log2FC", 0.0)))
        except (TypeError, ValueError):
            fc = 0.0
        effect_bonus = min(1.0, fc / EFFECT_NORM_FC)
        score = RERANK_W_COSINE * float(cos) + RERANK_W_PATHWAY * pw_bonus + RERANK_W_EFFECT * effect_bonus
        if hint and hint in str(d.get("stressor", "")).lower():
            score += RERANK_STRESSOR_BONUS
        scored.append((round(float(score), 4), str(d.get("experiment_id", "")),
                       str(d.get("symbol", "")), idx))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(idx, score) for score, _, _, idx in scored]


def parse_query(nl: str) -> ParsedQuery:
    """Parse natural-language query -> {disease, stressor_hint, pathway_hint}."""
    q = (nl or "").strip()
    ql = q.lower()
    disease = ""
    stressor_hint = ""
    pathway_hint = ""
    # stressor hints
    if any(k in ql for k in ("microgravity", "weightlessness", "unloading",
                             "disuse", "spaceflight", "space flight", "iss")):
        stressor_hint = "microgravity"
    if any(k in ql for k in ("radiation", "cosmic", "galactic", "solar particle",
                             "high-let", "high let", "dna damage", "genomic")):
        stressor_hint = "radiation"
    if any(k in ql for k in ("isolation", "confinement", "bed rest", "bedrest")):
        stressor_hint = stressor_hint or "isolation"
    # pathway hints (lightweight keyword map)
    if any(k in ql for k in ("wnt", "ossification", "osteoblast", "rankl", "bone resorption")):
        pathway_hint = "Wnt signaling"
    elif any(k in ql for k in ("atrophy", "ubiquitin", "proteasome", "myostatin", "muscle")):
        pathway_hint = "ubiquitin-proteasome"
    elif any(k in ql for k in ("senescence", "inflammag", "sasp", "p16", "p21", "cytokine")):
        pathway_hint = "senescence"
    elif any(k in ql for k in ("dna repair", "p53", "double-strand", "double strand",
                               "gadd45", "brca", "atm")):
        pathway_hint = "DNA repair"
    elif any(k in ql for k in ("mitochondr", "pgc", "ampk", "oxidative")):
        pathway_hint = "mitochondrial"
    elif "alzheimer" in ql or "amyloid" in ql or "tau" in ql:
        pathway_hint = "Alzheimer disease"
    # disease mapping (order matters: specific panels first)
    if any(k in ql for k in ("radiation", "dna damage", "genomic instability",
                             "carcinogenesis", "cancer", "p53 signaling")):
        disease = "Space radiation carcinogenesis"
    elif any(k in ql for k in ("neuro", "cognition", "cognitive", "alzheimer",
                               "dementia", "brain", "hippocampus")):
        disease = "Neurodegeneration"
    elif any(k in ql for k in ("cardio", "heart", "cardiac", "ventricle",
                               "deconditioning", "orthostatic")):
        disease = "Cardiovascular deconditioning"
    elif any(k in ql for k in ("immune", "senescence", "inflammag", "sasp",
                               "inflammation", "cytokine", "pbmc", "thymus", "spleen")):
        disease = "Immune senescence"
    elif any(k in ql for k in ("muscle", "atrophy", "sarcopenia", "wasting",
                               "myostatin", "soleus", "gastrocnemius", "quadriceps", "aging")):
        # "muscle aging" -> sarcopenia panel
        disease = "sarcopenia"
    elif any(k in ql for k in ("bone", "osteoporosis", "osteoblast", "osteoclast",
                               "sost", "sclerostin", "fracture", "bmd", "calvaria",
                               "tibia", "femur")):
        disease = "osteoporosis"
    if not disease:
        disease = q  # fall back to raw text for _find_disease substring match
    return {"disease": disease, "stressor_hint": stressor_hint,
            "pathway_hint": pathway_hint}


def _classify_evidence(similarity: float, log2fc: float, in_ref: bool) -> EvidenceLevel:
    # Weak retrieval matches are hypotheses, even if the flight DEG itself is real.
    # Threshold is backend-aware (see inferred_threshold()).
    if float(similarity) < inferred_threshold() or abs(float(log2fc)) < MIN_ABS_FC_DETECT:
        return "inferred"
    if in_ref:
        return "established"
    if abs(float(log2fc)) >= MIN_ABS_FC_OBSERVED:
        return "observed"
    return "inferred"


_CAP = {"observed": "Observed", "established": "Established", "inferred": "Inferred"}


@dataclass(frozen=True)
class ResolvedQuery:
    """Everything retrieval needs to know about the disease up front.

    Unknown input yields ref=None, which downstream treats as
    observed/inferred only — never established.
    """
    raw: str                # verbatim user input (echoed back, never matched on)
    lookup_name: str        # panel name from parse_query (what matching uses)
    stressor_hint: str
    ref: dict | None        # Earth disease panel, if the input matched one
    ref_genes: frozenset    # uppercased symbols in this panel (empty if ref is None)
    disease_label: str      # display name: panel name if known, else raw input
    mech_default: str       # fallback mechanism text when a gene has none


def _ensure_index() -> None:
    """Prefer the persisted index (offline fast path); build fresh otherwise."""
    if _EMB is None:
        if not load_index():
            build_index()


def _resolve_disease(disease_name: str, disease_refs: list) -> ResolvedQuery:
    """Map raw input to a disease panel without touching the index."""
    parsed = parse_query(disease_name)
    lookup_name = parsed.get("disease") or disease_name
    ref = _find_disease(lookup_name, disease_refs)
    label = (disease_name or "").strip() or "query"
    genes: frozenset = frozenset()
    mech_default = "shared pathway"
    if ref:
        label = str(ref.get("disease") or ref.get("name") or label)
        genes = frozenset(s for s in (_gene_symbol(g) for g in (ref.get("genes") or [])) if s)
        mechs = _ref_mechanisms(ref)
        mech_default = str(mechs[0]) if mechs else mech_default
    return ResolvedQuery(
        raw=disease_name,
        lookup_name=lookup_name,
        stressor_hint=parsed.get("stressor_hint", ""),
        ref=ref,
        ref_genes=genes,
        disease_label=label,
        mech_default=mech_default,
    )


def _retrieve_ranked(qtext: str, ref: dict | None, stressor_hint: str,
                     top_k: int) -> list[tuple[int, float]]:
    """Embed the disease text, take a wide cosine net, bio re-rank, slice top-k."""
    qv = get_embedding([qtext])[0]
    n_cand = max(int(top_k) * CANDIDATE_POOL_MULT, CANDIDATE_POOL_MIN)
    candidates = _cosine_search(qv, n_cand)
    return _rerank(candidates, ref, stressor_hint)[: max(1, int(top_k))]


def _build_hit(doc: dict, rank: int, sim: float, resolved: ResolvedQuery,
               earth_pubmed_list: list) -> Hit:
    """Assemble one ranked hit dict from a corpus doc. Pure formatting — no retrieval."""
    ref = resolved.ref
    gene = doc["symbol"]
    fc = float(doc.get("log2FC", 0.0))
    # Established only when the gene is in THIS disease's Earth ref panel.
    # Unknown diseases (ref=None) yield observed/inferred, never established.
    in_ref = gene in resolved.ref_genes if ref else False
    ev = _classify_evidence(float(sim), fc, bool(in_ref))
    is_established = bool(in_ref)
    is_observed = abs(fc) >= MIN_ABS_FC_DETECT
    gene_mech = _gene_mechanism(ref, gene, resolved.mech_default)
    stressor_list = [s.strip() for s in str(doc["stressor"]).split(",") if s.strip()] or [doc["stressor"]]
    exp_refs = list(doc.get("pubmed_refs", []) or [])
    go_terms = str(doc.get("go_terms", "") or "")
    pmid = _extract_first_pmid(exp_refs, earth_pubmed_list)
    change = f"{doc['direction']}regulated (log2FC {fc:.2f})"
    hit: Hit = {
        "gene": gene,
        "experiment_id": doc["experiment_id"],
        "mission": doc["mission"],
        "organism": doc["organism"],
        "tissue": doc["tissue"],
        "stressor": doc["stressor"],
        "log2FC": fc,
        "direction": doc["direction"],
        "pathway": doc.get("pathway", ""),
        "earth_disease": resolved.disease_label,
        "earth_mechanism": gene_mech,
        "similarity": round(float(sim), 4),
        "evidence_level": ev,
        "osdr_url": doc["osdr_url"],
        "rationale_template": "",
        # --- aliases for Streamlit frontend compat (engine keys above are canonical) ---
        "experiment": doc["experiment_id"],
        "stressors": stressor_list,
        "evidence": _CAP[ev],
        "osdr_link": doc["osdr_url"],
        "rationale": "",
        "rank": rank,
        "disease": resolved.disease_label,
        "earth_score": round(float(sim), 4),
        "pubmed": pmid,
        "observed": f"{gene} {doc['direction']}regulated (log2FC {fc:.2f}) in {doc['tissue']} ({doc['experiment_id']}) — potential biological connection, supported by available evidence.",
        "established": (f"Earth link: {resolved.disease_label} via {gene_mech} — gene reported in Earth literature; "
                        f"candidate target for further investigation." if is_established
                        else f"Potential biological connection to {resolved.disease_label} via {gene_mech}; candidate for further investigation."),
        "hypothesis": "Potential biological connection — candidate for further investigation. Hypothesis — requires experimental validation.",
        "pubmed_refs": exp_refs,
        "earth_pubmed": earth_pubmed_list,
        "go_terms": go_terms,
        "evidence_chain": {
            "experiment": doc["experiment_id"],
            "stressor": doc["stressor"],
            "tissue": doc["tissue"],
            "gene": gene,
            "change": change,
            "pathway": doc.get("pathway", ""),
            "disease": resolved.disease_label,
            "mechanism": gene_mech,
        },
        "is_observed": is_observed,
        "is_established": is_established,
    }
    hit["rationale_template"] = generate_rationale(hit)
    hit["rationale"] = hit["rationale_template"]
    return hit


def query(disease_name: str, top_k: int = 10) -> list[Hit]:
    """Retrieve top-k (experiment, gene) pairs for a disease.

    Pipeline: resolve disease → ensure index → retrieve + re-rank → build hits.
    Returns ranked Hits; see Hit for the key contract (core keys + frontend aliases).
    """
    _, disease_refs, _ = load_data()
    docs = get_corpus()
    if not docs:
        return []
    _ensure_index()
    resolved = _resolve_disease(disease_name, disease_refs)
    qtext = _disease_query_text(resolved.lookup_name, resolved.ref)
    ranked = _retrieve_ranked(qtext, resolved.ref, resolved.stressor_hint, top_k)
    earth_pubmed_list = _format_earth_pubmed(resolved.ref)
    return [_build_hit(docs[idx], rank, sim, resolved, earth_pubmed_list)
            for rank, (idx, sim) in enumerate(ranked, start=1)]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "osteoporosis"
    for h in query(q, top_k=3):
        print(h)
