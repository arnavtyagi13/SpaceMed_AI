# SpaceMed AI — Space-to-Earth Knowledge Discovery

Linking microgravity omics to terrestrial disease for faster therapeutic insight.

## Live demo

> **Live demo URL:** _coming soon — deploy `app.py` to Streamlit Community Cloud and paste the public link here._

Run it locally instead with the [Quickstart](#quickstart) below (or one-click `run_demo.bat`).

## Why this matters

Spaceflight acts as an accelerated aging model: within weeks, astronauts exposed to microgravity, cosmic radiation, and confinement develop osteoporosis-like bone loss (~1%/month), sarcopenia-like muscle atrophy, and immune senescence/inflammaging that normally take decades on Earth. SpaceMed AI systematically maps these spaceflight gene signatures to Earth diseases to surface candidate drug targets and repurposing hypotheses — e.g. the sclerostin/Wnt axis (SOST, target of romosozumab) and RANKL/OPG axis (TNFRSF11B/TNFSF11, target of denosumab) for osteoporosis, the myostatin/FOXO-atrogene program (MSTN, FBXO32, FOXO1/3) for sarcopenia, and senescence/SASP axes (CDKN2A/p16, IL6) guiding senolytics research for immune senescence.

## Architecture

```
OSDR JSON (data/osdr_experiments.json, earth_disease_refs.json, gene_annotations.csv)
   |  cached curated subset mirroring OSDR study IDs (30 experiments, not live ingestion)
   v
corpus (spacemed/engine.py: build_corpus — one doc per experiment x gene pair, 227 docs)
   |
   v
NL parse_query routing (natural language -> {disease, stressor_hint, pathway_hint})
   |
   v
embeddings: sBERT-preferred (sentence-transformers all-MiniLM-L6-v2) + sklearn TF-IDF offline fallback
   |
   v
retrieval: FAISS (IndexFlatIP) if installed, else numpy brute-force cosine (wide candidate net)
   |
   v
re-ranking: Disease -> mechanism -> gene -> experiment
   (0.6*cosine + 0.25*pathway_overlap + 0.15*effect-size, + stressor-hint tie-break;
   weights live in spacemed/engine.py retrieval configuration)
   |
   v
query() pipeline (spacemed/engine.py): resolve disease → ensure index →
retrieve + re-rank → build Hit records (observed/established/hypothesis)
   |
   v
display shaping (spacemed/present.py, pure functions, no Streamlit):
summarize_answer / group_genes / group_experiments / collect_references /
table_rows — the UI renders these structures without retrieval logic
   |
   v
Streamlit research interface (app.py): landing (hero + Explore + 5 example questions +
How-it-works) → Your question → Explore the evidence → What changed in space? →
Follow the biological trail → See where the evidence comes from (OSDR links) →
What we know (PubMed) → Why does it matter? → What we're still investigating →
Data & limitations → optional visual exploration (space-vs-Earth heatmap,
experiment→gene→pathway→disease graph) → results table + CSV → technical details
(Simple/Research view toggle) → disclaimer.

Data model (`spacemed/schema.py`): required vs optional OSDR record fields with
warn-only validation (`python -m spacemed.build_index` prints a validation summary),
so future compatible studies can extend the verified 30-experiment demo corpus
without breaking retrieval.
```

Index persistence: `data/faiss.index` (numpy array) + `data/corpus_meta.json`.
Stale/incompatible indexes self-heal (rebuilt automatically on dimension mismatch).

Backend selection: sBERT is preferred when `sentence-transformers` is installed;
otherwise the offline sklearn TF-IDF path is used. Evidence thresholds are
backend-aware (inferred below 0.65 sBERT / 0.30 TF-IDF). For bit-identical
reproduction of the tables below, pin the offline path:

```bat
set SPACEMED_BACKEND=tfidf
python -m spacemed.build_index
```

## Quickstart

```bat
pip install -r requirements.txt
python -m spacemed.build_index
streamlit run app.py
```

Or one-click on Windows: `run_demo.bat`.

## Testing

```bat
pip install pytest
python -m pytest tests/ -q
```

`tests/test_spacemed.py` pins `SPACEMED_BACKEND=tfidf` for speed and determinism
(sBERT returns the same flagship genes at higher scores by design). It covers
NL routing, flagship-gene retrieval, ranking determinism and top-k stability,
the evidence rules (unknown/weak can never be established), schema validation,
annotation coverage, and the pure display helpers in `spacemed/present.py`.

## Example queries (verified live outputs, TF-IDF backend; sBERT agrees on genes)

Flagship genes are backend-invariant: with sBERT, the same panels top the
rankings at higher scores (osteoporosis → SOST ×5, sim ~0.88–0.90; sarcopenia →
FBXO32 ×5, sim ~0.87–0.88; immune senescence → CDKN2A/IL1B/IL6, sim ~0.82–0.85;
radiation → GADD45A/CDKN1A, sim ~0.91). Unknown queries score ~0.5 (sBERT) /
~0.2 (TF-IDF) and are labelled inferred/hypothesis-only on both backends.

1. `osteoporosis` (flagship) → **SOST (sclerostin) up + RANKL up / OPG down** — Wnt inhibition plus raised RANKL/OPG ratio
   - Top-5: SOST (OSD-105, sim 0.5791), SOST (OSD-258, 0.5708), TNFSF11/RANKL (OSD-48, 0.5637), TNFRSF11B/OPG (OSD-48, 0.5635), SOST (OSD-48, 0.5606)
2. `sarcopenia` → **FBXO32 (Atrogin-1) atrogene program** (ubiquitin-proteasome; TRIM63/FOXO also in panel)
   - Top-5: FBXO32 (OSD-21, 0.5726), FBXO32 (OSD-256, 0.5717), FBXO32 (OSD-162, 0.5617), FBXO32 (OSD-104, 0.5598), FBXO32 (OSD-3, 0.5578)
3. `immune senescence` → **CDKN2A (p16) / IL6 / SERPINE1** (SASP/senescence signature)
   - Top-5: CDKN2A (OSD-101, 0.5582), IL6 (OSD-101, 0.5553), CDKN2A (OSD-99, 0.5450), IL6 (OSD-99, 0.5406), SERPINE1/PAI-1 (OSD-101, 0.5372)
   - NL routing: "What does microgravity reveal about osteoporosis?" → osteoporosis panel; "Which spaceflight genes are associated with muscle aging?" → sarcopenia; "What molecular pathways respond to space radiation?" → Space radiation carcinogenesis (top hit GADD45A, DNA-damage response).

Reproduce via:

```bat
python -c "from spacemed.engine import query; print(query('osteoporosis', top_k=5))"
```

See `demo_queries.md` for full top-5 tables.

## Demo workflow (what the judge sees, top to bottom)

Flagship — osteoporosis (microgravity bone loss):
short answer ("What does microgravity reveal about osteoporosis?")
→ key findings (SOST ↑, RANKL ↑, OPG ↓ cards from live retrieval)
→ evidence trail (OSD-105/48/258 → microgravity → tibia/femur bone → gene → log2FC change → Wnt / bone-resorption pathway → osteoporosis)
→ experiment cards with "View NASA source →" OSDR links
→ scientific references (PubMed IDs incl. Sost/sclerostin PMID 18323415)
→ clearly labelled hypothesis (e.g. RANKL-axis modulation as a candidate countermeasure — AI-inferred, requires validation).

Secondaries (one click each): sarcopenia → FBXO32/TRIM63 atrophy program;
immune aging → IL6/CDKN2A (p16) senescence signature; radiation effects →
GADD45A/CDKN1A DNA-damage response. Unrelated queries (e.g. `xyzzy`) show a
"Not enough evidence" panel with suggestions instead of invented links.

## Scientific limitations

- Curated 30-experiment cache, not live OSDR ingestion: study IDs mirror OSDR (`https://osdr.nasa.gov/bio/repo/data/studies/OSD-XX`) but records are a frozen demo subset.
- Transcriptomics only: differential expression (log2FC/direction); no proteomics, epigenomics, or phenotypic validation in the loop.
- Rodent-to-human translation gap: most flight experiments are murine; Earth disease links are largely human — cross-species inference is uncertain.
- Similarity ≠ causation: retrieval + re-rank scores measure text/signature resemblance, not causal evidence.
- All hypotheses require experimental validation before any translational use.
- No clinical claims: nothing here is medical advice or a treatment recommendation.

## Evidence-level legend (shown in the app as Observed / Established / Hypothesis)

- 🔵 **Observed** — directly measured in the NASA dataset.
- 🟢 **Established** — supported by existing scientific literature.
- 🟡 **Hypothesis** — a possible connection suggested by the system. Requires further research.

## Citations

- NASA Open Science Data Repository (OSDR): https://osdr.nasa.gov/
- This demo ships a cached curated subset mirroring OSDR study IDs (patterns `https://osdr.nasa.gov/bio/repo/data/studies/OSD-XX`); it is not a live OSDR feed.
- Per-hit OSDR records, e.g. https://osdr.nasa.gov/bio/repo/data/studies/OSD-48, https://osdr.nasa.gov/bio/repo/data/studies/OSD-21, https://osdr.nasa.gov/bio/repo/data/studies/OSD-99
- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- Representative PubMed IDs used in the frontend demo data: 31654570 (SOST/sclerostin), 30642874 (RANKL/OPG), 32848243 (myostatin), 33051659 (p16 senescence), 32976898 (IL-6 inflammaging). Per-hit PubMed links appear in the app evidence panel.

## Disclaimer

All outputs labelled **"AI-inferred hypothesis"** are computational hypotheses describing a **potential biological connection**, **not facts** and not clinical advice. Any nominated gene is a **candidate target for further investigation** only — **not ready for clinical testing**. Hypotheses require experimental validation before any translational or therapeutic use.
