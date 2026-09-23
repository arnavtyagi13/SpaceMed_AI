# SpaceMed AI — Demo queries (live engine outputs; TF-IDF tables + sBERT cross-check)

Reproduce TF-IDF tables exactly with: `set SPACEMED_BACKEND=tfidf` then
`python -c "from spacemed.engine import query; print(query('<disease>', top_k=5))"`.
Default (auto) prefers sBERT when installed: same flagship genes at higher scores
(osteoporosis → SOST ×5 ~0.88–0.90; sarcopenia → FBXO32 ×5 ~0.87–0.88;
immune senescence → CDKN2A/IL1B/IL6 ~0.82–0.85; radiation → GADD45A/CDKN1A ~0.91).
Unknown queries (e.g. `xyzzy`) score ~0.5 sBERT / ~0.2 TF-IDF and are labelled
inferred/hypothesis-only on both backends.

Reproduce with: `python -c "from spacemed.engine import query; print(query('<disease>', top_k=5))"`

## NL questions → panel mapping (verified via `spacemed.engine.parse_query`)

| # | natural-language question | routed disease panel | stressor hint | pathway hint |
|---|---------------------------|----------------------|---------------|--------------|
| 1 | How does microgravity cause bone loss like osteoporosis? | osteoporosis | microgravity | — |
| 2 | What muscle aging genes overlap with sarcopenia? | sarcopenia | — | ubiquitin-proteasome |
| 3 | How does space radiation damage DNA and raise cancer risk? | Space radiation carcinogenesis | radiation | — |

Backend note: tables below are live TF-IDF outputs from `spacemed.engine.query()`
(re-ranked 0.6 cosine + 0.25 pathway-overlap + 0.15 effect size; 227 docs).
Pin with `set SPACEMED_BACKEND=tfidf` before running. Hypotheses are AI-inferred
and require validation — not facts, not clinical claims.

## 1. osteoporosis — flagship: SOST up + RANKL up / OPG down

| rank | gene | experiment | tissue | direction | log2FC | similarity | evidence |
|------|------|------------|--------|-----------|--------|------------|----------|
| 1 | SOST (sclerostin) | OSD-105 | tibia bone | up | 1.84 | 0.5791 | Established |
| 2 | SOST (sclerostin) | OSD-258 | tibia bone | up | 1.72 | 0.5708 | Established |
| 3 | TNFSF11 (RANKL) | OSD-48 | femur bone | up | 1.25 | 0.5637 | Established |
| 4 | TNFRSF11B (OPG) | OSD-48 | femur bone | down | -0.83 | 0.5635 | Established |
| 5 | SOST (sclerostin) | OSD-48 | femur bone | up | 1.68 | 0.5606 | Established |

Interpretation (potential biological connection, supported by available evidence): sclerostin up
(Wnt inhibition) + RANKL up / OPG down → raised RANKL/OPG ratio and osteoclastogenesis;
Earth link osteoporosis via the sclerostin-Wnt axis (romosozumab context) and RANKL/OPG axis
(denosumab context) — candidate targets for further investigation, not clinical claims.

## 2. sarcopenia — expected top genes: FBXO32 / TRIM63 / FOXO1

| rank | gene | experiment | tissue | direction | log2FC | similarity | evidence |
|------|------|------------|--------|-----------|--------|------------|----------|
| 1 | FBXO32 (Atrogin-1) | OSD-21 | soleus muscle | up | 2.31 | 0.5726 | Established |
| 2 | FBXO32 (Atrogin-1) | OSD-256 | soleus muscle | up | 2.12 | 0.5717 | Established |
| 3 | FBXO32 (Atrogin-1) | OSD-162 | quadriceps muscle | up | 2.02 | 0.5617 | Established |
| 4 | FBXO32 (Atrogin-1) | OSD-104 | soleus muscle | up | 2.45 | 0.5598 | Established |
| 5 | FBXO32 (Atrogin-1) | OSD-3 | soleus muscle | up | 2.02 | 0.5578 | Established |

Interpretation: FOXO-driven atrogene program (Atrogin-1/MuRF1); Earth link sarcopenia via ubiquitin-proteasome atrophy program.

## 3. immune senescence — expected top genes: IL6 / CDKN2A

| rank | gene | experiment | tissue | direction | log2FC | similarity | evidence |
|------|------|------------|--------|-----------|--------|------------|----------|
| 1 | CDKN2A (p16) | OSD-101 | spleen | up | 1.41 | 0.5582 | Established |
| 2 | IL6 | OSD-101 | spleen | up | 1.34 | 0.5553 | Established |
| 3 | CDKN2A (p16) | OSD-99 | thymus | up | 1.22 | 0.5450 | Established |
| 4 | IL6 | OSD-99 | thymus | up | 1.08 | 0.5406 | Established |
| 5 | SERPINE1 (PAI-1) | OSD-101 | spleen | up | 1.18 | 0.5372 | Established |

Interpretation: SASP/senescence signature (IL-6 + p16 + PAI-1); Earth link immune senescence/inflammaging, guiding senolytics hypotheses.

Note: all rows above are live outputs from `spacemed.engine.query()` against `data/*.json` (not hardcoded). Hypotheses are AI-inferred and require validation — not facts.
