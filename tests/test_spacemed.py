"""Contract tests for SpaceMed AI — scientific behavior + display shaping.

Run from the repo root:  pip install pytest  &&  python -m pytest tests/ -q

The embedding backend is pinned to TF-IDF for speed and determinism.
sBERT returns the same flagship genes at higher scores by design, so these
tests assert genes/labels/properties — never exact cross-backend scores.
"""
import os

os.environ["SPACEMED_BACKEND"] = "tfidf"

from spacemed import engine as E
from spacemed import present as P
from spacemed import schema as S


# ---------------------------------------------------------------------------
# Query parsing: NL routing contracts (order-sensitive by design)
# ---------------------------------------------------------------------------
def test_parse_osteoporosis():
    p = E.parse_query("osteoporosis")
    assert p["disease"] == "osteoporosis"


def test_parse_muscle_aging():
    p = E.parse_query("Which genes change during spaceflight that are relevant to muscle aging?")
    assert p["disease"] == "sarcopenia"


def test_parse_immune_aging():
    p = E.parse_query("What NASA experiments are relevant to immune aging?")
    assert p["disease"] == "Immune senescence"


def test_parse_radiation():
    p = E.parse_query("What biological pathways are associated with radiation-related changes?")
    assert p["disease"] == "Space radiation carcinogenesis"
    assert p["stressor_hint"] == "radiation"


def test_parse_unknown_falls_back_to_raw():
    p = E.parse_query("xyzzy")
    assert p["disease"] == "xyzzy"


def test_parse_empty():
    p = E.parse_query("")
    assert p == {"disease": "", "stressor_hint": "", "pathway_hint": ""}


# ---------------------------------------------------------------------------
# Retrieval: flagship genes, determinism, top-k semantics
# ---------------------------------------------------------------------------
def _genes(q, k=5):
    return [h["gene"] for h in E.query(q, top_k=k)]


def test_osteoporosis_flagship_genes():
    genes = _genes("osteoporosis")
    assert "SOST" in genes[:3]
    assert {"TNFSF11", "TNFRSF11B"} & set(genes)


def test_sarcopenia_flagship_genes():
    assert "FBXO32" in _genes("sarcopenia")[:3]


def test_immune_flagship_genes():
    assert {"CDKN2A", "IL6"} & set(_genes("immune senescence")[:3])


def test_radiation_flagship_genes():
    assert "GADD45A" in _genes("What biological pathways are associated with radiation-related changes?")[:3]


def test_ranking_deterministic():
    a = [(h["gene"], h["experiment_id"], h["similarity"]) for h in E.query("osteoporosis", top_k=5)]
    b = [(h["gene"], h["experiment_id"], h["similarity"]) for h in E.query("osteoporosis", top_k=5)]
    assert a == b


def test_top_k_respected_and_stable_prefix():
    assert len(E.query("osteoporosis", top_k=1)) == 1
    assert len(E.query("osteoporosis", top_k=30)) == 30
    top1 = E.query("osteoporosis", top_k=1)[0]
    top5 = E.query("osteoporosis", top_k=5)
    assert (top1["gene"], top1["experiment_id"]) == (top5[0]["gene"], top5[0]["experiment_id"])


def test_hit_schema_and_ranges():
    for h in E.query("sarcopenia", top_k=5):
        for key in ("gene", "experiment_id", "mission", "organism", "tissue",
                    "log2FC", "direction", "similarity", "evidence_level",
                    "osdr_url", "earth_disease", "rationale"):
            assert key in h, key
        assert 0.0 <= h["similarity"] <= 1.0
        assert h["evidence_level"] in ("observed", "established", "inferred")
        assert h["osdr_url"].startswith("https://osdr.nasa.gov/bio/repo/data/studies/OSD-")


# ---------------------------------------------------------------------------
# Evidence: weak/unknown can never be established
# ---------------------------------------------------------------------------
def test_unknown_never_established():
    for h in E.query("xyzzy", top_k=10):
        assert h["evidence_level"] in ("observed", "inferred")


def test_empty_query_safe():
    hits = E.query("", top_k=5)
    assert isinstance(hits, list) and hits
    assert all(h["evidence_level"] in ("observed", "inferred") for h in hits)


def test_classify_evidence_rules():
    assert E._classify_evidence(0.10, 1.5, True) == "inferred"   # weak similarity wins
    assert E._classify_evidence(0.90, 0.10, True) == "inferred"  # negligible fold-change
    assert E._classify_evidence(0.90, 1.50, True) == "established"
    assert E._classify_evidence(0.90, 1.50, False) == "observed"
    assert E._classify_evidence(0.90, 0.40, False) == "inferred"  # between cutoffs


def test_inferred_threshold_tfidf():
    assert E.inferred_threshold() == 0.30


# ---------------------------------------------------------------------------
# Data + schema validation
# ---------------------------------------------------------------------------
def test_live_corpus_counts():
    exps, refs, _ = E.load_data()
    assert len(exps) == 30
    assert len(refs) == 6
    assert sum(len(e.get("genes", [])) for e in exps) == 227


def test_live_corpus_validates_clean():
    exps, _, _ = E.load_data()
    summary = S.validate_corpus(exps)
    assert summary["n_experiments"] == 30
    assert summary["n_warnings"] == 0


def test_validate_rejects_bad_record():
    assert S.validate_experiment({})  # missing everything
    assert S.validate_experiment(None)
    assert S.validate_experiment({"osdr_id": "OSD-X", "genes": [{"symbol": ""}]})
    ok = {
        "osdr_id": "OSD-1", "organism": "Mus musculus", "tissue": "liver",
        "mission": "ISS", "stressors": ["microgravity"],
        "genes": [{"symbol": "SOST", "log2FC": 1.0, "direction": "up"}],
    }
    assert S.validate_experiment(ok) == []


def test_annotations_cover_experiment_genes():
    import csv
    exps, _, _ = E.load_data()
    used = {g.get("symbol") for e in exps for g in e.get("genes", [])}
    with open("data/gene_annotations.csv", encoding="utf-8") as f:
        have = {r["symbol"] for r in csv.DictReader(f)}
    assert used <= have


# ---------------------------------------------------------------------------
# Presentation layer: pure formatting contracts
# ---------------------------------------------------------------------------
def test_shorten_pathway():
    assert P.shorten_pathway("KEGG:mmu04310 Wnt signaling pathway") == "Wnt signaling pathway"
    assert P.shorten_pathway("GO:0001503 ossification; KEGG:mmu04310 Wnt") == "ossification"
    assert P.shorten_pathway("") == ""


def test_format_fc():
    assert P.format_fc(1.234) == "+1.23"
    assert P.format_fc(-0.5) == "-0.50"
    assert P.format_fc("nope") == "—"


def test_evidence_display_never_says_inferred():
    assert P.evidence_label("inferred") == "Hypothesis"
    assert P.evidence_label("observed") == "Observed"
    assert P.evidence_label("established") == "Established"


def test_relevance_hides_raw_scores():
    assert P.relevance_word(0, "established") == "High"
    assert P.relevance_word(9, "observed") == "Moderate"
    assert P.relevance_word(0, "inferred") == "Limited"


def test_direction_words():
    assert P.direction_word("Up") == "Increased"
    assert P.direction_word("down") == "Decreased"
    assert P.arrow("Up") == "↑" and P.arrow("Down") == "↓"


def test_csv_filename_safe():
    assert P.csv_filename("Immune Senescence!") == "spacemed_immune_senescence.csv"
    assert P.csv_filename("") == "spacemed_query.csv"


def test_findings_pool_size():
    assert P.findings_pool_size(5) == 12
    assert P.findings_pool_size(30) == 30


def test_group_genes_orders_and_counts():
    pool = E.query("osteoporosis", top_k=12)
    groups = P.group_genes(pool, {})
    assert groups and len(groups) <= 4
    assert groups[0].gene == "SOST"
    assert groups[0].n_total >= 1 and groups[0].consistency.endswith("experiments")
    assert all(g.experiments for g in groups)


def test_apply_filters_is_pure_subset():
    rows = E.query("osteoporosis", top_k=10)
    sub = P.apply_filters(rows, ["microgravity"], "All", [])
    assert sub and len(sub) <= len(rows)
    assert [r["gene"] for r in sub] == [r["gene"] for r in rows if r in sub]
    assert P.apply_filters(rows, ["no-such-stressor"], "All", []) == []


def test_collect_references_dedupes():
    rows = E.query("osteoporosis", top_k=5)
    refs = P.collect_references(rows)
    assert refs
    pmids = [p for p, _ in refs if p]
    assert pmids and all(len(p) in (7, 8) and p.isdigit() for p in pmids)


def test_table_rows_gate_scores():
    rows = E.query("osteoporosis", top_k=3)
    simple = P.table_rows(rows, research_view=False)
    assert all("score" not in r for r in simple)
    research = P.table_rows(rows, research_view=True)
    assert all("score" in r for r in research)


def test_hypothesis_genes_lists_inferred_only():
    pool = E.query("xyzzy", top_k=12)
    groups = P.group_genes(pool, {})
    hyps = P.hypothesis_genes(pool, [g.gene for g in groups])
    assert hyps  # unknown query: everything is hypothesis
    assert all(isinstance(g, str) and isinstance(m, str) for g, m in hyps)


def test_summarize_answer_grounded():
    pool = E.query("osteoporosis", top_k=12)
    s = P.summarize_answer(pool, E.query("osteoporosis", top_k=5))
    assert "SOST" in s["gene_phrase"]
    assert s["panel_label"] == "Osteoporosis"
    assert s["exp_ids"] and s["tissues"] and "Wnt" in s["text"] or "ossification" in s["text"]


def test_best_hit_picks_max_and_tolerates_malformed():
    rows = [
        {"gene": "X", "similarity": 0.2},
        {"gene": "X", "similarity": "not-a-number"},
        {"gene": "X"},
        {"gene": "X", "similarity": 0.9},
    ]
    assert P.best_hit(rows)["similarity"] == 0.9
    assert P.safe_similarity({"similarity": "bad"}) == 0.0
    assert P.safe_similarity({}) == 0.0


def test_dominant_direction_tie_breaks_deterministically():
    pool = [
        {"gene": "TIE", "direction": "Up", "similarity": 0.5, "experiment": "OSD-1",
         "tissue": "t", "organism": "o", "mission": "m", "stressor": "s",
         "stressors": ["s"], "log2FC": 1.0, "pathway": "p", "evidence": "Observed",
         "earth_mechanism": "m", "osdr_link": "https://osdr.nasa.gov/"},
        {"gene": "TIE", "direction": "Down", "similarity": 0.4, "experiment": "OSD-2",
         "tissue": "t", "organism": "o", "mission": "m", "stressor": "s",
         "stressors": ["s"], "log2FC": -1.0, "pathway": "p", "evidence": "Observed",
         "earth_mechanism": "m", "osdr_link": "https://osdr.nasa.gov/"},
    ]
    first = P.group_genes(pool, {})[0].dominant_direction
    assert first in ("Up", "Down")
    # Alphabetical tie-break ("down" < "up") holds in every process.
    assert P.group_genes(pool, {})[0].dominant_direction == "Down"
    assert P.group_genes(pool, {})[0].mixed_directions is True


def test_unknown_evidence_tier_degrades_to_inferred():
    assert P.evidence_key("Observd") == "Inferred"
    assert P.evidence_key("  ") == "Inferred"
    assert P.evidence_label("whatever") == "Hypothesis"
