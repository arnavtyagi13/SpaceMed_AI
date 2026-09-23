"""Build + persist the SpaceMed AI retrieval index.

Usage:  python -m spacemed.build_index [--query osteoporosis]
Saves:  data/faiss.index + data/corpus_meta.json (numpy/sklearn, no hard faiss dep).
"""
from __future__ import annotations

from spacemed.engine import build_corpus, build_index, save_index, query, EMBEDDING_BACKEND


def main() -> None:
    import spacemed.engine as E

    docs = build_corpus()
    build_index()
    idx_path, meta_path = save_index()
    print(f"corpus_docs={len(docs)} backend={E.EMBEDDING_BACKEND}")
    print(f"saved index -> {idx_path}")
    print(f"saved meta  -> {meta_path}")
    try:
        from spacemed.schema import validate_corpus

        exps, _, _ = E.load_data()
        summary = validate_corpus(exps)
        print(
            f"validation: experiments={summary.get('n_experiments')} "
            f"records={summary.get('n_records')} "
            f"warnings={summary.get('n_warnings')} "
            f"assays={summary.get('assay_types')}"
        )
        for exp_id, warns in (summary.get("warnings_by_experiment") or {}).items():
            for w in warns:
                print(f"validation warning [{exp_id}]: {w}")
    except Exception as exc:
        print(f"validation skipped: {exc}")


if __name__ == "__main__":
    import sys

    main()
    if "--query" in sys.argv:
        i = sys.argv.index("--query")
        q = sys.argv[i + 1] if i + 1 < len(sys.argv) else "osteoporosis"
        for h in query(q, top_k=3):
            print(h)
