"""SpaceMed AI record model (spec section 13) — verified demo vs expanded dataset.

Record model
------------
A verified demo-dataset experiment record is a dict with REQUIRED fields:

- ``osdr_id`` (str): OSDR accession, e.g. ``"OSD-21"``.
- ``organism`` (str): e.g. ``"Mus musculus"``.
- ``tissue`` (str): e.g. ``"soleus muscle"``.
- ``mission`` (str): e.g. ``"STS-135"``.
- ``stressors`` (list[str]): e.g. ``["microgravity"]``.
- ``genes`` (list[dict]): non-empty; each gene has REQUIRED
  ``symbol`` (str), ``log2FC`` (numeric), ``direction`` (``"up"``/``"down"``).

Forward-compatible OPTIONAL experiment fields (present today on most demo
records, required for future OSDR studies but never required for validation):

- ``study_version``, ``assay``, ``duration_days``, ``pathways``,
  ``pubmed_refs``, ``osdr_url``, ``provenance``, ``ingestion_date``,
  ``processing_status``.

Forward-compatible OPTIONAL gene-level fields:

- ``p_value``, ``pathway``, ``go_terms``.

Scope
-----
- "Verified demo dataset": the current curated corpus in
  ``data/osdr_experiments.json`` (~30 experiments) that the engine, app, and
  index are tested against.
- "Expanded dataset": future OSDR studies which may populate any optional
  field above (new assays, ``study_version``, provenance, per-gene statistics).
  Validators accept and ignore such additions so the demo corpus keeps working.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_EXPERIMENT_FIELDS = [
    "osdr_id",
    "organism",
    "tissue",
    "mission",
    "stressors",
    "genes",
]

OPTIONAL_EXPERIMENT_FIELDS = [
    "study_version",
    "assay",
    "duration_days",
    "pathways",
    "pubmed_refs",
    "osdr_url",
    "provenance",
    "ingestion_date",
    "processing_status",
]

REQUIRED_GENE_FIELDS = [
    "symbol",
    "log2FC",
    "direction",
]

OPTIONAL_GENE_FIELDS = [
    "p_value",
    "pathway",
    "go_terms",
]

_VALID_DIRECTIONS = ("up", "down")


def _experiment_label(exp: dict, idx: int = 0) -> str:
    """Best-effort human label for an experiment dict (never raises)."""
    try:
        for key in ("osdr_id", "experiment_id", "osd_id", "accession", "id", "experiment"):
            val = exp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if val is not None and not isinstance(val, (dict, list)):
                text = str(val).strip()
                if text:
                    return text
    except Exception:
        pass
    return f"record-{idx}"


def validate_experiment(exp: dict) -> list[str]:
    """Validate one experiment record, returning human-readable warnings.

    Never raises on bad input: non-dict input returns ``["not a record ..."]``.
    """
    try:
        if not isinstance(exp, dict):
            return ["not a record: expected a dict experiment record"]
        warnings: list[str] = []

        for field in REQUIRED_EXPERIMENT_FIELDS:
            if field not in exp or exp[field] is None:
                warnings.append(f"missing required field: {field}")
            elif isinstance(exp[field], str) and not exp[field].strip():
                warnings.append(f"missing required field: {field}")

        # stressors must be a list when present.
        if "stressors" in exp and exp["stressors"] is not None:
            if not isinstance(exp["stressors"], list):
                warnings.append("stressors should be a list of strings")

        # genes must be a non-empty list when present.
        genes = exp.get("genes")
        if "genes" in exp and genes is not None:
            if not isinstance(genes, list) or len(genes) == 0:
                warnings.append("empty genes: expected a non-empty list of gene records")
                genes = []
        else:
            genes = []

        if isinstance(genes, list):
            for i, gene in enumerate(genes):
                prefix = f"gene[{i}]"
                if not isinstance(gene, dict):
                    warnings.append(f"{prefix}: not a record (expected dict)")
                    continue
                symbol = gene.get("symbol")
                if not isinstance(symbol, str) or not symbol.strip():
                    warnings.append(f"{prefix}: missing gene symbol")
                # log2FC must be numeric.
                if "log2FC" not in gene or gene["log2FC"] is None or (
                    isinstance(gene["log2FC"], str) and not gene["log2FC"].strip()
                ):
                    warnings.append(f"{prefix}: missing log2FC")
                else:
                    try:
                        float(gene["log2FC"])
                    except (TypeError, ValueError):
                        warnings.append(f"{prefix}: non-numeric log2FC: {gene.get('log2FC')!r}")
                # direction must be up/down (case-insensitive).
                direction = gene.get("direction")
                if not isinstance(direction, str) or direction.strip().lower() not in _VALID_DIRECTIONS:
                    warnings.append(
                        f"{prefix}: direction should be 'up' or 'down' (got {direction!r})"
                    )

        return warnings
    except Exception as exc:  # defensive: never raise on bad input
        return [f"not a record: validation failed ({exc})"]


def validate_corpus(experiments: list) -> dict:
    """Validate a list of experiment records.

    Returns ``{n_experiments, n_records, n_warnings, warnings_by_experiment,
    assay_types}`` using only currently-present fields. Never raises.
    """
    try:
        if not isinstance(experiments, list):
            return {
                "n_experiments": 0,
                "n_records": 0,
                "n_warnings": 1,
                "warnings_by_experiment": {"unknown": ["not a corpus: expected a list of records"]},
                "assay_types": [],
            }
        warnings_by_experiment: dict[str, list[str]] = {}
        assay_set: set[str] = set()
        n_records = 0
        for idx, exp in enumerate(experiments):
            if isinstance(exp, dict):
                genes = exp.get("genes")
                if isinstance(genes, list):
                    n_records += len(genes)
                assay = exp.get("assay")
                if isinstance(assay, str) and assay.strip():
                    assay_set.add(assay.strip())
                label = _experiment_label(exp, idx)
                try:
                    warnings = validate_experiment(exp)
                except Exception as exc:  # pragma: no cover - validate never raises
                    warnings = [f"not a record: validation failed ({exc})"]
                if warnings:
                    warnings_by_experiment[label] = warnings
            else:
                n_records += 0
                warnings_by_experiment[f"record-{idx}"] = ["not a record: expected a dict experiment record"]
        n_warnings = sum(len(v) for v in warnings_by_experiment.values())
        return {
            "n_experiments": len(experiments),
            "n_records": n_records,
            "n_warnings": n_warnings,
            "warnings_by_experiment": warnings_by_experiment,
            "assay_types": sorted(assay_set),
        }
    except Exception:
        return {
            "n_experiments": 0,
            "n_records": 0,
            "n_warnings": 1,
            "warnings_by_experiment": {"unknown": ["not a record: corpus validation failed"]},
            "assay_types": [],
        }


def schema_summary() -> str:
    """One-paragraph human description of the record model + corpus scope."""
    try:
        data_path = Path(__file__).resolve().parent.parent / "data" / "osdr_experiments.json"
        blob = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            for key in ("experiments", "data", "items"):
                if isinstance(blob.get(key), list):
                    blob = blob[key]
                    break
            else:
                blob = [blob]
        if not isinstance(blob, list) or not blob:
            raise ValueError("empty corpus")
        n_exp = len(blob)
        n_rec = sum(len(e.get("genes", []) or []) for e in blob if isinstance(e, dict))
        assays = sorted({str(e.get("assay", "")).strip() for e in blob if isinstance(e, dict) and str(e.get("assay", "")).strip()})
        assay_phrase = ", ".join(assays) if assays else "microarray and RNA-seq"
        return (
            "SpaceMed experiment records require osdr_id, organism, tissue, mission, "
            "stressors, and a non-empty genes list (each gene with symbol, numeric log2FC, "
            "and up/down direction), while optional forward-compatible fields (study_version, "
            "assay, duration_days, pathways, pubmed_refs, osdr_url, provenance, ingestion_date, "
            "processing_status, plus gene-level p_value, pathway, and go_terms) allow future OSDR "
            f"studies to be added without breaking validation; the current verified demo dataset holds "
            f"{n_exp} curated NASA experiments ({n_rec} gene records; assays: {assay_phrase}), and the "
            "expanded-dataset design accepts additional studies carrying any of those optional fields."
        )
    except Exception:
        return (
            "SpaceMed experiment records require osdr_id, organism, tissue, mission, "
            "stressors, and a non-empty genes list (each gene with symbol, numeric log2FC, "
            "and up/down direction), while optional forward-compatible fields (study_version, "
            "assay, duration_days, pathways, pubmed_refs, osdr_url, provenance, ingestion_date, "
            "processing_status, plus gene-level p_value, pathway, and go_terms) allow future OSDR "
            "studies to be added without breaking validation; the current verified demo dataset holds "
            "30 curated NASA experiments, and the expanded-dataset design accepts additional studies "
            "carrying any of those optional fields."
        )
