"""Layer A eval sets — frozen probe sets, generators, runner (wave LA-1).

Implements the Layer A surface of docs/specs/eval-methodology.md: the §7
probe-class taxonomy (:mod:`cortex.evalsets.taxonomy`), deterministic
procedural generators + frozen-set fingerprinting/loading
(:mod:`cortex.evalsets.generate`), the probe topics disjoint from the
train seed list (:mod:`cortex.evalsets.topics`), and the set runner
(:mod:`cortex.evalsets.runner`).

Purity: no store, no network, no vectors in the committed sets — every
scoring pass computes features through ``cortex.features.pair.features``.
"""

from __future__ import annotations

from cortex.evalsets.generate import (
    DEFAULT_SEED,
    MERGE_V1,
    RELEASE_V1,
    EvalProbe,
    EvalSet,
    EvalSetError,
    EvalSetRecipe,
    build_eval_set,
    eval_set_jsonl,
    eval_set_manifest_bytes,
    eval_set_sha256,
    load_eval_set,
    load_llm_slots,
    per_class_counts,
    probe_row,
)
from cortex.evalsets.runner import (
    ClassBreakdown,
    GateEvaluation,
    LayerAReport,
    evaluate_gates,
    report_to_dict,
    run_layer_a,
)
from cortex.evalsets.taxonomy import (
    ALL_PROBE_CLASSES,
    CANONICAL_PROBE_CLASSES,
    COVERAGE_MATRIX,
    LLM_SLOT_PROBE_CLASSES,
    ProbeClassSpec,
)

__all__ = [
    "ALL_PROBE_CLASSES",
    "CANONICAL_PROBE_CLASSES",
    "COVERAGE_MATRIX",
    "DEFAULT_SEED",
    "LLM_SLOT_PROBE_CLASSES",
    "MERGE_V1",
    "RELEASE_V1",
    "ClassBreakdown",
    "EvalProbe",
    "EvalSet",
    "EvalSetError",
    "EvalSetRecipe",
    "GateEvaluation",
    "LayerAReport",
    "ProbeClassSpec",
    "build_eval_set",
    "eval_set_jsonl",
    "eval_set_manifest_bytes",
    "eval_set_sha256",
    "evaluate_gates",
    "load_eval_set",
    "load_llm_slots",
    "per_class_counts",
    "probe_row",
    "report_to_dict",
    "run_layer_a",
]
