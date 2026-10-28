"""Record-quality package — the deterministic phase-1 scorer lives in
scorer.py (spec: docs/specs/record-quality-phase1.md; ADR 0002 duty #2,
phase 1 = code over ``canon_validate``). The ML phase 2 (conditional, a
merge with duty #8) would live BESIDE this module — feature core shared
as code, never as weights.
"""

from cortex.quality.scorer import (
    QUALITY_WEIGHTS_HUNDREDTHS,
    SCORER_REVISION,
    CanonValidator,
    EngineLoaderError,
    QualityAggregate,
    QualityError,
    QualityViolation,
    RecordQuality,
    UnknownViolationCode,
    aggregate,
    canon_scope,
    load_canon_validator,
    score_record,
    score_violations,
)

__all__ = [
    "QUALITY_WEIGHTS_HUNDREDTHS",
    "SCORER_REVISION",
    "CanonValidator",
    "EngineLoaderError",
    "QualityAggregate",
    "QualityError",
    "QualityViolation",
    "RecordQuality",
    "UnknownViolationCode",
    "aggregate",
    "canon_scope",
    "load_canon_validator",
    "score_record",
    "score_violations",
]
