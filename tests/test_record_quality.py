"""Record-quality phase-1 tests — synthetic fixtures only.

Data hygiene (data-contract §6): no real store row ever enters this file —
every record below is constructed in-code. The core suite (weight map,
clamping, scope gate, aggregation) runs without the engine; integration
tests against the pinned ``vesma.canon_validate`` skip honestly when the
engine checkout is absent (CI).
"""

from __future__ import annotations

import os
import sys
import tomllib
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from cortex.quality import (
    QUALITY_WEIGHTS_HUNDREDTHS,
    SCORER_REVISION,
    CanonValidator,
    EngineLoaderError,
    QualityAggregate,
    RecordQuality,
    UnknownViolationCode,
    aggregate,
    canon_scope,
    load_canon_validator,
    score_record,
    score_violations,
)

#: Pinned engine checkout (release/6.6.0, package ``vesma``); override via
#: the env var. Missing tree → integration tests skip honestly.
ENGINE_SRC_ENV = "VESMA_ENGINE_SRC"
_DEFAULT_ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")


def _engine_src() -> Path | None:
    raw = os.environ.get(ENGINE_SRC_ENV) or str(_DEFAULT_ENGINE_SRC)
    candidate = Path(raw)
    return candidate if (candidate / "vesma" / "canon_validate.py").is_file() else None


ENGINE_SRC = _engine_src()
requires_engine = pytest.mark.skipif(
    ENGINE_SRC is None, reason="pinned engine checkout not available — skips honestly"
)


# ── synthetic fixtures ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FakeViolation:
    """Engine CanonViolation shape (code/rule/detail) without the engine."""

    code: str
    rule: str = "canon §test"
    detail: str = "synthetic"


def _canon(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "1",
        "type": "checkpoint",
        "status": "active",
        "language": "ru",
        "session_ref": "sess-synth-1",
    }
    base.update(overrides)
    return {"canon": base}


def _task_canon(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "1",
        "type": "task",
        "status": "active",
        "language": "ru",
        "owner_slug": "synthetic",
        "priority": "P2",
        "size": "S",
    }
    base.update(overrides)
    return {"canon": base}


CHECKPOINT_BODY = (
    "## Goals\nG\n## Completed\nC\n## In Progress\nI\n## Decisions\nD\n## Context\nX"
)
TASK_BODY_OK = "## Why\nW\n## Acceptance\nA\n## Out of scope\nO\n## References\nR"
TASK_BODY_NO_REFS = "## Why\nW\n## Acceptance\nA\n## Out of scope\nO"


class RecordingModule:
    """Stub for the engine module — counts calls, emits nothing."""

    __file__ = "/tmp/engine-synth/vesma/canon_validate.py"

    def __init__(self) -> None:
        self.calls = 0

    def validate_canon_record(
        self, *, content: str, title: str | None, metadata: object
    ) -> list[object]:
        self.calls += 1
        return []


def _stub_validator(module: RecordingModule) -> CanonValidator:
    return CanonValidator(Path("/tmp/engine-synth"), "0.0.0-synth", module)


# ── weight map contract ───────────────────────────────────────────────────────


def test_weight_map_completeness_anchor() -> None:
    # One violation of EVERY class = exactly 100 hundredths = score 0.0.
    assert sum(QUALITY_WEIGHTS_HUNDREDTHS.values()) == 100


def test_weight_map_matches_spec_classes() -> None:
    assert set(QUALITY_WEIGHTS_HUNDREDTHS) == {
        "CANON-E-ENVELOPE",
        "CANON-E-LINEAGE",
        "CANON-E-SECTION",
        "CANON-E-STATUS",
        "CANON-E-LANGUAGE",
        "CANON-E-TITLE",
        "CANON-E-DATE",
    }


def test_scorer_revision_is_pinned() -> None:
    assert SCORER_REVISION == "record-quality-phase1/v1"


# ── score_violations: class → weight ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "weight"),
    sorted(QUALITY_WEIGHTS_HUNDREDTHS.items()),
)
def test_single_violation_scores_one_minus_weight(code: str, weight: int) -> None:
    result = score_violations([FakeViolation(code)])
    assert result.in_scope is True
    assert result.penalty_hundredths == weight
    assert result.score == (100 - weight) / 100
    assert result.components == {code: 1}
    assert len(result.violations) == 1
    assert result.violations[0].weight_hundredths == weight


def test_anchor_all_classes_lands_at_zero() -> None:
    violations = [FakeViolation(code) for code in QUALITY_WEIGHTS_HUNDREDTHS]
    result = score_violations(violations)
    assert result.penalty_hundredths == 100
    assert result.score == 0.0


def test_repeated_class_accumulates_linearly() -> None:
    result = score_violations([FakeViolation("CANON-E-DATE")] * 3)
    assert result.penalty_hundredths == 15
    assert result.score == 0.85
    assert result.components == {"CANON-E-DATE": 3}


def test_penalty_clamps_score_at_zero_but_stays_raw() -> None:
    result = score_violations([FakeViolation("CANON-E-DATE")] * 21)  # 105 hundredths
    assert result.penalty_hundredths == 105
    assert result.score == 0.0


def test_unknown_code_refuses_loudly() -> None:
    with pytest.raises(UnknownViolationCode, match="CANON-E-FUTURE"):
        score_violations([FakeViolation("CANON-E-FUTURE")])


def test_determinism_bit_for_bit() -> None:
    violations = [FakeViolation("CANON-E-TITLE"), FakeViolation("CANON-E-DATE")]
    assert score_violations(violations) == score_violations(violations)


# ── scope gate (canon §9) ─────────────────────────────────────────────────────


@pytest.mark.parametrize("metadata", [None, {}, {"canon": "not-a-dict"}, "junk"])
def test_out_of_scope_scores_none(metadata: object) -> None:
    assert canon_scope(metadata) is False


def test_score_record_out_of_scope_never_touches_engine() -> None:
    module = RecordingModule()
    result = score_record("любой текст", None, {}, validator=_stub_validator(module))
    assert module.calls == 0
    assert result.in_scope is False
    assert result.score is None
    assert result.violations == ()


def test_score_record_in_scope_calls_engine_once() -> None:
    module = RecordingModule()
    result = score_record(
        CHECKPOINT_BODY,
        "Короткий заголовок",
        _canon(),
        validator=_stub_validator(module),
    )
    assert module.calls == 1
    assert result.in_scope is True
    assert result.score == 1.0


# ── aggregation: in-scope only, counters only ────────────────────────────────


def test_aggregate_averages_in_scope_only() -> None:
    clean = score_violations([])
    broken = score_violations([FakeViolation("CANON-E-ENVELOPE")])  # 0.75
    out = RecordQuality(in_scope=False, score=None, penalty_hundredths=0)
    stats = aggregate([clean, broken, out])
    assert stats == QualityAggregate(
        total=3,
        in_scope=2,
        out_of_scope=1,
        mean_penalty_hundredths=(0 + 25) / 2,
        score_histogram={"1.00": 1, "0.75": 1},
        code_counts={"CANON-E-ENVELOPE": 1},
    )


def test_aggregate_empty_and_all_out_of_scope() -> None:
    empty = aggregate([])
    assert empty.total == 0
    assert empty.mean_penalty_hundredths is None
    all_out = aggregate(
        [RecordQuality(in_scope=False, score=None, penalty_hundredths=0)]
    )
    assert all_out.in_scope == 0
    assert all_out.mean_penalty_hundredths is None


# ── engine integration (pinned vesma checkout; skips honestly in CI) ─────────


def _engine_validator() -> CanonValidator:
    assert ENGINE_SRC is not None
    return load_canon_validator(ENGINE_SRC)


@requires_engine
def test_engine_clean_checkpoint_scores_one() -> None:
    validator = _engine_validator()
    result = score_record(
        CHECKPOINT_BODY, "Короткий заголовок", _canon(), validator=validator
    )
    assert result.score == 1.0
    assert result.violations == ()


@requires_engine
def test_engine_task_sections_intact_scores_one() -> None:
    validator = _engine_validator()
    result = score_record(TASK_BODY_OK, "Заголовок", _task_canon(), validator=validator)
    assert result.score == 1.0


@requires_engine
@pytest.mark.parametrize(
    ("metadata", "title", "body", "expected_code"),
    [
        (_canon(oops_field=1), "Заголовок", CHECKPOINT_BODY, "CANON-E-ENVELOPE"),
        (_task_canon(), "Заголовок", TASK_BODY_NO_REFS, "CANON-E-SECTION"),
        (_canon(status="resolved"), "Заголовок", CHECKPOINT_BODY, "CANON-E-STATUS"),
        (_canon(language="fr"), "Заголовок", CHECKPOINT_BODY, "CANON-E-LANGUAGE"),
        (_canon(), "х" * 81, CHECKPOINT_BODY, "CANON-E-TITLE"),
        (
            _canon(),
            "Заголовок",
            CHECKPOINT_BODY + "\n отчёт от 27.09.2026",
            "CANON-E-DATE",
        ),
        (
            _canon(lineage_marks=[{"kind": "split-from", "at": "не-дата"}]),
            "Заголовок",
            CHECKPOINT_BODY,
            "CANON-E-LINEAGE",
        ),
    ],
)
def test_engine_class_emits_expected_code(
    metadata: dict, title: str, body: str, expected_code: str
) -> None:
    validator = _engine_validator()
    result = score_record(body, title, metadata, validator=validator)
    codes = [v.code for v in result.violations]
    assert codes == [expected_code], f"expected only {expected_code}, got {codes}"
    weight = QUALITY_WEIGHTS_HUNDREDTHS[expected_code]
    assert result.score == (100 - weight) / 100


@requires_engine
def test_engine_scope_mirror_agrees() -> None:
    validator = _engine_validator()
    garbage_body = CHECKPOINT_BODY + "\n вчера и 27.09"
    legacy_metadata = {"tags": ["x"]}
    assert canon_scope(legacy_metadata) is False
    assert validator.validate(garbage_body, None, legacy_metadata) == []
    assert canon_scope(_canon()) is True


@requires_engine
def test_engine_pycache_hygiene() -> None:
    assert ENGINE_SRC is not None
    cache = ENGINE_SRC / "vesma" / "__pycache__"
    before = sorted(p.name for p in cache.iterdir()) if cache.is_dir() else []
    validator = _engine_validator()
    score_record(CHECKPOINT_BODY, None, _canon(), validator=validator)
    after = sorted(p.name for p in cache.iterdir()) if cache.is_dir() else []
    assert after == before, "engine checkout must never grow __pycache__"


@requires_engine
def test_engine_provenance_pins_checkout() -> None:
    assert ENGINE_SRC is not None
    validator = _engine_validator()
    engine_root = ENGINE_SRC if isinstance(ENGINE_SRC, Path) else Path(ENGINE_SRC)
    with (engine_root.parent / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]
    assert validator.provenance["scorer_revision"] == SCORER_REVISION
    assert validator.provenance["engine_package"] == "vesma"
    assert validator.provenance["engine_version"] == expected_version
    assert validator.provenance["engine_src"] == str(engine_root.resolve())


@requires_engine
def test_loader_identity_guard_fails_on_foreign_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ENGINE_SRC is not None
    foreign = Path("/tmp/foreign-engine/vesma/canon_validate.py")

    class _Foreign:
        __file__ = str(foreign)

    # Pre-seed BOTH import keys: the parent must lack the submodule
    # attribute so IMPORT_FROM falls back to sys.modules, where the
    # foreign module sits. The loader must refuse it — the identity of
    # the imported validator is the provenance foundation.
    monkeypatch.setitem(sys.modules, "vesma", types.ModuleType("vesma"))
    monkeypatch.setitem(sys.modules, "vesma.canon_validate", _Foreign())
    with pytest.raises(EngineLoaderError, match="not the pinned tree"):
        load_canon_validator(ENGINE_SRC)


@requires_engine
def test_loader_refuses_tree_without_validator(tmp_path: Path) -> None:
    with pytest.raises(EngineLoaderError, match="no vesma/canon_validate.py"):
        load_canon_validator(tmp_path)
