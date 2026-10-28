"""Record-quality scorer, phase 1 — deterministic, over canon_validate.

Spec: docs/specs/record-quality-phase1.md (this revision's frozen weight
map is ``QUALITY_WEIGHTS_HUNDREDTHS``, the revision string is
``SCORER_REVISION``). ADR 0002 duty #2, phase 1 = CODE: the engine's
``canon_validate`` already formalizes canon-discipline violations (the
stable CANON-E-* warn codes, canon §9); this module turns that output
into a typed score. No ML artifact, no thresholds, no gates — the Score
is an informational metric.

Formula (integer hundredths, one division per record — bit-for-bit
reproducible on any IEEE-754 platform):

    score = max(0, 100 − Σ_class w_class · n_class) / 100

Scope discipline (canon §9 transitional rule): a record WITHOUT
``metadata.canon`` is out of canon scope — it scores ``None`` (NO-DATA),
never a perfect 1.0. Aggregation averages in-scope records only and
reports the in-scope share separately.

Engine loading (the pinned engine checkout is read-only property):

- ``vesma.canon_validate`` is imported from a caller-provided engine
  ``src/`` tree (the engine package is named ``vesma`` since 6.4.0);
  the tree's import entry is added for the import window and REMOVED
  afterwards — the caller's ``sys.path`` is restored exactly;
- the window entry sits at ``sys.path[0]``, not appended: a user-site
  installed ``vesma`` distribution (found in the wild: 6.4.0 CLI install)
  shadows an APPEND and would silently answer for the pinned checkout —
  path order alone is not trusted, the post-import identity assert below
  is the real guarantee;
- during the import ``sys.dont_write_bytecode`` is forced — the engine
  checkout must never grow a ``__pycache__``;
- after the import the loaded module's identity is asserted
  (``canon_validate.__file__`` under the pinned root) — a stray cached or
  installed ``vesma`` package fails loudly instead of scoring silently
  wrong;
- an unknown violation code (the engine added a class, the frozen weight
  map was not extended deliberately) raises :class:`UnknownViolationCode`
  — the same "typo must fail loudly" discipline as the canon envelope's
  ``unevaluatedProperties: false``.
"""

from __future__ import annotations

import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

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

#: Revision of the frozen weight map — part of every result's provenance.
#: Changing the weights = a new revision string + a spec commit BEFORE the
#: next statistics run (preregistration discipline, eval-methodology §6).
SCORER_REVISION: Final[str] = "record-quality-phase1/v1"

#: Violation class → penalty weight, in HUNDREDTHS of a point. The keys
#: are the stable CANON-E-* warn codes of ``vesma.canon_validate`` (canon
#: §9) — the only contract surface; ``detail`` strings are human prose and
#: never parsed. The weights sum to exactly 100: one violation of EVERY
#: class lands the score at 0.0 (the completeness anchor of the formula).
QUALITY_WEIGHTS_HUNDREDTHS: Final[Mapping[str, int]] = {
    "CANON-E-ENVELOPE": 25,  # canon §2 — schema-invalid envelope (structural)
    "CANON-E-LINEAGE": 25,  # ADR-0037 Д2 — invalid lineage_marks (structural)
    "CANON-E-SECTION": 15,  # canon §3 — missing required body sections
    "CANON-E-STATUS": 10,  # canon §2/§7 — bad status enum / status x type
    "CANON-E-LANGUAGE": 10,  # canon §2/§6 — language outside ru/en
    "CANON-E-TITLE": 10,  # canon §3 — empty / >80 chars / multi-line title
    "CANON-E-DATE": 5,  # canon §5 — non-ISO date in body (narrow checker)
}

assert sum(QUALITY_WEIGHTS_HUNDREDTHS.values()) == 100, "completeness anchor"


class QualityError(ValueError):
    """Base for record-quality scoring failures."""


class EngineLoaderError(QualityError):
    """The pinned engine tree is missing/incomplete or imported wrongly."""


class UnknownViolationCode(QualityError):
    """A violation carries a code outside the frozen weight map.

    Raised, never silently zero-weighted: an engine that added a class
    must be met with a deliberate map extension (new scorer revision),
    not with scores that quietly drift.
    """

    def __init__(self, code: object, known: list[str]) -> None:
        super().__init__(
            f"violation code {code!r} is not in the frozen weight map "
            f"(scorer revision {SCORER_REVISION}); known codes: {known}"
        )
        self.code = code


@dataclass(frozen=True)
class QualityViolation:
    """One canon violation, normalized with its frozen class weight."""

    code: str
    rule: str
    detail: str
    weight_hundredths: int


@dataclass(frozen=True)
class RecordQuality:
    """Typed result for ONE record.

    ``score`` is ``None`` exactly when ``in_scope`` is false (out of canon
    scope — NO-DATA discipline). ``components`` maps violation code →
    count. ``provenance`` carries the scorer revision and, when the engine
    adapter was used, the engine identity it scored with.
    """

    in_scope: bool
    score: float | None
    penalty_hundredths: int
    violations: tuple[QualityViolation, ...] = field(default_factory=tuple)
    components: Mapping[str, int] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)


def canon_scope(metadata: object) -> bool:
    """True when the record carries a canon envelope (canon §9 scope).

    Mirrors the engine's own scope gate in ``validate_canon_record``: a
    Mapping metadata holding a dict ``metadata["canon"]`` is in scope;
    everything else is pre-canon legacy — never validated, never scored.
    Pinned against the engine by an integration test when the engine tree
    is available.
    """
    return isinstance(metadata, Mapping) and isinstance(metadata.get("canon"), dict)


def score_violations(
    violations: Iterable[Any], *, provenance: Mapping[str, str] | None = None
) -> RecordQuality:
    """Score a validator's violation list (the record is IN SCOPE by
    construction).

    Accepts anything shaped like the engine's ``CanonViolation`` (objects
    with ``code``/``rule``/``detail`` attributes). An unknown code raises
    :class:`UnknownViolationCode` — never silently zero-weighted.
    """
    parts: list[QualityViolation] = []
    counts: Counter[str] = Counter()
    penalty = 0
    for violation in violations:
        code = getattr(violation, "code", None)
        weight = QUALITY_WEIGHTS_HUNDREDTHS.get(code)  # type: ignore[arg-type]
        if weight is None:
            raise UnknownViolationCode(code, sorted(QUALITY_WEIGHTS_HUNDREDTHS))
        penalty += weight
        counts[code] += 1
        parts.append(
            QualityViolation(
                code=code,
                rule=str(getattr(violation, "rule", "")),
                detail=str(getattr(violation, "detail", "")),
                weight_hundredths=weight,
            )
        )
    score = max(0, 100 - penalty) / 100
    return RecordQuality(
        in_scope=True,
        score=score,
        penalty_hundredths=penalty,
        violations=tuple(parts),
        components=dict(counts),
        provenance=dict(provenance or {}),
    )


class CanonValidator:
    """Adapter over a pinned engine's ``vesma.canon_validate``.

    Construct via :func:`load_canon_validator` — it enforces the tree
    checks, the no-bytecode discipline and the module identity assert.
    """

    def __init__(self, engine_src: Path, engine_version: str, module: Any) -> None:
        self._engine_src = engine_src
        self._engine_version = engine_version
        self._module = module
        self.provenance: Final[Mapping[str, str]] = {
            "scorer_revision": SCORER_REVISION,
            "engine_package": "vesma",
            "engine_version": engine_version,
            "engine_src": str(engine_src),
            "validator_module": str(module.__file__),
        }

    def validate(
        self, content: str, title: str | None, metadata: Mapping[str, Any]
    ) -> list[Any]:
        """Run the engine's ``validate_canon_record`` — violations out,
        never raises (the engine's own contract)."""
        return self._module.validate_canon_record(
            content=content, title=title, metadata=metadata
        )


def load_canon_validator(engine_src: str | Path) -> CanonValidator:
    """Load ``vesma.canon_validate`` from a pinned engine checkout.

    Import window discipline: the engine ``src/`` root is placed at
    ``sys.path[0]`` for the import and removed afterwards (an APPEND loses
    to a user-site installed ``vesma`` — see the module docstring);
    ``sys.dont_write_bytecode`` is forced for the import so the read-only
    engine tree never grows ``__pycache__``; the post-import identity
    assert on ``canon_validate.__file__`` refuses any module that is not
    the pinned tree's.
    """
    root = Path(engine_src).expanduser().resolve()
    entry = root / "vesma" / "canon_validate.py"
    if not entry.is_file():
        raise EngineLoaderError(
            f"engine tree has no vesma/canon_validate.py under {root}"
        )

    path_entry = str(root)
    previous_no_bytecode = sys.dont_write_bytecode
    already_on_path = path_entry in sys.path
    sys.dont_write_bytecode = True
    try:
        if not already_on_path:
            sys.path.insert(0, path_entry)
        import vesma.canon_validate as module
    except ImportError as exc:
        raise EngineLoaderError(
            f"cannot import vesma.canon_validate from {root}: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_no_bytecode
        if not already_on_path:
            try:
                sys.path.remove(path_entry)
            except ValueError:
                pass

    module_file_raw = getattr(module, "__file__", None)
    if not isinstance(module_file_raw, str) or not module_file_raw:
        raise EngineLoaderError(
            f"imported vesma.canon_validate without a source file (namespace package?) from {root}"
        )
    module_file = Path(module_file_raw).resolve()
    if not module_file.is_relative_to(root):
        raise EngineLoaderError(
            f"imported vesma.canon_validate from {module_file}, not the pinned tree {root}"
        )
    return CanonValidator(root, _engine_version(root), module)


def _engine_version(root: Path) -> str:
    """Engine package version from the checkout's pyproject.toml.

    Read from the PINNED TREE, not importlib.metadata: an unrelated
    installed ``vesma`` distribution must never answer for this checkout.
    """
    pyproject = root.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        return str(data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def score_record(
    content: str,
    title: str | None,
    metadata: Mapping[str, Any],
    *,
    validator: CanonValidator,
) -> RecordQuality:
    """Validate ONE record with the engine and score it.

    Out-of-scope records (no ``metadata.canon``, canon §9) return
    ``score=None`` without touching the validator.
    """
    if not canon_scope(metadata):
        return RecordQuality(
            in_scope=False,
            score=None,
            penalty_hundredths=0,
            violations=(),
            components={},
            provenance=validator.provenance,
        )
    return score_violations(
        validator.validate(content, title, metadata), provenance=validator.provenance
    )


@dataclass(frozen=True)
class QualityAggregate:
    """Distribution over a batch of :class:`RecordQuality` results.

    ``mean_penalty_hundredths`` averages the IN-SCOPE records' penalty
    (None when no in-scope record exists); the mean SCORE is
    ``1 − mean_penalty_hundredths / 100``. ``score_histogram`` is the
    exact-value histogram ("%.2f" → count) — threshold-free by
    construction.
    """

    total: int
    in_scope: int
    out_of_scope: int
    mean_penalty_hundredths: float | None
    score_histogram: Mapping[str, int]
    code_counts: Mapping[str, int]


def aggregate(qualities: Iterable[RecordQuality]) -> QualityAggregate:
    """Aggregate batch results — counters only, no content (data-contract
    §6 hygiene: distributions and counts, never record text)."""
    total = 0
    in_scope = 0
    penalty_sum = 0
    histogram: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    for quality in qualities:
        total += 1
        if not quality.in_scope:
            continue
        in_scope += 1
        penalty_sum += quality.penalty_hundredths
        assert quality.score is not None
        histogram[f"{quality.score:.2f}"] += 1
        codes.update(quality.components)
    mean = penalty_sum / in_scope if in_scope else None
    return QualityAggregate(
        total=total,
        in_scope=in_scope,
        out_of_scope=total - in_scope,
        mean_penalty_hundredths=mean,
        score_histogram=dict(histogram),
        code_counts=dict(codes),
    )
