"""Layer A runner — frozen eval-set scoring + gate evaluation.

One command over one bundle: loads a frozen eval set (re-verifying
``eval_set_sha256`` — a mismatch voids the run), scores EVERY probe
through ``cortex.features.pair.features`` + the artifact graph, and
emits the report: BA at the runner's scoring cut, per-class breakdown,
mechanical gate evaluations, verdict by role.

Gate semantics (eval-methodology §4; values live in the frozen
``gate_contract.json`` — this module only CONSUMES it):

- role ``merge``   → bundle-contract invariants + invariant probe classes
  (self-pair floor, ladder monotonicity, pair symmetry) block;
- role ``release`` → additionally the corridor classes (near-identity
  floor, far-negative ceiling). The BA prev_adopt corridor is LA-3 CI
  wiring (needs the previous ADOPT BA) — the runner REPORTS BA, the CI
  gate compares.

Reusable-by-design: unlike the Layer B holdout there is NO single-shot
run-log here — Layer A sets run on every PR (eval-methodology §3);
their tamper-evidence is the frozen fingerprint, not burn-once.

Scoring reuses the sanity suite's session/scoring path (single source —
the same code that produces the #480 suite numbers, so suite and set
results stay comparable); the FEATURE math is and stays
``cortex.features.pair.features`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from cortex.eval.sanity import (
    MONOTONICITY_TOLERANCE,
    NEAR_BOUNDARY_MIN,
    SELF_PAIR_MIN,
    UNRELATED_MAX,
    SanityCheck,
)

# The bundle load/score path is deliberately REUSED from the sanity suite
# (private on purpose there): one scoring implementation keeps the suite's
# numbers and the eval-set numbers byte-comparable — duplicating it would
# recreate the exact local-copy drift the #480 lesson warns about. The
# feature math itself is public contract: cortex.features.pair.features.
from cortex.eval.sanity import (
    _contract_checks as _sanity_contract_checks,
)
from cortex.eval.sanity import (
    _load_manifest as _sanity_load_manifest,
)
from cortex.eval.sanity import (
    _open_session as _sanity_open_session,
)
from cortex.eval.sanity import (
    _resolve_bundle as _sanity_resolve_bundle,
)
from cortex.eval.sanity import (
    _score_pair as _sanity_score_pair,
)
from cortex.eval.runner import DUPLICATE_THRESHOLD_PROBABILITY
from cortex.evalsets.generate import (
    LABEL_DUPLICATE,
    LABEL_NOT_DUPLICATE,
    EvalProbe,
    EvalSet,
    EvalSetError,
)
from cortex.evalsets.taxonomy import (
    GATE_CORRIDOR,
    GATE_INVARIANT,
    PROBE_CLASS_FAR_NEGATIVE,
    PROBE_CLASS_IDENTITY_SELF,
    PROBE_CLASS_MONOTONICITY,
    PROBE_CLASS_NEAR_IDENTITY,
    PROBE_CLASS_PAIR_SYMMETRY,
)
from cortex.features.pair import features

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ClassBreakdown",
    "GateEvaluation",
    "LayerAReport",
    "evaluate_gates",
    "report_to_dict",
    "run_layer_a",
]

#: Report schema stamp (machine-readable JSON surface).
REPORT_SCHEMA_VERSION: Final[int] = 1

_LADDER_LABEL_DUPLICATE_MIN_COS: Final[float] = 0.95


# ── report types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassBreakdown:
    """Per-probe-class confusion + mean probability (informational)."""

    probe_class: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float | None  # None when the class has no positives
    specificity: float | None  # None when the class has no negatives
    mean_probability: float


@dataclass(frozen=True)
class GateEvaluation:
    """One mechanical gate evaluation over a probe class."""

    name: str
    gate_kind: str  # GATE_INVARIANT | GATE_CORRIDOR
    probe_class: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LayerAReport:
    """Full Layer A result for one (bundle, frozen eval set) run."""

    set_id: str
    role: str
    eval_set_sha256: str
    weights_sha256: str
    bundle: str
    threshold: float
    n_pairs: int
    balanced_accuracy: float
    per_class: tuple[ClassBreakdown, ...]
    gate_evaluations: tuple[GateEvaluation, ...]
    bundle_contract: tuple[SanityCheck, ...]
    passed: bool
    run_at: str

    @property
    def failed_gates(self) -> tuple[GateEvaluation, ...]:
        return tuple(gate for gate in self.gate_evaluations if not gate.passed)


def report_to_dict(report: LayerAReport) -> dict:
    """Machine-readable JSON payload (stdout of the runner script)."""
    payload = asdict(report)
    return payload


# ── scoring ───────────────────────────────────────────────────────────────────


def _score_eval_set(session, eval_set: EvalSet) -> list[float]:
    """P(dup) for every probe — features through the frozen contract only."""
    probabilities: list[float] = []
    for probe in eval_set.probes:
        vector = features(probe.record, probe.candidate, probe.similarity)
        probabilities.append(_sanity_score_pair(session, vector))
    return probabilities


def _class_breakdowns(
    probes: Sequence[EvalProbe], probabilities: Sequence[float]
) -> tuple[ClassBreakdown, ...]:
    by_class: dict[str, list[tuple[int, float]]] = {}
    for probe, probability in zip(probes, probabilities):
        by_class.setdefault(probe.probe_class, []).append(
            (1 if probe.label == LABEL_DUPLICATE else 0, probability)
        )
    breakdowns = []
    for probe_class in sorted(by_class):
        pairs = by_class[probe_class]
        tp = sum(1 for y, p in pairs if y == 1 and p >= DUPLICATE_THRESHOLD_PROBABILITY)
        fn = sum(1 for y, p in pairs if y == 1 and p < DUPLICATE_THRESHOLD_PROBABILITY)
        tn = sum(1 for y, p in pairs if y == 0 and p < DUPLICATE_THRESHOLD_PROBABILITY)
        fp = sum(1 for y, p in pairs if y == 0 and p >= DUPLICATE_THRESHOLD_PROBABILITY)
        positives = tp + fn
        negatives = tn + fp
        breakdowns.append(
            ClassBreakdown(
                probe_class=probe_class,
                n=len(pairs),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                sensitivity=(tp / positives) if positives else None,
                specificity=(tn / negatives) if negatives else None,
                mean_probability=sum(p for _, p in pairs) / len(pairs),
            )
        )
    return tuple(breakdowns)


# ── mechanical gate evaluations (pure — unit-testable without ONNX) ───────────


def evaluate_gates(
    probes: Sequence[EvalProbe], probabilities: Sequence[float]
) -> tuple[GateEvaluation, ...]:
    """Evaluate the invariant/corridor probe classes over scored probes.

    Pure: takes the probe rows and their P(dup) values, returns the gate
    verdicts. A gated class with ZERO probes in the set is a FAIL (an
    eval set structurally blind in a gated class defeats its purpose).
    """
    by_class: dict[str, list[float]] = {}
    for probe, probability in zip(probes, probabilities):
        by_class.setdefault(probe.probe_class, []).append(probability)

    def _floor(class_name: str, floor: float) -> GateEvaluation:
        values = by_class.get(class_name, [])
        if not values:
            return GateEvaluation(
                name=f"{class_name}_floor",
                gate_kind=GATE_INVARIANT,
                probe_class=class_name,
                passed=False,
                detail="no probes of this class in the set — structural blindness",
            )
        minimum = min(values)
        failures = sum(1 for value in values if value < floor)
        return GateEvaluation(
            name=f"{class_name}_floor",
            gate_kind=GATE_INVARIANT,
            probe_class=class_name,
            passed=minimum >= floor,
            detail=(
                f"P≥{floor} on {len(values) - failures}/{len(values)} probes "
                f"(min {minimum:.4f})"
            ),
        )

    def _ceiling(class_name: str, ceiling: float) -> GateEvaluation:
        values = by_class.get(class_name, [])
        if not values:
            return GateEvaluation(
                name=f"{class_name}_ceiling",
                gate_kind=GATE_CORRIDOR,
                probe_class=class_name,
                passed=False,
                detail="no probes of this class in the set — structural blindness",
            )
        maximum = max(values)
        failures = sum(1 for value in values if value >= ceiling)
        return GateEvaluation(
            name=f"{class_name}_ceiling",
            gate_kind=GATE_CORRIDOR,
            probe_class=class_name,
            passed=maximum < ceiling,
            detail=(
                f"P<{ceiling} on {len(values) - failures}/{len(values)} probes "
                f"(max {maximum:.4f})"
            ),
        )

    def _grouped(
        class_name: str,
        group_map: dict[str, list[tuple[float, float]]],
        check,
    ) -> GateEvaluation:
        if not group_map:
            return GateEvaluation(
                name=class_name,
                gate_kind=GATE_INVARIANT,
                probe_class=class_name,
                passed=False,
                detail="no probes of this class in the set — structural blindness",
            )
        details = []
        passed = True
        for group in sorted(group_map):
            # Sort by the ORDER key, never by probability (which would
            # reorder the sequence under test — caught by the prod-B1
            # smoke run).
            ordered = sorted(group_map[group])
            ok, detail = check(ordered)
            passed = passed and ok
            details.append(f"{group}: {detail}")
        return GateEvaluation(
            name=class_name,
            gate_kind=GATE_INVARIANT,
            probe_class=class_name,
            passed=passed,
            detail="; ".join(details),
        )

    # Ladder groups order by the ASSIGNED cosine DESCENDING — the
    # protocol ladder (COS_LADDER: 1.0 → 0.5) and the monotonicity rule
    # "P non-increasing as cosine drops". Symmetry groups order fwd
    # before rev.
    ladders: dict[str, list[tuple[float, float]]] = {}
    symmetries: dict[str, list[tuple[float, float]]] = {}
    for probe, probability in zip(probes, probabilities):
        if probe.group is None:
            continue
        if probe.probe_class == PROBE_CLASS_MONOTONICITY:
            ladders.setdefault(probe.group, []).append((-probe.similarity, probability))
        elif probe.probe_class == PROBE_CLASS_PAIR_SYMMETRY:
            order = 0.0 if probe.pair_id.endswith("-fwd") else 1.0
            symmetries.setdefault(probe.group, []).append((order, probability))

    def _monotone(pairs: Sequence[tuple[float, float]]) -> tuple[bool, str]:
        values = [p for _, p in pairs]
        ok = all(
            values[i] >= values[i + 1] - MONOTONICITY_TOLERANCE
            for i in range(len(values) - 1)
        )
        text = ", ".join(f"cos={-key:.2f}:{p:.4f}" for key, p in pairs)
        return ok, ("non-increasing " if ok else "INCREASING ") + f"[{text}]"

    def _symmetric(pairs: Sequence[tuple[float, float]]) -> tuple[bool, str]:
        values = [p for _, p in pairs]
        delta = abs(values[0] - values[1])
        ok = delta <= MONOTONICITY_TOLERANCE
        return ok, f"|P(x,y)−P(y,x)|={delta:.2e} (tol {MONOTONICITY_TOLERANCE})"

    ladder_gate = replace(
        _grouped(PROBE_CLASS_MONOTONICITY, ladders, _monotone), name="monotonicity"
    )
    symmetry_gate = replace(
        _grouped(PROBE_CLASS_PAIR_SYMMETRY, symmetries, _symmetric),
        name="pair_symmetry",
    )

    return (
        _floor(PROBE_CLASS_IDENTITY_SELF, SELF_PAIR_MIN),
        ladder_gate,
        symmetry_gate,
        _floor(PROBE_CLASS_NEAR_IDENTITY, NEAR_BOUNDARY_MIN),
        _ceiling(PROBE_CLASS_FAR_NEGATIVE, UNRELATED_MAX),
    )


# ── the run ───────────────────────────────────────────────────────────────────


def run_layer_a(bundle: Path, eval_set: EvalSet) -> LayerAReport:
    """Run the frozen eval set against one bundle (dir or .onnx path).

    Raises :class:`SanityLoadError` when the bundle cannot be loaded and
    :class:`EvalSetError` when the set is structurally invalid (the set
    itself is verified at load time — see ``load_eval_set``).
    """
    if not eval_set.probes:
        raise EvalSetError("eval set is empty")
    labels = {probe.label for probe in eval_set.probes}
    if labels != {LABEL_DUPLICATE, LABEL_NOT_DUPLICATE}:
        raise EvalSetError(
            f"eval set {eval_set.set_id!r} must carry both label classes "
            f"(BA undefined), got {sorted(labels)}"
        )

    model_path, manifest_path = _sanity_resolve_bundle(Path(bundle))
    manifest = _sanity_load_manifest(manifest_path)
    from cortex.artifacts import sha256_file

    weights_sha = sha256_file(model_path) if model_path.is_file() else ""
    session = _sanity_open_session(model_path)
    meta = dict(session.get_modelmeta().custom_metadata_map)

    bundle_contract = _sanity_contract_checks(manifest, meta, weights_sha)
    probabilities = _score_eval_set(session, eval_set)
    gates = evaluate_gates(eval_set.probes, probabilities)

    y_true = [1 if p.label == LABEL_DUPLICATE else 0 for p in eval_set.probes]
    y_pred = [1 if p >= DUPLICATE_THRESHOLD_PROBABILITY else 0 for p in probabilities]
    from sklearn.metrics import balanced_accuracy_score

    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    invariants_ok = all(
        gate.passed for gate in gates if gate.gate_kind == GATE_INVARIANT
    )
    corridors_ok = all(gate.passed for gate in gates if gate.gate_kind == GATE_CORRIDOR)
    contract_ok = all(check.passed for check in bundle_contract)
    passed = (
        contract_ok
        and invariants_ok
        and (corridors_ok if eval_set.role == "release" else True)
    )

    return LayerAReport(
        set_id=eval_set.set_id,
        role=eval_set.role,
        eval_set_sha256=eval_set.eval_set_sha256,
        weights_sha256=weights_sha,
        bundle=str(model_path),
        threshold=DUPLICATE_THRESHOLD_PROBABILITY,
        n_pairs=len(eval_set.probes),
        balanced_accuracy=balanced_accuracy,
        per_class=_class_breakdowns(eval_set.probes, probabilities),
        gate_evaluations=gates,
        bundle_contract=bundle_contract,
        passed=passed,
        run_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
