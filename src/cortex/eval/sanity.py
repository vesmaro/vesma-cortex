"""Adversarial sanity suite — the mandatory pre-ADOPT instrument (#480).

Why it exists: B2 (weights ``71f0572d…``) scored sens 0.989 on the
dataset-v3 holdout while the SHIPPED artifact graded a record against
ITSELF at P(dup)=0.0108 — an inversion (must be ~1.0). The eval surface
contained NO exact self-pairs, so the holdout metrics were blind to the
defect (vesmaro/vesma#480). Root suspicion: feature-column-order desync
between the export and the frozen 13-feature contract.

Every ADOPT claim must pass the four adversarial checks BELOW, before any
other metric is quoted (docs/experiments/calibration-b2-edge-neighborhood.md,
«adversarial sanity suite» section; roadmap-v2.md §5 checklist):

(a) ``self_pair``      P(dup | record vs itself, cos=1.0) ≥ 0.9
(b) ``near_boundary``  P(dup | light perturbation, cos≈0.99) ≥ 0.5 (cut)
(c) ``unrelated``      P(dup | different-topic pair, cos=0.578) < 0.5
(d) ``monotonicity``   over a FIXED pair, P non-increasing as cosine drops
                       along cos ∈ {1.0, 0.99, 0.95, 0.8, 0.5}

The self-pair check is content-independent by construction: record vs
itself pins every feature (n-gram Jaccard/containment = 1.0, deltas = 0,
type/lang match = 1.0, cos = 1.0), so the number is comparable across
probes and reproductions — the #480 measurement (0.0108 through the
provider wrapper) must equal this suite's number for the same weights.

Bundle-contract preconditions run first (integrity before semantics):
sha256(model.onnx) == manifest ``sha256``; manifest and ONNX-metadata
feature order == the frozen :data:`cortex.features.pair.FEATURE_NAMES`;
``feature_set_sha256`` == digest of that order; ``candidate == "d-boost"``
(n-head needs vector sidecars — out of suite scope, the CLI-eval
precedent).

Pure and deterministic: synthetic probe records only — no store, no
network (the repo-wide isolation discipline); same bundle → same report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from cortex.artifacts import CANDIDATE_D, features_digest, sha256_file
from cortex.features.pair import FEATURE_NAMES, FeatureVector, PairRecord, features

__all__ = [
    "SELF_PAIR_MIN",
    "NEAR_BOUNDARY_COS",
    "NEAR_BOUNDARY_MIN",
    "UNRELATED_COS",
    "UNRELATED_MAX",
    "COS_LADDER",
    "MONOTONICITY_TOLERANCE",
    "MODEL_FILENAME",
    "MANIFEST_FILENAME",
    "PROBE_RECORD",
    "UNRELATED_RECORD",
    "SanityCheck",
    "SanityReport",
    "SanityLoadError",
    "run_sanity_suite",
    "ladder_probabilities",
]

# ── thresholds (the D1 protocol requirements, frozen here) ───────────────────

#: (a) self-pair floor: a healthy artifact grades a record against itself
#: as a duplicate with P ≥ 0.9 (inversion = #480 defect class).
SELF_PAIR_MIN: Final[float] = 0.9

#: (b) near-boundary probe cosine.
NEAR_BOUNDARY_COS: Final[float] = 0.99

#: (b) the probability cut (runner's DUPLICATE_THRESHOLD_PROBABILITY).
NEAR_BOUNDARY_MIN: Final[float] = 0.5

#: (c) unrelated-probe cosine — the #480 unrelated-example point.
UNRELATED_COS: Final[float] = 0.578

#: (c) unrelated ceiling.
UNRELATED_MAX: Final[float] = 0.5

#: (d) the monotonicity cosine ladder (fixed, protocol text).
COS_LADDER: Final[tuple[float, ...]] = (1.0, 0.99, 0.95, 0.8, 0.5)

#: (d) float32 graph-noise allowance for the non-increase comparison.
MONOTONICITY_TOLERANCE: Final[float] = 1e-6

MODEL_FILENAME: Final[str] = "model.onnx"
MANIFEST_FILENAME: Final[str] = "manifest.json"


# ── deterministic synthetic probes (no store, no network) ────────────────────

#: The fixed probe record every check is built from. Content is arbitrary
#: but FROZEN here: suite results stay comparable across runs and repos.
PROBE_RECORD: Final[PairRecord] = PairRecord(
    title="Заметка: кэш-инвалидация в веб-приложениях",
    body=(
        "Кэш-инвалидация — одна из двух классических трудностей информатики. "
        "Запись фиксирует стратегию: TTL, версия ключа, точечная инвалидация "
        "по событию. Пример ключа: cart:{user_id} с инкрементом версии."
    ),
    tags=("memory", "engineering", "cache"),
    language="ru",
    record_type="note",
)

#: The unrelated-side record (different topical anchor, disjoint tags).
UNRELATED_RECORD: Final[PairRecord] = PairRecord(
    title="Рецепт борща с пампушками",
    body=(
        "Классический борщ: свёкла, капуста, картофель, морковь, томатная "
        "паста. Пампушки — чесночные булочки, подаются горячими. Варить на "
        "медленном огне около сорока минут."
    ),
    tags=("cooking",),
    language="ru",
    record_type="note",
)


def _perturbed(record: PairRecord) -> PairRecord:
    """Mechanical perturbation of the probe (the W4c-positive twin family,
    same idiom as tests/synth.make_pair_rows: title + ``!``)."""
    return replace(record, title=f"{record.title}!")


# ── report types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SanityCheck:
    """One named check with its verdict and a human-readable detail line."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SanityReport:
    """Full suite verdict for one bundle."""

    bundle: str
    weights_sha256: str
    checks: tuple[SanityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> tuple[SanityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


class SanityLoadError(ValueError):
    """Bundle cannot be loaded/validated (missing files, bad manifest,
    ONNX load failure, contract-violating graph output) — fail loud."""


# ── bundle loading (boundary validation lives here) ──────────────────────────


def _resolve_bundle(bundle: Path) -> tuple[Path, Path]:
    """(model.onnx, manifest) from a bundle DIR or a direct .onnx path.

    Two manifest conventions exist and both are accepted, first existing
    wins: a shipped bundle DIR carries ``manifest.json`` (the engine's
    ``vesmaro/models/<name>/`` layout); a loose exported artifact carries
    a sibling ``<stem>.manifest.json`` (the export-artifact convention).
    """
    bundle = Path(bundle)
    if bundle.is_dir():
        model_path = bundle / MODEL_FILENAME
        candidates = (bundle / MANIFEST_FILENAME, bundle / f"{model_path.stem}.manifest.json")
    elif bundle.is_file():
        model_path = bundle
        candidates = (bundle.with_suffix(".manifest.json"),)
    else:
        raise SanityLoadError(f"bundle path does not exist: {bundle}")
    for candidate in candidates:
        if candidate.is_file():
            return model_path, candidate
    return model_path, candidates[0]


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise SanityLoadError(f"manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SanityLoadError(f"manifest is not valid JSON: {manifest_path} — {exc}") from exc
    if not isinstance(manifest, dict):
        raise SanityLoadError(f"manifest must be a JSON object, got {type(manifest).__name__}")
    for key in ("sha256", "features", "candidate"):
        if key not in manifest:
            raise SanityLoadError(f"manifest missing required key {key!r}: {manifest_path}")
    if not isinstance(manifest["features"], list) or not all(
        isinstance(name, str) for name in manifest["features"]
    ):
        raise SanityLoadError("manifest 'features' must be a list of feature-name strings")
    return manifest


def _open_session(model_path: Path):
    import onnxruntime as ort

    if not model_path.is_file():
        raise SanityLoadError(f"artifact not found: {model_path}")
    try:
        # ORT load failures raise pybind11-state exceptions that are plain
        # Exception subclasses — normalized to the typed boundary error.
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise SanityLoadError(f"artifact load failed — {exc}") from exc


def _score_pair(session, vector: FeatureVector) -> float:
    """One P(dup) through the graph: the frozen contract is a single
    float probability per feature row (d_boost export: float32[K] → [1];
    the [None, K] export variant is tolerated on the input rank)."""
    row = np.asarray(vector.values, dtype=np.float32)
    graph_input = session.get_inputs()[0]
    matrix = row.reshape(1, -1) if len(graph_input.shape) == 2 else row
    try:
        (output,) = session.run(None, {graph_input.name: matrix})
    except Exception as exc:
        raise SanityLoadError(f"artifact inference failed — {exc}") from exc
    values = np.ravel(np.asarray(output))
    if values.shape != (1,):
        raise SanityLoadError(
            f"artifact returned {values.shape} values — expected exactly one probability"
        )
    value = float(values[0])
    if not 0.0 <= value <= 1.0:
        raise SanityLoadError(f"artifact probability {value} outside [0, 1]")
    return value


# ── the suite ────────────────────────────────────────────────────────────────


def _contract_checks(manifest: dict, meta: dict[str, str], weights_sha: str) -> list[SanityCheck]:
    checks: list[SanityCheck] = []

    declared_sha = str(manifest.get("sha256", ""))
    checks.append(
        SanityCheck(
            name="bundle_integrity",
            passed=bool(declared_sha) and weights_sha == declared_sha,
            detail=(
                f"sha256(model.onnx)={weights_sha[:12]}… vs manifest={declared_sha[:12]}… "
                + ("match" if weights_sha == declared_sha else "MISMATCH")
            ),
        )
    )

    declared = tuple(manifest["features"])
    meta_features = tuple(meta.get("features", "").split("\n"))
    digest_ok = manifest.get("feature_set_sha256") == features_digest(declared)
    order_ok = declared == FEATURE_NAMES
    meta_ok = meta_features == declared
    checks.append(
        SanityCheck(
            name="feature_contract",
            passed=order_ok and meta_ok and digest_ok,
            detail=(
                f"manifest order {'== frozen FEATURE_NAMES' if order_ok else '!= frozen FEATURE_NAMES (#480 desync signature)'}; "
                f"ONNX metadata {'matches manifest' if meta_ok else 'differs from manifest'}; "
                f"feature_set_sha256 {'ok' if digest_ok else 'MISMATCH'}"
            ),
        )
    )

    candidate = str(manifest.get("candidate", ""))
    checks.append(
        SanityCheck(
            name="candidate_supported",
            passed=candidate == CANDIDATE_D,
            detail=(
                "d-boost"
                if candidate == CANDIDATE_D
                else f"{candidate!r} — n-head needs vector sidecars, out of suite scope (CLI-eval precedent)"
            ),
        )
    )
    return checks


def _adversarial_checks(session) -> list[SanityCheck]:
    checks: list[SanityCheck] = []
    twin = _perturbed(PROBE_RECORD)

    # (a) self-pair — content-independent by construction (see module docstring).
    p_self = _score_pair(session, features(PROBE_RECORD, PROBE_RECORD, 1.0))
    checks.append(
        SanityCheck(
            name="self_pair",
            passed=p_self >= SELF_PAIR_MIN,
            detail=(
                f"P(dup | probe vs itself, cos=1.0) = {p_self:.4f} "
                f"(required ≥ {SELF_PAIR_MIN})"
            ),
        )
    )

    # (b) near-boundary: light perturbation at cos≈0.99 must clear the cut.
    p_near = _score_pair(session, features(PROBE_RECORD, twin, NEAR_BOUNDARY_COS))
    checks.append(
        SanityCheck(
            name="near_boundary",
            passed=p_near >= NEAR_BOUNDARY_MIN,
            detail=(
                f"P(dup | probe vs twin, cos={NEAR_BOUNDARY_COS}) = {p_near:.4f} "
                f"(required ≥ {NEAR_BOUNDARY_MIN})"
            ),
        )
    )

    # (c) unrelated: different topic at the #480 probe cosine must stay below.
    p_unrelated = _score_pair(session, features(PROBE_RECORD, UNRELATED_RECORD, UNRELATED_COS))
    checks.append(
        SanityCheck(
            name="unrelated",
            passed=p_unrelated < UNRELATED_MAX,
            detail=(
                f"P(dup | probe vs unrelated, cos={UNRELATED_COS}) = {p_unrelated:.4f} "
                f"(required < {UNRELATED_MAX})"
            ),
        )
    )

    # (d) monotonicity over the FIXED probe-vs-twin pair as cosine drops.
    ladder_ps = [
        _score_pair(session, features(PROBE_RECORD, twin, cos)) for cos in COS_LADDER
    ]
    monotone = all(
        ladder_ps[i] >= ladder_ps[i + 1] - MONOTONICITY_TOLERANCE
        for i in range(len(ladder_ps) - 1)
    )
    ladder_text = ", ".join(
        f"cos={cos:.2f}:{p:.4f}" for cos, p in zip(COS_LADDER, ladder_ps)
    )
    checks.append(
        SanityCheck(
            name="monotonicity",
            passed=monotone,
            detail=(
                ("non-increasing" if monotone else "INCREASING somewhere")
                + f" over the fixed pair [{ladder_text}] "
                f"(tolerance {MONOTONICITY_TOLERANCE})"
            ),
        )
    )
    return checks


def run_sanity_suite(bundle: Path) -> SanityReport:
    """Run the full suite against one bundle (dir or .onnx path).

    Returns a :class:`SanityReport`; raises :class:`SanityLoadError` only
    when the bundle cannot be loaded at all (every loadable defect is a
    FAILED check in the report, not an exception).
    """
    model_path, manifest_path = _resolve_bundle(Path(bundle))
    manifest = _load_manifest(manifest_path)
    weights_sha = sha256_file(model_path) if model_path.is_file() else ""
    session = _open_session(model_path)
    meta = dict(session.get_modelmeta().custom_metadata_map)

    checks = _contract_checks(manifest, meta, weights_sha)
    checks.extend(_adversarial_checks(session))
    return SanityReport(
        bundle=str(model_path), weights_sha256=weights_sha, checks=tuple(checks)
    )


def ladder_probabilities(session) -> Sequence[float]:
    """Expose the monotonicity ladder's raw probabilities (used by tests and
    by the post-adopt diagnosis flow — the shape of the P(cos) response)."""
    twin = _perturbed(PROBE_RECORD)
    return [
        _score_pair(session, features(PROBE_RECORD, twin, cos)) for cos in COS_LADDER
    ]
