"""Adversarial sanity suite — the mandatory pre-ADOPT instrument (#480).

Why it exists: B2 (weights ``71f0572d…``) scored sens 0.989 on the
dataset-v3 holdout while the SHIPPED artifact graded a record against
ITSELF at P(dup)=0.0108 — an inversion (must be ~1.0). The eval surface
contained NO exact self-pairs, so the holdout metrics were blind to the
defect (vesmaro/vesma#480). Root suspicion: feature-column-order desync
between the export and the frozen 13-feature contract.

Every ADOPT claim must pass the policy-aware exam v2 BELOW, before any
other metric is quoted (eval-methodology §10 + §4 gate table;
gate_contract.json ``sanity_v2`` — the constants here are coherent with
that section, pinned by test_eval_sanity.py):

    ``self_pair``        P(dup | record vs itself, cos=1.0) ≥ 0.9
    ``cosmetic_twin``    P(dup | title-punctuation twin, cos≈0.99) ≥ 0.5
                         (policy v1.1 §8.2: punctuation = cosmetic/T1 =
                         dup; supersedes the v1 ``near_boundary`` probe)
    ``fact_edit_twin``   P(dup | key-fact token edited in the body,
                         cos≈0.99) < 0.5 (policy v1.1 §8.2: names/numbers
                         = razor/T3 = NOT dup — the anti-dominance side of
                         the two-valued razor zone, the B2-prime lesson:
                         a corpus 192:72 negative-dominant in the band
                         taught «close = not dup», near_boundary 0.0261)
    ``envelope_variant`` P(dup | envelope-only delta, cos=1.0) ≥ 0.5
                         (policy v1.1 §8.1 whitelist; feature-degenerate
                         by contract — see _envelope_variant)
    ``unrelated``        P(dup | different-topic pair, cos=0.578) < 0.5
    ``monotonicity``     ZONED v2 over a FIXED pair: the razor zone
                         (0.95; 1.0) is two-valued by policy v1.1 — no
                         monotonicity asserted inside it; outside the
                         zone P is non-increasing (operationally
                         P(0.95) ≥ P(0.8) ≥ P(0.5), tol 1e-6;
                         eval-methodology §10.2)

Two-sidedness is the point of v2: in the razor band the exam demands BOTH
correct answers — a positive on cosmetics/envelope AND a negative on a
fact edit — instead of the v1 single side of the zone.

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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from cortex.artifacts import CANDIDATE_D, features_digest, sha256_file
from cortex.features.pair import FEATURE_NAMES, FeatureVector, PairRecord, features

__all__ = [
    "COS_LADDER",
    "COSMETIC_TWIN_MIN",
    "ENVELOPE_DATES",
    "ENVELOPE_VARIANT_COS",
    "ENVELOPE_VARIANT_MIN",
    "EXAM_V1",
    "EXAM_V2",
    "FACT_EDIT_ANCHOR",
    "FACT_EDIT_COS",
    "FACT_EDIT_REPLACEMENT",
    "FACT_EDIT_TWIN_MAX",
    "MANIFEST_EXAM_KEY",
    "MANIFEST_FILENAME",
    "MODEL_FILENAME",
    "MONOTONICITY_TOLERANCE",
    "NEAR_BOUNDARY_COS",
    "NEAR_BOUNDARY_MIN",
    "PROBE_RECORD",
    "RAZOR_ZONE_COS_LOW",
    "SELF_PAIR_MIN",
    "UNRELATED_COS",
    "UNRELATED_MAX",
    "UNRELATED_RECORD",
    "SanityCheck",
    "SanityLoadError",
    "SanityReport",
    "ladder_probabilities",
    "run_sanity_suite",
]

# ── exam cohort (manifest stamp; eval-methodology §10 change-control) ────────

#: The exam version a bundle was CERTIFIED under. The manifest ``stamp``
#: (written by ``cortex export-artifact``); its ABSENCE = v1 — the
#: historical cohort (artifacts trained under the pre-policy exam, e.g.
#: the registry B1 weights). Cohort semantics: an artifact certified under
#: exam v1 is REQUIRED to pass suite v1; one certified/stamped v2 is
#: REQUIRED to pass suite v2 (the two-sided policy-aware exam). Mixing —
#: running v2 probes on a v1-certified bundle or vice versa — would test
#: an artifact against rules it was never trained or ratified under
#: (the ratification asymmetry: policy v1.1 changed the ground truth).
EXAM_V1: Final[str] = "1"
EXAM_V2: Final[str] = "2"

#: Manifest key holding the exam stamp.
MANIFEST_EXAM_KEY: Final[str] = "sanity_exam"

# ── thresholds (the D1 protocol requirements, frozen here) ───────────────────

#: (a) self-pair floor: a healthy artifact grades a record against itself
#: as a duplicate with P ≥ 0.9 (inversion = #480 defect class).
SELF_PAIR_MIN: Final[float] = 0.9

#: (b) near-boundary probe cosine.
NEAR_BOUNDARY_COS: Final[float] = 0.99

#: (b) the probability cut (runner's DUPLICATE_THRESHOLD_PROBABILITY).
NEAR_BOUNDARY_MIN: Final[float] = 0.5

# ── sanity v2 exam thresholds (eval-methodology §10; frozen D1 pins above
#    stay as history and live on in the evalsets surface; coherence of the
#    v2 numbers with gate_contract.json `sanity_v2` is pinned by test) ──────

#: (i) cosmetic-twin floor: the title-punctuation twin of the probe
#: (policy v1.1 §8.2 re-classed it cosmetic/T1 = dup) in the razor band
#: must clear the duplicate cut.
COSMETIC_TWIN_MIN: Final[float] = 0.5

#: (ii) fact-edit twin ceiling: a key-fact token edited in the body (the
#: 'names' class of policy v1.1 §8.2) in the razor band must stay BELOW
#: the cut — the anti-dominance side of the two-valued razor zone.
FACT_EDIT_TWIN_MAX: Final[float] = 0.5

#: (ii) the fact-edit probe cosine point (the razor band, where corpus
#: T3 negatives live).
FACT_EDIT_COS: Final[float] = 0.99

#: (iii) envelope-variant floor: the same record differing only in an
#: envelope attribute (whitelist, policy v1.1 §8.1) must grade dup.
ENVELOPE_VARIANT_MIN: Final[float] = 0.5

#: (iii) the envelope-variant probe cosine point: identical text measures
#: cos 1.0 through the embedder (the corpus P-envelope class).
ENVELOPE_VARIANT_COS: Final[float] = 1.0

#: (d, v2) the razor zone's lower edge — the zone is the OPEN interval
#: (0.95; 1.0): ladder steps starting strictly above this line are not
#: checked, the zone is two-valued by policy v1.1 (monotonicity is not
#: asserted inside it; eval-methodology §10.2).
RAZOR_ZONE_COS_LOW: Final[float] = 0.95

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


#: The key-fact anchor of the probe body (the cache-key example) and its
#: edited form — the 'names' key-fact class of policy v1.1 §8.2: renaming
#: the key from a cart to a card changes WHAT the record states. Same
#: length by design: the pair keeps the razor-zone shape (zero len-delta,
#: minimal char signature), exactly the corpus T3 geometry.
FACT_EDIT_ANCHOR: Final[str] = "cart:{user_id}"
FACT_EDIT_REPLACEMENT: Final[str] = "card:{user_id}"

#: The envelope dates of the variant twin (policy v1.1 §8.1 whitelist:
#: «дата фиксации/события в конверте» — the same message re-observed
#: later). NOT representable in :class:`PairRecord`: the frozen contract
#: surface carries no timestamp (features/pair.py — ``created_at``
#: deliberately NOT here, OQ-2), so this delta is invisible to the
#: 13-feature vector BY CONTRACT. Kept as documentation of what the
#: corpus-level P-envelope class varies.
ENVELOPE_DATES: Final[tuple[str, str]] = ("2026-09-01", "2026-10-01")


def _fact_edited(record: PairRecord) -> PairRecord:
    """The razor-side twin: ONE key-fact token in the body replaced (the
    cache-key name ``cart`` → ``card`` — NOT dup by policy v1.1 §8.2).
    Fails loud if the frozen probe body no longer carries the anchor —
    a drifted probe must be repaired, never silently skipped."""
    body = record.body
    if body.count(FACT_EDIT_ANCHOR) != 1:
        raise ValueError(
            "PROBE_RECORD body lost the fact-edit anchor "
            f"{FACT_EDIT_ANCHOR!r} (occurrences: {body.count(FACT_EDIT_ANCHOR)}) "
            "— the frozen probe drifted; repair the probe or the anchor"
        )
    return replace(
        record, body=body.replace(FACT_EDIT_ANCHOR, FACT_EDIT_REPLACEMENT, 1)
    )


def _envelope_variant(record: PairRecord) -> PairRecord:
    """The whitelist-side twin: the same record re-observed under a
    different envelope date (``ENVELOPE_DATES``) = dup by policy v1.1
    §8.1 («то же сообщение, зафиксированное позже» — same thought, same
    facts).

    Feature-degenerate BY CONTRACT: the envelope delta rides OUTSIDE the
    frozen 13-feature surface (RecordLike carries no timestamp — OQ-2),
    so the probe vector equals the self-pair vector. That degeneracy is
    pinned by test (features(envelope_variant) == features(self)): the
    check keeps the policy verdict asserted and becomes LOAD-BEARING the
    moment the feature surface grows envelope fields (OQ-2 resolution) —
    at that point this twin must grow the real delta with it.
    """
    return replace(record)


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
    exam_version: str = EXAM_V2

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
        candidates = (
            bundle / MANIFEST_FILENAME,
            bundle / f"{model_path.stem}.manifest.json",
        )
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
        raise SanityLoadError(
            f"manifest is not valid JSON: {manifest_path} — {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SanityLoadError(
            f"manifest must be a JSON object, got {type(manifest).__name__}"
        )
    for key in ("sha256", "features", "candidate"):
        if key not in manifest:
            raise SanityLoadError(
                f"manifest missing required key {key!r}: {manifest_path}"
            )
    if not isinstance(manifest["features"], list) or not all(
        isinstance(name, str) for name in manifest["features"]
    ):
        raise SanityLoadError(
            "manifest 'features' must be a list of feature-name strings"
        )
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


def _contract_checks(
    manifest: dict, meta: dict[str, str], weights_sha: str
) -> list[SanityCheck]:
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


def _adversarial_checks_v1(session) -> list[SanityCheck]:
    """The HISTORICAL one-sided exam (v1, the pre-policy generation):
    the razor-band probe demanded a POSITIVE on the near-boundary twin —
    «any cos≈0.99 twin = dup». Superseded by the ratified policy v1.1
    (two-sided zone), kept ONLY for cohorts certified under it (the
    registry B1 weights); it is the exam that B1's ADOPT was graded by.
    """
    checks: list[SanityCheck] = []
    twin = _perturbed(PROBE_RECORD)

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

    p_near = _score_pair(session, features(PROBE_RECORD, twin, NEAR_BOUNDARY_COS))
    checks.append(
        SanityCheck(
            name="near_boundary",
            passed=p_near >= NEAR_BOUNDARY_MIN,
            detail=(
                f"P(dup | probe vs near-boundary twin, cos={NEAR_BOUNDARY_COS}) = "
                f"{p_near:.4f} (required ≥ {NEAR_BOUNDARY_MIN}; v1 exam — "
                "superseded by the v2 policy-aware probes for v2 cohorts)"
            ),
        )
    )

    p_unrelated = _score_pair(
        session, features(PROBE_RECORD, UNRELATED_RECORD, UNRELATED_COS)
    )
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

    ladder_ps = ladder_probabilities(session)
    monotone = all(
        ladder_ps[i] >= ladder_ps[i + 1] - MONOTONICITY_TOLERANCE
        for i in range(len(COS_LADDER) - 1)
    )
    ladder_text = ", ".join(
        f"cos={cos:.2f}:{p:.4f}" for cos, p in zip(COS_LADDER, ladder_ps)
    )
    checks.append(
        SanityCheck(
            name="monotonicity",
            passed=monotone,
            detail=(
                ("non-increasing" if monotone else "INCREASE")
                + f" over the fixed pair [{ladder_text}] "
                "(v1 full ladder; tolerance "
                f"{MONOTONICITY_TOLERANCE})"
            ),
        )
    )
    return checks


def _adversarial_checks(session) -> list[SanityCheck]:
    """The policy-aware two-sided exam (v2, eval-methodology §10):
    the probe generation the ratified policy v1.1 entails — see the
    module docstring for the check list. Applied to bundles stamped
    ``sanity_exam: 2``."""
    checks: list[SanityCheck] = []
    twin = _perturbed(PROBE_RECORD)
    fact_twin = _fact_edited(PROBE_RECORD)

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

    # (i) cosmetic twin (v2, ex near_boundary): the punctuation-titled twin
    # in the razor band must clear the cut (policy v1.1: cosmetic = dup).
    p_cosmetic = _score_pair(session, features(PROBE_RECORD, twin, NEAR_BOUNDARY_COS))
    checks.append(
        SanityCheck(
            name="cosmetic_twin",
            passed=p_cosmetic >= COSMETIC_TWIN_MIN,
            detail=(
                f"P(dup | probe vs cosmetic twin, cos={NEAR_BOUNDARY_COS}) = "
                f"{p_cosmetic:.4f} (required ≥ {COSMETIC_TWIN_MIN}; policy v1.1 "
                "§8.2: title punctuation is the cosmetic/T1 class)"
            ),
        )
    )

    # (ii) fact-edit twin: a key-fact token replaced in the body must stay
    # BELOW the cut — the anti-dominance side of the two-valued razor zone.
    p_fact = _score_pair(session, features(PROBE_RECORD, fact_twin, FACT_EDIT_COS))
    checks.append(
        SanityCheck(
            name="fact_edit_twin",
            passed=p_fact < FACT_EDIT_TWIN_MAX,
            detail=(
                f"P(dup | probe vs fact-edit twin, cos={FACT_EDIT_COS}) = "
                f"{p_fact:.4f} (required < {FACT_EDIT_TWIN_MAX}; policy v1.1 "
                "§8.2: a key-fact edit is razor/T3 = NOT dup)"
            ),
        )
    )

    # (iii) envelope-variant: an envelope-only delta (feature-degenerate by
    # contract, see _envelope_variant) must grade dup.
    p_env = _score_pair(
        session,
        features(PROBE_RECORD, _envelope_variant(PROBE_RECORD), ENVELOPE_VARIANT_COS),
    )
    checks.append(
        SanityCheck(
            name="envelope_variant",
            passed=p_env >= ENVELOPE_VARIANT_MIN,
            detail=(
                f"P(dup | probe vs envelope-variant, cos={ENVELOPE_VARIANT_COS}) = "
                f"{p_env:.4f} (required ≥ {ENVELOPE_VARIANT_MIN}; policy v1.1 "
                "§8.1 whitelist; feature-degenerate to the self-pair by OQ-2 — "
                "degeneracy pinned by test)"
            ),
        )
    )

    # (c) unrelated: different topic at the #480 probe cosine must stay below.
    p_unrelated = _score_pair(
        session, features(PROBE_RECORD, UNRELATED_RECORD, UNRELATED_COS)
    )
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

    # (d) monotonicity v2 — ZONED: the razor zone (0.95; 1.0) is two-valued
    # by policy v1.1, steps starting strictly above its low edge are not
    # asserted; outside the zone P must be non-increasing (operationally
    # P(0.95) ≥ P(0.8) ≥ P(0.5)).
    ladder_ps = [
        _score_pair(session, features(PROBE_RECORD, twin, cos)) for cos in COS_LADDER
    ]
    monotone = True
    for i in range(len(COS_LADDER) - 1):
        if COS_LADDER[i] > RAZOR_ZONE_COS_LOW:
            continue  # step starts inside the razor zone / identity point
        if ladder_ps[i] < ladder_ps[i + 1] - MONOTONICITY_TOLERANCE:
            monotone = False
            break
    ladder_text = ", ".join(
        f"cos={cos:.2f}:{p:.4f}" for cos, p in zip(COS_LADDER, ladder_ps)
    )
    checks.append(
        SanityCheck(
            name="monotonicity",
            passed=monotone,
            detail=(
                (
                    "non-increasing outside the razor zone"
                    if monotone
                    else "INCREASE outside the razor zone"
                )
                + f" over the fixed pair [{ladder_text}] "
                f"(zoned v2: steps above cos={RAZOR_ZONE_COS_LOW} unchecked — "
                f"two-valued zone; tolerance {MONOTONICITY_TOLERANCE})"
            ),
        )
    )
    return checks


def run_sanity_suite(bundle: Path) -> SanityReport:
    """Run the full suite against one bundle (dir or .onnx path).

    Returns a :class:`SanityReport`; raises :class:`SanityLoadError` only
    when the bundle cannot be loaded at all (every loadable defect is a
    FAILED check in the report, not an exception).

    Exam cohort (eval-methodology §10 change-control): the manifest
    ``sanity_exam`` stamp selects the probe generation — absent/v1 stamps
    run the historical one-sided exam (v1: near_boundary requires
    P(dup) ≥ 0.5, no two-sidedness), a v2 stamp runs the policy-aware
    two-sided exam. An artifact certified under one exam must not be
    re-graded under another it was not ratified against.
    """
    model_path, manifest_path = _resolve_bundle(Path(bundle))
    manifest = _load_manifest(manifest_path)
    exam = _exam_cohort(manifest, manifest_path)
    weights_sha = sha256_file(model_path) if model_path.is_file() else ""
    session = _open_session(model_path)
    meta = dict(session.get_modelmeta().custom_metadata_map)

    checks = _contract_checks(manifest, meta, weights_sha)
    checks.extend(
        _adversarial_checks(session)
        if exam == EXAM_V2
        else _adversarial_checks_v1(session)
    )
    return SanityReport(
        bundle=str(model_path),
        weights_sha256=weights_sha,
        checks=tuple(checks),
        exam_version=exam,
    )


def _exam_cohort(manifest: dict, manifest_path: Path) -> str:
    """The bundle's exam cohort from the manifest stamp (absent = v1).

    Unknown stamp values fail LOUD (SanityLoadError): a mistyped stamp
    must not silently degrade the exam to either generation.
    """
    stamp = manifest.get(MANIFEST_EXAM_KEY)
    if stamp is None or stamp == EXAM_V1:
        return EXAM_V1
    if stamp == EXAM_V2:
        return EXAM_V2
    raise SanityLoadError(
        f"manifest {manifest_path} carries unknown {MANIFEST_EXAM_KEY} "
        f"stamp {stamp!r} — expected {EXAM_V1!r} or {EXAM_V2!r}"
    )


def ladder_probabilities(session) -> Sequence[float]:
    """Expose the monotonicity ladder's raw probabilities (used by tests and
    by the post-adopt diagnosis flow — the shape of the P(cos) response)."""
    twin = _perturbed(PROBE_RECORD)
    return [
        _score_pair(session, features(PROBE_RECORD, twin, cos)) for cos in COS_LADDER
    ]
