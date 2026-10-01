"""Artifact contract: vesma-cortex-v1 identity, metadata, fingerprint.

Single source of truth for the artifact NAME (ADR 0001 П1: the constant
lives in exactly ONE place — see tests/test_skeleton.py guard). The full
inference contract is docs/specs/inference-v1.md; this module is its code
anchor: metadata_props assembly, the size gate, and the weights fingerprint
(sha256 over the .onnx bytes — the engine-side twin is
``mnema_weights_sha256`` in ``src/vesmaro/embeddings/__init__.py``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Mapping, Sequence

__all__ = [
    "ARTIFACT_NAME",
    "MODEL_NAME",
    "METADATA_VERSION",
    "MAX_ARTIFACT_BYTES",
    "METADATA_KEYS",
    "CANDIDATE_D",
    "CANDIDATE_N",
    "sha256_file",
    "build_metadata_props",
    "assert_artifact_size",
    "features_digest",
    "set_onnx_metadata",
]

#: Artifact identity, ONE constant for the whole repo (ADR 0001 П1).
#: The engine bundles it under src/vesmaro/models/<ARTIFACT_NAME>/.
ARTIFACT_NAME: Final[str] = "vesma-cortex-v1"

#: Model name inside metadata_props (name ≠ artifact dir name: the dir
#: carries the major, the metadata carries the family name — engine
#: precedent: vesma-embed-v1 bundle, manifest "name" field).
MODEL_NAME: Final[str] = "vesma-cortex"

#: Metadata schema major. Weight refresh within v1 bumps to "1.<n>" with a
#: new sha256 = recalibration event (inference-v1.md §8).
METADATA_VERSION: Final[str] = "1"

#: Hard size gate (ADR 0001 V3: single ONNX ≤ 5 MB, self-contained).
MAX_ARTIFACT_BYTES: Final[int] = 5 * 1024 * 1024

#: metadata_props keys (inference-v1.md §4). Frozen set — W5d validates
#: against exactly these.
METADATA_KEYS: Final[tuple[str, ...]] = (
    "name",
    "version",
    "embedder_pin",
    "corpus_fingerprint",
    "trained_at",
    "candidate",
    "features",
    "feature_set_sha256",
)

#: Which ladder candidate the artifact carries (ADR 0001 V1).
CANDIDATE_D: Final[str] = "d-boost"
CANDIDATE_N: Final[str] = "n-head"

#: sha256 over the "\n"-joined feature names — the compact feature-contract
#: assert (inference-v1.md §4, W5d step 8).
FEATURES_JOIN: Final[str] = "\n"


def sha256_file(path: Path) -> str:
    """sha256 over file bytes — the weights fingerprint discipline.

    Same scheme as the engine's ``mnema_weights_sha256`` (streamed,
    1 MiB chunks): the number that changes exactly when the weights
    change, i.e. the recalibration key (ADR-0021 discipline).
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def features_digest(feature_names: Sequence[str]) -> str:
    """sha256 hex of the "\n"-joined ordered feature names."""
    joined = FEATURES_JOIN.join(feature_names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_metadata_props(
    *,
    version: str = METADATA_VERSION,
    embedder_pin: str,
    corpus_fingerprint: str,
    trained_at: str,
    candidate: str,
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    """Assemble the ONNX metadata_props for a vesma-cortex artifact.

    Contract (inference-v1.md §4): ``embedder_pin`` MUST be the live
    engine fingerprint string (``nano:sha256:<hex>``), ``corpus_fingerprint``
    the BLAKE2b-256 of the training-pair manifest, ``feature_names`` the
    frozen ordered list (cortex.features.pair.FEATURE_NAMES) joined by
    newlines plus its sha256. Raises ValueError on unknown candidate.
    """
    if candidate not in (CANDIDATE_D, CANDIDATE_N):
        raise ValueError(
            f"unknown ladder candidate {candidate!r} — expected {CANDIDATE_D!r} or {CANDIDATE_N!r}"
        )
    if not embedder_pin:
        raise ValueError("embedder_pin is required (live engine fingerprint, nano:sha256:<hex>)")
    if not corpus_fingerprint:
        raise ValueError("corpus_fingerprint is required (BLAKE2b-256 of the pair manifest)")
    if not feature_names:
        raise ValueError("feature_names must be a non-empty ordered sequence")
    return {
        "name": MODEL_NAME,
        "version": version,
        "embedder_pin": embedder_pin,
        "corpus_fingerprint": corpus_fingerprint,
        "trained_at": trained_at,
        "candidate": candidate,
        "features": FEATURES_JOIN.join(feature_names),
        "feature_set_sha256": features_digest(feature_names),
    }


def assert_artifact_size(onnx_path: Path) -> None:
    """Enforce the ≤5 MB gate at export time (fail-loud, pre-bundle)."""
    size = Path(onnx_path).stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"artifact {onnx_path} is {size} bytes — exceeds the "
            f"{MAX_ARTIFACT_BYTES}-byte gate (ADR 0001 V3, inference-v1.md §4)"
        )


def set_onnx_metadata(model_proto, props: Mapping[str, str]) -> None:
    """Write ``props`` into the ONNX model metadata_props (in place).

    Lazy ``onnx`` import: the artifact module stays import-light for
    callers that never export (onnx arrives with the skl2onnx/onnxmltools
    train chain, never as a runtime dep of the engine).
    """
    import onnx  # local: export-path only

    existing = {entry.key: entry for entry in model_proto.metadata_props}
    for key, value in props.items():
        if key in existing:
            existing[key].value = value
        else:
            model_proto.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=value)
            )
