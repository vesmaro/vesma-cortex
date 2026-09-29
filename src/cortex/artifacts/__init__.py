"""Artifact contract: mnema-cortex-v1 identity, metadata, fingerprint.

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
from typing import Final

#: Artifact identity, ONE constant for the whole repo (ADR 0001 П1).
#: The engine bundles it under src/vesmaro/models/<ARTIFACT_NAME>/.
ARTIFACT_NAME: Final[str] = "mnema-cortex-v1"

#: Model name inside metadata_props (name ≠ artifact dir name: the dir
#: carries the major, the metadata carries the family name — engine
#: precedent: mnema-embed-v1 bundle, manifest "name" field).
MODEL_NAME: Final[str] = "mnema-cortex"

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


def sha256_file(path: Path) -> str:
    """sha256 over file bytes — the weights fingerprint discipline.

    Same scheme as the engine's ``mnema_weights_sha256`` (streamed,
    1 MiB chunks): the number that changes exactly when the weights
    change, i.e. the recalibration key (ADR-0021 discipline).
    """
    raise NotImplementedError("A3b+: implemented together with export-artifact")


def build_metadata_props(
    *,
    version: str = METADATA_VERSION,
    embedder_pin: str,
    corpus_fingerprint: str,
    trained_at: str,
    candidate: str,
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    """Assemble the ONNX metadata_props for a mnema-cortex artifact.

    Contract (inference-v1.md §4): ``embedder_pin`` MUST be the live
    engine fingerprint string (``nano:sha256:<hex>``), ``corpus_fingerprint``
    the BLAKE2b-256 of the training-pair manifest, ``feature_names`` the
    frozen ordered list (cortex.features.pair.FEATURE_NAMES) joined by
    newlines plus its sha256. Raises ValueError on unknown candidate.
    """
    raise NotImplementedError("A3b+: implemented together with export-artifact")


def assert_artifact_size(onnx_path: Path) -> None:
    """Enforce the ≤5 MB gate at export time (fail-loud, pre-bundle)."""
    raise NotImplementedError("A3b+: implemented together with export-artifact")
