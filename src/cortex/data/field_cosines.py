"""Field-cosine sidecar (A2 docompute) — reader and pair helper.

ADR 0001 V1 ablation "with/without field cosines": the three optional
features (``cos_title``, ``cos_body``, ``cos_tags``) are docomputed by the
bundle embedder (vesma-embed-v1) on CPU — this library carries no embedder
(cortex.features.pair refuses ``field_cosines=True`` loudly). The docompute
itself runs OUTSIDE ``src/cortex`` in ``scripts/a2_field_cosines.py`` under
the ENGINE's environment (onnxruntime + tokenizers live there), so the
network-import isolation of ``src/cortex`` stays intact and the engine is
never imported into cortex processes.

Sidecar artifact (``data/vectors/<corpus-id>/field_vecs.npz``, gitignored):

- ``ids``            — (N,) str, hygiene-passed record ids (export order);
- ``title_vecs`` / ``body_vecs`` / ``tags_vecs`` — (N, 384) float32,
  L2-normalized bundle-embedder outputs over each field separately
  (``tags`` text = space-joined tags; an empty field embeds like any
  text — two empty fields therefore yield cosine 1.0, matching the
  ``_jaccard`` empty-set convention of cortex.features.pair);
- ``embedder_fingerprint`` / ``embedder_model`` / ``corpus_id`` /
  ``created_at_utc`` — provenance scalars.

Field VECTORS are stored per record (strictly more informative than
pre-derived cosines: any pair consumer derives them deterministically);
:func:`FieldSidecar.cosines_for_pair` computes the three attach-ready
values for one pair. The sidecar step ALSO writes ready per-pair cosines
for the near-duplicate bases (``near_dup_field_cosines.jsonl``).

Privacy invariant: the sidecar script reads ONLY the exported
``records.jsonl`` — i.e. records that already passed the frozen hygiene
chain. No store content is re-read there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np

from cortex.features.pair import FeatureVector, attach_field_cosines

__all__ = [
    "SIDE_NPZ_NAME",
    "SIDE_PAIR_COSINES_NAME",
    "FieldSidecarError",
    "FieldSidecar",
    "cosines_for_pair",
    "attach_to_features",
]

SIDE_NPZ_NAME: Final[str] = "field_vecs.npz"
SIDE_PAIR_COSINES_NAME: Final[str] = "near_dup_field_cosines.jsonl"


class FieldSidecarError(ValueError):
    """The sidecar artifact is missing, malformed, or lacks a requested id."""


class FieldSidecar:
    """Reader over the A2 field-vector sidecar npz (pure numpy)."""

    def __init__(self, npz_path: str | Path) -> None:
        path = Path(npz_path)
        if not path.is_file():
            raise FieldSidecarError(f"field sidecar not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"ids", "title_vecs", "body_vecs", "tags_vecs", "embedder_fingerprint"}
            missing = required - set(data.files)
            if missing:
                raise FieldSidecarError(f"field sidecar {path} misses arrays: {sorted(missing)}")
            self._ids = np.asarray(data["ids"])
            self._title = np.asarray(data["title_vecs"], dtype=np.float32)
            self._body = np.asarray(data["body_vecs"], dtype=np.float32)
            self._tags = np.asarray(data["tags_vecs"], dtype=np.float32)
            self.fingerprint = str(data["embedder_fingerprint"])
            self.model = str(data["embedder_model"]) if "embedder_model" in data.files else ""
            self.corpus_id = str(data["corpus_id"]) if "corpus_id" in data.files else ""

        if not (len(self._ids) == len(self._title) == len(self._body) == len(self._tags)):
            raise FieldSidecarError(f"field sidecar {path} has ragged arrays")
        if self._title.ndim != 2 or self._title.shape[1] == 0:
            raise FieldSidecarError(f"field sidecar {path} has bad vector shape {self._title.shape}")
        self._index: dict[str, int] = {str(value): i for i, value in enumerate(self._ids)}
        if len(self._index) != len(self._ids):
            raise FieldSidecarError(f"field sidecar {path} has duplicate ids")

    @property
    def ids(self) -> list[str]:
        return [str(value) for value in self._ids]

    @property
    def dimension(self) -> int:
        return int(self._title.shape[1])

    def _row(self, record_id: str) -> int:
        row = self._index.get(record_id)
        if row is None:
            raise FieldSidecarError(f"id {record_id!r} not in field sidecar")
        return row

    def cosines_for_pair(self, id_a: str, id_b: str) -> tuple[float, float, float]:
        """(cos_title, cos_body, cos_tags) for one pair — attach-ready."""
        row_a, row_b = self._row(id_a), self._row(id_b)

        def cosine(block: np.ndarray) -> float:
            vec_a, vec_b = block[row_a], block[row_b]
            denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            if denominator == 0.0:
                return 1.0  # two zero vectors (empty field) — identical by convention
            # float32 dots on near-collinear unit vectors overshoot 1.0 by
            # ~1e-7 — clamp to the mathematical range (attach_field_cosines
            # rejects anything outside [-1, 1]).
            return min(1.0, max(-1.0, float(np.dot(vec_a, vec_b) / denominator)))

        return cosine(self._title), cosine(self._body), cosine(self._tags)


def cosines_for_pair(
    sidecar: FieldSidecar, id_a: str, id_b: str
) -> tuple[float, float, float]:
    """Module-level form of :meth:`FieldSidecar.cosines_for_pair`."""
    return sidecar.cosines_for_pair(id_a, id_b)


def attach_to_features(
    vector: FeatureVector, sidecar: FieldSidecar, id_a: str, id_b: str
) -> FeatureVector:
    """Attach the sidecar's three cosines to a core feature vector.

    Thin composition over :func:`cortex.features.pair.attach_field_cosines`
    (validation lives there: core-variant check, [-1, 1] bounds).
    """
    cos_title, cos_body, cos_tags = sidecar.cosines_for_pair(id_a, id_b)
    return attach_field_cosines(vector, cos_title=cos_title, cos_body=cos_body, cos_tags=cos_tags)


def load_pair_cosines(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Read the ready per-pair cosines file (pair_id → 3 cosines)."""
    out: dict[str, tuple[float, float, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["pair_id"])] = (
            float(row["cos_title"]),
            float(row["cos_body"]),
            float(row["cos_tags"]),
        )
    return out
