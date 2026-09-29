"""Fingerprint discipline — the frozen preregistration v2 scheme.

Prereg v2 (canon, frozen): canonical JSON of each pair (sorted keys, UTF-8)
→ sha256; manifest = lines ``pair_id <sha256>`` sorted by pair_id; corpus
fingerprint = BLAKE2b-256 (hex) of the manifest bytes. Labels get their own
file with their own fingerprint (same scheme). Both are fixed in W5b before
the run and re-verified at run time — a mismatch voids the run.

Hashing is stdlib-only (hashlib: sha256 + blake2b) — no pyblake2 dependency
(ADR note in pyproject).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Iterable

__all__ = [
    "CANONICAL_JSON_KWARGS",
    "canonical_json",
    "pair_sha256",
    "manifest_bytes",
    "corpus_fingerprint",
    "labels_fingerprint",
]

#: The exact json.dumps kwargs that make a canonical JSON: sorted keys, real
#: UTF-8 (ensure_ascii=False), no incidental whitespace. Any change here is
#: a fingerprint-format break — never change without a prereg addendum.
CANONICAL_JSON_KWARGS: Final[dict[str, Any]] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}


def canonical_json(obj: Any) -> str:
    """Canonical JSON text (sorted keys, UTF-8, tight separators)."""
    return json.dumps(obj, **CANONICAL_JSON_KWARGS)


def pair_sha256(pair: dict[str, Any]) -> str:
    """sha256 of the canonical JSON bytes of ONE corpus pair (hex).

    The fingerprinted object is the CORPUS pair (data-contract.md §3):
    ``{"record": <side>, "candidate": <side>, "similarity": float}`` where
    each side carries the prereg export composition — title, body, tags,
    language, record_type, created_at. pair_id/stratum/label/store-ids ride
    the manifest line, never the fingerprinted object (ids are assignment,
    not content). The ARTIFACT input pair (inference-v1.md §1) is a
    projection of this object with created_at dropped — a DIFFERENT
    canonical object with its own discipline.
    """
    return hashlib.sha256(canonical_json(pair).encode("utf-8")).hexdigest()


def manifest_bytes(entries: Iterable[tuple[str, str]]) -> bytes:
    """Manifest bytes: ``pair_id <sha256>`` lines, sorted by pair_id.

    Sorting is part of the frozen scheme (byte-identical manifests for
    identical corpora regardless of generation order). Every line is
    ``\\n``-terminated, including the last (data-contract.md §5.3).
    """
    lines = [f"{pair_id} {digest}" for pair_id, digest in sorted(entries)]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def corpus_fingerprint(manifest: bytes) -> str:
    """BLAKE2b-256 (hex) of the manifest bytes — the corpus fingerprint."""
    return hashlib.blake2b(manifest, digest_size=32).hexdigest()


def labels_fingerprint(labels: dict[str, str]) -> str:
    """Fingerprint of the label file (same scheme, own fingerprint)."""
    return hashlib.blake2b(canonical_json(labels).encode("utf-8"), digest_size=32).hexdigest()
