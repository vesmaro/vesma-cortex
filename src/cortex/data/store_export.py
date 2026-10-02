"""Store export for the A2 pretrain corpus — hygiene-frozen pipeline.

Contract anchors (frozen, do not reorder):

- **data-contract.md §6** — the export hygiene ORDER is frozen: (1) SQL
  candidate selection from the single engine store, read-only URI only;
  (2) per record — secrets scanner + ``no-federate`` tag + quarantine
  check BEFORE any output; a hit excludes the record with a per-reason
  counter, never with content; (3) only then the minimal export
  composition (title, body, tags, language, record_type, created_at,
  content_hash, vector); (4) fingerprints last.
- **ADR 0001 V2** — unlabeled store pairs are PRETRAIN-only; this export
  carries no labels and never extends the supervised train definition.
- **ADR 0001 П3 / prereg hygiene** — no raw store line ever enters the
  repo: content lands under ``data/`` (gitignored), the repo gets
  counters and fingerprints only.

What this module exports (contract resolution per data-contract §4):

- the §4 ``pairs.jsonl`` manifest (source ∈ {weak-positive, hard-negative},
  ``transform`` from cortex.pretrain.corruption) is generated DOWNSTREAM by
  ``cortex pretrain`` from a records pool — so this exporter's primary
  artifact is the POOL (``records.jsonl``), each row already carrying the
  store vector (``vec``) that candidate N's pretrain consumes;
- NEAR-DUPLICATE candidate pairs (store-vector cosine inside a frozen
  band, default [0.85, 0.97)) are exported alongside as unlabeled
  corruption BASES (``near_dup_candidates.jsonl`` + manifest + corpus
  fingerprint per the §5 scheme) — they are diagnostics/base material,
  NOT the §4 manifest and carry no ``source``/``transform``.

Engine reuse (ADR 0001 V2 hygiene): the secrets/danger detectors are the
ENGINE's own modules (``vesmaro.secrets_detector``, ``vesmaro.danger_detectors``)
loaded from a caller-provided engine source tree — both are stdlib-only,
so the import is clean and the engine checkout is never modified. The
engine import lives behind :func:`load_engine_scanner` and is injected
into :func:`export_store_corpus`; when the engine tree is unavailable a
clearly-marked local FALLBACK scanner is used instead (provenance
``"fallback scanner, not engine"`` is required in every report then).

Field mapping mirrors the engine's projection where the contract cares:

- ``language`` — ``metadata.canon.language`` (the engine's
  ``CanonRecordView.from_memory`` reads the envelope there); a pre-canon
  row yields ``None`` — the export stays honest about what the store does
  not say;
- ``record_type`` — the ``memory_type`` COLUMN (note/snippet/fact), NOT
  ``canon.type``: on the live store ``canon.type`` carries pipeline
  enrichment values (``checkpoint``), while the corruption transform
  cycle (corruption.py) is frozen over note/snippet/fact. Documented
  deviation, deliberate.

Store access is strictly read-only: both databases open via
``file:…?mode=ro`` URIs plus ``PRAGMA query_only=ON`` belt-and-braces —
the store serves a live MCP server and is never written, never locked
long (open → read → close).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Protocol, Sequence

import numpy as np

from cortex.data.fingerprints import (
    canonical_json,
    corpus_fingerprint,
    manifest_bytes,
    pair_sha256,
)

__all__ = [
    "NO_FEDERATE_TAG",
    "EMBEDDING_DIM",
    "VECTOR_BLOB_BYTES",
    "UNIT_NORM_TOLERANCE",
    "DEFAULT_LIMIT_POOL",
    "DEFAULT_MIN_COSINE",
    "DEFAULT_MAX_COSINE",
    "StoreRecord",
    "PairBase",
    "HygieneCounters",
    "TextScanner",
    "EngineScanner",
    "FallbackScanner",
    "EngineScannerError",
    "load_engine_scanner",
    "StoreOpenError",
    "ExportResult",
    "export_store_corpus",
    "load_ids_file",
    "edge_node_ids",
    "expand_one_hop",
    "SilverEdge",
    "TargetedExportResult",
    "export_targeted_corpus",
]

#: The engine's federation-exclusion tag (vesmaro.models.NO_FEDERATE_TAG).
#: Repeated here as a literal to keep this module import-free of the engine;
#: pinned against the engine source by tests/test_store_export.py when the
#: engine tree is available.
NO_FEDERATE_TAG: Final[str] = "mnemos:no-federate"

#: vesma-embed-v1 geometry: 384-dim float32 → 1536-byte blob.
EMBEDDING_DIM: Final[int] = 384
VECTOR_BLOB_BYTES: Final[int] = EMBEDDING_DIM * 4
#: Store vectors are unit-normalized (ArchCom A1 store inspection); a row
#: outside this band is a store-side anomaly → excluded with a counter,
#: never silently re-normalized (modify-nothing discipline).
UNIT_NORM_TOLERANCE: Final[float] = 1e-3

DEFAULT_LIMIT_POOL: Final[int] = 800
DEFAULT_MIN_COSINE: Final[float] = 0.85
DEFAULT_MAX_COSINE: Final[float] = 0.97

#: Store ids must not contain the pair_id separator (data-contract §2):
#: a violation refuses the WHOLE export (assignment integrity), it is not
#: a per-record hygiene exclusion.
_PAIR_ID_SEPARATOR: Final[str] = "--"


# ── Data shapes ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoreRecord:
    """One hygiene-passed store record in the minimal export composition."""

    id: str
    title: str
    body: str
    tags: tuple[str, ...]
    language: str | None
    record_type: str | None
    created_at: str
    content_hash: str
    vec_sha256: str
    vector: tuple[float, ...]

    def side(self) -> dict[str, object]:
        """The §3 fingerprinted-side composition (no vector, no id)."""
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "language": self.language,
            "record_type": self.record_type,
            "created_at": self.created_at,
        }

    def pool_row(self) -> dict[str, object]:
        """The records.jsonl row: minimal composition + vector provenance."""
        return {
            "id": self.id,
            **self.side(),
            "content_hash": self.content_hash,
            "vec_sha256": self.vec_sha256,
            "vec": list(self.vector),
        }

    def pool_fingerprint_payload(self) -> dict[str, object]:
        """Object hashed into the pool manifest (vector via its sha256)."""
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "language": self.language,
            "record_type": self.record_type,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "vec_sha256": self.vec_sha256,
        }


@dataclass(frozen=True)
class PairBase:
    """One unlabeled near-duplicate pair base (NOT a §4 manifest row)."""

    pair_id: str
    id_a: str
    id_b: str
    similarity: float
    record: StoreRecord
    candidate: StoreRecord

    def row(self) -> dict[str, object]:
        """Lean pair-base row: the sides are NOT duplicated here.

        One popular record participates in hundreds of band pairs —
        inlining both §3 sides per row would explode the file ~40× with
        repeated content. The fingerprinted object is computed BEFORE
        writing (pair_sha256 pins the full §3 sides); a verifier rebuilds
        the sides by joining records.jsonl on id_a/id_b and recomputes
        the hash — any side edit breaks the manifest fingerprint.
        """
        return {
            "pair_id": self.pair_id,
            "id_a": self.id_a,
            "id_b": self.id_b,
            "similarity": self.similarity,
            "pair_sha256": pair_sha256(self.fingerprinted_object()),
        }

    def fingerprinted_object(self) -> dict[str, object]:
        """§3 fingerprinted object: {record, candidate, similarity}."""
        return {
            "record": self.record.side(),
            "candidate": self.candidate.side(),
            "similarity": self.similarity,
        }


@dataclass
class HygieneCounters:
    """Counters only — never content (prereg hygiene §3 of data-contract)."""

    candidates_sql: int = 0
    pool: int = 0
    pairs: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def exclude(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @property
    def excluded_total(self) -> int:
        return sum(self.reasons.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates_sql": self.candidates_sql,
            "pool": self.pool,
            "pairs": self.pairs,
            "excluded_total": self.excluded_total,
            "reasons": dict(sorted(self.reasons.items())),
        }


# ── Scanners (secrets + danger), injected — engine imports stay outside ──────


class TextScanner(Protocol):
    """Hygiene scanner over one record's text.

    Returns exclusion REASON names (pattern names only — never matched
    values, prereg «без содержания в логах»). An empty tuple = clean.
    """

    provenance: str

    def scan(self, title: str, body: str, tags: Sequence[str]) -> tuple[str, ...]: ...


class EngineScannerError(RuntimeError):
    """The engine detector tree could not be loaded for read-only reuse."""


class EngineScanner:
    """Scanner over the ENGINE's own detector modules (single source of truth).

    The engine modules are stdlib-only; they are imported from a
    caller-provided engine source tree (a dedicated read-only worktree —
    the primary checkout is never touched). Findings keep pattern names
    and counts only: neither ``SecretFinding.matched_value`` nor any
    matched text crosses this boundary.
    """

    def __init__(self, engine_src: str | Path) -> None:
        src = Path(engine_src).resolve()
        if not (src / "vesmaro" / "secrets_detector.py").is_file():
            raise EngineScannerError(
                f"engine source tree has no vesmaro/secrets_detector.py under {src}"
            )
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from vesmaro.danger_detectors import detect as danger_detect
            from vesmaro.secrets_detector import detect_secrets, findings_by_pattern
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise EngineScannerError(f"engine detector import failed: {exc}") from exc
        self._detect_secrets = detect_secrets
        self._findings_by_pattern = findings_by_pattern
        self._danger_detect = danger_detect
        self.provenance = "engine"

    def scan(self, title: str, body: str, tags: Sequence[str]) -> tuple[str, ...]:
        text = "\n".join([title or "", body or "", " ".join(tags or ())])
        reasons: list[str] = []
        for pattern in self._findings_by_pattern(self._detect_secrets(text)):
            reasons.append(f"secret:{pattern}")
        result = self._danger_detect(text, title=title)
        if result.error is not None:
            # Fail-closed: a scan that could not complete never passes a
            # record through — the record is excluded and counted.
            return ("scanner-error",)
        for finding in result.findings:
            if finding.detector_class == "secret":
                continue  # already covered by detect_secrets above (DRY)
            reasons.append(f"{finding.detector_class}:{finding.pattern_name}")
        return tuple(reasons)


class FallbackScanner:
    """Local regex fallback — used ONLY when the engine tree is unavailable.

    Provenance ``"fallback scanner, not engine"`` MUST surface in every
    report/manifest produced with this scanner (weaker coverage than the
    engine's single source of truth).
    """

    #: Key-shaped patterns (structural, low-noise). Deliberately a SUBSET
    #: of the engine's catalogue — the fallback is a safety net, not a
    #: replacement; provenance always discloses the downgrade.
    _PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
        (name, re.compile(pattern, re.IGNORECASE))
        for name, pattern in (
            ("aws-key", r"AKIA[0-9A-Z]{16}"),
            ("github-token", r"gh[pousr]_[A-Za-z0-9]{36,}"),
            ("gitlab-token", r"glpat-[A-Za-z0-9_\-]{20,}"),
            ("slack-token", r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
            ("openai-key", r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
            ("anthropic-key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
            ("google-api-key", r"AIza[0-9A-Za-z_\-]{35}"),
            ("telegram-bot-token", r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b"),
            ("private-key-block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            ("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
            ("generic-secret-assignment", r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        )
    )

    provenance = "fallback scanner, not engine"

    def scan(self, title: str, body: str, tags: Sequence[str]) -> tuple[str, ...]:
        text = "\n".join([title or "", body or "", " ".join(tags or ())])
        return tuple(
            f"secret:{name}" for name, pattern in self._PATTERNS if pattern.search(text)
        )


def load_engine_scanner(engine_src: str | Path) -> EngineScanner:
    """Load the engine detectors from a read-only source tree.

    Raises:
        EngineScannerError: tree missing/incomplete or import failed —
            callers then decide on the fallback (and must disclose it).
    """
    return EngineScanner(engine_src)


# ── Store access (strictly read-only) ────────────────────────────────────────


class StoreOpenError(ValueError):
    """The store could not be opened read-only (missing files, bad URI)."""


def _ro_connection(db_path: Path) -> sqlite3.Connection:
    """Open one store database strictly read-only, then lock writes off."""
    if not db_path.is_file():
        raise StoreOpenError(f"store database not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise StoreOpenError(f"cannot open {db_path} read-only: {exc}") from exc
    conn.execute("PRAGMA query_only=ON")
    return conn


def resolve_store_databases(
    store_path: str | Path | None, store_uri: str | None
) -> tuple[Path, Path]:
    """Resolve (mnemos.db, vectors.db) paths from CLI-facing arguments.

    ``--store-path`` is the data DIRECTORY holding both databases;
    ``--store-uri`` is the contract-anchored read-only URI of mnemos.db
    (data-contract §6.1) — vectors.db is its sibling. Exactly one must be
    given; a URI without ``mode=ro`` is refused (read-only is not
    negotiable, the store serves a live server).
    """
    if (store_path is None) == (store_uri is None):
        raise StoreOpenError("exactly one of --store-path / --store-uri is required")
    if store_path is not None:
        base = Path(store_path).expanduser().resolve()
        if not base.is_dir():
            raise StoreOpenError(f"store path is not a directory: {base}")
        return base / "mnemos.db", base / "vectors.db"
    assert store_uri is not None
    if "mode=ro" not in store_uri:
        raise StoreOpenError("--store-uri must be a read-only URI (file:...?mode=ro)")
    raw = store_uri
    if raw.startswith("file:"):
        raw = raw[len("file:"):]
    raw = raw.split("?", 1)[0]
    vesma = Path(raw).expanduser().resolve()
    return vesma, vesma.parent / "vectors.db"


_SQL_CANDIDATES: Final[str] = """
SELECT id, title, content, tags, memory_type, created_at, metadata, quarantine_reason
FROM memories
WHERE status = 'published'
  AND memory_type IN ('note', 'snippet', 'fact')
  AND length(content) >= 80
ORDER BY created_at ASC, id ASC
LIMIT :limit
"""

_EMBEDDINGS_SELECT: Final[str] = "SELECT id, vector, metadata FROM embeddings"


def _load_embeddings(vectors_conn: sqlite3.Connection) -> dict[str, tuple[bytes, str]]:
    """Snapshot the vector table (id → blob, metadata) — read once, close."""
    rows = vectors_conn.execute(_EMBEDDINGS_SELECT).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _decode_vector(blob: bytes) -> tuple[tuple[float, ...] | None, str]:
    """Decode a store vector blob; returns (vector, "") or (None, reason)."""
    if len(blob) != VECTOR_BLOB_BYTES:
        return None, "bad-vector-size"
    values = struct.unpack(f"<{EMBEDDING_DIM}f", blob)
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not math.isfinite(norm) or abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        return None, "bad-vector-norm"
    return values, ""


# ── Export pipeline (hygiene ORDER is frozen) ────────────────────────────────

ProgressFn = Callable[[str], None]


#: One memories row as read by the candidate SQL (positional, mirrors the
#: SELECT column order below).
_MemoryRow = tuple


def _compose_record(
    row: _MemoryRow,
    embeddings: dict[str, tuple[bytes, str]],
    counters: HygieneCounters,
    scanner: TextScanner,
) -> StoreRecord | None:
    """Steps 2–3 of the frozen hygiene order for ONE candidate row.

    Per-record hygiene BEFORE any composition: no-federate tag →
    quarantine → secrets/danger scan → vector and content-hash
    validation. Exclusions carry reason names only. Returns ``None`` when
    the record is excluded (the reason is already counted).
    """
    (
        record_id,
        title,
        content,
        tags_json,
        memory_type,
        created_at,
        metadata_json,
        quarantine_reason,
    ) = row
    if _PAIR_ID_SEPARATOR in record_id:
        raise ValueError(
            f"store id contains the pair_id separator {_PAIR_ID_SEPARATOR!r} "
            "(data-contract §2) — refusing the export"
        )
    tags = tuple(str(tag) for tag in json.loads(tags_json or "[]"))
    if NO_FEDERATE_TAG in tags:
        counters.exclude("no-federate")
        return None
    if quarantine_reason:
        counters.exclude("quarantine")
        return None

    embedding = embeddings.get(record_id)
    if embedding is None:
        counters.exclude("missing-embedding")
        return None
    blob, emb_metadata_json = embedding
    emb_metadata = json.loads(emb_metadata_json or "{}")
    fingerprint = emb_metadata.get("model_fingerprint")
    if not fingerprint:
        counters.exclude("unpinned-embedding")
        return None
    content_hash = emb_metadata.get("content_hash")
    if not content_hash:
        counters.exclude("missing-content-hash")
        return None

    metadata = json.loads(metadata_json or "{}")
    canon = metadata.get("canon") if isinstance(metadata.get("canon"), dict) else {}
    language = canon.get("language")
    vector, reason = _decode_vector(blob)
    if vector is None:
        counters.exclude(reason)
        return None

    reasons = scanner.scan(title or "", content or "", tags)
    if reasons:
        for reason in reasons:
            counters.exclude(reason)
        return None

    return StoreRecord(
        id=record_id,
        title=title or "",
        body=content or "",
        tags=tags,
        language=language if isinstance(language, str) else None,
        record_type=memory_type,
        created_at=created_at,
        content_hash=str(content_hash),
        vec_sha256=hashlib.sha256(blob).hexdigest(),
        vector=vector,
    )


def _select_records(
    mnemos_conn: sqlite3.Connection,
    embeddings: dict[str, tuple[bytes, str]],
    counters: HygieneCounters,
    scanner: TextScanner,
    limit: int,
) -> list[StoreRecord]:
    """Steps 1–3 of the frozen hygiene order for the candidate set.

    SQL selection first, then per-record hygiene BEFORE any composition
    (see :func:`_compose_record`). Exclusions carry reason names only.
    """
    counters.candidates_sql = mnemos_conn.execute(
        """
        SELECT count(*) FROM memories
        WHERE status = 'published'
          AND memory_type IN ('note', 'snippet', 'fact')
          AND length(content) >= 80
        """
    ).fetchone()[0]

    records: list[StoreRecord] = []
    for row in mnemos_conn.execute(_SQL_CANDIDATES, {"limit": limit}):
        record = _compose_record(row, embeddings, counters, scanner)
        if record is not None:
            records.append(record)
    return records


# Module-level "current scan context" removed: the pipeline threads the
# scanner and pool limit explicitly (no process-global state).


def _build_pair_bases(
    records: Sequence[StoreRecord], min_cosine: float, max_cosine: float
) -> list[PairBase]:
    """Near-duplicate pair bases over hygiene-passed pool vectors only.

    Half-open band ``[min_cosine, max_cosine)`` (prereg band convention).
    Pairs are unordered-deduplicated; ``a`` is the earlier record by
    ``created_at`` (tie: lexicographically smaller id); the list is sorted
    by ``pair_id`` (data-contract §2 sorting discipline).
    """
    if not records:
        return []
    matrix = np.asarray([record.vector for record in records], dtype=np.float32)
    similarities = matrix @ matrix.T
    order = {record.id: (record.created_at, record.id) for record in records}
    by_id = {record.id: record for record in records}
    ids = [record.id for record in records]
    pairs: list[PairBase] = []
    seen: set[tuple[str, str]] = set()
    n = len(ids)
    for i in range(n):
        row = similarities[i]
        for j in range(i + 1, n):
            similarity = float(row[j])
            if not (min_cosine <= similarity < max_cosine):
                continue
            id_i, id_j = ids[i], ids[j]
            first, second = sorted((id_i, id_j), key=lambda k: order[k])
            pair_id = f"{first}{_PAIR_ID_SEPARATOR}{second}"
            if pair_id in seen:
                continue
            seen.add(pair_id)
            pairs.append(
                PairBase(
                    pair_id=pair_id,
                    id_a=first,
                    id_b=second,
                    similarity=round(similarity, 4),
                    record=by_id[first],
                    candidate=by_id[second],
                )
            )
    pairs.sort(key=lambda pair: pair.pair_id)
    return pairs


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]], progress: ProgressFn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    progress(f"wrote {path} ({len(rows)} rows)")


@dataclass(frozen=True)
class ExportResult:
    """Machine-readable export outcome — counters and fingerprints only."""

    corpus_id: str
    out_dir: Path
    records_path: Path
    pairs_path: Path | None
    records_manifest_path: Path
    pairs_manifest_path: Path | None
    pool_fingerprint: str
    corpus_fingerprint: str | None
    counters: HygieneCounters
    scanner_provenance: str
    embedder_fingerprints: tuple[str, ...]
    sidecar_hint: str
    timings_sec: dict[str, float]

    def summary(self) -> dict[str, object]:
        """JSON summary for stdout (no content, no ids beyond counts)."""
        return {
            "corpus_id": self.corpus_id,
            "out_dir": str(self.out_dir),
            "records": str(self.records_path),
            "pairs": str(self.pairs_path) if self.pairs_path else None,
            "pool_size": self.counters.pool,
            "pairs_count": self.counters.pairs,
            "pool_fingerprint": self.pool_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "hygiene": self.counters.to_dict(),
            "scanner_provenance": self.scanner_provenance,
            "embedder_fingerprints": list(self.embedder_fingerprints),
            "sidecar_hint": self.sidecar_hint,
            "timings_sec": {key: round(value, 3) for key, value in self.timings_sec.items()},
        }


def export_store_corpus(
    *,
    store_path: str | Path | None = None,
    store_uri: str | None = None,
    out_dir: str | Path,
    corpus_id: str,
    scanner: TextScanner,
    limit_pool: int = DEFAULT_LIMIT_POOL,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_cosine: float = DEFAULT_MAX_COSINE,
    with_pairs: bool = True,
    progress: ProgressFn = lambda message: None,
) -> ExportResult:
    """Run the frozen hygiene pipeline and write the A2 corpus files.

    Order (data-contract §6, frozen): SQL selection → per-record hygiene
    scan (no-federate, quarantine, secrets, danger) → minimal composition
    → fingerprints. Content lands ONLY under ``out_dir`` (which must live
    under the gitignored ``data/`` tree); the repo receives counters and
    fingerprints through :meth:`ExportResult.summary`.

    Raises:
        StoreOpenError: store missing or arguments ambiguous.
        ValueError: contract violations (pair-id separator, band order).
    """
    if min_cosine >= max_cosine:
        raise ValueError(f"cosine band must be [min, max): got [{min_cosine}, {max_cosine})")
    if limit_pool <= 0:
        raise ValueError(f"limit_pool must be positive, got {limit_pool}")

    timings: dict[str, float] = {}
    started = time.perf_counter()

    mnemos_path, vectors_path = resolve_store_databases(store_path, store_uri)
    counters = HygieneCounters()
    progress(f"opening store read-only: {mnemos_path}")
    mnemos_conn = _ro_connection(mnemos_path)
    try:
        vectors_conn = _ro_connection(vectors_path)
        try:
            t0 = time.perf_counter()
            embeddings = _load_embeddings(vectors_conn)
            timings["load_embeddings_sec"] = time.perf_counter() - t0
            progress(f"embeddings snapshot: {len(embeddings)} rows")

            t0 = time.perf_counter()
            records = _select_records(mnemos_conn, embeddings, counters, scanner, limit_pool)
            timings["selection_hygiene_sec"] = time.perf_counter() - t0
            counters.pool = len(records)
            progress(
                f"hygiene done: pool={counters.pool} "
                f"excluded={counters.excluded_total} reasons={sorted(counters.reasons)}"
            )
        finally:
            vectors_conn.close()
    finally:
        mnemos_conn.close()

    out_root = Path(out_dir).expanduser().resolve() / "pretrain" / corpus_id

    t0 = time.perf_counter()
    records_rows = [record.pool_row() for record in records]
    records_path = out_root / "records.jsonl"
    _write_jsonl(records_path, records_rows, progress)
    pool_entries = [
        (record.id, pair_sha256({"record": record.pool_fingerprint_payload()}))
        for record in records
    ]
    pool_manifest = out_root / "records_manifest.txt"
    pool_manifest.parent.mkdir(parents=True, exist_ok=True)
    pool_manifest.write_bytes(manifest_bytes(pool_entries))
    pool_fingerprint = corpus_fingerprint(manifest_bytes(pool_entries))
    timings["write_records_sec"] = time.perf_counter() - t0

    pairs_path: Path | None = None
    pairs_manifest_path: Path | None = None
    corpus_fp: str | None = None
    if with_pairs:
        t0 = time.perf_counter()
        pair_bases = _build_pair_bases(records, min_cosine, max_cosine)
        counters.pairs = len(pair_bases)
        pairs_path = out_root / "near_dup_candidates.jsonl"
        _write_jsonl(pairs_path, [pair.row() for pair in pair_bases], progress)
        pairs_manifest_path = out_root / "near_dup_manifest.txt"
        pairs_manifest_path.write_bytes(
            manifest_bytes((pair.pair_id, pair.row()["pair_sha256"]) for pair in pair_bases)
        )
        corpus_fp = corpus_fingerprint(pairs_manifest_path.read_bytes())
        timings["pairs_sec"] = time.perf_counter() - t0
        progress(f"pair bases: {counters.pairs} in [{min_cosine}, {max_cosine})")

    timings["total_sec"] = time.perf_counter() - started

    fingerprint_set: set[str] = set()
    for _, meta in _embeddings_meta_snapshot(vectors_path, {record.id for record in records}):
        pin = json.loads(meta or "{}").get("model_fingerprint")
        if pin:
            fingerprint_set.add(str(pin))
    fingerprints = sorted(fingerprint_set)
    hint = (
        "python scripts/a2_field_cosines.py --records "
        f"{records_path} --pairs {pairs_path if pairs_path else '-'} "
        f"--out-npz <data/vectors/{corpus_id}/field_vecs.npz> --engine-src <engine>/src"
    )
    return ExportResult(
        corpus_id=corpus_id,
        out_dir=Path(out_dir).expanduser().resolve(),
        records_path=records_path,
        pairs_path=pairs_path,
        records_manifest_path=pool_manifest,
        pairs_manifest_path=pairs_manifest_path,
        pool_fingerprint=pool_fingerprint,
        corpus_fingerprint=corpus_fp,
        counters=counters,
        scanner_provenance=scanner.provenance,
        embedder_fingerprints=tuple(fingerprints),
        sidecar_hint=hint,
        timings_sec=timings,
    )


# ── Targeted export: the edge-neighborhood mode (B2 dataset-v3 source) ───────


def load_ids_file(path: str | Path) -> list[str]:
    """Read a seed-id file: one store id per line; blank lines skipped.

    Contract violations refuse the load (data-contract §2: store ids never
    contain the ``--`` separator — it would corrupt pair_id joinability).
    Order is preserved; the caller counts duplicates if it cares.
    """
    ids: list[str] = []
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if _PAIR_ID_SEPARATOR in stripped:
            raise ValueError(
                f"{path}:{line_no}: id contains the pair_id separator "
                f"{_PAIR_ID_SEPARATOR!r} (data-contract §2) — refusing the load"
            )
        ids.append(stripped)
    if not ids:
        raise ValueError(f"{path}: no ids to export")
    return ids


def edge_node_ids(mnemos_conn: sqlite3.Connection) -> list[str]:
    """Distinct node ids touched by memory_edges (sorted, deterministic).

    This is the seed set of the B2 edge-neighborhood export: every node
    that is an endpoint of at least one edge.
    """
    rows = mnemos_conn.execute(
        """
        SELECT DISTINCT id FROM (
            SELECT from_memory_id AS id FROM memory_edges
            UNION
            SELECT to_memory_id AS id FROM memory_edges
        )
        ORDER BY id ASC
        """
    ).fetchall()
    return [row[0] for row in rows]


def expand_one_hop(
    mnemos_conn: sqlite3.Connection, seeds: set[str]
) -> set[str]:
    """Seeds ∪ every endpoint of an edge touching a seed (undirected 1-hop).

    Both endpoints of a touching edge enter the candidate set — this is
    what makes edge coverage well-defined: an edge is exportable iff both
    its ends survived hygiene, and both ends were candidates.
    """
    expanded = set(seeds)
    if not seeds:
        return expanded
    ordered = sorted(seeds)
    for chunk_start in range(0, len(ordered), 500):
        chunk = ordered[chunk_start:chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for src, dst in mnemos_conn.execute(
            f"""
            SELECT from_memory_id, to_memory_id FROM memory_edges
            WHERE from_memory_id IN ({placeholders})
               OR to_memory_id IN ({placeholders})
            """,
            tuple(chunk) + tuple(chunk),
        ):
            expanded.add(src)
            expanded.add(dst)
    return expanded


_SQL_MEMORY_COLUMNS: Final[str] = (
    "SELECT id, title, content, tags, memory_type, created_at, metadata, "
    "quarantine_reason FROM memories"
)


def _select_targeted_records(
    mnemos_conn: sqlite3.Connection,
    embeddings: dict[str, tuple[bytes, str]],
    counters: HygieneCounters,
    scanner: TextScanner,
    id_set: set[str],
    limit: int,
) -> tuple[list[StoreRecord], int]:
    """Targeted candidates → the SAME hygiene chain → (records, trimmed).

    SQL selection by the candidate id set (chunked IN(), deterministic
    created_at/id order — the pool convention). The ``limit`` applies at
    the SQL stage, EXACTLY like the A2 ``_SQL_CANDIDATES`` LIMIT: hygiene
    runs on the limited candidate set, so a limit trim never hides a
    hygiene exclusion — it shrinks the candidate set itself (the trim is
    reported separately, it is not an exclusion reason).
    """
    ordered_ids = sorted(id_set)
    matched: list[_MemoryRow] = []
    for chunk_start in range(0, len(ordered_ids), 500):
        chunk = ordered_ids[chunk_start:chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        matched.extend(
            mnemos_conn.execute(
                f"""
                {_SQL_MEMORY_COLUMNS}
                WHERE status = 'published'
                  AND memory_type IN ('note', 'snippet', 'fact')
                  AND length(content) >= 80
                  AND id IN ({placeholders})
                """,
                tuple(chunk),
            ).fetchall()
        )
    matched.sort(key=lambda row: (row[5], row[0]))  # created_at ASC, id ASC
    counters.candidates_sql = len(matched)

    trimmed = 0
    if len(matched) > limit:
        trimmed = len(matched) - limit
        matched = matched[:limit]

    records: list[StoreRecord] = []
    for row in matched:
        record = _compose_record(row, embeddings, counters, scanner)
        if record is not None:
            records.append(record)
    return records, trimmed


def _count_ids_in_memories(
    mnemos_conn: sqlite3.Connection, ids: list[str]
) -> int:
    """How many of the given ids exist in memories at ALL (any status)."""
    found = 0
    ordered = sorted(set(ids))
    for chunk_start in range(0, len(ordered), 500):
        chunk = ordered[chunk_start:chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        found += mnemos_conn.execute(
            f"SELECT count(*) FROM memories WHERE id IN ({placeholders})",
            tuple(chunk),
        ).fetchone()[0]
    return found


_EdgesRow = tuple


def _load_edges(mnemos_conn: sqlite3.Connection) -> list[_EdgesRow]:
    """All memory_edges rows (from, to, kind, provenance) — read once."""
    return mnemos_conn.execute(
        "SELECT from_memory_id, to_memory_id, kind, provenance FROM memory_edges"
    ).fetchall()


@dataclass(frozen=True)
class SilverEdge:
    """One store edge whose BOTH ends are in the exported pool.

    Graph-silver pair material for dataset v3: ``kind`` + ``provenance``
    ride along; ``pair_id`` follows the §2 convention (``a`` = earlier
    ``created_at``) extended with the kind so one node pair can carry
    several edge rows. Sides are NOT inlined — the verifier joins
    records.jsonl on id_a/id_b and recomputes ``edge_sha256``.
    """

    pair_id: str
    id_a: str
    id_b: str
    id_from: str
    id_to: str
    kind: str
    provenance: str
    record: StoreRecord
    candidate: StoreRecord

    def row(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "id_a": self.id_a,
            "id_b": self.id_b,
            "id_from": self.id_from,
            "id_to": self.id_to,
            "kind": self.kind,
            "provenance": self.provenance,
            "edge_sha256": pair_sha256(self.fingerprinted_object()),
        }

    def fingerprinted_object(self) -> dict[str, object]:
        return {
            "record": self.record.side(),
            "candidate": self.candidate.side(),
            "kind": self.kind,
            "provenance": self.provenance,
        }


def _build_silver_edges(
    edges: Sequence[_EdgesRow],
    records_by_id: dict[str, StoreRecord],
) -> tuple[list[SilverEdge], dict[str, int]]:
    """Edges fully inside the pool → SilverEdge rows (+ coverage counters).

    An edge is silver iff both endpoints are in the pool. Counters
    (counts only, never content): ``edges_total``, ``edges_fully_inside``,
    ``edges_partial`` (exactly one end in the pool), ``edges_absent``
    (both ends absent from the pool — the pool alone cannot attribute an
    absence to "never a candidate" vs "excluded by hygiene"; that
    attribution lives in the hygiene counters, so this stays a bare
    count).
    """
    coverage = {
        "edges_total": len(edges),
        "edges_fully_inside": 0,
        "edges_partial": 0,
        "edges_absent": 0,
    }
    silver: list[SilverEdge] = []
    pair_ids_seen: dict[str, int] = {}
    for id_from, id_to, kind, provenance in edges:
        record = records_by_id.get(id_from)
        candidate = records_by_id.get(id_to)
        if record is None or candidate is None:
            if record is None and candidate is None:
                coverage["edges_absent"] += 1
            else:
                coverage["edges_partial"] += 1
            continue
        coverage["edges_fully_inside"] += 1
        first, second = sorted(
            (id_from, id_to), key=lambda k: (records_by_id[k].created_at, k)
        )
        base_pair_id = f"{first}{_PAIR_ID_SEPARATOR}{second}{_PAIR_ID_SEPARATOR}{kind}"
        seen = pair_ids_seen.get(base_pair_id, 0)
        pair_ids_seen[base_pair_id] = seen + 1
        pair_id = base_pair_id if seen == 0 else f"{base_pair_id}#{seen + 1}"
        silver.append(
            SilverEdge(
                pair_id=pair_id,
                id_a=first,
                id_b=second,
                id_from=id_from,
                id_to=id_to,
                kind=kind,
                provenance=provenance or "",
                record=records_by_id[first],
                candidate=records_by_id[second],
            )
        )
    silver.sort(key=lambda edge: edge.pair_id)
    return silver, coverage


@dataclass(frozen=True)
class TargetedExportResult:
    """Machine-readable outcome of the targeted (edge-neighborhood) export."""

    corpus_id: str
    out_dir: Path
    records_path: Path
    pairs_path: Path | None
    silver_path: Path
    records_manifest_path: Path
    pairs_manifest_path: Path | None
    silver_manifest_path: Path
    coverage_path: Path
    pool_fingerprint: str
    corpus_fingerprint: str | None
    silver_fingerprint: str
    counters: HygieneCounters
    scanner_provenance: str
    embedder_fingerprints: tuple[str, ...]
    seeds: int
    expanded_ids: int
    ids_not_in_memories: int
    trimmed_by_limit: int
    edge_coverage: dict[str, int]
    timings_sec: dict[str, float]

    def summary(self) -> dict[str, object]:
        """JSON summary for stdout (no content, no ids beyond counts)."""
        return {
            "corpus_id": self.corpus_id,
            "out_dir": str(self.out_dir),
            "records": str(self.records_path),
            "pairs": str(self.pairs_path) if self.pairs_path else None,
            "silver_edges": str(self.silver_path),
            "coverage_report": str(self.coverage_path),
            "pool_size": self.counters.pool,
            "pairs_count": self.counters.pairs,
            "silver_edges_count": self.edge_coverage.get("edges_fully_inside", 0),
            "pool_fingerprint": self.pool_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "silver_fingerprint": self.silver_fingerprint,
            "hygiene": self.counters.to_dict(),
            "scanner_provenance": self.scanner_provenance,
            "embedder_fingerprints": list(self.embedder_fingerprints),
            "ids": {
                "seeds": self.seeds,
                "expanded": self.expanded_ids,
                "not_in_memories": self.ids_not_in_memories,
                "trimmed_by_limit": self.trimmed_by_limit,
            },
            "edge_coverage": dict(self.edge_coverage),
            "timings_sec": {key: round(value, 3) for key, value in self.timings_sec.items()},
        }


def export_targeted_corpus(
    *,
    ids_file: str | Path,
    store_path: str | Path | None = None,
    store_uri: str | None = None,
    out_dir: str | Path,
    corpus_id: str,
    scanner: TextScanner,
    limit_pool: int = 1200,
    min_cosine: float = DEFAULT_MIN_COSINE,
    max_cosine: float = DEFAULT_MAX_COSINE,
    with_pairs: bool = True,
    progress: ProgressFn = lambda message: None,
) -> TargetedExportResult:
    """Export the 1-hop edge-neighborhood of the seed ids, hygiene-frozen.

    Same frozen hygiene order as :func:`export_store_corpus`
    (data-contract §6): SQL selection over the candidate id set
    (seeds ∪ 1-hop) → per-record hygiene BEFORE composition → minimal
    export with store vectors → fingerprints. On top of the A2 family
    this writes:

    - ``silver_edges.jsonl`` — every memory_edges row whose BOTH ends are
      in the hygiene-passed pool (kind + provenance preserved; sides
      joinable via records.jsonl; edge_sha256 tamper-evident);
    - ``coverage.json`` — edge-coverage counters + all fingerprints
      (repo-facing handshake material, content-free).

    Raises:
        StoreOpenError: store missing or arguments ambiguous.
        ValueError: contract violations (separator in ids, band order,
            non-positive limit).
    """
    if min_cosine >= max_cosine:
        raise ValueError(f"cosine band must be [min, max): got [{min_cosine}, {max_cosine})")
    if limit_pool <= 0:
        raise ValueError(f"limit_pool must be positive, got {limit_pool}")

    timings: dict[str, float] = {}
    started = time.perf_counter()

    seeds = load_ids_file(ids_file)
    duplicates = len(seeds) - len(set(seeds))
    if duplicates:
        progress(f"ids file carries {duplicates} duplicate lines — collapsing")
    seed_set = set(seeds)

    mnemos_path, vectors_path = resolve_store_databases(store_path, store_uri)
    counters = HygieneCounters()
    progress(f"opening store read-only: {mnemos_path}")
    mnemos_conn = _ro_connection(mnemos_path)
    try:
        ids_not_in_memories = len(seed_set) - _count_ids_in_memories(mnemos_conn, seeds)
        if ids_not_in_memories:
            progress(
                f"{ids_not_in_memories} seed ids match no memories row "
                "(counted; ids themselves are not reported)"
            )

        t0 = time.perf_counter()
        expanded = expand_one_hop(mnemos_conn, seed_set)
        timings["expand_one_hop_sec"] = time.perf_counter() - t0
        progress(f"1-hop expansion: {len(seed_set)} seeds → {len(expanded)} candidate ids")

        vectors_conn = _ro_connection(vectors_path)
        try:
            t0 = time.perf_counter()
            embeddings = _load_embeddings(vectors_conn)
            timings["load_embeddings_sec"] = time.perf_counter() - t0
            progress(f"embeddings snapshot: {len(embeddings)} rows")

            t0 = time.perf_counter()
            records, trimmed = _select_targeted_records(
                mnemos_conn, embeddings, counters, scanner, expanded, limit_pool
            )
            timings["selection_hygiene_sec"] = time.perf_counter() - t0
            counters.pool = len(records)
            progress(
                f"hygiene done: pool={counters.pool} "
                f"excluded={counters.excluded_total} reasons={sorted(counters.reasons)} "
                f"trimmed_by_limit={trimmed}"
            )

            t0 = time.perf_counter()
            edges = _load_edges(mnemos_conn)
            silver, edge_coverage = _build_silver_edges(edges, {r.id: r for r in records})
            timings["silver_edges_sec"] = time.perf_counter() - t0
            progress(
                f"silver edges: {edge_coverage['edges_fully_inside']}/"
                f"{edge_coverage['edges_total']} fully inside the pool"
            )
        finally:
            vectors_conn.close()
    finally:
        mnemos_conn.close()

    out_root = Path(out_dir).expanduser().resolve() / "pretrain" / corpus_id

    t0 = time.perf_counter()
    records_rows = [record.pool_row() for record in records]
    records_path = out_root / "records.jsonl"
    _write_jsonl(records_path, records_rows, progress)
    pool_entries = [
        (record.id, pair_sha256({"record": record.pool_fingerprint_payload()}))
        for record in records
    ]
    pool_manifest = out_root / "records_manifest.txt"
    pool_manifest.parent.mkdir(parents=True, exist_ok=True)
    pool_manifest.write_bytes(manifest_bytes(pool_entries))
    pool_fingerprint = corpus_fingerprint(manifest_bytes(pool_entries))
    timings["write_records_sec"] = time.perf_counter() - t0

    pairs_path: Path | None = None
    pairs_manifest_path: Path | None = None
    corpus_fp: str | None = None
    if with_pairs:
        t0 = time.perf_counter()
        pair_bases = _build_pair_bases(records, min_cosine, max_cosine)
        counters.pairs = len(pair_bases)
        pairs_path = out_root / "near_dup_candidates.jsonl"
        _write_jsonl(pairs_path, [pair.row() for pair in pair_bases], progress)
        pairs_manifest_path = out_root / "near_dup_manifest.txt"
        pairs_manifest_path.write_bytes(
            manifest_bytes((pair.pair_id, pair.row()["pair_sha256"]) for pair in pair_bases)
        )
        corpus_fp = corpus_fingerprint(pairs_manifest_path.read_bytes())
        timings["pairs_sec"] = time.perf_counter() - t0
        progress(f"pair bases: {counters.pairs} in [{min_cosine}, {max_cosine})")

    t0 = time.perf_counter()
    silver_rows = [edge.row() for edge in silver]
    silver_path = out_root / "silver_edges.jsonl"
    _write_jsonl(silver_path, silver_rows, progress)
    silver_manifest = out_root / "silver_edges_manifest.txt"
    silver_manifest.write_bytes(
        manifest_bytes(
            (edge.pair_id, edge.row()["edge_sha256"]) for edge in silver
        )
    )
    silver_fingerprint = corpus_fingerprint(silver_manifest.read_bytes())
    timings["write_silver_sec"] = time.perf_counter() - t0

    timings["total_sec"] = time.perf_counter() - started

    fingerprint_set: set[str] = set()
    for _, meta in _embeddings_meta_snapshot(vectors_path, {record.id for record in records}):
        pin = json.loads(meta or "{}").get("model_fingerprint")
        if pin:
            fingerprint_set.add(str(pin))
    fingerprints = sorted(fingerprint_set)

    coverage_payload: dict[str, object] = {
        "corpus_id": corpus_id,
        "scanner_provenance": scanner.provenance,
        "embedder_fingerprints": fingerprints,
        "ids": {
            "seeds": len(seed_set),
            "duplicates_in_file": duplicates,
            "expanded_candidates": len(expanded),
            "not_in_memories": ids_not_in_memories,
            "trimmed_by_limit": trimmed,
        },
        "hygiene": counters.to_dict(),
        "edge_coverage": dict(edge_coverage),
        "pool_fingerprint": pool_fingerprint,
        "corpus_fingerprint": corpus_fp,
        "silver_fingerprint": silver_fingerprint,
        "records": str(records_path),
        "silver_edges": str(silver_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    coverage_path = out_root / "coverage.json"
    coverage_path.write_text(
        json.dumps(coverage_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress(f"wrote {coverage_path}")

    hint = (
        "python scripts/a2_field_cosines.py --records "
        f"{records_path} --pairs {pairs_path if pairs_path else '-'} "
        f"--out-npz <data/vectors/{corpus_id}/field_vecs.npz> --engine-src <engine>/src"
    )
    return TargetedExportResult(
        corpus_id=corpus_id,
        out_dir=Path(out_dir).expanduser().resolve(),
        records_path=records_path,
        pairs_path=pairs_path,
        silver_path=silver_path,
        records_manifest_path=pool_manifest,
        pairs_manifest_path=pairs_manifest_path,
        silver_manifest_path=silver_manifest,
        coverage_path=coverage_path,
        pool_fingerprint=pool_fingerprint,
        corpus_fingerprint=corpus_fp,
        silver_fingerprint=silver_fingerprint,
        counters=counters,
        scanner_provenance=scanner.provenance,
        embedder_fingerprints=tuple(fingerprints),
        seeds=len(seed_set),
        expanded_ids=len(expanded),
        ids_not_in_memories=ids_not_in_memories,
        trimmed_by_limit=trimmed,
        edge_coverage=edge_coverage,
        timings_sec=timings,
    )


def _embeddings_meta_snapshot(
    vectors_path: Path, ids: set[str]
) -> list[tuple[str, str]]:
    """Re-read ONLY the metadata of the exported ids (embedder pin provenance).

    Second short read-only open — the exporter has already closed its
    connections; this keeps fingerprint provenance without retaining
    vector blobs in memory.
    """
    if not ids:
        return []
    conn = _ro_connection(vectors_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, metadata FROM embeddings WHERE id IN ({placeholders})",
            tuple(sorted(ids)),
        ).fetchall()
        return [(row[0], row[1] or "{}") for row in rows]
    finally:
        conn.close()


#: Environment override used by the CLI (--engine-src wins over it).
ENGINE_SRC_ENV: Final[str] = "CORTEX_ENGINE_SRC"

__all__.append("ENGINE_SRC_ENV")

#: Re-export for the CLI: build the scanner with a disclosed fallback.
def make_scanner(engine_src: str | Path | None) -> TextScanner:
    """Engine scanner when a source tree is given and loads; else fallback.

    The fallback is LOUD: callers must surface the provenance string in
    every report (prereg hygiene — the reader must know the scan was not
    the engine's single source of truth).
    """
    if engine_src is None:
        engine_src = os.environ.get(ENGINE_SRC_ENV)
    if engine_src:
        try:
            return load_engine_scanner(engine_src)
        except EngineScannerError as exc:
            progress_warn = f"engine scanner unavailable ({exc}); using fallback scanner"
            print(progress_warn, file=sys.stderr)
    return FallbackScanner()
