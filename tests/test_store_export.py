"""A2 store export tests — hygiene chain, read-only discipline, fingerprints.

The REAL store is never touched here: every test builds a throwaway
mock store (mnemos.db + vectors.db, same schema surface the exporter
reads) in tmp_path. The synthetic corpus deliberately includes records
that MUST be excluded with counters:

- an AWS key in the body (secret scanner hit);
- a ``mnemos:no-federate`` tag;
- a non-empty ``quarantine_reason``;
- an unpinned embedding (no ``model_fingerprint``);
- a too-short record (< 80 chars), a non-published record, a wrong-type
  record, a record with no embedding, a non-unit-norm vector.

Engine-detector integration is exercised ONLY when a read-only engine
source tree exists on this machine (skipped otherwise) — CI runs the
fallback-scanner path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from cortex.data.store_export import (
    DEFAULT_LIMIT_POOL,
    NO_FEDERATE_TAG,
    EngineScannerError,
    ExportResult,
    FallbackScanner,
    HygieneCounters,
    StoreRecord,
    export_store_corpus,
    make_scanner,
    resolve_store_databases,
)

ENGINE_SRC = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/wt/a2-engine-readonly/src"
)

_PIN = "nano:sha256:" + "ab" * 32

# ── mock store fixture ────────────────────────────────────────────────────────


def _unit_vec(seed: int = 0) -> bytes:
    """Deterministic float32 unit vector from a seed (rng, zero-mean)."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(384).astype(np.float32)
    return struct.pack("<384f", *((values / np.linalg.norm(values)).tolist()))


def _rotated_unit_vec(seed: int, base_seed: int, cosine_target: float) -> bytes:
    """Unit vector at exactly ``cosine_target`` from the base vector.

    v = cosθ·u + sinθ·w where u is the base unit vector and w a unit
    vector in an independent random direction orthogonalized against u —
    so cos(v, u) = cosθ exactly, and vectors rotated in DIFFERENT planes
    stay mutually far apart (cos(v2, v3) = cosθ2·cosθ3).
    """
    rng_base = np.random.default_rng(base_seed)
    u = rng_base.standard_normal(384).astype(np.float32)
    u /= np.linalg.norm(u)
    rng_w = np.random.default_rng(seed + 10_000)
    w = rng_w.standard_normal(384).astype(np.float32)
    w -= float(np.dot(w, u)) * u
    w /= np.linalg.norm(w)
    theta = math.acos(min(1.0, max(-1.0, cosine_target)))
    v = math.cos(theta) * u + math.sin(theta) * w
    return struct.pack("<384f", *(v.astype(np.float32).tolist()))


def _memory_row(
    record_id: str,
    title: str,
    body: str,
    *,
    tags: list[str] | None = None,
    memory_type: str = "note",
    status: str = "published",
    created_at: str = "2026-09-01T10:00:00.000000+00:00",
    quarantine_reason: str | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "id": record_id,
        "title": title,
        "content": body,
        "tags": json.dumps(tags or []),
        "memory_type": memory_type,
        "created_at": created_at,
        "metadata": json.dumps(metadata or {}),
        "status": status,
        "quarantine_reason": quarantine_reason,
    }


def build_mock_store(store_dir: Path) -> None:
    """Create mnemos.db + vectors.db with the synthetic hygiene corpus.

    Seeded classes (tests assert exact counter outcomes):
    4 good records (one near-dup pair at cosine 0.90, one at 0.60 — out
    of band), 1 aws-key secret, 1 no-federate, 1 quarantined, 3
    SQL-filtered (short / draft / conversation), 1 no-embedding, 1
    unpinned embedding, 1 non-unit-norm vector.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    vesma = store_dir / "mnemos.db"
    vectors = store_dir / "vectors.db"
    m = sqlite3.connect(vesma)
    v = sqlite3.connect(vectors)
    m.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, title TEXT,
            tags TEXT NOT NULL DEFAULT '[]', memory_type TEXT NOT NULL DEFAULT 'note',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'raw',
            embedding_id TEXT, quarantine_reason TEXT
        )
        """
    )
    v.execute(
        """
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY, vector BLOB NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )

    good = [
        _memory_row(
            "id-good-1",
            "Python decorators",
            "A " + ("long enough body " * 6),
            tags=["python"],
        ),
        _memory_row(
            "id-good-2",
            "Python decorators notes",
            "A " + ("long enough body " * 6) + "plus a bit more",
            tags=["python"],
            created_at="2026-09-02T10:00:00.000000+00:00",
        ),
        _memory_row(
            "id-good-3",
            "Rust lifetimes",
            "Rust " + ("lifetime notes " * 8),
            memory_type="snippet",
            metadata={"canon": {"language": "ru"}},
        ),
        _memory_row(
            "id-good-4",
            "Totally different topic",
            "Cooking " + ("pasta recipes " * 8),
            memory_type="fact",
        ),
    ]
    secrets = [
        # AWS access key id — engine detector pattern (and fallback pattern).
        _memory_row(
            "id-secret", "creds", "my key AKIAIOSFODNN7EXAMPLE inside a long body " * 2
        ),
    ]
    no_federate = [
        _memory_row(
            "id-nofederate",
            "private note",
            "long enough private body " * 6,
            tags=[NO_FEDERATE_TAG],
        ),
    ]
    quarantined = [
        _memory_row(
            "id-quarantine",
            "bad record",
            "long enough quarantined body " * 6,
            quarantine_reason="danger-detector",
        ),
    ]
    excluded_sql = [
        _memory_row("id-short", "tiny", "too short"),
        _memory_row("id-draft", "draft", "long enough draft body " * 8, status="draft"),
        _memory_row(
            "id-conversation",
            "chat",
            "long enough chat body " * 8,
            memory_type="conversation",
        ),
    ]
    rows = good + secrets + no_federate + quarantined + excluded_sql
    for row in rows:
        m.execute(
            "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
            "VALUES (:id, :content, :title, :tags, :memory_type, :created_at, :created_at, :metadata, :status, :id, :quarantine_reason)",
            row,
        )
        v.execute(
            "INSERT INTO embeddings (id, vector, metadata) VALUES (:id, :vector, :metadata)",
            {
                "id": row["id"],
                "vector": _unit_vec(zlib.crc32(row["id"].encode()) % 1000),
                "metadata": json.dumps(
                    {"model_fingerprint": _PIN, "content_hash": "hash-" + row["id"]}
                ),
            },
        )
    # near-duplicate pair inside the [0.85, 0.97) band: good-2 at cosine
    # 0.90 from good-1 (different rotation plane than good-3 at 0.60).
    v.execute(
        "UPDATE embeddings SET vector = ? WHERE id = 'id-good-1'", (_unit_vec(1),)
    )
    v.execute(
        "UPDATE embeddings SET vector = ? WHERE id = 'id-good-2'",
        (_rotated_unit_vec(2, 1, 0.90),),
    )
    v.execute(
        "UPDATE embeddings SET vector = ? WHERE id = 'id-good-3'",
        (_rotated_unit_vec(3, 1, 0.60),),
    )
    # no embedding at all
    m.execute(
        "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
        "VALUES ('id-noemb', 'long enough body without vectors long enough body without vectors long enough body without vectors ', 'noemb', '[]', 'note', '2026-09-03T10:00:00.000000+00:00', '2026-09-03T10:00:00.000000+00:00', '{}', 'published', NULL, NULL)"
    )
    # unpinned embedding (legacy row)
    m.execute(
        "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
        "VALUES ('id-unpinned', 'long enough legacy body long enough legacy body long enough legacy body long enough legacy body long enough legacy body long enough legacy body ', 'legacy', '[]', 'note', '2026-09-04T10:00:00.000000+00:00', '2026-09-04T10:00:00.000000+00:00', '{}', 'published', 'id-unpinned', NULL)"
    )
    v.execute(
        "INSERT INTO embeddings (id, vector, metadata) VALUES ('id-unpinned', ?, '{}')",
        (_unit_vec(4),),
    )
    # non-unit-norm vector
    m.execute(
        "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
        "VALUES ('id-badnorm', 'long enough anomalous body long enough anomalous body long enough anomalous body long enough anomalous body long enough anomalous body long enough anomalous body ', 'badnorm', '[]', 'note', '2026-09-05T10:00:00.000000+00:00', '2026-09-05T10:00:00.000000+00:00', '{}', 'published', 'id-badnorm', NULL)"
    )
    v.execute(
        "INSERT INTO embeddings (id, vector, metadata) VALUES ('id-badnorm', ?, ?)",
        (
            struct.pack("<384f", *([0.5] * 384)),
            json.dumps({"model_fingerprint": _PIN, "content_hash": "hash-badnorm"}),
        ),
    )
    m.commit()
    v.commit()
    counts = m.execute("SELECT count(*) FROM memories").fetchone()[0]
    m.close()
    v.close()
    assert counts == len(rows) + 3


@pytest.fixture()
def mock_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    build_mock_store(store)
    return store


@pytest.fixture()
def fallback_export(mock_store: Path, tmp_path: Path) -> ExportResult:
    out = tmp_path / "data"
    return export_store_corpus(
        store_path=mock_store,
        out_dir=out,
        corpus_id="pretrain-test",
        scanner=FallbackScanner(),
        limit_pool=DEFAULT_LIMIT_POOL,
    )


# ── read-only discipline ──────────────────────────────────────────────────────


def test_resolve_store_requires_ro_uri(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode=ro"):
        resolve_store_databases(None, f"file:{tmp_path}/mnemos.db")


def test_resolve_store_path_and_uri_agree(tmp_path: Path) -> None:
    from_path = resolve_store_databases(tmp_path, None)
    from_uri = resolve_store_databases(
        None, f"file:{(tmp_path / 'mnemos.db').as_posix()}?mode=ro"
    )
    assert from_path == from_uri
    with pytest.raises(ValueError, match="exactly one"):
        resolve_store_databases(tmp_path, "file:x?mode=ro")


def test_store_opened_query_only(mock_store: Path) -> None:
    """Both databases open with mode=ro + PRAGMA query_only — writes fail."""
    from cortex.data.store_export import _ro_connection

    for name in ("mnemos.db", "vectors.db"):
        conn = _ro_connection(mock_store / name)
        try:
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(
                    "UPDATE memories SET title = 'tampered' WHERE id = 'id-good-1'"
                )
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM embeddings")
        finally:
            conn.close()


def test_store_bytes_unchanged_by_export(
    mock_store: Path, tmp_path: Path, fallback_export: ExportResult
) -> None:
    """The export leaves the store byte-identical (live-server discipline)."""
    before = {p.name: p.read_bytes() for p in sorted(mock_store.iterdir())}
    export_store_corpus(
        store_path=mock_store,
        out_dir=tmp_path / "data2",
        corpus_id="pretrain-test2",
        scanner=FallbackScanner(),
    )
    after = {p.name: p.read_bytes() for p in sorted(mock_store.iterdir())}
    assert before == after


# ── hygiene chain (order frozen: scan BEFORE composition) ────────────────────


def test_hygiene_exclusions_with_counters(fallback_export: ExportResult) -> None:
    counters = fallback_export.counters
    # 13 seeded rows − short/draft/conversation filtered by SQL = 10 candidates
    assert counters.candidates_sql == 10
    # pool: exactly the 4 good records
    assert counters.pool == 4
    reasons = counters.reasons
    assert reasons["no-federate"] == 1
    assert reasons["quarantine"] == 1
    assert reasons["missing-embedding"] == 1
    assert reasons["unpinned-embedding"] == 1
    assert reasons["bad-vector-norm"] == 1
    assert counters.excluded_total == 6  # 10 candidates − 4 pool
    assert fallback_export.scanner_provenance == "fallback scanner, not engine"


def test_secret_record_excluded_by_fallback(
    fallback_export: ExportResult, tmp_path: Path
) -> None:
    assert fallback_export.counters.reasons.get("secret:aws-key") == 1
    # no raw content of the excluded record may leak into the pool file
    pool_text = (
        tmp_path / "data" / "pretrain" / "pretrain-test" / "records.jsonl"
    ).read_text(encoding="utf-8")
    assert "AKIAIOSFODNN7EXAMPLE" not in pool_text
    assert "id-secret" not in pool_text


def test_pool_composition_and_engine_scanner_path(
    mock_store: Path, tmp_path: Path
) -> None:
    if not (ENGINE_SRC / "vesmaro" / "secrets_detector.py").is_file():
        pytest.skip("engine read-only worktree not available on this machine")
    scanner = make_scanner(ENGINE_SRC)
    assert scanner.provenance == "engine"
    result = export_store_corpus(
        store_path=mock_store,
        out_dir=tmp_path / "data",
        corpus_id="pretrain-test",
        scanner=scanner,
    )
    assert result.counters.pool == 4
    # the mocked aws key must hit the ENGINE detector (not just the fallback)
    assert result.counters.reasons.get("secret:aws-key") == 1
    assert result.embedder_fingerprints == (_PIN,)


def test_pool_rows_carry_minimal_composition_and_vec(
    fallback_export: ExportResult, tmp_path: Path
) -> None:
    records_path = tmp_path / "data" / "pretrain" / "pretrain-test" / "records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    row = rows[0]
    assert set(row) == {
        "id",
        "title",
        "body",
        "tags",
        "language",
        "record_type",
        "created_at",
        "content_hash",
        "vec_sha256",
        "vec",
    }
    assert len(row["vec"]) == 384
    norm = math.sqrt(sum(v * v for v in row["vec"]))
    assert abs(norm - 1.0) < 1e-3
    assert {r["id"] for r in rows} == {
        "id-good-1",
        "id-good-2",
        "id-good-3",
        "id-good-4",
    }
    record_with_lang = next(r for r in rows if r["id"] == "id-good-3")
    assert record_with_lang["language"] == "ru"
    assert record_with_lang["record_type"] == "snippet"


# ── near-duplicate pair bases ─────────────────────────────────────────────────


def test_near_dup_pair_bands_and_ordering(
    fallback_export: ExportResult, tmp_path: Path
) -> None:
    pairs_path = (
        tmp_path / "data" / "pretrain" / "pretrain-test" / "near_dup_candidates.jsonl"
    )
    rows = [
        json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert fallback_export.counters.pairs == len(rows) == 1
    pair = rows[0]
    assert pair["pair_id"] == "id-good-1--id-good-2"  # a = earlier created_at
    assert 0.85 <= pair["similarity"] < 0.97
    assert pair["id_a"] == "id-good-1" and pair["id_b"] == "id-good-2"
    # lean rows: no side inlining, no label/stratum/source (unlabeled bases)
    assert set(pair) == {"pair_id", "id_a", "id_b", "similarity", "pair_sha256"}
    assert "--" not in pair["id_a"] and "--" not in pair["id_b"]


def test_pair_sha256_verifiable_from_records_join(
    fallback_export: ExportResult, tmp_path: Path
) -> None:
    """Verifier path: rebuild the §3 sides from records.jsonl and recompute
    pair_sha256 — the lean row stays tamper-evident."""
    from cortex.data.fingerprints import pair_sha256 as fp

    corpus_dir = tmp_path / "data" / "pretrain" / "pretrain-test"
    records = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (corpus_dir / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    for pair in (
        json.loads(line)
        for line in (corpus_dir / "near_dup_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):

        def side(record_id: str) -> dict:
            row = records[record_id]
            return {
                key: row[key]
                for key in (
                    "title",
                    "body",
                    "tags",
                    "language",
                    "record_type",
                    "created_at",
                )
            }

        rebuilt = {
            "record": side(pair["id_a"]),
            "candidate": side(pair["id_b"]),
            "similarity": pair["similarity"],
        }
        assert fp(rebuilt) == pair["pair_sha256"]


def test_pair_fingerprints_reproducible_across_runs(
    mock_store: Path, tmp_path: Path
) -> None:
    outs = []
    for i in (1, 2):
        result = export_store_corpus(
            store_path=mock_store,
            out_dir=tmp_path / f"data{i}",
            corpus_id="pretrain-test",
            scanner=FallbackScanner(),
        )
        outs.append(result)
    first, second = outs
    assert first.pool_fingerprint == second.pool_fingerprint
    assert first.corpus_fingerprint == second.corpus_fingerprint
    manifest = (
        tmp_path / "data1" / "pretrain" / "pretrain-test" / "near_dup_manifest.txt"
    ).read_bytes()
    assert (
        manifest
        == (
            tmp_path / "data2" / "pretrain" / "pretrain-test" / "near_dup_manifest.txt"
        ).read_bytes()
    )
    # manifest scheme: sorted "pair_id <sha256>" lines, \n-terminated (§5)
    lines = manifest.decode("utf-8").splitlines()
    assert lines == sorted(lines) and manifest.endswith(b"\n")


def test_tampered_pool_breaks_fingerprint(mock_store: Path, tmp_path: Path) -> None:
    first = export_store_corpus(
        store_path=mock_store,
        out_dir=tmp_path / "d1",
        corpus_id="pretrain-test",
        scanner=FallbackScanner(),
    )
    # touch a memory's body → a new store vintage → fingerprint MUST move
    conn = sqlite3.connect(mock_store / "mnemos.db")
    conn.execute("UPDATE memories SET content = content || ' x' WHERE id = 'id-good-1'")
    conn.commit()
    conn.close()
    second = export_store_corpus(
        store_path=mock_store,
        out_dir=tmp_path / "d2",
        corpus_id="pretrain-test",
        scanner=FallbackScanner(),
    )
    assert first.pool_fingerprint != second.pool_fingerprint


def test_store_id_with_separator_refuses_export(
    mock_store: Path, tmp_path: Path
) -> None:
    conn = sqlite3.connect(mock_store / "mnemos.db")
    conn.execute(
        "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
        "VALUES ('id--bad', 'long enough body for the separator case long enough body for the separator case long enough body for the separator case ', 'bad', '[]', 'note', '2026-09-06T10:00:00.000000+00:00', '2026-09-06T10:00:00.000000+00:00', '{}', 'published', NULL, NULL)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="separator"):
        export_store_corpus(
            store_path=mock_store,
            out_dir=tmp_path / "d3",
            corpus_id="pretrain-test",
            scanner=FallbackScanner(),
        )


# ── CLI contract ──────────────────────────────────────────────────────────────


def test_cli_export_corpus_happy_path(
    mock_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cortex.cli.main import main

    out = tmp_path / "data"
    code = main(
        [
            "export-corpus",
            "--store-path",
            str(mock_store),
            "--out-dir",
            str(out),
            "--corpus-id",
            "pretrain-test",
            "--limit-pool",
            "800",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["pool_size"] == 4
    assert summary["hygiene"]["reasons"]["no-federate"] == 1
    assert summary["corpus_fingerprint"]
    assert summary["scanner_provenance"] == "fallback scanner, not engine"
    assert captured.err.strip(), "progress must be on stderr"


def test_cli_export_corpus_rejects_bad_args(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cortex.cli.main import main

    code = main(
        [
            "export-corpus",
            "--store-path",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "o"),
            "--min-cosine",
            "0.97",
            "--max-cosine",
            "0.85",
        ]
    )
    assert code == 2
    assert "min-cosine" in capsys.readouterr().err

    code = main(
        [
            "export-corpus",
            "--store-path",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "o"),
            "--corpus-id",
            "../escape",
        ]
    )
    assert code == 2
    assert "corpus-id" in capsys.readouterr().err


# ── engine scanner loader ─────────────────────────────────────────────────────


def test_engine_scanner_missing_tree_raises(tmp_path: Path) -> None:
    from cortex.data.store_export import load_engine_scanner

    with pytest.raises(EngineScannerError):
        load_engine_scanner(tmp_path / "no-such-tree")


def test_no_federate_tag_pinned_to_engine() -> None:
    """The repeated literal must match the engine's NO_FEDERATE_TAG when
    the read-only tree is available (single source of truth, DRY guard).

    Text-level check on purpose: engine models.py imports pydantic, which
    the cortex environment deliberately does not carry — read-only reuse
    must not force engine deps into this library."""
    import re

    models_py = ENGINE_SRC / "vesmaro" / "models.py"
    if not models_py.is_file():
        pytest.skip("engine read-only worktree not available on this machine")
    match = re.search(
        r'NO_FEDERATE_TAG(?::\s*str)?\s*=\s*"([^"]+)"',
        models_py.read_text(encoding="utf-8"),
    )
    assert match, "engine models.py lost the NO_FEDERATE_TAG constant"
    assert match.group(1) == NO_FEDERATE_TAG


# ── hygiene counters unit ─────────────────────────────────────────────────────


def test_hygiene_counters_shape() -> None:
    counters = HygieneCounters()
    counters.candidates_sql = 10
    counters.exclude("quarantine")
    counters.exclude("quarantine")
    counters.exclude("no-federate")
    counters.pool = 7
    shape = counters.to_dict()
    assert shape == {
        "candidates_sql": 10,
        "pool": 7,
        "pairs": 0,
        "excluded_total": 3,
        "reasons": {"no-federate": 1, "quarantine": 2},
    }


def test_store_record_side_composition() -> None:
    record = StoreRecord(
        id="x",
        title="t",
        body="b",
        tags=("a",),
        language="ru",
        record_type="note",
        created_at="2026-09-01T00:00:00Z",
        content_hash="h",
        vec_sha256="v",
        vector=(1.0,) * 384,
    )
    assert record.side() == {
        "title": "t",
        "body": "b",
        "tags": ["a"],
        "language": "ru",
        "record_type": "note",
        "created_at": "2026-09-01T00:00:00Z",
    }


# ── field-cosine sidecar reader (A2 docompute consumer) ───────────────────────


def _write_sidecar_npz(
    path: Path, ids: list[str], fp: str = "nano:sha256:ff" * 8
) -> None:
    rng = np.random.default_rng(7)
    title = rng.standard_normal((len(ids), 384)).astype(np.float32)
    title /= np.linalg.norm(title, axis=1, keepdims=True)
    body = rng.standard_normal((len(ids), 384)).astype(np.float32)
    body /= np.linalg.norm(body, axis=1, keepdims=True)
    tags = rng.standard_normal((len(ids), 384)).astype(np.float32)
    tags /= np.linalg.norm(tags, axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.asarray(ids),
        title_vecs=title,
        body_vecs=body,
        tags_vecs=tags,
        embedder_fingerprint=np.asarray(fp),
        embedder_model=np.asarray("mnema-embed-v1"),
        corpus_id=np.asarray("pretrain-test"),
    )


def test_field_sidecar_cosines_and_attach(tmp_path: Path) -> None:
    from cortex.data.field_cosines import (
        FieldSidecar,
        FieldSidecarError,
        attach_to_features,
    )
    from cortex.features.pair import (
        FEATURE_NAMES,
        FIELD_COSINE_FEATURES,
        PairRecord,
        features,
    )

    npz = tmp_path / "vectors" / "pretrain-test" / "field_vecs.npz"
    _write_sidecar_npz(npz, ["id-a", "id-b"], fp="nano:sha256:" + "ff" * 32)
    sidecar = FieldSidecar(npz)
    assert sidecar.dimension == 384
    assert sidecar.fingerprint == "nano:sha256:" + "ff" * 32

    # identical id → cosine 1.0 on every field
    assert sidecar.cosines_for_pair("id-a", "id-a") == pytest.approx(
        (1.0, 1.0, 1.0), abs=1e-6
    )
    cos_title, cos_body, cos_tags = sidecar.cosines_for_pair("id-a", "id-b")
    for value in (cos_title, cos_body, cos_tags):
        assert -1.0 <= value <= 1.0

    # attach-ready: core feature vector + sidecar → ablation variant
    core = features(PairRecord("t", "b"), PairRecord("t", "b"), similarity=1.0)
    assert core.names == FEATURE_NAMES
    ablation = attach_to_features(core, sidecar, "id-a", "id-b")
    assert ablation.names == FEATURE_NAMES + FIELD_COSINE_FEATURES
    assert ablation.values[-3:] == (cos_title, cos_body, cos_tags)

    with pytest.raises(FieldSidecarError, match="not in field sidecar"):
        sidecar.cosines_for_pair("id-a", "id-missing")
    with pytest.raises(FieldSidecarError, match="not found"):
        FieldSidecar(tmp_path / "missing.npz")


def test_field_sidecar_rejects_ragged_arrays(tmp_path: Path) -> None:
    from cortex.data.field_cosines import FieldSidecar, FieldSidecarError

    npz = tmp_path / "ragged.npz"
    np.savez(
        npz,
        ids=np.asarray(["a", "b"]),
        title_vecs=np.zeros((2, 384), dtype=np.float32),
        body_vecs=np.zeros((1, 384), dtype=np.float32),
        tags_vecs=np.zeros((2, 384), dtype=np.float32),
        embedder_fingerprint=np.asarray("nano:sha256:x"),
    )
    with pytest.raises(FieldSidecarError, match="ragged"):
        FieldSidecar(npz)


def test_field_sidecar_clamps_float32_overshoot(tmp_path: Path) -> None:
    """Near-collinear float32 unit vectors dot to 1.0 + ~1e-7 — the reader
    must clamp so attach_field_cosines ([-1, 1] validation) accepts it."""
    from cortex.data.field_cosines import FieldSidecar, attach_to_features
    from cortex.features.pair import PairRecord, features

    base = np.random.default_rng(3).standard_normal(384).astype(np.float32)
    base /= np.linalg.norm(base)
    npz = tmp_path / "vectors" / "c" / "field_vecs.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz,
        ids=np.asarray(["a", "b"]),
        title_vecs=np.stack([base, base]),  # identical → float32 overshoot
        body_vecs=np.stack([base, base]),
        tags_vecs=np.stack([base, base]),
        embedder_fingerprint=np.asarray("nano:sha256:x"),
    )
    sidecar = FieldSidecar(npz)
    cosines = sidecar.cosines_for_pair("a", "b")
    assert all(-1.0 <= value <= 1.0 for value in cosines)
    attach_to_features(
        features(PairRecord("t", "b"), PairRecord("t", "b"), similarity=0.9),
        sidecar,
        "a",
        "b",
    )
