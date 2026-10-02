"""B2 targeted export tests — edge-neighborhood mode of the A2 exporter.

The targeted mode (``export_targeted_corpus``) must preserve the frozen
A2 hygiene chain EXACTLY (same per-record exclusions, same counters,
same fingerprints schemes) while changing only the candidate SET: the
1-hop edge neighborhood of a seed-id list over ``memory_edges``. These
tests pin:

- 1-hop expansion and the seed-file contract (blank lines, ``--`` refusal);
- hygiene exclusions INSIDE the targeted selection (secret / no-federate /
  missing-embedding edge endpoints drop out with counters);
- silver edges = edges fully inside the pool, kind + provenance carried,
  edge_sha256 verifiable from a records.jsonl join;
- edge-coverage counters (fully inside / partial / absent);
- read-only discipline (byte-identical store, query_only);
- fingerprint reproducibility and the deterministic limit trim.

The REAL store is never touched: a throwaway mock store with the same
schema surface (memories + memory_edges + embeddings) is built in
tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from cortex.data.fingerprints import pair_sha256
from cortex.data.store_export import (
    FallbackScanner,
    expand_one_hop,
    export_targeted_corpus,
    export_store_corpus,
    load_ids_file,
)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "b2_targeted_export.py"
_spec = importlib.util.spec_from_file_location("b2_targeted_export", _SCRIPT)
b2_script = importlib.util.module_from_spec(_spec)
sys.modules["b2_targeted_export"] = b2_script
_spec.loader.exec_module(b2_script)

_PIN = "nano:sha256:" + "cd" * 32

_LONG = "long enough body for the graph fixture "


# ── mock graph store ──────────────────────────────────────────────────────────


def _unit_vec(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(384).astype(np.float32)
    return struct.pack("<384f", *((values / np.linalg.norm(values)).tolist()))


def _rotated_unit_vec(seed: int, base_seed: int, cosine_target: float) -> bytes:
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


# Topology (all records are published notes with long enough bodies so
# they reach the per-record hygiene chain):
#   good nodes:      n-a < n-b < n-c < n-d (created_at order), n-x (non-graph)
#   hygiene drops:   n-secret (aws key), n-nofed (no-federate), n-noemb (no embedding)
#   edges:  e1 n-a→n-b relates_to · e2 n-b→n-c relates_to · e3 n-c→n-d supersedes
#           e4 n-a→n-secret · e5 n-b→n-nofed · e6 n-c→n-noemb · e7 n-secret→n-nofed
#   expected pool: {n-a, n-b, n-c, n-d}; coverage: 3 inside / 3 partial / 1 absent
_TOPOLOGY = {
    "n-a": {"day": "2026-09-01", "vec": _unit_vec(1)},
    "n-b": {"day": "2026-09-02", "vec": _rotated_unit_vec(2, 1, 0.90)},
    "n-c": {"day": "2026-09-03", "vec": _rotated_unit_vec(3, 1, 0.60)},
    "n-d": {"day": "2026-09-04", "vec": _unit_vec(4)},
    "n-secret": {"day": "2026-09-05", "vec": _unit_vec(5), "body": f"my key AKIAIOSFODNN7EXAMPLE {_LONG * 2}"},
    "n-nofed": {"day": "2026-09-06", "vec": _unit_vec(6), "tags": ["mnemos:no-federate"]},
    "n-noemb": {"day": "2026-09-07", "vec": None},
    "n-x": {"day": "2026-09-08", "vec": _unit_vec(8)},
}
_EDGES = [
    ("n-a", "n-b", "relates_to"),
    ("n-b", "n-c", "relates_to"),
    ("n-c", "n-d", "supersedes"),
    ("n-a", "n-secret", "relates_to"),
    ("n-b", "n-nofed", "relates_to"),
    ("n-c", "n-noemb", "relates_to"),
    ("n-secret", "n-nofed", "relates_to"),
]


def build_mock_graph_store(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    m = sqlite3.connect(store_dir / "mnemos.db")
    v = sqlite3.connect(store_dir / "vectors.db")
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
    m.execute(
        """
        CREATE TABLE memory_edges (
            from_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            to_memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            kind           TEXT NOT NULL CHECK (kind IN ('supersedes', 'relates_to')),
            created_at     TEXT NOT NULL,
            weight         REAL NOT NULL DEFAULT 1.0,
            provenance     TEXT NOT NULL DEFAULT 'declared',
            scope_project  TEXT,
            scope_agent    TEXT,
            PRIMARY KEY (from_memory_id, to_memory_id, kind),
            CHECK (from_memory_id <> to_memory_id)
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
    for node_id, spec in _TOPOLOGY.items():
        body = spec.get("body", f"{node_id} topic: {_LONG * 4}")
        tags = spec.get("tags", [])
        created = f"{spec['day']}T10:00:00.000000+00:00"
        m.execute(
            "INSERT INTO memories (id, content, title, tags, memory_type, created_at, updated_at, metadata, status, embedding_id, quarantine_reason) "
            "VALUES (?, ?, ?, ?, 'note', ?, ?, '{}', 'published', ?, NULL)",
            (node_id, body, f"title {node_id}", json.dumps(tags), created, created, node_id),
        )
        if spec["vec"] is not None:
            v.execute(
                "INSERT INTO embeddings (id, vector, metadata) VALUES (?, ?, ?)",
                (node_id, spec["vec"], json.dumps({"model_fingerprint": _PIN, "content_hash": "hash-" + node_id})),
            )
    for i, (src, dst, kind) in enumerate(_EDGES):
        m.execute(
            "INSERT INTO memory_edges (from_memory_id, to_memory_id, kind, created_at, weight, provenance) "
            "VALUES (?, ?, ?, ?, 1.0, ?)",
            (src, dst, kind, f"2026-09-10T0{i}:00:00.000000+00:00", "auto-dedupe"),
        )
    m.commit()
    v.commit()
    m.close()
    v.close()


@pytest.fixture()
def graph_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    build_mock_graph_store(store)
    return store


@pytest.fixture()
def seeds_file(graph_store: Path, tmp_path: Path) -> Path:
    """Seeds = all memory_edges endpoints (the live-run seed contract)."""
    conn = sqlite3.connect(f"file:{(graph_store / 'mnemos.db').as_posix()}?mode=ro", uri=True)
    try:
        ids = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT id FROM (
                    SELECT from_memory_id AS id FROM memory_edges
                    UNION SELECT to_memory_id FROM memory_edges
                ) ORDER BY id
                """
            )
        ]
    finally:
        conn.close()
    path = tmp_path / "seeds.txt"
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def targeted_export(graph_store: Path, seeds_file: Path, tmp_path: Path):
    return export_targeted_corpus(
        ids_file=seeds_file,
        store_path=graph_store,
        out_dir=tmp_path / "data",
        corpus_id="b2-edges-test",
        scanner=FallbackScanner(),
    )


# ── seed file contract + 1-hop expansion ─────────────────────────────────────


def test_load_ids_file_contract(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("\nn-one\n\n  n-two  \n\n", encoding="utf-8")
    assert load_ids_file(path) == ["n-one", "n-two"]

    empty = tmp_path / "empty.txt"
    empty.write_text("\n \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ids"):
        load_ids_file(empty)

    bad = tmp_path / "bad.txt"
    bad.write_text("id--with-separator\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separator"):
        load_ids_file(bad)


def test_expand_one_hop_is_undirected_and_closed(graph_store: Path) -> None:
    conn = sqlite3.connect(f"file:{(graph_store / 'mnemos.db').as_posix()}?mode=ro", uri=True)
    try:
        expanded = expand_one_hop(conn, {"n-a"})
        # n-a touches e1 (→n-b) and e4 (→n-secret): both endpoints enter.
        assert expanded == {"n-a", "n-b", "n-secret"}
        assert expand_one_hop(conn, set()) == set()
        # the full endpoint set is a fixed point of the expansion
        everything = expand_one_hop(conn, {"n-a", "n-b", "n-c", "n-d", "n-secret", "n-nofed", "n-noemb"})
        assert everything == {"n-a", "n-b", "n-c", "n-d", "n-secret", "n-nofed", "n-noemb"}
    finally:
        conn.close()


# ── hygiene chain inside the targeted selection ───────────────────────────────


def test_targeted_pool_and_hygiene_counters(targeted_export) -> None:
    counters = targeted_export.counters
    # 7 edge endpoints reach the SQL stage (n-x is never a candidate);
    # 4 pass hygiene; the 3 dirty endpoints drop with their reasons.
    assert counters.candidates_sql == 7
    assert counters.pool == 4
    assert counters.reasons["secret:aws-key"] == 1
    assert counters.reasons["no-federate"] == 1
    assert counters.reasons["missing-embedding"] == 1
    assert counters.excluded_total == 3
    assert targeted_export.ids_not_in_memories == 0
    assert targeted_export.trimmed_by_limit == 0


def test_targeted_pool_is_exactly_the_clean_neighborhood(targeted_export, tmp_path: Path) -> None:
    records_path = targeted_export.records_path
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in rows} == {"n-a", "n-b", "n-c", "n-d"}
    row = rows[0]
    assert set(row) == {"id", "title", "body", "tags", "language", "record_type",
                        "created_at", "content_hash", "vec_sha256", "vec"}
    assert len(row["vec"]) == 384
    # the secret record's content never enters the export
    pool_text = records_path.read_text(encoding="utf-8")
    assert "AKIAIOSFODNN7EXAMPLE" not in pool_text and "n-secret" not in pool_text


def test_targeted_near_dup_pairs_same_band(targeted_export, tmp_path: Path) -> None:
    # n-b sits at cosine 0.90 from n-a → exactly one in-band pair.
    assert targeted_export.counters.pairs == 1
    assert targeted_export.corpus_fingerprint
    rows = [json.loads(line) for line in targeted_export.pairs_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["pair_id"] == "n-a--n-b"
    assert 0.85 <= rows[0]["similarity"] < 0.97


# ── silver edges + coverage ───────────────────────────────────────────────────


def test_silver_edges_only_fully_inside_pool(targeted_export) -> None:
    rows = [json.loads(line) for line in targeted_export.silver_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    by_pair = {row["pair_id"]: row for row in rows}
    # pair_id: earlier-created end first, kind appended
    assert set(by_pair) == {"n-a--n-b--relates_to", "n-b--n-c--relates_to", "n-c--n-d--supersedes"}
    edge = by_pair["n-a--n-b--relates_to"]
    assert edge["id_a"] == "n-a" and edge["id_b"] == "n-b"
    assert edge["id_from"] == "n-a" and edge["id_to"] == "n-b"  # store orientation kept
    assert edge["kind"] == "relates_to"
    assert edge["provenance"] == "auto-dedupe"
    assert set(edge) == {"pair_id", "id_a", "id_b", "id_from", "id_to", "kind", "provenance", "edge_sha256"}
    # coverage: 3 inside, 3 partial (one dirty end), 1 absent (both dirty)
    assert targeted_export.edge_coverage == {
        "edges_total": 7,
        "edges_fully_inside": 3,
        "edges_partial": 3,
        "edges_absent": 1,
    }
    assert targeted_export.summary()["silver_edges_count"] == 3


def test_silver_edge_sha256_verifiable_from_records_join(targeted_export, tmp_path: Path) -> None:
    records = {
        row["id"]: row
        for row in (json.loads(line) for line in targeted_export.records_path.read_text(encoding="utf-8").splitlines())
    }
    for edge in (json.loads(line) for line in targeted_export.silver_path.read_text(encoding="utf-8").splitlines()):
        def side(record_id: str) -> dict:
            row = records[record_id]
            return {key: row[key] for key in ("title", "body", "tags", "language", "record_type", "created_at")}

        rebuilt = {
            "record": side(edge["id_a"]),
            "candidate": side(edge["id_b"]),
            "kind": edge["kind"],
            "provenance": edge["provenance"],
        }
        assert pair_sha256(rebuilt) == edge["edge_sha256"]


def test_silver_manifest_scheme_sorted_and_terminated(targeted_export) -> None:
    manifest = targeted_export.silver_manifest_path.read_bytes()
    lines = manifest.decode("utf-8").splitlines()
    assert lines == sorted(lines)
    assert manifest.endswith(b"\n")
    assert len(lines) == 3


# ── ids integrity + limit trim ────────────────────────────────────────────────


def test_ids_missing_from_memories_are_counted(graph_store: Path, tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("n-a\nn-ghost\nn-phantom\n", encoding="utf-8")
    result = export_targeted_corpus(
        ids_file=seeds,
        store_path=graph_store,
        out_dir=tmp_path / "data",
        corpus_id="b2-edges-test",
        scanner=FallbackScanner(),
    )
    # n-a expands to n-b + n-secret; ghosts counted, never reported by id
    assert result.seeds == 3
    assert result.ids_not_in_memories == 2
    # the expansion is seeds ∪ touched endpoints: ghosts stay in the
    # candidate set (they simply match no memories row)
    assert result.expanded_ids == 5
    assert result.counters.pool == 2  # n-a + n-b; n-secret drops on the aws key
    assert result.edge_coverage["edges_fully_inside"] == 1  # n-a→n-b
    # partial: e4 (dirty end), e2/e5 (clean end in pool, other end never
    # a candidate — the seed set was n-a only)
    assert result.edge_coverage["edges_partial"] == 3
    assert result.edge_coverage["edges_absent"] == 3


def test_limit_trims_deterministically_after_hygiene(graph_store: Path, seeds_file: Path, tmp_path: Path) -> None:
    result = export_targeted_corpus(
        ids_file=seeds_file,
        store_path=graph_store,
        out_dir=tmp_path / "data",
        corpus_id="b2-edges-test",
        scanner=FallbackScanner(),
        limit_pool=2,
    )
    assert result.trimmed_by_limit == 5  # 7 candidates − limit 2, SQL-stage trim
    assert result.counters.pool == 2
    rows = [json.loads(line) for line in result.records_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["n-a", "n-b"]  # created_at order holds
    # A2 SQL-LIMIT semantics: hygiene sees the LIMITED candidate set only,
    # so nothing was "excluded" here — the dirty endpoints were trimmed
    # away before the hygiene stage ever saw them (no leak either way).
    assert result.counters.excluded_total == 0


# ── read-only discipline ──────────────────────────────────────────────────────


def test_targeted_export_leaves_store_byte_identical(graph_store: Path, seeds_file: Path, tmp_path: Path) -> None:
    before = {p.name: p.read_bytes() for p in sorted(graph_store.iterdir())}
    export_targeted_corpus(
        ids_file=seeds_file,
        store_path=graph_store,
        out_dir=tmp_path / "data1",
        corpus_id="b2-edges-test",
        scanner=FallbackScanner(),
    )
    after = {p.name: p.read_bytes() for p in sorted(graph_store.iterdir())}
    assert before == after


# ── fingerprints ──────────────────────────────────────────────────────────────


def test_targeted_fingerprints_reproducible(graph_store: Path, seeds_file: Path, tmp_path: Path) -> None:
    results = [
        export_targeted_corpus(
            ids_file=seeds_file,
            store_path=graph_store,
            out_dir=tmp_path / f"data{i}",
            corpus_id="b2-edges-test",
            scanner=FallbackScanner(),
        )
        for i in (1, 2)
    ]
    first, second = results
    assert first.pool_fingerprint == second.pool_fingerprint
    assert first.silver_fingerprint == second.silver_fingerprint
    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.embedder_fingerprints == (_PIN,)
    cov1 = json.loads(first.coverage_path.read_text(encoding="utf-8"))
    cov2 = json.loads(second.coverage_path.read_text(encoding="utf-8"))
    for key in ("pool_fingerprint", "silver_fingerprint", "corpus_fingerprint", "edge_coverage", "hygiene"):
        assert cov1[key] == cov2[key]


def test_targeted_pool_fingerprint_matches_pool_scheme(graph_store: Path, seeds_file: Path, tmp_path: Path) -> None:
    """The targeted pool fingerprint uses the SAME §5 scheme as A2: the
    records_manifest.txt of both modes over the SAME id set must agree —
    proving the targeted mode changed only the candidate set."""
    targeted = export_targeted_corpus(
        ids_file=seeds_file,
        store_path=graph_store,
        out_dir=tmp_path / "d1",
        corpus_id="c",
        scanner=FallbackScanner(),
        with_pairs=False,
    )
    targeted_rows = {
        row["id"]: row
        for row in (json.loads(line) for line in targeted.records_path.read_text(encoding="utf-8").splitlines())
    }
    # rebuild the pool fingerprint from the written rows (§5 scheme)
    entries = [
        (record_id, pair_sha256({"record": {
            "title": row["title"], "body": row["body"], "tags": row["tags"],
            "language": row["language"], "record_type": row["record_type"],
            "created_at": row["created_at"], "content_hash": row["content_hash"],
            "vec_sha256": row["vec_sha256"],
        }}))
        for record_id, row in targeted_rows.items()
    ]
    from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes

    assert corpus_fingerprint(manifest_bytes(entries)) == targeted.pool_fingerprint


# ── CLI-facing script surface ─────────────────────────────────────────────────


def test_script_generates_seeds_and_exports(
    graph_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code = b2_script.main([
        "--store-path", str(graph_store),
        "--out", str(tmp_path / "data"),
        "--corpus-id", "b2-edges-test",
        "--engine-src", "",
    ])
    assert code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["pool_size"] == 4
    assert summary["silver_edges_count"] == 3
    assert summary["edge_coverage"]["edges_total"] == 7
    assert summary["hygiene"]["reasons"]["secret:aws-key"] == 1
    assert summary["scanner_provenance"] == "fallback scanner, not engine"
    assert captured.err.strip(), "progress must be on stderr"
    # the generated seed file exists and holds exactly the 7 endpoints
    ids_files = list((tmp_path / "data" / "ids").glob("edge-node-ids-*.txt"))
    assert len(ids_files) == 1
    ids = [line for line in ids_files[0].read_text(encoding="utf-8").splitlines() if line]
    assert len(ids) == 7


def test_script_rejects_bad_args(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = b2_script.main(["--out", str(tmp_path / "o")])
    assert code == 2
    assert "exactly one" in capsys.readouterr().err

    code = b2_script.main([
        "--store-path", str(tmp_path),
        "--out", str(tmp_path / "o"),
        "--min-cosine", "0.97",
        "--max-cosine", "0.85",
    ])
    assert code == 2
    assert "min-cosine" in capsys.readouterr().err


def test_script_missing_ids_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = b2_script.main([
        "--store-path", str(tmp_path),
        "--out", str(tmp_path / "o"),
        "--ids-file", str(tmp_path / "nope.txt"),
    ])
    assert code == 2
    assert "not found" in capsys.readouterr().err


# ── regression: the plain A2 export is untouched by the refactor ──────────────


def test_plain_export_still_works_on_graph_store(graph_store: Path, tmp_path: Path) -> None:
    result = export_store_corpus(
        store_path=graph_store,
        out_dir=tmp_path / "data",
        corpus_id="plain",
        scanner=FallbackScanner(),
    )
    # all 8 published notes reach hygiene: 5 clean (incl. n-x), 3 dirty
    assert result.counters.candidates_sql == 8
    assert result.counters.pool == 5
    assert result.counters.reasons["secret:aws-key"] == 1
    assert result.counters.reasons["no-federate"] == 1
    assert result.counters.reasons["missing-embedding"] == 1
