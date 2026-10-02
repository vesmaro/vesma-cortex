#!/usr/bin/env python3
"""B2 graph-evidence sidecar — reads the ENGINE store read-only.

Wave B2 of the v2 duty program (roadmap-v2.md §4; ADR 0002: is-duplicate
rev 1.n, the graph block is ADDITIVE — the attach precedent is the A2
field-cosines sidecar). Like ``scripts/a2_field_cosines.py`` this script
lives OUTSIDE src/cortex on purpose: it opens the live store and carries
store-side logic that must never leak into the library's network-import
isolation boundary (tests/test_skeleton.py scans src/cortex).

Source of truth (READ-ONLY, not negotiable — the store serves a live
engine):

- ``<store>/mnemos.db``  — memories (id, title, content, tags, status) +
  memory_edges (kind CHECK IN ('supersedes', 'relates_to'));
- ``<store>/vectors.db`` — embeddings (id = memory id, 384-dim float32
  unit-norm blobs; content_hash + model pin live in the metadata JSON).

Both databases open via ``file:…?mode=ro`` URIs plus
``PRAGMA query_only=ON`` — belt and braces, same pattern as
``cortex.data.store_export._ro_connection``. The script never writes,
never re-embeds (drift ban, ADR 0001): vectors come from the store as-is.

Join strategy (dataset pairs reference store records through CONTENT):
1. PRIMARY — the dataset's ``source_id`` / ``source_id_a`` / ``source_id_b``
   columns (direct memory ids, assigned at dataset creation);
2. VERIFICATION — for every store-resident side the (title, body) pair
   must match the store row exactly (content-hash discipline); any drift
   is counted and reported, never silently accepted;
3. FALLBACK for sides without a resolvable source_id — exact
   (title, body) match against the store; ambiguous (title collisions)
   refuse the join with a counter;
4. sides with no store identity (e.g. synthetic corruption variants of
   the POS strata) are LEGITIMATELY graph-empty: has_graph_evidence=false,
   zero-valued counters — absence of evidence is evidence of absence here
   by dataset construction, and the report says so explicitly.

Per pair (both sides store-resident only):

- ``edge_ab`` — a direct memory_edges row between the two sides in ANY
  direction (undirected presence); ``edge_kinds`` lists the observed
  kinds;
- common neighbors — 1-hop undirected neighborhoods N(a), N(b) over all
  edge kinds; ``common_neighbors`` = |N(a) ∩ N(b)|, ``neighbor_jaccard``
  = |∩| / |∪| (0.0 when both are empty — unlike the text-Jaccard
  convention, two empty neighborhoods mean NO graph evidence, not
  identity);
- cosines to common neighbors — for every c ∈ N(a) ∩ N(b) with a store
  vector: cos(vec_a, vec_c) and cos(vec_b, vec_c); ``cos_to_common_max``
  / ``cos_to_common_mean`` aggregate over ALL (side, neighbor) values.
  When no vectors are available the fields are omitted (honest absence —
  counters only), never defaulted.

Outputs (gitignored data/ tree — no raw store content enters the repo):

- ``<out>.jsonl`` — one evidence row per pair, sorted by pair_id;
- ``<out>.coverage.json`` — coverage/counters report + sidecar
  fingerprint (BLAKE2b-256 over the sorted per-row sha256 manifest, the
  prereg scheme of cortex.data.fingerprints — copied, not imported, to
  keep this script dependency-free of the library).

Progress goes to stderr, a JSON summary to stdout. Stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

#: vesma-embed-v1 geometry (mirrors cortex.data.store_export — pinned there).
EMBEDDING_DIM: Final[int] = 384
VECTOR_BLOB_BYTES: Final[int] = EMBEDDING_DIM * 4
UNIT_NORM_TOLERANCE: Final[float] = 1e-3

#: Edge kinds the store CHECK constraint admits today (contradicts is a
#: future migration — outside this wave's slice, ADR 0002 §R1).
EDGE_KINDS: Final[frozenset[str]] = frozenset({"supersedes", "relates_to"})

CANONICAL_JSON_KWARGS: Final[dict[str, object]] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="b2_graph_sidecar",
        description="graph-evidence sidecar from the engine store (READ-ONLY)",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="dataset-v2 catalog dir with pairs-pos.jsonl and pairs-neg.jsonl",
    )
    parser.add_argument(
        "--store-path",
        help="store data DIRECTORY holding mnemos.db + vectors.db",
    )
    parser.add_argument(
        "--store-uri",
        help="read-only URI of mnemos.db (file:...?mode=ro); vectors.db is its sibling",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output sidecar path (jsonl); <out>.coverage.json is written next to it",
    )
    return parser.parse_args(argv)


def _progress(message: str) -> None:
    print(f"b2_graph_sidecar: {message}", file=sys.stderr)


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _row_sha256(row: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _sidecar_fingerprint(rows: list[dict[str, object]]) -> str:
    """BLAKE2b-256 over sorted ``pair_id <row-sha256>`` manifest lines.

    Same scheme as cortex.data.fingerprints.corpus_fingerprint (copied —
    this script stays library-free); any row edit changes the fingerprint.
    """
    lines = sorted(f"{row['pair_id']} {_row_sha256(row)}\n" for row in rows)
    return hashlib.blake2b("".join(lines).encode("utf-8"), digest_size=32).hexdigest()


def _ro_connection(db_path: Path) -> sqlite3.Connection:
    """Open one store database strictly read-only, then lock writes off."""
    if not db_path.is_file():
        raise SystemExit(f"store database not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise SystemExit(f"cannot open {db_path} read-only: {exc}") from exc
    conn.execute("PRAGMA query_only=ON")
    return conn


def _resolve_store(store_path: str | None, store_uri: str | None) -> tuple[Path, Path]:
    """Resolve (mnemos.db, vectors.db); exactly one argument must be given."""
    if (store_path is None) == (store_uri is None):
        raise SystemExit("exactly one of --store-path / --store-uri is required")
    if store_path is not None:
        base = Path(store_path).expanduser().resolve()
        if not base.is_dir():
            raise SystemExit(f"store path is not a directory: {base}")
        return base / "mnemos.db", base / "vectors.db"
    assert store_uri is not None
    if "mode=ro" not in store_uri:
        raise SystemExit("--store-uri must be a read-only URI (file:...?mode=ro)")
    raw = store_uri[len("file:"):] if store_uri.startswith("file:") else store_uri
    vesma = Path(raw.split("?", 1)[0]).expanduser().resolve()
    return vesma, mnemos.parent / "vectors.db"


class StoreSnapshot:
    """Read-only snapshot of the graph + vectors, opened last, closed first.

    Loads in-memory maps in one pass each, then releases both connections
    (open → read → close: the store serves a live server).
    """

    def __init__(self, mnemos: Path, vectors: Path) -> None:
        self.counters: dict[str, int] = {}

        conn = _ro_connection(vesma)
        try:
            self.bodies: dict[str, str] = {}
            titles: dict[str, list[str]] = {}
            for memory_id, title, content in conn.execute(
                "SELECT id, title, content FROM memories"
            ):
                self.bodies[memory_id] = content or ""
                titles.setdefault(title or "", []).append(memory_id)
            self.title_ids: dict[str, list[str]] = titles

            self.adjacency: dict[str, set[str]] = {}
            kind_index: dict[tuple[str, str], set[str]] = {}
            self.edges_total = 0
            for src, dst, kind in conn.execute(
                "SELECT from_memory_id, to_memory_id, kind FROM memory_edges"
            ):
                self.edges_total += 1
                if src not in self.bodies or dst not in self.bodies:
                    self.counters["edges_dangling"] = (
                        self.counters.get("edges_dangling", 0) + 1
                    )
                    continue
                self.adjacency.setdefault(src, set()).add(dst)
                self.adjacency.setdefault(dst, set()).add(src)
                kind_index.setdefault((src, dst), set()).add(kind)
            self._kind_index = kind_index
        finally:
            conn.close()

        self.vectors: dict[str, np.ndarray] = {}
        conn = _ro_connection(vectors)
        try:
            for memory_id, blob in conn.execute("SELECT id, vector FROM embeddings"):
                vector = self._decode(blob)
                if vector is None:
                    self.counters["bad-vectors"] = self.counters.get("bad-vectors", 0) + 1
                    continue
                self.vectors[memory_id] = vector
        finally:
            conn.close()

    @staticmethod
    def _decode(blob: bytes) -> np.ndarray | None:
        if len(blob) != VECTOR_BLOB_BYTES:
            return None
        vector = np.frombuffer(blob, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            return None
        return vector

    def side_in_store(self, memory_id: str | None, title: str, body: str) -> bool:
        """A side resolves into the store: id present AND content matches.

        The content check is the join verification (title+body exact) — a
        present id with drifted content is counted, never accepted.
        """
        if memory_id is None or memory_id not in self.bodies:
            return False
        if self.bodies[memory_id] != body or memory_id not in self.title_ids.get(title, []):
            self.counters["content-drift"] = self.counters.get("content-drift", 0) + 1
            return False
        return True

    def edge_kinds_between(self, id_a: str, id_b: str) -> list[str]:
        kinds: set[str] = set()
        kinds |= self._kind_index.get((id_a, id_b), set())
        kinds |= self._kind_index.get((id_b, id_a), set())
        return sorted(kinds & EDGE_KINDS)

    def cosine(self, id_u: str, id_v: str) -> float | None:
        vec_u, vec_v = self.vectors.get(id_u), self.vectors.get(id_v)
        if vec_u is None or vec_v is None:
            return None
        denominator = float(np.linalg.norm(vec_u) * np.linalg.norm(vec_v))
        if denominator == 0.0:
            return None
        return min(1.0, max(-1.0, float(np.dot(vec_u, vec_v) / denominator)))


def _load_pairs(dataset_dir: Path) -> list[tuple[str, dict[str, object]]]:
    """(kind, pair) rows from pairs-pos.jsonl / pairs-neg.jsonl, in order."""
    rows: list[tuple[str, dict[str, object]]] = []
    for name, kind in (("pairs-pos.jsonl", "pos"), ("pairs-neg.jsonl", "neg")):
        path = dataset_dir / name
        if not path.is_file():
            raise SystemExit(f"dataset file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
                raise SystemExit(f"{path}: row without a string pair_id")
            rows.append((kind, pair))
    return rows


def _side_of(pair: dict[str, object], side_key: str, id_key: str) -> tuple[dict[str, object], str | None]:
    side = pair.get(side_key)
    if not isinstance(side, dict):
        raise SystemExit(f"pair {pair.get('pair_id')!r}: side {side_key!r} is not an object")
    source_id = pair.get(id_key)
    return side, (source_id if isinstance(source_id, str) and source_id else None)


def _resolve_side(
    snapshot: StoreSnapshot,
    side: dict[str, object],
    source_id: str | None,
    join_methods: dict[str, int],
) -> str | None:
    """Store id of one pair side — source_id first, exact-content fallback."""
    title, body = str(side.get("title", "")), str(side.get("body", ""))
    if source_id is not None:
        if snapshot.side_in_store(source_id, title, body):
            join_methods["source_id"] = join_methods.get("source_id", 0) + 1
            return source_id
        join_methods["source_id-unresolved"] = join_methods.get("source_id-unresolved", 0) + 1
    candidates = snapshot.title_ids.get(title, [])
    exact = [mid for mid in candidates if snapshot.bodies[mid] == body]
    if len(exact) == 1:
        join_methods["content-exact"] = join_methods.get("content-exact", 0) + 1
        return exact[0]
    if len(exact) > 1:
        join_methods["content-ambiguous"] = join_methods.get("content-ambiguous", 0) + 1
    else:
        join_methods["none"] = join_methods.get("none", 0) + 1
    return None


def _evidence_row(
    snapshot: StoreSnapshot,
    pair: dict[str, object],
    id_a: str | None,
    id_b: str | None,
) -> tuple[dict[str, object], bool]:
    """One sidecar row + whether it carries any graph evidence."""
    row: dict[str, object] = {
        "pair_id": pair["pair_id"],
        "stratum": pair.get("stratum"),
        "label": str(pair.get("label", "")).replace("_", "-"),
        "a_store_id": id_a,
        "b_store_id": id_b,
        "edge_ab": False,
        "edge_kinds": [],
        "common_neighbors": 0,
        "neighbor_jaccard": 0.0,
        "has_graph_evidence": False,
    }
    if id_a is None or id_b is None:
        return row, False
    if id_a == id_b:
        # A dataset anomaly (a "pair" of one store record) — counted, never
        # normalized into evidence: the store CHECK forbids self-loops, so
        # graph evidence here would be a join artifact, not store truth.
        snapshot.counters["self-pairs"] = snapshot.counters.get("self-pairs", 0) + 1
        return row, False

    kinds = snapshot.edge_kinds_between(id_a, id_b)
    neighbors_a = snapshot.adjacency.get(id_a, set())
    neighbors_b = snapshot.adjacency.get(id_b, set())
    common = neighbors_a & neighbors_b
    union = neighbors_a | neighbors_b
    row["edge_ab"] = bool(kinds)
    row["edge_kinds"] = kinds
    row["common_neighbors"] = len(common)
    row["neighbor_jaccard"] = (len(common) / len(union)) if union else 0.0

    cosine_values: list[float] = []
    missing_vectors = 0
    for side_id in (id_a, id_b):
        for neighbor_id in sorted(common):
            value = snapshot.cosine(side_id, neighbor_id)
            if value is None:
                missing_vectors += 1
            else:
                cosine_values.append(value)
    if missing_vectors:
        snapshot.counters["neighbor-vectors-missing"] = (
            snapshot.counters.get("neighbor-vectors-missing", 0) + missing_vectors
        )
    if cosine_values:
        row["cos_to_common_max"] = max(cosine_values)
        row["cos_to_common_mean"] = sum(cosine_values) / len(cosine_values)

    has_evidence = bool(kinds) or len(common) > 0
    row["has_graph_evidence"] = has_evidence
    return row, has_evidence


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    vesma, vectors = _resolve_store(args.store_path, args.store_uri)
    _progress(f"opening store read-only: {vesma}")

    snapshot = StoreSnapshot(vesma, vectors)
    _progress(
        f"store snapshot: {len(snapshot.bodies)} memories, {snapshot.edges_total} edges, "
        f"{len(snapshot.vectors)} vectors"
    )

    join_methods: dict[str, int] = {}
    out_rows: list[dict[str, object]] = []
    evidence_pairs = 0
    pairs_without_store_sides = 0
    for kind, pair in _load_pairs(dataset_dir):
        if kind == "pos":
            raw_sides = (
                _side_of(pair, "original", "source_id"),
                _side_of(pair, "variant", "source_id_variant"),
            )
        else:
            raw_sides = (
                _side_of(pair, "a", "source_id_a"),
                _side_of(pair, "b", "source_id_b"),
            )
        id_a = _resolve_side(snapshot, raw_sides[0][0], raw_sides[0][1], join_methods)
        id_b = _resolve_side(snapshot, raw_sides[1][0], raw_sides[1][1], join_methods)
        if id_a is None or id_b is None:
            pairs_without_store_sides += 1

        row, has_evidence = _evidence_row(snapshot, pair, id_a, id_b)
        out_rows.append(row)
        if has_evidence:
            evidence_pairs += 1

    out_rows.sort(key=lambda row: str(row["pair_id"]))
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(_canonical_json(row) + "\n")
    _progress(f"wrote {out_path} ({len(out_rows)} rows)")

    strata_evidence: dict[str, int] = {}
    for row in out_rows:
        if row["has_graph_evidence"]:
            key = str(row["stratum"])
            strata_evidence[key] = strata_evidence.get(key, 0) + 1

    total = len(out_rows)
    coverage = {
        "pairs_total": total,
        "pairs_with_graph_evidence": evidence_pairs,
        "graph_evidence_rate": round(evidence_pairs / total, 4) if total else 0.0,
        "pairs_without_store_sides": pairs_without_store_sides,
        "join_methods": dict(sorted(join_methods.items())),
        "strata_with_evidence": dict(sorted(strata_evidence.items())),
        "store": {
            "memories": len(snapshot.bodies),
            "edges_total": snapshot.edges_total,
            "edges_dangling": snapshot.counters.get("edges_dangling", 0),
            "vectors": len(snapshot.vectors),
            "bad_vectors": snapshot.counters.get("bad-vectors", 0),
            "content_drift": snapshot.counters.get("content-drift", 0),
            "neighbor_vectors_missing": snapshot.counters.get("neighbor-vectors-missing", 0),
        },
        "sidecar_fingerprint": _sidecar_fingerprint(out_rows),
        "out": str(out_path),
        "seconds": round(time.perf_counter() - started, 1),
    }
    coverage_path = out_path.with_name(out_path.stem + ".coverage.json")
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _progress(f"wrote {coverage_path}")
    print(json.dumps(coverage, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
