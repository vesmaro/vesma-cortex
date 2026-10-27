#!/usr/bin/env python3
"""Cut and seal the g4-spec-v1 evaluation surface (spec leg of gate g4').

Law: docs/specs/b2-graph-prereg-v3-ACTIVATION-NOTE-g4-spec-surface.md
(commit 73ffa50, 2026-10-23, BINDING activation record — NOT an addendum;
no threshold/population/formula change to ACTIVATED ADDENDUM-2).

Surface definition (ONE reading for the gate cycle):
  1 source: live store, read-only probe of the census recipe §5.1
    (hygiene gates 1:1 with scripts/b2_store_probe_v45.py); snapshot
    fingerprinted;
  2 rows: pairs (record, candidate) with evidence flag
    (edge_ab=1 OR common_neighbors>=1), BOTH sides resolved, cos >= 0.97
    measured by the FROZEN embedder with the corpus builder formula
    (NanoProvider over title\\nbody\\ntags, round(min(1,max(0,dot)),6)),
    silver §A -> NOT dup (disputed -> NOT dup, §8.3; the whole mint zone
    is NOT-dup per the census); a norm-equal (silver-dup) pair cannot sit
    on this surface — it is quarantined with a counter and disclosed;
  3 exclusions (cut asserts, pair-level, both orientations = sorted side
    sha tuples + content-hash sets): train v4.5 (fp 3bf7ed53...), strata A
    (the full probe snapshot pair keys — stricter than the corpus rows),
    historical holdout surfaces v4.1/v4.2/v4.3/v4.4, holdout v4.5
    (seal 9eae5ec1...);
  4 NO subsampling: the surface is the whole qualifying population; if
    fewer than 30 pairs qualify — STOP and escalate to TL;
  5 freeze: evidence columns fixed at cut + fresh sidecar agreement assert
    (A2.5.3 discipline); ONE orientation per pair (lexicographic store-id
    order — deterministic; seed recorded for the record, no RNG used);
  6 seal: corpus_fingerprint over [(pair_id, pair_sha256)] (the canonical
    builder function); one gate-runner read; immutable after sealing.

Outputs (gitignored data/ tree):
  <out>/g4-spec-surface.jsonl       — one row per pair (sorted by pair_id)
  <out>/g4-spec-surface-ids.json    — ids + sha256 + surface fingerprint
  <out>/g4-spec-manifest.json       — funnel counters, snapshot, seal
  <out>/g4-spec-run-log.jsonl       — append-only run log

Exit codes: 0 ok | 1 gate refusal (incl. the <30 floor) | 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src"
BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
V45_DIR = REPO_ROOT / "data/b2-graph-corpus/v45"
POOL_JSONL = REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl"

PIN_PREFIX = "nano:sha256:3b752e06"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
COS_MIN = 0.97
FLOOR = 30
SEED = 42  # recorded; the orientation rule is deterministic (no RNG used)


class GateRefusal(SystemExit):
    """A cut assert failed — the surface is NOT sealed."""


def refuse(msg: str) -> None:
    raise GateRefusal(f"[g4-spec] GATE REFUSAL: {msg}")


def log(runlog: list[dict], event: str, **fields) -> None:
    entry = {"event": event, **fields}
    runlog.append(entry)
    print(
        f"[g4-spec] {event}: " + " ".join(f"{k}={v}" for k, v in fields.items()),
        file=sys.stderr,
    )


def norm1(s: str) -> str:
    return " ".join((s or "").lower().split())


def text_hash(text: str) -> str:
    n = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def side_sha(s: dict) -> str:
    payload = json.dumps(
        {
            "title": s.get("title"),
            "body": s.get("body"),
            "tags": sorted(s.get("tags") or []),
            "language": s.get("language"),
            "record_type": s.get("record_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_key(row_or_pair) -> tuple[str, str]:
    if isinstance(row_or_pair, dict) and "record" in row_or_pair:
        return tuple(
            sorted(
                (side_sha(row_or_pair["record"]), side_sha(row_or_pair["candidate"]))
            )
        )
    a, b = row_or_pair
    return tuple(sorted((side_sha(a), side_sha(b))))


def corpus_rows(path: Path) -> list[dict]:
    if not path.is_file():
        refuse(f"frozen input missing: {path}")
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def ro(db: Path):
    import sqlite3

    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="b2_graph_g4_spec_surface")
    p.add_argument(
        "--store-uri",
        required=True,
        help="read-only URI of vesma.db (file:...?mode=ro)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output directory (default: data/b2-graph-corpus/v45)",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "mode=ro" not in args.store_uri:
        refuse("--store-uri must be a read-only URI (file:...?mode=ro)")
    vesma_db = Path(args.store_uri.removeprefix("file:").split("?", 1)[0])
    vectors_db = vesma_db.parent / "vectors.db"
    runlog: list[dict] = []

    # ── phase 1: read-only store probe (census recipe §5.1, hygiene 1:1) ────
    scanner = make_scanner(ENGINE_SRC)
    conn = ro(vesma_db)
    mem: dict[str, tuple] = {}
    for row in conn.execute(
        "SELECT id, title, content, tags, memory_type, created_at, metadata,"
        " quarantine_reason, status FROM memories"
    ):
        mem[row[0]] = row
    edges = list(
        conn.execute(
            "SELECT from_memory_id, to_memory_id, kind, provenance, weight"
            " FROM memory_edges"
        )
    )
    conn.close()
    conn = ro(vectors_db)
    embeddings: dict[str, tuple[bytes, str]] = {}
    for mid, blob, meta_json in conn.execute(
        "SELECT id, vector, metadata FROM embeddings"
    ):
        if isinstance(blob, bytes) and len(blob) == BLOB:
            embeddings[mid] = (blob, meta_json or "{}")
    conn.close()

    counters = HygieneCounters()
    counters.candidates_sql = len(mem)
    passed: dict[str, dict] = {}
    for mid, row in mem.items():
        (rid, title, content, tags_json, mtype, created, meta_json, quar, status) = row
        if status != "published":
            counters.exclude("sql-gate:status")
            continue
        if mtype not in ("note", "snippet", "fact"):
            counters.exclude("sql-gate:memory_type")
            continue
        if len(content or "") < 80:
            counters.exclude("sql-gate:length<80")
            continue
        tags = tuple(str(t) for t in json.loads(tags_json or "[]"))
        if "mnemos:no-federate" in tags:
            counters.exclude("no-federate")
            continue
        if quar:
            counters.exclude("quarantine")
            continue
        emb = embeddings.get(rid)
        if emb is None:
            counters.exclude("missing-embedding")
            continue
        blob, meta_j = emb
        meta = json.loads(meta_j)
        pin = meta.get("model_fingerprint")
        if not pin or not str(pin).startswith(PIN_PREFIX):
            counters.exclude("unpinned/mismatch")
            continue
        if not meta.get("content_hash"):
            counters.exclude("missing-content-hash")
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
            counters.exclude("bad-vector")
            continue
        if scanner.scan(title or "", content or "", list(tags)):
            counters.exclude("scanner")
            continue
        passed[rid] = {
            "body": content or "",
            "title": title or "",
            "vec": vec / nrm,
            "tags": list(tags),
            "record_type": mtype,
            "language": (json.loads(meta_json or "{}").get("canon") or {}).get(
                "language"
            ),
            "created_at": created,
        }

    snapshot_fp = hashlib.blake2b(
        "\n".join(
            f"{rid} {text_hash(passed[rid]['title'] + chr(10) + passed[rid]['body'])}"
            for rid in sorted(passed)
        ).encode("utf-8")
        + b"\n",
        digest_size=32,
    ).hexdigest()
    log(
        runlog,
        "store_probe",
        uri=args.store_uri,
        store_snapshot={
            "memories": len(mem),
            "edges": len(edges),
            "vectors": len(embeddings),
        },
        hygiene=counters.to_dict(),
        passed=len(passed),
        snapshot_fingerprint=snapshot_fp,
    )

    # adjacency (undirected) over ALL edges — the evidence flag is a graph fact
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_pairs: set[frozenset] = set()
    edge_kinds: dict[frozenset, set[str]] = defaultdict(set)
    for s, d, kind, _prov, _w in edges:
        if s == d:
            continue
        adjacency[s].add(d)
        adjacency[d].add(s)
        edge_pairs.add(frozenset((s, d)))
        edge_kinds[frozenset((s, d))].add(kind)

    # ── phase 2: enumerate EVIDENCE pairs (edge_ab=1 OR common_neighbors>=1)
    evidence_ids: set[frozenset] = set(edge_pairs)
    for node, nbrs in adjacency.items():
        nl = sorted(nbrs)
        for i in range(len(nl)):
            for j in range(i + 1, len(nl)):
                evidence_ids.add(frozenset((nl[i], nl[j])))
    # keep only pairs where BOTH sides resolve (hygiene-passed)
    evidence_ids = {fs for fs in evidence_ids if all(m in passed for m in fs)}
    log(
        runlog,
        "evidence_pairs",
        raw=len(evidence_ids),
        with_edge=sum(1 for fs in evidence_ids if fs in edge_pairs),
    )

    # ── phase 3: frozen-embedder cosine (builder formula), filter >= 0.97 ───
    if ENGINE_SRC not in sys.path:
        sys.path.insert(0, ENGINE_SRC)
    from vesmaro.embeddings import NanoProvider  # noqa: E402

    provider = NanoProvider()
    pin = provider.fingerprint
    if not pin.startswith("nano:sha256:3b752e06"):
        refuse(f"embedder pin {pin!r} differs from the frozen prefix")
    cache: dict[str, list[float]] = {}

    def embed_text(text: str) -> list[float]:
        text = text[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    def embed_side(s: dict) -> list[float]:
        tags = " ".join(s.get("tags") or [])
        return embed_text(f"{s.get('title') or ''}\n{s.get('body') or ''}\n{tags}")

    def corpus_cos(a: dict, b: dict) -> float:
        va, vb = embed_side(a), embed_side(b)
        return round(min(1.0, max(0.0, sum(x * y for x, y in zip(va, vb)))), 6)

    qualified: list[dict] = []
    below_band = 0
    silver_dup = 0
    for fs in sorted(evidence_ids, key=lambda s: sorted(s)):
        ia, ib = sorted(fs)
        a, b = passed[ia], passed[ib]
        sim = corpus_cos(a, b)
        if sim < COS_MIN:
            below_band += 1
            continue
        if (
            norm1(a["body"]) == norm1(b["body"])
            and norm1(a["title"]) == norm1(b["title"])
            and set(a["tags"]) == set(b["tags"])
            and a["record_type"] == b["record_type"]
            and a["language"] == b["language"]
        ):
            # silver §A says dup — the surface is negatives-only; §8.3
            # conservative: quarantined with a counter, never squeezed in
            silver_dup += 1
            continue
        qualified.append(
            {
                "ids": (ia, ib),
                "a": a,
                "b": b,
                "similarity": sim,
                "has_edge": fs in edge_pairs,
                "kinds": sorted(edge_kinds.get(fs, set())),
                "cn": len(adjacency[ia] & adjacency[ib]),
            }
        )
    log(
        runlog,
        "cos_band_filter",
        cos_min=COS_MIN,
        below_band=below_band,
        silver_dup_quarantined=silver_dup,
        qualified=len(qualified),
    )
    if len(qualified) < FLOOR:
        refuse(
            f"qualifying pairs {len(qualified)} < FLOOR {FLOOR}; the "
            "definition is not to be squeezed; STOP and escalate to TL"
        )

    # ── phase 4: exclusions (pair-level, both orientations + content hashes)
    surfaces: dict[str, set] = {}
    content_surfaces: dict[str, set] = {}
    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        rows = corpus_rows(BASE / name)
        surfaces[f"historical:{name}"] = {pair_key(r) for r in rows}
        content_surfaces[f"historical:{name}"] = {
            (
                text_hash(r["record"]["title"] + "\n" + r["record"]["body"]),
                text_hash(r["candidate"]["title"] + "\n" + r["candidate"]["body"]),
            )
            for r in rows
        }
    v44_rows = corpus_rows(REPO_ROOT / "data/b2-graph-corpus/v44/holdout.jsonl")
    surfaces["historical:v4.4-holdout"] = {pair_key(r) for r in v44_rows}
    content_surfaces["historical:v4.4-holdout"] = {
        (
            text_hash(r["record"]["title"] + "\n" + r["record"]["body"]),
            text_hash(r["candidate"]["title"] + "\n" + r["candidate"]["body"]),
        )
        for r in v44_rows
    }
    hold45 = corpus_rows(V45_DIR / "holdout.jsonl")
    surfaces["holdout-v4.5"] = {pair_key(r) for r in hold45}
    content_surfaces["holdout-v4.5"] = {
        (
            text_hash(r["record"]["title"] + "\n" + r["record"]["body"]),
            text_hash(r["candidate"]["title"] + "\n" + r["candidate"]["body"]),
        )
        for r in hold45
    }
    train45 = corpus_rows(V45_DIR / "train.jsonl")
    surfaces["train-v4.5"] = {pair_key(r) for r in train45}
    content_surfaces["train-v4.5"] = {
        (
            text_hash(r["record"]["title"] + "\n" + r["record"]["body"]),
            text_hash(r["candidate"]["title"] + "\n" + r["candidate"]["body"]),
        )
        for r in train45
    }
    # strata A: the FULL probe snapshot (stricter than the corpus rows)
    snap = corpus_rows(V45_DIR / "strata-a-snapshot.jsonl")
    snap_keys = set()
    snap_content = set()
    for p in snap:
        k = pair_key((p["record"], p["candidate"]))
        snap_keys.add(k)
        snap_content.add(
            (
                text_hash(p["record"]["title"] + "\n" + p["record"]["body"]),
                text_hash(p["candidate"]["title"] + "\n" + p["candidate"]["body"]),
            )
        )
    surfaces["strata-A"] = snap_keys
    content_surfaces["strata-A"] = snap_content

    kept: list[dict] = []
    excluded: Counter = Counter()
    for q in qualified:
        a, b = q["a"], q["b"]
        pk = pair_key((a, b))
        ch = (
            text_hash(a["title"] + "\n" + a["body"]),
            text_hash(b["title"] + "\n" + b["body"]),
        )
        hit = None
        for name, keys in surfaces.items():
            if pk in keys:
                hit = f"{name}(pair-key)"
                break
        if hit is None:
            chs = {ch, (ch[1], ch[0])}
            for name, chs_set in content_surfaces.items():
                if chs & chs_set:
                    hit = f"{name}(content-hash)"
                    break
        if hit is not None:
            excluded[hit] += 1
            continue
        kept.append(q)
    log(
        runlog,
        "exclusions",
        total_surfaces=len(surfaces) + len(content_surfaces),
        excluded=dict(sorted(excluded.items())),
        kept=len(kept),
    )
    if len(kept) < FLOOR:
        refuse(
            f"after exclusions {len(kept)} pairs < FLOOR {FLOOR}; the "
            "definition is not to be squeezed; STOP and escalate to TL"
        )

    # ── phase 5: rows (one orientation: lexicographic store-id order) ───────
    rows: list[dict] = []
    for n, q in enumerate(sorted(kept, key=lambda q: q["ids"]), start=1):
        ia, ib = q["ids"]
        a, b = q["a"], q["b"]
        rec = {
            "title": a["title"],
            "body": a["body"],
            "tags": a["tags"],
            "language": a["language"],
            "record_type": a["record_type"],
            "created_at": a["created_at"],
        }
        cand = {
            "title": b["title"],
            "body": b["body"],
            "tags": b["tags"],
            "language": b["language"],
            "record_type": b["record_type"],
            "created_at": b["created_at"],
        }
        pair_obj = {"record": rec, "candidate": cand, "similarity": q["similarity"]}
        nbrs_a = adjacency[ia]
        nbrs_b = adjacency[ib]
        common = nbrs_a & nbrs_b
        union = nbrs_a | nbrs_b
        row = {
            "pair_id": f"G4SPEC-{n:04d}",
            "label": "not-duplicate",
            "stratum": "g4-spec-v1",
            "record": rec,
            "candidate": cand,
            "similarity": q["similarity"],
            "pair_sha256": pair_sha256(pair_obj),
            "label_rule": "silver §A -> NOT dup (disputed -> NOT dup §8.3; "
            "whole mint zone NOT-dup per census)",
            "id_a": ia,
            "id_b": ib,
            "orientation_rule": "lexicographic store-id order (deterministic)",
            "seed": SEED,
            "edge_ab": int(q["has_edge"]),
            "edge_kind_supersedes": int(1 if "supersedes" in q["kinds"] else 0),
            "edge_kinds": q["kinds"],
            "common_neighbors": len(common),
            "neighbor_jaccard": (round(len(common) / len(union), 6) if union else 0.0),
            "has_graph_evidence": bool(q["has_edge"] or len(common) >= 1),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["pair_id"])

    # frozen evidence columns must be self-consistent (ONE reading)
    for r in rows:
        ev = r["edge_ab"] == 1 or r["common_neighbors"] >= 1
        if ev != r["has_graph_evidence"]:
            refuse(f"evidence reading drift on {r['pair_id']}; STOP")
        if not r["has_graph_evidence"]:
            refuse(
                f"row {r['pair_id']} without evidence on an evidence-only surface; STOP"
            )

    # ── phase 6: fresh sidecar agreement (A2.5.3 discipline) ────────────────
    cat = out / "g4-spec-sidecar-catalog"
    side_out = out / "g4-spec-graph-sidecar.jsonl"
    pre_jsonl = out / "g4-spec-precarve.jsonl"
    write_jsonl(pre_jsonl, rows)
    py = sys.executable
    r1 = subprocess.run(
        [
            py,
            str(REPO_ROOT / "scripts" / "b2_rev1n_sidecar_input.py"),
            "--rows",
            str(pre_jsonl),
            "--out-dir",
            str(cat),
        ],
        capture_output=True,
        text=True,
    )
    if r1.returncode != 0:
        refuse(f"sidecar input converter failed: {r1.stderr[-400:]}")
    r2 = subprocess.run(
        [
            py,
            str(REPO_ROOT / "scripts" / "b2_graph_sidecar.py"),
            "--dataset-dir",
            str(cat),
            "--store-uri",
            args.store_uri,
            "--out",
            str(side_out),
        ],
        capture_output=True,
        text=True,
    )
    if r2.returncode != 0:
        refuse(f"fresh sidecar run failed: {r2.stderr[-400:]}")
    side = {r["pair_id"]: r for r in corpus_rows(side_out)}
    mismatch = 0
    for r in rows:
        sr = side.get(r["pair_id"])
        if sr is None:
            refuse(f"sidecar output missing {r['pair_id']}; STOP")
        if (
            int(sr.get("edge_ab") or 0) != r["edge_ab"]
            or int(sr.get("common_neighbors") or 0) != r["common_neighbors"]
        ):
            mismatch += 1
        if not sr.get("a_store_id") or not sr.get("b_store_id"):
            refuse(f"side {r['pair_id']} unresolved by the fresh sidecar; STOP")
    if mismatch:
        refuse(f"fresh sidecar disagrees with frozen columns on {mismatch} rows; STOP")
    cov = json.loads(
        (out / "g4-spec-graph-sidecar.coverage.json").read_text(encoding="utf-8")
    )
    sidecar_fp = cov.get("sidecar_fingerprint")
    log(
        runlog,
        "evidence_columns_frozen",
        rows=len(rows),
        sidecar_fp=sidecar_fp,
        mismatch=0,
        unresolved=0,
    )

    # side-overlap flags for the gate runner (informational; the exclusion
    # asserts above already removed every overlapping pair)
    flag_sets: dict[str, set] = {
        "holdout_side_overlap": set(),
        "gate_side_overlap": set(),
        "burned_side_overlap": set(),
    }
    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        for r in corpus_rows(BASE / name):
            for s in (r["record"], r["candidate"]):
                flag_sets["holdout_side_overlap"].add(
                    text_hash(s["title"] + "\n" + s["body"])
                )
    R4 = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
    )
    R5 = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
    )
    for gp in (
        R4 / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4 / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ):
        if gp.is_file():
            for row in corpus_rows(gp):
                for key in ("text_src", "text_dst", "text"):
                    if row.get(key):
                        flag_sets["gate_side_overlap"].add(text_hash(row[key]))
    for bp in (
        R4 / "data/embed-r4/corpus/train.jsonl",
        R4 / "data/embed-r4/corpus/val.jsonl",
        R5 / "data/embed-r5/corpus/train.jsonl",
        R5 / "data/embed-r5/corpus/val.jsonl",
        R5 / "data/embed-r7/corpus/train.jsonl",
        R5 / "data/embed-r7/corpus/val.jsonl",
    ):
        if bp.is_file():
            for row in corpus_rows(bp):
                for side_key in ("record", "candidate"):
                    s = row.get(side_key)
                    if s:
                        flag_sets["burned_side_overlap"].add(
                            text_hash(s["title"] + "\n" + s["body"])
                        )
    flag_counts: Counter = Counter()
    for r in rows:
        h = text_hash(r["record"]["title"] + "\n" + r["record"]["body"])
        for flag, fset in flag_sets.items():
            r[flag] = h in fset
            if r[flag]:
                flag_counts[flag] += 1
    log(runlog, "side_overlap_flags", **dict(flag_counts))

    # ── phase 7: seal ────────────────────────────────────────────────────────
    surface_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in rows])
    )
    write_jsonl(out / "g4-spec-surface.jsonl", rows)
    ids = sorted(r["pair_id"] for r in rows)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    (out / "g4-spec-surface-ids.json").write_text(
        json.dumps(
            {
                "surface": "g4-spec-v1",
                "holdout_ids": ids,
                "surface_ids_sha256": ids_sha,
                "surface_fingerprint": surface_fp,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "surface": "g4-spec-v1 (spec leg of gate g4'; eval-only sealed slice)",
        "law": [
            "docs/specs/b2-graph-prereg-v3-ACTIVATION-NOTE-g4-spec-surface.md "
            "(commit 73ffa50, BINDING activation record)",
            "docs/specs/b2-graph-prereg-v3-ADDENDUM-2.md §A2.7 (spec >= 0.90 "
            "unchanged)",
        ],
        "definition": {
            "evidence_flag": "edge_ab=1 OR common_neighbors>=1",
            "cos": ">= 0.97, frozen embedder (nano:sha256:3b752e06...), corpus "
            "builder formula",
            "label": "not-duplicate (silver §A; disputed -> NOT dup §8.3)",
            "orientation": "one per pair, lexicographic store-id order",
            "subsampling": "none (whole qualifying population)",
            "floor": FLOOR,
        },
        "funnel": {
            "store_snapshot": {
                "memories": len(mem),
                "edges": len(edges),
                "vectors": len(embeddings),
            },
            "hygiene_passed": len(passed),
            "snapshot_fingerprint": snapshot_fp,
            "evidence_pairs": len(evidence_ids),
            "below_cos_band": below_band,
            "silver_dup_quarantined": silver_dup,
            "excluded_by_surface": dict(sorted(excluded.items())),
            "qualified_final": len(kept),
        },
        "excluded_surfaces": [
            "train-v4.5 (fp 3bf7ed53..., pair-level, both orientations + "
            "content hashes)",
            "strata-A (full probe snapshot, 64 distinct pairs)",
            "historical: v4.3-holdout / v4.2-holdout (dataset-v4-holdout) / "
            "v4.1-holdout (b2) / v4.4-holdout",
            "holdout-v4.5 (seal 9eae5ec1...)",
        ],
        "evidence_columns": {
            "frozen_at_cut": True,
            "fresh_sidecar_fp": sidecar_fp,
            "agreement_mismatches": 0,
            "reading": "edge_ab=1 OR common_neighbors>=1",
        },
        "fingerprints": {
            "scheme": "manifest(pair_id pair_sha256 sorted) -> blake2b-256 "
            "(data-contract §5)",
            "surface_seal": surface_fp,
            "surface_ids_sha256": ids_sha,
            "rows": len(rows),
        },
        "labels": {"not-duplicate": len(rows)},
        "seal": {
            "surface_fingerprint": surface_fp,
            "rows": len(rows),
            "sealed_at_cut": True,
            "note": "sealed; one gate-runner read; immutable afterwards",
        },
    }
    (out / "g4-spec-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "g4-spec-run-log.jsonl").write_text(
        "".join(
            json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in runlog
        ),
        encoding="utf-8",
    )
    print(
        f"[g4-spec] DONE rows={len(rows)} "
        f"surface_seal={surface_fp[:16]}... snapshot={snapshot_fp[:16]}..."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateRefusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
