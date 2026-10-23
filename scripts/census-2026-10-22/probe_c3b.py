#!/usr/bin/env python3
"""C3 follow-up: edge mass in the mint zone, dup cluster structure, pool linkage."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/b2-graph-corpus")
sys.path.insert(0, str(REPO / "src"))
from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

STORE = Path("/var/home/abyss/.vesma/data")
PIN_PREFIX = "nano:sha256:3b752e06"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
MINT_COS, BAND_CAP = 0.92, 0.97


def ro(db: Path):
    import sqlite3
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def norm1(s):
    return " ".join((s or "").lower().split())


def detect_lang(body):
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def main() -> int:
    scanner = make_scanner("/tmp/b2_engine_detectors")
    conn = ro(STORE / "vesma.db")
    mem = {}
    for row in conn.execute(
        "SELECT id, title, content, tags, memory_type, created_at, metadata,"
        " quarantine_reason, status FROM memories"
    ):
        mem[row[0]] = row
    edges = list(conn.execute(
        "SELECT from_memory_id, to_memory_id, kind, provenance, weight FROM memory_edges"))
    conn.close()
    conn = ro(STORE / "vectors.db")
    embeddings = {}
    for mid, blob, meta_json in conn.execute("SELECT id, vector, metadata FROM embeddings"):
        if isinstance(blob, bytes) and len(blob) == BLOB:
            embeddings[mid] = (blob, meta_json or "{}")
    conn.close()

    counters = HygieneCounters()
    counters.candidates_sql = len(mem)
    passed: dict[str, dict] = {}
    for mid, row in mem.items():
        (rid, title, content, tags_json, mtype, created, meta_json, quar, status) = row
        if status != "published":
            counters.exclude("sql-gate:status"); continue
        if mtype not in ("note", "snippet", "fact"):
            counters.exclude("sql-gate:memory_type"); continue
        if len(content or "") < 80:
            counters.exclude("sql-gate:length<80"); continue
        tags = tuple(str(t) for t in json.loads(tags_json or "[]"))
        if "mnemos:no-federate" in tags:
            counters.exclude("no-federate"); continue
        if quar:
            counters.exclude("quarantine"); continue
        emb = embeddings.get(rid)
        if emb is None:
            counters.exclude("missing-embedding"); continue
        blob, meta_j = emb
        meta = json.loads(meta_j)
        pin = meta.get("model_fingerprint")
        if not pin or not str(pin).startswith(PIN_PREFIX):
            counters.exclude("unpinned/mismatch"); continue
        if not meta.get("content_hash"):
            counters.exclude("missing-content-hash"); continue
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
            counters.exclude("bad-vector"); continue
        if scanner.scan(title or "", content or "", tags):
            counters.exclude("scanner"); continue
        passed[rid] = {"body": content or "", "title": title or "", "vec": vec / nrm,
                       "tags": tags, "record_type": mtype,
                       "language": (json.loads(meta_json or "{}").get("canon") or {}).get("language"),
                       "created_at": created}

    ids = sorted(passed)
    idx = {m: i for i, m in enumerate(ids)}
    mat = np.stack([passed[m]["vec"] for m in ids])

    adjacency = defaultdict(set)
    edge_lookup = {}
    for s, d, kind, prov, w in edges:
        if s == d:
            continue
        adjacency[s].add(d); adjacency[d].add(s)
        edge_lookup[frozenset((s, d))] = (kind, prov, w)

    edge_zone = Counter()      # edge pairs among cos>=0.92, by band
    edge_kind_zone = Counter()
    edge_cos = []
    dup_rows = []
    all97 = 0
    edge97 = 0
    B = 512
    for i0 in range(0, len(ids), B):
        block = mat[i0:i0 + B] @ mat.T
        for li, rowv in enumerate(block):
            i = i0 + li
            rowv[i] = -1.0
            for j in np.nonzero(rowv >= MINT_COS)[0]:
                j = int(j)
                if j <= i:
                    continue
                cos = float(rowv[j])
                ia, ib = ids[i], ids[j]
                band = "[0.92,0.97)" if cos < BAND_CAP else "[0.97,1.0]"
                if band == "[0.97,1.0]":
                    all97 += 1
                fro = frozenset((ia, ib))
                edge = edge_lookup.get(fro)
                if edge:
                    edge_zone[band] += 1
                    edge_kind_zone[f"{edge[0]}|{edge[1]}|{band}"] += 1
                    edge_cos.append(cos)
                    if band == "[0.97,1.0]":
                        edge97 += 1
                la, lb = detect_lang(passed[ia]["body"]), detect_lang(passed[ib]["body"])
                if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
                    continue
                A, Bc = passed[ia], passed[ib]
                body_eq = norm1(A["body"]) == norm1(Bc["body"])
                if (body_eq and norm1(A["title"]) == norm1(Bc["title"])
                        and set(A["tags"]) == set(Bc["tags"])
                        and A["record_type"] == Bc["record_type"]
                        and A["language"] == Bc["language"]):
                    dup_rows.append((ia, ib, cos, bool(edge)))

    print("edge pairs in zone cos>=0.92 by band:", dict(edge_zone))
    print("edge kind|prov|band:", dict(edge_kind_zone))
    if edge_cos:
        ec = sorted(edge_cos)
        gaps = [b - a for a, b in zip(ec, ec[1:])]
        print(f"edge-pair cos: n={len(ec)} min={ec[0]:.4f} med={ec[len(ec)//2]:.4f} max={ec[-1]:.4f} max-gap={max(gaps):.4f}")
    print("all pairs cos>=0.97:", all97, "with edge:", edge97)

    # dup cluster structure
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for ia, ib, cos, edge in dup_rows:
        union(ia, ib)
    clusters = Counter()
    for m in list(parent):
        clusters[find(m)] += 1
    sizes = Counter(clusters.values())
    print(f"\nsilver-dup pairs: {len(dup_rows)}; distinct records: {len(parent)}; "
          f"clusters: {len(clusters)}; size histogram {dict(sizes)}")
    print("dup pairs with edge:", sum(1 for *_x, e in dup_rows if e))
    # dup-copies per record: any record in >1 dup pair?
    cnt = Counter()
    for ia, ib, *_ in dup_rows:
        cnt[ia] += 1; cnt[ib] += 1
    print("records in >1 dup pair:", sum(1 for v in cnt.values() if v > 1))

    # pool linkage: pool endpoints among dup sides
    pool_ids = set()
    with (REPO / "datasets/b2-graph/pool_manifest.txt").open() as fh:
        for line in fh:
            a, b = line.split()[0].split("--")
            pool_ids.add(a); pool_ids.add(b)
    inpool = sum(1 for m in parent if m in pool_ids)
    print(f"dup-cluster records that are pool endpoints: {inpool}/{len(parent)}")

    # silver-dup record age spread (envelope delta shape)
    deltas = []
    for ia, ib, cos, edge in dup_rows[:200]:
        ta, tb = passed[ia]["created_at"], passed[ib]["created_at"]
        if ta and tb:
            from datetime import datetime
            dta = datetime.fromisoformat(ta); dtb = datetime.fromisoformat(tb)
            deltas.append(abs((dta - dtb).total_seconds()) / 3600.0)
    if deltas:
        deltas.sort()
        print(f"created_at delta hours: min={deltas[0]:.1f} med={deltas[len(deltas)//2]:.1f} max={deltas[-1]:.1f}")

    # potential positive-with-evidence pool IF owner labels near-dups: edge pairs >=0.97
    print("\nalt-inputs: edge pairs at [0.97,1.0] =", edge97,
          "(candidate pool for owner gold labeling, silver stays NOT-dup until labeled)")

    Path("/tmp/b2_census/c3b_numbers.json").write_text(json.dumps({
        "edge_zone_by_band": dict(edge_zone),
        "edge_kind_zone": dict(edge_kind_zone),
        "all97": all97, "edge97": edge97,
        "dup_pairs": len(dup_rows), "dup_records": len(parent),
        "cluster_sizes": dict(sizes),
        "dup_records_in_pool_endpoints": inpool,
    }, indent=1))
    print("wrote /tmp/b2_census/c3b_numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
