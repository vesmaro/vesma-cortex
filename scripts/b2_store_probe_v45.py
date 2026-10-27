#!/usr/bin/env python3
"""Strata A probe — real re-save (silver-dup) snapshot from the live store.

Law: docs/specs/b2-graph-prereg-v3-ADDENDUM-2.md §A2.4 (ACTIVATED,
owner-ratified 2026-10-22); probe recipe canonized as
docs/experiments/b2-m11-evidence-census.md §5.1 (scripts:
scripts/census-2026-10-22/probe_c3b.py — hygiene gates byte-identical).

READ-ONLY, not negotiable: the store serves a live engine. Both databases
open via file:...?mode=ro + PRAGMA query_only=ON; the probe never writes,
never re-embeds (vectors come from the store as-is, drift ban ADR-0001).
Content exists only in memory and in the gitignored data/ tree.

Silver-dup predicate (deterministic; A2.4 letter):
  norm1(body_a) == norm1(body_b) AND norm1(title_a) == norm1(title_b)
  AND set(tags_a) == set(tags_b) AND record_type equal AND language equal
  (as stored — double-null counts as equal, whitelist §8.1 envelope)
  AND |created_at_a - created_at_b| <= 600 s ("differs by minutes")
Cross-language pairs cannot occur under full-metadata equality, but the
script-detect guard of the census is kept as a belt-and-braces quarantine.

Strata A contract (A2.4):
  - clusters via union-find over the silver pairs; clusters must NOT be
    split between train and holdout (enforced downstream by the builder);
  - every pair must carry a ZERO graph vector by construction:
    edge_ab = 0 AND common_neighbors = 0 in the live store at probe time
    (asserted here — a non-zero vector refuses the probe);
  - store-vector cos must be 1.0 (census geometry; asserted >= 0.9999);
  - the population growth vs the census snapshot (68 pairs, 2026-10-22)
    is DISCLOSED, not hidden.

Outputs (gitignored data/ tree):
  <out>/strata-a-snapshot.jsonl  — one row per silver pair (sorted)
  <out>/strata-a-probe.json      — numbers + snapshot fingerprint

Usage:
  /usr/bin/python3 scripts/b2_store_probe_v45.py \
      --store-uri 'file:/var/home/abyss/.vesma/data/vesma.db?mode=ro' \
      --out data/b2-graph-corpus/v45

Exit codes: 0 ok | 1 gate refusal | 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

CENSUS_PAIRS = 68  # evidence-census snapshot 2026-10-22 (disclosure anchor)
CENSUS_CLUSTERS = 64
PIN_PREFIX = "nano:sha256:3b752e06"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
MINT_COS = 0.92
CREATED_DELTA_MAX_S = 600  # A2.4: "created_at differs by minutes"


class ProbeRefusal(SystemExit):
    """A probe assert failed — no snapshot is issued."""


def refuse(msg: str) -> None:
    raise ProbeRefusal(f"[probe-v45] REFUSAL: {msg}")


def norm1(s: str) -> str:
    return " ".join((s or "").lower().split())


def detect_lang(body: str | None) -> str:
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def ro(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        refuse(f"store database not found: {db}")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="b2_store_probe_v45")
    p.add_argument(
        "--store-uri",
        required=True,
        help="read-only URI of vesma.db (file:...?mode=ro); vectors.db is its sibling",
    )
    p.add_argument(
        "--out", required=True, help="output directory (gitignored data/ tree)"
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

    ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src"
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
            "SELECT from_memory_id, to_memory_id, kind, provenance, weight FROM memory_edges"
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
        import numpy as np

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

    # adjacency for the zero-graph-vector assert
    adjacency: dict[str, set[str]] = {}
    for s, d, _kind, _prov, _w in edges:
        if s == d:
            continue
        adjacency.setdefault(s, set()).add(d)
        adjacency.setdefault(d, set()).add(s)

    ids = sorted(passed)
    _index = {m: i for i, m in enumerate(ids)}
    pairs: list[dict] = []
    quarantined_cross = 0
    for ii, ia in enumerate(ids):
        a = passed[ia]
        for ib in ids[ii + 1 :]:
            b = passed[ib]
            if norm1(a["body"]) != norm1(b["body"]):
                continue
            if norm1(a["title"]) != norm1(b["title"]):
                continue
            if set(a["tags"]) != set(b["tags"]):
                continue
            if a["record_type"] != b["record_type"]:
                continue
            if a["language"] != b["language"]:
                continue
            la, lb = detect_lang(a["body"]), detect_lang(b["body"])
            if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
                quarantined_cross += 1
                continue
            ta = a["created_at"]
            tb = b["created_at"]
            if not ta or not tb:
                continue
            delta = abs(
                (
                    datetime.fromisoformat(tb) - datetime.fromisoformat(ta)
                ).total_seconds()
            )
            if delta > CREATED_DELTA_MAX_S:
                continue
            # store-vector cosine (both sides pinned unit vectors)
            dot = sum(x * y for x, y in zip(a["vec"], b["vec"]))
            # zero-graph-vector assert (A2.4): no edge, no common neighbor
            n_a = adjacency.get(ia, set())
            n_b = adjacency.get(ib, set())
            if ib in n_a or ia in n_b:
                refuse(f"silver-dup pair carries an edge: {ia[:8]}..--{ib[:8]}..; STOP")
            cn = len(n_a & n_b)
            if cn:
                refuse(
                    f"silver-dup pair has common neighbors ({cn}): {ia[:8]}..--{ib[:8]}..; STOP"
                )
            pairs.append(
                {
                    "a_store_id": ia,
                    "b_store_id": ib,
                    "cluster_key": None,
                    "store_cos": round(min(1.0, max(0.0, float(dot))), 9),
                    "created_delta_s": round(delta, 3),
                    "detect_lang": la if la == lb else f"{la}|{lb}",
                    "record": {
                        "title": a["title"],
                        "body": a["body"],
                        "tags": a["tags"],
                        "language": a["language"],
                        "record_type": a["record_type"],
                        "created_at": a["created_at"],
                    },
                    "candidate": {
                        "title": b["title"],
                        "body": b["body"],
                        "tags": b["tags"],
                        "language": b["language"],
                        "record_type": b["record_type"],
                        "created_at": b["created_at"],
                    },
                }
            )

    if not pairs:
        refuse(
            "silver-dup population is empty at probe time; strata A cannot be built; STOP"
        )

    # union-find clusters over shared members
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p["a_store_id"]), find(p["b_store_id"])
        if ra != rb:
            parent[ra] = rb
    clusters: dict[str, list[dict]] = {}
    for p in pairs:
        clusters.setdefault(find(p["a_store_id"]), []).append(p)
    for key, members in clusters.items():
        for p in members:
            p["cluster_key"] = key
    cluster_sizes = Counter(len(m) for m in clusters.values())

    bad_cos = [p for p in pairs if p["store_cos"] < 0.9999]
    if bad_cos:
        refuse(
            f"{len(bad_cos)} silver pairs with store cos < 0.9999; census geometry violated; STOP"
        )

    pairs.sort(key=lambda p: (p["a_store_id"], p["b_store_id"]))
    snapshot_fp = hashlib.blake2b(
        "\n".join(
            f"{p['a_store_id']}--{p['b_store_id']} {p['store_cos']:.9f}" for p in pairs
        ).encode("utf-8")
        + b"\n",
        digest_size=32,
    ).hexdigest()

    with (out / "strata-a-snapshot.jsonl").open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")

    numbers = {
        "probe": "b2-store-probe-v45 (A2.4 strata A snapshot)",
        "store_uri": args.store_uri,
        "store_snapshot": {
            "memories": len(mem),
            "edges": len(edges),
            "vectors": len(embeddings),
        },
        "hygiene": counters.to_dict() | {"passed": len(passed)},
        "silver_dup_pairs": len(pairs),
        "clusters": len(clusters),
        "cluster_size_histogram": {str(k): v for k, v in sorted(cluster_sizes.items())},
        "census_anchor": {
            "pairs": CENSUS_PAIRS,
            "clusters": CENSUS_CLUSTERS,
            "date": "2026-10-22",
        },
        "growth_vs_census_pairs": len(pairs) - CENSUS_PAIRS,
        "growth_vs_census_clusters": len(clusters) - CENSUS_CLUSTERS,
        "store_cos_range": [
            min(p["store_cos"] for p in pairs),
            max(p["store_cos"] for p in pairs),
        ],
        "created_delta_s_max": max(p["created_delta_s"] for p in pairs),
        "created_delta_s_median": sorted(p["created_delta_s"] for p in pairs)[
            len(pairs) // 2
        ],
        "quarantined_cross_lang": quarantined_cross,
        "graph_vector_assert": "edge_ab=0 AND common_neighbors=0 for every pair (refused otherwise)",
        "snapshot_fingerprint": snapshot_fp,
        "snapshot_rows": len(pairs),
    }
    (out / "strata-a-probe.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: numbers[k]
                for k in (
                    "silver_dup_pairs",
                    "clusters",
                    "cluster_size_histogram",
                    "growth_vs_census_pairs",
                    "snapshot_fingerprint",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProbeRefusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
