#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 2: hygiene FIRST (frozen order В3).

The hygiene chain of the 2026-10-22 census (scripts/census-2026-10-22/
probe_c3.py, `_compose_record` order) reused verbatim over the FROZEN
snapshot (never the live store):

  status=published ∧ memory_type ∈ {note, snippet, fact} ∧ len(body) ≥ 80
  ∧ no `mnemos:no-federate` tag ∧ no quarantine_reason ∧ embedding present
  ∧ pin startswith nano:sha256:3b752e06 ∧ content_hash present
  ∧ unit-norm vector (tol 1e-3) ∧ engine secret-scan clean.

Additions required by the A-retrial prereg (§5.2, §7.3 assert 1):
  - whole-table PIN census: 100% of ALL snapshot vectors carry the prod pin;
  - edge-endpoint hygiene census: how many edge ENDPOINTS fail hygiene
    (DE boundary: ≤189 endpoints in hygiene fail → FR-P ≈ 1520);
  - quarantine_reason histogram of edge endpoints (ratification table В4).

Outputs (gitignored): hygiene/records.jsonl + vectors.npy + ids.json +
hygiene.json (counters only — no raw text outside the data dir).
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "src"))
from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

SNAP = WORKTREE / "data" / "a-retrial" / "snapshot"
OUT = WORKTREE / "data" / "a-retrial" / "hygiene"
ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/wt/embed-r7-train/src"
NO_FEDERATE_TAG = "mnemos:no-federate"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
PIN_PREFIX = "nano:sha256:3b752e06"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scanner = make_scanner(ENGINE_SRC)
    assert scanner.provenance == "engine", scanner.provenance

    conn = __import__("sqlite3").connect(
        f"file:{(SNAP / 'vesma.db').as_posix()}?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only=ON")
    mem = {}
    for row in conn.execute(
        "SELECT id, title, content, tags, memory_type, created_at, metadata,"
        " quarantine_reason, status FROM memories"
    ):
        mem[row[0]] = row
    edges = list(
        conn.execute("SELECT from_memory_id, to_memory_id, kind FROM memory_edges")
    )
    conn.close()

    conn = __import__("sqlite3").connect(
        f"file:{(SNAP / 'vectors.db').as_posix()}?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only=ON")
    embeddings = {}
    pin_prefix_hits = 0
    total_vectors = 0
    for mid, blob, meta_json in conn.execute("SELECT id, vector, metadata FROM embeddings"):
        total_vectors += 1
        if isinstance(blob, bytes) and len(blob) == BLOB:
            meta = json.loads(meta_json or "{}")
            pin = str(meta.get("model_fingerprint") or "")
            if pin.startswith(PIN_PREFIX):
                pin_prefix_hits += 1
            embeddings[mid] = (blob, meta_json or "{}")
    conn.close()

    assert total_vectors > 0
    pin_share = pin_prefix_hits / total_vectors
    assert pin_share == 1.0, f"pin assert FAILED: {pin_prefix_hits}/{total_vectors}"
    print(f"pin census: {pin_prefix_hits}/{total_vectors} vectors carry prod pin (100%)")

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
        if NO_FEDERATE_TAG in tags:
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
        if not pin:
            counters.exclude("unpinned-embedding")
            continue
        if not str(pin).startswith(PIN_PREFIX):
            counters.exclude("pin-mismatch")
            continue
        if not meta.get("content_hash"):
            counters.exclude("missing-content-hash")
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
            counters.exclude("bad-vector")
            continue
        mmeta = json.loads(meta_json or "{}")
        canon = mmeta.get("canon") if isinstance(mmeta.get("canon"), dict) else {}
        language = canon.get("language")
        language = language if isinstance(language, str) else None
        reasons = scanner.scan(title or "", content or "", tags)
        if reasons:
            for r in reasons:
                counters.exclude(r)
            continue
        passed[rid] = {
            "title": title or "",
            "body": content or "",
            "tags": list(tags),
            "language": language,
            "record_type": mtype,
            "created_at": created,
        }

    # edge-endpoint hygiene census (DE boundary + В4 histogram)
    endpoint_ids = set()
    for s, d, _kind in edges:
        if s != d:
            endpoint_ids.add(s)
            endpoint_ids.add(d)
    endpoint_fail: Counter[str] = Counter()
    fail_ids = set()
    for rid in sorted(endpoint_ids):
        if rid not in passed:
            # classify by re-running the gates in order (same reason names)
            row = mem[rid]
            (r_id, title, content, tags_json, mtype, created, meta_json, quar, status) = row
            tags = tuple(str(t) for t in json.loads(tags_json or "[]"))
            reason = "unknown"
            if status != "published":
                reason = "sql-gate:status"
            elif mtype not in ("note", "snippet", "fact"):
                reason = "sql-gate:memory_type"
            elif len(content or "") < 80:
                reason = "sql-gate:length<80"
            elif NO_FEDERATE_TAG in tags:
                reason = "no-federate"
            elif quar:
                reason = "quarantine"
            elif rid not in embeddings:
                reason = "missing-embedding"
            else:
                blob, meta_j = embeddings[rid]
                meta = json.loads(meta_j)
                pin = str(meta.get("model_fingerprint") or "")
                if not pin:
                    reason = "unpinned-embedding"
                elif not pin.startswith(PIN_PREFIX):
                    reason = "pin-mismatch"
                elif not meta.get("content_hash"):
                    reason = "missing-content-hash"
                else:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    nrm = float(np.linalg.norm(vec))
                    if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
                        reason = "bad-vector"
                    else:
                        rs = scanner.scan(title or "", content or "", tags)
                        reason = rs[0] if rs else "unknown"
            endpoint_fail[reason] += 1
            fail_ids.add(rid)

    # edges dropped by hygiene (any endpoint failing)
    dropped_edges = sum(1 for s, d, _ in edges if s in fail_ids or d in fail_ids)
    alive_edges = len(edges) - dropped_edges

    ids = sorted(passed)
    mat = np.stack(
        [np.frombuffer(embeddings[i][0], dtype=np.float32) for i in ids]
    )
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(OUT / "vectors.npy", mat)
    (OUT / "ids.json").write_text(json.dumps(ids) + "\n")
    with (OUT / "records.jsonl").open("w", encoding="utf-8") as fh:
        for rid in ids:
            fh.write(json.dumps({"id": rid, **passed[rid]}, ensure_ascii=False) + "\n")

    summary = {
        "snapshot_counters": {
            "memories": len(mem),
            "edges": len(edges),
            "vectors": total_vectors,
        },
        "pin_census": {"with_prod_pin": pin_prefix_hits, "total": total_vectors},
        "hygiene_passed": len(passed),
        "hygiene_excluded": counters.to_dict(),
        "edge_endpoints_total": len(endpoint_ids),
        "edge_endpoints_fail_total": len(fail_ids),
        "edge_endpoints_fail_reasons": dict(endpoint_fail),
        "edges_total": len(edges),
        "edges_dropped_by_hygiene": dropped_edges,
        "edges_post_hygiene": alive_edges,
    }
    (OUT / "hygiene.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "hygiene_excluded"}, indent=1))
    print("excluded reasons:", counters.reasons)
    print(
        f"DE boundary check: endpoints failing hygiene = {len(fail_ids)} "
        f"(boundary ≤189); FR-P post-hygiene ≈ {alive_edges}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
