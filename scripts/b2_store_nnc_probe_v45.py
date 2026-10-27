#!/usr/bin/env python3
"""N-near tag-clone probe — real near-topic same-tag pairs from the store.

The v4.3 near-topic heritage (stratum "near-topic", 46 rows in the v4.4-r2
base) is real x real pairing. For the ADDENDUM-2 N-near tag-clone addition
(A2.3) the authored template fallback covers most of the cos range [0.78,
0.92], but the template-parallel geometry bottoms out near 0.806 while the
ratified bands start at 0.78. This probe harvests REAL store pairs with the
natural phrasing diversity the class needs, strictly read-only.

READ-ONLY, not negotiable (the store serves a live engine): mode=ro +
PRAGMA query_only on both databases; content stays in memory and in the
gitignored data/ tree.

Candidate predicate (deterministic, all measured on the snapshot):
  - both records pass the census hygiene gates (1:1 with §5.1);
  - identical NON-EMPTY tag sets (tag_jaccard = 1.0 by construction);
  - same detected body language (script-detect, ru/en confident);
  - |len(body_a) - len(body_b)| <= 60 (bld cap, A2.3);
  - norm1-distinct bodies (never a re-save; §8.3 conservative);
  - store-vector cos in [0.78, 0.92] (A2.3 range);
  - the pair key is NOT a pool pair, and neither side sits on a holdout/
    gate/burned surface (the same exclusion recipe the builder applies);
  - BOTH sides resolve uniquely in the store (ids recorded for the sidecar
    primary join).

Outputs (gitignored data/ tree):
  <out>/nnc-real-snapshot.jsonl  — one row per candidate pair (sorted)
  <out>/nnc-real-probe.json      — numbers + snapshot fingerprint

Usage:
  /usr/bin/python3 scripts/b2_store_nnc_probe_v45.py \
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src"
PIN_PREFIX = "nano:sha256:3b752e06"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
NN_LOW, NN_HIGH = 0.78, 0.92
NN_BLD_MAX = 60


class ProbeRefusal(SystemExit):
    """A probe assert failed — no snapshot is issued."""


def refuse(msg: str) -> None:
    raise ProbeRefusal(f"[nnc-probe-v45] REFUSAL: {msg}")


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


def ro(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        refuse(f"store database not found: {db}")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="b2_store_nnc_probe_v45")
    p.add_argument("--store-uri", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "mode=ro" not in args.store_uri:
        refuse("--store-uri must be a read-only URI (file:...?mode=ro)")
    vesma_db = Path(args.store_uri.removeprefix("file:").split("?", 1)[0])
    vectors_db = vesma_db.parent / "vectors.db"

    scanner = make_scanner(ENGINE_SRC)

    # historical surface exclusion (same recipe as the builder)
    BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
    excluded: set = set()

    def corpus_rows(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]

    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        for r in corpus_rows(BASE / name):
            for s in (r["record"], r["candidate"]):
                excluded.add(text_hash(s["title"] + "\n" + s["body"]))
                excluded.add(text_hash(s["body"]))
                excluded.add(text_hash(s["title"]))
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
                        excluded.add(text_hash(row[key]))
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
                        excluded.add(text_hash(s["title"] + "\n" + s["body"]))
                        excluded.add(text_hash(s["body"]))
    pool_pair_keys: set = set()
    pool = REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl"
    if pool.is_file():
        for r in corpus_rows(pool):
            pool_pair_keys.add(
                tuple(sorted((side_sha(r["record"]), side_sha(r["candidate"]))))
            )

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
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
            counters.exclude("bad-vector")
            continue
        if scanner.scan(title or "", content or "", list(tags)):
            counters.exclude("scanner")
            continue
        body = content or ""
        lang = detect_lang(body)
        probes = (
            text_hash(title or "" + "\n" + body),
            text_hash(body),
            text_hash(title or ""),
        )
        if lang not in ("ru", "en"):
            counters.exclude("lang")
            continue
        if any(p in excluded for p in probes):
            counters.exclude("historical-side")
            continue
        if not tags:
            counters.exclude("no-tags")
            continue
        passed[rid] = {
            "body": body,
            "title": title or "",
            "vec": vec / nrm,
            "tags": list(tags),
            "record_type": mtype,
            "language": lang,
            "created_at": created,
        }

    # bucket by exact tag set + language, then pairwise cos within buckets
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for rid, rec in passed.items():
        key = (tuple(sorted(set(rec["tags"]))), rec["language"])
        buckets[key].append(rid)

    pairs: list[dict] = []
    band_hist: Counter = Counter()
    occupied: set = set()
    for key, ids in sorted(buckets.items(), key=lambda kv: kv[0]):
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        for i in range(len(ids)):
            ia = ids[i]
            a = passed[ia]
            for j in range(i + 1, len(ids)):
                ib = ids[j]
                b = passed[ib]
                if norm1(a["body"]) == norm1(b["body"]):
                    continue  # a re-save, not a near-clone
                bld = abs(len(a["body"]) - len(b["body"]))
                if bld > NN_BLD_MAX:
                    continue
                cos = float(np.dot(a["vec"], b["vec"]))
                if not (NN_LOW <= cos <= NN_HIGH):
                    continue
                pk = tuple(sorted((side_sha(a), side_sha(b))))
                if pk in pool_pair_keys or pk in occupied:
                    continue
                occupied.add(pk)
                band_hist[f"{int(cos * 100) / 100:.2f}"] += 1
                pairs.append(
                    {
                        "a_store_id": ia,
                        "b_store_id": ib,
                        "store_cos": round(cos, 9),
                        "bld": bld,
                        "tags": list(key[0]),
                        "language": key[1],
                        "record": {
                            "title": a["title"],
                            "body": a["body"],
                            "tags": a["tags"],
                            "language": None,
                            "record_type": a["record_type"],
                            "created_at": a["created_at"],
                        },
                        "candidate": {
                            "title": b["title"],
                            "body": b["body"],
                            "tags": b["tags"],
                            "language": None,
                            "record_type": b["record_type"],
                            "created_at": b["created_at"],
                        },
                    }
                )

    pairs.sort(key=lambda p: (p["a_store_id"], p["b_store_id"]))
    snap_fp = hashlib.blake2b(
        "\n".join(
            f"{p['a_store_id']}--{p['b_store_id']} {p['store_cos']:.9f}" for p in pairs
        ).encode("utf-8")
        + b"\n",
        digest_size=32,
    ).hexdigest()
    with (out / "nnc-real-snapshot.jsonl").open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    numbers = {
        "probe": "b2-store-nnc-probe-v45 (A2.3 real near-clone source)",
        "store_uri": args.store_uri,
        "store_snapshot": {
            "memories": len(mem),
            "edges": len(edges),
            "vectors": len(embeddings),
        },
        "hygiene": counters.to_dict() | {"passed_after_lang_and_sides": len(passed)},
        "buckets": len(buckets),
        "pairs": len(pairs),
        "per_0_01_band": dict(sorted(band_hist.items())),
        "cos_range": [
            min((p["store_cos"] for p in pairs), default=None),
            max((p["store_cos"] for p in pairs), default=None),
        ],
        "snapshot_fingerprint": snap_fp,
    }
    (out / "nnc-real-probe.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: numbers[k]
                for k in (
                    "pairs",
                    "buckets",
                    "per_0_01_band",
                    "cos_range",
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
