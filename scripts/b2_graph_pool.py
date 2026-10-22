#!/usr/bin/env python3
"""B2-graph corpus pool builder — slice-1 (v4.4 candidate pool, graph stratum).

Assembles the HYGIENIC CANDIDATE POOL for corpus v4.4 on top of the inherited
v4.3 universe (3131 pairs, untouched). The NEW graph stratum follows the
dataset-v3 construction: pool = edge endpoints of the live engine store,
candidate pairs = store-vector cosine band [0.85, 0.97).

Strata (per pair):
  edge-direct — direct relates_to edge in band  → HARD NEGATIVE
                (silver: edge provenance auto-dedupe weight 2.0; wave-B2
                verdict: relates_to = connectivity link, median body-ratio 0.04)
  edge-common — no direct edge, >=1 common neighbour, cos < mint → NEGATIVE
                (silver: labeling-policy-b2 §8, policy v1.1; 30-pair
                double-check sample mandatory)
  control     — no edge, no common neighbour, cos < mint → NEGATIVE (control,
                "graph adds nothing"; quota small, pre-gate decides)
  mint-top    — cos >= mint (0.92) < 0.97, no direct edge, top-K per record →
                DUPLICATE CANDIDATES ("top-1..3 mint"), silver §8 mechanical
                sifter; dedup by pair-hash

Hygiene: the SAME frozen engine chain as b2_targeted_export
(cortex.data.store_export): SQL gates → no-federate → quarantine →
missing/unpinned embedding → vector decode → ENGINE secrets/danger scanner.
Exclusions carry reason names only. Final pool numbers are counted AFTER
hygiene. Store access strictly read-only (mode=ro + PRAGMA query_only).

Disjointness: pair-level hard asserts vs ALL existing holdouts (v42-276,
b2-180, v43-holdout), the dataset-v3 600, the inherited v4.3 universe,
embed gate sets cb367577 (r5_gate) + r5_gate_v2, and burned surfaces
(v43 translated part = V43-TDUP/TSIB 288, embed r4/r5/r7 corpora).
Side-hash joins are reported per surface (record pools are shared by
corpus-family design; the pre-gate carves eval sets from the flagged map).

Determinism: seed 42; no wall-clock in outputs; double run must be
byte-identical (checked externally). Numbers only leave this script toward
the repo; record content stays under gitignored data/.

Output layout (default):
  <out>/pool_candidates.jsonl      full populations, §3 sides + metadata
  <out>/pool_manifest.txt          pair_id pair_sha256 lines (sorted)
  <out>/coverage.json              counters, fingerprints, hygiene reasons
  <out>/silver-edge-common-30.jsonl double-check sample (features, no content)
  <out>/quotas_proposal.json       400/600/800 variants + component allocation
  <out>/run-log.jsonl              append-only run log
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    corpus_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.data.store_export import (  # noqa: E402
    HygieneCounters,
    _compose_record,
    _ro_connection,
    make_scanner,
)

STORE_DIR = Path("/var/home/abyss/.vesma/data")
STORE_DB = STORE_DIR / "vesma.db"
VECTORS_DB = STORE_DIR / "vectors.db"
ENGINE_SRC_DEFAULT = Path("/tmp/b2_engine_detectors")

BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
R4_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
)
R5_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
)

UNIVERSE = [
    ("v43-train", BASE / "v43-localbuild/train.jsonl"),
    ("v43-holdout", BASE / "v43-holdout/holdout.jsonl"),
    ("v42-holdout-276", BASE / "dataset-v4-holdout/holdout.jsonl"),
    ("b2-holdout-180", BASE / "b2/holdout.jsonl"),
]
DATASET_V3 = [
    ("b2-train-420", BASE / "b2/train.jsonl"),
    ("b2-holdout-180", BASE / "b2/holdout.jsonl"),
]
HOLDOUT_FILES = {
    "v43-holdout": BASE / "v43-holdout/holdout.jsonl",
    "v42-holdout-276": BASE / "dataset-v4-holdout/holdout.jsonl",
    "b2-holdout-180": BASE / "b2/holdout.jsonl",
}
GATE_FILES = {
    "r5_gate-cb367577": R4_WT / "data/embed-r4/r5_gate/gate_set.jsonl",
    "r5_gate-tatoeba_notes": R4_WT / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
    "r5_gate_v2": R5_WT / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
    "r5_gate_v2-run-b": R5_WT / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
    "r5_gate_v2-frozen_pool": R5_WT / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
}
BURNED_EMBED = {
    "embed-r4-train": R4_WT / "data/embed-r4/corpus/train.jsonl",
    "embed-r4-val": R4_WT / "data/embed-r4/corpus/val.jsonl",
    "embed-r4-gen-llm0": R4_WT / "data/embed-r4/gen/llm-translations.shard0.jsonl",
    "embed-r4-gen-llm1": R4_WT / "data/embed-r4/gen/llm-translations.shard1.jsonl",
    "embed-r4-gen-real0": R4_WT / "data/embed-r4/gen/real-translations.shard0.jsonl",
    "embed-r4-gen-real1": R4_WT / "data/embed-r4/gen/real-translations.shard1.jsonl",
    "embed-r5-train": R5_WT / "data/embed-r5/corpus/train.jsonl",
    "embed-r5-val": R5_WT / "data/embed-r5/corpus/val.jsonl",
    "embed-r7-train": R5_WT / "data/embed-r7/corpus/train.jsonl",
    "embed-r7-val": R5_WT / "data/embed-r7/corpus/val.jsonl",
    "embed-r7-gen-llm0": R5_WT / "data/embed-r7/gen/llm-translations-r7.shard0.jsonl",
    "embed-r7-gen-llm1": R5_WT / "data/embed-r7/gen/llm-translations-r7.shard1.jsonl",
}

EMBED_DIM = 384
BLOB = EMBED_DIM * 4
UNIT_TOL = 1e-3
BAND_LO, BAND_HI = 0.85, 0.97
MINT_THRESHOLD = 0.92
TOP_K_MINT = 3
SEED = 42
DOUBLE_CHECK_SAMPLE = 30

NO_FEDERATE_TAG = "mnemos:no-federate"

DATE_RE = re.compile(
    r"\b\d{1,4}[-./]\d{1,2}[-./]\d{1,4}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"
)
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")
HASHY_RE = re.compile(r"\b[0-9a-f]{8,64}\b", re.IGNORECASE)
ENV_MARKERS = (
    "записан",
    "записано",
    "записана",
    "фиксир",
    "зафиксир",
    "сохран",
    "создан",
    "created",
    "logged",
    "saved",
    "captured",
    "note at",
    "конспект",
    "обновл",
    "updated",
    "редакц",
)


def log(runlog: list[dict], event: str, **fields) -> None:
    entry = {"event": event, **fields}
    runlog.append(entry)
    print(
        f"[pool] {event}: " + " ".join(f"{k}={v}" for k, v in fields.items()),
        file=sys.stderr,
    )


# ── text helpers (content stays local; only counts/hash prefixes leave) ──────


def text_hash(text: str) -> str:
    """NFC → lower → collapse whitespace → sha256 (embed-gate side-hash fn)."""
    n = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def norm_tokens(text: str) -> list[str]:
    n = unicodedata.normalize("NFC", (text or "").lower())
    return n.split()


def is_fact_token(tok: str) -> bool:
    """§8.2 key-fact markers: numbers, dates, versions, sums, hash-like ids."""
    core = tok.strip("\"'()[]{}.,;:!?—«»")
    if not core:
        return False
    if any(ch.isdigit() for ch in core):
        return True
    if DATE_RE.search(core) or VERSION_RE.search(core) or HASHY_RE.search(core):
        return True
    return False


def envelope_context(tokens: list[str], idx: int, window: int = 4) -> bool:
    lo, hi = max(0, idx - window), min(len(tokens), idx + window + 1)
    ctx = " ".join(tokens[lo:hi])
    return any(m in ctx for m in ENV_MARKERS)


def side_sha(s: dict) -> str:
    """Artifact-side projection hash (title, body, tags, language, record_type)."""
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


def pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((side_sha(row["record"]), side_sha(row["candidate"]))))


def corpus_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ── §8 mechanical sifter (silver; conservative; disputed bucket honest) ──────


def sift_pair(rec: dict, cand: dict) -> dict:
    """Mechanical labeling-policy-b2 §8 sifter over the artifact sides.

    Classes: dup-silver (whitelist/cosmetic-only delta, metadata identical),
    not-dup-identity (byte/normalization identity with metadata delta → T2),
    not-dup-razor (>=1 key-fact token delta → T3/§8.1 key content),
    not-dup-metadata (content identical, metadata differs → §1.3/T2),
    disputed (mixed evidence the sifter cannot separate).
    """
    t_a = norm_tokens(rec["title"] + " \n " + rec["body"])
    t_b = norm_tokens(cand["title"] + " \n " + cand["body"])
    sm = difflib.SequenceMatcher(a=t_a, b=t_b, autojunk=False)
    razor, envelope, other = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for origin, other_side, base in ((t_a, t_b, i1), (t_b, t_a, j1)):
            for off, tok in enumerate(
                origin[i1:i2] if origin is t_a else origin[j1:j2]
            ):
                core = tok.strip("\"'()[]{}.,;:!?—«»")
                if not core:
                    continue
                idx = base + off
                if is_fact_token(tok):
                    near = envelope_context(origin, idx) or (
                        idx < len(other_side) and envelope_context(other_side, idx)
                    )
                    if near:
                        envelope.append(core)
                    else:
                        razor.append(core)
                else:
                    other.append(core)
    meta_delta = {
        "tags_delta": sorted(rec.get("tags") or []) != sorted(cand.get("tags") or []),
        "type_delta": rec.get("record_type") != cand.get("record_type"),
        "lang_delta": rec.get("language") != cand.get("language"),
    }
    crosslang = (
        bool(rec.get("language"))
        and bool(cand.get("language"))
        and rec.get("language") != cand.get("language")
    )
    if crosslang:
        # mechanical token-diff is monolingual: translation twins reformat
        # dates/numbers without changing facts — no silver verdict here.
        return {
            "verdict": "disputed-crosslang",
            "razor": len(razor),
            "envelope": len(envelope),
            "other_tokens": len(other),
            "meta_delta": meta_delta,
            "disputed": True,
            "crosslang": True,
        }
    content_ident = t_a == t_b
    if content_ident:
        if any(meta_delta.values()):
            return {
                "verdict": "not-dup-metadata",
                "razor": 0,
                "envelope": 0,
                "other_tokens": 0,
                "meta_delta": meta_delta,
                "crosslang": False,
            }
        return {
            "verdict": "dup-silver",
            "razor": 0,
            "envelope": 0,
            "other_tokens": 0,
            "meta_delta": meta_delta,
            "crosslang": False,
        }
    if razor and envelope and not other:
        # both classes touched: key fact changed → not-dup wins (§1.2/§8.1),
        # but flag disputed so the double-check can audit the envelope reading
        return {
            "verdict": "not-dup-razor",
            "razor": len(razor),
            "envelope": len(envelope),
            "other_tokens": len(other),
            "meta_delta": meta_delta,
            "disputed": True,
            "crosslang": False,
        }
    if razor:
        return {
            "verdict": "not-dup-razor",
            "razor": len(razor),
            "envelope": len(envelope),
            "other_tokens": len(other),
            "meta_delta": meta_delta,
            "crosslang": False,
        }
    if envelope or other:
        return {
            "verdict": "dup-silver",
            "razor": 0,
            "envelope": len(envelope),
            "other_tokens": len(other),
            "meta_delta": meta_delta,
            "crosslang": False,
        }
    return {
        "verdict": "disputed",
        "razor": 0,
        "envelope": 0,
        "other_tokens": 0,
        "meta_delta": meta_delta,
        "disputed": True,
        "crosslang": False,
    }


# ── store load (strictly read-only; mirrors probe slice-0) ───────────────────


def load_store() -> dict:
    conn = _ro_connection(STORE_DB)
    try:
        mem_rows: dict[str, tuple] = {}
        for row in conn.execute(
            "SELECT id, title, content, tags, memory_type, created_at, metadata,"
            " quarantine_reason, status FROM memories"
        ):
            mem_rows[row[0]] = row
        edges = []
        for src, dst, kind, prov, weight, created in conn.execute(
            "SELECT from_memory_id, to_memory_id, kind, provenance, weight,"
            " created_at FROM memory_edges"
        ):
            edges.append((src, dst, kind, prov, weight, created))
    finally:
        conn.close()

    vectors: dict[str, np.ndarray] = {}
    conn = _ro_connection(VECTORS_DB)
    try:
        for mid, blob, meta_json in conn.execute(
            "SELECT id, vector, metadata FROM embeddings"
        ):
            if isinstance(blob, bytes) and len(blob) == BLOB:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if np.isfinite(norm) and abs(norm - 1.0) <= UNIT_TOL:
                    vectors[mid] = (blob, vec, meta_json or "{}")
    finally:
        conn.close()
    return {"mem_rows": mem_rows, "edges": edges, "vectors": vectors}


def components_of(edges: list[tuple]) -> dict[str, int]:
    adj: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for s, d, *_ in edges:
        if s != d:
            adj[s].add(d)
            adj[d].add(s)
            nodes.add(s)
            nodes.add(d)
    comp: dict[str, int] = {}
    cid = 0
    for n in sorted(nodes):
        if n in comp:
            continue
        cid += 1
        stack = [n]
        comp[n] = cid
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp[y] = cid
                    stack.append(y)
    return comp


def project_tags(tags_json: str) -> list[str]:
    try:
        tags = json.loads(tags_json or "[]")
    except json.JSONDecodeError:
        return []
    return sorted(t for t in tags if isinstance(t, str) and t.startswith("project:"))


# ── main build ────────────────────────────────────────────────────────────────


def build(out_dir: Path) -> int:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runlog: list[dict] = []

    scanner = make_scanner(ENGINE_SRC_DEFAULT)
    log(runlog, "scanner", provenance=scanner.provenance)
    if scanner.provenance != "engine":
        print(
            "[pool] FATAL: prereg hygiene expects the ENGINE detectors", file=sys.stderr
        )
        return 2

    store = load_store()
    log(
        runlog,
        "store_snapshot",
        memories=len(store["mem_rows"]),
        edges=len(store["edges"]),
        vectors=len(store["vectors"]),
    )

    comp = components_of(store["edges"])
    comp_sizes = Counter(comp.values())
    log(
        runlog,
        "components",
        count=len(comp_sizes),
        top10=sorted(comp_sizes.values(), reverse=True)[:10],
    )

    edge_lookup: dict[frozenset, dict] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for s, d, kind, prov, weight, created in store["edges"]:
        if s == d:
            continue
        edge_lookup[frozenset((s, d))] = {
            "kind": kind,
            "provenance": prov,
            "weight": weight,
            "created_at": created,
        }
        adjacency[s].add(d)
        adjacency[d].add(s)

    # ── step 1: endpoint pool through the frozen hygiene chain ───────────
    endpoints = sorted({x for s, d, *_ in store["edges"] for x in (s, d)})
    counters = HygieneCounters()
    counters.candidates_sql = len(endpoints)

    embeddings = {
        mid: (payload[0], payload[2]) for mid, payload in store["vectors"].items()
    }
    pool_records: dict[str, dict] = {}
    for mid in endpoints:
        row = store["mem_rows"].get(mid)
        if row is None:
            counters.exclude("not-in-memories")
            continue
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
        rec = _compose_record(
            (rid, title, content, tags_json, mtype, created, meta_json, quar),
            embeddings,
            counters,
            scanner,
        )
        if rec is None:
            continue
        pool_records[rec.id] = {
            "record": rec,
            "component": comp.get(mid, 0),
            "projects": project_tags(tags_json),
            "title": rec.title,
            "body": rec.body,
        }
    log(
        runlog,
        "hygiene_pool",
        endpoints=len(endpoints),
        passed=len(pool_records),
        excluded=counters.excluded_total,
        reasons=dict(sorted(counters.reasons.items())),
    )

    # ── step 2: band pairs + strata over the hygiene-passed pool ─────────
    ids = sorted(pool_records)
    vecs = np.stack([pool_records[i]["record"].vector for i in ids])
    cos = vecs @ vecs.T
    iu = np.triu_indices(len(ids), k=1)
    vals = cos[iu]
    mask = (vals >= BAND_LO) & (vals < BAND_HI)
    bi, bj = iu[0][mask], iu[1][mask]
    log(runlog, "band_pairs", total=int(mask.sum()))

    mint_neighbors: dict[str, list[tuple[float, str]]] = defaultdict(list)
    strata_counter = Counter()
    pair_rows: list[dict] = []
    for x, y in zip(bi.tolist(), bj.tolist()):
        a, b = ids[x], ids[y]
        similarity = round(float(cos[x, y]), 4)
        fs = frozenset((a, b))
        edge = edge_lookup.get(fs)
        common = adjacency[a] & adjacency[b]
        ra, rb = pool_records[a], pool_records[b]
        if edge is not None:
            stratum = "edge-direct"
        elif similarity >= MINT_THRESHOLD:
            stratum = "mint-top"
        elif common:
            stratum = "edge-common"
        else:
            stratum = "control"
        strata_counter[stratum] += 1

        first, second = sorted(
            (a, b), key=lambda m: (pool_records[m]["record"].created_at, m)
        )
        pair_id = f"{first}--{second}"
        rec_obj = pool_records[first]["record"]
        cand_obj = pool_records[second]["record"]
        pair = {
            "record": rec_obj.side(),
            "candidate": cand_obj.side(),
            "similarity": similarity,
        }
        sifter = sift_pair(
            {
                "title": rec_obj.title,
                "body": rec_obj.body,
                "tags": rec_obj.tags,
                "language": rec_obj.language,
                "record_type": rec_obj.record_type,
            },
            {
                "title": cand_obj.title,
                "body": cand_obj.body,
                "tags": cand_obj.tags,
                "language": cand_obj.language,
                "record_type": cand_obj.record_type,
            },
        )
        if stratum == "mint-top":
            mint_neighbors[a].append((similarity, pair_id))
            mint_neighbors[b].append((similarity, pair_id))

        pair_rows.append(
            {
                "pair_id": pair_id,
                "id_a": first,
                "id_b": second,
                "stratum": stratum,
                "similarity": similarity,
                "edge": edge,
                "common_neighbors": len(common),
                "neighbor_jaccard": round(
                    len(common) / len(adjacency[a] | adjacency[b]), 4
                )
                if (adjacency[a] | adjacency[b])
                else 0.0,
                "component_a": ra["component"],
                "component_b": rb["component"],
                "projects_a": ra["projects"],
                "projects_b": rb["projects"],
                "sifter": sifter,
                "pair": pair,
                "pair_sha256": pair_sha256(pair),
            }
        )

    pair_rows.sort(key=lambda r: r["pair_id"])
    log(runlog, "strata_populations", **dict(sorted(strata_counter.items())))

    # mint-top: keep only top-K per record (top-1..3 mint), then dedup by pair-hash
    topk_pair_ids: set[str] = set()
    for mid, lst in mint_neighbors.items():
        lst_sorted = sorted(lst, key=lambda t: (-t[0], t[1]))
        for _, pid in lst_sorted[:TOP_K_MINT]:
            topk_pair_ids.add(pid)
    mint_before = strata_counter["mint-top"]
    pair_rows = [
        r
        for r in pair_rows
        if r["stratum"] != "mint-top" or r["pair_id"] in topk_pair_ids
    ]
    log(
        runlog,
        "mint_topk",
        before=mint_before,
        after=len(topk_pair_ids),
        top_k=TOP_K_MINT,
    )

    # ── step 3: dedup by pair-hash vs existing corpora + within pool ─────
    # three layers: artifact-side projection, raw text pair, store id pair.
    occupied_art: dict[tuple[str, str], str] = {}
    occupied_text: dict[tuple[str, str], str] = {}
    resolve_index: dict[tuple[str, str], str] = {}
    for mid, pr in pool_records.items():
        resolve_index.setdefault((pr["record"].title, pr["record"].body), mid)
    resolve_stats = Counter()
    occupied_ids: set[frozenset] = set()
    for source, files in (("universe-v43", UNIVERSE), ("dataset-v3", DATASET_V3)):
        for name, path in files:
            for row in corpus_rows(path):
                occupied_art.setdefault(
                    pair_key(row), f"{source}:{name}:{row.get('pair_id', '')}"
                )
                tk = tuple(
                    sorted(
                        (
                            text_hash(
                                row["record"]["title"] + "\n" + row["record"]["body"]
                            ),
                            text_hash(
                                row["candidate"]["title"]
                                + "\n"
                                + row["candidate"]["body"]
                            ),
                        )
                    )
                )
                occupied_text.setdefault(
                    tk, f"{source}:{name}:{row.get('pair_id', '')}"
                )
                ia = resolve_index.get((row["record"]["title"], row["record"]["body"]))
                ib = resolve_index.get(
                    (row["candidate"]["title"], row["candidate"]["body"])
                )
                if ia is not None and ib is not None and ia != ib:
                    occupied_ids.add(frozenset((ia, ib)))
                else:
                    resolve_stats["unresolved-or-partial"] += 1
    burned_translated_keys: set[tuple[str, str]] = set()
    for row in corpus_rows(BASE / "v43-localbuild/train.jsonl"):
        if str(row.get("part", "")).startswith("translated") or str(
            row.get("pair_id", "")
        ).startswith(("V43-TDUP", "V43-TSIB")):
            burned_translated_keys.add(pair_key(row))

    dedup = Counter()
    seen_keys: set[tuple[str, str]] = set()
    final_rows: list[dict] = []
    for r in pair_rows:
        ck = pair_key(r["pair"])
        tk = tuple(
            sorted(
                (
                    text_hash(
                        r["pair"]["record"]["title"]
                        + "\n"
                        + r["pair"]["record"]["body"]
                    ),
                    text_hash(
                        r["pair"]["candidate"]["title"]
                        + "\n"
                        + r["pair"]["candidate"]["body"]
                    ),
                )
            )
        )
        if ck in occupied_art:
            dedup["occupied-artifact-key"] += 1
            continue
        if tk in occupied_text:
            dedup["occupied-text-key"] += 1
            continue
        if frozenset((r["id_a"], r["id_b"])) in occupied_ids:
            dedup["occupied-store-id-pair"] += 1
            continue
        if ck in burned_translated_keys:
            dedup["occupied-burned-translated"] += 1
            continue
        if ck in seen_keys:
            dedup["within-pool"] += 1
            continue
        seen_keys.add(ck)
        r["content_pair_key"] = "|".join(ck)
        final_rows.append(r)
    log(
        runlog,
        "dedup",
        dropped=dict(sorted(dedup.items())),
        kept=len(final_rows),
        corpus_rows_unresolved=resolve_stats["unresolved-or-partial"],
    )

    # ── step 4: disjointness — pair-level hard asserts + side-hash joins ─
    holdout_pairs: dict[str, set] = {}
    holdout_sides: dict[str, set] = {}
    for name, path in HOLDOUT_FILES.items():
        rows = corpus_rows(path)
        hp = set()
        hs = set()
        for row in rows:
            hp.add(pair_key(row))
            for side in (row["record"], row["candidate"]):
                hs.add(text_hash(side["title"] + "\n" + side["body"]))
                hs.add(text_hash(side["body"]))
                hs.add(text_hash(side["title"]))
        holdout_pairs[name] = hp
        holdout_sides[name] = hs

    gate_side_hashes: dict[str, set] = {}
    gate_pair_keys: set = set()
    for name, path in GATE_FILES.items():
        if not path.is_file():
            log(runlog, "gate_file_missing", name=name, path=str(path))
            continue
        hs = set()
        for row in corpus_rows(path):
            if "text_sha_src" in row:
                hs.add(row["text_sha_src"])
                hs.add(row["text_sha_dst"])
            else:
                hs.add(text_hash(row.get("text_src", "")))
                hs.add(text_hash(row.get("text_dst", "")))
            if row.get("text_src") and row.get("text_dst"):
                gate_pair_keys.add(
                    tuple(
                        sorted(
                            (
                                text_hash(
                                    row["text_src"] + "\n" + (row.get("lang_src") or "")
                                ),
                                text_hash(
                                    row["text_dst"] + "\n" + (row.get("lang_dst") or "")
                                ),
                            )
                        )
                    )
                )
            src_text = row.get("text_src") or ""
            dst_text = row.get("text_dst") or ""
            if src_text:
                hs.add(text_hash(src_text))
            if dst_text:
                hs.add(text_hash(dst_text))
        gate_side_hashes[name] = hs

    burned_side_hashes: dict[str, set] = {}
    for name, path in BURNED_EMBED.items():
        if not path.is_file():
            log(runlog, "burned_file_missing", name=name, path=str(path))
            continue
        hs = set()
        for row in corpus_rows(path):
            for key in ("text", "text_src", "text_dst"):
                if row.get(key):
                    hs.add(text_hash(row[key]))
        burned_side_hashes[name] = hs

    asserts: dict[str, object] = {}
    side_join_report: dict[str, dict] = {}
    surviving: list[dict] = []
    for r in final_rows:
        ck = pair_key(r["pair"])
        overlap_holdout = [n for n, hp in holdout_pairs.items() if ck in hp]
        if overlap_holdout:
            asserts.setdefault("pair_overlap_holdout", []).append(
                {"pair_id": r["pair_id"], "surfaces": overlap_holdout}
            )
            continue
        rec_t = r["pair"]["record"]
        cand_t = r["pair"]["candidate"]
        my_sides = {
            text_hash(rec_t["title"] + "\n" + rec_t["body"]),
            text_hash(rec_t["body"]),
            text_hash(rec_t["title"]),
            text_hash(cand_t["title"] + "\n" + cand_t["body"]),
            text_hash(cand_t["body"]),
            text_hash(cand_t["title"]),
        }
        gate_hits = sorted(
            name for name, hs in gate_side_hashes.items() if my_sides & hs
        )
        if gate_hits:
            asserts.setdefault("side_overlap_gate_FLAGGED", []).append(
                {"pair_id": r["pair_id"], "surfaces": gate_hits}
            )
        burned_hits = sorted(
            name for name, hs in burned_side_hashes.items() if my_sides & hs
        )
        if burned_hits:
            asserts.setdefault("side_overlap_burned_embed_FLAGGED", []).append(
                {"pair_id": r["pair_id"], "surfaces": burned_hits}
            )
        side_flags = {}
        for name, hs in holdout_sides.items():
            hit = bool(my_sides & hs)
            side_flags[name] = hit
            cell = side_join_report.setdefault(
                name, {"pairs_touching_holdout_sides": 0}
            )
            if hit:
                cell["pairs_touching_holdout_sides"] += 1
        if gate_hits:
            side_join_report.setdefault("gate-side-flags", {"pairs": 0})["pairs"] += 1
        if burned_hits:
            side_join_report.setdefault("burned-embed-side-flags", {"pairs": 0})[
                "pairs"
            ] += 1
        r["holdout_side_overlap"] = side_flags
        r["gate_side_overlap"] = gate_hits
        r["burned_side_overlap"] = burned_hits
        surviving.append(r)
    hard = {k: len(v) for k, v in asserts.items()}
    log(
        runlog,
        "disjointness_asserts",
        violations=hard,
        side_join=dict(side_join_report),
    )
    if asserts:
        (out_dir / "assert_violations.json").write_text(
            json.dumps(
                {k: v[:200] for k, v in asserts.items()},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # ── step 5: control trim + write pool + manifest + fingerprints ──────
    CONTROL_FILE_CAP = 5000
    control_rows = [r for r in surviving if r["stratum"] == "control"]
    by_comp: dict[int, list[dict]] = defaultdict(list)
    for r in sorted(control_rows, key=lambda r: r["pair_id"]):
        by_comp[r["component_a"]].append(r)
    control_kept_ids: set[str] = set()
    comps_sorted = sorted(by_comp)
    while len(control_kept_ids) < CONTROL_FILE_CAP:
        progressed = False
        for c in comps_sorted:
            if by_comp[c] and len(control_kept_ids) < CONTROL_FILE_CAP:
                control_kept_ids.add(by_comp[c].pop(0)["pair_id"])
                progressed = True
        if not progressed:
            break
    file_rows = [
        r
        for r in surviving
        if r["stratum"] != "control" or r["pair_id"] in control_kept_ids
    ]
    file_rows.sort(key=lambda r: r["pair_id"])
    log(
        runlog,
        "control_file_trim",
        population=len(control_rows),
        kept=len(control_kept_ids),
        cap=CONTROL_FILE_CAP,
        selection="deterministic round-robin over components, pair_id order",
    )

    pool_path = out_dir / "pool_candidates.jsonl"
    with pool_path.open("w", encoding="utf-8") as fh:
        for r in file_rows:
            out_row = {k: v for k, v in r.items() if k != "pair"}
            out_row["record"] = r["pair"]["record"]
            out_row["candidate"] = r["pair"]["candidate"]
            fh.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in file_rows])
    (out_dir / "pool_manifest.txt").write_bytes(manifest)
    pool_fp = corpus_fingerprint(manifest)

    strata_final = Counter(r["stratum"] for r in file_rows)
    sift_final = Counter((r["stratum"], r["sifter"]["verdict"]) for r in file_rows)
    comp_dist = Counter(r["component_a"] for r in file_rows)
    proj_counter = Counter()
    for r in file_rows:
        for p in r["projects_a"] or ["project:none"]:
            proj_counter[p] += 1
    log(
        runlog,
        "pool_final",
        pairs=len(file_rows),
        strata=dict(sorted(strata_final.items())),
        pool_fingerprint=pool_fp,
    )

    # ── step 6: silver double-check sample — 30 EDGE-COMMON pairs ────────
    edge_common = [r for r in file_rows if r["stratum"] == "edge-common"]
    rng = random.Random(SEED)
    sample = sorted(
        rng.sample(edge_common, min(DOUBLE_CHECK_SAMPLE, len(edge_common))),
        key=lambda r: r["pair_id"],
    )
    sample_rows = []
    for r in sample:
        s = r["sifter"]
        sample_rows.append(
            {
                "pair_id": r["pair_id"],
                "similarity": r["similarity"],
                "common_neighbors": r["common_neighbors"],
                "neighbor_jaccard": r["neighbor_jaccard"],
                "component_a": r["component_a"],
                "projects_a": r["projects_a"],
                "silver_label": s["verdict"],
                "silver_basis": "labeling-policy-b2 §8 policy-v1.1 mechanical-sifter",
                "razor_tokens": s["razor"],
                "envelope_tokens": s["envelope"],
                "other_tokens": s["other_tokens"],
                "meta_delta": s["meta_delta"],
                "disputed": bool(s.get("disputed")),
                "record_sha": side_sha(r["pair"]["record"])[:16],
                "candidate_sha": side_sha(r["pair"]["candidate"])[:16],
            }
        )
    (out_dir / "silver-edge-common-30.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sample_rows
        ),
        encoding="utf-8",
    )
    sample_verdicts = Counter(row["silver_label"] for row in sample_rows)
    log(
        runlog,
        "double_check_sample",
        size=len(sample_rows),
        verdicts=dict(sorted(sample_verdicts.items())),
        seed=SEED,
    )

    # ── step 7: quota proposals 400 / 600 / 800 ──────────────────────────
    # EMPIRICAL PIVOT (slice-1 finding): the mechanical §8 sifter finds ZERO
    # dup-silver pairs inside the graph stratum — the edge layer already
    # absorbed the duplicate mass at mint time (relates_to auto-dedupe ≥0.92;
    # the non-edge mint zone is fact-drifted twins = T3-class hard negatives).
    # The graph stratum is therefore an ALL-NEGATIVE stratum; the v4.3-style
    # 46/54 class balance needs positive injection (T0/T1 synthesis per
    # labeling-policy §4, or an LLM §8 pass over mint-top) — a pre-gate v3
    # decision, NOT a pool-construction decision. Quota variants below split
    # the negative mass by stratum role and compute the pool-level class
    # consequence honestly.
    pop = {
        "edge-direct": [r for r in file_rows if r["stratum"] == "edge-direct"],
        "edge-common": [r for r in file_rows if r["stratum"] == "edge-common"],
        "control": [r for r in file_rows if r["stratum"] == "control"],
        "mint-top": [r for r in file_rows if r["stratum"] == "mint-top"],
    }
    NEG_SPLIT = {
        "edge-direct": 0.40,
        "edge-common": 0.25,
        "mint-top": 0.25,
        "control": 0.10,
    }
    base_pairs, base_dup = 3131, 1267  # inherited v4.3 universe (measured)
    quotas = {}
    for total in (400, 600, 800):
        per_comp_cap = max(1, round(total * 0.15))
        variant_rows = {}
        for stratum in ("edge-direct", "edge-common", "control", "mint-top"):
            quota = round(total * NEG_SPLIT[stratum])
            rows = sorted(pop[stratum], key=lambda r: r["pair_id"])
            per_comp: dict[int, int] = defaultdict(int)
            picked = []
            for r in rows:
                if len(picked) >= quota:
                    break
                c = r["component_a"]
                if per_comp[c] >= per_comp_cap:
                    continue
                per_comp[c] += 1
                picked.append(r["pair_id"])
            if len(picked) < quota:  # relax the cap only if unreachable
                picked_set = set(picked)
                for r in rows:
                    if len(picked) >= quota:
                        break
                    if r["pair_id"] not in picked_set:
                        picked.append(r["pair_id"])
                        picked_set.add(r["pair_id"])
            variant_rows[stratum] = sorted(picked)
        comp_cov = set()
        for stratum, pids in variant_rows.items():
            byid = {r["pair_id"]: r for r in pop[stratum]}
            for pid in pids:
                comp_cov.add(byid[pid]["component_a"])
        dup_after = base_dup  # graph stratum adds zero duplicates
        pool_total_after = base_pairs + total
        quotas[f"graph-stratum-{total}"] = {
            "total": total,
            "allocation": {k: len(v) for k, v in sorted(variant_rows.items())},
            "stratum_label_semantics": {
                "edge-direct": "not-duplicate (HARD NEG, silver: edge provenance auto-dedupe)",
                "edge-common": "not-duplicate (silver §8, 30-pair double-check)",
                "mint-top": "not-duplicate razor-zone T3-class (silver §8 mechanical; 0 dup-silver found)",
                "control": "not-duplicate (no graph evidence)",
            },
            "label_mix": {"duplicate": 0, "not-duplicate": total},
            "pool_level_class_consequence": {
                "universe_before": {"duplicate": base_dup, "total": base_pairs},
                "universe_after": {"duplicate": dup_after, "total": pool_total_after},
                "dup_share_after": round(dup_after / pool_total_after, 4),
                "dups_needed_for_4654": max(
                    0, round(0.46 * pool_total_after) - base_dup
                ),
                "positive_injection_source": "pre-gate v3 decision (T0/T1 synthesis per policy §4 or LLM §8 pass over mint-top)",
            },
            "components_covered": len(comp_cov),
            "components_total": len(comp_sizes),
            "per_component_cap": per_comp_cap,
            "feasible": all(
                len(variant_rows[s]) == round(total * NEG_SPLIT[s])
                for s in variant_rows
            ),
        }
    (out_dir / "quotas_proposal.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "empirical_finding": "graph stratum contains 0 mechanical dup-silver pairs "
                "(sifter self-test: negative-confirmation strong, dup verdicts "
                "low-precision) — all-negative stratum; 46/54 needs positive "
                "injection pinned by pre-gate v3",
                "negative_split": NEG_SPLIT,
                "population_available": {k: len(v) for k, v in sorted(pop.items())},
                "variants": quotas,
                "note": "pool-level quota pinning belongs to pre-gate v3 (owner ratification); "
                "these are feasibility-checked proposals, not a sealed corpus",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ── step 8: sifter self-test + feature distributions ─────────────────
    v43_rows = corpus_rows(BASE / "v43-localbuild/train.jsonl")
    rng_st = random.Random(SEED)
    st_sample = rng_st.sample(v43_rows, 300)
    st_confusion: Counter = Counter()
    st_mono = st_cross = 0
    for row in st_sample:
        v = sift_pair(row["record"], row["candidate"])
        if v.get("crosslang"):
            st_cross += 1
            continue
        st_mono += 1
        lab = row.get("label")
        ok = (lab == "duplicate" and v["verdict"] == "dup-silver") or (
            lab == "not-duplicate" and v["verdict"].startswith("not-dup")
        )
        st_confusion[(lab, v["verdict"], ok)] += 1
    st_agree = sum(n for (lab_k, vd_k, ok_k), n in st_confusion.items() if ok_k)
    st_dup_verdicts = sum(
        n for (lab_k, vd_k, ok_k), n in st_confusion.items() if vd_k == "dup-silver"
    )
    st_dup_correct = sum(
        n
        for (lab_k, vd_k, ok_k), n in st_confusion.items()
        if vd_k == "dup-silver" and lab_k == "duplicate"
    )
    selftest = {
        "sample": 300,
        "seed": SEED,
        "surface": "v43-train labeled rows",
        "monolingual_rows": st_mono,
        "crosslang_rows_excluded": st_cross,
        "row_agreement": round(st_agree / max(st_mono, 1), 4),
        "dup_silver_precision": round(st_dup_correct / max(st_dup_verdicts, 1), 4),
        "confusion_top": [
            {"label": lab_k, "verdict": vd_k, "correct": ok_k, "rows": n}
            for (lab_k, vd_k, ok_k), n in sorted(
                st_confusion.items(), key=lambda x: -x[1]
            )[:8]
        ],
        "reading": "mechanical sifter is SILVER-GRADE for negative confirmation "
        "only; dup verdicts are low-precision and never mint positives "
        "in this pool",
    }

    def bucket(n: int) -> str:
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 10:
            return "4-10"
        return "11+"

    razor_dist: dict[str, Counter] = {}
    crosslang_share: Counter = Counter()
    for r in file_rows:
        s = r["sifter"]
        razor_dist.setdefault(r["stratum"], Counter())[bucket(s["razor"])] += 1
        crosslang_share[r["stratum"]] += 1 if s.get("crosslang") else 0
    coverage_extra = {
        "sifter_selftest": selftest,
        "razor_token_buckets_by_stratum": {
            s: dict(sorted(c.items())) for s, c in sorted(razor_dist.items())
        },
        "crosslang_share_by_stratum": dict(sorted(crosslang_share.items())),
        "control_file": {
            "population_total": len(control_rows),
            "file_cap": CONTROL_FILE_CAP,
            "file_kept": len(control_kept_ids),
            "selection": "deterministic round-robin over components, pair_id order",
        },
    }
    log(
        runlog,
        "sifter_selftest",
        agreement=selftest["row_agreement"],
        dup_precision=selftest["dup_silver_precision"],
    )

    # ── step 9: coverage.json (numbers only) ─────────────────────────────
    coverage = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "store_snapshot": {
            "memories": len(store["mem_rows"]),
            "edges": len(store["edges"]),
            "vectors": len(store["vectors"]),
        },
        "components": {
            "count": len(comp_sizes),
            "sizes_top10": sorted(comp_sizes.values(), reverse=True)[:10],
        },
        "hygiene": counters.to_dict(),
        "scanner_provenance": scanner.provenance,
        "band": [BAND_LO, BAND_HI],
        "mint_threshold": MINT_THRESHOLD,
        "mint_top_k": TOP_K_MINT,
        "populations_pre_dedup": dict(sorted(strata_counter.items())),
        "mint_topk": {"before": mint_before, "after": len(topk_pair_ids)},
        "dedup_dropped": dict(sorted(dedup.items())),
        "disjointness": {
            "pair_level_violations": hard,
            "side_join_report": side_join_report,
        },
        "pool_final": {
            "pairs_in_file": len(file_rows),
            "pairs_population_full": len(surviving),
            "by_stratum": dict(sorted(strata_final.items())),
            "by_sifter": {f"{s}|{v}": n for (s, v), n in sorted(sift_final.items())},
            "components_covered": len(comp_dist),
            "projects_top15": dict(proj_counter.most_common(15)),
        },
        "double_check_sample": {
            "size": len(sample_rows),
            "seed": SEED,
            "verdicts": dict(sorted(sample_verdicts.items())),
        },
        **coverage_extra,
        "pool_fingerprint": pool_fp,
        "pool_fingerprint_scheme": "manifest(pair_id pair_sha256 sorted) → blake2b-256",
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run-log.jsonl").write_text(
        "".join(
            json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in runlog
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pool_fingerprint": pool_fp,
                "pairs": len(file_rows),
                "strata": dict(sorted(strata_final.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="b2_graph_pool")
    parser.add_argument("--out", default=str(WT_ROOT / "data" / "b2-graph-corpus"))
    args = parser.parse_args()
    return build(Path(args.out).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
