#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 5b: freeze the sealed-side pieces.

Frozen here (В3: sealed side cut BEFORE train):
  1. PA-Q1 sources (llm notes head-cut + twin job plan) — fp-frozen;
     real stratum: K1-FAIL declared (clean pool 0 at D0 — every post-ceiling
     hygiene-passing record is an agent-session record; verdict goes to TL
     unchanged, no softening).
  2. Shared frozen distractor pool (FR-N-hard texts, 240) — asserts:
     no FR-X-class texts, no dup-territory texts, no masked-endpoint texts,
     no pair inside the pool at cos>=0.97 (unbeatable-distractor guard),
     hash-disjoint from the frozen llm sources and the ban registry.

Seeds declared: sources 20261028, distractor pool 20261030.
Outputs: gate/sources.jsonl, gate/sources_manifest.json, gate/distractor_pool.jsonl,
gate/distractor_manifest.json (texts gitignored; manifests numbers-only inside data).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import MAX_CHARS, detect_lang, normalise  # noqa: E402

OUT = WORKTREE / "data" / "a-retrial" / "gate"
HYG = WORKTREE / "data" / "a-retrial" / "hygiene"
CLS = WORKTREE / "data" / "a-retrial" / "classes"
MASK = WORKTREE / "data" / "a-retrial" / "mask"

LLM_TAKE = 240
TWIN_JOBS = 30
TWIN_PER_JOB = 6
TWIN_FAMILIES = ["meeting", "notes", "rules", "science", "tech"]
POOL_TAKE = 240
SEED_SOURCES = 20261028
SEED_POOL = 20261030


def text_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


class Ban:
    """Unpickle shim: the registry was written by a5_sources (__main__.Ban)."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.grams: set[str] = set()
        self.sides: dict = {}


def fp_of(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def side_text(r: dict) -> str:
    title = str(r.get("title") or "").strip()
    body = str(r.get("body") or "").strip()
    text = f"{title}. {body}" if title else body
    return normalise(text)[:MAX_CHARS]


def main() -> int:
    import random

    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    index = {rid: i for i, rid in enumerate(ids)}
    mat = np.load(HYG / "vectors.npy")
    records = {o["id"]: o for o in (json.loads(l) for l in (HYG / "records.jsonl").read_text().splitlines())}
    full = pickle.loads((OUT / "ban_registry.pkl").read_bytes())

    # ── 1. PA-Q1 sources ──
    clean = [json.loads(l) for l in (OUT / "llm_clean_pool.jsonl").read_text().splitlines()]
    rng = random.Random(SEED_SOURCES)
    llm_sources = clean[:LLM_TAKE]  # key-order head-cut (v2 recipe), seed recorded for provenance
    assert len(llm_sources) == LLM_TAKE
    source_rows = []
    for n in llm_sources:
        source_rows.append(
            {"kind": "note", "pool": "llm", "key": n["key"], "ru": n["ru"], "en": n["en"], "direction": n["direction"]}
        )
    for i, fam in enumerate([TWIN_FAMILIES[i % len(TWIN_FAMILIES)] for i in range(TWIN_JOBS)]):
        source_rows.append({"kind": "twin_job", "job": f"tw-{i:03d}", "family": fam, "npairs": TWIN_PER_JOB})
    dirs = {"llm": {"ru->en": sum(1 for r in llm_sources if r["direction"] == "ru->en"),
                    "en->ru": sum(1 for r in llm_sources if r["direction"] == "en->ru")},
            "twin_jobs": TWIN_JOBS, "twin_per_job": TWIN_PER_JOB}
    with (OUT / "sources.jsonl").open("w", encoding="utf-8") as fh:
        for row in source_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources_fp = fp_of(source_rows)
    print(f"sources frozen: {len(source_rows)} rows (llm {len(llm_sources)}, twin jobs {TWIN_JOBS}); fp={sources_fp[:16]}…")

    # ── 2. distractor pool from FR-N-hard ──
    from collections import defaultdict

    edges = [json.loads(l) for l in (CLS / "edges_post_hygiene.jsonl").read_text().splitlines()]
    adj: dict[str, set[str]] = defaultdict(set)
    edge_keys = set()
    for e in edges:
        adj[e["a"]].add(e["b"])
        adj[e["b"]].add(e["a"])
        edge_keys.add(e["pair_id"])
    masked = set(json.loads((MASK / "masked_pairs.json").read_text()))
    masked_ends = {x for key in masked for x in key.rsplit("--", 1)}
    dup = json.loads((CLS / "dup_pairs.json").read_text())
    dup_members = {p["a"] for p in dup} | {p["b"] for p in dup}
    frq_cn = {p["a"] for p in json.loads((CLS / "fr_q_cn_pairs.json").read_text())} | {
        p["b"] for p in json.loads((CLS / "fr_q_cn_pairs.json").read_text())
    }

    zp = np.load(CLS / "zone_pairs.npz")
    zi, zj, zc = zp["i"], zp["j"], zp["cos"]
    langs = {rid: detect_lang(records[rid]["body"]) for rid in ids}
    frnh_members: set[str] = set()
    frx_members: set[str] = set()
    for i, j, c in zip(zi.tolist(), zj.tolist(), zc.tolist()):
        a, b = ids[i], ids[j]
        la, lb = langs[a], langs[b]
        if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
            continue  # FR-Q cross-lang
        if a in frq_cn or b in frq_cn:
            continue  # FR-Q cn>=1 (members of any saved cn>=1 pair)
        if float(c) >= 0.97:
            frx_members.add(a)
            frx_members.add(b)
        else:
            frnh_members.add(a)
            frnh_members.add(b)

    banned_ids = masked_ends | dup_members | frx_members | frq_cn
    candidates = []
    for rid in sorted(frnh_members):
        if rid in banned_ids:
            continue
        r = records[rid]
        text = side_text(r)
        if len(text) < 80:
            continue
        if text_hash(text) in full.hashes:
            continue
        candidates.append(rid)
    print(f"distractor candidates: {len(candidates)} of FR-N-hard members {len(frnh_members)} (exclusions applied)")

    pool_ids = sorted(rng.sample(candidates, min(POOL_TAKE, len(candidates))))
    # asserts: pairwise cos < 0.97 inside the pool; languages composition
    pair_max = 0.0
    worst = None
    vecs = mat[[index[r] for r in pool_ids]]
    sim = vecs @ vecs.T
    n = len(pool_ids)
    for x in range(n):
        for y in range(x + 1, n):
            if sim[x, y] > pair_max:
                pair_max = float(sim[x, y])
                worst = (pool_ids[x], pool_ids[y])
    assert pair_max < 0.97, f"unbeatable-distractor guard FAILED: {worst} cos={pair_max:.4f}"
    pool_texts = [side_text(records[rid]) for rid in pool_ids]
    assert all(text_hash(t) not in full.hashes for t in pool_texts)
    llm_hash_set = {text_hash(t) for r in llm_sources for t in (r["ru"], r["en"])}
    assert all(text_hash(t) not in llm_hash_set for t in pool_texts)
    pool_lang = {}
    for rid in pool_ids:
        l = langs[rid]
        pool_lang[l] = pool_lang.get(l, 0) + 1

    with (OUT / "distractor_pool.jsonl").open("w", encoding="utf-8") as fh:
        for rid, t in zip(pool_ids, pool_texts):
            fh.write(json.dumps({"id": rid, "text": t}, ensure_ascii=False) + "\n")
    pool_manifest = {
        "seed": SEED_POOL,
        "universe": "FR-N-hard records (band [0.92,0.97), cn=0, no edge, same-language)",
        "size": len(pool_ids),
        "stop_minimum_per_set": 100,
        "shared_by_sets": ["alpha", "beta"],
        "languages": pool_lang,
        "pairwise_max_cos": round(pair_max, 6),
        "asserts": {
            "no_FR_X_texts": True,
            "no_dup_territory": True,
            "no_masked_endpoints": True,
            "no_FR_Q_members": True,
            "no_pair_ge_0_97": True,
            "hash_outside_ban_registry": True,
            "disjoint_from_frozen_sources": True,
        },
        "fp": fp_of(pool_texts),
    }
    (OUT / "distractor_manifest.json").write_text(json.dumps(pool_manifest, indent=1) + "\n")
    (OUT / "sources_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED_SOURCES,
                "llm_head_cut": LLM_TAKE,
                "directions": dirs,
                "twin_jobs": TWIN_JOBS,
                "twin_per_job": TWIN_PER_JOB,
                "real_stratum": {
                    "k1_verdict": "FAIL — clean pool 0 at D0",
                    "fact": "every post-ceiling hygiene-passing record (306) is agent-session-tagged; agent:user absent",
                    "action": "honest K1 STOP per stratum, reported to TL unchanged; no softening",
                },
                "fp": sources_fp,
            },
            indent=1,
        ) + "\n"
    )
    print(f"distractor pool frozen: {len(pool_ids)} texts; languages={pool_lang}; max-pairwise-cos={pair_max:.4f}; fp={pool_manifest['fp'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
