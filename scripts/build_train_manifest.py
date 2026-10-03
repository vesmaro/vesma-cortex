#!/usr/bin/env python3
"""Assemble a train/holdout split from a dataset directory (B-wave pipeline).

Docomputes similarity + vectors (engine NanoProvider, CPU, zero network),
fingerprints the corpus (BLAKE2b over sorted sha256 manifest, data-contract
§3/§5) and splits stratified by label: 70% train / 30% holdout (deterministic
stride over pair_id sort). Holdout ids land in <out-dir>/holdout-ids.json —
sealed BEFORE any training run.

Usage (ENGINE venv — imports vesmaro.embeddings):
    PYTHONPATH=<engine-src> python3 scripts/build_train_manifest.py \
        --in-dir <dataset-dir> --out-dir data/stage2/<name> [--holdout-frac 0.3] [--seed 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--holdout-frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

sys.path.insert(0, "/var/home/abyss/LABs/Projects/Project-Vesma/wt/a2-engine-readonly/src")
from vesmaro.embeddings import NanoProvider  # noqa: E402

provider = NanoProvider()
print(f"embedder pin: {provider.fingerprint}", file=sys.stderr)
cache: dict[str, list[float]] = {}


def vec(side):
    tags = " ".join(side.get("tags") or [])
    text = f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}"[:4096]
    k = hashlib.sha256(text.encode()).hexdigest()
    if k not in cache:
        cache[k] = provider.embed(text)
    return cache[k]


rows = []
for l in open(f"{args.in_dir}/pairs-pos.jsonl", encoding="utf-8"):
    r = json.loads(l)
    rows.append({"pair_id": r["pair_id"], "label": "duplicate",
                 "record": r["original"],
                 "candidate": r.get("variant") or r.get("candidate"),
                 "stratum": r.get("stratum", "constructed")})
for l in open(f"{args.in_dir}/pairs-neg.jsonl", encoding="utf-8"):
    r = json.loads(l)
    rows.append({"pair_id": r["pair_id"], "label": "not-duplicate",
                 "record": r["a"], "candidate": r["b"],
                 "stratum": r.get("stratum", "constructed")})
for r in rows:
    va, vb = vec(r["record"]), vec(r["candidate"])
    dot = sum(x * y for x, y in zip(va, vb))
    r["similarity"] = round(min(1.0, max(0.0, dot)), 6)
    r["vec_a"] = [round(x, 6) for x in va]
    r["vec_b"] = [round(x, 6) for x in vb]
    print(f"  embedded {r['pair_id']}", file=sys.stderr)

rows.sort(key=lambda r: r["pair_id"])
lines = sorted(
    f"{r['pair_id']} {hashlib.sha256(json.dumps({k: r[k] for k in ('record', 'candidate', 'similarity')}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}"
    for r in rows)
fp = hashlib.blake2b(("\n".join(lines) + "\n").encode(), digest_size=32).hexdigest()

hold_ids = set()
for cls in ("duplicate", "not-duplicate"):
    ids = [r["pair_id"] for r in rows if r["label"] == cls]
    hold_count = max(1, int(round(len(ids) * args.holdout_frac)))
    step = len(ids) / hold_count
    hold_ids.update(ids[min(len(ids) - 1, int(i * step))] for i in range(hold_count))
train = [r for r in rows if r["pair_id"] not in hold_ids]
hold = [r for r in rows if r["pair_id"] in hold_ids]

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
for name, rs in (("train.jsonl", train), ("holdout.jsonl", hold)):
    with open(out / name, "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
labels_fp = hashlib.blake2b(
    json.dumps({r["pair_id"]: r["label"] for r in rows}, sort_keys=True).encode(),
    digest_size=32).hexdigest()
with open(out / "holdout-ids.json", "w", encoding="utf-8") as fh:
    json.dump({"holdout_ids": sorted(hold_ids), "corpus_fingerprint": fp,
               "label_fingerprint_input": labels_fp,
               "seed": args.seed, "split": f"stratified stride, frac={args.holdout_frac}"}, fh, indent=1)
print(json.dumps({"total": len(rows), "corpus_fingerprint": fp[:16] + "…",
                  "train": dict(Counter(r["label"] for r in train)),
                  "holdout": dict(Counter(r["label"] for r in hold))}))
