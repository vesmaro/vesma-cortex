#!/usr/bin/env python3
"""S1 similarity docompute for the synthetic corpus — engine environment.

Runs OUTSIDE src/cortex (imports the engine NanoProvider — bundle
mnema-embed-v1, CPU, zero network). Adds the measured `similarity` field to
synthetic pairs so they satisfy the train-manifest shape (record/candidate/
similarity, data-contract §3). Per A2s: the enriched corpus is a DERIVED
corpus with its own fingerprint (the 621cb23f handshake covers the
pre-similarity corpus).

Embedding text per prereg v2: title + body + tags, cut 4096. Run as:

    PYTHONPATH=<engine-src> <engine-venv>/bin/python scripts/s1_synth_similarity.py \
        --pairs data/synth/pairs.jsonl --out data/synth/pairs.sim.jsonl

JSON summary on stdout; progress on stderr. No cortex imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def embed_text(side: dict) -> str:
    tags = " ".join(side.get("tags") or [])
    return f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}"[:4096]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vesmaro.embeddings import NanoProvider

    provider = NanoProvider()
    print(f"embedder fingerprint: {provider.fingerprint}", file=sys.stderr)

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    cache: dict[str, list[float]] = {}

    def vec(side: dict) -> list[float]:
        text = embed_text(side)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in cache:
            cache[key] = provider.embed(text)
        return cache[key]

    out_rows = []
    for i, row in enumerate(rows, 1):
        a, b = vec(row["record"]), vec(row["candidate"])
        dot = sum(x * y for x, y in zip(a, b))
        row["similarity"] = round(max(0.0, min(1.0, dot)), 6)
        # candidate N consumes the 4x384 block — attach store-shape vectors
        row["vec_a"] = [round(x, 6) for x in a]
        row["vec_b"] = [round(x, 6) for x in b]
        out_rows.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")

    lines = sorted(
        f"{r['pair_id']} {hashlib.sha256(json.dumps({k: r[k] for k in ('record', 'candidate', 'similarity')}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}"
        for r in out_rows
    )
    manifest = ("\n".join(lines) + "\n").encode()
    fp = hashlib.blake2b(manifest, digest_size=32).hexdigest()
    sims = [r["similarity"] for r in out_rows]
    print(json.dumps({
        "rows": len(out_rows),
        "derived_corpus_fingerprint": fp,
        "similarity_min": min(sims),
        "similarity_mean": round(sum(sims) / len(sims), 4),
        "similarity_max": max(sims),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
