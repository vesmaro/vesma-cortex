#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 4: cut-closure mask (§6.2, MLC 28.10).

Frozen order (В3): hygiene → classes → MASK → freeze α/β → train on remainder.

  seed-mask  ρ=20% of post-hygiene edges (rng seed declared in manifest,
             sorted pair_id population — snapshot-relative, deterministic)
  closure    knock-on quarantine to a FIXED POINT: every unmasked edge that
             participates in a 2-path u—w—v of a masked pair (u,v) is
             quarantined; repeat until no masked pair has common neighbors
  G′         feature graph = G − (masked ∪ knock-on)

Per-pair assert (§7.3 assert 2): the graph features of EVERY masked pair —
recomputed on G′ under the frozen contract semantics of
cortex.features.graph (edge_ab / edge_kinds / common_neighbors /
neighbor_jaccard / cos_to_common_max; rows validated through
GraphEvidence.from_row) — must equal the sidecar AND be exactly zero.

Closure cost histogram (per-step quarantine counts) → manifest.

Outputs (gitignored): mask/masked_pairs.json, mask/knockon_pairs.json,
mask/mask.json (seed/rho/histogram/fixed point), mask/sidecar_G.jsonl
(graph features on G′ for positives + masked + knock-on pairs), mask/mask_report.json.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "src"))
from cortex.features.graph import GraphEvidence  # noqa: E402  (frozen contract)

HYG = WORKTREE / "data" / "a-retrial" / "hygiene"
CLS = WORKTREE / "data" / "a-retrial" / "classes"
OUT = WORKTREE / "data" / "a-retrial" / "mask"

MASK_SEED = 20261028  # declared at build, fixed in the manifest
RHO = 0.20


def pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a < b else (b, a)
    return f"{x}--{y}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    index = {rid: i for i, rid in enumerate(ids)}
    mat = np.load(HYG / "vectors.npy")

    edges = [json.loads(line) for line in (CLS / "edges_post_hygiene.jsonl").read_text().splitlines()]
    edge_keys = sorted(e["pair_id"] for e in edges)
    edge_by_key = {e["pair_id"]: e for e in edges}
    n_edges = len(edge_keys)
    k = int(round(RHO * n_edges))
    rng = random.Random(MASK_SEED)
    masked_keys = sorted(rng.sample(edge_keys, k))
    masked_set = set(masked_keys)

    def build_adjacency(active: set[str]) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for key in active:
            e = edge_by_key[key]
            adj[e["a"]].add(e["b"])
            adj[e["b"]].add(e["a"])
        return adj

    masked_pairs = [tuple(key.split("--")) if "--" in key else None for key in masked_keys]
    masked_pairs = []
    for key in masked_keys:
        a, b = key.rsplit("--", 1)
        masked_pairs.append((a, b))

    # closure to fixed point
    active = set(edge_keys) - masked_set
    knockon_steps: list[dict] = []
    knockon_keys: set[str] = set()
    step = 0
    while True:
        adj = build_adjacency(active)
        bad: set[str] = set()
        for a, b in masked_pairs:
            common = adj[a] & adj[b]
            for w in common:
                bad.add(pair_key(a, w))
                bad.add(pair_key(w, b))
        bad &= active  # only active (unmasked, un-quarantined) edges can be knocked on
        if not bad:
            break
        step += 1
        active -= bad
        knockon_keys |= bad
        knockon_steps.append({"step": step, "quarantined": len(bad)})
        if step > 50:
            raise AssertionError("closure did not converge in 50 steps")

    adj_g1 = build_adjacency(active)

    # graph features on G' under the frozen contract semantics
    def features(a: str, b: str) -> dict:
        edge = None
        key = pair_key(a, b)
        if key in active:
            edge = edge_by_key[key]
        kinds = [edge["kind"]] if edge else []
        common = adj_g1[a] & adj_g1[b]
        cn = len(common)
        union = len(adj_g1[a] | adj_g1[b])
        nj = (cn / union) if union else 0.0
        cmax = 0.0
        if cn:
            ia, ib = index[a], index[b]
            for w in common:
                iw = index[w]
                cmax = max(cmax, float(mat[ia] @ mat[iw]), float(mat[ib] @ mat[iw]))
        return {
            "pair_id": key,
            "edge_ab": bool(edge),
            "edge_kinds": kinds,
            "common_neighbors": cn,
            "neighbor_jaccard": round(nj, 6),
            "cos_to_common_max": round(cmax, 6),
        }

    # sidecar over positives + masked + knock-on pairs; contract validation
    sidecar_pairs = sorted(active | masked_set | knockon_keys)
    rows = []
    for key in sidecar_pairs:
        a, b = key.rsplit("--", 1)
        row = features(a, b)
        GraphEvidence.from_row(row)  # frozen-contract validation, fail-loud
        rows.append(row)
    with (OUT / "sidecar_G.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # PER-PAIR ASSERT: masked pairs must be zero in G' and match the sidecar
    side_index = {row["pair_id"]: row for row in rows}
    recomputed_zero = 0
    for a, b in masked_pairs:
        key = pair_key(a, b)
        row = features(a, b)
        side = side_index[key]
        for field in ("edge_ab", "edge_kinds", "common_neighbors", "neighbor_jaccard", "cos_to_common_max"):
            assert row[field] == side[field], f"sidecar mismatch for masked {key}: {field}"
        assert row["edge_ab"] is False and row["edge_kinds"] == [], f"masked {key} has an edge in G'"
        assert row["common_neighbors"] == 0, f"masked {key} cn={row['common_neighbors']} in G'"
        assert row["neighbor_jaccard"] == 0.0
        assert row["cos_to_common_max"] == 0.0
        recomputed_zero += 1
    assert recomputed_zero == len(masked_pairs)

    # leakage sanity: knock-on pairs must not appear as train positives; masked∪knockon disjoint
    assert not (masked_set & knockon_keys)
    assert masked_set | knockon_keys <= set(edge_keys)

    hist_bins = Counter()
    for a, b in masked_pairs:
        key = pair_key(a, b)
        e = edge_by_key[key]
        c = e["cos"]
        band = "<0.92" if c < 0.92 else "[0.92,0.94)" if c < 0.94 else "[0.94,0.96)" if c < 0.96 else "[0.96,0.97)" if c < 0.97 else "[0.97,1.0]"
        hist_bins[band] += 1

    masked_ids_text = {
        "masked": sorted(masked_set),
        "knockon": sorted(knockon_keys),
    }
    (OUT / "masked_pairs.json").write_text(json.dumps(masked_ids_text["masked"], indent=0) + "\n")
    (OUT / "knockon_pairs.json").write_text(json.dumps(masked_ids_text["knockon"], indent=0) + "\n")

    survivors = sorted(active)
    summary = {
        "mask_seed": MASK_SEED,
        "rho": RHO,
        "edges_post_hygiene": n_edges,
        "masked_count": len(masked_set),
        "knockon_count": len(knockon_keys),
        "closure_steps": knockon_steps,
        "fixed_point_reached": True,
        "train_positive_pool": len(survivors),
        "masked_band_hist": dict(hist_bins),
        "asserts": {
            "per_pair_recompute_equals_sidecar": True,
            "per_pair_masked_zero_features": True,
            "masked_count": recomputed_zero,
            "masked_knockon_disjoint": True,
            "sidecar_rows_contract_validated": len(rows),
        },
    }
    (OUT / "mask.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
