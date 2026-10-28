"""B-leg stage 1: build train/val surfaces + eval funnel from the frozen
Phase-1 corpus (a-retrial D0 snapshot).

Ratified mechanics (run-log 2026-10-29 opening entry):
- train positives = TR-POS 919 (sidecar G' features, edge_ab=1);
- negatives from TR-NEG universe: FR-N-hard 3x (2757), FR-N-far 1x (919),
  FR-X=0, FR-Q=0, knockon/masked/val excluded;
- val negatives (selection surface, TR-VAL 56): hard 4x56 + far 56,
  pair-keys disjoint from train negatives (sampled first);
- funnel per masked query: cosine top-K=100 over ALL 4942 snapshot vectors
  (K >= 2*limit=50) + injection of the query's masked positives;
- g_spec pair list: all extractable masked positives + 3 FR-N-hard partners
  per masked query sampled from its own funnel pool (seed 20261104).

Outputs (gitignored data/a-retrial/b-leg/): train_rows.jsonl, val_rows.jsonl,
funnel.json, gspec_pairs.json, build_report.json (numbers only).

Interpreter: vesma-cortex/.venv; PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cortex.features.graph import GraphEvidence  # noqa: E402
from a_retrial_b_leg_common import (  # noqa: E402
    BLEG,
    CLASSES,
    EMBED_PIN,
    HYG,
    MASK,
    ROLES,
    ABLATED_FEATURES,
    FUSION_FEATURES,
    assert_fingerprints,
    band_of,
    build_gprime,
    cos64,
    feature_row,
    load_edges_direction,
    load_manifest,
    load_published_meta,
    load_sidecar,
    load_snapshot_vectors,
    pair_graph_features,
    pair_key,
)

SEED_TRAIN_NEG = 20261101
SEED_VAL_NEG = 20261102
SEED_GSPEC = 20261104

N_TRAIN_POS = 919
QUOTA_HARD = 3  # x positives -> 2757, within ratified corridor <=3-4x
QUOTA_FAR = 1  # x positives -> 919, within ratified corridor 1-2x
N_VAL_POS = 56
QUOTA_VAL_HARD = 4
QUOTA_VAL_FAR = 1

K_POOL = 100
ENGINE_LIMIT = 50
N_GSPEC_HARD_PER_QUERY = 3


def main() -> int:
    BLEG.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "b-leg-surfaces", "date": "2026-10-29"}

    # ── frozen inputs ──────────────────────────────────────────────────
    man = load_manifest()
    fps = assert_fingerprints(man)
    report["fps_verified"] = sorted(fps.keys())
    print(f"[fp] {len(fps)} frozen inputs sha-verified against manifest")

    snap_ids, snap_vec, pin_ok = load_snapshot_vectors()
    assert len(snap_ids) == 4942 and pin_ok == 4942, (len(snap_ids), pin_ok)
    snap_idx = {rid: i for i, rid in enumerate(snap_ids)}
    print(f"[pin] snapshot pin assert {pin_ok}/{len(snap_ids)} = {EMBED_PIN[:22]}…")
    report["pin_assert"] = {"ok": pin_ok, "total": len(snap_ids)}

    meta = load_published_meta()
    assert len(meta) == 4942
    hyg_ids: list[str] = json.loads((HYG / "ids.json").read_text())
    hyg_vec = np.load(HYG / "vectors.npy")
    assert hyg_vec.shape == (3442, 384)
    # hygiene vectors must match the snapshot vectors of the same records
    # (same D0 content; observed float round-trip delta 7.45e-08 — snapshot
    # is the canonical frozen surface, ALL B-leg math reads the snapshot)
    probe = np.arange(0, 3442, 17)
    dmax = 0.0
    for i in probe:
        dmax = max(dmax, float(np.abs(hyg_vec[i] - snap_vec[snap_idx[hyg_ids[i]]]).max()))
    assert dmax <= 1e-6, f"hygiene vs snapshot vector divergence {dmax}"
    print(f"[vec] hygiene vectors == snapshot up to float round-trip "
          f"(max abs delta {dmax:.2e}, {len(probe)} probes); canonical = snapshot")

    sidecar = load_sidecar()
    assert len(sidecar) == 1648
    adj = build_gprime(snap_ids, snap_idx, sidecar)
    n_gprime = sum(1 for ev in sidecar.values() if ev.edge_ab)
    assert n_gprime == 975, n_gprime
    print(f"[G'] feature graph: {n_gprime} edges, adjacency over {len(adj)} records")
    report["gprime_edges"] = n_gprime

    edges_dir = load_edges_direction()
    assert len(edges_dir) == 1648

    train_rows_in = [json.loads(l) for l in open(ROLES / "train_pairs.jsonl", encoding="utf-8") if l.strip()]
    val_rows_in = [json.loads(l) for l in open(ROLES / "val_pairs.jsonl", encoding="utf-8") if l.strip()]
    assert len(train_rows_in) == N_TRAIN_POS and len(val_rows_in) == N_VAL_POS

    masked_keys: list[str] = json.loads((MASK / "masked_pairs.json").read_text())
    assert len(masked_keys) == 330
    knockon_keys: list[str] = json.loads((MASK / "knockon_pairs.json").read_text())
    assert len(knockon_keys) == 343
    masked_set, knockon_set = set(masked_keys), set(knockon_keys)
    train_keys = {r["pair_id"] for r in train_rows_in}
    val_keys = {r["pair_id"] for r in val_rows_in}

    # ── per-pair recompute asserts (§7.3-2 re-check at B-leg build) ────
    # (a) masked pairs: all graph features zero on G'
    for key in masked_keys:
        a, b = key.split("--")
        cn, nj, ctcm = pair_graph_features(a, b, adj, snap_idx, snap_vec, snap_ids)
        assert (cn, nj, ctcm) == (0, 0.0, 0.0), f"masked {key} nonzero on G'"
        assert not (adj.get(a, set()) & {b}), f"masked {key} present as G' edge"
    print("[assert] masked 330/330 graph features all-zero on G' (cut-closure recompute)")

    # (b) train/val positives: recompute == sidecar (cn/nj exact, ctcm atol)
    ctcm_max_delta = 0.0
    cos_max_delta = 0.0
    for rows_in in (train_rows_in, val_rows_in):
        for r in rows_in:
            key = r["pair_id"]
            a, b = r["a"], r["b"]
            assert pair_key(a, b) == key
            ev = sidecar.get(key)
            assert ev is not None and ev.edge_ab, f"{key} not a G' edge in sidecar"
            cn, nj, ctcm = pair_graph_features(a, b, adj, snap_idx, snap_vec, snap_ids)
            # sidecar stores nj rounded to 6 decimals — tolerance is storage precision
            assert cn == ev.common_neighbors and abs(nj - ev.neighbor_jaccard) <= 1e-6, key
            if ev.cos_to_common_max is not None:
                ctcm_max_delta = max(ctcm_max_delta, abs(ctcm - ev.cos_to_common_max))
            c64 = cos64(snap_vec[snap_idx[a]], snap_vec[snap_idx[b]])
            cos_max_delta = max(cos_max_delta, abs(c64 - float(r["cos"])))
    assert ctcm_max_delta <= 1e-6, ctcm_max_delta
    assert cos_max_delta <= 1e-6, cos_max_delta
    print(f"[assert] train/val positives recompute==sidecar: ctcm_max_delta={ctcm_max_delta:.2e}, "
          f"cos64-vs-stored_max_delta={cos_max_delta:.2e}")

    # ── negatives ──────────────────────────────────────────────────────
    uni = np.load(ROLES / "neg_universe.npz")
    ui, uj, uc = uni["i"], uni["j"], uni["cos"]
    assert ui.max() < 3442 and uj.max() < 3442
    ukeys = [pair_key(hyg_ids[i], hyg_ids[j]) for i, j in zip(ui.tolist(), uj.tolist())]
    assert len(set(ukeys)) == len(ukeys), "universe pair keys not unique"
    # spot recompute of stored universe cosines (float64 vs stored float32)
    rng = np.random.default_rng(SEED_TRAIN_NEG)
    probe = rng.choice(len(ukeys), size=1000, replace=False)
    dmax = 0.0
    for p in probe.tolist():
        i, j = int(ui[p]), int(uj[p])
        dmax = max(dmax, abs(cos64(snap_vec[snap_idx[hyg_ids[i]]], snap_vec[snap_idx[hyg_ids[j]]]) - float(uc[p])))
    assert dmax <= 2e-6, dmax
    print(f"[assert] universe cos recompute max delta {dmax:.2e} (1000 probes)")

    excluded = masked_set | knockon_set | train_keys | val_keys
    hard_pool_idx = [p for p, k in enumerate(ukeys) if k not in excluded]
    assert len(hard_pool_idx) >= QUOTA_VAL_HARD * N_VAL_POS + QUOTA_HARD * N_TRAIN_POS

    rng_val = np.random.default_rng(SEED_VAL_NEG)
    val_hard_sel = rng_val.permutation(sorted(hard_pool_idx))[: QUOTA_VAL_HARD * N_VAL_POS]
    val_hard_keys = {ukeys[p] for p in val_hard_sel.tolist()}
    rng_tr = np.random.default_rng(SEED_TRAIN_NEG)
    train_pool = sorted(set(hard_pool_idx) - set(val_hard_sel.tolist()))
    train_hard_sel = rng_tr.permutation(np.array(train_pool))[: QUOTA_HARD * N_TRAIN_POS]
    print(f"[neg] hard sampled: train {len(train_hard_sel)} (quota 3x), val {len(val_hard_sel)} (quota 4x56)")

    # far negatives: below-zone (cos<0.92) hygiene pairs, cn(G')=0, excluded keys
    def sample_far(n: int, seed: int, taken: set[str]) -> list[tuple[str, str]]:
        rng_f = np.random.default_rng(seed)
        out: list[tuple[str, str]] = []
        seen: set[str] = set(taken)
        guard = 0
        while len(out) < n:
            guard += 1
            assert guard < 2_000_000, "far sampler guard"
            i = int(rng_f.integers(0, 3442))
            j = int(rng_f.integers(0, 3442))
            if i == j:
                continue
            a, b = hyg_ids[i], hyg_ids[j]
            key = pair_key(a, b)
            if key in seen or key in excluded:
                continue
            c64 = cos64(snap_vec[snap_idx[a]], snap_vec[snap_idx[b]])
            if c64 >= 0.92:
                continue
            cn, _nj, _ctcm = pair_graph_features(a, b, adj, snap_idx, snap_vec, snap_ids)
            if cn != 0:
                continue  # FR-Q quarantine
            seen.add(key)
            out.append((a, b))
        return out

    val_far = sample_far(QUOTA_VAL_FAR * N_VAL_POS, SEED_VAL_NEG, set())
    train_far = sample_far(QUOTA_FAR * N_TRAIN_POS, SEED_TRAIN_NEG, {pair_key(a, b) for a, b in val_far})
    print(f"[neg] far sampled: train {len(train_far)} (quota 1x), val {len(val_far)} (1x56)")

    # ── assemble feature rows ──────────────────────────────────────────
    def rows_from_universe(sel: np.ndarray) -> list[dict]:
        out = []
        for p in sel.tolist():
            i, j, c = int(ui[p]), int(uj[p]), float(uc[p])
            a, b = hyg_ids[i], hyg_ids[j]
            ev = sidecar.get(pair_key(a, b))
            assert ev is None or not ev.edge_ab  # negatives carry no G' edge
            zero_ev = GraphEvidence()
            c = cos64(snap_vec[snap_idx[a]], snap_vec[snap_idx[b]])
            out.append({
                "pair_id": pair_key(a, b), "q": a, "c": b, "label": 0,
                "stratum": "FR-N-hard", "cos": c,
                "feats": dict(zip(FUSION_FEATURES,
                                  feature_row(c, zero_ev, meta[a], meta[b]))),
            })
        return out

    def rows_from_pairs(pairs: list[tuple[str, str]], stratum: str) -> list[dict]:
        out = []
        for a, b in pairs:
            i, j = snap_idx[a], snap_idx[b]
            c64 = cos64(snap_vec[i], snap_vec[j])
            zero_ev = GraphEvidence()
            out.append({
                "pair_id": pair_key(a, b), "q": a, "c": b, "label": 0,
                "stratum": stratum, "cos": c64,
                "feats": dict(zip(FUSION_FEATURES,
                                  feature_row(c64, zero_ev, meta[a], meta[b]))),
            })
        return out

    def positive_rows(rows_in: list[dict]) -> list[dict]:
        out = []
        for r in rows_in:
            key, a, b = r["pair_id"], r["a"], r["b"]
            ev = sidecar[key]
            i, j = snap_idx[a], snap_idx[b]
            c64 = cos64(snap_vec[i], snap_vec[j])
            feats = feature_row(c64, ev, meta[a], meta[b])
            assert feats[1] == 1.0  # edge_ab=1 + label (run-log semantics entry)
            out.append({
                "pair_id": key, "q": a, "c": b, "label": 1, "stratum": "TR-POS",
                "cos": c64, "feats": dict(zip(FUSION_FEATURES, feats)),
            })
        return out

    train_rows = (positive_rows(train_rows_in)
                  + rows_from_universe(train_hard_sel)
                  + rows_from_pairs(train_far, "FR-N-far"))
    val_rows = (positive_rows(val_rows_in)
                + rows_from_universe(val_hard_sel)
                + rows_from_pairs(val_far, "FR-N-far"))
    assert len(train_rows) == N_TRAIN_POS + QUOTA_HARD * N_TRAIN_POS + QUOTA_FAR * N_TRAIN_POS
    assert len(val_rows) == N_VAL_POS + QUOTA_VAL_HARD * N_VAL_POS + QUOTA_VAL_FAR * N_VAL_POS
    tr_keys_all = {r["pair_id"] for r in train_rows}
    vl_keys_all = {r["pair_id"] for r in val_rows}
    assert not (tr_keys_all & masked_set), "train x masked overlap"
    assert not (vl_keys_all & masked_set), "val x masked overlap"
    assert not (tr_keys_all & vl_keys_all), "train x val overlap"
    assert not (tr_keys_all & knockon_set), "train x knockon overlap"
    assert not (vl_keys_all & knockon_set), "val x knockon overlap"
    print(f"[rows] train {len(train_rows)} (pos {N_TRAIN_POS}), val {len(val_rows)} (pos {N_VAL_POS})")

    band_hist = Counter(band_of(r["cos"]) for r in train_rows if r["label"] == 1)
    report["train_pos_band_hist"] = dict(band_hist)

    with open(BLEG / "train_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r) + "\n")
    with open(BLEG / "val_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in val_rows:
            fh.write(json.dumps(r) + "\n")

    # ── eval funnel (§6.3): top-K + injection ──────────────────────────
    vec64 = snap_vec.astype(np.float64)
    norms = np.linalg.norm(vec64, axis=1)
    masked_by_pair = []
    for key in masked_keys:
        a, b = edges_dir[key]  # (query=from, positive=to)
        assert a in snap_idx and b in snap_idx, f"masked endpoint w/o vector: {key}"
        masked_by_pair.append({"pair_id": key, "q": a, "pos": b})
    n_queries = len({p["q"] for p in masked_by_pair})
    print(f"[funnel] masked pairs {len(masked_by_pair)} -> queries {n_queries}")

    pools: dict[str, list[str]] = {}
    pos_by_query: dict[str, list[str]] = {}
    for p in masked_by_pair:
        pos_by_query.setdefault(p["q"], []).append(p["pos"])
    assert K_POOL >= 2 * ENGINE_LIMIT, "K >= 2xlimit (MLC 28.10)"
    for q in sorted(pos_by_query):
        qi = snap_idx[q]
        sims = (vec64 @ vec64[qi]) / (norms[qi] * norms)
        sims[qi] = -2.0
        top = np.argsort(-sims)[:K_POOL]
        pool = [snap_ids[t] for t in top.tolist()]
        for pos in pos_by_query[q]:
            if pos not in pool:
                pool.append(pos)
        pools[q] = pool
    extraction = {
        "pairs_total": len(masked_by_pair),
        "pairs_extractable": sum(1 for p in masked_by_pair
                                 if p["q"] in snap_idx and p["pos"] in snap_idx),
        "stop_minimum": 215,
    }
    assert extraction["pairs_extractable"] >= extraction["stop_minimum"]
    print(f"[funnel] extraction filter: {extraction['pairs_extractable']}/{extraction['pairs_total']} "
          f">= STOP {extraction['stop_minimum']} (range П6 215–330)")
    pool_sizes = {q: len(p) for q, p in pools.items()}
    report["funnel"] = {
        "K": K_POOL, "engine_limit_declared": ENGINE_LIMIT,
        "assert_K_ge_2x_limit": K_POOL >= 2 * ENGINE_LIMIT,
        "queries": n_queries, "pairs": len(masked_by_pair),
        "pool_size_min": min(pool_sizes.values()), "pool_size_max": max(pool_sizes.values()),
        "candidate_universe": len(snap_ids),
        "extraction": extraction,
    }
    with open(BLEG / "funnel.json", "w", encoding="utf-8") as fh:
        json.dump({"pairs": masked_by_pair, "pools": pools}, fh)

    # ── g_spec pair list (declared form; hard partners from own pool) ──
    rng_g = np.random.default_rng(SEED_GSPEC)
    gspec_hard = []
    for q in sorted(pos_by_query):
        cands = []
        for x in pools[q]:
            if x in pos_by_query[q]:
                continue
            key = pair_key(q, x)
            if key in knockon_set or key in masked_set:
                continue
            cn, _nj, _ct = pair_graph_features(q, x, adj, snap_idx, snap_vec, snap_ids)
            c64 = cos64(snap_vec[snap_idx[q]], snap_vec[snap_idx[x]])
            if cn == 0 and 0.92 <= c64 < 0.97:
                cands.append((c64, x))
        cands.sort()
        pick = rng_g.choice(len(cands), size=min(N_GSPEC_HARD_PER_QUERY, len(cands)),
                            replace=False)
        for pi in sorted(pick.tolist()):
            gspec_hard.append({"q": q, "c": cands[pi][1], "cos": cands[pi][0]})
    gspec_pos = [{"pair_id": p["pair_id"], "q": p["q"], "c": p["pos"]} for p in masked_by_pair]
    base_rate = len(gspec_pos) / (len(gspec_pos) + len(gspec_hard))
    print(f"[gspec] positives {len(gspec_pos)} + hard {len(gspec_hard)} (base rate {base_rate:.4f})")
    report["gspec"] = {"positives": len(gspec_pos), "hard": len(gspec_hard),
                       "base_rate": base_rate, "hard_per_query": N_GSPEC_HARD_PER_QUERY}
    with open(BLEG / "gspec_pairs.json", "w", encoding="utf-8") as fh:
        json.dump({"positives": gspec_pos, "hard": gspec_hard}, fh)

    report["seeds"] = {"train_neg": SEED_TRAIN_NEG, "val_neg": SEED_VAL_NEG,
                       "gspec": SEED_GSPEC, "bootstrap": 20261101, "B": 10000}
    report["features"] = {"fusion": list(FUSION_FEATURES), "ablated": list(ABLATED_FEATURES)}
    (BLEG / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[done] surfaces written to", BLEG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
