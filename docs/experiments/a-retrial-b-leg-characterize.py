"""B-leg stage 4: characterization — noise halves via stratified permutation
of edge labels (partner reshuffle WITHIN the positive's cos band, DE 28.10/В7),
retrain-inclusive draws (MLC 28.10), >= 3 draws.

Null mechanics: each TRAIN positive (q, c) gets its partner rewired to a
random record x with cos64(q, x) in the SAME band, subject to: x among the
3442 hygiene records, (q, x) not in {masked, knockon, train, val} pairs,
x not a G'-neighbor of q (the permuted positive is a no-edge pair, matching
the worst-case eval view). If no candidate exists in the band, the pair
stays original (counted). Negatives are unchanged. Each draw retrains
fusion (selected config) + ablated (selected config) and recomputes ALL
gate statistics on the frozen funnel/gspec — the floors are max |value|.

Interpreter: vesma-cortex/.venv; PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a_retrial_b_leg_common import (  # noqa: E402
    ABLATED_FEATURES,
    BLEG,
    FUSION_FEATURES,
    band_of,
    cos64,
    feature_row,
    load_manifest,
    assert_fingerprints,
    load_published_meta,
    load_sidecar,
    load_snapshot_vectors,
    pair_graph_features,
    pair_key,
)
from cortex.features.graph import GraphEvidence  # noqa: E402
from a_retrial_b_leg_eval import EvalContext, bootstrap_ci_delta  # noqa: E402

SEED_BASE = 20261103  # draws d=1..3 -> 20261104..20261106
DRAWS = 3
ART = BLEG / "artifacts" / "fusion-ranker-b-v1"


def load_selection() -> dict:
    return json.loads((BLEG / "selection.json").read_text())


def train_on(X, y, params, feature_names) -> lgb.Booster:
    base = dict(objective="binary", verbosity=-1, seed=20261101,
                deterministic=True, force_row_wise=True, num_threads=4,
                min_data_in_leaf=20, feature_fraction=1.0, bagging_fraction=1.0,
                bagging_freq=0, lambda_l2=1.0)
    ds = lgb.Dataset(X, label=y, feature_name=list(feature_names), free_raw_data=False)
    return lgb.train({**base, **params}, ds, num_boost_round=params["num_boost_round"])


def rewire_train_positives(ctx: EvalContext, train_rows: list[dict],
                           seed: int) -> tuple[list[dict], dict]:
    hyg_set = set(ctx.hyg_ids)
    masked = {p["pair_id"] for p in ctx.pairs}
    val_keys = {r["pair_id"] for r in (json.loads(l) for l in
                open(BLEG / "val_rows.jsonl", encoding="utf-8"))}
    train_keys = {r["pair_id"] for r in train_rows}
    knockon = set(json.loads((BLEG.parents[0] / "mask" / "knockon_pairs.json")
                             .read_text()))
    excluded = masked | val_keys | train_keys | knockon

    rng = np.random.default_rng(seed)
    kept = 0
    rows = []
    vec64 = ctx.snap_vec.astype(np.float64)
    norms = np.linalg.norm(vec64, axis=1)
    hyg_idx = [ctx.snap_idx[r] for r in ctx.hyg_ids]
    for r in train_rows:
        if r["label"] != 1:
            rows.append(r)
            continue
        q = r["q"]
        qi = ctx.snap_idx[q]
        sims = (vec64 @ vec64[qi]) / (norms[qi] * norms)
        band = band_of(float(r["cos"]))
        lo, hi = (0.0, 0.92) if band == "<0.92" else (
            (0.92, 0.94) if band == "[0.92,0.94)" else
            (0.94, 0.96) if band == "[0.94,0.96)" else
            (0.96, 0.97) if band == "[0.96,0.97)" else (0.97, 1.01))
        cands = []
        for hi_idx, rid in enumerate(ctx.hyg_ids):
            if rid == q:
                continue
            c = float(sims[hyg_idx[hi_idx]])
            if not (lo <= c < hi):
                continue
            key = pair_key(q, rid)
            if key in excluded:
                continue
            if rid in ctx.adj.get(q, set()):
                continue
            cands.append((rid, c, key))
        if not cands:
            kept += 1
            rows.append(r)
            continue
        pick = cands[int(rng.integers(0, len(cands)))]
        rid, c64, key = pick
        cn, nj, ctcm = pair_graph_features(q, rid, ctx.adj, ctx.snap_idx,
                                           ctx.snap_vec, ctx.snap_ids)
        ev = GraphEvidence(edge_ab=False, edge_kind_supersedes=False,
                           common_neighbors=cn, neighbor_jaccard=nj,
                           cos_to_common_max=ctcm if cn else 0.0)
        feats = feature_row(c64, ev, ctx.meta[q], ctx.meta[rid])
        rows.append({"pair_id": key, "q": q, "c": rid, "label": 1,
                     "stratum": "PERM-POS", "cos": c64,
                     "feats": dict(zip(FUSION_FEATURES, feats))})
    return rows, {"kept_original_no_band_candidate": kept,
                  "permuted": sum(1 for r in rows
                                  if r.get("stratum") == "PERM-POS")}


def evaluate(ctx: EvalContext, booster_f, booster_a) -> dict:
    pool_sc = {
        "fusion": booster_f.predict(ctx.X_fusion),
        "ablated": booster_a.predict(ctx.X_abl),
    }
    m = ctx.query_metrics(pool_sc)
    rec_f, mrr_f = m["fusion"]
    rec_a, mrr_a = m["ablated"]
    rec_c, mrr_c = m["cosine"]
    d_rec = rec_f - rec_c
    d_mrr = mrr_f - mrr_c
    d_abl = rec_f - rec_a
    nh = ctx.nharm_mask()
    d_nharm = (mrr_f[nh] - mrr_c[nh])
    gs_f = booster_f.predict(ctx.gspec_X)
    order = np.argsort(-gs_f)
    top5 = order[:5]
    p5 = float(ctx.gspec_y[top5].sum()) / 5.0
    gs_cos = ctx.gspec_X[:, FUSION_FEATURES.index("cos_ab")]
    top5c = np.argsort(-gs_cos)[:5]
    p5_cos = float(ctx.gspec_y[top5c].sum()) / 5.0
    return {
        "delta_recall5": float(d_rec.mean()),
        "delta_mrr10": float(d_mrr.mean()),
        "delta_abl_recall5": float(d_abl.mean()),
        "nharm_stratum": int(nh.sum()),
        "delta_mrr10_nharm": float(d_nharm.mean()) if len(d_nharm) else 0.0,
        "gspec_p5_fusion": p5,
        "gspec_p5_cosine": p5_cos,
        "base_rate": ctx.gspec_base_rate,
    }


def main() -> int:
    t0 = time.perf_counter()
    assert_fingerprints(load_manifest())
    sel = load_selection()
    cfg_f = sel["configs"]["fusion"][sel["configs"]["fusion"]["chosen"]]["params"]
    cfg_a = sel["configs"]["ablated"][sel["configs"]["ablated"]["chosen"]]["params"]

    ctx = EvalContext()
    train_rows = [json.loads(l) for l in open(BLEG / "train_rows.jsonl", encoding="utf-8") if l.strip()]
    y_tr = np.array([r["label"] for r in train_rows], dtype=np.int32)

    def rows_to_matrix(rows):
        X = np.array([[r["feats"][f] for f in FUSION_FEATURES] for r in rows])
        return X, np.array([r["label"] for r in rows], dtype=np.int32)

    X_all, _ = rows_to_matrix(train_rows)
    abl_idx = [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]

    print(f"[context] queries {len(ctx.queries)}, pool rows {len(ctx.pool_ids)}, "
          f"nharm stratum {int(ctx.nharm_mask().sum())}, "
          f"competitor edge share {ctx.competitor_edge_share():.4f}, "
          f"gspec base rate {ctx.gspec_base_rate:.4f}")

    results = {"draws": [], "seed_base": SEED_BASE, "draws_n": DRAWS}
    for d in range(1, DRAWS + 1):
        t1 = time.perf_counter()
        perm_rows, info = rewire_train_positives(ctx, train_rows, SEED_BASE + d)
        Xp, yp = rows_to_matrix(perm_rows)
        bf = train_on(Xp, yp, cfg_f, FUSION_FEATURES)
        ba = train_on(Xp[:, abl_idx], yp, cfg_a, ABLATED_FEATURES)
        stats = evaluate(ctx, bf, ba)
        stats["draw"] = d
        stats["seed"] = SEED_BASE + d
        stats["rewire"] = info
        results["draws"].append(stats)
        print(f"[draw {d}] {json.dumps({k: round(v, 6) if isinstance(v, float) else v
                                        for k, v in stats.items()
                                        if k not in ('rewire',)})} "
              f"({time.perf_counter() - t1:.1f}s)")

    def floor(key):
        return max(abs(r[key]) for r in results["draws"])

    results["floors"] = {
        "delta_recall5": floor("delta_recall5"),
        "delta_mrr10": floor("delta_mrr10"),
        "delta_abl_recall5": floor("delta_abl_recall5"),
        "delta_mrr10_nharm": floor("delta_mrr10_nharm"),
        "gspec_p5_dev_from_base": max(abs(r["gspec_p5_fusion"] - r["base_rate"])
                                     for r in results["draws"]),
    }
    results["tolerances"] = {
        "note": "ML-STD-0001: tol >= 2x max floor; g_spec tol = max(2x floor_spec, 0.20); "
                "K7 vacuum if tol_spec >= 0.50",
        "tol_mrr": max(2 * results["floors"]["delta_mrr10"], 0.0),
        "tol_abl_floor_component": 2 * results["floors"]["delta_abl_recall5"],
        "tol_nharm": 2 * results["floors"]["delta_mrr10_nharm"],
        "tol_spec": max(2 * results["floors"]["gspec_p5_dev_from_base"], 0.20),
    }
    results["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (BLEG / "floors.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[floors] {json.dumps(results['floors'])}")
    print(f"[tol] {json.dumps(results['tolerances'])}")
    print(f"[done] characterization in {results['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
