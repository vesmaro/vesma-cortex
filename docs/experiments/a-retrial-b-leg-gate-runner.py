"""B-leg stage 5: SINGLE-SHOT gate runner (one run; re-rolls forbidden, П8).

Prolog: swap_check -> env pins -> frozen-input fps -> artifact sha re-verify
-> frozen tolerances (floors.json) -> gate_start. Gates per framing §6.3 with
the pre-declared K7 frame outcome (run-log 2026-10-29 02:35Z): g_mrr and
g_nharm tolerances are K7-vacuum (2x floors 0.378 / 0.371 > 0.20 declared
vacuum threshold) — their numbers are published but they do not gate this
run; the question goes to TL/MLC ratification. Runnable gates: g_re
(CI-leg), g_abl (>= max(0.01, 2*floor)), g_spec (>= 1 - tol_spec).

Also writes the owner spot-check (N=20 train-side predictions, stratified,
ids+scores+class only, no texts) and gates.json.

Interpreter: vesma-cortex/.venv; PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a_retrial_b_leg_common import (  # noqa: E402
    ABLATED_FEATURES,
    BLEG,
    FUSION_FEATURES,
    assert_fingerprints,
    load_manifest,
)
from a_retrial_b_leg_eval import EvalContext, bootstrap_ci_delta  # noqa: E402

ART = BLEG / "artifacts" / "fusion-ranker-b-v1"
SEED_BOOTSTRAP = 20261101
B_BOOTSTRAP = 10000
DELTA_RE = 0.02   # MLC 28.10 §10-в.3
DELTA_MRR = 0.01  # MLC 28.10 §10-в.3
DELTA_ABL = 0.01  # MLC 28.10 §10-в.3
K7_VACUUM_TOL = 0.20  # declared in run-log BEFORE the run
SEED_SPOTCHECK = 20261105
N_SPOTCHECK = 20

W = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    prolog: dict = {"gate_start_utc": W, "single_shot": True}

    # П1 swap_check
    swapon = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout.strip()
    free = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()[1]
    prolog["swap_check"] = {"swapon": swapon, "free_mem_line": free}
    print(f"[P1 swap_check] {swapon.splitlines()[0]} | {free}")

    # П2 env pins
    from importlib.metadata import version
    prolog["env_pins"] = {m: version(m) for m in
                          ("lightgbm", "onnxruntime", "onnxmltools", "scikit-learn", "numpy")}
    print(f"[P2 env] {prolog['env_pins']}")

    # П5 frozen inputs
    assert_fingerprints(load_manifest())
    prolog["frozen_inputs"] = "fp 11/11 re-verified"
    art_manifest = json.loads((ART / "manifest.json").read_text())
    onnx_sha = sha256_file(ART / "model.onnx")
    assert onnx_sha == art_manifest["model_onnx_sha256"], "artifact sha drift"
    prolog["artifact"] = {"model_onnx_sha256": onnx_sha,
                          "feature_set_sha256": art_manifest["feature_set_sha256"]}

    floors = json.loads((BLEG / "floors.json").read_text())
    tol = floors["tolerances"]
    tol_mrr = max(2 * floors["floors"]["delta_mrr10"], 0.0)
    tol_nharm = 2 * floors["floors"]["delta_mrr10_nharm"]
    tol_spec = tol["tol_spec"]
    thr_abl = max(DELTA_ABL, 2 * floors["floors"]["delta_abl_recall5"])
    prolog["tolerances"] = {"tol_mrr": tol_mrr, "tol_nharm": tol_nharm,
                            "tol_spec": tol_spec, "thr_abl": thr_abl,
                            "delta_re": DELTA_RE}
    k7_mrr = tol_mrr > K7_VACUUM_TOL
    k7_nharm = tol_nharm > K7_VACUUM_TOL
    assert k7_mrr and k7_nharm, "pre-declared K7 frame changed — do not re-roll"
    print(f"[tolerances] g_mrr tol {tol_mrr:.4f} (K7={k7_mrr}), "
          f"g_nharm tol {tol_nharm:.4f} (K7={k7_nharm}), "
          f"g_abl thr {thr_abl:.4f}, g_spec thr {1 - tol_spec:.4f}")

    # ── gate_start ─────────────────────────────────────────────────────
    print(f"[gate_start] {W} — single shot begins")

    import lightgbm as lgb
    import onnxruntime as ort
    ctx = EvalContext()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1  # MLC runtime posture
    sess = ort.InferenceSession(str(ART / "model.onnx"), sess_opts,
                                providers=["CPUExecutionProvider"])
    prob_out = [o for o in sess.get_outputs() if o.type == "tensor(float)"]
    assert len(prob_out) == 1
    in_name = sess.get_inputs()[0].name

    x32 = ctx.X_fusion.astype(np.float32)
    p_fusion = sess.run([prob_out[0].name], {in_name: x32})[0][:, 1].astype(np.float64)
    booster_a = lgb.Booster(model_file=str(ART / "booster_ablated.txt"))
    p_ablated = booster_a.predict(ctx.X_abl)

    m = ctx.query_metrics({"fusion": p_fusion, "ablated": p_ablated})
    rec_f, mrr_f = m["fusion"]
    rec_a, mrr_a = m["ablated"]
    rec_c, mrr_c = m["cosine"]
    n = len(ctx.queries)

    d_rec = rec_f - rec_c
    d_mrr = mrr_f - mrr_c
    d_abl = rec_f - rec_a
    nh = ctx.nharm_mask()
    d_nharm = mrr_f[nh] - mrr_c[nh]

    ci_rec = bootstrap_ci_delta(d_rec, B_BOOTSTRAP, SEED_BOOTSTRAP)
    ci_mrr = bootstrap_ci_delta(d_mrr, B_BOOTSTRAP, SEED_BOOTSTRAP)
    ci_abl = bootstrap_ci_delta(d_abl, B_BOOTSTRAP, SEED_BOOTSTRAP)

    wins = int((d_rec > 0).sum())
    losses = int((d_rec < 0).sum())

    gates: dict = {}

    # g_re (PRIMARY)
    thr_re = max(2 * ci_rec["halfwidth"], DELTA_RE)
    gates["g_re"] = {
        "delta_recall5": ci_rec["mean"], "ci95": [ci_rec["ci95_low"], ci_rec["ci95_high"]],
        "halfwidth": ci_rec["halfwidth"], "threshold": thr_re,
        "wins": wins, "losses": losses, "n_queries": n,
        "pass": bool(ci_rec["mean"] >= thr_re),
        "formula": "delta >= max(2xCI95halfwidth, delta_re=0.02)",
    }
    # g_mrr (K7-vacuum this run; numbers published)
    gates["g_mrr"] = {
        "delta_mrr10": ci_mrr["mean"], "ci95": [ci_mrr["ci95_low"], ci_mrr["ci95_high"]],
        "tol_mrr": tol_mrr, "delta_mrr ratified": DELTA_MRR,
        "k7_vacuum": True, "k7_note": "tol 2xfloor exceeds meaningful band (declared pre-run)",
        "would_pass_at_tol": bool(ci_mrr["mean"] >= tol_mrr),
    }
    # g_abl
    gates["g_abl"] = {
        "delta_recall5_fusion_minus_ablated": ci_abl["mean"],
        "ci95": [ci_abl["ci95_low"], ci_abl["ci95_high"]],
        "threshold": thr_abl, "delta_abl_ratified": DELTA_ABL,
        "pass": bool(ci_abl["mean"] >= thr_abl),
        "formula": "delta >= max(delta_abl=0.01, 2xfloor)",
    }
    # g_nharm (K7-vacuum this run; numbers published)
    gates["g_nharm"] = {
        "stratum_queries": int(nh.sum()),
        "delta_mrr10": float(d_nharm.mean()) if len(d_nharm) else 0.0,
        "tol_nharm": tol_nharm, "k7_vacuum": True,
        "would_pass_at_tol": bool(d_nharm.mean() >= -tol_nharm),
        "formula": "delta_mrr10 >= -tol_nharm",
    }
    # g_spec
    gspec_scores = sess.run([prob_out[0].name], {in_name: ctx.gspec_X.astype(np.float32)})[0][:, 1]
    order = np.argsort(-gspec_scores)
    p5 = float(ctx.gspec_y[order[:5]].sum()) / 5.0
    gs_cos = ctx.gspec_X[:, FUSION_FEATURES.index("cos_ab")]
    p5_cos = float(ctx.gspec_y[np.argsort(-gs_cos)[:5]].sum()) / 5.0
    gs_ab = booster_a.predict(ctx.gspec_X[:, [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]])
    p5_ab = float(ctx.gspec_y[np.argsort(-gs_ab)[:5]].sum()) / 5.0
    gates["g_spec"] = {
        "precision_at5_fusion": p5, "precision_at5_cosine_reference": p5_cos,
        "precision_at5_ablated_reference": p5_ab,
        "base_rate": ctx.gspec_base_rate, "threshold": 1 - tol_spec,
        "tol_spec": tol_spec, "pass": bool(p5 >= 1 - tol_spec),
        "pairs": {"positives": int(ctx.gspec_y.sum()), "total": int(len(ctx.gspec_y))},
    }

    # g_lat (protocol context from the export manifest; not a gate here)
    gates["g_lat_context"] = {"offline_latency": art_manifest["latency_offline"],
                              "g6a_class_ceiling_ms": 45}

    absolutes = {
        "cosine_recall5": float(rec_c.mean()), "cosine_mrr10": float(mrr_c.mean()),
        "fusion_recall5": float(rec_f.mean()), "fusion_mrr10": float(mrr_f.mean()),
        "ablated_recall5": float(rec_a.mean()), "ablated_mrr10": float(mrr_a.mean()),
        "competitor_visible_edge_share": ctx.competitor_edge_share(),
        "nharm_stratum": int(nh.sum()), "queries": n, "pairs": len(ctx.pairs),
    }

    runnable = ["g_re", "g_abl", "g_spec"]
    all_pass = all(gates[g]["pass"] for g in runnable)
    if all_pass and not (k7_mrr or k7_nharm):
        verdict = "ADOPT"
    elif all_pass:
        verdict = "ABORT-K7 (runnable gates green; frame legs vacuum — ratification required)"
    else:
        verdict = "REJECT"
    results = {
        "kind": "a-retrial-b-leg-gates",
        "gate_start_utc": W, "prolog": prolog,
        "single_shot": True, "re_rolls": "forbidden (ML-STD-0003 П8)",
        "absolutes": absolutes, "gates": gates,
        "k7_frame": {"legs": ["g_mrr", "g_nharm"],
                     "note": "vacuum tolerances from characterization floors; "
                             "numbers published; ratification question to TL/MLC"},
        "verdict": verdict,
        "verdict_detail": {
            "runnable_green": all_pass,
            "adopt_possible": False,
            "reason": "K7 frame legs (pre-declared) + numeric outcomes on runnable legs",
        },
    }
    (ART / "gates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("[absolutes]", json.dumps({k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in absolutes.items()}))
    for g in ("g_re", "g_mrr", "g_abl", "g_nharm", "g_spec"):
        print(f"[gate {g}]", json.dumps(
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in gates[g].items() if k not in ("formula",)}))
    print(f"[VERDICT] {verdict}")

    # ── owner spot-check (train-side, monitored, outside sealed surfaces) ──
    rows = [json.loads(l) for l in open(BLEG / "train_rows.jsonl", encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(SEED_SPOTCHECK)
    by_stratum = {}
    for r in rows:
        key = "TR-POS" if r["label"] == 1 else r["stratum"]
        by_stratum.setdefault(key, []).append(r)
    picks = []
    quota = {"TR-POS": 8, "FR-N-hard": 8, "FR-N-far": 4}
    for strat, k in quota.items():
        pool = sorted(by_stratum[strat], key=lambda r: r["pair_id"])
        for idx in rng.choice(len(pool), size=k, replace=False).tolist():
            picks.append(pool[idx])
    Xsp = np.array([[r["feats"][f] for f in FUSION_FEATURES] for r in picks])
    sp_fusion = sess.run([prob_out[0].name], {in_name: Xsp.astype(np.float32)})[0][:, 1]
    sp_ablated = booster_a.predict(
        Xsp[:, [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]])
    spot = [{"pair_id": r["pair_id"], "stratum": ("TR-POS" if r["label"] == 1 else r["stratum"]),
             "label": r["label"],
             "p_fusion_onnx": round(float(a), 8), "p_ablated": round(float(b), 8)}
            for r, a, b in zip(picks, sp_fusion, sp_ablated)]
    spot.sort(key=lambda r: r["pair_id"])
    (ART / "spotcheck_n20.json").write_text(json.dumps(spot, indent=1), encoding="utf-8")
    repo_copy = Path(__file__).resolve().parent / "a-retrial-b-leg-spotcheck.json"
    repo_copy.write_text(json.dumps(
        {"kind": "a-retrial-b-leg-spotcheck", "n": len(spot), "seed": SEED_SPOTCHECK,
         "note": "ids+scores+class only, no texts (owner spot-check, monitored)",
         "rows": spot}, indent=1), encoding="utf-8")
    print(f"[spot-check] {len(spot)} rows -> {repo_copy.name} + artifact copy "
          f"(sha {sha256_file(repo_copy)[:16]}…)")
    print("[done] single-shot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
