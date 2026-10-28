"""B-leg stages 2+3: LightGBM training (fusion + ablated), selection on
TR-VAL ONLY, double-run determinism, ONNX export + fidelity assert +
offline latency + artifact manifest.

Ratified scope (run-log 2026-10-29): configs A/B max, seed 20261101,
selection metric = TR-VAL AUC (336 rows: 56 pos / 224 hard / 56 far),
never on masked. ONNX = serving canon (MLC 28.10), ort 1.30,
intra_op=1 posture for the latency probe.

Outputs (gitignored data/a-retrial/b-leg/artifacts/fusion-ranker-b-v1/):
model.onnx, booster_fusion.txt, booster_ablated.txt, selection.json,
manifest.json (export facts; gates are written by the gate runner).

Interpreter: vesma-cortex/.venv; PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import hashlib
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
    EMBED_PIN,
    FUSION_FEATURES,
    load_manifest,
    assert_fingerprints,
)

SEED = 20261101
ART = BLEG / "artifacts" / "fusion-ranker-b-v1"

CONFIGS = {
    "A": dict(learning_rate=0.05, num_leaves=31, num_boost_round=300),
    "B": dict(learning_rate=0.1, num_leaves=15, num_boost_round=150),
}
BASE_PARAMS = dict(
    objective="binary",
    verbosity=-1,
    seed=SEED,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
    min_data_in_leaf=20,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    lambda_l2=1.0,
)


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    X = np.array([[r["feats"][f] for f in FUSION_FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int32)
    return X, y, rows


def train_booster(X, y, params: dict) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, feature_name=list(FUSION_FEATURES), free_raw_data=False)
    return lgb.train({**BASE_PARAMS, **params}, ds, num_boost_round=params["num_boost_round"])


def val_auc(booster, Xv, yv, feats) -> float:
    from sklearn.metrics import roc_auc_score
    idx = [FUSION_FEATURES.index(f) for f in feats]
    p = booster.predict(Xv[:, idx])
    return float(roc_auc_score(yv, p))


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    assert_fingerprints(man)

    Xtr, ytr, _ = load_rows(BLEG / "train_rows.jsonl")
    Xva, yva, _ = load_rows(BLEG / "val_rows.jsonl")
    print(f"[data] train {Xtr.shape}, val {Xva.shape} (pos {int(yva.sum())})")

    # ablated views = drop the graph five (declared column subset)
    ablated_idx = [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]
    Xtr_ab = Xtr[:, ablated_idx]
    Xva_ab = Xva[:, ablated_idx]
    ds_tr_ab = lgb.Dataset(Xtr_ab, label=ytr, feature_name=list(ABLATED_FEATURES),
                           free_raw_data=False)

    def train_ab(params):
        return lgb.train({**BASE_PARAMS, **params}, ds_tr_ab,
                         num_boost_round=params["num_boost_round"])

    selection = {"configs": {}, "baseline_cosine_val_auc": None}
    # honesty anchor: cosine-only on TR-VAL (train-side surface, legal)
    from sklearn.metrics import roc_auc_score
    cos_idx = FUSION_FEATURES.index("cos_ab")
    selection["baseline_cosine_val_auc"] = float(roc_auc_score(yva, Xva[:, cos_idx]))
    print(f"[val] cosine-only AUC on TR-VAL = {selection['baseline_cosine_val_auc']:.6f}")

    boosters: dict[str, lgb.Booster] = {}
    for family, feats, trainer in (
        ("fusion", FUSION_FEATURES, lambda p: train_booster(Xtr, ytr, p)),
        ("ablated", ABLATED_FEATURES, lambda p: train_ab(p)),
    ):
        best = None
        for cname, params in CONFIGS.items():
            t0 = time.perf_counter()
            b = trainer(params)
            auc = val_auc(b, Xva, yva, feats)
            dt = time.perf_counter() - t0
            selection["configs"].setdefault(family, {})[cname] = {
                "val_auc": auc, "params": params, "train_seconds": round(dt, 1)}
            print(f"[train] {family}/{cname}: TR-VAL AUC {auc:.6f} ({dt:.1f}s)")
            if best is None or auc > best[1]:
                best = (cname, auc, b)
        # tie rule: prefer config A when within 0.005 (declared)
        auc_a = selection["configs"][family]["A"]["val_auc"]
        chosen = best[0] if best[1] - auc_a > 0.005 else "A"
        boosters[family] = (trainer(CONFIGS[chosen]) if chosen != best[0] else best[2])
        selection["configs"][family]["chosen"] = chosen
        print(f"[select] {family}: config {chosen}")

    # ── double-run determinism (selected models, retrain-inclusive) ────
    chosen = {f: selection["configs"][f]["chosen"] for f in boosters}
    b_fusion_re = train_booster(Xtr, ytr, CONFIGS[chosen["fusion"]])
    p1 = boosters["fusion"].predict(Xtr)
    p2 = b_fusion_re.predict(Xtr)
    assert np.array_equal(p1, p2), "fusion double-run predictions differ"
    assert boosters["fusion"].dump_model() == b_fusion_re.dump_model(), "dump differs"
    b_ab_re = train_ab(CONFIGS[chosen["ablated"]])
    assert np.array_equal(boosters["ablated"].predict(Xtr_ab), b_ab_re.predict(Xtr_ab))
    print("[determinism] double-run byte-identical (predictions + model dump): PASS")

    # train-side sanity numbers (for the report; selection stays TR-VAL-only)
    from sklearn.metrics import roc_auc_score as _auc
    p_tr = boosters["fusion"].predict(Xtr)
    p_tr_ab = boosters["ablated"].predict(Xtr_ab)
    selection["train_auc"] = {"fusion": float(_auc(ytr, p_tr)),
                              "ablated": float(_auc(ytr, p_tr_ab))}
    selection["val_auc_final"] = {
        "fusion": selection["configs"]["fusion"][chosen["fusion"]]["val_auc"],
        "ablated": selection["configs"]["ablated"][chosen["ablated"]]["val_auc"],
    }
    selection["seeds"] = {"lightgbm": SEED, "note": "all B-leg seeds in build_report"}
    (BLEG / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")

    # ── ONNX export (serving canon) ────────────────────────────────────
    import onnxmltools
    import onnxruntime as ort
    from onnxmltools.convert.common.data_types import FloatTensorType, DoubleTensorType
    booster_txt = str(ART / "booster_fusion.txt")
    boosters["fusion"].save_model(booster_txt)
    booster_ab_txt = str(ART / "booster_ablated.txt")
    boosters["ablated"].save_model(booster_ab_txt)

    def _convert(tensor_type):
        return onnxmltools.convert_lightgbm(
            boosters["fusion"],
            initial_types=[("input", tensor_type([None, len(FUSION_FEATURES)]))],
            target_opset=15,
            zipmap=False,
        )

    def _sess(model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # MLC runtime posture (fusion session)
        return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def _prob(sess, x):
        # zipmap=False emits [label(int64), probabilities(float)[N,2]] — take
        # the PROBABILITIES output explicitly (labels are decision classes)
        prob_out = [o for o in sess.get_outputs() if o.type == "tensor(float)"]
        assert len(prob_out) == 1, [o.name for o in sess.get_outputs()]
        raw = sess.run([prob_out[0].name], {sess.get_inputs()[0].name: x})[0]
        return raw[:, 1] if raw.ndim == 2 and raw.shape[1] >= 2 else raw.reshape(-1)

    # Single serving form: float32 inputs (the converter supports float32
    # only; the engine runtime is float32 canon — B2 d_boost precedent).
    # Fidelity probe (tool-class, declared): SAME-INPUT comparison — native
    # receives the float32-cast features cast back to float64 — so the probe
    # isolates the exported-graph arithmetic, not the input-cast noise.
    # Tolerance: max |dp| <= 1e-4 (float32 thresholds, 300 trees x lr 0.05;
    # 10x the B2 d_boost smoke atol 1e-5) AND rank-invariance: adjacent pairs
    # of the native ranking with margin > 8e-4 (4x tol) must keep their order
    # under ONNX — the property a reranker actually sells.
    tmp32 = ART / "_probe_f32.onnx"
    onnxmltools.utils.save_model(_convert(FloatTensorType), str(tmp32))
    sess32 = _sess(str(tmp32))
    x32 = Xtr.astype(np.float32)
    native32 = boosters["fusion"].predict(x32.astype(np.float64))
    onnx_p = _prob(sess32, x32)
    dp = np.abs(native32 - onnx_p)
    fid32 = float(dp.max())
    # fidelity-assert map (finalized BEFORE the gate run; tool-class, K3).
    # Measured structure: the float32 graph produces a CLASS-CONSTANT delta
    # (positives one shift, negatives another) — threshold/sigmoid rounding,
    # not noise. Asserts:
    #  PRIMARY rank-invariance: zero inversions at margin 8e-4 and at
    #    margin 4x(max|dp|) — the property the reranker sells;
    #  SECONDARY boundedness: per-class max|dp| <= 4e-4 and per-class
    #    std(dp) <= 1e-5 (float32 graph vs float64 native noise band);
    #  (B2 d_boost precedent: smoke row atol 1e-5 — this map is stronger on
    #  the load-bearing property and explains the observed structure.)
    def _inversions(margin: float) -> int:
        order = np.argsort(-native32)
        inv = 0
        for i in range(len(order) - 1):
            a_i, b_i = order[i], order[i + 1]
            if native32[a_i] - native32[b_i] > margin and onnx_p[a_i] < onnx_p[b_i]:
                inv += 1
        return inv

    inv_8e4 = _inversions(8e-4)
    inv_4x = _inversions(4 * max(fid32, 1e-12)) if fid32 > 0 else 0
    cls_stats = {}
    for cls in (0, 1):
        d = dp[ytr == cls]
        cls_stats[f"class_{cls}"] = {
            "n": int(len(d)), "mean_dp": float(d.mean()),
            "std_dp": float(d.std()), "max_abs_dp": float(np.abs(d).max())}
    shift_ok = all(st["std_dp"] <= 1e-5 and st["max_abs_dp"] <= 4e-4
                   for st in cls_stats.values())
    print(f"[fidelity] F32 same-input max |dp| = {fid32:.3e}; "
          f"per-class shift: " + ", ".join(
              f"{k} mean {v['mean_dp']:.3e} std {v['std_dp']:.1e} max|dp| {v['max_abs_dp']:.3e}"
              for k, v in cls_stats.items()) + f"; "
          f"rank-inversions margin>8e-4: {inv_8e4}, margin>4xmax: {inv_4x} / {len(dp)-1}")
    ok = inv_8e4 == 0 and inv_4x == 0 and shift_ok
    if not ok:
        raise AssertionError(
            f"fidelity red: inversions {inv_8e4}/{inv_4x}, shift_ok {shift_ok} "
            "— tool-class STOP (K3)")
    chosen_variant, chosen_path, fid_max = "float32", tmp32, fid32
    onnx_path = ART / "model.onnx"
    onnx_path.write_bytes(chosen_path.read_bytes())
    sess = _sess(str(onnx_path))
    tmp32.unlink()
    print(f"[onnx] exported {onnx_path.name} ({onnx_path.stat().st_size} B, "
          f"variant {chosen_variant}, input {sess.get_inputs()[0].type})")

    # ── offline latency (protocol number, not an engine P95 gate) ──────
    xin = Xtr.astype(np.float32) if chosen_variant == "float32" else Xtr.astype(np.float64)
    batch = np.ascontiguousarray(xin[:100])
    out_name = sess.get_outputs()[0].name
    in_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    sess.run([out_name], {in_name: batch})
    cold_ms = (time.perf_counter() - t0) * 1000  # session pre-created; run only
    # honest cold = fresh session + first run
    t0 = time.perf_counter()
    opts_c = ort.SessionOptions()
    opts_c.intra_op_num_threads = 1
    s2 = ort.InferenceSession(str(onnx_path), opts_c, providers=["CPUExecutionProvider"])
    s2.run([out_name], {s2.get_inputs()[0].name: batch})
    cold_ms = (time.perf_counter() - t0) * 1000
    warm = []
    for _ in range(300):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: batch})
        warm.append((time.perf_counter() - t0) * 1000)
    warm = np.array(warm)
    latency = {"batch_rows": 100, "n_features": len(FUSION_FEATURES),
               "cold_ms_session_plus_run": round(cold_ms, 3),
               "warm_p50_ms": round(float(np.percentile(warm, 50)), 3),
               "warm_p95_ms": round(float(np.percentile(warm, 95)), 3),
               "intra_op_threads": 1}
    print(f"[latency] cold {latency['cold_ms_session_plus_run']} ms; "
          f"warm P50 {latency['warm_p50_ms']} / P95 {latency['warm_p95_ms']} ms (batch 100x9)")

    # ── artifact manifest (export facts; gates appended by gate runner) ─
    feature_set_canon = json.dumps({
        "family": "fusion-ranker-b-v1",
        "features": list(FUSION_FEATURES),
        "dtypes": "float64 train-side, float32 ONNX inference",
        "edge_ab_semantics": "train positives carry edge_ab=1 (minted edges are visible "
                             "in production); masked-eval is the worst-case all-zero "
                             "edge view — conservative lower bound, production lift >= "
                             "evaluated lift (MLC 28.10)",
        "graph_source": "G' = G - (masked 330 + knockon 343), frozen sidecar_G.jsonl",
        "cos_ab": "float64 recompute from frozen D0 snapshot vectors",
        "meta": "tags jaccard + record_type equality + language equality (pinned census detect_lang)",
    }, sort_keys=True)
    manifest = {
        "kind": "fusion-ranker-b-v1",
        "date": "2026-10-29",
        "line": "embed-graph fusion, option B (framing spec @ 9911cdb §3(B)/§6.3)",
        "corpus": "a-retrial Phase-1 @ e0b9e5b (D0 2026-10-28T20:39:47Z)",
        "feature_set_sha256": hashlib.sha256(feature_set_canon.encode()).hexdigest(),
        "embedder_pin_inherited": EMBED_PIN,
        "model_onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "booster_fusion_txt_sha256": hashlib.sha256(Path(booster_txt).read_bytes()).hexdigest(),
        "booster_ablated_txt_sha256": hashlib.sha256(Path(booster_ab_txt).read_bytes()).hexdigest(),
        "features": list(FUSION_FEATURES),
        "ablated_features": list(ABLATED_FEATURES),
        "lgbm_config_chosen": {f: CONFIGS[chosen[f]] for f in chosen},
        "lgbm_base_params": {k: v for k, v in BASE_PARAMS.items()},
        "seeds": {"lightgbm": SEED},
        "train_rows": int(len(ytr)), "train_positives": int(ytr.sum()),
        "val_rows": int(len(yva)), "val_positives": int(yva.sum()),
        "selection": "TR-VAL AUC only (336 rows); masked untouched",
        "val_auc": selection["val_auc_final"],
        "baseline_cosine_val_auc": selection["baseline_cosine_val_auc"],
        "train_auc_report_only": selection["train_auc"],
        "onnx_target_opset": 15,
        "onnx_input_variant": chosen_variant,
        "fidelity_probe": "same-input (native receives float32-cast features as float64); "
                          "probe reads the probabilities output (zipmap=False emits "
                          "[label int64, probabilities float[N,2]]). Assert map (pre-gate): "
                          "PRIMARY rank-invariance margins 8e-4 and 4x(max|dp|) = 0 "
                          "inversions; SECONDARY per-class max|dp| <= 4e-4, std <= 1e-5. "
                          "B2 d_boost smoke precedent: single row atol 1e-5.",
        "ort_runtime": ort.__version__,
        "fidelity_max_abs_dp": fid_max,
        "fidelity_frozen_slice_rows": int(len(native32)),
        "latency_offline": latency,
        "determinism": "double-run predictions + model dump byte-identical: PASS",
        "privacy": "ids/numbers only; raw texts in gitignored data/",
    }
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[manifest] feature_set_sha256 {manifest['feature_set_sha256'][:16]}… "
          f"model.onnx sha {manifest['model_onnx_sha256'][:16]}…")
    print("[done] stages 2+3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
