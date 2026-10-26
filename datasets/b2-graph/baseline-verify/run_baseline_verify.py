#!/usr/bin/env python3
"""Baseline verification: prod v42 (00fc6b71) re-measured on holdout-276 with
the attempt-4 rev1n1 instrument (owner directive 2026-10-25).

DIAGNOSTIC, not a re-roll: every verdict of wave M1.1 stands as it fell; the
seals (holdout v4.5 9eae5ec1, g4-spec-v1 af167dc4) are NOT touched — this run
reads ONLY the open v42-276 surface (the same surface every g3 corridor reads)
and the already-exported model artifacts. No training, no network, no store
access (the sidecar is a file).

One pipeline for all three scorings (the SAME instrument that measured the
candidate in attempt-4 g3):
  - build_dataset from b2_rev1n_lib (frozen feature contract),
  - the attempt-4 fresh v42-276 sidecar (byte-pinned below),
  - OnnxScorer.predict from b2_gate_rev1n1 (serving-surface contract,
    row-by-row float32 -> probability[1]),
  - _metrics from b2_gate_rev1n1 (imported, not re-implemented).

Models scored:
  1. prod v42   — models/vesma-cortex-v1/model.onnx, weights 00fc6b71, 13 features
  2. cand full  — attempt-3 run-rev1n1-a model-full.onnx,  4fa3dd93, 13+5 features
  3. cand ablat — attempt-3 run-rev1n1-a model-ablated.onnx, 772c6ba1, 13 features

Determinism: every model is scored TWICE on the same in-memory vectors;
predictions must be byte-identical (uint8 view equality).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.append(str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import b2_gate_rev1n1 as gate  # noqa: E402
from b2_rev1n_lib import (  # noqa: E402
    ABLATED_NAMES,
    GATED_NAMES,
    build_dataset,
    features_digest_of,
    load_jsonl,
)
from cortex.features.graph import GraphSidecar  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
V42_DIR = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex"
    "/data/stage2/dataset-v4-holdout"
)

# ── identity pins (step 1, fail-closed) ──────────────────────────────────────
V42_WEIGHTS_SHA = "00fc6b71c62dbfa8268db358b39c0a4d9f30148805a4a18986487bc70e8346b2"
V42_FEATURE_SET_SHA = "dd86228f8c634f28d8a15b2d8279da1b99735d68e309be02d30f1c24698fa6af"
V42_MANIFEST_SHA = "bd9fd28a7fe1013bdf2d66a6aa87bb0537951b9f987b632aa044275f64ff168f"
PROD_ONNX = REPO_ROOT / "models/vesma-cortex-v1/model.onnx"
PROD_MANIFEST = REPO_ROOT / "models/vesma-cortex-v1/manifest.json"
RUN_A = REPO_ROOT / "data/b2-graph-corpus/v45/attempt-3/run-rev1n1-a"
SIDECAR = REPO_ROOT / "datasets/b2-graph/v45-1n1-attempt4/graph-sidecar-fresh-v42-276.jsonl"
# known context records (NOT asserted here, reported for provenance):
#   data/stage2/artifact-b2/model.onnx   = 71f0572d… — RECALLED B2 (2026-10-03)
#   data/stage2/artifact-ds1000/model.onnx = 281bd0fd… — prod bundle A5 (ds1000)

G3_RUNLOG_FULL_BA = 0.90625  # attempt-4 run-log, raw value, both passes
N_DUPS, N_NOT_DUPS = 144, 132


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProdScorer(gate.OnnxScorer):
    """The attempt-4 OnnxScorer with an explicit registry-layout manifest
    (models/vesma-cortex-v1/ carries manifest.json, not model.manifest.json).
    predict() is inherited VERBATIM — the serving-surface contract."""

    def __init__(self, onnx_path: Path, manifest_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        manifest = json.loads(Path(manifest_path).read_text())
        self.feature_names = tuple(manifest["features"])
        self.weights_sha256 = manifest["sha256"]


def main() -> int:
    t0 = time.monotonic()
    gate._log(OUT_DIR, "run_start", {
        "purpose": "baseline verify — v42 re-measured on holdout-276 with the "
                   "attempt-4 instrument (owner directive 2026-10-25)",
        "worktree_head": "f26ba3f",
        "diagnostic": "numbers stand as they fell; no verdict re-roll",
    })

    # ══ prolog: env pins + host (frozen §D.1/§D.2 checks, fail-closed) ══
    gate.prolog_env(OUT_DIR)
    gate.prolog_host(OUT_DIR)

    # ══ step 1 — prod artifact identity (fail-closed) ════════════════════
    onnx_sha = _sha256_file(PROD_ONNX)
    manifest = json.loads(PROD_MANIFEST.read_text())
    manifest_sha = _sha256_file(PROD_MANIFEST)
    import cortex

    identity = {
        "prod_onnx_path": str(PROD_ONNX),
        "prod_manifest_path": str(PROD_MANIFEST),
        "onnx_sha256": onnx_sha,
        "manifest_declared_sha256": manifest.get("sha256"),
        "manifest_file_sha256": manifest_sha,
        "registry_eval_manifest_sha256": manifest_sha == V42_MANIFEST_SHA,
        "weights_pin": V42_WEIGHTS_SHA,
        "trained_at": manifest.get("trained_at"),
        "candidate": manifest.get("candidate"),
        "features_match_ablated_contract": tuple(manifest["features"]) == ABLATED_NAMES,
        "feature_set_sha256": manifest.get("feature_set_sha256"),
        "feature_set_digest_recomputed": features_digest_of(ABLATED_NAMES),
        "cortex_package": cortex.__file__,
    }
    checks = {
        "onnx_sha == 00fc6b71 pin": onnx_sha == V42_WEIGHTS_SHA,
        "manifest sha256 == onnx sha": manifest.get("sha256") == onnx_sha,
        "manifest file sha == eval-results pin": manifest_sha == V42_MANIFEST_SHA,
        "13-feature contract (order) == ABLATED_NAMES":
            tuple(manifest["features"]) == ABLATED_NAMES,
        "feature_set digest == dd86228f pin":
            manifest.get("feature_set_sha256") == V42_FEATURE_SET_SHA
            and features_digest_of(ABLATED_NAMES) == V42_FEATURE_SET_SHA,
    }
    failed = [k for k, ok in checks.items() if not ok]
    identity["checks"] = checks
    if failed:
        gate._log(OUT_DIR, "abort_identity", {"failed": failed})
        print(json.dumps({"verdict": "IDENTITY UNRESOLVED — STOP", "failed": failed,
                          "identity": identity}, indent=1, ensure_ascii=False))
        return 3
    gate._log(OUT_DIR, "identity_resolved", identity)

    # candidate artifact byte tripwire (§C pins of the gate runner)
    cand_sha = {}
    for tag in ("full", "ablated"):
        got = _sha256_file(RUN_A / f"model-{tag}.onnx")
        if got != gate.ONNX_SHA[tag]:
            gate._log(OUT_DIR, "abort_identity", {
                "failed": f"candidate {tag} onnx {got} != {gate.ONNX_SHA[tag]}"})
            print(f"IDENTITY UNRESOLVED — candidate {tag} tripwire failed", file=sys.stderr)
            return 3
        cand_sha[tag] = got
    sidecar_sha = _sha256_file(SIDECAR)
    gate._log(OUT_DIR, "artifact_pins", {
        "candidates": cand_sha, "sidecar_sha256": sidecar_sha})

    # ══ surface: v42-276 rows + labels cross-check ═══════════════════════
    rows = load_jsonl(V42_DIR / "holdout.jsonl")
    label_map = {str(r["pair_id"]): str(r["label"]) for r in load_jsonl(V42_DIR / "labels.jsonl")}
    mism = [str(r["pair_id"]) for r in rows
            if label_map.get(str(r["pair_id"])) != str(r["label"])]
    dups = sum(1 for r in rows if str(r["label"]) == "duplicate")
    if mism or len(rows) != 276 or dups != N_DUPS:
        gate._log(OUT_DIR, "abort_surface", {
            "label_mismatches": mism[:5], "rows": len(rows), "dups": dups})
        print("SURFACE ASSERT FAILED (labels/counts) — STOP", file=sys.stderr)
        return 3
    gate._log(OUT_DIR, "surface", {
        "rows": len(rows), "dups": dups, "not_dups": len(rows) - dups,
        "labels_file_agrees": True, "path": str(V42_DIR / "holdout.jsonl")})

    # sidecar: the attempt-4 fresh file; zero-evidence cross-check
    side = GraphSidecar(SIDECAR)
    raw_rows = load_jsonl(SIDECAR)
    zero_evidence = sum(
        1 for r in raw_rows
        if not r.get("has_graph_evidence") and not r.get("a_store_id")
        and not r.get("b_store_id") and int(r.get("common_neighbors") or 0) == 0
    )
    gate._log(OUT_DIR, "sidecar", {
        "path": str(SIDECAR), "sha256": sidecar_sha, "rows": len(raw_rows),
        "zero_evidence_rows": zero_evidence,
        "pairs_without_store_sides": 276 if zero_evidence == 276 else None})

    # ══ step 2 — three scorings, ONE pipeline, double pass each ══════════
    scorers = {
        "prod": (ProdScorer(PROD_ONNX, PROD_MANIFEST), ABLATED_NAMES),
        "cand_full": (gate.OnnxScorer(RUN_A / "model-full.onnx"), GATED_NAMES),
        "cand_ablated": (gate.OnnxScorer(RUN_A / "model-ablated.onnx"), ABLATED_NAMES),
    }
    results: dict[str, dict] = {}
    for tag, (scorer, names) in scorers.items():
        vectors, labels, prov = build_dataset(rows, side, names)
        if vectors[0].names != tuple(scorer.feature_names):
            gate._log(OUT_DIR, "abort_contract", {
                "model": tag, "built": list(vectors[0].names),
                "artifact": list(scorer.feature_names)})
            print(f"FEATURE CONTRACT MISMATCH on {tag} — STOP", file=sys.stderr)
            return 3
        probs1 = scorer.predict(vectors)
        probs2 = scorer.predict(vectors)  # determinism: double scoring pass
        byte_identical = bool(np.array_equal(
            probs1.view(np.uint8), probs2.view(np.uint8)))
        if not byte_identical:
            gate._log(OUT_DIR, "abort_determinism", {"model": tag})
            print(f"DETERMINISM BROKEN on {tag} (double pass differs) — STOP",
                  file=sys.stderr)
            return 3
        m = gate._metrics(labels, probs1)
        csv_path = OUT_DIR / f"{tag}-v42-276.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pair_id", "label", "p_dup", "pred"])
            for p_row, prob in zip(prov, probs1):
                w.writerow([p_row["pair_id"], p_row["label"],
                            repr(float(prob)), int(prob >= gate.CUT)])
        results[tag] = {
            "metrics": m,
            "csv": str(csv_path),
            "weights_sha256": scorer.weights_sha256,
            "n_features": len(vectors[0].names),
            "determinism_double_pass_byte_identical": byte_identical,
        }
        gate._log(OUT_DIR, "scored", {"model": tag, "metrics": m,
                                      "csv": csv_path.name,
                                      "determinism": byte_identical})

    # prod and cand_ablated share the 13-feature contract: identical inputs
    vecs_p, _, _ = build_dataset(rows, side, ABLATED_NAMES)
    vecs_a, _, _ = build_dataset(rows, side, ABLATED_NAMES)
    same_inputs = all(np.array_equal(np.asarray(a.values, dtype=np.float32),
                                     np.asarray(b.values, dtype=np.float32))
                      for a, b in zip(vecs_p, vecs_a))

    # ══ control: candidate full must reproduce the g3 number ═════════════
    full_ba = results["cand_full"]["metrics"]["ba"]
    control_ok = abs(full_ba - G3_RUNLOG_FULL_BA) < 1e-12
    gate._log(OUT_DIR, "control_cand_full_vs_g3", {
        "today": full_ba, "attempt4_runlog": G3_RUNLOG_FULL_BA, "ok": control_ok})
    if not control_ok:
        print("CONTROL FAILED: cand_full BA today != attempt-4 g3 value — "
              "instrument defect, STOP (brief §step2.2)", file=sys.stderr)
        return 3

    # ══ step 3 — gap analysis ════════════════════════════════════════════
    prod_ba = results["prod"]["metrics"]["ba"]
    ci95 = gate.CI95_Z * float(np.sqrt(
        gate.PREV_BA * (1.0 - gate.PREV_BA) / gate.PREV_N))
    lower, upper = gate.PREV_BA - ci95, gate.PREV_BA + ci95
    if abs(prod_ba - gate.PREV_BA) < 1e-12:
        branch = "a_exact"
    elif lower <= prod_ba <= upper:
        branch = "b_within_ci"
    else:
        branch = "c_outside_ci"

    # per-pair prod vs cand_full
    def read_csv(tag: str) -> dict[str, dict]:
        out = {}
        with (OUT_DIR / f"{tag}-v42-276.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["pair_id"]] = {"label": row["label"],
                                       "p_dup": float(row["p_dup"]),
                                       "pred": int(row["pred"])}
        return out

    pp, pf, pa = read_csv("prod"), read_csv("cand_full"), read_csv("cand_ablated")
    assert set(pp) == set(pf) == set(pa)
    flips_pf = [
        {"pair_id": pid, "label": pp[pid]["label"],
         "p_prod": pp[pid]["p_dup"], "p_full": pf[pid]["p_dup"],
         "pred_prod": pp[pid]["pred"], "pred_full": pf[pid]["pred"],
         "abs_dp": abs(pp[pid]["p_dup"] - pf[pid]["p_dup"])}
        for pid in pp if pp[pid]["pred"] != pf[pid]["pred"]
    ]
    flips_by_class = {
        "duplicate_flips": sum(1 for f in flips_pf if f["label"] == "duplicate"),
        "not_duplicate_flips": sum(1 for f in flips_pf if f["label"] == "not-duplicate"),
    }
    top5_divergent = sorted(
        ({"pair_id": pid, "label": pp[pid]["label"],
          "p_prod": pp[pid]["p_dup"], "p_full": pf[pid]["p_dup"],
          "abs_dp": abs(pp[pid]["p_dup"] - pf[pid]["p_dup"])}
         for pid in pp),
        key=lambda d: d["abs_dp"], reverse=True)[:5]
    near_cut_flips = [f for f in flips_pf if min(f["p_prod"], f["p_full"]) < 0.5
                      <= max(f["p_prod"], f["p_full"])]

    full_vs_abl_flips = sum(
        1 for pid in pf if pf[pid]["pred"] != pa[pid]["pred"])

    report = {
        "schema_version": 1,
        "purpose": "baseline verification — prod v42 re-measured on holdout-276 "
                   "with the attempt-4 rev1n1 instrument (owner directive 2026-10-25)",
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "diagnostic_not_reroll": True,
        "seals_touched": "NONE — holdout v4.5 and g4-spec-v1 not read by this run",
        "identity": identity,
        "context_records": {
            "brief_literal_path": str(REPO_ROOT.parent.parent / "data/stage2/artifact-b2"),
            "brief_literal_path_sha256": "71f0572de76c0657c6dd33df81c1155ede4f3e05b4e44f1c75e0080ff367baa7",
            "brief_literal_path_verdict": "RECALLED B2 (2026-10-03 self-pair inversion) "
                                          "— NOT the v42 model; identity resolved via the "
                                          "recorded weights fingerprint 00fc6b71 instead",
            "prod_bundle_a5_ds1000_sha256": "281bd0fd39bf9935c86a8b32fed68191fa9b7da95a83b53eb10844a5fd100ac7",
            "prod_bundle_a5_role": "engine-side serving bundle (superseded by B1 "
                                   "beb0a65d); not the holder of the 0.9662 number",
        },
        "surface": {"path": str(V42_DIR / "holdout.jsonl"), "rows": 276,
                    "dups": dups, "not_dups": len(rows) - dups,
                    "labels_jsonl_agrees": True},
        "sidecar": {"path": str(SIDECAR), "sha256": sidecar_sha,
                    "rows": len(raw_rows), "zero_evidence_rows": zero_evidence},
        "pin_inputs": {"onnx": {**{"prod": onnx_sha}, **cand_sha},
                       "sidecar": sidecar_sha},
        "prod_inputs_equal_ablated_inputs_float32": same_inputs,
        "control_cand_full": {"today_ba": full_ba,
                              "attempt4_g3_runlog_ba": G3_RUNLOG_FULL_BA,
                              "ok": control_ok},
        "metrics": {tag: results[tag]["metrics"] for tag in results},
        "csv": {tag: results[tag]["csv"] for tag in results},
        "determinism": {tag: results[tag]["determinism_double_pass_byte_identical"]
                        for tag in results},
        "analysis": {
            "prev_ba": gate.PREV_BA, "prev_n": gate.PREV_N,
            "ci95_halfwidth": round(ci95, 6),
            "ci_band": [round(lower, 6), round(upper, 6)],
            "prod_ba_today": prod_ba,
            "delta_vs_prev": round(prod_ba - gate.PREV_BA, 6),
            "branch": branch,
            "prod_vs_cand_full": {
                "flips_total": len(flips_pf),
                "flips_by_class": flips_by_class,
                "flips_straddling_cut": len(near_cut_flips),
                "top5_divergent": top5_divergent,
            },
            "cand_full_vs_cand_ablated": {
                "ba_full": results["cand_full"]["metrics"]["ba"],
                "ba_ablated": results["cand_ablated"]["metrics"]["ba"],
                "delta_pp": round(100 * (results["cand_full"]["metrics"]["ba"]
                                         - results["cand_ablated"]["metrics"]["ba"]), 4),
                "pred_flips": full_vs_abl_flips,
            },
            "prod_vs_cand_ablated": {
                "flips_total": sum(1 for pid in pp if pp[pid]["pred"] != pa[pid]["pred"]),
            },
            "env_note": "onnxruntime pinned 1.30.0 (prolog §D.1 fail-closed); "
                        "known embedder env-sensitivity Δcos ≤ 0.005 — see report",
        },
        "wall_s": round(time.monotonic() - t0, 1),
    }
    (OUT_DIR / "verification-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    gate._log(OUT_DIR, "done", {"branch": branch, "prod_ba": prod_ba,
                                "control_ok": control_ok})
    print(json.dumps({
        "prod": results["prod"]["metrics"],
        "cand_full": results["cand_full"]["metrics"],
        "cand_ablated": results["cand_ablated"]["metrics"],
        "control_ok": control_ok, "branch": branch,
        "prod_determinism": results["prod"]["determinism_double_pass_byte_identical"],
        "flips_prod_vs_full": len(flips_pf),
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
