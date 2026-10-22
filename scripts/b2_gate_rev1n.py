#!/usr/bin/env python3
"""Single-shot gate runner for B2 is-duplicate revision 1.n (prereg v3 §4).

DISCIPLINE (hard order, the b2p-v41 precedent):

  1. g1 sanity suite v2 FIRST — 9/9 required on the gated full artifact.
     A red suite aborts the run: the holdout is not read AT ALL.
  2. Corpus verification (g5) — the 47-check builder-independent verifier.
  3. Pair-level disjointness re-asserts over every eval surface.
  4. The sealed v44 holdout is read EXACTLY ONCE, here, by this runner,
     through the exported ONNX artifacts. No re-rolls: infra failures are
     logged and the numbers stand as they fell.
  5. Gates g2/g3/g4/g6 evaluate on those numbers; a FAIL is a FAIL.

No secrets, no network. Store access only through the frozen sidecar
script (read-only), twice: the v44 holdout catalog and the v42-276 surface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from b2_rev1n_lib import (  # noqa: E402
    ABLATED_NAMES,
    GATED_NAMES,
    build_dataset,
    load_jsonl,
)

STORE_PATH = "/var/home/abyss/.local/share/vesma/core/data"

#: g3 no-harm corridor constants (prereg v3 §4; eval-methodology §4).
PREV_BA = 0.9662
PREV_N = 276
CI95_Z = 1.96
G2 = {"ba": 0.93, "sens": 0.95, "spec": 0.90, "brier": 0.05}
G4_MIN_DELTA = 0.02
G6_MIN_MARGIN = 0.05
CUT = 0.5


def _log(gate_dir: Path, event: str, payload: dict) -> None:
    entry = {"event": event, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
    with (gate_dir / "gate-run-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), file=sys.stderr)


def _metrics(labels, probs) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss

    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs >= CUT).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(labels)),
        "dups": int(labels.sum()),
        "not_dups": int(len(labels) - labels.sum()),
        "ba": float(balanced_accuracy_score(labels, pred)),
        "sens": float(sens),
        "spec": float(spec),
        "brier": float(brier_score_loss(labels, probs)),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


class OnnxScorer:
    """P(dup) through the exported artifact (the serving-surface contract)."""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        manifest = json.loads(onnx_path.with_suffix(".manifest.json").read_text())
        self.feature_names = tuple(manifest["features"])
        self.weights_sha256 = manifest["sha256"]

    def predict(self, vectors) -> np.ndarray:
        if vectors[0].names != self.feature_names:
            raise ValueError("feature contract mismatch vs artifact manifest")
        # The d-boost export contract is float32[K] -> probability[1]: ONE
        # pair per call (the graph's Reshape pins shape_2d=[1, K], so batch
        # feeding is undefined). Row-by-row, same path as sanity._score_pair
        # and the engine's serving wrapper.
        probs = []
        for vector in vectors:
            row = np.asarray(vector.values, dtype=np.float32)
            (out,) = self.session.run(None, {self.input_name: row})
            values = np.ravel(np.asarray(out, dtype=np.float64))
            if values.shape != (1,):
                raise ValueError(
                    f"artifact returned {values.shape} values — expected one probability"
                )
            probs.append(float(values[0]))
        return np.asarray(probs, dtype=np.float64)


def phase1_sanity(gate_dir: Path, bundles: list[Path]) -> dict:
    """g1 — sanity v2 on every artifact; 9/9 required (full is the gate)."""
    from cortex.eval.sanity import run_sanity_suite

    reports = {}
    for bundle in bundles:
        report = run_sanity_suite(bundle)
        payload = {
            "bundle": str(bundle),
            "weights_sha256": report.weights_sha256,
            "passed": report.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in report.checks
            ],
        }
        reports[bundle.name] = payload
        (gate_dir / f"sanity-{bundle.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log(gate_dir, "sanity", {"bundle": bundle.name, "passed": report.passed,
                                  "n_checks": len(report.checks),
                                  "failed": [c.name for c in report.failed]})
    full = next(p for p in bundles if p.stem.endswith("full"))
    gate_payload = reports[full.name]
    verdict = {
        "gate": "g1",
        "what": "sanity suite v2 on the gated full artifact",
        "required": "9/9",
        "measured": f"{sum(c['passed'] for c in gate_payload['checks'])}/{len(gate_payload['checks'])}",
        "pass": gate_payload["passed"] and len(gate_payload["checks"]) == 9,
    }
    return {"reports": reports, "g1": verdict}


def phase2_corpus_verify(gate_dir: Path, corpus_dir: Path, train_dir: Path) -> dict:
    """g5 part 1 — verifiers re-run: the ORIGINAL v44 seal verifier (universe
    + sealed holdout) AND the independent r2 train verifier (ADDENDUM-1)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "b2_graph_corpus_verify.py"),
         "--dir", str(corpus_dir)],
        capture_output=True, text=True, timeout=1800,
    )
    surfaces = [
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2/dataset-v4-holdout/holdout.jsonl",
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2/v43-holdout/holdout.jsonl",
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2/b2/holdout.jsonl",
    ]
    sidecar = REPO_ROOT / "data/b2-graph-corpus/rev1n/graph-sidecar-train.jsonl"
    proc_r2 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "b2_graph_corpus_verify_r2.py"),
         "--train-dir", str(train_dir), "--corpus-dir", str(corpus_dir),
         "--sidecar", str(sidecar), "--holdout-surfaces", *surfaces],
        capture_output=True, text=True, timeout=1800,
    )
    (gate_dir / "corpus-verify.txt").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr
        + "\n\n===== r2 verifier =====\n" + proc_r2.stdout
        + "\n--- stderr ---\n" + proc_r2.stderr,
        encoding="utf-8",
    )
    line = next((ln for ln in proc.stdout.splitlines() if "checks passed" in ln), "")
    line_r2 = next(
        (ln for ln in proc_r2.stdout.splitlines() if "checks" in ln), ""
    )
    _log(gate_dir, "corpus_verify", {"exit": proc.returncode, "line": line.strip(),
                                     "exit_r2": proc_r2.returncode, "line_r2": line_r2.strip()})
    return {
        "gate": "g5",
        "what": "corpus verifiers re-run: v44 seal (47 checks) + v44-r2 train (independent)",
        "required": "all checks, both exit 0",
        "measured": {"v44_seal": line.strip(), "v44_r2_train": line_r2.strip()},
        "pass": proc.returncode == 0 and proc_r2.returncode == 0,
    }


def phase3_disjointness(gate_dir: Path, corpus_dir: Path, v42_dir: Path,
                        train_dir: Path) -> dict:
    """g5 part 2 — pair-level disjointness of the eval surfaces."""
    train_ids = {r["pair_id"] for r in load_jsonl(train_dir / "train.jsonl")}
    hold_ids = {r["pair_id"] for r in load_jsonl(corpus_dir / "holdout.jsonl")}
    v42_ids = {r["pair_id"] for r in load_jsonl(v42_dir / "holdout.jsonl")}
    checks = {
        "holdout_v44_vs_train": len(hold_ids & train_ids) == 0,
        "v42_276_vs_v44_train": len(v42_ids & train_ids) == 0,
        "v42_276_vs_v44_holdout": len(v42_ids & hold_ids) == 0,
    }
    _log(gate_dir, "disjointness", checks)
    return {
        "gate": "g5b",
        "what": "pair-level disjointness of eval surfaces",
        "required": "0 intersections",
        "measured": checks,
        "pass": all(checks.values()),
    }


def phase4_sidecars(gate_dir: Path, corpus_dir: Path, v42_dir: Path) -> dict:
    """Frozen sidecar snapshots for the two eval surfaces (store read-only)."""
    results = {}
    for tag, rows_path in (
        ("holdout-v44", corpus_dir / "holdout.jsonl"),
        ("v42-276", v42_dir / "holdout.jsonl"),
    ):
        catalog = gate_dir / f"sidecar-in-{tag}"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "b2_rev1n_sidecar_input.py"),
             "--rows", str(rows_path), "--out-dir", str(catalog)],
            capture_output=True, text=True, check=True,
        )
        out = gate_dir / f"graph-sidecar-{tag}.jsonl"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "b2_graph_sidecar.py"),
             "--dataset-dir", str(catalog), "--store-path", STORE_PATH,
             "--out", str(out)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sidecar {tag} failed: {proc.stderr[-800:]}")
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
        results[tag] = summary
        _log(gate_dir, "sidecar", {"tag": tag, "fingerprint": summary["sidecar_fingerprint"],
                                   "evidence_pairs": summary["pairs_with_graph_evidence"]})
    return results


def _stratum_table(provenance, labels, probs) -> dict:
    out = {}
    for key in sorted({p["stratum"] for p in provenance}):
        idx = [i for i, p in enumerate(provenance) if p["stratum"] == key]
        out[key] = _metrics([labels[i] for i in idx], [probs[i] for i in idx])
    return out


def main(argv: list[str] | None = None) -> int:
    t0 = time.monotonic()
    parser = argparse.ArgumentParser(prog="b2-gate-rev1n")
    parser.add_argument("--corpus-dir", required=True,
                        help="sealed v44 corpus dir (holdout + seal manifest)")
    parser.add_argument("--train-dir", default=None,
                        help="train dir (rev1n-retry: the v44-r2 rebuild); "
                             "defaults to --corpus-dir")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--v42-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    train_dir = Path(args.train_dir) if args.train_dir else corpus_dir
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    v42_dir = Path(args.v42_dir)
    gate_dir = Path(args.gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)

    full_onnx = run_a / "model-full.onnx"
    ablated_onnx = run_a / "model-ablated.onnx"

    # ── PHASE 1: g1 sanity FIRST — a red suite means the holdout is never read ──
    sanity = phase1_sanity(gate_dir, [full_onnx, ablated_onnx])
    if not sanity["g1"]["pass"]:
        report = {"verdict": "ABORT — g1 red: holdout not read",
                  "gates": [sanity["g1"]], "wall_s": round(time.monotonic() - t0, 1)}
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log(gate_dir, "abort_g1_red", {"wall_s": report["wall_s"]})
        return 2

    # ── PHASE 2/3: g5 corpus verification + disjointness asserts ────────────
    gates = [sanity["g1"], phase2_corpus_verify(gate_dir, corpus_dir, train_dir),
             phase3_disjointness(gate_dir, corpus_dir, v42_dir, train_dir)]
    if not all(g["pass"] for g in gates):
        report = {"verdict": "ABORT — g5 red", "gates": gates,
                  "wall_s": round(time.monotonic() - t0, 1)}
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log(gate_dir, "abort_g5_red", {"wall_s": report["wall_s"]})
        return 2

    # ── PHASE 4: sidecars for the eval surfaces (store read-only) ──────────
    sidecars = phase4_sidecars(gate_dir, corpus_dir, v42_dir)

    from cortex.features.graph import GraphSidecar

    # ── PHASE 5: THE single holdout read + the v42-276 corridor ────────────
    hold_rows = load_jsonl(corpus_dir / "holdout.jsonl")
    hold_labels_rows = {
        r["pair_id"]: str(r["label"]) for r in load_jsonl(corpus_dir / "labels-holdout.jsonl")
    }
    for r in hold_rows:
        if hold_labels_rows.get(str(r["pair_id"])) != str(r["label"]):
            raise ValueError(f"holdout label mismatch on {r['pair_id']!r}")

    sc_hold = GraphSidecar(gate_dir / "graph-sidecar-holdout-v44.jsonl")
    sc_v42 = GraphSidecar(gate_dir / "graph-sidecar-v42-276.jsonl")

    scorer_full = OnnxScorer(full_onnx)
    scorer_abl = OnnxScorer(ablated_onnx)

    def eval_surface(rows, sidecar, scorer_full, scorer_abl):
        vecs_f, labels, prov = build_dataset(rows, sidecar, GATED_NAMES)
        vecs_a, labels_a, prov_a = build_dataset(rows, sidecar, ABLATED_NAMES)
        if labels != labels_a or [p["pair_id"] for p in prov] != [p["pair_id"] for p in prov_a]:
            raise RuntimeError("full/ablated dataset construction diverged")
        pf = scorer_full.predict(vecs_f)
        pa = scorer_abl.predict(vecs_a)
        return {"labels": labels, "prov": prov, "probs_full": pf, "probs_abl": pa}

    hold = eval_surface(hold_rows, sc_hold, scorer_full, scorer_abl)
    _log(gate_dir, "holdout_read_once", {"n": len(hold["labels"])})

    v42_rows = load_jsonl(v42_dir / "holdout.jsonl")
    v42 = eval_surface(v42_rows, sc_v42, scorer_full, scorer_abl)
    _log(gate_dir, "v42_read", {"n": len(v42["labels"])})

    # ── metrics ─────────────────────────────────────────────────────────────
    hold_full = _metrics(hold["labels"], hold["probs_full"])
    hold_abl = _metrics(hold["labels"], hold["probs_abl"])
    strata_full = _stratum_table(hold["prov"], hold["labels"], hold["probs_full"])
    strata_abl = _stratum_table(hold["prov"], hold["labels"], hold["probs_abl"])

    ev_idx = [i for i, p in enumerate(hold["prov"]) if p["has_graph_evidence"]]
    ev_labels = [hold["labels"][i] for i in ev_idx]
    ev_full = _metrics(ev_labels, [hold["probs_full"][i] for i in ev_idx])
    ev_abl = _metrics(ev_labels, [hold["probs_abl"][i] for i in ev_idx])
    g4_delta = ev_full["ba"] - ev_abl["ba"]

    ctrl_idx = [i for i, p in enumerate(hold["prov"]) if p["stratum"] == "control"]
    ctrl_full = _metrics([hold["labels"][i] for i in ctrl_idx],
                         [hold["probs_full"][i] for i in ctrl_idx]) if ctrl_idx else None

    ci95 = CI95_Z * float(np.sqrt(PREV_BA * (1.0 - PREV_BA) / PREV_N))
    corridor = max(0.02, ci95)
    g3_floor = PREV_BA - corridor

    v42_full = _metrics(v42["labels"], v42["probs_full"])
    v42_abl = _metrics(v42["labels"], v42["probs_abl"])
    v42_strata_full = _stratum_table(v42["prov"], v42["labels"], v42["probs_full"])

    # ── gates ───────────────────────────────────────────────────────────────
    gates.extend([
        {
            "gate": "g2", "what": "sealed v4.4 holdout (75), full model",
            "required": "BA>=0.93, sens>=0.95, spec>=0.90, Brier<=0.05",
            "measured": {k: round(hold_full[k], 4) for k in ("ba", "sens", "spec", "brier")},
            "pass": (hold_full["ba"] >= G2["ba"] and hold_full["sens"] >= G2["sens"]
                     and hold_full["spec"] >= G2["spec"] and hold_full["brier"] <= G2["brier"]),
        },
        {
            "gate": "g3", "what": "no-harm corridor on v42-276 (full model)",
            "required": f"BA >= {PREV_BA} - max(0.02; CI95) = {g3_floor:.4f} "
                        f"(CI95={ci95:.4f} from BA(prev) at n={PREV_N})",
            "measured": {"ba": round(v42_full["ba"], 4)},
            "pass": v42_full["ba"] >= g3_floor,
        },
        {
            "gate": "g4", "what": "graph ablation on the holdout evidence stratum",
            "required": f"BA(full) - BA(ablated) >= +{G4_MIN_DELTA}",
            "measured": {"ba_full": round(ev_full["ba"], 4),
                         "ba_ablated": round(ev_abl["ba"], 4),
                         "delta_pp": round(g4_delta * 100, 2),
                         "evidence_rows": ev_full["n"]},
            "pass": g4_delta >= G4_MIN_DELTA,
        },
        {
            "gate": "g6", "what": "anti-collapse: control spec vs evidence spec (full)",
            "required": "spec(control) >= spec(evidence) - 5pp",
            "measured": {"spec_control": None if ctrl_full is None else round(ctrl_full["spec"], 4),
                         "spec_evidence": round(ev_full["spec"], 4),
                         "control_rows": 0 if ctrl_full is None else ctrl_full["n"]},
            "pass": ctrl_full is not None
                    and ctrl_full["spec"] >= ev_full["spec"] - G6_MIN_MARGIN,
        },
    ])

    # ── determinism double-run (runA vs runB) on the same single read ──────
    full_b = OnnxScorer(run_b / "model-full.onnx")
    abl_b = OnnxScorer(run_b / "model-ablated.onnx")
    hold_f_b = full_b.predict(build_dataset(hold_rows, sc_hold, GATED_NAMES)[0])
    hold_a_b = abl_b.predict(build_dataset(hold_rows, sc_hold, ABLATED_NAMES)[0])
    determinism = {
        "full_byte_identical": bool(np.array_equal(hold_f_b.view(np.uint8),
                                                   hold["probs_full"].view(np.uint8))),
        "ablated_byte_identical": bool(np.array_equal(hold_a_b.view(np.uint8),
                                                      hold["probs_abl"].view(np.uint8))),
        "booster_txt_identical": (
            (run_a / "model-full" / "booster.txt").read_bytes()
            == (run_b / "model-full" / "booster.txt").read_bytes()
            and (run_a / "model-ablated" / "booster.txt").read_bytes()
            == (run_b / "model-ablated" / "booster.txt").read_bytes()
        ),
    }
    _log(gate_dir, "determinism", determinism)

    report = {
        "verdict": "ADOPT-candidate" if all(g["pass"] for g in gates) else "FAIL",
        "prereg": "docs/specs/b2-graph-prereg-v3.md (ACTIVATED 2026-10-22) "
                  "+ ADDENDUM-1 (NA-amended; thresholds unchanged)",
        "holdout_reads": (
            "2 — attempt 1 (16:28:41Z) crashed in the scorer harness "
            "(rank-2 batch feed vs the float32[K] single-pair export "
            "contract) BEFORE any probability or gate number existed; "
            "zero eval signal emitted (logged: infra_failure_holdout_read_1); "
            "attempt 2 produced these numbers. No metric-driven re-roll."
        ),
        "gates": gates,
        "holdout_v44": {
            "full": hold_full, "ablated": hold_abl,
            "per_stratum_full": strata_full, "per_stratum_ablated": strata_abl,
            "evidence_stratum": {"full": ev_full, "ablated": ev_abl},
            "control_stratum_full": ctrl_full,
            "evidence_definition": "pool rows with frozen-column evidence OR sidecar has_graph_evidence",
        },
        "v42_276": {
            "full": v42_full, "ablated": v42_abl, "per_stratum_full": v42_strata_full,
            "corridor_floor": round(g3_floor, 4), "ci95": round(ci95, 4),
        },
        "artifacts": {
            "full": {"sha256": scorer_full.weights_sha256,
                     "feature_set_sha256": None},
            "ablated": {"sha256": scorer_abl.weights_sha256},
        },
        "sidecars": {k: {"fingerprint": v["sidecar_fingerprint"],
                         "evidence_pairs": v["pairs_with_graph_evidence"]}
                     for k, v in sidecars.items()},
        "determinism_double_run": determinism,
        "wall_s": round(time.monotonic() - t0, 1),
    }
    manifest_full = json.loads(full_onnx.with_suffix(".manifest.json").read_text())
    manifest_abl = json.loads(ablated_onnx.with_suffix(".manifest.json").read_text())
    report["artifacts"]["full"]["feature_set_sha256"] = manifest_full["feature_set_sha256"]
    report["artifacts"]["ablated"]["feature_set_sha256"] = manifest_abl["feature_set_sha256"]
    report["artifacts"]["full"]["cv_config"] = manifest_full.get("cv_selected_config")
    report["artifacts"]["ablated"]["cv_config"] = manifest_abl.get("cv_selected_config")

    (gate_dir / "gate-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(gate_dir, "gate_done", {"verdict": report["verdict"],
                                 "wall_s": report["wall_s"]})
    print(json.dumps(report["gates"], indent=1, ensure_ascii=False))
    return 0 if all(g["pass"] for g in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
