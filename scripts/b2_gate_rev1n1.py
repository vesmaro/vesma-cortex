#!/usr/bin/env python3
"""Single-shot gate runner for B2 is-duplicate revision 1.n+1 (corpus v4.5).

DISCIPLINE (hard order, b2p-v41 precedent; prereg v3 §6 + ADDENDUM-2):

  PHASE 1 — pre-gates, sealed artifacts UNTOUCHED:
    g1  sanity suite v2, 9/9 hard on BOTH models
        (red ⇒ ABORT, the holdout is never read);
    g1b flora of unseen scaffolds (A2.6), both models, 4 probe classes;
    g3  no-harm corridor on v42-276, BA(full) ≥ 0.9449 (frozen;
        CI methodology reported informationally);
  PHASE 2 — SINGLE-SHOT (ONE read of both sealed artifacts):
    seals verified (holdout v4.5 + g4-spec-v1 fingerprints) BEFORE the read;
    fresh sidecars (store read-only) + A2.5.3 evidence-column agreement
    assert (unresolved native sides → quarantine counter, never a silent
    minus; resolved disagreement → ABORT);
    the read: holdout v4.5 + g4-spec-v1 through the exported ONNX artifacts;
    g2  BA ≥ 0.93 ∧ sens ≥ 0.95 ∧ spec ≥ 0.90 ∧ Brier ≤ 0.05 (full);
    g4' recall(full, holdout strata A) = 1.0 ∧ spec(full, g4-spec-v1) ≥ 0.90;
    g5  pair-level disjointness re-assert over every eval surface;
    g6  spec(full, holdout evidence-zero negatives) ≥ spec(full, g4-spec-v1) − 0.05;
    A2.8 characterization (NON-gating): graph-block importances +
    ΔBA(full − ablated) on g4-spec-v1 — answers ONLY the specificity
    question (pre-declared);
    determinism: run-a vs run-b boosters byte-identical + predictions
    byte-identical on the same single read.

No secrets, no network. Store touched only by the frozen sidecar script
(read-only). Any infra failure is logged; numbers stand as they fell —
no metric-driven re-rolls (single-shot).
"""

from __future__ import annotations

import argparse
import hashlib
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
    load_jsonl,
)
from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    manifest_bytes,
)
from cortex.features.graph import GRAPH_FEATURES, GraphEvidence  # noqa: E402
from cortex.features.pair import FeatureVector, PairRecord, features  # noqa: E402

STORE_URI_DEFAULT = "file:/var/home/abyss/.vesma/data/vesma.db?mode=ro"

# ── frozen seals (data/b2-graph-corpus/v45/manifest.json + g4-spec-manifest.json)
HOLDOUT_SEAL = "9eae5ec12b0be03fb1d29a7b1b72aedec7d2386efaf2d6e97eb5c6182a35b492"
HOLDOUT_IDS_SHA = "8961445dd483f77004a3e2d82ae99f76bc8af8ab51f20851844fdbd24716b657"
HOLDOUT_ROWS = 85
G4SPEC_SEAL = "af167dc4d192b9cc7c2fa10b44964fa5b5dc0436b9e265e36b04e4696c997a09"
G4SPEC_IDS_SHA = "fcecd4391f6e60b7ea4cde6790ce38cc640bac5cd82bd9bc5568e5957ddd34a2"
G4SPEC_ROWS = 186

# ── frozen gate thresholds (prereg v3 §4 as ratified; ADDENDUM-2 §A2.6/A2.7;
#    design package §5 — NOT editable here)
G1B = {
    "paradup": (">=", 0.95),   # P(dup) ≥ 0.95 on every probe
    "factedit": ("<", 0.5),    # P(dup) < 0.5 on every probe
    "xlangdup": (">=", 0.5),   # P(dup) ≥ 0.5 on every probe
    "xlangsib": ("<", 0.5),    # P(dup) < 0.5 on every probe
}
G3_FLOOR = 0.9449
PREV_BA = 0.9662
PREV_N = 276
CI95_Z = 1.96
G2 = {"ba": 0.93, "sens": 0.95, "spec": 0.90, "brier": 0.05}
G4SPEC_MIN = 0.90
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
        # single-pair inference: the d-boost export contract is float32[K]
        # → probability[1] (the rev-1.n infra-failure lesson; row by row).
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


def _row_record(side: dict) -> PairRecord:
    return PairRecord(
        title=str(side.get("title", "")),
        body=str(side.get("body", "")),
        tags=tuple(str(t) for t in side.get("tags") or []),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def frozen_row_evidence(row: dict, cos_to_common_max: float | None) -> GraphEvidence:
    """THE one frozen reading (A2.5.3): evidence columns fixed at build."""
    return GraphEvidence(
        edge_ab=bool(int(row.get("edge_ab") or 0)),
        edge_kind_supersedes=bool(int(row.get("edge_kind_supersedes") or 0)),
        common_neighbors=int(row.get("common_neighbors") or 0),
        neighbor_jaccard=float(row.get("neighbor_jaccard") or 0.0),
        cos_to_common_max=cos_to_common_max,
    )


def build_rows_dataset(rows, cos_max_by_pair, gated: bool):
    """Feature matrix from the FROZEN row columns (one reading)."""
    names = GATED_NAMES if gated else ABLATED_NAMES
    vectors, labels, prov = [], [], []
    for row in rows:
        sim = float(row["similarity"])
        base = features(_row_record(row["record"]), _row_record(row["candidate"]), sim)
        if gated:
            ev = frozen_row_evidence(
                row, cos_max_by_pair.get(str(row["pair_id"])))
            vector = FeatureVector(
                GATED_NAMES,
                base.values + (
                    1.0 if ev.edge_ab else 0.0,
                    1.0 if ev.edge_kind_supersedes else 0.0,
                    float(ev.common_neighbors),
                    ev.neighbor_jaccard,
                    0.0 if ev.cos_to_common_max is None else ev.cos_to_common_max,
                ),
            )
        else:
            vector = base
        if vector.names != names:
            raise ValueError(f"row {row['pair_id']!r}: contract mismatch")
        vectors.append(vector)
        labels.append(1 if str(row["label"]) == "duplicate" else 0)
        prov.append({
            "pair_id": str(row["pair_id"]),
            "stratum": str(row.get("stratum", "")),
            "label": str(row["label"]),
            "cluster_key": str(row.get("cluster_key", "")),
            "native": bool(row.get("id_a")),
        })
    return vectors, labels, prov


def fresh_sidecar(gate_dir: Path, tag: str, rows_path: Path, store_uri: str) -> Path:
    """Frozen sidecar pipeline over a surface (store READ-ONLY)."""
    catalog = gate_dir / f"sidecar-in-{tag}"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "b2_rev1n_sidecar_input.py"),
         "--rows", str(rows_path), "--out-dir", str(catalog)],
        capture_output=True, text=True, check=True,
    )
    out = gate_dir / f"graph-sidecar-fresh-{tag}.jsonl"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "b2_graph_sidecar.py"),
         "--dataset-dir", str(catalog), "--store-uri", store_uri, "--out", str(out)],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fresh sidecar {tag} failed: {proc.stderr[-800:]}")
    _log(gate_dir, "fresh_sidecar", {"tag": tag,
        "summary": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None})
    return out


def _seal_check(rows_path: Path, expected_fp: str, expected_ids_sha: str,
                expected_rows: int) -> dict:
    """Fingerprint-level seal verification (ids + pair hashes, no eval use)."""
    rows = load_jsonl(rows_path)
    fp = corpus_fingerprint(manifest_bytes(
        (r["pair_id"], r["pair_sha256"]) for r in rows))
    ids = sorted(r["pair_id"] for r in rows)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    ok = fp == expected_fp and ids_sha == expected_ids_sha and len(rows) == expected_rows
    return {"path": str(rows_path), "rows": len(rows), "fingerprint": fp,
            "ids_sha256": ids_sha, "ok": ok}


def phase1_pregates(gate_dir: Path, run_a: Path, args) -> tuple[list[dict], bool]:
    """g1 + g1b + g3 — sealed artifacts untouched."""
    gates: list[dict] = []
    from cortex.eval.sanity import run_sanity_suite

    # ── g1: sanity v2, 9/9 on BOTH models ────────────────────────────────
    g1_reports = {}
    g1_ok = True
    for onnx in (run_a / "model-full.onnx", run_a / "model-ablated.onnx"):
        report = run_sanity_suite(onnx)
        payload = {
            "bundle": str(onnx), "weights_sha256": report.weights_sha256,
            "passed": report.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in report.checks],
        }
        g1_reports[onnx.stem] = payload
        (gate_dir / f"sanity-{onnx.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        n_pass = sum(c["passed"] for c in payload["checks"])
        _log(gate_dir, "sanity", {"bundle": onnx.name, "passed": report.passed,
                                  "n_pass": n_pass, "n_checks": len(payload["checks"]),
                                  "failed": [c.name for c in report.failed]})
        g1_ok = g1_ok and report.passed and len(payload["checks"]) == 9
    gates.append({
        "gate": "g1", "what": "sanity suite v2, BOTH models",
        "required": "9/9 on full AND ablated",
        "measured": {k: f"{sum(c['passed'] for c in v['checks'])}/{len(v['checks'])}"
                     for k, v in g1_reports.items()},
        "pass": g1_ok,
    })
    if not g1_ok:
        return gates, False

    # ── g1b: flora of unseen scaffolds, BOTH models ──────────────────────
    probe_rows = load_jsonl(gate_dir / "g1b-probes.jsonl")
    scorers = {tag: OnnxScorer(run_a / f"model-{tag}.onnx")
               for tag in ("full", "ablated")}
    per_model = {}
    g1b_ok = True
    for tag, scorer in scorers.items():
        gated = tag == "full"
        per_probe = {}
        for row in probe_rows:
            base = features(_row_record(row["record"]), _row_record(row["candidate"]),
                            float(row["similarity"]))
            vector = (
                FeatureVector(
                    GATED_NAMES,
                    base.values + (0.0, 0.0, 0.0, 0.0, 0.0),
                )
                if gated else base
            )
            p = float(scorer.predict([vector])[0])
            per_probe.setdefault(row["probe_class"], []).append(
                {"pair_id": row["pair_id"], "p": p, "cos": row["similarity"]})
        cls_report = {}
        model_ok = True
        for cls, (op, thr) in G1B.items():
            ps = [x["p"] for x in per_probe.get(cls, [])]
            if not ps:
                raise RuntimeError(f"g1b class {cls}: no probes")
            stat = min(ps) if op == ">=" else max(ps)
            ok = stat >= thr if op == ">=" else stat < thr
            cls_report[cls] = {
                "op": op, "threshold": thr, "n": len(ps),
                "min": round(min(ps), 6), "max": round(max(ps), 6),
                "mean": round(sum(ps) / len(ps), 6),
                "binding": round(stat, 6), "pass": bool(ok),
            }
            model_ok = model_ok and ok
        per_model[tag] = {"classes": cls_report, "pass": model_ok}
        (gate_dir / f"g1b-{tag}.json").write_text(
            json.dumps({"model": tag, "classes": cls_report,
                        "per_probe": per_probe},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        _log(gate_dir, "g1b", {"model": tag, "pass": model_ok,
                               **{k: v["binding"] for k, v in cls_report.items()}})
        g1b_ok = g1b_ok and model_ok
    gates.append({
        "gate": "g1b", "what": "flora of unseen scaffolds (A2.6), both models",
        "required": "; ".join(f"{c}: P(dup) {op} {t}" for c, (op, t) in G1B.items()),
        "measured": {t: {c: v["binding"] for c, v in m["classes"].items()}
                     for t, m in per_model.items()},
        "pass": g1b_ok,
    })
    if not g1b_ok:
        return gates, False

    # ── g3: no-harm corridor on v42-276 (fresh sidecar, store read-only) ──
    v42_dir = Path(args.v42_dir)
    sc_path = fresh_sidecar(gate_dir, "v42-276", v42_dir / "holdout.jsonl",
                            args.store_uri)
    from cortex.features.graph import GraphSidecar

    side = GraphSidecar(sc_path)
    from b2_rev1n_lib import build_dataset

    v42_rows = load_jsonl(v42_dir / "holdout.jsonl")
    vecs_f, labels_f, prov_f = build_dataset(v42_rows, side, GATED_NAMES)
    probs_f = scorers["full"].predict(vecs_f)
    m_full = _metrics(labels_f, probs_f)
    ci95 = CI95_Z * float(np.sqrt(PREV_BA * (1.0 - PREV_BA) / PREV_N))
    gates.append({
        "gate": "g3", "what": "no-harm corridor on v42-276 (full model)",
        "required": f"BA >= {G3_FLOOR} (frozen; = {PREV_BA} - max(0.02, CI95), "
                    f"CI95={ci95:.4f} informational)",
        "measured": {"ba": round(m_full["ba"], 4), "ci95_informational": round(ci95, 4),
                     "sidecar_fingerprint": json.loads(
                         (gate_dir / "graph-sidecar-fresh-v42-276.coverage.json")
                         .read_text()).get("sidecar_fingerprint")},
        "pass": m_full["ba"] >= G3_FLOOR,
    })
    _log(gate_dir, "g3", {"ba": m_full["ba"], "floor": G3_FLOOR})
    return gates, True


def main(argv: list[str] | None = None) -> int:
    t0 = time.monotonic()
    parser = argparse.ArgumentParser(prog="b2-gate-rev1n1")
    parser.add_argument("--corpus-dir", default="data/b2-graph-corpus/v45")
    parser.add_argument("--v42-dir",
                        default="/var/home/abyss/LABs/Projects/Project-Vesma"
                                "/vesma-cortex/data/stage2/dataset-v4-holdout")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--store-uri", default=STORE_URI_DEFAULT)
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    gate_dir = Path(args.gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = corpus_dir / "holdout.jsonl"
    g4spec_path = corpus_dir / "g4-spec-surface.jsonl"
    train_path = corpus_dir / "train.jsonl"

    _log(gate_dir, "gate_start", {
        "revision": "1.n+1", "corpus_dir": str(corpus_dir),
        "sealed_holdout": HOLDOUT_SEAL[:16], "sealed_g4spec": G4SPEC_SEAL[:16],
        "store_uri": args.store_uri})

    # ══ PHASE 1: pre-gates (sealed artifacts untouched) ════════════════════
    gates, ok = phase1_pregates(gate_dir, run_a, args)
    if not ok:
        report = {"verdict": "ABORT — pre-gate red: sealed artifacts not read",
                  "gates": gates, "wall_s": round(time.monotonic() - t0, 1)}
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(gate_dir, "abort_pregate_red", {"wall_s": report["wall_s"]})
        return 2

    # ══ PHASE 2: SINGLE-SHOT ═══════════════════════════════════════════════
    # ── seals FIRST (fingerprint-level, before any eval read) ─────────────
    seal_hold = _seal_check(holdout_path, HOLDOUT_SEAL, HOLDOUT_IDS_SHA, HOLDOUT_ROWS)
    seal_g4 = _seal_check(g4spec_path, G4SPEC_SEAL, G4SPEC_IDS_SHA, G4SPEC_ROWS)
    _log(gate_dir, "seals", {"holdout_ok": seal_hold["ok"], "g4spec_ok": seal_g4["ok"]})
    if not (seal_hold["ok"] and seal_g4["ok"]):
        report = {"verdict": "ABORT — seal verification failed; no eval read",
                  "seals": {"holdout": seal_hold, "g4spec": seal_g4},
                  "wall_s": round(time.monotonic() - t0, 1)}
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2
    gates.append({
        "gate": "seals", "what": "sealed artifacts fingerprints re-derived",
        "required": "holdout v4.5 = 9eae5ec1…, g4-spec-v1 = af167dc4…, row counts",
        "measured": {"holdout_rows": seal_hold["rows"], "g4spec_rows": seal_g4["rows"],
                     "both_match": True},
        "pass": True,
    })

    # ── corpus verifier (builder-independent, corpus-stage instrument) ────
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "b2_graph_corpus_verify_v45.py"),
         "--dir", str(corpus_dir)],
        capture_output=True, text=True, timeout=1800,
    )
    (gate_dir / "corpus-verify.txt").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    _log(gate_dir, "corpus_verify", {"exit": proc.returncode,
                                     "line": next((ln for ln in proc.stdout.splitlines()
                                                   if "OK" in ln or "FAIL" in ln), "")})
    if proc.returncode != 0:
        report = {"verdict": "ABORT — corpus verifier red; no eval read",
                  "gates": gates, "wall_s": round(time.monotonic() - t0, 1)}
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2

    # ── fresh sidecars + A2.5.3 agreement assert ──────────────────────────
    sc_hold_path = fresh_sidecar(gate_dir, "holdout-v45", holdout_path, args.store_uri)
    sc_g4_path = fresh_sidecar(gate_dir, "g4-spec-v1", g4spec_path, args.store_uri)
    # RAW sidecar rows (the GraphSidecar reader drops a_store_id/b_store_id,
    # which the unresolved-sides rule needs)
    raw_hold = {str(r["pair_id"]): r for r in load_jsonl(sc_hold_path)}
    raw_g4 = {str(r["pair_id"]): r for r in load_jsonl(sc_g4_path)}

    def agreement(rows, raw: dict, surface: str):
        """A2.5.3: frozen columns vs fresh sidecar; unresolved native sides
        → quarantine counter (excluded from metrics, reported); resolved
        disagreement → hard abort; synthetic sides must stay unresolved and
        evidence-free."""
        quarantine, checked, mismatches = 0, 0, []
        for row in rows:
            pid = str(row["pair_id"])
            fresh = raw.get(pid)
            if not row.get("id_a"):
                # synthetic: store-external by construction — fresh sidecar
                # must NOT resolve it and must NOT show evidence
                if fresh is not None and (
                        fresh.get("a_store_id") or fresh.get("b_store_id")
                        or fresh.get("edge_ab") or int(fresh.get("common_neighbors") or 0) > 0):
                    raise RuntimeError(
                        f"{surface} {pid}: synthetic row resolved with "
                        "evidence/identity — by-construction violation")
                checked += 1
                continue
            if fresh is None or not fresh.get("a_store_id") or not fresh.get("b_store_id"):
                quarantine += 1
                continue
            if (int(fresh.get("edge_ab") or 0) != int(row.get("edge_ab") or 0)
                    or int(fresh.get("common_neighbors") or 0)
                    != int(row.get("common_neighbors") or 0)):
                mismatches.append(pid)
                continue
            checked += 1
        if mismatches:
            raise RuntimeError(
                f"{surface}: fresh sidecar disagrees with frozen evidence columns "
                f"on {len(mismatches)} rows (e.g. {mismatches[:5]}) — A2.5.3 abort")
        return {"quarantine": quarantine, "agreed": checked, "rows": len(rows)}

    hold_rows = load_jsonl(holdout_path)
    g4_rows = load_jsonl(g4spec_path)
    ag_hold = agreement(hold_rows, raw_hold, "holdout-v45")
    ag_g4 = agreement(g4_rows, raw_g4, "g4-spec-v1")
    _log(gate_dir, "evidence_agreement", {"holdout": ag_hold, "g4spec": ag_g4})

    # frozen cos_to_common_max for the g4-spec surface rides the FROZEN
    # sidecar snapshot cut at carve time (part of the sealed export)
    frozen_g4_side = {}
    for r in load_jsonl(corpus_dir / "g4-spec-graph-sidecar.jsonl"):
        frozen_g4_side[str(r["pair_id"])] = r.get("cos_to_common_max")
    # holdout rows carry their frozen cos_to_common_max column directly;
    # missing column (older synthetic rows) → None → 0.0 fallback in attach
    def holdout_cos_max(row):
        v = row.get("cos_to_common_max")
        return float(v) if v is not None else None

    hold_cos = {str(r["pair_id"]): holdout_cos_max(r) for r in hold_rows}

    # quarantined native rows: excluded from ALL metrics (counter, not minus)
    def quarantined_ids(rows, raw: dict):
        out = set()
        for row in rows:
            if not row.get("id_a"):
                continue
            fresh = raw.get(str(row["pair_id"]))
            if fresh is None or not fresh.get("a_store_id") or not fresh.get("b_store_id"):
                out.add(str(row["pair_id"]))
        return out

    q_hold = quarantined_ids(hold_rows, raw_hold)
    q_g4 = quarantined_ids(g4_rows, raw_g4)
    hold_rows_eval = [r for r in hold_rows if str(r["pair_id"]) not in q_hold]
    g4_rows_eval = [r for r in g4_rows if str(r["pair_id"]) not in q_g4]

    # ── THE read (once, both surfaces, both models) ───────────────────────
    scorers = {tag: OnnxScorer(run_a / f"model-{tag}.onnx")
               for tag in ("full", "ablated")}
    scorers_b = {tag: OnnxScorer(run_b / f"model-{tag}.onnx")
                 for tag in ("full", "ablated")}

    def read_surface(rows, cos_map, tag):
        gated = tag == "full"
        return build_rows_dataset(rows, cos_map, gated)

    surface_preds = {}
    for surface, rows, cos_map in (
            ("holdout", hold_rows_eval, hold_cos),
            ("g4spec", g4_rows_eval, frozen_g4_side)):
        for tag in ("full", "ablated"):
            vecs, labels, prov = read_surface(rows, cos_map, tag)
            probs = scorers[tag].predict(vecs)
            probs_b = scorers_b[tag].predict(vecs)
            if not np.array_equal(probs.view(np.uint8), probs_b.view(np.uint8)):
                raise RuntimeError(
                    f"determinism broken: run-b predictions differ on {surface}/{tag}")
            surface_preds[(surface, tag)] = {"labels": labels, "probs": probs,
                                             "prov": prov}
    _log(gate_dir, "sealed_read_once", {
        "holdout_rows": len(hold_rows_eval), "g4spec_rows": len(g4_rows_eval),
        "quarantine_holdout": len(q_hold), "quarantine_g4spec": len(q_g4)})

    # ── g2: sealed holdout v4.5, FULL ─────────────────────────────────────
    hf = surface_preds[("holdout", "full")]
    m_hold_full = _metrics(hf["labels"], hf["probs"])
    ha = surface_preds[("holdout", "ablated")]
    m_hold_abl = _metrics(ha["labels"], ha["probs"])
    gates.append({
        "gate": "g2", "what": "sealed holdout v4.5 (85; 19 dup / 66 not-dup), full",
        "required": "BA>=0.93, sens>=0.95, spec>=0.90, Brier<=0.05",
        "measured": {k: round(m_hold_full[k], 4)
                     for k in ("ba", "sens", "spec", "brier")},
        "pass": (m_hold_full["ba"] >= G2["ba"] and m_hold_full["sens"] >= G2["sens"]
                 and m_hold_full["spec"] >= G2["spec"]
                 and m_hold_full["brier"] <= G2["brier"]),
    })

    # per-stratum holdout table (both models)
    def strata_table(pred):
        out = {}
        for key in sorted({p["stratum"] for p in pred["prov"]}):
            idx = [i for i, p in enumerate(pred["prov"]) if p["stratum"] == key]
            out[key] = _metrics([pred["labels"][i] for i in idx],
                                [pred["probs"][i] for i in idx])
        return out

    strata_full = strata_table(hf)
    strata_abl = strata_table(ha)

    # ── g4': recall(full, strata A) = 1.0 ∧ spec(full, g4-spec-v1) ≥ 0.90 ──
    a_idx = [i for i, p in enumerate(hf["prov"]) if p["stratum"] == "A-resave"]
    a_rows = [{"pair_id": hf["prov"][i]["pair_id"],
               "cluster_key": hf["prov"][i]["cluster_key"],
               "p": float(hf["probs"][i]),
               "pred_dup": bool(hf["probs"][i] >= CUT)} for i in a_idx]
    a_recall = (sum(1 for r in a_rows if r["pred_dup"]) / len(a_rows)) if a_rows else 0.0
    clusters = {}
    for r in a_rows:
        clusters.setdefault(r["cluster_key"] or r["pair_id"], []).append(r["pred_dup"])
    gf = surface_preds[("g4spec", "full")]
    m_g4_full = _metrics(gf["labels"], gf["probs"])
    ga = surface_preds[("g4spec", "ablated")]
    m_g4_abl = _metrics(ga["labels"], ga["probs"])
    g4_pass = a_rows and a_recall == 1.0 and m_g4_full["spec"] >= G4SPEC_MIN
    gates.append({
        "gate": "g4'", "what": "re-saves recall + evidence-hard-negative specificity",
        "required": "recall(full, strata A holdout) = 1.0 ∧ spec(full, g4-spec-v1) >= 0.90",
        "measured": {"recall_strata_a": round(a_recall, 4), "strata_a_rows": len(a_rows),
                     "strata_a_clusters": {k: all(v) for k, v in clusters.items()},
                     "spec_g4spec_v1": round(m_g4_full["spec"], 4)},
        "pass": bool(g4_pass),
    })

    # ── g5: pair-level disjointness over every eval surface ───────────────
    train_rows = load_jsonl(train_path)
    v42_rows = load_jsonl(Path(args.v42_dir) / "holdout.jsonl")
    id_sets = {
        "train-v45": {str(r["pair_id"]) for r in train_rows},
        "holdout-v45": {str(r["pair_id"]) for r in hold_rows},
        "g4-spec-v1": {str(r["pair_id"]) for r in g4_rows},
        "v42-276": {str(r["pair_id"]) for r in v42_rows},
    }
    inter = {}
    names = list(id_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = len(id_sets[names[i]] & id_sets[names[j]])
            if n:
                inter[f"{names[i]}∩{names[j]}"] = n
    sha_sets = {k: {str(r["pair_sha256"]) for r in rs if r.get("pair_sha256")}
                for k, rs in (("train-v45", train_rows), ("holdout-v45", hold_rows),
                              ("g4-spec-v1", g4_rows))}
    sha_inter = {f"{a}∩{b}": len(sha_sets[a] & sha_sets[b])
                 for i, a in enumerate(("train-v45", "holdout-v45", "g4-spec-v1"))
                 for b in ("train-v45", "holdout-v45", "g4-spec-v1")[i + 1:]
                 if sha_sets[a] & sha_sets[b]}
    gates.append({
        "gate": "g5", "what": "pair-level disjointness re-assert, all eval surfaces",
        "required": "0 intersections (pair ids; pair hashes where frozen)",
        "measured": {"pair_id_intersections": inter or "none",
                     "pair_sha_intersections": sha_inter or "none"},
        "pass": not inter and not sha_inter,
    })

    # ── g6: anti-collapse (evidence-zero holdout negatives vs surface) ────
    gates.append({
        "gate": "g6",
        "what": "spec(full, holdout evidence-zero negatives) vs spec(full, g4-spec-v1)",
        "required": f"spec(holdout not-dup) >= spec(g4-spec-v1) - {G6_MIN_MARGIN}",
        "measured": {"spec_holdout_notdup": round(m_hold_full["spec"], 4),
                     "spec_g4spec_v1": round(m_g4_full["spec"], 4)},
        "pass": m_hold_full["spec"] >= m_g4_full["spec"] - G6_MIN_MARGIN,
    })

    # ── A2.8 characterization (NON-gating, pre-declared scope) ────────────
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(run_a / "model-full" / "booster.txt"))
    feat_names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain = float(gain.sum()) or 1.0
    importances = {n: {"gain": float(g), "gain_share": round(float(g) / total_gain, 6),
                       "splits": int(s)}
                   for n, g, s in zip(feat_names, gain, split)}
    graph_imp = {n: importances[n] for n in GRAPH_FEATURES}
    graph_gain_share = round(sum(v["gain"] for v in graph_imp.values()) / total_gain, 6)
    core_names = [n for n in feat_names if n not in GRAPH_FEATURES]
    dead_core = [n for n in core_names if importances[n]["gain"] <= 0.0]
    delta_surface = m_g4_full["spec"] - m_g4_abl["spec"]  # all-negative surface:
    # accuracy == spec there, so ΔBA(full − ablated) = Δspec (pre-declared)
    characterization = {
        "scope": "A2.8 pre-declaration: specificity question ONLY (not a gate)",
        "graph_block_gain_share": graph_gain_share,
        "graph_features": graph_imp,
        "core_features_dead_gain": dead_core,
        "delta_spec_g4spec_surface_full_minus_ablated": round(delta_surface, 4),
        "spec_full": round(m_g4_full["spec"], 4),
        "spec_ablated": round(m_g4_abl["spec"], 4),
        "surface_rows": m_g4_full["n"],
    }
    (gate_dir / "characterization-a2.8.json").write_text(
        json.dumps(characterization, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(gate_dir, "a2.8_characterization", {
        "graph_gain_share": graph_gain_share, "delta_spec": round(delta_surface, 4),
        "dead_core": dead_core})

    # ── determinism double-run (artifacts, not just predictions) ──────────
    det = {
        "booster_txt_identical": all(
            (run_a / f"model-{tag}" / "booster.txt").read_bytes()
            == (run_b / f"model-{tag}" / "booster.txt").read_bytes()
            for tag in ("full", "ablated")),
        "onnx_sha_identical": all(
            hashlib.sha256((run_a / f"model-{tag}.onnx").read_bytes()).hexdigest()
            == hashlib.sha256((run_b / f"model-{tag}.onnx").read_bytes()).hexdigest()
            for tag in ("full", "ablated")),
        "predictions_byte_identical": True,  # asserted during the read above
    }
    _log(gate_dir, "determinism", det)

    verdict = "ADOPT" if all(g["pass"] for g in gates) else "REJECT"
    red = [g["gate"] for g in gates if not g["pass"]]
    report = {
        "verdict": verdict,
        "red_gates": red,
        "prereg": "docs/specs/b2-graph-prereg-v3.md (ACTIVATED 22.10) + ADDENDUM-1/2 "
                  "(ACTIVATED 22.10) + ACTIVATION-NOTE g4-spec-v1 (73ffa50); "
                  "thresholds frozen, single-shot",
        "gates": gates,
        "holdout_v45": {
            "full": m_hold_full, "ablated": m_hold_abl,
            "per_stratum_full": strata_full, "per_stratum_ablated": strata_abl,
            "quarantine": len(q_hold),
        },
        "g4_spec_v1": {
            "full": m_g4_full, "ablated": m_g4_abl,
            "quarantine": len(q_g4),
        },
        "strata_a_detail": a_rows,
        "evidence_agreement": {"holdout": ag_hold, "g4spec": ag_g4},
        "sidecars_fresh": {
            "holdout": json.loads((gate_dir / "graph-sidecar-fresh-holdout-v45.coverage.json")
                                  .read_text()).get("sidecar_fingerprint"),
            "g4spec": json.loads((gate_dir / "graph-sidecar-fresh-g4-spec-v1.coverage.json")
                                 .read_text()).get("sidecar_fingerprint"),
        },
        "characterization_a2.8": characterization,
        "determinism_double_run": det,
        "artifacts": {
            tag: {"sha256": scorers[tag].weights_sha256,
                  "feature_set_sha256": json.loads(
                      (run_a / f"model-{tag}.manifest.json").read_text())["feature_set_sha256"],
                  "cv_config": json.loads(
                      (run_a / f"model-{tag}.manifest.json").read_text()).get("cv_selected_config")}
            for tag in ("full", "ablated")
        },
        "wall_s": round(time.monotonic() - t0, 1),
    }
    (gate_dir / "gate-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(gate_dir, "gate_done", {"verdict": verdict, "red": red,
                                 "wall_s": report["wall_s"]})
    print(json.dumps({"verdict": verdict, "red_gates": red,
                      "gates": [{k: g[k] for k in ("gate", "required", "measured", "pass")}
                                for g in gates]},
                     indent=1, ensure_ascii=False))
    return 0 if verdict == "ADOPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
