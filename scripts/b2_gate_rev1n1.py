#!/usr/bin/env python3
"""Single-shot gate runner for B2 is-duplicate revision 1.n+1 (corpus v4.5).

ATTEMPT 4 of the gate cycle under **ADDENDUM-4** (ratified: owner delegation
24.10 + unanimous MLC verdict 25.10, revision A2.6). Attempt 3 aborted at
pre-gate g1b on the sib leg (xlangsib full 0.9889 = 1/8 < 0.5 at the
pre-registered forecast); the red is the material confirmation of the
13-feature contract limit, not a model defect — retraining is a
metric-driven re-roll and is FORBIDDEN. Attempt 4 is a CONTROLLED
REPRODUCTION on the byte-stable attempt-1/2/3 boosters (no training), with:

  PROLOG (ADDENDUM-4 §D, enforced §D numbering, BEFORE any eval read):
    swap_check (brief: iron rule, first run-log event) → env (interpreter +
    package pins, fail-closed, §D.1) → hostname (§D.2) → byte tripwire
    (boosters + ONNX sha equality vs the §C pins, RUNNER-SIDE fail-closed,
    §D.3) → addendum_activation ("ADDENDUM-4 ACTIVATED" commit strictly
    before gate_start, §D.4) → frozen inputs (probes sha + 32/24/16/8
    count assert; quarantine ledger sha + 370 + §A apply-log cross-check;
    kNN train-sidecar sha; v4.5 train corpus fingerprint, §D.5);
  g1   sanity suite v2, 9/9 hard on BOTH models under the ADDENDUM-3 A3.1
       plateau rule implemented RUNNER-SIDE (src/cortex/eval/sanity.py is
       frozen prod code — the M3 migration on ADOPT is a separate decision);
  g1b  REDUCED gating composition (A2.6): paradup (32, P(dup) ≥ 0.95 on
       every probe) + factedit (24, P(dup) < 0.5 on every probe), BOTH
       models; thresholds and abort semantics untouched (red g1b = the
       holdout is never read). The xlang legs (xlangdup 16 / xlangsib 8)
       are COMPUTED on the same probes, both models, published, and do NOT
       enter the verdict conjunction (characterization, pre-declared scope:
       does the 13-feature contract limit reproduce on the cross-lang axis);
  DET  determinism assert vs attempt 3: per-probe P(dup) vectors, 80 × 2
       models = 160 values, printed-precision comparison (%.4f, string),
       references sha-pinned (attempt-3 g1b-full/g1b-ablated.json on
       commit bd410ff). Any divergence = instrument-class abort, not a gate;
  KNNE characterization block (§E, non-gating, BEFORE single-shot): kNN
       manifold check — cosine, k=5, train-to-train null leave-one-out
       (self-neighbor excluded), quantiles 0.99 and 0.01 from ONE code
       path; the frozen pre-built train sidecar is sha-pinned in the
       prolog; the full 80-distance vector + both quantiles + raw counters
       go to the run-log; off-manifold ≥ 8/80 is a REPORT branch (never an
       abort); near-tail ≥ 1 attaches the "memorization cannot be excluded"
       qualifier to the +0.021 margin disclosure. No RNG;
  g3   no-harm corridor on v42-276, BA(full) ≥ 0.9449 (frozen);
  PHASE 2 — SINGLE-SHOT (ONE read of both sealed artifacts):
    seals verified (holdout v4.5 + g4-spec-v1 fingerprints) BEFORE the read;
    fresh sidecars (store read-only) + A2.5.3 evidence-column agreement
    assert; the read: holdout v4.5 + g4-spec-v1 through the exported ONNX
    artifacts;
    g2  BA ≥ 0.93 ∧ sens ≥ 0.95 ∧ spec ≥ 0.90 ∧ Brier ≤ 0.05 (full);
    g4' recall(full, holdout strata A) = 1.0 ∧ spec(full, g4-spec-v1) ≥ 0.90;
    g5  pair-level disjointness re-assert over every eval surface;
    g6  spec(full, holdout evidence-zero negatives) ≥ spec(full, g4-spec-v1) − 0.05;
    A2.8 characterization (NON-gating): graph-block importances +
    ΔBA(full − ablated) on g4-spec-v1 — answers ONLY the specificity
    question (pre-declared);
    determinism section = the vector assert vs attempt 3 + a DOUBLE SCORING
    PASS over the already-read in-memory data (predictions byte-identical).
    NO training — the attempt-2 train double-run sections are NOT executed.

Disclosures (ADDENDUM-4 §F, canonical text in gate-report): contract limit;
two-sided edge geometry of the dup leg (with the near-tail qualifier iff it
fires); flora power 1/30 (S24); §D alignment report-only; §A quarantine as
the load-bearing mechanism; both population numbers (6/3654 and 370).

No secrets, no network. Store touched only by the frozen sidecar script
(read-only). Any infra failure is logged; numbers stand as they fell —
no metric-driven re-rolls (single-shot). Regeneration of probes and
retraining are FORBIDDEN (ADDENDUM-4 §C).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# APPEND, not insert(0) — same reason as b2_g1b_probes.py: cortex lives in
# src/, vesma/vesmaro/mnemos must resolve via ENGINE_SRC (prepended by the
# builder at make_embedder time); inserting src/ first would shadow the
# engine package with the local namespace stub.
sys.path.append(str(REPO_ROOT / "src"))

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
from cortex.features.pair import (  # noqa: E402
    FEATURE_NAMES,
    FeatureVector,
    PairRecord,
    features,
)

STORE_URI_DEFAULT = "file:/var/home/abyss/.vesma/data/vesma.db?mode=ro"

# ── frozen seals (data/b2-graph-corpus/v45/manifest.json + g4-spec-manifest.json)
HOLDOUT_SEAL = "9eae5ec12b0be03fb1d29a7b1b72aedec7d2386efaf2d6e97eb5c6182a35b492"
HOLDOUT_IDS_SHA = "8961445dd483f77004a3e2d82ae99f76bc8af8ab51f20851844fdbd24716b657"
HOLDOUT_ROWS = 85
G4SPEC_SEAL = "af167dc4d192b9cc7c2fa10b44964fa5b5dc0436b9e265e36b04e4696c997a09"
G4SPEC_IDS_SHA = "fcecd4391f6e60b7ea4cde6790ce38cc640bac5cd82bd9bc5568e5957ddd34a2"
G4SPEC_ROWS = 186

# ── frozen gate thresholds (prereg v3 §4 as ratified; ADDENDUM-2 §A2.6 as
#    revised by ADDENDUM-4 §B; design package §5 — NOT editable here)
G1B_GATING = {
    "paradup": (">=", 0.95),  # P(dup) ≥ 0.95 on every probe
    "factedit": ("<", 0.5),  # P(dup) < 0.5 on every probe
}
G1B_CHARACTERIZED = ("xlangdup", "xlangsib")  # §B: characterization, NON-gating
G3_FLOOR = 0.9449
PREV_BA = 0.9662
PREV_N = 276
CI95_Z = 1.96
G2 = {"ba": 0.93, "sens": 0.95, "spec": 0.90, "brier": 0.05}
G4SPEC_MIN = 0.90
G6_MIN_MARGIN = 0.05
CUT = 0.5

# ── ADDENDUM-3 (ACTIVATED 2026-10-23) ────────────────────────────────────────
# A3.1 plateau exclusion — RUNNER-SIDE only (the frozen suite keeps its 1e-6
# constant; prod migration is an M3 question on ADOPT).
A31_PLATEAU_TOL = 1e-4
A31_PLATEAU_FLOOR = 0.99

# ── ADDENDUM-4 (ACTIVATED before the attempt-4 run; §C frozen inputs) ────────
ADDENDUM4_PATH = "docs/specs/b2-graph-prereg-v3-ADDENDUM-4.md"
ADDENDUM4_ACTIVATION_PREFIX = "docs(specs): ADDENDUM-4 ACTIVATED"

# Byte tripwire — fail-closed, RUNNER-SIDE, BEFORE any read (§D.3). The
# attempt-4 boosters MUST be byte-identical to attempt 1/2/3.
BOOSTER_SHA = {
    "full": "7e9cacb4d62fa6a4006b9ff0d60e03b71b7b3e095d414a417cd6877aba68a6a6",
    "ablated": "a7721923404be6c9476f499dffc4beb071ae1295d8ed726818c04efc0c5af6e3",
}
# ONNX artifacts equal the attempt-3 manifests (§C; A != B only by the
# trained_at stamp — run-b ONNX is NOT pinned, booster.txt continuity is).
ONNX_SHA = {
    "full": "4fa3dd93f03040d9df77cb185a1626485e0642db47edeaea5f0686600ac88a9b",
    "ablated": "772c6ba1bed2e212a3925477c63a119c5ebc0993a6a2d4df0655abf443520ac7",
}

# Probe set: frozen, regeneration FORBIDDEN (§C). Count assert 32/24/16/8.
PROBES_SHA = "82dbff7b91519862cdafa1a39a2cd2fae7bb6ee37ae4cda853fe2daf1d5e381f"
PROBE_COUNTS = {"paradup": 32, "factedit": 24, "xlangdup": 16, "xlangsib": 8}

# Quarantine ledger §A (v44-r2 manifest): sha-pinned, pool_pairs_quarantined
# = 370, cross-checked against the §A apply log of the same build.
LEDGER_SHA = "6777f49365fc75ed53719e3ca3651d9fe302677fcb1316e2270054e08bfad7b8"
LEDGER_QUARANTINED = 370
LEDGER_FIELD = (
    "addendum1_numbers",
    "cross_language_quarantine",
    "pool_pairs_quarantined",
)

# Determinism references (attempt 3, commit bd410ff, sha-pinned §C).
REF_FULL_SHA = "b61e8d6dda958f14771ab018f6094dc5cf0d9820a8bb4f0cbfccc715c5c6fd16"
REF_ABLATED_SHA = "0e8c0dd0a11ab394c9603478f6ac10d510acb7eee8e4388f2d81d792b6afa1a3"

# v4.5 train corpus fingerprint (§C prolog assert) and the PRE-BUILT kNN
# train sidecar (§E — built BEFORE gate_start by b2_knn_train_sidecar.py;
# rebuilds are NOT byte-identical by design, the pre-built file is the
# frozen input pinned here like the boosters).
CORPUS_TRAIN_FP = "3bf7ed53e1dbf9092ec1e557c539b6fedd202ffd8fdbe9798643e2f6fc828a70"
KNN_SIDECAR_SHA = "8c6461ec72085e489c6e83b0ba76288d6c22bf8abbadd71fb0d6735f3e070664"

# Environment pins (§D.1; embedder env-sensitivity Δcos ≤ 0.005 — exact pins,
# fail-closed). `python3` from PATH is a broken toolbox shim — the runner
# MUST be invoked through the venv interpreter.
VENV_INTERPRETER_SUFFIX = "vesma-cortex/.venv/bin/python"
PYTHON_PIN = "3.12.13"
PACKAGE_PINS = {
    "lightgbm": "4.7.0",
    "onnxruntime": "1.30.0",
    "scikit-learn": "1.9.1",
    "tokenizers": "0.22.2",
}

# kNN characterization block (§E; MLC pins: cosine, k=5, LOO null, quantiles
# 0.99/0.01 from ONE code path, quantile method fixed and published; off-
# manifold ≥ 8/80 = report branch; no RNG). The pair→scalar reduction of a
# probe (two sides) is the ML-fixed convention (MLC Р1): probe distance =
# MAX of its two side distances (worst side — conservative for the manifold-
# validity reading); both per-side vectors (160 values) are published
# alongside, nothing is hidden.
KNN_K = 5
KNN_METRIC = "cosine (1 - mean of k=5 nearest cosine similarities)"
KNN_QUANTILE_METHOD = "numpy.quantile(method='linear')"
KNN_Q_HI, KNN_Q_LO = 0.99, 0.01
KNN_OFFMANIFOLD_MIN = 8  # of 80 — report branch threshold, paired with Q_HI
KNN_PROBE_REDUCTION = (
    "max of the two side distances (worst side); per-side vectors published in full"
)

# ══════════════════════════════════════════════════════════════════════════════


class InstrumentAbort(RuntimeError):
    """Abort class «инструмент» (ADDENDUM-4 §D.6): a defect of the tool or
    of a frozen input — never a gate verdict. Seals stay untouched."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _log(gate_dir: Path, event: str, payload: dict) -> None:
    entry = {
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
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
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
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
            ev = frozen_row_evidence(row, cos_max_by_pair.get(str(row["pair_id"])))
            vector = FeatureVector(
                GATED_NAMES,
                base.values
                + (
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
        prov.append(
            {
                "pair_id": str(row["pair_id"]),
                "stratum": str(row.get("stratum", "")),
                "label": str(row["label"]),
                "cluster_key": str(row.get("cluster_key", "")),
                "native": bool(row.get("id_a")),
            }
        )
    return vectors, labels, prov


def fresh_sidecar(gate_dir: Path, tag: str, rows_path: Path, store_uri: str) -> Path:
    """Frozen sidecar pipeline over a surface (store READ-ONLY)."""
    catalog = gate_dir / f"sidecar-in-{tag}"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "b2_rev1n_sidecar_input.py"),
            "--rows",
            str(rows_path),
            "--out-dir",
            str(catalog),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = gate_dir / f"graph-sidecar-fresh-{tag}.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "b2_graph_sidecar.py"),
            "--dataset-dir",
            str(catalog),
            "--store-uri",
            store_uri,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fresh sidecar {tag} failed: {proc.stderr[-800:]}")
    _log(
        gate_dir,
        "fresh_sidecar",
        {
            "tag": tag,
            "summary": proc.stdout.strip().splitlines()[-1]
            if proc.stdout.strip()
            else None,
        },
    )
    return out


def _seal_check(
    rows_path: Path, expected_fp: str, expected_ids_sha: str, expected_rows: int
) -> dict:
    """Fingerprint-level seal verification (ids + pair hashes, no eval use)."""
    rows = load_jsonl(rows_path)
    fp = corpus_fingerprint(
        manifest_bytes((r["pair_id"], r["pair_sha256"]) for r in rows)
    )
    ids = sorted(r["pair_id"] for r in rows)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    ok = (
        fp == expected_fp and ids_sha == expected_ids_sha and len(rows) == expected_rows
    )
    return {
        "path": str(rows_path),
        "rows": len(rows),
        "fingerprint": fp,
        "ids_sha256": ids_sha,
        "ok": ok,
    }


def _a31_for_bundle(onnx_path: Path) -> dict:
    """ADDENDUM-3 A3.1 plateau re-evaluation of the zoned ladder (RUNNER-SIDE).

    Raw probabilities come from the FROZEN suite helper (same probes, same
    order, same graph-empty gating) — the suite itself stays untouched and
    keeps its 1e-6 constant; this function adds the ratified zone:
    a controlled step passes at tol 1e-4 IFF both its points >= 0.99,
    else at 1e-6; steps starting strictly above RAZOR_ZONE_COS_LOW are not
    controlled (zoned v2, unchanged)."""
    import onnxruntime as ort

    from cortex.eval.sanity import (
        COS_LADDER,
        MONOTONICITY_TOLERANCE,
        RAZOR_ZONE_COS_LOW,
        ladder_probabilities,
    )

    manifest = json.loads(onnx_path.with_suffix(".manifest.json").read_text())
    graph_gated = tuple(manifest["features"]) == FEATURE_NAMES + GRAPH_FEATURES
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ps = [float(p) for p in ladder_probabilities(session, graph_gated=graph_gated)]
    steps: list[dict] = []
    ok = True
    for i in range(len(COS_LADDER) - 1):
        lo, hi = COS_LADDER[i], COS_LADDER[i + 1]
        if lo > RAZOR_ZONE_COS_LOW:
            steps.append(
                {
                    "from_cos": lo,
                    "to_cos": hi,
                    "controlled": False,
                    "tol": None,
                    "pass": None,
                }
            )
            continue
        tol = (
            A31_PLATEAU_TOL
            if ps[i] >= A31_PLATEAU_FLOOR and ps[i + 1] >= A31_PLATEAU_FLOOR
            else MONOTONICITY_TOLERANCE
        )
        step_pass = bool(ps[i] >= ps[i + 1] - tol)
        steps.append(
            {
                "from_cos": lo,
                "to_cos": hi,
                "controlled": True,
                "p_from": ps[i],
                "p_to": ps[i + 1],
                "delta_p_to_minus_p_from": ps[i + 1] - ps[i],
                "tol": tol,
                "pass": step_pass,
            }
        )
        ok = ok and step_pass
    return {
        "raw_ladder": {str(cos): p for cos, p in zip(COS_LADDER, ps)},
        "steps": steps,
        "pass": bool(ok),
    }


def _a31_crosscheck(monotonicity_detail: str, raw_ladder: dict) -> None:
    """Drift guard: the recomputed raw ladder must reproduce the numbers the
    frozen suite itself printed (4-decimal) — a divergence means the probe or
    scoring contract moved between the suite run and this re-evaluation."""
    import re

    printed = [
        (cos, float(p))
        for cos, p in re.findall(r"cos=([\d.]+):([\d.]+)", monotonicity_detail)
    ]
    if not printed:
        raise RuntimeError(
            "monotonicity detail carries no ladder values — "
            "sanity suite output contract changed; STOP"
        )
    for cos, p_printed in printed:
        # the suite prints two-decimal cosines ("cos=0.80"); raw_ladder is
        # keyed by str(COS_LADDER floats) ("0.8") — normalize through float.
        p_raw = raw_ladder.get(str(float(cos)))
        if p_raw is None or f"{p_raw:.4f}" != f"{p_printed:.4f}":
            raise RuntimeError(
                f"raw ladder drift at cos={cos}: recomputed {p_raw!r} vs "
                f"suite-printed {p_printed!r} — STOP, do not gate"
            )


# ════════════════════════════════════ PROLOG (ADDENDUM-4 §D) ══════════════════


def prolog_swap_check(gate_dir: Path) -> None:
    """Hard iron rule: swap state captured into the run-log BEFORE the run."""
    parts = {}
    for name, cmd in (("swapon", ["swapon", "--show"]), ("free", ["free", "-h"])):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise InstrumentAbort(
                f"swap-check command {cmd} failed: {proc.stderr[-200:]}"
            )
        parts[name] = proc.stdout.strip()
    if not parts["swapon"].strip():
        raise InstrumentAbort("swapon --show is empty — no swap configured; STOP")
    _log(gate_dir, "swap_check", parts)


def prolog_env(gate_dir: Path) -> None:
    """§D.1: interpreter + package pins, fail-closed, into the run-log."""
    exe = str(sys.executable)  # NOT resolved: the venv python is a symlink
    # to the uv-managed cpython — resolving would hide the venv identity.
    pyver = platform.python_version()
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    versions = {}
    problems = []
    for dist, pin in PACKAGE_PINS.items():
        try:
            got = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            got = None
        versions[dist] = got
        if got != pin:
            problems.append(f"{dist}: got {got!r}, pin {pin!r}")
    if not exe.endswith(VENV_INTERPRETER_SUFFIX):
        problems.append(
            f"interpreter {exe!r} is not the pinned venv (*{VENV_INTERPRETER_SUFFIX})"
        )
    if not in_venv:
        problems.append(
            "not running inside a virtualenv (sys.prefix == sys.base_prefix)"
        )
    if pyver != PYTHON_PIN:
        problems.append(f"python: got {pyver!r}, pin {PYTHON_PIN!r}")
    if problems:
        raise InstrumentAbort("environment pins violated: " + "; ".join(problems))
    _log(
        gate_dir,
        "env",
        {
            "interpreter": exe,
            "python": pyver,
            "pins": versions,
            "note": "ADDENDUM-4 §D.1 — fail-closed exact pins",
        },
    )


def prolog_host(gate_dir: Path) -> None:
    """§D.2: hostname into the run-log (attempt 4 on the host of 1–3;
    a host change is a TL decision BEFORE the run, never silent)."""
    _log(
        gate_dir,
        "host",
        {
            "hostname": socket.gethostname(),
            "expectation": "the host of attempts 1-3 (core-51); "
            "cross-host ONNX FP determinism not guaranteed",
        },
    )


def prolog_activation(gate_dir: Path) -> str:
    """§D.4: the «ADDENDUM-4 ACTIVATED» event — the addendum must be
    ACTIVATED at HEAD and the activation commit must exist on HEAD's
    ancestry; the event is written HERE, strictly before gate_start."""
    addendum = REPO_ROOT / ADDENDUM4_PATH
    text = addendum.read_text(encoding="utf-8")
    status_line = next(
        (ln for ln in text.splitlines() if ln.startswith("> Статус:")), ""
    )
    if "ACTIVATED" not in status_line or "DRAFT" in status_line:
        raise InstrumentAbort(
            f"ADDENDUM-4 is not ACTIVATED at HEAD (status line: {status_line!r}); "
            "activation must be committed BEFORE gate_start"
        )
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "log",
            "--format=%H%x1f%s",
            "-n",
            "100",
            "--",
            ADDENDUM4_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    activation = None
    for line in proc.stdout.splitlines():
        sha, _, subject = line.partition("\x1f")
        if subject.startswith(ADDENDUM4_ACTIVATION_PREFIX):
            activation = {"commit": sha, "subject": subject}
            break
    if activation is None:
        raise InstrumentAbort(
            f"no activation commit found for {ADDENDUM4_PATH} "
            f"(message prefix {ADDENDUM4_ACTIVATION_PREFIX!r}); order assert fails"
        )
    _log(
        gate_dir,
        "ADDENDUM-4 ACTIVATED",
        {
            **activation,
            "status_line": status_line.strip(),
            "order": "this event is written BEFORE gate_start (runner-enforced)",
        },
    )
    return activation["commit"]


def prolog_tripwire(gate_dir: Path, run_a: Path, run_b: Path | None) -> dict:
    """§D.3: byte tripwire — boosters AND ONNX artifacts vs the §C pins,
    verified RUNNER-SIDE BEFORE any read; plus run-b booster byte-continuity
    (attempt-3 convention: booster.txt byte-identical across runs)."""
    out = {"boosters": {}, "onnx": {}, "run_b_booster_eq_a": None}
    for tag, pin in BOOSTER_SHA.items():
        path = run_a / f"model-{tag}" / "booster.txt"
        got = _sha256_file(path)
        out["boosters"][tag] = got
        if got != pin:
            raise InstrumentAbort(
                f"byte tripwire FAILED on booster {tag}: {got} != {pin}; "
                "the frozen model state moved — hard abort before any read"
            )
    for tag, pin in ONNX_SHA.items():
        path = run_a / f"model-{tag}.onnx"
        got = _sha256_file(path)
        out["onnx"][tag] = got
        if got != pin:
            raise InstrumentAbort(
                f"ONNX tripwire FAILED on {tag}: {got} != {pin} (attempt-3 "
                "manifest equality, §C) — hard abort before any read"
            )
    if run_b is not None:
        eq = {}
        for tag in BOOSTER_SHA:
            a = (run_a / f"model-{tag}" / "booster.txt").read_bytes()
            b = (run_b / f"model-{tag}" / "booster.txt").read_bytes()
            eq[tag] = a == b
            if a != b:
                raise InstrumentAbort(
                    f"booster byte-continuity broken: run-b {tag} differs from "
                    "run-a — contradicts the frozen state (attempt-3 determinism)"
                )
        out["run_b_booster_eq_a"] = eq
    _log(gate_dir, "byte_tripwire", out)
    return out


def prolog_frozen_inputs(
    gate_dir: Path,
    probes_path: Path,
    ledger_manifest: Path,
    ledger_run_log: Path,
    knn_sidecar: Path,
    corpus_dir: Path,
) -> dict:
    """§C/§D.5: count asserts — probes (sha + 32/24/16/8), quarantine ledger
    (manifest sha + 370 + §A apply-log cross-check), the pre-built kNN train
    sidecar (sha + meta), and the v4.5 train corpus fingerprint."""
    # probes
    probes_sha = _sha256_file(probes_path)
    if probes_sha != PROBES_SHA:
        raise InstrumentAbort(
            f"probes sha256 {probes_sha} != frozen {PROBES_SHA} — regeneration "
            "is FORBIDDEN (§C); abort class «инструмент»"
        )
    probe_rows = load_jsonl(probes_path)
    counts = {}
    for row in probe_rows:
        cls = str(row.get("probe_class"))
        if cls not in PROBE_COUNTS:
            raise InstrumentAbort(f"unknown probe class {cls!r} in the frozen set")
        counts[cls] = counts.get(cls, 0) + 1
    if counts != PROBE_COUNTS:
        raise InstrumentAbort(
            f"probe count assert FAILED: {counts} != {PROBE_COUNTS} (§C table); "
            "abort class «инструмент»"
        )
    ids = [str(r["pair_id"]) for r in probe_rows]
    if len(set(ids)) != len(ids):
        raise InstrumentAbort("duplicate pair_ids inside the frozen probe set")

    # quarantine ledger §A
    ledger_sha = _sha256_file(ledger_manifest)
    if ledger_sha != LEDGER_SHA:
        raise InstrumentAbort(
            f"ledger manifest sha256 {ledger_sha} != pinned {LEDGER_SHA}"
        )
    manifest = json.loads(ledger_manifest.read_text(encoding="utf-8"))
    node = manifest
    for key in LEDGER_FIELD:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            break
    if node != LEDGER_QUARANTINED:
        raise InstrumentAbort(
            f"ledger {'.'.join(LEDGER_FIELD)} = {node!r} != {LEDGER_QUARANTINED}; "
            "abort class «инструмент»"
        )
    apply_events = []
    for line in ledger_run_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "cross_language_quarantine":
            apply_events.append(entry)
    if (
        len(apply_events) != 1
        or int(apply_events[0].get("quarantined_pairs", -1)) != LEDGER_QUARANTINED
    ):
        raise InstrumentAbort(
            f"§A apply-log cross-check FAILED: {len(apply_events)} "
            "cross_language_quarantine event(s), expected exactly one with "
            f"quarantined_pairs == {LEDGER_QUARANTINED}"
        )

    # kNN train sidecar (pre-built, §E)
    sidecar_sha = _sha256_file(knn_sidecar)
    if sidecar_sha != KNN_SIDECAR_SHA:
        raise InstrumentAbort(
            f"kNN train sidecar sha256 {sidecar_sha} != pinned {KNN_SIDECAR_SHA}; "
            "rebuilds are NOT byte-identical by design — the pre-built sidecar "
            "is the frozen input (§E)"
        )

    # v4.5 train corpus fingerprint (§C)
    train_rows = load_jsonl(corpus_dir / "train.jsonl")
    fp = corpus_fingerprint(
        manifest_bytes((r["pair_id"], r["pair_sha256"]) for r in train_rows)
    )
    if fp != CORPUS_TRAIN_FP:
        raise InstrumentAbort(
            f"train corpus fingerprint {fp} != pinned {CORPUS_TRAIN_FP}"
        )

    out = {
        "probes": {"sha256": probes_sha, "counts": counts, "total": len(probe_rows)},
        "ledger": {
            "sha256": ledger_sha,
            "field": ".".join(LEDGER_FIELD),
            "value": node,
            "apply_log_events": len(apply_events),
        },
        "knn_sidecar": {"sha256": sidecar_sha},
        "train_fingerprint": fp,
    }
    _log(gate_dir, "frozen_inputs", out)
    return probe_rows


def _ensure_vesmaro_alias(engine_src: str) -> None:
    """Engine 6.4.0 renamed the package vesmaro -> vesma (engine commit
    5e0e4b4); the frozen builder still imports ``vesmaro.embeddings``.
    Alias the renamed home under the old name BEFORE make_embedder runs.
    Verbatim pattern of b2_g1b_probes.py / b2_knn_train_sidecar.py."""
    import importlib
    import types

    try:
        importlib.import_module("vesmaro.embeddings")
        return
    except ImportError:
        pass
    if engine_src not in sys.path:
        sys.path.insert(0, engine_src)
    vesmaro = types.ModuleType("vesmaro")
    vesmaro.__path__ = [str(Path(engine_src) / "vesma")]
    sys.modules["vesmaro"] = vesmaro
    emb = importlib.import_module("vesma.embeddings")
    sys.modules["vesmaro.embeddings"] = emb
    vesmaro.embeddings = emb


# ════════════════════════════ kNN CHARACTERIZATION (§E) ═══════════════════════


def _knn_dist(
    matrix: np.ndarray, query: np.ndarray, k: int, exclude: int | None = None
) -> float:
    """Distance for a query side against the train manifold.

    matrix/query are L2-normalized float64; the distance is the cosine
    distance to the MEAN of the k nearest neighbors (self excluded for the
    LOO null). Deterministic (full sort — no RNG, stable tie handling).
    The MLC Р5 "one code path" pin is on the QUANTILES: 0.99 and 0.01 are
    read from the ONE train-to-train null distribution computed once in
    knn_characterization; this function computes the probe-side distances
    with the same formula (1 − mean of the top-5 cosine similarities)."""
    sims = matrix @ query
    if exclude is not None:
        sims[exclude] = -np.inf
    top = np.sort(sims)[-k:]
    return 1.0 - float(top.mean())


def knn_characterization(
    gate_dir: Path, probe_rows: list[dict], knn_sidecar: Path
) -> dict:
    """§E.2: ONE computational procedure, two pre-declared interpretations.

    - manifold: BOTH sides of every v4.5 train row, frozen embedder
      (pin-asserted live), deduplicated by the sha of the truncated embed
      text — the pre-built sha-pinned sidecar (prolog);
    - null: train-to-train leave-one-out, k=5, cosine, same code path;
    - quantiles 0.99 and 0.01 of the null (method fixed, published);
    - probes: both sides embedded, probe distance = max (worst side), the
      full 80-vector + both per-side vectors + raw counters → run-log;
    - off-manifold ≥ 8/80 → REPORT branch (never an abort); near-tail ≥ 1
      → the «запоминание исключить нельзя» qualifier on the +0.021 margin
      disclosure. No RNG anywhere."""
    from b2_graph_corpus_v45 import ENGINE_SRC, make_embedder

    _ensure_vesmaro_alias(ENGINE_SRC)

    npz = np.load(knn_sidecar, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    vecs = npz["vecs"].astype(np.float64)  # storage float32 → compute float64
    n = int(vecs.shape[0])
    if meta.get("manifold_points") != n or meta.get("embedder_dim") != vecs.shape[1]:
        raise InstrumentAbort("kNN sidecar meta/shape mismatch")
    if meta.get("train_fingerprint") != CORPUS_TRAIN_FP:
        raise InstrumentAbort("kNN sidecar meta train fingerprint mismatch")
    embed_side, embed_text, pin = make_embedder()
    if pin != meta.get("embedder_pin"):
        raise InstrumentAbort(
            f"live embedder pin {pin!r} != sidecar pin {meta.get('embedder_pin')!r}"
        )
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise InstrumentAbort(
            f"sidecar vectors not L2-normalized ({norms.min():.6f}..{norms.max():.6f})"
        )

    # ── null: train-to-train leave-one-out, chunked, deterministic ────────
    t0 = time.monotonic()
    null = np.empty(n, dtype=np.float64)
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = vecs[start:stop] @ vecs.T  # (chunk, n)
        for j in range(stop - start):
            block[j, start + j] = -np.inf  # LOO: self-neighbor OUT
        null[start:stop] = 1.0 - np.sort(block, axis=1)[:, -KNN_K:].mean(axis=1)
    q_hi = float(np.quantile(null, KNN_Q_HI, method="linear"))
    q_lo = float(np.quantile(null, KNN_Q_LO, method="linear"))

    # ── probes: both sides, one distance per side, max-reduction ─────────
    per_probe, per_side_flat = [], []
    for row in probe_rows:
        side_d = []
        for side in (row["record"], row["candidate"]):
            text = (
                f"{side.get('title') or ''}\n{side.get('body') or ''}\n"
                f"{' '.join(side.get('tags') or [])}"
            )[:4096]
            v = np.asarray(embed_text(text), dtype=np.float64)
            if v.shape != (vecs.shape[1],):
                raise InstrumentAbort(f"probe embed dim {v.shape} != {vecs.shape[1]}")
            side_d.append(_knn_dist(vecs, v, KNN_K))
        per_side_flat.append(side_d)
        per_probe.append(
            {
                "pair_id": str(row["pair_id"]),
                "probe_class": str(row["probe_class"]),
                "d_side_record": side_d[0],
                "d_side_candidate": side_d[1],
                "d_probe": max(side_d),
            }
        )
    off_manifold = [p for p in per_probe if p["d_probe"] > q_hi]
    near_tail = [p for p in per_probe if p["d_probe"] < q_lo]
    per_class = {}
    for cls in PROBE_COUNTS:
        ds = [p["d_probe"] for p in per_probe if p["probe_class"] == cls]
        per_class[cls] = {
            "n": len(ds),
            "off_manifold": sum(
                1 for p in per_probe if p["probe_class"] == cls and p["d_probe"] > q_hi
            ),
            "near_tail": sum(
                1 for p in per_probe if p["probe_class"] == cls and p["d_probe"] < q_lo
            ),
            "d_min": round(min(ds), 6),
            "d_max": round(max(ds), 6),
            "d_mean": round(sum(ds) / len(ds), 6),
        }
    out = {
        "scope": "ADDENDUM-4 §E.2 — ONE procedure, two pre-declared "
        "interpretations; NON-gating (report branches only)",
        "sidecar_sha256": KNN_SIDECAR_SHA,
        "sidecar_points": n,
        "embedder_pin": pin,
        "metric": KNN_METRIC,
        "k": KNN_K,
        "loo_null": {"n": n, "self_neighbor": "excluded (leave-one-out)"},
        "quantile_method": KNN_QUANTILE_METHOD,
        "q099_train_to_train": round(q_hi, 6),
        "q001_train_to_train": round(q_lo, 6),
        "probe_reduction": KNN_PROBE_REDUCTION,
        "probes": len(per_probe),
        "off_manifold": {
            "threshold": f"> q099, paired with the {KNN_OFFMANIFOLD_MIN}/80 "
            "report branch",
            "count": len(off_manifold),
            "pair_ids": [p["pair_id"] for p in off_manifold],
            "report_branch": len(off_manifold) >= KNN_OFFMANIFOLD_MIN,
        },
        "near_tail": {
            "threshold": "< q001",
            "count": len(near_tail),
            "pair_ids": [p["pair_id"] for p in near_tail],
            "qualifier": "запоминание исключить нельзя" if near_tail else None,
        },
        "per_class": per_class,
        "d80_vector": [{"pair_id": p["pair_id"], "d": p["d_probe"]} for p in per_probe],
        "d160_side_vector": per_side_flat,
        "null_dists_sha_prefix": hashlib.sha256(
            np.asarray(null, dtype=np.float64).tobytes()
        ).hexdigest()[:16],
        "wall_s": round(time.monotonic() - t0, 1),
    }
    (gate_dir / "knn-manifold.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(
        gate_dir,
        "knn_manifold",
        {
            "sidecar_sha256": KNN_SIDECAR_SHA,
            "points": n,
            "k": KNN_K,
            "q099": round(q_hi, 6),
            "q001": round(q_lo, 6),
            "off_manifold": out["off_manifold"]["count"],
            "off_manifold_ids": out["off_manifold"]["pair_ids"],
            "near_tail": out["near_tail"]["count"],
            "near_tail_ids": out["near_tail"]["pair_ids"],
            "report_branch_off_manifold": out["off_manifold"]["report_branch"],
            "d80": out["d80_vector"],
            "per_class": per_class,
        },
    )
    return out


# ══════════════════ DETERMINISM vs ATTEMPT 3 (§D.6, vector assert) ═══════════


def assert_determinism_vs_attempt3(
    gate_dir: Path, ref_dir: Path, per_probe_by_model: dict[str, dict]
) -> dict:
    """160 printed-precision string comparisons (%.4f, both sides) against
    the sha-pinned attempt-3 references. ANY divergence = instrument-class
    abort (a defect of the tool, never a gate). Aggregates are derivatives —
    NOT asserted (double counting removed by the ratified challenge)."""
    refs = {"full": ref_dir / "g1b-full.json", "ablated": ref_dir / "g1b-ablated.json"}
    ref_pins = {"full": REF_FULL_SHA, "ablated": REF_ABLATED_SHA}
    checked = 0
    mismatches = []
    for model, ref_path in refs.items():
        got_sha = _sha256_file(ref_path)
        if got_sha != ref_pins[model]:
            raise InstrumentAbort(
                f"reference {ref_path.name} sha256 {got_sha} != pinned "
                f"{ref_pins[model]} — the determinism baseline moved"
            )
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        mine = per_probe_by_model[model]
        for cls, entries in ref["per_probe"].items():
            mine_entries = mine.get(cls, [])
            if len(entries) != len(mine_entries):
                raise InstrumentAbort(
                    f"determinism vector length mismatch {model}/{cls}: "
                    f"{len(mine_entries)} vs {len(entries)}"
                )
            for e3, e4 in zip(entries, mine_entries):
                checked += 1
                if (
                    e3["pair_id"] != e4["pair_id"]
                    or f"{e3['p']:.4f}" != f"{e4['p']:.4f}"
                ):
                    mismatches.append(
                        {
                            "model": model,
                            "class": cls,
                            "pair_id_3": e3["pair_id"],
                            "pair_id_4": e4["pair_id"],
                            "p3": f"{e3['p']:.4f}",
                            "p4": f"{e4['p']:.4f}",
                        }
                    )
    if mismatches:
        raise InstrumentAbort(
            f"determinism vs attempt 3 FAILED on {len(mismatches)}/160 values "
            f"(e.g. {mismatches[:4]}) — abort class «инструмент»"
        )
    out = {
        "values_compared": checked,
        "mismatches": 0,
        "precision": "%.4f string comparison (both sides)",
        "references": {
            m: {"path": str(p), "sha256": ref_pins[m]} for m, p in refs.items()
        },
        "pass": True,
    }
    _log(gate_dir, "determinism_vs_attempt3", out)
    return out


# ════════════════════════════════════ PHASE 1 ════════════════════════════════


def phase1_pregates(
    gate_dir: Path, run_a: Path, args, probe_rows: list[dict]
) -> tuple[list[dict], bool, dict]:
    """g1 + g1b (reduced) + determinism + characterization (§E) + g3 —
    sealed artifacts untouched."""
    gates: list[dict] = []
    pre_out: dict = {}
    from cortex.eval.sanity import run_sanity_suite

    # ── g1: sanity v2, 9/9 on BOTH models under ADDENDUM-3 A3.1 ──────────
    g1_reports = {}
    g1_ok = True
    for onnx in (run_a / "model-full.onnx", run_a / "model-ablated.onnx"):
        report = run_sanity_suite(onnx)
        a31 = _a31_for_bundle(onnx)
        mono = next(c for c in report.checks if c.name == "monotonicity")
        _a31_crosscheck(mono.detail, a31["raw_ladder"])
        others_ok = all(c.passed for c in report.checks if c.name != "monotonicity")
        model_ok = others_ok and len(report.checks) == 9 and a31["pass"]
        payload = {
            "bundle": str(onnx),
            "weights_sha256": report.weights_sha256,
            "passed": report.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in report.checks
            ],
            "a3_1": {
                "rule": "ADDENDUM-3 A3.1 (ACTIVATED 23.10): controlled step "
                "tol = 1e-4 iff BOTH points >= 0.99, else 1e-6; steps "
                "starting strictly above cos=0.95 uncontrolled (zoned "
                "v2); implemented RUNNER-SIDE — sanity.py untouched",
                "production_suite_passed": report.passed,
                "raw_ladder": a31["raw_ladder"],
                "steps": a31["steps"],
                "pass_under_a3_1": a31["pass"],
                "model_pass_under_a3_1": model_ok,
            },
        }
        g1_reports[onnx.stem] = payload
        (gate_dir / f"sanity-{onnx.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        n_pass = sum(c["passed"] for c in payload["checks"])
        _log(
            gate_dir,
            "sanity",
            {
                "bundle": onnx.name,
                "production": report.passed,
                "under_a3_1": model_ok,
                "n_pass": n_pass,
                "n_checks": len(payload["checks"]),
                "failed": [c.name for c in report.failed],
                "raw_ladder": a31["raw_ladder"],
                "steps": a31["steps"],
            },
        )
        g1_ok = g1_ok and model_ok
    gates.append(
        {
            "gate": "g1",
            "what": "sanity suite v2, BOTH models (A3.1 plateau rule)",
            "required": "9/9 on full AND ablated under ADDENDUM-3 A3.1 "
            "(raw ladders + per-step tolerances in sanity-*.json)",
            "measured": {
                k: {
                    "checks_production_suite": f"{sum(c['passed'] for c in v['checks'])}/{len(v['checks'])}",
                    "under_a3_1": "9/9"
                    if v["a3_1"]["pass_under_a3_1"]
                    and all(
                        c["passed"] for c in v["checks"] if c["name"] != "monotonicity"
                    )
                    else "RED",
                    "raw_ladder": v["a3_1"]["raw_ladder"],
                    "step_tolerances": {
                        f"{s['from_cos']}->{s['to_cos']}": (
                            s["tol"] if s["controlled"] else "uncontrolled"
                        )
                        for s in v["a3_1"]["steps"]
                    },
                }
                for k, v in g1_reports.items()
            },
            "pass": g1_ok,
        }
    )
    if not g1_ok:
        return gates, False, pre_out

    # ── g1b: REDUCED gating composition (ADDENDUM-4 §B), BOTH models; ────
    #    xlang legs computed + published (characterization, non-gating);
    #    determinism assert vs attempt 3 BEFORE the gate evaluation.
    scorers = {
        tag: OnnxScorer(run_a / f"model-{tag}.onnx") for tag in ("full", "ablated")
    }
    per_model, per_probe_by_model = {}, {}
    g1b_ok = True
    for tag, scorer in scorers.items():
        gated = tag == "full"
        per_probe = {}
        for row in probe_rows:
            base = features(
                _row_record(row["record"]),
                _row_record(row["candidate"]),
                float(row["similarity"]),
            )
            vector = (
                FeatureVector(
                    GATED_NAMES,
                    base.values + (0.0, 0.0, 0.0, 0.0, 0.0),
                )
                if gated
                else base
            )
            p = float(scorer.predict([vector])[0])
            per_probe.setdefault(row["probe_class"], []).append(
                {"pair_id": row["pair_id"], "p": p, "cos": row["similarity"]}
            )
        per_probe_by_model[tag] = per_probe
        cls_report = {}
        model_ok = True
        for cls, (op, thr) in G1B_GATING.items():
            ps = [x["p"] for x in per_probe.get(cls, [])]
            if not ps:
                raise InstrumentAbort(f"g1b class {cls}: no probes")
            stat = min(ps) if op == ">=" else max(ps)
            ok = stat >= thr if op == ">=" else stat < thr
            cls_report[cls] = {
                "op": op,
                "threshold": thr,
                "n": len(ps),
                "min": round(min(ps), 6),
                "max": round(max(ps), 6),
                "mean": round(sum(ps) / len(ps), 6),
                "binding": round(stat, 6),
                "pass": bool(ok),
                "gating": True,
            }
            model_ok = model_ok and ok
        # characterized legs: published, never gated (§B)
        char_report = {}
        for cls in G1B_CHARACTERIZED:
            ps = [x["p"] for x in per_probe.get(cls, [])]
            if not ps:
                raise InstrumentAbort(f"g1b class {cls}: no probes")
            char_report[cls] = {
                "n": len(ps),
                "min": round(min(ps), 6),
                "max": round(max(ps), 6),
                "mean": round(sum(ps) / len(ps), 6),
                "n_p_ge_cut": sum(1 for p in ps if p >= CUT),
                "n_p_lt_cut": sum(1 for p in ps if p < CUT),
                "gating": False,
            }
        per_model[tag] = {
            "gating": cls_report,
            "characterized": char_report,
            "pass": model_ok,
        }
        (gate_dir / f"g1b-{tag}.json").write_text(
            json.dumps(
                {
                    "model": tag,
                    "gating": cls_report,
                    "characterized": char_report,
                    "per_probe": per_probe,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _log(
            gate_dir,
            "g1b",
            {
                "model": tag,
                "pass": model_ok,
                **{k: v["binding"] for k, v in cls_report.items()},
                **{
                    f"{k}_char": {"min": v["min"], "max": v["max"]}
                    for k, v in char_report.items()
                },
            },
        )
        g1b_ok = g1b_ok and model_ok

    # determinism vs attempt 3 (BEFORE the verdict logic — a divergence is
    # an instrument defect, not a red leg)
    pre_out["determinism"] = assert_determinism_vs_attempt3(
        gate_dir, Path(args.ref_dir), per_probe_by_model
    )
    # xlang characterization rides out for the report (both models), also on
    # the abort path (the brief: full numbers in every gate-report)
    pre_out["xlang"] = {t: m["characterized"] for t, m in per_model.items()}

    gates.append(
        {
            "gate": "g1b",
            "what": "flora of unseen scaffolds — REDUCED gating composition "
            "(ADDENDUM-4 §B: paradup + factedit gate; xlang legs "
            "characterized, non-gating)",
            "required": "; ".join(
                f"{c}: P(dup) {op} {t}" for c, (op, t) in G1B_GATING.items()
            ),
            "measured": {
                t: {
                    "gating": {c: v["binding"] for c, v in m["gating"].items()},
                    "characterized": {
                        c: {
                            "min": v["min"],
                            "max": v["max"],
                            "n_p_ge_cut": v["n_p_ge_cut"],
                            "n_p_lt_cut": v["n_p_lt_cut"],
                        }
                        for c, v in m["characterized"].items()
                    },
                }
                for t, m in per_model.items()
            },
            "pass": g1b_ok,
        }
    )
    if not g1b_ok:
        return gates, False, pre_out

    # ── §E.2: kNN manifold characterization (non-gating, pre-single-shot) ─
    pre_out["knn"] = knn_characterization(gate_dir, probe_rows, Path(args.knn_sidecar))

    # ── g3: no-harm corridor on v42-276 (fresh sidecar, store read-only) ──
    v42_dir = Path(args.v42_dir)
    sc_path = fresh_sidecar(
        gate_dir, "v42-276", v42_dir / "holdout.jsonl", args.store_uri
    )
    from cortex.features.graph import GraphSidecar

    side = GraphSidecar(sc_path)
    from b2_rev1n_lib import build_dataset

    v42_rows = load_jsonl(v42_dir / "holdout.jsonl")
    vecs_f, labels_f, prov_f = build_dataset(v42_rows, side, GATED_NAMES)
    probs_f = scorers["full"].predict(vecs_f)
    m_full = _metrics(labels_f, probs_f)
    ci95 = CI95_Z * float(np.sqrt(PREV_BA * (1.0 - PREV_BA) / PREV_N))
    gates.append(
        {
            "gate": "g3",
            "what": "no-harm corridor on v42-276 (full model)",
            "required": f"BA >= {G3_FLOOR} (frozen; = {PREV_BA} - max(0.02, CI95), "
            f"CI95={ci95:.4f} informational)",
            "measured": {
                "ba": round(m_full["ba"], 4),
                "ci95_informational": round(ci95, 4),
                "sidecar_fingerprint": json.loads(
                    (gate_dir / "graph-sidecar-fresh-v42-276.coverage.json").read_text()
                ).get("sidecar_fingerprint"),
            },
            "pass": m_full["ba"] >= G3_FLOOR,
        }
    )
    _log(gate_dir, "g3", {"ba": m_full["ba"], "floor": G3_FLOOR})
    # g3 red MUST abort phase 1 like g1/g1b (ADDENDUM-4 §B: red pre-gate =
    # ABORT before the sealed read). Found by TL read: this return was
    # unconditional True, so a red g3 flowed into single-shot prep.
    pregates_ok = g1_ok and g1b_ok
    return gates, pregates_ok and (m_full["ba"] >= G3_FLOOR), pre_out


# ════════════════════════════════════ MAIN ═══════════════════════════════════


def _write_abort(gate_dir: Path, t0: float, verdict: str, extra: dict) -> None:
    report = {
        "verdict": verdict,
        "sealed_artifacts": "NOT READ",
        "wall_s": round(time.monotonic() - t0, 1),
        **extra,
    }
    (gate_dir / "gate-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    t0 = time.monotonic()
    parser = argparse.ArgumentParser(prog="b2-gate-rev1n1")
    parser.add_argument("--corpus-dir", default="data/b2-graph-corpus/v45")
    parser.add_argument(
        "--v42-dir",
        default="/var/home/abyss/LABs/Projects/Project-Vesma"
        "/vesma-cortex/data/stage2/dataset-v4-holdout",
    )
    parser.add_argument(
        "--run-a",
        default="data/b2-graph-corpus/v45/attempt-3/run-rev1n1-a",
        help="attempt-4 scores through the byte-stable "
        "attempt-3 artifacts (no training, ADDENDUM-4 §C)",
    )
    parser.add_argument(
        "--run-b",
        default="data/b2-graph-corpus/v45/attempt-3/run-rev1n1-b",
        help="booster byte-continuity assert only (run-b ONNX "
        "differs by the trained_at stamp by convention)",
    )
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--store-uri", default=STORE_URI_DEFAULT)
    parser.add_argument(
        "--probes", default="datasets/b2-graph/v45-1n1-attempt3/g1b-probes.jsonl"
    )
    parser.add_argument("--ref-dir", default="datasets/b2-graph/v45-1n1-attempt3")
    parser.add_argument(
        "--ledger-manifest", default="data/b2-graph-corpus/v44-r2/manifest.json"
    )
    parser.add_argument(
        "--ledger-run-log", default="data/b2-graph-corpus/v44-r2/run-log.jsonl"
    )
    parser.add_argument(
        "--knn-sidecar",
        default="data/b2-graph-corpus/v45/attempt-4/knn-train-sidecar.npz",
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    run_a, run_b = Path(args.run_a), (Path(args.run_b) if args.run_b else None)
    gate_dir = Path(args.gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = corpus_dir / "holdout.jsonl"
    g4spec_path = corpus_dir / "g4-spec-surface.jsonl"
    train_path = corpus_dir / "train.jsonl"

    # ── run-log resume guard: single-shot must never consume the seals twice;
    #    honest appends after an infra crash before the read are allowed. ──
    log_path = gate_dir / "gate-run-log.jsonl"
    if log_path.exists():
        prior_events = [
            json.loads(ln)["event"]
            for ln in log_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if "sealed_read_once" in prior_events or "gate_done" in prior_events:
            print(
                "REFUSE: the run-log already carries sealed_read_once/gate_done "
                "— single-shot already consumed; a re-run would be a re-roll",
                file=sys.stderr,
            )
            return 4
        _log(
            gate_dir,
            "run_log_resume",
            {
                "mode": "append",
                "prior_events": len(prior_events),
                "note": "honest append-only continuation after an infra stop "
                "BEFORE the sealed read",
            },
        )

    # ══ PROLOG (ADDENDUM-4 §D) — enforced order, BEFORE any eval read ═════
    try:
        prolog_swap_check(gate_dir)
        prolog_env(gate_dir)
        prolog_host(gate_dir)
        tripwire = prolog_tripwire(gate_dir, run_a, run_b)
        activation_commit = prolog_activation(gate_dir)
        probe_rows = prolog_frozen_inputs(
            gate_dir,
            Path(args.probes),
            Path(args.ledger_manifest),
            Path(args.ledger_run_log),
            Path(args.knn_sidecar),
            corpus_dir,
        )
    except InstrumentAbort as exc:
        _log(gate_dir, "abort_instrument", {"reason": str(exc), "stage": "prolog"})
        _write_abort(gate_dir, t0, f"ABORT — instrument: {exc}", {"stage": "prolog"})
        print(f"INSTRUMENT ABORT: {exc}", file=sys.stderr)
        return 3

    _log(
        gate_dir,
        "gate_start",
        {
            "revision": "1.n+1",
            "attempt": 4,
            "addendum": "ADDENDUM-4 (A2.6 revision) ACTIVATED — see the "
            "ADDENDUM-4 ACTIVATED event above",
            "activation_commit": activation_commit,
            "corpus_dir": str(corpus_dir),
            "sealed_holdout": HOLDOUT_SEAL[:16],
            "sealed_g4spec": G4SPEC_SEAL[:16],
            "store_uri": args.store_uri,
            "training": "NONE — controlled reproduction on byte-stable boosters",
        },
    )

    # ══ PHASE 1: pre-gates (sealed artifacts untouched) ════════════════════
    try:
        gates, ok, pre_out = phase1_pregates(gate_dir, run_a, args, probe_rows)
    except InstrumentAbort as exc:
        _log(gate_dir, "abort_instrument", {"reason": str(exc), "stage": "phase1"})
        _write_abort(gate_dir, t0, f"ABORT — instrument: {exc}", {"stage": "phase1"})
        print(f"INSTRUMENT ABORT: {exc}", file=sys.stderr)
        return 3
    if not ok:
        report = {
            "verdict": "ABORT — pre-gate red: sealed artifacts not read",
            "gates": gates,
            "addendum4_prolog": {
                "activation_commit": activation_commit,
                "tripwire": tripwire,
            },
            "characterization_pre_gate": pre_out,
            "wall_s": round(time.monotonic() - t0, 1),
        }
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log(gate_dir, "abort_pregate_red", {"wall_s": report["wall_s"]})
        return 2

    # ══ PHASE 2: SINGLE-SHOT ═══════════════════════════════════════════════
    # ── seals FIRST (fingerprint-level, before any eval read) ─────────────
    seal_hold = _seal_check(holdout_path, HOLDOUT_SEAL, HOLDOUT_IDS_SHA, HOLDOUT_ROWS)
    seal_g4 = _seal_check(g4spec_path, G4SPEC_SEAL, G4SPEC_IDS_SHA, G4SPEC_ROWS)
    _log(gate_dir, "seals", {"holdout_ok": seal_hold["ok"], "g4spec_ok": seal_g4["ok"]})
    if not (seal_hold["ok"] and seal_g4["ok"]):
        report = {
            "verdict": "ABORT — seal verification failed; no eval read",
            "seals": {"holdout": seal_hold, "g4spec": seal_g4},
            "wall_s": round(time.monotonic() - t0, 1),
        }
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 2
    gates.append(
        {
            "gate": "seals",
            "what": "sealed artifacts fingerprints re-derived",
            "required": "holdout v4.5 = 9eae5ec1…, g4-spec-v1 = af167dc4…, row counts",
            "measured": {
                "holdout_rows": seal_hold["rows"],
                "g4spec_rows": seal_g4["rows"],
                "both_match": True,
            },
            "pass": True,
        }
    )

    # ── corpus verifier (builder-independent, corpus-stage instrument) ────
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "b2_graph_corpus_verify_v45.py"),
            "--dir",
            str(corpus_dir),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (gate_dir / "corpus-verify.txt").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )
    _log(
        gate_dir,
        "corpus_verify",
        {
            "exit": proc.returncode,
            "line": next(
                (ln for ln in proc.stdout.splitlines() if "OK" in ln or "FAIL" in ln),
                "",
            ),
        },
    )
    if proc.returncode != 0:
        report = {
            "verdict": "ABORT — corpus verifier red; no eval read",
            "gates": gates,
            "wall_s": round(time.monotonic() - t0, 1),
        }
        (gate_dir / "gate-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
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
                    fresh.get("a_store_id")
                    or fresh.get("b_store_id")
                    or fresh.get("edge_ab")
                    or int(fresh.get("common_neighbors") or 0) > 0
                ):
                    raise RuntimeError(
                        f"{surface} {pid}: synthetic row resolved with "
                        "evidence/identity — by-construction violation"
                    )
                checked += 1
                continue
            if (
                fresh is None
                or not fresh.get("a_store_id")
                or not fresh.get("b_store_id")
            ):
                quarantine += 1
                continue
            if int(fresh.get("edge_ab") or 0) != int(row.get("edge_ab") or 0) or int(
                fresh.get("common_neighbors") or 0
            ) != int(row.get("common_neighbors") or 0):
                mismatches.append(pid)
                continue
            checked += 1
        if mismatches:
            raise RuntimeError(
                f"{surface}: fresh sidecar disagrees with frozen evidence columns "
                f"on {len(mismatches)} rows (e.g. {mismatches[:5]}) — A2.5.3 abort"
            )
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
            if (
                fresh is None
                or not fresh.get("a_store_id")
                or not fresh.get("b_store_id")
            ):
                out.add(str(row["pair_id"]))
        return out

    q_hold = quarantined_ids(hold_rows, raw_hold)
    q_g4 = quarantined_ids(g4_rows, raw_g4)
    hold_rows_eval = [r for r in hold_rows if str(r["pair_id"]) not in q_hold]
    g4_rows_eval = [r for r in g4_rows if str(r["pair_id"]) not in q_g4]

    # ── THE read (once, both surfaces, both models) ───────────────────────
    scorers = {
        tag: OnnxScorer(run_a / f"model-{tag}.onnx") for tag in ("full", "ablated")
    }

    def read_surface(rows, cos_map, tag):
        gated = tag == "full"
        return build_rows_dataset(rows, cos_map, gated)

    surface_preds = {}
    for surface, rows, cos_map in (
        ("holdout", hold_rows_eval, hold_cos),
        ("g4spec", g4_rows_eval, frozen_g4_side),
    ):
        for tag in ("full", "ablated"):
            vecs, labels, prov = read_surface(rows, cos_map, tag)
            probs = scorers[tag].predict(vecs)
            surface_preds[(surface, tag)] = {
                "vectors": vecs,
                "labels": labels,
                "probs": probs,
                "prov": prov,
            }
    _log(
        gate_dir,
        "sealed_read_once",
        {
            "holdout_rows": len(hold_rows_eval),
            "g4spec_rows": len(g4_rows_eval),
            "quarantine_holdout": len(q_hold),
            "quarantine_g4spec": len(q_g4),
        },
    )

    # ── determinism: DOUBLE SCORING PASS over the already-read in-memory ──
    #    data (ADDENDUM-4: the attempt-2 train double-run is NOT executed —
    #    attempt 4 trains nothing). Predictions must be byte-identical.
    double_pass = {}
    for surface, rows, _cos_map in (
        ("holdout", hold_rows_eval, None),
        ("g4spec", g4_rows_eval, None),
    ):
        for tag in ("full", "ablated"):
            pred = surface_preds[(surface, tag)]
            probs2 = scorers[tag].predict(pred["vectors"])
            identical = bool(
                np.array_equal(pred["probs"].view(np.uint8), probs2.view(np.uint8))
            )
            double_pass[f"{surface}/{tag}"] = identical
            if not identical:
                raise RuntimeError(
                    f"determinism broken: double scoring pass differs on "
                    f"{surface}/{tag} — single-read contract violated; STOP"
                )
    _log(gate_dir, "double_scoring_pass", {"byte_identical": double_pass})

    hf = surface_preds[("holdout", "full")]
    m_hold_full = _metrics(hf["labels"], hf["probs"])
    ha = surface_preds[("holdout", "ablated")]
    m_hold_abl = _metrics(ha["labels"], ha["probs"])
    gates.append(
        {
            "gate": "g2",
            "what": "sealed holdout v4.5 (85; 19 dup / 66 not-dup), full",
            "required": "BA>=0.93, sens>=0.95, spec>=0.90, Brier<=0.05",
            "measured": {
                k: round(m_hold_full[k], 4) for k in ("ba", "sens", "spec", "brier")
            },
            "pass": (
                m_hold_full["ba"] >= G2["ba"]
                and m_hold_full["sens"] >= G2["sens"]
                and m_hold_full["spec"] >= G2["spec"]
                and m_hold_full["brier"] <= G2["brier"]
            ),
        }
    )

    # per-stratum holdout table (both models)
    def strata_table(pred):
        out = {}
        for key in sorted({p["stratum"] for p in pred["prov"]}):
            idx = [i for i, p in enumerate(pred["prov"]) if p["stratum"] == key]
            out[key] = _metrics(
                [pred["labels"][i] for i in idx], [pred["probs"][i] for i in idx]
            )
        return out

    strata_full = strata_table(hf)
    strata_abl = strata_table(ha)

    # ── g4': recall(full, strata A) = 1.0 ∧ spec(full, g4-spec-v1) ≥ 0.90 ──
    a_idx = [i for i, p in enumerate(hf["prov"]) if p["stratum"] == "A-resave"]
    a_rows = [
        {
            "pair_id": hf["prov"][i]["pair_id"],
            "cluster_key": hf["prov"][i]["cluster_key"],
            "p": float(hf["probs"][i]),
            "pred_dup": bool(hf["probs"][i] >= CUT),
        }
        for i in a_idx
    ]
    a_recall = (
        (sum(1 for r in a_rows if r["pred_dup"]) / len(a_rows)) if a_rows else 0.0
    )
    clusters = {}
    for r in a_rows:
        clusters.setdefault(r["cluster_key"] or r["pair_id"], []).append(r["pred_dup"])
    gf = surface_preds[("g4spec", "full")]
    m_g4_full = _metrics(gf["labels"], gf["probs"])
    ga = surface_preds[("g4spec", "ablated")]
    m_g4_abl = _metrics(ga["labels"], ga["probs"])
    g4_pass = a_rows and a_recall == 1.0 and m_g4_full["spec"] >= G4SPEC_MIN
    gates.append(
        {
            "gate": "g4'",
            "what": "re-saves recall + evidence-hard-negative specificity",
            "required": "recall(full, strata A holdout) = 1.0 ∧ spec(full, g4-spec-v1) >= 0.90",
            "measured": {
                "recall_strata_a": round(a_recall, 4),
                "strata_a_rows": len(a_rows),
                "strata_a_clusters": {k: all(v) for k, v in clusters.items()},
                "spec_g4spec_v1": round(m_g4_full["spec"], 4),
            },
            "pass": bool(g4_pass),
        }
    )

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
    sha_sets = {
        k: {str(r["pair_sha256"]) for r in rs if r.get("pair_sha256")}
        for k, rs in (
            ("train-v45", train_rows),
            ("holdout-v45", hold_rows),
            ("g4-spec-v1", g4_rows),
        )
    }
    sha_inter = {
        f"{a}∩{b}": len(sha_sets[a] & sha_sets[b])
        for i, a in enumerate(("train-v45", "holdout-v45", "g4-spec-v1"))
        for b in ("train-v45", "holdout-v45", "g4-spec-v1")[i + 1 :]
        if sha_sets[a] & sha_sets[b]
    }
    gates.append(
        {
            "gate": "g5",
            "what": "pair-level disjointness re-assert, all eval surfaces",
            "required": "0 intersections (pair ids; pair hashes where frozen)",
            "measured": {
                "pair_id_intersections": inter or "none",
                "pair_sha_intersections": sha_inter or "none",
            },
            "pass": not inter and not sha_inter,
        }
    )

    # ── g6: anti-collapse (evidence-zero holdout negatives vs surface) ────
    gates.append(
        {
            "gate": "g6",
            "what": "spec(full, holdout evidence-zero negatives) vs spec(full, g4-spec-v1)",
            "required": f"spec(holdout not-dup) >= spec(g4-spec-v1) - {G6_MIN_MARGIN}",
            "measured": {
                "spec_holdout_notdup": round(m_hold_full["spec"], 4),
                "spec_g4spec_v1": round(m_g4_full["spec"], 4),
            },
            "pass": m_hold_full["spec"] >= m_g4_full["spec"] - G6_MIN_MARGIN,
        }
    )

    # ── A2.8 characterization (NON-gating, pre-declared scope) ────────────
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(run_a / "model-full" / "booster.txt"))
    feat_names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain = float(gain.sum()) or 1.0
    importances = {
        n: {
            "gain": float(g),
            "gain_share": round(float(g) / total_gain, 6),
            "splits": int(s),
        }
        for n, g, s in zip(feat_names, gain, split)
    }
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
        json.dumps(characterization, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(
        gate_dir,
        "a2.8_characterization",
        {
            "graph_gain_share": graph_gain_share,
            "delta_spec": round(delta_surface, 4),
            "dead_core": dead_core,
        },
    )

    # ── §D alignment report-only counters (recomputed from the frozen ─────
    #    probes file — disclosed in §F item 4, never asserted)
    sd_alignment = {}
    for cls in ("paradup", "factedit"):
        flags = [
            bool(r.get("shadow", {}).get("in_range"))
            for r in probe_rows
            if r["probe_class"] == cls
        ]
        sd_alignment[cls] = {"in_range": sum(flags), "n": len(flags)}

    # ── ADDENDUM-4 §F disclosures (canonical text lives in gate-report) ───
    knn = pre_out["knn"]
    margin_qualifier = ""
    if knn["near_tail"]["count"] >= 1:
        margin_qualifier = (
            f" (квалификатор §E: запоминание исключить нельзя — "
            f"near-tail-счётчик {knn['near_tail']['count']} ≥ 1)"
        )
    disclosures = {
        "1_contract_limit": "Предел контракта: 13-фичевый контракт не несёт "
        "слот-семантики cross-lang (lang_match — бинарный флаг); "
        "материально подтверждён пре-регистрацией попытки 3 (dup 16/16 "
        "CONFIRMED 0.5208 = запас +0.021; sib 1/8, binding 0.9889; ablated "
        "12/16 / 0.4247 и 2/8 / 0.9867) — граница применимости, не дефект "
        "модели.",
        "2_edge_geometry_two_sided": "Краевая геометрия dup-ноги двусторонне: "
        "full 0.5208 = запас +0.021 над порогом" + margin_qualifier + "; "
        "ablated 0.4247 (12/16) НИЖЕ порога — продуктового заявления о "
        "cross-lang dup-детекции нет.",
        "3_flora_power": "Декларация мощности флоры 1/30 (скаффолд S24).",
        "4_d_alignment_report_only": "§D-выравнивание retained-ног — "
        f"report-only: в диапазоне страт §D paradup "
        f"{sd_alignment['paradup']['in_range']}/{sd_alignment['paradup']['n']}, "
        f"factedit {sd_alignment['factedit']['in_range']}/"
        f"{sd_alignment['factedit']['n']}; проба-набор заморожен по sha "
        "82dbff7b… без регенерации (регенерация под выравнивание после "
        "взгляда на красный = порог-фиттинг).",
        "5_quarantine_load_bearing": "Карантинная опора: защита продуктового "
        "контура — карантин §A(i) скрипт-детектором (отказ = "
        "консервативный сбой); карантин становится несущим механизмом.",
        "6_population_both_numbers": "Популяция cross-lang — обе цифры с "
        "базами: 6/3 654 = доля в эвиденс-срезе пула (перепись 22.10); "
        "370 = карантин §A в сборке v4.4-r2 (пул+контроль; леджер-ассерт "
        "пролога); вложенность 6 ≤ 370.",
        "reopen_hooks": "Reopen-хуки (4 триггера §G): снятие карантина §A; "
        "кросс-языковой минт (дрейф-монитор A2.9); фича-запрос D; "
        "off-manifold ≥ 8/80. Возврат xlang-ног в гейтующий состав — "
        "ТОЛЬКО новый прег + решение владельца (реанимация правкой порога "
        "запрещена); старт чартера 1.n+2 ≠ автоматический возврат ног.",
    }

    # ── determinism section (attempt-4 shape: vector assert + double ──────
    #    scoring pass; train double-run NOT executed — no training)
    det = {
        "vector_assert_vs_attempt3": pre_out["determinism"],
        "double_scoring_pass_single_read": {
            "byte_identical": double_pass,
            "note": "second pass over the ALREADY-READ in-memory data",
        },
        "booster_byte_continuity": {
            "run_a_pins": tripwire["boosters"],
            "run_b_eq_a": tripwire["run_b_booster_eq_a"],
        },
        "onnx_eq_attempt3_manifest": tripwire["onnx"],
        "train_double_run": "NOT EXECUTED — attempt 4 trains nothing "
        "(ADDENDUM-4: controlled reproduction)",
    }
    _log(
        gate_dir,
        "determinism",
        {"vector_assert": "PASS (160/160)", "double_scoring": double_pass},
    )

    verdict = "ADOPT" if all(g["pass"] for g in gates) else "REJECT"
    red = [g["gate"] for g in gates if not g["pass"]]
    report = {
        "verdict": verdict,
        "red_gates": red,
        "prereg": "docs/specs/b2-graph-prereg-v3.md (ACTIVATED 22.10) + ADDENDUM-1/2 "
        "(ACTIVATED 22.10) + ADDENDUM-3 (ACTIVATED 23.10) + ADDENDUM-4 "
        f"(A2.6 revision, ACTIVATED before this run, commit "
        f"{activation_commit[:12]}); thresholds frozen, single-shot; "
        "gating composition: g1 + g1b(paradup, factedit) + g3 + g2 + "
        "g4′ + g5 + g6",
        "gates": gates,
        "holdout_v45": {
            "full": m_hold_full,
            "ablated": m_hold_abl,
            "per_stratum_full": strata_full,
            "per_stratum_ablated": strata_abl,
            "quarantine": len(q_hold),
        },
        "g4_spec_v1": {
            "full": m_g4_full,
            "ablated": m_g4_abl,
            "quarantine": len(q_g4),
        },
        "strata_a_detail": a_rows,
        "evidence_agreement": {"holdout": ag_hold, "g4spec": ag_g4},
        "sidecars_fresh": {
            "holdout": json.loads(
                (gate_dir / "graph-sidecar-fresh-holdout-v45.coverage.json").read_text()
            ).get("sidecar_fingerprint"),
            "g4spec": json.loads(
                (gate_dir / "graph-sidecar-fresh-g4-spec-v1.coverage.json").read_text()
            ).get("sidecar_fingerprint"),
        },
        "characterization_a2.8": characterization,
        "addendum4": {
            "activation_commit": activation_commit,
            "prolog": {
                "tripwire": tripwire,
                "env_pins": dict(PACKAGE_PINS, python=PYTHON_PIN),
            },
            "characterization": {
                "xlang_legs_non_gating": pre_out["xlang"],
                "knn_manifold": knn,
            },
            "disclosures_f": disclosures,
        },
        "determinism": det,
        "artifacts": {
            tag: {
                "sha256": scorers[tag].weights_sha256,
                "feature_set_sha256": json.loads(
                    (run_a / f"model-{tag}.manifest.json").read_text()
                )["feature_set_sha256"],
                "cv_config": json.loads(
                    (run_a / f"model-{tag}.manifest.json").read_text()
                ).get("cv_selected_config"),
            }
            for tag in ("full", "ablated")
        },
        "wall_s": round(time.monotonic() - t0, 1),
    }
    (gate_dir / "gate-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(
        gate_dir,
        "gate_done",
        {"verdict": verdict, "red": red, "wall_s": report["wall_s"]},
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "red_gates": red,
                "gates": [
                    {k: g[k] for k in ("gate", "required", "measured", "pass")}
                    for g in gates
                ],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "ADOPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
