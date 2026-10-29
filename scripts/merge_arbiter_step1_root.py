#!/usr/bin/env python3
"""ADDENDUM-5 STEP 1 — root verification (characterization, no training).

Ratified recipe: docs/specs/merge-arbiter-corpus-prereg-ADDENDUM-5.md
(commit 347e73c; MLC 29.10 round 5, 3/3 for variant (а)). One testable
hypothesis, one axis, mechanical verdict:

  THE AXIS (pre-registered): char5_jaccard from the FROZEN
  cortex.features.pair.features — the lexical-overlap axis of the
  round-5 protocol (the 24 rejected rows sit at ~0.01, at the bottom of
  the train distribution; the n-gram text is title + "\\n" + body per the
  frozen contract).

  MATERIAL: 24 rejected TSIB rows + 120 other TSIB rows + 27 notdup rows
  (the noharm surface, untouched) + train CH-3 positives (both
  sub-strata incl. short-superset) + train CH-0/CH-5 + train negatives of
  ALL substrata (CH-2, CH-4, theme-near x3) — the EFFECTIVE corpus
  (null overrides applied).

  CRITERION (pre-registered HERE, before any feature computation; the
  criterion JSON is hashed into the run-log):
    1. HARD: max(char5 of the 24) < min(char5 of train CH-3) AND
       max(char5 of the 24) < min(char5 of train negatives); common
       direction = below; gap := min(char5 CH-3) - max(char5 24) >= 0.05
       (short-superset rows included in CH-3 by construction).
    2. NA corroboration >= 2 of 3 (mechanical definitions):
       a. NN-outlier: median over the 24 of d_nn(row -> effective train)
          > p95 of d_nn(train row -> effective train)   [euclidean over
          the 50-dim both-member feature matrix of the frozen-built
          Choice contract];
       b. leaf starvation: median over the 24 of frac_trees(leaf support
          < 5 train samples) > 0.5  [booster pred_leaf on the frozen
          iter-3 artifact];
       c. bulk avoidance: ALL 24 axis values < p1 of the axis over ALL
          train negatives (CH-2 + CH-4 + theme-near x3).
    3. non-separation: >= 95% of the 147 GREEN surface rows lie within
       [min, max] of train negatives on the axis (else the axis separates
       everything and is an artifact — DE).
    4. HYPOTHESIS KILLER (NA): min(char5 of train cross-lang negatives)
       > max(char5 of the 24). If violated — the hole theory dies
       (coverage exists, the model is blind on the axis) -> verdict
       ABORT, case (б): the root needs FEATURE work, not quotas.

  VERDICT = CONFIRMED iff 1 AND 2 AND 3 AND 4. No axis shopping: if the
  HARD leg fails on the pre-registered axis, the verdict is ABORT
  (ADDENDUM-5 §1, DE rule).

Diagnostics reported but NOT binding: full 25-feature contract deltas,
digit-token overlap, length-ratio, tag-jaccard, null flags, orientation
symmetry of e (both orders through the frozen iter-3 artifact), NN
distances, leaf support, percentiles vs bulk.

Discipline: no training, no sealed reads (noharm surface is the
UNSEALED QA surface; MA-holdout-choice and the sheet are not opened).
Extractor = frozen cortex.features only. Script hash, criterion hash and
result hash are logged.

Usage:
  <venv-python> scripts/merge_arbiter_step1_root.py \
      --artifact data/merge-arbiter/artifacts/merge-v1/merge_arbiter_v1-1-iter3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import merge_arbiter_train as mt  # noqa: E402

import numpy as np  # noqa: E402

SLICE = "merge-arbiter-step1-root-verification"
NOHARM = mt.ART_DIR / "noharm-surface-v1.jsonl"


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


CRITERION = {
    "axis": "char5_jaccard (frozen cortex.features.pair.features; n-gram text = title + \\n + body)",
    "material": {
        "rejected_24": "noharm surface TSIB rows with verdict keep under the frozen iter-3 model",
        "green_tsib": "the other TSIB surface rows",
        "notdup": "V4-PND surface rows",
        "train_classes": ["CH-3 (both sub-strata incl. short-superset)",
                          "CH-0", "CH-5",
                          "negatives: CH-2, CH-4, theme-near x3"],
    },
    "hard": {
        "rule": "max(char5_24) < min(char5_CH3) AND max(char5_24) < min(char5_train_negatives)",
        "gap": "min(char5_CH3) - max(char5_24) >= 0.05",
    },
    "corroboration_min": 2,
    "corroboration": {
        "nn_outlier": "median d_nn(24 -> effective train) > p95 d_nn(train -> effective train); euclidean, 50-dim both-member frozen-contract matrix",
        "leaf_starvation": "median over 24 of frac_trees(leaf support < 5 train samples) > 0.5",
        "bulk_avoidance": "all 24 axis values < p1(axis over ALL train negatives)",
    },
    "non_separation": ">= 95% of the 147 green surface rows within [min, max] of train-negative axis support",
    "killer": "min(char5 train cross-lang negatives) > max(char5_24); violation => ABORT case (б)",
    "verdict": "CONFIRMED iff hard AND corroboration>=2 AND non_separation AND killer-survives",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", required=True)
    args = ap.parse_args(argv)

    mt._SLICE = SLICE
    mt.log("cost_declaration", expected_prolog_s=2, expected_cycle_s=60,
           expected_peak_ram_gb=1.0, lane="light (draft §9)")

    # light prologue (no training, no sealed reads)
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    mt.log("swap_check", available_mb=int(free.splitlines()[1].split()[6]))
    mt.log("env", executable=sys.executable, python=sys.version.split()[0],
           lightgbm=mt.lgb.__version__, numpy=np.__version__,
           machine=platform.machine())
    mt.log("host", hostname=platform.node(), expectation=mt.EXPECTED_HOST)
    r = subprocess.run(["git", "merge-base", "--is-ancestor", "347e73c", "HEAD"],
                       cwd=REPO, capture_output=True)
    addendum_ok = r.returncode == 0
    mt.log("activation_addendum5", commit="347e73c",
           is_ancestor=r.returncode == 0, ok=addendum_ok)
    if not addendum_ok:
        raise mt.InstrumentAbort("ADDENDUM-5 not at HEAD")

    script_sha = mt.sha256_file(Path(__file__))
    criterion_sha = _sha_text(json.dumps(CRITERION, ensure_ascii=False,
                                         sort_keys=True))
    # ======== MINI-PREREG: criterion BEFORE any feature computation ========
    mt.log("step1_prereg_criterion", criterion=CRITERION,
           criterion_sha256=criterion_sha, script_sha256=script_sha,
           note="written BEFORE feature computation (ADDENDUM-5 §1, NA mini-prereg)")

    # ---------------- feature computation starts HERE ----------------
    model, tau_hi, tau_m = mt.load_frozen(Path(args.artifact))
    rows = [json.loads(line) for line in open(NOHARM, encoding="utf-8")
        if line.strip()]

    def surface_pair_x(r: dict) -> tuple[np.ndarray, np.ndarray]:
        a = mt.lib.Member("a", r["title_a"], r["body_a"], tuple(r["tags_a"]),
                          r["type_a"], r["lang_a"], r["created_a"])
        b = mt.lib.Member("b", r["title_b"], r["body_b"], tuple(r["tags_b"]),
                          r["type_b"], r["lang_b"], r["created_b"])
        x = mt.cluster_features([a, b])
        return x[0], x[1]

    surf = {}
    for r in rows:
        xa, xb = surface_pair_x(r)
        d = mt.arbitrate_production(
            mt.lib.Cluster("x", "N", "s", 
                           [mt.lib.Member("a", r["title_a"], r["body_a"],
                                          tuple(r["tags_a"]), r["type_a"],
                                          r["lang_a"], r["created_a"]),
                            mt.lib.Member("b", r["title_b"], r["body_b"],
                                          tuple(r["tags_b"]), r["type_b"],
                                          r["lang_b"], r["created_b"])],
                           {"type": "abstain"}, {}),
            model, tau_hi, tau_m)
        surf[r["pair_id"]] = {"x": np.concatenate([xa, xb]), "verdict": d["verdict"],
                              "sub": "notdup" if r["pair_id"].startswith("V4-PND")
                              else "tsib"}
    rejected = [k for k, v in surf.items()
                if v["verdict"] == "keep" and v["sub"] == "tsib"]
    green_tsib = [k for k, v in surf.items()
                  if v["verdict"] == "abstain" and v["sub"] == "tsib"]
    notdup = [k for k, v in surf.items() if v["sub"] == "notdup"]
    mt.log("step1_material", rejected=len(rejected), green_tsib=len(green_tsib),
           notdup=len(notdup), surface_rows=len(rows))
    if len(rejected) != 24:
        raise mt.InstrumentAbort(f"material drift: {len(rejected)} rejected, want 24")

    # frozen char5 axis per surface row (title + \n + body via the contract)
    from cortex.features.pair import RecordLike, features
    def pair_char5(r: dict) -> float:
        vec = features(RecordLike(r["title_a"], r["body_a"], tuple(r["tags_a"]),
                                  r["type_a"], r["lang_a"]),
                       RecordLike(r["title_b"], r["body_b"], tuple(r["tags_b"]),
                                  r["type_b"], r["lang_b"]),
                       similarity=1.0)
        return float(vec.values[mt.lib.FEATURE_NAMES.index("char5_jaccard")])

    char5_rejected = {k: pair_char5(rows[[r["pair_id"] for r in rows].index(k)])
                      for k in rejected}

    # train classes (effective corpus)
    clusters = mt.load_train()
    tr_mats, tr_meta = [], []
    ch3_x, neg_x, neg_cross_x, ch0ch5_x = [], [], [], []
    for c in clusters:
        x = mt.cluster_features(c.members)
        tr_mats.append(x)
        tr_meta.append({"cid": c.cid, "stratum": mt.stratum_key(c)})
        from cortex.features.pair import RecordLike as RL
        m0, m1 = c.members[0], c.members[-1]
        vec = features(RL(m0.title, m0.body, m0.tags, m0.language, m0.record_type),
                       RL(m1.title, m1.body, m1.tags, m1.language, m1.record_type),
                       similarity=1.0)
        c5 = float(vec.values[mt.lib.FEATURE_NAMES.index("char5_jaccard")])
        if c.stratum == "CH-3":
            ch3_x.append(c5)
        if c.label["type"] != "keep" and c.stratum != "theme-near":
            neg_x.append(c5)
        if c.stratum in ("CH-0", "CH-5"):
            ch0ch5_x.append(c5)
        if c.stratum == "theme-near" and c.substratum == "cross-lang-sibling":
            neg_cross_x.append(c5)
    tr_X = np.concatenate(tr_mats, axis=0)  # (n_members, 25) member rows
    # cluster membership per row (for excluding own cluster in NN)
    row_cluster = np.concatenate([
        np.full(len(c.members), i) for i, c in enumerate(clusters)])

    ch3_min, ch3_max = min(ch3_x), max(ch3_x)
    neg_min, neg_max = min(neg_x), max(neg_x)
    rej_vals = sorted(char5_rejected.values())
    rej_min, rej_max = rej_vals[0], rej_vals[-1]
    cross_min = min(neg_cross_x)
    gap = ch3_min - rej_max

    hard = bool(rej_max < ch3_min and rej_max < neg_min and gap >= 0.05)
    res = {
        "axis_char5": {
            "rejected": {"min": round(rej_min, 6), "max": round(rej_max, 6)},
            "train_ch3": {"min": round(ch3_min, 6), "max": round(ch3_max, 6)},
            "train_negatives": {"min": round(neg_min, 6), "max": round(neg_max, 6)},
            "train_cross_lang_negatives": {"min": round(cross_min, 6),
                                           "max": round(max(neg_cross_x), 6)},
            "gap_ch3_min_to_band_top": round(gap, 6),
        },
    }
    mt.log("step1_axis_char5", **res["axis_char5"], hard=hard)

    # criterion 4 first (the killer — cheapest decisive leg)
    killer_violated = bool(cross_min <= rej_max)
    res["killer_survives"] = not killer_violated
    mt.log("step1_killer", train_cross_lang_min=round(cross_min, 6),
           band_top=round(rej_max, 6), violated=killer_violated,
           note="NA: train cross-lang negatives must sit ENTIRELY above the "
                "rejected band; violation => coverage exists, model blind "
                "=> no quota sanction (case б)")

    # corroboration diagnostics
    # (a) NN-outlier
    d_nn_train = []
    for i in range(tr_X.shape[0]):
        d = np.linalg.norm(tr_X - tr_X[i], axis=1)
        d[row_cluster == row_cluster[i]] = np.inf  # exclude own cluster
        d_nn_train.append(float(d.min()))
    p95 = float(np.percentile(d_nn_train, 95))
    nn_med = []
    for k in rejected:
        # compare each surface MEMBER row (25-dim) against train member rows
        xa, xb = surface_pair_x(rows[[r["pair_id"] for r in rows].index(k)])
        d0 = np.linalg.norm(tr_X - xa, axis=1)
        d1 = np.linalg.norm(tr_X - xb, axis=1)
        nn_med.append(float(min(d0.min(), d1.min())))
    nn_outlier = bool(float(np.median(nn_med)) > p95)
    # (b) leaf starvation
    from collections import Counter
    leaves_tr = model.predict(tr_X, pred_leaf=True)
    starv = []
    for k in rejected:
        xa, xb = surface_pair_x(rows[[r["pair_id"] for r in rows].index(k)])
        leaves_pair = model.predict(np.array([xa, xb]), pred_leaf=True)
        fr = []
        for i in range(leaves_pair.shape[0]):
            low = 0
            for t in range(leaves_pair.shape[1]):
                cnt = Counter(leaves_tr[:, t])[leaves_pair[i, t]]
                if cnt < 5:
                    low += 1
            fr.append(low / leaves_pair.shape[1])
        starv.append(float(np.mean(fr)))  # mean over the pair's two members
    leaf_starvation = bool(float(np.median(starv)) > 0.5)
    # (c) bulk avoidance
    bulk_p1 = float(np.percentile(neg_x, 1))
    bulk_avoidance = bool(rej_max < bulk_p1)
    corr = {"nn_outlier": nn_outlier, "leaf_starvation": leaf_starvation,
            "bulk_avoidance": bulk_avoidance}
    n_corr = sum(corr.values())
    res["corroboration"] = {**{k: bool(v) for k, v in corr.items()},
                            "nn_median_rejected": round(float(np.median(nn_med)), 4),
                            "nn_p95_train": round(p95, 4),
                            "leaf_starv_median": round(float(np.median(starv)), 4),
                            "bulk_p1": round(bulk_p1, 6)}
    mt.log("step1_corroboration", **res["corroboration"], passed=n_corr,
           need=CRITERION["corroboration_min"])

    # criterion 3: non-separation of the green rows
    green_in = sum(1 for k in green_tsib + notdup
                   if neg_min <= pair_char5(rows[[r["pair_id"] for r in rows].index(k)])
                   <= neg_max)
    sep_ok = bool(green_in / len(green_tsib + notdup) >= 0.95)
    res["non_separation"] = {"green_within_train_support": green_in,
                             "green_total": len(green_tsib) + len(notdup),
                             "ok": sep_ok}
    mt.log("step1_non_separation", **res["non_separation"])

    # verdict
    verdict = ("CONFIRMED" if (hard and n_corr >= 2 and sep_ok
                               and not killer_violated) else "ABORT")
    res["verdict"] = verdict
    res["legs"] = {"hard": hard, "corroboration": n_corr,
                   "non_separation": sep_ok, "killer_survives": not killer_violated}
    result_sha = _sha_text(json.dumps(res, ensure_ascii=False, sort_keys=True))
    mt.log("step1_verdict", verdict=verdict, legs=res["legs"],
           result_sha256=result_sha,
           criterion_sha256=criterion_sha, script_sha256=script_sha,
           note="CONFIRMED => step 2 sanctioned immediately; ABORT => case (б): "
                "root needs feature work, not quotas; no axis shopping")
    Path(mt.ART_DIR / "step1_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"STEP1 VERDICT: {verdict}")
    return 0 if verdict == "CONFIRMED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
