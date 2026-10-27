#!/usr/bin/env python3
"""Pre-flight environment checks for the rev1n-retry gate — HOLDOUT-FREE.

Re-derives, independently of the r2 builder:
  1. r2 train + labels fingerprints (data-contract §5 scheme) vs the
     sealed r2 manifest;
  2. pair_id containment: original v44 train ⊆ r2 train; every NEW row
     carries a generator key (ZERO evidence by construction);
  3. sidecar coverage: every non-pool, non-generator r2 row resolves in
     the frozen rev1n train sidecar;
  4. §C razor-subzone census (parity + 3 bins) and §D probe coverage
     (band ±0.005 ∧ equal tag_jaccard, K>=30, continuity max-gap <=
     0.005) per the r2 manifest `probe_procedure_for_gate_runner`.

No secrets, no network, no holdout reads.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/b2-graph-corpus"
)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from b2_rev1n_lib import (  # noqa: E402
    GRAPH_STRATA,
    assert_corpus_identity,
    load_jsonl,
)
from cortex.features.graph import GraphSidecar  # noqa: E402

R2_DIR = REPO_ROOT / "data/b2-graph-corpus/v44-r2"
V44_DIR = REPO_ROOT / "data/b2-graph-corpus/v44"
OLD_SIDECAR = REPO_ROOT / "data/b2-graph-corpus/rev1n/graph-sidecar-train.jsonl"

# Sealed r2 manifest pins (datasets/b2-graph/v44-r2/manifest.json mirror).
EXPECT_TRAIN_FP = "bb1ca943d6beabf56a88450c977cd04f829bdee89d75b81dc1b5f394b1314106"
EXPECT_LABELS_FP = "e11e329849cd0f5fdca13a242db471ad4106f517c5e09e0277cb89af621b2f16"

# §D probe points (src/cortex/eval/sanity.py frozen constants x strata).
PROBE_CLASSES = (
    ("P-identity", 1.0, 1.0, "self_pair"),
    ("P-envelope", 1.0, 1.0, "envelope_variant"),
    ("P-cosmetic", 0.99, 1.0, "cosmetic_twin"),
    ("N-fact-edit", 0.99, 1.0, "fact_edit_twin"),
    ("N-far", 0.578, 0.0, "unrelated"),
)
PROBE_DELTA = 0.005
PROBE_K = 30


def tag_jaccard(a: dict, b: dict) -> float:
    ta = {str(t) for t in (a.get("tags") or [])}
    tb = {str(t) for t in (b.get("tags") or [])}
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    out: dict = {}

    # ── 1. corpus identity (fail-loud) ────────────────────────────────────
    asserts = assert_corpus_identity(
        R2_DIR / "train.jsonl",
        R2_DIR / "labels-train.jsonl",
        R2_DIR / "manifest.json",
        EXPECT_TRAIN_FP,
        EXPECT_LABELS_FP,
    )
    out["corpus_identity"] = {
        "rows": asserts.rows,
        "dups": asserts.dups,
        "not_dups": asserts.not_dups,
        "train_fingerprint": asserts.train_fingerprint,
        "labels_fingerprint": asserts.labels_fingerprint,
        "ok": True,
    }
    print(json.dumps({"step": "corpus_identity", **out["corpus_identity"]}))

    # ── 2. pair_id containment + new-row generator audit ─────────────────
    old_ids = {r["pair_id"] for r in load_jsonl(V44_DIR / "train.jsonl")}
    rows = load_jsonl(R2_DIR / "train.jsonl")
    r2_ids = {r["pair_id"] for r in rows}
    new_rows = [r for r in rows if r["pair_id"] not in old_ids]
    new_no_gen = [r["pair_id"] for r in new_rows if "generator" not in r]
    out["containment"] = {
        "old_train_rows": len(old_ids),
        "r2_rows": len(r2_ids),
        "new_rows": len(new_rows),
        "new_rows_without_generator": len(new_no_gen),
        "old_subset_of_r2": old_ids <= r2_ids,
        "ok": old_ids <= r2_ids and not new_no_gen,
    }
    print(json.dumps({"step": "containment", **out["containment"]}))

    # ── 3. sidecar coverage of non-pool non-generator rows ───────────────
    sidecar = GraphSidecar(str(OLD_SIDECAR))
    missing = []
    kinds = Counter()
    for r in rows:
        stratum = str(r.get("stratum", ""))
        if stratum in GRAPH_STRATA or "generator" in r:
            continue
        if sidecar.evidence_for(str(r["pair_id"])) is None:
            missing.append(str(r["pair_id"]))
        kinds["sidecar"] += 1
    out["sidecar_coverage"] = {
        "sidecar_rows": len(sidecar),
        "nonpool_nongen_rows": sum(kinds.values()),
        "uncovered": len(missing),
        "ok": not missing,
    }
    print(json.dumps({"step": "sidecar_coverage", **out["sidecar_coverage"]}))

    # ── 4. §C census + §D probe bands (independent re-derivation) ────────
    sub = [
        r
        for r in rows
        if r["similarity"] >= 0.95 and tag_jaccard(r["record"], r["candidate"]) == 1.0
    ]
    pos = [r for r in sub if r["label"] == "duplicate"]
    neg = [r for r in sub if r["label"] == "not-duplicate"]
    bin_of = lambda c: (  # noqa: E731
        "<0.95"
        if c < 0.95
        else "[0.95,0.97)"
        if c < 0.97
        else "[0.97,0.985)"
        if c < 0.985
        else "[0.985,1.0]"
    )
    bins = Counter(bin_of(r["similarity"]) for r in neg)
    c12 = [r for r in pos if r["stratum"] in ("P-identity", "P-cosmetic")]
    # §C parity, the three readings sealed in the r2 manifest:
    #   dynamic  = neg vs (census floor 854 + NEW class-1/2 subzone rows)
    #   floor    = neg vs static census floor 854
    #   strict   = neg vs classes-1/2 subzone positives
    c12_new = [
        r
        for r in new_rows
        if r["stratum"] in ("P-identity", "P-cosmetic")
        and r["similarity"] >= 0.95
        and tag_jaccard(r["record"], r["candidate"]) == 1.0
    ]
    census = {
        "subzone_pos": len(pos),
        "subzone_neg": len(neg),
        "pos_classes_1_2_strict": len(c12),
        "new_class12_subzone_rows": len(c12_new),
        "parity_dynamic": [len(neg), 854 + len(c12_new)],
        "parity_floor": [len(neg), 854],
        "parity_strict": [len(neg), len(c12)],
        "bins": {
            b: bins.get(b, 0) for b in ("[0.95,0.97)", "[0.97,0.985)", "[0.985,1.0]")
        },
    }
    bands = {}
    for stratum, pc, ptj, check in PROBE_CLASSES:
        band = [
            r
            for r in rows
            if r["stratum"] == stratum
            and abs(r["similarity"] - pc) <= PROBE_DELTA + 1e-12
            and abs(tag_jaccard(r["record"], r["candidate"]) - ptj) < 1e-9
        ]
        cs = sorted(r["similarity"] for r in band)
        gaps = [round(b - a, 9) for a, b in zip(cs, cs[1:])]
        max_gap = max(gaps) if gaps else 0.0
        bands[stratum] = {
            "check": check,
            "count": len(band),
            "max_gap": max_gap,
            "k_ok": len(band) >= PROBE_K,
            "continuity_ok": max_gap <= PROBE_DELTA + 1e-9,
        }
    out["census_c"] = {
        **census,
        "parity_ok": all(
            neg_side >= pos_side
            for neg_side, pos_side in (
                census["parity_dynamic"],
                census["parity_floor"],
                census["parity_strict"],
            )
        ),
        "bins_ok": all(v >= 30 for v in census["bins"].values()),
    }
    out["probe_bands_d"] = {
        k: {**v, "ok": v["k_ok"] and v["continuity_ok"]} for k, v in bands.items()
    }
    print(json.dumps({"step": "census_c", **out["census_c"]}))
    for k, v in out["probe_bands_d"].items():
        print(json.dumps({"step": "band", "probe": k, **v}))

    (Path(__file__).parent / "preflight.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok_all = (
        out["corpus_identity"]["ok"]
        and out["containment"]["ok"]
        and out["sidecar_coverage"]["ok"]
        and out["census_c"]["parity_ok"]
        and out["census_c"]["bins_ok"]
        and all(v["ok"] for v in out["probe_bands_d"].values())
    )
    print(json.dumps({"preflight_ok": ok_all}))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
