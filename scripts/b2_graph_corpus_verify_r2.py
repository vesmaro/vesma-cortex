#!/usr/bin/env python3
"""Independent verifier for the v4.4-r2 TRAIN rebuild (rev1n-retry gate, g5).

Builder-independent re-derivation from disk — never calls the builder:

  1. r2 train + labels fingerprints (data-contract §5) vs the sealed
     r2 manifest; counts and strata tables;
  2. the SEALED holdout v4.4: fingerprint, ids-sha, row count, label-file
     consistency (reads holdout content ONLY to re-derive the seal —
     the sanctioned gate phase before the single eval read);
  3. containment: original v44 train ⊆ r2 train; every new row carries an
     addendum1-* generator key (ZERO evidence by construction);
  4. §C razor-subzone parity (three sealed readings) + 3 cos bins >= 30;
  5. §D probe coverage: band |cos - probe| <= 0.005 ∧ equal tag_jaccard,
     K >= 30 producing-class rows, continuity max-gap <= 0.005
     (r2 manifest `probe_procedure_for_gate_runner`);
  6. scaffold anti-collapse on the 765 minted rows (>= 20 scaffolds per
     class, <= 5 % rows per scaffold);
  7. pair-level disjointness: r2 train vs every holdout surface;
  8. sidecar coverage: every non-pool, non-generator row resolves in the
     frozen rev1n train sidecar.

No secrets, no network. Exits non-zero on ANY failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from b2_rev1n_lib import GRAPH_STRATA, load_jsonl  # noqa: E402
from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.features.graph import GraphSidecar  # noqa: E402

SEAL_HOLDOUT_FP = "a9760d1f06ecdd40530bcc2dc37311f5e5bcd23aad653e03c87a49fe36761a97"
SEAL_IDS_SHA = "42d690578723fed34549532f2837f4540181a3a034097ce208f922dd710ce229"
R2_TRAIN_FP = "bb1ca943d6beabf56a88450c977cd04f829bdee89d75b81dc1b5f394b1314106"
R2_LABELS_FP = "e11e329849cd0f5fdca13a242db471ad4106f517c5e09e0277cb89af621b2f16"

PROBE_CLASSES = (
    ("P-identity", 1.0, 1.0, "self_pair"),
    ("P-envelope", 1.0, 1.0, "envelope_variant"),
    ("P-cosmetic", 0.99, 1.0, "cosmetic_twin"),
    ("N-fact-edit", 0.99, 1.0, "fact_edit_twin"),
    ("N-far", 0.578, 0.0, "unrelated"),
)
PROBE_DELTA = 0.005
PROBE_K = 30
PARITY_FLOOR_POS = 854
MINTED_CLASSES = (
    "N-fact-edit",
    "N-far",
    "P-cosmetic",
    "P-identity",
    "P-lexsub",
    "P-para-light",
)

CHECKS: list[str] = []
FAILS: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append(name)
    mark = "PASS" if cond else "FAIL"
    print(
        f"[verify-r2] {mark} {name}" + (f" — {detail}" if detail else ""),
        file=sys.stderr,
    )
    if not cond:
        FAILS.append(f"{name}: {detail}")


def tag_jaccard(a: dict, b: dict) -> float:
    ta = {str(t) for t in (a.get("tags") or [])}
    tb = {str(t) for t in (b.get("tags") or [])}
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b2-graph-corpus-verify-r2")
    parser.add_argument("--train-dir", required=True, help="v44-r2 dir (train side)")
    parser.add_argument("--corpus-dir", required=True, help="v44 dir (holdout seal)")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument(
        "--holdout-surfaces",
        nargs="*",
        default=[],
        help="extra holdout jsonl paths for pair-level disjointness",
    )
    parser.add_argument(
        "--no-seal",
        action="store_true",
        help="TEST-ONLY shakeout mode: skip every holdout-reading check; "
        "the gate always runs the FULL verifier (no flag)",
    )
    args = parser.parse_args(argv)
    train_dir = Path(args.train_dir)
    corpus_dir = Path(args.corpus_dir)

    manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf-8"))
    train = load_jsonl(train_dir / "train.jsonl")
    lab_t = {
        r["pair_id"]: str(r["label"])
        for r in load_jsonl(train_dir / "labels-train.jsonl")
    }

    # ── 1. identity ───────────────────────────────────────────────────────
    entries = (
        (
            r["pair_id"],
            pair_sha256(
                {
                    "record": r["record"],
                    "candidate": r["candidate"],
                    "similarity": r["similarity"],
                }
            ),
        )
        for r in train
    )
    fp_train = corpus_fingerprint(manifest_bytes(entries))
    ok("r2_fp_train", fp_train == R2_TRAIN_FP, fp_train[:16])
    ok("r2_fp_labels", labels_fingerprint(lab_t) == R2_LABELS_FP)
    ok("r2_rows", len(train) == manifest["counts"]["train_rows"], str(len(train)))
    ok(
        "r2_labels_consistent",
        all(str(r["label"]) == lab_t[str(r["pair_id"])] for r in train),
    )
    ok(
        "r2_strata",
        Counter(r["stratum"] for r in train)
        == Counter(manifest["counts"]["train_strata"]),
    )
    ok(
        "r2_label_split",
        Counter(r["label"] for r in train)
        == Counter(manifest["counts"]["train_labels"]),
    )

    # ── 2. holdout seal ───────────────────────────────────────────────────
    hold: list[dict] = []
    lab_h: dict[str, str] = {}
    if args.no_seal:
        ok("seal_skipped_test_only", True, "--no-seal shakeout: holdout unread")
    else:
        hold = load_jsonl(corpus_dir / "holdout.jsonl")
        lab_h = {
            r["pair_id"]: str(r["label"])
            for r in load_jsonl(corpus_dir / "labels-holdout.jsonl")
        }
        h_entries = (
            (
                r["pair_id"],
                pair_sha256(
                    {
                        "record": r["record"],
                        "candidate": r["candidate"],
                        "similarity": r["similarity"],
                    }
                ),
            )
            for r in hold
        )
        fp_hold = corpus_fingerprint(manifest_bytes(h_entries))
        ok("seal_fp_holdout", fp_hold == SEAL_HOLDOUT_FP, fp_hold[:16])
        # holdout_ids_sha256 = sha256(canonical_json(sorted(pair_ids))) — the
        # slice-2 seal scheme (scripts/b2_graph_corpus.py, fingerprints.py).
        ids_sha = hashlib.sha256(
            canonical_json(sorted(str(r["pair_id"]) for r in hold)).encode("utf-8")
        ).hexdigest()
        ok("seal_ids_sha", ids_sha == SEAL_IDS_SHA, ids_sha[:16])
        ok("seal_rows", len(hold) == 75, str(len(hold)))
        ok(
            "seal_labels_consistent",
            all(str(r["label"]) == lab_h[str(r["pair_id"])] for r in hold),
        )

    # ── 3. containment + new-row generator audit ──────────────────────────
    old_ids = {r["pair_id"] for r in load_jsonl(corpus_dir / "train.jsonl")}
    r2_ids = {r["pair_id"] for r in train}
    new_rows = [r for r in train if r["pair_id"] not in old_ids]
    ok(
        "containment_old_subset_r2",
        old_ids <= r2_ids,
        f"old={len(old_ids)} r2={len(r2_ids)}",
    )
    ok(
        "new_rows_all_generator",
        all(
            "generator" in r and str(r.get("source", "")).startswith("addendum1")
            for r in new_rows
        ),
        f"new={len(new_rows)}",
    )
    ok(
        "new_rows_count",
        len(new_rows) == manifest["counts"]["new_rows"],
        str(len(new_rows)),
    )

    # ── 4. §C parity + bins ───────────────────────────────────────────────
    sub = [
        r
        for r in train
        if r["similarity"] >= 0.95 and tag_jaccard(r["record"], r["candidate"]) == 1.0
    ]
    pos = [r for r in sub if r["label"] == "duplicate"]
    neg = [r for r in sub if r["label"] == "not-duplicate"]
    c12 = [r for r in pos if r["stratum"] in ("P-identity", "P-cosmetic")]
    c12_new = [r for r in c12 if r["pair_id"] not in old_ids]

    def bin_of(c: float) -> str:
        if c < 0.95:
            return "<0.95"
        if c < 0.97:
            return "[0.95,0.97)"
        if c < 0.985:
            return "[0.97,0.985)"
        return "[0.985,1.0]"

    bins = Counter(bin_of(r["similarity"]) for r in neg)
    ok(
        "c_parity_dynamic",
        len(neg) >= PARITY_FLOOR_POS + len(c12_new),
        f"{len(neg)} vs {PARITY_FLOOR_POS + len(c12_new)}",
    )
    ok("c_parity_floor", len(neg) >= PARITY_FLOOR_POS, str(len(neg)))
    ok("c_parity_strict", len(neg) >= len(c12), f"{len(neg)} vs {len(c12)}")
    for b in ("[0.95,0.97)", "[0.97,0.985)", "[0.985,1.0]"):
        ok(f"c_bin{b}", bins.get(b, 0) >= 30, str(bins.get(b, 0)))

    # ── 5. §D probe coverage ──────────────────────────────────────────────
    for stratum, pc, ptj, check in PROBE_CLASSES:
        band = [
            r
            for r in train
            if r["stratum"] == stratum
            and abs(r["similarity"] - pc) <= PROBE_DELTA + 1e-12
            and abs(tag_jaccard(r["record"], r["candidate"]) - ptj) < 1e-9
        ]
        cs = sorted(r["similarity"] for r in band)
        gaps = [round(b - a, 9) for a, b in zip(cs, cs[1:])]
        max_gap = max(gaps) if gaps else 0.0
        ok(f"d_band_{check}", len(band) >= PROBE_K, f"count={len(band)}")
        ok(f"d_continuity_{check}", max_gap <= PROBE_DELTA + 1e-9, f"max_gap={max_gap}")

    # ── 6. scaffold anti-collapse (minted rows) ───────────────────────────
    minted = [
        r
        for r in train
        if "generator" in r and str(r.get("source", "")).startswith("addendum1")
    ]
    for st in MINTED_CLASSES:
        sub_rows = [r for r in minted if r["stratum"] == st]
        sc = Counter(r["scaffold"] for r in sub_rows)
        ok(
            f"scaffold_{st}",
            len(sc) >= 20 and max(sc.values()) / len(sub_rows) <= 0.05,
            f"scaffolds={len(sc)} max_share={round(max(sc.values()) / len(sub_rows), 4)}",
        )

    # ── 7. pair-level disjointness ────────────────────────────────────────
    surfaces: dict[str, list[dict]] = {}
    if not args.no_seal:
        surfaces["holdout_v44"] = hold
    for p in args.holdout_surfaces:
        surfaces[f"{Path(p).parent.name}"] = load_jsonl(p)
    for name, rows in surfaces.items():
        inter = r2_ids & {r["pair_id"] for r in rows}
        ok(f"disjoint_train_r2_vs_{name}", not inter, str(len(inter)))

    # ── 8. sidecar coverage ───────────────────────────────────────────────
    sidecar = GraphSidecar(args.sidecar)
    uncovered = []
    for r in train:
        stratum = str(r.get("stratum", ""))
        if stratum in GRAPH_STRATA or "generator" in r:
            continue
        if sidecar.evidence_for(str(r["pair_id"])) is None:
            uncovered.append(str(r["pair_id"]))
    ok("sidecar_coverage", not uncovered, f"uncovered={len(uncovered)}")

    print(f"[verify-r2] {len(CHECKS)} checks, {len(FAILS)} failed", file=sys.stderr)
    if FAILS:
        for f in FAILS:
            print(f"[verify-r2] FAIL: {f}", file=sys.stderr)
        return 1
    print(f"[verify-r2] OK: {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
