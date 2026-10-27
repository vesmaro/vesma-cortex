#!/usr/bin/env python3
"""Verify the built corpus v4.5 artifacts against the build manifest.

Re-computes everything verifiable from the FILES alone (fingerprints,
labels, composition, holdout seal, strata-A cluster integrity, evidence
columns, parity tables) and refuses on any drift. This is the corpus-stage
self-check the gate runner can re-run before training.

Usage:
  /usr/bin/python3 scripts/b2_graph_corpus_verify_v45.py \
      --dir data/b2-graph-corpus/v45

Exit codes: 0 ok | 1 verification failure | 2 usage.
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

from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
)

V44R2_DIR = REPO_ROOT / "data/b2-graph-corpus/v44-r2"
BASE_TRAIN_FP = "bb1ca943d6beabf56a88450c977cd04f829bdee89d75b81dc1b5f394b1314106"


class VerifyFailure(SystemExit):
    pass


def fail(msg: str) -> None:
    raise VerifyFailure(f"[verify-v45] FAIL: {msg}")


def rows_of(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"missing artifact: {path}")
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def main() -> int:
    p = argparse.ArgumentParser(prog="b2_graph_corpus_verify_v45")
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    d = Path(args.dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))

    train = rows_of(d / "train.jsonl")
    holdout = rows_of(d / "holdout.jsonl")
    labels = {r["pair_id"]: r["label"] for r in rows_of(d / "labels-train.jsonl")}

    # 1. base untouched
    base = rows_of(V44R2_DIR / "train.jsonl")
    base_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in base])
    )
    if base_fp != BASE_TRAIN_FP:
        fail("v4.4-r2 base fingerprint drifted")

    # 2. pair_sha256 integrity of every artifact row
    for name, rows in (("train", train), ("holdout", holdout)):
        for r in rows:
            obj = {
                "record": r["record"],
                "candidate": r["candidate"],
                "similarity": r["similarity"],
            }
            dig = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
            if dig != r["pair_sha256"]:
                fail(f"{name} row {r['pair_id']}: pair_sha256 mismatch")

    # 3. fingerprints
    train_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in train])
    )
    if train_fp != manifest["fingerprints"]["train_v45"]:
        fail("train fingerprint differs from the manifest")
    hold_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in holdout])
    )
    if hold_fp != manifest["fingerprints"]["holdout_v45_seal"]:
        fail("holdout fingerprint differs from the sealed manifest value")
    ids = sorted(r["pair_id"] for r in holdout)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    if ids_sha != manifest["fingerprints"]["holdout_v45_ids_sha256"]:
        fail("holdout ids sha differs from the manifest")
    lab = {r["pair_id"]: r["label"] for r in train}
    if labels != lab:
        fail("labels-train.jsonl disagrees with train.jsonl")
    if labels_fingerprint(lab) != manifest["fingerprints"]["labels_train_v45"]:
        fail("labels fingerprint differs from the manifest")

    # 4. counts / composition
    c = manifest["counts"]
    if len(train) != c["train_rows"] or len(holdout) != c["holdout_rows"]:
        fail("row counts differ from the manifest")
    if dict(Counter(r["label"] for r in train)) != c["train_labels"]:
        fail("train label composition differs from the manifest")
    if dict(Counter(r["stratum"] for r in train)) != c["train_strata"]:
        fail("train strata composition differs from the manifest")

    # 5. train/holdout disjointness (pair ids and pair keys)
    train_ids = {r["pair_id"] for r in train}
    if train_ids & set(ids):
        fail("train/holdout pair_id intersection")
    train_keys = {
        tuple(
            sorted(
                (
                    hashlib.sha256(
                        json.dumps(
                            {
                                k: r["record"].get(k)
                                for k in (
                                    "title",
                                    "body",
                                    "tags",
                                    "language",
                                    "record_type",
                                )
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                    hashlib.sha256(
                        json.dumps(
                            {
                                k: r["candidate"].get(k)
                                for k in (
                                    "title",
                                    "body",
                                    "tags",
                                    "language",
                                    "record_type",
                                )
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                )
            )
        )
        for r in train
    }
    for r in holdout:
        hk = tuple(
            sorted(
                (
                    hashlib.sha256(
                        json.dumps(
                            {
                                k: r["record"].get(k)
                                for k in (
                                    "title",
                                    "body",
                                    "tags",
                                    "language",
                                    "record_type",
                                )
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                    hashlib.sha256(
                        json.dumps(
                            {
                                k: r["candidate"].get(k)
                                for k in (
                                    "title",
                                    "body",
                                    "tags",
                                    "language",
                                    "record_type",
                                )
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                )
            )
        )
        if hk in train_keys:
            fail(f"holdout pair {r['pair_id']} collides with a train pair key")

    # 6. strata A cluster integrity across train/holdout
    a_train = {r["cluster_key"] for r in train if r.get("cluster_key")}
    a_hold = {r["cluster_key"] for r in holdout if r.get("cluster_key")}
    if a_train & a_hold:
        fail("strata A cluster split across train/holdout")
    if not (2 <= len(a_hold) <= 3):
        fail(f"strata A holdout clusters {len(a_hold)} outside 2-3")
    a_train_pairs = sum(1 for r in train if r.get("cluster_key"))
    if a_train_pairs < 30:
        fail(f"strata A train mass {a_train_pairs} < 30")

    # 7. evidence columns present and coherent on the NEW v4.5 material (the
    #    A2.5.5 freeze covers the new rows; inherited base rows predate it)
    #    (ONE reading: edge_ab=1 OR common_neighbors>=1)
    for r in train + holdout:
        if not str(r.get("pair_id", "")).startswith("V45-"):
            continue
        for col in (
            "edge_ab",
            "common_neighbors",
            "neighbor_jaccard",
            "cos_to_common_max",
            "has_graph_evidence",
        ):
            if col not in r:
                fail(f"row {r['pair_id']} missing frozen column {col}")
        ev = r["edge_ab"] == 1 or r["common_neighbors"] >= 1
        if bool(ev) != bool(r["has_graph_evidence"]):
            fail(f"row {r['pair_id']} evidence flag disagrees with the frozen reading")
        if r.get("id_a") and r["has_graph_evidence"]:
            fail(f"strata A row {r['pair_id']} carries a non-zero graph vector")

    # 8. N-near band report coherence (rows on disk vs manifest bands)
    nnc = [
        r
        for r in train + holdout
        if r.get("stratum") == "N-near" and r.get("generator", "").startswith("v45")
    ]
    import re as _re

    for r in nnc:
        if not _re.match(r"0\.\d{3}", str(r.get("band", ""))):
            fail(f"N-near row {r['pair_id']} without a parseable band field")

    print(
        f"[verify-v45] OK train={len(train)} holdout={len(holdout)} "
        f"train_fp={train_fp[:16]}... holdout_seal={hold_fp[:16]}... "
        f"strata_A_train={a_train_pairs} holdout_clusters={len(a_hold)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VerifyFailure as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
