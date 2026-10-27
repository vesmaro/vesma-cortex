#!/usr/bin/env python3
"""Independent verifier for the b2-graph corpus v4.4 (slice-2).

Re-reads the SHIPPED artifacts from disk and re-derives every load-bearing
number WITHOUT calling the builder: fingerprints, counts, the 21-pair base
cleaning, graph-strata quotas/cap, injection construction (T0/T1 policy,
vectors, similarity), hygiene over synthetic sides, and pair/side-level
disjointness of the fresh holdout. Exits non-zero on ANY mismatch.

Run:  .venv/bin/python scripts/b2_graph_corpus_verify.py \
          [--dir data/b2-graph-corpus/v44]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.data.store_export import make_scanner  # noqa: E402

BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")
R4_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
)
R5_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
)

CHECKS: list[str] = []
FAILS: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append(name)
    mark = "PASS" if cond else "FAIL"
    print(
        f"[verify] {mark} {name}" + (f" — {detail}" if detail else ""), file=sys.stderr
    )
    if not cond:
        FAILS.append(f"{name}: {detail}")


def rows_of(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def text_hash(t: str) -> str:
    n = unicodedata.normalize("NFC", t or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def side_sha(s: dict) -> str:
    payload = json.dumps(
        {
            "title": s.get("title"),
            "body": s.get("body"),
            "tags": sorted(s.get("tags") or []),
            "language": s.get("language"),
            "record_type": s.get("record_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_key(r: dict) -> tuple[str, str]:
    return tuple(sorted((side_sha(r["record"]), side_sha(r["candidate"]))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "data/b2-graph-corpus/v44")
    args = ap.parse_args()
    d = args.dir
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    train = rows_of(d / "train.jsonl")
    hold = rows_of(d / "holdout.jsonl")
    lab_t = {r["pair_id"]: r["label"] for r in rows_of(d / "labels-train.jsonl")}
    lab_h = {r["pair_id"]: r["label"] for r in rows_of(d / "labels-holdout.jsonl")}

    # ── counts + labels ──────────────────────────────────────────────────────
    ok("train_rows", len(train) == manifest["counts"]["train_rows"], str(len(train)))
    ok("holdout_rows", len(hold) == manifest["counts"]["holdout_rows"], str(len(hold)))
    ok(
        "labels_match_rows",
        lab_t == {r["pair_id"]: r["label"] for r in train}
        and lab_h == {r["pair_id"]: r["label"] for r in hold},
    )
    ok(
        "label_counters",
        Counter(r["label"] for r in train)
        == Counter(manifest["counts"]["train_labels"])
        and Counter(r["label"] for r in hold)
        == Counter(manifest["counts"]["holdout_labels"]),
    )
    ok(
        "stratum_counters",
        Counter(r["stratum"] for r in train)
        == Counter(manifest["counts"]["train_strata"])
        and Counter(r["stratum"] for r in hold)
        == Counter(manifest["counts"]["holdout_strata"]),
    )
    ok(
        "pair_ids_unique",
        len({r["pair_id"] for r in train + hold}) == len(train) + len(hold),
    )

    # ── fingerprints ─────────────────────────────────────────────────────────
    def fp(rows: list[dict]) -> str:
        return corpus_fingerprint(
            manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in rows])
        )

    ok("fp_train", fp(train) == manifest["fingerprints"]["train"])
    ok("fp_holdout", fp(hold) == manifest["fingerprints"]["holdout"])
    ok(
        "fp_corpus",
        corpus_fingerprint(
            manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in train + hold])
        )
        == manifest["fingerprints"]["corpus_v44"],
    )
    ok(
        "fp_labels",
        labels_fingerprint(lab_t) == manifest["fingerprints"]["labels_train"]
        and labels_fingerprint(lab_h) == manifest["fingerprints"]["labels_holdout"],
    )

    # ── base integrity: v4.3 train minus exactly the 21 removed copies ──────
    src = rows_of(BASE / "v43-localbuild/train.jsonl")
    removed_ids = {r["train_pair_id"] for r in manifest["base_clean"]["removed"]}
    expect_base = [r for r in src if r["pair_id"] not in removed_ids]
    ship_base = [r for r in train if not r.get("source")]
    ok(
        "base_rows_2378",
        len(ship_base) == 2378 and len(expect_base) == 2378,
        str(len(ship_base)),
    )
    same = all(
        a["pair_id"] == b["pair_id"]
        and a["record"] == b["record"]
        and a["candidate"] == b["candidate"]
        and a["label"] == b["label"]
        and a["stratum"] == b["stratum"]
        for a, b in zip(ship_base, expect_base)
    )
    ok("base_rows_byte_equal_to_source_minus_21", same)
    ok(
        "removed_ids_absent",
        not (removed_ids & {r["pair_id"] for r in train + hold}),
    )
    # every removed pair: label-identical copy of a live holdout row
    for entry in manifest["base_clean"]["removed"]:
        hr = next(
            r
            for r in rows_of(
                BASE
                / {
                    "v43-holdout": "v43-holdout/holdout.jsonl",
                    "v42-holdout-276": "dataset-v4-holdout/holdout.jsonl",
                    "b2-holdout-180": "b2/holdout.jsonl",
                }[entry["holdout_surface"]]
            )
            if r["pair_id"] == entry["holdout_pair_id"]
        )
        tr = next(r for r in src if r["pair_id"] == entry["train_pair_id"])
        if pair_key(hr) != pair_key(tr) or hr["label"] != tr["label"]:
            ok(
                f"removed_pair_{entry['train_pair_id']}",
                False,
                "pair-key/label mismatch",
            )
            break
    else:
        ok("removed_21_pairkey_and_label_equal_to_holdout_copies", True)

    # ── graph rows vs pool ───────────────────────────────────────────────────
    pool = {
        r["pair_id"]: r
        for r in rows_of(REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl")
    }
    graph_ship = [r for r in train + hold if r.get("source") == "b2-graph-pool-230745b"]
    per_stratum = Counter(r["stratum"] for r in graph_ship)
    ok(
        "graph_quotas",
        per_stratum
        == Counter(
            {"edge-direct": 240, "edge-common": 150, "mint-top": 150, "control": 60}
        ),
        str(dict(per_stratum)),
    )
    fields_equal = True
    for g in graph_ship:
        p = pool[g["pair_id"]]
        for k, v in p.items():
            if g.get(k) != v:
                fields_equal = False
                ok(f"graph_field_{g['pair_id'][:20]}_{k}", False, "differs from pool")
                break
    ok("graph_rows_untouched_vs_pool", fields_equal)
    comp = Counter(r["component_a"] for r in graph_ship)
    ok("graph_component_cap_90", max(comp.values()) <= 90, f"max={max(comp.values())}")
    ok(
        "graph_labels",
        all(r["label"] == "not-duplicate" for r in graph_ship),
    )

    # ── injection construction ───────────────────────────────────────────────
    inj = [r for r in train + hold if r.get("source") == "injection-policy-s4"]
    ok("injection_rows_429", len(inj) == 429, str(len(inj)))
    cls = Counter(r["pair_id"].rsplit("-", 1)[0] for r in inj)
    ok(
        "injection_split_2to1",
        cls["V44-INJ-T0S"] == 143
        and cls["V44-INJ-T0C"] == 143
        and cls["V44-INJ-T1C"] == 143,
        str(dict(cls)),
    )
    t0s = [r for r in inj if r["pair_id"].startswith("V44-INJ-T0S")]
    t0c = [r for r in inj if r["pair_id"].startswith("V44-INJ-T0C")]
    t1c = [r for r in inj if r["pair_id"].startswith("V44-INJ-T1C")]
    ok(
        "T0S_is_self_pair",
        all(r["record"] == r["candidate"] for r in t0s),
    )
    unit_t0s = {r["unit"] for r in t0s}
    unit_t0c = {r["unit"] for r in t0c}
    ok("T0S_T0C_share_units", len(unit_t0s) == len(unit_t0c) == 143)
    # each T0C clone == its unit's self base
    by_unit = {r["unit"]: r for r in t0s}
    ok(
        "T0C_clone_of_unit_base",
        all(
            r["candidate"] == by_unit[r["unit"]]["record"]
            and r["record"] == by_unit[r["unit"]]["record"]
            for r in t0c
        ),
    )
    ok(
        "T1C_body_swapcase",
        all(
            r["candidate"]["body"] == r["record"]["body"].swapcase()
            and r["candidate"]["title"] == r["record"]["title"]
            and r["candidate"]["tags"] == r["record"]["tags"]
            for r in t1c
        ),
    )
    ok(
        "T0_vectors_identical",
        all(r["vec_a"] == r["vec_b"] for r in t0s + t0c),
    )

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    ok(
        "injection_similarity_remeasured",
        all(
            r["similarity"] == round(min(1.0, max(0.0, dot(r["vec_a"], r["vec_b"]))), 6)
            for r in inj
        ),
    )
    ok(
        "injection_pair_sha256",
        all(
            r["pair_sha256"]
            == pair_sha256(
                {
                    "record": r["record"],
                    "candidate": r["candidate"],
                    "similarity": r["similarity"],
                }
            )
            for r in inj
        ),
    )
    ok(
        "injection_labels_by_construction",
        all(r["label"] == "duplicate" for r in inj)
        and {r["stratum"] for r in t0s + t0c} == {"P-identity"}
        and {r["stratum"] for r in t1c} == {"P-cosmetic"},
    )

    # hygiene: engine detectors over EVERY synthetic side (fresh scan)
    scanner = make_scanner(ENGINE_SRC)
    hits = []
    for r in inj:
        for side_name in ("record", "candidate"):
            s = r[side_name]
            found = scanner.scan(
                s.get("title") or "", s.get("body") or "", s.get("tags") or []
            )
            if found:
                hits.append((r["pair_id"], side_name, found))
    ok("hygiene_rescan_injection_0_hits", not hits, str(hits[:3]))

    # injection pair-level disjointness vs pool + historical surfaces
    inj_keys = {pair_key(r) for r in inj}
    pool_keys = {pair_key(r) for r in pool.values()}
    uni_keys = set()
    for name, p in (
        ("train", BASE / "v43-localbuild/train.jsonl"),
        ("h43", BASE / "v43-holdout/holdout.jsonl"),
        ("h42", BASE / "dataset-v4-holdout/holdout.jsonl"),
        ("b2", BASE / "b2/holdout.jsonl"),
        ("b2train", BASE / "b2/train.jsonl"),
    ):
        uni_keys |= {pair_key(r) for r in rows_of(p)}
    ok("injection_pairkey_0_vs_pool", not (inj_keys & pool_keys))
    ok("injection_pairkey_0_vs_universe_ds3", not (inj_keys & uni_keys))
    # injection bases: no side touching historical eval surfaces
    hold_sides = set()
    for p in (
        BASE / "v43-holdout/holdout.jsonl",
        BASE / "dataset-v4-holdout/holdout.jsonl",
        BASE / "b2/holdout.jsonl",
    ):
        for r in rows_of(p):
            for s in (r["record"], r["candidate"]):
                hold_sides |= {
                    text_hash(s["title"] + "\n" + s["body"]),
                    text_hash(s["body"]),
                    text_hash(s["title"]),
                }
    gate_sides = set()
    for gp in (
        R4_WT / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4_WT / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ):
        if gp.is_file():
            for r in rows_of(gp):
                for k in ("text_src", "text_dst", "text"):
                    if r.get(k):
                        gate_sides.add(text_hash(r[k]))
    bad_sides = 0
    for r in inj:
        for s in (r["record"], r["candidate"]):
            probes = (
                text_hash(s["title"] + "\n" + s["body"]),
                text_hash(s["body"]),
                text_hash(s["title"]),
            )
            if any(x in hold_sides or x in gate_sides for x in probes):
                bad_sides += 1
    ok("injection_sides_0_vs_holdouts_gates", bad_sides == 0, str(bad_sides))

    # ── fresh holdout ────────────────────────────────────────────────────────
    ok(
        "holdout_new_material_only",
        all(
            r["pair_id"].startswith("V44-")
            or r.get("source") == "b2-graph-pool-230745b"
            for r in hold
        ),
    )
    flags_bad = [
        r["pair_id"]
        for r in hold
        if any((r.get("holdout_side_overlap") or {}).values())
        or r.get("gate_side_overlap")
        or r.get("burned_side_overlap")
    ]
    ok("holdout_rows_flag_free", not flags_bad, str(flags_bad[:3]))
    t_keys = {pair_key(r) for r in train}
    h_keys = {pair_key(r) for r in hold}
    ok("holdout_pairkey_0_vs_train", not (t_keys & h_keys))
    # g5 surfaces = historical eval surfaces (the pool/injection ARE the new
    # material the holdout is carved FROM — identity there is by design)
    ok("holdout_pairkey_0_vs_all_surfaces", not (h_keys & uni_keys))
    side_hits = 0
    for r in hold:
        for s in (r["record"], r["candidate"]):
            probes = (
                text_hash(s["title"] + "\n" + s["body"]),
                text_hash(s["body"]),
                text_hash(s["title"]),
            )
            if any(x in hold_sides or x in gate_sides for x in probes):
                side_hits += 1
    ok("holdout_sides_0_vs_eval_surfaces", side_hits == 0, str(side_hits))
    seal = json.loads((d / "holdout-ids.json").read_text(encoding="utf-8"))
    ok(
        "seal_ids_match",
        seal["holdout_ids"] == sorted(r["pair_id"] for r in hold)
        and seal["holdout_ids_sha256"]
        == hashlib.sha256(
            canonical_json(sorted(r["pair_id"] for r in hold)).encode("utf-8")
        ).hexdigest(),
    )
    # T0 units never split
    t0_units_hold = {r["unit"] for r in hold if r.get("unit")}
    ok(
        "T0_units_atomic",
        not (t0_units_hold & {r["unit"] for r in train if r.get("unit")}),
    )

    # ── holdout carve arithmetic (10% ceil per cell, eval-eligible) ─────────
    all_new = [r for r in train + hold if r.get("source")]
    eligible = [
        r
        for r in all_new
        if not (
            any((r.get("holdout_side_overlap") or {}).values())
            or r.get("gate_side_overlap")
            or r.get("burned_side_overlap")
        )
    ]
    expected_hold_rows = 0
    for cell, meta in manifest["holdout"]["per_cell"].items():
        expected_hold_rows += meta["holdout_rows"]
        ok(
            f"cell_{cell}_ceil10pct",
            meta["holdout_units"] == math.ceil(0.10 * meta["units"]),
            f"{meta['holdout_units']}/{meta['units']}",
        )
    ok(
        "holdout_rows_match_per_cell_sum",
        len(hold) == expected_hold_rows,
        f"{len(hold)} vs {expected_hold_rows}",
    )
    ok(
        "holdout_share_of_eligible",
        abs(len(hold) / len(eligible) - 0.1061) < 0.01,
        f"{len(hold)}/{len(eligible)}",
    )

    print(
        f"\n[verify] {len(CHECKS) - len(FAILS)}/{len(CHECKS)} checks passed",
        file=sys.stderr,
    )
    if FAILS:
        for f in FAILS:
            print(f"[verify] FAILED: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
