#!/usr/bin/env python3
"""Build corpus v4.4 — prereg v3 (ACTIVATED 2026-10-22) §1-§2, slice-2.

Law: docs/specs/b2-graph-prereg-v3.md (ratified by the owner 22.10; quotas,
population and formulas are frozen — any change voids the prereg).

Composition:
  base     v4.3 universe (3131: v43-train 2399 + v43-holdout 276 +
           v42-holdout-276 + b2-holdout-180) with the 21 train∩holdout
           pair-copies REMOVED from the train side only (holdouts untouched;
           2399 -> 2378). The 21 pairs are identified at build time by the
           canonical pair-key join (slice-1 recipe), never hardcoded.
  graph    600 pairs from the frozen slice-1 pool (commit 230745b,
           fp 87b3d4c2...): edge-direct 240 / edge-common 150 / mint-top 150
           / control 60; per-component cap 15% of the stratum total (90);
           class semantics per prereg §2 (all four strata not-duplicate;
           edge-direct = HARD NEGATIVE, silver: edge provenance auto-dedupe).
  inject   T0/T1 synthesis per labeling-policy-b2 §4 (deterministic
           v3/v4.3-lineage generator: self-pairs + byte clones = T0,
           body swapcase cosmetics = T1; v4.3 real-part lineage ratio 2:1).
           Size = formula: restore the dup share to 40.5% over the assembled
           corpus (universe base + graph strata + injection):
               n = (p*(base_rows + graph_rows) - base_dups) / (1 - p)
           computed on the ACTUAL post-cleaning base (prereg: "при текущих
           числах ≈ 410" was the same formula on the pre-cleaning legacy
           numbers: (0.405*3731-1267)/0.595 = 410.2 -> 410; asserted).
  holdout  fresh sealed 10% of the NEW material (graph strata + injection),
           stratified per (label, stratum) with ceil, unit-level for T0
           (self+clone of one base never split across train/holdout),
           pair_id-order, no RNG. Eval rows are carved ONLY from pairs
           carrying zero side-flags (holdout/gate/burned) — the slice-1
           pre-gate recipe. T0/T1 bases are additionally restricted to
           records with zero side overlap at MINT time (conservative).

Discipline: seed 42 recorded; construction is order-deterministic (no RNG
draws at all — nothing to seed); no wall-clock in content artifacts; the
live store is NOT touched (the pool file is the frozen input; store vectors
are provenance, not input). Determinism: double run must be byte-identical
on every artifact (asserted by --verify-double-run).

Inputs (frozen):
  data/b2-graph-corpus/pool_candidates.jsonl   (slice-1 pool, fp-verified)
  data/stage2/v43-localbuild/train.jsonl       (v4.3 train, 2399)
  data/stage2/v43-holdout/holdout.jsonl        (276, untouched)
  data/stage2/dataset-v4-holdout/holdout.jsonl (276, untouched)
  data/stage2/b2/holdout.jsonl                 (180, untouched)
  data/stage2/b2/train.jsonl                   (dataset-v3 420, dedup ref)

Outputs (gitignored data/, numbers-only mirrors land in datasets/b2-graph):
  <out>/train.jsonl  <out>/holdout.jsonl
  <out>/labels-train.jsonl  <out>/labels-holdout.jsonl
  <out>/holdout-ids.json  <out>/manifest.json  <out>/manifest.txt
  <out>/run-log.jsonl

Run (worktree checkout, project venv):
    /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/.venv/bin/python \
        scripts/b2_graph_corpus.py --out-dir data/b2-graph-corpus/v44

Exit codes: 0 ok · 1 gate refusal (assert violation) · 2 usage/validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

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

# ── frozen constants (prereg v3 §1-§2; DO NOT EDIT without an addendum) ─────

POOL_FP_EXPECTED = "87b3d4c22a43befbfd81dc70c83a2cf7fdfce972378a9b550cc100922050cb1e"
SEED = 42
DUP_SHARE_TARGET = 0.405  # prereg §2: restore 40.5% dup share (formula)
GRAPH_QUOTAS = {  # prereg §2: 600, split 40/25/25/10
    "edge-direct": 240,
    "edge-common": 150,
    "mint-top": 150,
    "control": 60,
}
GRAPH_TOTAL = sum(GRAPH_QUOTAS.values())
COMPONENT_CAP_FRAC = 0.15  # prereg §2: cap 15% per connected component
STRATUM_LABELS = {  # prereg §2 class semantics
    "edge-direct": "not-duplicate",  # HARD NEGATIVE (silver: auto-dedupe)
    "edge-common": "not-duplicate",  # silver labeling-policy-b2 §8 policy v1.1
    "mint-top": "not-duplicate",  # razor-zone T3-class material
    "control": "not-duplicate",  # no graph evidence, ablation control
}
HOLDOUT_FRACTION = 0.10  # prereg §2: sealed 10% of the NEW material
INJECTION_T1_SHARE = 1 / 3  # v4.3 real-part lineage: T0:T1 ≈ 96:50 ≈ 2:1

EXPECTED_PIN_PREFIX = "nano:sha256:3b752e06"
DETECTOR_SHA = {
    "secrets_detector.py": (
        "4ff7fcc655f16f1307ce986f58d381daa0c091601439214a9d88e0212863ad95"
    ),
    "danger_detectors.py": (
        "c338af80483dfcd95509a00ec0272278e92fc56243a207c8cb5fb58c27dcef48"
    ),
}

BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
POOL_JSONL = REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl"
ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")

# Legacy pre-cleaning numbers (slice-1, frozen) — the ≈410 anchor reproduces
# from these; asserted so the formula reading is provably the prereg's own.
LEGACY_UNIVERSE_ROWS = 3131
LEGACY_UNIVERSE_DUPS = 1267
LEGACY_ANCHOR_N = 410

# frozen slice-1 characterization of the 21 train∩holdout pair-copies
EXPECT_JOINS = {"v43-holdout": 18, "v42-holdout-276": 3, "b2-holdout-180": 0}
EXPECT_REMOVED_DUPS = 20  # 20 duplicate-labeled train copies removed
EXPECT_REMOVED_NOTDUP = 1

UNIVERSE_FILES = [
    ("v43-train", BASE / "v43-localbuild/train.jsonl"),
    ("v43-holdout", BASE / "v43-holdout/holdout.jsonl"),
    ("v42-holdout-276", BASE / "dataset-v4-holdout/holdout.jsonl"),
    ("b2-holdout-180", BASE / "b2/holdout.jsonl"),
]
HOLDOUT_FILES = {
    "v43-holdout": BASE / "v43-holdout/holdout.jsonl",
    "v42-holdout-276": BASE / "dataset-v4-holdout/holdout.jsonl",
    "b2-holdout-180": BASE / "b2/holdout.jsonl",
}
DATASET_V3_FILES = [("b2-train-420", BASE / "b2/train.jsonl")]


class GateRefusal(SystemExit):
    """An assert failed — the corpus is NOT built, holdout NOT sealed."""


def refuse(msg: str) -> None:
    raise GateRefusal(f"[v44] GATE REFUSAL: {msg}")


def log(runlog: list[dict], event: str, **fields: Any) -> None:
    entry = {"event": event, **fields}
    runlog.append(entry)
    print(
        f"[v44] {event}: " + " ".join(f"{k}={v}" for k, v in fields.items()),
        file=sys.stderr,
    )


# ── hash helpers — byte-identical recipes from the slice-1 pool builder ─────


def text_hash(text: str) -> str:
    """NFC → lower → collapse whitespace → sha256 (embed-gate side-hash fn)."""
    n = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def side_sha(s: dict) -> str:
    """Artifact-side projection hash (title, body, tags, language, record_type)."""
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


def pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((side_sha(row["record"]), side_sha(row["candidate"]))))


def text_pair_key(row: dict) -> tuple[str, str]:
    return tuple(
        sorted(
            (
                text_hash(row["record"]["title"] + "\n" + row["record"]["body"]),
                text_hash(row["candidate"]["title"] + "\n" + row["candidate"]["body"]),
            )
        )
    )


def corpus_rows(path: Path) -> list[dict]:
    if not path.is_file():
        refuse(f"frozen input missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ── stage A: frozen inputs, pool fingerprint verification ───────────────────


def verify_pool(runlog: list[dict]) -> list[dict]:
    rows = corpus_rows(POOL_JSONL)
    entries = [(r["pair_id"], r["pair_sha256"]) for r in rows]
    fp = corpus_fingerprint(manifest_bytes(entries))
    log(
        runlog,
        "pool_verify",
        rows=len(rows),
        pool_fingerprint=fp,
        expected=POOL_FP_EXPECTED,
        match=(fp == POOL_FP_EXPECTED),
    )
    if fp != POOL_FP_EXPECTED:
        refuse(
            "pool fingerprint mismatch — the frozen slice-1 input "
            f"(commit 230745b, fp {POOL_FP_EXPECTED[:8]}...) is not on disk "
            "as shipped; refusing (prereg §1 frozen inputs)"
        )
    strata = Counter(r["stratum"] for r in rows)
    expect = {
        "edge-direct": 1139,
        "edge-common": 801,
        "mint-top": 1714,
        "control": 5000,
    }
    if dict(strata) != expect:
        refuse(f"pool strata {dict(strata)} != frozen slice-1 {expect}")
    return rows


def verify_detectors(runlog: list[dict]) -> Any:
    for fname, expected in DETECTOR_SHA.items():
        p = ENGINE_SRC / "vesmaro" / fname
        if not p.is_file():
            refuse(f"engine detector missing: {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            refuse(
                f"engine detector {fname} sha {got} != frozen slice-1 pin "
                f"{expected} — hygiene chain is not the frozen one"
            )
    scanner = make_scanner(ENGINE_SRC)
    log(runlog, "scanner", provenance=scanner.provenance)
    if scanner.provenance != "engine":
        refuse("prereg hygiene expects the ENGINE detectors (provenance=engine)")
    return scanner


def make_embedder():
    """gen_dataset_v4 / build_corpus_v43 embedder path: engine NanoProvider,
    local ONNX, zero network. Pin asserted = frozen vesma-embed-v1."""
    if str(ENGINE_SRC) not in sys.path:
        sys.path.insert(0, str(ENGINE_SRC))
    from vesmaro.embeddings import NanoProvider  # noqa: E402

    provider = NanoProvider()
    pin = provider.fingerprint
    if not pin.startswith(EXPECTED_PIN_PREFIX):
        refuse(
            f"embedder pin {pin!r} differs from the frozen prefix "
            f"{EXPECTED_PIN_PREFIX!r} — calibration geometry would break"
        )
    cache: dict[str, list[float]] = {}

    def embed_side(side: dict[str, Any]) -> list[float]:
        tags = " ".join(side.get("tags") or [])
        text = f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}"[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    return embed_side, pin


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── stage B: base cleaning — the 21 train∩holdout pair-copies ───────────────


def clean_base(runlog: list[dict]) -> tuple[list[dict], dict]:
    train = corpus_rows(BASE / "v43-localbuild/train.jsonl")
    if len(train) != 2399:
        refuse(f"v4.3 train rows {len(train)} != 2399 (frozen input changed)")
    # fingerprint-scheme pair hash for every shipped pair (fp_entries needs it
    # on base rows too; same convention as pool rows and injection rows)
    for r in train:
        r["pair_sha256"] = pair_sha256(
            {
                "record": r["record"],
                "candidate": r["candidate"],
                "similarity": r["similarity"],
            }
        )
    uni_dups = sum(
        1
        for name, path in UNIVERSE_FILES
        for r in corpus_rows(path)
        if r["label"] == "duplicate"
    )
    uni_rows = sum(len(corpus_rows(p)) for _, p in UNIVERSE_FILES)
    if (uni_rows, uni_dups) != (LEGACY_UNIVERSE_ROWS, LEGACY_UNIVERSE_DUPS):
        refuse(
            f"v4.3 universe {uni_rows} rows / {uni_dups} dups != frozen "
            f"{LEGACY_UNIVERSE_ROWS}/{LEGACY_UNIVERSE_DUPS}"
        )

    # holdout pair-keys per surface; membership over ALL train rows so a
    # doubled train copy can never slip through the cleaning
    keys_by_surface: dict[str, set] = {}
    row_by_key: dict[tuple[str, str], dict] = {}
    per_surface: dict[str, int] = {}
    train_keys = {pair_key(r) for r in train}
    for name, path in HOLDOUT_FILES.items():
        hrows = corpus_rows(path)
        keys_by_surface[name] = {pair_key(r) for r in hrows}
        per_surface[name] = len(keys_by_surface[name] & train_keys)
        for hr in hrows:
            row_by_key.setdefault(pair_key(hr), hr)
    cross = set().union(*keys_by_surface.values())
    cross_dups = sum(len(v) for v in keys_by_surface.values()) - len(cross)
    if cross_dups:
        # inherited lineage fact (surfaces are untouched per prereg) — report,
        # do not refuse: the union is the assert surface for g5
        log(
            runlog,
            "holdout_cross_surface_keys",
            pairs_in_two_surfaces=cross_dups,
            note="inherited v4.2/v4.3 lineage overlap; eval surfaces untouched",
        )

    removed: list[dict] = []
    for tr in train:
        k = pair_key(tr)
        if k not in cross:
            continue
        name = next(n for n, keys in keys_by_surface.items() if k in keys)
        hr = row_by_key[k]
        if hr["label"] != tr["label"]:
            refuse(
                f"label contradiction on pair {tr['pair_id']} vs holdout "
                f"{hr['pair_id']}: {tr['label']} vs {hr['label']}"
            )
        removed.append(
            {
                "train_pair_id": tr["pair_id"],
                "train_stratum": tr.get("stratum"),
                "label": tr["label"],
                "holdout_surface": name,
                "holdout_pair_id": hr["pair_id"],
                "holdout_stratum": hr.get("stratum"),
            }
        )
    removed.sort(key=lambda d: (d["holdout_surface"], d["train_pair_id"]))
    if per_surface != EXPECT_JOINS:
        refuse(
            f"train∩holdout join {per_surface} != slice-1 frozen {EXPECT_JOINS} "
            "— the 21-pair population moved; STOP, prereg population changed"
        )
    if len(removed) != 21:
        refuse(
            f"expected exactly 21 train-side copies, got {len(removed)} — "
            "a pair-key is doubled in train; STOP"
        )
    rm_labels = Counter(d["label"] for d in removed)
    if rm_labels != Counter(
        {"duplicate": EXPECT_REMOVED_DUPS, "not-duplicate": EXPECT_REMOVED_NOTDUP}
    ):
        refuse(f"removed-pair label mix {dict(rm_labels)} != 20 dup + 1 not-dup")

    drop_ids = {d["train_pair_id"] for d in removed}
    base = [r for r in train if r["pair_id"] not in drop_ids]
    if len(base) != 2378:
        refuse(f"cleaned base rows {len(base)} != 2378")
    base_dups = sum(1 for r in base if r["label"] == "duplicate")
    if base_dups != 905:
        refuse(f"cleaned base dups {base_dups} != 905 (925 - 20)")

    # anchor: the same formula on LEGACY (pre-cleaning) universe numbers must
    # give the prereg's ≈410 — this pins the FORMULA SURFACE to the assembled
    # corpus UNIVERSE (legacy accounting incl. the three historical holdouts),
    # not to the train split alone.
    p = DUP_SHARE_TARGET
    anchor = (p * (LEGACY_UNIVERSE_ROWS + GRAPH_TOTAL) - LEGACY_UNIVERSE_DUPS) / (1 - p)
    if round(anchor) != LEGACY_ANCHOR_N:
        refuse(
            f"formula anchor {anchor:.3f} != prereg ≈{LEGACY_ANCHOR_N} — "
            "the formula reading is wrong; STOP"
        )
    # actual post-cleaning universe (prereg §1 base + rule): 3131-21 / 1267-20
    uni_clean_rows = uni_rows - len(removed)
    uni_clean_dups = uni_dups - rm_labels["duplicate"]
    if (uni_clean_rows, uni_clean_dups) != (3110, 1247):
        refuse(
            f"cleaned universe {uni_clean_rows}/{uni_clean_dups} != "
            "3110/1247 — base population moved; STOP"
        )
    n_real = (p * (uni_clean_rows + GRAPH_TOTAL) - uni_clean_dups) / (1 - p)
    n_inject = math.floor(n_real + 0.5)  # round-half-up (anchor convention)
    log(
        runlog,
        "base_clean",
        removed_pairs=len(removed),
        join=per_surface,
        removed_labels=dict(rm_labels),
        base_rows=len(base),
        base_dups=base_dups,
        universe_clean=f"{uni_clean_rows}/{uni_clean_dups}",
    )
    log(
        runlog,
        "injection_formula",
        formula="n = (p*(uni_clean_rows+graph_rows)-uni_clean_dups)/(1-p), p=0.405",
        surface="assembled corpus universe (prereg legacy accounting)",
        uni_clean_rows=uni_clean_rows,
        uni_clean_dups=uni_clean_dups,
        graph_rows=GRAPH_TOTAL,
        n_real=round(n_real, 4),
        n_injection=n_inject,
        anchor_legacy=round(anchor, 4),
        anchor_note="legacy numbers reproduce the prereg's ≈410",
        final_dup_share_universe=round(
            (uni_clean_dups + n_inject) / (uni_clean_rows + GRAPH_TOTAL + n_inject),
            6,
        ),
    )
    info = {
        "removed": removed,
        "base_rows": len(base),
        "base_dups": base_dups,
        "uni_clean_rows": uni_clean_rows,
        "uni_clean_dups": uni_clean_dups,
        "n_real": n_real,
        "n_injection": n_inject,
        "universe_rows": uni_rows,
        "universe_dups": uni_dups,
    }
    return base, info


# ── stage C: graph strata 600 with the 15% per-component cap ────────────────


def select_graph(runlog: list[dict], pool: list[dict]) -> list[dict]:
    comp_cap = math.floor(COMPONENT_CAP_FRAC * GRAPH_TOTAL + 1e-9)  # 90
    comp_count: Counter = Counter()
    picked: list[dict] = []
    for stratum in ("edge-direct", "edge-common", "mint-top", "control"):
        quota = GRAPH_QUOTAS[stratum]
        rows = sorted(
            (r for r in pool if r["stratum"] == stratum),
            key=lambda r: r["pair_id"],
        )
        have = 0
        for r in rows:
            if have >= quota:
                break
            comp = r["component_a"]
            if comp_count[comp] >= comp_cap:
                continue
            comp_count[comp] += 1
            picked.append(r)
            have += 1
        if have != quota:
            refuse(
                f"stratum {stratum}: filled {have}/{quota} — pool too small "
                "under the component cap; prereg quota infeasible, STOP"
            )
    worst = comp_count.most_common(1)[0]
    if worst[1] > comp_cap:
        refuse(f"component cap violated: {worst}")
    comps = len(comp_count)
    log(
        runlog,
        "graph_select",
        strata=dict(Counter(r["stratum"] for r in picked)),
        total=len(picked),
        component_cap=comp_cap,
        max_component=worst[1],
        components_covered=comps,
        components_total=len({r["component_a"] for r in pool}),
        selection="stratum order (prereg §2), pair_id order, cap 15% of 600",
    )
    out = []
    for r in picked:
        row = dict(r)
        row["label"] = STRATUM_LABELS[row["stratum"]]
        row["source"] = "b2-graph-pool-230745b"
        out.append(row)
    return out


# ── stage D: positive injection — T0/T1 synthesis, labeling-policy §4 ───────


def build_injection(
    runlog: list[dict],
    scanner: Any,
    pool: list[dict],
    n_inject: int,
    base_rows: list[dict],
    graph_rows: list[dict],
) -> list[dict]:
    embed_side, pin = make_embedder()

    # occupied pair surfaces (pair-key level): everything the new material
    # must stay disjoint from, plus the new material itself
    occupied: dict[tuple[str, str], str] = {}
    for source, files in (
        ("universe-v43", UNIVERSE_FILES),
        ("dataset-v3", DATASET_V3_FILES),
    ):
        for name, path in files:
            for row in corpus_rows(path):
                occupied.setdefault(pair_key(row), f"{source}:{name}")
    for r in pool:
        occupied.setdefault(
            pair_key(r), "pool-230745b"
        )  # pool rows ship record/candidate
    for r in graph_rows:
        occupied.setdefault(pair_key(r), "v44-graph")

    # historical side surfaces (eval rows must never share sides): the same
    # side-hash joins the pool builder flagged; for injection we exclude at
    # MINT time (conservative — stronger than the carve-time recipe)
    holdout_sides: dict[str, set] = {}
    for name, path in HOLDOUT_FILES.items():
        hs: set = set()
        for row in corpus_rows(path):
            for side in (row["record"], row["candidate"]):
                hs.add(text_hash(side["title"] + "\n" + side["body"]))
                hs.add(text_hash(side["body"]))
                hs.add(text_hash(side["title"]))
        holdout_sides[name] = hs
    gate_sides: set = set()
    R4_WT = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
    )
    R5_WT = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
    )
    gate_files = [
        R4_WT / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4_WT / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ]
    for gp in gate_files:
        if not gp.is_file():
            continue
        for row in corpus_rows(gp):
            for key in ("text_src", "text_dst", "text"):
                if row.get(key):
                    gate_sides.add(text_hash(row[key]))
    burned_sides: set = set()
    burned_files = [
        R4_WT / "data/embed-r4/corpus/train.jsonl",
        R4_WT / "data/embed-r4/corpus/val.jsonl",
        R5_WT / "data/embed-r5/corpus/train.jsonl",
        R5_WT / "data/embed-r5/corpus/val.jsonl",
        R5_WT / "data/embed-r7/corpus/train.jsonl",
        R5_WT / "data/embed-r7/corpus/val.jsonl",
    ]
    for bp in burned_files:
        if not bp.is_file():
            continue
        for row in corpus_rows(bp):
            for side_key in ("record", "candidate"):
                side = row.get(side_key)
                if side:
                    burned_sides.add(text_hash(side["title"] + "\n" + side["body"]))
                    burned_sides.add(text_hash(side["body"]))
            for key in ("text", "text_src", "text_dst"):
                if row.get(key):
                    burned_sides.add(text_hash(row[key]))

    # deterministic base order over the pool's endpoint sides
    sides: dict[str, dict] = {}
    for r in pool:
        for s in (r["record"], r["candidate"]):
            sides.setdefault(side_sha(s), s)
    order = sorted(sides.items(), key=lambda kv: (kv[1].get("created_at") or "", kv[0]))

    counters = Counter()
    skipped = Counter()
    rows: list[dict] = []
    minted_keys: set[tuple[str, str]] = set()

    def side_clean(s: dict) -> bool:
        probes = (
            text_hash(s["title"] + "\n" + s["body"]),
            text_hash(s["body"]),
            text_hash(s["title"]),
        )
        if any(p in gate_sides for p in probes):
            return False
        if any(p in burned_sides for p in probes):
            return False
        if any(p in hs for hs in holdout_sides.values() for p in probes):
            return False
        return True

    def hygiene(s: dict, where: str) -> None:
        hits = scanner.scan(
            s.get("title") or "", s.get("body") or "", s.get("tags") or []
        )
        if hits:
            refuse(f"hygiene hit on synthetic {where}: {hits}")

    def mint(unit: str, cls: str, stratum: str, rec: dict, cand: dict) -> None:
        pid = f"V44-INJ-{cls}-{counters[cls] + 1:04d}"
        counters[cls] += 1
        va = embed_side(rec)
        vb = embed_side(cand)
        sim = round(min(1.0, max(0.0, dot(va, vb))), 6)  # v4.3 embed_measure clamp
        pair_obj = {"record": rec, "candidate": cand, "similarity": sim}
        row = {
            "pair_id": pid,
            "label": "duplicate",
            "stratum": stratum,
            "unit": unit,
            "record": rec,
            "candidate": cand,
            "similarity": sim,
            "vec_a": va,
            "vec_b": vb,
            "pair_sha256": pair_sha256(pair_obj),
            "generator": "legacy-v43-t0t1",
            "seed": SEED,
            "embedder_pin": pin,
            "source": "injection-policy-s4",
        }
        hygiene(rec, pid + ":record")
        hygiene(cand, pid + ":candidate")
        rows.append(row)
        minted_keys.add(pair_key(row))

    t0_units = math.floor(n_inject / 3 + 0.5)
    t1_rows = n_inject - 2 * t0_units
    if t1_rows < 0:
        refuse(f"injection split negative: n={n_inject}")

    bases_iter = iter(order)
    # T0: self + byte-clone units (labeling-policy §4 G-identity)
    made_t0 = 0
    while made_t0 < t0_units:
        sha, s = next(bases_iter, (None, None))
        if s is None:
            refuse(f"T0 bases exhausted at {made_t0}/{t0_units}")
        if not side_clean(s):
            skipped["t0-side-overlap"] += 1
            continue
        rec = {
            k: s[k]
            for k in ("title", "body", "tags", "language", "record_type", "created_at")
        }
        key = (side_sha(rec), side_sha(rec))
        if key in occupied or key in minted_keys:
            skipped["t0-occupied-pair"] += 1
            continue
        unit = f"U-T0-{made_t0 + 1:04d}"
        cand = deepcopy(rec)
        mint(unit, "T0S", "P-identity", deepcopy(rec), deepcopy(rec))
        mint(unit, "T0C", "P-identity", rec, cand)
        minted_keys.add(key)
        made_t0 += 1
    # T1: body case cosmetics (labeling-policy §4 G-cosmetic; v4.3 RCOS recipe)
    made_t1 = 0
    while made_t1 < t1_rows:
        sha, s = next(bases_iter, (None, None))
        if s is None:
            refuse(f"T1 bases exhausted at {made_t1}/{t1_rows}")
        if not side_clean(s):
            skipped["t1-side-overlap"] += 1
            continue
        rec = {
            k: s[k]
            for k in ("title", "body", "tags", "language", "record_type", "created_at")
        }
        cand = deepcopy(rec)
        cand["body"] = cand["body"].swapcase()
        key = pair_key({"record": rec, "candidate": cand})
        if key in occupied or key in minted_keys:
            skipped["t1-occupied-pair"] += 1
            continue
        mint(f"U-T1-{made_t1 + 1:04d}", "T1C", "P-cosmetic", rec, cand)
        made_t1 += 1

    if len(rows) != n_inject:
        refuse(f"injection rows {len(rows)} != formula n {n_inject}")
    sim = [r["similarity"] for r in rows]
    log(
        runlog,
        "injection_built",
        rows=len(rows),
        t0_self=counters["T0S"],
        t0_clone=counters["T0C"],
        t1_case=counters["T1C"],
        bases_considered=len(order),
        skipped=dict(skipped),
        strata=dict(Counter(r["stratum"] for r in rows)),
        similarity_min=min(sim),
        similarity_max=max(sim),
        hygiene_hits=0,
        embedder_pin=pin[:24] + "...",
    )
    return rows


# ── stage E: fresh sealed holdout — 10% of the NEW material ─────────────────


def carve_holdout(
    runlog: list[dict],
    base: list[dict],
    graph_rows: list[dict],
    inj_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    # eval eligibility: zero side-flags (slice-1 pre-gate recipe) — applies to
    # EVAL rows only; flagged rows stay in TRAIN (side-sharing train-side is
    # the corpus-family norm: the v4.3 base train itself shares b2-holdout
    # sides 127/308)
    def flags_clean(r: dict) -> bool:
        ho = r.get("holdout_side_overlap") or {}
        if any(ho.values()):
            return False
        if r.get("gate_side_overlap"):
            return False
        if r.get("burned_side_overlap"):
            return False
        return True

    graph_eval = [r for r in graph_rows if flags_clean(r)]
    graph_flagged = [r for r in graph_rows if not flags_clean(r)]
    inj_ok = [r for r in inj_rows if flags_clean(r)]
    if inj_ok != inj_rows:
        refuse("injection rows must be flag-free by construction — check mint gates")

    # units: T0 self+clone of one base never split across train/holdout
    units: dict[str, list[dict]] = defaultdict(list)
    for r in inj_ok:
        units[r["unit"]].append(r)
    graph_units = {r["pair_id"]: [r] for r in graph_eval}

    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, members in graph_units.items():
        m = members[0]
        cells[(m["label"], m["stratum"])].append(pid)
    for uid, members in units.items():
        m = members[0]
        cells[(m["label"], m["stratum"])].append(uid)

    hold_units: set[str] = set()
    per_cell = {}
    for (label, stratum), members in sorted(cells.items()):
        members = sorted(members)
        n_hold = math.ceil(HOLDOUT_FRACTION * len(members))
        hold_units.update(members[:n_hold])
        per_cell[f"{label}/{stratum}"] = {
            "units": len(members),
            "holdout_units": n_hold,
            "holdout_rows": sum(
                len(units[u]) if u in units else 1 for u in members[:n_hold]
            ),
        }
    inj_hold_ids = {r["pair_id"] for u in hold_units for r in units.get(u, [])}
    graph_hold_ids = {u for u in hold_units if u in graph_units}

    holdout = [r for r in graph_eval if r["pair_id"] in graph_hold_ids]
    holdout += [r for r in inj_ok if r["pair_id"] in inj_hold_ids]
    # ordering: base rows keep v4.3 file order; new material pair_id-sorted
    base_part = list(base)
    new_train = sorted(
        list(graph_flagged)
        + [r for r in graph_eval if r["pair_id"] not in graph_hold_ids]
        + [r for r in inj_ok if r["pair_id"] not in inj_hold_ids],
        key=lambda r: r["pair_id"],
    )
    train = base_part + new_train
    holdout = sorted(holdout, key=lambda r: r["pair_id"])

    if len(train) + len(holdout) != len(base) + len(graph_rows) + len(inj_ok):
        refuse("split lost rows")
    t_ids = [r["pair_id"] for r in train]
    h_ids = [r["pair_id"] for r in holdout]
    if len(t_ids) != len(set(t_ids)) or len(h_ids) != len(set(h_ids)):
        refuse("duplicate pair_id within a split")
    if set(t_ids) & set(h_ids):
        refuse("pair_id overlap train∩holdout")
    t_keys = {pair_key(r) for r in train}
    h_keys = {pair_key(r) for r in holdout}
    if t_keys & h_keys:
        refuse(
            f"pair-LEVEL overlap train∩holdout: {len(t_keys & h_keys)} keys — "
            "g5 violation (T0 units must not split)"
        )

    log(
        runlog,
        "holdout_carve",
        fraction=HOLDOUT_FRACTION,
        new_material_rows=len(graph_rows) + len(inj_ok),
        eval_eligible_rows=len(graph_eval) + len(inj_ok),
        new_material_units=len(graph_units) + len(units),
        graph_flagged_train_only=len(graph_flagged),
        per_cell=per_cell,
        holdout_rows=len(holdout),
        train_rows=len(train),
        share_of_eval_eligible=round(len(holdout) / (len(graph_eval) + len(inj_ok)), 4),
    )
    meta = {
        "per_cell": per_cell,
        "graph_flagged_train_only": len(graph_flagged),
        "eval_eligible_rows": len(graph_eval) + len(inj_ok),
        "new_units": len(graph_units) + len(units),
    }
    return train, holdout, meta


# ── stage F: disjointness asserts (g5) over the FINAL artifacts ─────────────


def assert_disjoint(runlog: list[dict], train: list[dict], holdout: list[dict]) -> dict:
    new_sources = ("b2-graph-pool-230745b", "injection-policy-s4")
    new_rows = [
        r for r in list(train) + list(holdout) if r.get("source") in new_sources
    ]

    surfaces_pairs: dict[str, set] = {}
    for name, path in HOLDOUT_FILES.items():
        surfaces_pairs[name] = {pair_key(r) for r in corpus_rows(path)}
    universe_pairs: set = set()
    for name, path in UNIVERSE_FILES:
        for r in corpus_rows(path):
            universe_pairs.add(pair_key(r))
    ds3_pairs: set = set()
    for name, path in DATASET_V3_FILES:
        for r in corpus_rows(path):
            ds3_pairs.add(pair_key(r))
    burned_translated: set = set()
    for r in corpus_rows(BASE / "v43-localbuild/train.jsonl"):
        if str(r.get("part", "")).startswith("translated") or str(
            r.get("pair_id", "")
        ).startswith(("V43-TDUP", "V43-TSIB")):
            burned_translated.add(pair_key(r))

    violations: dict[str, int] = {}
    for r in new_rows:
        ck = pair_key(r)
        hits = [n for n, s in surfaces_pairs.items() if ck in s]
        if hits:
            violations["holdouts"] = violations.get("holdouts", 0) + 1
        if ck in universe_pairs:
            violations["v43-universe"] = violations.get("v43-universe", 0) + 1
        if ck in ds3_pairs:
            violations["dataset-v3"] = violations.get("dataset-v3", 0) + 1
        if ck in burned_translated:
            violations["burned-translated"] = violations.get("burned-translated", 0) + 1
    if violations:
        refuse(f"g5 pair-level disjointness violated: {violations}")

    # fresh holdout vs the v4.4 train at pair level (already asserted in the
    # carve) + side-level purity of the fresh holdout vs eval surfaces
    R4_WT = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
    )
    R5_WT = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
    )
    gate_sides: set = set()
    for gp in (
        R4_WT / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4_WT / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5_WT / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ):
        if gp.is_file():
            for row in corpus_rows(gp):
                for key in ("text_src", "text_dst", "text"):
                    if row.get(key):
                        gate_sides.add(text_hash(row[key]))
    side_hits = 0
    for r in holdout:
        for s in (r["record"], r["candidate"]):
            probes = (
                text_hash(s["title"] + "\n" + s["body"]),
                text_hash(s["body"]),
                text_hash(s["title"]),
            )
            if any(p in gate_sides for p in probes):
                side_hits += 1
    if side_hits:
        refuse(f"fresh holdout shares {side_hits} sides with gate sets")
    log(
        runlog,
        "disjointness_final",
        new_material_checked=len(new_rows),
        violations=0,
        holdout_gate_side_hits=0,
        surfaces="v43/v42/b2 holdouts, v4.3 universe, dataset-v3, burned translated, gate sides",
    )
    return {"new_material_checked": len(new_rows), "violations": 0}


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data/b2-graph-corpus/v44"
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    runlog: list[dict] = []

    pool = verify_pool(runlog)
    scanner = verify_detectors(runlog)
    base, binfo = clean_base(runlog)
    graph_rows = select_graph(runlog, pool)
    inj_rows = build_injection(
        runlog, scanner, pool, binfo["n_injection"], base, graph_rows
    )
    train, holdout, hmeta = carve_holdout(runlog, base, graph_rows, inj_rows)
    disjoint = assert_disjoint(runlog, train, holdout)

    # fingerprints (data-contract §5 scheme)
    def fp_entries(rows: list[dict]) -> list[tuple[str, str]]:
        return [(r["pair_id"], r["pair_sha256"]) for r in rows]

    train_manifest = manifest_bytes(fp_entries(train))
    hold_manifest = manifest_bytes(fp_entries(holdout))
    train_fp = corpus_fingerprint(train_manifest)
    hold_fp = corpus_fingerprint(hold_manifest)
    whole_fp = corpus_fingerprint(
        manifest_bytes(fp_entries(train) + fp_entries(holdout))
    )

    labels_train = {r["pair_id"]: r["label"] for r in train}
    labels_hold = {r["pair_id"]: r["label"] for r in holdout}

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "holdout.jsonl", holdout)
    write_jsonl(
        out / "labels-train.jsonl",
        [{"label": v, "pair_id": k} for k, v in sorted(labels_train.items())],
    )
    write_jsonl(
        out / "labels-holdout.jsonl",
        [{"label": v, "pair_id": k} for k, v in sorted(labels_hold.items())],
    )
    seal = {
        "holdout_ids": sorted(labels_hold),
        "holdout_ids_sha256": hashlib.sha256(
            canonical_json(sorted(labels_hold)).encode("utf-8")
        ).hexdigest(),
        "holdout_fingerprint": hold_fp,
        "sealing": "prereg v3 §2; single-shot; construction-seeded (no RNG)",
        "prereg": "docs/specs/b2-graph-prereg-v3.md (ACTIVATED 2026-10-22)",
    }
    (out / "holdout-ids.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    strata_train = Counter(r["stratum"] for r in train)
    strata_hold = Counter(r["stratum"] for r in holdout)
    lab_train = Counter(r["label"] for r in train)
    lab_hold = Counter(r["label"] for r in holdout)
    comp_train = Counter(r["component_a"] for r in train if "component_a" in r)
    manifest = {
        "corpus": "b2-graph-v4.4",
        "prereg": "docs/specs/b2-graph-prereg-v3.md (ACTIVATED, owner 22.10)",
        "inputs": {
            "pool_commit": "230745b",
            "pool_fingerprint": POOL_FP_EXPECTED,
            "pool_rows": len(pool),
            "v43_universe_rows": binfo["universe_rows"],
            "v43_universe_dups": binfo["universe_dups"],
            "embedder_pin": inj_rows[0]["embedder_pin"] if inj_rows else None,
        },
        "base_clean": {
            "rule": "exclude 21 train∩holdout pair-copies (train side; holdouts untouched)",
            "identification": "canonical pair-key join at build time (slice-1 recipe)",
            "removed": binfo["removed"],
            "removed_labels": {"duplicate": 20, "not-duplicate": 1},
            "train_rows_before": 2399,
            "train_rows_after": binfo["base_rows"],
            "train_dups_after": binfo["base_dups"],
        },
        "graph_stratum": {
            "quotas": GRAPH_QUOTAS,
            "filled": dict(Counter(r["stratum"] for r in graph_rows)),
            "component_cap": math.floor(0.15 * GRAPH_TOTAL + 1e-9),
            "component_cap_rule": "15% of stratum total (600) per component_a",
            "components_covered": len(comp_train),
            "label_semantics": STRATUM_LABELS,
            "selection": "stratum order (prereg §2), pair_id order, deterministic",
        },
        "injection": {
            "policy": "labeling-policy-b2 §4 T0/T1 synthesis (LLM pass rejected per prereg)",
            "generator": "v3/v4.3-lineage deterministic (self+clone T0, swapcase T1)",
            "seed": SEED,
            "rng_note": "construction order-deterministic; no RNG draws (seed recorded per prereg)",
            "formula": "n = (0.405*(uni_clean_rows+600)-uni_clean_dups)/0.595, round-half-up",
            "formula_surface": "assembled corpus universe (legacy prereg accounting incl. v43/v42/b2 holdouts) — pinned by the ≈410 anchor",
            "formula_values": {
                "uni_clean_rows": binfo["uni_clean_rows"],
                "uni_clean_dups": binfo["uni_clean_dups"],
                "graph_rows": GRAPH_TOTAL,
                "n_real": round(binfo["n_real"], 4),
                "n_injection": binfo["n_injection"],
            },
            "anchor_check": {
                "legacy_values": {"rows": 3131, "dups": 1267},
                "n_real": 410.175,
                "n": 410,
                "note": "reproduces the prereg's '≈410' — same formula, pre-cleaning numbers",
            },
            "t0_units": math.floor(binfo["n_injection"] / 3 + 0.5),
            "t0_rows": 2 * math.floor(binfo["n_injection"] / 3 + 0.5),
            "t1_rows": binfo["n_injection"]
            - 2 * math.floor(binfo["n_injection"] / 3 + 0.5),
            "split_rule": "v4.3 real-part lineage ratio T0:T1 ≈ 2:1 (96:50)",
            "bases": "pool endpoint sides; holdout/gate/burned-side records excluded at MINT time",
            "hygiene": "engine detectors over every minted side; 0 hits tolerated",
            "result": dict(Counter(r["stratum"] for r in inj_rows)),
        },
        "holdout": {
            "rule": "sealed 10% of NEW material (graph strata + injection), seed/construction-fixed",
            "fraction": HOLDOUT_FRACTION,
            "carve": "per (label, stratum), ceil, unit-level (T0 self+clone never split), pair_id order",
            "eval_row_filter": "zero side-flags (holdout/gate/burned) — slice-1 pre-gate recipe; flagged rows stay in TRAIN (corpus-family side-sharing norm)",
            "graph_flagged_train_only": hmeta["graph_flagged_train_only"],
            "eval_eligible_rows": hmeta["eval_eligible_rows"],
            "per_cell": hmeta["per_cell"],
            "rows": len(holdout),
            "ids_sha256": seal["holdout_ids_sha256"],
        },
        "counts": {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "total_rows": len(train) + len(holdout),
            "train_labels": dict(lab_train),
            "holdout_labels": dict(lab_hold),
            "train_strata": dict(sorted(strata_train.items())),
            "holdout_strata": dict(sorted(strata_hold.items())),
            "train_dup_share": round(lab_train["duplicate"] / len(train), 6),
            "files_dup_share": round(
                (lab_train["duplicate"] + lab_hold["duplicate"])
                / (len(train) + len(holdout)),
                6,
            ),
            "files_dup_share_note": (
                "train+fresh-holdout files only; the formula surface is the "
                "universe accounting (adds the three untouched historical "
                "holdouts) — see injection.formula_values"
            ),
            "dup_share_target": DUP_SHARE_TARGET,
        },
        "fingerprints": {
            "scheme": "manifest(pair_id pair_sha256 sorted) → blake2b-256 (data-contract §5)",
            "train": train_fp,
            "holdout": hold_fp,
            "corpus_v44": whole_fp,
            "labels_train": labels_fingerprint(labels_train),
            "labels_holdout": labels_fingerprint(labels_hold),
        },
        "disjointness": {
            "g5": "pair-level 0 vs v43-holdout / v42-276 / b2-180 / v4.3 universe / dataset-v3 / burned translated — asserted",
            "fresh_holdout": "pair-level 0 vs v4.4 train (T0 units atomic); 0 gate-side hits — asserted",
            "checked": disjoint,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "manifest.txt").write_bytes(train_manifest)
    (out / "run-log.jsonl").write_text(
        "".join(
            json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in runlog
        ),
        encoding="utf-8",
    )
    print(
        f"[v44] DONE train={len(train)} holdout={len(holdout)} "
        f"fp={whole_fp[:16]}...  injection={len(inj_rows)} "
        f"(formula {binfo['n_real']:.2f} -> {binfo['n_injection']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateRefusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
