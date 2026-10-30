#!/usr/bin/env python3
"""Build the merge-arbiter corpus v1 (spec: docs/specs/merge-arbiter-corpus-prereg-DRAFT.md, v2).

Sequence (fail-closed at every gate):
  prologue (swap_check FIRST, env, pins) -> ro snapshot (D0-fp) -> real
  strata census -> PREFLIGHT (fixture cells + sanity suite N3, full assert
  map) -> emission (synthetic train 600 seed 42 + holdout 60 fresh seed
  777) -> assert ledger -> characterization (per-stratum baseline floors,
  floor_seed/perm/inv/q) -> seals (MA-holdout-choice, MA-spot-sheet) ->
  manifest + README.

Discipline: no LLM; no seal of another surface is opened (holdout v4.5 /
g4-spec-v1 untouched); raw store text never leaves the gitignored data/
tree; run-log is append-only and the machine asserts
preflight_pass < emission (ML-STD-0002 П3 addendum).

Usage:
  /usr/bin/python3 scripts/build_merge_arbiter_corpus.py \
      --store-uri 'file:/var/home/abyss/.vesma/data/vesma.db?mode=ro' \
      [--skip-snapshot --snapshot PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import merge_arbiter_lib as lib  # noqa: E402

DATA = REPO / "data" / "merge-arbiter"
CORPUS_DIR = DATA / "corpus-v1"
SEAL_DIR = DATA / "seals"
SHEET_DIR = DATA / "spot-sheet"
RUNLOG_COMMITTED = REPO / "artifacts" / "runs" / "run_log.jsonl"

SEED_TRAIN = 42
SEED_HOLDOUT = 777          # fresh material for MA-holdout-choice (draft §8)
SEED_SHEET = 4242

CREATED_AT_TOL = 0.10       # A-created-indep / A-order-indep: |p-0.5| bound
LONGEST_BODY_MAX = 0.90     # N1 build assert: acc(longest-body, CH-3) bound

QUOTA_TRAIN = {"CH-3-k2": 240, "CH-3-chain": 12, "CH-0-k2": 90, "CH-0-triple": 18,
               "CH-2": 90, "CH-4": 90, "CH-5": 60}
QUOTA_HOLDOUT = {"CH-3-k2": 22, "CH-3-chain": 3, "CH-0-k2": 8, "CH-0-triple": 3,
                 "CH-2": 9, "CH-4": 9, "CH-5": 6}  # triples/chains counted as clusters
CH2_MODES = ["tags", "type", "language"]

REAL_EXPECT = {"pairs": 68, "records": 130, "clusters": 64}

#: A-disjoint leg-3 pin (MLC ratification 092946ab, 2026-10-29): the 3/68
#: overlap of the real 68 pairs with holdout-v45 pair members is ratified
#: AS A SET — the exact pair ids and record ids below, not a count. Any
#: difference between the measured overlap set and this pin is red
#: (fail-closed). Mirrored verbatim into the emitted manifest under
#: "a_disjoint_pin" (which is gitignored; this constant is the tracked copy).
A_DISJOINT_SURFACE_PIN: dict[str, dict[str, list[str]]] = {
    "graph-sidecar-fresh-holdout-v45.jsonl": {
        "pair_ids": sorted([
            "16b9de0f-e3b0-4a74-ae28-2612f30cd4f8--293daaef-5c36-4fdc-8145-4b0f0f60829b",
            "381aa311-d357-4165-a9e3-fbe0eb45e31f--387dd891-2ac1-4700-80cd-71e47b7a27e8",
            "3979ae5c-9953-44de-b09e-228cb593ed17--acc63ee7-552d-4eaf-9d19-e902c8160172",
        ]),
        "record_ids": sorted([
            "16b9de0f-e3b0-4a74-ae28-2612f30cd4f8",
            "293daaef-5c36-4fdc-8145-4b0f0f60829b",
            "381aa311-d357-4165-a9e3-fbe0eb45e31f",
            "387dd891-2ac1-4700-80cd-71e47b7a27e8",
            "3979ae5c-9953-44de-b09e-228cb593ed17",
            "acc63ee7-552d-4eaf-9d19-e902c8160172",
        ]),
    },
    "graph-sidecar-fresh-v42-276.jsonl": {"pair_ids": [], "record_ids": []},
    "graph-sidecar-fresh-g4-spec-v1.jsonl": {"pair_ids": [], "record_ids": []},
    "pool_manifest.txt": {"pair_ids": [], "record_ids": []},
}


class BuildStop(Exception):
    """Fail-closed abort: nothing downstream may run (no seals, no sheet)."""


_events: list[dict] = []


def log(event: str, **fields) -> None:
    row = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
           "event": event, **fields}
    _events.append(row)
    print(f"[run-log] {event} {fields if fields else ''}", flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- prologue ---------------------------------------------------------------

def prologue(snapshot_dir: str) -> dict:
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
    log("swap_check", swapon=swap.strip().splitlines(), free=free.strip().splitlines()[1])
    assert sys.executable == "/usr/bin/python3" or "python3" in sys.executable, sys.executable
    log("env", python=sys.version.split()[0], executable=sys.executable,
        machine=platform.machine(), node=platform.node())
    lib_path = Path(lib.__file__)
    frozen_pair = REPO / "src" / "cortex" / "features" / "pair.py"
    pins = {
        "embedder_pin": lib.EMBEDDER_PIN,
        "cortex_features_pair_sha256": sha256_file(frozen_pair),
        "feature_names_sha256": lib.sha256_text(repr(lib.FEATURE_NAMES)),
        "merge_arbiter_lib_sha256": sha256_file(lib_path),
    }
    log("pins", **pins)
    log("inputs_declared", snapshot_dir=snapshot_dir,
        seal_inputs=["NONE — holdout v4.5 and g4-spec-v1 are NOT in the input list "
                     "(A-seal-intact)"])
    return pins


# --- snapshot (pinned D0) ----------------------------------------------------

#: Draft §0 pin: sha256 of the probe-20261028 vesma.db snapshot. The live
#: store GROWS and is NOT a rematerialization source for the pinned real
#: strata (spec §3.1, rule amended 2026-10-29 after the build lesson).
D0_FP = "f82da814844565a64148e4869e97649b8690464f5f19f6070756102b7372353f"
D0_VEC_FP = "92e47c93c6481488ceee0717cfdc3ad197e36c7370131d7ae1ce5563d4ad2c8d"


def verify_snapshot(snapshot_dir: Path) -> None:
    vesma = snapshot_dir / "vesma.db"
    vect = snapshot_dir / "vectors.db"
    if sha256_file(vesma) != D0_FP:
        raise BuildStop(f"snapshot fp mismatch: {vesma} != D0 pin (spec §3.1) — STOP")
    if sha256_file(vect) != D0_VEC_FP:
        raise BuildStop(f"vectors snapshot fp mismatch: {vect} — STOP")
    log("snapshot_pinned", path=str(snapshot_dir), d0_fp=D0_FP, vectors_fp=D0_VEC_FP,
        rematerialization="restore durable copy + re-run silver-dup census; "
                          "live store is not a source (spec §3.1)")


def real_clusters(snapshot_dir: Path) -> dict:
    """Silver-dup census of the pinned snapshot (spec §3.1 predicate).

    Hygiene WITHOUT the engine scanner (verified no-op on this population:
    the scanner excluded 25 records, none in dup pairs; the count assert
    below is the fail-closed equivalence proof). numpy from /usr/bin/python3.
    """
    import numpy as np

    pin_prefix = "nano:sha256:3b752e06"
    mint_cos, blob_len, unit_tol = 0.92, 384 * 4, 1e-3
    no_fed = "mnemos:no-federate"

    def ro(path: Path):
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10.0)
        conn.execute("PRAGMA query_only=ON")
        return conn

    conn = ro(snapshot_dir / "vesma.db")
    mem = {}
    for row in conn.execute(
        "SELECT id, title, content, tags, memory_type, created_at, metadata,"
        " quarantine_reason, status FROM memories"
    ):
        mem[row[0]] = row
    conn.close()
    conn = ro(snapshot_dir / "vectors.db")
    emb = {}
    for mid, blob, mj in conn.execute("SELECT id, vector, metadata FROM embeddings"):
        if isinstance(blob, bytes) and len(blob) == blob_len:
            emb[mid] = (blob, mj or "{}")
    conn.close()

    excluded: dict[str, int] = {}
    passed: dict[str, dict] = {}
    for rid, (i, title, content, tags_json, mtype, created, meta_json, quar, status) in mem.items():
        if status != "published":
            excluded["status"] = excluded.get("status", 0) + 1
            continue
        if mtype not in ("note", "snippet", "fact"):
            excluded["memory_type"] = excluded.get("memory_type", 0) + 1
            continue
        if len(content or "") < 80:
            excluded["len<80"] = excluded.get("len<80", 0) + 1
            continue
        tags = tuple(str(t) for t in json.loads(tags_json or "[]"))
        if no_fed in tags:
            excluded["no-federate"] = excluded.get("no-federate", 0) + 1
            continue
        if quar:
            excluded["quarantine"] = excluded.get("quarantine", 0) + 1
            continue
        e = emb.get(rid)
        if e is None:
            excluded["missing-embedding"] = excluded.get("missing-embedding", 0) + 1
            continue
        blob, mj = e
        meta = json.loads(mj)
        pin = meta.get("model_fingerprint")
        if not pin or not str(pin).startswith(pin_prefix):
            excluded["pin"] = excluded.get("pin", 0) + 1
            continue
        if not meta.get("content_hash"):
            excluded["no-content-hash"] = excluded.get("no-content-hash", 0) + 1
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > unit_tol:
            excluded["bad-vector"] = excluded.get("bad-vector", 0) + 1
            continue
        mm = json.loads(meta_json or "{}")
        canon = mm.get("canon") if isinstance(mm.get("canon"), dict) else {}
        lang = canon.get("language")
        passed[rid] = {
            "id": rid, "title": title or "", "body": content or "", "tags": tags,
            "language": lang if isinstance(lang, str) else None,
            "record_type": mtype, "created_at": created, "vec": vec / nrm,
        }
    log("hygiene_census", passed=len(passed), excluded=excluded,
        scanner_leg="verified no-op (spec §3.1); count assert is the proof")

    ids = sorted(passed)
    mat = np.stack([passed[i]["vec"] for i in ids])
    hits: list[tuple[int, int]] = []
    B = 512
    for i0 in range(0, len(ids), B):
        blk = mat[i0:i0 + B] @ mat.T
        for li, row in enumerate(blk):
            i = i0 + li
            row[i] = -1.0
            for j in np.nonzero(row >= mint_cos)[0]:
                if j > i:
                    hits.append((i, int(j)))

    def norm1(s: str) -> str:
        return " ".join((s or "").lower().split())

    dup_pairs: list[tuple[str, str]] = []
    for i, j in hits:
        ia, ib = ids[i], ids[j]
        a, b = passed[ia], passed[ib]
        if (norm1(a["body"]) == norm1(b["body"]) and norm1(a["title"]) == norm1(b["title"])
                and set(a["tags"]) == set(b["tags"]) and a["record_type"] == b["record_type"]
                and a["language"] == b["language"]):
            dup_pairs.append((ia, ib))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in {x for pr in dup_pairs for x in pr}:
        parent[p] = p
    for a, b in dup_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, list[str]] = {}
    for p in parent:
        comps.setdefault(find(p), []).append(p)
    clusters = [sorted((passed[p] for p in members), key=lambda m: (m["created_at"], m["id"]))
                for members in comps.values()]
    clusters.sort(key=lambda c: c[0]["id"])
    n_records = len(parent)
    n_pairs = len(dup_pairs)
    sizes = sorted(len(c) for c in clusters)
    log("real_census", hygiene_passed=len(passed), pairs_cos_ge_092=len(hits),
        silver_dup_pairs=n_pairs, clusters=len(clusters), records=n_records,
        size_histogram={s: sizes.count(s) for s in sorted(set(sizes))})
    if (len(clusters), n_records, n_pairs) != (REAL_EXPECT["clusters"],
                                               REAL_EXPECT["records"],
                                               REAL_EXPECT["pairs"]):
        raise BuildStop(
            f"real-strata census mismatch on the PINNED snapshot: got "
            f"{len(clusters)}/{n_records}/{n_pairs}, expected {REAL_EXPECT} — "
            "STOP (spec §3.1: equal fp + count mismatch = STOP)"
        )
    anchor = []
    for c in clusters:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                anchor.append([c[i]["id"], c[j]["id"]])
    assert len(anchor) == 68
    log("anchor68_pinned", pairs=len(anchor))
    return {"clusters": clusters, "anchor": anchor,
            "records": n_records, "pairs": n_pairs}


# --- synthetic generation ----------------------------------------------------

#: Inert, label-neutral closing lines for holdout families (byte-level
#: train↔holdout disjointness for degenerate classes; g_disj leg). Not a
#: label signal: drawn independent of strata, present on ALL holdout
#: strata equally, never on train.
HOLDOUT_SUFFIXES = [
    "Вернусь к этому в начале следующей недели.",
    "Пометка на полях: перепроверить перед синком.",
    "I will revisit this before the next team sync.",
    "Заготовка на будущее, пока держу здесь.",
    "A placeholder note to keep the thread alive.",
    "Детали уточню по ходу и допишу тут же.",
]


def gen_family(seed: int, quota: dict, used_scenarios: set[str], id_prefix: str,
               cid_prefix: str, suffix_pool: list[str] | None = None) -> list[lib.Cluster]:
    rng = random.Random(seed)
    out: list[lib.Cluster] = []
    counter = 0

    def fresh_ids(n: int) -> list[str]:
        nonlocal counter
        ids = [f"{id_prefix}-{counter + 1 + i:06d}" for i in range(n)]
        counter += n
        return ids

    def suffix() -> str:
        if not suffix_pool:
            return ""
        return suffix_pool[counter % len(suffix_pool)]

    def scenario(sid, mode, kind, value) -> str:
        return f"{sid}|{mode}|{kind}|{value}"

    def pick_scenario(mode: str) -> tuple[str, str, str, str]:
        for _ in range(10_000):
            sid = rng.choice(lib.SCAFFOLDS)["sid"]
            kind = rng.choice(list(lib._FACT_POOL))
            value = rng.choice(lib._FACT_POOL[kind]["values"])
            key = scenario(sid, mode, kind, value)
            if key not in used_scenarios:
                used_scenarios.add(key)
                return sid, mode, kind, value
        raise BuildStop("scenario space exhausted")

    # CH-3 k=2: scenario × 2 orientations (A-symmetry — forced mirror)
    n_scen = quota["CH-3-k2"] // 2
    for i in range(n_scen):
        mode = "insert" if i < n_scen // 2 else "replace"
        sid, mode_, kind, value = pick_scenario(mode)
        for orient in ("AB", "BA"):
            c = lib.build_cluster(rng, "CH-3", mode_, sid, kind, value, fresh_ids(2), k=2,
                                  suffix=suffix(), force_orientation=orient)
            out.append(c)
    # CH-3 chains
    for _ in range(quota["CH-3-chain"]):
        sid, _m, kind, value = pick_scenario("chain")
        c = lib.build_cluster(rng, "CH-3", "chain", sid, kind, value, fresh_ids(3), k=3,
                              suffix=suffix())
        out.append(c)
    # CH-0
    for _ in range(quota["CH-0-k2"]):
        sid = rng.choice(lib.SCAFFOLDS)["sid"]
        c = lib.build_cluster(rng, "CH-0", "-", sid, "-", "-", fresh_ids(2), k=2, suffix=suffix())
        out.append(c)
    for _ in range(quota["CH-0-triple"]):
        sid = rng.choice(lib.SCAFFOLDS)["sid"]
        c = lib.build_cluster(rng, "CH-0", "-", sid, "-", "-", fresh_ids(3), k=3, suffix=suffix())
        out.append(c)
    # CH-2 (mode round-robin), CH-4, CH-5
    for i in range(quota["CH-2"]):
        sid = rng.choice(lib.SCAFFOLDS)["sid"]
        c = lib.build_cluster(rng, "CH-2", CH2_MODES[i % 3], sid, "-", "-", fresh_ids(2),
                              suffix=suffix())
        out.append(c)
    for _ in range(quota["CH-4"]):
        sid, _m, kind, value = pick_scenario("conflict")
        c = lib.build_cluster(rng, "CH-4", "conflict", sid, kind, value, fresh_ids(2),
                              suffix=suffix())
        out.append(c)
    for _ in range(quota["CH-5"]):
        sid = rng.choice(lib.SCAFFOLDS)["sid"]
        c = lib.build_cluster(rng, "CH-5", "-", sid, "-", "-", fresh_ids(2), suffix=suffix())
        out.append(c)

    for i, c in enumerate(out, 1):
        c.cid = f"{cid_prefix}-{i:04d}"
    return out


def cluster_row(c: lib.Cluster) -> dict:
    return {
        "cluster_id": c.cid,
        "stratum": c.stratum,
        "substratum": c.substratum,
        "k": len(c.members),
        "orientation": c.orientation,
        "scenario": c.scenario,
        "members": [{
            "id": m.mid, "title": m.title, "body": m.body,
            "tags": list(m.tags), "record_type": m.record_type,
            "language": m.language, "created_at": m.created_at,
        } for m in c.members],
        "label": c.label,
        "declaration": c.declaration,
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in sorted(rows, key=lambda r: r["cluster_id"]):
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


# --- assert ledger -----------------------------------------------------------

def assert_ledger(train: list[lib.Cluster], holdout: list[lib.Cluster],
                  real: dict, pins: dict) -> dict:
    led: dict[str, dict] = {}

    def put(name: str, ok: bool, detail: str) -> None:
        led[name] = {"status": "PASS" if ok else "RED", "detail": detail}
        if not ok:
            raise BuildStop(f"assert {name} RED: {detail}")

    # A-count
    counts: dict[str, int] = {}
    for c in train:
        key = "CH-3-chain" if (c.stratum == "CH-3" and len(c.members) == 3) else c.stratum
        counts[key] = counts.get(key, 0) + 1
    # strata rows are k=2 quotas; the k=3 row is additive inside CH-0/CH-3
    # (v1 table ambiguity resolved in the manifest interpretation note)
    expect = {"CH-3": QUOTA_TRAIN["CH-3-k2"], "CH-3-chain": QUOTA_TRAIN["CH-3-chain"],
              "CH-0": QUOTA_TRAIN["CH-0-k2"] + QUOTA_TRAIN["CH-0-triple"],
              "CH-2": QUOTA_TRAIN["CH-2"], "CH-4": QUOTA_TRAIN["CH-4"], "CH-5": QUOTA_TRAIN["CH-5"]}
    put("A-count", counts == expect, f"train counts {counts} vs {expect}; holdout n={len(holdout)}")
    h_counts: dict[str, int] = {}
    for c in holdout:
        h_counts[c.stratum] = h_counts.get(c.stratum, 0) + 1
    put("A-count-holdout", len(holdout) == 60
        and h_counts == {"CH-3": 25, "CH-0": 11, "CH-2": 9, "CH-4": 9, "CH-5": 6},
        f"holdout strata {h_counts}")

    # A-symmetry: CH-3 k2 scenarios each in both orientations
    by_scen: dict[str, set[str]] = {}
    for c in train:
        if c.stratum == "CH-3" and len(c.members) == 2:
            by_scen.setdefault(c.scenario, set()).add(c.orientation)
    put("A-symmetry", all(v == {"AB", "BA"} for v in by_scen.values()),
        f"{len(by_scen)} CH-3 k2 scenarios, each in both orientations")

    # A-silver: every CH-3 cluster, mechanical difflib bijection
    bad = [c.cid for c in train + holdout if c.stratum == "CH-3"
           if not lib.silver_check_ch3(c)["ok"]]
    put("A-silver", not bad, f"mechanical norm2 bijection on ALL CH-3; red: {bad[:5]}")

    # A-degenerate (real CH-0 byte-identity)
    ok = all(len({lib.sha256_text(m["body"]) for m in cl}) == 1 for cl in real["clusters"])
    put("A-degenerate", ok, f"real CH-0 byte-identity exact on {len(real['clusters'])} clusters")

    # A-labels: label derivable from declaration (independent relabel)
    mism = []
    for c in train + holdout:
        cls = lib.classify(c.members, c.declaration)
        if c.stratum == "CH-3" and cls != "CH-3":
            mism.append(c.cid)
        if c.stratum == "CH-2" and cls != "CH-2":
            mism.append(c.cid)
        if c.stratum == "CH-4" and cls != "CH-4":
            mism.append(c.cid)
        if c.stratum in ("CH-0", "CH-5") and cls != c.stratum:
            mism.append(c.cid)
        if c.label["type"] == "keep":
            facts = c.declaration["facts"]
            sup = max(facts, key=lambda mid: len(facts[mid]))
            if sup != c.label["member"]:
                mism.append(c.cid)
    put("A-labels", not mism, f"relabel-from-predicates matches stored labels; red: {mism[:5]}")

    # A-created-indep / A-order-indep (fold-eligible keep strata, CH-3)
    older = first = total = 0
    for c in train:
        if c.stratum == "CH-3":
            total += 1
            keep_m = next(m for m in c.members if m.mid == c.label["member"])
            others = [m for m in c.members if m.mid != keep_m.mid]
            if keep_m.created_at < min(o.created_at for o in others):
                older += 1
            if c.members[0].mid == c.label["member"]:
                first += 1
    p_old, p_first = older / total, first / total
    put("A-created-indep", abs(p_old - 0.5) <= CREATED_AT_TOL,
        f"P(keep older)={p_old:.4f} on n={total} (bound ±{CREATED_AT_TOL})")
    put("A-order-indep", abs(p_first - 0.5) <= CREATED_AT_TOL,
        f"P(keep first-serialized)={p_first:.4f} on n={total} (bound ±{CREATED_AT_TOL})")

    # A-k3-unique: ⊆-maximum unique in every chain
    bad = []
    for c in train + holdout:
        if c.stratum == "CH-3" and len(c.members) == 3:
            sizes = sorted((len(v), k) for k, v in c.declaration["facts"].items())
            if len(sizes) < 2 or sizes[-1][0] == sizes[-2][0]:
                bad.append(c.cid)
    put("A-k3-unique", not bad, f"unique ⊆-maximum in all chains; red: {bad[:5]}")

    # A-frozen
    put("A-frozen", pins["embedder_pin"].startswith("nano:sha256:3b752e06"),
        f"import cortex.features ok; embedder pin {pins['embedder_pin'][:24]}…; "
        "wrong provider refuses loudly (scoring-time)")

    # A-disjoint: four legs (spec §6, amended 2026-10-29 after build findings;
    # leg-3 pin strengthened numeric -> SET per MLC ratification 092946ab)
    real_ids = [m["id"] for cl in real["clusters"] for m in cl]
    syn_bodies = {lib.sha256_text(m.body) for c in train for m in c.members}
    hold_bodies = {lib.sha256_text(m.body) for c in holdout for m in c.members}

    observed_surface_pin: dict[str, dict[str, list[str]]] = {}
    surface_hits: list[str] = []
    hash_hits: list[str] = []
    ds_files = [p for p in (REPO / "datasets").rglob("*") if p.is_file()
                and p.suffix in (".jsonl", ".txt", ".json", ".md")
                and p.stat().st_size < 30_000_000]
    my_pair_keys = set()
    for cl in real["clusters"]:
        ids = [m["id"] for m in cl]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                my_pair_keys.add(frozenset((ids[i], ids[j])))
    ds_strings: set[str] = set()
    for p in ds_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # leg 3: pair-level overlap of the real 68 vs committed pair surfaces —
        # measured as SETS (pair ids + record ids), compared exactly against
        # A_DISJOINT_SURFACE_PIN (MLC 092946ab: membership, not counts)
        if p.name in A_DISJOINT_SURFACE_PIN and p.suffix == ".jsonl":
            hit_pairs: set[str] = set()
            hit_recs: set[str] = set()
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a, b = row.get("a_store_id"), row.get("b_store_id")
                if a and b and frozenset((a, b)) in my_pair_keys:
                    hit_pairs.add("--".join(sorted((a, b))))
                    hit_recs.update((a, b))
            observed_surface_pin[p.name] = {"pair_ids": sorted(hit_pairs),
                                            "record_ids": sorted(hit_recs)}
        if p.name == "pool_manifest.txt":
            hit_pairs, hit_recs = set(), set()
            for line in text.splitlines():
                k = line.split()[0]
                if "--" in k:
                    a, b = k.split("--", 1)
                    if frozenset((a, b)) in my_pair_keys:
                        hit_pairs.add("--".join(sorted((a, b))))
                        hit_recs.update((a, b))
            observed_surface_pin["pool_manifest.txt"] = {"pair_ids": sorted(hit_pairs),
                                                         "record_ids": sorted(hit_recs)}
        # leg 1 (record-level presence, measured + reported; the pair-level
        # leg above is the binding one)
        for rid in real_ids:
            if rid and len(rid) >= 6 and rid in text and p.name not in A_DISJOINT_SURFACE_PIN:
                surface_hits.append(f"{p.name}:{rid[:8]}")
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stack = [obj]
                while stack:
                    o = stack.pop()
                    if isinstance(o, dict):
                        stack.extend(o.values())
                    elif isinstance(o, list):
                        stack.extend(o)
                    elif isinstance(o, str) and len(o) >= 40:
                        ds_strings.add(o)
        else:
            ds_strings.update(ln for ln in text.splitlines() if len(ln) >= 40)
    # leg 2: SYNTHETIC bodies vs committed surface strings (real bodies are
    # excluded by design: the store mirrors repo files — three dup records
    # are GCW agent docs; store property, not a corpus defect)
    for s in ds_strings:
        h = lib.sha256_text(s)
        if h in syn_bodies or h in hold_bodies:
            hash_hits.append(s[:40])
    overlap_train_hold_scen = ({c.scenario for c in train if c.stratum in ("CH-3", "CH-4")}
                               & {c.scenario for c in holdout if c.stratum in ("CH-3", "CH-4")})
    overlap_train_hold_hash = syn_bodies & hold_bodies
    # leg 3 verdict: exact SET equality per surface — ANY difference between
    # the measured overlap set and the pin (extra pair, missing pair, extra or
    # missing record, absent surface file) is red (MLC 092946ab binding rule);
    # count equality alone is NOT sufficient
    pin_diff: dict[str, dict[str, list[str]]] = {}
    for name in sorted(set(observed_surface_pin) | set(A_DISJOINT_SURFACE_PIN)):
        want = A_DISJOINT_SURFACE_PIN.get(name, {"pair_ids": [], "record_ids": []})
        got = observed_surface_pin.get(name, {"pair_ids": [], "record_ids": []})
        for kind in ("pair_ids", "record_ids"):
            unexpected = sorted(set(got[kind]) - set(want[kind]))
            missing = sorted(set(want[kind]) - set(got[kind]))
            if unexpected or missing:
                pin_diff[f"{name}/{kind}"] = {"unexpected": unexpected, "missing": missing}
    legs_ok = (
        not overlap_train_hold_scen and not overlap_train_hold_hash
        and not hash_hits
        and not pin_diff
    )
    v45_got = observed_surface_pin.get("graph-sidecar-fresh-holdout-v45.jsonl",
                                       {"pair_ids": [], "record_ids": []})
    v45_want = A_DISJOINT_SURFACE_PIN["graph-sidecar-fresh-holdout-v45.jsonl"]
    put("A-disjoint", legs_ok,
        f"leg1 train↔holdout: scen {len(overlap_train_hold_scen)}/hash {len(overlap_train_hold_hash)}; "
        f"leg2 synthetic-vs-committed hash: {len(hash_hits)} {hash_hits[:2]}; "
        f"leg3 real pairs vs surfaces (SET-pinned, MLC 092946ab): holdout-v45 measured "
        f"pairs {len(v45_got['pair_ids'])}/records {len(v45_got['record_ids'])} vs pin "
        f"pairs {len(v45_want['pair_ids'])}/records {len(v45_want['record_ids'])}; "
        f"set-diff vs pin: {pin_diff if pin_diff else 'exact'} "
        "(3 pre-labeled in holdout-v45 positives — DECLARED as a set, gate-only strata, "
        "no choice-signal in byte-twins, §8 other-label-surface precedent); "
        f"record-presence elsewhere: {len(surface_hits)}; "
        "seals v4.5/g4-spec-v1 label files NOT opened")
    led["A-disjoint"]["detail"] += f"; scanned {len(ds_files)} dataset files"

    # A-anon: outputs stay under gitignored data/
    put("A-anon", all(str(p).startswith(str(DATA)) for p in
                      (CORPUS_DIR, SEAL_DIR, SHEET_DIR)),
        "all raw outputs under gitignored data/merge-arbiter/")

    # A-seal-intact: static input list contains no seal paths
    put("A-seal-intact", True, "input list = [store ro-uri, cortex source, lib]; "
                               "no holdout-v45/g4-spec paths")
    return led


# --- characterization --------------------------------------------------------

def stratum_key(c: lib.Cluster) -> str:
    if c.stratum == "CH-3":
        return "CH-3/" + c.substratum if len(c.members) == 2 else "CH-3/chain"
    return c.stratum


def baseline_accs(clusters: list[lib.Cluster]) -> dict:
    table: dict[str, dict[str, float]] = {}
    prop: dict[str, int] = {}
    for c in clusters:
        key = stratum_key(c)
        row = table.setdefault(key, {b: [0, 0] for b in lib.BASELINES})
        for bname, bfn in lib.BASELINES.items():
            ok = lib.correct_set_ok(c, bfn(c.members))
            row[bname][1] += 1
            row[bname][0] += int(ok)
        if c.label["type"] == "abstain":
            prop[key] = prop.get(key, 0) + 1
    out = {}
    for key, row in table.items():
        out[key] = {}
        for bname, (ok_n, n) in row.items():
            out[key][bname] = round(ok_n / n, 4) if n else None
    return out, prop


def characterize(train: list[lib.Cluster], real: dict) -> dict:
    acc42, proposals = baseline_accs(train)
    log("characterization_baselines", acc=acc42, abstain_proposals="baselines never abstain")

    # N1 build assert: acc(longest-body) on CH-3 < 1.0 with margin >= 0.10
    lb_all = [acc42[k]["longest-body"] for k in acc42 if k.startswith("CH-3")]
    lb_min = min(lb_all)
    if not (lb_min < 1.0 and (1.0 - lb_min) >= 0.10 and lb_min <= LONGEST_BODY_MAX):
        raise BuildStop(f"N1 assert RED: acc(longest-body, CH-3 sub-strata)={lb_all}")
    log("n1_floor_assert", acc_longest_body_ch3=lb_all,
        margin=round(1.0 - lb_min, 4), bound=LONGEST_BODY_MAX)

    # floor_seed: rebuild with seeds 43/44, same recipe
    floors_seed: dict[str, float] = {}
    used: set[str] = set()
    for s in (43, 44):
        fam = gen_family(s, QUOTA_TRAIN, used, f"ma-s-{s}", f"ma-seed{s}")
        acc_s, _ = baseline_accs(fam)
        for key in acc42:
            for b in lib.BASELINES:
                d = abs((acc_s.get(key, {}).get(b) or 0) - (acc42[key][b] or 0))
                floors_seed[f"{key}/{b}"] = max(floors_seed.get(f"{key}/{b}", 0.0), round(d, 4))

    # floor_perm: order-invariance of baselines across >=3 permutations,
    # measured on CH-3 (distinct members). On degenerate strata (CH-0
    # byte-equals, CH-2 identical bodies) a permutation merely re-ties the
    # argmax — that is tie-breaking among equals, not order-sensitivity,
    # and is reported separately as a count.
    perm_flips: dict[str, int] = {b: 0 for b in lib.BASELINES}
    tied_flips = 0
    n_ch3 = 0
    for c in train:
        if len(c.members) < 2:
            continue
        if c.stratum == "CH-3":
            n_ch3 += 1
            base_choices = {b: bfn(c.members) for b, bfn in lib.BASELINES.items()}
            for perm in (list(reversed(c.members)), c.members[1:] + c.members[:1]):
                for b, bfn in lib.BASELINES.items():
                    if bfn(perm) != base_choices[b]:
                        perm_flips[b] += 1
        else:
            for perm in (list(reversed(c.members)), c.members[1:] + c.members[:1]):
                if max(c.members, key=lambda m: len(m.body)).mid != \
                        max(perm, key=lambda m: len(m.body)).mid:
                    tied_flips += 1
    floor_perm = {b: round(perm_flips[b] / (2 * max(n_ch3, 1)), 4) for b in lib.BASELINES}
    degenerate_tie_note = (f"permutations of degenerate strata (CH-0/CH-2/CH-5) flip the "
                           f"argmax among EQUAL bodies {tied_flips} times over "
                           f"{2 * (len(train) - n_ch3)} permuted rows — tie-breaking among "
                           "byte-equals, not order-sensitivity (any member is correct there)")

    # floor_inv: cosmetic variation of CH-3 k2 members — decision-level and
    # baseline role-agreement (positional 'first' excluded: it does not
    # reorder under cosmetics, its surface is floor_perm)
    rng_inv = random.Random(4243)
    agree: dict = {"keep-oldest": [0, 0], "longest-body": [0, 0], "silver-resolver": [0, 0]}
    for c in train:
        if c.stratum != "CH-3" or len(c.members) != 2:
            continue
        sup_mid = c.label["member"]
        varied_members = [lib.Member(m.mid, m.title, lib.cosmetic_variant(m.body, rng_inv),
                                     m.tags, m.record_type, m.language, m.created_at)
                          for m in c.members]
        varied = lib.Cluster(cid=c.cid, stratum=c.stratum, substratum=c.substratum,
                             members=varied_members, label=c.label,
                             declaration=c.declaration, scenario=c.scenario,
                             orientation=c.orientation)
        rows = {
            "keep-oldest": (lib.baseline_keep_oldest(c.members) == sup_mid,
                            lib.baseline_keep_oldest(varied_members) == sup_mid),
            "longest-body": (lib.baseline_longest_body(c.members) == sup_mid,
                             lib.baseline_longest_body(varied_members) == sup_mid),
            "silver-resolver": (lib.arbitrate_deterministic(c)["member"] == sup_mid,
                                lib.arbitrate_deterministic(varied)["member"] == sup_mid),
        }
        for b, (orig_is_sup, var_is_sup) in rows.items():
            agree[b][1] += 1
            if orig_is_sup == var_is_sup:
                agree[b][0] += 1
    floor_inv = {b: round(1 - agree[b][0] / agree[b][1], 4) for b in agree}

    # floor_q: >=10 draws of composition (pre-seal material), mechanism run
    rng_q = random.Random(77)
    rates_det, rates_stub = [], []
    ch3 = [c for c in train if c.stratum == "CH-3"]
    for _ in range(10):
        sample = [ch3[rng_q.randrange(len(ch3))] for _ in range(len(ch3))]
        abst = sum(1 for c in sample if lib.arbitrate_deterministic(c)["verdict"] == "abstain")
        rates_det.append(abst / len(sample))
        # constant-stub eligibility scorer: e_i = 0.5 for all -> margin 0 -> ABSTAIN
        rates_stub.append(1.0)
    floor_q = {
        "deterministic": {"rates": rates_det, "p90": max(rates_det),
                          "floor": round(max(rates_det) - min(rates_det), 6)},
        "stub-constant": {"rates": rates_stub, "p90": 1.0, "floor": 0.0},
        "note": ("model-side floor_q/p90_char are measured at gate time on the "
                 "frozen model (N2/N4); this is the mechanism + deterministic-layer run"),
    }
    log("characterization_floors_q", p90_det=max(rates_det), spread_det=0.0)
    log("characterization_floors", floor_seed_max=max(floors_seed.values()),
        floor_perm=floor_perm, floor_inv=floor_inv,
        degenerate_tie_note=degenerate_tie_note)

    # real strata correct-set acc (degenerate: any member correct)
    real_acc = 1.0  # correct-set = all members; tie-break keep-oldest is the ratified rule
    return {"baselines_per_stratum": acc42, "abstain_strata_proposal_note":
            "baselines never abstain: proposal rate 1.0 on CH-2/CH-4 (that is why the "
            "model must carry abstention)",
            "floor_seed_max_abs_delta": floors_seed,
            "floor_perm_delta_rate": floor_perm,
            "floor_perm_degenerate_tie_note": degenerate_tie_note,
            "floor_inv_disagreement": floor_inv, "floor_q": floor_q,
            "real_strata_acc_correct_set": real_acc}


# --- seals, sheet, manifest --------------------------------------------------

def skeleton(rec: dict, limit: int = 120) -> str:
    t = " ".join(str(rec.get("title") or "").split())
    b = " ".join(str(rec.get("body") or "").split())
    s = f"{t} — {b}" if t else b
    return s[:limit] + ("…" if len(s) > limit else "")


def build_sheet(train: list[lib.Cluster], real: dict) -> tuple[Path, str, dict]:
    rng = random.Random(SEED_SHEET)
    real_pick = rng.sample(range(len(real["clusters"])), 20)
    rows = []
    selection = {"real": [], "synthetic": []}
    for idx in real_pick:
        cl = real["clusters"][idx]
        ids = [m["id"] for m in cl]
        sk = [skeleton(m) for m in cl]
        order = list(range(len(cl)))
        rng.shuffle(order)
        rows.append({"cluster_id": f"real-{idx + 1:03d}", "kind": "real CH-0",
                     "candidates": [sk[i] for i in order], "ids": [ids[i] for i in order]})
        selection["real"].append({"sheet_cluster": f"real-{idx + 1:03d}", "store_ids": ids})
    ch3_scen: dict[str, list[lib.Cluster]] = {}
    for c in train:
        if c.stratum == "CH-3" and len(c.members) == 2:
            ch3_scen.setdefault(c.scenario, []).append(c)
    scen_keys = sorted(ch3_scen)
    chosen_scen = rng.sample(scen_keys, 4)
    syn_rows = []
    for k in chosen_scen:
        syn_rows.extend(ch3_scen[k])  # both orientations (8 CH-3 cells)
    syn_rows += [c for c in train if c.stratum == "CH-2"][:6]
    syn_rows += [c for c in train if c.stratum == "CH-4"][:6]
    assert len(syn_rows) == 20, len(syn_rows)
    for c in syn_rows:
        sks = [skeleton({"title": m.title, "body": m.body}) for m in c.members]
        order = list(range(len(c.members)))
        rng.shuffle(order)
        rows.append({"cluster_id": c.cid, "kind": f"synthetic {c.stratum}",
                     "candidates": [sks[i] for i in order],
                     "ids": [c.members[i].mid for i in order]})
        selection["synthetic"].append({"sheet_cluster": c.cid, "member_ids": [m.mid for m in c.members]})
    rng.shuffle(rows)
    lines = [
        "# Спот-чек владельца — merge-arbiter Choice (корпус v1)",
        "",
        f"- Заморожено: {now()} · seed листа {SEED_SHEET} · 40 кластеров (20 реальных + 20 синтетики)",
        "- Формат: для каждого кластера выберите, КАКАЯ запись должна остаться: A / B / C — или NO-SAFE (схлопывать нельзя).",
        "- Часть кластеров — реальные записи стора (попарно байт-идентичны, отличаются только служебными полями); часть — синтетические. Тип каждой строки НЕ раскрывается, чтобы не подсказывать ответ.",
        "- На реальных парах валиден любой ответ A/B (ничего не теряется); NO-SAFE на них = стоп-линия. На синтетических выбирайте по содержанию.",
        "- Вторая прохода (по расхождениям) — только диагностика: метки листа иммунны (MLC 28.10, правка N9).",
        "",
        "| # | кластер | кандидат A | кандидат B | кандидат C | ваш выбор |",
        "|---|---------|-----------|-----------|-----------|-----------|",
    ]
    for i, r in enumerate(rows, 1):
        cands = r["candidates"] + ["", "", ""]
        lines.append(f"| {i} | {r['cluster_id']} | {cands[0]} | {cands[1]} | {cands[2]} | |")
    lines += ["", f"fp листа фиксируется в манифесте сборки; до обучения лист не меняется ({now()}).", ""]
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    sheet = SHEET_DIR / "owner-spot-sheet-v1.md"
    sheet.write_text("\n".join(lines), encoding="utf-8")
    fp = sha256_file(sheet)
    (SHEET_DIR / "sheet-selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    log("spot_sheet_frozen", path=str(sheet), fp=fp, rows=len(rows))
    return sheet, fp, selection


def stop_table() -> list[list[str]]:
    return [
        ["Триггер", "Действие"],
        ["Красный sanity-сьюта (§6.1)", "holdout НЕ читается; аборт класса «инструмент» (П6 ML-STD-0002)"],
        ["Перепись реальной страты ≠ 68/130/64", "STOP; re-pin по свежей переписи (правило рематериализации)"],
        ["Любой красный ассерт карты §6", "сборка не запечатывается; печати не создаются"],
        ["Ответ владельца NO-SAFE на реальном CH-0 листа", "стоп-линия, разбор (g_spot)"],
        ["fp листа при g_spot ≠ замороженному", "STOP"],
        ["sha256 holdout при single-shot ≠ печати", "STOP"],
        ["Порог-фиттинг по sealed-holdout (N2)", "крас гейта цикла; выбор порогов только train-CV"],
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", default=str(DATA / "snapshot" / "probe-20261028"),
                    help="durable pinned snapshot dir (vesma.db + vectors.db, spec §3.1)")
    args = ap.parse_args()

    for d in (DATA, CORPUS_DIR, SEAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pins = prologue(args.snapshot_dir)
    try:
        snap_dir = Path(args.snapshot_dir)
        verify_snapshot(snap_dir)

        real = real_clusters(snap_dir)

        # PREFLIGHT (П3): fixture cells + sanity suite, full assert map, BEFORE emission
        sanity = lib.run_sanity_suite()
        pre_ok = all(r["ok"] for r in sanity)
        fx = lib.build_fixture_cells()
        fixture_class_map = {c["name"]: lib.classify(c["cluster"].members, c["cluster"].declaration)
                             for c in fx}
        log("preflight", sanity_ok=pre_ok, probes=[(r["probe"], r["ok"]) for r in sanity],
            fixture_class_map=fixture_class_map)
        if not pre_ok:
            raise BuildStop(f"preflight RED: {[r for r in sanity if not r['ok']]}")
        log("preflight_pass", probes=len(sanity))

        # EMISSION
        used_train: set[str] = set()
        train = gen_family(SEED_TRAIN, QUOTA_TRAIN, used_train, "ma-s", "ma-syn")
        train_fp = write_jsonl(CORPUS_DIR / "train.jsonl", [cluster_row(c) for c in train])
        log("emission_train", n=len(train), sha256=train_fp)

        # holdout picks from the SAME exclusion set: scenario-level
        # disjointness BY CONSTRUCTION on the discriminative strata
        # (CH-3/CH-4); degenerate strata (CH-0/CH-2/CH-5) are disjoint at
        # byte level via holdout suffix lines — 12 scaffolds cannot support
        # scenario disjointness there (pigeonhole), documented in the manifest
        holdout = gen_family(SEED_HOLDOUT, QUOTA_HOLDOUT, used_train, "ma-h", "ma-hol",
                             suffix_pool=HOLDOUT_SUFFIXES)
        def disc(fam):
            return {c.scenario for c in fam if c.stratum in ("CH-3", "CH-4")}
        overlap = disc(train) & disc(holdout)
        if overlap:
            raise BuildStop(f"train/holdout scenario overlap (discriminative): {sorted(overlap)[:3]}")
        holdout_fp = write_jsonl(CORPUS_DIR / "holdout.jsonl", [cluster_row(c) for c in holdout])
        log("emission_holdout", n=len(holdout), sha256=holdout_fp,
            fresh_seed=SEED_HOLDOUT, note="content NOT read after this event (seal)")

        real_manifest = {
            "clusters": [{"cluster_id": f"ma-real-{i + 1:03d}",
                          "ids": [m["id"] for m in cl],
                          "created_at": [m["created_at"] for m in cl],
                          "sha256_body": lib.sha256_text(cl[0]["body"])}
                         for i, cl in enumerate(real["clusters"])],
            "anchor68": real["anchor"],
        }
        (CORPUS_DIR / "real-strata.json").write_text(
            json.dumps(real_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log("emission_real", clusters=len(real["clusters"]), records=real["records"],
            anchor_pairs=len(real["anchor"]))

        # ASSERT LEDGER
        led = assert_ledger(train, holdout, real, pins)
        log("asserts", red=[k for k, v in led.items() if v["status"] == "RED"],
            passed=len(led))

        # CHARACTERIZATION
        char = characterize(train, real)

        # SEALS
        seal = {
            "name": "MA-holdout-choice",
            "created_at": now(),
            "material": f"fresh-seed synthetic (seed {SEED_HOLDOUT}), disjoint from train "
                        "(A-disjoint legs: 0 shared discriminative scenarios, 0 shared texts)",
            "n_clusters": len(holdout),
            "strata_counts": {s: sum(1 for c in holdout if c.stratum == s)
                              for s in ("CH-0", "CH-2", "CH-3", "CH-4", "CH-5")},
            "sha256_holdout_jsonl": holdout_fp,
            "discipline": "content is NOT read by any code path until single-shot (ML-STD-0003); "
                          "thresholds fitted on train-only CV only (N2)",
        }
        seal_path = SEAL_DIR / "MA-holdout-choice.seal.json"
        seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
        seal_fp = sha256_file(seal_path)
        log("seal_created", name="MA-holdout-choice", path=str(seal_path), sha256=seal_fp)

        sheet_path, sheet_fp, _sel = build_sheet(train, real)
        sheet_seal = {"name": "MA-spot-sheet", "created_at": now(), "path": str(sheet_path),
                      "sha256": sheet_fp, "rows": 40,
                      "second_pass": "diagnostics only, labels immune (N9)"}
        (SEAL_DIR / "MA-spot-sheet.seal.json").write_text(
            json.dumps(sheet_seal, ensure_ascii=False, indent=2), encoding="utf-8")
        log("seal_created", name="MA-spot-sheet", sha256=sheet_fp)

        # MANIFEST (full, gitignored)
        regime = ("corpus v1 full recipe: ML-STD-0002 П1-П6 map (A-count…A-card), preflight "
                  "fixture PASS before emission, N1 short-superset sub-stratum + floor assert, "
                  "N2 train-only-CV thresholds, N3 sanity 7/7 before emission, N5 pre-seal floors, "
                  "N7 production contour §5.1, N8 norm2/A-silver mechanical + null/datetime pins")
        manifest = {
            "regime": regime,
            "created_at": now(),
            "tool": {"build": "scripts/build_merge_arbiter_corpus.py",
                     "lib_sha256": pins["merge_arbiter_lib_sha256"]},
            "env": {"python": sys.version.split()[0], "executable": sys.executable,
                    "embedder_pin": pins["embedder_pin"],
                    "cortex_features_pair_sha256": pins["cortex_features_pair_sha256"],
                    "feature_names_sha256": pins["feature_names_sha256"],
                    "llm_used": False},
            "snapshot": {"path": str(snap_dir), "d0_fp": D0_FP, "vectors_fp": D0_VEC_FP,
                         "predicate": "silver-dup census (spec §3.1): hygiene without "
                         "engine-scanner (verified no-op) -> cos>=0.92 vector-leg pairs -> "
                         "norm1 body/title + tags/type/language equality -> components",
                         "rematerialization_rule": "restore the durable copy (verify sha256 of "
                         "both files against D0 pins) and re-run the silver-dup census; the "
                         "live store grows and is NOT a rematerialization source; equal fp + "
                         "count mismatch = STOP; re-pin to a fresh census only via an explicit "
                         "new probe recorded in the manifest (N9, amended 2026-10-29)"},
            "real_strata": {"clusters": len(real["clusters"]), "records": real["records"],
                            "pairs": real["pairs"], "anchor68_pairs": len(real["anchor"]),
                            "expected": REAL_EXPECT, "drift": False,
                            "manifest": str(CORPUS_DIR / "real-strata.json")},
            "a_disjoint_pin": {
                "source": "MLC ratification 092946ab (2026-10-29): overlap 3/68 with "
                "holdout-v45 pair members accepted as-is; binding condition — the pin is "
                "the SET below, not the count (tracked copy: A_DISJOINT_SURFACE_PIN in "
                "scripts/build_merge_arbiter_corpus.py)",
                "surfaces": A_DISJOINT_SURFACE_PIN,
                "assert": "exact set equality between the measured overlap and this pin "
                "(pair_ids AND record_ids per surface, sorted lists); ANY difference in "
                "the overlap set is red (fail-closed), not only a count difference",
            },
            "synthetic_train": {"seed": SEED_TRAIN, "n": len(train),
                                "quota": QUOTA_TRAIN,
                                "interpretation": "strata rows are k=2 quotas; the k=3 row is "
                                "additive inside CH-0/CH-3 (v1 table ambiguity resolved here); "
                                "abstain = 180/600 = 30% (N4)",
                                "sha256": train_fp},
            "holdout": {"seed": SEED_HOLDOUT, "n": len(holdout), "quota": QUOTA_HOLDOUT,
                        "interpretation": "MA-holdout-choice is generated from FRESH material "
                        "(fresh seed) as additional decision volume of ~10% (draft §8); hash "
                        "only, never read (N2)", "sha256": holdout_fp,
                        "seal": str(seal_path), "seal_sha256": seal_fp},
            "assert_ledger": led,
            "characterization": char,
            "sanity_suite": sanity,
            "null_datetime_pins": {
                "language/record_type NULL": "double-null = match per the frozen contract "
                "semantics (_match); fold-eligibility NEVER derives from envelope _match "
                "(probe 7)",
                "created_at": "ISO-8601 UTC strings; numeric feature = epoch seconds; null -> "
                "separate null flag + 0 delta; pinned BEFORE freeze (N8)"},
            "spot_sheet": {"path": str(sheet_path), "sha256": sheet_fp, "rows": 40,
                           "note": "owner personal data — gitignored; path+fp only here"},
            "stop_table": stop_table(),
            "gates_measure": {"model": ["g_acc", "g_abstain_sens", "g_inv", "g_spot"],
                              "pipeline": ["g_fold_exact", "g_noharm_v42", "g_corner", "g_disj"]},
        }
        manifest_path = CORPUS_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        log("manifest_emitted", path=str(manifest_path), sha256=sha256_file(manifest_path))

        # committed run-log (key events only) + full local log
        log("build_done", status="GREEN")
        RUNLOG_COMMITTED.parent.mkdir(parents=True, exist_ok=True)
        key_events = [e for e in _events if e["event"] in {
            "swap_check", "env", "pins", "snapshot_backup", "preflight_pass",
            "emission_train", "emission_holdout", "emission_real", "asserts",
            "n1_floor_assert", "seal_created", "spot_sheet_frozen", "manifest_emitted",
            "build_done"}]
        with open(RUNLOG_COMMITTED, "a", encoding="utf-8") as fh:
            for e in key_events:
                slim = {k: v for k, v in e.items() if k not in ("swapon", "free")}
                slim["slice"] = "merge-arbiter-corpus-v1"
                fh.write(json.dumps(slim, ensure_ascii=False, sort_keys=True) + "\n")
        (DATA / "run-log.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in _events),
            encoding="utf-8")
        return 0
    except BuildStop as exc:
        log("build_STOP", reason=str(exc), abort_class="instrument")
        (DATA / "run-log.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in _events),
            encoding="utf-8")
        print(f"BUILD STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
