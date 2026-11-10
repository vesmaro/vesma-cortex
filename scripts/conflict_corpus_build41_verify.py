#!/usr/bin/env python3
"""build4.1 — mechanical pair_id collision fix for train-v4 (ADDENDUM-7 §6,
TL verdict @ d61f2de, slice conflict-detector-corpus-v2-build4).

DEFECT: train-v4 (sha 7645ae46…) carried 4 pair_id collisions
(tr4-C1-00009/00010/00013/00016, x4 rows each): the two id lanes of
conflict_corpus_carriers_v4.py held private ctr=[0] counters under the SAME
prefix tr4-C1.

FIX (this lane): content-IDENTICAL re-emission through the repaired
machinery — the shared-namespace rekey in conflict_corpus_carriers_v4.py
(lane-1 fresh-base ids are immutable: fresh rung rng streams are keyed by
the base scenario_id; lane-2 colliding rung ids are rekeyed beyond the
namespace max). Pairs/labels/orbits/scaffolds are bit-untouched; ONLY the
colliding pair_ids move. NOT a carrier edit — a machine-room fix with a
correction record (G1 content freeze intact).

What this script does:
  1. G1 re-checks: banks modules vs freeze pins; carriers CONFIG content
     sha vs freeze pin (the predicate-set freeze proof); train-v2
     fingerprint vs manifest.
  2. Re-runs the C1 census (deterministic; embedder-pinned) and asserts
     the pinned measure-then-fix quotas (T-carrier-quota) — the proof the
     repaired machinery selects the SAME 384 carriers.
  3. Re-emits: the build4 rows VERBATIM with ONLY the rekeyed pair_ids
     swapped (§6: nothing but pair_id may change). measured.cos_target
     keeps the pinned build4-time nano measurement — the census re-run
     reproduces the selection exactly while re-measuring cos within
     +-2e-4 (instrument drift vs the build4-time embedder state,
     disclosed); re-measuring would break the artifact's internal
     measurement consistency. CONTENT IDENTITY is asserted against the
     on-disk artifact: sorted content tuples (every field except
     pair_id/scenario_id) — 0 discrepancies.
  4. Re-runs the train-leg assert map on the census carriers (the fixed
     machinery): T-carrier-silver, T-labels+invariance, T-cosmetic-proof,
     T-carrier-disjoint, T-carrier-shape (exact ports of the cmd_build4
     blocks) + the NEW T-id-unique (GLOBAL pair_id uniqueness, the
     trainer's P5 contract: (pair_id, orientation) unique AND pair_id
     unique outside orientations, over all 984x2 rows).
  5. --apply: writes train.jsonl atomically; updates pin-v4.json (train_v4
     sha + correction_build4_1 record), build-manifest.json +
     artifacts/runs/seal_v4_report.json (fingerprints.train.jsonl,
     pin_sha256, correction block); appends run-log events. The EVAL side
     of the pin (seal 76c12ad1…, census, transfer markup) is NOT touched —
     seal v4 stays valid, holdout files are never opened.

Dry-run is the default: everything is computed and asserted, nothing is
written. Run: taskset -c 0-12 /usr/bin/python3 scripts/\
conflict_corpus_build41_verify.py [--apply]
"""

from __future__ import annotations

import collections
import json
import random
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO / "scripts"))

import build_conflict_detector_corpus as drv          # noqa: E402
import conflict_corpus_lib as lib                     # noqa: E402
import conflict_corpus_carriers_v4 as c1m             # noqa: E402

CORPUS_V4 = drv.CORPUS_DIR_V4
TRAIN_V4 = CORPUS_V4 / "train.jsonl"
TRAIN_V2 = drv.CORPUS_DIR_V2 / "train.jsonl"
PIN = drv.PIN_V4_PATH
MANIFEST = CORPUS_V4 / "build-manifest.json"
SEAL_REPORT = drv.RUNLOG.parent / "seal_v4_report.json"
BACKUP_TRAIN = CORPUS_V4 / "train.jsonl.pre-build41"
BACKUP_PIN = drv.DATA / "pin-v4.pre-build41.json"

#: the defective build4 emission (run-log seal_v4 event, train_sha)
OLD_TRAIN_SHA = "7645ae4601f25b8bd0cfc30dcd95ef0a6646b6f98adfd66029cee83563e794bf"
#: carriers module freeze pin (prep.carriers_c1.file_sha256) — changed by
#: this correction (disclosed); the CONFIG content sha must NOT change
OLD_MODULE_SHA = "b197ed715843861bb7c680fb77c90ab597137f7c2c627356694a40ffabffee11"
VERDICT_REF = "ADDENDUM-7 §6 (build4.1, TL verdict @ d61f2de)"
#: ids the ML lane reported as colliding (x4 rows each) — the artifact-visible
#: collision set; the rekey may rename more ENUMERATED rungs whose lane-2
#: partner was culled by the census (disclosed in the report)
DEFECT_IDS = {"tr4-C1-00009", "tr4-C1-00010", "tr4-C1-00013", "tr4-C1-00016"}
ID_FIELDS = ("pair_id", "scenario_id")


def fail(msg: str) -> None:
    print(f"BUILD41-STOP: {msg}", file=sys.stderr)
    sys.exit(1)


def content_key(row: dict) -> str:
    """The CONTENT identity key: every field except the id fields
    (pair_id == scenario_id by construction). Nested declaration rides
    along; sort_keys pins the JSON form."""
    slim = {k: v for k, v in row.items() if k not in ID_FIELDS}
    return json.dumps(slim, ensure_ascii=False, sort_keys=True)


def main() -> int:
    t0 = time.time()
    apply_mode = "--apply" in sys.argv
    mode = "APPLY" if apply_mode else "DRY-RUN"
    print(f"== build4.1 ({mode}) — {VERDICT_REF} ==")

    # --- 1. baseline + G1 re-checks -----------------------------------------
    old_text = TRAIN_V4.read_text(encoding="utf-8")
    old_sha = lib.sha(old_text)
    if old_sha != OLD_TRAIN_SHA:
        fail(f"on-disk train-v4 sha {old_sha[:16]} != defective baseline "
             f"{OLD_TRAIN_SHA[:16]} — wrong baseline, abort")
    new_module_sha = drv._module_sha("scripts/conflict_corpus_carriers_v4.py")
    prep = json.loads(PIN.read_text(encoding="utf-8"))["prep"]
    if prep["carriers_c1"]["file_sha256"] != OLD_MODULE_SHA:
        fail("freeze pin carriers_c1 file_sha256 drifted")
    if new_module_sha == OLD_MODULE_SHA:
        fail("carriers module unchanged — the rekey fix is missing")
    c1_content = drv._c1_content_sha(drv.SEED_HOLD_V4)
    if c1_content != prep["carriers_c1"]["content_sha256"]:
        fail(f"carriers CONFIG content sha {c1_content[:12]} != frozen "
             f"{prep['carriers_c1']['content_sha256'][:12]} — the frozen "
             "predicate set changed, this is NOT a machine-room fix")
    for name, rel in (("banks_v3_continuity",
                       "scripts/conflict_corpus_banks_v3.py"),
                      ("banks_v4", "scripts/conflict_corpus_banks_v4.py")):
        if drv._module_sha(rel) != prep[name]["file_sha256"]:
            fail(f"G1: {name} module drifted vs freeze pin")
    man2_fp = drv.file_sha256(TRAIN_V2)
    man2 = json.loads((drv.CORPUS_DIR_V2 / "build-manifest.json")
                      .read_text(encoding="utf-8"))
    if man2_fp != man2["fingerprints"]["train.jsonl"]:
        fail("train-v2 fingerprint RED")
    print(f"baseline ok: old train sha {old_sha[:16]}, module {new_module_sha[:12]} "
          f"(was {OLD_MODULE_SHA[:12]}), config content sha frozen "
          f"{c1_content[:12]}, banks modules intact, train-v2 {man2_fp[:12]}")

    # --- 2. census re-run (deterministic, embedder-pinned) -------------------
    print("== census re-run (C1 survival: pred+razor+channel+F10+caps) ==")
    cos = drv.Cos()
    tr_texts, _ = drv.disk_text_shas(TRAIN_V2)
    sources = {
        "train-v2": tr_texts,
        "v3-seal": drv.sealed_generation_sha_set(
            drv.SEAL_V3_PATH, drv.CORPUS_DIR_V3 / "holdout-pool.jsonl"),
        "v2-seal": drv.sealed_generation_sha_set(
            drv.SEAL_V2_PATH, drv.CORPUS_DIR_V2 / "holdout-pool.jsonl"),
        "v1-seal": drv.sealed_generation_sha_set(
            drv.SEAL_V1_PATH, drv.CORPUS_DIR / "holdout-pool.jsonl"),
    }
    c1 = drv._surviving_carriers(cos, sources)
    carriers = c1["carriers"]
    quota_pin = prep["carriers_c1_quota"]["per_stratum"]
    c1_by_s = collections.Counter(c["sc"].stratum for c in carriers)
    if dict(c1_by_s) != {s: quota_pin[s] for s in quota_pin}:
        fail(f"census quotas drifted: {dict(c1_by_s)} != {quota_pin}")
    rekey = c1m.enumerate_orbits(drv.SEED_HOLD_V4)["id_rekey"]
    print(f"census ok: {len(carriers)} carriers == pin (measure-then-fix); "
          f"cull: {dict(c1['report']['cull'])}; "
          f"enumerated rekey: {rekey['collisions']} collisions "
          f"{sorted(rekey['renamed'])} -> {sorted(rekey['renamed'].values())}")

    # --- 3. re-emission: old rows VERBATIM, only rekeyed pair_ids move -------
    # measured.cos_target is the PINNED build4-time nano measurement. The
    # census re-run reproduces the SELECTION exactly (quotas == pin) but
    # re-measures cos within +-2e-4 (instrument drift vs the build4-time
    # embedder state; disclosed) — re-measuring would break the artifact's
    # internal measurement consistency (the rest of the corpus carries
    # build4-time measurements). ADDENDUM-7 §6: ONLY pair_id may change,
    # so the emitted rows are the build4 rows verbatim with the rekeyed
    # ids; the census re-run above is the proof the machinery behind them
    # is deterministic and green.
    old_rows = [json.loads(l) for l in old_text.splitlines()]
    old_carriers = [r for r in old_rows if r["pair_id"].startswith("tr4-")]

    def pair_key(r: dict) -> tuple:
        d = r["declaration"]
        return (r["stratum"], d.get("base_id"), d.get("rung"),
                d.get("orbit_kind"))

    old_by_pair: dict[tuple, dict] = {}
    old_twins: dict[tuple, list] = {}
    for r in old_carriers:
        old_twins.setdefault(pair_key(r), []).append(r)
        if r["orientation"] != "a>b":
            continue  # the a>b row carries the pair
        if pair_key(r) in old_by_pair:
            fail(f"old artifact has duplicate carrier pair key {pair_key(r)}")
        old_by_pair[pair_key(r)] = r

    moved: dict[str, str] = {}   # old pair_id -> new pair_id (surviving only)
    unmoved = 0
    #: content-key -> the ONE id this row must carry after re-emission
    #: (content-key based because old ids are double-assigned — the defect)
    new_id_by_key: dict[str, str] = {}
    pair_rows: list[tuple[tuple, str, dict]] = []  # (sort key, new_id, old a>b row)
    for c in carriers:
        sc = c["sc"]
        key = (sc.stratum, c["base"].scenario_id, c["rung"], c["kind"])
        old_row = old_by_pair.get(key)
        if old_row is None:
            fail(f"census carrier has no old artifact pair: {key} — "
                 "selection drifted, abort")
        new_id = sc.scenario_id
        if new_id != old_row["pair_id"]:
            if old_row["pair_id"] in moved:
                fail(f"two census carriers map from old id "
                     f"{old_row['pair_id']}")
            moved[old_row["pair_id"]] = new_id
        else:
            unmoved += 1
        for t in old_twins[key]:
            new_id_by_key[content_key(t)] = new_id
        pair_rows.append(((sc.stratum, new_id), new_id, old_row))
    if len(pair_rows) != len(old_by_pair):
        fail(f"carrier count drift: census {len(pair_rows)} != artifact "
             f"{len(old_by_pair)}")
    if not moved:
        fail("no surviving carrier was rekeyed — the fix is not in effect")
    # every surviving collider of the ENUMERATED namespace moves (the
    # source-level fix); the artifact-visible defect set must be covered
    renamed_all = sorted(rekey["renamed"])
    if not set(moved) <= set(renamed_all):
        fail(f"moved ids {sorted(moved)} exceed the enumerated rekey "
             f"{renamed_all}")
    if not DEFECT_IDS <= set(moved):
        fail(f"artifact-visible defect ids {sorted(DEFECT_IDS - set(moved))} "
             "did not move")
    surviving_defect = sorted(moved)

    carrier_rows = []
    for (_, new_id, old_row) in sorted(pair_rows, key=lambda x: x[0]):
        key = pair_key(old_row)
        twins = sorted(old_twins[key],
                       key=lambda r: 0 if r["orientation"] == "a>b" else 1)
        if {t["orientation"] for t in twins} != {"a>b", "b>a"}:
            fail(f"old pair {old_row['pair_id']} ({key}) lacks its two "
                 "orientations")
        for t in twins:
            nr = json.loads(json.dumps(t, ensure_ascii=False))
            nr["pair_id"] = new_id
            nr["scenario_id"] = new_id
            carrier_rows.append(nr)
    tr2_lines = TRAIN_V2.read_text(encoding="utf-8")
    new_text = tr2_lines + "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in carrier_rows)
    new_sha = lib.sha(new_text)
    if not old_text.startswith(tr2_lines) or not new_text.startswith(tr2_lines):
        fail("classic prefix broken — train-v2 verbatim violated")

    # --- 4. content identity (THE build4.1 assert) ----------------------------
    new_rows = [json.loads(l) for l in new_text.splitlines()]
    if len(old_rows) != len(new_rows):
        fail(f"row count changed: {len(old_rows)} -> {len(new_rows)}")
    new_carriers = [r for r in new_rows if r["pair_id"].startswith("tr4-")]
    old_keys = sorted(content_key(r) for r in old_carriers)
    new_keys = sorted(content_key(r) for r in new_carriers)
    disc = collections.Counter(old_keys)
    disc.subtract(collections.Counter(new_keys))
    disc = {k: v for k, v in disc.items() if v}
    if disc:
        fail(f"CONTENT IDENTITY RED: {len(disc)} content tuples differ "
             f"(first: {next(iter(disc))[:120]})")
    # row-level id-swap consistency: each new row's content key must carry
    # EXACTLY the id the census assigns it (old ids are DOUBLE-assigned in
    # the defective artifact — that is the defect — so the mapping is
    # content-key based, not id based)
    row_delta_bad = []
    seen_keys: set = set()
    for r in new_carriers:
        ck = content_key(r)
        want = new_id_by_key.get(ck)
        if want is None:
            row_delta_bad.append(f"{r['pair_id']}: unknown content key")
        elif r["pair_id"] != want:
            row_delta_bad.append(f"{r['pair_id']} (want {want})")
        seen_keys.add(ck)
    if row_delta_bad:
        fail(f"id-swap consistency RED: {row_delta_bad[:4]}")
    if len(seen_keys) != len(new_id_by_key):
        fail("content-key coverage drift: some old rows have no new twin")
    changed = dict(moved)
    n_changed_rows = 2 * len(changed)
    print(f"content identity: {len(old_carriers)} carrier rows compared, "
          f"0 discrepancies (rows verbatim, ids only); "
          f"pair_id map: {unmoved} unchanged, {len(changed)} moved "
          f"{surviving_defect} -> {[changed[i] for i in surviving_defect]}; "
          f"{n_changed_rows} of {len(new_rows)} rows differ in id only; "
          f"census re-measure drift disclosed: selection unchanged "
          f"(quotas == pin)")

    # --- 5. train-leg assert map on the new emission --------------------------
    print("== train-leg asserts (build4 map + build4.1 T-id-unique) ==")
    legs: dict[str, dict] = {}

    def rec(name: str, ok, detail) -> None:
        legs[name] = {"ok": bool(ok), **detail}

    # T-carrier-silver / T-labels+invariance / T-cosmetic-proof
    # (exact port of cmd_build4 lines 4929-4959)
    c1_pred_bad = []
    for c in carriers:
        if c["rung"] == "base":
            if not lib.mechanical_ok(c["sc"]):
                c1_pred_bad.append(c["sc"].scenario_id)
        elif not c1m.orbit_ok(c["sc"], c["base"]):
            c1_pred_bad.append(c["sc"].scenario_id)
    lab_c1 = [c["sc"].scenario_id for c in carriers
              if c["sc"].label != ("contradicts"
                                   if c["sc"].stratum.startswith("CD")
                                   else "not-contradicts")
              or c["sc"].label != c["base"].label]
    raz_c1 = [c["sc"].scenario_id for c in carriers
              if c["sc"]._cos < c1m.RAZOR_LO]
    ch_c1 = [c["sc"].scenario_id for c in carriers
             if c["sc"].length_channel != c["base"].length_channel]
    cp_bad = [c["base"].scenario_id for c in carriers
              if not c1m.cosmetic_proof(c["base"].side_a)
              or not c1m.cosmetic_proof(c["base"].side_b)]
    rec("T-carrier-silver", not c1_pred_bad and not raz_c1 and not ch_c1,
        {"pred_bad": c1_pred_bad[:4], "below_razor": raz_c1[:4],
         "channel_shift": ch_c1[:4]})
    rec("T-labels+invariance", not lab_c1, {"bad": lab_c1[:4]})
    rec("T-cosmetic-proof", not cp_bad, {"bad": cp_bad[:4]})

    # T-carrier-disjoint (port of 4960-4989)
    leak_seal, leak_train = [], []
    for c in carriers:
        sc = c["sc"]
        new_sides = [t for t in (sc.text_a(), sc.text_b())
                     if t != c["base"].text_a() and t != c["base"].text_b()]
        for t in new_sides:
            h = lib.sha(t)
            if drv._forbidden_source(h, h, {k: sources[k] for k in
                                            ("v1-seal", "v2-seal", "v3-seal")}):
                leak_seal.append(sc.scenario_id)
            if h in sources["train-v2"]:
                leak_train.append(sc.scenario_id)
    for c in carriers:
        if c["kind"] == "fresh":
            sc = c["sc"]
            for t in (sc.text_a(), sc.text_b()):
                h = lib.sha(t)
                if drv._forbidden_source(h, h, {k: sources[k] for k in
                                                ("v1-seal", "v2-seal",
                                                 "v3-seal")}):
                    leak_seal.append(sc.scenario_id)
                if h in sources["train-v2"]:
                    leak_train.append(sc.scenario_id)
    rec("T-carrier-disjoint", not leak_seal and not leak_train,
        {"seal_hits": leak_seal[:4], "train_side_hits": leak_train[:4]})

    # T-carrier-shape (port of 4990-5048)
    c1_pairs: dict[tuple, dict] = {}
    dup_pairs = 0
    meta_distinguished = 0
    for c in carriers:
        sc = c["sc"]
        pr = tuple(sorted((lib.sha(sc.text_a()), lib.sha(sc.text_b()))))
        if pr in c1_pairs:
            prev = c1_pairs[pr]
            meta_diff = (prev.side_a.record_type != sc.side_a.record_type
                         or prev.side_b.record_type != sc.side_b.record_type
                         or prev.side_a.language != sc.side_a.language
                         or prev.side_b.language != sc.side_b.language
                         or list(prev.side_a.tags) != list(sc.side_a.tags)
                         or list(prev.side_b.tags) != list(sc.side_b.tags))
            if meta_diff:
                meta_distinguished += 1
            else:
                dup_pairs += 1
        c1_pairs[pr] = sc
    null_c = sum(1 for c in carriers if c["sc"].both_null)
    mixed_c = sum(1 for c in carriers
                  if (c["sc"].side_a.created_at is None)
                  != (c["sc"].side_b.created_at is None))
    both_fit = 0
    side_a_n = 0
    for c in carriers:
        if c["rung"] not in ("para1", "para2"):
            continue
        base = c["base"]
        k = 1 if c["rung"] == "para1" else 2
        fit = []
        for side in ("a", "b"):
            body = (base.side_a if side == "a" else base.side_b).body
            try:
                lib.benign_substitute(body, k,
                                      (base.side_a if side == "a"
                                       else base.side_b).language or "ru",
                                      random.Random(0))
                fit.append(side)
            except ValueError:
                pass
        if len(fit) == 2:
            both_fit += 1
            d = c["sc"].benign[0]["side"]
            side_a_n += 1 if d == "a" else 0
    rec("T-carrier-shape", dup_pairs == 0 and mixed_c == 0
        and (both_fit == 0 or abs(side_a_n / both_fit - 0.5) <= 0.10),
        {"pair_dups": dup_pairs,
         "sha_pair_collisions_meta_distinguished": meta_distinguished,
         "mixed_null": mixed_c, "e7_both_fit": both_fit, "side_a": side_a_n,
         "null_share_carriers": round(null_c / max(len(carriers), 1), 3)})

    # T-id-unique (build4.1): in-memory carrier block + FULL disk artifact
    c1_ids = collections.Counter(c["sc"].scenario_id for c in carriers)
    id_dups = sorted(i for i, n in c1_ids.items() if n > 1)
    disk_pairs = collections.Counter(r["pair_id"] for r in new_rows)
    bad_count = sorted(p for p, n in disk_pairs.items() if n != 2)
    orient_bad = []
    by_pair: dict[str, set] = collections.defaultdict(set)
    for r in new_rows:
        by_pair[r["pair_id"]].add(r["orientation"])
    for p, os_ in by_pair.items():
        if os_ != {"a>b", "b>a"}:
            orient_bad.append(p)
    classic_ids = {r["pair_id"] for r in new_rows
                   if not r["pair_id"].startswith("tr4-")}
    ns_overlap = sorted(classic_ids & set(c1_ids))
    rec("T-id-unique", not id_dups and not bad_count and not orient_bad
        and not ns_overlap,
        {"carrier_dups": id_dups[:6], "rows_not_x2": bad_count[:6],
         "bad_orientations": orient_bad[:6],
         "classic_ns_overlap": ns_overlap[:6],
         "distinct_pair_ids": len(disk_pairs),
         "rows": len(new_rows)})

    for name, v in legs.items():
        print(f"  {name}: {'PASS' if v['ok'] else 'RED'} {v}")
    reds = [k for k, v in legs.items() if not v["ok"]]
    if reds:
        fail(f"asserts RED: {reds}")
    print(f"asserts: {len(legs)}/{len(legs)} PASS")

    # --- 6. apply --------------------------------------------------------------
    if not apply_mode:
        print(f"DRY-RUN complete — nothing written. New train sha would be "
              f"{new_sha[:16]}")
        return 0

    BACKUP_TRAIN.write_text(old_text, encoding="utf-8")
    shutil.copyfile(PIN, BACKUP_PIN)
    tmp = TRAIN_V4.parent / "train.jsonl.build41.tmp"
    tmp.write_text(new_text, encoding="utf-8")
    if lib.sha(tmp.read_text(encoding="utf-8")) != new_sha:
        fail("tmp write sha mismatch")
    tmp.replace(TRAIN_V4)

    # pin-v4.json: train_v4 sha + correction record (EVAL side untouched)
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    pin["train_v4"]["sha256"] = new_sha
    pin["train_v4"]["sha_note"] = ("build4.1 content-identical re-emission "
                                   "(pair_id collision fix); classic "
                                   "prefix byte-identical")
    pin["correction_build4_1"] = {
        "ts": drv.now(),
        "verdict": VERDICT_REF,
        "reason": "4 pair_id collisions (tr4-C1-00009/00010/00013/00016, "
                  "x4 rows each) — trainer P5 duplicate (pair_id, "
                  "orientation); root: two private ctr counters under one "
                  "tr4-C1 prefix (conflict_corpus_carriers_v4.py)",
        "old_train_sha256": OLD_TRAIN_SHA,
        "new_train_sha256": new_sha,
        "old_carriers_module_sha256": OLD_MODULE_SHA,
        "new_carriers_module_sha256": new_module_sha,
        "carriers_config_content_sha256_unchanged": c1_content,
        "id_scheme": {
            "before": "two private per-lane ctr=[0] counters (fresh "
                      "bases; classic+fresh rungs) under ONE prefix "
                      "tr4-C1 — overlapping number spaces",
            "after": "ONE shared tr4-C1 namespace: lane-1 (fresh base) "
                     "ids immutable (fresh rung rng keyed by base id), "
                     "lane-2 colliding rung ids rekeyed beyond namespace "
                     f"max {renamed_all}",
        },
        "content_identity_check": "PASS (0 discrepancies; sorted content "
                                  "tuples, all fields except pair_id/"
                                  f"scenario_id; {n_changed_rows} rows "
                                  "differ in id only)",
        "changed_pair_ids": changed,
        "eval_side_untouched": True,
        "seal_v4_unchanged": "76c12ad1947aaf7856541cd72745adbc8a0e775365"
                             "27e67120d64382b7ce986f",
    }
    pin_text = json.dumps(pin, ensure_ascii=False, indent=2)
    PIN.write_text(pin_text, encoding="utf-8")
    new_pin_sha = lib.sha(PIN.read_text(encoding="utf-8"))

    correction_short = {
        "verdict": VERDICT_REF,
        "old_train_sha256": OLD_TRAIN_SHA,
        "new_train_sha256": new_sha,
        "id_scheme": pin["correction_build4_1"]["id_scheme"]["after"],
        "content_identity_check": "PASS (0 discrepancies)",
        "changed_pair_ids": changed,
        "asserts_build41": {k: v["ok"] for k, v in legs.items()},
    }
    for mpath in (MANIFEST, SEAL_REPORT):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["fingerprints"]["train.jsonl"] = new_sha
        m["pin_sha256"] = new_pin_sha
        m["correction_build4_1"] = correction_short
        mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # run-log (slice conflict-detector-corpus-v2-build4)
    drv.log4("build4_1_correction_start", verdict=VERDICT_REF,
             reason="4 pair_id collisions x4 rows (trainer P5)",
             old_train_sha256=OLD_TRAIN_SHA,
             old_carriers_module_sha256=OLD_MODULE_SHA,
             new_carriers_module_sha256=new_module_sha)
    drv.log4("build4_1_census", verdict="pass", total=len(carriers),
             quotas=dict(c1_by_s),
             enumerated_rekey={"collisions": rekey["collisions"],
                               "renamed": rekey["renamed"]})
    drv.log4("build4_1_content_identity", method="sorted content tuples "
             "(all fields except pair_id/scenario_id)", rows=len(new_rows),
             carrier_rows=len(new_carriers), discrepancies=0,
             changed_pair_ids=changed, changed_rows=n_changed_rows,
             check="PASS")
    drv.log4("build4_1_asserts", verdict="PASS",
             legs={k: v["ok"] for k, v in legs.items()},
             new_leg="T-id-unique (global pair_id uniqueness)")
    drv.log4_md("build4_1_correction", old_sha256=OLD_TRAIN_SHA,
                new_sha256=new_sha,
                id_scheme=pin["correction_build4_1"]["id_scheme"]["after"],
                content_identity_check="PASS",
                asserts=f"{len(legs)}/{len(legs)} PASS",
                changed_pair_ids=changed)
    drv.log4_md("build4_1_asserts", verdict="PASS",
                legs={k: v["ok"] for k, v in legs.items()},
                new_leg="T-id-unique")
    drv.log4("build4_1_correction", old_sha256=OLD_TRAIN_SHA,
             new_sha256=new_sha,
             id_scheme=pin["correction_build4_1"]["id_scheme"]["after"],
             content_identity_check="PASS")
    drv.log4("build4_1_done", ok=True, mode=mode,
             elapsed_s=round(time.time() - t0, 1),
             pin_sha256=new_pin_sha, eval_side_untouched=True)

    print(f"APPLIED: train-v4 {old_sha[:16]} -> {new_sha[:16]}; "
          f"pin sha {new_pin_sha[:16]}; backups: {BACKUP_TRAIN.name}, "
          f"{BACKUP_PIN.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
