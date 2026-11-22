#!/usr/bin/env python3
"""B6-conflict cycle-2 §7.2 LEG — the NEW artifact on the SAME E2 matter
(ADDENDUM-7 §7 рулинг, постановление 2; one-shot), cycle2_e2_char.py shape.

The cycle-2 retrain artifact (models/conflict-detector-v4, capacity trim C1,
orbit carriers into train) runs the EXACT cycle2_e2_char.py matter:
 - the SAME G10-census clean pool supply (mechanical_ok -> F10 sha-cull ->
   side-text dedup), reproduced machine-identically with the reproduction
   assert against the census_v4_g10 event;
 - the SAME seal legality: CD-holdout-v4 (76c12ad1…) is SEALED — the 180
   seal ids are METADATA, excluded from the contour population BEFORE any
   verdict; the sealed file is never opened, never hashed, never evaluated;
 - the SAME match-cells band x len Delta table with Wilson 95% per
   proportion + Newcombe hybrid-score CI, the SAME ratified band pin
   (|Delta| <= 0.15 AND the interval covers 0 — NOT modified);
 - the SAME scope NC-2/NC-3/NC-5 (NC-4 = C3 promotion record, out of scope;
   CD-*/NC-1 = no v4-bank rows).

EXPECTATION (the §7.2 leg): NC-2 Delta -> 0 (|Delta| <= 0.15 AND CI covers
0). The E2 RED of the OLD model (char2, +0.3577) was the EXPECTED exposure
of the diagnosed cycle-1 defect (non-monotonic razor boundary); if the NEW
artifact ALSO flips the new NC-2 refine combos, THIS IS A CYCLE-2 DEFECT
(model/bank) -> full STOP per the F1 path (instrument verdict RED, no gate,
escalation TL/MLC) — never a re-roll, never a threshold search.

ONE-SHOT (P8-class): the slice ``cycle2-e2-new-artifact`` accepts exactly
one run — a re-roll refuses at the run-log guard; a crash after
cycle2_char_start spends the run (visible instrument_abort, no silent
retry).

Run-log slice: cycle2-e2-new-artifact. Numbers-only copy:
artifacts/runs/cycle2_b6_e2_new_artifact_<date>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402 (frozen corpus lib)
import build_conflict_detector_corpus as cb  # noqa: E402 (frozen build recipe)
import conflict_corpus_banks_v3 as banks3  # noqa: E402 (frozen v3 banks, G1)
import conflict_corpus_banks_v4 as banks4  # noqa: E402 (frozen v4 banks, G1)

ART = REPO / "artifacts" / "runs"
RUNLOG = ART / "run_log.jsonl"
RUNLOG_MD = REPO / "docs" / "experiments" / "conflict-detector-train-runlog.md"
PIN_V4 = REPO / "data" / "conflict-detector" / "pin-v4.json"
ADDENDUM7 = REPO / "docs" / "experiments" / "conflict-detector-corpus-design-ADDENDUM-7.md"
OUT_JSON = ART / "cycle2_b6_e2_new_artifact_2026-11-22.json"

SLICE = "cycle2-e2-new-artifact"
SEED_HOLD_V4 = 7047
V3_PREFIX = "ho3"
V4_PREFIX = "ho4"

#: Pre-registered scope and band (ADDENDUM-7 §2/§4 pin — verbatim form).
CHAR_STRATA = ("NC-2", "NC-3", "NC-5")
GATE_ABS_DELTA = 0.15
Z95 = 1.959963985  # two-sided 95% (same constant as char2 / runner E8-ii)

#: The NEW cycle-2 artifact identity (dir pin; the onnx sha is recorded at
#: run time and cross-checked against the bundle manifest — the trim was
#: ratified with a fresh artifact, the sha was never pre-registered).
NEW_ARTIFACT_DIR = "models/conflict-detector-v4"

#: The census target event (machine target, no hand-copied numbers).
CENSUS_EVENT = "census_v4_g10"
CENSUS_SLICE = "conflict-detector-corpus-v2-build4"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields) -> None:
    row = {"ts": now(), "event": event, "slice": SLICE, **fields}
    line = json.dumps(row, ensure_ascii=False)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(f"[run-log] {event} {fields if fields else ''}", flush=True)


class InstrumentAbort(Exception):
    """Fail-closed abort (runner convention)."""


#: selfcheck mode: instrument validation — NO run-log events (an event here
#: would spend the one-shot before the real run ever started).
SELFCHECK = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die_abort(msg: str) -> None:
    if not SELFCHECK:
        log("instrument_abort", reason=msg, abort_class="instrument",
            note="one-shot discipline: a rerun of this slice REFUSES at the "
                 "run-log guard (P8-class); escalation TL/MLC, no re-roll")
    print(f"CYCLE2 §7.2 INSTRUMENT ABORT: {msg}", file=sys.stderr)
    sys.exit(4)


# --- statistics (pre-registered CI machinery — byte-same as char2) -----------

def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denom
    half = (Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))) / denom
    return center - half, center + half


def newcombe_delta(k1: int, n1: int, k0: int, n0: int) -> dict:
    """Newcombe hybrid-score interval for p1 - p0 (method 10, Newcombe 1998)
    built from the two Wilson intervals — the pre-registered Delta CI."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    lo = (p1 - p0) - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = (p1 - p0) + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return {
        "delta": round(p1 - p0, 4),
        "p_v4": round(p1, 4), "k_v4": k1, "n_v4": n1,
        "p_old": round(p0, 4), "k_old": k0, "n_old": n0,
        "wilson95_v4": [round(l1, 4), round(u1, 4)],
        "wilson95_old": [round(l0, 4), round(u0, 4)],
        "delta_ci95_newcombe": [round(lo, 4), round(hi, 4)],
        "covers_zero": bool(lo <= 0.0 <= hi),
    }


# --- one-shot guard (P8-class) -----------------------------------------------

def one_shot_guard() -> None:
    for path in (RUNLOG, RUNLOG_MD):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("slice") == SLICE and ev.get("event") in (
                "cycle2_char_start", "cycle2_char_done", "instrument_abort",
            ):
                print(
                    f"REFUSE: slice {SLICE} already has event "
                    f"'{ev['event']}' ({ev.get('ts')}) in {path.name} — "
                    "ONE-SHOT spent, re-rolls forbidden (P8-class)",
                    file=sys.stderr,
                )
                sys.exit(3)


# --- frozen-banks / ratification ancestry (G1) --------------------------------

def census_target() -> dict:
    """The build4 census event — machine target for the reproduction assert."""
    got = None
    for path in (RUNLOG,):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == CENSUS_EVENT and ev.get("slice") == CENSUS_SLICE:
                got = ev
    if got is None:
        die_abort(f"run-log has no {CENSUS_EVENT} event (slice {CENSUS_SLICE}) "
                  "— census must precede the §7.2 leg")
    if got.get("verdict") != "pass":
        die_abort("build4 census verdict is not pass — the §7.2 leg refuses "
                  "to run on a red census")
    return got["g10"]


def assert_ancestry() -> dict:
    if "ADDENDUM-7 РАТИФИЦИРОВАН" not in ADDENDUM7.read_text(encoding="utf-8"):
        die_abort("ADDENDUM-7 ratified marker missing at HEAD (fail-closed)")
    pin = json.loads(PIN_V4.read_text(encoding="utf-8"))
    prep = pin.get("prep", {})
    correction = pin.get("correction_build4_1") or {}
    frozen_checks = (
        ("banks_v4", "scripts/conflict_corpus_banks_v4.py"),
        ("banks_v3_continuity", "scripts/conflict_corpus_banks_v3.py"),
    )
    shas = {}
    basis = {}
    for key, rel in frozen_checks:
        node = prep.get(key)
        if not node or "file_sha256" not in node:
            die_abort(f"pin-v4.json prep.{key} missing (freeze first, G1)")
        cur = sha256_file(REPO / rel)
        if cur != node["file_sha256"]:
            die_abort(f"G1 LOCK VIOLATION: {rel} sha {cur[:16]} != frozen "
                      f"{node['file_sha256'][:16]} — edited after freeze")
        shas[key] = cur
        basis[key] = "prep-freeze"
    # carriers_c1: the G1 freeze covers the CONTENT config; the build4.1
    # correction (ADDENDUM-7 §6, TL verdict @ d61f2de) LEGALLY replaced the
    # emission machinery (id-namespace counter fix) — the correction record
    # pins the new module sha and proves the content config unchanged. The
    # module sha must match EITHER the prep freeze OR the correction record;
    # the config content sha must be UNCHANGED either way.
    node_c = prep.get("carriers_c1")
    if not node_c or "file_sha256" not in node_c:
        die_abort("pin-v4.json prep.carriers_c1 missing (freeze first, G1)")
    cur_c = sha256_file(REPO / "scripts/conflict_corpus_carriers_v4.py")
    want_freeze = node_c["file_sha256"]
    want_correction = correction.get("new_carriers_module_sha256")
    if cur_c == want_freeze:
        basis["carriers_c1"] = "prep-freeze"
    elif want_correction and cur_c == want_correction:
        frozen_cfg = node_c.get("content_sha256")
        unchanged = correction.get("carriers_config_content_sha256_unchanged")
        if frozen_cfg and unchanged and frozen_cfg != unchanged:
            die_abort("build4.1 correction basis RED: the correction record's "
                      "config content sha does not match the frozen "
                      "content_sha256 — content drift under a machinery fix")
        basis["carriers_c1"] = "build4.1-correction (ADDENDUM-7 §6; module "
        "machinery replaced content-identically)"
    else:
        die_abort(f"G1 LOCK VIOLATION: scripts/conflict_corpus_carriers_v4.py "
                  f"sha {cur_c[:16]} matches NEITHER the frozen sha "
                  f"{want_freeze[:16]} NOR the build4.1 correction sha "
                  f"{(want_correction or '')[:16]} — edited outside the "
                  "ratified correction")
    shas["carriers_c1"] = cur_c
    if basis["carriers_c1"].startswith("build4.1") and \
            not correction.get("eval_side_untouched"):
        die_abort("build4.1 correction record does not assert "
                  "eval_side_untouched — seal legality cannot be inherited")
    train_node = pin.get("train_v4") or {}
    train_sha = train_node.get("train.jsonl_sha256") or train_node.get("sha256")
    if not train_sha:
        die_abort("pin-v4.json train_v4 sha missing — the corrected train-v4 "
                  "is not pinned (build4.1)")
    cur_train = sha256_file(REPO / "data/conflict-detector/corpus-v4/train.jsonl")
    if cur_train != train_sha:
        die_abort(f"train-v4 sha RED: {cur_train[:16]}… != pin {train_sha[:16]}…")
    if correction and correction.get("new_train_sha256") and \
            cur_train != correction["new_train_sha256"]:
        die_abort("train-v4 sha does not match the build4.1 correction "
                  "record's new_train_sha256")
    seal_ids = pin.get("seal_ids")
    if not isinstance(seal_ids, list) or len(seal_ids) != 180:
        die_abort("pin-v4.json seal_ids missing or != 180 — seal metadata "
                  "incomplete, the §7.2 leg cannot exclude the sealed surface")
    g10 = census_target()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=REPO).stdout.strip().splitlines()
    return {
        "head": head,
        "banks_v4_file_sha256": shas["banks_v4"],
        "banks_v3_file_sha256": shas["banks_v3_continuity"],
        "carriers_c1_file_sha256": shas["carriers_c1"],
        "g1_basis": basis,
        "train_v4_sha256": cur_train,
        "seal_ids_n": len(seal_ids),
        "dirty_paths": [d.split()[-1] for d in dirty],
        "sealed_files_opened": False,
    }, g10, seal_ids


# --- pool build + census clean replication (G10 supply) -----------------------

def build_clean_supply(log_fn, g10: dict) -> dict[str, dict[str, list]]:
    """Reproduce the build4 census clean supply for the characterization
    strata (the build's own machine recipe, byte-same call shapes)."""
    classic = cb.gen_pool(SEED_HOLD_V4, "ho", rotate=True)
    v3pool = banks3.gen_banks_v3_pool(SEED_HOLD_V4, V3_PREFIX, rotate=True)
    v4pool = banks4.gen_banks_v4_pool(SEED_HOLD_V4, V4_PREFIX, rotate=True)
    cells = [x for x in classic + v3pool + v4pool if x.stratum in CHAR_STRATA]
    t0 = time.time()
    cos = cb.Cos()  # pinned nano verified at construction (census instrument)
    cb.measure_all(cells, cos)
    log_fn("cycle2_embed_pass", texts_embedded=cos.n, cells=len(cells),
           elapsed_s=round(time.time() - t0, 1),
           instrument="cb.Cos pinned nano (census instrument)",
           note="embed pass covers the full census pool incl. sealed-id rows "
                "(the build's own machine pass, counts only); contour "
                "evaluation below EXCLUDES every sealed id")

    tr_texts, _ = cb.disk_text_shas(cb.CORPUS_DIR_V2 / "train.jsonl")
    v3_shas = cb.sealed_generation_sha_set(cb.SEAL_V3_PATH,
                                           cb.CORPUS_DIR_V3 / "holdout-pool.jsonl")
    v2_shas = cb.sealed_generation_sha_set(cb.SEAL_V2_PATH,
                                           cb.CORPUS_DIR_V2 / "holdout-pool.jsonl")
    v1_shas = cb.sealed_generation_sha_set(cb.SEAL_V1_PATH,
                                           cb.CORPUS_DIR / "holdout-pool.jsonl")
    sources = {"train-v2": tr_texts, "v3-seal": v3_shas, "v2-seal": v2_shas,
               "v1-seal": v1_shas}

    out: dict[str, dict[str, list]] = {}
    for s in CHAR_STRATA:
        sc_cells = [x for x in cells if x.stratum == s]
        live = [x for x in sc_cells if lib.mechanical_ok(x)]
        live.sort(key=lambda x: (-x._cos, x.scenario_id))
        seen: set[str] = set()
        culled: dict[str, int] = {}
        dups = 0
        clean_rows = []
        for x in live:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            src = cb._forbidden_source(ha, hb, sources)
            if src:
                culled[src] = culled.get(src, 0) + 1
                continue
            if ha in seen or hb in seen:
                dups += 1
                continue
            seen.update((ha, hb))
            clean_rows.append(x)
        origin_v4 = sum(1 for x in clean_rows
                        if x.scenario_id.startswith(V4_PREFIX + "-"))
        origin_v3 = sum(1 for x in clean_rows
                        if x.scenario_id.startswith(V3_PREFIX + "-"))
        got = {
            "pool": len(sc_cells),
            "preflight_attrited": len(sc_cells) - len(live),
            "culled_f10": culled,
            "dup_blocked": dups,
            "clean": len(clean_rows),
            "clean_v4": origin_v4,
            "clean_v3": origin_v3,
            "clean_classic": len(clean_rows) - origin_v4 - origin_v3,
        }
        want = {k: g10[s][k] for k in got}
        if got != want:
            die_abort(
                f"census reproduction RED for {s}: got {got} vs census {want} "
                "— clean supply is not the ratified build4 census supply"
            )
        v4_rows = [x for x in clean_rows
                   if x.scenario_id.startswith(V4_PREFIX + "-")]
        old_rows = [x for x in clean_rows
                    if not x.scenario_id.startswith(V4_PREFIX + "-")]
        out[s] = {"v4": v4_rows, "old": old_rows, "clean": clean_rows}
        log_fn("cycle2_clean_reproduced", stratum=s, **got)
    return out


def apply_seal_exclusion(supply, seal_ids, log_fn):
    """Seal legality (cycle-2): the 180 sealed ids are METADATA — excluded
    from the contour population BEFORE any verdict. Counts published."""
    seal_set = set(seal_ids)
    out = {}
    for s in CHAR_STRATA:
        rows = supply[s]["clean"]
        excluded = [x for x in rows if x.scenario_id in seal_set]
        kept = [x for x in rows if x.scenario_id not in seal_set]
        exc_v4 = sum(1 for x in excluded
                     if x.scenario_id.startswith(V4_PREFIX + "-"))
        exc_v3 = sum(1 for x in excluded
                     if x.scenario_id.startswith(V3_PREFIX + "-"))
        kept_v4 = [x for x in kept
                   if x.scenario_id.startswith(V4_PREFIX + "-")]
        kept_old = [x for x in kept
                    if not x.scenario_id.startswith(V4_PREFIX + "-")]
        out[s] = {"v4": kept_v4, "old": kept_old}
        log_fn("cycle2_seal_exclusion", stratum=s,
               seal_ids_total=len(seal_ids),
               pool_rows_excluded=len(excluded),
               excluded_origins={"v4bank": exc_v4, "v3bank": exc_v3,
                                 "classic_or_other": len(excluded) - exc_v4 - exc_v3},
               post_seal_v4=len(kept_v4), post_seal_old=len(kept_old),
               legality="seal ids used as METADATA only (pin-v4 seal_ids); "
               "sealed rows NEVER evaluated by the contour, never printed; "
               "cut was hardest-first -> post-cut remainder disclosed",
               sealed_file_opened=False)
    return out


# --- production contour over pool rows (the NEW cycle-2 artifact) --------------

def scenario_row(sc) -> dict:
    """Pool row dict in the corpus convention (char2 verbatim shape)."""
    return {
        "pair_id": sc.scenario_id,
        "stratum": sc.stratum,
        "label": sc.label,
        "orientation": "a>b",
        "measured": {"cos_target": sc._cos},
        "envelope_a": cb.side_dict(sc.side_a),
        "envelope_b": cb.side_dict(sc.side_b),
    }


def run_contour(rows: list, booster) -> list[dict]:
    import train_conflict_detector as tcd

    out = []
    for sc in rows:
        row = scenario_row(sc)
        rec_a, rec_b = tcd.row_records(row)
        d = tcd.decide_pipeline(rec_a, rec_b, float(row["measured"]["cos_target"]),
                                booster)
        d["band"] = sc._band
        d["len"] = sc.length_channel
        out.append(d)
    return out


def characterize_stratum(s: str, pop: dict, booster, log_fn) -> dict:
    v4_res = run_contour(pop["v4"], booster)
    old_res = run_contour(pop["old"], booster)

    def cellize(res):
        cells: dict[tuple[str, str], dict] = {}
        for d in res:
            c = cells.setdefault((d["band"], d["len"]), {"n": 0, "k": 0})
            c["n"] += 1
            c["k"] += int(d["verdict"] == "contradicts")
        return cells

    v4_cells, old_cells = cellize(v4_res), cellize(old_res)
    match = sorted(set(v4_cells) & set(old_cells))
    per_cell = [
        {"band": b, "len": ln,
         "v4": {"k": v4_cells[(b, ln)]["k"], "n": v4_cells[(b, ln)]["n"]},
         "old": {"k": old_cells[(b, ln)]["k"], "n": old_cells[(b, ln)]["n"]}}
        for b, ln in match
    ]
    v4_only = sorted(set(v4_cells) - set(old_cells))
    old_only = sorted(set(old_cells) - set(v4_cells))

    info = {
        "stratum": s,
        "cells_v4": {f"{b}/{ln}": v4_cells[(b, ln)] for b, ln in sorted(v4_cells)},
        "cells_old": {f"{b}/{ln}": old_cells[(b, ln)] for b, ln in sorted(old_cells)},
        "match_cells": per_cell,
        "v4_only_cells": [f"{b}/{ln}" for b, ln in v4_only],
        "old_only_cells": [f"{b}/{ln}" for b, ln in old_only],
        "leg_v4": dict(__import__("collections").Counter(d["leg"] for d in v4_res)),
        "leg_old": dict(__import__("collections").Counter(d["leg"] for d in old_res)),
    }
    if not match:
        info.update({
            "match": False,
            "note": "honest NO MATCH: no (band, len) cell holds clean rows of "
                    "both the v4 bank and the old material — no Delta is "
                    "fabricated (G4)",
        })
        log_fn("cycle2_stratum_no_match", stratum=s,
               cells_v4=info["cells_v4"], cells_old=info["cells_old"])
        return info

    k1 = sum(v4_cells[c]["k"] for c in match)
    n1 = sum(v4_cells[c]["n"] for c in match)
    k0 = sum(old_cells[c]["k"] for c in match)
    n0 = sum(old_cells[c]["n"] for c in match)
    st = newcombe_delta(k1, n1, k0, n0)
    gate = (abs(st["delta"]) <= GATE_ABS_DELTA) and st["covers_zero"]
    info.update({
        "match": True,
        "stats": st,
        "gate": {"band": "|Delta| <= 0.15 AND delta CI95 covers 0",
                 "abs_delta_ok": bool(abs(st["delta"]) <= GATE_ABS_DELTA),
                 "ci_covers_zero": st["covers_zero"],
                 "verdict": "GREEN" if gate else "RED"},
    })
    log_fn("cycle2_stratum_verdict", stratum=s, delta=st["delta"],
           delta_ci95=st["delta_ci95_newcombe"], k_v4=k1, n_v4=n1,
           k_old=k0, n_old=n0, gate=info["gate"]["verdict"])
    return info


# --- selfcheck (instrument validation; NO events, NO embeds, NO model) --------

def selfcheck() -> int:
    anc, g10, seal_ids = assert_ancestry()
    print("selfcheck: ancestry OK", anc["head"][:12],
          "banks_v4", anc["banks_v4_file_sha256"][:12],
          "train_v4", anc["train_v4_sha256"][:12],
          "seal_ids", anc["seal_ids_n"])
    classic = cb.gen_pool(SEED_HOLD_V4, "ho", rotate=True)
    v3pool = banks3.gen_banks_v3_pool(SEED_HOLD_V4, V3_PREFIX, rotate=True)
    v4pool = banks4.gen_banks_v4_pool(SEED_HOLD_V4, V4_PREFIX, rotate=True)
    for s in CHAR_STRATA:
        n_cl = sum(1 for x in classic if x.stratum == s)
        n_v3 = sum(1 for x in v3pool if x.stratum == s)
        n_v4 = sum(1 for x in v4pool if x.stratum == s)
        pool_want = g10[s]["pool"]
        assert n_cl + n_v3 + n_v4 == pool_want, (s, n_cl, n_v3, n_v4, pool_want)
        print(f"selfcheck: {s} pool classic={n_cl} v3={n_v3} v4={n_v4} "
              f"== census pool {pool_want}")
    pool_ids = {x.scenario_id for x in classic + v3pool + v4pool}
    hits = [i for i in seal_ids if i in pool_ids]
    print(f"selfcheck: seal ids present in regenerated pools: {len(hits)}/180 "
          "(the rest are carrier/train-origin rows outside the eval pools)")
    print("selfcheck PASS — instrument is sound for the one-shot run")
    return 0


# --- main (ONE-SHOT) ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true",
                    help="instrument validation only: no events, no embeds, "
                         "no model load (does NOT spend the one-shot)")
    args = ap.parse_args()
    if args.selfcheck:
        global SELFCHECK
        SELFCHECK = True
        return selfcheck()

    one_shot_guard()
    anc, g10, seal_ids = assert_ancestry()

    log("cost_declaration",
        expected_prolog_s=5,
        expected_cycle_s=420,
        expected_peak_ram_gb=2.0,
        lane="§7.2 leg (ADDENDUM-7 §7 ruling item 2): the NEW cycle-2 "
             "artifact over the SAME E2 matter as char2 (census-clean pool "
             "rows of NC-2/NC-3/NC-5, match-cells band x len, Newcombe "
             "Delta CI), seal-180 ids excluded at metadata level; pool-only, "
             "sealed holdout file NOT opened, weights loaded frozen")

    # Pre-registration BEFORE start (the §7.2 expectation, pinned verbatim).
    log("cycle2_preregister_new_artifact",
        band_pin="|Delta| <= 0.15 with a CI covering 0 (ADDENDUM-7 §2/§4 — "
                 "same ratified form as ADDENDUM-6 §5 G4 / char2, NOT "
                 "modified)",
        expectation="§7.2: the NEW artifact must land the NC-2 Delta -> 0 "
                    "(|Delta| <= 0.15 AND CI covers 0); a RED of the NEW "
                    "model on the E2 matter = a cycle-2 defect (model/bank) "
                    "-> STOP per the F1 path (no gate, escalation TL/MLC)",
        anchor="char2 RED (+0.3577) on the OLD model over label-correct new "
               "banks in the known defect area = expected exposure of the "
               "training target (ADDENDUM-7 §7 ruling); the arbiter is the "
               "NEW model on the SAME matter",
        delta_definition="per-stratum P(contradicts | v4-bank rows) - "
                         "P(contradicts | old-material rows) pooled over "
                         "MATCH-CELLS stratum x band x len (cells with clean "
                         "rows of both groups); Wilson 95% per proportion + "
                         "Newcombe hybrid-score interval for Delta",
        scope=list(CHAR_STRATA),
        out_of_scope={
            "NC-4": "C3 promotion record lives in the seal; no v4-bank pool "
                    "rows — disclosed out-of-scope, not skipped silently",
            "CD-*/NC-1": "no v4-bank rows in the census (clean_v4 = 0)",
            "C1 carriers": "train-side material, not exam rows",
        },
        population="build4 G10 census clean supply (mechanical_ok -> F10 vs "
                   "train-v2/v1-seal/v2-seal/v3-seal -> dedup), origins by id "
                   "prefix (ho4- / ho3- / classic); reproduction assert vs "
                   "the census_v4_g10 run-log event",
        seal_legality="CD-holdout-v4 SEALED: the 180 seal ids excluded from "
                      "the contour population at metadata level BEFORE any "
                      "verdict; sealed rows ride the census-reproduction "
                      "embed pass only (counts, no verdict, no content "
                      "logged); cut was hardest-first -> post-cut remainder "
                      "disclosed; single-shot gate verdict stays fresh",
        contour="production: version-guard (one-sided NOT) -> model leg, "
                "cut 0.5, no threshold search; the NEW cycle-2 artifact "
                f"{NEW_ARTIFACT_DIR} (capacity trim C1, orbit carriers)",
        one_shot="P8-class: single run; re-rolls forbidden; all numbers from "
                 "this run only",
        **{k: v for k, v in anc.items() if k != "census_event_ts"})

    log("cycle2_char_start",
        ratification="ADDENDUM-7 §7 рулинг (постановление 2) over the "
                     "ratified §2/§4 form",
        legality="pool-only: pool rows + seal-manifest sha sets + pin seal-id "
                 "metadata only; sealed holdout.jsonl v4 NOT opened; prints "
                 "not read; weights unchanged; nothing cut, nothing sealed")

    t0 = time.time()
    import train_conflict_detector as tcd  # after sys.path setup

    ns = argparse.Namespace(
        corpus_dir=str(REPO / "data" / "conflict-detector" / "corpus-v4"),
        pin=None, seal=None, artifact_dir=None,
    )
    tcd.apply_input_params(ns)
    if tcd.ART_DIR != REPO / NEW_ARTIFACT_DIR:
        die_abort(f"artifact dir RED: {tcd.ART_DIR} != {NEW_ARTIFACT_DIR}")
    booster, manifest = tcd.load_frozen()
    if str(manifest.get("candidate") or manifest.get("name")) != "conflict-detector-v4":
        die_abort("artifact identity RED: manifest candidate/name is not "
                  "conflict-detector-v4")
    if "ADDENDUM-7 §2 C1" not in str(manifest.get("artifact_lineage", "")):
        die_abort("artifact lineage RED: the cycle-2 C1 ratification marker "
                  "is missing from manifest.artifact_lineage")
    if manifest["lgb_params"].get("num_leaves") != 8 or \
            manifest["lgb_params"].get("min_child_samples") != 20:
        die_abort("capacity trim RED: the manifest lgb_params are not the "
                  "ratified C1 trim (num_leaves 8, min_child_samples 20)")
    log("cycle2_model_loaded", artifact=NEW_ARTIFACT_DIR,
        manifest_sha256=manifest["sha256"], trained_at=manifest.get("trained_at"),
        feature_contract="CORE-13 frozen", cut=0.5,
        lgb_params=manifest["lgb_params"],
        bundle_integrity="sha256(model.onnx) == manifest.sha256 (load_frozen)")

    supply = build_clean_supply(log, g10)
    population = apply_seal_exclusion(supply, seal_ids, log)

    per_stratum = {}
    reds = []
    no_match = []
    for s in CHAR_STRATA:
        info = characterize_stratum(s, population[s], booster, log)
        per_stratum[s] = info
        if not info.get("match"):
            no_match.append(s)
        elif info["gate"]["verdict"] == "RED":
            reds.append(s)

    band_verdict = "RED" if reds else "GREEN"
    if no_match:
        band_verdict = f"GREEN-ON-MATCH (no match: {', '.join(no_match)})"
    verdict = "RED" if reds else "GREEN"

    result = {
        "ts": now(),
        "slice": SLICE,
        "head": anc["head"],
        "ratification": "ADDENDUM-7 §7 item 2 (form of ADDENDUM-6 §5 G4)",
        "artifact": {"dir": NEW_ARTIFACT_DIR, "sha256": manifest["sha256"]},
        "band_pin": "|Delta| <= 0.15 AND delta CI95 covers 0",
        "scope": list(CHAR_STRATA),
        "seal_exclusion": "180 seal ids excluded at metadata level; "
                          "post-cut remainder population; sealed file never opened",
        "per_stratum": {
            s: {k: v for k, v in info.items() if k != "stratum"}
            for s, info in per_stratum.items()
        },
        "no_match_strata": no_match,
        "band_verdict": band_verdict,
        "verdict": verdict,
        "anchor": "char GREEN (§7.2 expectation) -> gate proceeds per the "
                  "TL command; char RED = cycle-2 defect (model/bank) -> "
                  "STOP per the F1 path, no gate",
        "numbers_only": True,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    log("cycle2_char_done",
        verdict=verdict,
        band_verdict=band_verdict,
        delta={s: (per_stratum[s]["stats"]["delta"] if per_stratum[s].get("match")
                   else "no-match") for s in CHAR_STRATA},
        delta_ci95={s: (per_stratum[s]["stats"]["delta_ci95_newcombe"]
                        if per_stratum[s].get("match") else None)
                    for s in CHAR_STRATA},
        artifact_json=str(OUT_JSON.relative_to(REPO)),
        elapsed_s=result["elapsed_s"],
        next=("§7.2 GREEN -> the cycle-2 pipeline (train/floors/sanity) is "
              "instrument-complete; the gate is TL-commanded"
              if verdict == "GREEN"
              else "RED -> STOP per the F1 path: cycle-2 defect (model/bank "
                   "class), full teardown report, NO gate, escalation "
                   "TL/MLC; no re-roll, no threshold search"))

    print(f"\nCYCLE2 §7.2 NEW-ARTIFACT VERIFICATION: {verdict}")
    for s in CHAR_STRATA:
        info = per_stratum[s]
        if info.get("match"):
            st = info["stats"]
            print(f"  {s}: Delta={st['delta']:+.4f} "
                  f"CI95 [{st['delta_ci95_newcombe'][0]:+.4f}, "
                  f"{st['delta_ci95_newcombe'][1]:+.4f}] "
                  f"(v4 {st['k_v4']}/{st['n_v4']}, old {st['k_old']}/{st['n_old']}) "
                  f"gate={info['gate']['verdict']}")
        else:
            print(f"  {s}: NO MATCH (honest)")
    return 0 if verdict == "GREEN" else 2


if __name__ == "__main__":
    sys.exit(main())
