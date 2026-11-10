#!/usr/bin/env python3
"""B6-conflict cycle-2 E2 CLASS CHARACTERIZATION — ADDENDUM-7 §2/§4 (E2-класс,
one-shot), same ratified form as char2 (ADDENDUM-6 §5 G4 precedent, a9ac15d).

The FROZEN conflict-detector-v2 production contour (artifact 2534fcbe) over
the G10-census clean pool rows of the build4 universes, BEFORE any v4
training. ONE-SHOT (P8-class): the slice ``conflict-detector-v2-cycle2``
accepts exactly one characterization run — a re-roll refuses at the run-log
guard; a crash after cycle2_char_start spends the run (visible
instrument_abort, no silent retry).

PRE-REGISTERED BAND (ratification pin, NOT modified):
    per-stratum Delta = P(contradicts | v4-bank rows) - P(contradicts |
    old-material rows) on MATCH-CELLS stratum x band x len (cells where both
    groups hold clean rows); Wilson 95% CI per proportion, Newcombe
    hybrid-score interval for Delta; gate per stratum: |Delta| <= 0.15 AND
    the Delta interval covers 0. RED => STOP + MLC escalation — NOT a
    template edit (banks are frozen under G1).

Scope (pinned pre-run): NC-2, NC-3, NC-5 — the strata holding v4-bank
(clean_v4) rows in the build4 census. NC-4 is OUT OF SCOPE (C3: merged
promotion record lives in the seal, no v4-bank pool rows); CD-* and NC-1
carry no v4-bank rows (disclosed, never fabricated). C1 orbit carriers are
TRAIN-side material — not exam rows, out of characterization scope.

Population (pinned pre-run): the G10 census clean supply (mechanical_ok ->
F10 sha-cull vs train-v2 / v1-seal / v2-seal / v3-seal -> side-text dedup),
origins split by id prefix (ho4- = v4 bank, ho3- = v3 bank, else classic).
Reproduction assert against the census_v4_g10 event of the build4 lane in
the run-log — any mismatch is an instrument abort, never a silent
re-derivation.

SEAL LEGALITY (cycle-2 difference vs char2): CD-holdout-v4 is SEALED
(single legal read belongs to the gate runner). The 180 seal ids are used
as METADATA ONLY: every sealed id is excluded from the contour population
BEFORE any verdict is computed. Sealed rows ride NOTHING but the census
reproduction embed pass (the build's own machine pass; counts only, no
verdict, no content logged). The cut was hardest-first, so the
characterization runs on the POST-CUT REMAINDER — disclosed in every
report. The single-shot gate verdict stays fresh: no sealed row is
evaluated, printed, or unsealed here.

Instrument: engine venv python + APPENDED vesma engine path; pinned nano
verified at Cos construction; taskset -c 0-12; PYTHONDONTWRITEBYTECODE off.
Run-log slice: conflict-detector-v2-cycle2.
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
OUT_JSON = ART / "cycle2_b6_e2_char_2026-11-10.json"

SLICE = "conflict-detector-v2-cycle2"
SEED_HOLD_V4 = 7047
V3_PREFIX = "ho3"
V4_PREFIX = "ho4"

#: Pre-registered scope and band (ADDENDUM-7 §2/§4 pin — verbatim form).
CHAR_STRATA = ("NC-2", "NC-3", "NC-5")
GATE_ABS_DELTA = 0.15
Z95 = 1.959963985  # two-sided 95% (same constant as char2 / runner E8-ii)

#: Artifact identity pin from the dispatch: frozen v2 artifact 2534fcbe.
V2_ARTIFACT_SHA_PREFIX = "2534fcbe"

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
    print(f"CYCLE2 E2 INSTRUMENT ABORT: {msg}", file=sys.stderr)
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
                  "— census must precede E2")
    if got.get("verdict") != "pass":
        die_abort("build4 census verdict is not pass — E2 refuses to run "
                  "on a red census")
    return got["g10"]


def assert_ancestry() -> dict:
    if "ADDENDUM-7 РАТИФИЦИРОВАН" not in ADDENDUM7.read_text(encoding="utf-8"):
        die_abort("ADDENDUM-7 ratified marker missing at HEAD (fail-closed)")
    pin = json.loads(PIN_V4.read_text(encoding="utf-8"))
    prep = pin.get("prep", {})
    frozen_checks = (
        ("banks_v4", "scripts/conflict_corpus_banks_v4.py"),
        ("banks_v3_continuity", "scripts/conflict_corpus_banks_v3.py"),
        ("carriers_c1", "scripts/conflict_corpus_carriers_v4.py"),
    )
    shas = {}
    for key, rel in frozen_checks:
        node = prep.get(key)
        if not node or "file_sha256" not in node:
            die_abort(f"pin-v4.json prep.{key} missing (freeze first, G1)")
        cur = sha256_file(REPO / rel)
        if cur != node["file_sha256"]:
            die_abort(f"G1 LOCK VIOLATION: {rel} sha {cur[:16]} != frozen "
                      f"{node['file_sha256'][:16]} — edited after freeze")
        shas[key] = cur
    seal_ids = pin.get("seal_ids")
    if not isinstance(seal_ids, list) or len(seal_ids) != 180:
        die_abort("pin-v4.json seal_ids missing or != 180 — seal metadata "
                  "incomplete, E2 cannot exclude the sealed surface")
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
        "seal_ids_n": len(seal_ids),
        "census_event_ts": None,  # filled by caller if needed
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


# --- production contour over pool rows (frozen v2 artifact) --------------------

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
        lane="E2 class characterization cycle-2 (ADDENDUM-7 §2/§4): frozen "
             "v2 contour 2534fcbe over census-clean pool rows of the v4-bearing "
             "strata NC-2/NC-3/NC-5, seal-180 ids excluded at metadata level; "
             "pool-only, sealed holdout file NOT opened, weights unchanged")

    # Pre-registration BEFORE start (the ratification pin, quoted verbatim).
    log("cycle2_preregister",
        band_pin="|Delta| <= 0.15 with a CI covering 0 (ADDENDUM-7 §2/§4 — "
                 "same ratified form as ADDENDUM-6 §5 G4, NOT modified)",
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
                      "disclosed; single-shot verdict stays fresh",
        contour="production: version-guard (one-sided NOT) -> model leg, "
                "cut 0.5, no threshold search; frozen artifact "
                f"models/conflict-detector-v2 (manifest sha {V2_ARTIFACT_SHA_PREFIX}…)",
        one_shot="P8-class: single run; re-rolls forbidden; all numbers from "
                 "this run only",
        **{k: v for k, v in anc.items() if k != "census_event_ts"})

    log("cycle2_char_start",
        ratification="ADDENDUM-7 §2/§4 (MLC-7 3-leg convergence, commit 154f7af)",
        legality="pool-only: pool rows + seal-manifest sha sets + pin seal-id "
                 "metadata only; sealed holdout.jsonl v4 NOT opened; prints "
                 "not read; weights unchanged; nothing cut, nothing sealed")

    t0 = time.time()
    import train_conflict_detector as tcd  # after sys.path setup

    ns = argparse.Namespace(
        corpus_dir=str(REPO / "data" / "conflict-detector" / "corpus-v2"),
        pin=None, seal=None, artifact_dir=None,
    )
    tcd.apply_input_params(ns)
    booster, manifest = tcd.load_frozen()
    if not str(manifest["sha256"]).startswith(V2_ARTIFACT_SHA_PREFIX):
        die_abort(f"artifact identity RED: manifest sha {manifest['sha256'][:16]}… "
                  f"!= dispatch pin {V2_ARTIFACT_SHA_PREFIX}…")
    log("cycle2_model_loaded", artifact="models/conflict-detector-v2",
        manifest_sha256=manifest["sha256"], trained_at=manifest.get("trained_at"),
        feature_contract="CORE-13 frozen", cut=0.5)

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
        "ratification": "ADDENDUM-7 §2/§4 (form of ADDENDUM-6 §5 G4)",
        "artifact": {"dir": "models/conflict-detector-v2",
                     "sha256": manifest["sha256"]},
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
        "anchor": "char GREEN + v4 gate RED = model defect; "
                  "char RED + v4 gate RED = bank drift (G4 form)",
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
        next=("GREEN -> retrain v4 (ёмкость-trim C1) после build4.1 "
              "re-emission"
              if verdict == "GREEN"
              else "RED -> STOP: эскалация MLC (G4), шаблоны НЕ правятся "
                   "(заморожены G1); retrain НЕ исполняется"))

    print(f"\nCYCLE2 E2 CHARACTERIZATION: {verdict}")
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
