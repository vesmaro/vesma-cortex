#!/usr/bin/env python3
"""B6-conflict E2 PRE-CUT CHARACTERIZATION — ADDENDUM-6 §5 G4 (ML-E2).

Mixed run of the FROZEN conflict-detector-v2 production contour over the
clean-after-preflight pool rows of the NEW banks (gen_pool_v3 eval-only
fork) BEFORE the single v3 cut. ONE-SHOT (P8-class): the slice
``conflict-detector-v2-char2`` accepts exactly one run — a re-roll
refuses at the run-log guard, a crash after char2_pre_cut_start spends
the run (visible instrument_abort, no silent retry).

PRE-REGISTERED BAND (ratification pin, NOT modified):
    per-stratum Delta = P(contradicts | new bank) - P(contradicts | old
    bank) on MATCH-CELLS stratum x band x len (cells where both banks
    have clean rows); Wilson 95% CI per proportion, Newcombe hybrid-score
    interval for Delta; gate per stratum: |Delta| <= 0.15 AND the Delta
    interval covers 0. Strata without old parity => honest "no match"
    (disclosed, never fabricated). RED => STOP + MLC escalation — NOT a
    template edit (banks are frozen under G1).

Scope (pinned pre-run): NC-1, NC-3, NC-5 — the strata the v3 cut
actually draws from the NEW banks. NC-4 is OUT OF SCOPE by ratification
(G7: fallback-margin only, never cut from the new bank; the brief
excludes it from the probes for the same reason).

Population (pinned pre-run): the G10 census clean supply (mechanical_ok
-> F10 sha-cull vs train-v2/v1-seal/v2-seal -> side-text dedup), split
classic (prefix "ho-") vs v3 (prefix "ho3-"); reproduction assert
against artifacts/runs/banks_v3_census.json — any mismatch is an
instrument abort, never a silent re-derivation.

Probes (pinned pre-run): §6.3 stability ladder raw/cosmetic/para1
(probe 6: verdict invariant) + cosmetic-twin leg (probe 2: NOT
contradicts) on ONE fixture cell of each new bank NC-1/NC-3/NC-5 — the
same declared preflight fixture cells (identical builders, identical
name-derived seeds); expectations as §6.3; through the production
contour (version-guard -> model leg, cut 0.5, no threshold search).

Legality: pool-only (F7). Sealed holdouts v1/v2 are NEVER opened; F10
sha sets ride seal manifests + pool origins (the census instrument).
Weights unchanged; nothing cut, nothing sealed, nothing printed.

Anchor (G4, for the v3 gate report): characterization GREEN + gate RED
= model defect; characterization RED + gate RED = bank drift.

Instrument: engine venv python + APPENDED vesma engine path; pinned
nano verified at Cos/NanoLeg construction; taskset -c 0-12;
PYTHONDOTWRITEBYTECODE off. Run-log slice: conflict-detector-v2-char2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402 (frozen corpus lib)
import build_conflict_detector_corpus as cb  # noqa: E402 (frozen build recipe)
import conflict_corpus_banks_v3 as banks  # noqa: E402 (frozen NEW banks, G1)

ART = REPO / "artifacts" / "runs"
RUNLOG = ART / "run_log.jsonl"
RUNLOG_MD = REPO / "docs" / "experiments" / "conflict-detector-train-runlog.md"
PIN_V3 = REPO / "data" / "conflict-detector" / "pin-v3.json"
CENSUS_JSON = ART / "banks_v3_census.json"
ADDENDUM6 = REPO / "docs" / "experiments" / "conflict-detector-corpus-design-ADDENDUM-6.md"
OUT_JSON = ART / "char2_pre_cut_2026-11-10.json"

SLICE = "conflict-detector-v2-char2"
SEED_HOLD_V3 = 6117
V3_PREFIX = "ho3"

#: Pre-registered scope and band (ADDENDUM-6 §5 G4 pin — verbatim, unmodified).
CHAR_STRATA = ("NC-1", "NC-3", "NC-5")
GATE_ABS_DELTA = 0.15
Z95 = 1.959963985  # two-sided 95% (same constant as the runner's E8-ii CI)

#: Probe fixtures: the SAME declared preflight cells (builders + name-derived
#: seeds identical to scripts/banks_v3_lane.py preflight declarations).
PROBE_PLAN = {
    "NC-1": {"kind": "nc1", "name": "nc1-single", "args": (0, 0, False)},
    "NC-3": {"kind": "nc3", "name": "nc3-fam0-fl0", "args": (0, "fl0")},
    "NC-5": {"kind": "nc5", "name": None, "args": ("cross",)},  # name from subject
}
#: Ladder RNG — the g_stab machine constant (declared pre-run).
LADDER_SEED = 21

#: Artifact identity pin from the dispatch: v2 artifact 2534fcbe (manifest sha).
V2_ARTIFACT_SHA_PREFIX = "2534fcbe"


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
    print(f"CHAR2 INSTRUMENT ABORT: {msg}", file=sys.stderr)
    sys.exit(4)


# --- statistics (pre-registered CI machinery) --------------------------------

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
        "p_new": round(p1, 4), "k_new": k1, "n_new": n1,
        "p_old": round(p0, 4), "k_old": k0, "n_old": n0,
        "wilson95_new": [round(l1, 4), round(u1, 4)],
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
                "char2_pre_cut_start", "char2_pre_cut_done", "instrument_abort",
            ):
                print(
                    f"REFUSE: slice {SLICE} already has event "
                    f"'{ev['event']}' ({ev.get('ts')}) in {path.name} — "
                    "ONE-SHOT spent, re-rolls forbidden (P8-class)",
                    file=sys.stderr,
                )
                sys.exit(3)


# --- frozen-banks / ratification ancestry (G1) --------------------------------

def assert_ancestry() -> dict:
    if "РАТИФИЦИРОВАНО" not in ADDENDUM6.read_text(encoding="utf-8"):
        die_abort("ADDENDUM-6 ratified marker missing at HEAD (fail-closed)")
    prep = json.loads(PIN_V3.read_text(encoding="utf-8")).get("prep", {}).get("banks_v3")
    if not prep:
        die_abort("pin-v3.json has no banks_v3 prep section (freeze first, G1)")
    cur = sha256_file(REPO / "scripts" / "conflict_corpus_banks_v3.py")
    if cur != prep["file_sha256"]:
        die_abort(f"G1 LOCK VIOLATION: banks module sha {cur[:16]} != frozen "
                  f"{prep['file_sha256'][:16]} — bank edited after freeze")
    if not CENSUS_JSON.exists():
        die_abort("banks_v3_census.json missing — census (G10) must precede E2")
    census = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))
    if census.get("verdict") != "pass":
        die_abort("census verdict is not pass — E2 refuses to run on a red census")
    if census.get("frozen_banks") != prep["content_sha256"]:
        die_abort("census frozen_banks sha != pin prep content sha")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=REPO).stdout.strip().splitlines()
    return {
        "head": head,
        "banks_commit": prep.get("commit") or "066eea6",
        "banks_file_sha256": prep["file_sha256"],
        "banks_content_sha256": prep["content_sha256"],
        "census_ts": census.get("ts"),
        "dirty_paths": [d.split()[-1] for d in dirty],
        "sealed_files_opened": False,
    }


# --- pool build + census clean replication (G10 supply) -----------------------

def build_clean_supply(log_fn) -> dict[str, dict[str, list]]:
    """Reproduce the census clean supply for the characterization strata."""
    classic = cb.gen_pool(SEED_HOLD_V3, "ho", rotate=True)
    v3pool = banks.gen_banks_v3_pool(SEED_HOLD_V3, V3_PREFIX, rotate=True)
    cells = [x for x in classic + v3pool if x.stratum in CHAR_STRATA]
    t0 = time.time()
    cos = cb.Cos()  # pinned nano verified at construction (census instrument)
    cb.measure_all(cells, cos)
    log_fn("char2_embed_pass", texts_embedded=cos.n, cells=len(cells),
           elapsed_s=round(time.time() - t0, 1),
           instrument="cb.Cos pinned nano (census instrument)")

    tr_texts, _ = cb.disk_text_shas(cb.CORPUS_DIR_V2 / "train.jsonl")
    v2_shas = cb.sealed_generation_sha_set(cb.SEAL_V2_PATH,
                                           cb.CORPUS_DIR_V2 / "holdout-pool.jsonl")
    v1_shas = cb.sealed_generation_sha_set(cb.SEAL_V1_PATH,
                                           cb.CORPUS_DIR / "holdout-pool.jsonl")
    sources = {"train-v2": tr_texts, "v2-seal": v2_shas, "v1-seal": v1_shas}

    census = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))["strata"]
    out: dict[str, dict[str, list]] = {}
    for s in CHAR_STRATA:
        sc_cells = [x for x in cells if x.stratum == s]
        live = [x for x in sc_cells if lib.mechanical_ok(x)]
        live.sort(key=lambda x: (-x._cos, x.scenario_id))
        seen: set[str] = set()
        culled: Counter = Counter()
        dups = 0
        clean_rows = []
        for x in live:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            src = cb._forbidden_source(ha, hb, sources)
            if src:
                culled[src] += 1
                continue
            if ha in seen or hb in seen:
                dups += 1
                continue
            seen.update((ha, hb))
            clean_rows.append(x)
        want = census[s]
        got = {
            "pool": len(sc_cells),
            "preflight_attrited": len(sc_cells) - len(live),
            "culled_f10": dict(culled),
            "dup_blocked": dups,
            "clean": len(clean_rows),
            "clean_v3": sum(1 for x in clean_rows
                            if x.scenario_id.startswith(f"{V3_PREFIX}-")),
            "clean_classic": sum(1 for x in clean_rows
                                 if not x.scenario_id.startswith(f"{V3_PREFIX}-")),
        }
        if got != want:
            die_abort(
                f"census reproduction RED for {s}: got {got} vs census {want} "
                "— clean supply is not the ratified census supply"
            )
        v3_rows = [x for x in clean_rows if x.scenario_id.startswith(f"{V3_PREFIX}-")]
        cl_rows = [x for x in clean_rows if not x.scenario_id.startswith(f"{V3_PREFIX}-")]
        out[s] = {"v3": v3_rows, "classic": cl_rows}
        log_fn("char2_clean_reproduced", stratum=s, **got)
    return out


# --- production contour over pool rows ----------------------------------------

def scenario_row(sc) -> dict:
    """Pool row dict in the corpus convention: envelope dicts + measured
    cos_target = the pinned build-time nano measurement (rows_of form)."""
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
        d["pair_id"] = sc.scenario_id
        d["band"] = sc._band
        d["len"] = sc.length_channel
        out.append(d)
    return out


def characterize_stratum(s: str, supply: dict, booster, log_fn) -> dict:
    v3_rows = supply[s]["v3"]
    cl_rows = supply[s]["classic"]
    v3_res = run_contour(v3_rows, booster)
    cl_res = run_contour(cl_rows, booster)

    def cellize(res):
        cells: dict[tuple[str, str], dict] = {}
        for d in res:
            c = cells.setdefault((d["band"], d["len"]), {"n": 0, "k": 0})
            c["n"] += 1
            c["k"] += int(d["verdict"] == "contradicts")
        return cells

    v3_cells, cl_cells = cellize(v3_res), cellize(cl_res)
    match = sorted(set(v3_cells) & set(cl_cells))
    per_cell = [
        {"band": b, "len": ln,
         "v3": {"k": v3_cells[(b, ln)]["k"], "n": v3_cells[(b, ln)]["n"]},
         "classic": {"k": cl_cells[(b, ln)]["k"], "n": cl_cells[(b, ln)]["n"]}}
        for b, ln in match
    ]
    v3_only = sorted(set(v3_cells) - set(cl_cells))
    cl_only = sorted(set(cl_cells) - set(v3_cells))

    info = {
        "stratum": s,
        "cells_v3": {f"{b}/{ln}": v3_cells[(b, ln)] for b, ln in sorted(v3_cells)},
        "cells_classic": {f"{b}/{ln}": cl_cells[(b, ln)] for b, ln in sorted(cl_cells)},
        "match_cells": per_cell,
        "v3_only_cells": [f"{b}/{ln}" for b, ln in v3_only],
        "classic_only_cells": [f"{b}/{ln}" for b, ln in cl_only],
        "leg_v3": dict(Counter(d["leg"] for d in v3_res)),
        "leg_classic": dict(Counter(d["leg"] for d in cl_res)),
    }
    if not match:
        info.update({
            "match": False,
            "note": "honest NO MATCH: no (band, len) cell holds clean rows of "
                    "both banks — no Delta is fabricated (G4)",
        })
        log_fn("char2_stratum_no_match", **{k: info[k] for k in
                                            ("stratum", "cells_v3", "cells_classic")})
        return info

    k1 = sum(c["k"] for c in (v3_cells[c] for c in match))
    n1 = sum(c["n"] for c in (v3_cells[c] for c in match))
    k0 = sum(c["k"] for c in (cl_cells[c] for c in match))
    n0 = sum(c["n"] for c in (cl_cells[c] for c in match))
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
    log_fn("char2_stratum_verdict", stratum=s, delta=st["delta"],
           delta_ci95=st["delta_ci95_newcombe"], k_new=k1, n_new=n1,
           k_old=k0, n_old=n0, gate=info["gate"]["verdict"])
    return info


# --- §6.3 probes on new-template fixture cells ---------------------------------

def build_probe_fixtures() -> dict:
    """The declared preflight fixture cells (identical builders + seeds)."""
    def nc1(name, ci, fl, long_=False):
        r = random.Random(f"pf:NC1V3:{name}")
        if long_:
            combo = tuple(i % len(banks.INSERT_PHRASES_V3)
                          for i in range(ci * 3, ci * 3 + 30))
            ins = " ".join(banks.INSERT_PHRASES_V3[i] for i in combo)
            body = banks._nc1_body_long(ci * 3)
            tag = f"v3xL{ci}"
        else:
            combo = banks.COMBOS_V3[ci]
            ins = " ".join(banks.INSERT_PHRASES_V3[i] for i in combo)
            body = banks._nc1_body(ci, fl)
            tag = f"v3x{ci}"
        subj = lib.TECH[3]
        sc = banks.build_nc1_v3(f"pf-{name}", subj, ins, body)
        sc.scaffold_id = f"NC-1:{subj}:sib:{tag}"
        return banks._finish_rotate(sc, SEED_HOLD_V3, False, r)

    def nc3(name, fam, channel):
        r = random.Random(f"pf:NC3V3:{name}")
        subj = lib.TECH[5]
        vp = banks.VERSION_PAIRS_V3[fam]
        sc = banks.build_nc3_v3(f"pf-{name}", subj, vp,
                                banks._nc3_body(fam, 5, channel))
        tag = vp[1] if channel != "long" else vp[1] + "L"
        sc.scaffold_id = f"NC-3:{subj}:{vp[0]}:{tag}"
        return banks._finish_rotate(sc, SEED_HOLD_V3, False, r)

    def nc5(name, sa, sb, kind):
        with banks.v3_bank_context():
            r = random.Random(f"pf:NC5V3:{name}")
            sibs = sorted(set(list(banks.SUBJECTS_V3) + list(lib.SUBJECTS)) - {sa})
            ca = sibs.index(sb) % 6
            sc = banks.build_nc5_v3(f"pf-{name}", sa, sb, kind, ca, ca)
            return banks._finish_rotate(sc, SEED_HOLD_V3, False, r, force_nn=True)

    out = {}
    for stratum, plan in PROBE_PLAN.items():
        if plan["kind"] == "nc5":
            subj = next(iter(banks.SUBJECTS_V3))  # declaration order (preflight)
            name = f"nc5-{subj}-{plan['args'][0]}"
            sc = nc5(name, subj, lib.TECH[0], plan["args"][0])
        elif plan["kind"] == "nc1":
            name = plan["name"]
            sc = nc1(name, *plan["args"])
        else:
            name = plan["name"]
            sc = nc3(name, *plan["args"])
        out[stratum] = {"fixture": name, "sc": sc}
    return out


def run_probes(booster, log_fn) -> dict:
    import train_conflict_detector as tcd

    nano = tcd.NanoLeg()  # runner instrument (float64-hardened, pinned)
    rng = random.Random(LADDER_SEED)
    results = {}
    all_ok = True
    for stratum, fx in build_probe_fixtures().items():
        sc = fx["sc"]
        name = fx["fixture"]
        row = {
            "pair_id": sc.scenario_id,
            "stratum": sc.stratum,
            "label": sc.label,
            "orientation": "a>b",
            "measured": {"cos_target": round(
                nano.pair_cos(sc.side_a, sc.side_b, site=f"char2:{name}"), 4)},
            "envelope_a": cb.side_dict(sc.side_a),
            "envelope_b": cb.side_dict(sc.side_b),
        }
        rec_a, rec_b = tcd.row_records(row)
        cos = float(row["measured"]["cos_target"])
        raw = tcd.decide_pipeline(rec_a, rec_b, cos, booster)
        # orientation swap invariance (sanity-suite run_pair practice)
        sw = tcd.orientation_swap(row)
        ra2, rb2 = tcd.row_records(sw)
        raw_swap = tcd.decide_pipeline(ra2, rb2, cos, booster)
        # cosmetic surface-invariance machine proof (runner assert)
        tcd.assert_cosmetic_surface_invariance(row, rng)
        # stability ladder raw/cosmetic/para1 on side_b (g_stab practice)
        lang = (row["envelope_b"].get("language") or "ru")
        para_lang = "en" if str(lang).lower().startswith("en") else "ru"
        ladder = {"raw": raw}
        applied_body = row["envelope_b"]["body"]
        for step in ("cosmetic", "para1"):
            if step == "cosmetic":
                body = tcd.cosmetic_variant(row["envelope_b"]["body"], rng)
            else:
                body = lib.para(row["envelope_b"]["body"], 1, para_lang, rng)
            applied_body = body
            pert = tcd.row_with_body(row, "envelope_b", body)
            pa, pb = tcd.row_records(pert)
            c = nano.pair_cos(pa, pb, site=f"char2:{name}:{step}")
            ladder[step] = tcd.decide_pipeline(pa, pb, c, booster)
            ladder[step]["cos"] = round(c, 4)
        verdicts = [ladder[s]["verdict"] for s in ("raw", "cosmetic", "para1")]
        checks = {
            "raw_not_contradicts": raw["verdict"] == "not-contradicts",
            "ladder_invariant": len(set(verdicts)) == 1,
            "cosmetic_twin_not": ladder["cosmetic"]["verdict"] == "not-contradicts",
            "swap_invariant": raw_swap["verdict"] == raw["verdict"],
        }
        ok = all(checks.values())
        all_ok &= ok
        results[stratum] = {
            "fixture": name, "scaffold_id": sc.scaffold_id,
            "cos_target": cos, "para_lang": para_lang,
            "para1_changed_body": applied_body != row["envelope_b"]["body"],
            "ladder": {s: {"verdict": d["verdict"], "leg": d["leg"], "p": d["p"],
                           **({"cos": d["cos"]} if "cos" in d else {})}
                       for s, d in ladder.items()},
            "swap_verdict": raw_swap["verdict"],
            "checks": checks, "ok": ok,
        }
        log_fn("char2_probe_63", stratum=stratum, fixture=name,
               expectation="§6.3 probe 6 ladder invariant + probe 2 cosmetic "
                           "twin NOT contradicts; raw verdict NOT contradicts "
                           "(NC class, false-positive axis)",
               verdicts=verdicts, ladder=results[stratum]["ladder"],
               swap_verdict=raw_swap["verdict"], checks=checks, ok=ok)
    return {"probes": results, "all_ok": all_ok}


# --- selfcheck (instrument validation; NO events, NO embeds, NO model) --------

def selfcheck() -> int:
    anc = assert_ancestry()
    print("selfcheck: ancestry OK", anc["head"][:12],
          "banks", anc["banks_file_sha256"][:12])
    classic = cb.gen_pool(SEED_HOLD_V3, "ho", rotate=True)
    v3pool = banks.gen_banks_v3_pool(SEED_HOLD_V3, V3_PREFIX, rotate=True)
    for s in CHAR_STRATA:
        n_cl = sum(1 for x in classic if x.stratum == s)
        n_v3 = sum(1 for x in v3pool if x.stratum == s)
        census = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))["strata"][s]
        assert n_cl + n_v3 == census["pool"], (s, n_cl, n_v3, census["pool"])
        print(f"selfcheck: {s} pool classic={n_cl} v3={n_v3} == census pool "
              f"{census['pool']}")
    fx = build_probe_fixtures()
    for st, f in fx.items():
        assert lib.mechanical_ok(f["sc"]), (st, "fixture mechanical_ok RED")
        assert f["sc"].label == "not-contradicts", (st, "fixture label RED")
        print(f"selfcheck: probe fixture {st} {f['fixture']} mechanical_ok")
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
        SELFCHECK = True
        return selfcheck()

    one_shot_guard()
    anc = assert_ancestry()

    log("cost_declaration",
        expected_prolog_s=5,
        expected_cycle_s=300,
        expected_peak_ram_gb=2.0,
        lane="E2 pre-cut characterization (ADDENDUM-6 §5 G4): frozen v2 "
             "contour over census-clean pool rows of the NEW banks NC-1/NC-3/NC-5 "
             "+ §6.3 probes on new-template fixture cells; pool-only (F7), "
             "sealed holdouts NOT opened, weights unchanged")

    # Pre-registration BEFORE start (the ratification pin, quoted verbatim).
    log("char2_preregister",
        band_pin="|Delta| <= 0.15 with a CI covering 0 "
                 "(ADDENDUM-6 §5 G4 ratification pin, NOT modified)",
        delta_definition="per-stratum P(contradicts | new bank) - "
                         "P(contradicts | old bank) pooled over MATCH-CELLS "
                         "stratum x band x len (cells with clean rows of both "
                         "banks); Wilson 95% per proportion + Newcombe "
                         "hybrid-score interval for Delta",
        scope=list(CHAR_STRATA),
        out_of_scope={"NC-4": "fallback-margin per G7, never cut from the "
                              "new bank — disclosed out-of-scope, not skipped "
                              "silently"},
        population="G10 census clean supply (mechanical_ok -> F10 -> dedup), "
                   "classic prefix 'ho-' vs v3 prefix 'ho3-'; reproduction "
                   "assert against banks_v3_census.json",
        contour="production: version-guard (one-sided NOT) -> model leg, "
                "cut 0.5, no threshold search; artifact "
                f"models/conflict-detector-v2 (manifest sha {V2_ARTIFACT_SHA_PREFIX}…)",
        probes="§6.3 ladder raw/cosmetic/para1 (probe 6 invariant) + "
               "cosmetic-twin (probe 2 NOT contradicts) on the declared "
               "preflight fixture cells nc1-single / nc3-fam0-fl0 / "
               "nc5-<first-subject>-cross; ladder RNG seed "
               f"{LADDER_SEED} (g_stab machine constant)",
        one_shot="P8-class: single run; re-rolls forbidden; all numbers from "
                 "this run only",
        **anc)

    log("char2_pre_cut_start",
        ratification="ADDENDUM-6 §5 G4 (MLC synthesis 10.11)",
        legality="pool-only F7: pool rows + seal-manifest sha sets only; "
                 "sealed holdout.jsonl v1/v2 NOT opened; prints not read; "
                 "weights unchanged; nothing cut, nothing sealed")

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
    log("char2_model_loaded", artifact="models/conflict-detector-v2",
        manifest_sha256=manifest["sha256"], trained_at=manifest.get("trained_at"),
        feature_contract="CORE-13 frozen", cut=0.5)

    supply = build_clean_supply(log)

    per_stratum = {}
    reds = []
    no_match = []
    for s in CHAR_STRATA:
        info = characterize_stratum(s, supply, booster, log)
        per_stratum[s] = info
        if not info.get("match"):
            no_match.append(s)
        elif info["gate"]["verdict"] == "RED":
            reds.append(s)

    probes = run_probes(booster, log)

    band_verdict = "RED" if reds else "GREEN"
    if no_match:
        band_verdict = f"GREEN-ON-MATCH (no match: {', '.join(no_match)})"
    probes_verdict = "GREEN" if probes["all_ok"] else "RED"
    verdict = "GREEN" if (not reds and probes["all_ok"]) else "RED"

    result = {
        "ts": now(),
        "slice": SLICE,
        "head": anc["head"],
        "ratification": "ADDENDUM-6 §5 G4",
        "artifact": {"dir": "models/conflict-detector-v2",
                     "sha256": manifest["sha256"]},
        "band_pin": "|Delta| <= 0.15 AND delta CI95 covers 0",
        "scope": list(CHAR_STRATA),
        "per_stratum": {
            s: {k: v for k, v in info.items() if k != "stratum"}
            for s, info in per_stratum.items()
        },
        "no_match_strata": no_match,
        "band_verdict": band_verdict,
        "probes_verdict": probes_verdict,
        "probes": probes["probes"],
        "verdict": verdict,
        "anchor": "char GREEN + v3 gate RED = model defect; "
                  "char RED + v3 gate RED = bank drift (G4)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    log("char2_pre_cut_done",
        verdict=verdict,
        band_verdict=band_verdict,
        probes_verdict=probes_verdict,
        delta={s: (per_stratum[s]["stats"]["delta"] if per_stratum[s].get("match")
                   else "no-match") for s in CHAR_STRATA},
        delta_ci95={s: (per_stratum[s]["stats"]["delta_ci95_newcombe"]
                        if per_stratum[s].get("match") else None)
                    for s in CHAR_STRATA},
        report="docs/experiments/conflict-detector-char2-precut-2026-11-10.md",
        artifact_json=str(OUT_JSON.relative_to(REPO)),
        elapsed_s=result["elapsed_s"],
        next=("GREEN -> DE ценз+резка+seal v3 по G2-цепочке"
              if verdict == "GREEN"
              else "RED -> STOP: эскалация MLC (G4), шаблоны НЕ правятся "
                   "(заморожены G1); единственная резка НЕ исполняется"))

    print(f"\nCHAR2 PRE-CUT CHARACTERIZATION: {verdict}")
    for s in CHAR_STRATA:
        info = per_stratum[s]
        if info.get("match"):
            st = info["stats"]
            print(f"  {s}: Delta={st['delta']:+.4f} "
                  f"CI95 [{st['delta_ci95_newcombe'][0]:+.4f}, "
                  f"{st['delta_ci95_newcombe'][1]:+.4f}] "
                  f"(new {st['k_new']}/{st['n_new']}, old {st['k_old']}/{st['n_old']}) "
                  f"gate={info['gate']['verdict']}")
        else:
            print(f"  {s}: NO MATCH (honest)")
    print(f"  probes §6.3: {probes_verdict}")
    return 0 if verdict == "GREEN" else 2


if __name__ == "__main__":
    sys.exit(main())
