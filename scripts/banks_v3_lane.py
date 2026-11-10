#!/usr/bin/env python3
"""B6-conflict banks lane driver — ADDENDUM-6 §5 (G1 freeze / G6-G10).

Subcommands (ratified lane order):
  preflight  declarations-vs-text on fixture cells of every NEW bank class
             (П3 style) + measured NC-5 cross envelope + pool-wide
             mechanical attrition + anti-collapse bank-level caps (§4.5)
  freeze     pin bank shas into pin-v3.json prep section (G1: AFTER the
             banks commit, BEFORE the census; bank edits are LOCKED after)
  census     E9-mini over clean-after-preflight-attrition supply (G10):
             F10 sha-disjointness vs the three sealed generations +
             within-cut side-text uniqueness + per-stratum clean counts
             vs the ratified 2x quotas; style-neutrality numbers (G5)
             in the same pass (shared embeddings)

Run-log slice: conflict-detector-corpus-v2-seal3-banks (own events, no
re-pin of seeds — the seed event is a re-affirmation only).
Instrument: /usr/bin/python3 + APPENDED vesma engine path; pinned nano
verified at Cos construction (wrong provider refuses loudly, A-frozen).
The train path is not touched: this lane writes NO corpus artifacts, it
only reads pin/manifests/pool origins (sha level) and appends run-log
events + prep reports under artifacts/runs/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402
import conflict_corpus_banks_v3 as banks  # noqa: E402
import build_conflict_detector_corpus as drv  # noqa: E402

DATA = REPO / "data" / "conflict-detector"
PIN_V3_PATH = DATA / "pin-v3.json"
ART = REPO / "artifacts" / "runs"
RUNLOG = REPO / "artifacts" / "runs" / "run_log.jsonl"
RUNLOG_MD = REPO / "docs" / "experiments" / "conflict-detector-train-runlog.md"
SLICE = "conflict-detector-corpus-v2-seal3-banks"

SEED_HOLD_V3 = drv.SEED_HOLD_V3  # 6117
V3_PREFIX = "ho3"

#: G10 ratified census quotas (seal share -> 2x; NC-4 strict per the
#: ratified table — the 2x depth is WAIVED and disclosed, not silently
#: relabeled; NC-4 clean must clear 11 for the strict cens).
CENSUS_QUOTAS = {"NC-1": 40, "NC-2": 44, "NC-3": 32, "NC-4": 11, "NC-5": 50,
                 "N1": 22, "N2": 14, "N3": 14}
N1_HOST_NEEDS = {h: 2 * q for h, q in drv.SEAL_N1_HOSTS.items()}
N1_K_NEEDS = {1: 2 * drv.SEAL_N1_K[0], 2: 2 * drv.SEAL_N1_K[1]}
N2_K_NEEDS = {1: 2 * drv.SEAL_N2_K[0], 2: 2 * drv.SEAL_N2_K[1]}
#: NC-5 envelope gates (G9: "RU/EN одного субъекта заведомо не переводы;
#: перевод-пара только BASE — конверт cross cos ~0.486 не покидать").
#: CALIBRATION (disclosed in the prep report): the "~0.486" record is the
#: corpus measured FLOOR (lower edge), not the band center. The frozen
#: classical bank itself measures cross cells in [0.35..0.80] (p50 0.66)
#: and same-subject phrase pairs in [0.47..0.87] (p50 0.74). The gates are
#: therefore anchored to the FROZEN CORPUS, computed live:
#:   CELL gate  : new cross cells in [0.486 (recorded floor), old_bank_max]
#:   PHRASE gate: new same-subject phrase pairs within the old bank's
#:                same-subject phrase-pair envelope (-0.03 slack at the low
#:                edge) — the new bank may never be MORE translation-like
#:                than the corpus maximum.
NC5_CELL_FLOOR = 0.486


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields) -> None:
    row = {"ts": now(), "event": event, "slice": SLICE, **fields}
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def die(msg: str) -> None:
    print(f"BANKS-LANE STOP: {msg}", file=sys.stderr)
    log("build_stop", reason=msg,
        note="pre-cut BuildStop does not spend the one-shot F1 budget (G2)")
    sys.exit(1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)

    def p(q: float) -> float:
        return round(s[min(len(s) - 1, int(q * (len(s) - 1)))], 4)

    return {"n": len(s), "min": p(0.0), "p10": p(0.1), "p50": p(0.5),
            "p90": p(0.9), "max": p(1.0),
            "mean": round(sum(s) / len(s), 4)}


# ===========================================================================
# preflight
# ===========================================================================

def _nc1_fixture(name: str, combo_idx: int, fl: int, long_: bool = False):
    subj = lib.TECH[3]
    r = __import__("random").Random(f"pf:NC1V3:{name}")
    if long_:
        combo = tuple(i % len(banks.INSERT_PHRASES_V3)
                      for i in range(combo_idx * 3, combo_idx * 3 + 30))
        ins = " ".join(banks.INSERT_PHRASES_V3[i] for i in combo)
        body = banks._nc1_body_long(combo_idx * 3)
        sid_tag = f"v3xL{combo_idx}"
    else:
        combo = banks.COMBOS_V3[combo_idx]
        ins = " ".join(banks.INSERT_PHRASES_V3[i] for i in combo)
        body = banks._nc1_body(combo_idx, fl)
        sid_tag = f"v3x{combo_idx}"
    sc = banks.build_nc1_v3(f"pf-{name}", subj, ins, body)
    sc.scaffold_id = f"NC-1:{subj}:sib:{sid_tag}"
    return banks._finish_rotate(sc, 6117, False, r)


def _nc3_fixture(name: str, fam: int, channel: str):
    subj = lib.TECH[5]
    r = __import__("random").Random(f"pf:NC3V3:{name}")
    vp = banks.VERSION_PAIRS_V3[fam]
    sc = banks.build_nc3_v3(f"pf-{name}", subj, vp,
                            banks._nc3_body(fam, 5, channel))
    tag = vp[1] if channel != "long" else vp[1] + "L"
    sc.scaffold_id = f"NC-3:{subj}:{vp[0]}:{tag}"
    return banks._finish_rotate(sc, 6117, False, r)


def _nc4_fixture(name: str, subj: str, delta: str, channel: str):
    r = __import__("random").Random(f"pf:NC4V3:{name}")
    sc = banks.build_nc4_v3(f"pf-{name}", subj, delta,
                            banks._nc4_body(banks.NC4_DELTAS_V3.index(delta),
                                            channel))
    if channel == "long":
        sc.scaffold_id = f"NC-4:{subj}:{delta}:v3:L"
    return banks._finish_rotate(sc, 6117, False, r,
                                force_nn=(delta in ("type", "lang")))


def _nc5_fixture(name: str, sa: str, sb: str, kind: str):
    with banks.v3_bank_context():
        r = __import__("random").Random(f"pf:NC5V3:{name}")
        sibs = sorted(set(list(banks.SUBJECTS_V3) + list(lib.SUBJECTS)) - {sa})
        ca = sibs.index(sb) % 6
        sc = banks.build_nc5_v3(f"pf-{name}", sa, sb, kind, ca, ca)
        # fixture-scoped both-null exemption (N3/corner-twin precedent):
        # declarations verify the CONSTRUCTION, incl. visible languages;
        # the null envelope keeps its corpus-wide 1/3 draw on pool cells
        return banks._finish_rotate(sc, 6117, False, r, force_nn=True)


def cmd_preflight() -> int:
    t0 = time.time()

    # --- seed re-affirmation FIRST (no re-pin; the pin exists in the log) --
    _scan, prior_v3 = drv.pinned_seeds_from_runlog()
    if prior_v3 and any(p != SEED_HOLD_V3 for p in prior_v3):
        die(f"seed-family shopping RED: prior seeds_pinned_v3 hold={prior_v3}")
    log("seeds_reaffirmed_v3", seeds={"hold": SEED_HOLD_V3},
        reaffirmed=True, prior_pins=len(prior_v3),
        note="re-affirmation only — seeds_pinned_v3 already in the log, "
             "no new pin event by lane rule")

    # --- banks authored (authoring inventory; freeze comes later) ----------
    log("banks_authored",
        nc1={"phrases": len(banks.INSERT_PHRASES_V3),
             "combos": len(banks.COMBOS_V3),
             "cells": 13 * 12 * 2 + 2 * 12},
        nc5={"subjects": len(banks.SUBJECTS_V3),
             "phrases": 4 * len(banks.SUBJECTS_V3)},
        nc3={"families": len(banks.VERSION_PAIRS_V3)},
        nc4={"role": "fallback-margin", "subjects": len(banks.NC4_SUBJECTS_V3),
             "cells": 36},
        static_guards="PASS at import (fail-closed)")

    report: dict = {"ts": now(), "slice": SLICE, "ratification":
                    "ADDENDUM-6 §5 G6-G9", "static_guards": "PASS",
                    "declarations": [], "bank_checks": {}, "envelope": {},
                    "attrition": {}, "anti_collapse": {}}
    rows = []

    def decl(row: dict, ok: bool, why: str) -> bool:
        row["ok"] = ok
        row["fail"] = "" if ok else why
        report["declarations"].append(row)
        return ok

    ok = True

    # --- NC-1 fixtures: one per combo arity + long -------------------------
    for name, ci, fl, long_ in (("nc1-single", 0, 0, False),
                                ("nc1-pair", 3, 1, False),
                                ("nc1-triple", 6, 1, False),
                                ("nc1-quad", 9, 0, False),
                                ("nc1-quint", 11, 1, False),
                                ("nc1-long", 0, 1, True)):
        sc = _nc1_fixture(name, ci, fl, long_)
        regs = lib.regions(sc.side_a.body, sc.side_b.body)
        checks = {
            "mechanical_ok": lib.mechanical_ok(sc),
            "all_regions_insert": bool(regs) and all(t == "insert" for t, *_ in regs),
            "slot_bijection_ok": lib.slot_bijection_ok(sc),
            "region_is_version_false_truthful":
                all(not lib.region_is_version(t) for _, t, _, _ in regs),
            "len_channel_ok": lib.len_channel_ok(sc),
            "version_guard_false": lib.version_guard(sc) is False,
            "benign_pair_absent": lib.benign_pair_ok(sc) is False,
            "n3_ok_absent": lib.n3_ok(sc) is False,
            "label": sc.label == "not-contradicts",
        }
        ok &= decl({"class": "NC-1", "fixture": name, "checks": checks},
                   all(checks.values()),
                   ",".join(k for k, v in checks.items() if not v))
        rows.append((name, checks))

    # --- NC-3 fixtures ------------------------------------------------------
    for name, fam, channel in (("nc3-fam0-fl0", 0, "fl0"),
                               ("nc3-fam0-long", 0, "long"),
                               ("nc3-fam1-fl1", 1, "fl1"),
                               ("nc3-fam1-long", 1, "long")):
        sc = _nc3_fixture(name, fam, channel)
        regs = lib.regions(sc.side_a.body, sc.side_b.body)
        slot = sc.slots[0]
        checks = {
            "mechanical_ok": lib.mechanical_ok(sc),
            "regions_all_version": bool(regs) and
                all(lib.region_is_version(t) for _, t, _, _ in regs),
            "slot_is_version_truthful": bool(slot["is_version_token"]) and
                lib.region_is_version(str(slot["value_b"])),
            "slot_bijection_ok": lib.slot_bijection_ok(sc),
            "len_channel_ok": lib.len_channel_ok(sc),
            "version_guard_true": lib.version_guard(sc) is True,
            "label": sc.label == "not-contradicts",
        }
        ok &= decl({"class": "NC-3", "fixture": name, "checks": checks},
                   all(checks.values()),
                   ",".join(k for k, v in checks.items() if not v))

    # --- NC-4 fallback fixtures ---------------------------------------------
    for name, subj, delta, channel in (("nc4-tags-fl1", "cook", "tags", "fl1"),
                                       ("nc4-lang-long", "fit", "lang", "long"),
                                       ("nc4-type-fl1", "weather", "type", "fl1")):
        sc = _nc4_fixture(name, subj, delta, channel)
        a, b = sc.side_a, sc.side_b
        delta_present = (a.tags != b.tags) or (a.record_type != b.record_type) \
            or (a.language != b.language)
        checks = {
            "mechanical_ok": lib.mechanical_ok(sc),
            "identity_body": lib.n1(a.body) == lib.n1(b.body),
            "envelope_delta_present": delta_present,
            "len_channel_ok": lib.len_channel_ok(sc),
            "version_guard_false": lib.version_guard(sc) is False,
            "fallback_role": subj in banks.NC4_SUBJECTS_V3,
        }
        ok &= decl({"class": "NC-4", "fixture": name, "checks": checks},
                   all(checks.values()),
                   ",".join(k for k, v in checks.items() if not v))

    # --- NC-5 fixtures: every new subject, each leg shape --------------------
    tech0 = lib.TECH[0]
    for subj in banks.SUBJECTS_V3:
        for kind, sa, sb in (("same", subj, tech0), ("cross", subj, tech0),
                             ("long", subj, tech0), ("cross-rev", tech0, subj)):
            sc = _nc5_fixture(f"nc5-{subj}-{kind}", sa, sb, kind)
            a, b = sc.side_a, sc.side_b
            checks = {
                "mechanical_ok": lib.mechanical_ok(sc),
                "subjects_differ": sc.subject_a != sc.subject_b,
                "slots_empty_truthful": sc.slots == [],
                "len_channel_ok": lib.len_channel_ok(sc),
                "benign_pair_absent": lib.benign_pair_ok(sc) is False,
                "n3_ok_absent": lib.n3_ok(sc) is False,
            }
            if kind == "cross" or kind == "cross-rev":
                en_first = banks.SPEC_V3[sb][2] if sb in banks.SPEC_V3 \
                    else lib.SPEC[sb][2]
                checks["languages_differ"] = (a.language != b.language)
                checks["b_short_envelope"] = len(b.body) <= 200
                checks["en_body_is_first_en_sentence"] = b.body == en_first
            if kind == "same":
                checks["same_language"] = (a.language == b.language)
                checks["cross_domain"] = (sa in lib.TECH) != (sb in lib.TECH)
            if kind == "long":
                blk = lib.filler_block("ru", 35)
                checks["shared_boilerplate_tail"] = \
                    a.body.endswith(blk) and b.body.endswith(blk)
            ok &= decl({"class": "NC-5", "fixture": f"{subj}:{kind}",
                        "checks": checks},
                       all(checks.values()),
                       ",".join(k for k, v in checks.items() if not v))

    # --- control legs: old-bank carriers (the battery discriminates) --------
    cos = drv.Cos()
    n1c = drv.gen_n1_pool(6117, "ctl")
    n3c = drv.gen_n3_pool(6117, "ctl")
    ctl_n1 = None
    for x in n1c:
        drv.measure_all([x], cos)
        if lib.mechanical_ok(x) and lib.benign_pair_ok(x) and x._cos >= 0.95:
            ctl_n1 = x
            break
    ctl_n3 = next(x for x in n3c if lib.n3_ok(x))
    for name, cell, want_benign, want_n3 in (
            ("ctl-N1-classic", ctl_n1, True, False),
            ("ctl-N3-classic", ctl_n3, False, True)):
        checks = {
            "mechanical_ok": lib.mechanical_ok(cell),
            "benign_pair": lib.benign_pair_ok(cell) == want_benign,
            "n3_ok": lib.n3_ok(cell) == want_n3,
        }
        ok &= decl({"class": "CONTROL", "fixture": name, "checks": checks},
                   all(checks.values()),
                   ",".join(k for k, v in checks.items() if not v))

    # --- measured NC-5 envelope (G9), anchored to the frozen classical bank -
    base_pair = cos.pair(lib.n1(lib.BASE_RU[0]), lib.n1(lib.BASE_EN[0]))
    subs_old = list(lib.SPEC)

    def _ru_pair(s: str) -> str:
        src = banks.SPEC_V3 if s in banks.SPEC_V3 else lib.SPEC
        return f"{src[s][0]} {src[s][1]}"

    def _en0(s: str) -> str:
        return banks.SPEC_V3[s][2] if s in banks.SPEC_V3 else lib.SPEC[s][2]

    old_cell_vals = [cos.pair(lib.n1(_ru_pair(a)), lib.n1(_en0(b)))
                     for a in subs_old for b in subs_old if a != b]
    old_phrase_vals = [cos.pair(lib.n1(a), lib.n1(b))
                       for s in subs_old
                       for a, b in ((lib.SPEC[s][0], lib.SPEC[s][2]),
                                    (lib.SPEC[s][0], lib.SPEC[s][3]),
                                    (lib.SPEC[s][1], lib.SPEC[s][2]),
                                    (lib.SPEC[s][1], lib.SPEC[s][3]))]
    cell_hi = round(max(old_cell_vals), 4)
    phrase_lo = round(min(old_phrase_vals) - 0.03, 4)
    phrase_hi = round(max(old_phrase_vals), 4)
    report["envelope"]["old_bank_reference"] = {
        "cross_cell": _json_stat(old_cell_vals),
        "phrase_pair": _json_stat(old_phrase_vals)}
    report["envelope"]["declared_bounds"] = {
        "cell": [NC5_CELL_FLOOR, cell_hi],
        "phrase": [phrase_lo, phrase_hi],
        "base_translation_pair": round(base_pair, 4)}

    new_cell_vals = {}
    new_subs = list(banks.SPEC_V3)
    for sa in new_subs:
        for sb in new_subs + subs_old:
            if sa == sb:
                continue
            c = round(cos.pair(lib.n1(_ru_pair(sa)), lib.n1(_en0(sb))), 4)
            new_cell_vals[f"{sa}->{sb}"] = c
            if not (NC5_CELL_FLOOR <= c <= cell_hi):
                ok = False
                report["envelope"].setdefault("violations", []).append(
                    {"kind": "cell", "pair": f"{sa}->{sb}", "cos": c})
    new_phrase_rows = {}
    for subj, (ru0, ru1, en0, en1) in banks.SPEC_V3.items():
        pairs = {}
        for i, ru in ((0, ru0), (1, ru1)):
            for j, en in ((0, en0), (1, en1)):
                c = round(cos.pair(lib.n1(ru), lib.n1(en)), 4)
                pairs[f"ru{i}-en{j}"] = c
                if not (phrase_lo <= c <= phrase_hi):
                    ok = False
                    report["envelope"].setdefault("violations", []).append(
                        {"kind": "phrase", "subject": subj,
                         "pair": f"ru{i}-en{j}", "cos": c})
        new_phrase_rows[subj] = pairs
    report["envelope"]["new_cross_cells"] = {
        "n": len(new_cell_vals),
        "stats": _json_stat(list(new_cell_vals.values())),
        "out_of_band": {k: v for k, v in new_cell_vals.items()
                        if not (NC5_CELL_FLOOR <= v <= cell_hi)}}
    report["envelope"]["same_subject_phrase_pairs"] = new_phrase_rows

    # --- pool-wide attrition + anti-collapse (§4.5 bank-level) --------------
    pool = banks.gen_banks_v3_pool(6117, V3_PREFIX, rotate=True)
    by_stratum: dict[str, list] = {}
    for x in pool:
        by_stratum.setdefault(x.stratum, []).append(x)
    for s, cells in sorted(by_stratum.items()):
        mech_fail = sum(1 for x in cells if not lib.mechanical_ok(x))
        len_fail = sum(1 for x in cells
                       if lib.mechanical_ok(x) and not lib.len_channel_ok(x))
        chan = Counter(x.length_channel for x in cells)
        n = len(cells)
        shares = {c: round(v / n, 3) for c, v in sorted(chan.items())}
        max_share = max(Counter(x.scaffold_id for x in cells).values()) / n
        scaf = len({x.scaffold_id for x in cells})
        anti_ok = scaf >= 20 and max_share <= lib.CAP + 1e-9
        report["attrition"][s] = {
            "pool": n, "mech_fail": mech_fail, "len_fail": len_fail,
            "post_preflight": n - mech_fail - len_fail,
            "channels": shares, "scaffolds": scaf,
            "max_scaffold_share": round(max_share, 4), "anti_collapse_ok": anti_ok}
        report["anti_collapse"][s] = {"scaffolds": scaf, "max_share": round(max_share, 4),
                                      "ok": anti_ok}
        ok &= anti_ok

    report["preflight_ok"] = bool(ok)
    report["elapsed_s"] = round(time.time() - t0, 1)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "banks_v3_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("preflight_v3_banks", verdict="pass" if ok else "RED",
        fixtures=len(report["declarations"]),
        envelope_new_cells=report["envelope"]["new_cross_cells"]["n"],
        pool=len(pool), elapsed_s=report["elapsed_s"])
    print(f"preflight: {'PASS' if ok else 'RED'} "
          f"({len(report['declarations'])} fixture declarations, "
          f"pool {len(pool)}, {report['elapsed_s']}s)")
    return 0 if ok else 1


# ===========================================================================
# freeze (G1)
# ===========================================================================

def cmd_freeze() -> int:
    banks_path = REPO / "scripts" / "conflict_corpus_banks_v3.py"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(banks_path)], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        die(f"freeze requires the banks file COMMITTED first "
            f"(dirty: {dirty})")
    file_sha = file_sha256(banks_path)
    content_sha = lib.sha(lib.json_dumps_sorted(banks.bank_content()))
    pin = {}
    if PIN_V3_PATH.exists():
        pin = json.loads(PIN_V3_PATH.read_text(encoding="utf-8"))
    if "prep" in pin and pin["prep"].get("banks_v3", {}).get("content_sha256") \
            not in (None, content_sha):
        die("pin-v3.json prep already pins a DIFFERENT bank content — "
            "re-freeze is a new incident, not an iteration")
    pin["prep"] = {
        "banks_v3": {
            "frozen_at": now(),
            "banks_commit": head,
            "module": "scripts/conflict_corpus_banks_v3.py",
            "file_sha256": file_sha,
            "content_sha256": content_sha,
            "nc1": {"phrases": len(banks.INSERT_PHRASES_V3),
                    "combos": len(banks.COMBOS_V3)},
            "nc5": {"subjects": len(banks.SUBJECTS_V3),
                    "phrases": 4 * len(banks.SUBJECTS_V3)},
            "nc3": {"families": len(banks.VERSION_PAIRS_V3)},
            "nc4": {"role": "fallback-margin",
                    "subjects": len(banks.NC4_SUBJECTS_V3)},
            "ratification": "ADDENDUM-6 §5 G1/G6/G7/G8/G9",
            "lock": "bank edits FORBIDDEN after this pin (G1); a census or "
                    "gate failure is a BuildStop + NEW addendum",
        },
        "seeds": {"hold": SEED_HOLD_V3, "status": "pinned (seeds_pinned_v3)",
                  "reaffirmed_by_lane": True},
    }
    PIN_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIN_V3_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    freeze_report = {"ts": now(), "banks_commit": head,
                     "file_sha256": file_sha, "content_sha256": content_sha,
                     "pin_v3": str(PIN_V3_PATH)}
    (ART / "banks_v3_freeze.json").write_text(
        json.dumps(freeze_report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("banks_freeze_v3", banks_commit=head[:12], file_sha256=file_sha,
        content_sha256=content_sha, pin="data/conflict-detector/pin-v3.json#prep",
        note="G1: banks LOCKED; census may observe from here")
    print(f"freeze OK: commit {head[:12]} content {content_sha[:16]}")
    return 0


def _assert_frozen() -> dict:
    if not PIN_V3_PATH.exists():
        die("freeze first: pin-v3.json prep section missing (G1 order)")
    prep = json.loads(PIN_V3_PATH.read_text(encoding="utf-8")).get("prep", {})
    bv = prep.get("banks_v3")
    if not bv:
        die("freeze first: pin-v3.json has no banks_v3 prep section")
    cur = file_sha256(REPO / "scripts" / "conflict_corpus_banks_v3.py")
    if cur != bv["file_sha256"]:
        die(f"G1 LOCK VIOLATION: banks module sha {cur[:16]} != frozen "
            f"{bv['file_sha256'][:16]} — bank edited after freeze")
    return bv


# ===========================================================================
# census (G10) + style neutrality (G5) — one embedding pass
# ===========================================================================

def cmd_census() -> int:
    t0 = time.time()
    frozen = _assert_frozen()
    cos = drv.Cos()

    print("== pools: classic hold generation + v3 banks (seed 6117) ==")
    classic = drv.gen_pool(SEED_HOLD_V3, "ho", rotate=True)
    n1 = drv.gen_n1_pool(SEED_HOLD_V3, "ho")
    n2 = drv.gen_n2_pool(SEED_HOLD_V3, "ho")
    n3 = drv.gen_n3_pool(SEED_HOLD_V3, "ho")
    v3pool = banks.gen_banks_v3_pool(SEED_HOLD_V3, V3_PREFIX, rotate=True)
    print(f"classic {len(classic)}, carriers n1={len(n1)} n2={len(n2)} "
          f"n3={len(n3)}, v3 banks {len(v3pool)}")
    drv.measure_all(classic + v3pool + n1 + n2 + n3, cos)
    print(f"embedded {cos.n} unique texts in {time.time()-t0:.0f}s")

    # --- F10 sha sets (three sealed generations; sealed content unopened) ---
    print("== F10 disjointness sources ==")
    tr_texts, _ = drv.disk_text_shas(drv.CORPUS_DIR_V2 / "train.jsonl")
    v2_shas = drv.sealed_generation_sha_set(drv.SEAL_V2_PATH,
                                            drv.CORPUS_DIR_V2 / "holdout-pool.jsonl")
    v1_shas = drv.sealed_generation_sha_set(drv.SEAL_V1_PATH,
                                            drv.CORPUS_DIR / "holdout-pool.jsonl")
    sources = {"train-v2": tr_texts, "v2-seal": v2_shas, "v1-seal": v1_shas}
    print(f"sha sets: train-v2 {len(tr_texts)}, v2-seal {len(v2_shas)}, "
          f"v1-seal {len(v1_shas)}")

    def clean_count(cells: list, mech_fn) -> dict:
        """G10 census: preflight attrition -> F10 cull -> within-cut
        side-text uniqueness; attribution classic vs v3 by id prefix."""
        live = [x for x in cells if mech_fn(x)]
        attrited = len(cells) - len(live)
        live.sort(key=lambda x: (-x._cos, x.scenario_id))
        seen: set[str] = set()
        culled: Counter = Counter()
        dups = 0
        clean_rows = []
        for x in live:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            src = drv._forbidden_source(ha, hb, sources)
            if src:
                culled[src] += 1
                continue
            if ha in seen or hb in seen:
                dups += 1
                continue
            seen.update((ha, hb))
            clean_rows.append(x)
        origin = Counter("v3" if x.scenario_id.startswith(f"{V3_PREFIX}-")
                         else "classic" for x in clean_rows)
        return {"pool": len(cells), "preflight_attrited": attrited,
                "culled_f10": dict(culled), "dup_blocked": dups,
                "clean": len(clean_rows),
                "clean_v3": origin.get("v3", 0),
                "clean_classic": origin.get("classic", 0)}

    census: dict = {"ts": now(), "frozen_banks": frozen["content_sha256"],
                    "quotas": CENSUS_QUOTAS, "strata": {}, "carriers": {}}
    red: list[str] = []

    print("== classic strata census (G10 clean-after-preflight) ==")
    by_s: dict[str, list] = {}
    for x in classic + v3pool:
        by_s.setdefault(x.stratum, []).append(x)
    for s in drv.ORDER:
        row = clean_count(by_s.get(s, []), lib.mechanical_ok)
        census["strata"][s] = row
        want = CENSUS_QUOTAS.get(s)
        if want is not None:
            mark = "OK" if row["clean"] >= want else "RED"
            if row["clean"] < want:
                red.append(f"{s} clean {row['clean']} < {want}")
            print(f"  {s}: clean {row['clean']} (v3 {row['clean_v3']} / "
                  f"classic {row['clean_classic']}) vs {want} [{mark}]")
        else:
            print(f"  {s}: clean {row['clean']} (informational)")

    print("== carrier census (E9-mini legs, mirrors census_v3) ==")
    cen_n1 = clean_count(n1, lambda x: drv.carrier_suitable(x))
    cen_n2 = clean_count(n2, lambda x: drv.carrier_suitable(x))
    cen_n3 = clean_count(n3, lib.mechanical_ok)
    hosts = {h: sum(1 for x in n1 if drv.carrier_suitable(x) and x.stratum == h)
             for h in drv.SEAL_N1_HOSTS}
    kmix1 = Counter(len(x.benign) for x in n1 if drv.carrier_suitable(x))
    kmix2 = Counter(len(x.benign) for x in n2 if drv.carrier_suitable(x))
    census["carriers"] = {"n1": cen_n1, "n2": cen_n2, "n3": cen_n3,
                          "n1_hosts": hosts,
                          "n1_k": dict(kmix1), "n2_k": dict(kmix2)}
    checks = [("N1", cen_n1["clean"], CENSUS_QUOTAS["N1"]),
              ("N2", cen_n2["clean"], CENSUS_QUOTAS["N2"]),
              ("N3", cen_n3["clean"], CENSUS_QUOTAS["N3"])]
    for h, need in N1_HOST_NEEDS.items():
        checks.append((f"N1/{h}", hosts.get(h, 0), need))
    for k, need in N1_K_NEEDS.items():
        checks.append((f"N1/k{k}", kmix1.get(k, 0), need))
    for k, need in N2_K_NEEDS.items():
        checks.append((f"N2/k{k}", kmix2.get(k, 0), need))
    for name, got, need in checks:
        mark = "OK" if got >= need else "RED"
        if got < need:
            red.append(f"{name} {got} < {need}")
        print(f"  {name}: {got} vs {need} [{mark}]")

    # --- style neutrality (G5): per-stratum old/new bank distributions ------
    print("== style neutrality (measured-cos / char5_jaccard / channels) ==")
    style: dict = {"ts": now(), "per_stratum": {}}
    for s in ("NC-1", "NC-3", "NC-4", "NC-5"):
        row = {}
        for bank, cells in (("classic", [x for x in classic if x.stratum == s]),
                            ("v3", [x for x in v3pool if x.stratum == s])):
            if not cells:
                continue
            cs = [x._cos for x in cells]
            j5 = [lib.pair_char5_jaccard(x.side_a, x.side_b, x._cos) for x in cells]
            chan = Counter(x.length_channel for x in cells)
            n = len(cells)
            a_len_dev = {c: round(abs(v / n - 1 / 3), 3) for c, v in chan.items()
                         if v >= 30}
            row[bank] = {
                "n": n, "cos": _json_stat(cs), "char5_jaccard": _json_stat(j5),
                "channels": {c: round(v / n, 3) for c, v in sorted(chan.items())},
                "a_len_indep_max_dev_n_ge_30": a_len_dev,
                "bands": dict(Counter(x._band for x in cells))}
        if row:
            style["per_stratum"][s] = row
            print(f"  {s}: classic cos {_json_stat([x._cos for x in classic if x.stratum == s])['mean']}"
                  f" / v3 cos {_json_stat([x._cos for x in v3pool if x.stratum == s])['mean']}")

    verdict = "pass" if not red else "RED"
    census["style"] = style
    census["verdict"] = verdict
    census["red"] = red
    census["elapsed_s"] = round(time.time() - t0, 1)
    (ART / "banks_v3_census.json").write_text(
        json.dumps(census, ensure_ascii=False, indent=2), encoding="utf-8")
    if red:
        log("census_v3_banks", verdict="RED", red=red,
            clean={s: census["strata"][s]["clean"] for s in census["strata"]},
            note="G10 census on clean-after-preflight supply")
        die("census G10 RED: " + "; ".join(red))
    log("census_v3_banks", verdict="pass",
        clean={s: census["strata"][s]["clean"] for s in census["strata"]},
        carriers={"n1": cen_n1["clean"], "n2": cen_n2["clean"],
                  "n3": cen_n3["clean"]},
        elapsed_s=census["elapsed_s"])
    log("style_neutrality_v3",
        per_stratum={s: {b: {"cos_mean": d[b]["cos"]["mean"],
                             "jaccard_mean": d[b]["char5_jaccard"]["mean"],
                             "channels": d[b]["channels"]}
                         for b in d} for s, d in style["per_stratum"].items()},
        note="G5: distribution comparison old/new bank; full G4 "
             "characterization belongs to the ML lane")
    print(f"census G10 PASS in {census['elapsed_s']}s — report: "
          f"{ART/'banks_v3_census.json'}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight":
        return cmd_preflight()
    if cmd == "freeze":
        return cmd_freeze()
    if cmd == "census":
        return cmd_census()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
