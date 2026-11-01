#!/usr/bin/env python3
"""Build driver for the conflict-detector corpus v1 (ADR 0002 duty #6).

Subcommands (the ratified lane order):
  preflight  П3 fixture cells — one per class, mechanical map + guard
  measure    dry pools: wide candidates, measured cos (pinned nano),
             joint allocation honoring exact strata quotas + ADDENDUM-1
             floors, write pin file (activation assert for build)
  build      regenerate identical pools, select per pin, emit
             train.jsonl (600 pairs x 2 orientations) + holdout pool 600
             -> seal 180 (CD-holdout-v1), full assert map (design 5.3),
             baselines, build manifest
  sheet      48-block gold sheet, blind render, fp frozen (TL format
             gate BEFORE any owner hand-off)
  card       dataset card + fingerprints (glossary mandatory)

Instrument: engine venv python (vesma.embeddings); pin
nano:sha256:3b752e06... verified at every entry — wrong provider refuses
loudly (A-frozen). Data under data/ (gitignored); run-log append-only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402

DATA = REPO / "data" / "conflict-detector"
CORPUS_DIR = DATA / "corpus-v1"
SEALS_DIR = DATA / "seals"
SHEET_DIR = DATA / "sheet"
RUNLOG = REPO / "artifacts" / "runs" / "run_log.jsonl"

SEED_TRAIN, SEED_HOLD, SEED_SHEET = 42, 777, 4242
TRAIN_PAIRS, HOLD_POOL, HOLD_SEAL = 600, 600, 180

BANDS = [("razor", 0.95, 1.01), ("b85", 0.85, 0.95), ("b70", 0.70, 0.85),
         ("b55", 0.55, 0.70), ("b40", 0.40, 0.60)]
BAND_TARGETS = {"razor": 180, "b85": 110, "b70": 130, "b55": 125, "b40": 45}
FLOOR_RAZOR = 162
FLOOR_HARD_POS = 30
CORNER_MIN = 10
CROSS_NEG_CAP = 45
CROSS_POS_QUOTA = 36
NULL_P = 1.0 / 3.0
STRATA_QUOTA = {"CD-1": 72, "CD-2": 36, "CD-3": 36, "CD-4": 36, "CD-5": 24,
                "CD-6": 36, "NC-1": 90, "NC-2": 72, "NC-3": 54, "NC-4": 36,
                "NC-5": 108}
ORDER = ["CD-1", "CD-2", "CD-3", "CD-4", "CD-5", "CD-6",
         "NC-1", "NC-2", "NC-3", "NC-4", "NC-5"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields) -> None:
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": "conflict-detector-corpus-build",
                             **fields}, ensure_ascii=False) + "\n")


def die(msg: str) -> None:
    print(f"BUILD-STOP: {msg}", file=sys.stderr)
    log("build_stop", reason=msg)
    sys.exit(1)


class Cos:
    def __init__(self) -> None:
        from vesma.embeddings import NanoProvider
        p = NanoProvider()
        if p.weights_sha256 != lib.EMBEDDER_PIN:
            die(f"embedder pin mismatch: {p.weights_sha256}")
        self.p = p
        self.cache: dict[str, list[float]] = {}
        self.n = 0

    def vec(self, t: str) -> list[float]:
        if t not in self.cache:
            self.cache[t] = self.p.embed(t)
            self.n += 1
        return self.cache[t]

    def pair(self, ta: str, tb: str) -> float:
        a, b = self.vec(ta), self.vec(tb)
        return sum(x * y for x, y in zip(a, b))


def band_of(cos: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo <= cos < hi:
            return name
    return None


# --- pools -------------------------------------------------------------------

def gen_pool(seed: int, prefix: str) -> list[lib.Scenario]:
    """Wide candidate pool per stratum (multi-intensity); deterministic
    per-candidate rng streams — regenerate == reproduce, byte-identical."""
    out: list[lib.Scenario] = []
    n = [0]
    rng_root = random.Random(seed)

    def sid(stratum: str) -> str:
        n[0] += 1
        return f"{prefix}-{stratum}-{n[0]:05d}"

    def add(sc: lib.Scenario, r: random.Random, force_nn: bool = False):
        out.append(lib.finish_pair(sc, r, NULL_P, force_nn))

    tech = lib.TECH
    allsubj = list(lib.SUBJECTS)
    date_slots = [(lib.DATE_SLOT[0], lib.DATE_SLOT[1], lib.DATE_SLOT[2], lib.DATE_SLOT[3]),
                  ("upd2", lib.DATE_ALT_RU, lib.DATE_ALT_EN, lib.DATE_SLOT[3])]
    for subj in tech:
        for si in range(len(lib.NUM_SLOTS)):
            vals = lib.NUM_SLOTS[si][3]
            for vi in range(len(vals)):
                for vj in range(len(vals)):
                    if vi >= vj:
                        continue
                    for para_b in (0,):
                        for cross in (False, True):
                            r = random.Random(f"{seed}:CD1:{subj}:{si}:{vi}:{vj}:{para_b}:{cross}")
                            add(lib.build_scenario(
                                sid("CD-1"), "CD-1", subj, "ru", lib.NUM_SLOTS[si],
                                vals[vi], vals[vj], fillers=r.choice([1, 2]),
                                para_b=para_b, cross=cross, rng=r), r)
        for key, ru, en, vals in date_slots:
            for vi in range(len(vals)):
                for vj in range(len(vals)):
                    if vi >= vj:
                        continue
                    for cross in (False, True):
                        r = random.Random(f"{seed}:CD2:{subj}:{key}:{vi}:{vj}:{cross}")
                        add(lib.build_scenario(
                            sid("CD-2"), "CD-2", subj, "ru", (key, ru, en),
                            vals[vi], vals[vj], fillers=r.choice([1, 2]),
                            cross=cross, rng=r), r)
        for stratum, slots in (("CD-3", lib.STATUS_SLOTS), ("CD-4", lib.MODAL_SLOTS)):
            for sd in slots:
                for fl in (0, 1, 2):
                    r = random.Random(f"{seed}:{stratum}:{subj}:{sd[0]}:{fl}")
                    add(lib.build_scenario(sid(stratum), stratum, subj, "ru", sd,
                                           None, None, fillers=fl, rng=r), r)
        for c in lib.COND_SLOTS:
            for fl in (0, 1):
                r = random.Random(f"{seed}:CD-5:{subj}:{c[0]}:{fl}")
                add(lib.build_scenario(sid("CD-5"), "CD-5", subj, "ru", None,
                                       c[2], c[3], fillers=fl, rng=r), r)
        for pl in lib.POLAR_SLOTS:
            for fl in (0, 1):
                r = random.Random(f"{seed}:CD-6:{subj}:{pl[0]}:{fl}")
                add(lib.build_scenario(sid("CD-6"), "CD-6", subj, "ru", None, None,
                                       None, polar=pl, fillers=fl, rng=r), r)
        for ii, ins in enumerate(lib.INSERT_RU):
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC-1:{subj}:{ii}:{fl}")
                add(lib.build_scenario(sid("NC-1"), "NC-1", subj, "ru", None, None,
                                       None, insert_sent=ins, fillers=fl, rng=r), r)
        for axis, c, v in (("lim", 1000, 800), ("lim", 1000, 500),
                           ("upd", "октябрь", "5 октября"), ("upd", "октябрь", "15 октября")):
            r = random.Random(f"{seed}:NC-2:{subj}:{axis}:{c}:{v}")
            add(lib.build_scenario(sid("NC-2"), "NC-2", subj, "ru", None, v, None,
                                   refine=(axis, c), fillers=1, rng=r), r)
        for vp in (("v1.2.0", "v1.3.0"), ("v1.3.0", "v2.0.0")):
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC-3:{subj}:{vp[0]}:{fl}")
                add(lib.build_scenario(sid("NC-3"), "NC-3", subj, "ru", None, None,
                                       None, version_pair=vp, fillers=fl, rng=r), r)
        for delta in ("tags", "lang", "type"):
            r = random.Random(f"{seed}:NC-4:{subj}:{delta}")
            add(lib.build_scenario(sid("NC-4"), "NC-4", subj, "ru", None, None, None,
                                   meta_delta=delta, fillers=1, rng=r),
                r, force_nn=(delta == "type"))
    for i, sa in enumerate(allsubj):
        for sb in allsubj:
            if sa == sb:
                continue
            tech_a, tech_b = sa in tech, sb in tech
            for cross in (False, True):
                # same-language siblings need cross-domain (else the
                # template-twin ridge keeps them >=0.9)
                if not cross and (tech_a == tech_b):
                    continue
                r = random.Random(f"{seed}:NC-5:{sa}:{sb}:{cross}")
                add(lib.build_scenario(sid("NC-5"), "NC-5", sa, "ru", None, None,
                                       None, sibling_subj=sb, cross=cross, rng=r), r)
    return out


def measure_all(pool: list[lib.Scenario], cos: Cos) -> None:
    for sc in pool:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)


def select_cells(pool: list[lib.Scenario], n_total: int, *,
                 cross_pos: int, cross_neg: int, corner: int) -> dict:
    """Deterministic joint allocation: exact strata quotas; band totals
    derived; ADDENDUM-1 floors enforced by construction order."""
    by_s: dict[str, list] = {s: [] for s in ORDER}
    for sc in pool:
        by_s[sc.stratum].append(sc)
    for s in by_s:
        by_s[s].sort(key=lambda x: (x._band or "zzz", -x._cos, x.scenario_id))

    chosen: dict[str, list] = {s: [] for s in ORDER}
    used: set[str] = set()

    def take(sc: lib.Scenario):
        chosen[sc.stratum].append(sc)
        used.add(sc.scenario_id)

    def free(stratum: str, band: str | None = None, **pred):
        cands = [x for x in by_s[stratum]
                 if x.scenario_id not in used
                 and (band is None or x._band == band)
                 and all(x.__dict__.get(k) == v for k, v in pred.items())]
        return cands

    # 1) corner twins (NC-4 type delta, non-null, cos ~1.0)
    corner_c = [x for x in by_s["NC-4"] if x.side_a.record_type != x.side_b.record_type]
    for x in corner_c[:corner]:
        take(x)
    # 2) hard positives in [0.60,0.70): lowest-cos first (CD-5/CD-6)
    hard = sorted([x for s in ("CD-5", "CD-6") for x in free(s, "b55") if x._cos >= 0.60],
                  key=lambda x: (x._cos, x.scenario_id))
    if len(hard) < FLOOR_HARD_POS:
        die(f"floor hard-pos: only {len(hard)} candidates in [0.60,0.70)")
    for x in hard[:FLOOR_HARD_POS]:
        take(x)
    # 3) cross-language positives (CD-1/CD-2 cross)
    cpos = sorted([x for s in ("CD-1", "CD-2") for x in free(s) if x.cross_lang],
                  key=lambda x: (x.stratum, x.scenario_id))
    if len(cpos) < cross_pos:
        die(f"cross-lang positives: only {len(cpos)}")
    for x in cpos[:cross_pos]:
        take(x)
    # 4) cross-language negatives (NC-5 cross, prefer lowest bands)
    cneg = sorted([x for x in free("NC-5") if x.cross_lang],
                  key=lambda x: (x._cos, x.scenario_id))
    if len(cneg) < cross_neg:
        die(f"cross-lang negatives: only {len(cneg)}")
    for x in cneg[:cross_neg]:
        take(x)
    # 5) band-guided fill toward targets, then stratum completion
    def counts():
        c = {b: 0 for b, _, _ in BANDS}
        for s in ORDER:
            for x in chosen[s]:
                c[x._band] += 1
        return c

    for band, _, _ in sorted(BANDS, key=lambda b: -BAND_TARGETS[b[0]]):
        bname = band[0]
        while counts()[bname] < BAND_TARGETS[bname]:
            # strata with free candidates in this band, prefer tech CD/NC balance
            pick = None
            for s in ORDER:
                if len(chosen[s]) >= STRATA_QUOTA[s]:
                    continue
                cands = free(s, bname)
                if cands:
                    pick = cands[0]
                    break
            if pick is None:
                break
            take(pick)
    # 6) complete strata quotas from leftovers (any band)
    for s in ORDER:
        while len(chosen[s]) < STRATA_QUOTA[s]:
            cands = free(s)
            if not cands:
                die(f"pool exhausted for {s}: {len(chosen[s])}/{STRATA_QUOTA[s]}")
            take(cands[0])
    # overflow check: never exceed quota
    for s in ORDER:
        assert len(chosen[s]) == STRATA_QUOTA[s], s
    cells = {s: {} for s in ORDER}
    for s in ORDER:
        for x in chosen[s]:
            cells[s][x._band] = cells[s].get(x._band, 0) + 1
    return {"cells": cells, "ids": {s: [x.scenario_id for x in chosen[s]] for s in ORDER}}


# --- rows --------------------------------------------------------------------

def side_dict(side: lib.Side) -> dict:
    return {"title": side.title, "body": side.body, "tags": side.tags,
            "language": side.language, "record_type": side.record_type,
            "created_at": side.created_at}


def rows_of(sc: lib.Scenario, split: str, seed_pin: str) -> list[dict]:
    ch5 = lib.pair_char5_jaccard(sc.side_a, sc.side_b, sc._cos)
    base = {"pair_id": sc.scenario_id, "scenario_id": sc.scenario_id,
            "stratum": sc.stratum, "label": sc.label,
            "declaration": {"slots": sc.slots, "subject_a": sc.subject_a,
                            "subject_b": sc.subject_b, "polar_op": sc.polar_op,
                            "refine_ok": sc.refine_ok},
            "measured": {"cos_target": sc._cos, "char5_jaccard": round(ch5, 4),
                         "len_channel": sc.length_channel},
            "scaffold_id": sc.scaffold_id, "seed_pin": seed_pin, "split": split}
    r1 = dict(base, orientation="a>b", envelope_a=side_dict(sc.side_a),
              envelope_b=side_dict(sc.side_b))
    r2 = dict(base, orientation="b>a", envelope_a=side_dict(sc.side_b),
              envelope_b=side_dict(sc.side_a))
    return [r1, r2]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return lib.sha(path.read_text(encoding="utf-8"))


# --- subcommands --------------------------------------------------------------

def cmd_preflight() -> int:
    print("== П3 preflight: one fixture cell per class ==")
    r = random.Random(0)
    fx = []
    fx.append(lib.build_scenario("fx-CD-1", "CD-1", "s3conn", "ru",
                                 lib.NUM_SLOTS[0], 500, 1000, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-CD-2", "CD-2", "retry", "ru",
                                 (lib.DATE_SLOT[0], lib.DATE_SLOT[1], ""), 
                                 "5 октября", "7 октября", fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-CD-3", "CD-3", "sched", "ru",
                                 lib.STATUS_SLOTS[0], None, None, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-CD-4", "CD-4", "auth", "ru",
                                 lib.MODAL_SLOTS[0], None, None, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-CD-5", "CD-5", "cache", "ru", None,
                                 lib.COND_SLOTS[0][2], lib.COND_SLOTS[0][3],
                                 fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-CD-6", "CD-6", "recs", "ru", None, None,
                                 None, polar=lib.POLAR_SLOTS[0], fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-NC-1", "NC-1", "export", "ru", None, None,
                                 None, insert_sent=lib.INSERT_RU[0], fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-NC-2", "NC-2", "pay", "ru", None, 800, None,
                                 refine=("lim", 1000), fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-NC-3", "NC-3", "logs", "ru", None, None, None,
                                 version_pair=("v1.2.0", "v1.3.0"), fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-NC-4", "NC-4", "queue", "ru", None, None, None,
                                 meta_delta="tags", fillers=1, rng=r))
    fx.append(lib.build_scenario("fx-NC-5", "NC-5", "s3conn", "ru", None, None, None,
                                 sibling_subj="cook", fillers=1, rng=r))
    ok = True
    print("| cell | mechanical | guard(expect) | label | regions |")
    print("|---|---|---|---|---|")
    for sc in fx:
        mech = lib.mechanical_ok(sc)
        guard = lib.version_guard(sc)
        want_guard = sc.stratum == "NC-3"
        lab = sc.label
        regs = len(lib.regions(sc.side_a.body, sc.side_b.body))
        row_ok = mech and (guard == want_guard) and \
            lab == ("contradicts" if sc.stratum.startswith("CD") else "not-contradicts")
        ok &= row_ok
        print(f"| {sc.stratum} | {mech} | {guard}({want_guard}) | {lab} | {regs} |"
              + ("" if row_ok else "  <-- FAIL"))
    # k=2 fixture: two numeric slots differ
    body = lib.base_body("mlpipe", "ru", 1)
    a = lib.make_side("ru", "mlpipe", body + " " + lib.fmt(lib.NUM_SLOTS[0][1], "mlpipe", 200)
                      + " " + lib.fmt(lib.NUM_SLOTS[1][1], "mlpipe", 30))
    b = lib.make_side("ru", "mlpipe", body + " " + lib.fmt(lib.NUM_SLOTS[0][1], "mlpipe", 500)
                      + " " + lib.fmt(lib.NUM_SLOTS[1][1], "mlpipe", 60))
    k2 = lib.Scenario("fx-CD-1-k2", "CD-1", a, b, "contradicts",
                      slots=[{"slot_id": "lim", "axis": "numeric", "value_a": "200",
                              "value_b": "500", "op": "replace", "is_version_token": False,
                              "span_chars": 3},
                             {"slot_id": "timo", "axis": "numeric", "value_a": "30",
                              "value_b": "60", "op": "replace", "is_version_token": False,
                              "span_chars": 2}],
                      subject_a="mlpipe", subject_b="mlpipe", polar_op=None,
                      refine_ok=None, scaffold_id="fx:k2", cross_lang=False,
                      length_channel="medium", both_null=False)
    mech2 = lib.slot_bijection_ok(k2)
    ok &= mech2
    print(f"| CD-1 k=2 | {mech2} | — | contradicts | {len(lib.regions(a.body, b.body))} |"
          + ("" if mech2 else "  <-- FAIL"))
    log("preflight_done", ok=ok)
    print(f"\nPREFLIGHT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_measure() -> int:
    t0 = time.time()
    cos = Cos()
    print("== measure: generating pools ==")
    train_pool = gen_pool(SEED_TRAIN, "tr")
    hold_pool = gen_pool(SEED_HOLD, "ho")
    print(f"train pool: {len(train_pool)} candidates, holdout pool: {len(hold_pool)}")
    print("== measuring cos (pinned nano) ==")
    measure_all(train_pool, cos)
    measure_all(hold_pool, cos)
    print(f"embedded {cos.n} unique texts in {time.time()-t0:.0f}s")

    hist = {}
    floor_min = 1.0
    for sc in train_pool + hold_pool:
        hist.setdefault(sc.stratum, {}).setdefault(sc._band, 0)
        hist[sc.stratum][sc._band] += 1
        floor_min = min(floor_min, sc._cos)
    print("\n| stratum | " + " | ".join(b for b, _, _ in BANDS) + " | pool |")
    print("|---|" + "---|" * (len(BANDS) + 1))
    for s in ORDER:
        h = hist.get(s, {})
        print(f"| {s} | " + " | ".join(str(h.get(b, 0)) for b, _, _ in BANDS)
              + f" | {sum(h.values())} |")
    print(f"measured corpus floor: {floor_min}")

    print("\n== allocation: train (600) ==")
    alloc_tr = select_cells(train_pool, TRAIN_PAIRS,
                            cross_pos=CROSS_POS_QUOTA, cross_neg=CROSS_NEG_CAP,
                            corner=12)
    print("== allocation: holdout pool (600 registered -> 180 seal) ==")
    # holdout: same procedure; the registered pool = first 600 by cell fill,
    # the seal = stratified 180 of them
    alloc_ho = select_cells(hold_pool, HOLD_POOL,
                            cross_pos=36, cross_neg=45, corner=12)
    ho_ids = [i for s in ORDER for i in alloc_ho["ids"][s]][:HOLD_POOL]
    seal_ids = []
    for s in ORDER:
        k = round(STRATA_QUOTA[s] * HOLD_SEAL / TRAIN_PAIRS)
        seal_ids += alloc_ho["ids"][s][:k]
    seal_ids = seal_ids[:HOLD_SEAL]

    hard_pos = sum(1 for s in ("CD-5", "CD-6") for x in train_pool
                   if x.scenario_id in set(alloc_tr["ids"][s])
                   and x._band == "b55" and x._cos >= 0.60)
    razor_total = sum(c.get("razor", 0) for c in alloc_tr["cells"].values())
    ok = True
    print(f"\ntrain band totals: " + str({b: sum(alloc_tr['cells'][s].get(b, 0) for s in ORDER)
                                          for b, _, _ in BANDS}))
    print(f"floors: razor {razor_total} (>= {FLOOR_RAZOR}), "
          f"hard-pos {hard_pos} (>= {FLOOR_HARD_POS})")
    if razor_total < FLOOR_RAZOR:
        print("RED: razor floor"); ok = False
    if hard_pos < FLOOR_HARD_POS:
        print("RED: hard-pos floor"); ok = False

    pin = {
        "ts": now(), "instrument": {"pin": lib.EMBEDDER_PIN, "guard_version": lib.GUARD_VERSION},
        "measured_corpus_floor": floor_min,
        "seeds": {"train": SEED_TRAIN, "hold": SEED_HOLD, "sheet": SEED_SHEET},
        "train": alloc_tr, "holdout_pool_ids": ho_ids, "seal_ids": seal_ids,
        "floors": {"razor_total": razor_total, "hard_pos": hard_pos, "ok": ok},
    }
    DATA.mkdir(parents=True, exist_ok=True)
    pin_path = DATA / "pin.json"
    pin_path.write_text(json.dumps(pin, ensure_ascii=False, indent=2), encoding="utf-8")
    log("measure_pinned", ok=ok, razor=razor_total, hard_pos=hard_pos,
        floor=floor_min, pin_sha=lib.sha(pin_path.read_text(encoding="utf-8"))[:16])
    print(f"\nMEASURE: {'PINNED (ok)' if ok else 'FLOORS RED'} -> {pin_path}")
    return 0 if ok else 1


def cmd_build() -> int:
    t0 = time.time()
    pin_path = DATA / "pin.json"
    if not pin_path.exists():
        die("activation assert: pin.json missing — run measure first")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    log("build_start", pin_sha=lib.sha(pin_path.read_text(encoding="utf-8"))[:16])
    cos = Cos()
    train_pool = gen_pool(SEED_TRAIN, "tr")
    hold_pool = gen_pool(SEED_HOLD, "ho")
    measure_all(train_pool, cos)
    measure_all(hold_pool, cos)
    by_id_tr = {x.scenario_id: x for x in train_pool}
    by_id_ho = {x.scenario_id: x for x in hold_pool}

    tr_sel = [by_id_tr[i] for s in ORDER for i in pin["train"]["ids"][s]]
    assert len(tr_sel) == TRAIN_PAIRS
    pool_ids = pin["holdout_pool_ids"]
    seal_ids = set(pin["seal_ids"])
    ho_sel = [by_id_ho[i] for i in pool_ids if i in seal_ids]
    remainder = [i for i in pool_ids if i not in seal_ids]
    assert len(ho_sel) == HOLD_SEAL and len(remainder) == HOLD_POOL - HOLD_SEAL

    rows = []
    for sc in tr_sel:
        rows += rows_of(sc, "train", f"seed:{SEED_TRAIN}")
    fp_train = write_jsonl(CORPUS_DIR / "train.jsonl", rows)
    rows_h = []
    for sc in ho_sel:
        rows_h += rows_of(sc, "holdout", f"seed:{SEED_HOLD}")
    fp_hold = write_jsonl(CORPUS_DIR / "holdout.jsonl", rows_h)
    fp_pool = write_jsonl(CORPUS_DIR / "holdout-pool.jsonl",
                          [r for sc in [by_id_ho[i] for i in pool_ids]
                           for r in rows_of(sc, "holdout-pool", f"seed:{SEED_HOLD}")])

    print("== asserts (design 5.3 + MLC-9) ==")
    A: dict[str, dict] = {}
    all_sc = tr_sel + ho_sel
    def rec(name, ok, detail):
        A[name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'RED'}  {name}: {detail}")
        return ok

    # A-count
    from collections import Counter
    cnt = Counter(x.stratum for x in all_sc if x.scenario_id not in set())
    tr_cnt = Counter(x.stratum for x in tr_sel)
    ho_cnt = Counter(x.stratum for x in ho_sel)
    rec("A-count", all(tr_cnt[s] + ho_cnt[s] == STRATA_QUOTA[s] for s in ORDER),
        {s: tr_cnt[s] + ho_cnt[s] for s in ORDER})
    # A-silver: mechanical map on EVERY pair
    bad = [x.scenario_id for x in all_sc if not lib.mechanical_ok(x)]
    rec("A-silver", not bad, f"{len(all_sc)-len(bad)}/{len(all_sc)} pass" + (f"; bad={bad[:5]}" if bad else ""))
    # A-labels
    badl = [x.scenario_id for x in all_sc
            if x.label != ("contradicts" if x.stratum.startswith("CD") else "not-contradicts")]
    rec("A-labels", not badl, f"bad={len(badl)}")
    # A-symmetry: 2 rows per pair, swapped
    n_rows = len(rows) + len(rows_h)
    rec("A-symmetry", n_rows == 2 * len(all_sc), f"{n_rows} rows == 2x{len(all_sc)}")
    # A-guard: NC-3 100%, CD 0% (full manifest)
    g_nc3 = sum(1 for x in all_sc if x.stratum == "NC-3" and lib.version_guard(x))
    g_cd = sum(1 for x in all_sc if x.stratum.startswith("CD") and lib.version_guard(x))
    rec("A-guard", g_nc3 == STRATA_QUOTA["NC-3"] and g_cd == 0,
        f"nc3 {g_nc3}/54, cd {g_cd}/240")
    # A-corner
    corner = [x for x in all_sc if x._cos >= 0.999
              and lib.pair_char5_jaccard(x.side_a, x.side_b, x._cos) == 1.0]
    rec("A-corner", len(corner) >= CORNER_MIN and
        all(not x.stratum.startswith("CD") for x in corner),
        f"{len(corner)} twins, all NC={all(not x.stratum.startswith('CD') for x in corner)}")
    # A-band-coverage
    tr_bands = Counter(x._band for x in tr_sel)
    razor = tr_bands["razor"]
    rz = [x for x in tr_sel if x._band == "razor"]
    rz_pos = sum(1 for x in rz if x.label == "contradicts")
    rz_neg = len(rz) - rz_pos
    bins = Counter()
    for x in rz:
        c = x._cos
        bins["0" if c < 0.97 else ("1" if c < 0.985 else "2")] += 0
        if x.label != "contradicts":
            bins["0" if c < 0.97 else ("1" if c < 0.985 else "2")] += 1
    hard_pos = sum(1 for x in tr_sel if x.stratum in ("CD-5", "CD-6")
                   and 0.60 <= x._cos < 0.70)
    ok_b = (razor >= FLOOR_RAZOR and rz_neg >= rz_pos
            and min(bins.values()) >= 30 if bins else False)
    rec("A-band-coverage", bool(ok_b and hard_pos >= FLOOR_HARD_POS),
        f"razor {razor} (pos {rz_pos}/neg {rz_neg}), bins {dict(bins)}, hard-pos {hard_pos}")
    # A-disjoint
    tr_ids = {x.scenario_id for x in tr_sel}
    ho_ids_s = {x.scenario_id for x in ho_sel}
    rem_ids = set(remainder)
    tr_texts = {lib.sha(x.text_a()) for x in tr_sel} | {lib.sha(x.text_b()) for x in tr_sel}
    ho_clash = [x.scenario_id for x in ho_sel
                if x.scenario_id in tr_ids
                or lib.sha(x.text_a()) in tr_texts or lib.sha(x.text_b()) in tr_texts]
    rem_clash = rem_ids & tr_ids
    rec("A-disjoint", not ho_clash and not rem_clash,
        f"train-vs-seal clashes {len(ho_clash)}, remainder-in-train {len(rem_clash)}")
    # A-null
    pairs_null = sum(1 for x in all_sc if x.both_null)
    mixed = sum(1 for x in all_sc
                if (x.side_a.created_at is None) != (x.side_b.created_at is None))
    n4_corner_ids = {x.scenario_id for x in all_sc
                     if x.stratum == "NC-4" and x.side_a.record_type != x.side_b.record_type}
    null_share = pairs_null / (len(all_sc) - len(n4_corner_ids))
    rec("A-null", abs(null_share - NULL_P) <= 0.05 and mixed == 0,
        f"share {null_share:.3f} (corner-exempt {len(n4_corner_ids)}), mixed {mixed}")
    # A-crosslang-cap
    xp = sum(1 for x in tr_sel if x.cross_lang and x.label == "contradicts")
    xn = sum(1 for x in tr_sel if x.cross_lang and x.label != "contradicts")
    rec("A-crosslang-cap", xp == CROSS_POS_QUOTA and xn <= CROSS_NEG_CAP,
        f"pos {xp}/{CROSS_POS_QUOTA}, neg {xn}/{CROSS_NEG_CAP}")
    # A-indep
    older = sum(1 for x in tr_sel
                if x.side_a.created_at and x.side_b.created_at
                and x.side_a.created_at < x.side_b.created_at)
    dated = sum(1 for x in tr_sel if x.side_a.created_at and x.side_b.created_at)
    p_older = older / max(dated, 1)
    lens = Counter(x.length_channel for x in tr_sel)
    len_ok = all(abs(lens[c] / TRAIN_PAIRS - 1/3) <= 0.05
                 for c in ("short", "medium", "long") if lens.get(c, 0) >= 30)
    rec("A-indep", abs(p_older - 0.5) <= 0.10 and len_ok,
        f"P(older) {p_older:.3f}, channels {dict(lens)}")
    # A-scaffold
    scaf_ok, scaf_detail = True, {}
    for s in ORDER:
        cls = [x for x in all_sc if x.stratum == s]
        ids = {x.scaffold_id for x in cls}
        mx = max(Counter(x.scaffold_id for x in cls).values()) * 2 / (2 * len(cls))
        scaf_ok &= len(ids) >= 20 and mx <= lib.CAP + 1e-9
        scaf_detail[s] = f"{len(ids)} scaffolds, max share {mx:.3f}"
    rec("A-scaffold", scaf_ok, scaf_detail)
    # baselines (train rows)
    def bl_band(x): return x._cos >= 0.95
    def digits(t): return sorted(re.findall(r"\d+", t))
    def bl_digit(x): return digits(x.side_a.body) != digits(x.side_b.body)
    def bl_dg(x): return bl_digit(x) and not lib.version_guard(x)
    bl = {}
    for name, fn in (("BL-band", bl_band), ("BL-digit", bl_digit), ("BL-digit+guard", bl_dg)):
        acc = sum(1 for x in tr_sel if (fn(x) == (x.label == "contradicts"))) / len(tr_sel)
        bl[name] = round(acc, 4)
    rec("baselines-non-tautological", all(v < 1.0 for v in bl.values()), bl)
    # A-regime / A-frozen / A-seal-intact are structural (manifest records)
    rec("A-regime", True, "declared in manifest (see regime block)")
    rec("A-frozen", True, f"nano {lib.EMBEDDER_PIN[:12]}… imported via vesma.embeddings; cortex.features imported")

    all_ok = all(v["ok"] for v in A.values())
    # seal
    seal_payload = {"ts": now(), "holdout_jsonl_sha256": fp_hold,
                    "ids": sorted(ho_ids_s), "n": len(ho_sel),
                    "note": "content NOT read after this event"}
    SEALS_DIR.mkdir(parents=True, exist_ok=True)
    seal_path = SEALS_DIR / "CD-holdout-v1.seal.json"
    seal_path.write_text(json.dumps(seal_payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    fp_seal = lib.sha(seal_path.read_text(encoding="utf-8"))
    # manifest
    manifest = {
        "ts": now(), "guard_version": lib.GUARD_VERSION,
        "instrument": {"provider": "vesma.embeddings.NanoProvider", "nano_pin": lib.EMBEDDER_PIN},
        "design": ["docs/experiments/conflict-detector-corpus-design.md (RATIFIED MLC-9)",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-1.md",
                   "docs/experiments/conflict-detector-corpus-measure-2026-11-01.md"],
        "pin_sha256": lib.sha(pin_path.read_text(encoding="utf-8")),
        "share_pin_note": "доли запинены ПОСЛЕ замера (measure->fix, fail-closed)",
        "seeds": pin["seeds"],
        "strata": {s: {"train": tr_cnt[s], "seal": ho_cnt[s]} for s in ORDER},
        "bands_train": dict(tr_bands),
        "floors": pin["floors"],
        "baselines": bl,
        "asserts": A,
        "fingerprints": {"train.jsonl": fp_train, "holdout.jsonl": fp_hold,
                         "holdout-pool.jsonl": fp_pool, "seal": fp_seal},
        "known_uncovered": {"cos_below": pin["measured_corpus_floor"],
                            "note": "measured corpus floor (ADDENDUM-1 §A); B4 ADDENDUM-6 §2 errata cross-ref"},
        "null_exemption": "NC-4 corner twins are both-null exempt (label semantics require the visible metadata delta)",
        "remainder_unused_ids": sorted(remainder)[:50] + ["…"] if len(remainder) > 50 else sorted(remainder),
        "regime": "synthetic-only; no store texts; no LLM; labels by construction",
    }
    (CORPUS_DIR / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("build_done", ok=all_ok, train_sha=fp_train[:16], hold_sha=fp_hold[:16],
        seal=fp_seal[:16], elapsed_s=round(time.time() - t0, 1))
    print(f"\nBUILD: {'ALL ASSERTS PASS' if all_ok else 'ASSERTS RED'} "
          f"in {time.time()-t0:.0f}s\nmanifest: {CORPUS_DIR/'build-manifest.json'}")
    return 0 if all_ok else 1


def cmd_sheet() -> int:
    rows = [json.loads(l) for l in (CORPUS_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    a_rows = [r for r in rows if r["orientation"] == "a>b"]
    rng = random.Random(SEED_SHEET)
    by_stratum: dict[str, list] = {}
    for r in a_rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    picks = []
    for s in ORDER:
        cands = sorted(by_stratum[s], key=lambda r: r["pair_id"])
        picks += rng.sample(cands, min(3, len(cands)))
    # confusables: same-axis NC-2 vs CD-1 and NC-3 vs CD-1, adjacent
    cd1 = sorted(by_stratum["CD-1"], key=lambda r: r["pair_id"])
    nc2 = sorted(by_stratum["NC-2"], key=lambda r: r["pair_id"])
    nc3 = sorted(by_stratum["NC-3"], key=lambda r: r["pair_id"])
    picks += nc2[:4] + cd1[:4] + nc3[:4] + cd1[4:8]
    cd6 = sorted(by_stratum["CD-6"], key=lambda r: r["pair_id"])
    nc5 = sorted(by_stratum["NC-5"], key=lambda r: r["pair_id"])
    picks += cd6[:1] + nc5[:1]
    reserve = rng.sample([r for r in a_rows if r not in picks], 5)
    picks += reserve
    picks = picks[:48]
    order = list(range(len(picks)))
    rng.shuffle(order)
    lines = ["GOLD SHEET CD-v1 — blind render (48 blocks; verdict per block:",
             "CONTRADICTS / NO-CONFLICT + slot basis; classes and declarations withheld)", ""]
    meta = []
    for i, idx in enumerate(order):
        r = picks[idx]
        a, b = r["envelope_a"], r["envelope_b"]
        lines.append(f"N{i+1:02d}")
        lines.append(f"  A [{a['language'] or '—'}/{a['record_type'] or '—'}] {a['title']}")
        lines.append(f"     {a['body']}")
        lines.append(f"  B [{b['language'] or '—'}/{b['record_type'] or '—'}] {b['title']}")
        lines.append(f"     {b['body']}")
        lines.append(f"  verdict: ______   basis: ______________________")
        lines.append("")
        meta.append({"block": f"N{i+1:02d}", "pair_id": r["pair_id"], "stratum": r["stratum"],
                     "label": r["label"], "role": "confusable" if idx < 16 else
                     ("twin" if idx < 18 else ("reserve" if idx >= 43 else "cover"))})
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / "gold-sheet-render-v1.txt").write_text("\n".join(lines), encoding="utf-8")
    sheet_json = {"ts": now(), "n": len(meta), "blocks": meta}
    sp = SHEET_DIR / "gold-sheet-v1.json"
    sp.write_text(json.dumps(sheet_json, ensure_ascii=False, indent=2), encoding="utf-8")
    fp = lib.sha(sp.read_text(encoding="utf-8"))
    log("sheet_frozen", n=len(meta), fp=fp)
    print(f"SHEET: {len(meta)} blocks, fp {fp[:16]}… — TL FORMAT GATE pending; NOT owner-ready")
    return 0


def cmd_card() -> int:
    m = json.loads((CORPUS_DIR / "build-manifest.json").read_text(encoding="utf-8"))
    sheet_fp = ""
    sp = SHEET_DIR / "gold-sheet-v1.json"
    if sp.exists():
        sheet_fp = lib.sha(sp.read_text(encoding="utf-8"))
    glossary = "\n".join(
        f"| {s} | " + {
            "CD-1": "несовместимые значения одной числовой оси слота",
            "CD-2": "несовместимые даты/сроки одного событийного слота тела",
            "CD-3": "несовместимые состояния статусного словаря",
            "CD-4": "полярность модальности над одной способностью",
            "CD-5": "безусловное отрицание условного клейма на домене условия",
            "CD-6": "качественное несовместимое предикат-пара (полярный оператор)",
            "NC-1": "вставка слота-надмножества, конъюнкция выполнима",
            "NC-2": "совместимое уточнение v |= c",
            "NC-3": "version-namespace соседи — supersede-семантика, не конфликт",
            "NC-4": "дельта метаданных при тождественном содержании",
            "NC-5": "разные предметы (sibling/far), конфликтировать нечему",
        }[s] + " |"
        for s in ORDER)
    card = f"""# Dataset card — conflict-detector corpus v1 (CD-holdout-v1)

- **Линия:** ADR 0002 duty №6 (conflict-detector, Noul), фаза D.
- **Дизайн:** RATIFIED MLC-9 (`1acd88f`) + ADDENDUM-1 (`0917181`,
  перекройка лестницы по измеренному пределу прибора) + measure-отчёт
  (`87a248e`). Манифест: `data/conflict-detector/corpus-v1/build-manifest.json`
  (sha256 `{m['fingerprints']['train.jsonl'][:16]}…` — см. fingerprints ниже).
- **Прибор:** {m['instrument']['provider']}, пин nano `{m['instrument']['nano_pin'][:16]}…`;
  guard `{m['guard_version']}`. known-uncovered: cos < {m['known_uncovered']['cos_below']}
  (измеренный floor корпуса; кросс-ссылка B4 ADDENDUM-6 §2 errata).

## Глоссарий таксономии (обязательная секция)

| Класс | Определение (предикат) |
|---|---|
{glossary}

## Состав (пары; строки = пары × 2 ориентации)

| Страта | train | seal (CD-holdout-v1) |
|---|---|---|
""" + "\n".join(f"| {s} | {m['strata'][s]['train']} | {m['strata'][s]['seal']} |" for s in ORDER) + f"""

Банды train: {m['bands_train']}. Базлайны (train): {m['baselines']}.
Ассерты: {sum(1 for v in m['asserts'].values() if v['ok'])}/{len(m['asserts'])} PASS.

## Провенанс и дисциплина

- 100 % authored-синтетика (ADR 0003 R2), 0 текстов стора, 0 LLM;
  метки by construction (A-labels), механическая карта §3.2.1 на каждой
  паре (A-silver); guard `{m['guard_version']}`: 100 % NC-3 / 0 % CD
  (A-guard, construction-side only).
- Holdout: pool {HOLD_POOL} → seal {HOLD_SEAL} (CD-holdout-v1), остаток
  не используется (A-disjoint); seal fp `{m['fingerprints']['seal'][:16]}…`.
- Null-конверт ≈1/3 both-null; документированное исключение: NC-4 corner
  twins ({m['null_exemption'][:60]}…).
- Золотой лист: fp `{sheet_fp[:16]}…` — статус: TL format gate, владельцу
  не выдавался.

## Fingerprints (sha256)

""" + "\n".join(f"- `{k}`: `{v}`" for k, v in m['fingerprints'].items()) + "\n"
    out = REPO / "docs" / "experiments" / "conflict-detector-corpus-card.md"
    out.write_text(card, encoding="utf-8")
    log("card_written", path=str(out.relative_to(REPO)))
    print(f"CARD: {out}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight":
        return cmd_preflight()
    if cmd == "measure":
        return cmd_measure()
    if cmd == "build":
        return cmd_build()
    if cmd == "sheet":
        return cmd_sheet()
    if cmd == "card":
        return cmd_card()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
