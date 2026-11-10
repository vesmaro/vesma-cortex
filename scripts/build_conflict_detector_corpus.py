#!/usr/bin/env python3
"""Build driver for the conflict-detector corpus v1 (ADR 0002 duty #6).

Subcommands (the ratified lane order):
  preflight  П3 fixture cells — one per class, mechanical map + guard
  measure    dry pools: wide candidates, measured cos (pinned nano),
             joint allocation honoring exact strata quotas + ADDENDUM-1/2
             floors (hard window [0.70,0.75) per ADDENDUM-2), write pin
             file (activation assert for build)
  build      regenerate identical pools, select per pin, emit
             train.jsonl (600 pairs x 2 orientations) + holdout pool 600
             -> seal 180 (CD-holdout-v1), full assert map (design 5.3),
             baselines, build manifest
  sheet      48-block gold sheet, blind render, fp frozen (TL format
             gate BEFORE any owner hand-off)
  card       dataset card + fingerprints (glossary mandatory)
  seal3      CD-holdout-v3 rotation (ADDENDUM-5 §4 F6-F12): POOL-ONLY
             lane — fresh hold seed -> census E9-mini -> cut 180 ->
             seal + pin-v3; train/sheet absent by construction;
             measure2/build2 untouched byte-for-byte (F7)

Instrument: engine venv python (vesma.embeddings); pin
nano:sha256:3b752e06... verified at every entry — wrong provider refuses
loudly (A-frozen). Data under data/ (gitignored); run-log append-only.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import re
import sys
import time
from collections import Counter
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
# ADDENDUM-2 §A: the hard window is a POPULATION CONSTRAINT inside the
# b70 band [0.70,0.85) — no separate allocation cell.
HARD_WIN_LO, HARD_WIN_HI = 0.70, 0.75
# §4.3 razor bins: >=30 negatives each; continuity <=0.005 inside razor
RAZOR_BINS = [(0.95, 0.97), (0.97, 0.985), (0.985, 1.01)]
RAZOR_BIN_FLOOR = 30
RAZOR_MAX_GAP = 0.005
CORNER_MIN = 10
CROSS_NEG_CAP = 45
CROSS_POS_QUOTA = 36
NULL_P = 1.0 / 3.0
# §4.4 length channels ~1/3 each: long-channel reservation per stratum
# (sum 190/600 = 0.317)
LONG_RESERVE = {"CD-1": 24, "CD-2": 12, "CD-3": 12, "CD-4": 12, "CD-5": 8,
                "CD-6": 12, "NC-1": 20, "NC-2": 24, "NC-3": 18, "NC-4": 12,
                "NC-5": 36}
STRATA_QUOTA = {"CD-1": 72, "CD-2": 36, "CD-3": 36, "CD-4": 36, "CD-5": 24,
                "CD-6": 36, "NC-1": 90, "NC-2": 72, "NC-3": 54, "NC-4": 36,
                "NC-5": 108}
ORDER = ["CD-1", "CD-2", "CD-3", "CD-4", "CD-5", "CD-6",
         "NC-1", "NC-2", "NC-3", "NC-4", "NC-5"]
# §4.5 scaffold cap: max pairs of a stratum on one scaffold across
# train+seal (5% of the projected union, floor 1)
SCAFFOLD_CAP = {s: max(1, int(0.065 * STRATA_QUOTA[s])) for s in ORDER}


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

def _sents(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=\.)\s+", body.strip()) if s]


def rotate_pair(a_body: str, b_body: str, seed: int) -> tuple[str, str]:
    """Seed-dependent sentence-order rotation (A-disjoint surface jitter):
    the train/holdout pools share fixed-text families (cross templates,
    SPEC bodies, long boilerplate) whose texts would be byte-identical
    across seeds. BOTH sides rotate by the SAME offset k (1 <= k < base
    length), so the base block stays contiguous inside the b-side and the
    mechanical region map survives (NC-1 stays all-insert, NC-4 stays
    identity-body). Single-sentence sides pass through unchanged."""
    sa, sb = _sents(a_body), _sents(b_body)
    if len(sa) >= 2:
        k = 1 + (seed % (len(sa) - 1))
        return (" ".join(sa[k:] + sa[:k]),
                " ".join(sb[k:] + sb[:k]) if len(sb) >= 2 else b_body)
    if len(sb) >= 2:
        k = 1 + (seed % (len(sb) - 1))
        return (a_body, " ".join(sb[k:] + sb[:k]))
    return a_body, b_body


def gen_pool(seed: int, prefix: str, rotate: bool | None = None) -> list[lib.Scenario]:
    """Wide candidate pool per stratum (multi-intensity); deterministic
    per-candidate rng streams — regenerate == reproduce, byte-identical.
    The holdout pool rotates sentence order so its texts never collide
    with train texts (A-disjoint §5.3). rotate: None keeps the v1
    behavior (rotate iff seed != SEED_TRAIN); v2 passes it EXPLICITLY —
    the v1 seed-equality guard breaks under a fresh train seed (the v2
    train seed != 42 would rotate the train pool too, with the same
    k for equal sentence counts, producing literal text twins)."""
    out: list[lib.Scenario] = []
    n = [0]
    rng_root = random.Random(seed)
    do_rotate = (seed != SEED_TRAIN) if rotate is None else rotate

    def sid(stratum: str) -> str:
        n[0] += 1
        return f"{prefix}-{stratum}-{n[0]:05d}"

    def add(sc: lib.Scenario, r: random.Random, force_nn: bool = False):
        if do_rotate:
            sc.side_a.body, sc.side_b.body = rotate_pair(
                sc.side_a.body, sc.side_b.body, seed)
        out.append(lib.finish_pair(sc, r, NULL_P, force_nn))

    tech = lib.TECH
    allsubj = list(lib.SUBJECTS)
    date_slots = [(lib.DATE_SLOT[0], lib.DATE_SLOT[1], lib.DATE_SLOT[2],
                   lib.DATE_SLOT[3], lib.DATE_SLOT[4]),
                  ("upd2", lib.DATE_ALT_RU, lib.DATE_ALT_EN,
                   lib.DATE_SLOT[3], lib.DATE_SLOT[4])]
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
                                vals[vi], vals[vj], fillers=r.choice([0, 1, 2]),
                                para_b=para_b, cross=cross, rng=r), r)
                    # ratified hard-floor source family "CD-1-paraphrase"
                    # (design 4.3 / ADDENDUM-1 B): cross pairs with the
                    # paraphrase dial engaged
                    for para_b in (1, 2):
                        r = random.Random(f"{seed}:CD1p:{subj}:{si}:{vi}:{vj}:{para_b}")
                        add(lib.build_scenario(
                            sid("CD-1"), "CD-1", subj, "ru", lib.NUM_SLOTS[si],
                            vals[vi], vals[vj], fillers=r.choice([1, 2]),
                            para_b=para_b, cross=True, rng=r), r)
                    # long channel (fillers=35 -> >=2000 chars, same-lang
                    # only: long cross bodies would dive out of their bands)
                    r = random.Random(f"{seed}:CD1L:{subj}:{si}:{vi}:{vj}")
                    add(lib.build_scenario(
                        sid("CD-1"), "CD-1", subj, "ru", lib.NUM_SLOTS[si],
                        vals[vi], vals[vj], fillers=35, rng=r), r)
        for key, ru, en, vals_ru, vals_en in date_slots:
            for vi in range(len(vals_ru)):
                for vj in range(len(vals_ru)):
                    if vi >= vj:
                        continue
                    for cross in (False, True):
                        vals = vals_en if cross else vals_ru
                        r = random.Random(f"{seed}:CD2:{subj}:{key}:{vi}:{vj}:{cross}")
                        add(lib.build_scenario(
                            sid("CD-2"), "CD-2", subj, "ru", (key, ru, en),
                            vals[vi], vals[vj], fillers=r.choice([0, 1, 2]),
                            cross=cross, rng=r,
                            scaffold_tag="x" if cross else "s"), r)
                    r = random.Random(f"{seed}:CD2L:{subj}:{key}:{vi}:{vj}")
                    add(lib.build_scenario(
                        sid("CD-2"), "CD-2", subj, "ru", (key, ru, en),
                        vals_ru[vi], vals_ru[vj], fillers=35, rng=r,
                        scaffold_tag="L"), r)
        for stratum, slots in (("CD-3", lib.STATUS_SLOTS), ("CD-4", lib.MODAL_SLOTS)):
            for sd in slots:
                for fl in (0, 1, 2):
                    r = random.Random(f"{seed}:{stratum}:{subj}:{sd[0]}:{fl}")
                    add(lib.build_scenario(sid(stratum), stratum, subj, "ru", sd,
                                           None, None, fillers=fl, rng=r,
                                           scaffold_tag=f"f{fl}"), r)
                r = random.Random(f"{seed}:{stratum}L:{subj}:{sd[0]}")
                add(lib.build_scenario(sid(stratum), stratum, subj, "ru", sd,
                                       None, None, fillers=35, rng=r,
                                       scaffold_tag="L"), r)
        for c in lib.COND_SLOTS:
            for fl in (0, 1):
                r = random.Random(f"{seed}:CD-5:{subj}:{c[0]}:{fl}")
                add(lib.build_scenario(sid("CD-5"), "CD-5", subj, "ru", None,
                                       c[2], c[3], fillers=fl, rng=r,
                                       scaffold_tag=f"{c[0]}f{fl}"), r)
            r = random.Random(f"{seed}:CD-5L:{subj}:{c[0]}")
            add(lib.build_scenario(sid("CD-5"), "CD-5", subj, "ru", None,
                                   c[2], c[3], fillers=35, rng=r,
                                   scaffold_tag=f"{c[0]}L"), r)
        for pl in lib.POLAR_SLOTS:
            pe0 = next(e for e in lib.EN_POLAR if e[0] == pl[0])
            for fl in (0, 1):
                r = random.Random(f"{seed}:CD-6:{subj}:{pl[0]}:{fl}")
                add(lib.build_scenario(sid("CD-6"), "CD-6", subj, "ru", None, None,
                                       None, polar=pl, fillers=fl, rng=r,
                                       scaffold_tag=f"{pl[0]}f{fl}"), r)
                # deviant EN surface chosen deterministically per cell:
                # spreads CD-6x across the lower bands (surface probe)
                surfaces = lib.EN_NEG_SURFACES[pl[0]]
                rng_s = random.Random(f"{seed}:CD-6x:{subj}:{pl[0]}:{fl}")
                vi = rng_s.randrange(len(surfaces))
                surf = surfaces[vi]
                pe = (pe0[0], surf, surf, surf)
                r2 = random.Random(f"{seed}:CD-6xr:{subj}:{pl[0]}:{fl}")
                add(lib.build_scenario(sid("CD-6"), "CD-6", subj, "ru", None, None,
                                       None, polar=pl, polar_en=pe,
                                       cross=True, rng=r2,
                                       scaffold_tag=f"x{pl[0]}v{vi}"), r2)
            r = random.Random(f"{seed}:CD-6L:{subj}:{pl[0]}")
            add(lib.build_scenario(sid("CD-6"), "CD-6", subj, "ru", None, None,
                                   None, polar=pl, fillers=35, rng=r,
                                   scaffold_tag=f"{pl[0]}L"), r)
        ins_pool = lib.INSERT_POOL
        n_ins = len(ins_pool)
        for combo_idx, combo in enumerate(
                [(i,) for i in range(3)]                       # single inserts
                + [(0, 1), (0, 2), (1, 2)]                     # pairs (existing)
                + [(i, (i + 1) % n_ins, (i + 3) % n_ins) for i in range(3)]   # triples
                + [tuple(sorted((i, (i + 1) % n_ins, (i + 3) % n_ins, (i + 5) % n_ins)))
                   for i in range(2)]                          # quads -> [0.95,0.97)
                + [(0, 1, 3, 4, 6), (1, 2, 4, 5, 7)]):         # quints (some <0.95)
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC-1:{subj}:{combo_idx}:{fl}")
                ins = " ".join(ins_pool[i] for i in combo)
                add(lib.build_scenario(sid("NC-1"), "NC-1", subj, "ru", None, None,
                                       None, insert_sent=ins, fillers=fl, rng=r,
                                       scaffold_tag=f"c{combo_idx}"), r)
        for li in range(2):  # long channel: 30 inserts -> >=2000 chars, b85
            combo = tuple(i % n_ins for i in range(li * 3, li * 3 + 30))
            r = random.Random(f"{seed}:NC-1L:{subj}:{li}")
            ins = " ".join(ins_pool[i] for i in combo)
            add(lib.build_scenario(sid("NC-1"), "NC-1", subj, "ru", None, None,
                                   None, insert_sent=ins, fillers=1, rng=r,
                                   scaffold_tag=f"L{li}"), r)
        for axis, c, v in (("lim", 1000, 800), ("lim", 1000, 500), ("lim", 1000, 200),
                           ("upd", "октябрь", "5 октября"), ("upd", "октябрь", "15 октября"),
                           ("timo", 120, 60), ("timo", 120, 90)):
            r = random.Random(f"{seed}:NC-2:{subj}:{axis}:{c}:{v}")
            add(lib.build_scenario(sid("NC-2"), "NC-2", subj, "ru", None, v, None,
                                   refine=(axis, c), fillers=r.choice([0, 1]), rng=r), r)
            r = random.Random(f"{seed}:NC-2L:{subj}:{axis}:{c}:{v}")
            add(lib.build_scenario(sid("NC-2"), "NC-2", subj, "ru", None, v, None,
                                   refine=(axis, c), fillers=35, rng=r,
                                   scaffold_tag="L"), r)
        for vp in (("v1.2.0", "v1.3.0"), ("v1.3.0", "v2.0.0"), ("v1.2.0", "v2.0.0")):
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC-3:{subj}:{vp[0]}:{fl}")
                add(lib.build_scenario(sid("NC-3"), "NC-3", subj, "ru", None, None,
                                       None, version_pair=vp, fillers=fl, rng=r,
                                       scaffold_tag=vp[1]), r)
            r = random.Random(f"{seed}:NC-3L:{subj}:{vp[0]}")
            add(lib.build_scenario(sid("NC-3"), "NC-3", subj, "ru", None, None,
                                   None, version_pair=vp, fillers=35, rng=r,
                                   scaffold_tag=vp[1] + "L"), r)
        for delta in ("tags", "lang", "type"):
            # type/lang deltas live in envelope fields that both-null
            # nulling would erase on BOTH sides (degenerate identity pair)
            # — same scoped exemption as the corner twins
            r = random.Random(f"{seed}:NC-4:{subj}:{delta}")
            add(lib.build_scenario(sid("NC-4"), "NC-4", subj, "ru", None, None, None,
                                   meta_delta=delta, fillers=1, rng=r),
                r, force_nn=(delta in ("type", "lang")))
            r = random.Random(f"{seed}:NC-4L:{subj}:{delta}")
            add(lib.build_scenario(sid("NC-4"), "NC-4", subj, "ru", None, None, None,
                                   meta_delta=delta, fillers=35, rng=r,
                                   scaffold_tag="L"),
                r, force_nn=(delta in ("type", "lang")))
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
            # long channel: same-language cross-domain pairs with a SHARED
            # boilerplate block (razor-safe: probe 0.988-0.993)
            if tech_a != tech_b:
                r = random.Random(f"{seed}:NC-5L:{sa}:{sb}")
                add(lib.build_scenario(sid("NC-5"), "NC-5", sa, "ru", None, None,
                                       None, sibling_subj=sb, fillers=35, rng=r), r)
    return out


def measure_all(pool: list[lib.Scenario], cos: Cos) -> None:
    for sc in pool:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)


def select_cells(pool: list[lib.Scenario], n_total: int, *,
                 cross_pos: int, cross_neg: int, corner: int,
                 prior_scaf: dict[str, int] | None = None,
                 quotas: dict[str, int] | None = None,
                 scaffold_cap: dict[str, int] | None = None,
                 long_reserve: dict[str, int] | None = None) -> dict:
    """Deterministic joint allocation: exact strata quotas; band totals
    derived; ADDENDUM-1/2 floors enforced by construction order;
    scaffold-cap + channel-balanced picking (§4.4/§4.5).

    prior_scaf: scaffold counts from the TRAIN allocation — the holdout
    call honors the union scaffold cap (§4.5 is computed on train+seal).
    quotas/scaffold_cap: v2 overrides (ADDENDUM-3 net quotas + caps on
    class totals); defaults keep the v1 behavior byte-compatible.
    """
    strata_quota = quotas or STRATA_QUOTA
    scaf_cap_tbl = scaffold_cap or SCAFFOLD_CAP
    long_tbl = long_reserve or LONG_RESERVE
    by_s: dict[str, list] = {s: [] for s in ORDER}
    for sc in pool:
        by_s[sc.stratum].append(sc)
    for s in by_s:
        by_s[s].sort(key=lambda x: (x._band or "zzz", -x._cos, x.scenario_id))

    chosen: dict[str, list] = {s: [] for s in ORDER}
    used: set[str] = set()
    scaf: dict[str, int] = dict(prior_scaf or {})
    ch_cnt: dict[str, int] = {"short": 0, "medium": 0, "long": 0}

    def take(sc: lib.Scenario):
        chosen[sc.stratum].append(sc)
        used.add(sc.scenario_id)
        scaf[sc.scaffold_id] = scaf.get(sc.scaffold_id, 0) + 1
        ch_cnt[x_key(sc)] = ch_cnt.get(x_key(sc), 0) + 1

    def x_key(sc: lib.Scenario) -> str:
        return sc.length_channel if sc.length_channel in ("short", "medium", "long") \
            else "medium"

    def cap_ok(sc: lib.Scenario) -> bool:
        return scaf.get(sc.scaffold_id, 0) < scaf_cap_tbl[sc.stratum]

    def take_spread(cands: list[lib.Scenario], k: int,
                    key=lambda x: (x._cos, x.scenario_id)) -> int:
        """Take up to k candidates minimizing scaffold concentration
        first, then by `key`; skips scaffold-capped candidates. Returns
        the number taken (fail-closed callers check it)."""
        rest = sorted(cands, key=key)
        out = 0
        for _ in range(k):
            feasible = [x for x in rest if cap_ok(x)]
            if not feasible:
                break
            best = min(feasible, key=lambda x: (scaf.get(x.scaffold_id, 0),) + key(x))
            rest.remove(best)
            take(best)
            out += 1
        return out

    def free(stratum: str, band: str | None = None, **pred):
        cands = [x for x in by_s[stratum]
                 if x.scenario_id not in used
                 and (band is None or x._band == band)
                 and all(x.__dict__.get(k) == v for k, v in pred.items())]
        return cands

    def pick_one(cands: list[lib.Scenario],
                 key=lambda x: (-x._cos, x.scenario_id)):
        """Least-filled channel, then scaffold spread, then `key`;
        scaffold-capped candidates are NOT eligible (hard cap, §4.5)."""
        feasible = [x for x in cands if cap_ok(x)]
        if not feasible:
            return None
        return min(feasible,
                   key=lambda x: (ch_cnt.get(x_key(x), 0),
                                  scaf.get(x.scaffold_id, 0)) + key(x))

    def cross_quota_left() -> tuple[int, int]:
        xp = sum(1 for s in ORDER for x in chosen[s]
                 if x.cross_lang and x.label == "contradicts")
        xn = sum(1 for s in ORDER for x in chosen[s]
                 if x.cross_lang and x.label != "contradicts")
        return max(cross_pos - xp, 0), max(cross_neg - xn, 0)

    # 1) corner twins (NC-4 type delta, non-null, cos ~1.0)
    corner_c = [x for x in by_s["NC-4"] if x.side_a.record_type != x.side_b.record_type]
    if take_spread(corner_c, corner, key=lambda x: (x.scenario_id,)) < corner:
        die(f"corner twins: only {len(corner_c)} candidates")
    # 2) razor-bin negative reservations (§4.3: >=30 negatives per bin,
    #    continuity <=0.005 inside [0.95,1.0]): spread evenly over each
    #    bin so no empty stretch opens at the zone edges
    for lo, hi in RAZOR_BINS:
        cands = [x for s in ORDER for x in free(s)
                 if x.label != "contradicts" and lo <= x._cos < hi]
        if len(cands) < RAZOR_BIN_FLOOR:
            die(f"razor bin [{lo},{hi}): only {len(cands)} negatives")
        step = (len(cands) - 1) / (RAZOR_BIN_FLOOR - 1)
        idx = [round(i * step) for i in range(RAZOR_BIN_FLOOR)]
        taken = 0
        seen = set()
        # even-spread positions first, then the rest; the scaffold cap is
        # re-checked at take time (candidates on one scaffold sit adjacent)
        order = [i for i in idx if i not in seen and not seen.add(i)]
        order += [i for i in range(len(cands)) if i not in seen]
        for i in order:
            if taken >= RAZOR_BIN_FLOOR:
                break
            if cap_ok(cands[i]):
                take(cands[i])
                taken += 1
        if taken < RAZOR_BIN_FLOOR:
            die(f"razor bin [{lo},{hi}): scaffold cap leaves {taken}/{RAZOR_BIN_FLOOR}")
    # 2b) long-channel reservation (§4.4: three length channels ~1/3
    #     each; long variants are scarce and would be outcompeted)
    for s in ORDER:
        cands = sorted([x for x in free(s) if x.length_channel == "long"],
                       key=lambda x: (x.scenario_id,))
        k = 0
        for x in cands:
            if k >= long_tbl[s]:
                break
            if cap_ok(x):
                take(x)
                k += 1
    # 2c) b40 carrier reservation (ADDENDUM-2 §B: the thin known-uncovered
    #     floor evidence — every free NC-5 pair below 0.60 is taken; they
    #     are scarce and would otherwise be outcompeted by the b55/b70
    #     fills inside the NC-5 quota)
    for x in sorted([x for x in free("NC-5") if x._band == "b40"],
                    key=lambda x: (x._cos, x.scenario_id)):
        if cap_ok(x):
            take(x)
    # 3) hard-pos floor in [0.70,0.75) (ADDENDUM-2 §A: population
    #    constraint inside the b70 band) + the cross-positive quota:
    #    the floor's 30 window positives are taken first, the quota is
    #    completed with the LOWEST-cos remaining cross positives
    #    (hardest first) — quota total stays exact (36)
    if cross_pos < FLOOR_HARD_POS:
        die("cross quota smaller than the hard floor")
    win = [x for s in ("CD-1", "CD-2", "CD-5", "CD-6")
           for x in free(s) if x.cross_lang
           and HARD_WIN_LO <= x._cos < HARD_WIN_HI]
    if len(win) < FLOOR_HARD_POS:
        die(f"floor hard-pos: only {len(win)} positives in "
            f"[{HARD_WIN_LO},{HARD_WIN_HI}) (need {FLOOR_HARD_POS}); see "
            f"data/conflict-detector/measure/window-census.json")
    if take_spread(win, FLOOR_HARD_POS) < FLOOR_HARD_POS:
        die("floor hard-pos: scaffold cap leaves no feasible window picks")
    rest = [x for s in ("CD-1", "CD-2", "CD-5", "CD-6")
            for x in free(s) if x.cross_lang]
    need_rest = cross_pos - FLOOR_HARD_POS
    if len(rest) < need_rest:
        die(f"cross-lang positives: only {len(rest) + FLOOR_HARD_POS}")
    take_spread(rest, need_rest)
    # 4) cross-language negatives (NC-5 cross, prefer lowest bands)
    cneg = [x for x in free("NC-5") if x.cross_lang]
    if len(cneg) < cross_neg:
        die(f"cross-lang negatives: only {len(cneg)}")
    take_spread(cneg, cross_neg)
    # 5) band-guided fill toward targets, scarce bands first (ascending
    #    target); cross-language quotas stay closed after their steps
    def counts():
        c = {b: 0 for b, _, _ in BANDS}
        for s in ORDER:
            for x in chosen[s]:
                if x._band is not None:
                    c[x._band] += 1
        return c

    for band in sorted(BANDS, key=lambda b: BAND_TARGETS[b[0]]):
        bname = band[0]
        while counts()[bname] < BAND_TARGETS[bname]:
            xp_left, xn_left = cross_quota_left()
            pick = None
            for s in ORDER:
                if len(chosen[s]) >= strata_quota[s]:
                    continue
                cands = [x for x in free(s, bname)
                         if not (x.cross_lang and x.label == "contradicts" and xp_left == 0)
                         and not (x.cross_lang and x.label != "contradicts" and xn_left == 0)]
                if cands:
                    pick = pick_one(cands)
                    if pick is not None:
                        break
            if pick is None:
                break
            take(pick)
    # 6) complete strata quotas from leftovers (any band)
    for s in ORDER:
        while len(chosen[s]) < strata_quota[s]:
            xp_left, xn_left = cross_quota_left()
            cands = [x for x in free(s)
                     if not (x.cross_lang and x.label == "contradicts" and xp_left == 0)
                     and not (x.cross_lang and x.label != "contradicts" and xn_left == 0)]
            if not cands:
                die(f"pool exhausted for {s}: {len(chosen[s])}/{strata_quota[s]}")
            pick = pick_one(cands, key=lambda x: (x._cos, x.scenario_id))
            if pick is None:
                die(f"scaffold cap exhausted for {s}: every remaining scaffold "
                    f"already at {scaf_cap_tbl[s]} pairs (train+seal union)")
            take(pick)
    # overflow check: never exceed quota
    for s in ORDER:
        assert len(chosen[s]) == strata_quota[s], s
    cells = {s: {} for s in ORDER}
    for s in ORDER:
        for x in chosen[s]:
            cells[s][x._band] = cells[s].get(x._band, 0) + 1
    return {"cells": cells, "ids": {s: [x.scenario_id for x in chosen[s]] for s in ORDER},
            "scaf": dict(scaf)}


# --- v2 rebuild (ADDENDUM-3, ratified 09.11) ---------------------------------

CORPUS_DIR_V2 = DATA / "corpus-v2"
PIN_V2_PATH = DATA / "pin-v2.json"
SEAL_V2_PATH = SEALS_DIR / "CD-holdout-v2.seal.json"
RUNLOG_MD = REPO / "docs" / "experiments" / "conflict-detector-train-runlog.md"

# E11: fresh seed family — != {42,777,4242} and != {43..52}
SEEDS_V2 = {"train": 2113, "hold": 7919, "sheet": 9377}
FORBIDDEN_SEEDS = {42, 777, 4242} | set(range(43, 53))
assert not (set(SEEDS_V2.values()) & FORBIDDEN_SEEDS), "seed family collides"

# §8.2 ratified NET (classic) quotas per stratum; carriers add inside the
# host stratum; class totals = v1 masses (CD-1 72, CD-2 36, CD-4 36,
# NC-2 96, NC-5 108, NC-1 66 shrunk — the donor table of E1).
STRATA_NET_V2 = {"CD-1": 56, "CD-2": 28, "CD-3": 36, "CD-4": 24, "CD-5": 24,
                 "CD-6": 36, "NC-1": 66, "NC-2": 72, "NC-3": 54, "NC-4": 36,
                 "NC-5": 84}
CARRIER_QUOTA_V2 = {"N1": {"CD-1": 16, "CD-2": 8, "CD-4": 12},
                    "N2": {"NC-2": 24}, "N3": {"NC-5": 24}}
CLASS_TOTALS_V2 = dict(STRATA_NET_V2)
for car, hosts in CARRIER_QUOTA_V2.items():
    for s, q in hosts.items():
        CLASS_TOTALS_V2[s] += q
assert sum(STRATA_NET_V2.values()) == 516 and sum(CLASS_TOTALS_V2.values()) == 600
N1_K_SPLIT = {"CD-1": (10, 6), "CD-2": (5, 3), "CD-4": (7, 5)}   # (k=1, k=2) -> 22/14 = 60/40
N2_K_SPLIT = (14, 10)
CENSUS_FACTOR = 2          # E9: razor-suitable pool >= 2x quota
N1_QUOTA, N2_QUOTA, N3_QUOTA = 36, 24, 24
# long-channel reservation scaled to the NET (donor-cut) quotas — the v1
# table is sized for the v1 quotas and would force-allocate past the NC-5
# net quota (36+10+45=91 > 84). Same share of the stratum, floor 1.
LONG_RESERVE_V2 = {s: max(1, int(LONG_RESERVE[s] * STRATA_NET_V2[s]
                                 / STRATA_QUOTA[s])) for s in ORDER}
# channel rebalance (A-indep 1/3 ± 0.05): the scaled reserve under-fills
# long once carriers (mostly short) join the population — medium drifted
# to 0.392 in the first build2 dry-run; +10 long reservations pull the
# channels back inside the tube
LONG_RESERVE_V2["NC-1"] += 3
LONG_RESERVE_V2["NC-2"] += 3
LONG_RESERVE_V2["NC-3"] += 2
LONG_RESERVE_V2["NC-4"] += 2
LONG_RESERVE_V2["CD-2"] += 1
LONG_RESERVE_V2["CD-6"] += 1
# seal shares (§4.1 formula over the §8.2 partition, largest remainder)
def _seal_shares() -> dict[str, int]:
    rows = dict(STRATA_NET_V2)
    rows["N1"], rows["N2"], rows["N3"] = N1_QUOTA, N2_QUOTA, N3_QUOTA
    raw = {s: q * HOLD_SEAL / TRAIN_PAIRS for s, q in rows.items()}
    shares = {s: int(raw[s]) for s in rows}
    left = HOLD_SEAL - sum(shares.values())
    for s in sorted(rows, key=lambda s: (-(raw[s] - int(raw[s])), s)):
        if left <= 0:
            break
        shares[s] += 1
        left -= 1
    assert left == 0 and sum(shares.values()) == HOLD_SEAL
    return shares
SEAL_SHARES_V2 = _seal_shares()
SEAL_N1_HOSTS = {"CD-1": 5, "CD-2": 3, "CD-4": 3}     # 16/8/12 x 0.3, largest remainder
SEAL_N1_K = (7, 4)
SEAL_N2_K = (4, 3)
N3_SEAL = 7
# scaffold allocation caps sized on the TRAIN+SEAL projected population
# per stratum (§4.5 union rule; the A-scaffold assert allows 0.05 of the
# union — cap 1 would starve the holdout corner/long legs on NC-4)
_SEAL_CARRIER_IN_STRATUM = {"CD-1": SEAL_N1_HOSTS["CD-1"],
                            "CD-2": SEAL_N1_HOSTS["CD-2"],
                            "CD-4": SEAL_N1_HOSTS["CD-4"],
                            "NC-2": SEAL_SHARES_V2["N2"],
                            "NC-5": N3_SEAL}
SCAFFOLD_CAP_V2 = {s: max(1, int(0.05 * (CLASS_TOTALS_V2[s]
                                         + SEAL_SHARES_V2[s]
                                         + _SEAL_CARRIER_IN_STRATUM.get(s, 0))))
                   for s in ORDER}
# E8(i) pre-declared separability rule: N1 vs N2 slices must be
# distinguishable on at least ONE cheap surface metric (rank-sum);
# indistinguishable on BOTH = overlap = BuildStop.
E8_OVERLAP_ALPHA = 0.05

NC2_GRID = [("lim", 1000, 800), ("lim", 1000, 500), ("lim", 1000, 200),
            ("upd", "октябрь", "5 октября"), ("upd", "октябрь", "15 октября"),
            ("timo", 120, 60), ("timo", 120, 90)]
DATE_SLOTS_ALL = [(lib.DATE_SLOT[0], lib.DATE_SLOT[1], lib.DATE_SLOT[2],
                   lib.DATE_SLOT[3], lib.DATE_SLOT[4]),
                  ("upd2", lib.DATE_ALT_RU, lib.DATE_ALT_EN,
                   lib.DATE_SLOT[3], lib.DATE_SLOT[4])]


def log_md(event: str, **fields) -> None:
    """Append the same JSON-line event to the COMMITTED run-log."""
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": "conflict-detector-corpus-v2-build",
                             **fields}, ensure_ascii=False) + "\n")


def gen_n1_pool(seed: int, prefix: str) -> list[lib.Scenario]:
    """N1 candidates: SAME-LANGUAGE razor-positive donors CD-1/CD-2/CD-4
    (E1: outside the cross family; razor >= 0.95 excludes the hard window
    structurally) x side in (a,b) (E7 randomization) x k in (1,2)."""
    out: list[lib.Scenario] = []
    ctr = [0]
    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-N1-{ctr[0]:05d}"
    for subj in lib.TECH:
        for si in range(len(lib.NUM_SLOTS)):
            vals = lib.NUM_SLOTS[si][3]
            for vi in range(len(vals)):
                for vj in range(len(vals)):
                    if vi >= vj:
                        continue
                    for fl in (0, 1, 2):
                        for side in ("a", "b"):
                            for k in (1, 2):
                                dseed = f"{seed}:N1d:{subj}:{si}:{vi}:{vj}:{fl}:{side}:{k}"
                                d = lib.build_scenario(
                                    sid(), "CD-1", subj, "ru", lib.NUM_SLOTS[si],
                                    vals[vi], vals[vj], fillers=fl,
                                    rng=random.Random(dseed))
                                d.carrier = "N1"
                                r = random.Random(f"{seed}:N1v:{subj}:{si}:{vi}:{vj}:{fl}:{side}:{k}")
                                try:
                                    out.append(lib.apply_benign(d, side, k, r))
                                except ValueError:
                                    pass  # dictionary cannot host k here (deterministic unfit)
        for key, ru, en, vals_ru, _vals_en in DATE_SLOTS_ALL:
            for vi in range(len(vals_ru)):
                for vj in range(len(vals_ru)):
                    if vi >= vj:
                        continue
                    for fl in (0, 1, 2):
                        for side in ("a", "b"):
                            for k in (1, 2):
                                d = lib.build_scenario(
                                    sid(), "CD-2", subj, "ru", (key, ru, en),
                                    vals_ru[vi], vals_ru[vj], fillers=fl,
                                    scaffold_tag="n1",
                                    rng=random.Random(f"{seed}:N1d2:{subj}:{key}:{vi}:{vj}:{fl}:{side}:{k}"))
                                d.carrier = "N1"
                                r = random.Random(f"{seed}:N1v2:{subj}:{key}:{vi}:{vj}:{fl}:{side}:{k}")
                                try:
                                    out.append(lib.apply_benign(d, side, k, r))
                                except ValueError:
                                    pass
        for sd in lib.MODAL_SLOTS:
            for fl in (0, 1, 2):
                for side in ("a", "b"):
                    for k in (1, 2):
                        d = lib.build_scenario(
                            sid(), "CD-4", subj, "ru", sd, None, None,
                            fillers=fl, scaffold_tag="n1",
                            rng=random.Random(f"{seed}:N1d4:{subj}:{sd[0]}:{fl}:{side}:{k}"))
                        d.carrier = "N1"
                        r = random.Random(f"{seed}:N1v4:{subj}:{sd[0]}:{fl}:{side}:{k}")
                        try:
                            out.append(lib.apply_benign(d, side, k, r))
                        except ValueError:
                            pass
    return out


def gen_n2_pool(seed: int, prefix: str) -> list[lib.Scenario]:
    """N2 candidates: NC-2 refinement donors (same grid as classic NC-2)
    + the SAME benign dial — the anti-proxy control (same dictionary,
    same k-distribution, NOT label)."""
    out: list[lib.Scenario] = []
    ctr = [0]
    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-N2-{ctr[0]:05d}"
    for subj in lib.TECH:
        for axis, c, v in NC2_GRID:
            for fl in (0, 1):
                for side in ("a", "b"):
                    for k in (1, 2):
                        d = lib.build_scenario(
                            sid(), "NC-2", subj, "ru", None, v, None,
                            refine=(axis, c), fillers=fl, scaffold_tag="n2",
                            rng=random.Random(f"{seed}:N2d:{subj}:{axis}:{c}:{fl}:{side}:{k}"))
                        d.carrier = "N2"
                        r = random.Random(f"{seed}:N2v:{subj}:{axis}:{c}:{fl}:{side}:{k}")
                        try:
                            out.append(lib.apply_benign(d, side, k, r))
                        except ValueError:
                            pass
    return out


def gen_n3_pool(seed: int, prefix: str) -> list[lib.Scenario]:
    """N3 candidates: true translation — cross-language pair, same subject,
    canonical slot equality (numeric same value; date same index), op=none."""
    out: list[lib.Scenario] = []
    ctr = [0]
    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-N3-{ctr[0]:05d}"
    for subj in lib.TECH:
        for si, sd in enumerate(lib.NUM_SLOTS):
            for ci in range(len(sd[3])):
                out.append(lib.build_n3(
                    sid(), subj, lib.NUM_SLOTS[si], ci, fillers=(0 if ci == 0 else 1),
                    scaffold_tag=f"v{ci}"))
        for key, ru, en, vals_ru, vals_en in DATE_SLOTS_ALL:
            for ci in range(len(vals_ru)):
                out.append(lib.build_n3(
                    sid(), subj, (key, ru, en, vals_ru, vals_en), ci,
                    fillers=(0 if ci == 0 else 1), scaffold_tag=f"v{ci}"))
    return out


def carrier_suitable(sc: lib.Scenario) -> bool:
    """Razor-suitability for the E9 census: mechanical pass, razor band,
    channel-consistent, non-cross (E1 donor constraints)."""
    return (lib.mechanical_ok(sc) and sc._band == "razor"
            and lib.len_channel_ok(sc) and not sc.cross_lang)


def select_carriers(n1: list, n2: list, n3: list, *,
                    scaf: dict[str, int], log_fn=print) -> dict:
    """Deterministic carrier allocation per the ratified pins: N1 hosts
    16/8/12 with k-mix 60/40 per host, sides balanced; N2 24 with k
    14/10; N3 24. Scaffold caps are UNION-aware (classic counts come in
    via `scaf`). Returns {carrier: [scenarios]}."""
    def cap_ok(sc):
        return scaf.get(sc.scaffold_id, 0) < SCAFFOLD_CAP_V2[sc.stratum]

    def take(want: int, want_k: dict[int, int], pool: list,
             label: str) -> list:
        picked: list = []
        picked_ids: set[str] = set()
        for k, have_k in sorted(want_k.items()):
            # E7 side randomization with a deterministic 50/50 plan per
            # host-k cell: the top-of-list bias would otherwise drift the
            # sides (pool ids interleave a-then-b at equal cos)
            want_a = (have_k + 1) // 2
            want_b = have_k - want_a
            for side, quota_side in (("a", want_a), ("b", want_b)):
                cands = sorted([x for x in pool
                                if len(x.benign) == k
                                and x.benign[0]["side"] == side
                                and x.scenario_id not in picked_ids],
                               key=lambda x: (scaf.get(x.scaffold_id, 0), -x._cos,
                                              x.scenario_id))
                got = 0
                for x in cands:
                    if got >= quota_side:
                        break
                    if not cap_ok(x):   # re-check: counts move mid-list
                        continue
                    picked.append(x)
                    picked_ids.add(x.scenario_id)
                    scaf[x.scaffold_id] = scaf.get(x.scaffold_id, 0) + 1
                    got += 1
                if got < quota_side:
                    die(f"{label}: k={k} side-{side} exhausted ({got}/{quota_side})")
        if len(picked) != want:
            die(f"{label}: {len(picked)}/{want}")
        return picked

    sel: dict[str, list] = {}
    n1_pick: list = []
    for host, want in CARRIER_QUOTA_V2["N1"].items():
        k1, k2 = N1_K_SPLIT[host]
        got = take(want, {1: k1, 2: k2},
                   [x for x in n1 if x.stratum == host and carrier_suitable(x)],
                   f"N1/{host}")
        n1_pick += got
    # E7 side balance: |p(a) - 0.5| <= 0.1 per carrier kind
    def balance(cells: list, name: str) -> list:
        na = sum(1 for x in cells if x.benign[0]["side"] == "a")
        if abs(na / len(cells) - 0.5) > 0.1:
            die(f"{name}: side balance broken (a={na}/{len(cells)})")
        return cells
    sel["N1"] = balance(n1_pick, "N1")
    sel["N2"] = balance(
        take(N2_QUOTA, {1: N2_K_SPLIT[0], 2: N2_K_SPLIT[1]},
             [x for x in n2 if carrier_suitable(x)], "N2"), "N2")
    n3_cands = sorted([x for x in n3 if lib.mechanical_ok(x)],
                      key=lambda x: (scaf.get(x.scaffold_id, 0), -x._cos,
                                     x.scenario_id))
    got = 0
    sel["N3"] = []
    for x in n3_cands:
        if got >= N3_QUOTA:
            break
        if cap_ok(x):
            sel["N3"].append(x)
            scaf[x.scaffold_id] = scaf.get(x.scaffold_id, 0) + 1
            got += 1
    if got < N3_QUOTA:
        die(f"N3: pool exhausted ({got}/{N3_QUOTA})")
    return sel


def cmd_measure2() -> int:
    """Census E9 -> pins -> full measure/allocation -> pin-v2.json."""
    t0 = time.time()
    cos = Cos()
    # E11 BEFORE ANY GENERATION: the seed-family event
    log("seeds_pinned_v2", seeds=SEEDS_V2, forbidden="42/777/4242 + 43..52",
        family="fresh", check="disjoint-by-construction")
    log_md("seeds_pinned", seeds=SEEDS_V2, note="fresh family, != v1 {42,777,4242}, != {43..52}",
           forbidden_sets=sorted(FORBIDDEN_SEEDS))
    print(f"seeds v2 pinned: {SEEDS_V2}")

    print("== census E9: carrier pools (dry measure) ==")
    n1_pool = gen_n1_pool(SEEDS_V2["train"], "tr")
    n2_pool = gen_n2_pool(SEEDS_V2["train"], "tr")
    n3_pool = gen_n3_pool(SEEDS_V2["train"], "tr")
    for sc in n1_pool + n2_pool + n3_pool:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)
    cen_n1 = [x for x in n1_pool if carrier_suitable(x)]
    cen_n2 = [x for x in n2_pool if carrier_suitable(x)]
    cen = {
        "n1": {"pool": len(n1_pool), "suitable": len(cen_n1),
               "by_host": {h: sum(1 for x in cen_n1 if x.stratum == h)
                           for h in ("CD-1", "CD-2", "CD-4")},
               "by_k": {k: sum(1 for x in cen_n1 if len(x.benign) == k)
                        for k in (1, 2)}},
        "n2": {"pool": len(n2_pool), "suitable": len(cen_n2),
               "by_k": {k: sum(1 for x in cen_n2 if len(x.benign) == k)
                        for k in (1, 2)}},
        "n3": {"pool": len(n3_pool),
               "mech_ok": sum(1 for x in n3_pool if lib.mechanical_ok(x))},
    }
    print("census:", json.dumps(cen))
    # E9 gate: >= 2x quota of razor-suitable candidates, else BuildStop
    need_n1 = {"CD-1": 32, "CD-2": 16, "CD-4": 24}
    red = []
    if len(cen_n1) < CENSUS_FACTOR * N1_QUOTA:
        red.append(f"N1 suitable {len(cen_n1)} < {CENSUS_FACTOR * N1_QUOTA}")
    if len(cen_n2) < CENSUS_FACTOR * N2_QUOTA:
        red.append(f"N2 suitable {len(cen_n2)} < {CENSUS_FACTOR * N2_QUOTA}")
    for h, need in need_n1.items():
        if cen["n1"]["by_host"][h] < need:
            red.append(f"N1/{h} {cen['n1']['by_host'][h]} < {need}")
    for k, need in ((1, N1_K_SPLIT['CD-1'][0] + N1_K_SPLIT['CD-2'][0] + N1_K_SPLIT['CD-4'][0]),
                    (2, N1_K_SPLIT['CD-1'][1] + N1_K_SPLIT['CD-2'][1] + N1_K_SPLIT['CD-4'][1])):
        if cen["n1"]["by_k"][k] < need:
            red.append(f"N1 k={k} {cen['n1']['by_k'][k]} < {need}")
    for k, need in zip((1, 2), N2_K_SPLIT):
        if cen["n2"]["by_k"][k] < need:
            red.append(f"N2 k={k} {cen['n2']['by_k'][k]} < {need}")
    if red:
        die("census E9 RED: " + "; ".join(red))
    log("census_v2_pass", **cen)
    log_md("census_v2", verdict="pass", n1_suitable=cen["n1"]["suitable"],
           n2_suitable=cen["n2"]["suitable"],
           pins_after={"k_mix": "60/40 (N1 22/14, N2 14/10)",
                       "host_split": "16/8/12", "bands": "measure->fix below"})
    print("census E9 PASS: pins after census -> k-mix 60/40, host split 16/8/12")

    print("== classic pools (fresh v2 seeds) ==")
    train_pool = gen_pool(SEEDS_V2["train"], "tr", rotate=False)
    hold_pool = gen_pool(SEEDS_V2["hold"], "ho", rotate=True)
    print(f"train pool: {len(train_pool)}, holdout pool: {len(hold_pool)}, "
          f"carriers: N1 {len(n1_pool)} N2 {len(n2_pool)} N3 {len(n3_pool)}")
    measure_all(train_pool, cos)
    measure_all(hold_pool, cos)
    for sc in n1_pool + n2_pool + n3_pool:
        if not hasattr(sc, "_cos"):
            sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
            sc._band = band_of(sc._cos)
    print(f"embedded {cos.n} unique texts in {time.time()-t0:.0f}s")

    print("== allocation: classic net (516) ==")
    alloc_tr = select_cells(train_pool, sum(STRATA_NET_V2.values()),
                            cross_pos=CROSS_POS_QUOTA, cross_neg=CROSS_NEG_CAP,
                            corner=12, quotas=STRATA_NET_V2,
                            scaffold_cap=SCAFFOLD_CAP_V2,
                            long_reserve=LONG_RESERVE_V2)
    scaf_union = dict(alloc_tr["scaf"])
    print("== allocation: carriers (84) ==")
    carriers = select_carriers(n1_pool, n2_pool, n3_pool, scaf=scaf_union)
    tr_sel_ids = {i for s in ORDER for i in alloc_tr["ids"][s]}
    tr_sel_ids |= {x.scenario_id for c in ("N1", "N2", "N3") for x in carriers[c]}
    print(f"train total: {len(tr_sel_ids)}")

    # joint razor population: parity + bins + hard window (E1/NA-2 recheck)
    all_tr = ([x for s in ORDER for x in train_pool if x.scenario_id in alloc_tr["ids"][s]]
              + [x for c in ("N1", "N2", "N3") for x in carriers[c]])
    rz = [x for x in all_tr if x._band == "razor"]
    rz_pos = sum(1 for x in rz if x.label == "contradicts")
    rz_neg = len(rz) - rz_pos
    bins = Counter()
    for x in rz:
        if x.label != "contradicts":
            c = x._cos
            bins["0" if c < 0.97 else ("1" if c < 0.985 else "2")] += 1
    hard_pos = sum(1 for x in all_tr if x.label == "contradicts"
                   and HARD_WIN_LO <= x._cos < HARD_WIN_HI)
    ok_floors = (len(rz) >= FLOOR_RAZOR and rz_neg >= rz_pos
                 and min(bins.values()) >= RAZOR_BIN_FLOOR
                 and hard_pos >= FLOOR_HARD_POS)
    print(f"floors v2: razor {len(rz)} (pos {rz_pos}/neg {rz_neg}), "
          f"bins {dict(bins)}, hard-pos {hard_pos}")
    if not ok_floors:
        die(f"floors RED on joint population: razor {len(rz)}, pos {rz_pos}, "
            f"neg {rz_neg}, bins {dict(bins)}, hard {hard_pos}")

    # holdout: same procedure, fresh seeds, union scaffold caps
    print("== allocation: holdout pool (registered 600 -> seal 180) ==")
    hn1 = gen_n1_pool(SEEDS_V2["hold"], "ho")
    hn2 = gen_n2_pool(SEEDS_V2["hold"], "ho")
    hn3 = gen_n3_pool(SEEDS_V2["hold"], "ho")
    for sc in hn1 + hn2 + hn3:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)
    alloc_ho = select_cells(hold_pool, sum(STRATA_NET_V2.values()),
                            cross_pos=36, cross_neg=45, corner=12,
                            prior_scaf=scaf_union, quotas=STRATA_NET_V2,
                            scaffold_cap=SCAFFOLD_CAP_V2,
                            long_reserve=LONG_RESERVE_V2)
    scaf_ho = dict(alloc_ho["scaf"])
    carriers_ho = select_carriers(hn1, hn2, hn3, scaf=scaf_ho)
    ho_reg_ids = ([i for s in ORDER for i in alloc_ho["ids"][s]]
                  + [x.scenario_id for c in ("N1", "N2", "N3") for x in carriers_ho[c]])

    # seal 180: share table (largest remainder over the §8.2 partition),
    # culled for train-text disjointness (A-disjoint)
    tr_by_id = {x.scenario_id: x for s in ORDER for x in train_pool}
    tr_by_id.update({x.scenario_id: x for c in ("N1", "N2", "N3") for x in carriers[c]})
    ho_by_id = {x.scenario_id: x for s in ORDER for x in hold_pool}
    ho_by_id.update({x.scenario_id: x for c in ("N1", "N2", "N3")
                     for x in (carriers_ho["N1"] + carriers_ho["N2"] + carriers_ho["N3"])})
    tr_texts = {lib.sha(tr_by_id[i].text_a()) for i in tr_sel_ids} | \
               {lib.sha(tr_by_id[i].text_b()) for i in tr_sel_ids}
    tr_hash_owner: dict[str, str] = {}
    for i in tr_sel_ids:
        tr_hash_owner.setdefault(lib.sha(tr_by_id[i].text_a()), i)
        tr_hash_owner.setdefault(lib.sha(tr_by_id[i].text_b()), i)
    seal_ids: list[str] = []
    culled = 0
    cull_diag: dict[str, str] = {}
    want_classic = {s: SEAL_SHARES_V2[s] for s in ORDER}
    want_n1 = SEAL_N1_HOSTS
    remaining_n1 = SEAL_N1_K[0] + SEAL_N1_K[1]
    for s in ORDER:
        got = 0
        for i in alloc_ho["ids"][s]:
            if got >= want_classic[s]:
                break
            if i in tr_sel_ids:
                cull_diag.setdefault(i, "scenario_id in train")
                culled += 1
                continue
            ha, hb = lib.sha(ho_by_id[i].text_a()), lib.sha(ho_by_id[i].text_b())
            if ha in tr_texts or hb in tr_texts:
                cull_diag.setdefault(i, f"text == {tr_hash_owner.get(ha) or tr_hash_owner.get(hb)}")
                culled += 1
                continue
            seal_ids.append(i)
            got += 1
        if got < want_classic[s]:
            examples = {k: cull_diag[k] for k in list(cull_diag)[:5]}
            comp = Counter(len(_sents(ho_by_id[i].side_a.body))
                           for i in alloc_ho["ids"][s])
            die(f"seal disjointness: classic {s} exhausted ({got}/{want_classic[s]}); "
                f"registered a-side sentence-count hist {dict(comp)}; "
                f"cull examples {examples}")
    n1_ho = sorted([x for x in carriers_ho["N1"] if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    per_host = {h: 0 for h in want_n1}
    for x in n1_ho:
        if per_host[x.stratum] >= want_n1[x.stratum] or remaining_n1 <= 0:
            continue
        if scaf_ho.get(x.scaffold_id, 0) >= SCAFFOLD_CAP_V2[x.stratum]:
            continue
        if x.scenario_id in tr_sel_ids or lib.sha(x.text_a()) in tr_texts \
                or lib.sha(x.text_b()) in tr_texts:
            culled += 1
            continue
        seal_ids.append(x.scenario_id)
        scaf_ho[x.scaffold_id] = scaf_ho.get(x.scaffold_id, 0) + 1
        per_host[x.stratum] += 1
        remaining_n1 -= 1
    if remaining_n1 > 0:
        die(f"seal disjointness: N1 exhausted ({SEAL_N1_K[0]+SEAL_N1_K[1]-remaining_n1}/11)")
    n2_ho = sorted([x for x in carriers_ho["N2"] if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    got_n2 = 0
    for x in n2_ho:
        if got_n2 >= SEAL_SHARES_V2["N2"]:
            break
        if scaf_ho.get(x.scaffold_id, 0) >= SCAFFOLD_CAP_V2[x.stratum]:
            continue
        if x.scenario_id in tr_sel_ids or lib.sha(x.text_a()) in tr_texts \
                or lib.sha(x.text_b()) in tr_texts:
            culled += 1
            continue
        seal_ids.append(x.scenario_id)
        scaf_ho[x.scaffold_id] = scaf_ho.get(x.scaffold_id, 0) + 1
        got_n2 += 1
    if got_n2 < SEAL_SHARES_V2["N2"]:
        die(f"seal disjointness: N2 exhausted ({got_n2}/{SEAL_SHARES_V2['N2']})")
    n3_ho = sorted([x for x in carriers_ho["N3"] if lib.mechanical_ok(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    got_n3 = 0
    for x in n3_ho:
        if got_n3 >= N3_SEAL:
            break
        if scaf_ho.get(x.scaffold_id, 0) >= SCAFFOLD_CAP_V2[x.stratum]:
            continue
        if x.scenario_id in tr_sel_ids or lib.sha(x.text_a()) in tr_texts \
                or lib.sha(x.text_b()) in tr_texts:
            culled += 1
            continue
        seal_ids.append(x.scenario_id)
        scaf_ho[x.scaffold_id] = scaf_ho.get(x.scaffold_id, 0) + 1
        got_n3 += 1
    if got_n3 < N3_SEAL:
        die(f"seal disjointness: N3 exhausted ({got_n3}/{N3_SEAL})")
    assert len(seal_ids) == HOLD_SEAL, len(seal_ids)
    print(f"seal: {len(seal_ids)} pairs ({culled} culled for train-text collisions)")

    corpus_floor = min([x._cos for x in all_tr] + [ho_by_id[i]._cos for i in seal_ids])
    pin = {
        "ts": now(), "version": 2,
        "instrument": {"pin": lib.EMBEDDER_PIN, "guard_version": lib.GUARD_VERSION},
        "measured_corpus_floor": corpus_floor,
        "seeds": dict(SEEDS_V2),
        "seed_family_check": "fresh; disjoint from {42,777,4242} and {43..52}",
        "hard_window": [HARD_WIN_LO, HARD_WIN_HI],
        "razor_pos_quota": rz_pos,
        "census": cen,
        "pins_after_census": {"k_mix": "60/40", "host_split": "16/8/12"},
        "classic": alloc_tr,
        "carriers": {c: [x.scenario_id for x in carriers[c]]
                     for c in ("N1", "N2", "N3")},
        "carrier_cells": {c: [{"id": x.scenario_id, "cos": x._cos,
                               "k": len(x.benign),
                               "side": (x.benign[0]["side"] if x.benign else None)}
                              for x in carriers[c]]
                          for c in ("N1", "N2", "N3")},
        "holdout_classic": alloc_ho,
        "holdout_carriers": {c: [x.scenario_id for x in carriers_ho[c]]
                             for c in ("N1", "N2", "N3")},
        "holdout_pool_ids": ho_reg_ids,
        "seal_ids": seal_ids,
        "seal_shares": {**{s: SEAL_SHARES_V2[s] for s in ORDER},
                        "N1": sum(SEAL_N1_HOSTS.values()), "N2": SEAL_SHARES_V2["N2"],
                        "N3": N3_SEAL},
        "bands_train": {b: sum(1 for x in all_tr if x._band == b)
                        for b, _, _ in BANDS},
        "floors": {"razor_total": len(rz), "razor_pos": rz_pos, "razor_neg": rz_neg,
                   "bins": dict(bins), "hard_pos": hard_pos, "ok": ok_floors},
    }
    DATA.mkdir(parents=True, exist_ok=True)
    PIN_V2_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log("measure2_pinned", ok=True, razor=len(rz), razor_pos=rz_pos,
        hard_pos=hard_pos, floor=corpus_floor,
        pin_sha=lib.sha(PIN_V2_PATH.read_text(encoding="utf-8"))[:16])
    log_md("measure_v2_pinned", razor=len(rz), razor_pos=rz_pos,
           razor_neg=rz_neg, hard_pos=hard_pos, bins=dict(bins),
           corpus_floor=corpus_floor, n1=len(carriers["N1"]),
           n2=len(carriers["N2"]), n3=len(carriers["N3"]),
           seal=len(seal_ids))
    print(f"\nMEASURE2: PINNED -> {PIN_V2_PATH} ({time.time()-t0:.0f}s)")
    return 0


def cmd_build2() -> int:
    """Generate v2 per pin-v2.json; FULL assert map (§5.3 + E1-E8);
    stop-first: any red = BuildStop, NOTHING is written or sealed."""
    t0 = time.time()
    if not PIN_V2_PATH.exists():
        die("activation assert: pin-v2.json missing — run measure2 first")
    pin = json.loads(PIN_V2_PATH.read_text(encoding="utf-8"))
    log("build2_start", pin_sha=lib.sha(PIN_V2_PATH.read_text(encoding="utf-8"))[:16])
    cos = Cos()
    train_pool = gen_pool(SEEDS_V2["train"], "tr", rotate=False)
    hold_pool = gen_pool(SEEDS_V2["hold"], "ho", rotate=True)
    n1_pool = gen_n1_pool(SEEDS_V2["train"], "tr")
    n2_pool = gen_n2_pool(SEEDS_V2["train"], "tr")
    n3_pool = gen_n3_pool(SEEDS_V2["train"], "tr")
    hn1 = gen_n1_pool(SEEDS_V2["hold"], "ho")
    hn2 = gen_n2_pool(SEEDS_V2["hold"], "ho")
    hn3 = gen_n3_pool(SEEDS_V2["hold"], "ho")
    measure_all(train_pool, cos)
    measure_all(hold_pool, cos)
    for sc in n1_pool + n2_pool + n3_pool + hn1 + hn2 + hn3:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)
    by_id: dict[str, lib.Scenario] = {}
    for pool in (train_pool, hold_pool, n1_pool, n2_pool, n3_pool, hn1, hn2, hn3):
        for x in pool:
            by_id[x.scenario_id] = x
    tr_classic = [by_id[i] for s in ORDER for i in pin["classic"]["ids"][s]]
    tr_carriers = {c: [by_id[i] for i in pin["carriers"][c]]
                   for c in ("N1", "N2", "N3")}
    tr_sel = tr_classic + [x for c in ("N1", "N2", "N3") for x in tr_carriers[c]]
    assert len(tr_sel) == TRAIN_PAIRS, len(tr_sel)
    seal_ids = pin["seal_ids"]
    pool_ids = pin["holdout_pool_ids"]
    ho_sel = [by_id[i] for i in seal_ids]
    remainder = [i for i in pool_ids if i not in set(seal_ids)]
    assert len(ho_sel) == HOLD_SEAL
    all_sc = tr_sel + ho_sel

    print("== asserts (design 5.3 + ADDENDUM-3 E1-E8), stop-first ==")
    A: dict[str, dict] = {}
    def rec(name, ok, detail):
        A[name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'RED'}  {name}: {detail}")
        return bool(ok)

    from collections import Counter as _C
    tr_cnt = _C(x.stratum for x in tr_sel)
    ho_cnt = _C(x.stratum for x in ho_sel)
    car_cnt = {c: len(tr_carriers[c]) for c in ("N1", "N2", "N3")}
    class_ok = all(tr_cnt[s] == CLASS_TOTALS_V2[s] for s in ORDER)
    net_ok = all(
        tr_cnt[s] - sum(q for h, qs in (("N1", CARRIER_QUOTA_V2["N1"]),
                                        ("N2", CARRIER_QUOTA_V2["N2"]),
                                        ("N3", CARRIER_QUOTA_V2["N3"]))
                        for hs, q in qs.items() if hs == s) == STRATA_NET_V2[s]
        for s in ORDER)
    host_ok = all(
        sum(1 for x in tr_carriers["N1"] if x.stratum == h) == q
        for h, q in CARRIER_QUOTA_V2["N1"].items())
    k_ok = (
        sorted(len(x.benign) for x in tr_carriers["N1"]).count(1) == 22
        and sorted(len(x.benign) for x in tr_carriers["N1"]).count(2) == 14
        and sorted(len(x.benign) for x in tr_carriers["N2"]).count(1) == 14
        and sorted(len(x.benign) for x in tr_carriers["N2"]).count(2) == 10)
    rec("A-count", class_ok and net_ok and host_ok and k_ok
        and car_cnt == {"N1": 36, "N2": 24, "N3": 24}
        and sum(ho_cnt.values()) == HOLD_SEAL,
        {"train_class": dict(tr_cnt), "carriers": car_cnt,
         "seal_total": sum(ho_cnt.values()), "seal": dict(ho_cnt)})
    # A-silver (carrier-aware) + E4 len-channel on every carrier cell
    bad = [x.scenario_id for x in all_sc if not lib.mechanical_ok(x)]
    len_bad = [x.scenario_id for x in all_sc
               if x.carrier in ("N1", "N2") and not lib.len_channel_ok(x)]
    rec("A-silver+len4", not bad and not len_bad,
        f"{len(all_sc)-len(bad)}/{len(all_sc)} mechanical pass; "
        f"E4 len-channel violations {len(len_bad)}")
    badl = [x.scenario_id for x in all_sc
            if x.label != ("contradicts" if x.stratum.startswith("CD")
                           else "not-contradicts")]
    rec("A-labels", not badl, f"bad={len(badl)}")
    n_rows = 2 * len(tr_sel) + 2 * len(ho_sel)
    rec("A-symmetry", n_rows == 2 * len(all_sc), f"{n_rows} == 2x{len(all_sc)}")
    g_nc3 = sum(1 for x in all_sc if x.stratum == "NC-3" and lib.version_guard(x))
    g_cd = sum(1 for x in all_sc if x.stratum.startswith("CD")
               and lib.version_guard(x))
    n_nc3 = sum(1 for x in all_sc if x.stratum == "NC-3")
    rec("A-guard", g_nc3 == n_nc3 and g_cd == 0,
        f"nc3 {g_nc3}/{n_nc3}, cd {g_cd}/0")
    corner = [x for x in all_sc if x._cos >= 0.999
              and lib.pair_char5_jaccard(x.side_a, x.side_b, x._cos) == 1.0]
    rec("A-corner", len(corner) >= CORNER_MIN
        and all(not x.stratum.startswith("CD") for x in corner),
        f"{len(corner)} twins")
    tr_bands = _C(x._band for x in tr_sel)
    rz = [x for x in tr_sel if x._band == "razor"]
    rz_pos = sum(1 for x in rz if x.label == "contradicts")
    rz_neg = len(rz) - rz_pos
    bins = _C()
    for x in rz:
        if x.label != "contradicts":
            c = x._cos
            bins["0" if c < 0.97 else ("1" if c < 0.985 else "2")] += 1
    rz_sorted = sorted(x._cos for x in rz)
    max_gap = max((b - a for a, b in zip(rz_sorted, rz_sorted[1:])), default=0.0)
    hard_pos = sum(1 for x in tr_sel if x.label == "contradicts"
                   and HARD_WIN_LO <= x._cos < HARD_WIN_HI)
    quota_pin = pin["razor_pos_quota"]
    ok_b = (len(rz) >= FLOOR_RAZOR and rz_neg >= rz_pos
            and min(bins.values()) >= RAZOR_BIN_FLOOR
            and max_gap <= RAZOR_MAX_GAP and rz_pos == quota_pin
            and hard_pos >= FLOOR_HARD_POS)
    rec("A-band-coverage", bool(ok_b),
        f"razor {len(rz)} (pos {rz_pos}/neg {rz_neg}, quota pin {quota_pin}), "
        f"bins {dict(bins)}, max-gap {max_gap:.4f}, hard-pos {hard_pos}")
    tr_ids = {x.scenario_id for x in tr_sel}
    tr_texts = {lib.sha(x.text_a()) for x in tr_sel} | \
               {lib.sha(x.text_b()) for x in tr_sel}
    ho_clash = [x.scenario_id for x in ho_sel
                if x.scenario_id in tr_ids
                or lib.sha(x.text_a()) in tr_texts
                or lib.sha(x.text_b()) in tr_texts]
    rem_clash = set(remainder) & tr_ids
    rec("A-disjoint", not ho_clash and not rem_clash,
        f"train-vs-seal clashes {len(ho_clash)}, remainder-in-train {len(rem_clash)}")
    n4_corner_ids = {x.scenario_id for x in all_sc
                     if x.stratum == "NC-4"
                     and x.side_a.record_type != x.side_b.record_type}
    n3_ids = {x.scenario_id for x in all_sc if x.carrier == "N3"}
    exempt = n4_corner_ids | n3_ids
    pairs_null = sum(1 for x in all_sc if x.both_null and x.scenario_id not in exempt)
    mixed = sum(1 for x in all_sc
                if (x.side_a.created_at is None) != (x.side_b.created_at is None)
                and x.scenario_id not in exempt)
    null_share = pairs_null / (len(all_sc) - len(exempt))
    rec("A-null", abs(null_share - NULL_P) <= 0.05 and mixed == 0,
        f"share {null_share:.3f} (exempt {len(exempt)}: N4-corner+N3), mixed {mixed}")
    xp = sum(1 for x in tr_sel if x.cross_lang and x.label == "contradicts")
    xn = sum(1 for x in tr_sel if x.cross_lang and x.label != "contradicts")
    rec("A-crosslang-cap", xp == CROSS_POS_QUOTA and xn <= 2 * CROSS_NEG_CAP
        and xn / 360 <= 0.5,
        f"pos {xp}/{CROSS_POS_QUOTA}, neg {xn} (classic cap {CROSS_NEG_CAP} + N3 24), "
        f"neg share {xn/360:.3f} <= 1/2")
    older = sum(1 for x in tr_sel
                if x.side_a.created_at and x.side_b.created_at
                and x.side_a.created_at < x.side_b.created_at)
    dated = sum(1 for x in tr_sel if x.side_a.created_at and x.side_b.created_at)
    p_older = older / max(dated, 1)
    lens = _C(x.length_channel for x in tr_sel)
    len_ok = all(abs(lens[c] / TRAIN_PAIRS - 1 / 3) <= 0.05
                 for c in ("short", "medium", "long") if lens.get(c, 0) >= 30)
    side_a_n1 = sum(1 for x in tr_carriers["N1"] if x.benign[0]["side"] == "a")
    side_a_n2 = sum(1 for x in tr_carriers["N2"] if x.benign[0]["side"] == "a")
    rec("A-indep+side", abs(p_older - 0.5) <= 0.10 and len_ok
        and abs(side_a_n1 / 36 - 0.5) <= 0.1 and abs(side_a_n2 / 24 - 0.5) <= 0.1,
        f"P(older) {p_older:.3f}, channels {dict(lens)}, N1 side-a {side_a_n1}/36, "
        f"N2 side-a {side_a_n2}/24")
    scaf_ok, scaf_detail = True, {}
    for s in ORDER:
        cls = [x for x in all_sc if x.stratum == s]
        ids = {x.scaffold_id for x in cls}
        mx = max(_C(x.scaffold_id for x in cls).values()) / len(cls)
        scaf_ok &= len(ids) >= 20 and mx <= lib.CAP + 1e-9
        scaf_detail[s] = f"{len(ids)} scaffolds, max share {mx:.3f}"
    rec("A-scaffold", scaf_ok, scaf_detail)
    # E2: dictionary sha pin (two-level guarantee level 1 is the import assert)
    rec("A-para-dict", True, {"ru_sha256": lib.PARA_DICT_SHA["ru"],
                              "en_sha256": lib.PARA_DICT_SHA["en"],
                              "note": "import-time FACT_ALPHA whitelist asserted; "
                                      "per-pair no-digit/no-fact checked in A-silver"})
    # E8(i): N1 vs N2 separability on the cheap surface (cos, char5)
    def mwu(xs, ys):
        import math
        n1, n2 = len(xs), len(ys)
        tagged = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
        ranks = [0.0] * len(tagged)
        i = 0
        while i < len(tagged):
            j = i
            while j + 1 < len(tagged) and tagged[j + 1][0] == tagged[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                ranks[t] = avg
            i = j + 1
        r1 = sum(r for r, (_v, g) in zip(ranks, tagged) if g == 0)
        u1 = r1 - n1 * (n1 + 1) / 2
        mu = n1 * n2 / 2
        tie = sum(t ** 3 - t for t in _C(v for v, _ in tagged).values())
        n = len(tagged)
        sd = math.sqrt((n1 * n2 / 12) * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0.0
        if sd == 0:
            return 1.0, u1
        z = (u1 - mu) / sd
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return p, round(u1, 1)
    n1_cells = tr_carriers["N1"]
    n2_cells = tr_carriers["N2"]
    n1_cos = [x._cos for x in n1_cells]
    n2_cos = [x._cos for x in n2_cells]
    n1_ch5 = [round(lib.pair_char5_jaccard(x.side_a, x.side_b, x._cos), 4)
              for x in n1_cells]
    n2_ch5 = [round(lib.pair_char5_jaccard(x.side_a, x.side_b, x._cos), 4)
              for x in n2_cells]
    p_cos, u_cos = mwu(n1_cos, n2_cos)
    p_ch5, u_ch5 = mwu(n1_ch5, n2_ch5)
    overlap = p_cos >= E8_OVERLAP_ALPHA and p_ch5 >= E8_OVERLAP_ALPHA
    rec("A-carrier-sep(E8i)", not overlap,
        {"n1_cos": [min(n1_cos), max(n1_cos)], "n2_cos": [min(n2_cos), max(n2_cos)],
         "n1_char5": [min(n1_ch5), max(n1_ch5)],
         "n2_char5": [min(n2_ch5), max(n2_ch5)],
         "mwu_p_cos": round(p_cos, 5), "mwu_p_char5": round(p_ch5, 5),
         "rule": f"overlap = p_cos>= {E8_OVERLAP_ALPHA} AND p_char5>= {E8_OVERLAP_ALPHA}"})
    # E8(iii): para2 canary — the para_b=2 family stays contradicts
    para2 = [x for x in tr_sel if x.stratum == "CD-1" and x.para_b == 2]
    para_hist = dict(_C(x.para_b for x in tr_sel))
    rec("A-canary-para2(E8iii)", len(para2) >= 3
        and all(x.label == "contradicts" and lib.mechanical_ok(x) for x in para2),
        f"{len(para2)} para2 cells selected (all contradicts), para_b hist {para_hist}")
    # baselines
    def bl_band(x): return x._cos >= 0.95
    def digits(t): return sorted(re.findall(r"\d+", t))
    def bl_digit(x): return digits(x.side_a.body) != digits(x.side_b.body)
    def bl_dg(x): return bl_digit(x) and not lib.version_guard(x)
    bl = {}
    for name, fn in (("BL-band", bl_band), ("BL-digit", bl_digit),
                     ("BL-digit+guard", bl_dg)):
        bl[name] = round(sum(1 for x in tr_sel
                             if (fn(x) == (x.label == "contradicts")))
                         / len(tr_sel), 4)
    rec("baselines-non-tautological", all(v < 1.0 for v in bl.values()), bl)
    rec("A-regime", True, "declared in manifest")
    rec("A-frozen", True, f"nano {lib.EMBEDDER_PIN[:12]}…; cortex.features imported")

    all_ok = all(v["ok"] for v in A.values())
    if not all_ok:
        reds = [k for k, v in A.items() if not v["ok"]]
        log("build_stop", reason=f"asserts RED: {reds}")
        log_md("build_stop", reds=reds, note="stop-first: NOTHING written or sealed")
        print(f"BUILD-STOP: asserts RED: {reds}", file=sys.stderr)
        return 1

    # --- stop-first passed: emit ---
    rows = []
    for sc in tr_sel:
        rows += rows_of(sc, "train", f"seed:{SEEDS_V2['train']}")
    fp_train = write_jsonl(CORPUS_DIR_V2 / "train.jsonl", rows)
    rows_h = []
    for sc in ho_sel:
        rows_h += rows_of(sc, "holdout", f"seed:{SEEDS_V2['hold']}")
    fp_hold = write_jsonl(CORPUS_DIR_V2 / "holdout.jsonl", rows_h)
    fp_pool = write_jsonl(CORPUS_DIR_V2 / "holdout-pool.jsonl",
                          [r for i in pool_ids
                           for r in rows_of(by_id[i], "holdout-pool",
                                            f"seed:{SEEDS_V2['hold']}")])
    seal_payload = {"ts": now(), "version": "CD-holdout-v2",
                    "holdout_jsonl_sha256": fp_hold, "ids": sorted(set(seal_ids)),
                    "n": len(ho_sel),
                    "v1_archive": {
                        "paths": ["data/conflict-detector/corpus-v1/holdout.jsonl",
                                  "data/conflict-detector/corpus-v1/holdout-pool.jsonl",
                                  "data/conflict-detector/seals/CD-holdout-v1.seal.json",
                                  "data/conflict-detector/sheet/verdicts/KEY-cd-v1.sealed.json"],
                        "reads_in_this_build": 0,
                        "note": "CD-holdout-v1 archived UNTOUCHED; 0 reads in its "
                                "entire history (run-log fact, E11)"},
                    "note": "content NOT read after this event"}
    SEALS_DIR.mkdir(parents=True, exist_ok=True)
    SEAL_V2_PATH.write_text(json.dumps(seal_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    fp_seal = lib.sha(SEAL_V2_PATH.read_text(encoding="utf-8"))
    manifest = {
        "ts": now(), "version": "corpus-v2 (ADDENDUM-3 rebuild, B6-conflict)",
        "guard_version": lib.GUARD_VERSION,
        "instrument": {"provider": "vesma.embeddings.NanoProvider",
                       "nano_pin": lib.EMBEDDER_PIN},
        "design": ["docs/experiments/conflict-detector-corpus-design.md (RATIFIED MLC-9)",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-1.md",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-2.md",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-3.md (RATIFIED amend-then-ratify)"],
        "pin_sha256": lib.sha(PIN_V2_PATH.read_text(encoding="utf-8")),
        "seeds": dict(SEEDS_V2),
        "seed_family_check": pin["seed_family_check"],
        "strata": {s: {"net": STRATA_NET_V2[s],
                       "carrier": {c: CARRIER_QUOTA_V2[c].get(s, 0)
                                   for c in ("N1", "N2", "N3")},
                       "class_total": tr_cnt[s],
                       "seal": ho_cnt[s]} for s in ORDER},
        "carriers": {"N1": {"quota": 36, "hosts": "CD-1/2/4 = 16/8/12",
                            "k_mix": "22/14 (60/40)",
                            "E6": "scaffold inherited from donor (CV-grouped)"},
                     "N2": {"quota": 24, "k_mix": "14/10",
                            "role": "anti-proxy control (same dictionary, same k, NOT)"},
                     "N3": {"quota": 24, "predicate": "n3_ok branch (E5)",
                            "note": "NC-5 subclass; classic NC-5 predicate would fail every N3 pair"}},
        "para_dict_sha256": lib.PARA_DICT_SHA,
        "e8_separability": A["A-carrier-sep(E8i)"]["detail"],
        "e8_canary_para2": A["A-canary-para2(E8iii)"]["detail"],
        "bands_train": dict(tr_bands),
        "floors": pin["floors"],
        "baselines": bl,
        "asserts": A,
        "fingerprints": {"train.jsonl": fp_train, "holdout.jsonl": fp_hold,
                         "holdout-pool.jsonl": fp_pool, "seal": fp_seal},
        "known_uncovered": {"cos_below": pin["measured_corpus_floor"],
                            "note": "measured corpus floor over the emitted v2 corpus"},
        "null_exemption": "NC-4 corner twins + N3 carriers are both-null exempt "
                          "(N3 mechanical predicate requires visible languages, E5)",
        "seal_intact_inputs": seal_payload["v1_archive"],
        "remainder_unused_ids": sorted(remainder)[:50] + ["…"] if len(remainder) > 50
                                else sorted(remainder),
        "regime": "synthetic-only; no store texts; no LLM in construction; "
                  "labels by construction",
    }
    CORPUS_DIR_V2.mkdir(parents=True, exist_ok=True)
    (CORPUS_DIR_V2 / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("build2_done", ok=True, train_sha=fp_train[:16], hold_sha=fp_hold[:16],
        seal=fp_seal[:16], elapsed_s=round(time.time() - t0, 1))
    log_md("build_v2_done", ok=True, train_sha=fp_train, hold_sha=fp_hold,
           seal_sha=fp_seal, seal_n=len(ho_sel),
           asserts=f"{sum(1 for v in A.values() if v['ok'])}/{len(A)} PASS")
    log_md("seal_v2", seal="CD-holdout-v2", n=len(ho_sel), sha=fp_seal,
           v1_archive="untouched, 0 reads in entire history (E11)",
           holdout_content_reads=0)
    print(f"\nBUILD2: ALL ASSERTS PASS in {time.time()-t0:.0f}s\n"
          f"manifest: {CORPUS_DIR_V2/'build-manifest.json'}\nseal: {SEAL_V2_PATH}")
    return 0


# --- seal3: CD-holdout-v3 rotation (ADDENDUM-5 §4 F6-F12, ratified 10.11) ------
# POOL-ONLY lane (F7): measure2/build2 stay byte-for-byte untouched; a fresh
# hold-seed generation is cut to 180 and sealed as CD-holdout-v3. train.jsonl
# and the sheet are ABSENT BY CONSTRUCTION here. This is the slice rule
# (fresh eval surface), NOT a corpus rebuild and NOT amend-once: the E13
# amend budget is NOT spent, the train-v2 artifact (train sha a549dc0c…,
# model 2534fcbe…) stays frozen (F3 legs re-checked at runtime).

CORPUS_DIR_V3 = DATA / "corpus-v3"
PIN_V3_PATH = DATA / "pin-v3.json"
SEAL_V3_PATH = SEALS_DIR / "CD-holdout-v3.seal.json"
SEAL_V1_PATH = SEALS_DIR / "CD-holdout-v1.seal.json"
SEAL3_SLICE = "conflict-detector-corpus-v2-seal3"

# F8: the v3 seed family is a SINGLE-MEMBER family (hold only) — the train
# and sheet members are absent by construction in the pool-only lane. The
# seed must differ from the v2 family {2113,7919,9377}, the v1 family
# {42,777,4242} ∪ {43..52} (FORBIDDEN_SEEDS pins the v1 sets), and from
# EVERY seed recorded by any seeds_pinned* run-log event of any slice
# (ad-hoc pin guard — re-scanned at runtime BEFORE the pin event).
SEED_HOLD_V3 = 6117
SEEDS_FORBIDDEN_V3 = set(FORBIDDEN_SEEDS) | set(SEEDS_V2.values())
assert SEED_HOLD_V3 not in SEEDS_FORBIDDEN_V3, "v3 hold seed collides with a pinned family"

# F9: seal-slice k targets. The ratified k totals SEAL_N1_K=(7,4) spread over
# the host split SEAL_N1_HOSTS={CD-1:5, CD-2:3, CD-4:3}; N2 uses SEAL_N2_K.
SEAL_N1_K_TARGETS = {"CD-1": (3, 2), "CD-2": (2, 1), "CD-4": (2, 1)}
assert (sum(t[0] for t in SEAL_N1_K_TARGETS.values()),
        sum(t[1] for t in SEAL_N1_K_TARGETS.values())) == SEAL_N1_K
assert {h: t[0] + t[1] for h, t in SEAL_N1_K_TARGETS.items()} == SEAL_N1_HOSTS

# F9: NO CV minimums on eval cells (the train-CV rule stays a train-side
# instrument); small eval strata are reported descriptively, and no gate
# verdict may rest on an eval cell with n < 10.
SEAL3_REPORTING_RULE = {
    "cv_minimums": "none on eval cells (F9: train-CV rule is train-side)",
    "per_stratum_n_lt_10": "Wilson CI + descriptive-only; no verdict may "
                           "rest on an eval cell with n < 10",
}

# ADDENDUM-6 §5 (path B): the NC template universes are EXTENDED via the
# gen_pool_v3 eval-only fork banks (scripts/conflict_corpus_banks_v3.py,
# frozen 066eea6, sha pinned in pin-v3.json#prep.banks_v3). The v3 cut
# draws NC-1/NC-3/NC-5 from the MERGED pool (classic + new banks); NC-4
# stays OLD-bank only (G7: the v3 axis is fallback-margin, engaged only
# if post-preflight attrition kills the old-bank supply — disclosed,
# never silent). New-bank rows are the G3 transfer surface.
SEAL3_V3_PREFIX = "ho3"  # lane-distinct id prefix (same as census/char2 lanes)

# G10 census quotas (mirror of banks_v3_lane.CENSUS_QUOTAS NC legs — the
# ratified 2x depth on clean-after-preflight supply; NC-4 strict 11 has
# the 2x WAIVED per G7 and must clear 11 to cut). Carriers keep their
# E9-mini 2x legs (N1 22 / N2 14 / N3 14 + hosts + k) computed on CLEAN
# rows. A census failure is a BuildStop + NEW addendum (G1) — never a
# bank iteration.
SEAL3_G10_QUOTAS = {"NC-1": 40, "NC-2": 44, "NC-3": 32, "NC-4": 11,
                    "NC-5": 50}


def log3(event: str, **fields) -> None:
    """Append a seal3 event to the run-log (own slice tag)."""
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": SEAL3_SLICE, **fields},
                            ensure_ascii=False) + "\n")


def log3_md(event: str, **fields) -> None:
    """E11 dual format: the same event into the COMMITTED md run-log."""
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": SEAL3_SLICE, **fields},
                            ensure_ascii=False) + "\n")


def pinned_seeds_from_runlog() -> tuple[dict[str, list[int]], list[int]]:
    """F8 ad-hoc pin guard: every integer recorded under a seed-ish key in
    ANY seeds_pinned* run-log event (any slice, any generation). Returns
    (per-event map, prior seeds_pinned_v3 hold-seed pins)."""
    found: dict[str, list[int]] = {}
    prior_v3: list[int] = []
    if not RUNLOG.exists():
        return found, prior_v3
    ints: list[int] = []

    def walk(v) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            ints.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    for line in RUNLOG.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        name = str(ev.get("event", ""))
        if not name.startswith("seeds_pinned"):
            continue
        ints.clear()
        for k, v in ev.items():
            if "seed" in k.lower() or k == "per_substratum":
                walk(v)
        if ints:
            found[str(ev.get("ts", "?"))] = sorted(set(ints))
        if name == "seeds_pinned_v3":
            seeds = ev.get("seeds")
            if isinstance(seeds, dict) and isinstance(seeds.get("hold"), int):
                prior_v3.append(seeds["hold"])
    return found, prior_v3


def file_sha256(path: Path) -> str:
    """Whole-file digest. Used ONLY as an integrity seal on archived
    artifacts (byte digest, no parsing — sealed content stays unopened)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _disk_env_sha(env: dict) -> str:
    """sha over the PINNED embed-text convention (lib.embed_text) rebuilt
    from a jsonl envelope — the F6 disk-side identity."""
    return lib.sha(f"{lib.n1(env['title'])}\n{lib.n1(env['body'])}\n"
                   f"{lib.n1(' '.join(env['tags']))}")


def disk_text_shas(jsonl_path: Path, ids: set[str] | None = None
                   ) -> tuple[set[str], set[str]]:
    """sha(text_a/b) set over jsonl rows (both orientations, deduped). With
    `ids`, restrict to those pair ids (id->pool-origin disjointness) and
    return the requested ids NOT found on disk. NEVER pointed at a sealed
    holdout.jsonl — sealed content is only reached through its pool origin
    (ADDENDUM-5 F6/F10)."""
    out: set[str] = set()
    seen_ids: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pid = row.get("pair_id") or row.get("scenario_id")
            if ids is not None and pid not in ids:
                continue
            seen_ids.add(pid)
            out.add(_disk_env_sha(row["envelope_a"]))
            out.add(_disk_env_sha(row["envelope_b"]))
    missing = (ids - seen_ids) if ids is not None else set()
    return out, missing


def sealed_generation_sha_set(seal_path: Path, pool_path: Path
                              ) -> set[str]:
    """F10: text-sha set of a SEALED generation WITHOUT opening the sealed
    holdout.jsonl — seal-manifest ids -> pool-origin rows on disk -> shas.
    Fail-closed on manifest/pool divergence."""
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    ids = set(seal["ids"])
    if seal.get("n") is not None and len(ids) != seal["n"]:
        die(f"seal manifest {seal_path.name}: {len(ids)} unique ids != n={seal['n']}")
    shas, missing = disk_text_shas(pool_path, ids)
    if missing:
        die(f"seal manifest {seal_path.name}: {len(missing)} ids absent from "
            f"pool origin {pool_path.name} (provenance broken)")
    return shas


def _forbidden_source(ha: str, hb: str,
                      sources: dict[str, set[str]]) -> str | None:
    for name, sha_set in sources.items():
        if ha in sha_set or hb in sha_set:
            return name
    return None


def cmd_seal3() -> int:
    """CD-holdout-v3 rotation (ADDENDUM-5 §4, F6-F12): pool-only lane,
    stop-first (any RED assert writes NOTHING and seals NOTHING)."""
    t0 = time.time()

    # --- legality preconditions (F3/F7 legs DE owns) ------------------------
    gate_done = 0
    if RUNLOG.exists():
        for line in RUNLOG.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "gate_done" \
                    and ev.get("slice") == "conflict-detector-v2-gate":
                gate_done += 1
    if gate_done:
        die(f"F3 freeze violated: v2 gate already reached a verdict "
            f"({gate_done} gate_done events) — rotation is illegal")
    man2 = json.loads((CORPUS_DIR_V2 / "build-manifest.json")
                      .read_text(encoding="utf-8"))
    train_fp_start = file_sha256(CORPUS_DIR_V2 / "train.jsonl")
    if train_fp_start != man2["fingerprints"]["train.jsonl"]:
        die("train-v2 artifact integrity RED: disk train.jsonl digest != "
            "build-manifest fingerprint — F7 forbids the whole lane")
    hold2_fp = file_sha256(CORPUS_DIR_V2 / "holdout.jsonl")
    seal2_fp = file_sha256(SEAL_V2_PATH)
    hold1_fp = file_sha256(CORPUS_DIR / "holdout.jsonl")
    archives_intact = {
        "corpus-v2/holdout.jsonl": hold2_fp == man2["fingerprints"]["holdout.jsonl"],
        "seals/CD-holdout-v2.seal.json": seal2_fp == man2["fingerprints"]["seal"],
        "corpus-v1/holdout.jsonl": hold1_fp == json.loads(
            SEAL_V1_PATH.read_text(encoding="utf-8"))["holdout_jsonl_sha256"],
    }
    if not all(archives_intact.values()):
        die(f"archive integrity RED: {dict(archives_intact)} — someone touched a seal")
    print(f"preconditions: F3 gate_done=0; train-v2 fp {train_fp_start[:16]} "
          f"== manifest; v1/v2 archives digest-intact (bytes hashed, NOT parsed)")

    # --- G1: the banks freeze must hold BEFORE anything observes a census --
    # static guards fail-closed at import (G9); file+content shas must match
    # the pin-v3.json prep pin (commit 066eea6). A violation is an incident,
    # not an iteration: banks stay frozen, the lane BuildStops.
    import conflict_corpus_banks_v3 as banks_v3
    prep = {}
    if PIN_V3_PATH.exists():
        prep = json.loads(PIN_V3_PATH.read_text(encoding="utf-8")).get("prep", {})
    bv = prep.get("banks_v3")
    if not bv:
        die("G1: pin-v3.json has no banks_v3 prep pin — freeze first")
    banks_file_sha = file_sha256(REPO / "scripts" / "conflict_corpus_banks_v3.py")
    if banks_file_sha != bv["file_sha256"]:
        die(f"G1 LOCK VIOLATION: banks module sha {banks_file_sha[:16]} != "
            f"frozen {bv['file_sha256'][:16]} — bank edited after freeze")
    banks_content_sha = lib.sha(lib.json_dumps_sorted(banks_v3.bank_content()))
    if banks_content_sha != bv["content_sha256"]:
        die(f"G1 LOCK VIOLATION (content): banks content sha "
            f"{banks_content_sha[:16]} != frozen {bv['content_sha256'][:16]}")
    print(f"G1 banks freeze: commit {bv['banks_commit'][:12]}, file "
          f"{banks_file_sha[:16]}, content {banks_content_sha[:16]} — LOCKED")

    # --- F8: seed pin FIRST, before any generation ---------------------------
    scan, prior_v3 = pinned_seeds_from_runlog()
    observed = sorted({s for seeds in scan.values() for s in seeds})
    observed_elsewhere = sorted(set(observed) - set(prior_v3))
    if SEED_HOLD_V3 in observed_elsewhere:
        die(f"v3 hold seed {SEED_HOLD_V3} already pinned elsewhere in run_log "
            f"(ad-hoc pin guard)")
    sealed_v3 = 0
    for line in RUNLOG.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "seal_v3":
            sealed_v3 += 1
    if sealed_v3:
        die(f"rotation already consumed: {sealed_v3} seal_v3 events in run_log "
            "— a burned v3 would be a NEW incident, not a retry")
    for p in prior_v3:
        if p != SEED_HOLD_V3:
            die(f"seed-family shopping RED: prior seeds_pinned_v3 hold={p}, "
                f"this run pins {SEED_HOLD_V3}")
    reaffirm = len(prior_v3) > 0
    log3("seeds_pinned_v3", seeds={"hold": SEED_HOLD_V3},
         family="single-member (hold only; train/sheet absent by construction)",
         forbidden="2113/7919/9377 + 42/777/4242 + 43..52 + every seeds_pinned* seed",
         reaffirmed=bool(reaffirm),
         runlog_scan={"events": len(scan), "distinct_seeds": len(observed),
                      "seeds": observed})
    log3_md("seeds_pinned_v3", seeds={"hold": SEED_HOLD_V3},
            family="single-member (hold only)",
            forbidden_sets=sorted(SEEDS_FORBIDDEN_V3),
            reaffirmed=bool(reaffirm),
            runlog_scan_events=len(scan), runlog_scan_seeds=len(observed))
    print(f"seed v3 pinned: hold={SEED_HOLD_V3} "
          f"({'RE-AFFIRMED, ' + str(len(prior_v3)) + ' prior pin(s)' if reaffirm else 'fresh'}; "
          f"run-log scan: {len(scan)} seeds_pinned events, "
          f"{len(observed)} distinct seeds — no collision)")

    cos = Cos()
    print("== mini-census E9: hold-side carrier pools (fresh v3 seed) ==")
    hn1 = gen_n1_pool(SEED_HOLD_V3, "ho")
    hn2 = gen_n2_pool(SEED_HOLD_V3, "ho")
    hn3 = gen_n3_pool(SEED_HOLD_V3, "ho")
    for sc in hn1 + hn2 + hn3:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)
    cen_n1 = [x for x in hn1 if carrier_suitable(x)]
    cen_n2 = [x for x in hn2 if carrier_suitable(x)]
    cen = {
        "n1": {"pool": len(hn1), "suitable": len(cen_n1),
               "by_host": {h: sum(1 for x in cen_n1 if x.stratum == h)
                           for h in ("CD-1", "CD-2", "CD-4")},
               "by_k": {k: sum(1 for x in cen_n1 if len(x.benign) == k)
                        for k in (1, 2)}},
        "n2": {"pool": len(hn2), "suitable": len(cen_n2),
               "by_k": {k: sum(1 for x in cen_n2 if len(x.benign) == k)
                        for k in (1, 2)}},
        "n3": {"pool": len(hn3),
               "mech_ok": sum(1 for x in hn3 if lib.mechanical_ok(x))},
    }
    print("census:", json.dumps(cen))
    # E9-mini gate: every SEAL-slice quota needs 2x suitability depth
    need_n1_total = 2 * (SEAL_N1_K[0] + SEAL_N1_K[1])
    need_n1_host = {h: 2 * q for h, q in SEAL_N1_HOSTS.items()}
    need_n1_k = {1: 2 * SEAL_N1_K[0], 2: 2 * SEAL_N1_K[1]}
    need_n2 = 2 * SEAL_SHARES_V2["N2"]
    need_n2_k = {1: 2 * SEAL_N2_K[0], 2: 2 * SEAL_N2_K[1]}
    need_n3 = 2 * N3_SEAL
    red = []
    if len(cen_n1) < need_n1_total:
        red.append(f"N1 suitable {len(cen_n1)} < {need_n1_total}")
    for h, need in need_n1_host.items():
        if cen["n1"]["by_host"][h] < need:
            red.append(f"N1/{h} {cen['n1']['by_host'][h]} < {need}")
    for k, need in need_n1_k.items():
        if cen["n1"]["by_k"][k] < need:
            red.append(f"N1 k={k} {cen['n1']['by_k'][k]} < {need}")
    if len(cen_n2) < need_n2:
        red.append(f"N2 suitable {len(cen_n2)} < {need_n2}")
    for k, need in need_n2_k.items():
        if cen["n2"]["by_k"][k] < need:
            red.append(f"N2 k={k} {cen['n2']['by_k'][k]} < {need}")
    if cen["n3"]["mech_ok"] < need_n3:
        red.append(f"N3 mech_ok {cen['n3']['mech_ok']} < {need_n3}")
    if red:
        die("census E9-mini RED: " + "; ".join(red))
    log3("census_v3", verdict="pass",
         need={"n1_total": need_n1_total, "n1_host": need_n1_host,
               "n1_k": need_n1_k, "n2": need_n2, "n2_k": need_n2_k,
               "n3": need_n3})
    log3_md("census_v3", verdict="pass", n1_suitable=cen["n1"]["suitable"],
            n2_suitable=cen["n2"]["suitable"], n3_mech_ok=cen["n3"]["mech_ok"],
            note="E9-mini: 2x depth verified on every seal-slice quota")
    print("census E9-mini PASS: 2x depth on every seal quota "
          f"(N1 {need_n1_total}, N2 {need_n2}, N3 {need_n3})")

    print("== classic hold pool + v3 banks (F7: rotate=TRUE EXPLICIT — "
          "gen_pool trap; ADDENDUM-6 path B merged pool) ==")
    hold_pool = gen_pool(SEED_HOLD_V3, "ho", rotate=True) \
        + banks_v3.gen_banks_v3_pool(SEED_HOLD_V3, SEAL3_V3_PREFIX, rotate=True)
    pool_classic_n = sum(1 for x in hold_pool
                         if not x.scenario_id.startswith(SEAL3_V3_PREFIX + "-"))
    pool_v3_n = len(hold_pool) - pool_classic_n
    print(f"holdout pool: {len(hold_pool)} (classic {pool_classic_n}, "
          f"v3 banks {pool_v3_n})")
    measure_all(hold_pool, cos)
    print(f"embedded {cos.n} unique texts in {time.time()-t0:.0f}s")

    # --- union-cap prior = REAL train-v2 scaffold counts from disk --------
    # (the §4.5 union cap is honored against the frozen artifact during
    # picking, not only at assert time)
    tr_scaf: dict[str, int] = {}
    tr_scaf_by_s: dict[str, dict[str, int]] = {}
    seen_pairs: set[str] = set()
    with (CORPUS_DIR_V2 / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pid = row["pair_id"]
            if pid in seen_pairs:
                continue
            seen_pairs.add(pid)
            tr_scaf[row["scaffold_id"]] = tr_scaf.get(row["scaffold_id"], 0) + 1
            tr_scaf_by_s.setdefault(row["stratum"], {})
            tr_scaf_by_s[row["stratum"]][row["scaffold_id"]] = \
                tr_scaf_by_s[row["stratum"]].get(row["scaffold_id"], 0) + 1
    print(f"train-v2 artifact on disk: {len(seen_pairs)} pairs, "
          f"{len(tr_scaf)} scaffold classes (union-cap prior)")

    # --- disjointness sources (F6/F10); sealed content NEVER opened --------
    print("== disjointness sources (F6/F10) ==")
    tr_texts, _ = disk_text_shas(CORPUS_DIR_V2 / "train.jsonl")
    v2_shas = sealed_generation_sha_set(SEAL_V2_PATH,
                                        CORPUS_DIR_V2 / "holdout-pool.jsonl")
    v1_shas = sealed_generation_sha_set(SEAL_V1_PATH,
                                        CORPUS_DIR / "holdout-pool.jsonl")
    sources = {"train-v2": tr_texts, "v2-seal": v2_shas, "v1-seal": v1_shas}
    print(f"sha sets: train-v2 {len(tr_texts)}, v2-seal {len(v2_shas)} "
          f"(ids->pool-origin), v1-seal {len(v1_shas)} (ids->pool-origin); "
          "holdout.jsonl v1/v2 NOT opened")

    # F6 mechanics check (fail-fast, BEFORE anything depends on the disk
    # sha sets): the embed-convention reconstruction must be byte-exact.
    # Regenerate the v2 hold pool in-process (no embedding, sealed file
    # untouched) and compare per-id shas.
    disk_pool_by_id: dict[str, set[str]] = {}
    with (CORPUS_DIR_V2 / "holdout-pool.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            disk_pool_by_id.setdefault(row["pair_id"], set()).update(
                {_disk_env_sha(row["envelope_a"]),
                 _disk_env_sha(row["envelope_b"])})
    conv_pool = list(gen_pool(SEEDS_V2["hold"], "ho", rotate=True)) \
        + gen_n1_pool(SEEDS_V2["hold"], "ho") \
        + gen_n2_pool(SEEDS_V2["hold"], "ho") \
        + gen_n3_pool(SEEDS_V2["hold"], "ho")
    conv_n = 0
    for x in conv_pool:
        want = disk_pool_by_id.get(x.scenario_id)
        if want is None:
            continue  # candidate was not registered into the v2 pool file
        if {lib.sha(x.text_a()), lib.sha(x.text_b())} != want:
            die(f"sha convention RED at {x.scenario_id}: in-process texts "
                "diverge from the v2 pool-origin rows on disk")
        conv_n += 1
    conv_ok = conv_n == len(disk_pool_by_id)
    if not conv_ok:
        die(f"sha convention RED: only {conv_n}/{len(disk_pool_by_id)} v2 "
            "pool ids reproduce identical disk shas — disk-side sha sets "
            "untrustworthy, refusing the whole lane")

    # --- G10 census (ADDENDUM-6 §5): E9-mini over clean-after-preflight
    # supply — preflight/mechanical attrition -> F10 sha cull vs ALL three
    # sealed generations -> within-count side-text uniqueness. FRESH RUN on
    # every invocation (never cached); because seed, banks and instrument
    # are pinned, the fresh numbers must REPRODUCE the census-lane report
    # (artifacts/runs/banks_v3_census.json) exactly — divergence is an
    # instrument abort, not a bank defect and not a silent pass.
    print("== G10 census: clean-after-preflight supply vs 2x quotas ==")

    def _clean(cells: list, mech_fn) -> dict:
        live = [x for x in cells if mech_fn(x)]
        live.sort(key=lambda x: (-x._cos, x.scenario_id))
        seen: set[str] = set()
        culled: Counter = Counter()
        dups = 0
        rows = []
        for x in live:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            src = _forbidden_source(ha, hb, sources)
            if src:
                culled[src] += 1
                continue
            if ha in seen or hb in seen:
                dups += 1
                continue
            seen.update((ha, hb))
            rows.append(x)
        origin = Counter("v3" if x.scenario_id.startswith(SEAL3_V3_PREFIX + "-")
                         else "classic" for x in rows)
        return {"pool": len(cells),
                "preflight_attrited": len(cells) - len(live),
                "culled_f10": dict(culled), "dup_blocked": dups,
                "clean": len(rows), "clean_v3": origin.get("v3", 0),
                "clean_classic": origin.get("classic", 0), "_rows": rows}

    g10: dict = {}
    red10 = []
    by_s_g10: dict[str, list] = {}
    for x in hold_pool:
        by_s_g10.setdefault(x.stratum, []).append(x)
    for s in ORDER:
        row = _clean(by_s_g10.get(s, []), lib.mechanical_ok)
        g10[s] = {k: v for k, v in row.items() if k != "_rows"}
        want = SEAL3_G10_QUOTAS.get(s)
        if want is not None:
            mark = "OK" if row["clean"] >= want else "RED"
            if row["clean"] < want:
                red10.append(f"{s} clean {row['clean']} < {want}")
            print(f"  {s}: clean {row['clean']} (v3 {row['clean_v3']} / "
                  f"classic {row['clean_classic']}) vs {want} [{mark}]")
        else:
            print(f"  {s}: clean {row['clean']} (v3 {row['clean_v3']} / "
                  f"classic {row['clean_classic']}) (informational)")
    # carriers: the same G10 methodology, gated on CLEAN rows (the honest
    # depth a sha-disjoint cut can consume) — host/k legs at the 2x needs
    cen_n1c = _clean(hn1, carrier_suitable)
    cen_n2c = _clean(hn2, carrier_suitable)
    cen_n3c = _clean(hn3, lib.mechanical_ok)
    n1_hosts_c = Counter(x.stratum for x in cen_n1c["_rows"])
    n1_k_c = Counter(len(x.benign) for x in cen_n1c["_rows"])
    n2_k_c = Counter(len(x.benign) for x in cen_n2c["_rows"])
    g10["N1"] = {**{k: v for k, v in cen_n1c.items() if k != "_rows"},
                 "hosts": dict(n1_hosts_c), "k": dict(n1_k_c)}
    g10["N2"] = {**{k: v for k, v in cen_n2c.items() if k != "_rows"},
                 "k": dict(n2_k_c)}
    g10["N3"] = {k: v for k, v in cen_n3c.items() if k != "_rows"}
    for name, got, need in (
            ("N1", cen_n1c["clean"], need_n1_total),
            ("N1/CD-1", n1_hosts_c.get("CD-1", 0), need_n1_host["CD-1"]),
            ("N1/CD-2", n1_hosts_c.get("CD-2", 0), need_n1_host["CD-2"]),
            ("N1/CD-4", n1_hosts_c.get("CD-4", 0), need_n1_host["CD-4"]),
            ("N1/k1", n1_k_c.get(1, 0), need_n1_k[1]),
            ("N1/k2", n1_k_c.get(2, 0), need_n1_k[2]),
            ("N2", cen_n2c["clean"], need_n2),
            ("N2/k1", n2_k_c.get(1, 0), need_n2_k[1]),
            ("N2/k2", n2_k_c.get(2, 0), need_n2_k[2]),
            ("N3", cen_n3c["clean"], need_n3)):
        mark = "OK" if got >= need else "RED"
        if got < need:
            red10.append(f"{name} clean {got} < {need}")
        print(f"  {name}: clean {got} vs {need} [{mark}]")
    # determinism: reproduce the frozen census-lane numbers (same seed,
    # same banks sha, same pinned nano) — any drift aborts the instrument
    cen_ref_path = REPO / "artifacts" / "runs" / "banks_v3_census.json"
    reproduced_census = None
    if cen_ref_path.exists():
        ref = json.loads(cen_ref_path.read_text(encoding="utf-8"))
        dev = []
        for s, exp in ((s, ref.get("strata", {}).get(s, {}).get("clean"))
                       for s in ORDER):
            if exp is not None and g10[s]["clean"] != exp:
                dev.append(f"{s}: fresh {g10[s]['clean']} != census-lane {exp}")
        for name, key, exp in (("N1", "n1", ref.get("carriers", {})
                                .get("n1", {}).get("clean")),
                               ("N2", "n2", ref.get("carriers", {})
                                .get("n2", {}).get("clean")),
                               ("N3", "n3", ref.get("carriers", {})
                                .get("n3", {}).get("clean"))):
            if exp is not None and g10[name]["clean"] != exp:
                dev.append(f"{name}: fresh {g10[name]['clean']} != "
                           f"census-lane {exp}")
        if dev:
            die("G10 census determinism RED (fresh run diverges from the "
                "frozen census-lane report — instrument abort, NOT a bank "
                "defect, nothing written): " + "; ".join(dev))
        reproduced_census = True
        print("G10 determinism: fresh census reproduces the census-lane "
              "numbers exactly (strata + carriers)")
    if red10:
        die("census G10 RED (clean-after-preflight, fresh run): "
            + "; ".join(red10)
            + " — rotation lane BuildStop, NOTHING sealed (G1: a census "
              "failure is a new addendum, not a bank iteration)")
    log3("census_v3_g10", verdict="pass", g10=g10,
         quotas=SEAL3_G10_QUOTAS, fresh_run=True,
         reproduced_census_banks=reproduced_census)
    log3_md("census_v3_g10", verdict="pass",
            clean={s: g10[s]["clean"] for s in ORDER},
            carriers={c: g10[c]["clean"] for c in ("N1", "N2", "N3")},
            nc4_clean_classic=g10["NC-4"]["clean_classic"],
            fresh_run=True, reproduced_census_banks=reproduced_census,
            note="G10: E9-mini over clean-after-preflight supply "
                 "(attrition -> F10 cull -> within-count dedup); fresh "
                 "run, not cached")
    print("census G10 PASS: clean-after-preflight depth verified on every "
          "seal quota (strata 2x, NC-4 strict, carriers 2x incl. host/k)")

    # --- classic supply census (E9 philosophy over the WIDE pool): per
    # stratum, how many candidates are sha-clean against ALL three
    # generations (F10) under within-cut uniqueness. The classic cut has
    # NO ratified census gate of its own (the ratified E9-mini covers the
    # carrier careers), but a starved stratum would die mid-cut — the
    # census fails fast with the FULL table so a red run leaves complete
    # numbers, not a one-line die.
    print("== classic supply census (sha-clean depth vs seal shares) ==")
    by_stratum: dict[str, list] = {}
    for x in hold_pool:
        by_stratum.setdefault(x.stratum, []).append(x)
    for s in by_stratum:
        by_stratum[s].sort(key=lambda x: (-x._cos, x.scenario_id))
    supply = {}
    starved = []
    for s in ORDER:
        cands = by_stratum.get(s, [])
        seen: set[str] = set()
        clean = 0
        for x in cands:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            if _forbidden_source(ha, hb, sources):
                continue
            if ha in seen or hb in seen:
                continue
            seen.update((ha, hb))
            clean += 1
        want = SEAL_SHARES_V2[s]
        supply[s] = {"pool": len(cands), "want": want, "clean": clean}
        if clean < want:
            starved.append(f"{s} clean {clean} < share {want}")
    log3("supply_census_v3", supply=supply, starved=starved)
    log3_md("supply_census_v3", supply=supply, starved=starved,
            note="classic sha-clean depth (F6/F10) vs ratified seal shares; "
                 "seed-independent text universes do not replenish")
    print("supply:", json.dumps(supply))
    if starved:
        die("classic supply RED (F10 sha-disjointness vs seed-independent "
            "universes): " + "; ".join(starved)
            + " — the ratified share table is not fillable from a fresh "
              "generation; rotation lane BuildStop, NOTHING sealed")

    # --- cut 180 (F7 pool-only: the cut draws from the WIDE generated
    # pool, hardest-first, union scaffold caps, sha-only culling; F6
    # excludes scenario_id checks). The v2-era registered-516 window is a
    # train+hold joint-emission artifact that does not exist in this lane;
    # with three generations of sha culling it is starved by design
    # (NC-1 texts are largely seed-INDEPENDENT — the top of ANY window
    # collides with the sealed tops of v1/v2). Depth discipline comes
    # from the supply census above + BuildStop on exhaustion.
    print("== cut: 180 per SEAL_SHARES_V2 from the wide pool (sha-only) ==")
    scaf_ho: dict[str, int] = dict(tr_scaf)
    taken_shas: set[str] = set()
    seal_ids: list[str] = []
    seal_taken: set[str] = set()
    origin_of: dict[str, str] = {}   # G3: transfer surface marking
    culled = 0
    cull_by_src: Counter = Counter()
    cull_diag: dict[str, str] = {}
    want_classic = {s: SEAL_SHARES_V2[s] for s in ORDER}

    blocked: Counter = Counter()

    def cut_one(x: lib.Scenario) -> str | None:
        """Shared cut predicate. Returns 'taken' | 'culled:<src>' |
        'cap' | 'dup' | 'dup-id'."""
        nonlocal culled
        if x.scenario_id in seal_taken:
            blocked["dup-id"] += 1
            return "dup-id"
        ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
        src = _forbidden_source(ha, hb, sources)
        if src:
            cull_diag.setdefault(x.scenario_id, f"text sha in {src}")
            cull_by_src[src] += 1
            culled += 1
            return f"culled:{src}"
        if ha in taken_shas or hb in taken_shas:
            blocked["dup"] += 1
            return "dup"
        if scaf_ho.get(x.scaffold_id, 0) >= SCAFFOLD_CAP_V2[x.stratum]:
            blocked["cap"] += 1
            return "cap"
        seal_ids.append(x.scenario_id)
        seal_taken.add(x.scenario_id)
        origin_of[x.scenario_id] = (
            "transfer:v3" if x.scenario_id.startswith(SEAL3_V3_PREFIX + "-")
            else "indist:classic")
        scaf_ho[x.scaffold_id] = scaf_ho.get(x.scaffold_id, 0) + 1
        taken_shas.update((ha, hb))
        return "taken"

    depth_report = {}
    nc4_decision: dict = {}
    for s in ORDER:
        blocked.clear()
        cands = by_stratum.get(s, [])
        got = 0
        if s == "NC-4":
            # G7: NC-4 is cut from the OLD bank (the only honest in-dist
            # knife leg); the v3 non-tech axis serves ONLY as fallback
            # margin if post-preflight attrition kills the old-bank
            # supply — any fallback engagement is disclosed, never silent.
            classic_c = [x for x in cands
                         if not x.scenario_id.startswith(SEAL3_V3_PREFIX + "-")]
            v3_c = [x for x in cands
                    if x.scenario_id.startswith(SEAL3_V3_PREFIX + "-")]
            for x in classic_c:
                if got >= want_classic[s]:
                    break
                if cut_one(x) == "taken":
                    got += 1
            fallback_n = 0
            if got < want_classic[s]:
                for x in v3_c:
                    if got >= want_classic[s]:
                        break
                    if cut_one(x) == "taken":
                        got += 1
                        fallback_n += 1
            nc4_decision = {
                "bank": "classic(old)" if not fallback_n
                        else f"fallback:v3 ({fallback_n} rows)",
                "old_bank_pool": len(classic_c),
                "fallback_rows": fallback_n,
                "rule": "G7: NC-4 from the OLD bank; v3 axis is "
                        "fallback-margin only (disclosed)"}
            print(f"NC-4 G7: {nc4_decision['bank']} (old-bank pool "
                  f"{nc4_decision['old_bank_pool']})")
        else:
            for x in cands:
                if got >= want_classic[s]:
                    break
                if cut_one(x) == "taken":
                    got += 1
        depth_report[s] = {"pool": len(cands), "want": want_classic[s],
                           "got": got, "blocked": dict(blocked)}
        if got < want_classic[s]:
            examples = {k: cull_diag[k] for k in list(cull_diag)[:5]}
            die(f"seal3 disjointness: classic {s} exhausted "
                f"({got}/{want_classic[s]}) over the WIDE pool "
                f"({depth_report[s]}); cull examples {examples}")
    print("depth:", json.dumps(depth_report))
    # N1 slice: hosts 5/3/3 with per-host k targets meeting (7,4)
    n1_ho = sorted([x for x in hn1 if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    n1_host_got = {h: 0 for h in SEAL_N1_HOSTS}
    for h, want_h in SEAL_N1_HOSTS.items():
        for k, want_k in zip((1, 2), SEAL_N1_K_TARGETS[h]):
            got_k = 0
            for x in n1_ho:
                if got_k >= want_k:
                    break
                if x.stratum != h or len(x.benign) != k:
                    continue
                if cut_one(x) == "taken":
                    got_k += 1
                    n1_host_got[h] += 1
            if got_k < want_k:
                die(f"seal3 disjointness: N1/{h} k={k} exhausted "
                    f"({got_k}/{want_k})")
        if n1_host_got[h] < want_h:
            die(f"seal3 disjointness: N1/{h} exhausted "
                f"({n1_host_got[h]}/{want_h})")
    # N2 slice: 7 with k 4/3
    n2_ho = sorted([x for x in hn2 if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    n2_k_got = {1: 0, 2: 0}
    for k, want_k in zip((1, 2), SEAL_N2_K):
        for x in n2_ho:
            if n2_k_got[k] >= want_k:
                break
            if len(x.benign) != k:
                continue
            if cut_one(x) == "taken":
                n2_k_got[k] += 1
        if n2_k_got[k] < want_k:
            die(f"seal3 disjointness: N2 k={k} exhausted ({n2_k_got[k]}/{want_k})")
    # N3 slice: 7
    n3_ho = sorted([x for x in hn3 if lib.mechanical_ok(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    got_n3 = 0
    for x in n3_ho:
        if got_n3 >= N3_SEAL:
            break
        if cut_one(x) == "taken":
            got_n3 += 1
    if got_n3 < N3_SEAL:
        die(f"seal3 disjointness: N3 exhausted ({got_n3}/{N3_SEAL})")
    assert len(seal_ids) == HOLD_SEAL, len(seal_ids)
    ho_by_id = {x.scenario_id: x for x in hold_pool}
    ho_by_id.update({x.scenario_id: x for x in hn1 + hn2 + hn3})
    assert set(seal_ids) <= set(ho_by_id), "seal id outside generated pools"
    origins_total = Counter(origin_of[i] for i in seal_ids)
    log3("cut_v3", n=len(seal_ids), culled=culled,
         cull_by_src=dict(cull_by_src), depth=depth_report,
         origins=dict(origins_total), nc4=nc4_decision)
    log3_md("cut_v3", n=len(seal_ids), culled=culled,
            cull_by_src=dict(cull_by_src),
            nc4_bank=nc4_decision.get("bank"),
            origins_transfer=origins_total.get("transfer:v3", 0),
            origins_indist=origins_total.get("indist:classic", 0),
            depth_got={s: d["got"] for s, d in depth_report.items()},
            note="sha-only culling (F6) over the WIDE pool (classic + v3 "
                 "banks, ADDENDUM-6 path B); sealed v1/v2 content not opened")
    print(f"cut: {len(seal_ids)} pairs ({culled} culled: "
          f"{dict(cull_by_src)})")

    # --- asserts (F6/F9/F10), stop-first ------------------------------------
    print("== asserts (ADDENDUM-5 F6/F9/F10), stop-first ==")
    A: dict[str, dict] = {}

    def rec(name: str, ok, detail) -> bool:
        A[name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'RED'}  {name}: {detail}")
        return bool(ok)

    seal_sc = [ho_by_id[i] for i in seal_ids]
    cnt = Counter(x.stratum for x in seal_sc)
    classic_cnt = Counter(x.stratum for x in seal_sc if x.carrier is None)
    n1_sc = [x for x in seal_sc if x.carrier == "N1"]
    n2_sc = [x for x in seal_sc if x.carrier == "N2"]
    n3_sc = [x for x in seal_sc if x.carrier == "N3"]
    fold_expected = Counter()
    for s in ORDER:
        fold_expected[s] += SEAL_SHARES_V2[s]
    for h, q in SEAL_N1_HOSTS.items():
        fold_expected[h] += q
    fold_expected["NC-2"] += SEAL_SHARES_V2["N2"]
    fold_expected["NC-5"] += N3_SEAL
    classic_ok = all(classic_cnt[s] == SEAL_SHARES_V2[s] for s in ORDER)
    n1_ok = (Counter(x.stratum for x in n1_sc) == Counter(SEAL_N1_HOSTS)
             and Counter(len(x.benign) for x in n1_sc) ==
             Counter({1: SEAL_N1_K[0], 2: SEAL_N1_K[1]}))
    n2_ok = (len(n2_sc) == SEAL_SHARES_V2["N2"]
             and Counter(len(x.benign) for x in n2_sc) ==
             Counter({1: SEAL_N2_K[0], 2: SEAL_N2_K[1]}))
    n3_ok = len(n3_sc) == N3_SEAL
    rec("A-share-table(F9)", classic_ok and n1_ok and n2_ok and n3_ok
        and dict(cnt) == dict(fold_expected) and sum(cnt.values()) == HOLD_SEAL,
        {"classic": dict(classic_cnt),
         "N1_hosts": dict(Counter(x.stratum for x in n1_sc)),
         "N1_k": dict(Counter(len(x.benign) for x in n1_sc)),
         "N2_k": dict(Counter(len(x.benign) for x in n2_sc)),
         "N3": len(n3_sc), "fold": dict(cnt),
         "fold_expected": dict(fold_expected), "total": sum(cnt.values())})
    bad = [x.scenario_id for x in seal_sc if not lib.mechanical_ok(x)]
    len_bad = [x.scenario_id for x in seal_sc
               if x.carrier in ("N1", "N2") and not lib.len_channel_ok(x)]
    rec("A-silver+len4(F9)", not bad and not len_bad,
        f"{len(seal_sc)-len(bad)}/{len(seal_sc)} mechanical pass "
        f"(N1/N2 benign_pair_ok per-pair); E4 len-channel violations {len(len_bad)}")
    badl = [x.scenario_id for x in seal_sc
            if x.label != ("contradicts" if x.stratum.startswith("CD")
                           else "not-contradicts")]
    rec("A-labels(F9)", not badl, f"bad={len(badl)}")
    g_nc3 = sum(1 for x in seal_sc if x.stratum == "NC-3" and lib.version_guard(x))
    n_nc3 = sum(1 for x in seal_sc if x.stratum == "NC-3")
    g_cd = sum(1 for x in seal_sc if x.stratum.startswith("CD")
               and lib.version_guard(x))
    rec("A-guard(F9)", g_nc3 == n_nc3 and g_cd == 0,
        f"nc3 {g_nc3}/{n_nc3}, cd {g_cd}/0")
    scaf_bad, share_bad = [], []
    for s in ORDER:
        v3_cls = Counter(x.scaffold_id for x in seal_sc if x.stratum == s)
        for g, c in v3_cls.items():
            if tr_scaf.get(g, 0) + c > SCAFFOLD_CAP_V2[s]:
                scaf_bad.append(f"{s}/{g}: {tr_scaf.get(g, 0)}+{c} > cap "
                                f"{SCAFFOLD_CAP_V2[s]}")
        tr_cls = tr_scaf_by_s.get(s, {})
        total = sum(tr_cls.values()) + sum(v3_cls.values())
        mx = max([tr_cls.get(g, 0) + c for g, c in v3_cls.items()] + [0])
        if total and mx / total > lib.CAP + 1e-9:
            share_bad.append(f"{s}: max union share {mx/total:.3f} > {lib.CAP}")
    rec("A-scaffold-union(F9)", not scaf_bad and not share_bad,
        {"cap_violations": scaf_bad[:5], "share_violations": share_bad[:5],
         "rule": "train-v2 (disk) + seal-v3 <= SCAFFOLD_CAP_V2 per stratum; "
                 f"union max share <= {lib.CAP}"})
    seal_shas = []
    for x in seal_sc:
        seal_shas += [lib.sha(x.text_a()), lib.sha(x.text_b())]
    dup = len(seal_shas) - len(set(seal_shas))
    for name in ("train-v2", "v2-seal", "v1-seal"):
        hits = sum(1 for h in seal_shas if h in sources[name])
        rec(f"A-disjoint-{name}(F6/F10)", hits == 0,
            f"{hits} sha hits of {len(seal_shas)} against a "
            f"{len(sources[name])}-sha set"
            + (" (seal ids -> pool-origin shas; sealed file NOT opened)"
               if name.endswith("-seal") else " (disk train.jsonl)"))
    rec("A-sha-unique(F6)", dup == 0,
        f"{len(set(seal_shas))} distinct of {len(seal_shas)} (dup {dup})")
    rec("A-sha-convention(F6)", conv_ok,
        f"{conv_n}/{len(disk_pool_by_id)} v2 pool ids reproduce identical "
        "disk shas in-process (embed-convention reconstruction byte-exact; "
        "regenerated from the v2 hold SEED, sealed file not touched)")
    # G3/G7: transfer surface marking — every seal row's origin matches its
    # id prefix; the per-stratum split feeds the gate's g_spec_indist /
    # g_spec_transfer legs; NC-4 G7 discipline (old bank, fallback only
    # if disclosed).
    transfer_split = {}
    for s in ORDER:
        ids_s = [i for i in seal_ids if ho_by_id[i].stratum == s]
        transfer_split[s] = {
            "transfer": sum(1 for i in ids_s
                            if origin_of[i] == "transfer:v3"),
            "indist": sum(1 for i in ids_s
                          if origin_of[i] == "indist:classic")}
    mark_ok = all((i.startswith(SEAL3_V3_PREFIX + "-"))
                  == (origin_of[i] == "transfer:v3") for i in seal_ids)
    nc4_ids = [i for i in seal_ids if ho_by_id[i].stratum == "NC-4"]
    nc4_ok = (nc4_decision.get("fallback_rows", 0)
              == sum(1 for i in nc4_ids if origin_of[i] == "transfer:v3"))
    rec("A-origin-marking(G3/G7)", mark_ok and nc4_ok,
        {"transfer": origins_total.get("transfer:v3", 0),
         "indist": origins_total.get("indist:classic", 0),
         "per_stratum": transfer_split, "nc4": nc4_decision,
         "nc1_leg": "G3: NC-1 transfer-only (in-dist leg does not exist) "
                    f"— actual split {transfer_split.get('NC-1')}"})
    rec("A-frozen", True, f"nano {lib.EMBEDDER_PIN[:12]}…; cortex features "
        "imported; instrument pin verified at Cos init")
    rec("A-archive-integrity", all(archives_intact.values()),
        {**{k: ("digest-ok" if v else "RED") for k, v in archives_intact.items()},
         "note": "whole-file sha256 digests only — no content parsed"})

    all_ok = all(v["ok"] for v in A.values())
    if not all_ok:
        reds = [k for k, v in A.items() if not v["ok"]]
        log3("build_stop", reason=f"asserts RED: {reds}")
        log3_md("build_stop", reds=reds,
                note="stop-first: NOTHING written or sealed")
        print(f"BUILD-STOP: asserts RED: {reds}", file=sys.stderr)
        return 1

    # --- stop-first passed: emit (pool-only: NO train, NO sheet) -----------
    rows_h = []
    for sc in seal_sc:
        rows_h += rows_of(sc, "holdout", f"seed:{SEED_HOLD_V3}")
    fp_hold = write_jsonl(CORPUS_DIR_V3 / "holdout.jsonl", rows_h)
    pool_rows = []
    for sc in list(hold_pool) + list(hn1) + list(hn2) + list(hn3):
        pool_rows += rows_of(sc, "holdout-pool", f"seed:{SEED_HOLD_V3}")
    fp_pool = write_jsonl(CORPUS_DIR_V3 / "holdout-pool.jsonl", pool_rows)
    floor3 = min(x._cos for x in seal_sc)
    lineage = {
        "rotation_of": "CD-holdout-v2",
        "ancestor_holdout_sha256": man2["fingerprints"]["holdout.jsonl"],
        "ancestor_seal_sha256": man2["fingerprints"]["seal"],
        "ancestor_status": "burned",
        "burn_reason": "post-read instrument crash (ADDENDUM-5 §1-2): sealed "
                       "read consumed 2026-11-10 23:41:19.281 (sha c92a2caa…), "
                       "float32 pair_cos overshoot 1.0000001105987988 escaped "
                       "to traceback/stdout; zero gate numbers published",
        "ancestor_gate_state": "verdict NOT REACHED; gate_done absent in "
                               "run-log (F3 freeze leg re-verified at runtime)",
        "rotation_rule": "slice rule (fresh generation), NOT corpus amend-once: "
                         "E13 amend budget NOT spent; train corpus v2 untouched "
                         "byte-for-byte (F7)",
        "cut_rule": "SINGLE cut of 180 (SEAL_SHARES_V2 largest remainder, "
                    "hardest-first within stratum, sha-only culling F6, "
                    "union scaffold caps, G10 census-gated); one seed family "
                    "member (hold=6117); no re-cut on this artifact",
        "universes_extension": "ADDENDUM-6 path B (ratified MLC): NC-1/NC-3/"
                               "NC-5 drawn from the gen_pool_v3 eval-only fork "
                               f"banks (frozen commit {bv['banks_commit'][:12]}, "
                               "content sha in pin prep); NC-4 old-bank per G7 "
                               f"({nc4_decision.get('bank')}); new-bank rows "
                               "are the G3 transfer surface — split per "
                               "stratum in pin transfer_split / manifest",
        "disclosure_unseen_scaffold": "holdout v3 carries unseen-scaffold rows "
                                      "of the extended classes (ADDENDUM-6 §2.4/"
                                      "G12); the model card known-limitations "
                                      "must repeat this line verbatim",
        "characterization_anchor": "E2 pre-cut characterization GREEN "
                                   "(|Delta|=0.0 all strata, run a9ac15d): "
                                   "gate RED would mean model defect, not "
                                   "bank drift",
        "f1_single_shot": "a red gate v3 is FINAL — no v4 on the same "
                          "artifact+thresholds (F1)",
    }
    seal_payload = {"ts": now(), "version": "CD-holdout-v3",
                    "holdout_jsonl_sha256": fp_hold,
                    "ids": sorted(set(seal_ids)), "n": len(seal_sc),
                    "lineage": lineage,
                    "disjointness": {
                        "rule": "sha(text_a/b) ONLY (F6); scenario_id excluded "
                                "(deterministic template grid id-space)",
                        "checked_against": {
                            "train-v2": "disk train.jsonl text-sha set",
                            "CD-holdout-v2": "seal ids -> holdout-pool.jsonl "
                                             "origin text-shas",
                            "CD-holdout-v1": "seal ids -> holdout-pool.jsonl "
                                             "origin text-shas"},
                        "sealed_content_reads": {
                            "corpus-v1/holdout.jsonl": 0,
                            "corpus-v2/holdout.jsonl": 0}},
                    "archives_digest_intact": {k: bool(v) for k, v
                                               in archives_intact.items()},
                    "note": "content NOT read after this event"}
    SEALS_DIR.mkdir(parents=True, exist_ok=True)
    SEAL_V3_PATH.write_text(json.dumps(seal_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    fp_seal = lib.sha(SEAL_V3_PATH.read_text(encoding="utf-8"))
    # EXTEND the existing pin (the G1 freeze prep section is history —
    # never rewritten), then merge the seal3 main section.
    pin = {}
    if PIN_V3_PATH.exists():
        pin = json.loads(PIN_V3_PATH.read_text(encoding="utf-8"))
    if pin.get("prep", {}).get("banks_v3", {}).get("content_sha256") \
            not in (None, banks_content_sha):
        die("pin-v3.json prep pins a DIFFERENT bank content than the "
            "runtime-verified banks — G1 inconsistency, refusing")
    pin.update({
        "ts": now(), "version": 3,
        "lane": "seal3 pool-only (F7, ADDENDUM-6 path B): fresh hold-seed "
                "generation over classic + v3 bank universes -> G10 census "
                "-> single cut of 180 -> seal; train.jsonl/sheet ABSENT BY "
                "CONSTRUCTION; measure2/build2 untouched byte-for-byte",
        "instrument": {"pin": lib.EMBEDDER_PIN,
                       "guard_version": lib.GUARD_VERSION},
        "seeds": {"hold": SEED_HOLD_V3},
        "seed_family_check": "single-member family (hold only; train/sheet members "
                             "absent by construction, F8); != {2113,7919,9377}, "
                             "!= {42,777,4242} ∪ {43..52}, != all "
                             f"{len(observed)} seeds seen in seeds_pinned* "
                             "run-log events",
        "runlog_seed_scan": {"events": len(scan),
                             "distinct_seeds": len(observed), "seeds": observed},
        "banks_v3_frozen": {
            "banks_commit": bv["banks_commit"],
            "file_sha256": bv["file_sha256"],
            "content_sha256": banks_content_sha,
            "lock": "G1: bank edits forbidden; census/gate failure = "
                    "BuildStop + new addendum",
            "runtime_verified": True},
        "hard_window": [HARD_WIN_LO, HARD_WIN_HI],
        "measured_holdout_floor": floor3,
        "census": cen,
        "g10_census": {s: g10[s] for s in list(ORDER) + ["N1", "N2", "N3"]},
        "seal_shares": {**{s: SEAL_SHARES_V2[s] for s in ORDER},
                        "N1": sum(SEAL_N1_HOSTS.values()),
                        "N2": SEAL_SHARES_V2["N2"], "N3": N3_SEAL},
        "cut_rule": lineage["cut_rule"],
        "transfer_split": {
            "rule": "G3: rows from the NEW banks (id prefix "
                    f"{SEAL3_V3_PREFIX}-) are the transfer surface; rows "
                    "from the OLD bank are in-dist; the v3 gate report "
                    "splits g_spec_indist / g_spec_transfer by this "
                    "marking; thresholds unchanged",
            "per_stratum": transfer_split,
            "nc1_disclosure": "NC-1 transfer-only (in-dist leg does not "
                              "exist per G3)",
            "nc4": nc4_decision},
        "seal_fold_strata": dict(cnt),
        "seal_ids": seal_ids,
        "bands_seal": {b: sum(1 for x in seal_sc if x._band == b)
                       for b, _, _ in BANDS},
        "reporting_rule": SEAL3_REPORTING_RULE,
        "lineage": lineage,
        "train_v2_untouched": {"train.jsonl_sha256_start": train_fp_start},
    })
    CORPUS_DIR_V3.mkdir(parents=True, exist_ok=True)
    PIN_V3_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    manifest = {
        "ts": now(),
        "version": "corpus-v3-holdout (ADDENDUM-5 seal rotation + ADDENDUM-6 "
                   "path B universes extension, B6-conflict seal3)",
        "guard_version": lib.GUARD_VERSION,
        "instrument": {"provider": "vesma.embeddings.NanoProvider",
                       "nano_pin": lib.EMBEDDER_PIN},
        "design": ["docs/experiments/conflict-detector-gate-incident-2026-11-10.md "
                   "(ADDENDUM-5 RATIFIED, §4 F6-F12)",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-6.md "
                   "(RATIFIED MLC, path B, G1-G12)",
                   "docs/experiments/conflict-detector-corpus-design.md (RATIFIED MLC-9)",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-1.md",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-2.md",
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-3.md"],
        "banks_v3_frozen": {"banks_commit": bv["banks_commit"],
                            "file_sha256": bv["file_sha256"],
                            "content_sha256": banks_content_sha,
                            "runtime_verified": True},
        "pin_sha256": lib.sha(PIN_V3_PATH.read_text(encoding="utf-8")),
        "seeds": {"hold": SEED_HOLD_V3},
        "seed_family_check": pin["seed_family_check"],
        "emitted": ["holdout.jsonl (180 pairs, sealed)",
                    f"holdout-pool.jsonl ({len(pool_rows)} rows, wide "
                    "generation provenance)", "pin-v3.json",
                    "seals/CD-holdout-v3.seal.json"],
        "absent_by_construction": ["train.jsonl", "sheet/*"],
        "pool_scope": "the full generated hold-side pools (classic "
                      f"{pool_classic_n} + v3 banks {pool_v3_n} = "
                      f"{len(hold_pool)}, N1 {len(hn1)}, N2 {len(hn2)}, "
                      f"N3 {len(hn3)} scenarios) — the v2-era registered-600 "
                      "window does not exist in the pool-only lane; every "
                      "seal id resolves in this pool file",
        "pool_sizes": {"classic": pool_classic_n, "v3_banks": pool_v3_n,
                       "merged": len(hold_pool), "N1": len(hn1),
                       "N2": len(hn2), "N3": len(hn3)},
        "transfer_split": {
            "rule": "G3: new-bank rows (prefix " + SEAL3_V3_PREFIX +
                    "-) = transfer surface; the gate report splits "
                    "g_spec_indist / g_spec_transfer by this marking",
            "per_stratum": transfer_split, "nc4": nc4_decision},
        "cut_depth": depth_report,
        "strata_seal": {s: {"classic": classic_cnt[s],
                            "carrier": {"N1": sum(1 for x in n1_sc
                                                   if x.stratum == s),
                                        "N2": sum(1 for x in n2_sc
                                                  if x.stratum == s),
                                        "N3": sum(1 for x in n3_sc
                                                  if x.stratum == s)},
                            "fold": cnt[s]} for s in ORDER if cnt[s]},
        "carrier_slices": {"N1": {"hosts": dict(Counter(x.stratum
                                                        for x in n1_sc)),
                                  "k_mix": dict(Counter(len(x.benign)
                                                        for x in n1_sc))},
                           "N2": {"quota": len(n2_sc),
                                  "k_mix": dict(Counter(len(x.benign)
                                                        for x in n2_sc))},
                           "N3": {"quota": len(n3_sc)}},
        "disjointness": {"rule": "sha(text_a/b) only (F6)",
                         "culled": culled, "cull_by_src": dict(cull_by_src),
                         "sha_set_sizes": {k: len(v)
                                           for k, v in sources.items()},
                         "sealed_files_opened": 0},
        "census": cen,
        "g10_census": {s: g10[s] for s in list(ORDER) + ["N1", "N2", "N3"]},
        "asserts": A,
        "lineage": lineage,
        "reporting_rule": SEAL3_REPORTING_RULE,
        "para_dict_sha256": lib.PARA_DICT_SHA,
        "bands_seal": pin["bands_seal"],
        "known_uncovered": {"cos_below": floor3,
                            "note": "measured floor over the emitted v3 seal"},
        "seal_intact_inputs": {"v1_archive": "untouched (F2: permanent "
                                             "archive, digest-verified, "
                                             "0 content reads)",
                               "v2_archive": "burned; sealed file unopened "
                                             "by this lane (digest-verified "
                                             "only)"},
        "regime": "synthetic-only; no store texts; no LLM in construction; "
                  "labels by construction",
        "fingerprints": {"holdout.jsonl": fp_hold,
                         "holdout-pool.jsonl": fp_pool, "seal": fp_seal},
    }
    (CORPUS_DIR_V3 / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # byte-identity tail check: the lane proves it did not touch build2 outputs
    train_fp_end = file_sha256(CORPUS_DIR_V2 / "train.jsonl")
    untouched = train_fp_end == train_fp_start \
        == man2["fingerprints"]["train.jsonl"]
    n_pass = sum(1 for v in A.values() if v["ok"])
    log3("seal_v3", seal="CD-holdout-v3", n=len(seal_sc), sha=fp_seal,
         holdout_sha=fp_hold, pool_sha=fp_pool,
         lineage_ancestor="CD-holdout-v2 (c92a2caa…, burned)",
         holdout_content_reads=0)
    log3_md("seal_v3", seal="CD-holdout-v3", n=len(seal_sc), sha=fp_seal,
            holdout_sha=fp_hold, seal_n=len(seal_sc),
            asserts=f"{n_pass}/{len(A)} PASS", holdout_content_reads=0)
    log3("seal3_done", ok=True, train_v2_untouched=bool(untouched),
         elapsed_s=round(time.time() - t0, 1),
         asserts=f"{n_pass}/{len(A)} PASS")
    log3_md("seal3_done", ok=True, train_v2_untouched=bool(untouched),
            elapsed_s=round(time.time() - t0, 1),
            asserts=f"{n_pass}/{len(A)} PASS")
    print(f"train-v2 byte-identity after run: {'OK' if untouched else 'RED'} "
          f"({train_fp_end[:16]})")
    print(f"\nSEAL3: ALL ASSERTS PASS in {time.time()-t0:.0f}s\n"
          f"pin: {PIN_V3_PATH}\nseal: {SEAL_V3_PATH}\n"
          f"corpus: {CORPUS_DIR_V3} (holdout + pool only)")
    return 0


# --- rows --------------------------------------------------------------------

def side_dict(side: lib.Side) -> dict:
    return {"title": side.title, "body": side.body, "tags": side.tags,
            "language": side.language, "record_type": side.record_type,
            "created_at": side.created_at}


def rows_of(sc: lib.Scenario, split: str, seed_pin: str) -> list[dict]:
    ch5 = lib.pair_char5_jaccard(sc.side_a, sc.side_b, sc._cos)
    base = {"pair_id": sc.scenario_id, "scenario_id": sc.scenario_id,
            "stratum": sc.stratum, "label": sc.label,
            "carrier": sc.carrier,
            "declaration": {"slots": sc.slots, "subject_a": sc.subject_a,
                            "subject_b": sc.subject_b, "polar_op": sc.polar_op,
                            "refine_ok": sc.refine_ok,
                            "benign_regions": sc.benign},
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
    # holdout: same procedure under the UNION scaffold cap (§4.5 is
    # computed on train+seal); the registered pool = the allocation, the
    # seal = stratified 180 by largest remainder (§4.2 rule), CULLED for
    # train-text disjointness (A-disjoint §5.3: 0 shared text hashes)
    alloc_ho = select_cells(hold_pool, HOLD_POOL,
                            cross_pos=36, cross_neg=45, corner=12,
                            prior_scaf=alloc_tr["scaf"])
    ho_ids = [i for s in ORDER for i in alloc_ho["ids"][s]][:HOLD_POOL]

    tr_by_id = {x.scenario_id: x for x in train_pool}
    tr_texts = set()
    for s in ORDER:
        for i in alloc_tr["ids"][s]:
            x = tr_by_id[i]
            tr_texts.add(lib.sha(x.text_a()))
            tr_texts.add(lib.sha(x.text_b()))
    ho_by_id = {x.scenario_id: x for x in hold_pool}

    # largest-remainder seal shares (§4.2: holdout = доли × 180)
    raw = {s: STRATA_QUOTA[s] * HOLD_SEAL / TRAIN_PAIRS for s in ORDER}
    shares = {s: int(raw[s]) for s in ORDER}
    leftovers = HOLD_SEAL - sum(shares.values())
    for s in sorted(ORDER, key=lambda s: (-(raw[s] - int(raw[s])), ORDER.index(s))):
        if leftovers <= 0:
            break
        shares[s] += 1
        leftovers -= 1

    train_ids = {i for s in ORDER for i in alloc_tr["ids"][s]}
    seal_ids = []
    culled = 0
    for s in ORDER:
        got = 0
        for i in alloc_ho["ids"][s]:
            if got >= shares[s]:
                break
            if i in train_ids:
                culled += 1
                continue
            x = ho_by_id[i]
            if lib.sha(x.text_a()) in tr_texts or lib.sha(x.text_b()) in tr_texts:
                culled += 1
                continue
            seal_ids.append(i)
            got += 1
        if got < shares[s]:
            die(f"seal disjointness: {s} exhausted ({got}/{shares[s]} after culling)")
    assert len(seal_ids) == HOLD_SEAL
    print(f"seal: {len(seal_ids)} pairs, {culled} candidates culled for "
          f"train-text collisions")

    sel_ids = set()
    for s in ORDER:
        sel_ids |= set(alloc_tr["ids"][s])
    hard_pos = sum(1 for x in train_pool
                   if x.scenario_id in sel_ids and x.label == "contradicts"
                   and HARD_WIN_LO <= x._cos < HARD_WIN_HI)
    rz_sel = [x for x in train_pool
              if x.scenario_id in sel_ids and x._band == "razor"]
    razor_pos_quota = sum(1 for x in rz_sel if x.label == "contradicts")
    razor_total = sum(c.get("razor", 0) for c in alloc_tr["cells"].values())
    # measured corpus floor over the EMITTED corpus (train + seal), not
    # the unused remainder — known-uncovered documents the corpus itself
    corpus_floor = min([x._cos for x in train_pool if x.scenario_id in sel_ids]
                       + [ho_by_id[i]._cos for i in seal_ids])
    ok = True
    print(f"\ntrain band totals: " + str({b: sum(alloc_tr['cells'][s].get(b, 0) for s in ORDER)
                                          for b, _, _ in BANDS}))
    print(f"floors: razor {razor_total} (>= {FLOOR_RAZOR}), "
          f"hard-pos [{HARD_WIN_LO},{HARD_WIN_HI}) {hard_pos} (>= {FLOOR_HARD_POS})")
    if razor_total < FLOOR_RAZOR:
        print("RED: razor floor"); ok = False
    if hard_pos < FLOOR_HARD_POS:
        print("RED: hard-pos floor"); ok = False

    pin = {
        "ts": now(), "instrument": {"pin": lib.EMBEDDER_PIN, "guard_version": lib.GUARD_VERSION},
        "measured_corpus_floor": corpus_floor,
        "seeds": {"train": SEED_TRAIN, "hold": SEED_HOLD, "sheet": SEED_SHEET},
        "hard_window": [HARD_WIN_LO, HARD_WIN_HI],
        "razor_pos_quota": razor_pos_quota,
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

    # A-count: train == exact strata quota (§4.2); seal == the share
    # table of §4.1/MLC-9 п.11 (quota × 180/600, largest remainder) —
    # the seal is ADDITIONAL fresh material, not part of the 600
    from collections import Counter
    cnt = Counter(x.stratum for x in all_sc)
    tr_cnt = Counter(x.stratum for x in tr_sel)
    ho_cnt = Counter(x.stratum for x in ho_sel)
    raw = {s: STRATA_QUOTA[s] * HOLD_SEAL / TRAIN_PAIRS for s in ORDER}
    exp_seal = {s: int(raw[s]) for s in ORDER}
    left = HOLD_SEAL - sum(exp_seal.values())
    for s in sorted(ORDER, key=lambda s: (-(raw[s] - int(raw[s])), ORDER.index(s))):
        if left <= 0:
            break
        exp_seal[s] += 1
        left -= 1
    rec("A-count", all(tr_cnt[s] == STRATA_QUOTA[s] and ho_cnt[s] == exp_seal[s]
                       for s in ORDER),
        {"train": {s: tr_cnt[s] for s in ORDER},
         "seal": {s: ho_cnt[s] for s in ORDER}})
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
    # A-guard: NC-3 100%, CD 0% on the FULL manifest (train+seal, §5.3)
    g_nc3 = sum(1 for x in all_sc if x.stratum == "NC-3" and lib.version_guard(x))
    g_cd = sum(1 for x in all_sc if x.stratum.startswith("CD") and lib.version_guard(x))
    n_nc3 = sum(1 for x in all_sc if x.stratum == "NC-3")
    rec("A-guard", g_nc3 == n_nc3 and g_cd == 0,
        f"nc3 {g_nc3}/{n_nc3}, cd {g_cd}/0")
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
        if x.label != "contradicts":
            c = x._cos
            bins["0" if c < 0.97 else ("1" if c < 0.985 else "2")] += 1
    raz_sorted = sorted(x._cos for x in rz)
    max_gap = max((b - a for a, b in zip(raz_sorted, raz_sorted[1:])), default=0.0)
    hard_pos = sum(1 for x in tr_sel if x.label == "contradicts"
                   and HARD_WIN_LO <= x._cos < HARD_WIN_HI)
    quota_pin = pin.get("razor_pos_quota", 0)
    ok_b = (razor >= FLOOR_RAZOR and rz_neg >= rz_pos
            and min(bins.values()) >= RAZOR_BIN_FLOOR
            and max_gap <= RAZOR_MAX_GAP
            and rz_pos >= quota_pin
            and hard_pos >= FLOOR_HARD_POS)
    rec("A-band-coverage", bool(ok_b),
        f"razor {razor} (pos {rz_pos}/neg {rz_neg}, pos-quota pin {quota_pin}), "
        f"bins {dict(bins)}, max-gap {max_gap:.4f}, "
        f"hard-pos [{HARD_WIN_LO},{HARD_WIN_HI}) {hard_pos}")
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
                   "docs/experiments/conflict-detector-corpus-design-ADDENDUM-2.md",
                   "docs/experiments/conflict-detector-corpus-measure-2026-11-01.md",
                   "docs/experiments/conflict-detector-corpus-measure2-2026-11-02.md"],
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
                            "note": "measured corpus floor over the emitted corpus, conventional surface (ADDENDUM-2 §B); B4 ADDENDUM-6 §2 errata cross-ref"},
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


#: Neutral render fillers for long-channel bodies in the gold sheet:
#: the corpus long channel reuses one 8-sentence filler set (known
#: limitation, see dataset card) which is unreadable for a human verdict
#: pass — the SHEET renders long bodies with these UNIQUE sentences
#: instead (>= 35 distinct per body, 0 repeated sentences). Neutral by
#: design: no dates, no versions, no conflict semantics — the verdict of
#: a pair never depends on them.
RENDER_FILLERS_RU = [
    "Отчётные формы архивируются автоматически по окончании периода.",
    "Справочники синхронизируются с корпоративным реестром по расписанию.",
    "Ошибки интеграций группируются по источнику и попадают в сводку.",
    "Журнал действий администраторов доступен в отдельной панели.",
    "Ночные обновления справочников не влияют на работу пользователей.",
    "Доступ к панели управления разграничен по проектным ролям.",
    "Служебные уведомления доставляются в общий канал дежурных.",
    "Резервные копии проверяются автоматикой по контрольным суммам.",
    "Данные мониторинга хранятся в отдельном долгосрочном хранилище.",
    "Изменения настроек фиксируются с указанием автора правки.",
    "Шаблоны отчётов настраиваются на уровне рабочей области.",
    "События сбоя собираются в единый журнал разбора инцидентов.",
    "Документация по интеграциям обновляется вместе с релизами.",
    "Очередь фоновых заданий наблюдается на общей панели дежурных.",
    "Квоты на хранение пересматриваются по итогам квартала.",
    "Внешние вызовы проходят через единый шлюз с общими лимитами.",
    "Служба поддержки получает сводку обращений в начале смены.",
    "Новые версии зависимостей апробируются на выделенном стенде.",
    "Схемы обмена данными версионируются в общем репозитории.",
    "Уведомления о деградациях уходят ответственным автоматически.",
    "Итоговые показатели собираются в еженедельный отчёт команды.",
    "Проверки целостности данных выполняются по ночному расписанию.",
    "Служебные учётные записи пересматриваются при ротации команды.",
    "Журнал изменений доступен каждому участнику проекта на чтение.",
    "Временные файлы очищаются автоматически по завершении задач.",
    "Профили окружений описаны в конфигурации рабочей области.",
    "Согласование доступов проходит в рамках общей процедуры команды.",
    "Статистика использования собирается в обезличенном виде.",
    "Резервный контур разворачивается теми же средствами автоматизации.",
    "Расписание обслуживания публикуется в общем календаре команды.",
    "Служебные метрики доступности дублируются на внешнюю панель.",
    "Инструкции по восстановлению проверяются учебными прогонами.",
    "Журнал рассылок сопровождения хранится в общем архиве команды.",
    "Обновления документации согласуются с владельцами компонентов.",
    "Проверочные отчёты формируются перед каждым плановым окном.",
    "Очереди уведомлений наблюдаются по глубине и возрасту записей.",
    "Доступы пересматриваются по окончании проектных этапов.",
    "Служебные сценарии автоматизации хранятся рядом с проектом.",
    "Итоги проверок целостности публикуются в канале дежурных.",
    "Сводки по ошибкам рассылаются ответственным по расписанию.",
]

#: Worked examples for the sheet preamble (ADDENDUM-7 §4 issue 1, §3.3):
#: one worked example per class, instructive material — the class names
#: here are allowed; blindness applies to the 48 scorable blocks only.
WORKED_EXAMPLES = [
    ("CD-1 numeric", "несовместимые значения одной числовой оси слота",
     "A: «Лимит запросов к сервису — 200 в минуту.» / B: «Лимит запросов к сервису — 1000 в минуту.»",
     "CONTRADICTS", "одна числовая ось, значения несовместимы"),
    ("CD-2 date", "несовместимые даты/сроки одного событийного слота тела",
     "A: «Обновление намечено на 5 октября.» / B: «Обновление намечено на 15 октября.»",
     "CONTRADICTS", "одно событие — два несовместимых срока"),
    ("CD-3 status", "несовместимые состояния статусного словаря",
     "A: «Модуль активен в реестре сервисов.» / B: «Модуль архивирован в реестре сервисов.»",
     "CONTRADICTS", "статусы взаимно исключающие"),
    ("CD-4 modal", "полярность модальности над одной способностью",
     "A: «Виджет оплаты допускает анонимный доступ к ресурсам.» / B: «Виджет оплаты требует авторизацию для доступа к ресурсам.»",
     "CONTRADICTS", "allow vs require над одной способностью"),
    ("CD-5 conditional", "безусловное отрицание условного клейма на домене условия",
     "A: «При включённом флаге multipart модуль режет большие файлы на части.» / B: «Модуль никогда не резать большие файлы на части, независимо от флага multipart.»",
     "CONTRADICTS", "безусловное «никогда не» против условного «при флаге»"),
    ("CD-6 semantic", "качественное несовместимое предикат-пара (полярный оператор)",
     "A: «Компонент даёт заметный прирост скорости.» / B: «Компонент замедлил работу системы.»",
     "CONTRADICTS", "полярные оценки одного эффекта"),
    ("NC-1 addition", "вставка слота-надмножества, конъюнкция выполнима",
     "A: обычное описание сервиса. / B: то же + «Дополнительно поддерживается выгрузка результатов в CSV.»",
     "NO-CONFLICT", "добавка не отменяет сказанное — оба утверждения совместимы"),
    ("NC-2 refinement", "совместимое уточнение v |= c",
     "A: «Лимит запросов — не более 1000 в минуту.» / B: «Лимит запросов — 800 в минуту.»",
     "NO-CONFLICT", "800 укладывается в «не более 1000» — уточнение, не спор"),
    ("NC-3 supersede", "version-namespace соседи — supersede-семантика",
     "A: «Текущая версия модуля — v1.2.0.» / B: «Текущая версия модуля — v1.3.0.»",
     "NO-CONFLICT", "различие только в номере версии — смена поколения, не конфликт фактов"),
    ("NC-4 metadata", "дельта метаданных при тождественном содержании",
     "A и B: тела слово в слово одинаковы; различаются только тег/тип записи.",
     "NO-CONFLICT", "содержание тождественно — спорить нечему"),
    ("NC-5 theme", "разные предметы (sibling/far), конфликтировать нечему",
     "A: запись о планировщике фоновых задач. / B: запись о рецепте борща.",
     "NO-CONFLICT", "разные предметы — пересечение тем не создаёт противоречия"),
]


def _render_body(body: str, pair_id: str, channel: str) -> str:
    """Sheet readability render: replace REPEATED sentences (the corpus
    long channel reuses one filler set — known limitation) with unique
    neutral sentences from RENDER_FILLERS_RU. Slot/basis sentences are
    unique already and pass through untouched. Short/medium bodies are
    duplicate-free by construction (asserted)."""
    sents = _sents(body)
    dup = len(sents) - len(set(sents))
    if dup == 0:
        return body
    rng = random.Random(f"sheet-render:{pair_id}")
    order = list(range(len(RENDER_FILLERS_RU)))
    rng.shuffle(order)
    seen: set[str] = set(sents)
    out = []
    j = 0
    for s in sents:
        if s in seen and out and s in set(out):
            while j < len(order) and RENDER_FILLERS_RU[order[j]] in seen:
                j += 1
            if j >= len(order):
                die(f"sheet render: filler pool exhausted for {pair_id}")
            s = RENDER_FILLERS_RU[order[j]]
            j += 1
        seen.add(s)
        out.append(s)
    rendered = " ".join(out)
    if channel == "long" and len(rendered) < 2000:
        die(f"sheet render: long body for {pair_id} degraded below 2000 chars")
    return rendered


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
    lines = ["GOLD SHEET CD-v1 — blind render (48 scorable blocks; verdict per block:",
             "CONTRADICTS / NO-CONFLICT + slot basis; block classes and declarations withheld)", ""]
    lines += ["=" * 72,
              "ИНСТРУКТАЖ — определения классов с примерами (не скорируется;",
              "в 48 блоках ниже классы не называются). Ориентир времени: ~16 мин.", ""]
    for cls, definition, example, verdict, basis in WORKED_EXAMPLES:
        lines.append(f"{cls} — {definition}")
        lines.append(f"  пример: {example}")
        lines.append(f"  вердикт: {verdict} (основание: {basis})")
        lines.append("")
    lines += ["=" * 72, "СКОРИРУЕМЫЕ БЛОКИ (48):", ""]
    meta = []
    n_dedup = 0
    for i, idx in enumerate(order):
        r = picks[idx]
        a, b = r["envelope_a"], r["envelope_b"]
        ch = r["measured"]["len_channel"]
        body_a = _render_body(a["body"], r["pair_id"] + ":a", ch)
        body_b = _render_body(b["body"], r["pair_id"] + ":b", ch)
        for txt in (body_a, body_b):
            if len(_sents(txt)) != len(set(_sents(txt))):
                die(f"sheet render: duplicate sentences left in {r['pair_id']} ({ch})")
        if ch in ("short", "medium") and (body_a != a["body"] or body_b != b["body"]):
            n_dedup += 1  # medium-borderline NC-1 30-insert bodies (see card)
        lines.append(f"N{i+1:02d}")
        lines.append(f"  A [{a['language'] or '—'}/{a['record_type'] or '—'}] {a['title']}")
        lines.append(f"     {body_a}")
        lines.append(f"  B [{b['language'] or '—'}/{b['record_type'] or '—'}] {b['title']}")
        lines.append(f"     {body_b}")
        lines.append("  verdict: ______   basis: ______________________")
        lines.append("")
        meta.append({"block": f"N{i+1:02d}", "pair_id": r["pair_id"], "stratum": r["stratum"],
                     "label": r["label"], "len_channel": ch,
                     "render_dedup": body_a != a["body"] or body_b != b["body"],
                     "role": "confusable" if idx < 16 else
                     ("twin" if idx < 18 else ("reserve" if idx >= 43 else "cover"))})
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    (SHEET_DIR / "gold-sheet-render-v1.txt").write_text("\n".join(lines), encoding="utf-8")
    sheet_json = {"ts": now(), "n": len(meta),
                  "render_note": "long-channel bodies rendered with unique neutral fillers "
                                 "(corpus long channel repeats one filler set — known limitation, "
                                 "dataset card); slot/basis sentences untouched; 0 repeated "
                                 "sentences per rendered body (asserted)",
                  "blocks": meta}
    sp = SHEET_DIR / "gold-sheet-v1.json"
    sp.write_text(json.dumps(sheet_json, ensure_ascii=False, indent=2), encoding="utf-8")
    fp = lib.sha(sp.read_text(encoding="utf-8"))
    log("sheet_frozen", n=len(meta), fp=fp)
    print(f"SHEET: {len(meta)} blocks, fp {fp[:16]}… — TL FORMAT GATE pending; NOT owner-ready")
    print(f"render: dedup applied to {sum(1 for m in meta if m['render_dedup'])} blocks; "
          f"medium-borderline hits (NC-1 30-insert dial): {n_dedup}")
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
  перекройка лестницы по измеренному пределу прибора) + ADDENDUM-2
  (`8c756a0`, трудное окно [0.70, 0.75) как ограничение населения банда,
  конвенционные замеры) + measure-отчёты (`87a248e`, `deb440b`).
  Манифест: `data/conflict-detector/corpus-v1/build-manifest.json`
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

## Золото: владелец вне контура (решение 02.11.2026)

Владелец отказался от личной калибровки золотого листа (02.11.2026,
дословно: «не буду этим заниматься, у тебя есть ML-специалисты, пусть
занимаются, комитет перепроверит»). Активирован ратифицированный фолбэк
OQ-1: **золото = комитет по ADDENDUM-1**, гейт g_gold на месте
(≥ 45/50 + 0 системного паттерна на конфузиблах).

**Ограничение независимости:** gold-метки производны от by-construction
меток того же генератора, что и train/holdout — судейская нога не
независима от конструкции. Компенсации: (1) механический предикат
§3.2.1 вне судейского счёта (проверяет тела, не метки); (2) двойной
аудит «декларация vs тело» ≥ 30 пар + LLM-нога (слепой рендер, локальная
модель, пин + хэш промпта до прогона); (3) B0-телеметрия после миграции
(§7) как независимый живой сигнал.

## Известные ограничения (known limitations)

- **Long-канал тел корпуса использует повторяющийся филлер**: тела
  ≥ 2000 символов собраны из одного набора из 8 нейтральных
  предложений-филлеров, повторённого 4–8 раз; то же свойство у
  30-вставочного диала NC-1, тела которого лежат у верхней границы
  medium-канала (длина канала и нейтральность сохранены, конфликтные
  слоты не затронуты). Для человека-верdict-инструмента это нечитаемо —
  золотой лист рендерит такие тела с уникальными филлерами (см.
  render_note в gold-sheet-v1.json). Короткие тела и прочие средние —
  без повторов (проверено рендером листа).
- **Watch-item (characterization до обучения):** если модель выродится
  на длинных неповторяющихся текстах (деградация long-канала против
  short/medium на characterization-пробах), — генераторный фикс пула
  филлеров ДО обучения; окно есть: обучение стартует после миграции
  движковой ноги (§7). Короткие/средние каналы повторов не содержат
  (проверено рендером листа, 0 замен).

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


def n2_mirror_of(n1_sc: lib.Scenario, c: object, v: object,
                 mirror_id: str) -> lib.Scenario:
    """Build the E10 confusable twin of an N1 train cell: SAME prefix with
    the SAME benign substitutions, slot sentences swapped from the
    conflict pair to a satisfiable NC-2 refinement (c |= v) — the label
    differs ONLY by slot compatibility (same dialect, same k, same side)."""
    axis = n1_sc.slots[0]["axis"]
    key = n1_sc.slots[0]["slot_id"]
    if key == "upd2":
        key = "upd"  # alternate date template maps onto the same refine axis
    c_sent, v_sent, check = lib.REFINE_RU[key]
    subj_fmt = {"gen": lib.SUBJECTS[n1_sc.subject_a][2],
                "dat": lib.SUBJECTS[n1_sc.subject_a][3]}
    body_a = lib.benign_prefix_of(n1_sc, "a") + " " + c_sent.format(c=c, **subj_fmt)
    body_b = lib.benign_prefix_of(n1_sc, "b") + " " + v_sent.format(v=v, **subj_fmt)
    side_a = lib.Side(title=n1_sc.side_a.title, body=body_a,
                      tags=list(n1_sc.side_a.tags), language="ru",
                      record_type=n1_sc.side_a.record_type,
                      created_at=n1_sc.side_a.created_at)
    side_b = lib.Side(title=n1_sc.side_b.title, body=body_b,
                      tags=list(n1_sc.side_b.tags), language="ru",
                      record_type=n1_sc.side_b.record_type,
                      created_at=n1_sc.side_b.created_at)
    sc = lib.Scenario(
        scenario_id=mirror_id, stratum="NC-2", side_a=side_a, side_b=side_b,
        label="not-contradicts",
        slots=[{"slot_id": key, "axis": axis, "value_a": str(c),
                "value_b": str(v), "op": "replace", "is_version_token": False,
                "span_chars": max(len(str(v)), 6)}],
        subject_a=n1_sc.subject_a, subject_b=n1_sc.subject_a,
        polar_op=None, refine_ok=bool(check(v, c)),
        scaffold_id=f"sheet-mirror:{key}:{n1_sc.subject_a}",
        cross_lang=False, length_channel=lib.channel_for(body_a),
        both_null=False, carrier="N2",
        benign=[dict(d) for d in n1_sc.benign], para_b=0)
    return sc


def cmd_preflight2() -> int:
    """П3 v2: fixture cell of EVERY class (classic 11 + carriers N1/N2/N3)
    against the FULL assert map, BEFORE any emission (stop-first)."""
    print("== П3-v2 preflight: fixture cells x full assert map ==")
    r = random.Random(0)
    fx: list[lib.Scenario] = []
    fx.append(lib.build_scenario("fx2-CD-1", "CD-1", "s3conn", "ru",
                                 lib.NUM_SLOTS[0], 500, 1000, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-CD-2", "CD-2", "retry", "ru",
                                 (lib.DATE_SLOT[0], lib.DATE_SLOT[1], ""),
                                 "5 октября", "7 октября", fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-CD-3", "CD-3", "sched", "ru",
                                 lib.STATUS_SLOTS[0], None, None, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-CD-4", "CD-4", "auth", "ru",
                                 lib.MODAL_SLOTS[0], None, None, fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-CD-5", "CD-5", "cache", "ru", None,
                                 lib.COND_SLOTS[0][2], lib.COND_SLOTS[0][3],
                                 fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-CD-6", "CD-6", "recs", "ru", None, None,
                                 None, polar=lib.POLAR_SLOTS[0], fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-NC-1", "NC-1", "export", "ru", None, None,
                                 None, insert_sent=lib.INSERT_RU[0], fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-NC-2", "NC-2", "pay", "ru", None, 800, None,
                                 refine=("lim", 1000), fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-NC-3", "NC-3", "logs", "ru", None, None, None,
                                 version_pair=("v1.2.0", "v1.3.0"), fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-NC-4", "NC-4", "queue", "ru", None, None, None,
                                 meta_delta="tags", fillers=1, rng=r))
    fx.append(lib.build_scenario("fx2-NC-5", "NC-5", "s3conn", "ru", None, None, None,
                                 sibling_subj="cook", fillers=1, rng=r))
    # carriers: N1 on EVERY host, both k; N2; N3 numeric + date
    d = lib.build_scenario("fx2-N1a", "CD-1", "mlpipe", "ru", lib.NUM_SLOTS[0],
                           200, 500, fillers=1,
                           rng=random.Random("fx-n1a"))
    d.carrier = "N1"
    fx.append(lib.apply_benign(d, "b", 1, random.Random("fx-n1a-v")))
    d = lib.build_scenario("fx2-N1b", "CD-1", "pay", "ru", lib.NUM_SLOTS[1],
                           30, 60, fillers=1, rng=random.Random("fx-n1b"))
    d.carrier = "N1"
    fx.append(lib.apply_benign(d, "a", 2, random.Random("fx-n1b-v")))
    d = lib.build_scenario("fx2-N1c", "CD-2", "auth", "ru",
                           (lib.DATE_SLOT[0], lib.DATE_SLOT[1], ""),
                           "5 октября", "15 октября", fillers=1,
                           rng=random.Random("fx-n1c"))
    d.carrier = "N1"
    fx.append(lib.apply_benign(d, "a", 1, random.Random("fx-n1c-v")))
    d = lib.build_scenario("fx2-N1d", "CD-4", "queue", "ru",
                           lib.MODAL_SLOTS[0], None, None, fillers=1,
                           rng=random.Random("fx-n1d"))
    d.carrier = "N1"
    fx.append(lib.apply_benign(d, "b", 2, random.Random("fx-n1d-v")))
    d = lib.build_scenario("fx2-N2", "NC-2", "logs", "ru", None, 60, None,
                           refine=("timo", 120), fillers=1,
                           rng=random.Random("fx-n2"))
    d.carrier = "N2"
    fx.append(lib.apply_benign(d, "b", 1, random.Random("fx-n2-v")))
    fx.append(lib.build_n3("fx2-N3-num", "sched", lib.NUM_SLOTS[2], 1))
    fx.append(lib.build_n3("fx2-N3-date", "bill",
                           (lib.DATE_SLOT[0], lib.DATE_SLOT[1], lib.DATE_SLOT[2],
                            lib.DATE_SLOT[3], lib.DATE_SLOT[4]), 2))
    ok = True
    print("| cell | mechanical | E4 len | guard(expect) | label | regions | slots+benign |")
    print("|---|---|---|---|---|---|---|")
    for sc in fx:
        mech = lib.mechanical_ok(sc)
        # E4 scope is N1/N2 cells only (ratified); NC-5 declares its
        # channel on the summed-body convention (v1 behavior, untouched)
        len4 = lib.len_channel_ok(sc) if sc.carrier in ("N1", "N2") else True
        guard = lib.version_guard(sc)
        want_guard = sc.stratum == "NC-3"
        lab = sc.label
        regs = len(lib.regions(sc.side_a.body, sc.side_b.body))
        expect = f"{len(sc.slots)}+{len(sc.benign)}" if sc.benign else str(len(sc.slots))
        row_ok = (mech and len4 and (guard == want_guard)
                  and lab == ("contradicts" if sc.stratum.startswith("CD")
                              else "not-contradicts"))
        ok &= row_ok
        print(f"| {sc.carrier or sc.stratum} | {mech} | {len4} | {guard}({want_guard}) "
              f"| {lab} | {regs} | {expect} |" + ("" if row_ok else "  <-- FAIL"))
    ok &= bool(lib.FACT_ALPHA) and bool(lib.PARA_DICT_SHA["ru"])
    print(f"import-level E2: FACT_ALPHA {len(lib.FACT_ALPHA)} values, "
          f"para-dict sha ru {lib.PARA_DICT_SHA['ru'][:12]}… en {lib.PARA_DICT_SHA['en'][:12]}…")
    print("\nassert map applied per cell: A-silver(carrier-aware) + E2 per-pair "
          "no-digit/no-fact + E4 len-channel + A-labels + A-guard expectation")
    log("preflight2_done", ok=ok)
    log_md("preflight_v2", ok=ok, cells=len(fx),
           note="fixture cells of every class + carriers, full assert map, pre-emission")
    print(f"\nPREFLIGHT-V2: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_gold2() -> int:
    """E10 delta-gold: 12 blocks (>=4 N1-vs-N2 confusable twins, all three
    N1 host paths, mandatory RED-form CD-2+k=1+razor), blind render,
    worked examples of the 3 carriers, sealed key. Committee legs score
    the BLIND render; labels live only in the sealed key."""
    train_path = CORPUS_DIR_V2 / "train.jsonl"
    if not train_path.exists():
        die("gold2: corpus-v2/train.jsonl missing — run build2 first")
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l]
    a_rows = [r for r in rows if r["orientation"] == "a>b"]
    rng = random.Random(SEEDS_V2["sheet"])

    def row_to_sc(r: dict) -> lib.Scenario:
        ea, eb = r["envelope_a"], r["envelope_b"]
        dec = r["declaration"]
        return lib.Scenario(
            scenario_id=r["pair_id"], stratum=r["stratum"],
            side_a=lib.Side(**{k: ea[k] for k in ("title", "body", "tags",
                                                  "language", "record_type",
                                                  "created_at")}),
            side_b=lib.Side(**{k: eb[k] for k in ("title", "body", "tags",
                                                  "language", "record_type",
                                                  "created_at")}),
            label=r["label"], slots=dec["slots"],
            subject_a=dec["subject_a"], subject_b=dec["subject_b"],
            polar_op=dec.get("polar_op"), refine_ok=dec.get("refine_ok"),
            scaffold_id=r["scaffold_id"],
            cross_lang=(ea["language"] != eb["language"]),
            length_channel=r["measured"]["len_channel"], both_null=False,
            carrier=r.get("carrier"), benign=dec.get("benign_regions", []))

    by_carrier: dict[str, list] = {"N1": [], "N2": [], "N3": []}
    for r in a_rows:
        if r.get("carrier"):
            by_carrier[r["carrier"]].append(r)
    # N1 picks: twins on CD-1 (lim k1, timo k2, lim k2), RED-form CD-2 k1, CD-4 k1
    def n1_pick(host: str, axis_hint: str, k: int, exclude: set[str]) -> dict:
        cands = [r for r in by_carrier["N1"]
                 if r["stratum"] == host and len(r["declaration"]["benign_regions"]) == k
                 and r["pair_id"] not in exclude]
        if axis_hint:
            cands = [r for r in cands if r["declaration"]["slots"][0]["slot_id"] == axis_hint]
        if not cands:
            cands = [r for r in by_carrier["N1"]
                     if r["stratum"] == host and len(r["declaration"]["benign_regions"]) == k
                     and r["pair_id"] not in exclude]
        if not cands:
            die(f"gold2: no N1 candidate {host} k={k}")
        return sorted(cands, key=lambda r: r["pair_id"])[0]

    used: set[str] = set()
    t1 = n1_pick("CD-1", "lim", 1, used); used.add(t1["pair_id"])
    t2 = n1_pick("CD-1", "timo", 2, used); used.add(t2["pair_id"])
    red = n1_pick("CD-2", "upd", 1, used); used.add(red["pair_id"])
    t4 = n1_pick("CD-1", "lim", 2, used); used.add(t4["pair_id"])
    n1_cd4 = n1_pick("CD-4", "", 1, used); used.add(n1_cd4["pair_id"])
    red_cos = red["measured"]["cos_target"]
    if red_cos < 0.95:
        die(f"gold2: RED-form block cos {red_cos} outside razor")
    REFINE_C = {"lim": 1000, "timo": 120, "upd": "октябрь"}
    REFINE_V = {"lim": 800, "timo": 60, "upd": "5 октября"}
    blocks: list[dict] = []
    for i, (r, role, twin_axis) in enumerate(
            [(t1, "twin", "lim"), (t2, "twin", "timo"), (red, "red-form", "upd"),
             (t4, "twin", "lim")]):
        sc = row_to_sc(r)
        raw_ax = sc.slots[0]["slot_id"]
        ax = "upd" if raw_ax == "upd2" else raw_ax
        mir = n2_mirror_of(sc, REFINE_C[ax], REFINE_V[ax],
                           f"gold-mir-{i+1}")
        if not (lib.mechanical_ok(mir) and mir.refine_ok):
            die(f"gold2: twin mirror {i+1} fails mechanical map")
        blocks.append({"row": r, "role": "n1-" + role, "sc": sc})
        blocks.append({"row": None, "role": "n2-mirror", "sc": mir,
                       "twin_of": r["pair_id"]})
    sc4 = row_to_sc(n1_cd4)
    if not lib.mechanical_ok(sc4):
        die("gold2: CD-4 N1 block fails mechanical map")
    blocks.append({"row": n1_cd4, "role": "n1-host-cover", "sc": sc4})
    # N3 x2 (numeric + date — the two distinct axes the census carries) +
    # 1 classic CD-2 block (old-class generator drift control; E10 brief:
    # remainder "1-2 N3 blocks AND old classes"). Mechanical-checked.
    n3_sorted = sorted(by_carrier["N3"], key=lambda r: r["pair_id"])
    n3_picked: list[dict] = []
    seen_axes: set[str] = set()
    for r in n3_sorted:
        ax = r["declaration"]["slots"][0]["axis"]
        if ax in seen_axes or len(n3_picked) >= 2:
            continue
        seen_axes.add(ax)
        sc = row_to_sc(r)
        if not lib.mechanical_ok(sc):
            die(f"gold2: N3 block {r['pair_id']} fails mechanical map")
        n3_picked.append({"row": r, "role": "n3", "sc": sc})
    if len(n3_picked) != 2:
        die(f"gold2: only {len(n3_picked)} N3 blocks (want 2)")
    blocks += n3_picked
    classic_cands = sorted((r for r in a_rows if not r.get("carrier")
                            and r["stratum"] == "CD-2"
                            and r["pair_id"] not in used),
                           key=lambda r: r["pair_id"])
    if not classic_cands:
        die("gold2: no classic CD-2 candidate for drift control")
    cl = classic_cands[0]
    scl = row_to_sc(cl)
    if not lib.mechanical_ok(scl):
        die(f"gold2: classic block {cl['pair_id']} fails mechanical map")
    blocks.append({"row": cl, "role": "classic-drift", "sc": scl})
    if len(blocks) != 12:
        die(f"gold2: {len(blocks)} blocks != 12")
    twins = sum(1 for b in blocks if b["role"] == "n2-mirror")
    hosts = {b["sc"].stratum for b in blocks if b["sc"].carrier == "N1"}
    if twins < 4 or hosts != {"CD-1", "CD-2", "CD-4"}:
        die(f"gold2: E10 composition broken (twins {twins}, hosts {hosts})")

    order = list(range(len(blocks)))
    rng.shuffle(order)
    worked = [
        ("N1 — конфликт + бенигный парафраз",
         "конфликтный слот (несовместимые значения) + одна/две бенигные подстановки в другом предложении",
         "A: «Лимит запросов к сервису — 500 в минуту.» (+ бенигная подстановка в базовом предложении) / B: «Лимит запросов к сервису — 1000 в минуту.»",
         "CONTRADICTS", "слоты конфликтуют; бенигная правка не отменяет конфликта"),
        ("N2 — уточнение + тот же бенигный парафраз",
         "совместимое уточнение (v |= c) + такая же подстановка в другом предложении",
         "A: «Лимит запросов к сервису — не более 1000 в минуту.» (+ та же подстановка) / B: «Лимит запросов к сервису — 800 в минуту.»",
         "NO-CONFLICT", "800 укладывается в «не более 1000» — уточнение"),
        ("N3 — истинный перевод",
         "пара на разных языках, значения всех слотов канонически равны",
         "A: «Обновление намечено на 5 октября.» / B: «The update is scheduled for October 5.»",
         "NO-CONFLICT", "перевод с тем же фактом — противоречия нет"),
    ]
    lines = ["DELTA GOLD SHEET CD-v2 — blind render (12 scorable blocks; verdict per block:",
             "CONTRADICTS / NO-CONFLICT + slot basis; block classes and declarations withheld)",
             ""]
    lines += ["=" * 72,
              "ИНСТРУКТАЖ — три новых носителя v2 с примерами (не скорируется;",
              "в 12 блоках ниже классы не называются).", ""]
    for cls, definition, example, verdict, basis in worked:
        lines.append(f"{cls} — {definition}")
        lines.append(f"  пример: {example}")
        lines.append(f"  вердикт: {verdict} (основание: {basis})")
        lines.append("")
    lines += ["ПРАВИЛА ЛИСТА — как в GOLD SHEET CD-v1:",
              "  вердикт на блок: CONTRADICTS / NO-CONFLICT + слот-основание.",
              "  CONTRADICTS: несовместимые числа/даты одного слота, статусы,",
              "  модальность allow vs require, безусловное отрицание условия,",
              "  полярные оценки одного эффекта.",
              "  NO-CONFLICT: добавка, уточнение в границе, версия-сосед,",
              "  тождественные тела при дельте метаданных, разные предметы,",
              "  перевод с тем же фактом.",
              "  Длину текста, позицию и оформление не использовать как",
              "  аргумент — только факт-слоты.", ""]
    lines += ["=" * 72, "СКОРИРУЕМЫЕ БЛОКИ (12):", ""]
    meta = []
    for i, idx in enumerate(order):
        b = blocks[idx]
        sc = b["sc"]
        r = b["row"]
        ch = sc.length_channel
        body_a = _render_body(sc.side_a.body, sc.scenario_id + ":a", ch) \
            if r is not None else sc.side_a.body
        body_b = _render_body(sc.side_b.body, sc.scenario_id + ":b", ch) \
            if r is not None else sc.side_b.body
        lines.append(f"M{i+1:02d}")
        lines.append(f"  A [{sc.side_a.language or '—'}/{sc.side_a.record_type or '—'}] {sc.side_a.title}")
        lines.append(f"     {body_a}")
        lines.append(f"  B [{sc.side_b.language or '—'}/{sc.side_b.record_type or '—'}] {sc.side_b.title}")
        lines.append(f"     {body_b}")
        lines.append("  verdict: ______   basis: ______________________")
        lines.append("")
        meta.append({"block": f"M{i+1:02d}",
                     "pair_id": sc.scenario_id,
                     "role": b["role"],
                     "twin_of": b.get("twin_of"),
                     "stratum": sc.stratum, "carrier": sc.carrier,
                     "k": len(sc.benign),
                     "label": sc.label,
                     "mech_ok": lib.mechanical_ok(sc),
                     "cos_conventional": (r["measured"]["cos_target"]
                                          if r is not None else None),
                     "len_channel": ch})
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    V2_SHEET = SHEET_DIR  # same dir; v2 names
    render_path = V2_SHEET / "gold-sheet-v2-delta.txt"
    render_path.write_text("\n".join(lines), encoding="utf-8")
    labels_ok = {"contradicts", "not-contradicts"} == {m["label"] for m in meta}
    sheet_json = {"ts": now(), "n": len(meta),
                  "composition": {"blocks": len(meta), "twins": twins,
                                  "n1_hosts": sorted(hosts),
                                  "red_form": "CD-2 upd k=1 razor "
                                              f"(cos {red_cos})",
                                  "n3": 2, "classic_drift": 1},
                  "provenance": "N1/N3 blocks are train cells; N2 twin mirrors are "
                                "purpose-built from the SAME prefix+substitutions "
                                "(slot sentences swapped to a satisfiable refinement); "
                                "each block mechanical-verified",
                  "render_sha256": lib.sha(render_path.read_text(encoding="utf-8")),
                  "blocks": meta}
    sp = V2_SHEET / "gold-sheet-v2-delta.json"
    sp.write_text(json.dumps(sheet_json, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    key = {"kind": "gold-sheet-key", "sheet": "CD-v2-delta",
           "ts": now(),
           "rule": "SEALED — do not open before the committee synthesis",
           "render_fp_sha256": sheet_json["render_sha256"],
           "n": len(meta), "labels_ok": labels_ok,
           "blocks": [{"block": m["block"], "pair_id": m["pair_id"],
                       "stratum": m["stratum"], "carrier": m["carrier"],
                       "role": m["role"], "label": m["label"]} for m in meta]}
    key_path = V2_SHEET / "verdicts-v2" / "KEY-cd-v2.sealed.json"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_body = json.dumps(key, ensure_ascii=False, indent=1)
    key_path.write_text(key_body + "\n", encoding="utf-8")
    fp_key = lib.sha(key_body)
    log("sheet2_frozen", n=len(meta), fp=sheet_json["render_sha256"])
    log("key2_sealed", sheet="CD-v2-delta", n=len(meta), fp=fp_key,
        path=str(key_path.relative_to(REPO)))
    log_md("gold_v2_render", n=len(meta), twins=twins,
           n1_hosts=sorted(hosts), red_form=f"CD-2 k=1 razor cos {red_cos}",
           n3=2, classic_drift=1,
           render="data/conflict-detector/sheet/gold-sheet-v2-delta.txt",
           key_fp=fp_key, legs="LLM+DE now; TL+ML separately after DE report")
    print(f"GOLD2: {len(meta)} blocks (twins {twins}, hosts {sorted(hosts)}), "
          f"render {render_path}")
    print(f"render sha256: {sheet_json['render_sha256']}")
    print(f"KEY SEALED: {key_path} fp {fp_key[:16]}…")
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
    if cmd == "preflight2":
        return cmd_preflight2()
    if cmd == "measure2":
        return cmd_measure2()
    if cmd == "build2":
        return cmd_build2()
    if cmd == "gold2":
        return cmd_gold2()
    if cmd == "seal3":
        return cmd_seal3()
    if cmd == "banks4":
        return cmd_banks4()
    if cmd == "freeze4":
        return cmd_freeze4()
    if cmd == "build4":
        return cmd_build4()
    print(__doc__)
    return 2



# --- build4: C1 orbit carriers + C2/C3 bank refresh + CD-holdout-v4 -----------
# ADDENDUM-7 §2 C1-C9 (RATIFIED 10.11, commit 154f7af): cycle-2 corpus lane.
# C1 — orbit carriers into TRAIN (7 uncovered strata, label-invariance);
# C2 — eval bank refresh (NC-5/NC-2/NC-3, G11); C3 — NC-4 axis promotion.
# Slice rule (fresh generation), NOT amend-once: train-v2 stays byte-frozen,
# train-v4 = train-v2 (verbatim) + C1 carrier rows; seal v4 = fresh cut of
# 180 from the seed4 merged pool. Stop-first: any RED assert writes nothing.

CORPUS_DIR_V4 = DATA / "corpus-v4"
PIN_V4_PATH = DATA / "pin-v4.json"
SEAL_V4_PATH = SEALS_DIR / "CD-holdout-v4.seal.json"
V4_GATE_RECORD = DATA / "measure" / "gate-v3-single-shot-2026-11-10.json"
BANKS4_SLICE = "conflict-detector-corpus-v2-build4-banks"
BUILD4_SLICE = "conflict-detector-corpus-v2-build4"

SEED_HOLD_V4 = 7047
SEEDS_FORBIDDEN_V4 = set(FORBIDDEN_SEEDS) | set(SEEDS_V2.values()) \
    | {SEED_HOLD_V3}
assert SEED_HOLD_V4 not in SEEDS_FORBIDDEN_V4, "v4 hold seed collides with a pinned family"
V3_PREFIX = SEAL3_V3_PREFIX          # "ho3" — banks_v3 rows keep their prefix
V4_PREFIX = "ho4"                    # banks_v4 rows (C2 refresh)

#: G10 census quotas for v4 (2x the ratified rotation depth on
#: clean-after-preflight supply). NC-2 = 2x the NC-2 SEAL FOLD (29;
#: the brief's G11 arithmetic, stricter than the 2x22 classic-only form).
#: NC-4 = 2x11 — the G7 waiver dies with the C3 promotion.
G10_QUOTAS_V4 = {"NC-1": 40, "NC-2": 58, "NC-3": 32, "NC-4": 22, "NC-5": 50}

UNCOVERED_STRATA = ("CD-3", "CD-5", "CD-6", "NC-1", "NC-3", "NC-4", "NC-5")


def log4(event: str, **fields) -> None:
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": BUILD4_SLICE, **fields},
                            ensure_ascii=False) + "\n")


def log4_md(event: str, **fields) -> None:
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": BUILD4_SLICE, **fields},
                            ensure_ascii=False) + "\n")


def log4b(event: str, **fields) -> None:
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": event,
                             "slice": BANKS4_SLICE, **fields},
                            ensure_ascii=False) + "\n")


def _c1_content_sha(seed: int) -> str:
    import conflict_corpus_carriers_v4 as c1m
    return lib.sha(lib.json_dumps_sorted({
        "tag": c1m.TAG, "uncovered": list(c1m.UNCOVERED),
        "expected_razor_bases": c1m.EXPECTED_RAZOR_BASES,
        "fresh_cd5_n": c1m.FRESH_CD5_N, "razor_lo": c1m.RAZOR_LO,
        "seed": seed}))


def _module_sha(relpath: str) -> str:
    return file_sha256(REPO / relpath)


def _frozen_module_check(name: str, relpath: str, prep: dict) -> str:
    """G1: the module on disk must match the freeze pin byte-for-byte."""
    fsha = _module_sha(relpath)
    pin = prep.get(name)
    if not pin:
        die(f"G1: pin-v4.json prep has no {name} — run freeze4 first")
    if fsha != pin["file_sha256"]:
        die(f"G1 LOCK VIOLATION: {name} sha {fsha[:16]} != frozen "
            f"{pin['file_sha256'][:16]} — module edited after freeze")
    return fsha


def cmd_banks4() -> int:
    """C2/C3/C1 preflight (П3-style, fixtures only; runs BEFORE the banks
    freeze commit — G2 line: authorship -> preflight -> freeze -> census ->
    single cut). Writes artifacts/runs/banks_v4_preflight.json."""
    t0 = time.time()
    import conflict_corpus_banks_v4 as b4
    import conflict_corpus_carriers_v4 as c1m
    cos = Cos()
    report: dict = {"ts": now(), "slice": BANKS4_SLICE}

    # -- banks_v3 continuity: the frozen module must import byte-identical --
    prep3 = {}
    if PIN_V3_PATH.exists():
        prep3 = json.loads(PIN_V3_PATH.read_text(encoding="utf-8")).get("prep", {})
    bv3 = prep3.get("banks_v3")
    b3_sha = _module_sha("scripts/conflict_corpus_banks_v3.py")
    if not bv3 or b3_sha != bv3["file_sha256"]:
        die(f"banks_v3 continuity RED: file sha {b3_sha[:16]} != pin-v3 prep "
            f"{(bv3 or {}).get('file_sha256', '?')[:16]}")
    print(f"banks_v3 continuity OK ({b3_sha[:16]})")

    # -- fixture cells: one per NEW bank class ------------------------------
    fx = []
    with b4.v4_bank_context():
        r = random.Random(0)
        fx.append(b4.build_nc2_v4("fx4-NC-2", "pay", "lim", 3000, 1500, 1))
        fx.append(b4.build_nc3_v4("fx4-NC-3", "logs", b4.VERSION_PAIRS_V4[0],
                                  b4._nc3_body(0, 3, "fl1"), 0))
        fx.append(b4.build_nc5_v4("fx4-NC-5", "herbs", "retry", "cross", 0, 0))
    fx_ok = True
    print("| fixture | mechanical | label |")
    print("|---|---|---|")
    for sc in fx:
        mech = lib.mechanical_ok(sc)
        want = "not-contradicts"
        row_ok = mech and sc.label == want
        fx_ok &= row_ok
        print(f"| {sc.stratum} (v4) | {mech} | {sc.label} |"
              + ("" if row_ok else "  <-- FAIL"))
    report["fixtures"] = {"ok": fx_ok,
                          "cells": [x.scenario_id for x in fx]}

    # -- NC-2 refresh: checker truth over the whole combo table -------------
    chk = []
    for axis, c, v in b4.NC2_COMBOS_V4:
        chk.append(bool(lib.REFINE_RU[axis][2](v, c)))
    report["nc2_checker_truth"] = {"ok": all(chk), "combos": len(chk)}
    print(f"NC-2 v4 checker truth: {sum(chk)}/{len(chk)}")

    # -- NC-5 refresh: measured cross envelope over TRUE cross legs --------
    # (RU content vs EN content; the v3-parity "long"/"same" legs carry
    # RU bodies with lang flags — the v3 seal itself ships that shape,
    # see ho3-NC-5 transfer rows — they are NOT the G8 cross envelope)
    def _is_ascii(t: str) -> bool:
        return all(ord(c) < 128 for c in t)

    # same-subject RU/EN phrase pairs (the G8/G9 "knowingly not
    # translations" invariant) must stay inside the OLD-bank cross envelope
    # [0.486, 0.8024] (v3 declared bounds, banks_v3_preflight.json); the
    # cross-DOMAIN cell distribution is informational (feeds b70/b55)
    pairs = {}
    with b4.v4_bank_context():
        for subj, (ru0, ru1, en0, en1) in b4.SPEC_V4.items():
            s_ru = lib.make_side("ru", subj, ru0)
            s_ru1 = lib.make_side("ru", subj, ru1)
            s_en = lib.make_side("en", subj, en0)
            s_en1 = lib.make_side("en", subj, en1)
            pairs[f"{subj}:ru0-en0"] = round(cos.pair(lib.embed_text(s_ru), lib.embed_text(s_en)), 4)
            pairs[f"{subj}:ru0-en1"] = round(cos.pair(lib.embed_text(s_ru), lib.embed_text(s_en1)), 4)
            pairs[f"{subj}:ru1-en0"] = round(cos.pair(lib.embed_text(s_ru1), lib.embed_text(s_en)), 4)
            pairs[f"{subj}:ru1-en1"] = round(cos.pair(lib.embed_text(s_ru1), lib.embed_text(s_en1)), 4)
    band_lo, band_hi = 0.486, 0.8024
    out_band = {k: v for k, v in pairs.items()
                if not (band_lo <= v <= band_hi)}
    ok_env = not out_band
    cross_cells = []
    with b4.v4_bank_context():
        pool5 = b4.gen_nc5_v4(7047, "fx", rotate=False)
    for sc in pool5:
        if sc.cross_lang and _is_ascii(sc.side_b.body) \
                and not _is_ascii(sc.side_a.body):
            cross_cells.append(round(cos.pair(sc.text_a(), sc.text_b()), 4))
    report["nc5_cross_envelope"] = {
        "same_subject_pairs": pairs, "declared_band": [band_lo, band_hi],
        "out_of_band": out_band, "ok": bool(ok_env),
        "cross_cells_info": {"n": len(cross_cells),
                             "min": min(cross_cells),
                             "max": max(cross_cells),
                             "note": "cross-DOMAIN legs feed the b70/b55 "
                                     "bands; informational here"},
        "note": "v3 declared bounds [0.486, 0.8024] (old-bank envelope, "
                "banks_v3_preflight.json)"}
    print(f"NC-5 v4 same-subject RU/EN pairs: {len(pairs)}, out-of-band "
          f"{len(out_band)} (band [{band_lo}, {band_hi}]) "
          f"{'OK' if ok_env else 'RED'}; range "
          f"[{min(pairs.values())}, {max(pairs.values())}]; cross cells "
          f"[{min(cross_cells)}, {max(cross_cells)}] n={len(cross_cells)}")
    fx_ok &= ok_env

    # -- pool-wide mechanical attrition + anti-collapse (bank-level) --------
    with b4.v4_bank_context():
        pool4 = b4.gen_banks_v4_pool(7047, "fx", rotate=False)
    mech_ok = sum(1 for x in pool4 if lib.mechanical_ok(x))
    report["pool_attrition"] = {"pool": len(pool4), "mech_ok": mech_ok,
                                "ok": mech_ok == len(pool4)}
    print(f"banks_v4 pool: {len(pool4)} cells, mechanical {mech_ok}/{len(pool4)}")
    fx_ok &= mech_ok == len(pool4)
    scaf_by_class: dict[str, Counter] = {}
    for x in pool4:
        scaf_by_class.setdefault(x.stratum, Counter())[x.scaffold_id] += 1
    scaf_rep = {}
    for s, cnt in scaf_by_class.items():
        total = sum(cnt.values())
        mx = max(cnt.values())
        scaf_rep[s] = {"scaffolds": len(cnt), "max_share": round(mx / total, 4)}
        ok = len(cnt) >= 20 and mx / total <= lib.CAP + 1e-9
        scaf_rep[s]["ok"] = bool(ok)
        fx_ok &= ok
    report["anti_collapse"] = scaf_rep
    print("anti-collapse:", json.dumps(scaf_rep))

    # -- C1 fixture orbits: reconstruction fidelity + rung predicates -------
    bases = c1m.load_razor_bases()
    fid = 0
    total = 0
    for s, lst in bases.items():
        for base in lst:
            total += 1
            if lib.mechanical_ok(base):
                fid += 1
    report["c1_base_fidelity"] = {"bases": total, "mech_ok": fid,
                                  "ok": fid == total}
    print(f"C1 base reconstruction: mechanical {fid}/{total}")
    fx_ok &= fid == total
    enu = c1m.enumerate_orbits(SEED_HOLD_V4)
    rung_ok = sum(1 for o in enu["orbits"] for rg in o.rungs
                  if c1m.orbit_ok(rg, o.base))
    rung_n = sum(len(o.rungs) for o in enu["orbits"])
    lab_ok = all(rg.label == o.base.label
                 for o in enu["orbits"] for rg in o.rungs)
    scf_ok = all(rg.scaffold_id == o.base.scaffold_id
                 for o in enu["orbits"] for rg in o.rungs)
    cp_ok = all(c1m.cosmetic_proof(o.base.side_a)
                and c1m.cosmetic_proof(o.base.side_b) for o in enu["orbits"])
    report["c1_orbits"] = {
        "orbits": len(enu["orbits"]), "rungs": rung_n, "pred_ok": rung_ok,
        "label_invariance": bool(lab_ok), "scaffold_inheritance": bool(scf_ok),
        "cosmetic_proof": bool(cp_ok),
        "by_stratum": {s: sum(1 for o in enu["orbits"] if o.stratum == s)
                       for s in UNCOVERED_STRATA},
        "ok": rung_ok == rung_n and lab_ok and scf_ok and cp_ok}
    fx_ok &= rung_ok == rung_n and lab_ok and scf_ok and cp_ok
    print(f"C1 orbits: {len(enu['orbits'])}, rungs {rung_n}, pred {rung_ok}, "
          f"label/scaffold/cosmetic {lab_ok}/{scf_ok}/{cp_ok}")

    ok = bool(fx_ok)
    report["verdict"] = "PASS" if ok else "RED"
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    (RUNLOG.parent / "banks_v4_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log4b("banks4_preflight", ok=ok, verdict=report["verdict"],
          fixtures=report["fixtures"]["ok"],
          nc5_envelope=report["nc5_cross_envelope"],
          pool=f"{mech_ok}/{len(pool4)}",
          c1_rungs=f"{rung_ok}/{rung_n}")
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": "banks4_preflight",
                             "slice": BANKS4_SLICE, "ok": ok,
                             "verdict": report["verdict"],
                             "nc5_cross_envelope": report["nc5_cross_envelope"],
                             "c1_rungs": f"{rung_ok}/{rung_n}"},
                            ensure_ascii=False) + "\n")
    print(f"\nBANKS4 PREFLIGHT: {report['verdict']} ({time.time()-t0:.0f}s)")
    return 0 if ok else 1


def cmd_freeze4() -> int:
    """G1 freeze: pin the NEW module shas (banks_v4 + carriers_c1) and the
    v4 seed into pin-v4.json prep — AFTER the banks commit, BEFORE any
    census observation. Re-freeze or post-freeze edits are incidents."""
    if PIN_V4_PATH.exists():
        die("pin-v4.json already exists — freeze is one-shot; a re-freeze "
            "is an incident (G1)")
    import conflict_corpus_banks_v4 as b4
    import conflict_corpus_carriers_v4 as c1m
    prep3 = json.loads(PIN_V3_PATH.read_text(encoding="utf-8")).get("prep", {})
    bv3 = prep3.get("banks_v3")
    b3_sha = _module_sha("scripts/conflict_corpus_banks_v3.py")
    if not bv3 or b3_sha != bv3["file_sha256"]:
        die("banks_v3 continuity RED at freeze — G1 chain broken")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    if not head:
        die("git rev-parse failed — cannot record the banks commit")
    banks_commit = head
    file_b4 = _module_sha("scripts/conflict_corpus_banks_v4.py")
    file_c1 = _module_sha("scripts/conflict_corpus_carriers_v4.py")
    content_b4 = lib.sha(lib.json_dumps_sorted(b4.bank_content()))
    content_c1 = _c1_content_sha(SEED_HOLD_V4)
    with b4.v4_bank_context():
        n2 = b4.gen_nc2_v4(SEED_HOLD_V4, "smoke", rotate=False)
        n3 = b4.gen_nc3_v4(SEED_HOLD_V4, "smoke", rotate=False)
        n5 = b4.gen_nc5_v4(SEED_HOLD_V4, "smoke", rotate=False)
    bank_counts = {"nc2": len(n2), "nc3": len(n3), "nc5": len(n5)}
    enu = c1m.enumerate_orbits(SEED_HOLD_V4)
    carrier_counts = {
        "orbits": len(enu["orbits"]),
        "rungs": sum(len(o.rungs) for o in enu["orbits"]),
        "fresh_cd5": sum(1 for o in enu["orbits"] if o.kind == "fresh"),
        "by_stratum": {s: sum(1 for o in enu["orbits"] if o.stratum == s)
                       for s in UNCOVERED_STRATA}}
    pin = {
        "prep": {
            "banks_v3_continuity": {
                "pin_source": "pin-v3.json#prep.banks_v3",
                "banks_commit": bv3["banks_commit"],
                "file_sha256": bv3["file_sha256"],
                "runtime_verified": True},
            "banks_v4": {
                "frozen_at": now(),
                "banks_commit": banks_commit,
                "module": "scripts/conflict_corpus_banks_v4.py",
                "file_sha256": file_b4,
                "content_sha256": content_b4,
                "counts": bank_counts,
                "ratification": "ADDENDUM-7 §2 C2 (G11 refresh) + §2 C3 "
                                "(NC-4 promotion record)",
                "lock": "bank edits FORBIDDEN after this pin (G1); a census "
                        "or assert failure is a BuildStop + NEW addendum"},
            "carriers_c1": {
                "frozen_at": now(),
                "banks_commit": banks_commit,
                "module": "scripts/conflict_corpus_carriers_v4.py",
                "file_sha256": file_c1,
                "content_sha256": content_c1,
                "counts": carrier_counts,
                "ratification": "ADDENDUM-7 §2 C1 (orbit carriers, "
                                "label-invariance; measure-then-fix quotas)",
                "lock": "carrier edits FORBIDDEN after this pin (G1)"},
            "seeds": {
                "hold": SEED_HOLD_V4,
                "status": "pinned (seeds_pinned_v4)",
            },
        },
        "ts": now(),
    }
    PIN_V4_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIN_V4_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    pin_sha = lib.sha(PIN_V4_PATH.read_text(encoding="utf-8"))
    log4b("banks4_frozen", ok=True, banks_commit=banks_commit[:12],
          banks_v4={"file": file_b4[:16], "content": content_b4[:16]},
          carriers_c1={"file": file_c1[:16], "content": content_c1[:16]},
          seed=SEED_HOLD_V4, pin_sha=pin_sha[:16],
          counts={"banks_v4": bank_counts, "carriers_c1": carrier_counts})
    RUNLOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_MD.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now(), "event": "banks4_frozen",
                             "slice": BANKS4_SLICE, "ok": True,
                             "banks_commit": banks_commit[:12],
                             "seed": SEED_HOLD_V4,
                             "lock": "G1 (banks_v4 + carriers_c1)"},
                            ensure_ascii=False) + "\n")
    print(f"FREEZE4: G1 LOCKED (commit {banks_commit[:12]})\n"
          f"  banks_v4    file {file_b4[:16]} content {content_b4[:16]} {bank_counts}\n"
          f"  carriers_c1 file {file_c1[:16]} content {content_c1[:16]} "
          f"{carrier_counts['orbits']} orbits / {carrier_counts['rungs']} rungs\n"
          f"  seed hold={SEED_HOLD_V4}\n  pin: {PIN_V4_PATH} ({pin_sha[:16]})")
    return 0


def _v3_verdict_on_record() -> dict:
    """Cycle-2 authorization: the v3 single-shot record must exist and carry
    the final DECLINE verdict (F1: red v3 is FINAL; ADDENDUM-7 §4 order)."""
    if not V4_GATE_RECORD.exists():
        die(f"cycle-2 authorization RED: no v3 gate record at {V4_GATE_RECORD}")
    rec = json.loads(V4_GATE_RECORD.read_text(encoding="utf-8"))
    done = [e for e in rec if e.get("event") == "gate_done"]
    if not done:
        die("cycle-2 authorization RED: v3 gate record has no gate_done")
    v = done[-1]
    if v.get("verdict") != "DECLINE":
        die(f"cycle-2 authorization RED: v3 verdict {v.get('verdict')!r} "
            "is not DECLINE — no cycle-2 lane is legal")
    return v


def _surviving_carriers(cos, sources) -> dict:
    """C1 census (dry): enumerate orbits, measure, apply the survival rules
    (mechanical predicate, razor retention, E4 channel preservation, F10
    sha cleanliness vs ALL sealed generations + train-v2 pair identity,
    union scaffold caps). Returns the surviving carrier block + report."""
    import conflict_corpus_carriers_v4 as c1m
    enu = c1m.enumerate_orbits(SEED_HOLD_V4)
    for o in enu["orbits"]:
        if o.kind == "fresh":
            o.base._cos = round(cos.pair(o.base.text_a(), o.base.text_b()), 4)
        for rg in o.rungs:
            rg._cos = round(cos.pair(rg.text_a(), rg.text_b()), 4)

    tr_pair_shas: set[tuple[str, str]] = set()
    with (CORPUS_DIR_V2 / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            tr_pair_shas.add(tuple(sorted((
                _disk_env_sha(row["envelope_a"]),
                _disk_env_sha(row["envelope_b"])))))
    tr_scaf: dict[str, int] = {}
    tr_scaf_by_s: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    with (CORPUS_DIR_V2 / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["pair_id"] in seen:
                continue
            seen.add(row["pair_id"])
            tr_scaf[row["scaffold_id"]] = tr_scaf.get(row["scaffold_id"], 0) + 1
            tr_scaf_by_s.setdefault(row["stratum"], {})
            tr_scaf_by_s[row["stratum"]][row["scaffold_id"]] = \
                tr_scaf_by_s[row["stratum"]].get(row["scaffold_id"], 0) + 1

    seal_names = ("v1-seal", "v2-seal", "v3-seal")
    survivors: list[dict] = []
    rep: dict = {"orbits": len(enu["orbits"]),
                 "cull": Counter(), "per_stratum": {}}
    orbit_out: dict[str, list] = {}
    for o in enu["orbits"]:
        kept = []
        for rg in o.rungs:
            rung_name = "para1" if len(rg.benign) == 1 else "para2"
            if not c1m.orbit_ok(rg, o.base):
                rep["cull"]["pred"] += 1
                continue
            if rg._cos < c1m.RAZOR_LO:
                rep["cull"]["razor_loss"] += 1
                continue
            if rg.length_channel != o.base.length_channel:
                rep["cull"]["channel"] += 1
                continue
            ha, hb = lib.sha(rg.text_a()), lib.sha(rg.text_b())
            src = _forbidden_source(ha, hb, {n: sources[n] for n in seal_names})
            if src:
                rep["cull"][f"f10:{src}"] += 1
                continue
            pair = tuple(sorted((ha, hb)))
            if pair in tr_pair_shas:
                rep["cull"]["train_pair_dup"] += 1
                continue
            kept.append({"sc": rg, "rung": rung_name, "base": o.base,
                         "kind": o.kind, "sha_new": hb if rg.text_a() ==
                         o.base.text_a() else ha})
        if kept:
            orbit_out[f"{o.stratum}:{o.base.scenario_id}"] = kept
            survivors.extend(kept)

    # fresh CD-5 bases ride their orbit: a fresh orbit emits its base iff
    # its para1 rung survived (the orbit is the carrier unit)
    carriers = []
    dropped_fresh = 0
    for key, kept in orbit_out.items():
        o_kind = kept[0]["kind"]
        o_stratum = kept[0]["base"].stratum
        if o_kind == "fresh":
            if not any(k["rung"] == "para1" for k in kept):
                dropped_fresh += 1
                continue
            carriers.append({"sc": kept[0]["base"], "rung": "base",
                             "base": kept[0]["base"], "kind": "fresh",
                             "sha_new": None})
        carriers.extend(kept)
    rep["dropped_fresh_orbits"] = dropped_fresh

    # union scaffold caps for the carrier block (per stratum): CAP-AWARE
    # SELECTION at census time — the ratified §4.5 rule applied as a
    # selection rule, exactly like the v2 union-aware select_carriers
    # (lane-side selection is NOT a bank edit; the frozen enumeration
    # stays the candidate pool). Deterministic: keep-order (stratum, base
    # id, rung rank), cap recomputed from post-selection totals to a
    # fixed point, drops disclosed.
    rep["cap_selection_dropped"] = {}
    rung_rank = {"base": 0, "para1": 1, "para2": 2}
    for _pass in range(4):
        carriers.sort(key=lambda c: (c["sc"].stratum, c["base"].scenario_id,
                                     rung_rank.get(c["rung"], 9)))
        class_total_v4 = {s: CLASS_TOTALS_V2[s]
                          + sum(1 for c in carriers if c["sc"].stratum == s)
                          for s in ORDER}
        dropped = 0
        by_s: dict[str, list] = {}
        for c in carriers:
            by_s.setdefault(c["sc"].stratum, []).append(c)
        for s, lst in by_s.items():
            cap = max(1, int(lib.CAP * (class_total_v4[s] + SEAL_SHARES_V2[s]
                                        + _SEAL_CARRIER_IN_STRATUM.get(s, 0))
                              + 1e-9))
            cnt: Counter = Counter()
            keep = []
            for c in lst:
                g = c["sc"].scaffold_id
                if tr_scaf.get(g, 0) + cnt[g] + 1 > cap:
                    dropped += 1
                    rep["cap_selection_dropped"].setdefault(
                        s, []).append(f"{c['rung']}:{c['sc'].scenario_id}"
                                      f" (scaffold {g}, cap {cap})")
                    continue
                cnt[g] += 1
                keep.append(c)
            lst[:] = keep
        carriers = [c for s in UNCOVERED_STRATA for c in by_s.get(s, [])]
        if dropped == 0:
            break
    carrier_rows = []
    for s in UNCOVERED_STRATA:
        lst = [c for c in carriers if c["sc"].stratum == s]
        cap = max(1, int(lib.CAP * (CLASS_TOTALS_V2[s] + len(lst)
                                    + SEAL_SHARES_V2[s]
                                    + _SEAL_CARRIER_IN_STRATUM.get(s, 0))
                          + 1e-9))
        cnt: Counter = Counter()
        red = []
        for c in lst:
            g = c["sc"].scaffold_id
            cnt[g] += 1
            if tr_scaf.get(g, 0) + cnt[g] > cap:
                red.append(f"{s}/{g}: {tr_scaf.get(g, 0)}+{cnt[g]} > cap {cap}")
        rep["per_stratum"][s] = {
            "carriers": len(lst),
            "class_total_v4": CLASS_TOTALS_V2[s] + len(lst),
            "cap_v4": cap,
            "scaffold_cap_red": red[:3],
            "scaffolds": len(set(c["sc"].scaffold_id for c in lst)),
            "full_orbits": sum(1 for k, kept in orbit_out.items()
                               if kept and kept[0]["base"].stratum == s
                               and len({k2["rung"] for k2 in kept}) >= 2
                               and kept[0]["kind"] == "classic"
                               and all(any(c["sc"].scenario_id ==
                                           k2["sc"].scenario_id
                                           for c in carriers)
                                       for k2 in kept)),
        }
        carrier_rows.extend(red)
    rep["scaffold_red"] = carrier_rows
    return {"carriers": carriers, "orbit_out": orbit_out, "report": rep,
            "tr_scaf": tr_scaf, "tr_scaf_by_s": tr_scaf_by_s}


def _carrier_rows4(sc: lib.Scenario, rung: str, base_id: str,
                   kind: str) -> list[dict]:
    ch5 = lib.pair_char5_jaccard(sc.side_a, sc.side_b, sc._cos)
    base_decl = {"slots": sc.slots, "subject_a": sc.subject_a,
                 "subject_b": sc.subject_b, "polar_op": sc.polar_op,
                 "refine_ok": sc.refine_ok, "benign_regions": sc.benign,
                 "rung": rung, "base_id": base_id, "orbit_kind": kind}
    row = {"pair_id": sc.scenario_id, "scenario_id": sc.scenario_id,
           "stratum": sc.stratum, "label": sc.label, "carrier": sc.carrier,
           "declaration": base_decl,
           "measured": {"cos_target": sc._cos,
                        "char5_jaccard": round(ch5, 4),
                        "len_channel": sc.length_channel},
           "scaffold_id": sc.scaffold_id,
           "seed_pin": f"seed:{SEED_HOLD_V4}:c1", "split": "train"}
    r1 = dict(row, orientation="a>b", envelope_a=side_dict(sc.side_a),
              envelope_b=side_dict(sc.side_b))
    r2 = dict(row, orientation="b>a", envelope_a=side_dict(sc.side_b),
              envelope_b=side_dict(sc.side_a))
    return [r1, r2]


def cmd_build4() -> int:
    """Cycle-2 corpus lane (ADDENDUM-7 §4): C1 carriers into train-v4 +
    C2/C3 eval refresh -> censuses -> SINGLE cut of 180 -> seal
    CD-holdout-v4. Stop-first: any RED census or assert writes NOTHING."""
    t0 = time.time()

    # --- 1. legality (F1/F3/F7 legs of cycle-2) ---------------------------
    verdict3 = _v3_verdict_on_record()
    print(f"cycle-2 authorization: v3 gate verdict DECLINE on record "
          f"(red={verdict3.get('red')}) — ADDENDUM-7 §4 order is legal")
    seal4_done = 0
    if RUNLOG.exists():
        for line in RUNLOG.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "seal_v4":
                seal4_done += 1
    if seal4_done:
        die(f"rotation already consumed: {seal4_done} seal_v4 events — "
            "a second v4 would be an incident, not a retry")
    man2 = json.loads((CORPUS_DIR_V2 / "build-manifest.json")
                      .read_text(encoding="utf-8"))
    man3 = json.loads((CORPUS_DIR_V3 / "build-manifest.json")
                      .read_text(encoding="utf-8"))
    train_fp_start = file_sha256(CORPUS_DIR_V2 / "train.jsonl")
    if train_fp_start != man2["fingerprints"]["train.jsonl"]:
        die("train-v2 integrity RED: disk digest != manifest fingerprint")
    archives_intact = {
        "corpus-v2/holdout.jsonl": file_sha256(CORPUS_DIR_V2 / "holdout.jsonl")
        == man2["fingerprints"]["holdout.jsonl"],
        "seals/CD-holdout-v2.seal.json": file_sha256(SEAL_V2_PATH)
        == man2["fingerprints"]["seal"],
        "corpus-v1/holdout.jsonl": file_sha256(CORPUS_DIR / "holdout.jsonl")
        == json.loads(SEAL_V1_PATH.read_text(encoding="utf-8"))
        ["holdout_jsonl_sha256"],
        "corpus-v3/holdout.jsonl": file_sha256(CORPUS_DIR_V3 / "holdout.jsonl")
        == man3["fingerprints"]["holdout.jsonl"],
        "seals/CD-holdout-v3.seal.json": file_sha256(SEAL_V3_PATH)
        == man3["fingerprints"]["seal"],
    }
    if not all(archives_intact.values()):
        die(f"archive integrity RED: {dict(archives_intact.values())}")

    # --- G1 freeze verification (banks_v3 continuity + new modules) -------
    import conflict_corpus_banks_v3 as banks_v3
    import conflict_corpus_banks_v4 as banks_v4
    import conflict_corpus_carriers_v4 as c1m
    prep3 = json.loads(PIN_V3_PATH.read_text(encoding="utf-8")).get("prep", {})
    bv3 = prep3.get("banks_v3")
    b3_sha = _module_sha("scripts/conflict_corpus_banks_v3.py")
    if not bv3 or b3_sha != bv3["file_sha256"]:
        die("G1 banks_v3 continuity RED at build4")
    b3_content = lib.sha(lib.json_dumps_sorted(banks_v3.bank_content()))
    if b3_content != bv3["content_sha256"]:
        die("G1 banks_v3 content RED at build4")
    prep4 = json.loads(PIN_V4_PATH.read_text(encoding="utf-8")).get("prep", {}) \
        if PIN_V4_PATH.exists() else {}
    b4_sha = _frozen_module_check("banks_v4", "scripts/conflict_corpus_banks_v4.py", prep4)
    c1_sha = _frozen_module_check("carriers_c1", "scripts/conflict_corpus_carriers_v4.py", prep4)
    b4_content = lib.sha(lib.json_dumps_sorted(banks_v4.bank_content()))
    if b4_content != prep4["banks_v4"]["content_sha256"]:
        die("G1 banks_v4 content RED (disk content != freeze pin)")
    c1_content = _c1_content_sha(SEED_HOLD_V4)
    if c1_content != prep4["carriers_c1"]["content_sha256"]:
        die("G1 carriers_c1 content RED (config snapshot != freeze pin)")
    print(f"G1 freeze verified: banks_v3 {b3_sha[:12]} (byte-frozen), "
          f"banks_v4 {b4_sha[:12]}, carriers_c1 {c1_sha[:12]} LOCKED")

    # --- 2. seed pin FIRST (before any generation) --------------------------
    scan, _prior_v3 = pinned_seeds_from_runlog()
    observed = sorted({s for seeds in scan.values() for s in seeds})
    prior_v4: list[int] = []
    if RUNLOG.exists():
        for line in RUNLOG.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "seeds_pinned_v4":
                seeds = ev.get("seeds")
                if isinstance(seeds, dict) and isinstance(seeds.get("hold"), int):
                    prior_v4.append(seeds["hold"])
    if SEED_HOLD_V4 in observed and not prior_v4:
        die(f"v4 hold seed {SEED_HOLD_V4} already pinned elsewhere in "
            "run_log (ad-hoc pin guard)")
    for p in prior_v4:
        if p != SEED_HOLD_V4:
            die(f"seed-family shopping RED: prior seeds_pinned_v4 hold={p}, "
                f"this run pins {SEED_HOLD_V4}")
    reaffirm = len(prior_v4) > 0
    log4("seeds_pinned_v4", seeds={"hold": SEED_HOLD_V4},
         family="single-member (hold only; train/sheet members absent by "
                "construction — C1 carriers derive from train-v2 + the "
                "frozen PARA dial, not from a train seed)",
         forbidden="2113/7919/9377/6117 + 42/777/4242 + 43..52 + every "
                   "seeds_pinned* seed",
         reaffirmed=bool(reaffirm),
         runlog_scan={"events": len(scan), "distinct_seeds": len(observed),
                      "seeds": observed})
    log4_md("seeds_pinned_v4", seeds={"hold": SEED_HOLD_V4},
            family="single-member (hold only)",
            forbidden_sets=sorted(SEEDS_FORBIDDEN_V4 | {SEED_HOLD_V3}),
            reaffirmed=bool(reaffirm))
    print(f"seed v4 pinned: hold={SEED_HOLD_V4} "
          f"({'RE-AFFIRMED' if reaffirm else 'fresh'}; run-log scan: "
          f"{len(scan)} events, {len(observed)} distinct seeds)")

    cos = Cos()

    # --- 3. disjointness sources (F6/F10; sealed content NEVER opened) -----
    print("== disjointness sources (4 generations + carriers) ==")
    tr_texts, _ = disk_text_shas(CORPUS_DIR_V2 / "train.jsonl")
    v3_shas = sealed_generation_sha_set(SEAL_V3_PATH,
                                        CORPUS_DIR_V3 / "holdout-pool.jsonl")
    v2_shas = sealed_generation_sha_set(SEAL_V2_PATH,
                                        CORPUS_DIR_V2 / "holdout-pool.jsonl")
    v1_shas = sealed_generation_sha_set(SEAL_V1_PATH,
                                        CORPUS_DIR / "holdout-pool.jsonl")
    sources = {"train-v2": tr_texts, "v3-seal": v3_shas, "v2-seal": v2_shas,
               "v1-seal": v1_shas}
    print(f"sha sets: train-v2 {len(tr_texts)}, v3-seal {len(v3_shas)}, "
          f"v2-seal {len(v2_shas)}, v1-seal {len(v1_shas)} "
          "(seal ids -> pool-origin shas; holdout.jsonl files NOT opened)")

    # --- 4. C1 census (dry) + measure-then-fix quota pin -------------------
    print("== C1 census: orbit carriers (survival: pred+razor+channel+F10+caps) ==")
    c1 = _surviving_carriers(cos, sources)
    c1_rep = c1["report"]
    carriers = c1["carriers"]
    if c1_rep["scaffold_red"]:
        die("C1 census RED (scaffold union caps): "
            + "; ".join(c1_rep["scaffold_red"][:4]))
    c1_by_stratum = {s: [c for c in carriers if c["sc"].stratum == s]
                     for s in UNCOVERED_STRATA}
    red_c1 = [s for s in UNCOVERED_STRATA if not c1_by_stratum[s]]
    if red_c1:
        die(f"C1 census RED: strata with zero surviving carriers: {red_c1}")
    quotas_c1 = {s: len(c1_by_stratum[s]) for s in UNCOVERED_STRATA}
    rungs_c1 = {s: sum(1 for c in c1_by_stratum[s] if c["rung"] != "base")
                for s in UNCOVERED_STRATA}
    total_carriers = len(carriers)
    print("C1 carriers per stratum:",
          json.dumps({s: len(c1_by_stratum[s]) for s in UNCOVERED_STRATA}))
    print(f"C1 cull ledger: {dict(c1_rep['cull'])}; fresh-orbit drops "
          f"{c1_rep['dropped_fresh_orbits']}")
    # measure-then-fix: pin the census numbers BEFORE emission (the pin is
    # a quota write, not a bank edit — G1 untouched)
    pin = json.loads(PIN_V4_PATH.read_text(encoding="utf-8"))
    if "carriers_c1_quota" in pin.get("prep", {}):
        if pin["prep"]["carriers_c1_quota"]["per_stratum"] != quotas_c1:
            die("C1 quota pin diverges from a prior pin — measure-then-fix "
                "is one-shot per freeze; diverging census is an instrument "
                "abort")
    else:
        pin["prep"]["carriers_c1_quota"] = {
            "pinned_at": now(),
            "per_stratum": quotas_c1,
            "per_stratum_rungs": rungs_c1,
            "total": total_carriers,
            "note": "measure-then-fix (ADDENDUM-7 C1): census-achieved "
                    "counts pinned BEFORE emission; emission asserts "
                    "equality; classic shares untouched"}
        PIN_V4_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    log4("census_v4_c1", verdict="pass", quotas=quotas_c1, rungs=rungs_c1,
         total=total_carriers, cull=dict(c1_rep["cull"]),
         per_stratum={s: c1_rep["per_stratum"][s] for s in UNCOVERED_STRATA})
    log4_md("census_v4_c1", verdict="pass", quotas=quotas_c1,
            total=total_carriers, cull=dict(c1_rep["cull"]))
    print("C1 census PASS (quotas pinned pre-emission)")

    # --- 5. eval pool (fresh seed4 generation over all universes) ----------
    print("== eval pool: classic + banks_v3 (ho3) + banks_v4 (ho4) + carriers ==")
    hold_pool = gen_pool(SEED_HOLD_V4, "ho", rotate=True) \
        + banks_v3.gen_banks_v3_pool(SEED_HOLD_V4, V3_PREFIX, rotate=True) \
        + banks_v4.gen_banks_v4_pool(SEED_HOLD_V4, V4_PREFIX, rotate=True)
    pool_classic_n = sum(1 for x in hold_pool if "-" not in x.scenario_id
                         or x.scenario_id.startswith("ho-"))
    pool_v3_n = sum(1 for x in hold_pool
                    if x.scenario_id.startswith(V3_PREFIX + "-"))
    pool_v4_n = sum(1 for x in hold_pool
                    if x.scenario_id.startswith(V4_PREFIX + "-"))
    hn1 = gen_n1_pool(SEED_HOLD_V4, "ho")
    hn2 = gen_n2_pool(SEED_HOLD_V4, "ho")
    hn3 = gen_n3_pool(SEED_HOLD_V4, "ho")
    print(f"holdout pool: {len(hold_pool)} (classic {pool_classic_n}, "
          f"v3 banks {pool_v3_n}, v4 banks {pool_v4_n}); carriers "
          f"N1 {len(hn1)} N2 {len(hn2)} N3 {len(hn3)}")
    measure_all(hold_pool, cos)
    for sc in hn1 + hn2 + hn3:
        sc._cos = round(cos.pair(sc.text_a(), sc.text_b()), 4)
        sc._band = band_of(sc._cos)
    print(f"embedded {cos.n} unique texts in {time.time()-t0:.0f}s")

    # --- 6. G10 census (clean-after-preflight vs 2x quotas) ----------------
    print("== G10 census: clean-after-preflight supply vs 2x quotas ==")
    tr_scaf_all: dict[str, int] = dict(c1["tr_scaf"])
    for c in carriers:
        g = c["sc"].scaffold_id
        tr_scaf_all[g] = tr_scaf_all.get(g, 0) + 1

    def _clean(cells: list, mech_fn) -> dict:
        live = [x for x in cells if mech_fn(x)]
        live.sort(key=lambda x: (-x._cos, x.scenario_id))
        seen: set[str] = set()
        culled: Counter = Counter()
        dups = 0
        rows = []
        for x in live:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            src = _forbidden_source(ha, hb, sources)
            if src:
                culled[src] += 1
                continue
            if ha in seen or hb in seen:
                dups += 1
                continue
            seen.update((ha, hb))
            rows.append(x)
        origin = Counter("v4" if x.scenario_id.startswith(V4_PREFIX + "-")
                         else ("v3" if x.scenario_id.startswith(V3_PREFIX + "-")
                               else "classic") for x in rows)
        return {"pool": len(cells),
                "preflight_attrited": len(cells) - len(live),
                "culled_f10": dict(culled), "dup_blocked": dups,
                "clean": len(rows), "clean_v4": origin.get("v4", 0),
                "clean_v3": origin.get("v3", 0),
                "clean_classic": origin.get("classic", 0), "_rows": rows}

    g10: dict = {}
    red10 = []
    by_s_g10: dict[str, list] = {}
    for x in hold_pool:
        by_s_g10.setdefault(x.stratum, []).append(x)
    for s in ORDER:
        row = _clean(by_s_g10.get(s, []), lib.mechanical_ok)
        g10[s] = {k: v for k, v in row.items() if k != "_rows"}
        want = G10_QUOTAS_V4.get(s)
        if want is not None:
            mark = "OK" if row["clean"] >= want else "RED"
            if row["clean"] < want:
                red10.append(f"{s} clean {row['clean']} < {want}")
            print(f"  {s}: clean {row['clean']} (v4 {row['clean_v4']} / "
                  f"v3 {row['clean_v3']} / classic {row['clean_classic']}) "
                  f"vs {want} [{mark}]")
    cen_n1c = _clean(hn1, carrier_suitable)
    cen_n2c = _clean(hn2, carrier_suitable)
    cen_n3c = _clean(hn3, lib.mechanical_ok)
    n1_hosts_c = Counter(x.stratum for x in cen_n1c["_rows"])
    n1_k_c = Counter(len(x.benign) for x in cen_n1c["_rows"])
    n2_k_c = Counter(len(x.benign) for x in cen_n2c["_rows"])
    g10["N1"] = {**{k: v for k, v in cen_n1c.items() if k != "_rows"},
                 "hosts": dict(n1_hosts_c), "k": dict(n1_k_c)}
    g10["N2"] = {**{k: v for k, v in cen_n2c.items() if k != "_rows"},
                 "k": dict(n2_k_c)}
    g10["N3"] = {k: v for k, v in cen_n3c.items() if k != "_rows"}
    need_n1_total = 2 * (SEAL_N1_K[0] + SEAL_N1_K[1])
    need_n1_host = {h: 2 * q for h, q in SEAL_N1_HOSTS.items()}
    need_n1_k = {1: 2 * SEAL_N1_K[0], 2: 2 * SEAL_N1_K[1]}
    need_n2 = 2 * SEAL_SHARES_V2["N2"]
    need_n2_k = {1: 2 * SEAL_N2_K[0], 2: 2 * SEAL_N2_K[1]}
    need_n3 = 2 * N3_SEAL
    for name, got, need in (
            ("N1", cen_n1c["clean"], need_n1_total),
            ("N1/CD-1", n1_hosts_c.get("CD-1", 0), need_n1_host["CD-1"]),
            ("N1/CD-2", n1_hosts_c.get("CD-2", 0), need_n1_host["CD-2"]),
            ("N1/CD-4", n1_hosts_c.get("CD-4", 0), need_n1_host["CD-4"]),
            ("N1/k1", n1_k_c.get(1, 0), need_n1_k[1]),
            ("N1/k2", n1_k_c.get(2, 0), need_n1_k[2]),
            ("N2", cen_n2c["clean"], need_n2),
            ("N2/k1", n2_k_c.get(1, 0), need_n2_k[1]),
            ("N2/k2", n2_k_c.get(2, 0), need_n2_k[2]),
            ("N3", cen_n3c["clean"], need_n3)):
        mark = "OK" if got >= need else "RED"
        if got < need:
            red10.append(f"{name} clean {got} < {need}")
        print(f"  {name}: clean {got} vs {need} [{mark}]")
    if red10:
        die("census G10 RED (clean-after-preflight, fresh run): "
            + "; ".join(red10)
            + " — build4 lane BuildStop, NOTHING sealed (G1: a census "
              "failure is a new addendum, not a bank iteration)")
    log4("census_v4_g10", verdict="pass", g10={s: g10[s] for s in list(ORDER)
                                               + ["N1", "N2", "N3"]},
         quotas=G10_QUOTAS_V4)
    log4_md("census_v4_g10", verdict="pass",
            clean={s: g10[s]["clean"] for s in ORDER},
            carriers={c: g10[c]["clean"] for c in ("N1", "N2", "N3")})
    print("census G10 PASS")

    # --- classic supply census (CD strata draw classic-only) ---------------
    print("== supply census: classic depth vs seal shares ==")
    supply = {}
    starved = []
    for s in ORDER:
        cands = by_s_g10.get(s, [])
        seen: set[str] = set()
        clean = 0
        for x in cands:
            ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
            if _forbidden_source(ha, hb, sources):
                continue
            if ha in seen or hb in seen:
                continue
            seen.update((ha, hb))
            clean += 1
        want = SEAL_SHARES_V2[s]
        supply[s] = {"pool": len(cands), "want": want, "clean": clean}
        if clean < want:
            starved.append(f"{s} clean {clean} < share {want}")
    log4("supply_census_v4", supply=supply, starved=starved)
    log4_md("supply_census_v4", supply=supply, starved=starved)
    if starved:
        die("supply census RED: " + "; ".join(starved)
            + " — BuildStop, NOTHING sealed")

    # --- F6 sha-convention mechanics (v2 pool reconstruction) --------------
    disk_pool_by_id: dict[str, set[str]] = {}
    with (CORPUS_DIR_V2 / "holdout-pool.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            disk_pool_by_id.setdefault(row["pair_id"], set()).update(
                {_disk_env_sha(row["envelope_a"]),
                 _disk_env_sha(row["envelope_b"])})
    conv_pool = list(gen_pool(SEEDS_V2["hold"], "ho", rotate=True)) \
        + gen_n1_pool(SEEDS_V2["hold"], "ho") \
        + gen_n2_pool(SEEDS_V2["hold"], "ho") \
        + gen_n3_pool(SEEDS_V2["hold"], "ho")
    conv_n = 0
    for x in conv_pool:
        want = disk_pool_by_id.get(x.scenario_id)
        if want is None:
            continue
        if {lib.sha(x.text_a()), lib.sha(x.text_b())} != want:
            die(f"sha convention RED at {x.scenario_id}")
        conv_n += 1
    conv_ok = conv_n == len(disk_pool_by_id)
    if not conv_ok:
        die(f"sha convention RED: {conv_n}/{len(disk_pool_by_id)} reproduce")

    # --- 7. cut 180 (SEAL_SHARES_V2; merged sources; C3 promotion) ---------
    print("== cut: 180 per SEAL_SHARES_V2 (sha-only culling, hardest-first) ==")
    scaf_ho: dict[str, int] = dict(tr_scaf_all)
    taken_shas: set[str] = set()
    seal_ids: list[str] = []
    seal_taken: set[str] = set()
    origin_of: dict[str, str] = {}
    carrier_of: dict[str, str] = {}
    culled = 0
    cull_by_src: Counter = Counter()
    blocked: Counter = Counter()
    want_classic = {s: SEAL_SHARES_V2[s] for s in ORDER}

    def cut_one(x: lib.Scenario) -> str | None:
        nonlocal culled
        if x.scenario_id in seal_taken:
            blocked["dup-id"] += 1
            return "dup-id"
        ha, hb = lib.sha(x.text_a()), lib.sha(x.text_b())
        src = _forbidden_source(ha, hb, sources)
        if src:
            cull_by_src[src] += 1
            culled += 1
            return f"culled:{src}"
        if ha in taken_shas or hb in taken_shas:
            blocked["dup"] += 1
            return "dup"
        if scaf_ho.get(x.scaffold_id, 0) >= SCAFFOLD_CAP_V2[x.stratum]:
            blocked["cap"] += 1
            return "cap"
        seal_ids.append(x.scenario_id)
        seal_taken.add(x.scenario_id)
        if x.scenario_id.startswith(V4_PREFIX + "-"):
            origin_of[x.scenario_id] = "transfer:v4bank"
        elif x.scenario_id.startswith(V3_PREFIX + "-"):
            origin_of[x.scenario_id] = "transfer:v3bank"
        else:
            origin_of[x.scenario_id] = "indist:classic"
        if x.carrier:
            carrier_of[x.scenario_id] = x.carrier
        scaf_ho[x.scaffold_id] = scaf_ho.get(x.scaffold_id, 0) + 1
        taken_shas.update((ha, hb))
        return "taken"

    def merged_candidates(s: str) -> list[lib.Scenario]:
        cands = sorted(by_s_g10.get(s, []), key=lambda x: (-x._cos, x.scenario_id))
        if s in ("CD-1", "CD-2", "CD-3", "CD-4", "CD-5", "CD-6"):
            return cands
        return cands  # NC strata draw merged (the pool IS merged)

    depth_report = {}
    nc4_split = {}
    for s in ORDER:
        blocked.clear()
        cands = merged_candidates(s)
        got = 0
        classic_n = sum(1 for x in cands if not x.scenario_id.startswith((V3_PREFIX + "-", V4_PREFIX + "-")))
        for x in cands:
            if got >= want_classic[s]:
                break
            if cut_one(x) == "taken":
                got += 1
        depth_report[s] = {"pool": len(cands), "want": want_classic[s],
                           "got": got, "blocked": dict(blocked),
                           "classic_pool": classic_n}
        if got < want_classic[s]:
            die(f"build4 disjointness: {s} exhausted "
                f"({got}/{want_classic[s]}) over the merged pool")
    print("depth:", json.dumps(depth_report))
    # N1 slice: per-host k targets (mirror of the ratified v2 spread)
    n1_ho = sorted([x for x in hn1 if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    n1_host_got = {h: 0 for h in SEAL_N1_HOSTS}
    for h, want_h in SEAL_N1_HOSTS.items():
        for k, want_k in zip((1, 2), SEAL_N1_K_TARGETS[h]):
            got_k = 0
            for x in n1_ho:
                if got_k >= want_k:
                    break
                if x.stratum != h or len(x.benign) != k:
                    continue
                if cut_one(x) == "taken":
                    got_k += 1
                    n1_host_got[h] += 1
            if got_k < want_k:
                die(f"build4 disjointness: N1/{h} k={k} exhausted ({got_k}/{want_k})")
        if n1_host_got[h] < want_h:
            die(f"build4 disjointness: N1/{h} exhausted ({n1_host_got[h]}/{want_h})")
    n2_ho = sorted([x for x in hn2 if carrier_suitable(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    n2_k_got = {1: 0, 2: 0}
    for k, want_k in zip((1, 2), SEAL_N2_K):
        for x in n2_ho:
            if n2_k_got[k] >= want_k:
                break
            if len(x.benign) != k:
                continue
            if cut_one(x) == "taken":
                n2_k_got[k] += 1
        if n2_k_got[k] < want_k:
            die(f"build4 disjointness: N2 k={k} exhausted ({n2_k_got[k]}/{want_k})")
    n3_ho = sorted([x for x in hn3 if lib.mechanical_ok(x)],
                   key=lambda x: (-x._cos, x.scenario_id))
    got_n3 = 0
    for x in n3_ho:
        if got_n3 >= N3_SEAL:
            break
        if cut_one(x) == "taken":
            got_n3 += 1
    if got_n3 < N3_SEAL:
        die(f"build4 disjointness: N3 exhausted ({got_n3}/{N3_SEAL})")
    assert len(seal_ids) == HOLD_SEAL, len(seal_ids)
    ho_by_id = {x.scenario_id: x for x in hold_pool}
    ho_by_id.update({x.scenario_id: x for x in hn1 + hn2 + hn3})
    assert set(seal_ids) <= set(ho_by_id), "seal id outside generated pools"
    nc4_split = {
        "old_bank": sum(1 for i in seal_ids
                        if ho_by_id[i].stratum == "NC-4"
                        and origin_of[i] == "indist:classic"),
        "promoted_axis_v3": sum(1 for i in seal_ids
                                if ho_by_id[i].stratum == "NC-4"
                                and origin_of[i] == "transfer:v3bank"),
        "rule": "C3: merged hardest-first (old bank + promoted v3 axis); "
                "the v3 G7 fallback-only rule is dead"}
    origins_total = Counter(origin_of[i] for i in seal_ids)
    log4("cut_v4", n=len(seal_ids), culled=culled,
         cull_by_src=dict(cull_by_src), depth=depth_report,
         origins=dict(origins_total), nc4=nc4_split,
         carriers=dict(Counter(carrier_of.values())))
    log4_md("cut_v4", n=len(seal_ids), culled=culled,
            cull_by_src=dict(cull_by_src),
            origins={k: v for k, v in origins_total.items()},
            nc4=nc4_split, carriers=dict(Counter(carrier_of.values())))
    print(f"cut: {len(seal_ids)} pairs ({culled} culled: {dict(cull_by_src)})")

    # --- 8. asserts (stop-first) -------------------------------------------
    print("== asserts (ADDENDUM-7 C1-C3 map), stop-first ==")
    A: dict[str, dict] = {}

    def rec(name: str, ok, detail) -> bool:
        A[name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'RED'}  {name}: {str(detail)[:200]}")
        return bool(ok)

    seal_sc = [ho_by_id[i] for i in seal_ids]
    cnt = Counter(x.stratum for x in seal_sc)
    n1_sc = [x for x in seal_sc if x.carrier == "N1"]
    n2_sc = [x for x in seal_sc if x.carrier == "N2"]
    n3_sc = [x for x in seal_sc if x.carrier == "N3"]
    fold_expected = Counter()
    for s in ORDER:
        fold_expected[s] += SEAL_SHARES_V2[s]
    for h, q in SEAL_N1_HOSTS.items():
        fold_expected[h] += q
    fold_expected["NC-2"] += SEAL_SHARES_V2["N2"]
    fold_expected["NC-5"] += N3_SEAL
    classic_cnt = Counter(x.stratum for x in seal_sc if x.carrier is None)
    classic_ok = all(classic_cnt[s] == SEAL_SHARES_V2[s] for s in ORDER)
    n1_ok = (Counter(x.stratum for x in n1_sc) == Counter(SEAL_N1_HOSTS)
             and Counter(len(x.benign) for x in n1_sc) ==
             Counter({1: SEAL_N1_K[0], 2: SEAL_N1_K[1]}))
    n2_ok = (len(n2_sc) == SEAL_SHARES_V2["N2"]
             and Counter(len(x.benign) for x in n2_sc) ==
             Counter({1: SEAL_N2_K[0], 2: SEAL_N2_K[1]}))
    n3_ok = len(n3_sc) == N3_SEAL
    rec("A-share-table", classic_ok and n1_ok and n2_ok and n3_ok
        and dict(cnt) == dict(fold_expected) and sum(cnt.values()) == HOLD_SEAL,
        {"classic": dict(classic_cnt),
         "N1_hosts": dict(Counter(x.stratum for x in n1_sc)),
         "N1_k": dict(Counter(len(x.benign) for x in n1_sc)),
         "N2_k": dict(Counter(len(x.benign) for x in n2_sc)),
         "N3": len(n3_sc), "fold": dict(cnt),
         "fold_expected": dict(fold_expected)})
    bad = [x.scenario_id for x in seal_sc if not lib.mechanical_ok(x)]
    len_bad = [x.scenario_id for x in seal_sc
               if x.carrier in ("N1", "N2") and not lib.len_channel_ok(x)]
    rec("A-silver+len4", not bad and not len_bad,
        f"{len(seal_sc)-len(bad)}/{len(seal_sc)} mechanical pass; "
        f"E4 len violations {len(len_bad)}")
    badl = [x.scenario_id for x in seal_sc
            if x.label != ("contradicts" if x.stratum.startswith("CD")
                           else "not-contradicts")]
    rec("A-labels", not badl, f"bad={len(badl)}")
    g_nc3 = sum(1 for x in seal_sc if x.stratum == "NC-3" and lib.version_guard(x))
    n_nc3 = sum(1 for x in seal_sc if x.stratum == "NC-3")
    g_cd = sum(1 for x in seal_sc if x.stratum.startswith("CD")
               and lib.version_guard(x))
    rec("A-guard", g_nc3 == n_nc3 and g_cd == 0,
        f"nc3 {g_nc3}/{n_nc3}, cd {g_cd}/0")

    # union scaffold caps (train-v4 + seal) per stratum
    scaf_bad, share_bad = [], []
    tr_scaf_v4: dict[str, int] = dict(c1["tr_scaf"])
    tr_scaf_v4_by_s: dict[str, dict[str, int]] = {s: dict(v) for s, v
                                                  in c1["tr_scaf_by_s"].items()}
    for c in carriers:
        g = c["sc"].scaffold_id
        s = c["sc"].stratum
        tr_scaf_v4[g] = tr_scaf_v4.get(g, 0) + 1
        tr_scaf_v4_by_s.setdefault(s, {})
        tr_scaf_v4_by_s[s][g] = tr_scaf_v4_by_s[s].get(g, 0) + 1
    for s in ORDER:
        v4_cls = Counter(x.scaffold_id for x in seal_sc if x.stratum == s)
        for g, c in v4_cls.items():
            if tr_scaf_v4.get(g, 0) + c > SCAFFOLD_CAP_V2[s]:
                scaf_bad.append(f"{s}/{g}: {tr_scaf_v4.get(g, 0)}+{c} > cap "
                                f"{SCAFFOLD_CAP_V2[s]}")
        tr_cls = tr_scaf_v4_by_s.get(s, {})
        total = sum(tr_cls.values()) + sum(v4_cls.values())
        mx = max([tr_cls.get(g, 0) + c for g, c in v4_cls.items()] + [0])
        if total and mx / total > lib.CAP + 1e-9:
            share_bad.append(f"{s}: union share {mx/total:.3f} > {lib.CAP}")
        if total and len(tr_cls) < 20:
            scaf_bad.append(f"{s}: train-v4 scaffolds {len(tr_cls)} < 20")
    rec("A-scaffold-union", not scaf_bad and not share_bad,
        {"cap_violations": scaf_bad[:4], "share_violations": share_bad[:4],
         "rule": "train-v4 (classic v2 + C1 carriers) + seal-v4 <= cap; "
                 "union max share <= 5%; >= 20 scaffolds per class"})

    seal_shas = []
    for x in seal_sc:
        seal_shas += [lib.sha(x.text_a()), lib.sha(x.text_b())]
    dup = len(seal_shas) - len(set(seal_shas))
    seal_pair_keys = [tuple(sorted((lib.sha(x.text_a()), lib.sha(x.text_b()))))
                      for x in seal_sc]
    pair_dups = len(seal_pair_keys) - len(set(seal_pair_keys))
    sha_owner: dict[str, str] = {}
    cross_pair_shares = 0
    for x in seal_sc:
        pid = x.scenario_id
        for h in (lib.sha(x.text_a()), lib.sha(x.text_b())):
            if h in sha_owner and sha_owner[h] != pid:
                cross_pair_shares += 1
            sha_owner.setdefault(h, pid)
    ab_same = sum(1 for x in seal_sc
                  if lib.sha(x.text_a()) == lib.sha(x.text_b()))
    c1_new_shas: set[str] = set()
    for c in carriers:
        sc = c["sc"]
        base = c["base"]
        if sc.text_a() != base.text_a():
            c1_new_shas.add(lib.sha(sc.text_a()))
        if sc.text_b() != base.text_b():
            c1_new_shas.add(lib.sha(sc.text_b()))
    for name, sha_set in (("train-v2", sources["train-v2"]),
                          ("c1-carriers", c1_new_shas),
                          ("v3-seal", sources["v3-seal"]),
                          ("v2-seal", sources["v2-seal"]),
                          ("v1-seal", sources["v1-seal"])):
        hits = sum(1 for h in seal_shas if h in sha_set)
        rec(f"A-disjoint-{name}", hits == 0,
            f"{hits} sha hits of {len(seal_shas)} against a {len(sha_set)}-sha "
            "set" + (" (substituted/fresh sides)" if name == "c1-carriers"
                     else " (seal ids -> pool-origin shas; sealed file NOT "
                     "opened)" if name.endswith("-seal") else " (disk)"))
    rec("A-sha-unique", pair_dups == 0 and cross_pair_shares == 0,
        {"unique_pairs": len(set(seal_pair_keys)), "of": HOLD_SEAL,
         "cross_pair_sha_shares": cross_pair_shares,
         "within_pair_a_eq_b": ab_same, "side_list_dup": dup,
         "note": "pair-level + cross-pair side uniqueness is the gate; "
                 "within-pair a==b is the NC-4 metadata-delta semantics "
                 "(embed-text convention excludes language/record_type; "
                 "the v3-literal 360-form assumed no promoted-axis rows; "
                 "C3 changes that) — disclosed, counted separately"})
    rec("A-sha-convention", conv_ok,
        f"{conv_n}/{len(disk_pool_by_id)} v2 pool ids reproduce identical "
        "disk shas in-process (F6 mechanics; sealed file untouched)")

    # G3/G7 marking + C3 promotion disclosure + carrier marking
    transfer_split = {}
    for s in ORDER:
        ids_s = [i for i in seal_ids if ho_by_id[i].stratum == s]
        transfer_split[s] = {
            "transfer_v4bank": sum(1 for i in ids_s
                                   if origin_of[i] == "transfer:v4bank"),
            "transfer_v3bank": sum(1 for i in ids_s
                                   if origin_of[i] == "transfer:v3bank"),
            "indist": sum(1 for i in ids_s
                          if origin_of[i] == "indist:classic")}
    mark_ok = all(
        (i.startswith(V4_PREFIX + "-"))
        == (origin_of[i] == "transfer:v4bank")
        and (i.startswith(V3_PREFIX + "-"))
        == (origin_of[i] == "transfer:v3bank")
        for i in seal_ids)
    rec("A-origin-marking", mark_ok,
        {"origins": dict(origins_total), "per_stratum": transfer_split,
         "nc4": nc4_split, "carriers_seal": dict(Counter(carrier_of.values())),
         "nc1_note": "NC-1 draws from merged (classic+v3-bank) — the "
                     "classic universe is exhausted at clean 0-1",
         "rule": "ho4- -> transfer:v4bank, ho3- -> transfer:v3bank, "
                 "ho- -> indist:classic; carrier field marks N1/N2/N3"})
    dated = [x for x in seal_sc if x.side_a.created_at and x.side_b.created_at]
    older = sum(1 for x in dated if x.side_a.created_at < x.side_b.created_at)
    p_older = older / max(len(dated), 1)
    lens_seal = Counter(x.length_channel for x in seal_sc)
    rec("A-indep+len", abs(p_older - 0.5) <= 0.10,
        {"P(a_older)": round(p_older, 3), "dated": len(dated),
         "gate": "|P - 0.5| <= 0.10", "channels_report": dict(lens_seal),
         "len_note": "the 1/3 length balance is a TRAIN-side design axis "
                     "(4.4 A-len-indep, asserted on train-v4 carriers); the "
                     "seal is hardest-first by cos and the razor band is "
                     "long-channel physics — the RATIFIED v3 seal shipped "
                     "long 128 / medium 38 / short 14; a v4 long-dominated "
                     "shape is v3-parity, reported not gated"})
    rec("A-frozen", True, f"nano {lib.EMBEDDER_PIN[:12]}…; cortex features "
        "imported; instrument pin verified at Cos init")
    rec("A-archive-integrity", all(archives_intact.values()),
        {k: ("digest-ok" if v else "RED") for k, v in archives_intact.items()})

    # ---- train-v4 legs ------------------------------------------------------
    quota = pin["prep"]["carriers_c1_quota"]["per_stratum"]
    carriers_ok = all(len(c1_by_stratum[s]) == quota[s] for s in UNCOVERED_STRATA)
    rec("T-carrier-quota", carriers_ok,
        {"pinned": quota, "emitted": {s: len(c1_by_stratum[s])
                                      for s in UNCOVERED_STRATA},
         "total": total_carriers,
         "note": "measure-then-fix: emission == census pin (exact)"})
    c1_pred_bad = []
    for c in carriers:
        if c["rung"] == "base":
            if not lib.mechanical_ok(c["sc"]):
                c1_pred_bad.append(c["sc"].scenario_id)
        elif not c1m.orbit_ok(c["sc"], c["base"]):
            c1_pred_bad.append(c["sc"].scenario_id)
    lab_c1 = [c["sc"].scenario_id for c in carriers
              if c["sc"].label != ("contradicts" if c["sc"].stratum.startswith("CD")
                                   else "not-contradicts")
              or c["sc"].label != c["base"].label]
    raz_c1 = [c["sc"].scenario_id for c in carriers if c["sc"]._cos < c1m.RAZOR_LO]
    ch_c1 = [c["sc"].scenario_id for c in carriers
             if c["sc"].length_channel != c["base"].length_channel]
    cp_bad = [c["base"].scenario_id for c in carriers
              if not c1m.cosmetic_proof(c["base"].side_a)
              or not c1m.cosmetic_proof(c["base"].side_b)]
    rec("T-carrier-silver", not c1_pred_bad and not raz_c1 and not ch_c1,
        {"pred_bad": c1_pred_bad[:4], "below_razor": raz_c1[:4],
         "channel_shift": ch_c1[:4],
         "note": "orbit_ok on rungs (sentence-pinned substitution), "
                 "mechanical_ok on fresh bases; razor retention; E4 channel"})
    rec("T-labels+invariance", not lab_c1,
        {"bad": lab_c1[:4],
         "note": "label derivable from stratum AND == base label "
                 "(NA label-invariance constraint)"})
    rec("T-cosmetic-proof", not cp_bad,
        {"bad": cp_bad[:4],
         "note": "case-flip variant leaves embed-text sha byte-identical "
                 "on every orbit base (the cosmetic rung, verified "
                 "not emitted)"})
    # carrier leakage guards: substituted/fresh sides vs seals + train-v2
    leak_seal = []
    leak_train = []
    for c in carriers:
        sc = c["sc"]
        new_sides = [t for t in (sc.text_a(), sc.text_b())
                     if t != c["base"].text_a() and t != c["base"].text_b()]
        for t in new_sides:
            h = lib.sha(t)
            if _forbidden_source(h, h, {k: sources[k] for k in
                                        ("v1-seal", "v2-seal", "v3-seal")}):
                leak_seal.append(sc.scenario_id)
            if h in sources["train-v2"]:
                leak_train.append(sc.scenario_id)
    for c in carriers:
        if c["kind"] == "fresh":
            sc = c["sc"]
            for t in (sc.text_a(), sc.text_b()):
                h = lib.sha(t)
                if _forbidden_source(h, h, {k: sources[k] for k in
                                            ("v1-seal", "v2-seal", "v3-seal")}):
                    leak_seal.append(sc.scenario_id)
                if h in sources["train-v2"]:
                    leak_train.append(sc.scenario_id)
    rec("T-carrier-disjoint", not leak_seal and not leak_train,
        {"seal_hits": leak_seal[:4], "train_side_hits": leak_train[:4],
         "note": "substituted/fresh side shas are new vs ALL sealed "
                 "generations and vs train-v2 sides (F10 leakage guard); "
                 "the UNCHANGED side legitimately shares its own base "
                 "(orbit structure)"})
    # train-v4 sha pair uniqueness + E7 side balance + null discipline
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
                lib.benign_substitute(body, k, (base.side_a if side == "a"
                                                else base.side_b).language
                                      or "ru", random.Random(0))
                fit.append(side)
            except ValueError:
                pass
        if len(fit) == 2:
            both_fit += 1
            d = c["sc"].benign[0]["side"]
            side_a_n += 1 if d == "a" else 0
    null_c = sum(1 for c in carriers if c["sc"].both_null)
    mixed_c = sum(1 for c in carriers
                  if (c["sc"].side_a.created_at is None)
                  != (c["sc"].side_b.created_at is None))
    rec("T-carrier-shape", dup_pairs == 0 and mixed_c == 0
        and (both_fit == 0 or abs(side_a_n / both_fit - 0.5) <= 0.10),
        {"pair_dups": dup_pairs,
         "sha_pair_collisions_meta_distinguished": meta_distinguished,
         "mixed_null": mixed_c,
         "e7_both_fit": both_fit, "side_a": side_a_n,
         "null_share_carriers": round(null_c / max(total_carriers, 1), 3),
         "note": "gate = no FULL-ROW duplicates (sha-pair collisions "
                 "count only when the rows are metadata-IDENTICAL); "
                 "sha-pair collisions between rows differing in "
                 "record_type/language/tags are the NC-4 metadata-delta "
                 "semantics (embed-text convention excludes those "
                 "fields) — disclosed; 0 mixed-null; E7 side balance "
                 "over both-side-fit rungs"})
    # build4.1 (ADDENDUM-7 §6): GLOBAL pair_id uniqueness — the gap the
    # trainer's P5 caught. pair_id must be unique OUTSIDE orientations
    # (one pair per id over the whole carrier block, fresh bases and
    # rungs in one shared tr4-C1 namespace); (pair_id, orientation)
    # uniqueness is structural in _carrier_rows4 (exactly 2 rows: a>b,
    # b>a) and is re-asserted on the disk artifact by the build4.1 lane.
    c1_ids = Counter(c["sc"].scenario_id for c in carriers)
    id_dups = sorted(i for i, n in c1_ids.items() if n > 1)
    rec("T-id-unique", not id_dups,
        {"dups": id_dups[:6], "distinct_pair_ids": len(c1_ids),
         "note": "build4.1 (ADDENDUM-7 §6): GLOBAL pair_id uniqueness "
                 "over the carrier block (fresh bases + rungs, ONE shared "
                 "tr4-C1 namespace) — pair_id unique outside orientations; "
                 "the classic train-v2 prefix is disjoint by id prefix"})

    all_ok = all(v["ok"] for v in A.values())
    if not all_ok:
        reds = [k for k, v in A.items() if not v["ok"]]
        log4("build_stop", reason=f"asserts RED: {reds}")
        log4_md("build_stop", reds=reds,
                note="stop-first: NOTHING written or sealed")
        print(f"BUILD-STOP: asserts RED: {reds}", file=sys.stderr)
        return 1

    # --- 9. emit (stop-first passed) ----------------------------------------
    tr2_lines = (CORPUS_DIR_V2 / "train.jsonl").read_text(encoding="utf-8")
    carrier_rows = []
    for c in sorted(carriers, key=lambda c: (c["sc"].stratum,
                                             c["sc"].scenario_id)):
        carrier_rows += _carrier_rows4(c["sc"], c["rung"],
                                       c["base"].scenario_id, c["kind"])
    CORPUS_DIR_V4.mkdir(parents=True, exist_ok=True)
    with (CORPUS_DIR_V4 / "train.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(tr2_lines)
        for r in carrier_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr4_text = (CORPUS_DIR_V4 / "train.jsonl").read_text(encoding="utf-8")
    prefix_ok = tr4_text.startswith(tr2_lines)
    fp_train = lib.sha(tr4_text)
    rows_h = []
    for sc in seal_sc:
        rows_h += rows_of(sc, "holdout", f"seed:{SEED_HOLD_V4}")
    fp_hold = write_jsonl(CORPUS_DIR_V4 / "holdout.jsonl", rows_h)
    pool_rows = []
    for sc in list(hold_pool) + list(hn1) + list(hn2) + list(hn3):
        pool_rows += rows_of(sc, "holdout-pool", f"seed:{SEED_HOLD_V4}")
    fp_pool = write_jsonl(CORPUS_DIR_V4 / "holdout-pool.jsonl", pool_rows)
    floor4 = min(x._cos for x in seal_sc)

    lineage = {
        "rotation_of": "CD-holdout-v3",
        "ancestor_holdout_sha256": man3["fingerprints"]["holdout.jsonl"],
        "ancestor_seal_sha256": man3["fingerprints"]["seal"],
        "ancestor_status": "red-gate-DECLINE",
        "ancestor_gate_state": "single legal read consumed "
                               "(sealed_read_once 2026-11-10T02:50:06Z, "
                               "sha 548eae2a…, n=180/360); verdict DECLINE "
                               "(g_stab 0.5833 — 35/60 pairs invariant, "
                               "threshold 1.0, floor_inv=0); 10 other "
                               "gates GREEN",
        "cycle2_authorization": "ADDENDUM-7 RATIFIED (MLC-7 three-leg "
                                "convergence, commit 154f7af): retrain "
                                "wave with orbit carriers (C1), bank "
                                "refresh (C2), NC-4 promotion (C3); "
                                "anchor G4: characterization GREEN + "
                                "gate RED = model defect — the corpus "
                                "was not the defect, the uncovered "
                                "carrier cell was the amplificator",
        "rotation_rule": "slice rule (fresh generation), NOT corpus "
                         "amend-once: E13 amend budget NOT spent; "
                         "train-v2 stays byte-frozen; train-v4 = "
                         "train-v2 verbatim + C1 carrier rows (declared "
                         "in this manifest)",
        "cut_rule": "SINGLE cut of 180 (SEAL_SHARES_V2 largest remainder, "
                    "hardest-first within stratum, sha-only culling F6 vs "
                    "ALL FOUR generations + C1 carriers, union scaffold "
                    "caps, G10 census-gated); one seed family member "
                    "(hold=7047); no re-cut on this artifact",
        "universes_extension": "C2 (G11): banks_v4 eval-only fork — NC-5 "
                               "6 new SPEC subjects, NC-2 8 fresh refine "
                               "combos, NC-3 2 new version families; "
                               "C3: the banks_v3 NC-4 non-tech axis "
                               "PROMOTED to the main bank (merged "
                               "hardest-first, transfer-marked)",
        "carriers_extension": "C1: para-orbits into train for the 7 "
                              "razor-uncovered strata (base + cosmetic + "
                              "para1 + para2, label-invariance); carrier "
                              "quotas pinned from the census before "
                              "emission (measure-then-fix); classic "
                              "shares untouched",
        "disclosure_unseen_scaffold": "holdout v4 carries unseen-scaffold "
                                      "rows of the extended classes "
                                      "(ADDENDUM-6 §2.4/G12 lineage rule "
                                      "carried forward); the model card "
                                      "known-limitations must repeat this "
                                      "line verbatim",
        "f1_single_shot": "a red gate v4 is FINAL on this artifact + "
                          "thresholds; any next attempt is a new cycle "
                          "with a fresh seal (F1/F3)",
    }
    seal_payload = {"ts": now(), "version": "CD-holdout-v4",
                    "holdout_jsonl_sha256": fp_hold,
                    "ids": sorted(set(seal_ids)), "n": len(seal_sc),
                    "lineage": lineage,
                    "disjointness": {
                        "rule": "sha(text_a/b) ONLY (F6); scenario_id "
                                "excluded (deterministic template grid "
                                "id-space)",
                        "checked_against": {
                            "train-v2": "disk train.jsonl text-sha set",
                            "train-v4-carriers": "C1 substituted/fresh "
                                                 "side shas (this build)",
                            "CD-holdout-v3": "seal ids -> holdout-pool.jsonl "
                                             "origin text-shas",
                            "CD-holdout-v2": "seal ids -> holdout-pool.jsonl "
                                             "origin text-shas",
                            "CD-holdout-v1": "seal ids -> holdout-pool.jsonl "
                                             "origin text-shas"},
                        "sealed_content_reads": {
                            "corpus-v1/holdout.jsonl": 0,
                            "corpus-v2/holdout.jsonl": 0,
                            "corpus-v3/holdout.jsonl": 0}},
                    "archives_digest_intact": {k: bool(v) for k, v
                                               in archives_intact.items()},
                    "note": "content NOT read after this event"}
    SEALS_DIR.mkdir(parents=True, exist_ok=True)
    SEAL_V4_PATH.write_text(json.dumps(seal_payload, ensure_ascii=False,
                                       indent=2), encoding="utf-8")
    fp_seal = lib.sha(SEAL_V4_PATH.read_text(encoding="utf-8"))

    pin = json.loads(PIN_V4_PATH.read_text(encoding="utf-8"))
    if pin.get("prep", {}).get("banks_v4", {}).get("content_sha256") \
            not in (None, b4_content) or \
            pin.get("prep", {}).get("carriers_c1", {}).get("content_sha256") \
            not in (None, c1_content):
        die("pin-v4 prep pins DIFFERENT module content than the "
            "runtime-verified modules — G1 inconsistency, refusing")
    pin.update({
        "ts": now(), "version": 4,
        "lane": "build4 (ADDENDUM-7 §4, cycle-2): train-v4 = train-v2 "
                "verbatim + C1 orbit carriers; eval pool = fresh seed4 "
                "generation over classic + banks_v3 + banks_v4 (C2) + "
                "promoted NC-4 axis (C3) -> G10 census -> single cut of "
                "180 -> seal CD-holdout-v4",
        "instrument": {"pin": lib.EMBEDDER_PIN,
                       "guard_version": lib.GUARD_VERSION},
        "seeds": {"hold": SEED_HOLD_V4},
        "seed_family_check": "single-member family (hold only; the C1 "
                             "train-side carriers derive from train-v2 + "
                             "the frozen PARA dial, not from a train seed); "
                             f"!= {sorted(SEEDS_FORBIDDEN_V4 | {SEED_HOLD_V3})}, "
                             f"!= all {len(observed)} seeds seen in "
                             "seeds_pinned* run-log events",
        "runlog_seed_scan": {"events": len(scan),
                             "distinct_seeds": len(observed),
                             "seeds": observed},
        "banks_v3_frozen": {"banks_commit": bv3["banks_commit"],
                            "file_sha256": b3_sha,
                            "content_sha256": b3_content,
                            "runtime_verified": True,
                            "note": "byte-frozen continuity from v3"},
        "banks_v4_frozen": {"banks_commit": prep4["banks_v4"]["banks_commit"],
                            "file_sha256": b4_sha,
                            "content_sha256": b4_content,
                            "counts": prep4["banks_v4"]["counts"],
                            "runtime_verified": True,
                            "lock": "G1: bank edits forbidden; census/"
                                    "assert failure = BuildStop + new "
                                    "addendum"},
        "carriers_c1_frozen": {"banks_commit":
                               prep4["carriers_c1"]["banks_commit"],
                               "file_sha256": c1_sha,
                               "content_sha256": c1_content,
                               "runtime_verified": True,
                               "lock": "G1: carrier edits forbidden"},
        "carriers_c1": {
            "quotas": quotas_c1,
            "rungs": rungs_c1,
            "total": total_carriers,
            "orbit_shape": "base + cosmetic (verified, not emitted) + "
                           "para1 + para2; NC-4 symmetric; fresh CD-5 "
                           "orbits = base + para1 (8)",
            "label_invariance": "NA constraint: pair + perturbed variants "
                                "carry ONE label",
            "cull_ledger": dict(c1_rep["cull"]),
            "per_stratum": c1_rep["per_stratum"]},
        "hard_window": [HARD_WIN_LO, HARD_WIN_HI],
        "measured_holdout_floor": floor4,
        "g10_census": {s: g10[s] for s in list(ORDER) + ["N1", "N2", "N3"]},
        "seal_shares": {**{s: SEAL_SHARES_V2[s] for s in ORDER},
                        "N1": sum(SEAL_N1_HOSTS.values()),
                        "N2": SEAL_SHARES_V2["N2"], "N3": N3_SEAL},
        "cut_rule": lineage["cut_rule"],
        "transfer_split": {
            "rule": "G3 carried forward: ho4- rows = transfer:v4bank, "
                    "ho3- rows = transfer:v3bank, ho- rows = indist; "
                    "the v4 gate report splits by this marking; "
                    "thresholds unchanged",
            "per_stratum": transfer_split,
            "nc4_promotion": nc4_split,
            "carrier_split": dict(Counter(carrier_of.values()))},
        "seal_fold_strata": dict(cnt),
        "seal_ids": seal_ids,
        "bands_seal": {b: sum(1 for x in seal_sc if x._band == b)
                       for b, _, _ in BANDS},
        "reporting_rule": SEAL3_REPORTING_RULE,
        "lineage": lineage,
        "train_v4": {"classic": "train-v2 verbatim (600 pairs, byte-identical "
                                "prefix)",
                     "carriers": {s: len(c1_by_stratum[s])
                                  for s in UNCOVERED_STRATA},
                     "total_pairs": 600 + total_carriers,
                     "train_v2_sha_unchanged": train_fp_start},
        "train_v2_untouched": {"train.jsonl_sha256_start": train_fp_start},
    })
    PIN_V4_PATH.write_text(json.dumps(pin, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    manifest = {
        "ts": now(),
        "version": "corpus-v4 (ADDENDUM-7 cycle-2: C1 orbit carriers + "
                   "C2/C3 bank refresh + seal CD-holdout-v4, B6-conflict "
                   "build4)",
        "guard_version": lib.GUARD_VERSION,
        "instrument": {"provider": "vesma.embeddings.NanoProvider",
                       "nano_pin": lib.EMBEDDER_PIN},
        "design": ["docs/experiments/conflict-detector-corpus-design-"
                   "ADDENDUM-7.md (RATIFIED, C1-C9, commit 154f7af)",
                   "docs/experiments/conflict-detector-corpus-design-"
                   "ADDENDUM-6.md (RATIFIED, path B, G1-G12)",
                   "docs/experiments/conflict-detector-corpus-design.md "
                   "(RATIFIED MLC-9)",
                   "docs/experiments/conflict-detector-corpus-design-"
                   "ADDENDUM-1.md",
                   "docs/experiments/conflict-detector-corpus-design-"
                   "ADDENDUM-3.md"],
        "banks_v4_frozen": {"banks_commit":
                            prep4["banks_v4"]["banks_commit"],
                            "file_sha256": b4_sha,
                            "content_sha256": b4_content},
        "carriers_c1_frozen": {"banks_commit":
                               prep4["carriers_c1"]["banks_commit"],
                               "file_sha256": c1_sha,
                               "content_sha256": c1_content},
        "pin_sha256": lib.sha(PIN_V4_PATH.read_text(encoding="utf-8")),
        "seeds": {"hold": SEED_HOLD_V4},
        "seed_family_check": pin["seed_family_check"],
        "strata_v4": {s: {"classic_v2": CLASS_TOTALS_V2[s],
                          "carriers_c1": len(c1_by_stratum.get(s, []))
                          if s in UNCOVERED_STRATA else 0,
                          "train_v4_total": CLASS_TOTALS_V2[s]
                          + (len(c1_by_stratum.get(s, []))
                             if s in UNCOVERED_STRATA else 0),
                          "seal": cnt[s]}
                      for s in ORDER},
        "carriers_c1": pin["carriers_c1"],
        "transfer_split": pin["transfer_split"],
        "g10_census": {s: g10[s] for s in list(ORDER) + ["N1", "N2", "N3"]},
        "asserts": A,
        "bands_seal": pin["bands_seal"],
        "lineage": lineage,
        "para_dict_sha256": lib.PARA_DICT_SHA,
        "known_uncovered": {"cos_below": floor4,
                            "note": "measured floor over the emitted v4 "
                                    "seal"},
        "disjointness": {"rule": "sha(text_a/b) only (F6)",
                         "culled": culled,
                         "cull_by_src": dict(cull_by_src),
                         "sha_set_sizes": {k: len(v) for k, v
                                           in sources.items()},
                         "sealed_files_opened": 0},
        "cut_depth": depth_report,
        "emitted": ["train.jsonl (600 classic verbatim + "
                    f"{total_carriers} C1 carriers)",
                    "holdout.jsonl (180 pairs, sealed)",
                    f"holdout-pool.jsonl ({len(pool_rows)} rows, wide "
                    "generation provenance)", "pin-v4.json",
                    "seals/CD-holdout-v4.seal.json"],
        "absent_by_construction": ["sheet/* (cycle-2 sheet lane is separate)"],
        "regime": "synthetic-only; no store texts; no LLM in construction; "
                  "labels by construction",
        "fingerprints": {"train.jsonl": fp_train,
                         "holdout.jsonl": fp_hold,
                         "holdout-pool.jsonl": fp_pool, "seal": fp_seal},
    }
    (CORPUS_DIR_V4 / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    (RUNLOG.parent / "seal_v4_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    train_fp_end = file_sha256(CORPUS_DIR_V2 / "train.jsonl")
    untouched = train_fp_end == train_fp_start \
        == man2["fingerprints"]["train.jsonl"]
    n_pass = sum(1 for v in A.values() if v["ok"])
    log4("seal_v4", seal="CD-holdout-v4", n=len(seal_sc), sha=fp_seal,
         holdout_sha=fp_hold, pool_sha=fp_pool,
         train_v4_sha=fp_train, train_carriers=total_carriers,
         lineage_ancestor="CD-holdout-v3 (red-gate-DECLINE)",
         holdout_content_reads=0)
    log4_md("seal_v4", seal="CD-holdout-v4", n=len(seal_sc), sha=fp_seal,
            holdout_sha=fp_hold, train_sha=fp_train,
            seal_n=len(seal_sc),
            asserts=f"{n_pass}/{len(A)} PASS", holdout_content_reads=0)
    log4("build4_done", ok=True, train_v2_untouched=bool(untouched),
         classic_prefix_ok=bool(prefix_ok),
         elapsed_s=round(time.time() - t0, 1),
         asserts=f"{n_pass}/{len(A)} PASS")
    log4_md("build4_done", ok=True, train_v2_untouched=bool(untouched),
            elapsed_s=round(time.time() - t0, 1),
            asserts=f"{n_pass}/{len(A)} PASS")
    print(f"train-v2 byte-identity after run: {'OK' if untouched else 'RED'} "
          f"({train_fp_end[:16]}); train-v4 classic prefix byte-identical: "
          f"{'OK' if prefix_ok else 'RED'}")
    print(f"\nBUILD4: ALL ASSERTS PASS in {time.time()-t0:.0f}s\n"
          f"pin: {PIN_V4_PATH}\nseal: {SEAL_V4_PATH}\n"
          f"corpus: {CORPUS_DIR_V4} (train-v4 + holdout + pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
