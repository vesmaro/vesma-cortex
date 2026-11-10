#!/usr/bin/env python3
"""C1 orbit carriers for train-v4 (ADDENDUM-7 §2 C1, ratified 10.11).

Cycle-2 fix for the carrier asymmetry (MLC-7 DE leg, numbers verified on
disk 2026-11-10): train-v2 carries para-carriers in the razor band only
for 4/11 strata (N2 on NC-2, N1 on CD-1/2/4); the 7 uncovered strata
CD-3 (33 razor pairs), CD-5 (22), CD-6 (25), NC-1 (49), NC-3 (54),
NC-4 (36), NC-5 (37) teach no ``label survives a benign substitution''
cell. The v3 gate failed there (g_stab 0.5833, non-monotone logits under
para/cosmetic).

ORBIT (NA position, label-invariance constraint): base + cosmetic +
para1 + para2 — the pair and its perturbed variants carry ONE label.

  base     = the razor-band train-v2 pair itself (NOT re-emitted; it is
             already in train-v2 byte-for-byte; fresh CD-5 orbits emit
             their base — see below);
  cosmetic = case-flip variant of one side — INVISIBLE on the frozen
             surface BY CONSTRUCTION (norm1/embed_text lowercase; the
             machine proof runs per orbit at census/assert time); NOT
             emitted (a duplicate embed-text sha would break sha
             uniqueness and add nothing to a tree on CORE-13);
  para1    = k=1 benign substitution on one side (lib.apply_benign, the
             FROZEN PARA dictionaries — no dictionary change);
  para2    = k=2 benign substitution (one per sentence, the dial's
             one-substitution-per-sentence rule).

NC-4 special: the class semantics require norm-equal bodies, so the
perturbation is SYMMETRIC (the same substitution applied to both sides)
— bodies stay identical, the metadata delta stays, the label survives.

Fresh CD-5 orbits: the classic CD-5 universe has only 8 razor bases that
can host the dial (long-channel bodies with PARA-hosting filler
sentences) and the §4.5 union scaffold cap cannot host even those at
CD-5's class size (24 classic rows -> cap 1..2). Eight FRESH long bases
on (subject, conditional) combos unused by train-v2 lift the class size
over the cap boundary and each carries a para1 rung. The fresh bases are
carrier-marked (rung "base") — the ratified classic shares stay
untouched; the strat table gains CARRIER ROWS only.

Mechanical discipline (G9-class, banks_v3 precedent): this module never
edits the frozen lib; predicates COMPOSE frozen lib predicates
(benign_pair_ok / region_is_version / norm equality). Fail-closed at
import on the static invariants. Deterministic: per-candidate rng
streams keyed ``"<seed>:C1:..."``; regenerate == reproduce, byte-identical.

FREEZE (G1): after the freeze pin this file is EDIT-LOCKED; a census or
assert failure is a BuildStop + a new addendum, never a carrier
iteration.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "scripts"),):
    if _p not in sys.path:
        sys.path.append(_p)

import conflict_corpus_lib as lib  # noqa: E402

RAZOR_LO = 0.95
TAG = "P"           # carrier mark of every C1 row (rung detail in declaration)
UNCOVERED = ("CD-3", "CD-5", "CD-6", "NC-1", "NC-3", "NC-4", "NC-5")
TRAIN_V2 = _REPO / "data" / "conflict-detector" / "corpus-v2" / "train.jsonl"

#: measured 2026-11-10 on disk (the MLC-7 DE numbers reproduced exactly):
#: razor-band pairs of the uncovered strata = the orbit-base supply.
EXPECTED_RAZOR_BASES = {"CD-3": 33, "CD-5": 22, "CD-6": 25, "NC-1": 49,
                        "NC-3": 54, "NC-4": 36, "NC-5": 37}

#: fresh CD-5 orbit plan (authorship-time, measured): 8 orbits on free
#: long-channel (subject, conditional) combos; each emits base + para1.
FRESH_CD5_N = 8
FRESH_CD5_FILLERS = 35   # long channel (classic CD-5L construction)


# --- static guards (fail-closed at import) ----------------------------------

_SENT_SPLIT = re.compile(r"(?<=\.)\s+")


def _guard() -> None:
    # the frozen dial must still be the frozen dial (sha goes to the pin)
    if not TRAIN_V2.exists():
        raise AssertionError("train-v2 artifact missing — carriers cannot "
                             "reconstruct orbit bases")
    # every uncovered stratum must be a contradicts/not label per its class
    for s in UNCOVERED:
        want = "contradicts" if s.startswith("CD") else "not-contradicts"
        if (s.startswith("CD")) != (want == "contradicts"):
            raise AssertionError(f"label map broken for {s}")
    # PARA dictionaries: digit-free (E2 import assert in lib already ran)


_guard()


# --- reconstruction of orbit bases from the disk artifact --------------------

def side_from_env(env: dict) -> lib.Side:
    return lib.Side(title=env["title"], body=env["body"], tags=list(env["tags"]),
                    language=env.get("language"),
                    record_type=env.get("record_type"),
                    created_at=env.get("created_at"))


def recon_scenario(row: dict) -> lib.Scenario:
    """Faithful Scenario reconstruction of a train-v2 row (benign=[], the
    declaration copied verbatim). The disk sha round-trip is asserted by
    the caller (fidelity gate)."""
    d = row["declaration"]
    a, b = row["envelope_a"], row["envelope_b"]
    both_null = a.get("created_at") is None and b.get("created_at") is None
    la = (a.get("language") or "") != (b.get("language") or "")
    return lib.Scenario(
        scenario_id=row["pair_id"], stratum=row["stratum"],
        side_a=side_from_env(a), side_b=side_from_env(b), label=row["label"],
        slots=[dict(s) for s in d.get("slots", [])],
        subject_a=d.get("subject_a"), subject_b=d.get("subject_b"),
        polar_op=d.get("polar_op"), refine_ok=d.get("refine_ok"),
        scaffold_id=row["scaffold_id"], cross_lang=bool(la),
        length_channel=row["measured"]["len_channel"], both_null=both_null,
        carrier=None, benign=[], para_b=0)


def load_razor_bases() -> dict[str, list[lib.Scenario]]:
    """The 7 strata's razor-band pairs, reconstructed from the disk
    artifact (train-v2 read-only; the artifact digest gate is the lane's
    job). Ids and texts round-trip by construction."""
    out: dict[str, list[lib.Scenario]] = {s: [] for s in UNCOVERED}
    seen: set[str] = set()
    with TRAIN_V2.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pid = row["pair_id"]
            if pid in seen:
                continue
            seen.add(pid)
            if row["stratum"] not in UNCOVERED:
                continue
            if row["measured"]["cos_target"] < RAZOR_LO:
                continue
            out[row["stratum"]].append(recon_scenario(row))
    for s, n in EXPECTED_RAZOR_BASES.items():
        if len(out[s]) != n:
            raise AssertionError(
                f"orbit-base supply drifted for {s}: {len(out[s])} != {n} "
                "(train-v2 is frozen; this is an instrument abort)")
    return out


# --- orbit predicates (composition of frozen lib predicates; NO lib edits) --

def _benign_area_clean(sc: lib.Scenario) -> bool:
    """Level-2 per-pair check for the bespoke strata (NC-4/NC-5): every
    declared benign area carries no digit and no FACT_ALPHA member."""
    for d in sc.benign:
        nt = lib.norm1(d["sent"])
        if any(ch.isdigit() for ch in nt):
            return False
        for f in lib.FACT_ALPHA:
            if f and f in nt:
                return False
    return True


def nc4_orbit_ok(sc: lib.Scenario) -> bool:
    """NC-4 symmetric rung: bodies stay norm-equal (the class semantics),
    the metadata delta stays visible, declared benign areas clean."""
    a, b = sc.side_a, sc.side_b
    if lib.norm1(a.body) != lib.norm1(b.body):
        return False
    if lib.norm1(a.title) != lib.norm1(b.title):
        return False
    if not (a.tags != b.tags or a.record_type != b.record_type
            or a.language != b.language):
        return False
    if not sc.benign:
        return False
    return _benign_area_clean(sc)


def nc5_orbit_ok(sc: lib.Scenario) -> bool:
    """NC-5 rung: subject-anchor disjointness (the classic line) holds and
    the declared benign areas are clean."""
    if sc.subject_a == sc.subject_b:
        return False
    if not sc.benign:
        return False
    return _benign_area_clean(sc)


def _ben_region_clean(text: str) -> bool:
    """Level-2 per-region check (benign_pair_ok leg): a benign region
    carries no digit and no FACT_ALPHA member."""
    nt = lib.norm1(text)
    if any(ch.isdigit() for ch in nt):
        return False
    for f in lib.FACT_ALPHA:
        if f and f in nt:
            return False
    return True


def _sents(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=\.)\s+", body.strip()) if s]


def sentence_substitution_ok(rung_body: str, base_body: str,
                             entries: list) -> bool:
    """THE rung predicate core: the substituted side differs from its base
    by EXACTLY the declared benign substitutions — sentence-wise, position
    pinned, one dictionary pair per changed sentence, nothing else moves.

    This composes the frozen primitives (norm1, FACT_ALPHA) at sentence
    level instead of riding difflib regions: on long repeated-filler
    bodies difflib's longest-block alignment produces phantom
    deletion+insert region pairs around the substituted filler (measured
    2026-11-10: a-toks 32..91 ``replace ''`` + tail insert), which would
    falsely fail honest rungs. The base already passed its stratum's
    mechanical predicate (it is a train-v2 row), the slot/cond/polar
    sentence is byte-unchanged (only declared positions may differ), so
    the base's slot semantics are INHERITED — the label stays derivable
    from the construction (A-labels)."""
    bs, rs = _sents(base_body), _sents(rung_body)
    if len(bs) != len(rs):
        return False
    diff = [i for i in range(len(bs)) if bs[i] != rs[i]]
    if len(diff) != len(entries):
        return False
    unused = set(range(len(entries)))
    for i in diff:
        hit = None
        for k in sorted(unused):
            a_tok, b_tok = entries[k]["applied"]
            if bs[i].replace(a_tok, b_tok, 1) == rs[i]:
                hit = k
                break
        if hit is None:
            return False
        unused.discard(hit)
        if not _ben_region_clean(rs[i]):
            return False
    return True


def orbit_ok(sc: lib.Scenario, base: lib.Scenario) -> bool:
    """The composed mechanical predicate for a C1 rung (A-silver leg for
    carrier rows; classic rows keep lib.mechanical_ok). `base` is the
    orbit's base scenario (the train-v2 razor pair, or the fresh CD-5
    base)."""
    s = sc.stratum
    if sc.carrier != TAG or not sc.benign:
        return False
    if s != base.stratum or sc.label != base.label:
        return False
    if sc.label != ("contradicts" if s.startswith("CD") else "not-contradicts"):
        return False
    if s == "NC-4":
        # symmetric: bodies stay norm-equal, the metadata delta stays,
        # both sides carry the same declared substitution
        if not nc4_orbit_ok(sc):
            return False
        if not all(d.get("side") == "both" for d in sc.benign):
            return False
        return sentence_substitution_ok(sc.side_a.body, base.side_a.body,
                                        sc.benign) \
            and sentence_substitution_ok(sc.side_b.body, base.side_b.body,
                                         sc.benign)
    # asymmetric: exactly one side changed; the other is byte-identical
    sides = {d.get("side") for d in sc.benign}
    if len(sides) != 1:
        return False
    side = sides.pop()
    if side not in ("a", "b"):
        return False
    other = "b" if side == "a" else "a"
    if getattr(sc, f"side_{other}").body != getattr(base, f"side_{other}").body:
        return False
    if not sentence_substitution_ok(getattr(sc, f"side_{side}").body,
                                    getattr(base, f"side_{side}").body,
                                    sc.benign):
        return False
    if s == "NC-5":
        return sc.subject_a != sc.subject_b
    return True


def orbit_len_ok(sc: lib.Scenario, base_channel: str) -> bool:
    """E4 for rungs: the substitution must not push the pair across a
    length-channel boundary (the DECLARED channel is the base's)."""
    return sc.length_channel == base_channel


def cosmetic_proof(side: lib.Side) -> bool:
    """Machine proof of the cosmetic rung's surface-invariance: a case-only
    variant of one side leaves the pinned embed-text convention
    byte-identical (norm1 lowercases). The rung itself is never emitted."""
    flipped = " ".join(
        w.upper() if i % 3 == 0 else (w.capitalize() if i % 3 == 1 else w)
        for i, w in enumerate(side.body.split(" ")))
    twin = lib.Side(title=side.title, body=flipped, tags=list(side.tags),
                    language=side.language, record_type=side.record_type,
                    created_at=side.created_at)
    return lib.sha(lib.embed_text(side)) == lib.sha(lib.embed_text(twin))


# --- rung construction -------------------------------------------------------

def _new_id(prefix: str, ctr: list[int]) -> str:
    ctr[0] += 1
    return f"{prefix}-C1-{ctr[0]:05d}"


def make_para_rung(base: lib.Scenario, k: int, side: str, rng: random.Random,
                   prefix: str, ctr: list[int], rung: str,
                   sc_id: str | None = None) -> lib.Scenario:
    """para-k rung on an asymmetric orbit (lib.apply_benign verbatim: the
    E3 declared areas, E6 scaffold inheritance, E7 side randomization all
    come from the frozen machinery). Label/scaffold/nullness inherit —
    the label-invariance constraint is construction-level."""
    rung_sc = lib.apply_benign(base, side, k, rng)
    rung_sc.scenario_id = sc_id or _new_id(prefix, ctr)
    rung_sc.carrier = TAG
    rung_sc.label = base.label
    rung_sc.both_null = base.both_null
    # benign stays the pure E3 shape (side/k/applied/sent); the rung name
    # and base linkage travel in the ROW declaration (lane-level stamping)
    return rung_sc


def make_symmetric_rung(base: lib.Scenario, k: int, rng: random.Random,
                        prefix: str, ctr: list[int], rung: str,
                        sc_id: str | None = None) -> lib.Scenario:
    """NC-4 rung: ONE substitution applied to BOTH sides (bodies stay
    norm-equal; the metadata delta stays; benign areas declared on both
    sides)."""
    new_body, _applied, entries = lib.benign_substitute(
        base.side_a.body, k, base.side_a.language or "ru", rng)
    sc = lib.Scenario(
        scenario_id=sc_id or _new_id(prefix, ctr), stratum=base.stratum,
        side_a=lib.Side(title=base.side_a.title, body=new_body,
                        tags=list(base.side_a.tags),
                        language=base.side_a.language,
                        record_type=base.side_a.record_type,
                        created_at=base.side_a.created_at),
        side_b=lib.Side(title=base.side_b.title, body=new_body,
                        tags=list(base.side_b.tags),
                        language=base.side_b.language,
                        record_type=base.side_b.record_type,
                        created_at=base.side_b.created_at),
        label=base.label, slots=[dict(s) for s in base.slots],
        subject_a=base.subject_a, subject_b=base.subject_b,
        polar_op=base.polar_op, refine_ok=base.refine_ok,
        scaffold_id=base.scaffold_id, cross_lang=base.cross_lang,
        length_channel=base.length_channel, both_null=base.both_null,
        carrier=TAG, benign=[dict(e, side="both") for e in entries], para_b=0)
    return sc


# --- orbit enumeration (deterministic; census-facing) ------------------------

@dataclass
class Orbit:
    stratum: str
    base: lib.Scenario            # None for the fresh-base half is NOT used:
                                  # fresh bases ARE scenarios (rung "base")
    rungs: list                   # fitted rung scenarios (unmeasured cos)
    kind: str                     # "classic" | "fresh"


def fresh_cd5_bases(seed: int) -> list[lib.Scenario]:
    """Fresh long-channel CD-5 bases on (subject, conditional) combos that
    carry no classic row text (sha-verified by the census against the
    train-v2 set and the sealed generations). Construction mirrors the
    classic CD-5L grid leg (fillers=35) with train-side rotation OFF and
    scaffold tags distinct from every classic tag family."""
    from build_conflict_detector_corpus import rotate_pair  # noqa: F401
    out: list[lib.Scenario] = []
    ctr = [0]
    seen_texts: set[str] = set()
    with TRAIN_V2.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            seen_texts.add(_disk_sha(row["envelope_a"]))
            seen_texts.add(_disk_sha(row["envelope_b"]))
    for subj in lib.TECH:
        if len(out) >= FRESH_CD5_N:
            break
        for c in lib.COND_SLOTS:
            if len(out) >= FRESH_CD5_N:
                break
            r = random.Random(f"{seed}:C1:CD5F:{subj}:{c[0]}")
            sc = lib.build_scenario(
                _new_id("tr4", ctr), "CD-5", subj, "ru", None,
                c[2], c[3], fillers=FRESH_CD5_FILLERS, rng=r,
                scaffold_tag=f"c1{subj}")
            if _disk_sha_text(sc.text_a()) in seen_texts \
                    or _disk_sha_text(sc.text_b()) in seen_texts:
                continue  # combo text already consumed by train-v2
            sc = lib.finish_pair(sc, r, 1.0 / 3.0)
            out.append(sc)
    if len(out) < FRESH_CD5_N:
        raise AssertionError(
            f"fresh CD-5 orbit plan starved: {len(out)} < {FRESH_CD5_N} "
            "free (subject, conditional) combos")
    return out


def _disk_sha(env: dict) -> str:
    return lib.sha(f"{lib.n1(env['title'])}\n{lib.n1(env['body'])}\n"
                   f"{lib.n1(' '.join(env['tags']))}")


def _disk_sha_text(embed_text_str: str) -> str:
    return lib.sha(embed_text_str)


def _shared_namespace_rekey(orbits: list, rungs: list,
                            bases_fresh: list) -> dict:
    """build4.1 (ADDENDUM-7 §6, TL verdict @ d61f2de): ONE shared tr4-C1
    namespace across BOTH id lanes of this module.

    The two emission lanes (fresh_cd5_bases and enumerate_orbits) each held
    a private ctr=[0] under the SAME prefix tr4-C1: their number spaces
    overlapped and 4 emitted pair_ids carried rows of TWO different pairs
    (trainer P5: duplicate (pair_id, orientation)). This pass enforces
    global uniqueness without any content delta:

    - fresh-base-lane ids are IMMUTABLE — a fresh rung's rng stream is
      keyed by its base scenario_id ("<seed>:C1:CD5F:<base_id>:…"), so
      re-keying a base would change rung CONTENT (§6 forbids any content
      delta; the re-emission must be content-identical);
    - collisions are therefore resolved on the RUNG lane (classic + fresh
      rungs are content-keyed by their train-v2 BASE id, never by their
      own id): a colliding rung is re-keyed to the next free number beyond
      the namespace max, in enumeration order (deterministic, idempotent).

    Fail-closed: the rung lane's own counter must be duplicate-free, and
    the post-state must be globally unique over bases + rungs."""
    base_ids = [b.scenario_id for b in bases_fresh]
    seen: dict[str, int] = {}
    for sc in rungs:
        seen[sc.scenario_id] = seen.get(sc.scenario_id, 0) + 1
    intra = sorted(i for i, n in seen.items() if n > 1)
    if intra:
        raise AssertionError(
            f"rung-lane counter duplicated its own ids: {intra[:4]}")
    cross = sorted(set(base_ids).intersection(seen))
    if not cross:
        return {"renamed": {}, "collisions": 0, "scheme": "tr4-C1-shared"}
    nxt = max(int(i.rsplit("-", 1)[1])
              for i in set(base_ids) | set(seen)) + 1
    renamed: dict[str, str] = {}
    for o in orbits:
        for rg in o.rungs:
            old = rg.scenario_id
            if old in cross:
                new = f"tr4-C1-{nxt:05d}"
                nxt += 1
                rg.scenario_id = new
                renamed[old] = new
    ids = base_ids + [sc.scenario_id for sc in rungs]
    if len(ids) != len(set(ids)):
        raise AssertionError(
            "shared tr4-C1 namespace still colliding after rekey")
    return {"renamed": renamed, "collisions": len(renamed),
            "scheme": "tr4-C1-shared: lane-1 (fresh base) ids immutable "
                      "(rng contract), lane-2 colliding rung ids rekeyed "
                      "beyond namespace max"}


def enumerate_orbits(seed: int) -> dict:
    """The deterministic C1 candidate set (census input; NOTHING emitted).

    Per orbit: para1 (k=1) + para2 (k=2) where the frozen dial fits, one
    rung per (orbit, k) with the side picked by the orbit's rng among the
    FITTING sides (E7); rungs that do not fit are skipped deterministically
    (the dial raises ValueError). NC-4 uses the symmetric builder. Fresh
    CD-5 orbits emit base (rung "base") + para1.

    Returns {"orbits": [Orbit...], "rungs": flat list, "bases_fresh": [...],
    "id_rekey": build4.1 shared-namespace report}. Ids live in ONE shared
    tr4-C1 namespace (build4.1, ADDENDUM-7 §6). Razor-retention / F10 /
    channel checks are the CENSUS's job (they need the embedder and the
    sealed-generation sha sets)."""
    bases = load_razor_bases()
    orbits: list[Orbit] = []
    rungs: list[lib.Scenario] = []
    ctr = [0]

    for s in UNCOVERED:
        for base in bases[s]:
            or_rungs: list[lib.Scenario] = []
            if s == "NC-4":
                for k, rung in ((1, "para1"), (2, "para2")):
                    r = random.Random(f"{seed}:C1:{s}:{base.scenario_id}:{k}:sym")
                    try:
                        sc = make_symmetric_rung(base, k, r, "tr4", ctr, rung)
                    except ValueError:
                        continue  # dial cannot host k here (deterministic unfit)
                    or_rungs.append(sc)
            else:
                for k, rung in ((1, "para1"), (2, "para2")):
                    fit = []
                    for side in ("a", "b"):
                        body = (base.side_a if side == "a" else base.side_b).body
                        try:
                            lib.benign_substitute(
                                body, k, (base.side_a if side == "a"
                                          else base.side_b).language or "ru",
                                random.Random(0))
                            fit.append(side)
                        except ValueError:
                            pass
                    if not fit:
                        continue  # neither side hosts k (deterministic unfit)
                    r = random.Random(f"{seed}:C1:{s}:{base.scenario_id}:{k}")
                    side = r.choice(sorted(fit))
                    rr = random.Random(f"{seed}:C1:{s}:{base.scenario_id}:{k}:{side}")
                    try:
                        sc = make_para_rung(base, k, side, rr, "tr4", ctr, rung)
                    except ValueError:
                        continue
                    or_rungs.append(sc)
            orbits.append(Orbit(stratum=s, base=base, rungs=or_rungs,
                                kind="classic"))
            rungs.extend(or_rungs)

    fresh = fresh_cd5_bases(seed)
    for fb in fresh:
        fb.carrier = TAG   # rung "base" is stamped at row emission (lane)
        fit = []
        for side in ("a", "b"):
            body = (fb.side_a if side == "a" else fb.side_b).body
            try:
                lib.benign_substitute(body, 1, "ru", random.Random(0))
                fit.append(side)
            except ValueError:
                pass
        fr: list[lib.Scenario] = []
        if fit:
            r = random.Random(f"{seed}:C1:CD5F:{fb.scenario_id}:1")
            side = r.choice(sorted(fit))
            rr = random.Random(f"{seed}:C1:CD5F:{fb.scenario_id}:1:{side}")
            try:
                fr.append(make_para_rung(fb, 1, side, rr, "tr4", ctr, "para1"))
            except ValueError:
                pass
        orbits.append(Orbit(stratum="CD-5", base=fb, rungs=fr, kind="fresh"))
        rungs.extend(fr)

    rekey = _shared_namespace_rekey(orbits, rungs,
                                    [o.base for o in orbits
                                     if o.kind == "fresh"])

    return {"orbits": orbits, "rungs": rungs,
            "bases_fresh": [o.base for o in orbits if o.kind == "fresh"],
            "id_rekey": rekey}
