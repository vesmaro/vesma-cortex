#!/usr/bin/env python3
"""gen_pool_v4 EVAL-ONLY fork — C2 bank refresh (ADDENDUM-7 §2 C2, ratified).

G11 preventive cens (the cycle-2 trigger, measured on the four-generation
disjointness): NC-5 stock after the v3 seal is 0 of the 50 needed,
NC-2 stock is 52 < 58, NC-3 classic margin is thin (G6 class). This
module extends the eval universes for exactly those classes, following
the gen_pool_v3 precedent (ADDENDUM-6 §5, banks commit 066eea6):

  NC-5  6 new SPEC subjects x 4 phrases RU+EN (G8 bounds; G9 invariants
        verbatim: RU/EN of one subject KNOWINGLY not translations, benign
        declarative, EN cross legs in the short envelope <= 200);
  NC-2  8 new refinement combos (v |= c through the FROZEN REFINE_RU
        checkers) x 12 tech subjects x fl{0,1} = 192 cells (>= 3x the
        one-distribution consumption 22, G11 sizing);
  NC-3  2 new version-pair families (G6 bound 1-2; 2 forced by the G11
        sizing 3x16 = 48 > 36 cells of one family) on bank-owned base
        sentences, 12 tech subjects x 3 channels = 72 cells.

C3 (NC-4 fate): the 36-cell non-tech fallback axis of banks_v3
(NC4_SUBJECTS_V3, frozen) is PROMOTED to a main bank for the v4 cut —
zero new authorship; the promotion is a CUT RULE (the v4 NC-4 slice
draws old-bank and promoted-axis candidates merged, hardest-first; the
G7 fallback-only rule of v3 is dead). The axis rows keep their v3-module
origin (transfer marking) and every report discloses the promotion.

The banks_v3 module stays BYTE-FROZEN (066eea6 / pin-v3 prep); this fork
IMPORTS its constants read-only (base sentences for NC-3 style parity)
and never mutates it. The train path never imports this module.

Determinism: pure content + explicit ``random.Random`` per-candidate
streams keyed ``"<seed>:<BANK>V4:..."``. Regenerate == reproduce.

FREEZE (G1): after the freeze commit this file is EDIT-LOCKED. A census
or gate failure is a BuildStop + a NEW addendum, never a bank iteration.
"""

from __future__ import annotations

import contextlib
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "scripts"),):
    if _p not in sys.path:
        sys.path.append(_p)

import conflict_corpus_lib as lib  # noqa: E402
import conflict_corpus_banks_v3 as v3  # noqa: E402 (read-only constants)

NULL_P = 1.0 / 3.0
BANK_TAG = "v4"


# ===========================================================================
# BANK CONTENT (the frozen artifact; sha256 of bank_content() is pinned)
# ===========================================================================

#: NC-5 bank — 6 new SPEC subjects (G8 bounds 4-6; 24 phrases RU+EN).
SUBJECTS_V4 = {
    "herbs":     ("Травы на подоконнике", "Window-sill herbs",
                  "трав на подоконнике", "травам на подоконнике"),
    "candles":   ("Свечи ручной работы", "Handmade candles",
                  "свечей ручной работы", "свечам ручной работы"),
    "sketch":    ("Скетчбук для зарисовок", "Sketchbook for studies",
                  "скетчбука для зарисовок", "скетчбуку для зарисовок"),
    "aquarium":  ("Домашний аквариум", "Home aquarium",
                  "домашнего аквариума", "домашнему аквариуму"),
    "terrace":   ("Цветы на террасе", "Terrace flowers",
                  "цветов на террасе", "цветам на террасе"),
    "sewing":    ("Швейная шкатулка", "Sewing box",
                  "швейной шкатулки", "швейной шкатулке"),
}

#: SPEC_V4[subj] = (ru0, ru1, en0, en1): RU/EN of one subject are KNOWINGLY
#: not translations (translation pairs stay BASE-only); benign declarative
#: (no negation, no polarity, no versions, no digits/number words —
#: slots=[] stays truthful); EN first sentence is the cross leg and must
#: stay in the short envelope (<= 200 chars).
SPEC_V4 = {
    "herbs":    ("Зелень на подоконнике клонится к стеклу всей массой.",
                 "Срезка для кухни заготавливается по мере надобности.",
                 "Window herbs lean toward the glass as a whole.",
                 "The little scissors rest in the jar beside the basil."),
    "candles":  ("Фитили подрезаются до длины в сантиметр перед каждым зажиганием.",
                 "Стеклянные баночки служат формами для свежих заливок.",
                 "Wicks get trimmed to a finger width before each lighting.",
                 "Glass jars serve as molds for fresh pours."),
    "sketch":   ("Зарисовки с натуры заполняют разворот за вечер.",
                 "Мягкие карандаши живут в кармашке обложки.",
                 "Life studies fill a spread within an evening.",
                 "A kneaded eraser sits in the side tuck."),
    "aquarium": ("Рыбки получают корм строго по расписанию.",
                 "Пещера из камушков служит убежищем мелким рыбкам.",
                 "Algae wipe off the front glass every few days.",
                 "A dim lamp on a timer mimics the sunset."),
    "terrace":  ("Горшки с пеларгонией выносятся на террасу после заморозков.",
                 "Полив на террасе проходит рано утром.",
                 "Geranium pots move onto the terrace after the last frosts.",
                 "Terrace watering happens early in the morning."),
    "sewing":   ("Ножницы для рукоделия хранятся отдельно от хозяйственных.",
                 "Лоскуты сортируются по цветам в жестяной коробке.",
                 "Craft scissors live apart from the household pair.",
                 "Fabric scraps sort by color into the tin box."),
}

#: NC-2 refresh combos — (axis, constraint c, value v) with v |= c through
#: the FROZEN lib.REFINE_RU checkers; every v and every numeric c is FRESH
#: (side-level sha freshness vs train-v2 / the sealed generations: the
#: value sentence and the constraint sentence must be new strings).
NC2_COMBOS_V4 = [
    ("lim", 3000, 1500), ("lim", 2000, 1200), ("lim", 1500, 900),
    ("timo", 300, 180), ("timo", 240, 150),
    ("upd", "октябрь", "1 октября"), ("upd", "октябрь", "20 октября"),
    ("upd", "октябрь", "31 октября"),
]

#: NC-3 refresh — 2 new version-pair families (G6 bound; 2 forced by the
#: G11 sizing rule 3x16=48 > 36 cells of one family — anti-collapse share
#: arithmetic: 2 families = 48 scaffolds, max 2/72 = 2.8% <= 5%).
VERSION_PAIRS_V4 = [("v3.1.0", "v3.2.0"), ("v4.1.0", "v4.2.0")]

#: NC-3 bank-owned base block (fresh wording, disjoint from BASE_V3B_RU by
#: construction-checked substring rules; version-keyword-free).
BASE_V4B_RU = [
    "Компонент значится в журнале учета команды платформы.",
    "Состояние компонента сверяется с каталогом по субботам.",
]

#: C3 promotion — the banks_v3 NC-4 non-tech axis (36 cells, frozen
#: module) joins the MAIN bank for the v4 cut. Documentation + the lane
#: reads the axis from the frozen banks_v3 generator (zero authorship).
NC4_PROMOTION_V4 = {
    "ratification": "ADDENDUM-7 §2 C3",
    "source_module": "scripts/conflict_corpus_banks_v3.py",
    "source_fn": "gen_nc4_v3",
    "cells": 36,
    "rule": "v4 NC-4 slice draws old-bank + promoted-axis merged, "
            "hardest-first; the v3 G7 fallback-only rule is dead; "
            "axis rows keep transfer marking (v3-module origin)",
}

BANK_META = {
    "ratification": "ADDENDUM-7 §2 C2 (G11 refresh, G8/G9-class invariants, "
                    "G6-class NC-3 micro-extension) + §2 C3 (NC-4 promotion "
                    "record, zero authorship)",
    "eval_only": True,
    "nc5": {"subjects": len(SUBJECTS_V4), "phrases": 4 * len(SUBJECTS_V4),
            "grid": "new<->old ordered pairs (cross always; same-lang and "
                    "long only cross-domain) + new<->new cross legs"},
    "nc2": {"combos": len(NC2_COMBOS_V4),
            "grid": "8 combos x 12 tech subjects x fl{0,1} = 192 cells",
            "checkers": "frozen lib.REFINE_RU (v |= c mechanical truth)"},
    "nc3": {"families": len(VERSION_PAIRS_V4),
            "grid": "2 families x 12 tech subjects x {fl0, fl1, long} = 72",
            "note": "G6 upper bound; 1 family breaches the G11 sizing 48"},
    "nc4": NC4_PROMOTION_V4,
}


# ===========================================================================
# STATIC GUARDS (fail-closed at import; PARA/banks_v3 precedent)
# ===========================================================================

_NEG_MORPHEMS_RU = ("не", "никогда", "только", "вместо")
_NEG_WORDS_EN = ("not", "never", "only", "instead")
_NUMBER_WORDS = ("один", "одна", "одно", "одним", "одного", "две", "двух", "два",
                 "три", "трёх", "четыре", "четырёх", "пять", "пяти", "шесть",
                 "шести", "семь", "семи", "восемь", "восьми", "девять", "девяти",
                 "десять", "десяти", "сто", "первый", "первая", "первой",
                 "второй", "вторая", "третий", "третья",
                 "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "hundred", "first", "second", "third")
_POLARITY_STEMS = ("падает", "падение", "замедл", "ускор", "прирост", "сократ",
                   "увелич", "сниз", "медленне", "быстрее", "хуже", "лучше",
                   "fails", "crash", "slow", "speedup", "faster", "slower",
                   "reduced", "increased", "worse", "better", "regression")

#: every fact/value string the classic universe has consumed (NC2_GRID,
#: NUM_SLOTS, DATE_SLOT values, REFINE defaults) — NC-2 refresh values must
#: be OUTSIDE this multiset for side-level sha freshness.
_CLASSIC_VALUES = frozenset(
    [str(x) for x in (800, 500, 200, 1000, 60, 90, 120, 30, 100, 14)]
    + ["5 октября", "7 октября", "15 октября", "октябрь"])


def _norm(t: str) -> str:
    return " ".join(t.split()).lower()


def _guard_spec_phrase(subj: str, lang: str, idx: int, ph: str) -> list[str]:
    errs = []
    nph = _norm(ph)
    if any(ch.isdigit() for ch in ph):
        errs.append(f"{subj}/{lang}{idx} digit present")
    for m in _NEG_MORPHEMS_RU:
        if m in nph:
            errs.append(f"{subj}/{lang}{idx} negation morpheme {m!r}")
    for w in _NEG_WORDS_EN:
        if re.search(rf"\b{w}\b", nph):
            errs.append(f"{subj}/{lang}{idx} EN negation word {w!r}")
    for w in _NUMBER_WORDS:
        if re.search(rf"\b{w}\b", nph):
            errs.append(f"{subj}/{lang}{idx} number word {w!r}")
    for s in _POLARITY_STEMS:
        if s in nph:
            errs.append(f"{subj}/{lang}{idx} polarity stem {s!r}")
    if lib.VERSION_RE.search(ph):
        errs.append(f"{subj}/{lang}{idx} version-like token")
    if lang == "en" and idx == 0 and len(ph) > 200:
        errs.append(f"{subj}/en0 crosses the short envelope ({len(ph)} > 200)")
    return errs


def _guard_base_sentence(idx: int, ph: str) -> list[str]:
    """Bank-owned base sentences follow the corpus house style («Компонент
    ...»), so the negation check here is TOKEN-level for «не» (the
    substring form would over-fire inside «компонент» — the classic
    BASE_RU carries the same word); никогда/только/вместо stay
    substring-checked."""
    errs = []
    nph = _norm(ph)
    if any(ch.isdigit() for ch in ph):
        errs.append(f"base{idx} digit present")
    tokens = nph.split(" ")
    if any(t.strip(".,:;—–-()\"'") == "не" for t in tokens):
        errs.append(f"base{idx} negation token «не»")
    for m in ("никогда", "только", "вместо"):
        if m in nph:
            errs.append(f"base{idx} negation morpheme {m!r}")
    if lib.VERSION_RE.search(ph):
        errs.append(f"base{idx} version-like token")
    hosts = list(lib.BASE_RU) + list(v3.BASE_V3_RU) + list(v3.BASE_V3B_RU)
    for h in hosts:
        nh = _norm(h)
        if nph in nh or nh in nph:
            errs.append(f"base{idx} substring collision with bank-owned bases")
    return errs


_STATIC_ERRORS: list[str] = []


def _run_static_guards() -> None:
    global _STATIC_ERRORS
    errs: list[str] = []
    if not 4 <= len(SUBJECTS_V4) <= 6:
        errs.append(f"NC-5 subject count {len(SUBJECTS_V4)} outside 4-6 (G8)")
    if not 16 <= 4 * len(SUBJECTS_V4) <= 24:
        errs.append("NC-5 phrase count outside 16-24 (G8)")
    if not 1 <= len(VERSION_PAIRS_V4) <= 2:
        errs.append(f"NC-3 family count {len(VERSION_PAIRS_V4)} outside 1-2 (G6)")
    if len(NC2_COMBOS_V4) * 12 * 2 < 3 * 22:
        errs.append("NC-2 refresh sizing below 3x one-distribution (G11)")
    known_versions = {v for pair in (("v1.2.0", "v1.3.0"), ("v1.3.0", "v2.0.0"),
                                     ("v1.2.0", "v2.0.0"),
                                     ("v0.9.0", "v1.0.0"), ("v2.1.0", "v2.2.0"))
                      for v in pair}
    seen_v: set[str] = set()
    for axis, c, val in NC2_COMBOS_V4:
        checker = lib.REFINE_RU[axis][2]
        if not checker(val, c):
            errs.append(f"NC-2 combo ({axis},{c},{val}) fails the frozen "
                        "v |= c checker")
        sv = str(val)
        if sv in _CLASSIC_VALUES:
            errs.append(f"NC-2 value {sv!r} reuses a consumed classic value "
                        "(side-level sha freshness)")
        if sv in seen_v:
            errs.append(f"NC-2 value {sv!r} duplicates another combo value")
        seen_v.add(sv)
        if axis in ("lim", "timo") and str(c) in _CLASSIC_VALUES:
            errs.append(f"NC-2 constraint {c!r} reuses a consumed classic "
                        "constraint value")
    for va, vb in VERSION_PAIRS_V4:
        for v in (va, vb):
            if not lib.VERSION_RE.fullmatch(v):
                errs.append(f"version token {v!r} outside the guard alphabet")
            if v in known_versions:
                errs.append(f"version token {v!r} reuses a consumed family")
        if va == vb:
            errs.append(f"degenerate version pair ({va})")
    for i, ph in enumerate(BASE_V4B_RU):
        errs.extend(_guard_base_sentence(i, ph))
    old_specs = set(lib.SPEC) | set(v3.SUBJECTS_V3)
    for subj, (ru0, ru1, en0, en1) in SPEC_V4.items():
        if subj in old_specs or subj in lib.SUBJECTS:
            errs.append(f"subject {subj} collides with a classical/v3 bank")
        for j, ph in ((0, ru0), (1, ru1)):
            errs.extend(_guard_spec_phrase(subj, "ru", j, ph))
        for j, ph in ((2, en0), (3, en1)):
            errs.extend(_guard_spec_phrase(subj, "en", j, ph))
    for subj, rows in SPEC_V4.items():
        for ph in rows:
            nph = _norm(ph)
            for os_, orow in list(lib.SPEC.items()) + list(v3.SPEC_V3.items()):
                for oph in orow:
                    if nph == _norm(oph):
                        errs.append(f"{subj} sentence duplicates {os_} SPEC")
    _STATIC_ERRORS = errs
    if errs:
        raise AssertionError(
            "BANKS-V4 STATIC GUARDS RED:\n  " + "\n  ".join(errs))


_run_static_guards()


def bank_content() -> dict:
    """Canonical content snapshot — sha256(json_dumps_sorted(...)) is the
    bank identity pinned at freeze (banks_v3 precedent)."""
    return {
        "subjects_v4": {k: list(v) for k, v in SUBJECTS_V4.items()},
        "spec_v4": {k: list(v) for k, v in SPEC_V4.items()},
        "nc2_combos_v4": [list(c) for c in NC2_COMBOS_V4],
        "version_pairs_v4": [list(p) for p in VERSION_PAIRS_V4],
        "base_v4b_ru": list(BASE_V4B_RU),
        "nc4_promotion_v4": dict(NC4_PROMOTION_V4),
        "meta": BANK_META,
    }


def static_guard_errors() -> list[str]:
    return list(_STATIC_ERRORS)


# ===========================================================================
# BODY COMPOSITION (channel arithmetic mirrors banks_v3)
# ===========================================================================

def _nc3_body(fam_idx: int, subj_idx: int, channel: str) -> str:
    if channel == "fl0":
        return BASE_V4B_RU[fam_idx % 2]
    if channel == "fl1":
        return " ".join([BASE_V4B_RU[0], BASE_V4B_RU[1],
                         lib.FILLERS_RU[(fam_idx * 3 + subj_idx) % len(lib.FILLERS_RU)]])
    fill = lib.FILLERS_RU
    tail = " ".join(fill[(fam_idx * 4 + i) % len(fill)] for i in range(35))
    return " ".join([BASE_V4B_RU[0], BASE_V4B_RU[1], tail])


#: NC-5 composition dial for the NEW side (6 variants — mirrors the banks_v3
#: dial: cells sharing a subject must not collapse into one side-sha).
def _nc5_comp(subj: str, idx: int) -> str:
    ru0, ru1, _, _ = SPEC_V4[subj]
    if idx == 0:
        return ru0
    if idx == 1:
        return ru1
    if idx == 2:
        return f"{ru0} {ru1}"
    if idx == 3:
        return f"{ru0} {lib.FILLERS_RU[3]}"
    if idx == 4:
        return f"{ru1} {lib.FILLERS_RU[6]}"
    return f"{ru0} {ru1} {lib.FILLERS_RU[1]}"


def _nc5_old_plain(subj: str) -> str:
    return f"{lib.SPEC[subj][0]} {lib.SPEC[subj][1]}"


# ===========================================================================
# BUILDERS (declaration dicts VERBATIM from the frozen constructors)
# ===========================================================================

def _finish_rotate(sc: lib.Scenario, seed: int, rotate: bool,
                   r: random.Random, force_nn: bool = False) -> lib.Scenario:
    from build_conflict_detector_corpus import rotate_pair
    if rotate:
        sc.side_a.body, sc.side_b.body = rotate_pair(
            sc.side_a.body, sc.side_b.body, seed)
    return lib.finish_pair(sc, r, NULL_P, force_nn)


def build_nc2_v4(sid: str, subj: str, axis: str, c, v,
                 fl: int) -> lib.Scenario:
    """Classic NC-2 branch, verbatim declaration; fresh (c, v)."""
    c_sent, v_sent, _check = lib.REFINE_RU[axis]
    subj_fmt = {"gen": lib.SUBJECTS[subj][2], "dat": lib.SUBJECTS[subj][3]}
    body = lib.base_body(subj, "ru", fl)
    body_a = body + " " + c_sent.format(c=c, **subj_fmt)
    body_b = body + " " + v_sent.format(v=v, **subj_fmt)
    side_a = lib.make_side("ru", subj, body_a)
    side_b = lib.make_side("ru", subj, body_b)
    return lib.Scenario(
        scenario_id=sid, stratum="NC-2", side_a=side_a, side_b=side_b,
        label="not-contradicts",
        slots=[{"slot_id": axis, "axis": axis, "value_a": str(c),
                "value_b": str(v), "op": "replace", "is_version_token": False,
                "span_chars": max(len(str(v)), 6)}],
        subject_a=subj, subject_b=subj, polar_op=None,
        refine_ok=bool(lib.REFINE_RU[axis][2](v, c)),
        scaffold_id=f"NC-2:{subj}:{axis}:v4c{c}", cross_lang=False,
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


def build_nc3_v4(sid: str, subj: str, vp: tuple[str, str],
                 body: str, fam_idx: int) -> lib.Scenario:
    gen = lib.SUBJECTS[subj][2]
    side_a = lib.make_side("ru", subj, body + " " + lib.VERSION_RU.format(gen=gen, v=vp[0]))
    side_b = lib.make_side("ru", subj, body + " " + lib.VERSION_RU.format(gen=gen, v=vp[1]))
    return lib.Scenario(
        scenario_id=sid, stratum="NC-3", side_a=side_a, side_b=side_b,
        label="not-contradicts",
        slots=[{"slot_id": "ver", "axis": "version", "value_a": vp[0],
                "value_b": vp[1], "op": "replace", "is_version_token": True,
                "span_chars": max(len(vp[0]), len(vp[1]))}],
        subject_a=subj, subject_b=subj, polar_op=None, refine_ok=None,
        scaffold_id=f"NC-3:{subj}:{vp[0]}:v4f{fam_idx}", cross_lang=False,
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


def build_nc5_v4(sid: str, sa: str, sb: str, kind: str,
                 comp_a: int, comp_b: int) -> lib.Scenario:
    """kind: 'same' | 'cross' | 'long' — mirrors banks_v3.build_nc5_v3."""
    blk = lib.filler_block("ru", 35)
    kind = "cross" if kind == "cross-rev" else kind
    if sa in SPEC_V4:
        body_a = _nc5_comp(sa, comp_a) if kind != "long" \
            else _nc5_comp(sa, 2) + " " + blk
        lang_a = "ru"
    else:
        body_a = _nc5_old_plain(sa) if kind != "long" \
            else _nc5_old_plain(sa) + " " + blk
        lang_a = "ru"
    if sb in SPEC_V4:
        body_b = _nc5_comp(sb, comp_b) if kind == "same" \
            else (SPEC_V4[sb][2] if kind == "cross" else _nc5_comp(sb, 2) + " " + blk)
        lang_b = "ru" if kind == "same" else "en"
    else:
        body_b = _nc5_old_plain(sb) if kind == "same" \
            else (lib.SPEC[sb][2] if kind == "cross" else _nc5_old_plain(sb) + " " + blk)
        lang_b = "ru" if kind == "same" else "en"
    side_a = lib.make_side(lang_a, sa, body_a)
    side_b = lib.make_side(lang_b, sb, body_b)
    return lib.Scenario(
        scenario_id=sid, stratum="NC-5", side_a=side_a, side_b=side_b,
        label="not-contradicts", slots=[],
        subject_a=sa, subject_b=sb, polar_op=None, refine_ok=None,
        scaffold_id=f"nc5:{sa}:{sb}:{BANK_TAG}",
        cross_lang=(lang_a != lang_b),
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


# ===========================================================================
# SCOPED SUBJECT INJECTION (reversible; integrity-asserted; banks_v3 style)
# ===========================================================================

@contextlib.contextmanager
def v4_bank_context():
    had_subj = set(SUBJECTS_V4) <= set(lib.SUBJECTS)
    had_spec = set(SPEC_V4) <= set(lib.SPEC)
    snap_subj = dict(lib.SUBJECTS)
    snap_spec = dict(lib.SPEC)
    if not had_subj:
        lib.SUBJECTS.update(SUBJECTS_V4)
    if not had_spec:
        lib.SPEC.update(SPEC_V4)
    try:
        yield
    finally:
        if not had_subj:
            lib.SUBJECTS.clear()
            lib.SUBJECTS.update(snap_subj)
        if not had_spec:
            lib.SPEC.clear()
            lib.SPEC.update(snap_spec)
        assert dict(lib.SUBJECTS) == snap_subj and dict(lib.SPEC) == snap_spec, \
            "v4_bank_context: lib state failed to restore byte-equal"


# ===========================================================================
# GENERATORS (deterministic per-candidate streams; rotate EXPLICIT)
# ===========================================================================

def gen_nc2_v4(seed: int, prefix: str, rotate: bool = True) -> list:
    out: list = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-2-{ctr[0]:05d}"

    for subj in lib.TECH:
        for axis, c, v in NC2_COMBOS_V4:
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC2V4:{subj}:{axis}:{c}:{fl}")
                sc = build_nc2_v4(sid(), subj, axis, c, v, fl)
                out.append(_finish_rotate(sc, seed, rotate, r))
    return out


def gen_nc3_v4(seed: int, prefix: str, rotate: bool = True) -> list:
    out: list = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-3-{ctr[0]:05d}"

    for si, subj in enumerate(lib.TECH):
        for fi, vp in enumerate(VERSION_PAIRS_V4):
            for channel in ("fl0", "fl1", "long"):
                r = random.Random(f"{seed}:NC3V4:{subj}:{fi}:{channel}")
                sc = build_nc3_v4(sid(), subj, vp,
                                  _nc3_body(fi, si, channel), fi)
                out.append(_finish_rotate(sc, seed, rotate, r))
    return out


def gen_nc5_v4(seed: int, prefix: str, rotate: bool = True) -> list:
    out: list = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-5-{ctr[0]:05d}"

    new = list(SUBJECTS_V4)
    old = [s for s in lib.SUBJECTS if s not in SUBJECTS_V4]
    tech = set(lib.TECH)

    def comp_for(a: str, b: str) -> int:
        sibs = sorted(set(new + old) - {a})
        return sibs.index(b) % 6

    def add(sa: str, sb: str, kind: str) -> None:
        r = random.Random(f"{seed}:NC5V4:{sa}:{sb}:{kind}")
        sc = build_nc5_v4(sid(), sa, sb, kind, comp_for(sa, sb), comp_for(sb, sa))
        out.append(_finish_rotate(sc, seed, rotate, r))

    for sa in new:
        for sb in old:
            add(sa, sb, "cross")
            if (sa in tech) != (sb in tech):
                add(sa, sb, "same")
                add(sa, sb, "long")
        for sb in new:
            if sb != sa:
                add(sa, sb, "cross")
    for sa in old:
        for sb in new:
            add(sa, sb, "cross")
            if (sa in tech) != (sb in tech):
                add(sa, sb, "same")
                add(sa, sb, "long")
    return out


def gen_banks_v4_pool(seed: int, prefix: str, rotate: bool = True) -> list:
    """The C2 refresh candidate pool (NC-2/NC-3/NC-5; NC-4 rides the frozen
    banks_v3 axis per C3)."""
    if prefix in {"tr", "ho"}:
        raise ValueError("v4 bank pools must use a lane-distinct prefix")
    with v4_bank_context():
        return (gen_nc2_v4(seed, prefix, rotate)
                + gen_nc3_v4(seed, prefix, rotate)
                + gen_nc5_v4(seed, prefix, rotate))


if __name__ == "__main__":
    with v4_bank_context():
        n2 = gen_nc2_v4(7047, "smoke", rotate=True)
        n3 = gen_nc3_v4(7047, "smoke", rotate=True)
        n5 = gen_nc5_v4(7047, "smoke", rotate=True)
    mech = sum(1 for x in n2 + n3 + n5 if lib.mechanical_ok(x))
    print(f"smoke: nc2={len(n2)} nc3={len(n3)} nc5={len(n5)} "
          f"total={len(n2)+len(n3)+len(n5)} mech_ok={mech}")
    print("lib dict integrity: SUBJECTS", len(lib.SUBJECTS), "SPEC", len(lib.SPEC))
