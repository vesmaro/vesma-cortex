#!/usr/bin/env python3
"""gen_pool_v3 EVAL-ONLY fork — new template banks (ADDENDUM-6 §5, G6-G9).

Ratified lane: B6-conflict bank authorship (slice
``conflict-detector-corpus-v2-seal3-banks``). This module is BANK CONTENT
ONLY: the new NC-1 insert bank, the new NC-5 SPEC subjects, the NC-3
version-pair micro-extension (G6) and the NC-4 non-tech fallback axis
(G7, fallback margin). It is imported ONLY by eval-side v3 lanes
(preflight / census / cut). The train path (measure2/build2, seeds
2113/7919/9377) never imports it. The new NC-5 subjects resolve through
a SCOPED, REVERSIBLE injection into ``lib.SUBJECTS``/``lib.SPEC``
(``v3_bank_context``: snapshot -> inject -> restore, with a byte-equality
integrity assert after restore; nested use is idempotent). Outside the
context the lib dicts are exactly their classical state — the train path
can never observe the new subjects. Builders otherwise mirror the
``build_n3`` precedent of bespoke constructors with VERBATIM declaration
dicts — no lib edits, no reimplementation of predicates.

Why the banks own their base blocks: the sealed generations consumed the
classical text universe by text-sha (F6/F10 culling is per-side
embed-text sha). A new insert phrase alone changes only the b-side; the
unchanged a-side would hit ``train-v2``/``v1-seal``/``v2-seal`` shas and
cull every cell again (the exact BuildStop mechanism of 581883c-era
runs: NC-1 clean 1/336). Therefore NC-1/NC-3 cells compose the a/b bodies
from NEW bank-owned base sentences (plus the shared corpus filler
inventory for channel depth), so BOTH sides carry fresh text shas. The
G9 declaration invariants are unchanged and statically guarded at import.

Determinism: pure content + explicit ``random.Random`` per-candidate
streams keyed ``"<seed>:<BANK>V3:..."`` (no collision with gen_pool
streams). Regenerate == reproduce, byte-identical.

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

NULL_P = 1.0 / 3.0
BANK_TAG = "v3"


# ===========================================================================
# BANK CONTENT (the frozen artifact; sha256 of _bank_content() goes into
# pin-v3.json prep section at freeze)
# ===========================================================================

#: NC-1 bank — 12 new superset-addition phrases (G8: 10-14; G9 invariants:
#: negation-morpheme-free («не/никогда/только/вместо» as substrings,
#: case-insensitive), digit-free, version-token-free, single sentence with
#: terminal period and no internal splitters, not a substring of any
#: BASE/FILLERS sentence either direction — all statically guarded below).
INSERT_PHRASES_V3 = [
    "Дополнительно формируется сводный отчёт по обращениям за день.",
    "Также собирается статистика времени ответа по каждому сценарию.",
    "Отдельным модулем ведётся синхронизация справочников с соседними сервисами.",
    "Дополнительно поддерживается ручной запуск задач из интерфейса команды.",
    "Также включено резервное копирование настроек в общее хранилище.",
    "Отдельно настраивается рассылка сводок для дежурных по расписанию.",
    "Поддерживается отображение статуса задач на общем мониторе.",
    "Дополнительно включён приём событий от смежных сервисов через шину.",
    "Также ведётся учёт обращений пользователей в базе знаний команды.",
    "Отдельным потоком выполняется предагрегация метрик для дашбордов.",
    "Поддерживается выгрузка промежуточных результатов в рабочую область.",
    "Дополнительно настраивается разметка задач по проектам.",
]

#: NC-1 combo table — 13 combos mirroring the classical shape (G8 "~12",
#: shape parity with the old bank: 3 singles / 3 pairs / 3 triples /
#: 2 quads / 2 quints; quads are the razor-source family per design 4.3).
COMBOS_V3 = [
    (0,), (4,), (8,),
    (1, 5), (3, 9), (6, 10),
    (0, 4, 8), (2, 6, 10), (3, 5, 11),
    (1, 4, 7, 10), (2, 5, 8, 11),
    (0, 3, 6, 9, 11), (1, 4, 7, 9, 10),
]

#: NC-1 bank-owned base block (new a-side texts; see module docstring).
#: Generic declarative component sentences, digit-free, version-token-free,
#: addition-compatible with every insert phrase.
BASE_V3_RU = [
    "Компонент обслуживается командой платформы и состоит в общем реестре сервисов.",
    "Рабочие показатели компонента собираются в общую панель наблюдения.",
    "Настройки компонента читаются из репозитория команды при каждом запуске.",
]

#: NC-3 bank-owned base block (disjoint from BASE_V3_RU by wording).
BASE_V3B_RU = [
    "Компонент опубликован во внутреннем каталоге сервисов команды.",
    "Дежурная смена получает сводку о состоянии компонента ежедневно.",
]

#: NC-3 micro-extension (G6): 2 version-pair families — the UPPER bound of
#: the ratified "1-2 семейства (+36-72 ячейки)". 2 (not 1) is forced by the
#: anti-collapse invariant: 1 family = 24 scaffolds but max share
#: 2/36 = 5.6% > 5%; 2 families = 48 scaffolds, max 2/72 = 2.8%.
VERSION_PAIRS_V3 = [("v0.9.0", "v1.0.0"), ("v2.1.0", "v2.2.0")]

#: NC-5 bank — 6 new SPEC subjects (G8 upper bound 4-6; 24 phrases RU+EN =
#: ratified upper bound 16-24). Six (not four) is forced by the census
#: arithmetic: NC-5 clean supply under F10 sha-culling + within-cut
#: side-text uniqueness is bounded by the NEW text inventory (new-only
#: legs are the cross legs between new subjects); 4 subjects top out below
#: the 2x census quota (50), 6 subjects clear it with margin. Disclosure
#: goes to the prep report (K4-class arithmetic correction, authoring-time).
SUBJECTS_V3 = {
    "garden":     ("Рассада на балконе", "Balcony seedling setup",
                   "рассады на балконе", "рассаде на балконе"),
    "tools":      ("Набор для мелкого ремонта", "Small repair kit",
                   "набора для мелкого ремонта", "набору для мелкого ремонта"),
    "photos":     ("Домашний фотоархив", "Home photo archive",
                   "домашнего фотоархива", "домашнему фотоархиву"),
    "tea":        ("Чайная полка на кухне", "Kitchen tea shelf",
                   "чайной полки на кухне", "чайной полке на кухне"),
    "bikes":      ("Обслуживание городского велосипеда", "City bike maintenance",
                   "городского велосипеда", "городскому велосипеду"),
    "boardgames": ("Полка настольных игр", "Board game shelf",
                   "настольных игр", "настольным играм"),
}

#: SPEC_V3[subj] = (ru0, ru1, en0, en1): RU/EN of one subject are KNOWINGLY
#: not translations (translation pairs are BASE-only — the cross cos
#: envelope ~0.486 must not be left; measured preflight gate [0.46, 0.60]).
#: All phrases benign-declarative: no negation, no polarity, no versions,
#: no digits/number words — slots=[] stays truthful. EN first sentence is
#: the cross-leg body and must stay in the short envelope (<= 200 chars).
SPEC_V3 = {
    "garden": ("Ящик с рассадой поворачивают к солнцу каждое утро.",
               "Полив проводят утром, чтобы грунт просох до вечера.",
               "Seedlings move to the glazed balcony after the last frosts.",
               "A shallow tray under the pots catches the extra water."),
    "tools":  ("Ящик с инструментами хранится в кладовке у стены.",
               "Мелкий ремонт начинают с поиска подходящей отвёртки.",
               "Screwdrivers sort by size into the folded cloth roll.",
               "A spare set of bits lives in the magnetic tray."),
    "photos": ("Снимки сортируются по папкам с названиями городов.",
               "Старые плёнки оцифровывают зимними вечерами.",
               "Duplicates land in a separate folder for later review.",
               "Favourite shots go into the printed family album."),
    "tea":    ("Заварка хранится в жестяных банках с плотными крышками.",
               "Вечерняя заварка оставляется на подносе до утра.",
               "The teapot warms up before the leaves go in.",
               "A small hourglass measures the steeping interval."),
    "bikes":  ("Цепь протирается после каждой поездки в сырую погоду.",
               "Седло настраивается по высоте без инструментов.",
               "The chain gets wiped after every wet ride.",
               "Tyre pressure is checked by squeezing the sidewall."),
    "boardgames": ("Коробки с играми стоят на стеллаже в гостиной.",
                   "Правила новой игры читают вслух перед партией.",
                   "Small pieces stay in labeled bags inside the box.",
                   "The scoreboard pencils live in the side drawer."),
}

#: NC-4 fallback axis (G7): the OLD bank stays the cut source (strict cens
#: 12 >= 11); these 6 OLD non-tech subjects extend the subject axis as
#: FALLBACK MARGIN ONLY — marked fallback here and in every report.
NC4_SUBJECTS_V3 = ["cook", "fit", "weather", "music", "travel", "books"]
NC4_DELTAS_V3 = ["tags", "lang", "type"]   # mirror of the classical deltas

BANK_META = {
    "ratification": "ADDENDUM-6 §5 (G1 freeze, G6 NC-3 micro, G7 NC-4 fallback, "
                    "G8 authorship plan, G9 invariants verbatim, G10 census on "
                    "clean-after-preflight-attrition)",
    "eval_only": True,
    "nc1": {"phrases": len(INSERT_PHRASES_V3), "combos": len(COMBOS_V3),
            "grid": "13 combos x 12 tech subjects x fl{0,1} + 2 long = 336 cells",
            "channels": "short (fl0), medium (fl1), long (2/subject)"},
    "nc5": {"subjects": len(SUBJECTS_V3),
            "phrases": 4 * len(SUBJECTS_V3),
            "grid": "new<->old ordered pairs (cross always; same-lang and long "
                    "only cross-domain) + new<->new cross legs = 534 cells"},
    "nc3": {"families": len(VERSION_PAIRS_V3),
            "grid": "2 families x 12 tech subjects x {fl0, fl1, long} = 72 cells",
            "note": "G6 upper bound; 1 family breaches the 5%/scaffold cap"},
    "nc4": {"role": "FALLBACK MARGIN ONLY (old bank stays the cut source, G7)",
            "subjects": len(NC4_SUBJECTS_V3),
            "grid": "6 old non-tech subjects x 3 deltas x {fl1, long} = 36 cells"},
}


# ===========================================================================
# STATIC GUARDS (fail-closed at import; PARA-dictionary precedent)
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
_SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")


def _norm(t: str) -> str:
    return " ".join(t.split()).lower()


def _guard_insert_phrase(idx: int, ph: str) -> list[str]:
    errs = []
    nph = _norm(ph)
    if any(ch.isdigit() for ch in ph):
        errs.append(f"[{idx}] digit present")
    for m in _NEG_MORPHEMS_RU:
        if m in nph:
            errs.append(f"[{idx}] negation morpheme {m!r}")
    if lib.VERSION_RE.search(ph):
        errs.append(f"[{idx}] version-like token")
    sents = [s for s in _SENT_SPLIT_RE.split(ph.strip()) if s]
    if len(sents) != 1 or not ph.endswith(".") or sents[0] != ph:
        errs.append(f"[{idx}] not a single terminal-period sentence")
    if any(c in ph for c in ";?!"):
        errs.append(f"[{idx}] internal splitter present")
    hosts = list(lib.BASE_RU) + list(lib.BASE_EN) + list(lib.FILLERS_RU) \
        + list(lib.FILLERS_EN) + list(BASE_V3_RU) + list(BASE_V3B_RU)
    for h in hosts:
        nh = _norm(h)
        if nph in nh or nh in nph:
            errs.append(f"[{idx}] substring collision with base/filler corpus")
    return errs


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


_STATIC_ERRORS: list[str] = []


def _run_static_guards() -> None:
    global _STATIC_ERRORS
    errs: list[str] = []
    if not 10 <= len(INSERT_PHRASES_V3) <= 14:
        errs.append(f"NC-1 phrase count {len(INSERT_PHRASES_V3)} outside 10-14 (G8)")
    if not 4 <= len(SUBJECTS_V3) <= 6:
        errs.append(f"NC-5 subject count {len(SUBJECTS_V3)} outside 4-6 (G8)")
    if not 16 <= 4 * len(SUBJECTS_V3) <= 24:
        errs.append("NC-5 phrase count outside 16-24 (G8)")
    if not 1 <= len(VERSION_PAIRS_V3) <= 2:
        errs.append(f"NC-3 family count {len(VERSION_PAIRS_V3)} outside 1-2 (G6)")
    arities = sorted(len(c) for c in COMBOS_V3)
    if sum(1 for a in arities if a == 4) < 2 or sum(1 for a in arities if a == 5) < 2:
        errs.append("combo table lacks quads/quints (razor-source family)")
    for i, ph in enumerate(INSERT_PHRASES_V3):
        errs.extend(_guard_insert_phrase(i, ph))
    if len(set(INSERT_PHRASES_V3)) != len(INSERT_PHRASES_V3):
        errs.append("duplicate insert phrases")
    old_specs = {s for s in lib.SPEC}
    for subj, (ru0, ru1, en0, en1) in SPEC_V3.items():
        if subj in old_specs or subj in lib.SUBJECTS:
            errs.append(f"subject {subj} collides with the classical bank")
        for j, ph in ((0, ru0), (1, ru1)):
            errs.extend(_guard_spec_phrase(subj, "ru", j, ph))
        for j, ph in ((0, en0), (1, en1)):
            errs.extend(_guard_spec_phrase(subj, "en", j, ph))
    known = {v for pair in (("v1.2.0", "v1.3.0"), ("v1.3.0", "v2.0.0"),
                            ("v1.2.0", "v2.0.0")) for v in pair}
    for va, vb in VERSION_PAIRS_V3:
        for v in (va, vb):
            if not lib.VERSION_RE.fullmatch(v):
                errs.append(f"version token {v!r} outside the guard alphabet")
            if v in known:
                errs.append(f"version token {v!r} reuses the classical family")
        if va == vb:
            errs.append(f"degenerate version pair ({va})")
    # distinctness of new SPEC sentences vs the classical SPEC corpus
    for subj, rows in SPEC_V3.items():
        for ph in rows:
            nph = _norm(ph)
            for os_, orow in lib.SPEC.items():
                for oph in orow:
                    if nph == _norm(oph):
                        errs.append(f"{subj} sentence duplicates {os_} SPEC")
    _STATIC_ERRORS = errs
    if errs:
        raise AssertionError(
            "BANKS-V3 STATIC GUARDS RED:\n  " + "\n  ".join(errs))


_run_static_guards()


def bank_content() -> dict:
    """Canonical content snapshot — sha256(json_dumps_sorted(...)) is the
    bank identity pinned at freeze (PARA-dictionary precedent)."""
    return {
        "insert_phrases_v3": list(INSERT_PHRASES_V3),
        "combos_v3": [list(c) for c in COMBOS_V3],
        "base_v3_ru": list(BASE_V3_RU),
        "base_v3b_ru": list(BASE_V3B_RU),
        "version_pairs_v3": [list(p) for p in VERSION_PAIRS_V3],
        "subjects_v3": {k: list(v) for k, v in SUBJECTS_V3.items()},
        "spec_v3": {k: list(v) for k, v in SPEC_V3.items()},
        "nc4_subjects_v3": list(NC4_SUBJECTS_V3),
        "nc4_deltas_v3": list(NC4_DELTAS_V3),
        "meta": BANK_META,
    }


def static_guard_errors() -> list[str]:
    return list(_STATIC_ERRORS)


# ===========================================================================
# BODY COMPOSITION (bank-owned a-side inventory; channel arithmetic
# mirrors base_body: 1 sentence -> short, 2-3 -> medium, +35 fillers -> long)
# ===========================================================================

def _nc1_body(combo_idx: int, fl: int) -> str:
    if fl == 0:
        return BASE_V3_RU[combo_idx % 3]
    return " ".join([BASE_V3_RU[0], BASE_V3_RU[1 + (combo_idx % 2)],
                     lib.FILLERS_RU[combo_idx % len(lib.FILLERS_RU)]])


def _nc1_body_long(phase: int) -> str:
    fill = lib.FILLERS_RU
    tail = " ".join(fill[(phase + i) % len(fill)] for i in range(35))
    return " ".join([BASE_V3_RU[0], BASE_V3_RU[1], tail])


def _nc3_body(fam_idx: int, subj_idx: int, channel: str) -> str:
    if channel == "fl0":
        return BASE_V3B_RU[fam_idx % 2]
    if channel == "fl1":
        return " ".join([BASE_V3B_RU[0], BASE_V3B_RU[1],
                         lib.FILLERS_RU[(fam_idx * 3 + subj_idx) % len(lib.FILLERS_RU)]])
    fill = lib.FILLERS_RU
    tail = " ".join(fill[(fam_idx * 4 + i) % len(fill)] for i in range(35))
    return " ".join([BASE_V3B_RU[0], BASE_V3B_RU[1], tail])


def _nc4_body(delta_idx: int, channel: str) -> str:
    if channel == "fl1":
        return " ".join([lib.BASE_RU[0], lib.BASE_RU[1],
                         lib.FILLERS_RU[delta_idx % len(lib.FILLERS_RU)]])
    fill = lib.FILLERS_RU
    tail = " ".join(fill[(delta_idx * 2 + i) % len(fill)] for i in range(35))
    return " ".join([lib.BASE_RU[0], lib.BASE_RU[1], tail])


#: NC-5 composition dial for the NEW side (6 variants — see module note:
#: the dial spreads a-side texts so cells sharing a subject do not collapse
#: into one side-sha at census/cut dedupe). Variants 3-5 carry ONE corpus
#: filler sentence (fillers <= 2 are free dial in the classical grid;
#: the G9 ">2 => shared boilerplate" rule is untouched).
def _nc5_comp(subj: str, idx: int) -> str:
    ru0, ru1, _, _ = SPEC_V3[subj]
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
# BUILDERS (declaration dicts copied VERBATIM from lib.build_scenario's
# branches — G9 declaration forms: slot ins/axis addition/op insert/
# value_b=full text; NC-3 slot ver/axis version/is_version_token=True)
# ===========================================================================

def _finish_rotate(sc: lib.Scenario, seed: int, rotate: bool,
                   r: random.Random, force_nn: bool = False) -> lib.Scenario:
    from build_conflict_detector_corpus import rotate_pair  # drv import safe
    if rotate:
        sc.side_a.body, sc.side_b.body = rotate_pair(
            sc.side_a.body, sc.side_b.body, seed)
    return lib.finish_pair(sc, r, NULL_P, force_nn)


def build_nc1_v3(sid: str, subj: str, ins: str, body: str) -> lib.Scenario:
    side_a = lib.make_side("ru", subj, body)
    side_b = lib.make_side("ru", subj, body + " " + ins)
    return lib.Scenario(
        scenario_id=sid, stratum="NC-1", side_a=side_a, side_b=side_b,
        label="not-contradicts",
        slots=[{"slot_id": "ins", "axis": "addition", "value_a": "",
                "value_b": ins, "op": "insert", "is_version_token": False,
                "span_chars": len(ins)}],
        subject_a=subj, subject_b=subj, polar_op=None, refine_ok=None,
        scaffold_id=f"NC-1:{subj}:sib:v3", cross_lang=False,
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


def build_nc3_v3(sid: str, subj: str, vp: tuple[str, str],
                 body: str) -> lib.Scenario:
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
        scaffold_id=f"NC-3:{subj}:{vp[0]}", cross_lang=False,
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


def build_nc4_v3(sid: str, subj: str, delta: str,
                 body: str) -> lib.Scenario:
    side_a = lib.make_side("ru", subj, body)
    side_b = lib.make_side("ru", subj, body)
    if delta == "type":
        side_b.record_type = "fact"
    elif delta == "lang":
        side_b.language = "en"
    elif delta == "tags":
        side_b.tags = sorted(set(side_b.tags + ["delta:meta"]))
    return lib.Scenario(
        scenario_id=sid, stratum="NC-4", side_a=side_a, side_b=side_b,
        label="not-contradicts", slots=[],
        subject_a=subj, subject_b=subj, polar_op=None, refine_ok=None,
        scaffold_id=f"NC-4:{subj}:{delta}:v3", cross_lang=False,
        length_channel=lib.channel_for(side_a.body), both_null=False,
        carrier=None, benign=[], para_b=0)


def build_nc5_v3(sid: str, sa: str, sb: str, kind: str,
                 comp_a: int, comp_b: int) -> lib.Scenario:
    """kind: 'same' | 'cross' | 'long'. G9 leg rules live in the generators;
    this builder composes the declared bodies:
      same  -> RU vs RU (both subjects' RU bodies)
      cross -> RU(a) vs FIRST EN sentence of the sibling (<= 200, short)
      long  -> both sides + the SHARED boilerplate block (fillers > 2).
    New-subject RU bodies come from the composition dial; old-subject RU
    bodies keep the classical plain form (style parity)."""
    blk = lib.filler_block("ru", 35)
    kind = "cross" if kind == "cross-rev" else kind  # lane leg alias
    if sa in SPEC_V3:
        body_a = _nc5_comp(sa, comp_a) if kind != "long" \
            else _nc5_comp(sa, 2) + " " + blk
        lang_a, anchor_a = "ru", sa
    else:
        body_a = _nc5_old_plain(sa) if kind != "long" \
            else _nc5_old_plain(sa) + " " + blk
        lang_a, anchor_a = "ru", sa
    if sb in SPEC_V3:
        body_b = _nc5_comp(sb, comp_b) if kind == "same" \
            else (SPEC_V3[sb][2] if kind == "cross" else _nc5_comp(sb, 2) + " " + blk)
        lang_b = "ru" if kind == "same" else "en"
    else:
        body_b = _nc5_old_plain(sb) if kind == "same" \
            else (lib.SPEC[sb][2] if kind == "cross" else _nc5_old_plain(sb) + " " + blk)
        lang_b = "ru" if kind == "same" else "en"
    side_a = lib.make_side(lang_a, anchor_a, body_a)
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
# SCOPED SUBJECT INJECTION (reversible; integrity-asserted)
# ===========================================================================

@contextlib.contextmanager
def v3_bank_context():
    """Make the new NC-5 subjects resolvable inside the context and restore
    the classical lib state afterwards (byte-equality assert). Nested use is
    idempotent: the inner snapshot is taken after the outer injection."""
    had_subj = set(SUBJECTS_V3) <= set(lib.SUBJECTS)
    had_spec = set(SPEC_V3) <= set(lib.SPEC)
    snap_subj = dict(lib.SUBJECTS)
    snap_spec = dict(lib.SPEC)
    if not had_subj:
        lib.SUBJECTS.update(SUBJECTS_V3)
    if not had_spec:
        lib.SPEC.update(SPEC_V3)
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
            "v3_bank_context: lib state failed to restore byte-equal"


# ===========================================================================
# GENERATORS (deterministic per-candidate streams; rotate EXPLICIT — the
# F7 trap: hold-side generation always rotates, rotate=True by default)
# ===========================================================================

def gen_nc1_v3(seed: int, prefix: str, rotate: bool = True) -> list[lib.Scenario]:
    out: list[lib.Scenario] = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-1-{ctr[0]:05d}"

    for subj in lib.TECH:
        for ci, combo in enumerate(COMBOS_V3):
            for fl in (0, 1):
                r = random.Random(f"{seed}:NC1V3:{subj}:{ci}:{fl}")
                ins = " ".join(INSERT_PHRASES_V3[i] for i in combo)
                sc = build_nc1_v3(sid(), subj, ins, _nc1_body(ci, fl))
                sc.scaffold_id = f"NC-1:{subj}:sib:v3x{ci}"
                out.append(_finish_rotate(sc, seed, rotate, r))
        for li in (0, 1):
            r = random.Random(f"{seed}:NC1V3L:{subj}:{li}")
            combo = tuple(i % len(INSERT_PHRASES_V3) for i in range(li * 3, li * 3 + 30))
            ins = " ".join(INSERT_PHRASES_V3[i] for i in combo)
            sc = build_nc1_v3(sid(), subj, ins, _nc1_body_long(li * 3))
            sc.scaffold_id = f"NC-1:{subj}:sib:v3xL{li}"
            out.append(_finish_rotate(sc, seed, rotate, r))
    return out


def gen_nc3_v3(seed: int, prefix: str, rotate: bool = True) -> list[lib.Scenario]:
    out: list[lib.Scenario] = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-3-{ctr[0]:05d}"

    for si, subj in enumerate(lib.TECH):
        for fi, vp in enumerate(VERSION_PAIRS_V3):
            for channel, tag in (("fl0", vp[1]), ("fl1", vp[1]), ("long", vp[1] + "L")):
                r = random.Random(f"{seed}:NC3V3:{subj}:{fi}:{channel}")
                sc = build_nc3_v3(sid(), subj, vp, _nc3_body(fi, si, channel))
                sc.scaffold_id = f"NC-3:{subj}:{vp[0]}:{tag}"
                out.append(_finish_rotate(sc, seed, rotate, r))
    return out


def gen_nc4_v3(seed: int, prefix: str, rotate: bool = True) -> list[lib.Scenario]:
    """FALLBACK MARGIN ONLY (G7): the old bank remains the cut source."""
    out: list[lib.Scenario] = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-4-{ctr[0]:05d}"

    for subj in NC4_SUBJECTS_V3:
        for di, delta in enumerate(NC4_DELTAS_V3):
            r = random.Random(f"{seed}:NC4V3:{subj}:{delta}:fl1")
            sc = build_nc4_v3(sid(), subj, delta, _nc4_body(di, "fl1"))
            out.append(_finish_rotate(sc, seed, rotate, r,
                                      force_nn=(delta in ("type", "lang"))))
            r = random.Random(f"{seed}:NC4V3:{subj}:{delta}:long")
            sc = build_nc4_v3(sid(), subj, delta, _nc4_body(di, "long"))
            sc.scaffold_id = f"NC-4:{subj}:{delta}:v3:L"
            out.append(_finish_rotate(sc, seed, rotate, r,
                                      force_nn=(delta in ("type", "lang"))))
    return out


def gen_nc5_v3(seed: int, prefix: str, rotate: bool = True) -> list[lib.Scenario]:
    out: list[lib.Scenario] = []
    ctr = [0]

    def sid() -> str:
        ctr[0] += 1
        return f"{prefix}-NC-5-{ctr[0]:05d}"

    new = list(SUBJECTS_V3)
    old = [s for s in lib.SUBJECTS if s not in SUBJECTS_V3]
    tech = set(lib.TECH)

    def comp_for(a: str, b: str) -> int:
        """Deterministic spread of a-side compositions over the sibling set."""
        sibs = sorted(set(new + old) - {a})
        return sibs.index(b) % 6

    def add(sa: str, sb: str, kind: str) -> None:
        r = random.Random(f"{seed}:NC5V3:{sa}:{sb}:{kind}")
        sc = build_nc5_v3(sid(), sa, sb, kind, comp_for(sa, sb), comp_for(sb, sa))
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


def gen_banks_v3_pool(seed: int, prefix: str, rotate: bool = True) -> list[lib.Scenario]:
    """The full new-bank candidate pool for the v3 census/cut lanes."""
    if prefix in {"tr", "ho"}:
        raise ValueError("v3 bank pools must use a lane-distinct prefix")
    with v3_bank_context():
        return (gen_nc1_v3(seed, prefix, rotate)
                + gen_nc5_v3(seed, prefix, rotate)
                + gen_nc3_v3(seed, prefix, rotate)
                + gen_nc4_v3(seed, prefix, rotate))


if __name__ == "__main__":
    with v3_bank_context():
        n1 = gen_nc1_v3(6117, "smoke", rotate=True)
        n5 = gen_nc5_v3(6117, "smoke", rotate=True)
        n3 = gen_nc3_v3(6117, "smoke", rotate=True)
        n4 = gen_nc4_v3(6117, "smoke", rotate=True)
    mech = sum(1 for x in n1 + n5 + n3 + n4 if lib.mechanical_ok(x))
    print(f"smoke: nc1={len(n1)} nc5={len(n5)} nc3={len(n3)} nc4={len(n4)} "
          f"total={len(n1)+len(n5)+len(n3)+len(n4)} mech_ok={mech}")
    print("lib dict integrity: SUBJECTS", len(lib.SUBJECTS), "SPEC", len(lib.SPEC))
