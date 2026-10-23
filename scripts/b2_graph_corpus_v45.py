#!/usr/bin/env python3
"""Build corpus v4.5 (train + sealed holdout) per prereg-v3 ADDENDUM-2.

Law (ACTIVATED, owner-ratified 2026-10-22, строго до прогона 1.n+1):
  docs/specs/b2-graph-prereg-v3.md (§4 thresholds frozen)
  docs/specs/b2-graph-prereg-v3-ADDENDUM-1.md (taxonomy §A, §C, §D)
  docs/specs/b2-graph-prereg-v3-ADDENDUM-2.md (BINDING recipe A2.1-A2.6/A2.8-A2.10)
  docs/experiments/b2-m11-design-package.md §3/§4 (ratified composition)
  docs/specs/labeling-policy-b2.md §8 policy v1.1 (whitelist, §8.3 disputed)

Base: v4.4-r2 train (4097 rows, fp bb1ca943...) — inherited strata are NOT
rebuilt. Integrity is asserted against the v4.4-r2 manifest BEFORE anything
else. The sealed holdout v4.4 (a9760d1f...) is NOT recreated, NOT rewritten
and never read for content (fingerprint discipline only).

Additions (ADDENDUM-2):
  1  translated-dup REGENERATE (slot bookkeeping; T verbatim / V free
     subfamilies per A2.1-A2.2). Parity per cos bin [0.55,0.65)/[0.65,0.75)/
     [0.75,0.86]: n_dup(new) >= n_sibling(frozen 144 block) and K >= 30 and
     cos continuity <= 0.005 (§D-б). The defective P-trans families of
     rev 1.n are REPLACED by this regeneration (g3 generation conflict;
     g3-analysis §2a/§3; build directive 2026-10-22). Sibling-144 and the
     v4.2 razor block are untouched.
  2  class-7 same-language (single-fact edit, short body): parity >= 1.0 in
     ch5j cells [0.80,0.85)/[0.85,0.90)/[0.90,0.95) at cos >= 0.985,
     lang_match = 1, K >= 30 per cell; every new row declares its lang group
     (the silent "addition landed in lm=0" failure of v4.4 is impossible).
  3  N-near tag-clone: same-language, tag_jaccard = 1.0, cos in [0.78,0.92]
     in ±0.005 bands, bld <= 60, K >= 30 per band, parity >= 1.0 against the
     positive mass of the same cell.
  4  strata A — real re-saves from the live store (read-only probe, census
     recipe §5.1): snapshot fingerprinted; growth vs 68 disclosed; clusters
     never split between train/holdout (holdout gets 2-3 FULL clusters,
     train mass >= 30 pairs); pair-level disjointness asserted; zero graph
     vector asserted by construction; §D-б continuity NOT applied
     (documented single-point-band exception, A2.4).

Sealed holdout v4.5 (A2.5): fresh 10% of the NEW v4.5 material only; seed
fixed; stratified label x evidence-flag x stratum; mint-top is split into
mint-top-evidence / mint-top-zero with separate quotas (declared 0/0 when
the new material carries no mint-top rows — disclosed, not silent); ONE
reading of "evidence row" = edge_ab=1 OR common_neighbors>=1, frozen at
build + fresh-sidecar agreement; unresolved-side rows -> quarantine counter;
pair-level disjointness + side-overlap flag slice at carve; cos_to_common_max
frozen in the export; §D representation assert extended to evidence material
where applicable.

Discipline: deterministic template-slot generator (label = function of
construction, A2.1/policy §4); NO live LLM calls (templates authored offline,
frozen before the build); dedup-aware generation (sample -> check -> refill
inside the generator); corpus-stage prereg asserts BEFORE the seal; seed 42;
order-deterministic; no wall-clock in content artifacts; holdout file never
opened for writing twice; double run must be byte-identical on every
artifact (the strata A snapshot is a pinned input, so re-runs reproduce).

Run (worktree checkout, project venv):
  /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/.venv/bin/python \
      scripts/b2_graph_corpus_v45.py \
      --snapshot data/b2-graph-corpus/v45/strata-a-snapshot.jsonl \
      --out-dir data/b2-graph-corpus/v45

Exit codes: 0 ok | 1 gate refusal | 2 usage/validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.features.pair import (  # noqa: E402
    _containment,
    _jaccard,
    _ngrams,
)

ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src"
BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
POOL_JSONL = REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl"
V44R2_DIR = REPO_ROOT / "data/b2-graph-corpus/v44-r2"
V44_HOLDOUT = REPO_ROOT / "data/b2-graph-corpus/v44/holdout.jsonl"

# ── frozen pins (v4.4-r2 manifest; DO NOT EDIT) ──────────────────────────────

BASE_TRAIN_FP = "bb1ca943d6beabf56a88450c977cd04f829bdee89d75b81dc1b5f394b1314106"
BASE_LABELS_FP = "e11e329849cd0f5fdca13a242db471ad4106f517c5e09e0277cb89af621b2f16"
BASE_ROWS = 4097
V44_HOLDOUT_FP = "a9760d1f06ecdd40530bcc2dc37311f5e5bcd23aad653e03c87a49fe36761a97"

# census anchors (evidence-census 2026-10-22; asserted, growth disclosed)
CENSUS_SIBLING_BINS = {"[0.55,0.65)": 42, "[0.65,0.75)": 66, "[0.75,0.86]": 36}
CENSUS_C7_CELLS = {"[0.80,0.85)": 26, "[0.85,0.90)": 14, "[0.90,0.95)": 112}
CENSUS_NNEAR_POS = 60

SEED = 42
HOLDOUT_SHARE = 0.10

# A2.2 constants
TVER_BINS = ((0.55, 0.65), (0.65, 0.75), (0.75, 0.86))
BIN_MIN_K = 30
CONTINUITY_DELTA = 0.005
BLD_T_MAX = 60
BLD_V_MAX = 130
SCAFFOLD_MIN = 20
SCAFFOLD_MAX_SHARE = 0.05

# A2.3 constants
C7_CH5J_CELLS = ((0.80, 0.85), (0.85, 0.90), (0.90, 0.95))
C7_COS_MIN = 0.985
C7_BODY_MAX = 400  # "short body" operationalization, logged
NN_LOW, NN_HIGH = 0.78, 0.92
NN_BAND = 0.005
NN_BLD_MAX = 60

PROBE_K = 30  # §D band K
SIDE_KEYS = ("title", "body", "tags", "language", "record_type", "created_at")

GRAPH_FEATURES = (
    "edge_ab",
    "edge_kind_supersedes",
    "common_neighbors",
    "neighbor_jaccard",
    "cos_to_common_max",
)


class GateRefusal(SystemExit):
    """An assert failed — the corpus is NOT sealed."""


def refuse(msg: str) -> None:
    raise GateRefusal(f"[v45] GATE REFUSAL: {msg}")


def log(runlog: list[dict], event: str, **fields: Any) -> None:
    entry = {"event": event, **fields}
    runlog.append(entry)
    print(
        f"[v45] {event}: " + " ".join(f"{k}={v}" for k, v in fields.items()),
        file=sys.stderr,
    )


# ── shared helpers (byte-identical recipes from the v44r2 builder) ───────────


def text_hash(text: str) -> str:
    n = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def side_sha(s: dict) -> str:
    payload = json.dumps(
        {
            "title": s.get("title"),
            "body": s.get("body"),
            "tags": sorted(s.get("tags") or []),
            "language": s.get("language"),
            "record_type": s.get("record_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_key(row_or_pair) -> tuple[str, str]:
    if isinstance(row_or_pair, dict) and "record" in row_or_pair:
        return tuple(
            sorted((side_sha(row_or_pair["record"]), side_sha(row_or_pair["candidate"])))
        )
    a, b = row_or_pair
    return tuple(sorted((side_sha(a), side_sha(b))))


def corpus_rows(path: Path) -> list[dict]:
    if not path.is_file():
        refuse(f"frozen input missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def tag_jaccard(a: dict, b: dict) -> float:
    sa, sb = set(a.get("tags") or []), set(b.get("tags") or [])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def detect_lang(body: str | None) -> str:
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_embedder():
    if ENGINE_SRC not in sys.path:
        sys.path.insert(0, ENGINE_SRC)
    from vesmaro.embeddings import NanoProvider  # noqa: E402

    provider = NanoProvider()
    pin = provider.fingerprint
    if not pin.startswith("nano:sha256:3b752e06"):
        refuse(f"embedder pin {pin!r} differs from the frozen prefix")
    cache: dict[str, list[float]] = {}

    def embed_text(text: str) -> list[float]:
        text = text[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    def embed_side(s: dict) -> list[float]:
        tags = " ".join(s.get("tags") or [])
        return embed_text(f"{s.get('title') or ''}\n{s.get('body') or ''}\n{tags}")

    return embed_side, embed_text, pin


def ch_ngrams(side: dict, n: int) -> frozenset[str]:
    text = " ".join(f"{side.get('title') or ''}\n{side.get('body') or ''}".split()).lower()
    return _ngrams(text, n)


def ch_jaccard(a_side: dict, b_side: dict, n: int) -> float:
    return _jaccard(ch_ngrams(a_side, n), ch_ngrams(b_side, n))


def ch_containment(a_side: dict, b_side: dict, n: int) -> float:
    return _containment(ch_ngrams(a_side, n), ch_ngrams(b_side, n))


def lang_match(a: dict, b: dict) -> float:
    la, lb = a.get("language"), b.get("language")
    if la is None or lb is None:
        return 0.0
    return 1.0 if la == lb else 0.0


def body_len_delta(a: dict, b: dict) -> int:
    return abs(len(a.get("body") or "") - len(b.get("body") or ""))


# ── frozen text utilities (v4.2 razor heritage) ──────────────────────────────

_FACT_RE = re.compile(r"\d+(?:[:.\-_/]\d+)*")


def fact_token_spans(body: str) -> list[tuple[int, int]]:
    return [
        m.span()
        for m in _FACT_RE.finditer(body)
        if any(c.isdigit() for c in m.group())
    ]


def cipher_span(body: str, span: tuple[int, int]) -> str:
    chars = list(body)
    for p in range(span[0], span[1]):
        if body[p].isdigit():
            chars[p] = "0" if body[p] != "0" else "7"
    return "".join(chars)


# v4.2-style fact-WORD substitutions (one word = one date/weekday fact edited;
# same register as the v4.2 razor heritage whose heavy single-fact edits are
# the exam geometry). Frozen table; the replacement is the NEXT entry
# (deterministic cyclic shift). Months/weekdays are §8 key facts (dates).
WORD_FACTS: dict[str, dict[str, list[str]]] = {
    "ru": {
        "months": ["января", "февраля", "марта", "апреля", "мая", "июня",
                   "июля", "августа", "сентября", "октября", "ноября", "декабря"],
        "weekdays": ["понедельник", "вторник", "среда", "четверг", "пятница",
                     "суббота", "воскресенье"],
    },
    "en": {
        "months": ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November",
                   "December"],
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"],
    },
}


def split_compound(text: str) -> str:
    """Structural lever for the near-clone side: split the first compound
    sentence at ', ' (order and facts unchanged, structure diverges)."""
    idx = text.find(", ")
    if idx < 0:
        return text
    return text[:idx] + ". " + text[idx + 2:]


def case_run(text: str, dose: float) -> str:
    """Legacy v4.2/v4.4-r2 band-placement lever (byte-identical recipe of
    scripts/b2_graph_corpus_v44r2.py): swap the case of the first ``dose``
    fraction of characters. Normalization-invisible (lowercase text is
    unchanged, so char geometry and tag geometry survive); the measured
    cosine is the only feature that moves.DOSE_LADDER matches v44r2."""
    n = len(text)
    length = int(n * dose)
    return "".join((c.swapcase() if p < length else c)
                   for p, c in enumerate(text))


DOSE_LADDER = (0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 1.0)


def word_fact_edits(body: str, lang: str) -> list[tuple[str, str]]:
    """All single-fact WORD edits of the body (each found fact word shifted to
    the next table entry). Returns (edited_body, fact_word) pairs."""
    out: list[tuple[str, str]] = []
    tables = WORD_FACTS[lang]
    for kind, words in tables.items():
        for i, w in enumerate(words):
            start = 0
            while True:
                idx = body.find(w, start)
                if idx < 0:
                    break
                end = idx + len(w)
                boundary = ((idx == 0 or not (body[idx - 1].isalnum()))
                            and (end == len(body) or not body[end].isalnum()))
                if boundary:
                    new = body[:idx] + words[(i + 1) % len(words)] + body[end:]
                    out.append((new, w))
                start = idx + len(w)
    return out


# ── number/date verbalization for the V subfamily (offline, frozen) ─────────

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")
_MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December")
_MONTHS_RU = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря")


def num_words_en(n: int) -> str:
    if n < 0:
        return "minus " + num_words_en(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + num_words_en(n % 100) if n % 100 else "")
    if n < 1_000_000:
        th = n // 1000
        th_txt = num_words_en(th) if th < 1000 else str(th)
        return th_txt + " thousand" + (" " + num_words_en(n % 1000) if n % 1000 else "")
    return str(n)


def date_parts(iso: str) -> tuple[int, int, int]:
    y, m, d = iso.split("-")
    return int(y), int(m), int(d)


def date_verbal_en(iso: str) -> str:
    y, m, d = date_parts(iso)
    return f"{_MONTHS_EN[m - 1]} {d}, {y}"


def date_verbal_ru(iso: str) -> str:
    y, m, d = date_parts(iso)
    return f"{d} {_MONTHS_RU[m - 1]} {y}"


# ── the frozen scaffold library (authored offline BEFORE the build; the LLM
#    role was draftsman only — the templates below are the frozen result;
#    label = function of construction) ────────────────────────────────────────
#
# Each scaffold carries:
#   id, tags (envelope: identical on every side of the scaffold), type,
#   title {ru, en} (static, no slots),
#   dup {ru, en} body templates + slot ladders   -> translated-dup pairs
#       (T subfamily renders each slot token verbatim on BOTH sides;
#        V subfamily renders ru naturally and verbalizes the en side)
#   sib {ru, en} body templates + slot ladders   -> same-language near-clone
#       pairs (class 8, tag-clone): disjoint facts, same topic, same tags;
#       born from the SAME scaffold as the dup family (alignment §D:
#       scaffold -> both families -> train instances + probes).
# dup and sib slot ladders are value-disjoint by design; the no-shared-slot-
#   token sibling assert (A2.2) is re-measured on rendered text.
# Slot value encoding (all §8 key-fact classes: numbers/dates/names/versions/
#   amounts): {"t", "ru_t", "en_t", "ru_v", "en_v"} — t is the canonical
#   token contained in both T-family renders; the slot map digest rides the
#   row, never the raw values.


def N(tok: str, en_v: str | None = None) -> dict:
    """Integer slot: verbatim everywhere; en V-family verbalizes."""
    return {"t": tok, "ru_t": tok, "en_t": tok, "ru_v": tok,
            "en_v": en_v or num_words_en(int(tok))}


def D(iso: str) -> dict:
    """ISO date slot: T verbatim ISO; V verbal (ru words / en words)."""
    return {"t": iso, "ru_t": iso, "en_t": iso,
            "ru_v": date_verbal_ru(iso), "en_v": date_verbal_en(iso)}


def M(tok: str, ru_unit: str, en_unit: str, en_v: str) -> dict:
    """Money/amount slot with a currency unit."""
    return {"t": tok, "ru_t": f"{tok} {ru_unit}", "en_t": f"{tok} {en_unit}",
            "ru_v": f"{tok} {ru_unit}", "en_v": f"{en_v} {en_unit}"}


def X(token: str, en: str | None = None) -> dict:
    """Proper-noun/token slot: identical on both sides in both subfamilies."""
    return {"t": token, "ru_t": token, "en_t": en or token,
            "ru_v": token, "en_v": en or token}


SCAFFOLDS: list[dict] = [
    {
        "id": "S01", "tags": ["tech", "storage"], "type": "note",
        "title": {"ru": "Чистка памяти планшета", "en": "Cleaning the tablet storage"},
        "dup": {
            "ru": "Залил {dev} под {pct} процентов, скинул архив в облако {iso} и освободил {gb} гигабайт. Кэш приложений чищу по расписанию.",
            "en": "Filled the {dev} up to {pct} percent, moved the archive to the cloud on {iso} and freed {gb} gigabytes. App cache is cleaned on a schedule.",
        },
        "slots": {
            "dev": [X("Pixel"), X("ThinkPad")],
            "pct": [N("95"), N("91")],
            "iso": [D("2026-04-04"), D("2026-03-21")],
            "gb": [N("20"), N("34")],
        },
        "sib": {
            "ru": "Купил карту памяти на {tgb} гигабайт и перенёс туда фотоархив {siso}. Планшет перестал тормозить на съёмке.",
            "en": "Bought a {tgb} gigabyte memory card and moved the photo archive there on {siso}. The tablet stopped lagging while shooting.",
            "slots": {
                "tgb": [N("128"), N("256")],
                "siso": [D("2026-05-02"), D("2026-04-18")],
            },
        },
    },
    {
        "id": "S02", "tags": ["tech", "hardware"], "type": "fact",
        "title": {"ru": "Батарея ноутбука", "en": "Laptop battery"},
        "dup": {
            "ru": "Поставил новую батарею в {dev}: автономность держится {hrs} часов, замена обошлась в {cost}. Нагрев ушёл.",
            "en": "Installed a new battery in the {dev}: runtime holds {hrs} hours, the replacement cost {cost}. The overheating is gone.",
        },
        "slots": {
            "dev": [X("ThinkPad"), X("MacBook")],
            "hrs": [N("8"), N("6")],
            "cost": [M("4900", "рублей", "rubles", "four thousand nine hundred"),
                     M("5300", "рублей", "rubles", "five thousand three hundred")],
        },
        "sib": {
            "ru": "Заказал батарею на {dev}, поставка через {days} дней, пока сижу на зарядке. Старая держит от силы {shrs} часа.",
            "en": "Ordered a battery for the {dev}, delivery in {days} days, so I stay plugged in for now. The old one holds barely {shrs} hours.",
            "slots": {
                "dev": [X("Asus"), X("Acer")],
                "days": [N("21"), N("14")],
                "shrs": [N("2"), N("3")],
            },
        },
    },
    {
        "id": "S03", "tags": ["money", "subscription"], "type": "note",
        "title": {"ru": "Подписка на сервис", "en": "Service subscription"},
        "dup": {
            "ru": "Продлил подписку {srv} на год {iso}, списали {cost}. Напоминание поставил за неделю.",
            "en": "Renewed the {srv} subscription for a year on {iso}, they charged {cost}. Set a reminder a week ahead.",
        },
        "slots": {
            "srv": [X("Figma"), X("Notion")],
            "iso": [D("2026-04-10"), D("2026-03-28")],
            "cost": [M("5900", "рублей", "rubles", "five thousand nine hundred"),
                     M("7200", "рублей", "rubles", "seven thousand two hundred")],
        },
        "sib": {
            "ru": "Отменил подписку {srv} {siso}: пользуемся раз в месяц, {scost} в год не оправдывают. Данные выгрузил.",
            "en": "Cancelled the {srv} subscription on {siso}: we use it once a month, {scost} a year is not justified. Exported the data.",
            "slots": {
                "srv": [X("Trello"), X("Linear")],
                "siso": [D("2026-05-06"), D("2026-04-22")],
                "scost": [M("8400", "рублей", "rubles", "eight thousand four hundred"),
                          M("9600", "рублей", "rubles", "nine thousand six hundred")],
            },
        },
    },
    {
        "id": "S04", "tags": ["health", "dentist"], "type": "fact",
        "title": {"ru": "Приём у стоматолога", "en": "Dentist appointment"},
        "dup": {
            "ru": "Был у стоматолога {iso}: вылечили {tooth} зуб, поставили пломбу за {cost}. Записался на контроль.",
            "en": "Saw the dentist on {iso}: they treated tooth {tooth}, the filling cost {cost}. Booked a check-up.",
        },
        "slots": {
            "iso": [D("2026-04-14"), D("2026-03-30")],
            "tooth": [N("36"), N("19")],
            "cost": [M("3800", "рублей", "rubles", "three thousand eight hundred"),
                     M("4500", "рублей", "rubles", "four thousand five hundred")],
        },
        "sib": {
            "ru": "Стоматолог посоветовал профгигиену раз в полгода, записался на {siso}, чистка стоит {scost}. Пасту поменял.",
            "en": "The dentist recommended a professional cleaning every six months, booked {siso}, the cleaning costs {scost}. Switched toothpaste.",
            "slots": {
                "siso": [D("2026-05-20"), D("2026-06-02")],
                "scost": [M("3200", "рублей", "rubles", "three thousand two hundred"),
                          M("3600", "рублей", "rubles", "three thousand six hundred")],
            },
        },
    },
    {
        "id": "S05", "tags": ["home", "plants"], "type": "note",
        "title": {"ru": "Пересадка фикуса", "en": "Repotting the ficus"},
        "dup": {
            "ru": "Пересадил фикус {iso} в горшок на {lit} литра, добавил дренаж {cm} см. Выбросил два листа.",
            "en": "Repotted the ficus on {iso} into a {lit} liter pot, added {cm} cm of drainage. It dropped two leaves.",
        },
        "slots": {
            "iso": [D("2026-04-05"), D("2026-03-15")],
            "lit": [N("3"), N("5")],
            "cm": [N("4"), N("6")],
        },
        "sib": {
            "ru": "Фикус дал {sn} новых листьев после подкормки {siso}, землю менял весной. Ставлю на светлый подоконник.",
            "en": "The ficus grew {sn} new leaves after feeding on {siso}, changed the soil in spring. Moving it to a bright windowsill.",
            "slots": {
                "sn": [N("7"), N("9")],
                "siso": [D("2026-05-11"), D("2026-04-25")],
            },
        },
    },
    {
        "id": "S06", "tags": ["auto", "service"], "type": "fact",
        "title": {"ru": "ТО автомобиля", "en": "Car maintenance"},
        "dup": {
            "ru": "Прошёл ТО на {car} {iso}: пробег {km} км, замена масла и фильтров обошлась в {cost}.",
            "en": "Did the maintenance on the {car} on {iso}: mileage {km} km, oil and filter change cost {cost}.",
        },
        "slots": {
            "car": [X("Kia"), X("Skoda")],
            "iso": [D("2026-04-08"), D("2026-03-25")],
            "km": [N("62000"), N("58500")],
            "cost": [M("12400", "рублей", "rubles", "twelve thousand four hundred"),
                     M("13800", "рублей", "rubles", "thirteen thousand eight hundred")],
        },
        "sib": {
            "ru": "На {scar} поменял тормозные колодки {siso}, ходят тихо, задние next season. Стоило {scost}.",
            "en": "Changed the brake pads on the {scar} on {siso}, they run quiet, rear ones next season. It cost {scost}.",
            "slots": {
                "scar": [X("Hyundai"), X("Volkswagen")],
                "siso": [D("2026-05-15"), D("2026-04-29")],
                "scost": [M("6800", "рублей", "rubles", "six thousand eight hundred"),
                          M("7400", "рублей", "rubles", "seven thousand four hundred")],
            },
        },
    },
    {
        "id": "S07", "tags": ["sport", "gym"], "type": "note",
        "title": {"ru": "Прогресс в зале", "en": "Gym progress"},
        "dup": {
            "ru": "Взял {w} кг в становой на {reps} повторов {iso}, спина в порядке. Пояс надел на последний подход.",
            "en": "Pulled {w} kg for {reps} reps in the deadlift on {iso}, back feels fine. Belted up for the last set.",
        },
        "slots": {
            "w": [N("120"), N("110")],
            "reps": [N("5"), N("3")],
            "iso": [D("2026-04-12"), D("2026-03-29")],
        },
        "sib": {
            "ru": "Перешёл на {sreps} подходов жима лёжа {siso}, вес пока {sw} кг, расту по 2.5 в неделю.",
            "en": "Switched to {sreps} sets on the bench on {siso}, weight is {sw} kg for now, adding 2.5 weekly.",
            "slots": {
                "sreps": [N("4"), N("6")],
                "siso": [D("2026-05-04"), D("2026-04-20")],
                "sw": [N("75"), N("80")],
            },
        },
    },
    {
        "id": "S08", "tags": ["money", "food"], "type": "note",
        "title": {"ru": "Бюджет на продукты", "en": "Grocery budget"},
        "dup": {
            "ru": "Сходил за продуктами {iso}, уложился в {sum}: мясо, овощи, крупы на неделю. Чек сохранил.",
            "en": "Did the grocery run on {iso}, stayed within {sum}: meat, vegetables, grains for a week. Kept the receipt.",
        },
        "slots": {
            "iso": [D("2026-04-11"), D("2026-04-01")],
            "sum": [M("5400", "рублей", "rubles", "five thousand four hundred"),
                    M("6100", "рублей", "rubles", "six thousand one hundred")],
        },
        "sib": {
            "ru": "Начал вести таблицу расходов {siso}: в среднем выходит {ssum} в месяц, доставка съедает лишнее.",
            "en": "Started an expense sheet on {siso}: it averages {ssum} a month, delivery eats the surplus.",
            "slots": {
                "siso": [D("2026-04-19"), D("2026-03-08")],
                "ssum": [M("21500", "рублей", "rubles", "twenty-one thousand five hundred"),
                         M("23800", "рублей", "rubles", "twenty-three thousand eight hundred")],
            },
        },
    },
    {
        "id": "S09", "tags": ["home", "kitchen"], "type": "fact",
        "title": {"ru": "Чайник: накипь", "en": "Kettle: limescale"},
        "dup": {
            "ru": "Почистил чайник лимонной кислотой {iso}: {pack} пакетика на {min} минут, налёт ушёл полностью.",
            "en": "Descaled the kettle with citric acid on {iso}: {pack} sachets for {min} minutes, the scale came off completely.",
        },
        "slots": {
            "iso": [D("2026-04-03"), D("2026-03-18")],
            "pack": [N("2"), N("3")],
            "min": [N("30"), N("45")],
        },
        "sib": {
            "ru": "Купил новый чайник {siso}, на {sl} л, с фильтром от накипи. Старый отдал на дачу.",
            "en": "Bought a new kettle on {siso}, {sl} liters, with a scale filter. Gave the old one to the dacha.",
            "slots": {
                "siso": [D("2026-04-26"), D("2026-05-09")],
                "sl": [X("1.7"), X("0.9")],
            },
        },
    },
    {
        "id": "S10", "tags": ["tech", "network"], "type": "note",
        "title": {"ru": "Настройка роутера", "en": "Router setup"},
        "dup": {
            "ru": "Перенастроил роутер {iso}: канал {ch}, ширина {mhz} МГц, скорость поднялась до {mbps} Мбит/с.",
            "en": "Reconfigured the router on {iso}: channel {ch}, width {mhz} MHz, speed went up to {mbps} Mbit/s.",
        },
        "slots": {
            "iso": [D("2026-04-06"), D("2026-03-22")],
            "ch": [N("6"), N("11")],
            "mhz": [N("40"), N("80")],
            "mbps": [N("340"), N("420")],
        },
        "sib": {
            "ru": "Поставил Mesh-точку {siso} в коридоре, покрытие теперь и на балконе. Гостевая сеть отдельная.",
            "en": "Added a mesh node on {siso} in the hallway, coverage now reaches the balcony. Guest network is separate.",
            "slots": {
                "siso": [D("2026-04-28"), D("2026-05-13")],
            },
        },
    },
    {
        "id": "S11", "tags": ["hobby", "reading"], "type": "note",
        "title": {"ru": "Книжный клуб", "en": "Book club"},
        "dup": {
            "ru": "Дочитал {book}: {pages} страниц за {days} дней, обсудили {iso}. Лягу спать позже.",
            "en": "Finished {book}: {pages} pages in {days} days, we discussed it on {iso}. Going to bed late.",
        },
        "slots": {
            "book": [X("Dune"), X("Solaris")],
            "pages": [N("412"), N("296")],
            "days": [N("9"), N("6")],
            "iso": [D("2026-04-09"), D("2026-03-27")],
        },
        "sib": {
            "ru": "Взял в клубе следующую книгу {siso}, {spages} страниц, чтение до июня. Веду конспект.",
            "en": "Picked the next club book on {siso}, {spages} pages, reading until June. Keeping notes.",
            "slots": {
                "siso": [D("2026-04-23"), D("2026-05-07")],
                "spages": [N("348"), N("274")],
            },
        },
    },
    {
        "id": "S12", "tags": ["travel", "rail"], "type": "fact",
        "title": {"ru": "Билеты на поезд", "en": "Train tickets"},
        "dup": {
            "ru": "Взял билеты {iso}: вагон {car}, места {seat}, заплатил {cost}. Поезд вечерний.",
            "en": "Got the tickets on {iso}: car {car}, seats {seat}, paid {cost}. The train is an evening one.",
        },
        "slots": {
            "iso": [D("2026-05-16"), D("2026-06-03")],
            "car": [N("7"), N("12")],
            "seat": [N("22"), N("35")],
            "cost": [M("4300", "рублей", "rubles", "four thousand three hundred"),
                     M("5100", "рублей", "rubles", "five thousand one hundred")],
        },
        "sib": {
            "ru": "Сдал один билет {siso} — поездка переносится, вернули {scost} из {sfull}. Возьму на другой день.",
            "en": "Returned one ticket on {siso} — the trip moves, got {scost} back out of {sfull}. Will book for another day.",
            "slots": {
                "siso": [D("2026-05-08"), D("2026-05-21")],
                "scost": [M("3300", "рублей", "rubles", "three thousand three hundred"),
                          M("4000", "рублей", "rubles", "four thousand")],
                "sfull": [M("5200", "рублей", "rubles", "five thousand two hundred"),
                          M("6400", "рублей", "rubles", "six thousand four hundred")],
            },
        },
    },
    {
        "id": "S13", "tags": ["health", "labs"], "type": "fact",
        "title": {"ru": "Анализы", "en": "Blood work"},
        "dup": {
            "ru": "Сдал анализы {iso}: витамин D — {val} нг/мл, врач сказал норма. Пересдача через полгода.",
            "en": "Did the blood work on {iso}: vitamin D at {val} ng/ml, the doctor says it is normal. Recheck in six months.",
        },
        "slots": {
            "iso": [D("2026-04-07"), D("2026-03-24")],
            "val": [N("38"), N("42")],
        },
        "sib": {
            "ru": "Записался на диспансеризацию {siso}, список анализов расширенный, натощак. Паспорт не забыть.",
            "en": "Booked a full check-up on {siso}, the test list is extended, fasting required. Do not forget the ID.",
            "slots": {
                "siso": [D("2026-05-26"), D("2026-06-10")],
            },
        },
    },
    {
        "id": "S14", "tags": ["tech", "backup"], "type": "note",
        "title": {"ru": "Бэкапы", "en": "Backups"},
        "dup": {
            "ru": "Прогнал полный бэкап {iso}: {gb} гигабайт за {min} минут, checksum сошёлся. Диск гудит ровно.",
            "en": "Ran the full backup on {iso}: {gb} gigabytes in {min} minutes, checksum matched. The drive hums evenly.",
        },
        "slots": {
            "iso": [D("2026-04-13"), D("2026-04-02")],
            "gb": [N("310"), N("285")],
            "min": [N("52"), N("47")],
        },
        "sib": {
            "ru": "Освободил {sgb} гигабайт на бэкап-диске {siso}, удалил старые снапшоты. Ротация теперь месяц.",
            "en": "Freed {sgb} gigabytes on the backup disk on {siso}, removed old snapshots. Rotation is monthly now.",
            "slots": {
                "sgb": [N("95"), N("120")],
                "siso": [D("2026-04-21"), D("2026-05-01")],
            },
        },
    },
    {
        "id": "S15", "tags": ["study", "language"], "type": "note",
        "title": {"ru": "Курс английского", "en": "English course"},
        "dup": {
            "ru": "Закрыл модуль {mod} на платформе {iso}: {words} новых слов, тест на {pct} процентов.",
            "en": "Completed module {mod} on the platform on {iso}: {words} new words, test at {pct} percent.",
        },
        "slots": {
            "mod": [N("5"), N("6")],
            "iso": [D("2026-04-10"), D("2026-03-31")],
            "words": [N("120"), N("95")],
            "pct": [N("88"), N("92")],
        },
        "sib": {
            "ru": "Записался на разговорный клуб {siso}, встречи по средам, {smin} минут через Zoom. Словарь повторяю.",
            "en": "Joined a speaking club on {siso}, meetings on Wednesdays, {smin} minutes over Zoom. Reviewing vocabulary.",
            "slots": {
                "siso": [D("2026-04-15"), D("2026-04-29")],
                "smin": [N("60"), N("90")],
            },
        },
    },
    {
        "id": "S16", "tags": ["sport", "bike"], "type": "fact",
        "title": {"ru": "Ремонт велосипеда", "en": "Bike repair"},
        "dup": {
            "ru": "Перебрал задний переключатель {iso}: тросик, рубашка, регулировка — {cost}. Передачи щёлкают чётко.",
            "en": "Overhauled the rear derailleur on {iso}: cable, housing, tuning — {cost}. The gears click crisply now.",
        },
        "slots": {
            "iso": [D("2026-04-04"), D("2026-03-20")],
            "cost": [M("1900", "рублей", "rubles", "one thousand nine hundred"),
                     M("2400", "рублей", "rubles", "two thousand four hundred")],
        },
        "sib": {
            "ru": "Заказал новые покрышки {siso}, {sinch} дюйма, летний протектор. Старые донор.",
            "en": "Ordered new tires on {siso}, {sinch} inches, summer tread. The old ones are spares.",
            "slots": {
                "siso": [D("2026-04-24"), D("2026-05-05")],
                "sinch": [N("27"), N("29")],
            },
        },
    },
    {
        "id": "S17", "tags": ["home", "rent"], "type": "fact",
        "title": {"ru": "Аренда квартиры", "en": "Apartment rent"},
        "dup": {
            "ru": "Внёс аренду за {mon} месяцев {iso}: {sum}, коммуналка отдельно. Договор продлили.",
            "en": "Paid the rent for {mon} months on {iso}: {sum}, utilities separate. The contract is extended.",
        },
        "slots": {
            "mon": [N("2"), N("1")],
            "iso": [D("2026-04-01"), D("2026-03-05")],
            "sum": [M("92000", "рублей", "rubles", "ninety-two thousand"),
                    M("46000", "рублей", "rubles", "forty-six thousand")],
        },
        "sib": {
            "ru": "Хозяин поднял аренду {siso} на {spct} процентов — договорились на компромисс. Ремонт за его счёт.",
            "en": "The landlord raised the rent by {spct} percent on {siso} — we agreed on a compromise. Repairs are on him.",
            "slots": {
                "siso": [D("2026-04-18"), D("2026-05-02")],
                "spct": [N("5"), N("7")],
            },
        },
    },
    {
        "id": "S18", "tags": ["food", "coffee"], "type": "note",
        "title": {"ru": "Кофе: зерно", "en": "Coffee beans"},
        "dup": {
            "ru": "Взял {grams} граммов зерна {iso}: обжарка {roast}, помол под воронку, {cost}. Аромат густой.",
            "en": "Got {grams} grams of beans on {iso}: {roast} roast, ground for the pour-over, {cost}. The aroma is rich.",
        },
        "slots": {
            "grams": [N("500"), N("250")],
            "iso": [D("2026-04-09"), D("2026-03-26")],
            "roast": [X("Ethiopia"), X("Colombia")],
            "cost": [M("1650", "рублей", "rubles", "one thousand six hundred fifty"),
                     M("1200", "рублей", "rubles", "one thousand two hundred")],
        },
        "sib": {
            "ru": "Поменял кофемолку {siso}: жернова, {sgrad} ступеней помола, шумит тише. Старую продал.",
            "en": "Swapped the grinder on {siso}: burrs, {sgrad} grind settings, runs quieter. Sold the old one.",
            "slots": {
                "siso": [D("2026-04-30"), D("2026-05-14")],
                "sgrad": [N("15"), N("40")],
            },
        },
    },
    {
        "id": "S19", "tags": ["docs", "government"], "type": "fact",
        "title": {"ru": "Замена паспорта", "en": "Passport renewal"},
        "dup": {
            "ru": "Подал на замену паспорта {iso}, готовность через {days} дней, пошлина {fee}. Заберу утром.",
            "en": "Filed the passport renewal on {iso}, ready in {days} days, the fee is {fee}. Will pick it up in the morning.",
        },
        "slots": {
            "iso": [D("2026-04-02"), D("2026-03-19")],
            "days": [N("10"), N("14")],
            "fee": [M("1500", "рублей", "rubles", "one thousand five hundred"),
                    M("3000", "рублей", "rubles", "three thousand")],
        },
        "sib": {
            "ru": "Записался на оформление загранпаспорта {siso}, биометрия на месте, срок изготовления {sweeks} недели.",
            "en": "Booked the international passport appointment on {siso}, biometrics on site, processing takes {sweeks} weeks.",
            "slots": {
                "siso": [D("2026-05-12"), D("2026-05-27")],
                "sweeks": [N("4"), N("6")],
            },
        },
    },
    {
        "id": "S20", "tags": ["hobby", "aquarium"], "type": "note",
        "title": {"ru": "Аквариум", "en": "The aquarium"},
        "dup": {
            "ru": "Почистил аквариум {iso}: подмена {pct} процентов воды, фильтр промыл, рыбки активны.",
            "en": "Cleaned the aquarium on {iso}: {pct} percent water change, rinsed the filter, the fish are active.",
        },
        "slots": {
            "iso": [D("2026-04-05"), D("2026-03-22")],
            "pct": [N("25"), N("30")],
        },
        "sib": {
            "ru": "Запустил новый фильтр {siso}: производительность {slph} л/ч, шумит заметно тише старого.",
            "en": "Started the new filter on {siso}: throughput {slph} l/h, noticeably quieter than the old one.",
            "slots": {
                "siso": [D("2026-04-27"), D("2026-05-10")],
                "slph": [N("240"), N("320")],
            },
        },
    },
    {
        "id": "S21", "tags": ["home", "heating"], "type": "fact",
        "title": {"ru": "Обслуживание котла", "en": "Boiler service"},
        "dup": {
            "ru": "Вызвал мастера по котлу {iso}: чистка теплообменника, {cost}, давление выставили {bar} бар.",
            "en": "Called the boiler technician on {iso}: heat exchanger cleaning, {cost}, pressure set to {bar} bar.",
        },
        "slots": {
            "iso": [D("2026-04-08"), D("2026-03-27")],
            "cost": [M("5200", "рублей", "rubles", "five thousand two hundred"),
                     M("4700", "рублей", "rubles", "four thousand seven hundred")],
            "bar": [X("1.5"), X("1.2")],
        },
        "sib": {
            "ru": "Заказал термостат {siso}, программатор на неделю, расход газа должен упасть. Ставлю сам.",
            "en": "Ordered a thermostat on {siso}, a weekly programmer, gas use should drop. Installing it myself.",
            "slots": {
                "siso": [D("2026-04-29"), D("2026-05-15")],
            },
        },
    },
    {
        "id": "S22", "tags": ["tech", "cloud"], "type": "note",
        "title": {"ru": "Облачное хранилище", "en": "Cloud storage"},
        "dup": {
            "ru": "Расширил облако до {gb} гигабайт {iso}, тариф {cost} в месяц. Автозагрузку фото включил.",
            "en": "Expanded the cloud to {gb} gigabytes on {iso}, the plan is {cost} a month. Enabled photo auto-upload.",
        },
        "slots": {
            "gb": [N("200"), N("500")],
            "iso": [D("2026-04-06"), D("2026-03-23")],
            "cost": [M("199", "рублей", "rubles", "one hundred ninety-nine"),
                     M("349", "рублей", "rubles", "three hundred forty-nine")],
        },
        "sib": {
            "ru": "Перенёс архив на другой провайдер {siso}: {sgb} гигабайт, миграция заняла {shrs} часов.",
            "en": "Moved the archive to another provider on {siso}: {sgb} gigabytes, the migration took {shrs} hours.",
            "slots": {
                "siso": [D("2026-04-25"), D("2026-05-08")],
                "sgb": [N("180"), N("240")],
                "shrs": [N("5"), N("7")],
            },
        },
    },
    {
        "id": "S23", "tags": ["sport", "running"], "type": "note",
        "title": {"ru": "Беговая тренировка", "en": "Running workout"},
        "dup": {
            "ru": "Пробежал {km} км {iso} за {min} минут, темп ровный, пульс до {hr}. Кроссовки отходили.",
            "en": "Ran {km} km on {iso} in {min} minutes, even pace, heart rate up to {hr}. The shoes held up.",
        },
        "slots": {
            "km": [N("10"), N("12")],
            "iso": [D("2026-04-11"), D("2026-03-28")],
            "min": [N("54"), N("61")],
            "hr": [N("158"), N("164")],
        },
        "sib": {
            "ru": "Зарегистрировался на полумарафон {siso}, старт в {sstart} утра, слоты уже подорожали.",
            "en": "Registered for the half marathon on {siso}, the start is at {sstart} am, the slots already cost more.",
            "slots": {
                "siso": [D("2026-05-24"), D("2026-06-14")],
                "sstart": [N("9"), N("11")],
            },
        },
    },
    {
        "id": "S24", "tags": ["home", "repair"], "type": "fact",
        "title": {"ru": "Электрик", "en": "Electrician visit"},
        "dup": {
            "ru": "Электрик приходил {iso}: заменил автомат на {amp} ампер, проверил щиток, {cost}. Розетки не греются.",
            "en": "The electrician came on {iso}: replaced the breaker with a {amp} amp one, checked the panel, {cost}. The outlets no longer heat up.",
        },
        "slots": {
            "iso": [D("2026-04-07"), D("2026-03-25")],
            "amp": [N("16"), N("25")],
            "cost": [M("2800", "рублей", "rubles", "two thousand eight hundred"),
                     M("3400", "рублей", "rubles", "three thousand four hundred")],
        },
        "sib": {
            "ru": "Планирую перенос розеток на кухне {siso}: {spts} точек, штробить будут под столешницу.",
            "en": "Planning to move the kitchen outlets on {siso}: {spts} points, they will chase the wall under the countertop.",
            "slots": {
                "siso": [D("2026-05-06"), D("2026-05-19")],
                "spts": [N("4"), N("6")],
            },
        },
    },
    {
        "id": "S25", "tags": ["food", "baking"], "type": "note",
        "title": {"ru": "Закваска", "en": "Sourdough starter"},
        "dup": {
            "ru": "Вывел закваску {iso}: мука {grams} граммов, вода 1:1, поднялась за {hrs} часов. Пахнет приятно.",
            "en": "Started the sourdough on {iso}: {grams} grams of flour, water 1:1, it rose in {hrs} hours. Smells pleasant.",
        },
        "slots": {
            "iso": [D("2026-04-03"), D("2026-03-21")],
            "grams": [N("50"), N("100")],
            "hrs": [N("6"), N("8")],
        },
        "sib": {
            "ru": "Испёк первый хлеб на закваске {siso}: {sgrams} граммов муки, пар держал {smin} минут. Корка хрустит.",
            "en": "Baked the first sourdough loaf on {siso}: {sgrams} grams of flour, kept steam for {smin} minutes. The crust crackles.",
            "slots": {
                "siso": [D("2026-04-12"), D("2026-04-26")],
                "sgrams": [N("450"), N("500")],
                "smin": [N("20"), N("25")],
            },
        },
    },
    {
        "id": "S26", "tags": ["family", "kindergarten"], "type": "fact",
        "title": {"ru": "Садик: взносы", "en": "Kindergarten fees"},
        "dup": {
            "ru": "Оплатил садик {iso}: {sum} за месяц, квитанция в приложении. Собрание {sday}-го числа.",
            "en": "Paid the kindergarten on {iso}: {sum} for the month, the receipt is in the app. The meeting is on the {sday}th.",
        },
        "slots": {
            "iso": [D("2026-04-05"), D("2026-03-06")],
            "sum": [M("3100", "рублей", "rubles", "three thousand one hundred"),
                    M("3450", "рублей", "rubles", "three thousand four hundred fifty")],
            "sday": [N("15"), N("20")],
        },
        "sib": {
            "ru": "Записал ребёнка на кружок в садике {siso}: логопедика, {sless} занятия в месяц, оплата вперёд.",
            "en": "Enrolled the kid in a kindergarten club on {siso}: speech therapy, {sless} sessions a month, paid upfront.",
            "slots": {
                "siso": [D("2026-04-14"), D("2026-04-28")],
                "sless": [N("8"), N("12")],
            },
        },
    },
    {
        "id": "S27", "tags": ["tech", "mobile"], "type": "fact",
        "title": {"ru": "Тариф телефона", "en": "Phone plan"},
        "dup": {
            "ru": "Сменил тариф {iso}: {gb} гигабайт и минуты без лимита, {cost} в месяц. Скорость не режут.",
            "en": "Switched the plan on {iso}: {gb} gigabytes and unlimited minutes, {cost} a month. No speed throttling.",
        },
        "slots": {
            "iso": [D("2026-04-01"), D("2026-03-17")],
            "gb": [N("30"), N("50")],
            "cost": [M("450", "рублей", "rubles", "four hundred fifty"),
                     M("600", "рублей", "rubles", "six hundred")],
        },
        "sib": {
            "ru": "Заказал eSIM для поездки {siso}: {sgb} гигабайт на {sdays} дней, активация по QR-коду.",
            "en": "Ordered an eSIM for the trip on {siso}: {sgb} gigabytes for {sdays} days, activation via QR code.",
            "slots": {
                "siso": [D("2026-05-18"), D("2026-06-01")],
                "sgb": [N("10"), N("20")],
                "sdays": [N("7"), N("14")],
            },
        },
    },
    {
        "id": "S28", "tags": ["health", "optics"], "type": "fact",
        "title": {"ru": "Новые очки", "en": "New glasses"},
        "dup": {
            "ru": "Забрал очки {iso}: диоптрии {od}/{os}, линзы с покрытием, оправа и линзы — {cost}. Мир резкий.",
            "en": "Picked up the glasses on {iso}: diopters {od}/{os}, coated lenses, frame plus lenses — {cost}. The world is sharp.",
        },
        "slots": {
            "iso": [D("2026-04-10"), D("2026-03-30")],
            "od": [X("-2.5"), X("-3.0")],
            "os": [X("-2.25"), X("-2.75")],
            "cost": [M("9800", "рублей", "rubles", "nine thousand eight hundred"),
                     M("11400", "рублей", "rubles", "eleven thousand four hundred")],
        },
        "sib": {
            "ru": "Записался на проверку зрения {siso}, подбор линз займёт {smin} минут, с собой старый рецепт.",
            "en": "Booked a vision test on {siso}, lens fitting takes {smin} minutes, bringing the old prescription.",
            "slots": {
                "siso": [D("2026-05-05"), D("2026-05-21")],
                "smin": [N("30"), N("40")],
            },
        },
    },
    {
        "id": "S29", "tags": ["home", "garden"], "type": "note",
        "title": {"ru": "Дача: деревья", "en": "Dacha: the trees"},
        "dup": {
            "ru": "Обрезал яблони {iso}: {cnt} дерева, спил {cm} см в диаметре максимум. Замазал садовым варом.",
            "en": "Pruned the apple trees on {iso}: {cnt} trees, the largest cut {cm} cm across. Sealed with garden pitch.",
        },
        "slots": {
            "iso": [D("2026-04-04"), D("2026-03-28")],
            "cnt": [N("3"), N("5")],
            "cm": [N("7"), N("9")],
        },
        "sib": {
            "ru": "Посадил саженцы {siso}: {scnt} штуки, сорт {sort}, полив через день. Колья поставил.",
            "en": "Planted saplings on {siso}: {scnt} of them, the {sort} variety, watering every other day. Stakes are in.",
            "slots": {
                "siso": [D("2026-04-26"), D("2026-05-03")],
                "scnt": [N("2"), N("4")],
                "sort": [X("Антоновка"), X("Грушовка")],
            },
        },
    },
    {
        "id": "S30", "tags": ["tech", "photos"], "type": "note",
        "title": {"ru": "Фотоархив", "en": "Photo archive"},
        "dup": {
            "ru": "Разобрал фотоархив {iso}: {cnt} снимков за год, {gb} гигабайт, дубли почистил. Папки по месяцам.",
            "en": "Sorted the photo archive on {iso}: {cnt} shots over the year, {gb} gigabytes, duplicates removed. Folders by month.",
        },
        "slots": {
            "iso": [D("2026-04-12"), D("2026-04-01")],
            "cnt": [N("4200"), N("3800")],
            "gb": [N("46"), N("39")],
        },
        "sib": {
            "ru": "Настроил резервную копию архива {siso}: синхронизация ночью, {smin} минут на инкремент.",
            "en": "Set up an archive backup on {siso}: sync at night, {smin} minutes per increment.",
            "slots": {
                "siso": [D("2026-04-22"), D("2026-05-06")],
                "smin": [N("12"), N("18")],
            },
        },
    },
]

# sib-family titles: the near-clone family carries its OWN title (distinct
# from the dup family of the same scaffold — real near-topic records do not
# share titles; sharing them pins cos >= 0.87 and starves the [0.78,0.84)
# bands).
TITLE_SIB: dict[str, dict[str, str]] = {
    "S01": {"ru": "Карта памяти", "en": "The memory card"},
    "S02": {"ru": "Заказ батареи", "en": "Battery order"},
    "S03": {"ru": "Отмена подписки", "en": "Cancelling the subscription"},
    "S04": {"ru": "Профгигиена", "en": "Professional cleaning"},
    "S05": {"ru": "Фикус: новые листья", "en": "Ficus: new leaves"},
    "S06": {"ru": "Колодки", "en": "Brake pads"},
    "S07": {"ru": "Программа жима", "en": "Bench program"},
    "S08": {"ru": "Таблица расходов", "en": "The expense sheet"},
    "S09": {"ru": "Новый чайник", "en": "The new kettle"},
    "S10": {"ru": "Mesh-точка", "en": "The mesh node"},
    "S11": {"ru": "Следующая книга", "en": "The next book"},
    "S12": {"ru": "Возврат билета", "en": "Ticket refund"},
    "S13": {"ru": "Диспансеризация", "en": "The check-up"},
    "S14": {"ru": "Чистка бэкап-диска", "en": "Backup disk cleanup"},
    "S15": {"ru": "Разговорный клуб", "en": "The speaking club"},
    "S16": {"ru": "Покрышки", "en": "New tires"},
    "S17": {"ru": "Аренда: компромисс", "en": "Rent: the compromise"},
    "S18": {"ru": "Кофемолка", "en": "The grinder"},
    "S19": {"ru": "Загранпаспорт", "en": "International passport"},
    "S20": {"ru": "Новый фильтр", "en": "The new filter"},
    "S21": {"ru": "Термостат", "en": "The thermostat"},
    "S22": {"ru": "Миграция архива", "en": "Archive migration"},
    "S23": {"ru": "Полумарафон", "en": "The half marathon"},
    "S24": {"ru": "Розетки на кухне", "en": "Kitchen outlets"},
    "S25": {"ru": "Первый хлеб", "en": "The first loaf"},
    "S26": {"ru": "Кружок в садике", "en": "The kindergarten club"},
    "S27": {"ru": "eSIM для поездки", "en": "Travel eSIM"},
    "S28": {"ru": "Проверка зрения", "en": "Vision test"},
    "S29": {"ru": "Саженцы", "en": "The saplings"},
    "S30": {"ru": "Резервная копия", "en": "The archive backup"},
}


# SIB2: rewritten sibling bodies (same slot ladders as the sib family —
# value disjointness from the dup family is inherited; the same topic, a
# different syntactic frame and lexicon). They are the extra pairing source
# for the low cos bands of the N-near tag-clone addition (the parallel
# template geometry bottoms out near 0.81; the exam near-clone floor is
# 0.801 with real-record phrasing, which SIB2 emulates).
SIB2: dict[str, dict[str, str]] = {
    "S01": {"ru": "Фотоархив с {siso} живёт на отдельной карте на {tgb} гигабайт — внутренняя память освобождена, съёмка не тормозит.",
            "en": "Since {siso} the photo archive lives on a separate {tgb} gigabyte card — internal storage freed, shooting no longer lags."},
    "S02": {"ru": "Батарея для {dev} ещё в пути, {days} дней доставки; старая выжимает {shrs} часа и гаснет на холоде.",
            "en": "The battery for the {dev} is still shipping, {days} days out; the old one squeezes out {shrs} hours and dies in the cold."},
    "S03": {"ru": "{srv} не продлеваю — решение от {siso}: за {scost} в год хватает бесплатного тарифа, всё выгружено заранее.",
            "en": "Not renewing {srv} — decided on {siso}: the free tier covers it for {scost} a year, everything was exported early."},
    "S04": {"ru": "Договорились о профилактике на приёме: чистка назначена на {siso}, {scost}, с полировкой и снимком.",
            "en": "Prevention was agreed at the visit: the cleaning is booked for {siso}, {scost}, with polishing and an X-ray."},
    "S05": {"ru": "После подкормки {siso} фикус выдал {sn} свежих листьев; землю не трогал, только свет добавил.",
            "en": "After the feeding on {siso} the ficus put out {sn} fresh leaves; the soil was left alone, only more light added."},
    "S06": {"ru": "Колодки на {scar} пришли вовремя, заменил {siso} — скрипа нет, {scost} с доставкой.",
            "en": "The pads for the {scar} arrived on time, fitted on {siso} — no more squeal, {scost} with delivery."},
    "S07": {"ru": "Жим перестроил {siso}: {sreps} подхода, рабочий вес {sw} кг, прибавка по субботам.",
            "en": "Rebuilt the bench plan on {siso}: {sreps} sets, working weight {sw} kg, adding weight on Saturdays."},
    "S08": {"ru": "Расходы веду в таблице с {siso}: выходит {ssum} за месяц, основная статья — доставка.",
            "en": "Expenses tracked in a sheet since {siso}: it comes to {ssum} a month, delivery is the biggest line."},
    "S09": {"ru": "Новый чайник с {siso} в деле: колба на {sl} л, фильтр держит накипь, на дачу старый уехал.",
            "en": "The new kettle from {siso} is in service: a {sl} liter jug, the scale filter works, the old one left for the dacha."},
    "S10": {"ru": "Покрытие расширилось после {siso}: меш-узел в коридоре добивает до балкона, гостевая сеть отдельно.",
            "en": "Coverage widened after {siso}: the hallway mesh node reaches the balcony now, guest network stays separate."},
    "S11": {"ru": "Клубная книга следующая выбрана {siso}: {spages} страниц, срок до июня, читаю с закладками.",
            "en": "The next club book was picked on {siso}: {spages} pages, due in June, reading with sticky notes."},
    "S12": {"ru": "Один билет сдан {siso}, поездка сдвинулась: вернули {scost} из {sfull}, новый куплю на другой день.",
            "en": "One ticket returned on {siso}, the trip moved: {scost} of {sfull} came back, the new one bought for another day."},
    "S13": {"ru": "На диспансеризацию записан {siso}: список расширенный, строго натощак, паспорт с собой.",
            "en": "Signed up for the full check-up on {siso}: extended test list, strictly fasting, ID in hand."},
    "S14": {"ru": "Старые снапшоты снесены {siso}: на бэкап-диске освободилось {sgb} гигабайт, ротация месячная.",
            "en": "Old snapshots were purged on {siso}: {sgb} gigabytes freed on the backup disk, monthly rotation now."},
    "S15": {"ru": "С {siso} хожу в разговорный клуб: среда, {smin} минут в Zoom, словарь перед сном.",
            "en": "Since {siso} I attend the speaking club: Wednesdays, {smin} minutes on Zoom, vocabulary before bed."},
    "S16": {"ru": "Покрышки {sinch} дюйма заказаны {siso}, летний протектор; старые оставлю как запаску.",
            "en": "{sinch} inch tires ordered on {siso}, summer tread; the old ones stay as spares."},
    "S17": {"ru": "Аренду подняли на {spct} процентов {siso}, сошлись на компромиссе: ремонт ляжет на хозяина.",
            "en": "The rent went up {spct} percent on {siso}, we struck a compromise: repairs fall to the landlord."},
    "S18": {"ru": "Кофемолка новая с {siso}: жернова, {sgrad} ступеней, работает тише прежней.",
            "en": "New grinder since {siso}: burrs, {sgrad} settings, runs quieter than the old one."},
    "S19": {"ru": "На загранпаспорт записан {siso}: биометрия на месте, изготовление {sweeks} недели.",
            "en": "The international passport appointment is set for {siso}: biometrics on site, {sweeks} weeks to process."},
    "S20": {"ru": "Фильтр заменил {siso}: {slph} л/ч, шум ниже, вода чище — рыбки подвижнее.",
            "en": "Swapped the filter on {siso}: {slph} l/h, less noise, clearer water — the fish are livelier."},
    "S21": {"ru": "Термостат заказан {siso}, недельный программатор; жду падения расхода газа после установки.",
            "en": "The thermostat was ordered on {siso}, a weekly programmer; expecting the gas bill to drop after installation."},
    "S22": {"ru": "Архив переехал к другому провайдеру {siso}: {sgb} гигабайт, дорога заняла {shrs} часов.",
            "en": "The archive moved to another provider on {siso}: {sgb} gigabytes, the trip took {shrs} hours."},
    "S23": {"ru": "Слот на полумарафон занял {siso}: старт в {sstart} утра, цены на слоты уже выросли.",
            "en": "Took a half marathon slot on {siso}: the start is {sstart} am, slot prices have already climbed."},
    "S24": {"ru": "Перенос розеток запланирован на {siso}: {spts} точек, штробление под столешницей.",
            "en": "The outlet move is scheduled for {siso}: {spts} points, chasing the wall under the countertop."},
    "S25": {"ru": "Первый каравай из закваски вышел {siso}: муки ушло {sgrams} граммов, пар держал {smin} минут.",
            "en": "The first sourdough loaf came out on {siso}: {sgrams} grams of flour went in, steam held for {smin} minutes."},
    "S26": {"ru": "Кружок логопедики в садике оформлен {siso}: {sless} занятия в месяц, оплата вперёд.",
            "en": "The kindergarten speech club is signed up on {siso}: {sless} sessions a month, paid upfront."},
    "S27": {"ru": "eSIM под поездку куплена {siso}: {sgb} гигабайт на {sdays} дней, активация по QR-коду.",
            "en": "A travel eSIM was bought on {siso}: {sgb} gigabytes for {sdays} days, activation via QR code."},
    "S28": {"ru": "Зрение проверю {siso}: подбор линз займёт {smin} минут, старый рецепт прихвачу.",
            "en": "The vision test is on {siso}: lens fitting takes {smin} minutes, bringing the old prescription along."},
    "S29": {"ru": "Саженцы в земле с {siso}: {scnt} штуки сорта {sort}, полив через день, колья стоят.",
            "en": "The saplings are in the ground since {siso}: {scnt} of the {sort} variety, watering every other day, stakes up."},
    "S30": {"ru": "Ночная синхронизация архива запущена {siso}: инкремент занимает {smin} минут, папки по месяцам.",
            "en": "The nightly archive sync started on {siso}: an increment takes {smin} minutes, folders by month."},
}


# global tails (faithful, fact-free; modulate cos via length/divergence)
TAILS: dict[str, dict[str, str]] = {
    "none": {"ru": "", "en": ""},
    "a": {
        "ru": " Проверял дважды — держится.",
        "en": " Checked twice — it holds.",
    },
    "b": {
        "ru": " Оставляю заметку, чтобы не искать заново при следующем апдейте, деталей здесь достаточно.",
        "en": " Keeping this note so I do not have to look it up again on the next update; the details here are enough.",
    },
    "c": {
        "ru": " Детали и контекст лежат в проекте; вернусь к записи, когда доберусь до этой части плана, и сверю с тем, что закладывал изначально.",
        "en": " Details and context live in the project; I will get back to this entry when I reach that part of the plan and cross-check it against what I originally intended.",
    },
    "d": {
        "ru": " Запись для себя: если ситуация изменится, допишу вторую часть и отмечу, чем новый расклад отличается от прежнего, чтобы к возвращению к теме не восстанавливать ход мыслей по памяти.",
        "en": " A note to myself: should things change, I will add the second half and mark how the new setup differs from the old one, so that coming back to this topic later does not mean reconstructing the whole line of thought from scratch.",
    },
}

# divergent SHORT tails: same register, DIFFERENT text — used pairwise (a on
# one side, a2 on the other) so the shared surface fraction drops on BOTH
# sides while body_len_delta stays within the A2.3 cap.
TAILS["a2"] = {
    "ru": " Дополню заметку, когда будет ответ.",
    "en": " This note gets an update once there is news.",
}
TAILS["a3"] = {
    "ru": " Пока держу это на контроле.",
    "en": " Keeping this one on my radar for now.",
}
# ── rendering ────────────────────────────────────────────────────────────────


def render_template(tmpl: str, values: dict[str, dict], lang: str, family: str) -> str:
    out = tmpl
    for name, val in values.items():
        if family == "T":
            piece = val["ru_t"] if lang == "ru" else val["en_t"]
        else:
            piece = val["ru_v"] if lang == "ru" else val["en_v"]
        out = out.replace("{" + name + "}", piece)
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def swap_sentences(text: str) -> str:
    """Legacy class-4 lever: swap the first two sentences (order-only change;
    tokens, slot map and lengths survive)."""
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) < 2:
        return text
    return " ".join([parts[1], parts[0], *parts[2:]])


def ladder_combos(ladders: dict[str, list[dict]], cap: int) -> list[dict[str, dict]]:
    """Deterministic slot combinations (sorted names, capped product)."""
    names = sorted(ladders)
    combos: list[dict[str, dict]] = []
    idx = [0] * len(names)
    total = 1
    for n in names:
        total *= len(ladders[n])
    for _ in range(min(total, cap)):
        combos.append({n: ladders[n][idx[i]] for i, n in enumerate(names)})
        for i in reversed(range(len(names))):
            idx[i] += 1
            if idx[i] < len(ladders[names[i]]):
                break
            idx[i] = 0
    return combos


def slot_tokens(values: dict[str, dict]) -> list[str]:
    return [v["t"] for v in values.values()]


# ── minting ──────────────────────────────────────────────────────────────────


class Minter:
    def __init__(self, runlog: list[dict], embed_side, pin: str, scanner: Any,
                 occupied: set, prefix: str):
        self.runlog = runlog
        self.embed_side = embed_side
        self.pin = pin
        self.scanner = scanner
        self.occupied = occupied
        self.prefix = prefix
        self.counters: Counter = Counter()
        self.hygiene_hits = 0

    def hygiene(self, s: dict, where: str) -> None:
        hits = self.scanner.scan(s.get("title") or "", s.get("body") or "",
                                 s.get("tags") or [])
        if hits:
            self.hygiene_hits += 1
            refuse(f"hygiene hit on minted side {where}: {hits}")

    def mint(self, cls: str, label: str, stratum: str, generator: str, source: str,
             rec: dict, cand: dict, scaffold: str | None = None,
             extra: dict | None = None, check_occupied: bool = True,
             allow_dup_key: bool = False) -> dict:
        self.counters[cls] += 1
        pid = f"{self.prefix}-{cls}-{self.counters[cls]:04d}"
        va = self.embed_side(rec)
        vb = self.embed_side(cand)
        sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
        pair_obj = {"record": rec, "candidate": cand, "similarity": sim}
        row = {
            "pair_id": pid,
            "label": label,
            "stratum": stratum,
            "record": rec,
            "candidate": cand,
            "similarity": sim,
            "vec_a": va,
            "vec_b": vb,
            "pair_sha256": pair_sha256(pair_obj),
            "generator": generator,
            "seed": SEED,
            "embedder_pin": self.pin,
            "source": source,
        }
        if scaffold is not None:
            row["scaffold"] = scaffold
        if extra:
            row.update(extra)
        key = pair_key(row)
        if allow_dup_key:
            # the swapped twin of the SAME pair: identical pair-key by design
            # (v44r2 convention — the key was validated at the first mint)
            pass
        elif check_occupied and key in self.occupied:
            refuse(f"minted pair collides with an occupied pair-key: {pid}")
        self.occupied.add(key)
        self.hygiene(rec, pid + ":record")
        self.hygiene(cand, pid + ":candidate")
        return row


# ── candidate generation: add-1 translated-dup (T/V) ─────────────────────────


def gen_tver_candidates(runlog, minter: Minter, embed_side) -> list[dict]:
    """A2.1-A2.2: slot-bookkeeping translated-dup candidates, T + V families,
    both orientations, tails modulating cos. Per-pair asserts measured here;
    selection happens per bin afterwards."""
    inst_cap = 16
    raw: list[dict] = []
    for scaf in SCAFFOLDS:
        for values in ladder_combos(scaf["slots"], inst_cap):
            tokens = slot_tokens(values)
            slot_map_sha = hashlib.sha256(
                canonical_json(dict(sorted((n, v["t"]) for n, v in values.items())))
                .encode("utf-8")).hexdigest()[:16]
            for tail_id in ("none", "a", "b", "c", "d"):
                for family in ("T", "V"):
                    # order-only levers (taxonomy class 5, dup-preserving):
                    # reorder the en side, the ru side, or neither
                    for re_side in ("none", "en", "ru"):
                        sides = {}
                        ok = True
                        for lang in ("ru", "en"):
                            body = render_template(scaf["dup"][lang], values, lang, family)
                            body += TAILS[tail_id][lang]
                            if re_side == lang:
                                body = swap_sentences(body)
                            title = scaf["title"][lang]
                            side = {"title": title, "body": body, "tags": list(scaf["tags"]),
                                    "language": lang, "record_type": scaf["type"]}
                            if family == "T":
                                full = (title + "\n" + body).lower()
                                for tok in tokens:
                                    # case-insensitive containment: the token
                                    # canonical form survives cosmetic case
                                    # (taxonomy class 2); matches the ch3c
                                    # measurement which lowercases
                                    if tok.lower() not in full:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            sides[lang] = side
                        if not ok:
                            continue
                        rec, cand = sides["ru"], sides["en"]
                        ch3c = ch_containment(rec, cand, 3)  # assert for T, monitor for V
                        va, vb = embed_side(rec), embed_side(cand)
                        sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
                        bld = body_len_delta(rec, cand)
                        base = {
                            "scaf": scaf["id"], "family": family, "tail": tail_id,
                            "reorder": re_side,
                            "slot_map_sha": slot_map_sha, "sim": sim, "bld": bld,
                            "ch3c": round(ch3c, 6), "rec": rec, "cand": cand,
                        }
                        if family == "T":
                            if bld > BLD_T_MAX:
                                continue
                            if ch3c < 0.01:
                                continue
                        else:
                            if bld > BLD_V_MAX:
                                continue
                        raw.append(base)
    log(runlog, "tver_candidates", total=len(raw),
        t=sum(1 for c in raw if c["family"] == "T"),
        v=sum(1 for c in raw if c["family"] == "V"),
        in_band=sum(1 for c in raw if 0.55 <= c["sim"] <= 0.86))
    return raw


def select_tver_bins(runlog, cands: list[dict], sibling_bins: dict[str, int],
                     minter: Minter, embed_side) -> tuple[list[dict], dict]:
    """A2.2 selection (the runs-6/7 architecture that filled all three bins
    with continuity, minus its cap bug): per-bin greedy spacing selection over
    scaffolds, class-global scaffold cap of 3 PAIRS (6 rows = 4.1% <= 5%),
    thin-bin deficits closed from the dose reserve above the bin (v4.4-r2
    case_run — normalization-invisible: ch3c/bld/tags/slot tokens invariant,
    only the measured cos moves; precedent: addendum-1 dose rows), then gap
    backfill. Both orientations minted per pair (v44r2 convention)."""
    edges = ((0.55, 0.65, "[0.55,0.65)"), (0.65, 0.75, "[0.65,0.75)"),
             (0.75, 0.8600001, "[0.75,0.86]"))
    sel_rows: list[dict] = []
    report = {}
    per_scaffold: Counter = Counter()
    # class-global scaffold cap in PAIRS: 6 rows <= 5% of the class quota
    quota_total = sum(max(sibling_bins.get(n, 0), BIN_MIN_K)
                      for _l, _h, n in edges)
    scaf_cap_pairs = max(1, int(SCAFFOLD_MAX_SHARE * quota_total) // 2)
    used_scaf: Counter = Counter()  # pair counts, class-global

    def gaps_of(coses):
        cs = sorted(coses)
        return [(b - a, a, b) for a, b in zip(cs, cs[1:])]

    dose_used_total = 0
    for lo, hi, name in edges:
        sib = sibling_bins.get(name, 0)
        quota_rows = max(sib, BIN_MIN_K)
        target_pairs = quota_rows // 2
        pool = sorted((c for c in cands if lo <= c["sim"] < hi),
                      key=lambda c: (c["sim"], c["scaf"], c["family"], c["tail"]))
        # greedy spacing selection across scaffolds
        spacing = min(0.003, (hi - lo) / max(target_pairs, 1))
        chosen: list[dict] = []
        last = -1.0
        for c in pool:
            if len(chosen) >= target_pairs:
                break
            if used_scaf[c["scaf"]] >= scaf_cap_pairs:
                continue
            if not chosen or c["sim"] - last >= spacing or hi - c["sim"] < 0.005:
                chosen.append(c)
                used_scaf[c["scaf"]] += 1
                last = c["sim"]
        # dose-fill: quota deficit or continuity holes are closed from the
        # reserve just above the bin (ladder pre-computed lazily, placements
        # bridge the largest gaps first)
        grid: list[tuple[float, float, dict]] | None = None
        used_reserve: set = set()

        def dose_place(gi: float, gj: float) -> dict | None:
            nonlocal grid
            if grid is None:
                reserve = sorted(
                    (c for c in cands if hi <= c["sim"] <= hi + 0.07
                     and id(c) not in {id(x) for x in chosen}),
                    key=lambda c: (c["scaf"], c["sim"], c["family"], c["tail"]))
                built: list[tuple[float, float, dict]] = []
                for c in reserve[:3000]:
                    for d in DOSE_LADDER:
                        da = dict(c["rec"])
                        da["body"] = case_run(c["rec"]["body"], d)
                        sim = round(min(1.0, max(0.0, dot(
                            embed_side(da), embed_side(c["cand"])))), 6)
                        if lo <= sim < hi:
                            built.append((sim, d, c))
                built.sort(key=lambda g: (g[0], g[2]["scaf"]))
                grid = built
            best = None
            best_key = None
            for sim, d, c in grid:
                if not (gi < sim < gj):
                    continue
                if id(c) in used_reserve:
                    continue
                if used_scaf[c["scaf"]] >= scaf_cap_pairs:
                    continue
                key = (abs(sim - (gi + gj) / 2), c["scaf"])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (sim, d, c)
            if best is None:
                return None
            sim, d, c = best
            used_reserve.add(id(c))
            used_scaf[c["scaf"]] += 1
            da = dict(c["rec"])
            da["body"] = case_run(c["rec"]["body"], d)
            return {
                "scaf": c["scaf"], "family": c["family"], "tail": c["tail"],
                "reorder": c.get("reorder", False),
                "slot_map_sha": c["slot_map_sha"], "sim": sim, "bld": c["bld"],
                "ch3c": c["ch3c"], "rec": da, "cand": c["cand"], "dose": d,
            }

        guard = 0
        while guard < 96:
            guard += 1
            coses = sorted(c["sim"] for c in chosen)
            gaps = gaps_of(coses)
            worst = max(gaps, default=(hi - lo, lo, hi))
            short = len(chosen) < target_pairs
            if not short and worst[0] <= CONTINUITY_DELTA + 1e-9:
                break
            if short:
                gi, gj = (worst[1], worst[2]) if worst[2] > worst[1] else (lo, hi)
                if gj - gi < 1e-6:
                    gi, gj = lo, hi
            else:
                gi, gj = worst[1], worst[2]
                if gj - gi < 1e-6:
                    break
            placed = dose_place(gi, gj)
            if placed is None and (gi, gj) != (lo, hi):
                placed = dose_place(lo, hi)
            if placed is None:
                refuse(f"tver bin {name}: dose fill cannot cover deficit/gap "
                       f"{(round(gi, 4), round(gj, 4))}; STOP")
            chosen.append(placed)
            dose_used_total += 1
        coses = sorted(c["sim"] for c in chosen)
        max_gap = max((b - a for a, b in zip(coses, coses[1:])), default=0.0)
        rows_n = len(chosen) * 2
        report[name] = {
            "quota_rows": quota_rows, "pairs_selected": len(chosen),
            "rows": rows_n, "max_gap": round(max_gap, 6),
            "k_ok": rows_n >= BIN_MIN_K, "parity_ok": rows_n >= sib,
            "continuity_ok": max_gap <= CONTINUITY_DELTA + 1e-9,
            "ok": (rows_n >= quota_rows and rows_n >= sib
                   and rows_n >= BIN_MIN_K
                   and max_gap <= CONTINUITY_DELTA + 1e-9),
        }
        if not report[name]["ok"]:
            refuse(f"tver bin {name} asserts failed: {report[name]}; STOP")
        for c in chosen:
            per_scaffold[c["scaf"]] += 2
            dose = c.get("dose")
            gen = (f"v45-regenerate slot-bookkeeping ({c['family']}-family, "
                   f"tail={c['tail']}, reorder={c.get('reorder', False)})")
            if dose is not None:
                gen += f" + v44r2 case-dose placement (dose={dose})"
            first = True
            for rec, cand in ((c["rec"], c["cand"]), (c["cand"], c["rec"])):
                extra = {"family": c["family"], "slot_map_sha": c["slot_map_sha"],
                         "tail": c["tail"], "reorder": c.get("reorder", False)}
                if dose is not None:
                    extra["dose"] = dose
                    extra["dose_placed"] = True
                sel_rows.append(minter.mint(
                    "TDUP" if c["family"] == "T" else "TDUPV",
                    "duplicate", "translated-dup", gen,
                    "addendum2-translated-dup", deepcopy(rec), deepcopy(cand),
                    scaffold=c["scaf"], extra=extra,
                    allow_dup_key=not first,
                ))
                first = False
    log(runlog, "tver_selection", bins=report,
        dose_placed_pairs_total=dose_used_total)
    # class-level scaffold anti-collapse (A2.2)
    total = len(sel_rows)
    n_scaf = len(per_scaffold)
    max_share = max(per_scaffold.values()) / total if total else 0.0
    log(runlog, "tver_scaffolds", scaffolds=n_scaf,
        max_share=round(max_share, 4), min_required=SCAFFOLD_MIN,
        max_allowed_share=SCAFFOLD_MAX_SHARE)
    if n_scaf < SCAFFOLD_MIN:
        refuse(f"tver scaffold anti-collapse: {n_scaf} scaffolds < {SCAFFOLD_MIN}; STOP")
    if max_share > SCAFFOLD_MAX_SHARE:
        refuse(f"tver scaffold anti-collapse: max share {max_share:.4f} > "
               f"{SCAFFOLD_MAX_SHARE}; STOP")
    return sel_rows, {"bins": report, "scaffolds": n_scaf,
                      "max_scaffold_share": round(max_share, 4), "rows": total}


# ── add-2: class-7 same-language (single-fact edit, short body) ──────────────


def collect_side_universes(runlog) -> tuple[dict[str, dict], dict[str, set]]:
    """The widened clean side pool (pool + dataset-v4 + v44-r2 train real
    sides) under the v44r2 exclusion recipe (holdout/gate/burned sides), plus
    the flag sets used for the side-overlap slice at holdout carve."""
    hs: set = set()
    for name in ("v43-holdout/holdout.jsonl", "dataset-v4-holdout/holdout.jsonl",
                 "b2/holdout.jsonl"):
        for r in corpus_rows(BASE / name):
            for s in (r["record"], r["candidate"]):
                hs.add(text_hash(s["title"] + "\n" + s["body"]))
                hs.add(text_hash(s["body"]))
                hs.add(text_hash(s["title"]))
    gate_sides: set = set()
    R4 = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
    R5 = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
    for gp in (
        R4 / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4 / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ):
        if gp.is_file():
            for row in corpus_rows(gp):
                for key in ("text_src", "text_dst", "text"):
                    if row.get(key):
                        gate_sides.add(text_hash(row[key]))
    burned: set = set()
    for bp in (
        R4 / "data/embed-r4/corpus/train.jsonl",
        R4 / "data/embed-r4/corpus/val.jsonl",
        R5 / "data/embed-r5/corpus/train.jsonl",
        R5 / "data/embed-r5/corpus/val.jsonl",
        R5 / "data/embed-r7/corpus/train.jsonl",
        R5 / "data/embed-r7/corpus/val.jsonl",
    ):
        if bp.is_file():
            for row in corpus_rows(bp):
                for side_key in ("record", "candidate"):
                    s = row.get(side_key)
                    if s:
                        burned.add(text_hash(s["title"] + "\n" + s["body"]))
                        burned.add(text_hash(s["body"]))
                for key in ("text", "text_src", "text_dst"):
                    if row.get(key):
                        burned.add(text_hash(row[key]))

    sides: dict[str, dict] = {}
    for src in (POOL_JSONL, BASE / "dataset-v4/train.jsonl",
                BASE / "v43-localbuild/train.jsonl", V44R2_DIR / "train.jsonl"):
        for r in corpus_rows(src):
            for s in (r["record"], r["candidate"]):
                sides.setdefault(text_hash(s["title"] + "\n" + s["body"]), s)
    clean: dict[str, dict] = {}
    skipped: Counter = Counter()
    for h, s in sides.items():
        probes = (h, text_hash(s["body"]), text_hash(s["title"]))
        if any(p in gate_sides for p in probes):
            skipped["gate-side"] += 1
            continue
        if any(p in burned for p in probes):
            skipped["burned-side"] += 1
            continue
        if any(p in hs for p in probes):
            skipped["holdout-side"] += 1
            continue
        clean[h] = s
    log(runlog, "side_universe", distinct=len(sides), clean=len(clean),
        skipped=dict(skipped))
    flags = {"holdout_side_overlap": hs, "gate_side_overlap": gate_sides,
             "burned_side_overlap": burned}
    return clean, flags


def gen_c7_candidates(runlog, minter: Minter, embed_side,
                      clean: dict[str, dict]) -> list[dict]:
    """A2.3 class-7 same-language: one fact-token digit cipher on clean short
    real sides; language detected and DECLARED per row (the v4.4 silent lm=0
    drift is structurally impossible); cells keyed by ch5_jaccard."""
    cands: list[dict] = []
    skipped: Counter = Counter()
    for h, s in sorted(clean.items()):
        body = s.get("body") or ""
        title = s.get("title") or ""
        if not body or len(body) > C7_BODY_MAX:
            skipped["long-or-empty"] += 1
            continue
        lang = detect_lang(body)
        if lang not in ("ru", "en"):
            skipped[f"lang-{lang}"] += 1
            continue
        spans = fact_token_spans(body)
        if not spans:
            skipped["no-fact-token"] += 1
            continue
        rec = {k: s[k] for k in SIDE_KEYS if k in s}
        rec["language"] = lang
        va = embed_side(rec)
        # candidate edits:
        #   - digit cipher per fact-token span;
        #   - single fact-WORD shifts (month/weekday);
        #   - COMBINED date-fact edit: a digit span + a month/weekday word in
        #     the same clause = ONE date fact rewritten to another date (the
        #     v4.2 exam mechanism, e.g. "5 октября" -> "7 ноября").
        edits: list[tuple[str, str]] = [
            (cipher_span(body, span), f"digit:{body[span[0]:span[1]]}")
            for span in spans
        ]
        edits += [(new, f"word:{w}") for new, w in word_fact_edits(body, lang)]
        fact_words = [(w, body.find(w)) for _, words in WORD_FACTS[lang].items()
                      for w in words if body.find(w) >= 0]
        for span in spans:
            for w, wpos in fact_words:
                if abs(wpos - span[0]) <= 40:
                    combined = cipher_span(body, span)
                    words = next(tbl for _kind, tbl in WORD_FACTS[lang].items()
                                 if w in tbl)
                    i = words.index(w)
                    start = combined.find(w)
                    if start >= 0:
                        end = start + len(w)
                        before_ok = start == 0 or not combined[start - 1].isalnum()
                        after_ok = end == len(combined) or not combined[end].isalnum()
                        if before_ok and after_ok:
                            combined = combined[:start] + words[(i + 1) % len(words)] + combined[end:]
                            edits.append((combined, f"date:{body[span[0]:span[1]]}+{w}"))
                    break
        for edited, kind in edits:
            cand = dict(rec)
            cand["body"] = edited
            j = ch_jaccard(rec, cand, 5)
            if not (0.80 <= j < 0.95):
                skipped["ch5j-outside-cells"] += 1
                continue
            sim = round(min(1.0, max(0.0, dot(va, embed_side(cand)))), 6)
            if sim < C7_COS_MIN:
                skipped["cos-below-0.985"] += 1
                continue
            cands.append({
                "sha": h, "rec": rec, "cand": cand, "sim": sim, "ch5j": j,
                "lang_group": lang, "edit_kind": kind,
            })
    log(runlog, "c7_candidates", total=len(cands), skipped=dict(skipped),
        cells={f"[{lo:.2f},{hi:.2f})": sum(1 for c in cands if lo <= c["ch5j"] < hi)
               for lo, hi in C7_CH5J_CELLS})
    return cands


def select_c7_cells(runlog, cands: list[dict], cell_pos: dict[str, int],
                    minter: Minter) -> tuple[list[dict], dict]:
    sel_rows: list[dict] = []
    report = {}
    occupied_skipped = 0
    for lo, hi in C7_CH5J_CELLS:
        name = f"[{lo:.2f},{hi:.2f})"
        quota = max(BIN_MIN_K, cell_pos.get(name, 0))
        pool = sorted((c for c in cands if lo <= c["ch5j"] < hi),
                      key=lambda c: (abs(c["ch5j"] - (lo + hi) / 2), c["sha"]))
        # a cipher candidate on a base already T3-minted by v4.4-r2 can
        # reproduce the SAME pair (identical deterministic cipher) — skip it
        # instead of refusing the build
        fresh: list[dict] = []
        for c in pool:
            probe = {"record": c["rec"], "candidate": c["cand"]}
            if pair_key(probe) in minter.occupied:
                occupied_skipped += 1
                continue
            fresh.append(c)
        pool = fresh
        if len(pool) < quota // 2:
            refuse(f"c7 cell {name}: pool {len(pool)} pairs < {quota // 2}; STOP")
        # spread across the cell: alternate from center outwards
        chosen: list[dict] = []
        used: set = set()
        # deterministic spread: sort by ch5j and pick evenly
        stride = max(1, len(pool) // (quota // 2 + 1))
        picked = 0
        i = 0
        while picked < quota // 2 and i < len(pool):
            c = pool[i]
            if id(c) not in used:
                used.add(id(c))
                chosen.append(c)
                picked += 1
            i += stride
        for c in pool:
            if picked >= quota // 2:
                break
            if id(c) not in used:
                used.add(id(c))
                chosen.append(c)
                picked += 1
        rows_n = len(chosen) * 2
        report[name] = {
            "quota_rows": quota, "pairs_selected": len(chosen), "rows": rows_n,
            "parity_ok": rows_n >= cell_pos.get(name, 0), "k_ok": rows_n >= BIN_MIN_K,
            "ok": rows_n >= quota,
        }
        if not report[name]["ok"]:
            refuse(f"c7 cell {name} asserts failed: {report[name]}; STOP")
        for c in chosen:
            first = True
            for rec, cand in ((c["rec"], c["cand"]), (c["cand"], c["rec"])):
                sel_rows.append(minter.mint(
                    "C7", "not-duplicate", "N-fact-edit",
                    "v45-class7-same-lang (single-fact digit cipher, short body)",
                    "addendum2-class7-same-lang", deepcopy(rec), deepcopy(cand),
                    scaffold=c["sha"],
                    extra={"lang_group": c["lang_group"],
                           "ch5j_cell": round(c["ch5j"], 6)},
                    allow_dup_key=not first,
                ))
                first = False
    # audit assert: every new class-7 row declares its lang group
    bad = [r["pair_id"] for r in sel_rows if not r.get("lang_group")]
    if bad:
        refuse(f"c7 audit: {len(bad)} rows without lang_group; STOP")
    bad_lm = [r["pair_id"] for r in sel_rows if lang_match(r["record"], r["candidate"]) != 1.0]
    if bad_lm:
        refuse(f"c7 audit: {len(bad_lm)} rows with lang_match != 1.0; STOP")
    log(runlog, "c7_selection", cells=report, occupied_skipped=occupied_skipped)
    return sel_rows, {"cells": report, "rows": len(sel_rows),
                      "occupied_skipped": occupied_skipped}


# ── add-3: N-near tag-clone (same-language near-topic, same scaffold) ────────


def gen_nnc_candidates(runlog, minter: Minter, embed_side) -> list[dict]:
    """A2.3 N-near tag-clone: dup-family body x sib-family body of the SAME
    scaffold, same language, cloned tags, disjoint slot tokens (asserted)."""
    cands: list[dict] = []
    digit_re = re.compile(r"\d+")
    monitor_overlap = Counter()
    for scaf in SCAFFOLDS:
        dup_insts = ladder_combos(scaf["slots"], 6)
        sib_insts = ladder_combos(scaf["sib"]["slots"], 6)
        dup_tokens = {v["t"] for vals in scaf["slots"].values() for v in vals}
        sib_tokens = {v["t"] for vals in scaf["sib"]["slots"].values() for v in vals}
        if dup_tokens & sib_tokens:
            refuse(f"scaffold {scaf['id']}: dup/sib slot tokens overlap "
                   f"{sorted(dup_tokens & sib_tokens)}; STOP")
        for dvals in dup_insts:
            for svals in sib_insts:
                for lang in ("ru", "en"):
                    for tmpl_b, tail_id in (
                            (t, tt)
                            for t in (scaf["sib"][lang], SIB2[scaf["id"]][lang])
                            for tt in ("none", "a", "b", "c", "d")):
                        # structural variants: plain / sibling-side reorder /
                        # sibling-side split / record-side split / both split;
                        # asymmetric-tail variants (negatives only — labels
                        # are by construction, and a one-sided short tail is
                        # legitimate not-dup content; bld stays <= 60):
                        #   ta-a / ta-b: short tail appended on ONE side only
                        #   pre-a / pre-b: short tail prepended on ONE side
                        for variant in ("plain", "reorder", "split", "split-a",
                                        "split-both", "ta-a", "ta-b",
                                        "pre-a", "pre-b", "da-a", "da-b",
                                        "dc-a", "dc-b"):
                            if variant.startswith(("ta-", "pre-")) and tail_id != "none":
                                continue  # asymmetric variants pair with no tail
                            body_a = render_template(scaf["dup"][lang], dvals, lang, "T")
                            body_a += TAILS[tail_id][lang]
                            body_b = render_template(tmpl_b, svals, lang, "T")
                            body_b += TAILS[tail_id][lang]
                            if variant == "reorder":
                                # legacy class-4 lever: order-only divergence
                                body_b = swap_sentences(body_b)
                            elif variant == "split":
                                body_b = split_compound(body_b)
                            elif variant == "split-a":
                                body_a = split_compound(body_a)
                            elif variant == "split-both":
                                body_a = split_compound(body_a)
                                body_b = split_compound(body_b)
                            elif variant == "ta-a":
                                body_a += TAILS["a"][lang]
                            elif variant == "ta-b":
                                body_b += TAILS["a"][lang]
                            elif variant == "pre-a":
                                body_a = TAILS["a"][lang].lstrip() + " " + body_a
                            elif variant == "pre-b":
                                body_b = TAILS["a"][lang].lstrip() + " " + body_b
                            elif variant == "da-a":
                                body_a += TAILS["a"][lang]
                                body_b += TAILS["a2"][lang]
                            elif variant == "da-b":
                                body_a += TAILS["a2"][lang]
                                body_b += TAILS["a"][lang]
                            elif variant == "dc-a":
                                body_a += TAILS["a"][lang]
                                body_b += TAILS["a3"][lang]
                            elif variant == "dc-b":
                                body_a += TAILS["a3"][lang]
                                body_b += TAILS["a"][lang]
                            da = set(digit_re.findall(body_a))
                            db = set(digit_re.findall(body_b))
                            if da & db:
                                # monitor-only: incidental digit runs (e.g. the
                                # shared calendar year inside ISO dates) are NOT
                                # slot tokens; the binding assert is the slot-token
                                # disjointness above (A2.2 "ни одного общего
                                # слот-токена"), which holds by construction.
                                monitor_overlap["digit-run-overlap"] += 1
                            rec = {"title": scaf["title"][lang],
                                   "body": body_a,
                                   "tags": list(scaf["tags"]), "language": lang,
                                   "record_type": scaf["type"]}
                            cand = {"title": TITLE_SIB[scaf["id"]][lang],
                                    "body": body_b,
                                    "tags": list(scaf["tags"]), "language": lang,
                                    "record_type": scaf["type"]}
                            bld = body_len_delta(rec, cand)
                            if bld > NN_BLD_MAX:
                                continue
                            va, vb = embed_side(rec), embed_side(cand)
                            sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
                            if not (NN_LOW <= sim <= NN_HIGH):
                                continue
                            cands.append({
                                "scaf": scaf["id"], "lang": lang, "tail": tail_id,
                                "variant": variant,
                                "rec": rec, "cand": cand, "sim": sim, "bld": bld,
                            })
    log(runlog, "nnc_candidates", total=len(cands),
        digit_run_overlap_skipped=monitor_overlap["digit-run-overlap"],
        bands={f"{c:.3f}": sum(1 for x in cands if abs(x["sim"] - c) <= NN_BAND + 1e-12)
               for c in [round(NN_LOW + NN_BAND + k * NN_BAND, 3)
                         for k in range(int((NN_HIGH - NN_LOW) / NN_BAND))]})
    return cands


def select_nnc_bands(runlog, cands: list[dict], pos_total: int,
                     pos_by_band: dict[str, int], minter: Minter,
                     embed_side) -> tuple[list[dict], dict]:
    """Global density selection: bands are ±0.005 and OVERLAP, so pairs are
    selected once (global spacing) and every band's K is verified against the
    FULL selected set (rows of the class in the band), never against a
    band-exclusive pick.

    The same-language near-clone geometry bottoms out at cos ≈ 0.806 across
    every construction (template-parallel, rewritten SIB2, asymmetric tails,
    real store pairs floor at 0.829) while the ratified bands start at 0.78 —
    the deficit is closed by the LINE'S OWN dose lever (v4.4-r2 case_run,
    normalization-invisible: ch5j/bld/tags/label unchanged, only the measured
    cos moves; v44r2 precedent: dose-placed rows closing the [0.97,0.985) bin).
    Candidates from the natural reserve are dose-placed into the thin bands
    [0.78, 0.81) via the deterministic ladder, and every dose is recorded on
    the row."""
    centers = [round(NN_LOW + NN_BAND + k * NN_BAND, 3)
               for k in range(int(round((NN_HIGH - NN_LOW) / NN_BAND)))]
    # target density: >= 30 rows (= 15 pairs) inside EVERY 0.01 window
    # -> one pair per ~0.00066 of cos span
    need_pairs_total = max(int((NN_HIGH - NN_LOW) / 0.00066) + 1,
                           pos_total, BIN_MIN_K)
    spacing = (NN_HIGH - NN_LOW) / need_pairs_total
    pool = sorted(cands, key=lambda c: (c["sim"], c["scaf"], c["lang"], c["tail"]))
    if len(pool) < need_pairs_total:
        refuse(f"nnc candidate pool {len(pool)} pairs < density need "
               f"{need_pairs_total}; STOP")

    # 0.810 included: its window [0.805,0.815] loses its natural mass to
    # the dose reserve below — covered by dose placement like the rest
    # thin-band fill via the tver-proven GRID: precompute every (sim, dose)
    # landing in [0.78, 0.815) from the lowest naturals, then place at ~0.00066
    # density so every ±0.005 band intersects >= 15 placements (= 30 rows)
    thin_centers = [c for c in centers if c <= 0.810 + 1e-9]
    # ONE class-global scaffold counter shared by dose grid, natural walk and
    # top-ups (a separate dose counter would double the per-scaffold share)
    per_scaf_pairs: Counter = Counter()
    # 220 (not the observed ~239): the top-up pass grows the class after
    # the cap is fixed — the conservative estimate keeps max share < 5%
    scaf_cap_pairs = 12  # 24 rows; the class-size floor below keeps <= 5%
    dose_placed: list[dict] = []
    used_ids: set = set()
    # upper reach 0.825: the dose reserve consumes the lowest naturals,
    # so band 0.820's window [0.815,0.825] must be grid-covered too
    thin_lo, thin_hi = NN_LOW, 0.825
    if thin_centers:
        # reserve from naturals >= 0.84 ONLY: the natural walk needs the
        # 0.81-0.84 mass for its own bands; heavy doses from >= 0.84 reach the
        # whole thin region [0.775, 0.84) (dose-1.0 drop is ~0.064)
        reserve = sorted((c for c in pool if 0.84 <= c["sim"] <= 0.90),
                         key=lambda c: (c["sim"], c["scaf"]))
        reserve = reserve[:4000]  # bounded deterministic subset
        grid: list[tuple[float, float, dict]] = []
        for c in reserve:
            for d in DOSE_LADDER:
                da = dict(c["rec"])
                da["body"] = case_run(c["rec"]["body"], d)
                sim = round(min(1.0, max(0.0, dot(embed_side(da),
                                                  embed_side(c["cand"])))), 6)
                if thin_lo <= sim < thin_hi:
                    grid.append((sim, d, c))
        grid.sort(key=lambda g: (g[0], g[2]["scaf"]))
        step = 0.00066
        last = -1.0
        target_n = int((thin_hi - thin_lo) / step) + 1
        for sim, d, c in grid:
            if len(dose_placed) >= target_n:
                break
            if id(c) in used_ids or per_scaf_pairs[c["scaf"]] >= scaf_cap_pairs:
                continue
            if not dose_placed or sim - last >= step:
                da = dict(c["rec"])
                da["body"] = case_run(c["rec"]["body"], d)
                dose_placed.append({
                    "scaf": c["scaf"], "lang": c["lang"], "tail": "none",
                    "variant": c["variant"], "rec": da, "cand": c["cand"],
                    "sim": sim, "bld": c["bld"], "dose": d,
                })
                used_ids.add(id(c))
                per_scaf_pairs[c["scaf"]] += 1
                last = sim
        if len(dose_placed) < target_n:
            # target_n is the internal density heuristic; the BINDING asserts
            # are the per-band K >= 30 checks below (they refuse if a band is
            # short) — a near-target placement count proceeds to them
            log(runlog, "nnc_dose_grid_soft_target",
                placed=len(dose_placed), target=target_n,
                note="below heuristic density; binding per-band K checks follow")
    log(runlog, "nnc_dose_placement", pairs=len(dose_placed),
        thin_bands=[f"{c:.3f}" for c in thin_centers],
        lever="v44r2 case_run (normalization-invisible; ch5j/bld/tags/label "
              "invariant; precedent: addendum-1 dose rows)")

    # the natural walk must cover [0.81, 0.92] at the SAME per-window density:
    # its budget and spacing are computed AFTER the dose placements took their
    # share of the total
    naturals_target = int((NN_HIGH - 0.81) / 0.00066) + 1
    need_pairs_total = len(dose_placed) + naturals_target
    spacing = (NN_HIGH - 0.81) / naturals_target
    natural_pool = [c for c in pool
                    if c["sim"] > 0.81 + 1e-9 and id(c) not in used_ids]
    chosen: list[dict] = []
    last = -1.0
    for c in natural_pool:
        if len(chosen) >= naturals_target:
            break
        if per_scaf_pairs[c["scaf"]] >= scaf_cap_pairs:
            continue
        if not chosen or c["sim"] - last >= spacing:
            chosen.append(c)
            per_scaf_pairs[c["scaf"]] += 1
            last = c["sim"]
    chosen.extend(dose_placed)
    # per-band K top-up: the boundary/half-windows at the range edges need
    # extra mass beyond the uniform walk spacing; top up from unused naturals
    # inside each failing window (additions only densify — continuity can only
    # improve; the binding verification follows)
    chosen_ids = {id(c) for c in chosen}
    for c0 in centers:
        name = f"{c0:.3f}"
        need_rows = max(BIN_MIN_K, pos_by_band.get(name, 0))
        guard = 0
        while guard < 64:
            guard += 1
            rows_n = sum(2 for c in chosen
                         if abs(c["sim"] - c0) <= NN_BAND + 1e-12)
            if rows_n >= need_rows:
                break
            lo_w, hi_w = c0 - NN_BAND, min(NN_HIGH, c0 + NN_BAND)
            best = None
            best_key = None
            for c in natural_pool:
                if id(c) in chosen_ids:
                    continue
                if per_scaf_pairs[c["scaf"]] >= scaf_cap_pairs:
                    continue
                if not (lo_w <= c["sim"] <= hi_w):
                    continue
                key = (abs(c["sim"] - c0), c["scaf"])
                if best_key is None or key < best_key:
                    best_key = key
                    best = c
            if best is None:
                break
            chosen.append(best)
            chosen_ids.add(id(best))
            per_scaf_pairs[best["scaf"]] += 1
    # class-size floor: at the 12-pair scaffold cap the 5% share holds iff the
    # class carries >= 240 pairs (24/480 = exactly 5%); the recipe floors are
    # minima, so the extra near-clone mass is legal and feeds the g3 cell
    filler_target = max(need_pairs_total, 240)
    guard = 0
    while len(chosen) < filler_target and guard < 40:
        guard += 1
        placed_any = False
        for c0 in centers:
            if len(chosen) >= filler_target:
                break
            lo_w, hi_w = max(NN_LOW, c0 - NN_BAND), min(NN_HIGH, c0 + NN_BAND)
            best = None
            best_key = None
            for c in natural_pool:
                if id(c) in chosen_ids:
                    continue
                if per_scaf_pairs[c["scaf"]] >= scaf_cap_pairs:
                    continue
                if not (lo_w <= c["sim"] <= hi_w):
                    continue
                key = (per_scaf_pairs[c["scaf"]], abs(c["sim"] - c0), c["scaf"])
                if best_key is None or key < best_key:
                    best_key = key
                    best = c
            if best is not None:
                chosen.append(best)
                chosen_ids.add(id(best))
                per_scaf_pairs[best["scaf"]] += 1
                placed_any = True
        if not placed_any:
            break
    # verify per-band K on the full selected set (overlapping bands)
    sel_rows: list[dict] = []
    report = {}
    for c0 in centers:
        name = f"{c0:.3f}"
        in_band = [c for c in chosen if abs(c["sim"] - c0) <= NN_BAND + 1e-12]
        rows_n = len(in_band) * 2
        report[name] = {
            "rows": rows_n, "k_ok": rows_n >= BIN_MIN_K,
            "parity_ok": rows_n >= pos_by_band.get(name, 0),
            "ok": rows_n >= BIN_MIN_K and rows_n >= pos_by_band.get(name, 0),
        }
        if not report[name]["ok"]:
            refuse(f"nnc band {name} asserts failed: {report[name]}; STOP")
    # continuity: max gap between consecutive selected cos <= delta
    sims = sorted(c["sim"] for c in chosen)
    max_gap = max((b - a for a, b in zip(sims, sims[1:])), default=0.0)
    if max_gap > CONTINUITY_DELTA + 1e-9:
        refuse(f"nnc continuity: max gap {max_gap:.6f} > {CONTINUITY_DELTA}; STOP")
    total_pos = sum(pos_by_band.values())
    total_rows = len(chosen) * 2
    if total_rows < total_pos:
        refuse(f"nnc total parity failed: {total_rows} rows < positives {total_pos}; STOP")
    # mint with band attribution = nearest center
    for c in chosen:
        c0 = min(centers, key=lambda x: abs(x - c["sim"]))
        name = f"{c0:.3f}"
        dose = c.get("dose")
        gen = "v45-nnear-tag-clone (same-scaffold near-topic, same-lang)"
        if dose is not None:
            gen += f" + v44r2 case-dose placement (dose={dose})"
        first = True
        for rec, cand in ((c["rec"], c["cand"]), (c["cand"], c["rec"])):
            extra = {"lang_group": c["lang"], "band": name}
            if dose is not None:
                extra["dose"] = dose
                extra["dose_placed"] = True
            sel_rows.append(minter.mint(
                "NNC", "not-duplicate", "N-near", gen,
                "addendum2-nnear-tag-clone", deepcopy(rec), deepcopy(cand),
                scaffold=c["scaf"] + "|nnc", extra=extra,
                allow_dup_key=not first,
            ))
            first = False
    # scaffold anti-collapse on the minted class
    per_scaf: Counter = Counter(r.get("scaffold") or "?" for r in sel_rows)
    max_share = max(per_scaf.values()) / total_rows
    if len(per_scaf) < SCAFFOLD_MIN:
        refuse(f"nnc scaffold anti-collapse: {len(per_scaf)} scaffolds < {SCAFFOLD_MIN}; STOP")
    if max_share > SCAFFOLD_MAX_SHARE:
        refuse(f"nnc scaffold anti-collapse: max share {max_share:.4f} > {SCAFFOLD_MAX_SHARE}; STOP")
    log(runlog, "nnc_selection", bands=report, total_rows=total_rows,
        total_parity_ok=total_rows >= total_pos, max_gap=round(max_gap, 6),
        scaffolds=len(per_scaf), max_scaffold_share=round(max_share, 4),
        density_target_pairs=need_pairs_total)
    return sel_rows, {"bands": report, "rows": total_rows,
                      "max_gap": round(max_gap, 6),
                      "scaffolds": len(per_scaf),
                      "max_scaffold_share": round(max_share, 4)}


# ── add-4: strata A (real re-saves from the probe snapshot) ──────────────────

CREATED_DELTA_MAX_S = 600  # A2.4 letter: "differs by minutes"
STRA_A_HOLDOUT_CLUSTERS = 3  # A2.4: 2-3 FULL clusters
STRA_A_TRAIN_MIN_PAIRS = 30


def load_strata_a(runlog, minter: Minter, snapshot_path: Path) -> dict:
    """Build strata A rows from the fingerprinted probe snapshot; allocate
    full clusters to holdout, the rest to train; assert cluster integrity,
    train mass, norm-equality, envelope, zero graph vector, corpus cos."""
    if not snapshot_path.is_file():
        refuse(f"strata A snapshot missing: {snapshot_path} — run "
               "scripts/b2_store_probe_v45.py first; STOP")
    snap = corpus_rows(snapshot_path)
    probe = json.loads(
        (snapshot_path.parent / "strata-a-probe.json").read_text(encoding="utf-8"))
    snap_fp = hashlib.blake2b(
        "\n".join(f"{p['a_store_id']}--{p['b_store_id']} {p['store_cos']:.9f}"
                  for p in snap).encode("utf-8") + b"\n",
        digest_size=32).hexdigest()
    if snap_fp != probe.get("snapshot_fingerprint"):
        refuse("strata A snapshot fingerprint mismatch vs probe numbers; STOP")
    log(runlog, "strata_a_snapshot", fp=snap_fp, pairs=len(snap),
        census_pairs=probe.get("census_anchor", {}).get("pairs"),
        growth_pairs=probe.get("growth_vs_census_pairs"),
        store_snapshot=probe.get("store_snapshot"))

    clusters: dict[str, list[dict]] = {}
    for p in snap:
        clusters.setdefault(p["cluster_key"], []).append(p)
    # deterministic holdout allocation: largest clusters first, tie by key
    order = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    holdout_keys = {k for k, _ in order[:STRA_A_HOLDOUT_CLUSTERS]}

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    quarantine = 0
    content_dedup = 0
    quarantine_log: list[dict] = []
    for key, members in sorted(clusters.items()):
        for p in members:
            a, b = p["record"], p["candidate"]
            # re-measure the silver predicate on the snapshot (trust, verify)
            if norm1eq(a["body"], b["body"]) is False:
                refuse(f"strata A pair {p['a_store_id'][:8]} body not norm-equal; STOP")
            delta = p.get("created_delta_s")
            if delta is None or delta > CREATED_DELTA_MAX_S:
                refuse(f"strata A pair {p['a_store_id'][:8]} created delta {delta}; STOP")
            sides_ok = a.get("language") == b.get("language") and \
                set(a.get("tags") or []) == set(b.get("tags") or []) and \
                a.get("record_type") == b.get("record_type")
            if not sides_ok:
                refuse(f"strata A pair {p['a_store_id'][:8]} envelope mismatch; STOP")
            extra = {
                "id_a": p["a_store_id"], "id_b": p["b_store_id"],
                "cluster_key": key, "store_cos": p["store_cos"],
            }
            target = holdout_rows if key in holdout_keys else train_rows
            # orientation: record=older side, candidate=younger (created_at);
            # exported sides follow the base convention: created_at is NOT part
            # of a corpus side (it rides the snapshot and the ids metadata)
            ra, rb = (a, b) if (a.get("created_at") or "") <= (b.get("created_at") or "") else (b, a)
            ra = {k: v for k, v in ra.items() if k != "created_at"}
            rb = {k: v for k, v in rb.items() if k != "created_at"}
            # content dedup BEFORE minting: every member of a re-save cluster
            # is byte-identical, so all pairs of one cluster share ONE pair-key
            # (dedup-before-splits). A repeated key = the same content pair
            # again — counted separately from the A2.5.3 quarantine (which is
            # for UNRESOLVED sides, not for identical content)
            ra_key = side_sha(ra)
            rb_key = side_sha(rb)
            dup_key = tuple(sorted((ra_key, rb_key)))
            if dup_key in {r["_pair_key"] for r in train_rows + holdout_rows}:
                content_dedup += 1
                continue
            try:
                row = minter.mint(
                    "RA", "duplicate", "A-resave",
                    "store-probe-snapshot (real re-save, whitelist §8.1 envelope)",
                    "addendum2-strata-A", deepcopy(ra), deepcopy(rb),
                    scaffold="strata-a:" + key, extra=extra,
                )
            except GateRefusal as exc:
                quarantine += 1
                quarantine_log.append({
                    "cluster": key[:16], "pair": f"{p['a_store_id'][:8]}--{p['b_store_id'][:8]}",
                    "reason": str(exc)[:200],
                })
                continue
            if row["similarity"] < 0.999:
                refuse(f"strata A corpus cos {row['similarity']} < 0.999 on "
                       f"{p['a_store_id'][:8]}; STOP")
            row["_pair_key"] = dup_key
            target.append(row)
    train_pairs = len(train_rows)
    hold_pairs = len(holdout_rows)
    hold_clusters = len({r["cluster_key"] for r in holdout_rows})
    ok = (train_pairs >= STRA_A_TRAIN_MIN_PAIRS and 2 <= hold_clusters <= 3)
    for r in train_rows + holdout_rows:
        r.pop("_pair_key", None)
    report = {
        "snapshot_pairs": len(snap), "clusters": len(clusters),
        "distinct_content_pairs": len(snap) - content_dedup,
        "content_dedup_collapsed": content_dedup,
        "train_pairs": train_pairs, "holdout_pairs": hold_pairs,
        "holdout_clusters": hold_clusters, "quarantine": quarantine,
        "k_ok": train_pairs >= STRA_A_TRAIN_MIN_PAIRS,
        "clusters_ok": 2 <= hold_clusters <= 3, "ok": ok,
    }
    log(runlog, "strata_a_split", **report)
    if quarantine:
        log(runlog, "strata_a_quarantine", count=quarantine,
            note="unresolvable sides counted, not silently dropped (A2.5.3)")
    if not ok:
        refuse(f"strata A split asserts failed: {report}; STOP")
    # cluster integrity: no cluster straddles train/holdout
    t_keys = {r["cluster_key"] for r in train_rows}
    h_keys = {r["cluster_key"] for r in holdout_rows}
    if t_keys & h_keys:
        refuse("strata A cluster integrity violated; STOP")
    report["quarantine_log"] = quarantine_log
    return {"train": train_rows, "holdout": holdout_rows, "probe": probe,
            "snapshot_fp": snap_fp, "report": report}


def norm1eq(a: str, b: str) -> bool:
    return " ".join((a or "").lower().split()) == " ".join((b or "").lower().split())


# ── evidence columns: frozen at build + fresh-sidecar agreement ──────────────


def freeze_evidence_columns(runlog, rows: list[dict], out: Path,
                            snapshot_adjacency_ok: dict[str, tuple[int, int]]) -> dict:
    """Run the frozen sidecar over the new material (read-only store), freeze
    the GRAPH_FEATURES columns onto every row, and assert agreement with the
    independent probe-snapshot adjacency for the store-native rows."""
    cat = out / "sidecar-catalog"
    side_out = out / "graph-sidecar-v45.jsonl"
    new_jsonl = out / "new-material-precarve.jsonl"
    write_jsonl(new_jsonl, rows)
    import subprocess
    py = sys.executable
    r1 = subprocess.run(
        [py, str(REPO_ROOT / "scripts" / "b2_rev1n_sidecar_input.py"),
         "--rows", str(new_jsonl), "--out-dir", str(cat)],
        capture_output=True, text=True)
    if r1.returncode != 0:
        refuse(f"sidecar input converter failed: {r1.stderr[-400:]}")
    r2 = subprocess.run(
        [py, str(REPO_ROOT / "scripts" / "b2_graph_sidecar.py"),
         "--dataset-dir", str(cat), "--store-uri", STORE_URI,
         "--out", str(side_out)],
        capture_output=True, text=True)
    if r2.returncode != 0:
        refuse(f"sidecar run failed: {r2.stderr[-400:]}")
    side = {}
    for r in corpus_rows(side_out):
        side[r["pair_id"]] = r
    quarantine = 0
    unresolved_native = 0
    synthetic_empty = 0
    for row in rows:
        sr = side.get(row["pair_id"])
        if sr is None:
            refuse(f"sidecar output missing row {row['pair_id']}; STOP")
        kinds = sr.get("edge_kinds") or []
        row["edge_ab"] = int(sr.get("edge_ab") or 0)
        row["edge_kind_supersedes"] = int(1 if "supersedes" in kinds else 0)
        row["common_neighbors"] = int(sr.get("common_neighbors") or 0)
        row["neighbor_jaccard"] = float(sr.get("neighbor_jaccard") or 0.0)
        c2c = sr.get("cos_to_common_max")
        row["cos_to_common_max"] = float(c2c) if c2c is not None else (
            0.0 if row["common_neighbors"] == 0 else None)
        if row["cos_to_common_max"] is None:
            refuse(f"cos_to_common_max unresolvable with cn>0 on {row['pair_id']}; STOP")
        row["has_graph_evidence"] = bool(
            row["edge_ab"] == 1 or row["common_neighbors"] >= 1)
        native = bool(row.get("id_a"))
        if native:
            a_id, b_id = row["id_a"], row["id_b"]
            row["a_store_id"], row["b_store_id"] = a_id, b_id
            if not sr.get("a_store_id") or not sr.get("b_store_id"):
                unresolved_native += 1
                continue
            # independent agreement: probe-snapshot adjacency
            deg_a, cn_ab = snapshot_adjacency_ok.get(a_id, (0, 0))
            if row["edge_ab"] != 0 or row["common_neighbors"] != 0:
                refuse(f"strata A row {row['pair_id']} carries a non-zero graph "
                       "vector at sidecar time; A2.4 violated; STOP")
            if row["has_graph_evidence"]:
                refuse(f"strata A row {row['pair_id']} evidence flag true; STOP")
        else:
            if sr.get("a_store_id") or sr.get("b_store_id"):
                refuse(f"synthetic row {row['pair_id']} unexpectedly resolved "
                       "against the store; STOP")
            if row["has_graph_evidence"]:
                refuse(f"synthetic row {row['pair_id']} unexpectedly has evidence; STOP")
            synthetic_empty += 1
    quarantine = unresolved_native
    log(runlog, "evidence_columns_frozen", rows=len(rows),
        sidecar_fp=(json.loads(r2.stdout or "{}").get("fingerprint")
                    if r2.stdout else None),
        unresolved_native=unresolved_native, quarantine=quarantine,
        synthetic_legitimately_empty=synthetic_empty,
        evidence_rows=sum(1 for r in rows if r["has_graph_evidence"]))
    return {"quarantine": quarantine, "unresolved_native": unresolved_native,
            "sidecar_rows": len(side)}


# ── stage 0: base integrity ──────────────────────────────────────────────────

STORE_URI: str = ""  # set by main(); used by the sidecar subprocess


def stage0_base(runlog) -> dict:
    rows = corpus_rows(V44R2_DIR / "train.jsonl")
    fp = corpus_fingerprint(manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in rows]))
    if fp != BASE_TRAIN_FP:
        refuse(f"base train fp {fp} != sealed {BASE_TRAIN_FP}; the v4.4-r2 base "
               "is not intact; STOP")
    labels = {}
    for r in corpus_rows(V44R2_DIR / "labels-train.jsonl"):
        labels[r["pair_id"]] = r["label"]
    lab_fp = labels_fingerprint(labels)
    if lab_fp != BASE_LABELS_FP:
        refuse(f"base labels fp {lab_fp} != sealed {BASE_LABELS_FP}; STOP")
    if len(rows) != BASE_ROWS:
        refuse(f"base rows {len(rows)} != {BASE_ROWS}; STOP")
    for r in rows:
        if labels.get(r["pair_id"]) != r["label"]:
            refuse(f"base row {r['pair_id']} label drift vs labels-train; STOP")
    # structure pins vs the census anchors
    tsib = [r for r in rows if r.get("stratum") == "translated-sibling"]
    ptr = [r for r in rows if r.get("stratum") == "P-trans"]
    razor = [r for r in rows if r.get("stratum") in ("N-fact-edit", "N-para-notdup", "P-cosmetic")]
    bins = Counter()
    for r in tsib:
        c = r["similarity"]
        if 0.55 <= c < 0.65:
            bins["[0.55,0.65)"] += 1
        elif 0.65 <= c < 0.75:
            bins["[0.65,0.75)"] += 1
        elif 0.75 <= c <= 0.86:
            bins["[0.75,0.86]"] += 1
    if dict(bins) != CENSUS_SIBLING_BINS:
        refuse(f"sibling bin pins drifted: {dict(bins)} != {CENSUS_SIBLING_BINS}; STOP")
    if len(tsib) != 144 or len(ptr) != 40:
        refuse(f"sibling/P-trans structure drifted: {len(tsib)}/{len(ptr)}; STOP")
    # the sealed v4.4 holdout: fingerprint only (ids/assignment, no content use)
    disk_hold_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in corpus_rows(V44_HOLDOUT)]))
    if disk_hold_fp != V44_HOLDOUT_FP:
        refuse("sealed v4.4 holdout fingerprint changed; STOP")
    log(runlog, "base_integrity", train_fp=fp, labels_fp=lab_fp, rows=len(rows),
        sibling_bins=dict(bins), p_trans=len(ptr), razor_block=len(razor),
        v44_holdout_fp=disk_hold_fp, untouched=True)
    return {"rows": rows, "labels": labels, "sibling_bins": dict(bins),
            "p_trans": ptr, "razor_block": razor}


def historical_surfaces(runlog) -> dict:
    surfaces: dict[str, set] = {}
    for name in ("v43-holdout/holdout.jsonl", "dataset-v4-holdout/holdout.jsonl",
                 "b2/holdout.jsonl"):
        surfaces[name] = {pair_key(r) for r in corpus_rows(BASE / name)}
    surfaces["burned-translated"] = {
        pair_key(r) for r in corpus_rows(BASE / "v43-localbuild/train.jsonl")
        if str(r.get("part", "")).startswith("translated")
        or str(r.get("pair_id", "")).startswith(("V43-TDUP", "V43-TSIB"))}
    surfaces["pool-pairs"] = {pair_key(r) for r in corpus_rows(POOL_JSONL)}
    # v4.4 sealed holdout: pair identity only (pair_id, pair_sha256), no sides
    v44_pairs = {(r["pair_id"], r["pair_sha256"])
                 for r in corpus_rows(V44_HOLDOUT)}
    log(runlog, "historical_surfaces", **{k: len(v) for k, v in surfaces.items()},
        v44_holdout_pair_ids=len(v44_pairs))
    return {"pair_keys": surfaces, "v44_pairs": v44_pairs}


# ── holdout v4.5 carve + seal ────────────────────────────────────────────────


def carve_holdout_v45(runlog, synthetic: list[dict], strata_a: dict,
                      flags: dict[str, set]) -> tuple[list[dict], list[dict], dict]:
    cells: dict[tuple, list[dict]] = {}
    for r in synthetic:
        cells.setdefault((r["label"], r["has_graph_evidence"], r["stratum"]), []).append(r)
    rng = random.Random(SEED)
    drawn_ids: set = set()
    draw_report: dict[str, dict] = {}
    # draw by PAIR GROUP (both orientations of one pair travel together —
    # an orientation twin split across train/holdout would leak the pair)
    pair_group: dict[tuple[str, str], list[dict]] = {}
    for r in synthetic:
        pair_group.setdefault(pair_key(r), []).append(r)
    for cell in sorted(cells, key=lambda c: tuple(str(x) for x in c)):
        rows = cells[cell]
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            groups.setdefault(pair_key(r), []).append(r)
        n = len(rows)
        n_groups = len(groups)
        n_h = round(HOLDOUT_SHARE * n)
        if n >= 10 and n_h == 0:
            n_h = 1  # representation floor for non-trivial cells
        # pairs to draw (round to groups; at least the floor)
        gids = sorted(groups)
        n_h_groups = min(n_groups, max(1, round(HOLDOUT_SHARE * n_groups))) \
            if n_groups else 0
        if n_h and n_h_groups == 0:
            n_h_groups = 1
        pick_groups = set(rng.sample(gids, n_h_groups)) if n_h_groups else set()
        cell_drawn = 0
        for g in pick_groups:
            for r in groups[g]:
                drawn_ids.add(r["pair_id"])
                cell_drawn += 1
        draw_report[f"{cell[0]}|evidence={cell[1]}|{cell[2]}"] = {
            "cell_n": n, "drawn": cell_drawn, "groups_drawn": len(pick_groups)}
    # mint-top quotas: declared, empty in the new material (disclosed)
    mint_top_present = sum(1 for r in synthetic if r.get("stratum") == "mint-top")
    draw_report["mint-top-evidence"] = {"cell_n": 0, "drawn": 0,
                                        "note": "no mint-top rows exist in the "
                                                "new v4.5 material (disclosed)"}
    draw_report["mint-top-zero"] = {"cell_n": 0, "drawn": 0,
                                    "note": "same disclosure"}
    holdout = [r for r in synthetic if r["pair_id"] in drawn_ids]
    holdout.extend(strata_a["holdout"])
    # side-overlap flags: slice at carve
    flag_counts: Counter = Counter()
    for r in synthetic + strata_a["holdout"] + strata_a["train"]:
        hashes = {
            text_hash(r["record"]["title"] + "\n" + r["record"]["body"]),
            text_hash(r["record"]["body"]),
            text_hash(r["candidate"]["title"] + "\n" + r["candidate"]["body"]),
            text_hash(r["candidate"]["body"]),
        }
        for flag, fset in flags.items():
            if hashes & fset:
                r[flag] = True
                flag_counts[flag] += 1
            else:
                r[flag] = False
    hold_flags = {f: sum(1 for r in holdout if r.get(f)) for f in flags}
    log(runlog, "holdout_carve", draw=draw_report, flags_total=dict(flag_counts),
        flags_in_holdout=hold_flags, mint_top_present=mint_top_present,
        evidence_reading="edge_ab=1 OR common_neighbors>=1 (A2.5.3, frozen)")
    # A2.5.4 asks for the flag SLICE at carve (reported, kept on the rows for
    # the gate runner), not a zero-refusal. TRUE holdout-surface overlap is a
    # leak and refuses; gate/burned overlaps are UNLABELED embed-corpus /
    # gate-set text (no label leakage — the embed corpora never carried dup
    # labels): store-native strata A rows legitimately carry them, disclosed.
    if hold_flags.get("holdout_side_overlap"):
        refuse(f"holdout rows overlap sealed holdout surfaces: {hold_flags}; STOP")
    log(runlog, "side_overlap_disclosure",
        gate_in_holdout=hold_flags.get("gate_side_overlap", 0),
        burned_in_holdout=hold_flags.get("burned_side_overlap", 0),
        note="unlabeled embed-corpus/gate text shared by store-native rows; "
             "no label leakage; rows keep their flag columns for the gate runner")
    return holdout, [r for r in synthetic if r["pair_id"] not in drawn_ids], {
        "draw": draw_report, "flags_total": dict(flag_counts),
        "flags_in_holdout": hold_flags}


# ── assembly + manifest ──────────────────────────────────────────────────────


def envelope_asserts(runlog, rows: list[dict], scope: str) -> None:
    """A2.1(3): tag_jaccard = 1.0, type_match = 1.0, tag_count_delta = 0."""
    for r in rows:
        if tag_jaccard(r["record"], r["candidate"]) != 1.0:
            refuse(f"{scope} {r['pair_id']}: tag_jaccard != 1.0; STOP")
        if (r["record"].get("record_type") or "") != (r["candidate"].get("record_type") or ""):
            refuse(f"{scope} {r['pair_id']}: record_type mismatch; STOP")
        if len(r["record"].get("tags") or []) != len(r["candidate"].get("tags") or []):
            refuse(f"{scope} {r['pair_id']}: tag_count_delta != 0; STOP")
    log(runlog, "envelope_asserts", scope=scope, rows=len(rows),
        tag_jaccard="1.0", type_match="1.0", tag_count_delta="0")


def disjointness_asserts(runlog, new_rows: list[dict], surfaces: dict) -> None:
    violations: Counter = Counter()
    for r in new_rows:
        ck = pair_key(r)
        for name, keys in surfaces["pair_keys"].items():
            if ck in keys:
                violations[name] += 1
        if (r["pair_id"], r["pair_sha256"]) in surfaces["v44_pairs"]:
            violations["v44-holdout"] += 1
    if violations:
        refuse(f"pair-level disjointness violated: {dict(violations)}; STOP")
    log(runlog, "disjointness_new_material", checked=len(new_rows), violations=0,
        surfaces=list(surfaces["pair_keys"]) + ["v44-holdout(pairs)"])


def build_manifest(runlog, base_info, train, holdout, new_rows, strata_a,
                   tver_report, c7_report, nnc_report, carve, evis, out: Path,
                   args) -> dict:
    entries = [(r["pair_id"], r["pair_sha256"]) for r in train]
    train_fp = corpus_fingerprint(manifest_bytes(entries))
    labels = {r["pair_id"]: r["label"] for r in train}
    hold_entries = [(r["pair_id"], r["pair_sha256"]) for r in holdout]
    hold_fp = corpus_fingerprint(manifest_bytes(hold_entries))
    ids = sorted(r["pair_id"] for r in holdout)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    strata = Counter(r["stratum"] for r in train)
    strata_h = Counter(r["stratum"] for r in holdout)
    lab = Counter(r["label"] for r in train)
    lab_h = Counter(r["label"] for r in holdout)
    base_strata = Counter(r["stratum"] for r in base_info["rows"])
    manifest = {
        "corpus": "b2-graph-v4.5 (ADDENDUM-2 maintenance build, rev 1.n+1 corpus)",
        "law": [
            "docs/specs/b2-graph-prereg-v3.md (ACTIVATED, owner 22.10; §4 frozen)",
            "docs/specs/b2-graph-prereg-v3-ADDENDUM-1.md (taxonomy §A, §C, §D)",
            "docs/specs/b2-graph-prereg-v3-ADDENDUM-2.md (ACTIVATED 22.10, binding)",
            "docs/experiments/b2-m11-design-package.md (ratified §3/§4)",
            "docs/specs/labeling-policy-b2.md §8 (policy v1.1)",
        ],
        "base": {
            "corpus": "v4.4-r2 train", "rows": len(base_info["rows"]),
            "fingerprint": BASE_TRAIN_FP, "integrity": "asserted pre-build",
            "inherited_strata_rebuilt": False,
        },
        "p_trans_replacement": {
            "removed_rows": 40, "stratum": "P-trans",
            "reason": "defective P-trans families of rev 1.n replaced by "
                      "regeneration (g3 generation conflict, g3-analysis §2a/§3; "
                      "build directive 2026-10-22)",
        },
        "addendum2_additions": {
            "translated_dup_regenerate": {
                "bins": tver_report, "scaffold_source": "SCAFFOLDS (frozen library, "
                "30 authored templates; probes g1b drawn from the same library)",
            },
            "class7_same_language": {"cells": c7_report,
                "lang_group_declared_per_row": True},
            "nnear_tag_clone": {"bands": nnc_report,
                "source": "same-scaffold dup-family x sib-family, same-language; "
                          "real x real pool yield measured 3 pairs (insufficient) "
                          "— authored fallback per generation discipline"},
            "strata_A": {
                "snapshot_fingerprint": strata_a["snapshot_fp"],
                "probe_numbers": {k: strata_a["probe"][k] for k in (
                    "store_snapshot", "silver_dup_pairs", "clusters",
                    "cluster_size_histogram", "growth_vs_census_pairs",
                    "created_delta_s_max", "store_cos_range")},
                "split": strata_a["report"],
                "exclusion": "§D-б continuity NOT applied to strata A "
                             "(single-point band cos≡1.0 — documented A2.4 exception)",
            },
        },
        "holdout_v45": {
            "share_of_new_material": HOLDOUT_SHARE,
            "stratification": "label x evidence-flag x stratum (A2.5.1)",
            "draw": carve["draw"],
            "evidence_reading": "edge_ab=1 OR common_neighbors>=1 (A2.5.3, one "
                                "reading, columns frozen at build)",
            "sidecar_agreement": evis,
            "quarantine": strata_a["report"]["quarantine"],
            "cos_to_common_max": "frozen in export columns (A2.5.5)",
            "side_overlap_flags": carve["flags_total"],
            "representation_assert": "§D extended where applicable; dup-with-"
                                     "evidence material absent (census 0/3654) "
                                     "— vacuous, disclosed (A2.5.6)",
        },
        "counts": {
            "train_rows": len(train),
            "train_labels": dict(lab),
            "train_strata": dict(sorted(strata.items())),
            "train_dup_share": round(lab["duplicate"] / len(train), 6),
            "holdout_rows": len(holdout),
            "holdout_labels": dict(lab_h),
            "holdout_strata": dict(sorted(strata_h.items())),
            "new_rows_total": len(new_rows),
            "composition_vs_v44r2": {
                s: {"v44r2": base_strata.get(s, 0), "v4.5_train": strata.get(s, 0)}
                for s in sorted(set(base_strata) | set(strata))
            },
        },
        "fingerprints": {
            "scheme": "manifest(pair_id pair_sha256 sorted) -> blake2b-256 (data-contract §5)",
            "base_v44r2_train": BASE_TRAIN_FP,
            "train_v45": train_fp,
            "labels_train_v45": labels_fingerprint(labels),
            "holdout_v45_seal": hold_fp,
            "holdout_v45_ids_sha256": ids_sha,
            "strata_a_snapshot": strata_a["snapshot_fp"],
        },
        "hygiene": {"engine_detectors": True, "hits": 0,
                    "minted_sides_scanned": len(new_rows) * 2},
        "determinism": "order-deterministic, seed 42, no wall-clock in content; "
                       "strata A snapshot is a pinned fingerprinted input; "
                       "double run byte-identical asserted",
        "disclosures": [
            "P-trans 40 rows removed and regenerated (mission directive; g3 conflict)",
            "strata A growth vs census 68: see addendum2_additions.strata_A.probe_numbers",
            "add-3 real x real pool yield = 3 pairs; authored same-scaffold fallback used",
            "mint-top-evidence/zero quotas declared 0/0: no mint-top rows in new material",
            "pool sides carry language=None (the v4.4 lm=0 mechanism); new rows set "
            "language=detect_lang(body) explicitly and declare lang_group",
            "store snapshot at probe: see probe_numbers.store_snapshot",
        ],
        "seal": {
            "holdout_fingerprint": hold_fp,
            "holdout_rows": len(holdout),
            "holdout_ids_sha256": ids_sha,
            "sealed_at_build": True,
            "note": "holdout v4.5 is SEALED after this build; do not read or modify",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (out / "train-manifest.txt").write_bytes(manifest_bytes(entries))
    (out / "run-log.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n"
                for e in runlog), encoding="utf-8")
    return manifest


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="b2_graph_corpus_v45")
    p.add_argument("--snapshot", required=True,
                   help="strata A snapshot jsonl (probe output, pinned input)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--store-uri", required=True,
                   help="read-only vesma.db URI for the sidecar join")
    return p.parse_args(argv)


def main() -> int:
    global STORE_URI
    args = parse_args()
    STORE_URI = args.store_uri
    if "mode=ro" not in STORE_URI:
        refuse("--store-uri must be a read-only URI")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runlog: list[dict] = []

    from cortex.data.store_export import make_scanner
    scanner = make_scanner(ENGINE_SRC)
    embed_side, embed_text, pin = make_embedder()
    log(runlog, "environment", embedder_pin=pin)

    base_info = stage0_base(runlog)
    surfaces = historical_surfaces(runlog)
    clean, flags = collect_side_universes(runlog)

    # occupied pair-key universe for mint-time collision checks
    occupied = {pair_key(r) for r in base_info["rows"]}
    for keys in surfaces["pair_keys"].values():
        occupied |= keys
    minter = Minter(runlog, embed_side, pin, scanner, occupied, prefix="V45")

    # census pins for add-2 cells and add-3 bands (measured, then asserted)
    dup_rows = [r for r in base_info["rows"] if r["label"] == "duplicate"]
    cell_pos: dict[str, int] = {}
    for r in dup_rows:
        if r["similarity"] >= C7_COS_MIN and lang_match(r["record"], r["candidate"]) == 1.0:
            j = ch_jaccard(r["record"], r["candidate"], 5)
            for lo, hi in C7_CH5J_CELLS:
                if lo <= j < hi:
                    cell_pos[f"[{lo:.2f},{hi:.2f})"] = cell_pos.get(f"[{lo:.2f},{hi:.2f})", 0) + 1
    if cell_pos != CENSUS_C7_CELLS:
        refuse(f"c7 cell positive pins drifted: {cell_pos} != {CENSUS_C7_CELLS}; STOP")
    centers = [round(NN_LOW + NN_BAND + k * NN_BAND, 3)
               for k in range(int(round((NN_HIGH - NN_LOW) / NN_BAND)))]
    pos_by_band = {f"{c:.3f}": 0 for c in centers}
    nnc_pos = 0
    for r in dup_rows:
        if (NN_LOW <= r["similarity"] <= NN_HIGH
                and lang_match(r["record"], r["candidate"]) == 1.0
                and tag_jaccard(r["record"], r["candidate"]) == 1.0):
            nnc_pos += 1
            for c in centers:
                if abs(r["similarity"] - c) <= NN_BAND + 1e-12:
                    pos_by_band[f"{c:.3f}"] += 1
    if nnc_pos != CENSUS_NNEAR_POS:
        refuse(f"nnc positive pins drifted: {nnc_pos} != {CENSUS_NNEAR_POS}; STOP")
    log(runlog, "census_pins", c7_cells=cell_pos, nnc_positives=nnc_pos,
        nnc_pos_by_band=pos_by_band)

    # additions
    tver_cands = gen_tver_candidates(runlog, minter, embed_side)
    tver_rows, tver_report = select_tver_bins(
        runlog, tver_cands, base_info["sibling_bins"], minter, embed_side)
    envelope_asserts(runlog, tver_rows, "tver")
    c7_cands = gen_c7_candidates(runlog, minter, embed_side, clean)
    c7_rows, c7_report = select_c7_cells(runlog, c7_cands, cell_pos, minter)
    nnc_cands = gen_nnc_candidates(runlog, minter, embed_side)
    nnc_rows, nnc_report = select_nnc_bands(
        runlog, nnc_cands, nnc_pos, pos_by_band, minter, embed_side)
    strata_a = load_strata_a(runlog, minter, Path(args.snapshot))

    synthetic = tver_rows + c7_rows + nnc_rows
    new_rows = synthetic + strata_a["train"] + strata_a["holdout"]
    envelope_asserts(runlog, strata_a["train"] + strata_a["holdout"], "strata-a")
    disjointness_asserts(runlog, new_rows, surfaces)

    # probe-snapshot adjacency for the independent evidence agreement
    adjacency: dict[str, set] = {}
    snap = corpus_rows(Path(args.snapshot))
    probe_edges = json.loads((Path(args.snapshot).parent / "strata-a-probe.json")
                             .read_text(encoding="utf-8"))
    log(runlog, "evidence_agreement_input", snapshot_pairs=len(snap),
        probe_fp=probe_edges.get("snapshot_fingerprint"))
    adj_ok: dict[str, tuple[int, int]] = {}
    for p in snap:
        adj_ok[p["a_store_id"]] = (0, 0)
        adj_ok[p["b_store_id"]] = (0, 0)

    evis = freeze_evidence_columns(runlog, new_rows, out, adj_ok)
    if evis["quarantine"]:
        refuse(f"strata A rows unresolved at sidecar time: {evis['quarantine']}; STOP")

    holdout, synth_train_part, carve = carve_holdout_v45(
        runlog, synthetic, strata_a, flags)

    removed = {r["pair_id"] for r in base_info["p_trans"]}
    train = [r for r in base_info["rows"] if r["pair_id"] not in removed]
    train += synth_train_part + strata_a["train"]
    train.sort(key=lambda r: r["pair_id"])
    holdout.sort(key=lambda r: r["pair_id"])

    # final seal asserts (BEFORE artifacts are written)
    if len(train) != len(base_info["rows"]) - 40 + len(synth_train_part) + len(strata_a["train"]):
        refuse("train assembly arithmetic mismatch; STOP")
    hold_ids = [r["pair_id"] for r in holdout]
    if len(set(hold_ids)) != len(hold_ids):
        refuse("holdout pair_id collision; STOP")
    train_ids = {r["pair_id"] for r in train}
    if any(r["pair_id"] in train_ids for r in holdout):
        refuse("train/holdout pair_id intersection; STOP")
    log(runlog, "preseal_asserts", train_rows=len(train), holdout_rows=len(holdout),
        all_ok=True)

    manifest = build_manifest(runlog, base_info, train, holdout, new_rows,
                              strata_a, tver_report, c7_report, nnc_report,
                              carve, evis, out, args)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "labels-train.jsonl",
                [{"label": v, "pair_id": k} for k, v in sorted(
                    {r["pair_id"]: r["label"] for r in train}.items())])
    write_jsonl(out / "holdout.jsonl", holdout)
    (out / "holdout-ids.json").write_text(
        json.dumps({"holdout_ids": hold_ids,
                    "holdout_ids_sha256": manifest["fingerprints"]["holdout_v45_ids_sha256"],
                    "holdout_fingerprint": manifest["fingerprints"]["holdout_v45_seal"]},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    # rewrite the run-log with the final events
    (out / "run-log.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n"
                for e in runlog), encoding="utf-8")
    print(f"[v45] DONE train={len(train)} holdout={len(holdout)} "
          f"train_fp={manifest['fingerprints']['train_v45'][:16]}... "
          f"holdout_seal={manifest['fingerprints']['holdout_v45_seal'][:16]}...")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateRefusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
