#!/usr/bin/env python3
"""Build the dataset-v4 (B2-prime) train + sealed holdout corpus.

Wave D4-1 of the dataset-v4 plan (docs/experiments/dataset-v4-plan.md).
Labels are BY CONSTRUCTION per the ratified labeling policy
(docs/specs/labeling-policy-b2.md): the generator knows exactly what it
changed and labels by tier. Content comes from authored batch files
(datasets/corpus-v4/*.jsonl — synthetic, zero store lines, ADR 0003 R2);
transforms over them are deterministic scans (no RNG anywhere), so a
double run is byte-identical.

Classes (tier → label):
    P-identity   T0 self-pairs + clones                    duplicate
    P-cosmetic   T1 case-only / whitespace-only twins      duplicate
    P-para-light sentence swap + one lexical replacement  duplicate
    P-para-sub   authored substantial paraphrase           duplicate
    P-para-struct authored structural reorganization       duplicate
    P-trans      RU/EN translation twins (sens 0.00 cure)  duplicate
    N-metadata   T2 tags / record_type / language only     not-duplicate
    N-fact-edit  T3 one fact token replaced (both orient.) not-duplicate
    N-near       same topic, different fact (hard neg.)    not-duplicate
    N-far        different topic                           not-duplicate

Pipeline: batches → validate (dedup, fact anchor, DISJOINTNESS vs synth
TOPICS / evalsets / LA-2 batch / sanity anchors) → pairs → measured
cosine + vectors (engine NanoProvider, CPU, zero network) → corner-QA
gate (cortex.data.corner_qa; refusal means NOTHING is written, policy
§5 sanction) → stratified 75/25 split per (label, stratum) → seal.

Outputs (gitignored data/ tree; repo gets counts + fingerprints):
    <train-dir>/train.jsonl        labeled train rows (stage2 format)
    <train-dir>/labels-train.jsonl {pair_id, label} for train pairs
    <train-dir>/manifest.txt       data-contract §5 pair manifest (ALL pairs)
    <train-dir>/holdout-ids.json   sealed holdout ids + fingerprints
    <train-dir>/corpus-report.json counts, corner-QA report, provenance
    <holdout-dir>/holdout.jsonl    labeled holdout rows (OUTSIDE train root)
    <holdout-dir>/labels.jsonl     holdout labels (physical isolation)
    <holdout-dir>/seal.json        the seal record

Run with the ENGINE venv (onnxruntime lives there):

    PYTHONPATH=<engine-src> python3 scripts/gen_dataset_v4.py \
        [--train-dir data/stage2/dataset-v4 \
         --holdout-dir data/stage2/dataset-v4-holdout]

Exit codes: 0 ok · 1 corner-QA refusal (nothing sealed) · 2 usage/validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.corner_qa import (  # noqa: E402 — repo src sys.path shim above
    CornerQAThresholds,
    thresholds_from_gate_contract,
)
from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.data.holdout import (  # noqa: E402
    LABEL_DUPLICATE,
    assert_labels_isolated,
    assert_no_pair_overlap,
)
from cortex.features.pair import FEATURE_NAMES  # noqa: E402

BATCH_DIR = REPO_ROOT / "datasets" / "corpus-v4"
BASE_RU_FILE = BATCH_DIR / "batch-base-ru.jsonl"
BASE_EN_FILE = BATCH_DIR / "batch-base-en.jsonl"
NEAR_FILE = BATCH_DIR / "batch-near.jsonl"
PARA_FILES = (BATCH_DIR / "batch-paras-1.jsonl", BATCH_DIR / "batch-paras-2.jsonl")
FACTS_FILE = BATCH_DIR / "batch-facts.jsonl"

DEFAULT_ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")
DEFAULT_TRAIN_DIR = REPO_ROOT / "data" / "stage2" / "dataset-v4"
DEFAULT_HOLDOUT_DIR = REPO_ROOT / "data" / "stage2" / "dataset-v4-holdout"

HOLDOUT_FRACTION = 0.25
RECORD_TYPES = ("note", "fact", "decision", "task")
ROLES = ("base", "near", "para-sub", "para-struct", "trans2")
LANGS = ("ru", "en")

#: Lexical replacement for P-para-light: first matching entry is applied to
#: the sentence-swapped body. Length- or token-changing on purpose — the
#: replacement (plus the swap) must push char5_jaccard BELOW the razor zone
#: (policy §5-G3: positives never carry the T3 char-signature). A body that
#: matches nothing aborts the build loudly: patch the batch, never silence.
SYNONYMS_RU: tuple[tuple[str, str], ...] = (
    ("раз в год", "ежегодно"),
    ("раз в месяц", "ежемесячно"),
    ("раз в неделю", "еженедельно"),
    ("раз в квартал", "ежеквартально"),
    ("раз в три месяца", "трижды в год"),
    ("раз в два месяца", "раз в 60 дней"),
    ("раз в три дня", "раз в 72 часа"),
    ("раз в два дня", "через день"),
    ("раз в сезон", "один раз за сезон"),
    ("Раз в год", "Ежегодно"),
    ("раз в", "каждые"),
    ("примерно", "порядка"),
    ("около", "приблизительно"),
    ("уже 14 месяцев", "уже больше года"),
    ("исчезли", "пропали"),
    ("исчез", "пропал"),
    ("ушёл", "пропал"),
    ("ушла", "пропала"),
    ("пропали", "исчезли"),
    ("пропал", "исчез"),
    ("появился", "возник"),
    ("появилась", "возникла"),
    ("заметно", "ощутимо"),
    ("нужна", "требуется"),
    ("нужно", "следует"),
    ("упал", "снизился"),
    ("вышло", "собралось"),
    ("стало", "сделалось"),
    ("перестал", "больше не"),
    ("прекратился", "остановился"),
    ("держится", "сохраняется"),
    ("вернулась", "восстановилась"),
    ("вернулось", "восстановилось"),
    ("вернулся", "восстановился"),
    ("гаснет", "тухнет"),
    ("сразу", "без промедления"),
    ("Завёл", "Создал"),
    ("Растапливаю", "Разогреваю"),
    ("закрылась", "решилась"),
    ("убрала", "устранила"),
    ("собралась", "сложилась"),
    ("случился", "произошёл"),
    ("Раз в месяц", "Каждый месяц"),
    ("заряжаю", "питаю"),
    ("Кладу", "Укладываю"),
    ("дважды в год", "по два раза в год"),
    ("Поставил", "Установил"),
    ("Забрал", "Получил"),
    ("живут", "растут"),
    ("Веду", "Поддерживаю"),
)
SYNONYMS_EN: tuple[tuple[str, str], ...] = (
    ("every three months", "quarterly"),
    ("every two months", "bimonthly"),
    ("every six weeks", "every 42 days"),
    ("every three days", "every 72 hours"),
    ("every two days", "every other day"),
    ("once a year", "annually"),
    ("once a season", "once per season"),
    ("twice a year", "semiannually"),
    ("about a tenth", "about ten percent"),
    ("instead of", "rather than"),
    ("noticeably", "markedly"),
    ("Once a month", "Each month"),
    ("The quarter saw", "The quarter recorded"),
    ("stayed away", "kept away"),
    ("stayed clean", "remained clean"),
    ("caught", "uncovered"),
    ("happen", "occur"),
    ("ends up at", "settles at"),
    ("showed", "confirmed"),
    ("Picked it up", "Collected it"),
    ("went through", "was processed"),
    ("wiped", "cleaned"),
    ("dropped by", "fell by"),
    ("missed it", "overlooked it"),
    ("is gone", "has disappeared"),
    ("is indistinguishable from fresh", "is as good as fresh"),
    ("I go", "I visit the studio"),
    ("on purchase day", "the same day"),
    ("I liquefy", "I re-liquefy"),
    ("closed based on", "was settled by"),
    (
        "I sow tomatoes on March 20 in covered trays and pot them up",
        "Tomato seeds go into covered trays on March 20 and are potted up",
    ),
    ("came together", "was assembled"),
    ("mattered more", "counted for more"),
    ("I moved to", "I switched to"),
    ("I paint the fence in", "The fence is painted in"),
    ("I put two drops", "Two drops go"),
    (
        "I submit water and electricity readings",
        "Water and electricity readings are submitted",
    ),
    ("I keep a dedicated", "A dedicated"),
    ("about", "roughly"),
    ("after", "following"),
    ("every", "each"),
)
#: Tags appended by the N-metadata transform (tag_jaccard must drop below 1).
META_TAG_POOL: tuple[str, ...] = ("imported", "archive", "follow-up")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _record_text_key(record: dict[str, Any]) -> str:
    """Content hash of a record WITHOUT the designed metadata surface —
    used for dedup of authored texts (identity clones repeat content BY
    DESIGN and are built in the generator, not in batches)."""
    return hashlib.sha256(
        canonical_json(
            {
                "title": record.get("title"),
                "body": record.get("body"),
                "language": record.get("language"),
            }
        ).encode("utf-8")
    ).hexdigest()


def load_batches() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Load and structurally validate all batches.

    Returns (base_ru, base_en, near_rows, para_rows by (key, lang), facts).
    """
    base_ru_rows = _read_jsonl(BASE_RU_FILE)
    base_en_rows = _read_jsonl(BASE_EN_FILE)
    near_rows = _read_jsonl(NEAR_FILE)
    para_rows: list[dict[str, Any]] = []
    for path in PARA_FILES:
        para_rows.extend(_read_jsonl(path))
    facts = _read_jsonl(FACTS_FILE)

    base_ru = {r["key"]: r for r in base_ru_rows}
    base_en = {r["key"]: r for r in base_en_rows}

    # ── structural validation (loud, before anything is built) ───────────
    if len(base_ru) != len(base_ru_rows) or len(base_en) != len(base_en_rows):
        raise SystemExit("validation: duplicate keys inside a base batch")
    if set(base_ru) != set(base_en):
        missing_ru = sorted(set(base_en) - set(base_ru))
        missing_en = sorted(set(base_ru) - set(base_en))
        raise SystemExit(
            f"validation: RU/EN base keys diverge (ru-only: {missing_ru[:3]}, "
            f"en-only: {missing_en[:3]})"
        )
    for row in base_ru_rows + base_en_rows + near_rows + para_rows:
        if row.get("role") not in ROLES:
            raise SystemExit(f"validation: bad role {row.get('role')!r}")
        if row.get("lang") not in LANGS:
            raise SystemExit(f"validation: bad lang {row.get('lang')!r}")
        if row.get("record_type") not in RECORD_TYPES:
            raise SystemExit(f"validation: bad record_type {row.get('record_type')!r}")
        for field in ("title", "body"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise SystemExit(
                    f"validation: empty {field} in {row['key']}/{row['lang']}"
                )
        if not isinstance(row.get("tags"), list) or not row["tags"]:
            raise SystemExit(f"validation: bad tags in {row['key']}/{row['lang']}")
        for tag in row["tags"]:
            if not isinstance(tag, str) or not tag or tag != tag.lower():
                raise SystemExit(
                    f"validation: bad tag {tag!r} in {row['key']}/{row['lang']}"
                )
        if (
            len(_SENTENCE_SPLIT.split(row["body"].strip())) < 2
            and row.get("role") == "base"
        ):
            raise SystemExit(
                f"validation: base body needs >= 2 sentences: {row['key']}/{row['lang']}"
            )

    # authored texts are unique (dedup by design, policy §4)
    seen: dict[str, str] = {}
    for row in base_ru_rows + base_en_rows + near_rows + para_rows:
        key = _record_text_key(row)
        if key in seen:
            raise SystemExit(
                f"validation: duplicate authored text {row['key']}/{row['lang']} == {seen[key]}"
            )
        seen[key] = f"{row['key']}/{row['lang']}"

    # near rows: same topic, genuinely different content, known key+lang
    for row in near_rows:
        base = base_ru if row["lang"] == "ru" else base_en
        if row["key"] not in base:
            raise SystemExit(
                f"validation: near row without a base: {row['key']}/{row['lang']}"
            )
        if _record_text_key(row) == _record_text_key(base[row["key"]]):
            raise SystemExit(
                f"validation: near row identical to its base: {row['key']}"
            )

    # paras: anchor to an existing base of the same key+lang
    for row in para_rows:
        base = base_ru if row["lang"] == "ru" else base_en
        if row["key"] not in base:
            raise SystemExit(
                f"validation: para row without a base: {row['key']}/{row['lang']}"
            )
        if _record_text_key(row) == _record_text_key(base[row["key"]]):
            raise SystemExit(
                f"validation: para row identical to its base: {row['key']}"
            )

    # facts: the find-anchor must occur EXACTLY ONCE in the base body
    for spec in facts:
        base = base_ru if spec["lang"] == "ru" else base_en
        if spec["key"] not in base:
            raise SystemExit(
                f"validation: fact spec without a base: {spec['key']}/{spec['lang']}"
            )
        body = base[spec["key"]]["body"]
        if body.count(spec["find"]) != 1:
            raise SystemExit(
                f"validation: fact anchor {spec['find']!r} occurs "
                f"{body.count(spec['find'])}x (need 1) in {spec['key']}/{spec['lang']}"
            )
        if spec["find"] == spec["replace"]:
            raise SystemExit(
                f"validation: no-op fact spec in {spec['key']}/{spec['lang']}"
            )

    paras: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in para_rows:
        paras.setdefault((row["key"], row["lang"]), []).append(row)
    return base_ru, base_en, near_rows, paras, facts


def check_disjointness(
    base_ru: dict[str, dict[str, Any]],
    base_en: dict[str, dict[str, Any]],
    near_rows: list[dict[str, Any]],
    para_rows: list[dict[str, Any]],
) -> None:
    """Corpus themes are disjoint from synth TOPICS, evalsets, the LA-2
    batch and the sanity anchors — by TEST-level discipline, enforced here
    at build time too (tests pin it independently)."""
    from cortex.eval.sanity import PROBE_RECORD, UNRELATED_RECORD
    from cortex.evalsets.topics import EVAL_TOPICS, anchor_records
    from cortex.synth.generate import TOPICS

    forbidden_keys: set[str] = {t.key for t in TOPICS} | {t.key for t in EVAL_TOPICS}
    forbidden_titles: set[str] = {t.title for t in TOPICS} | {
        t.title for t in EVAL_TOPICS
    }
    forbidden_titles.add(PROBE_RECORD.title)
    forbidden_titles.add(UNRELATED_RECORD.title)
    forbidden_texts: set[str] = {PROBE_RECORD.body, UNRELATED_RECORD.body}
    for rec in anchor_records():
        forbidden_titles.add(rec.title)
        forbidden_texts.add(rec.body)

    la2_path = (
        REPO_ROOT / "datasets" / "evalsets" / "llm-batches" / "la2-llm-batch-1.jsonl"
    )
    if la2_path.exists():
        for line in la2_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for side in ("record", "candidate"):
                forbidden_titles.add(row[side]["title"])
                forbidden_texts.add(row[side]["body"])

    all_rows = list(base_ru.values()) + list(base_en.values()) + near_rows + para_rows
    for row in all_rows:
        if row["key"] in forbidden_keys:
            raise SystemExit(f"disjointness: key collides: {row['key']}")
        if row["title"] in forbidden_titles:
            raise SystemExit(f"disjointness: title collides: {row['title']!r}")
        if row["body"] in forbidden_texts:
            raise SystemExit(f"disjointness: body collides in {row['key']}")


# ── deterministic transforms ─────────────────────────────────────────────────


def _swapcase_record(
    record: dict[str, Any], *, title: bool, body: bool
) -> dict[str, Any]:
    out = deepcopy(record)
    if title:
        out["title"] = out["title"].swapcase()
    if body:
        out["body"] = out["body"].swapcase()
    return out


def _add_spaces_record(record: dict[str, Any]) -> dict[str, Any]:
    """Whitespace-only variant: every sentence gap widens — erased by the
    feature normalization (policy T1), so the pair stays a duplicate."""
    out = deepcopy(record)
    out["body"] = out["body"].replace(". ", ".  ")
    return out


def _sentence_swapped(record: dict[str, Any]) -> dict[str, Any]:
    sentences = _SENTENCE_SPLIT.split(record["body"].strip())
    if len(sentences) < 2:
        raise SystemExit(f"light transform needs >= 2 sentences: {record['title']!r}")
    out = deepcopy(record)
    out["body"] = " ".join([sentences[1], sentences[0], *sentences[2:]])
    return out


def _lexical_replacement(record: dict[str, Any], lang: str) -> dict[str, Any]:
    table = SYNONYMS_RU if lang == "ru" else SYNONYMS_EN
    body = record["body"]
    for find, replace in table:
        if find in body:
            out = deepcopy(record)
            out["body"] = body.replace(find, replace, 1)
            return out
    raise SystemExit(
        f"light transform: no synonym entry matches body of "
        f"{record['title']!r} — extend the table or patch the batch"
    )


def _fact_edited(record: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(record)
    out["body"] = out["body"].replace(spec["find"], spec["replace"], 1)
    return out


def _with_meta(
    record: dict[str, Any],
    *,
    tags_delta: bool,
    type_delta: bool,
    lang_delta: bool,
    salt: int,
) -> dict[str, Any]:
    out = deepcopy(record)
    if tags_delta:
        out["tags"] = list(out["tags"]) + [META_TAG_POOL[salt % len(META_TAG_POOL)]]
    if type_delta:
        idx = RECORD_TYPES.index(out["record_type"])
        out["record_type"] = RECORD_TYPES[(idx + 1) % len(RECORD_TYPES)]
    if lang_delta:
        current = out.get("language") or out.get("lang")
        out["language"] = "en" if current == "ru" else "ru"
    return out


def _side_of(row: dict[str, Any]) -> dict[str, Any]:
    """Batch row → the frozen contract surface (RecordLike fields only)."""
    return {
        "title": row["title"],
        "body": row["body"],
        "tags": list(row["tags"]),
        "language": row.get("language") or row.get("lang"),
        "record_type": row.get("record_type"),
    }


def _first_spec_per_cell(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic T3 quota: the FIRST spec of every (key, lang) cell in
    file order — 48 keys × 2 langs = 96 specs → 192 pairs with both
    orientations. Surplus specs stay in the batch as documented reserve."""
    seen: set[tuple[str, str]] = set()
    picked: list[dict[str, Any]] = []
    for spec in facts:
        cell = (spec["key"], spec["lang"])
        if cell in seen:
            continue
        seen.add(cell)
        picked.append(spec)
    return picked


def build_pairs(
    base_ru: dict[str, dict[str, Any]],
    base_en: dict[str, dict[str, Any]],
    near_rows: list[dict[str, Any]],
    paras: dict[tuple[str, str], list[dict[str, Any]]],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The whole corpus, in deterministic construction order."""
    keys = sorted(base_ru)
    pairs: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    def add(
        cls: str,
        stratum: str,
        label: str,
        record: dict[str, Any],
        candidate: dict[str, Any],
    ) -> None:
        counters[cls] = counters.get(cls, 0) + 1
        pairs.append(
            {
                "pair_id": f"V4-{cls}-{counters[cls]:04d}",
                "label": label,
                "stratum": stratum,
                "record": _side_of(record),
                "candidate": _side_of(candidate),
            }
        )

    # P-identity: self-pairs (48) + clones (48) — the corner is POSITIVE now
    for i, key in enumerate(keys):
        rec = base_ru[key] if i % 2 == 0 else base_en[key]
        add("IDENT", "P-identity", LABEL_DUPLICATE, rec, rec)
    for i, key in enumerate(keys):
        rec = base_ru[key] if i % 2 == 1 else base_en[key]
        add("IDENT", "P-identity", LABEL_DUPLICATE, rec, deepcopy(rec))

    # P-cosmetic: case-only (48, zero len delta) + whitespace (24)
    for i, key in enumerate(keys):
        rec = base_ru[key] if i % 2 == 0 else base_en[key]
        add(
            "COSM",
            "P-cosmetic",
            LABEL_DUPLICATE,
            rec,
            _swapcase_record(rec, title=True, body=(i % 4 == 0)),
        )
    for i, key in enumerate(keys[:24]):
        rec = base_ru[key]
        add("COSM", "P-cosmetic", LABEL_DUPLICATE, rec, _add_spaces_record(rec))

    # P-para-light: sentence swap + one lexical replacement (96: 48 ru + 48 en)
    for key in keys:
        for rec in (base_ru[key], base_en[key]):
            add(
                "LIGHT",
                "P-para-light",
                LABEL_DUPLICATE,
                rec,
                _lexical_replacement(_sentence_swapped(rec), rec["lang"]),
            )

    # P-para-sub / P-para-struct: authored variants (48 + 36)
    for (key, lang), rows in sorted(paras.items()):
        base = base_ru if lang == "ru" else base_en
        for row in rows:
            if row["role"] == "para-sub":
                add("SUBST", "P-para-sub", LABEL_DUPLICATE, base[key], row)
            elif row["role"] == "para-struct":
                add("STRUCT", "P-para-struct", LABEL_DUPLICATE, base[key], row)
            elif row["role"] == "trans2":
                add("TRANS", "P-trans", LABEL_DUPLICATE, base_ru[key], row)
            else:  # pragma: no cover — validated at load
                raise SystemExit(f"unexpected para role {row['role']}")

    # P-trans: base RU/EN twins (48)
    for key in keys:
        add("TRANS", "P-trans", LABEL_DUPLICATE, base_ru[key], base_en[key])

    # N-metadata: tag / type / lang / combo deltas over identical text (112)
    for i, key in enumerate(keys[:32]):
        rec = base_ru[key] if i % 2 == 0 else base_en[key]
        add(
            "META",
            "N-metadata",
            "not-duplicate",
            rec,
            _with_meta(
                rec, tags_delta=True, type_delta=False, lang_delta=False, salt=i
            ),
        )
    for i, key in enumerate(keys[:32]):
        rec = base_ru[key] if i % 2 == 1 else base_en[key]
        add(
            "META",
            "N-metadata",
            "not-duplicate",
            rec,
            _with_meta(
                rec, tags_delta=False, type_delta=True, lang_delta=False, salt=i
            ),
        )
    for i, key in enumerate(keys[:32]):
        rec = base_ru[key] if i % 2 == 0 else base_en[key]
        add(
            "META",
            "N-metadata",
            "not-duplicate",
            rec,
            _with_meta(
                rec, tags_delta=False, type_delta=False, lang_delta=True, salt=i
            ),
        )
    for i, key in enumerate(keys[:16]):
        rec = base_en[key] if i % 2 == 0 else base_ru[key]
        add(
            "META",
            "N-metadata",
            "not-duplicate",
            rec,
            _with_meta(rec, tags_delta=True, type_delta=True, lang_delta=False, salt=i),
        )

    # N-fact-edit: T3 razor, BOTH orientations (96 specs × 2)
    for i, spec in enumerate(_first_spec_per_cell(facts)):
        base = base_ru if spec["lang"] == "ru" else base_en
        rec = base[spec["key"]]
        edited = _fact_edited(rec, spec)
        add("FACT", "N-fact-edit", "not-duplicate", rec, edited)
        add("FACT", "N-fact-edit", "not-duplicate", edited, rec)

    # N-near: same topic, different fact (authored, 48)
    for row in near_rows:
        base = base_ru if row["lang"] == "ru" else base_en
        add("NEAR", "N-near", "not-duplicate", base[row["key"]], row)

    # N-far: different topics; half cross-language (the embedder's lowest
    # band), quarter each same-language (108). The offset grows per 48-block
    # so no pair content repeats (7, then 8, then 9 — deterministic).
    for i in range(108):
        key_a = keys[i % len(keys)]
        key_b = keys[(i + 7 + i // len(keys)) % len(keys)]
        if key_a == key_b:  # pragma: no cover — 48 keys, offsets 7..9
            raise SystemExit("far transform: key collided with itself")
        mode = i % 4
        if mode == 0:
            add("FAR", "N-far", "not-duplicate", base_ru[key_a], base_ru[key_b])
        elif mode == 1:
            add("FAR", "N-far", "not-duplicate", base_en[key_a], base_en[key_b])
        else:
            add("FAR", "N-far", "not-duplicate", base_ru[key_a], base_en[key_b])

    return pairs


def embed_rows(
    rows: list[dict[str, Any]], engine_src: Path
) -> tuple[list[dict[str, Any]], str]:
    """Measured cosine + 384-dim vectors via the engine NanoProvider
    (frozen vesma-embed-v1, CPU, zero network). Text-keyed cache."""
    if str(engine_src) not in sys.path:
        sys.path.insert(0, str(engine_src))
    from vesmaro.embeddings import NanoProvider  # noqa: E402 — engine shim

    provider = NanoProvider()
    print(f"embedder pin: {provider.fingerprint}", file=sys.stderr)
    cache: dict[str, list[float]] = {}

    def vec(side: dict[str, Any]) -> list[float]:
        tags = " ".join(side.get("tags") or [])
        text = f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}"[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        va, vb = vec(row["record"]), vec(row["candidate"])
        dot = sum(x * y for x, y in zip(va, vb))
        out.append(
            {
                **row,
                "similarity": round(min(1.0, max(0.0, dot)), 6),
                "vec_a": va,
                "vec_b": vb,
            }
        )
        if i % 100 == 0:
            print(f"  embedded {i}/{len(rows)}", file=sys.stderr, flush=True)
    return out, provider.fingerprint


def stratified_split(
    rows: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (label, stratum): pair_id-sorted, first ceil(fraction·n) → holdout.
    Same discipline as the prereg split (cortex.data.holdout), localized to
    the stage2 layout with the wave's 75/25 proportion."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["label"], row["stratum"]), []).append(row)
    hold_ids: set[str] = set()
    for (_, _), members in sorted(cells.items()):
        members = sorted(members, key=lambda r: r["pair_id"])
        n_hold = math.ceil(fraction * len(members))
        hold_ids.update(r["pair_id"] for r in members[:n_hold])
    train = [r for r in rows if r["pair_id"] not in hold_ids]
    hold = [r for r in rows if r["pair_id"] in hold_ids]
    assert_no_pair_overlap([r["pair_id"] for r in train], [r["pair_id"] for r in hold])
    return train, hold


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    ap.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    ap.add_argument("--engine-src", type=Path, default=DEFAULT_ENGINE_SRC)
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="optional JSON with CornerQAThresholds overrides (default: the "
        "corner_qa section of gate_contract.json, falling back to the frozen "
        "code constants when the contract file is absent)",
    )
    args = ap.parse_args(argv)

    thresholds, thresholds_source = thresholds_from_gate_contract()
    print(f"corner-QA thresholds source: {thresholds_source}", file=sys.stderr)
    if args.thresholds is not None:
        overrides = json.loads(args.thresholds.read_text(encoding="utf-8"))
        thresholds = CornerQAThresholds(**overrides)

    base_ru, base_en, near_rows, paras, facts = load_batches()
    para_flat = [row for rows in paras.values() for row in rows]
    check_disjointness(base_ru, base_en, near_rows, para_flat)

    pairs = build_pairs(base_ru, base_en, near_rows, paras, facts)
    print(f"pairs built: {len(pairs)}", file=sys.stderr)

    ids = [p["pair_id"] for p in pairs]
    if len(ids) != len(set(ids)):
        raise SystemExit("validation: duplicate pair_id at construction")

    # NO network: embeddings are local CPU inference (engine NanoProvider)
    started = time.monotonic()
    rows, embedder_pin = embed_rows(pairs, args.engine_src)
    print(f"embedded in {time.monotonic() - started:.1f}s", file=sys.stderr)

    # corner-QA gate BEFORE anything is sealed (policy §5 sanction order)
    ok, report = _run_qa(rows, thresholds)
    if not ok:
        print(
            "CORNER-QA REFUSAL — corpus is invalid, nothing will be written:",
            file=sys.stderr,
        )
        for line in report["violations"]:
            print(f"  - {line}", file=sys.stderr)
        return 1

    train, hold = stratified_split(rows, HOLDOUT_FRACTION)

    # fingerprints (data-contract §5): pairs then labels
    entries = [
        (
            r["pair_id"],
            pair_sha256(
                {
                    "record": r["record"],
                    "candidate": r["candidate"],
                    "similarity": r["similarity"],
                }
            ),
        )
        for r in rows
    ]
    fp = corpus_fingerprint(manifest_bytes(entries))
    label_fp = labels_fingerprint({r["pair_id"]: r["label"] for r in rows})

    train_root = args.train_dir.resolve()
    train_root.mkdir(parents=True, exist_ok=True)
    labels_path = (args.holdout_dir / "labels.jsonl").resolve()
    assert_labels_isolated(train_root, labels_path)

    _write_jsonl(train_root / "train.jsonl", train)
    _write_jsonl(
        train_root / "labels-train.jsonl",
        [{"pair_id": r["pair_id"], "label": r["label"]} for r in train],
    )
    (train_root / "manifest.txt").write_bytes(manifest_bytes(entries))

    train_entries = [
        (r["pair_id"], d) for r, (pid, d) in _pair_entries(rows, entries, train)
    ]
    hold_entries = [
        (r["pair_id"], d) for r, (pid, d) in _pair_entries(rows, entries, hold)
    ]
    train_fp = corpus_fingerprint(manifest_bytes(train_entries))
    hold_fp = corpus_fingerprint(manifest_bytes(hold_entries))

    # holdout side — physically OUTSIDE the train root
    args.holdout_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.holdout_dir / "holdout.jsonl", hold)
    _write_jsonl(
        labels_path, [{"pair_id": r["pair_id"], "label": r["label"]} for r in hold]
    )

    batch_files = [BASE_RU_FILE, BASE_EN_FILE, NEAR_FILE, *PARA_FILES, FACTS_FILE]
    batch_shas = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in batch_files
    }

    seal = {
        "corpus_fingerprint": fp,
        "label_fingerprint": label_fp,
        "train_fingerprint": train_fp,
        "holdout_fingerprint": hold_fp,
        "holdout_ids": sorted(r["pair_id"] for r in hold),
        "split": f"stratified per (label, stratum), first ceil({HOLDOUT_FRACTION}*n) by pair_id",
        "holdout_fraction": HOLDOUT_FRACTION,
        "isolation": "holdout dir outside train root (assert_labels_isolated) + zero id overlap (assert_no_pair_overlap)",
    }
    # holdout-ids.json is byte-deterministic (no timestamps): a double run
    # must reproduce it exactly; the wall-clock seal record lives separately.
    (train_root / "holdout-ids.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (args.holdout_dir / "seal.json").write_text(
        json.dumps(
            {
                "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **{k: v for k, v in seal.items() if k != "holdout_ids"},
                "holdout_count": len(hold),
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    report_payload = {
        "kind": "dataset-v4-corpus-report",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": "dataset-v4 (B2-prime), wave D4-1",
        "total_pairs": len(rows),
        "train_pairs": len(train),
        "holdout_pairs": len(hold),
        "by_stratum": report["by_stratum"],
        "by_label": report["by_label"],
        "holdout_by_stratum": _counter(hold, "stratum"),
        "holdout_by_label": _counter(hold, "label"),
        "cos_histogram": report["cos_histogram"],
        "cos_below_055": report["cos_below_055"],
        "cos_below_070": report["cos_below_070"],
        "corner_qa": {k: v for k, v in report.items() if k != "violations"},
        "corner_qa_violations": report["violations"],
        "thresholds": {
            f: getattr(thresholds, f) for f in thresholds.__dataclass_fields__
        },
        "feature_names": list(FEATURE_NAMES),
        "fingerprints": {
            "corpus": fp,
            "labels": label_fp,
            "train": train_fp,
            "holdout": hold_fp,
        },
        "embedder_pin": embedder_pin,
        "network": "none (local CPU inference)",
        "batches": batch_shas,
        "determinism": "structural (scan-based, no RNG); double run is byte-identical",
    }
    (train_root / "corpus-report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "total": len(rows),
                "train": len(train),
                "holdout": len(hold),
                "by_stratum": report_payload["by_stratum"],
                "corpus_fingerprint": fp,
                "train_dir": str(train_root),
                "holdout_dir": str(args.holdout_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_qa(
    rows: list[dict[str, Any]], thresholds: CornerQAThresholds
) -> tuple[bool, dict[str, Any]]:
    from cortex.data.corner_qa import run_corner_qa

    return run_corner_qa(rows, thresholds)


def _pair_entries(
    rows: list[dict[str, Any]],
    entries: list[tuple[str, str]],
    subset: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], tuple[str, str]]]:
    subset_ids = {r["pair_id"] for r in subset}
    return [(r, e) for r, e in zip(rows, entries) if r["pair_id"] in subset_ids]


def _counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
