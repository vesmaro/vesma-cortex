"""Synthetic pair corpus — stage-1 training data (ADR 0001 addendum П3).

Owner decision П3 (2026-09-29): a synthetic corpus is stage ONE of training
(basic); the ~140 real owner pairs remain stage TWO (final supervised fit);
evaluation stays on the REAL holdout only. The model's input is NUMERIC pair
features (never raw text), so the LLM here is a GENERATOR OF TEXT PAIRS with
labels BY CONSTRUCTION — never a judge and never a language teacher:

- paraphrase          → label 1: the LLM rewrites one record keeping the
  meaning (synonyms, reordered arguments); light procedural meaning-free
  variations from cortex.pretrain.corruption are applied on top;
- near-topic          → label 0: the LLM writes a NEW record on the SAME
  topic/domain with a DIFFERENT fact/decision/conclusion — topically close,
  semantically another memory (the W4c hard zone: globally close, locally
  different);
- broken-field        → label 0: same text, one field broken — procedural
  composition over the EXISTING corruption hard negatives (no LLM);
- trivial-negative    → label 0: two records from DIFFERENT topics —
  procedural (no LLM).

Purity contract (tests/test_skeleton.py AST tripwire): this module is
NETWORK-FREE. The LLM enters ONLY as an injected callback — either the
one-prompt form ``llm_fn: Callable[[str], str]]`` (local ollama) or the
batched form ``llm_batch_fn: Callable[[Sequence[str]], Sequence[str]]``
(cloud chat/completions backends, 6–8 records per HTTP call); the HTTP
clients live outside src/cortex (scripts/gen_synth_corpus.py).

Determinism / provenance (П3): generation is seed-deterministic for the
procedural part; prompts carry PROMPT_VERSION; RU records get RU prompts,
EN records get EN ones. Pair content is NEVER committed — the corpus
fingerprint (BLAKE2b-256 over the manifest, data-contract §5 scheme via
cortex.data.fingerprints) plus counters are the committable provenance.

Repeat-basis mechanics (full-corpus scale): 40 seed topics against a
1200–1600 pair corpus means every base record is reused 7–10 times per
strategy. Two mechanisms keep that honest:

- VARIANT PROMPTS — at temperature 0 the same prompt returns the same text,
  so a repeated base gets an explicit "variant №k" instruction line; the
  prompt (hence the output) differs per occurrence. First occurrence of a
  base sends the bare template.
- BOUNDED RETRIES — a slot whose constructed pair collides with an already
  emitted pair_id retries up to MAX_ATTEMPTS times (LLM slots bump the
  variant, procedural slots re-draw the RNG); exhaustion drops the slot
  with a counter, never a silent repair and never a corpus-killing raise.

Translation-twin guard: the seed list holds each topic twice (RU + EN).
Two records sharing a topic ``key`` are the SAME memory in different
languages — pairing them as a negative would be a FALSE label by
construction. Trivial-negative therefore refuses same-key partners
(drop counter ``same-topic``).

Fingerprinted object per pair (docs/synth.md, differs from the eval corpus
§3 BY DESIGN — synth pairs carry their label inside): canonical JSON of
``{"record": <side>, "candidate": <side>, "label": "duplicate"|
"not-duplicate"}``. pair_id/strategy/seed are assignment, not content.
``similarity`` is NOT here: synth pairs have no store vector at generation
time — the measured cosine is attached later by the A2 docompute step, which
produces a DERIVED corpus with its own fingerprint.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from cortex.data.fingerprints import (
    canonical_json,
    corpus_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.features.pair import PairRecord
from cortex.pretrain.corruption import (
    HARD_NEGATIVE_TRANSFORMS,
    CollapseWhitespaceBody,
    FlipLanguage,
    FlipRecordType,
    NormalizePunctuationBody,
    ShuffleTags,
    SubstituteField,
    SwapCaseTitle,
    record_key,
)

__all__ = [
    "BODY_RATIO_MAX",
    "BODY_RATIO_MIN",
    "LABEL_DUPLICATE",
    "LABEL_NOT_DUPLICATE",
    "LLM_STRATEGIES",
    "MARKER_LINE_RE",
    "MARKER_PREFIX_RE",
    "MAX_ATTEMPTS",
    "MAX_TITLE_CHARS",
    "MIN_BODY_CHARS",
    "MIN_TITLE_CHARS",
    "NEAR_TOPIC_PROMPT_EN",
    "NEAR_TOPIC_PROMPT_RU",
    "PARAPHRASE_PROMPT_EN",
    "PARAPHRASE_PROMPT_RU",
    "PROMPT_VERSION",
    "STRATEGIES",
    "STRATEGY_BROKEN_FIELD",
    "STRATEGY_NEAR_TOPIC",
    "STRATEGY_PARAPHRASE",
    "STRATEGY_TRIVIAL_NEGATIVE",
    "TOPICS",
    "VARIANT_LINE_EN",
    "VARIANT_LINE_RU",
    "SynthPair",
    "SynthRecord",
    "SynthStats",
    "SynthTopic",
    "SynthTopicError",
    "corpus_fingerprint_of_pairs",
    "fingerprint_object",
    "generate_corpus",
    "label_name",
    "manifest_bytes_of_pairs",
    "manifest_entries",
    "pair_row",
    "pair_row_json",
    "parse_llm_record",
    "sample_quotas",
    "strip_protocol_markers",
    "validate_candidate",
]

#: Version of the prompt set below — provenance field of every corpus.
PROMPT_VERSION: Final[str] = "v1"

#: Strategy names (machine strings, latin-only — ADR 0001 V4).
STRATEGY_PARAPHRASE: Final[str] = "paraphrase"
STRATEGY_NEAR_TOPIC: Final[str] = "near-topic"
STRATEGY_BROKEN_FIELD: Final[str] = "broken-field"
STRATEGY_TRIVIAL_NEGATIVE: Final[str] = "trivial-negative"

#: Fixed emission order; the corpus fingerprint depends on content, the
#: order fixes topic cycling only.
STRATEGIES: Final[tuple[str, ...]] = (
    STRATEGY_PARAPHRASE,
    STRATEGY_NEAR_TOPIC,
    STRATEGY_BROKEN_FIELD,
    STRATEGY_TRIVIAL_NEGATIVE,
)

#: Strategies that call the injected LLM.
LLM_STRATEGIES: Final[tuple[str, ...]] = (STRATEGY_PARAPHRASE, STRATEGY_NEAR_TOPIC)

LABEL_DUPLICATE: Final[int] = 1
LABEL_NOT_DUPLICATE: Final[int] = 0

#: Backwards-friendly alias: a synth record IS a PairRecord shape —
#: the frozen feature contract consumes it unchanged (features()).
SynthRecord = PairRecord

#: LLM output corridor. The body must stay within [ratio_min, ratio_max] of
#: the base body (and at least MIN_BODY_CHARS) — outputs outside are dropped
#: with a counter, never silently repaired.
MIN_TITLE_CHARS: Final[int] = 3
MAX_TITLE_CHARS: Final[int] = 150
MIN_BODY_CHARS: Final[int] = 30
BODY_RATIO_MIN: Final[float] = 0.35
BODY_RATIO_MAX: Final[float] = 3.0

#: Per-slot retry bound on pair-id collisions (see module docstring).
MAX_ATTEMPTS: Final[int] = 4


class SynthTopicError(ValueError):
    """A topic definition violates the SynthRecord surface."""


class SynthTopic(NamedTuple):
    """One seed topic — a full record-shaped memory snippet.

    RU and EN halves by design; ``body`` must state a CONCRETE fact or
    decision so the near-topic strategy has something to differ from.
    ``key`` is the SEMANTIC identity of the topic (latin machine string,
    ADR 0001 V4): the RU and EN entries describing the same memory share
    one key, and the trivial-negative strategy refuses same-key partners —
    a translation twin labeled not-duplicate would be a false label.
    """

    title: str
    body: str
    tags: tuple[str, ...]
    language: str
    record_type: str
    key: str = ""

    def as_record(self) -> SynthRecord:
        return SynthRecord(
            title=self.title,
            body=self.body,
            tags=self.tags,
            language=self.language,
            record_type=self.record_type,
        )


@dataclass(frozen=True)
class SynthPair:
    """One generated pair with its label BY CONSTRUCTION (never LLM-judged)."""

    pair_id: str
    record_a: SynthRecord
    record_b: SynthRecord
    label: int  # LABEL_DUPLICATE | LABEL_NOT_DUPLICATE
    strategy: str
    seed: int  # per-pair seed (master-seed derived) — provenance


@dataclass
class SynthStats:
    """Generation counters — dropped pairs are counted, never hidden.

    ``llm_calls`` counts RECORD prompts (one per LLM slot attempt), while
    ``llm_batches``/``llm_json_retries`` are HTTP-level counters owned by the
    transport wrapper (0 on the sequential ollama path); ``llm_strips``
    counts protocol-marker cleanups (see :func:`strip_protocol_markers`).
    """

    attempts: int = 0
    emitted: int = 0
    dropped_empty: int = 0
    dropped_length: int = 0
    dropped_identical: int = 0
    dropped_duplicate: int = 0
    dropped_same_topic: int = 0
    llm_calls: int = 0
    llm_batches: int = 0
    llm_json_retries: int = 0
    llm_strips: int = 0

    def drop(self, reason: str) -> None:
        if reason == "empty":
            self.dropped_empty += 1
        elif reason == "length":
            self.dropped_length += 1
        elif reason == "identical":
            self.dropped_identical += 1
        elif reason == "duplicate":
            self.dropped_duplicate += 1
        elif reason == "same-topic":
            self.dropped_same_topic += 1
        else:
            raise ValueError(f"unknown drop reason: {reason!r}")

    def as_dict(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "emitted": self.emitted,
            "dropped_empty": self.dropped_empty,
            "dropped_length": self.dropped_length,
            "dropped_identical": self.dropped_identical,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_same_topic": self.dropped_same_topic,
            "llm_calls": self.llm_calls,
            "llm_batches": self.llm_batches,
            "llm_json_retries": self.llm_json_retries,
            "llm_strips": self.llm_strips,
        }


# ── Prompts (PROMPT_VERSION = "v1") — the LLM is a text generator, the label
#    comes from the STRATEGY, never from the model's judgement. ────────────────


PARAPHRASE_PROMPT_RU: Final[str] = (
    "Ты генерируешь синтетические данные для детектора дубликатов заметок.\n"
    "Перепиши запись ниже, полностью сохранив смысл: та же память, те же факты,\n"
    "то же решение и вывод. Измени формулировки: синонимы, другой порядок\n"
    "аргументов, другая структура фраз. Не добавляй новых фактов и не теряй старых.\n"
    "Ответь СТРОГО в формате (без пояснений, без кавычек, без markdown):\n"
    "строка 1 — новый заголовок;\n"
    "остальные строки — новое тело записи.\n"
    "\n"
    "Запись:\n"
    "Заголовок: {title}\n"
    "Тело: {body}\n"
    "Теги: {tags}"
)

PARAPHRASE_PROMPT_EN: Final[str] = (
    "You generate synthetic data for a note-duplicate detector.\n"
    "Rewrite the record below keeping the meaning fully intact: same memory,\n"
    "same facts, same decision and conclusion. Change the wording: synonyms,\n"
    "a different order of arguments, different sentence structure. Add no new\n"
    "facts and drop none.\n"
    "Answer STRICTLY in the format (no explanations, no quotes, no markdown):\n"
    "line 1 — the new title;\n"
    "remaining lines — the new body.\n"
    "\n"
    "Record:\n"
    "Title: {title}\n"
    "Body: {body}\n"
    "Tags: {tags}"
)

NEAR_TOPIC_PROMPT_RU: Final[str] = (
    "Ты генерируешь синтетические данные для детектора дубликатов заметок.\n"
    "Напиши НОВУЮ запись на ту же тему и в том же домене, что и запись ниже,\n"
    "но с ДРУГИМ фактом, решением или выводом: топикно близкая, а по содержанию —\n"
    "другая память. Не копируй формулировки оригинала и не повторяй его вывод.\n"
    "Ответь СТРОГО в формате (без пояснений, без кавычек, без markdown):\n"
    "строка 1 — заголовок новой записи;\n"
    "остальные строки — тело новой записи.\n"
    "\n"
    "Запись-ориентир:\n"
    "Заголовок: {title}\n"
    "Тело: {body}\n"
    "Теги: {tags}"
)

NEAR_TOPIC_PROMPT_EN: Final[str] = (
    "You generate synthetic data for a note-duplicate detector.\n"
    "Write a NEW record on the same topic and domain as the record below, but\n"
    "with a DIFFERENT fact, decision or conclusion: topically close, yet in\n"
    "content another memory. Do not copy the original wording and do not repeat\n"
    "its conclusion.\n"
    "Answer STRICTLY in the format (no explanations, no quotes, no markdown):\n"
    "line 1 — the new record's title;\n"
    "remaining lines — the new record's body.\n"
    "\n"
    "Reference record:\n"
    "Title: {title}\n"
    "Body: {body}\n"
    "Tags: {tags}"
)

#: Variant instruction appended for the 2nd+ occurrence of the same base
#: record (temperature 0 returns the same text for the same prompt — the
#: variant line is what decorrelates repeated bases). Formatted with {n}.
VARIANT_LINE_RU: Final[str] = (
    "ВАЖНО: это вариант №{n} для этой записи — используй другую лексику\n"
    "и другой порядок предложений, чтобы текст заметно отличался от предыдущих\n"
    "вариантов."
)
VARIANT_LINE_EN: Final[str] = (
    "IMPORTANT: this is variant #{n} for this record — use different vocabulary\n"
    "and a different sentence order so the text differs noticeably from the\n"
    "previous variants."
)


def _build_prompt(base: SynthRecord, *, paraphrase: bool, variant: int = 0) -> str:
    """Fill the language-matched template from the base record (RU↔RU, EN↔EN).

    ``variant > 0`` appends the numbered variant instruction (1-based
    numbering in the text; the bare template is variant 0's prompt).
    """
    if base.language == "en":
        template = PARAPHRASE_PROMPT_EN if paraphrase else NEAR_TOPIC_PROMPT_EN
        variant_line = VARIANT_LINE_EN
    else:
        template = PARAPHRASE_PROMPT_RU if paraphrase else NEAR_TOPIC_PROMPT_RU
        variant_line = VARIANT_LINE_RU
    prompt = template.format(
        title=base.title,
        body=base.body,
        tags=", ".join(base.tags) if base.tags else "-",
    )
    if variant > 0:
        prompt += "\n\n" + variant_line.format(n=variant + 1)
    return prompt


# ── LLM output validation (drop-with-counter, never silent repair) ────────────


#: Protocol markers some models write instead of following the bare
#: "line 1 title + body" contract (observed on 16/24 sides of the qwen2.5:7b
#: owner sample: «Заголовок:/Тело:/Теги:» and their EN twins). Case-insensitive;
#: ASCII and full-width colons.
_MARKER_TOKEN: Final[str] = r"(?P<token>заголовок|тело|теги|title|body|tags)"
MARKER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    _MARKER_TOKEN + r"\s*[:：]\s*", re.IGNORECASE
)
MARKER_LINE_RE: Final[re.Pattern[str]] = re.compile(
    _MARKER_TOKEN + r"\s*[:：]\s*", re.IGNORECASE
)


def strip_protocol_markers(raw: str) -> tuple[str, int]:
    """Strip protocol-marker noise from an LLM output; returns ``(text, strips)``.

    Deterministic PROTOCOL CLEANUP over the marker set
    {Заголовок, Тело, Теги, Title, Body, Tags} — counted in provenance
    (``llm_strips``), never a content repair; anything else still dies at
    validation. Rules, applied line by line (start-of-string is line 0):

    1. a line that is ONLY a marker (``"Тело:"``) is dropped whole;
    2. a line starting with a marker prefix loses the prefix
       (``"Заголовок: X"`` → ``"X"``, chained prefixes collapse);
       a ``Теги/Tags:``-prefixed line is dropped WHOLE — the model-invented
       tag list is protocol noise (the candidate inherits base tags).

    Mid-line markers are content and are never touched; an empty input maps
    to itself with zero strips.
    """
    if not raw:
        return raw, 0
    count = 0
    out: list[str] = []
    for line in raw.splitlines():
        work = line.strip()
        if not work:
            out.append(work)
            continue
        if MARKER_LINE_RE.fullmatch(work) is not None:
            count += 1
            continue
        match = MARKER_PREFIX_RE.match(work)
        if match is not None and match.group("token").lower() in ("теги", "tags"):
            count += 1
            continue  # model-invented tag list — dropped whole
        while match is not None:
            count += 1
            work = work[match.end() :]
            match = MARKER_PREFIX_RE.match(work)
        out.append(work)
    return "\n".join(out), count


def parse_llm_record(raw: str) -> tuple[str, str] | None:
    """Parse the ``title line + body`` protocol; None when output is empty.

    Line 1 is the title; the remaining non-empty lines are joined into the
    body. Markdown code fences are PROTOCOL NOISE, not content — lines that
    are bare fences (```` ``` ````, ```` ```text ````) are stripped before
    parsing; everything else dies at validation, nothing is "fixed up".
    """
    lines = [line.strip() for line in raw.strip().splitlines()]
    lines = [line for line in lines if line and not line.startswith("```")]
    if not lines:
        return None
    title = lines[0]
    body = " ".join(line for line in lines[1:] if line).strip()
    return title, body


def _norm_text(text: str) -> str:
    return " ".join(text.split()).lower()


def validate_candidate(base: SynthRecord, title: str, body: str) -> str | None:
    """Return a drop reason ("empty" | "length" | "identical") or None if OK."""
    if not title.strip() or not body.strip():
        return "empty"
    if not (MIN_TITLE_CHARS <= len(title) <= MAX_TITLE_CHARS):
        return "length"
    low = max(MIN_BODY_CHARS, int(len(base.body) * BODY_RATIO_MIN))
    high = max(low + 1, int(len(base.body) * BODY_RATIO_MAX))
    if not (low <= len(body) <= high):
        return "length"
    if _norm_text(f"{title}\n{body}") == _norm_text(f"{base.title}\n{base.body}"):
        return "identical"
    return None


# ── Serialization + fingerprint helpers (single source for CLI and tests) ─────


def label_name(label: int) -> str:
    """0/1 → the repo's manifest label convention (CLI train-manifest strings)."""
    if label == LABEL_DUPLICATE:
        return "duplicate"
    if label == LABEL_NOT_DUPLICATE:
        return "not-duplicate"
    raise ValueError(
        f"label must be {LABEL_DUPLICATE} or {LABEL_NOT_DUPLICATE}, got {label!r}"
    )


def _side_dict(record: SynthRecord) -> dict:
    return {
        "title": record.title,
        "body": record.body,
        "tags": list(record.tags),
        "language": record.language,
        "record_type": record.record_type,
    }


def pair_row(pair: SynthPair) -> dict:
    """The canonical JSONL row of one synth pair (docs/synth.md)."""
    return {
        "pair_id": pair.pair_id,
        "strategy": pair.strategy,
        "label": label_name(pair.label),
        "record": _side_dict(pair.record_a),
        "candidate": _side_dict(pair.record_b),
        "seed": pair.seed,
    }


def fingerprint_object(pair: SynthPair) -> dict:
    """The fingerprinted object: content + by-construction label.

    pair_id/strategy/seed are assignment and ride outside (data-contract §3
    discipline: ids are assignment, not content). ``similarity`` is absent —
    no store vector exists at generation time; the A2 docompute step will
    attach measured cosines as a DERIVED corpus with its own fingerprint.
    """
    return {
        "record": _side_dict(pair.record_a),
        "candidate": _side_dict(pair.record_b),
        "label": label_name(pair.label),
    }


def manifest_entries(pairs: Sequence[SynthPair]) -> list[tuple[str, str]]:
    """``(pair_id, pair_sha256)`` manifest lines (data-contract §5 scheme)."""
    return [(pair.pair_id, pair_sha256(fingerprint_object(pair))) for pair in pairs]


def manifest_bytes_of_pairs(pairs: Sequence[SynthPair]) -> bytes:
    """Manifest bytes for a synth corpus (sorted ``pair_id <sha256>`` lines)."""
    return manifest_bytes(manifest_entries(pairs))


def corpus_fingerprint_of_pairs(pairs: Sequence[SynthPair]) -> str:
    """BLAKE2b-256 corpus fingerprint over the manifest (data-contract §5)."""
    return corpus_fingerprint(manifest_bytes_of_pairs(pairs))


def pair_row_json(pair: SynthPair) -> str:
    """Canonical JSON text of one pair row (stable jsonl line)."""
    return canonical_json(pair_row(pair))


# ── Owner-validation sample quotas ────────────────────────────────────────────


def sample_quotas(n: int) -> dict[str, int]:
    """Split an owner-validation sample of ``n`` pairs across strategies.

    Each LLM strategy gets ceil(n/3) — the LLM output is what the owner is
    validating — and the two procedural strategies share the rest
    (broken-field takes the odd pair). Totals always equal ``n``:
    n=24 → 8/8/4/4 (the A2s sample gate). n < 3 is special-cased to the
    first ``n`` strategies in :data:`STRATEGIES` order.
    """
    if n < 0:
        raise ValueError(f"sample size must be >= 0, got {n}")
    if n == 0:
        return {strategy: 0 for strategy in STRATEGIES}
    if n < 3:
        quotas = {strategy: 0 for strategy in STRATEGIES}
        for strategy in STRATEGIES[:n]:
            quotas[strategy] = 1
        return quotas
    llm = -(-n // 3)
    rest = n - 2 * llm
    return {
        STRATEGY_PARAPHRASE: llm,
        STRATEGY_NEAR_TOPIC: llm,
        STRATEGY_BROKEN_FIELD: rest // 2 + rest % 2,
        STRATEGY_TRIVIAL_NEGATIVE: rest // 2,
    }


# ── Topic engine: 40 memory-domain seed topics, RU/EN halves ──────────────────


def _topic(
    title: str,
    body: str,
    tags: tuple[str, ...],
    language: str,
    record_type: str,
    key: str,
) -> SynthTopic:
    topic = SynthTopic(title, body, tags, language, record_type, key)
    if len(topic.body) < MIN_BODY_CHARS:
        raise SynthTopicError(f"topic body too short: {title!r}")
    if language not in ("ru", "en"):
        raise SynthTopicError(f"topic language must be ru|en: {title!r}")
    if (
        not key
        or not key.isascii()
        or not key.replace("-", "").isalnum()
        or key != key.lower()
    ):
        raise SynthTopicError(f"topic key must be a latin machine string: {key!r}")
    return topic


#: Deterministic seed list (frozen here; regenerating a corpus with a changed
#: list is a new corpus with a new fingerprint — provenance, not drift).
#: Layout: 20 RU topics then the SAME 20 as EN translations, same order —
#: matching positions share ``key`` (the translation-twin guard uses it).
TOPICS: Final[tuple[SynthTopic, ...]] = (
    # ── RU half ──
    _topic(
        "Настройка ruff для репы",
        "Вынесли конфиг ruff в pyproject: длина строки 100, включены правила E, F и I, форматтер запускается через pre-commit. Отступ четыре пробела, кавычки двойные.",
        ("tools", "linting"),
        "ru",
        "note",
        "ruff-config",
    ),
    _topic(
        "Решение: выбор очереди задач",
        "Выбрали Redis Streams вместо RabbitMQ: ноль новой инфраструктуры, конкурирующие консюмеры из коробки. Kafka отложили — нет операций на поддержку кластера.",
        ("decisions", "queue"),
        "ru",
        "note",
        "queue-choice",
    ),
    _topic(
        "Чекпоинт сессии 12 сентября",
        "Закрыли волну A3: контракты данных, алгоритмы D и N и CPU-smoke зелёные. Следующий шаг — синтетический корпус для претрейна. Блокеров нет.",
        ("checkpoint", "session"),
        "ru",
        "note",
        "session-checkpoint",
    ),
    _topic(
        "Переменные окружения dev-машины",
        "PATH дополняется через ~/.local/bin, OLLAMA_HOST держим 127.0.0.1:11434. Экспорты сессии не переживают новый терминал — постоянные значения переносим в .bashrc руками.",
        ("env", "config"),
        "ru",
        "fact",
        "env-vars",
    ),
    _topic(
        "Урок: тесты падали из-за таймзоны",
        "Нестабильные тесты на CI вылечились фиксацией TZ=UTC в конфиге pytest. Локально часы шли в MSK, а тест ждал UTC-метку — рассинхрон три часа.",
        ("lessons", "testing"),
        "ru",
        "note",
        "timezone-lesson",
    ),
    _topic(
        "Настройка ollama на ноутбуке",
        "Ollama слушает localhost:11434, модели лежат в /usr/share/ollama/.ollama. Модели на 7 миллиардов параметров хватает 8 ГБ памяти при контексте 4096.",
        ("tools", "ollama"),
        "ru",
        "fact",
        "ollama-setup",
    ),
    _topic(
        "Миграция БД: добавление индекса",
        "Миграция 0042 добавляет индекс на records(created_at), бэкфилл занимает около четырёх минут на трёх миллионах строк. Откат — DROP INDEX, данные не трогаются.",
        ("db", "migration"),
        "ru",
        "note",
        "db-migration-index",
    ),
    _topic(
        "Ревью PR 118: замечания",
        "Просили разбить диф на три коммита, вынести магическую константу таймаута в конфиг и добавить тест на пустой ввод. Мёртвый код в helpers удалить.",
        ("review", "pr"),
        "ru",
        "note",
        "pr-review",
    ),
    _topic(
        "Конфиг Docker-сборки движка",
        "Базовый образ ubuntu 24.04, сборка в два этапа: у сборщика есть gcc, у рантайма нет. Порт 8080 наружу не публикуем, только unix-сокет.",
        ("docker", "config"),
        "ru",
        "fact",
        "docker-build",
    ),
    _topic(
        "Инцидент: переполнение диска",
        "Диск заполнили логи ollama — сорок гигабайт за неделю. Вылечили ротацией logrotate: лимит два гигабайта, хранение семь дней. Проверку df добавили в cron.",
        ("incident", "ops"),
        "ru",
        "note",
        "disk-incident",
    ),
    _topic(
        "Хуки git: запрет коммитов в main",
        "Pre-commit хук блокирует прямой коммит в main и требует английское сообщение вида type(scope): описание. Обходной путь — флаг --no-verify, это осознанно.",
        ("git", "hooks"),
        "ru",
        "fact",
        "git-hooks",
    ),
    _topic(
        "План на неделю 40",
        "Приоритеты: закончить генератор корпуса, прогнать сэмпл из двадцати четырёх пар, отдать владельцу на валидацию. Вторая очередь — докомпьют полевых косинусов.",
        ("planning", "week"),
        "ru",
        "note",
        "week-plan",
    ),
    _topic(
        "Заметка про кэш эмбеддингов",
        "Кэш векторов живёт рядом со стором, ключ — fingerprint эмбеддера плюс хеш текста. Смена модели инвалидирует кэш целиком, это событие перекалибровки.",
        ("cache", "embeddings"),
        "ru",
        "note",
        "embedding-cache",
    ),
    _topic(
        "Настройка CI: матрица pytest",
        "CI гоняет pytest на питоне 3.12 и 3.13, uv-артефакты кэшируются. Тяжёлые тесты помечены словом slow и запускаются только ночным расписанием.",
        ("ci", "testing"),
        "ru",
        "fact",
        "ci-pytest-matrix",
    ),
    _topic(
        "Соглашение по тегам памяти",
        "Теги латиницей, до трёх на запись: домен, тип, статус. Кириллические теги разрешены только для русских заметок о встречах.",
        ("tags", "memory"),
        "ru",
        "fact",
        "tag-convention",
    ),
    _topic(
        "Документация API: метод /recall",
        "Эндпоинт /recall принимает query и число k, возвращает записи с косинусом выше порога 0.3. Ошибки отдаём по RFC-7807, лимит шестьдесят запросов в минуту.",
        ("api", "docs"),
        "ru",
        "note",
        "api-recall",
    ),
    _topic(
        "Настройки эмбеддера в движке",
        "mnema-embed-v1: вектора 384 размерности, float32, нормализация L2. Пин эмбеддера зашит в манифест стора, смена проходит только через перекалибровку.",
        ("embeddings", "engine"),
        "ru",
        "fact",
        "embedder-settings",
    ),
    _topic(
        "Ретроспектива волны A2",
        "Вышло хорошо: контракты заморозили до кода. Отстали на полдня из-за спора о формате pair_id, спор закрыло голосование комитета.",
        ("retro", "wave"),
        "ru",
        "note",
        "wave-retro",
    ),
    _topic(
        "Доступ по SSH к серверу сборки",
        "Ключ ed25519, парольную фразу держит агент. В ssh-config сборочный хост доступен под именем build-box на порту 2222, прямой root-вход запрещён.",
        ("ssh", "access"),
        "ru",
        "fact",
        "ssh-build-server",
    ),
    _topic(
        "Оптимизация запроса поиска",
        "Поиск тормозил на LIKE с процентами — переехали на FTS5-таблицу с русским стеммером. Время запроса упало с восьмисот миллисекунд до двенадцати на трёх миллионах документов.",
        ("db", "performance"),
        "ru",
        "note",
        "search-tuning",
    ),
    # ── EN half (translations of the RU half, same order, same keys) ──
    _topic(
        "Ruff configuration for the repo",
        "The ruff config lives in pyproject: line length 100, rules E, F and I enabled, the formatter runs through pre-commit. Indent is four spaces, double quotes.",
        ("tools", "linting"),
        "en",
        "note",
        "ruff-config",
    ),
    _topic(
        "Decision: task queue choice",
        "We picked Redis Streams over RabbitMQ: zero new infrastructure, competing consumers out of the box. Kafka is postponed — there is no ops budget for a cluster.",
        ("decisions", "queue"),
        "en",
        "note",
        "queue-choice",
    ),
    _topic(
        "Session checkpoint September 18",
        "Wave B is closed: schema freeze, importer green, docs synced. The next step is the eval corpus export. No blockers recorded this session.",
        ("checkpoint", "session"),
        "en",
        "note",
        "session-checkpoint",
    ),
    _topic(
        "Dev machine environment variables",
        "PATH gets ~/.local/bin appended, OLLAMA_HOST stays at 127.0.0.1:11434. Session exports do not survive a new terminal — permanent values move into .bashrc by hand.",
        ("env", "config"),
        "en",
        "fact",
        "env-vars",
    ),
    _topic(
        "Lesson: tests failed over a timezone",
        "Flaky CI tests were fixed by pinning TZ=UTC in the pytest config. The local clock ran MSK while the test expected a UTC stamp — a three hour skew.",
        ("lessons", "testing"),
        "en",
        "note",
        "timezone-lesson",
    ),
    _topic(
        "Ollama setup on the workstation",
        "Ollama listens on localhost:11434, models live under /usr/share/ollama/.ollama. A 7B parameter model fits into 8 GB of RAM with the context capped at 4096.",
        ("tools", "ollama"),
        "en",
        "fact",
        "ollama-setup",
    ),
    _topic(
        "DB migration: adding an index",
        "Migration 0042 adds an index on records(created_at); the backfill takes about four minutes over three million rows. Rollback is DROP INDEX, no data is touched.",
        ("db", "migration"),
        "en",
        "note",
        "db-migration-index",
    ),
    _topic(
        "PR review 204: findings",
        "We asked to split the diff into three commits, move the magic timeout constant into config, and add a test for empty input. Dead code in helpers goes away.",
        ("review", "pr"),
        "en",
        "note",
        "pr-review",
    ),
    _topic(
        "Engine Docker build config",
        "Base image ubuntu 24.04, a two-stage build: the builder carries gcc, the runtime does not. Port 8080 is never published — a unix socket only.",
        ("docker", "config"),
        "en",
        "fact",
        "docker-build",
    ),
    _topic(
        "Incident: disk full",
        "Ollama logs filled the disk — forty gigabytes in a week. Fixed with logrotate: a two gigabyte cap and seven day retention. A df check was added to cron.",
        ("incident", "ops"),
        "en",
        "note",
        "disk-incident",
    ),
    _topic(
        "Git hooks: blocking main commits",
        "A pre-commit hook blocks direct commits to main and enforces English messages of the form type(scope): description. The escape hatch is --no-verify, deliberately.",
        ("git", "hooks"),
        "en",
        "fact",
        "git-hooks",
    ),
    _topic(
        "Plan for week 41",
        "Priorities: finish the corpus generator, run the twenty-four pair sample, hand it to the owner for validation. Second in line is the field-cosine docompute.",
        ("planning", "week"),
        "en",
        "note",
        "week-plan",
    ),
    _topic(
        "Note on the embedding cache",
        "The vector cache sits next to the store, keyed by the embedder fingerprint plus the text hash. Swapping the model invalidates the whole cache — a recalibration event.",
        ("cache", "embeddings"),
        "en",
        "note",
        "embedding-cache",
    ),
    _topic(
        "CI setup: the pytest matrix",
        "CI runs pytest on Python 3.12 and 3.13 with uv artifacts cached. Heavy tests are marked slow and only run on the nightly schedule.",
        ("ci", "testing"),
        "en",
        "fact",
        "ci-pytest-matrix",
    ),
    _topic(
        "Memory tag convention",
        "Tags are latin, up to three per record: domain, type, status. Cyrillic tags are allowed only for Russian meeting notes.",
        ("tags", "memory"),
        "en",
        "fact",
        "tag-convention",
    ),
    _topic(
        "API docs: the /recall method",
        "The /recall endpoint takes a query and k, and returns records with cosine above the 0.3 threshold. Errors follow RFC-7807; the rate limit is sixty requests per minute.",
        ("api", "docs"),
        "en",
        "note",
        "api-recall",
    ),
    _topic(
        "Embedder settings in the engine",
        "mnema-embed-v1: 384-dim vectors, float32, L2 normalization. The embedder pin is baked into the store manifest; changing it goes through recalibration.",
        ("embeddings", "engine"),
        "en",
        "fact",
        "embedder-settings",
    ),
    _topic(
        "Wave A2 retrospective",
        "What worked: contracts were frozen before code. We slipped half a day arguing about the pair_id format — the committee vote settled it.",
        ("retro", "wave"),
        "en",
        "note",
        "wave-retro",
    ),
    _topic(
        "SSH access to the build server",
        "An ed25519 key with an agent-held passphrase. The ssh config exposes the build host as build-box on port 2222; direct root login is disabled.",
        ("ssh", "access"),
        "en",
        "fact",
        "ssh-build-server",
    ),
    _topic(
        "Search query tuning",
        "Search was slow on LIKE with wildcards — we moved to an FTS5 table with a Russian stemmer. Query time dropped from eight hundred ms to twelve on three million docs.",
        ("db", "performance"),
        "en",
        "note",
        "search-tuning",
    ),
)

_TOPIC_RECORDS: Final[tuple[SynthRecord, ...]] = tuple(t.as_record() for t in TOPICS)

#: Meaning-preserving procedural variations applied ON TOP of an LLM
#: paraphrase (composition over the existing corruption weak positives).
_WEAK_ON_TOP: Final[tuple] = (
    ShuffleTags(),
    CollapseWhitespaceBody(),
    NormalizePunctuationBody(),
    SwapCaseTitle(),
)

_DONOR_FIELD_BY_TRANSFORM: Final[dict[str, str]] = {
    "swap_title": "title",
    "swap_body": "body",
    "swap_tags": "tags",
}


def _pick_donor(
    rng: random.Random, base: SynthRecord, pool: Sequence[SynthRecord], field: str
) -> SynthRecord | None:
    """Deterministic donor search mirroring the corruption semantics: the
    donor must differ in ``field`` and not be content-identical to base."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    for index in order:
        donor = pool[index]
        if getattr(donor, field) != getattr(base, field) and record_key(
            donor
        ) != record_key(base):
            return donor
    return None


def _broken_field_transform(
    rng: random.Random, base: SynthRecord, pool: Sequence[SynthRecord]
) -> object:
    """One hard-negative transform composed over the corruption registry."""
    name = rng.choice(HARD_NEGATIVE_TRANSFORMS)
    field = _DONOR_FIELD_BY_TRANSFORM.get(name)
    if field is not None:
        donor = _pick_donor(rng, base, pool, field)
        if donor is None:  # degenerate pool → guaranteed-change fallback
            return FlipLanguage()
        return SubstituteField(name, field, donor)
    return FlipLanguage() if name == "flip_language" else FlipRecordType()


def _different_topic(
    keys: Sequence[str],
    index_a: int,
    index_b: int,
    record_a: SynthRecord,
    record_b: SynthRecord,
) -> bool:
    """True iff the two pool records are genuinely DIFFERENT memories.

    Same non-empty topic key (translation twins) → same memory → False.
    Empty keys (custom topic lists) fall back to content inequality.
    """
    if keys[index_a] and keys[index_b] and keys[index_a] == keys[index_b]:
        return False
    return record_key(record_a) != record_key(record_b)


def _resolve_quotas(
    pairs_per_strategy: int, quotas: Mapping[str, int] | None
) -> dict[str, int]:
    if pairs_per_strategy < 0:
        raise ValueError(f"pairs_per_strategy must be >= 0, got {pairs_per_strategy}")
    resolved = {strategy: pairs_per_strategy for strategy in STRATEGIES}
    if quotas is not None:
        unknown = set(quotas) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies in quotas: {sorted(unknown)}")
        for strategy, quota in quotas.items():
            if quota < 0:
                raise ValueError(f"quota for {strategy!r} must be >= 0, got {quota}")
            resolved[strategy] = quota
    return resolved


def _llm_candidate(
    base: SynthRecord,
    *,
    paraphrase: bool,
    variant: int,
    llm_fn: Callable[[str], str],
    stats: SynthStats,
) -> SynthRecord | None:
    """One LLM call + parse + corridor validation. None ⇒ dropped (counted).

    Transport failures propagate — the injected callback may raise, and a
    backend outage is NOT a droppable pair (the caller aborts loudly).
    """
    stats.llm_calls += 1
    raw = llm_fn(_build_prompt(base, paraphrase=paraphrase, variant=variant))
    clean, strips = strip_protocol_markers(raw)
    stats.llm_strips += strips
    parsed = parse_llm_record(clean)
    if parsed is None:
        stats.drop("empty")
        return None
    title, body = parsed
    reason = validate_candidate(base, title, body)
    if reason is not None:
        stats.drop(reason)
        return None
    return SynthRecord(
        title=title,
        body=body,
        tags=base.tags,
        language=base.language,
        record_type=base.record_type,
    )


def _generate_llm_strategy_batched(
    strategy: str,
    quota: int,
    ordinal: int,
    pairs_per_strategy: int,
    pool: Sequence[SynthRecord],
    master: random.Random,
    llm_batch_fn: Callable[[Sequence[str]], Sequence[str]],
    batch_size: int,
    counters: SynthStats,
    seen_ids: set[str],
    next_variant: dict[tuple[str, int], int],
) -> list[SynthPair]:
    """One LLM strategy through the BATCHED callback, in rounds (cloud path).

    Mirrors the sequential slot loop's semantics: monotonic variant numbers
    per base, at most MAX_ATTEMPTS prompts per slot, corridor drops are NOT
    retried (temperature 0 cannot fix validation), pair-id collisions retry
    in the NEXT round with a bumped variant and drop with a counter on
    exhaustion. Pair seeds are drawn from ``master`` in (round, slot) order —
    a collision-free, corridor-free run draws exactly the same sequence as
    the sequential loop, so equal prompts ⇒ equal corpus.
    """
    paraphrase = strategy == STRATEGY_PARAPHRASE
    base_indices = [
        (ordinal * pairs_per_strategy + i) % len(pool) for i in range(quota)
    ]
    sent: dict[int, int] = {i: 0 for i in range(quota)}
    corridor_dropped: set[int] = set()
    duplicate_dropped: set[int] = set()
    results: dict[int, SynthPair] = {}

    for _round in range(MAX_ATTEMPTS):
        active = [
            i
            for i in range(quota)
            if i not in results
            and i not in corridor_dropped
            and i not in duplicate_dropped
            and sent[i] < MAX_ATTEMPTS
        ]
        if not active:
            break
        jobs: list[tuple[int, SynthRecord, int, str]] = []
        for i in active:
            base = pool[base_indices[i]]
            pair_seed = master.getrandbits(32)
            variant_key = (strategy, base_indices[i])
            variant = next_variant.get(variant_key, 0)
            next_variant[variant_key] = variant + 1
            jobs.append(
                (
                    i,
                    base,
                    pair_seed,
                    _build_prompt(base, paraphrase=paraphrase, variant=variant),
                )
            )
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            counters.llm_calls += len(chunk)
            counters.llm_batches += 1
            outputs = llm_batch_fn([job[3] for job in chunk])
            if len(outputs) != len(chunk):
                raise ValueError(
                    f"llm_batch_fn returned {len(outputs)} outputs for "
                    f"{len(chunk)} prompts — a batched callback must keep "
                    "count and order"
                )
            for (i, base, pair_seed, _prompt), raw in zip(chunk, outputs):
                sent[i] += 1
                clean, strips = strip_protocol_markers(raw)
                counters.llm_strips += strips
                parsed = parse_llm_record(clean)
                if parsed is None:
                    counters.drop("empty")
                    corridor_dropped.add(i)
                    continue
                title, body = parsed
                reason = validate_candidate(base, title, body)
                if reason is not None:
                    counters.drop(reason)
                    corridor_dropped.add(i)
                    continue
                candidate = SynthRecord(
                    title=title,
                    body=body,
                    tags=base.tags,
                    language=base.language,
                    record_type=base.record_type,
                )
                if paraphrase:
                    pair_rng = random.Random(pair_seed)
                    candidate = pair_rng.choice(_WEAK_ON_TOP).apply(candidate, pair_rng)
                label = LABEL_DUPLICATE if paraphrase else LABEL_NOT_DUPLICATE
                try:
                    results[i] = _emit(
                        seen_ids, base, candidate, label, strategy, pair_seed
                    )
                except ValueError:
                    if sent[i] >= MAX_ATTEMPTS:
                        duplicate_dropped.add(i)

    for _slot in duplicate_dropped:
        counters.drop("duplicate")
    return [results[i] for i in sorted(results)]


def generate_corpus(
    topics: Sequence[SynthTopic],
    pairs_per_strategy: int,
    llm_fn: Callable[[str], str] | None,
    seed: int,
    *,
    quotas: Mapping[str, int] | None = None,
    stats: SynthStats | None = None,
    llm_batch_fn: Callable[[Sequence[str]], Sequence[str]] | None = None,
    batch_size: int = 8,
    emit_callback: Callable[[SynthPair], None] | None = None,
) -> list[SynthPair]:
    """Generate the stage-1 synth corpus (ADR 0001 П3) — pure, no network.

    Args:
        topics: seed topics (TOPICS by convention, passed explicitly for
            testability); cycled deterministically per strategy.
        pairs_per_strategy: uniform quota per strategy (overridden per
            strategy by ``quotas``).
        llm_fn: INJECTED text generator ``prompt -> output`` (sequential
            mode, local ollama); required when any LLM-strategy quota is
            positive AND ``llm_batch_fn`` is None. The callback may raise on
            transport failures — that aborts the generation loudly (a
            transport outage is not a droppable pair).
        llm_batch_fn: INJECTED batched generator ``[prompt] -> [output]``
            (cloud mode): receives a list of record prompts, returns the
            SAME number of raw outputs in the same order. Exactly one of
            ``llm_fn``/``llm_batch_fn`` must be supplied when an LLM
            strategy quota is positive. Failure semantics match ``llm_fn``.
        batch_size: max prompts per ``llm_batch_fn`` call (cloud batching,
            6–8 keeps one chat/completions call within output limits).
        emit_callback: optional ``SynthPair -> None`` hook called for every
            emitted pair IN EMISSION ORDER — the transport-side durability
            checkpoint (script-side partial-corpus journal).
        seed: master seed; every pair draws its own sub-seed from it.
        quotas: optional per-strategy override (strategy name → quota).
        stats: optional counter sink (mutated in place, returned via
            :meth:`SynthStats.as_dict` by the caller).

    Returns:
        list[SynthPair] in fixed strategy order; drop events are counted in
        ``stats`` (or discarded when not supplied) — never silently repaired.

    Raises:
        ValueError: on negative quotas, unknown strategy names, an empty
            topic list while a quota is positive, a missing ``llm_fn``/
            ``llm_batch_fn`` for an LLM strategy, both callbacks at once,
            a ``batch_size`` < 1, or a batched callback that breaks
            count/order. Duplicate pair ids survive MAX_ATTEMPTS
            retries only as a counted drop — a raise here would mean the
            retry mechanics itself is broken.
    """
    resolved = _resolve_quotas(pairs_per_strategy, quotas)
    topic_list = tuple(topics)
    if any(resolved[s] > 0 for s in STRATEGIES) and not topic_list:
        raise ValueError("topics is empty but at least one quota is positive")
    if any(resolved[s] > 0 for s in LLM_STRATEGIES):
        if llm_fn is None and llm_batch_fn is None:
            raise ValueError(
                f"llm_fn (or llm_batch_fn) is required for LLM strategies "
                f"{list(LLM_STRATEGIES)} (src/cortex is network-free — "
                "inject the callback)"
            )
        if llm_fn is not None and llm_batch_fn is not None:
            raise ValueError(
                "pass either llm_fn or llm_batch_fn — one callback mode per run"
            )
    if llm_batch_fn is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    counters = stats if stats is not None else SynthStats()
    master = random.Random(seed)
    pool = tuple(topic.as_record() for topic in topic_list)
    keys = tuple(topic.key for topic in topic_list)
    pairs: list[SynthPair] = []
    seen_ids: set[str] = set()
    # (strategy, base_index) → next variant number to send for this base.
    # MONOTONIC: every prompt sent for a base is unique (temperature 0 maps
    # equal prompts to equal outputs — repeating one would waste an LLM call
    # on a guaranteed collision). Retry attempts draw the next number.
    next_variant: dict[tuple[str, int], int] = {}

    for ordinal, strategy in enumerate(STRATEGIES):
        quota = resolved[strategy]
        if quota == 0:
            continue
        if strategy in LLM_STRATEGIES and llm_batch_fn is not None:
            for pair in _generate_llm_strategy_batched(
                strategy,
                quota,
                ordinal,
                pairs_per_strategy,
                pool,
                master,
                llm_batch_fn,
                batch_size,
                counters,
                seen_ids,
                next_variant,
            ):
                pairs.append(pair)
                counters.emitted += 1
                if emit_callback is not None:
                    emit_callback(pair)
            continue
        for i in range(quota):
            base_index = (ordinal * pairs_per_strategy + i) % len(pool)
            base = pool[base_index]
            counters.attempts += 1
            emitted: SynthPair | None = None
            # Why the slot produced nothing: "corridor" (LLM output failed
            # validation — ALREADY counted inside _llm_candidate),
            # "same-topic" (no honest negative partner exists),
            # "duplicate" (pair-id collision survived MAX_ATTEMPTS retries).
            slot_drop: str | None = None

            for attempt in range(MAX_ATTEMPTS):
                pair_seed = master.getrandbits(32)
                pair_rng = random.Random(pair_seed)

                if strategy == STRATEGY_PARAPHRASE:
                    variant_key = (strategy, base_index)
                    variant = next_variant.get(variant_key, 0)
                    next_variant[variant_key] = variant + 1
                    candidate = _llm_candidate(
                        base,
                        paraphrase=True,
                        variant=variant,
                        llm_fn=llm_fn,
                        stats=counters,  # type: ignore[arg-type]
                    )
                    if candidate is None:
                        slot_drop = "corridor"
                        break  # retry at temperature 0 cannot fix a corridor drop
                    record_b = candidate
                    weak = pair_rng.choice(_WEAK_ON_TOP)
                    record_b = weak.apply(record_b, pair_rng)
                    label = LABEL_DUPLICATE

                elif strategy == STRATEGY_NEAR_TOPIC:
                    variant_key = (strategy, base_index)
                    variant = next_variant.get(variant_key, 0)
                    next_variant[variant_key] = variant + 1
                    candidate = _llm_candidate(
                        base,
                        paraphrase=False,
                        variant=variant,
                        llm_fn=llm_fn,
                        stats=counters,  # type: ignore[arg-type]
                    )
                    if candidate is None:
                        slot_drop = "corridor"
                        break
                    record_b = candidate
                    label = LABEL_NOT_DUPLICATE

                elif strategy == STRATEGY_BROKEN_FIELD:
                    record_b = _broken_field_transform(pair_rng, base, pool).apply(
                        base, pair_rng
                    )  # type: ignore[attr-defined]
                    label = LABEL_NOT_DUPLICATE

                else:  # STRATEGY_TRIVIAL_NEGATIVE — different topics, procedural
                    partner = None
                    offset = i + attempt + 1  # retry shifts the scan start
                    for j in range(len(pool)):
                        partner_index = (base_index + offset + j) % len(pool)
                        candidate = pool[partner_index]
                        if _different_topic(
                            keys, base_index, partner_index, base, candidate
                        ):
                            partner = candidate
                            break
                    if partner is None:
                        # The scan covers the whole pool — no retry can help:
                        # every other record is the same memory (degenerate list).
                        slot_drop = "same-topic"
                        break
                    record_b = partner
                    label = LABEL_NOT_DUPLICATE

                try:
                    emitted = _emit(
                        seen_ids, base, record_b, label, strategy, pair_seed
                    )
                    break
                except ValueError:
                    if attempt == MAX_ATTEMPTS - 1:
                        slot_drop = "duplicate"
                        break
                    continue  # bump variant / re-draw and try again

            if emitted is None:
                if slot_drop is not None and slot_drop != "corridor":
                    counters.drop(slot_drop)
                continue

            pairs.append(emitted)
            counters.emitted += 1
            if emit_callback is not None:
                emit_callback(emitted)

    return pairs


def _emit(
    seen_ids: set[str],
    record_a: SynthRecord,
    record_b: SynthRecord,
    label: int,
    strategy: str,
    pair_seed: int,
) -> SynthPair:
    """Build the pair, enforce pair-id uniqueness."""
    pair_id = f"{record_key(record_a)}--{record_key(record_b)}"
    if pair_id in seen_ids:
        raise ValueError(
            f"duplicate synthetic pair id {pair_id!r} (strategy {strategy!r}) — "
            "the generator must never emit the same content pair twice"
        )
    seen_ids.add(pair_id)
    return SynthPair(
        pair_id=pair_id,
        record_a=record_a,
        record_b=record_b,
        label=label,
        strategy=strategy,
        seed=pair_seed,
    )
