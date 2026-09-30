"""A2s synth corpus generator — contract tests (no network, no ollama).

Covers the ADR 0001 addendum П3 generator: labels by construction per
strategy, drop-with-counter validation, seed determinism of the procedural
part, translation-twin guard, repeat-basis variant/retry mechanics, feature
path compatibility and the data-contract §5 fingerprint scheme. The CLI
wrapper (scripts/gen_synth_corpus.py) is tested for its argument surface
only — the LLM backend itself is exercised by the owner-validation sample
run, never in tests.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cortex.features.pair import FEATURE_NAMES, features
from cortex.synth import (
    MAX_ATTEMPTS,
    STRATEGY_BROKEN_FIELD,
    STRATEGY_NEAR_TOPIC,
    STRATEGY_PARAPHRASE,
    STRATEGY_TRIVIAL_NEGATIVE,
    SynthStats,
    TOPICS,
    SynthTopic,
    corpus_fingerprint_of_pairs,
    generate_corpus,
    manifest_bytes_of_pairs,
    manifest_entries,
    pair_row,
    pair_row_json,
    parse_llm_record,
    sample_quotas,
    validate_candidate,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_synth_corpus.py"


def _topic(title: str, body: str, language: str, key: str) -> SynthTopic:
    return SynthTopic(
        title=title,
        body=body,
        tags=("test", "memory"),
        language=language,
        record_type="note",
        key=key,
    )


#: Small deterministic topic list (bodies pass the 30-char topic floor).
TOPICS_TEST = (
    _topic("Настройка линтера", "Вынесли конфиг линтера в pyproject, длина строки сто, правило импортов включено.", "ru", "linter"),
    _topic("Выбор очереди", "Выбрали легковесную очередь вместо брокера сообщений: ноль новой инфраструктуры, консюмеры из коробки.", "ru", "queue"),
    _topic("Linter configuration", "The linter config lives in pyproject, line length one hundred, the import rule is enabled.", "en", "linter"),
    _topic("Queue choice", "We picked a lightweight queue over a message broker: zero new infrastructure, consumers out of the box.", "en", "queue"),
)


def _extract_base(prompt: str) -> tuple[str, str]:
    """Pull the record back out of a prompt (test-side prompt protocol)."""
    title = re.search(r"(?:Заголовок|Title): (.*)", prompt).group(1)
    body = re.search(r"(?:Тело|Body): (.*)", prompt).group(1)
    return title, body


def _mock_llm(prompt: str) -> str:
    """Deterministic mock: distinct rewrite per strategy and variant number."""
    title, _ = _extract_base(prompt)
    variant_match = re.search(r"вариант №(\d+)|variant #(\d+)", prompt)
    variant = (variant_match.group(1) or variant_match.group(2)) if variant_match else "1"
    paraphrase = ("Перепиши" in prompt) or ("Rewrite" in prompt)
    russian = "Перепиши" in prompt or "НОВУЮ" in prompt
    kind = "пересказ" if paraphrase else "новая память"
    if russian:
        new_title = f"{kind} вариант {variant}: {title[::-1]}"
        body = (
            f"Перефразированный текст варианта {variant}: смысл сохранён, "
            "формулировки и порядок аргументов изменены, добавлены синонимы."
        )
    else:
        new_title = f"{kind} variant {variant}: {title[::-1]}"
        body = (
            f"Rewritten text of variant {variant}: meaning kept, wording and "
            "argument order changed, synonyms used throughout the record."
        )
    return f"{new_title}\n{body}"


# ── sample quotas ─────────────────────────────────────────────────────────────


def test_sample_quotas_match_the_a2s_gate() -> None:
    assert sample_quotas(24) == {
        STRATEGY_PARAPHRASE: 8,
        STRATEGY_NEAR_TOPIC: 8,
        STRATEGY_BROKEN_FIELD: 4,
        STRATEGY_TRIVIAL_NEGATIVE: 4,
    }


@pytest.mark.parametrize("n", list(range(0, 61)))
def test_sample_quotas_total_exactly_n(n: int) -> None:
    assert sum(sample_quotas(n).values()) == n


def test_sample_quotas_reject_negative() -> None:
    with pytest.raises(ValueError):
        sample_quotas(-1)


# ── labels by construction ────────────────────────────────────────────────────


def test_strategies_carry_labels_by_construction() -> None:
    pairs = generate_corpus(
        TOPICS_TEST, 0, _mock_llm, 11, quotas={s: 2 for s in (STRATEGY_PARAPHRASE, STRATEGY_NEAR_TOPIC, STRATEGY_BROKEN_FIELD, STRATEGY_TRIVIAL_NEGATIVE)}
    )
    labels = {pair.strategy: pair.label for pair in pairs}
    assert labels == {
        STRATEGY_PARAPHRASE: 1,
        STRATEGY_NEAR_TOPIC: 0,
        STRATEGY_BROKEN_FIELD: 0,
        STRATEGY_TRIVIAL_NEGATIVE: 0,
    }
    for pair in pairs:
        if pair.strategy == STRATEGY_BROKEN_FIELD:
            # hard negative: SAME text, one field broken — the flip may be
            # exactly the language/type field, so equality is NOT expected
            continue
        # closeness by design: both sides share topic tags/language/type
        assert pair.record_b.language == pair.record_a.language
        assert pair.record_b.record_type == pair.record_a.record_type


def test_llm_strategies_reuse_base_tags() -> None:
    pairs = generate_corpus(
        TOPICS_TEST, 0, _mock_llm, 3, quotas={STRATEGY_PARAPHRASE: 2, STRATEGY_NEAR_TOPIC: 2}
    )
    for pair in pairs:
        assert pair.record_b.tags == pair.record_a.tags


def test_paraphrase_applies_meaning_free_variation_on_top() -> None:
    seen = set()
    for seed in range(20):
        pairs = generate_corpus(
            TOPICS_TEST, 0, _mock_llm, seed, quotas={STRATEGY_PARAPHRASE: 4}
        )
        for pair in pairs:
            record = pair.record_b
            seen.add(record.title + "|" + record.body + "|" + ",".join(record.tags))
    # the procedural layer must actually vary the LLM output: on clean text
    # that is swapcase + tag order — for 4 bases that is > 4 distinct records
    assert len(seen) > 8


# ── drop-with-counter validation ──────────────────────────────────────────────


def test_empty_llm_output_drops_with_counter() -> None:
    stats = SynthStats()
    pairs = generate_corpus(
        TOPICS_TEST, 0, lambda _prompt: "", 5, quotas={STRATEGY_PARAPHRASE: 2}, stats=stats
    )
    assert pairs == []
    assert stats.as_dict()["dropped_empty"] == 2
    assert stats.as_dict()["llm_calls"] == 2


def test_echo_llm_output_dropped_as_identical() -> None:
    def echo(prompt: str) -> str:
        title, body = _extract_base(prompt)
        return f"{title}\n{body}"

    stats = SynthStats()
    pairs = generate_corpus(
        TOPICS_TEST, 0, echo, 5, quotas={STRATEGY_NEAR_TOPIC: 2}, stats=stats
    )
    assert pairs == []
    assert stats.as_dict()["dropped_identical"] == 2


def test_oversized_llm_body_dropped_by_corridor() -> None:
    def bloated(prompt: str) -> str:
        title, _ = _extract_base(prompt)
        return f"{title} переписан\n" + ("длинное предложение про то же самое. " * 40)

    stats = SynthStats()
    pairs = generate_corpus(
        TOPICS_TEST, 0, bloated, 5, quotas={STRATEGY_PARAPHRASE: 1}, stats=stats
    )
    assert pairs == []
    assert stats.as_dict()["dropped_length"] == 1


def test_corridor_bounds() -> None:
    base = TOPICS_TEST[0].as_record()
    assert validate_candidate(base, "ab", "тело") == "length"  # title too short
    assert validate_candidate(base, "Хороший заголовок", "коротко") == "length"
    assert validate_candidate(base, "  ", "тело записи") == "empty"
    assert validate_candidate(base, base.title, base.body) == "identical"
    ok = validate_candidate(base, "Другой заголовок записи", "Другое тело записи той же памяти, смысл тот же самый.")
    assert ok is None


def test_llm_fn_required_for_llm_strategies() -> None:
    with pytest.raises(ValueError, match="llm_fn"):
        generate_corpus(TOPICS_TEST, 2, None, 1)
    # procedural-only quotas never need the callback
    generate_corpus(TOPICS_TEST, 0, None, 1, quotas={STRATEGY_BROKEN_FIELD: 2})


def test_transport_failure_aborts_loudly() -> None:
    def broken(_prompt: str) -> str:
        raise ConnectionError("backend down")

    with pytest.raises(ConnectionError):
        generate_corpus(TOPICS_TEST, 0, broken, 1, quotas={STRATEGY_NEAR_TOPIC: 1})


# ── determinism + repeat-basis mechanics ──────────────────────────────────────


def test_procedural_strategies_are_seed_deterministic() -> None:
    quotas = {STRATEGY_PARAPHRASE: 0, STRATEGY_NEAR_TOPIC: 0, STRATEGY_BROKEN_FIELD: 12, STRATEGY_TRIVIAL_NEGATIVE: 12}
    first = [pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)]
    second = [pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)]
    assert first == second
    other = generate_corpus(TOPICS_TEST, 0, None, 43, quotas=quotas)
    assert [p.seed for p in other] != [p.seed for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)]


def test_llm_generation_is_seed_deterministic() -> None:
    quotas = {STRATEGY_PARAPHRASE: 3, STRATEGY_NEAR_TOPIC: 3, STRATEGY_BROKEN_FIELD: 0, STRATEGY_TRIVIAL_NEGATIVE: 0}
    first = [pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, _mock_llm, 9, quotas=quotas)]
    second = [pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, _mock_llm, 9, quotas=quotas)]
    assert first == second


def test_translation_twins_never_form_a_negative() -> None:
    pairs = generate_corpus(
        TOPICS, 0, None, 7, quotas={STRATEGY_TRIVIAL_NEGATIVE: 40, STRATEGY_BROKEN_FIELD: 0, STRATEGY_PARAPHRASE: 0, STRATEGY_NEAR_TOPIC: 0}
    )
    key_by_title = {topic.title: topic.key for topic in TOPICS}
    for pair in pairs:
        assert key_by_title[pair.record_a.title] != key_by_title[pair.record_b.title]


def test_full_scale_procedural_run_has_no_collisions() -> None:
    """quota > pool size must not crash (old behavior raised on pair ids)."""
    pairs = generate_corpus(
        TOPICS, 0, None, 7, quotas={STRATEGY_BROKEN_FIELD: 40, STRATEGY_TRIVIAL_NEGATIVE: 40}
    )
    ids = [pair.pair_id for pair in pairs]
    assert len(set(ids)) == len(ids) == 80


def test_variant_prompts_rescue_repeated_bases() -> None:
    """One topic, three slots: the variant line makes outputs (and pair ids)
    unique even though the mock is deterministic per prompt."""
    single = (_topic("Одна тема", "Единственная запись в пуле, тело достаточной длины для коридора валидации.", "ru", "solo"),)
    calls: list[str] = []

    def tracking_mock(prompt: str) -> str:
        calls.append(prompt)
        return _mock_llm(prompt)

    pairs = generate_corpus(single, 0, tracking_mock, 5, quotas={STRATEGY_NEAR_TOPIC: 3})
    assert len(pairs) == 3
    assert len({p.pair_id for p in pairs}) == 3
    assert len(set(calls)) == len(calls), "prompts for the same base must never repeat"


def test_futile_variants_drop_with_counter() -> None:
    """A mock that ignores the variant line: the first slot emits, the second
    collides until the retry bound and drops with a counter — never raises."""
    single = (_topic("Одна тема", "Единственная запись в пуле, тело достаточной длины для коридора валидации.", "ru", "solo"),)
    stats = SynthStats()

    def stubborn(_prompt: str) -> str:
        return "Стабильный ответ\nОдин и тот же текст ответа, модель игнорирует строку варианта полностью."

    pairs = generate_corpus(single, 0, stubborn, 5, quotas={STRATEGY_NEAR_TOPIC: 2}, stats=stats)
    assert len(pairs) == 1  # the first slot is fine
    counters = stats.as_dict()
    assert counters["dropped_duplicate"] == 1
    # the second slot burned exactly MAX_ATTEMPTS calls before giving up
    assert counters["llm_calls"] == 1 + MAX_ATTEMPTS


# ── feature path + fingerprints ───────────────────────────────────────────────


def test_generated_pairs_flow_through_the_feature_path() -> None:
    pairs = generate_corpus(
        TOPICS_TEST, 0, _mock_llm, 21,
        quotas={s: 3 for s in (STRATEGY_PARAPHRASE, STRATEGY_NEAR_TOPIC, STRATEGY_BROKEN_FIELD, STRATEGY_TRIVIAL_NEGATIVE)},
    )
    for pair in pairs:
        vector = features(pair.record_a, pair.record_b, similarity=0.9)
        assert vector.names == FEATURE_NAMES
        assert all(value == value for value in vector.values)  # no NaN


def test_fingerprint_scheme_matches_data_contract() -> None:
    from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes

    pairs = generate_corpus(
        TOPICS_TEST, 0, _mock_llm, 33, quotas={STRATEGY_PARAPHRASE: 2, STRATEGY_NEAR_TOPIC: 2}
    )
    fingerprint = corpus_fingerprint_of_pairs(pairs)
    assert fingerprint == corpus_fingerprint(manifest_bytes(manifest_entries(pairs)))
    raw = manifest_bytes_of_pairs(pairs).decode("utf-8")
    lines = raw.splitlines()
    assert raw.endswith("\n")
    assert lines == sorted(lines)
    assert all(len(line.split(" ")) == 2 for line in lines)
    # content-sensitive: one edited body changes the corpus fingerprint
    edited = [replace(pairs[0], record_a=replace(pairs[0].record_a, body="изменённое тело записи"))] + pairs[1:]
    assert corpus_fingerprint_of_pairs(edited) != fingerprint


def test_pair_row_json_roundtrip() -> None:
    pairs = generate_corpus(
        TOPICS_TEST, 0, _mock_llm, 4, quotas={STRATEGY_PARAPHRASE: 1, STRATEGY_TRIVIAL_NEGATIVE: 1}
    )
    for pair in pairs:
        row = json.loads(pair_row_json(pair))
        assert row == pair_row(pair)
        assert row["label"] in ("duplicate", "not-duplicate")


# ── prompt protocol ───────────────────────────────────────────────────────────


def test_parse_llm_record_strips_markdown_fences() -> None:
    assert parse_llm_record("") is None
    assert parse_llm_record("   \n\n  ") is None
    fenced = "```markdown\nЗаголовок записи\nТело записи после забора.\n```"
    assert parse_llm_record(fenced) == ("Заголовок записи", "Тело записи после забора.")
    assert parse_llm_record("Просто заголовок\nПервое предложение.\nВторое предложение.") == (
        "Просто заголовок",
        "Первое предложение. Второе предложение.",
    )


def test_prompts_follow_record_language() -> None:
    from cortex.synth import VARIANT_LINE_RU
    from cortex.synth.generate import _build_prompt

    ru_base = TOPICS_TEST[0].as_record()
    en_base = TOPICS_TEST[3].as_record()
    paraphrase_ru = _build_prompt(ru_base, paraphrase=True)
    near_en = _build_prompt(en_base, paraphrase=False)
    assert paraphrase_ru.startswith("Ты генерируешь")
    assert near_en.startswith("You generate")
    assert "Заголовок: Настройка линтера" in paraphrase_ru
    assert "Reference record:" in near_en
    assert "Title: Queue choice" in near_en
    variant_prompt = _build_prompt(ru_base, paraphrase=True, variant=2)
    assert VARIANT_LINE_RU.format(n=3) in variant_prompt
    assert "вариант" not in paraphrase_ru


# ── CLI wrapper surface (backend itself is the owner sample gate) ─────────────


def test_script_help_exits_zero() -> None:
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "--sample" in proc.stdout


def test_script_requires_exactly_one_sampling_mode() -> None:
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert proc.stderr.strip()


def test_script_rejects_bad_sample_size() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--sample", "0"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2
