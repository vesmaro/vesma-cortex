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
import os
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
    TOPICS,
    SynthStats,
    SynthTopic,
    corpus_fingerprint_of_pairs,
    generate_corpus,
    manifest_bytes_of_pairs,
    manifest_entries,
    pair_row,
    pair_row_json,
    parse_llm_record,
    sample_quotas,
    strip_protocol_markers,
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
    _topic(
        "Настройка линтера",
        "Вынесли конфиг линтера в pyproject, длина строки сто, правило импортов включено.",
        "ru",
        "linter",
    ),
    _topic(
        "Выбор очереди",
        "Выбрали легковесную очередь вместо брокера сообщений: ноль новой инфраструктуры, консюмеры из коробки.",
        "ru",
        "queue",
    ),
    _topic(
        "Linter configuration",
        "The linter config lives in pyproject, line length one hundred, the import rule is enabled.",
        "en",
        "linter",
    ),
    _topic(
        "Queue choice",
        "We picked a lightweight queue over a message broker: zero new infrastructure, consumers out of the box.",
        "en",
        "queue",
    ),
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
    variant = (
        (variant_match.group(1) or variant_match.group(2)) if variant_match else "1"
    )
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


@pytest.mark.parametrize("n", list(range(61)))
def test_sample_quotas_total_exactly_n(n: int) -> None:
    assert sum(sample_quotas(n).values()) == n


def test_sample_quotas_reject_negative() -> None:
    with pytest.raises(ValueError):
        sample_quotas(-1)


# ── labels by construction ────────────────────────────────────────────────────


def test_strategies_carry_labels_by_construction() -> None:
    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        _mock_llm,
        11,
        quotas={
            s: 2
            for s in (
                STRATEGY_PARAPHRASE,
                STRATEGY_NEAR_TOPIC,
                STRATEGY_BROKEN_FIELD,
                STRATEGY_TRIVIAL_NEGATIVE,
            )
        },
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
        TOPICS_TEST,
        0,
        _mock_llm,
        3,
        quotas={STRATEGY_PARAPHRASE: 2, STRATEGY_NEAR_TOPIC: 2},
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
        TOPICS_TEST,
        0,
        lambda _prompt: "",
        5,
        quotas={STRATEGY_PARAPHRASE: 2},
        stats=stats,
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
    ok = validate_candidate(
        base,
        "Другой заголовок записи",
        "Другое тело записи той же памяти, смысл тот же самый.",
    )
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
    quotas = {
        STRATEGY_PARAPHRASE: 0,
        STRATEGY_NEAR_TOPIC: 0,
        STRATEGY_BROKEN_FIELD: 12,
        STRATEGY_TRIVIAL_NEGATIVE: 12,
    }
    first = [
        pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)
    ]
    second = [
        pair_row(p) for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)
    ]
    assert first == second
    other = generate_corpus(TOPICS_TEST, 0, None, 43, quotas=quotas)
    assert [p.seed for p in other] != [
        p.seed for p in generate_corpus(TOPICS_TEST, 0, None, 42, quotas=quotas)
    ]


def test_llm_generation_is_seed_deterministic() -> None:
    quotas = {
        STRATEGY_PARAPHRASE: 3,
        STRATEGY_NEAR_TOPIC: 3,
        STRATEGY_BROKEN_FIELD: 0,
        STRATEGY_TRIVIAL_NEGATIVE: 0,
    }
    first = [
        pair_row(p)
        for p in generate_corpus(TOPICS_TEST, 0, _mock_llm, 9, quotas=quotas)
    ]
    second = [
        pair_row(p)
        for p in generate_corpus(TOPICS_TEST, 0, _mock_llm, 9, quotas=quotas)
    ]
    assert first == second


def test_translation_twins_never_form_a_negative() -> None:
    pairs = generate_corpus(
        TOPICS,
        0,
        None,
        7,
        quotas={
            STRATEGY_TRIVIAL_NEGATIVE: 40,
            STRATEGY_BROKEN_FIELD: 0,
            STRATEGY_PARAPHRASE: 0,
            STRATEGY_NEAR_TOPIC: 0,
        },
    )
    key_by_title = {topic.title: topic.key for topic in TOPICS}
    for pair in pairs:
        assert key_by_title[pair.record_a.title] != key_by_title[pair.record_b.title]


def test_full_scale_procedural_run_has_no_collisions() -> None:
    """quota > pool size must not crash (old behavior raised on pair ids)."""
    pairs = generate_corpus(
        TOPICS,
        0,
        None,
        7,
        quotas={STRATEGY_BROKEN_FIELD: 40, STRATEGY_TRIVIAL_NEGATIVE: 40},
    )
    ids = [pair.pair_id for pair in pairs]
    assert len(set(ids)) == len(ids) == 80


def test_variant_prompts_rescue_repeated_bases() -> None:
    """One topic, three slots: the variant line makes outputs (and pair ids)
    unique even though the mock is deterministic per prompt."""
    single = (
        _topic(
            "Одна тема",
            "Единственная запись в пуле, тело достаточной длины для коридора валидации.",
            "ru",
            "solo",
        ),
    )
    calls: list[str] = []

    def tracking_mock(prompt: str) -> str:
        calls.append(prompt)
        return _mock_llm(prompt)

    pairs = generate_corpus(
        single, 0, tracking_mock, 5, quotas={STRATEGY_NEAR_TOPIC: 3}
    )
    assert len(pairs) == 3
    assert len({p.pair_id for p in pairs}) == 3
    assert len(set(calls)) == len(calls), "prompts for the same base must never repeat"


def test_futile_variants_drop_with_counter() -> None:
    """A mock that ignores the variant line: the first slot emits, the second
    collides until the retry bound and drops with a counter — never raises."""
    single = (
        _topic(
            "Одна тема",
            "Единственная запись в пуле, тело достаточной длины для коридора валидации.",
            "ru",
            "solo",
        ),
    )
    stats = SynthStats()

    def stubborn(_prompt: str) -> str:
        return "Стабильный ответ\nОдин и тот же текст ответа, модель игнорирует строку варианта полностью."

    pairs = generate_corpus(
        single, 0, stubborn, 5, quotas={STRATEGY_NEAR_TOPIC: 2}, stats=stats
    )
    assert len(pairs) == 1  # the first slot is fine
    counters = stats.as_dict()
    assert counters["dropped_duplicate"] == 1
    # the second slot burned exactly MAX_ATTEMPTS calls before giving up
    assert counters["llm_calls"] == 1 + MAX_ATTEMPTS


# ── feature path + fingerprints ───────────────────────────────────────────────


def test_generated_pairs_flow_through_the_feature_path() -> None:
    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        _mock_llm,
        21,
        quotas={
            s: 3
            for s in (
                STRATEGY_PARAPHRASE,
                STRATEGY_NEAR_TOPIC,
                STRATEGY_BROKEN_FIELD,
                STRATEGY_TRIVIAL_NEGATIVE,
            )
        },
    )
    for pair in pairs:
        vector = features(pair.record_a, pair.record_b, similarity=0.9)
        assert vector.names == FEATURE_NAMES
        assert all(value == value for value in vector.values)  # no NaN


def test_fingerprint_scheme_matches_data_contract() -> None:
    from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes

    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        _mock_llm,
        33,
        quotas={STRATEGY_PARAPHRASE: 2, STRATEGY_NEAR_TOPIC: 2},
    )
    fingerprint = corpus_fingerprint_of_pairs(pairs)
    assert fingerprint == corpus_fingerprint(manifest_bytes(manifest_entries(pairs)))
    raw = manifest_bytes_of_pairs(pairs).decode("utf-8")
    lines = raw.splitlines()
    assert raw.endswith("\n")
    assert lines == sorted(lines)
    assert all(len(line.split(" ")) == 2 for line in lines)
    # content-sensitive: one edited body changes the corpus fingerprint
    edited = [
        replace(
            pairs[0], record_a=replace(pairs[0].record_a, body="изменённое тело записи")
        )
    ] + pairs[1:]
    assert corpus_fingerprint_of_pairs(edited) != fingerprint


def test_pair_row_json_roundtrip() -> None:
    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        _mock_llm,
        4,
        quotas={STRATEGY_PARAPHRASE: 1, STRATEGY_TRIVIAL_NEGATIVE: 1},
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
    assert parse_llm_record(
        "Просто заголовок\nПервое предложение.\nВторое предложение."
    ) == (
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


# ── protocol-marker cleanup (llm_strips provenance counter) ───────────────────


def test_strip_protocol_markers_removes_prefixes_and_marker_lines() -> None:
    raw = (
        "Заголовок: Настройка ruff\n"
        "Тело: Конфиг в pyproject, длина строки сто.\n"
        "Теги: tools, linting"
    )
    assert strip_protocol_markers(raw) == (
        "Настройка ruff\nКонфиг в pyproject, длина строки сто.",
        3,
    )
    raw_en = (
        "Title: Ruff setup\n"
        "Body: The config lives in pyproject, line length one hundred.\n"
        "Tags: tools"
    )
    assert strip_protocol_markers(raw_en) == (
        "Ruff setup\nThe config lives in pyproject, line length one hundred.",
        3,
    )


def test_strip_protocol_markers_edge_cases() -> None:
    # marker-only line
    assert strip_protocol_markers("Тело:\nТекст после пустого маркера.") == (
        "Текст после пустого маркера.",
        1,
    )
    # chained prefixes collapse with one strip per marker
    assert strip_protocol_markers("Заголовок: Заголовок: Двойной") == ("Двойной", 2)
    # case-insensitive EN
    assert strip_protocol_markers("TITLE: X\nBODY: Y") == ("X\nY", 2)
    # clean text, empty text and mid-line markers pass through untouched
    clean = "Просто заголовок\nТело записи не помечено маркером."
    assert strip_protocol_markers(clean) == (clean, 0)
    assert strip_protocol_markers("") == ("", 0)
    mixed = "Заголовок\nВ тексте упомянуто Тело: посреди строки — это контент."
    assert strip_protocol_markers(mixed) == (mixed, 0)


# ── batched cloud path (llm_batch_fn) ─────────────────────────────────────────


def _batch_mock(prompts: list[str]) -> list[str]:
    return [_mock_llm(prompt) for prompt in prompts]


def test_batched_path_matches_sequential_on_clean_mock() -> None:
    """Collision-free run: same prompts, same seed-draw order, same corpus."""
    quotas = {STRATEGY_PARAPHRASE: 3, STRATEGY_NEAR_TOPIC: 3}
    sequential = [
        pair_row(p)
        for p in generate_corpus(TOPICS_TEST, 0, _mock_llm, 9, quotas=quotas)
    ]
    batched = [
        pair_row(p)
        for p in generate_corpus(
            TOPICS_TEST, 0, None, 9, quotas=quotas, llm_batch_fn=_batch_mock
        )
    ]
    assert batched == sequential


def test_batched_path_counts_batches_and_enforces_count_order() -> None:
    stats = SynthStats()
    calls: list[list[str]] = []

    def tracking_mock(prompts: list[str]) -> list[str]:
        calls.append(list(prompts))
        return _batch_mock(prompts)

    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        None,
        5,
        quotas={STRATEGY_PARAPHRASE: 3},
        stats=stats,
        llm_batch_fn=tracking_mock,
        batch_size=2,
    )
    assert len(pairs) == 3
    counters = stats.as_dict()
    assert counters["llm_batches"] == 2  # ceil(3/2)
    assert counters["llm_calls"] == 3
    assert [len(chunk) for chunk in calls] == [2, 1]

    def bad_count(prompts: list[str]) -> list[str]:
        return _batch_mock(prompts)[:-1]

    with pytest.raises(ValueError, match="count and order"):
        generate_corpus(
            TOPICS_TEST,
            0,
            None,
            5,
            quotas={STRATEGY_PARAPHRASE: 1},
            llm_batch_fn=bad_count,
        )


def test_batched_corridor_drops_are_not_retried_transport_aborts_loudly() -> None:
    stats = SynthStats()

    def empty(prompts: list[str]) -> list[str]:
        return [""] * len(prompts)

    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        None,
        5,
        quotas={STRATEGY_NEAR_TOPIC: 2},
        stats=stats,
        llm_batch_fn=empty,
    )
    assert pairs == []
    counters = stats.as_dict()
    assert counters["dropped_empty"] == 2
    assert counters["llm_calls"] == 2  # corridor drops get no second round

    def broken(prompts: list[str]) -> list[str]:
        raise ConnectionError("provider down")

    with pytest.raises(ConnectionError):
        generate_corpus(
            TOPICS_TEST,
            0,
            None,
            5,
            quotas={STRATEGY_NEAR_TOPIC: 1},
            llm_batch_fn=broken,
        )


def test_callback_modes_are_mutually_exclusive_and_batch_size_validated() -> None:
    with pytest.raises(ValueError, match="one callback mode"):
        generate_corpus(
            TOPICS_TEST,
            0,
            _mock_llm,
            5,
            quotas={STRATEGY_PARAPHRASE: 1},
            llm_batch_fn=_batch_mock,
        )
    with pytest.raises(ValueError, match="batch_size"):
        generate_corpus(
            TOPICS_TEST,
            0,
            None,
            5,
            quotas={STRATEGY_PARAPHRASE: 1},
            llm_batch_fn=_batch_mock,
            batch_size=0,
        )


def test_emit_callback_receives_every_pair_in_order() -> None:
    seen: list[str] = []
    pairs = generate_corpus(
        TOPICS_TEST,
        0,
        _mock_llm,
        21,
        quotas={
            s: 3
            for s in (
                STRATEGY_PARAPHRASE,
                STRATEGY_NEAR_TOPIC,
                STRATEGY_BROKEN_FIELD,
                STRATEGY_TRIVIAL_NEGATIVE,
            )
        },
        emit_callback=lambda pair: seen.append(pair.pair_id),
    )
    assert seen == [pair.pair_id for pair in pairs]


# ── cloud batch JSON contract (script-side parser) ────────────────────────────


def _script_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_synth_corpus_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_batch_json_tolerates_fences_and_trailing_commentary() -> None:
    g = _script_module()
    raw = (
        "```json\n"
        '[\n  {"title": "А", "body": "Первое тело.\\nТеги: t"},\n'
        '  {"title": "Б", "body": "Второе тело."}\n]\n'
        "```\n\n"
        "Обратите внимание: выше приведён формат для наглядности."
    )
    assert g.parse_batch_json(raw, 2) == ["А\nПервое тело.\nТеги: t", "Б\nВторое тело."]


def test_parse_batch_json_rejects_malformed_and_maps_bad_items_to_empty() -> None:
    g = _script_module()
    with pytest.raises(ValueError, match="no JSON array"):
        g.parse_batch_json("Просто текст без массива.", 1)
    with pytest.raises(ValueError, match="expected 2"):
        g.parse_batch_json('[{"title": "А", "body": "Б"}]', 2)
    # non-object items and missing fields map to "" → dropped at validation
    assert g.parse_batch_json(
        '[{"title": "А", "body": "Б"}, 5, {"title": "В"}]', 3
    ) == [
        "А\nБ",
        "",
        "",
    ]


def test_chat_completion_wall_cap_aborts_a_stalled_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway that defeats the socket inactivity timeout (openrouter keeps
    the connection warm while the model generates) is still bounded by the
    SIGALRM wall cap: the attempt aborts at ~timeout and ProviderDown wins."""
    import time as time_mod
    import urllib.request as urlreq

    g = _script_module()

    def stalled(_request, timeout=None):
        time_mod.sleep(3.0)
        raise TimeoutError("should have been killed by the wall cap first")

    monkeypatch.setattr(urlreq, "urlopen", stalled)
    started = time_mod.monotonic()
    with pytest.raises(g.ProviderDown, match="wall cap|transport failure"):
        g.chat_completion(
            "ping",
            name="openrouter",
            base_url="https://x/v1",
            model="m",
            api_key="k",
            seed=1,
            max_tokens=16,
            timeout=1.0,
            rate_retries=1,
        )
    # two attempts at ~1s wall cap each (no full 3s sleeper completion ×2)
    assert time_mod.monotonic() - started < 5.0


# ── CLI wrapper surface (backend itself is the owner sample gate) ─────────────


def test_script_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--sample" in proc.stdout


def test_script_requires_exactly_one_sampling_mode() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2
    assert proc.stderr.strip()


def test_script_rejects_bad_sample_size() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--sample", "0"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2


def test_script_cloud_provider_without_key_exits_two(tmp_path: Path) -> None:
    """Missing provider key = usage error (exit 2) BEFORE any generation;
    the error names the env var, never a value."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENROUTER_API_KEY", "GROQ_API_KEY")
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pairs-per-strategy",
            "1",
            "--out-dir",
            str(tmp_path / "synth"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 2
    assert "OPENROUTER_API_KEY" in proc.stderr
    assert not list(tmp_path.rglob("pairs.jsonl"))


def test_script_render_review_labels_are_not_inverted(tmp_path: Path) -> None:
    """The old hand-rendered REVIEW.md showed paraphrase as «НЕ дубликат» —
    the renderer pins the true mapping (duplicate → «дубликат»)."""
    pairs = generate_corpus(
        TOPICS,
        0,
        _mock_llm,
        3,
        quotas={
            s: 4
            for s in (
                STRATEGY_PARAPHRASE,
                STRATEGY_NEAR_TOPIC,
                STRATEGY_BROKEN_FIELD,
                STRATEGY_TRIVIAL_NEGATIVE,
            )
        },
    )
    corpus = tmp_path / "pairs.jsonl"
    corpus.write_text(
        "\n".join(pair_row_json(p) for p in pairs) + "\n", encoding="utf-8"
    )
    out = tmp_path / "REVIEW.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--render-review",
            str(corpus),
            "--review-out",
            str(out),
            "--review-n",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    assert "## Пара 1 [paraphrase] — метка: дубликат" in text
    assert "[near-topic] — метка: НЕ дубликат" in text
    assert "[broken-field]" in text
    assert "[trivial-negative]" in text
