"""Corruption-generator contract tests (A3b): seed determinism, registry
invariants, honest hard negatives (no false "negative" twins)."""

from __future__ import annotations

import pytest

from cortex.features.pair import PairRecord
from cortex.pretrain.corruption import (
    HARD_NEGATIVE_TRANSFORMS,
    WEAK_POSITIVE_TRANSFORMS,
    generate_pretrain_pairs,
    record_key,
)

RECORDS = [
    PairRecord(
        title="заметка о кофе",
        body="заметка о кофе — тело номер один",
        tags=("memory", "ru"),
        language="ru",
        record_type="note",
    ),
    PairRecord(
        title="поход в горы",
        body="поход в горы — тело номер два",
        tags=("travel",),
        language="en",
        record_type="fact",
    ),
    PairRecord(
        title="python tips",
        body="python tips — тело номер три",
        tags=("code", "draft"),
        language=None,
        record_type=None,
    ),
]


# ── seed determinism ──────────────────────────────────────────────────────────


def test_same_seed_same_pairs() -> None:
    first = generate_pretrain_pairs(RECORDS, seed=42)
    second = generate_pretrain_pairs(RECORDS, seed=42)
    assert first == second  # dataclass equality over frozen records


def test_registry_membership() -> None:
    pairs = generate_pretrain_pairs(RECORDS, seed=42)
    assert pairs, "a non-degenerate corpus must yield pairs"
    for pair in pairs:
        if pair.positive:
            assert pair.transform_name in WEAK_POSITIVE_TRANSFORMS
        else:
            assert pair.transform_name in HARD_NEGATIVE_TRANSFORMS


def test_pair_id_shape() -> None:
    for pair in generate_pretrain_pairs(RECORDS, seed=0):
        base12, transform = pair.pair_id.split("--")
        assert len(base12) == 12 and int(base12, 16) >= 0
        assert transform == pair.transform_name
        assert record_key(pair.record_a) == base12


def test_max_pairs_truncates_and_validates() -> None:
    pairs = generate_pretrain_pairs(RECORDS, seed=7, max_pairs=3)
    assert len(pairs) == 3
    assert generate_pretrain_pairs(RECORDS, seed=7, max_pairs=0) == []
    with pytest.raises(ValueError):
        generate_pretrain_pairs(RECORDS, seed=7, max_pairs=-1)


# ── weak-positive invariants (meaning-preserving) ─────────────────────────────


def test_weak_positive_transform_preserves_identity_signal() -> None:
    """Every weak positive keeps the record recognizable: the tag SET, the
    whitespace-collapsed body word sequence, or the title character multiset
    survive per transform (mechanical meaning-preservation, no LLM)."""
    for pair in generate_pretrain_pairs(RECORDS * 10, seed=3):
        if not pair.positive:
            continue
        a, b = pair.record_a, pair.record_b
        if pair.transform_name == "shuffle_tags":
            assert set(a.tags) == set(b.tags)
        elif pair.transform_name == "collapse_ws_body":
            assert " ".join(a.body.split()) == b.body
        elif pair.transform_name == "normalize_punct":
            assert b.title == a.title and b.tags == a.tags
            assert len(b.body) >= len(a.body)  # 1 typographic char -> N ascii
        elif pair.transform_name == "retitle_case":
            assert b.title == a.title.swapcase() and len(b.title) == len(a.title)


def test_weak_positive_never_breaks_a_field() -> None:
    """Weak positives never substitute a field from another record: language
    and record_type ride along unchanged."""
    for pair in generate_pretrain_pairs(RECORDS * 10, seed=11):
        if pair.positive:
            assert pair.record_a.language == pair.record_b.language
            assert pair.record_a.record_type == pair.record_b.record_type


# ── hard-negative invariants (same text, broken field) ────────────────────────


def test_hard_negative_differs_from_base() -> None:
    for pair in generate_pretrain_pairs(RECORDS * 10, seed=5):
        if pair.positive:
            continue
        assert record_key(pair.record_a) != record_key(pair.record_b), (
            "a labeled negative must never be content-identical to its base"
        )


def test_swap_transforms_keep_the_unbroken_fields() -> None:
    """swap_* negatives break exactly ONE content field; the untouched sides
    stay identical (the W4c twin: globally same, locally broken)."""
    seen = set()
    for pair in generate_pretrain_pairs(RECORDS * 20, seed=13):
        if pair.transform_name not in {"swap_title", "swap_body", "swap_tags"}:
            continue
        seen.add(pair.transform_name)
        a, b = pair.record_a, pair.record_b
        field = pair.transform_name.removeprefix("swap_")
        assert getattr(b, field) != getattr(a, field)
        for other in ("title", "body", "tags", "language", "record_type"):
            if other != field:
                assert getattr(b, other) == getattr(a, other)
    assert seen == {"swap_title", "swap_body", "swap_tags"}


def test_flip_transforms_always_change() -> None:
    for pair in generate_pretrain_pairs(RECORDS * 10, seed=17):
        if pair.transform_name not in {"flip_language", "flip_record_type"}:
            continue
        a, b = pair.record_a, pair.record_b
        if pair.transform_name == "flip_language":
            assert b.language is not None and b.language != a.language
            assert (b.title, b.body, b.tags, b.record_type) == (a.title, a.body, a.tags, a.record_type)
        else:
            assert b.record_type != a.record_type
            assert (b.title, b.body, b.tags, b.language) == (a.title, a.body, a.tags, a.language)


# ── degenerate corpora ────────────────────────────────────────────────────────


def test_no_qualifying_donor_means_skip() -> None:
    """All records sharing one title → swap_title has no honest donor and is
    skipped; whatever is emitted stays deterministic and label-honest."""
    twins = [PairRecord("одинаковый заголовок", f"тело {i}", ("t",), "ru", "note") for i in range(5)]
    pairs = generate_pretrain_pairs(twins, seed=1)
    assert not any(p.transform_name == "swap_title" for p in pairs)
    assert pairs == generate_pretrain_pairs(twins, seed=1)
    for pair in pairs:  # weak positives survive (they need no donor)
        if pair.positive:
            assert pair.transform_name in WEAK_POSITIVE_TRANSFORMS


def test_empty_corpus() -> None:
    assert generate_pretrain_pairs([], seed=0) == []
