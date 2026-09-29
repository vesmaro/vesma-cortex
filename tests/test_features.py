"""Pair-feature contract tests (A3b): determinism, exact values, borders."""

from __future__ import annotations

import math

import pytest

from cortex.features.pair import (
    FEATURE_NAMES,
    FIELD_COSINE_FEATURES,
    FeatureVector,
    PairRecord,
    attach_field_cosines,
    features,
)


def _names_mapping(vector: FeatureVector) -> dict[str, float]:
    return dict(zip(vector.names, vector.values))


R_A = PairRecord("Заметка", "Тело записи", ("tag1", "tag2"), "ru", "note")
R_B = PairRecord("Заметка", "Тело записи", ("tag1", "tag2"), "ru", "note")


# ── shape and determinism ─────────────────────────────────────────────────────


def test_feature_vector_contract() -> None:
    vector = features(R_A, R_B, 0.5)
    assert vector.names == FEATURE_NAMES
    assert len(vector.values) == len(FEATURE_NAMES) == 13


def test_deterministic_and_symmetric() -> None:
    left = features(R_A, R_B, 0.75)
    right = features(R_A, R_B, 0.75)
    assert left == right
    assert left == features(R_B, R_A, 0.75)  # |Δ| deltas + symmetric sets


# ── exact values on a hand-computed case ──────────────────────────────────────


def test_identical_records_all_max() -> None:
    values = _names_mapping(features(R_A, R_B, 0.9312))
    assert values["cos_target"] == 0.9312  # passthrough, never re-measured
    for name in ("char3_jaccard", "char4_jaccard", "char5_jaccard",
                 "char3_containment", "char4_containment", "char5_containment",
                 "tag_jaccard"):
        assert values[name] == 1.0
    assert values["title_len_delta"] == 0.0
    assert values["body_len_delta"] == 0.0
    assert values["tag_count_delta"] == 0.0
    assert values["type_match"] == 1.0
    assert values["lang_match"] == 1.0


def test_disjoint_texts_and_tags() -> None:
    # long enough texts, fully disjoint → zero overlap everywhere
    long_a = PairRecord("a" * 40, "b" * 40, ("t1",), "ru", "note")
    long_b = PairRecord("c" * 40, "d" * 40, ("t2",), "en", "fact")
    values = _names_mapping(features(long_a, long_b, 0.0))
    for name in ("char3_jaccard", "char4_jaccard", "char5_jaccard",
                 "char3_containment", "char4_containment", "char5_containment",
                 "tag_jaccard"):
        assert values[name] == 0.0, name
    assert values["type_match"] == 0.0
    assert values["lang_match"] == 0.0
    assert values["title_len_delta"] == 0.0  # equal lengths
    # containment vs jaccard: an appended-only edit keeps every original
    # gram (containment 1.0) while shrinking the jaccard union
    base = PairRecord("заметка о кофе", "тело", (), None, None)
    extended = PairRecord("заметка о кофе", "тело!!", (), None, None)
    values = _names_mapping(features(base, extended, 0.9))
    assert values["char3_containment"] == pytest.approx(1.0, abs=1e-9)
    assert values["char3_jaccard"] < 1.0  # extra grams grow the union
    # a mid-text edit kills junction grams → containment drops below 1.0
    mid = PairRecord("заметка о кофе!!", "тело", (), None, None)
    values = _names_mapping(features(base, mid, 0.9))
    assert 0.0 < values["char3_containment"] < 1.0


def test_unicode_ru_en_normalization() -> None:
    lower = PairRecord("Привет Мир", "Body текст", (), None, None)
    upper = PairRecord("привет мир", "BODY ТЕКСТ", (), None, None)
    values = _names_mapping(features(lower, upper, 0.9))
    assert values["char3_jaccard"] == 1.0  # lower() + ws-collapse absorb the case


def test_tag_order_and_deltas() -> None:
    a = PairRecord("t", "b", ("x", "y", "z"), "ru", "note")
    b = PairRecord("tt", "b", ("z", "y", "x", "w"), "ru", "note")
    values = _names_mapping(features(a, b, 0.5))
    assert values["tag_jaccard"] == pytest.approx(3 / 4)  # order-invariant set
    assert values["title_len_delta"] == 1.0
    assert values["tag_count_delta"] == 1.0


def test_match_semantics_null_vs_value() -> None:
    both_null = _names_mapping(features(
        PairRecord("a", "b", (), None, None), PairRecord("a", "b", (), None, None), 0.5))
    one_null = _names_mapping(features(
        PairRecord("a", "b", (), "ru", None), PairRecord("a", "b", (), "ru", "note"), 0.5))
    mismatch = _names_mapping(features(
        PairRecord("a", "b", (), "ru", "note"), PairRecord("a", "b", (), "en", "note"), 0.5))
    assert both_null["lang_match"] == 0.0 and both_null["type_match"] == 0.0
    assert one_null["type_match"] == 0.0 and one_null["lang_match"] == 1.0
    assert mismatch["lang_match"] == 0.0 and mismatch["type_match"] == 1.0


# ── borders: empty fields, short texts ────────────────────────────────────────


def test_empty_fields_both_sides() -> None:
    empty = PairRecord("", "", (), None, None)
    values = _names_mapping(features(empty, empty, 0.0))
    for name in ("char3_jaccard", "char4_jaccard", "char5_jaccard",
                 "char3_containment", "char4_containment", "char5_containment",
                 "tag_jaccard"):
        assert values[name] == 1.0, name  # two empty sets are identical


def test_empty_vs_nonempty() -> None:
    empty = PairRecord("", "", (), None, None)
    full = PairRecord("текст", "тело", ("t",), "ru", "note")
    values = _names_mapping(features(empty, full, 0.5))
    for name in ("char3_jaccard", "char5_containment", "tag_jaccard"):
        assert values[name] == 0.0, name


def test_text_shorter_than_ngram_order() -> None:
    a = PairRecord("ab", "", (), None, None)  # normalized text "ab" (3 chars incl. \n→space)
    values = _names_mapping(features(a, a, 1.0))
    assert values["char3_jaccard"] == 1.0  # identical short texts still match
    b = PairRecord("ab", "", (), None, None)
    c = PairRecord("ac", "", (), None, None)
    values = _names_mapping(features(b, c, 1.0))
    assert 0.0 <= values["char3_jaccard"] <= 1.0  # no crash on tiny gram sets


# ── validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [1.5, -1.01, float("nan"), "not-a-number"])
def test_similarity_bounds(bad) -> None:
    with pytest.raises(ValueError):
        features(R_A, R_B, bad)


def test_vector_twins_validation() -> None:
    vec = (0.1, 0.2)
    with pytest.raises(ValueError):
        features(R_A, R_B, 0.5, vec_a=vec)  # vec_b missing
    with pytest.raises(ValueError):
        features(R_A, R_B, 0.5, vec_a=vec, vec_b=(0.1,))  # length mismatch
    with pytest.raises(ValueError):
        features(R_A, R_B, 0.5, vec_a=(), vec_b=())  # empty
    # valid twins are accepted (and ignored for the core vector)
    assert features(R_A, R_B, 0.5, vec_a=vec, vec_b=(0.3, 0.4)).names == FEATURE_NAMES


# ── gated field cosines ───────────────────────────────────────────────────────


def test_field_cosines_gated() -> None:
    with pytest.raises(ValueError, match="attach_field_cosines"):
        features(R_A, R_B, 0.5, field_cosines=True)


def test_attach_field_cosines() -> None:
    core = features(R_A, R_B, 0.5)
    extended = attach_field_cosines(core, cos_title=0.9, cos_body=0.8, cos_tags=0.7)
    assert extended.names == FEATURE_NAMES + FIELD_COSINE_FEATURES
    assert extended.values[: len(FEATURE_NAMES)] == core.values
    assert extended.values[-3:] == (0.9, 0.8, 0.7)

    with pytest.raises(ValueError):
        attach_field_cosines(extended, cos_title=0.0, cos_body=0.0, cos_tags=0.0)  # no double attach
    with pytest.raises(ValueError):
        attach_field_cosines(core, cos_title=1.5, cos_body=0.0, cos_tags=0.0)  # out of range
    with pytest.raises(ValueError):
        attach_field_cosines(core, cos_title=float("nan"), cos_body=0.0, cos_tags=0.0)


def test_values_finite() -> None:
    values = features(R_A, R_B, 0.123).values
    assert all(math.isfinite(v) for v in values)
