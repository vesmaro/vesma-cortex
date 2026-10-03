"""Holdout-isolation contract tests (A3b): frozen prereg split, pair-id
overlap assert, physical label isolation (symlinks resolved)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes
from cortex.data.holdout import (
    HOLDOUT_FRACTION,
    SplitPair,
    assert_labels_isolated,
    assert_no_pair_overlap,
    split_holdout,
)


def _pair(n: int, stratum: str = "P1", label: str = "duplicate") -> SplitPair:
    return SplitPair(
        pair_id=f"mem-{n:04d}--mem-{n + 100:04d}",
        stratum=stratum,
        label=label,
        sha_a=f"a{n}",
        sha_b=f"b{n}",
        pair_sha256=f"h{n}",
    )


# ── the frozen split ──────────────────────────────────────────────────────────


def test_split_first_ceil_30pct_per_stratum_sorted_by_pair_id() -> None:
    pairs = [_pair(i, stratum=s) for s in ("P1", "N2") for i in range(10)]
    split = split_holdout(pairs)
    # every pair lands in exactly one side
    assert set(split.train_pair_ids) | set(split.holdout_pair_ids) == {
        p.pair_id for p in pairs
    }
    assert len(split.train_pair_ids) + len(split.holdout_pair_ids) == len(pairs)
    # per-stratum arithmetic: 10 members per stratum → ceil(3) holdout each
    assert len(split.holdout_pair_ids) == 6
    assert len(split.train_pair_ids) == 14
    for stratum in ("P1", "N2"):
        members = sorted(p.pair_id for p in pairs if p.stratum == stratum)
        n_holdout = math.ceil(HOLDOUT_FRACTION * len(members))
        assert set(members[:n_holdout]) <= set(split.holdout_pair_ids)  # first ids go
        assert set(members[n_holdout:]) <= set(split.train_pair_ids)


def test_split_rounding_small_strata() -> None:
    split = split_holdout([_pair(1, "P2"), _pair(2, "N1"), _pair(3, "N1")])
    # ceil(0.3·1) = 1 → the lone P2 pair is holdout; ceil(0.3·2) = 1 →
    # the pair_id-first N1 pair joins it, the second stays in train.
    # Collection order is by SORTED stratum name (N1 before P2).
    assert split.holdout_pair_ids == ("mem-0002--mem-0102", "mem-0001--mem-0101")
    assert split.train_pair_ids == ("mem-0003--mem-0103",)


def test_split_rejects_disputed() -> None:
    pairs = [_pair(1), _pair(2, label="disputed")]
    with pytest.raises(ValueError, match="disputed"):
        split_holdout(pairs)


def test_split_rejects_missing_pair_sha() -> None:
    broken = SplitPair("a--b", "P1", "duplicate", "x", "y", "")
    with pytest.raises(ValueError, match="pair_sha256"):
        split_holdout([broken])


def test_split_carries_honest_corpus_fingerprint() -> None:
    pairs = [_pair(i) for i in range(7)]
    split = split_holdout(pairs)
    expected = corpus_fingerprint(
        manifest_bytes((p.pair_id, p.pair_sha256) for p in pairs)
    )
    assert split.corpus_fingerprint == expected
    assert len(split.corpus_fingerprint) == 64  # BLAKE2b-256 hex


# ── overlap assert (single-shot integrity) ────────────────────────────────────


def test_no_overlap_assert() -> None:
    assert assert_no_pair_overlap(["a--b", "c--d"], ["e--f"]) is None
    with pytest.raises(AssertionError, match="intersection"):
        assert_no_pair_overlap(["a--b", "c--d"], ["c--d", "g--h"])


def test_split_produces_disjoint_sides() -> None:
    split = split_holdout([_pair(i) for i in range(20)])
    assert not set(split.train_pair_ids) & set(split.holdout_pair_ids)


# ── physical label isolation ──────────────────────────────────────────────────


def test_labels_inside_train_tree_refused(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    labels = train_root / "labels.jsonl"
    train_root.mkdir()
    with pytest.raises(AssertionError, match="physical isolation"):
        assert_labels_isolated(train_root, labels)
    with pytest.raises(AssertionError, match="physical isolation"):
        assert_labels_isolated(train_root, train_root / "sub" / "labels.jsonl")


def test_labels_outside_train_tree_accepted(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    labels = tmp_path / "labels-holdout" / "labels.jsonl"
    train_root.mkdir()
    (labels.parent).mkdir()
    assert assert_labels_isolated(train_root, labels) is None


def test_labels_via_symlink_into_train_tree_refused(tmp_path: Path) -> None:
    """A symlink pointing inside the train tree is caught by resolve() —
    the assert must not be fooled by an alias path."""
    train_root = tmp_path / "train"
    secret = train_root / "holdout-labels.jsonl"
    alias = tmp_path / "alias-labels.jsonl"
    train_root.mkdir()
    secret.touch()
    alias.symlink_to(secret)
    with pytest.raises(AssertionError, match="physical isolation"):
        assert_labels_isolated(train_root, alias)
