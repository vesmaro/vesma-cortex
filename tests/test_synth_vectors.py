"""Synthetic store-vector contract (A3c) — the input side of the N ladder.

The vectors are DATA (deterministic, label-consistent by construction),
so their contract is pinned like any other fixture generator: shape/unit
norm (store contract), determinism across calls, and the label semantics
of attach_store_vectors (duplicate ≈ anchor + whisper, not-duplicate =
the other topic's anchor)."""

from __future__ import annotations

import numpy as np
from synth import (
    attach_store_vectors,
    make_pair_rows,
    store_vector,
    write_records_jsonl,
)


def test_store_vector_shape_and_norm() -> None:
    vec = store_vector("заметка о кофе")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_store_vector_deterministic_and_keyed() -> None:
    assert np.array_equal(store_vector("a", seed=7), store_vector("a", seed=7))
    assert not np.array_equal(store_vector("a", seed=7), store_vector("b", seed=7))
    assert not np.array_equal(store_vector("a", seed=7), store_vector("a", seed=8))


def test_attach_store_vectors_label_semantics() -> None:
    rows = attach_store_vectors(make_pair_rows(12))
    for row in rows:
        assert len(row["vec_a"]) == len(row["vec_b"]) == 384
        vec_a = np.asarray(row["vec_a"], dtype=np.float32)
        vec_b = np.asarray(row["vec_b"], dtype=np.float32)
        cos = float(np.dot(vec_a, vec_b))
        if row["label"] == "duplicate":
            # same memory: near-identical store vector (whisper of noise)
            assert cos > 0.99, row["pair_id"]
        else:
            # different memory: an unrelated anchor vector
            assert cos < 0.2, row["pair_id"]


def test_attach_store_vectors_deterministic() -> None:
    first = attach_store_vectors(make_pair_rows(8))
    second = attach_store_vectors(make_pair_rows(8))
    assert first == second  # float-for-float: smoke determinism rides on this


def test_write_records_jsonl_carries_vec(tmp_path) -> None:
    import json

    from synth import make_records

    path = write_records_jsonl(tmp_path / "records.jsonl", make_records(6))
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    for row in rows:
        assert set(row) == {"title", "body", "tags", "language", "record_type", "vec"}
        assert len(row["vec"]) == 384
    # content-keyed: two distinct records never share a vector
    vecs = [tuple(row["vec"]) for row in rows]
    assert len(set(vecs)) == 6
