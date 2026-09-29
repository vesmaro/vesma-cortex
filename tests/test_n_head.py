"""Candidate N contract tests (A3b) — torch-gated.

The whole module skips when torch is absent (train extra): the default
``uv run pytest`` run stays green without it, exactly like the module's
lazy-import contract (src/cortex/candidates/n_head.py docstring)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F841 — module-level gate

import numpy as np  # noqa: E402

from cortex.candidates.n_head import (  # noqa: E402
    GRID_N,
    MAX_HEAD_PARAMS,
    NGridConfig,
    NHeadModel,
    VECTOR_BLOCK_ROWS,
    VECTOR_DIM,
    vector_block,
)
from cortex.features.pair import FEATURE_NAMES  # noqa: E402

from synth import make_vectors, unit_vec  # noqa: E402

VECTORS, LABELS, _ROWS = make_vectors(40)
RNG = np.random.RandomState(3)
BLOCKS = np.stack(
    [vector_block(unit_vec(RNG), unit_vec(RNG)) for _ in range(len(VECTORS))]
)


# ── frozen grid vs the parameter budget ───────────────────────────────────────


def test_grid_frozen_within_budget() -> None:
    assert 1 <= len(GRID_N) <= 8
    assert len({c.name for c in GRID_N}) == len(GRID_N)
    assert any(c.pretrain for c in GRID_N) and any(not c.pretrain for c in GRID_N)
    model = NHeadModel()
    for config in GRID_N:
        module = model._build_module(config)
        n_params = sum(p.numel() for p in module.parameters())
        assert n_params <= MAX_HEAD_PARAMS, f"{config.name}: {n_params} params"


def test_budget_exceeded_raises() -> None:
    oversized = NGridConfig("oversized", (1024, 1024), 0.0, 1e-3, False, 1)
    with pytest.raises(ValueError, match="budget"):
        NHeadModel()._build_module(oversized)


# ── vector block assembly ─────────────────────────────────────────────────────


def test_vector_block_layout() -> None:
    a = unit_vec(np.random.RandomState(0))
    b = unit_vec(np.random.RandomState(1))
    block = vector_block(a, b)
    assert block.shape == (VECTOR_BLOCK_ROWS, VECTOR_DIM) == (4, 384)
    assert block.dtype == np.float32
    np.testing.assert_allclose(block[0], a, rtol=1e-6)
    np.testing.assert_allclose(block[1], b, rtol=1e-6)
    np.testing.assert_allclose(block[2], np.abs(a - b), rtol=1e-6)
    np.testing.assert_allclose(block[3], a * b, rtol=1e-6)


def test_vector_block_validation() -> None:
    with pytest.raises(ValueError, match="twins"):
        vector_block((0.1, 0.2), (0.1))
    with pytest.raises(ValueError, match="twins"):
        vector_block((), ())


# ── train / predict ──────────────────────────────────────────────────────────


def test_train_predict_contract() -> None:
    model = NHeadModel()
    model.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS)
    probs = model.predict_proba(VECTORS, vector_blocks=BLOCKS)
    assert probs.shape == (len(VECTORS),)
    assert ((0.0 <= probs) & (probs <= 1.0)).all()


def test_fit_is_deterministic() -> None:
    first, second = NHeadModel(), NHeadModel()
    first.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS)
    second.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS)
    np.testing.assert_array_equal(
        first.predict_proba(VECTORS, vector_blocks=BLOCKS),
        second.predict_proba(VECTORS, vector_blocks=BLOCKS),
    )


def test_input_validation() -> None:
    model = NHeadModel()
    with pytest.raises(ValueError, match="at least one"):
        model.train([], [], GRID_N[0], vector_blocks=[])
    with pytest.raises(ValueError, match="binary"):
        model.train(VECTORS, [2] + LABELS[1:], GRID_N[0], vector_blocks=BLOCKS)
    with pytest.raises(ValueError, match="both classes"):
        model.train(VECTORS, [1] * len(VECTORS), GRID_N[0], vector_blocks=BLOCKS)
    with pytest.raises(ValueError, match="shape"):
        model.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS[:, :2, :])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(VECTORS, vector_blocks=BLOCKS)


def test_pretrain_then_train_overrides() -> None:
    """Corruption pretrain initializes; the supervised stage then owns the
    deployed behavior (ADR 0001 V2: owner labels override)."""
    pre_vectors = VECTORS[:20]
    pre_blocks = BLOCKS[:20]
    pre_labels = [1, 0] * 10  # labels by construction (allowed in pretrain)

    model = NHeadModel()
    model.pretrain(list(zip(pre_vectors, pre_blocks)), GRID_N[3], labels=pre_labels)
    assert model._pretrained
    after_pre = model.predict_proba(VECTORS[:4], vector_blocks=BLOCKS[:4]).copy()
    model.train(VECTORS, LABELS, GRID_N[3], vector_blocks=BLOCKS)
    after_fit = model.predict_proba(VECTORS[:4], vector_blocks=BLOCKS[:4])
    assert not np.array_equal(after_pre, after_fit), "supervised fit must move the weights"


def test_architecture_change_between_stages_refused() -> None:
    model = NHeadModel()
    model.pretrain(list(zip(VECTORS[:10], BLOCKS[:10])), GRID_N[3], labels=[1, 0] * 5)
    with pytest.raises(ValueError, match="architecture changed"):
        model.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS)


# ── export (inference-v1.md §4, variant N) ────────────────────────────────────


def test_export_onnx_roundtrip(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    import onnx

    model = NHeadModel()
    model.train(VECTORS, LABELS, GRID_N[0], vector_blocks=BLOCKS)
    out = tmp_path / "head.onnx"
    model.export_onnx(out)

    loaded = onnx.load(str(out))
    assert {op.domain for op in loaded.graph.node if op.op_type == "Sigmoid"}
    assert {opset.domain: opset.version for opset in loaded.opset_import}[""] == 15
    assert [i.name for i in loaded.graph.input] == ["scalars", "vectors"]
    assert loaded.graph.input[1].type.tensor_type.shape.dim[1].dim_value == 4
    assert loaded.graph.input[1].type.tensor_type.shape.dim[2].dim_value == 384
    assert out.stat().st_size <= 5 * 1024 * 1024

    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feeds = {
        "scalars": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        "vectors": np.zeros((4, 384), dtype=np.float32),
    }
    (probability,) = session.run(None, feeds)
    assert probability.shape == (1,)
    assert 0.0 <= float(probability[0]) <= 1.0


# ── save/load (npz weights, no pickle) ────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = NHeadModel()
    model.train(VECTORS, LABELS, GRID_N[2], vector_blocks=BLOCKS)
    model.save(tmp_path)
    loaded = NHeadModel.load(tmp_path)
    np.testing.assert_array_equal(
        model.predict_proba(VECTORS, vector_blocks=BLOCKS),
        loaded.predict_proba(VECTORS, vector_blocks=BLOCKS),
    )
    assert (tmp_path / "weights.npz").exists() and (tmp_path / "meta.json").exists()
