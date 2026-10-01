"""Candidate N — small MLP head over frozen vesma-embed-v1 vectors.

ADR 0001 V1: head ≤ ~0.5M parameters over [a, b, |a−b|, a⊙b] (4×384)
plus the scalar feature block; corruption pretrain (cortex.pretrain),
NO own encoder, NO external pretrain weights, cosine distillation
forbidden. Prod wiring is GATED on the canon ADR 0004 addendum (vectors
in CanonState); the eval runner (A5) reads vectors itself and is not
blocked. torch — train extra only (CPU; XPU walkthrough per charter §6).

A3b implementation notes (frozen here):

- torch is imported INSIDE every method: ``import cortex.candidates`` must
  stay green in a torch-less environment (default ``uv run pytest`` —
  tests gate themselves with ``pytest.importorskip("torch")``).
- Determinism: ``torch.manual_seed(config.seed)`` before module creation,
  single-threaded CPU fit, deterministic algorithms enforced for the fit
  window (previous global state restored afterwards). Same config + data
  → identical weights (pinned by tests).
- Vector block: [a, b, |a−b|, a⊙b] flattened (4×384=1536) concatenated
  with the scalar block (FEATURE_NAMES) — corruption pairs reuse the BASE
  record's store vector on both sides (self-pair convention,
  cortex.pretrain.corruption), so the head must read divergence from the
  scalars: exactly the W4c lesson as training signal.
- Pretrain is SUPERVISED-on-construction (weak positive=1, hard
  negative=0, labels by construction — allowed for PRETRAIN, ADR 0001
  V2/P3); ``train()`` then fine-tunes from the pretrained state so owner
  labels override the initialization (V2: "pretrain-инициализация обязана
  перебиваться владельческими метками").
- ``pretrain()`` gained the ``labels`` keyword (A3b signature fix): the
  corruption labels-by-construction ARE the signal; silently assuming a
  label would be dishonest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from cortex.features.pair import FEATURE_NAMES, FeatureVector

__all__ = ["NGridConfig", "GRID_N", "NHeadModel", "MAX_HEAD_PARAMS", "N_EPOCHS"]

#: Parameter budget (ADR 0001 V1: "голова ≤ ~0.5M параметров").
MAX_HEAD_PARAMS: Final[int] = 500_000

#: Vector block layout fed to the head: [a, b, |a−b|, a⊙b] → 4 × 384.
VECTOR_BLOCK_ROWS: Final[int] = 4
VECTOR_DIM: Final[int] = 384

#: Full-batch Adam epochs — fixed (not a grid axis) so the frozen grid
#: stays ≤ 8 configurations and fits stay deterministic.
N_EPOCHS: Final[int] = 100


@dataclass(frozen=True)
class NGridConfig:
    """One frozen grid point. Hidden layers 1–2, widths 32–128."""

    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    pretrain: bool  # ablation axis: with/without corruption pretrain
    seed: int


#: Frozen grid, ≤ 8 configurations (ADR 0001 V1). Three architectures over
#: the plain supervised path plus two corruption-pretrain points (the
#: ADR-mandated с/без-претрейна ablation).
GRID_N: Final[tuple[NGridConfig, ...]] = (
    NGridConfig("n-h64", hidden_dims=(64,), dropout=0.1, learning_rate=1e-3, pretrain=False, seed=1),
    NGridConfig("n-h128", hidden_dims=(128,), dropout=0.1, learning_rate=1e-3, pretrain=False, seed=1),
    NGridConfig("n-h64-32", hidden_dims=(64, 32), dropout=0.1, learning_rate=1e-3, pretrain=False, seed=1),
    NGridConfig("n-h64-pt", hidden_dims=(64,), dropout=0.1, learning_rate=1e-3, pretrain=True, seed=1),
    NGridConfig("n-h128-64-pt", hidden_dims=(128, 64), dropout=0.1, learning_rate=1e-3, pretrain=True, seed=1),
)


def vector_block(vec_a: Sequence[float], vec_b: Sequence[float]) -> np.ndarray:
    """Assemble the [a, b, |a−b|, a⊙b] block — shape (4, dim), float32.

    Store vectors must be non-empty twins; unit-normalization is the store
    contract (vesma-embed-v1) and is NOT re-imposed here (never re-measure
    — the block mirrors what the engine would pass).
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError(f"vector pair must be non-empty 1-D twins, got {a.shape} vs {b.shape}")
    return np.stack([a, b, np.abs(a - b), a * b])


class NHeadModel:
    """MLP head over frozen store vectors + scalar features.

    Shares the three-method candidate surface with DBoostModel. Vectors
    enter as the (4, 384) block; scalars as FEATURE_NAMES block — the
    exported graph takes BOTH inputs (inference-v1.md §4, variant N).
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __init__(self) -> None:
        self._module: object | None = None  # torch.nn.Module (lazy import)
        self._config: NGridConfig | None = None
        self._pretrained: bool = False

    # ── torch plumbing (lazy import boundary) ─────────────────────────────────

    def _build_module(self, config: NGridConfig) -> object:
        import torch
        from torch import nn

        class Head(nn.Module):
            def __init__(self, scalar_dim: int) -> None:
                super().__init__()
                dims = [scalar_dim + VECTOR_BLOCK_ROWS * VECTOR_DIM, *config.hidden_dims]
                layers: list[nn.Module] = []
                for in_dim, out_dim in zip(dims, dims[1:]):
                    layers.append(nn.Linear(in_dim, out_dim))
                    layers.append(nn.ReLU())
                    if config.dropout > 0:
                        layers.append(nn.Dropout(config.dropout))
                layers.append(nn.Linear(dims[-1], 1))
                self.net = nn.Sequential(*layers)

            def forward(self, scalars, vectors):
                flat = torch.flatten(vectors, start_dim=-2)
                x = torch.cat([scalars, flat], dim=-1)
                return torch.sigmoid(self.net(x))

        torch.manual_seed(config.seed)
        module = Head(len(FEATURE_NAMES))
        n_params = sum(p.numel() for p in module.parameters())
        if n_params > MAX_HEAD_PARAMS:
            raise ValueError(
                f"head has {n_params} parameters — exceeds the "
                f"{MAX_HEAD_PARAMS} budget (ADR 0001 V1)"
            )
        return module

    def _fit_loop(
        self,
        scalars: np.ndarray,
        blocks: np.ndarray,
        labels: np.ndarray,
        config: NGridConfig,
    ) -> None:
        """Full-batch Adam BCE fit — deterministic under config.seed."""
        import torch

        y = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(-1)
        x_scalars = torch.as_tensor(scalars, dtype=torch.float32)
        x_vectors = torch.as_tensor(blocks, dtype=torch.float32)

        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            if (
                self._module is not None
                and self._config is not None
                and self._config.hidden_dims != config.hidden_dims
            ):
                raise ValueError(
                    "architecture changed between pretrain and train — "
                    f"{self._config.hidden_dims} vs {config.hidden_dims}; "
                    "pretrained weights cannot carry over (rebuild explicitly)"
                )
            if self._module is None:
                self._module = self._build_module(config)
            module = self._module
            module.train()
            optimizer = torch.optim.Adam(module.parameters(), lr=config.learning_rate)
            loss_fn = torch.nn.BCELoss()
            for _ in range(N_EPOCHS):
                optimizer.zero_grad()
                probs = module(x_scalars, x_vectors)
                loss = loss_fn(probs, y)
                loss.backward()
                optimizer.step()
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic)
        self._config = config

    def _validate_input(
        self,
        vectors: Sequence[FeatureVector],
        vector_blocks: Sequence[np.ndarray],
        labels: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if not vectors:
            raise ValueError("training/inference requires at least one pair")
        names = {v.names for v in vectors}
        if len(names) != 1 or names != {FEATURE_NAMES}:
            raise ValueError(
                f"N head consumes the CORE scalar block ({len(FEATURE_NAMES)} features), "
                f"got {len(next(iter(names))) if len(names) == 1 else 'mixed'}"
            )
        blocks = np.asarray(vector_blocks, dtype=np.float32)
        if blocks.shape != (len(vectors), VECTOR_BLOCK_ROWS, VECTOR_DIM):
            raise ValueError(
                "vector_blocks must have shape "
                f"({len(vectors)}, {VECTOR_BLOCK_ROWS}, {VECTOR_DIM}), got {blocks.shape}"
            )
        scalars = np.asarray([v.values for v in vectors], dtype=np.float32)
        if labels is None:
            return scalars, blocks, None
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (len(vectors),):
            raise ValueError(f"labels length {len(y)} != vectors length {len(vectors)}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be binary {0, 1}")
        if len(np.unique(y)) < 2:
            raise ValueError("training requires both classes present")
        return scalars, blocks, y

    # ── candidate surface ─────────────────────────────────────────────────────

    def train(
        self,
        vectors: Sequence[FeatureVector],
        labels: Sequence[int],
        config: NGridConfig,
        *,
        vector_blocks: Sequence[np.ndarray],
    ) -> None:
        """Supervised fit; MUST override any pretrain initialization when
        pretrain weights are present (ADR 0001 V1: owner labels win).

        Override semantics: supervised training runs ON TOP of the
        corruption-initialized weights (two-stage scheme, addendum P3) —
        the final stage is fitted on owner labels only, so they dominate
        the deployed behavior.
        """
        scalars, blocks, y = self._validate_input(vectors, vector_blocks, labels)
        self._fit_loop(scalars, blocks, y, config)

    def pretrain(
        self,
        pairs: Sequence[tuple[FeatureVector, np.ndarray]],
        config: NGridConfig,
        *,
        labels: Sequence[int],
    ) -> None:
        """Corruption-based pretrain (cortex.pretrain.corruption pairs).
        Distillation of the cosine is FORBIDDEN (ADR 0001 V1).

        ``pairs`` carry (scalar features, vector block); ``labels`` are the
        labels-BY-CONSTRUCTION (weak positive=1, hard negative=0) — the
        only supervision allowed at this stage (ADR 0001 V2/P3).
        """
        if not pairs:
            raise ValueError("pretrain requires at least one corruption pair")
        vectors = [pair[0] for pair in pairs]
        blocks = [pair[1] for pair in pairs]
        scalars, block_array, y = self._validate_input(vectors, blocks, labels)
        self._fit_loop(scalars, block_array, y, config)
        self._pretrained = True

    def _require_module(self) -> object:
        if self._module is None:
            raise RuntimeError("NHeadModel is not fitted — call train()/pretrain() first")
        return self._module

    def predict_proba(
        self, vectors: Sequence[FeatureVector], *, vector_blocks: Sequence[np.ndarray]
    ) -> np.ndarray:
        """P(duplicate) per pair, shape (n,), values in [0, 1]."""
        import torch

        module = self._require_module()
        scalars, blocks, _ = self._validate_input(vectors, vector_blocks)
        module.eval()
        with torch.no_grad():
            probs = module(
                torch.as_tensor(scalars, dtype=torch.float32),
                torch.as_tensor(blocks, dtype=torch.float32),
            )
        return probs.squeeze(-1).numpy().astype(np.float64)

    def export_onnx(
        self, path: Path, *, metadata_props: Mapping[str, str] | None = None
    ) -> Path:
        """torch.onnx export: inputs scalars float32[K] + vectors
        float32[4, 384] → probability[1]; raises RuntimeError when the
        param budget (MAX_HEAD_PARAMS) or the artifact size gate is
        exceeded."""
        import torch

        module = self._require_module()
        module.eval()
        scalars_example = torch.zeros(len(FEATURE_NAMES), dtype=torch.float32)
        vectors_example = torch.zeros(VECTOR_BLOCK_ROWS, VECTOR_DIM, dtype=torch.float32)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # dynamo=False — the legacy exporter owns opset-15 static-shape
        # export (the dynamo path targets opset 18+; torch >= 2.9 defaults
        # to it, so pin the legacy path explicitly).
        torch.onnx.export(
            module,
            (scalars_example, vectors_example),
            str(out_path),
            opset_version=15,
            input_names=["scalars", "vectors"],
            output_names=["probability"],
            dynamo=False,
        )
        self._smoke_onnx(out_path)
        if metadata_props:
            import onnx

            from cortex.artifacts import set_onnx_metadata

            model = onnx.load(str(out_path))
            set_onnx_metadata(model, metadata_props)
            out_path.write_bytes(model.SerializeToString())
        from cortex.artifacts import assert_artifact_size

        assert_artifact_size(out_path)
        return out_path

    def _smoke_onnx(self, path: Path) -> None:
        """Eager ORT validation: zeros-in → probability[1] ∈ [0, 1],
        consistent with the library forward pass."""
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {
            "scalars": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            "vectors": np.zeros((VECTOR_BLOCK_ROWS, VECTOR_DIM), dtype=np.float32),
        }
        (prob,) = session.run(None, feeds)
        if prob.shape != (1,):
            raise RuntimeError(f"exported graph output shape {prob.shape} != (1,)")
        if not (0.0 <= float(prob[0]) <= 1.0):
            raise RuntimeError(f"exported probability {float(prob[0])} outside [0, 1]")
        reference = float(self.predict_proba(
            [FeatureVector(FEATURE_NAMES, (0.0,) * len(FEATURE_NAMES))],
            vector_blocks=[np.zeros((VECTOR_BLOCK_ROWS, VECTOR_DIM), dtype=np.float32)],
        )[0])
        if abs(float(prob[0]) - reference) > 1e-5:
            raise RuntimeError(
                f"exported graph diverges from library prediction: {float(prob[0])} vs {reference}"
            )

    # ── persistence (dev pipeline: npz weights, no pickle) ────────────────────

    def save(self, directory: Path) -> Path:
        """Persist fitted state as numpy weights (npz) + meta json — no
        pickle anywhere in the dev hand-off (defense in depth on top of
        the artifact-side pickle ban)."""
        import torch

        module = self._require_module()
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {name: param.detach().numpy() for name, param in module.state_dict().items()}
        np.savez(out_dir / "weights.npz", **arrays)
        meta = {
            "candidate": "n-head",
            "config": asdict(self._config) if self._config else None,
            "feature_names": list(FEATURE_NAMES),
            "pretrained": self._pretrained,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, directory: Path) -> "NHeadModel":
        import torch

        in_dir = Path(directory)
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("candidate") != "n-head":
            raise ValueError(f"{in_dir} is not an n-head model directory")
        config = NGridConfig(**meta["config"]) if meta.get("config") else None
        if config is None:
            raise ValueError(f"{in_dir} carries no grid config — cannot rebuild the head")
        model = cls()
        model._module = model._build_module(config)
        state = {name: torch.as_tensor(array) for name, array in np.load(in_dir / "weights.npz").items()}
        model._module.load_state_dict(state)
        model._config = config
        model._pretrained = bool(meta.get("pretrained", False))
        return model
