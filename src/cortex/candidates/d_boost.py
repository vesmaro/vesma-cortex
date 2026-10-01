"""Candidate D — gradient boosting over the frozen pair features.

ADR 0001 V1: D is an EQUAL ADOPT-candidate and the internal bar, not a
control afterthought. LightGBM here (train env only — never a runtime dep
of the engine, addendum P2). Grids ≤ 8 configurations, frozen in code at
A3 (this file); selection protocol lives in cortex.select.cv.

A3b implementation notes (frozen here):

- Determinism: every LightGBM fit runs single-threaded with
  ``deterministic=True`` + ``force_col_wise=True`` under the config seed —
  two fits on identical data produce identical margins (pinned by tests).
- Calibration: Platt (sigmoid over the raw margin, p = σ(A·z+B)) fitted on
  TRAIN ONLY via out-of-fold margins (internal StratifiedKFold, seed =
  config seed). Falls back to identity (A=1, B=0) with a warning when the
  train split cannot support CV (a class with < 2 members). Isotonic was
  rejected for ~140 labeled pairs (overfits the tails; Platt is the
  low-variance choice, matching the ladder philosophy).
- ONNX export (skl2onnx chain — the LightGBM converter ships in
  ``onnxmltools`` since skl2onnx 1.20): the converted
  TreeEnsembleClassifier emits σ(z) via post_transform; this module
  rewrites post_transform to NONE and appends the Platt transform
  (Mul/Add/Sigmoid) as explicit graph nodes, so the exported probability
  is EXACTLY the calibrated library ``predict_proba`` (cross-checked at
  export time through an onnxruntime smoke inference).
- Graph I/O contract (inference-v1.md §4, variant D): input
  ``features: float32[K]`` (1-D, single pair) → output
  ``probability: float32[1]``; opset 15 + ai.onnx.ml 1, IR 8.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from cortex.features.pair import FEATURE_NAMES, FIELD_COSINE_FEATURES, FeatureVector

__all__ = ["DGridConfig", "GRID_D", "DBoostModel", "D_N_ESTIMATORS"]

#: Tree count of every grid point — deliberately NOT a grid axis (the grid
#: cap is 8 configurations; depth/lr/min-data carry the capacity trade).
D_N_ESTIMATORS: Final[int] = 200

#: Internal CV folds for the Platt calibration (train-only signal).
PLATT_CV_FOLDS: Final[int] = 5


@dataclass(frozen=True)
class DGridConfig:
    """One frozen grid point (hyperparameters + feature variant)."""

    name: str
    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    field_cosines: bool  # ablation axis: with/without field cosines
    seed: int


#: Frozen grid, ≤ 8 configurations (ADR 0001 V1: "гриды ≤ 8 конфигураций,
#: замораживаются в коде A3"). Two depths × two learning rates over the
#: core features, plus one field-cosines ablation point (the ADR-mandated
#: "D-без-полевых is the cheapest runtime" axis).
GRID_D: Final[tuple[DGridConfig, ...]] = (
    DGridConfig("d-l7-lr005", num_leaves=7, learning_rate=0.05, min_data_in_leaf=5, field_cosines=False, seed=1),
    DGridConfig("d-l15-lr005", num_leaves=15, learning_rate=0.05, min_data_in_leaf=5, field_cosines=False, seed=1),
    DGridConfig("d-l7-lr010", num_leaves=7, learning_rate=0.10, min_data_in_leaf=10, field_cosines=False, seed=1),
    DGridConfig("d-l15-lr010", num_leaves=15, learning_rate=0.10, min_data_in_leaf=10, field_cosines=False, seed=1),
    DGridConfig("d-l15-lr005-fc", num_leaves=15, learning_rate=0.05, min_data_in_leaf=5, field_cosines=True, seed=1),
)

_CORE_NAMES: Final[tuple[str, ...]] = FEATURE_NAMES
_EXTENDED_NAMES: Final[tuple[str, ...]] = FEATURE_NAMES + FIELD_COSINE_FEATURES


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class DBoostModel:
    """LightGBM classifier over FEATURE_NAMES — train/predict/export API.

    The three-method surface is the candidate contract shared with
    NHeadModel: cv_select drives both through it, export-artifact writes
    the winner as a single ONNX (≤ 5 MB, metadata_props per
    cortex.artifacts).
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __init__(self) -> None:
        self._booster: object | None = None  # lightgbm.Booster
        self._platt: tuple[float, float] = (1.0, 0.0)
        self._config: DGridConfig | None = None

    # ── fitting ───────────────────────────────────────────────────────────────

    @staticmethod
    def _lgbm_kwargs(config: DGridConfig) -> dict[str, object]:
        return {
            "objective": "binary",
            "n_estimators": D_N_ESTIMATORS,
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "min_child_samples": config.min_data_in_leaf,
            "random_state": config.seed,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 1,
            "verbosity": -1,
        }

    def train(
        self,
        vectors: Sequence[FeatureVector],
        labels: Sequence[int],
        config: DGridConfig,
        *,
        calibrate: bool = True,
    ) -> None:
        """Fit on labeled train pairs (labels ∈ {0, 1}); deterministic
        under config.seed. Train-env only.

        ``calibrate=True`` (default) fits the Platt sigmoid on TRAIN-ONLY
        out-of-fold margins; the selection protocol passes
        ``calibrate=False`` (nested CV would spend 6× the budget on a
        monotone transform — the ranking metric is computed on the raw
        candidate, final training always calibrates).
        """
        import lightgbm as lgb  # local: train-env import (addendum P2)

        matrix, y = self._validate_train_input(vectors, labels, config)
        expected = _EXTENDED_NAMES if config.field_cosines else _CORE_NAMES
        clf = lgb.LGBMClassifier(**self._lgbm_kwargs(config))
        clf.fit(matrix, y)
        self._booster = clf.booster_
        self._config = config
        self.feature_names = expected
        self._platt = (
            self._fit_platt_params(matrix, y, config) if calibrate else (1.0, 0.0)
        )

    def _validate_train_input(
        self, vectors: Sequence[FeatureVector], labels: Sequence[int], config: DGridConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        if not vectors:
            raise ValueError("training requires at least one labeled pair")
        observed = {v.names for v in vectors}
        if len(observed) != 1:
            raise ValueError("all training vectors must share one feature-name contract")
        expected = _EXTENDED_NAMES if config.field_cosines else _CORE_NAMES
        names = observed.pop()
        if names != expected:
            kind = "with" if config.field_cosines else "without"
            raise ValueError(
                f"config {config.name} expects field cosines {kind} "
                f"({len(expected)} features), vectors carry {len(names)}"
            )
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (len(vectors),):
            raise ValueError(f"labels length {len(y)} != vectors length {len(vectors)}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be binary {0, 1}")
        if len(np.unique(y)) < 2:
            raise ValueError("training requires both classes present")
        matrix = np.asarray([v.values for v in vectors], dtype=np.float64)
        return matrix, y

    def _fit_platt_params(
        self, matrix: np.ndarray, y: np.ndarray, config: DGridConfig
    ) -> tuple[float, float]:
        """Platt sigmoid (A, B) over TRAIN-ONLY out-of-fold raw margins."""
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        counts = np.bincount(y)
        if counts.size < 2 or int(counts.min()) < 2:
            warnings.warn(
                "a class has < 2 members — Platt calibration skipped (identity)",
                stacklevel=2,
            )
            return (1.0, 0.0)
        n_splits = int(min(PLATT_CV_FOLDS, counts.min()))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        margins = np.zeros(len(y), dtype=np.float64)
        for train_idx, valid_idx in skf.split(matrix, y):
            fold = lgb.LGBMClassifier(**self._lgbm_kwargs(config))
            fold.fit(matrix[train_idx], y[train_idx])
            margins[valid_idx] = fold.predict(matrix[valid_idx], raw_score=True)
        lr = LogisticRegression(C=float("inf"), solver="lbfgs")  # unpenalized Platt fit
        lr.fit(margins.reshape(-1, 1), y)
        return (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    # ── inference ─────────────────────────────────────────────────────────────

    def _require_fitted(self) -> object:
        if self._booster is None:
            raise RuntimeError("DBoostModel is not fitted — call train() first")
        return self._booster

    def predict_proba(self, vectors: Sequence[FeatureVector]) -> np.ndarray:
        """P(duplicate) per pair, float64 array shape (n,), values in [0, 1]."""
        booster = self._require_fitted()
        if not vectors:
            return np.zeros(0, dtype=np.float64)
        first = vectors[0].names
        if first != self.feature_names:
            raise ValueError(
                f"vectors carry {len(first)} features, model expects {len(self.feature_names)}"
            )
        matrix = np.asarray([v.values for v in vectors], dtype=np.float64)
        margins = booster.predict(matrix, raw_score=True)
        platt_a, platt_b = self._platt
        return _sigmoid(platt_a * margins + platt_b)

    # ── export ────────────────────────────────────────────────────────────────

    def export_onnx(
        self, path: Path, *, metadata_props: Mapping[str, str] | None = None
    ) -> Path:
        """Export the fitted model as a single ONNX (skl2onnx chain), opset
        15, input float32[K] → probability[1].

        The Platt calibration is folded into the graph as explicit nodes
        (see module docstring). Raises RuntimeError if unfitted or the file
        would exceed cortex.artifacts.MAX_ARTIFACT_BYTES.
        """
        booster = self._require_fitted()
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        model = self._build_onnx_model(booster, convert_lightgbm, FloatTensorType)
        if metadata_props:
            from cortex.artifacts import set_onnx_metadata

            set_onnx_metadata(model, metadata_props)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(model.SerializeToString())
        from cortex.artifacts import assert_artifact_size

        assert_artifact_size(out_path)
        return out_path

    def _build_onnx_model(self, booster, convert_lightgbm, float_tensor_type):
        import numpy as _np
        import onnx
        from onnx import TensorProto, helper

        k = len(self.feature_names)
        converted = convert_lightgbm(
            booster,
            initial_types=[("features_in", float_tensor_type([None, k]))],
            target_opset=15,
        )
        tree = next(
            (n for n in converted.graph.node if n.op_type == "TreeEnsembleClassifier"),
            None,
        )
        if tree is None:
            raise RuntimeError("conversion produced no TreeEnsembleClassifier node")

        platt_a, platt_b = self._platt

        def tensor(name: str, values, dtype) -> onnx.TensorProto:
            array = _np.asarray(values)
            return helper.make_tensor(
                name, dtype, dims=array.shape, vals=array.flatten().tolist()
            )

        nodes = [
            helper.make_node("Reshape", ["features", "shape_2d"], ["features_2d"]),
            tree,
            helper.make_node(
                "Slice",
                ["raw_pair", "slice_start", "slice_end", "slice_axis"],
                ["raw_margin"],
            ),
            helper.make_node("Mul", ["raw_margin", "platt_a"], ["scaled_margin"]),
            helper.make_node("Add", ["scaled_margin", "platt_b"], ["calibrated_logit"]),
            helper.make_node("Sigmoid", ["calibrated_logit"], ["probability_2d"]),
            helper.make_node("Reshape", ["probability_2d", "shape_out"], ["probability"]),
        ]
        # Rewire the tree node into the wrapped graph (raw margins out).
        # TreeEnsembleClassifier REQUIRES two outputs (label + scores) — the
        # label twin stays unwired on purpose.
        del tree.input[:]
        tree.input.extend(["features_2d"])
        del tree.output[:]
        tree.output.extend(["label_unused", "raw_pair"])
        for attr in tree.attribute:
            if attr.name == "post_transform":
                attr.s = b"NONE"  # Platt lives in explicit graph nodes

        initializers = [
            tensor("shape_2d", [1, k], TensorProto.INT64),
            tensor("slice_start", [1], TensorProto.INT64),
            tensor("slice_end", [2], TensorProto.INT64),
            tensor("slice_axis", [1], TensorProto.INT64),
            tensor("platt_a", [platt_a], TensorProto.FLOAT),
            tensor("platt_b", [platt_b], TensorProto.FLOAT),
            tensor("shape_out", [1], TensorProto.INT64),
        ]
        graph = helper.make_graph(
            nodes,
            name="vesma-cortex-d-boost",
            inputs=[
                helper.make_tensor_value_info("features", TensorProto.FLOAT, [k])
            ],
            outputs=[
                helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1])
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 15),
                helper.make_opsetid("ai.onnx.ml", 1),
            ],
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        self._smoke_onnx(model)
        return model

    def _smoke_onnx(self, model) -> None:
        """Eager ORT validation: the graph must reproduce the calibrated
        library probability on a synthetic zero feature row."""
        import onnxruntime as ort

        k = len(self.feature_names)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        zeros = np.zeros(k, dtype=np.float32)
        graph_prob = float(session.run(None, {"features": zeros})[0][0])
        reference = float(
            self.predict_proba([FeatureVector(self.feature_names, tuple(zeros.tolist()))])[0]
        )
        if not np.isclose(graph_prob, reference, atol=1e-5):
            raise RuntimeError(
                f"exported graph diverges from library prediction: {graph_prob} vs {reference}"
            )

    # ── persistence (dev pipeline: no pickle — LightGBM text format) ──────────

    def save(self, directory: Path) -> Path:
        """Persist fitted state (booster text + meta json). Dev-only format —
        NEVER the artifact (sklearn/lightgbm pickle artifacts are forbidden,
        ADR 0001 V3; this text dump is the train-epoch hand-off between the
        train and export CLI steps)."""
        booster = self._require_fitted()
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        booster_path = out_dir / "booster.txt"
        booster.save_model(str(booster_path))
        meta = {
            "candidate": "d-boost",
            "config": asdict(self._config) if self._config else None,
            "feature_names": list(self.feature_names),
            "platt": list(self._platt),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, directory: Path) -> "DBoostModel":
        import lightgbm as lgb

        in_dir = Path(directory)
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("candidate") != "d-boost":
            raise ValueError(f"{in_dir} is not a d-boost model directory")
        model = cls()
        model._booster = lgb.Booster(model_file=str(in_dir / "booster.txt"))
        model._platt = (float(meta["platt"][0]), float(meta["platt"][1]))
        model.feature_names = tuple(meta["feature_names"])
        model._config = DGridConfig(**meta["config"]) if meta.get("config") else None
        return model
