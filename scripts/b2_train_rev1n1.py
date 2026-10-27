#!/usr/bin/env python3
"""Train the B2 is-duplicate revision 1.n+1 models (prereg v3 + ADDENDUM-2).

Adapted from ``b2_train_rev1n.py`` (rev 1.n, v44-r2) — the recipe is the
FROZEN line-A1 convention and is NOT re-derived: LightGBM boosting over the
frozen pair features, the frozen grid GRID_D (non-field-cosines points),
selection by the frozen CV protocol (stratified 5-fold × 20 seeds, BA mean
then Brier — pre-calibration), final fit with Platt calibration, ONNX export
with metadata_props pinning ``features`` + ``feature_set_sha256`` +
``corpus_fingerprint``. Determinism: single-threaded deterministic LightGBM
under the config seed; a second identical invocation must produce
byte-identical boosters and predictions (the gate re-asserts).

v4.5 changes vs rev 1.n (constants only, no recipe drift):
- corpus dir ``data/b2-graph-corpus/v45``; sealed train fp ``3bf7ed53…``
  and labels fp ``cad44ab1…`` (manifest fingerprints, verified before any
  compute — a mismatch voids the run);
- ``corpus_fingerprint`` metadata pin = the v4.5 TRAIN fingerprint (the
  trained-on data; the sealed holdout is never read here);
- revision stamp ``1.n+1``.

Usage:
    python scripts/b2_train_rev1n1.py --run-dir <out> \
        [--sidecar <merged-train-sidecar.jsonl>] \
        [--corpus-dir data/b2-graph-corpus/v45]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from b2_rev1n_lib import (  # noqa: E402
    ABLATED_NAMES,
    GATED_NAMES,
    assert_corpus_identity,
    build_dataset,
    features_digest_of,
    load_jsonl,
)
from cortex.artifacts import (  # noqa: E402
    ARTIFACT_NAME,
    CANDIDATE_D,
    METADATA_VERSION,
    build_metadata_props,
    sha256_file,
)
from cortex.candidates.d_boost import GRID_D, DBoostModel  # noqa: E402
from cortex.features.pair import FeatureVector  # noqa: E402
from cortex.select.cv import CV_FOLDS, CV_SEEDS, run_cv  # noqa: E402

# v4.5 sealed fingerprints (data/b2-graph-corpus/v45/manifest.json,
# scheme manifest(pair_id pair_sha256 sorted) -> blake2b-256). Verified
# against disk BEFORE any compute; mismatch = STOP.
TRAIN_FP = "3bf7ed53e1dbf9092ec1e557c539b6fedd202ffd8fdbe9798643e2f6fc828a70"
LABELS_TRAIN_FP = "cad44ab10797c507f090fa2998b6dfc5edfc63f30a892506eadc75e4e98fdaac"
EMBEDDER_PIN = (
    "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
)
REVISION = "1.n+1"

#: Frozen eval threshold of the line (prereg lineage; NOT a g2 gate here).
DECISION_CUT = 0.5


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(run_dir: Path, event: str, payload: dict) -> None:
    entry = {"event": event, "ts": _utc_now(), **payload}
    with (run_dir / "run-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), file=sys.stderr)


def _vectors_from_matrix(names: tuple[str, ...], matrix: np.ndarray):
    return [FeatureVector(names, tuple(row)) for row in matrix]


def _cv_selector(names: tuple[str, ...], config):
    from cortex.candidates.d_boost import DBoostModel, DGridConfig

    gated = DGridConfig(
        name=config.name,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        min_data_in_leaf=config.min_data_in_leaf,
        field_cosines=False,
        seed=config.seed,
        feature_contract=tuple(names) if names != ABLATED_NAMES else None,
    )

    def train_fn(features_train, labels_train, seed):
        vectors = _vectors_from_matrix(names, features_train)
        model = DBoostModel()
        model.train(vectors, list(labels_train), gated, calibrate=False)
        return lambda x: model.predict_proba(_vectors_from_matrix(names, x))

    return train_fn


def select_config(names: tuple[str, ...], vectors, labels, run_dir: Path, tag: str):
    """Frozen CV protocol per grid point; pick by (BA mean, −Brier mean).

    Graph-incompatible grid points (field-cosines variants) are skipped —
    the same explicit-skip rule the CLI select uses without an A2 sidecar.
    """
    usable = [c for c in GRID_D if not c.field_cosines]
    matrix = np.asarray([v.values for v in vectors], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    reports = []
    for config in usable:
        t0 = time.monotonic()
        ba_mean, ba_std, brier_mean, brier_std = run_cv(
            _cv_selector(names, config), matrix, y
        )
        reports.append(
            {
                "config": config.name,
                "config_payload": {
                    "num_leaves": config.num_leaves,
                    "learning_rate": config.learning_rate,
                    "min_data_in_leaf": config.min_data_in_leaf,
                    "seed": config.seed,
                    "feature_contract": list(names),
                },
                "ba_mean": ba_mean,
                "ba_std": ba_std,
                "brier_mean": brier_mean,
                "brier_std": brier_std,
                "wall_s": round(time.monotonic() - t0, 1),
            }
        )
        _log(run_dir, "cv_point", {"model": tag, **reports[-1]})
    best = max(reports, key=lambda r: (r["ba_mean"], -r["brier_mean"]))
    return best, reports


def _per_stratum(provenance, probs, labels) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    out: dict[str, dict] = {}
    for key in sorted({p["stratum"] for p in provenance}):
        idx = [i for i, p in enumerate(provenance) if p["stratum"] == key]
        truth = [labels[i] for i in idx]
        pred = [int(probs[i] >= DECISION_CUT) for i in idx]
        dups = sum(truth)
        out[key] = {
            "n": len(idx),
            "dups": dups,
            "ba": float(balanced_accuracy_score(truth, pred))
            if 0 < dups < len(idx)
            else None,
            "pos_rate_pred": float(np.mean(pred)),
        }
    return out


def train_one(
    tag: str,
    names: tuple[str, ...],
    vectors,
    labels,
    provenance,
    run_dir: Path,
    corpus_fp: str,
) -> dict:
    """CV select → final calibrated fit → ONNX artifact. Returns the summary."""
    best, reports = select_config(names, vectors, labels, run_dir, tag)
    config = next(c for c in GRID_D if c.name == best["config"])
    gated_config = DGridConfigShim(config, names)

    t0 = time.monotonic()
    model = DBoostModel()
    model.train(vectors, labels, gated_config, calibrate=True)
    fit_s = round(time.monotonic() - t0, 1)

    in_sample = model.predict_proba(vectors)
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
    )

    char = {
        "model": tag,
        "feature_names": list(names),
        "feature_set_sha256": features_digest_of(names),
        "cv_selected": best,
        "cv_protocol": {
            "folds": CV_FOLDS,
            "seeds": list(CV_SEEDS),
            "calibration": "platt-at-final-fit-only",
        },
        "final_fit_wall_s": fit_s,
        "in_sample_ba": float(
            balanced_accuracy_score(labels, in_sample >= DECISION_CUT)
        ),
        "in_sample_brier": float(brier_score_loss(labels, in_sample)),
        "per_stratum_in_sample": _per_stratum(provenance, in_sample, labels),
        "platt": list(model._platt),  # noqa: SLF001 — run-log provenance
    }
    (run_dir / f"characterization-{tag}.json").write_text(
        json.dumps(char, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(
        run_dir,
        "characterization",
        {
            "model": tag,
            "in_sample_ba": char["in_sample_ba"],
            "in_sample_brier": char["in_sample_brier"],
            "cv_best": best["config"],
        },
    )

    model_dir = run_dir / f"model-{tag}"
    model.save(model_dir)
    out_onnx = run_dir / f"model-{tag}.onnx"
    trained_at = _utc_now()
    props = build_metadata_props(
        version=METADATA_VERSION,
        embedder_pin=EMBEDDER_PIN,
        corpus_fingerprint=corpus_fp,
        trained_at=trained_at,
        candidate=CANDIDATE_D,
        feature_names=model.feature_names,
    )
    model.export_onnx(out_onnx, metadata_props=props)
    weights_sha = sha256_file(out_onnx)
    manifest = {
        "name": ARTIFACT_NAME,
        "version": METADATA_VERSION,
        "sha256": weights_sha,
        "embedder_pin": EMBEDDER_PIN,
        "corpus_fingerprint": corpus_fp,
        "train_fingerprint": TRAIN_FP,
        "labels_fingerprint": LABELS_TRAIN_FP,
        "trained_at": trained_at,
        "candidate": CANDIDATE_D,
        "features": list(model.feature_names),
        "feature_set_sha256": props["feature_set_sha256"],
        "size_bytes": out_onnx.stat().st_size,
        "weights_path": str(out_onnx),
        "sanity_exam": "2",
        "revision": REVISION,
        "variant": tag,
        "cv_selected_config": best["config"],
    }
    manifest_path = out_onnx.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(
        run_dir,
        "artifact",
        {
            "model": tag,
            "sha256": weights_sha,
            "manifest": str(manifest_path),
            "feature_set_sha256": manifest["feature_set_sha256"],
        },
    )
    return {
        "model": tag,
        "weights_sha256": weights_sha,
        "feature_set_sha256": manifest["feature_set_sha256"],
        "cv_best": best["config"],
        "onnx": str(out_onnx),
    }


def DGridConfigShim(config, names: tuple[str, ...]):
    """Rebuild a grid point with the explicit gated feature contract."""
    from cortex.candidates.d_boost import DGridConfig

    return DGridConfig(
        name=config.name,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        min_data_in_leaf=config.min_data_in_leaf,
        field_cosines=False,
        seed=config.seed,
        feature_contract=tuple(names) if names != ABLATED_NAMES else None,
    )


def main(argv: list[str] | None = None) -> int:
    t_start = time.monotonic()
    parser = argparse.ArgumentParser(prog="b2-train-rev1n1")
    parser.add_argument("--corpus-dir", default="data/b2-graph-corpus/v45")
    parser.add_argument(
        "--sidecar",
        required=True,
        help="merged train sidecar (v44-r2 base + v45 new material)",
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── corpus identity (fail-loud before any compute) ────────────────────
    asserts = assert_corpus_identity(
        corpus_dir / "train.jsonl",
        corpus_dir / "labels-train.jsonl",
        corpus_dir / "manifest.json",
        TRAIN_FP,
        LABELS_TRAIN_FP,
    )
    _log(
        run_dir,
        "corpus_identity",
        {
            "rows": asserts.rows,
            "dups": asserts.dups,
            "not_dups": asserts.not_dups,
            "train_fingerprint": asserts.train_fingerprint,
            "labels_fingerprint": asserts.labels_fingerprint,
            "corpus_fingerprint_pin": TRAIN_FP,
            "revision": REVISION,
        },
    )

    rows = load_jsonl(corpus_dir / "train.jsonl")
    from cortex.features.graph import GraphSidecar

    sidecar = GraphSidecar(args.sidecar)
    _log(run_dir, "sidecar_loaded", {"path": args.sidecar, "rows": len(sidecar)})

    import resource

    results = {}
    for tag, names in (("full", GATED_NAMES), ("ablated", ABLATED_NAMES)):
        vectors, labels, provenance = build_dataset(rows, sidecar, names)
        kinds = {}
        for p in provenance:
            kinds[p["evidence_kind"]] = kinds.get(p["evidence_kind"], 0) + 1
        _log(
            run_dir,
            "matrix_built",
            {
                "model": tag,
                "n": len(vectors),
                "width": len(names),
                "feature_set_sha256": features_digest_of(names),
                "evidence_kinds": kinds,
                "maxrss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                ),
            },
        )
        results[tag] = train_one(
            tag, names, vectors, labels, provenance, run_dir, TRAIN_FP
        )

    summary = {
        "models": results,
        "wall_s": round(time.monotonic() - t_start, 1),
        "maxrss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "corpus_fingerprint": TRAIN_FP,
        "revision": REVISION,
        "sidecar": args.sidecar,
        "python": sys.version.split()[0],
    }
    (run_dir / "train-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(run_dir, "train_done", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
