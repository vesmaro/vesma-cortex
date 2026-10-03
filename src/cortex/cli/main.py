"""cortex CLI — pipeline entry points.

Subcommands mirror the pipeline order: export-corpus (A2, prereg hygiene +
fingerprints) → pretrain (A3b, corruption pairs) → train (A3b, D/N) →
select (A3b, frozen CV) → export-artifact (A6, ONNX + metadata_props) →
eval (A5, single-shot runner).

Exit codes (machine-checkable, latin-only on stderr):

- 0 — success;
- 2 — usage error / invalid input (bad store, contract violation);
- 3 — train-env dependency missing (torch for the N candidate);
- 4 — single-shot run refused (this corpus was already evaluated).

A3b data convention: the TRAIN-side manifest is a labeled jsonl — one
data-contract §3 pair line (pair_id, record, candidate, similarity, ...)
plus a ``label`` field ("duplicate"/"not-duplicate"/"disputed") and,
for candidate N, optional ``vec_a``/``vec_b`` (384-dim store embeddings).
Disputed rows are skipped with a counter. The physically-split holdout
labels never appear here (cortex.data.holdout owns that side).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from cortex.data.fingerprints import (
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.features.pair import FEATURE_NAMES, PairRecord, features

__all__ = ["build_parser", "main"]

#: Exit code for a missing train-env dependency (torch extra).
ENV_MISSING_EXIT: Final[int] = 3

#: Exit code for a refused single-shot run (prereg guard).
RUN_REFUSED_EXIT: Final[int] = 4

#: corpus-id shape (data-contract §2 + ADR 0001 V4: machine strings are
#: latin-only; also a path-safety guard — it becomes a directory name).
_CORPUS_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


# ── shared loading helpers (train-side manifest) ──────────────────────────────


def _record_like(side: dict[str, Any]) -> PairRecord:
    return PairRecord(
        title=str(side.get("title", "")),
        body=str(side.get("body", "")),
        tags=tuple(str(tag) for tag in side.get("tags", ())),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def _load_labeled_pairs(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load the labeled train manifest; disputed rows are skipped+counted."""
    rows: list[dict[str, Any]] = []
    disputed = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label")
        if label == "disputed":
            disputed += 1
            continue
        if label not in ("duplicate", "not-duplicate"):
            raise ValueError(
                f"{path}: pair {row.get('pair_id')!r} carries bad label {label!r}"
            )
        if "record" not in row or "candidate" not in row or "similarity" not in row:
            raise ValueError(
                f"{path}: pair {row.get('pair_id')!r} misses record/candidate/similarity"
            )
        rows.append(row)
    if not rows:
        raise ValueError(
            f"{path}: no usable labeled pairs (disputed skipped: {disputed})"
        )
    return rows, disputed


def _manifest_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    """Corpus fingerprint over the loaded manifest rows (frozen scheme)."""
    entries = (
        (
            row["pair_id"],
            pair_sha256(
                {
                    "record": row["record"],
                    "candidate": row["candidate"],
                    "similarity": row["similarity"],
                }
            ),
        )
        for row in rows
    )
    return corpus_fingerprint(manifest_bytes(entries))


def _pair_feature_vector(row: dict[str, Any]):
    return features(
        _record_like(row["record"]),
        _record_like(row["candidate"]),
        float(row["similarity"]),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _d_config_by_name(name: str):
    from cortex.candidates.d_boost import GRID_D

    for config in GRID_D:
        if config.name == name:
            return config
    raise ValueError(
        f"unknown D grid config {name!r} — frozen grid: {[c.name for c in GRID_D]}"
    )


def _n_config_by_name(name: str):
    from cortex.candidates.n_head import GRID_N

    for config in GRID_N:
        if config.name == name:
            return config
    raise ValueError(
        f"unknown N grid config {name!r} — frozen grid: {[c.name for c in GRID_N]}"
    )


# ── command handlers ──────────────────────────────────────────────────────────


def _cmd_export_corpus(args: argparse.Namespace) -> int:
    from cortex.data.store_export import (
        export_store_corpus,
        make_scanner,
    )

    if args.min_cosine >= args.max_cosine:
        raise ValueError(
            f"--min-cosine must be < --max-cosine, got [{args.min_cosine}, {args.max_cosine})"
        )
    if args.limit_pool <= 0:
        raise ValueError(f"--limit-pool must be positive, got {args.limit_pool}")
    if (args.store_path is None) == (args.store_uri is None):
        raise ValueError("exactly one of --store-path / --store-uri is required")
    corpus_id = args.corpus_id or f"pretrain-{datetime.now(UTC).strftime('%Y%m%d')}"
    if not _CORPUS_ID_RE.fullmatch(corpus_id):
        raise ValueError(
            f"--corpus-id must match {_CORPUS_ID_RE.pattern!r} (machine string, ADR 0001 V4), "
            f"got {corpus_id!r}"
        )

    scanner = make_scanner(args.engine_src)
    if scanner.provenance != "engine":
        print(
            f"cortex export-corpus: WARNING scanner provenance is {scanner.provenance!r} — "
            "the prereg hygiene expects the ENGINE detectors (--engine-src / $CORTEX_ENGINE_SRC); "
            "any report built from this export must disclose the fallback",
            file=sys.stderr,
        )

    def progress(message: str) -> None:
        print(f"cortex export-corpus: {message}", file=sys.stderr)

    result = export_store_corpus(
        store_path=args.store_path,
        store_uri=args.store_uri,
        out_dir=args.out,
        corpus_id=corpus_id,
        scanner=scanner,
        limit_pool=args.limit_pool,
        min_cosine=args.min_cosine,
        max_cosine=args.max_cosine,
        with_pairs=not args.no_pairs,
        progress=progress,
    )
    if result.corpus_fingerprint is None:
        progress("near-duplicate pair bases skipped (--no-pairs)")
    print(json.dumps(result.summary(), ensure_ascii=False))
    return 0


def _cmd_pretrain(args: argparse.Namespace) -> int:
    from cortex.pretrain.corruption import generate_pretrain_pairs

    records: list[PairRecord] = []
    vectors_by_key: dict[tuple[str, ...], list[float]] = {}
    for line in Path(args.corpus).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        record = _record_like(raw)
        records.append(record)
        if "vec" in raw:  # store embedding rides along when exported (N pretrain)
            vectors_by_key[(record.title, record.body, record.tags)] = list(raw["vec"])
    if not records:
        raise ValueError(f"{args.corpus}: no records to corrupt")
    pairs = generate_pretrain_pairs(records, seed=args.seed, max_pairs=args.max_pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            row = {
                "pair_id": pair.pair_id,
                "record": {
                    "title": pair.record_a.title,
                    "body": pair.record_a.body,
                    "tags": list(pair.record_a.tags),
                    "language": pair.record_a.language,
                    "record_type": pair.record_a.record_type,
                },
                "candidate": {
                    "title": pair.record_b.title,
                    "body": pair.record_b.body,
                    "tags": list(pair.record_b.tags),
                    "language": pair.record_b.language,
                    "record_type": pair.record_b.record_type,
                },
                "similarity": 1.0,  # self-pair convention (corruption.py docstring)
                "source": "weak-positive" if pair.positive else "hard-negative",
                "transform": pair.transform_name,
            }
            # Self-pair: the BASE record's store vector stands for both
            # sides (corruption.py module docstring) — pass it through when
            # the records export carried one.
            key = (pair.record_a.title, pair.record_a.body, pair.record_a.tags)
            if key in vectors_by_key:
                row["vec_a"] = row["vec_b"] = vectors_by_key[key]
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[row["source"]] = counts.get(row["source"], 0) + 1
            entries.append(
                (
                    pair.pair_id,
                    pair_sha256(
                        {
                            "record": row["record"],
                            "candidate": row["candidate"],
                            "similarity": row["similarity"],
                        }
                    ),
                )
            )
    fingerprint = corpus_fingerprint(manifest_bytes(entries))
    (out_path.parent / "manifest.txt").write_bytes(manifest_bytes(entries))
    print(
        json.dumps(
            {
                "out": str(out_path),
                "pairs": len(pairs),
                "weak_positive": counts.get("weak-positive", 0),
                "hard_negative": counts.get("hard-negative", 0),
                "seed": args.seed,
                "corpus_fingerprint": fingerprint,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _train_d(
    rows: list[dict[str, Any]], config_name: str, out_dir: Path, fingerprint: str
) -> int:
    from cortex.candidates.d_boost import DBoostModel

    config = _d_config_by_name(config_name)
    vectors = [_pair_feature_vector(row) for row in rows]
    labels = [1 if row["label"] == "duplicate" else 0 for row in rows]
    model = DBoostModel()
    model.train(vectors, labels, config)
    model.save(out_dir)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["corpus_fingerprint"] = fingerprint
    meta["n_pairs"] = len(rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate": "d-boost",
                "config": config.name,
                "out": str(out_dir),
                "pairs": len(rows),
                "corpus_fingerprint": fingerprint,
            }
        )
    )
    return 0


def _train_n(
    rows: list[dict[str, Any]],
    config_name: str | None,
    out_dir: Path,
    fingerprint: str,
    pretrain_corpus: Path | None,
) -> int:
    try:
        import torch  # noqa: F401 — train extra presence probe
    except ImportError:
        print(
            "cortex train: candidate n needs torch (train extra) — "
            "sync with `uv sync --extra train`",
            file=sys.stderr,
        )
        return ENV_MISSING_EXIT

    from cortex.candidates.n_head import GRID_N, NHeadModel, vector_block

    config = _n_config_by_name(config_name or GRID_N[0].name)
    missing_vecs = [
        row["pair_id"] for row in rows if "vec_a" not in row or "vec_b" not in row
    ]
    if missing_vecs:
        print(
            f"cortex train: candidate n needs vec_a/vec_b on every pair "
            f"({len(missing_vecs)} missing, first: {missing_vecs[:3]})",
            file=sys.stderr,
        )
        return ENV_MISSING_EXIT

    vectors = [_pair_feature_vector(row) for row in rows]
    labels = [1 if row["label"] == "duplicate" else 0 for row in rows]
    blocks = [vector_block(row["vec_a"], row["vec_b"]) for row in rows]

    model = NHeadModel()
    if pretrain_corpus is not None:
        pretrain_rows = [
            json.loads(line)
            for line in Path(pretrain_corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        no_vec = [
            r["pair_id"] for r in pretrain_rows if "vec_a" not in r or "vec_b" not in r
        ]
        if no_vec:
            print(
                f"cortex train: pretrain corpus lacks store vectors on {len(no_vec)} "
                f"pairs (first: {no_vec[:3]}) — regenerate it from a records export "
                "that carries 'vec'",
                file=sys.stderr,
            )
            return ENV_MISSING_EXIT
        pre_vectors = [
            features(
                _record_like(r["record"]),
                _record_like(r["candidate"]),
                float(r["similarity"]),
            )
            for r in pretrain_rows
        ]
        pre_blocks = [vector_block(r["vec_a"], r["vec_b"]) for r in pretrain_rows]
        pre_labels = [1 if r["source"] == "weak-positive" else 0 for r in pretrain_rows]
        model.pretrain(list(zip(pre_vectors, pre_blocks)), config, labels=pre_labels)
    model.train(vectors, labels, config, vector_blocks=blocks)
    model.save(out_dir)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["corpus_fingerprint"] = fingerprint
    meta["n_pairs"] = len(rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate": "n-head",
                "config": config.name,
                "out": str(out_dir),
                "pairs": len(rows),
                "pretrain": str(pretrain_corpus) if pretrain_corpus else None,
                "corpus_fingerprint": fingerprint,
            }
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from cortex.candidates.d_boost import GRID_D

    rows, disputed = _load_labeled_pairs(Path(args.train_manifest))
    if disputed:
        print(f"cortex train: skipped {disputed} disputed pairs", file=sys.stderr)
    fingerprint = _manifest_fingerprint(rows)
    out_dir = Path(args.out)
    config_name = args.config  # None → first frozen grid point per candidate

    if args.candidate == "d":
        return _train_d(
            rows, config_name or GRID_D[0].name, out_dir / "d-boost", fingerprint
        )
    if args.candidate == "n":
        return _train_n(
            rows, config_name, out_dir / "n-head", fingerprint, args.pretrain_corpus
        )
    code_d = _train_d(
        rows, config_name or GRID_D[0].name, out_dir / "d-boost", fingerprint
    )
    code_n = _train_n(
        rows, config_name, out_dir / "n-head", fingerprint, args.pretrain_corpus
    )
    return code_d if code_d != 0 else code_n


def _cv_matrix(rows: list[dict[str, Any]], with_vectors: bool):
    """Feature matrix for the CV protocol: scalars (D) or scalars +
    flattened vector block (N — the block rides the same fold indices)."""
    import numpy as np

    scalars = np.asarray(
        [_pair_feature_vector(row).values for row in rows], dtype=np.float64
    )
    if not with_vectors:
        return scalars
    from cortex.candidates.n_head import vector_block

    blocks = np.asarray(
        [vector_block(row["vec_a"], row["vec_b"]) for row in rows], dtype=np.float64
    )
    return np.concatenate([scalars, blocks.reshape(len(rows), -1)], axis=1)


def _select_d(matrix, labels, config) -> object:
    from cortex.candidates.d_boost import DBoostModel
    from cortex.features.pair import FeatureVector

    def train_fn(features_train, labels_train, seed):
        vectors = [FeatureVector(FEATURE_NAMES, tuple(row)) for row in features_train]
        model = DBoostModel()
        model.train(vectors, list(labels_train), config, calibrate=False)
        return lambda x: model.predict_proba(
            [FeatureVector(FEATURE_NAMES, tuple(r)) for r in x]
        )

    return train_fn


def _select_n(matrix, labels, config):
    from cortex.candidates.n_head import NHeadModel
    from cortex.features.pair import FeatureVector

    scalar_dim = len(FEATURE_NAMES)

    def train_fn(features_train, labels_train, seed):
        vectors = [
            FeatureVector(FEATURE_NAMES, tuple(row[:scalar_dim]))
            for row in features_train
        ]
        blocks = features_train[:, scalar_dim:].reshape(-1, 4, 384)
        model = NHeadModel()
        model.train(vectors, list(labels_train), config, vector_blocks=blocks)
        return lambda x: model.predict_proba(
            [FeatureVector(FEATURE_NAMES, tuple(r[:scalar_dim])) for r in x],
            vector_blocks=x[:, scalar_dim:].reshape(-1, 4, 384),
        )

    return train_fn


def _cmd_select(args: argparse.Namespace) -> int:
    import numpy as np

    from cortex.candidates.d_boost import GRID_D
    from cortex.select.cv import CvReport, run_cv, select_candidate

    rows, disputed = _load_labeled_pairs(Path(args.train_manifest))
    if disputed:
        print(f"cortex select: skipped {disputed} disputed pairs", file=sys.stderr)
    labels = np.asarray(
        [1 if row["label"] == "duplicate" else 0 for row in rows], dtype=np.int64
    )

    reports: list[CvReport] = []
    d_matrix = _cv_matrix(rows, with_vectors=False)
    # The CLI feature pass is core-only (text features); field-cosines
    # ablation points need the A2 sidecar (attach_field_cosines) — skip
    # them explicitly instead of failing the whole protocol run.
    usable_d = [config for config in GRID_D if not config.field_cosines]
    skipped_d = [config.name for config in GRID_D if config.field_cosines]
    if skipped_d:
        print(
            f"cortex select: field-cosines grid points skipped (no A2 sidecar "
            f"in this manifest): {skipped_d}",
            file=sys.stderr,
        )
    for config in usable_d:
        ba_mean, ba_std, brier_mean, brier_std = run_cv(
            _select_d(d_matrix, labels, config), d_matrix, labels
        )
        reports.append(
            CvReport("d-boost", config.name, ba_mean, ba_std, brier_mean, brier_std)
        )
    d_best = max(reports, key=lambda r: (r.balanced_accuracy_mean, -r.brier_mean))

    n_best: CvReport | None = None
    if args.candidate in ("n", "both"):
        try:
            import torch  # noqa: F401

            from cortex.candidates.n_head import GRID_N

            n_matrix = _cv_matrix(rows, with_vectors=True)
            n_reports: list[CvReport] = []
            for config in GRID_N:
                ba_mean, ba_std, brier_mean, brier_std = run_cv(
                    _select_n(n_matrix, labels, config), n_matrix, labels
                )
                n_reports.append(
                    CvReport(
                        "n-head", config.name, ba_mean, ba_std, brier_mean, brier_std
                    )
                )
            n_best = max(
                n_reports, key=lambda r: (r.balanced_accuracy_mean, -r.brier_mean)
            )
            reports.extend(n_reports)
        except ImportError:
            print(
                "cortex select: torch absent — N candidate skipped (train extra)",
                file=sys.stderr,
            )
        except KeyError as exc:
            print(
                f"cortex select: N needs vec_a/vec_b on every pair ({exc})",
                file=sys.stderr,
            )

    verdict = select_candidate(d_best, n_best)

    def _margin(value: float) -> float | str:
        # json.dumps would emit the non-standard literal `Infinity`
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return value

    payload = {
        "reports": [report.__dict__ for report in reports],
        "d_best": d_best.__dict__,
        "n_best": n_best.__dict__ if n_best else None,
        "verdict": {
            "winner": verdict.winner,
            "margin_stds": _margin(verdict.margin_stds),
        },
        "note": "selection metrics are pre-calibration (calibrate=False): "
        "the Platt sigmoid is fitted at final training only",
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["verdict"]))
    if args.out:
        print(f"report: {args.out}")
    return 0


def _cmd_export_artifact(args: argparse.Namespace) -> int:
    from cortex.artifacts import (
        ARTIFACT_NAME,
        CANDIDATE_D,
        CANDIDATE_N,
        METADATA_VERSION,
        build_metadata_props,
        sha256_file,
    )

    model_dir = Path(args.model)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    candidate = meta.get("candidate")
    corpus_fp = args.corpus_fingerprint or meta.get("corpus_fingerprint", "")
    if not corpus_fp:
        print(
            "cortex export-artifact: no corpus fingerprint (meta lacks it and "
            "no --corpus-fingerprint) — artifact metadata would be a lie",
            file=sys.stderr,
        )
        return 2
    trained_at = _utc_now()

    out_path = Path(args.out)
    if candidate == "d-boost":
        from cortex.candidates.d_boost import DBoostModel

        model = DBoostModel.load(model_dir)
        props = build_metadata_props(
            version=METADATA_VERSION,
            embedder_pin=args.embedder_pin,
            corpus_fingerprint=corpus_fp,
            trained_at=trained_at,
            candidate=CANDIDATE_D,
            feature_names=model.feature_names,
        )
        model.export_onnx(out_path, metadata_props=props)
    elif candidate == "n-head":
        from cortex.candidates.n_head import NHeadModel

        model = NHeadModel.load(model_dir)
        props = build_metadata_props(
            version=METADATA_VERSION,
            embedder_pin=args.embedder_pin,
            corpus_fingerprint=corpus_fp,
            trained_at=trained_at,
            candidate=CANDIDATE_N,
            feature_names=model.feature_names,
        )
        model.export_onnx(out_path, metadata_props=props)
    else:
        print(
            f"cortex export-artifact: unknown candidate {candidate!r}", file=sys.stderr
        )
        return 2

    weights_sha = sha256_file(out_path)
    manifest = {
        "name": ARTIFACT_NAME,
        "version": METADATA_VERSION,
        "sha256": weights_sha,
        "embedder_pin": args.embedder_pin,
        "corpus_fingerprint": corpus_fp,
        "trained_at": trained_at,
        "candidate": candidate,
        "features": list(model.feature_names),
        "feature_set_sha256": props["feature_set_sha256"],
        "size_bytes": out_path.stat().st_size,
        "weights_path": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path),
                "sha256": weights_sha,
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


class _OrtDModel:
    """ScoredModel adapter over an exported d-boost ONNX artifact."""

    def __init__(self, session, feature_names: tuple[str, ...]) -> None:
        self._session = session
        self._feature_names = feature_names

    def predict_proba(self, pairs: Sequence[object]) -> Sequence[float]:
        import numpy as np

        vectors = [
            features(
                _record_like(pair.record or {}),
                _record_like(pair.candidate or {}),
                float(pair.similarity),
            )
            for pair in pairs
        ]
        matrix = np.asarray([v.values for v in vectors], dtype=np.float32)
        if matrix.shape[1] != len(self._feature_names):
            raise ValueError(
                f"artifact expects {len(self._feature_names)} features, "
                f"pairs yield {matrix.shape[1]} — feature contract mismatch"
            )
        out: list[float] = []
        for row in matrix:
            (probability,) = self._session.run(None, {"features": row})
            value = float(probability[0])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"artifact probability {value} outside [0, 1]")
            out.append(value)
        return out


def _cmd_eval(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    from cortex.artifacts import MODEL_NAME
    from cortex.eval.runner import (
        RunLog,
        RunLogRefusalError,
        load_holdout_pairs,
        run_single_shot,
    )
    from cortex.features.pair import FEATURE_NAMES as CORE_NAMES

    artifact = Path(args.artifact)
    try:
        session = ort.InferenceSession(
            str(artifact), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        # ORT load failures raise pybind11-state subclasses of Exception
        # (not ValueError/RuntimeError) — the CLI boundary maps them to the
        # documented exit-2 family instead of an uncaught traceback (exit 1).
        print(f"cortex eval: artifact load failed — {exc}", file=sys.stderr)
        return 2
    meta = dict(session.get_modelmeta().custom_metadata_map)
    if meta.get("name") != MODEL_NAME:
        print(
            f"cortex eval: artifact name {meta.get('name')!r} != {MODEL_NAME!r}",
            file=sys.stderr,
        )
        return 2
    candidate = meta.get("candidate")
    if candidate == "n-head":
        print(
            "cortex eval: n-head artifacts need vector sidecars — A5 runner "
            "wiring (ADR 0001 V1 dependency)",
            file=sys.stderr,
        )
        return 2
    feature_names = tuple(meta.get("features", "").split("\n"))
    model = _OrtDModel(session, feature_names or CORE_NAMES)

    holdout_dir = Path(args.holdout)
    pairs = load_holdout_pairs(
        holdout_dir / "pairs.jsonl", holdout_dir / "labels.jsonl"
    )
    if args.corpus_fingerprint:
        corpus_fp = args.corpus_fingerprint
    else:
        rows = [
            json.loads(line)
            for line in (holdout_dir / "pairs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        corpus_fp = _manifest_fingerprint(rows)
    labels_map: dict[str, str] = {}
    for line in (holdout_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            label_row = json.loads(line)
            labels_map[label_row["pair_id"]] = label_row["label"]

    from cortex.artifacts import sha256_file

    run_log = RunLog(Path(args.run_log))
    try:
        report = run_single_shot(
            model,
            pairs,
            run_log=run_log,
            weights_sha256=sha256_file(artifact),
            corpus_fingerprint=corpus_fp,
            label_fingerprint=labels_fingerprint(labels_map),
        )
    except RunLogRefusalError as exc:
        print(f"cortex eval: refused — {exc}", file=sys.stderr)
        return RUN_REFUSED_EXIT
    print(json.dumps(report.__dict__, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The frozen CLI surface: subcommand names are contract (runbooks and
    the charter's CPU-smoke gate reference them); options may grow later."""
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="vesma-cortex development pipeline (train epoch library)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "export-corpus",
        help="export eval/pretrain corpus from the store (prereg hygiene, fingerprints)",
    )
    p.add_argument(
        "--store-path",
        default=None,
        help="store data DIRECTORY holding mnemos.db + vectors.db (opened strictly read-only)",
    )
    p.add_argument(
        "--store-uri",
        default=None,
        help="contract-form read-only URI of mnemos.db (file:...?mode=ro); vectors.db is its sibling",
    )
    p.add_argument(
        "--out",
        "--out-dir",
        dest="out",
        required=True,
        help="output root under data/ (corpus lands in <out>/pretrain/<corpus-id>/)",
    )
    p.add_argument("--corpus-id", default=None, help="default: pretrain-<UTC date>")
    p.add_argument(
        "--limit-pool",
        type=int,
        default=800,
        help="max records in the pool (default 800)",
    )
    p.add_argument(
        "--min-cosine",
        type=float,
        default=0.85,
        help="near-dup pair band lower bound (inclusive)",
    )
    p.add_argument(
        "--max-cosine",
        type=float,
        default=0.97,
        help="near-dup pair band upper bound (exclusive)",
    )
    p.add_argument(
        "--no-pairs", action="store_true", help="skip near-duplicate pair bases"
    )
    p.add_argument(
        "--engine-src",
        default=None,
        help="engine source tree for the hygiene detectors (default: $CORTEX_ENGINE_SRC); "
        "absent → local fallback scanner, disclosed in provenance",
    )

    p = sub.add_parser(
        "pretrain", help="generate corruption pretrain pairs (candidate N)"
    )
    p.add_argument(
        "--corpus",
        required=True,
        help="records jsonl (title/body/tags/language/record_type rows)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", required=True, help="output pairs.jsonl (data-contract §4 shape)"
    )
    p.add_argument("--max-pairs", type=int, default=None)

    p = sub.add_parser("train", help="train D and N candidates on labeled train pairs")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--candidate", choices=["d", "n", "both"], default="both")
    p.add_argument(
        "--config",
        default=None,
        help="frozen grid config name (default: first of the grid)",
    )
    p.add_argument(
        "--pretrain-corpus",
        default=None,
        help="corruption pairs jsonl (candidate N pretrain stage)",
    )
    p.add_argument("--out", required=True, help="output directory for the model state")

    p = sub.add_parser(
        "select", help="frozen CV protocol: stratified 5-fold × 20 seeds"
    )
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--candidate", choices=["d", "n", "both"], default="both")
    p.add_argument("--out", default=None, help="selection report json path")

    p = sub.add_parser(
        "export-artifact", help="export the winning candidate as vesma-cortex-v1 ONNX"
    )
    p.add_argument(
        "--model", required=True, help="trained model directory (train --out)"
    )
    p.add_argument("--out", required=True, help="output model.onnx path")
    p.add_argument(
        "--embedder-pin",
        required=True,
        help="live engine embedder fingerprint (nano:sha256:<hex>)",
    )
    p.add_argument(
        "--corpus-fingerprint",
        default=None,
        help="override the fingerprint recorded in model meta",
    )

    p = sub.add_parser(
        "eval", help="single-shot holdout evaluation (prereg v2, append-only run log)"
    )
    p.add_argument("--artifact", required=True)
    p.add_argument(
        "--holdout", required=True, help="directory with pairs.jsonl + labels.jsonl"
    )
    p.add_argument("--run-log", required=True)
    p.add_argument(
        "--corpus-fingerprint",
        default=None,
        help="default: fingerprint of the given pairs.jsonl",
    )

    return parser


_HANDLERS = {
    "export-corpus": _cmd_export_corpus,
    "pretrain": _cmd_pretrain,
    "train": _cmd_train,
    "select": _cmd_select,
    "export-artifact": _cmd_export_artifact,
    "eval": _cmd_eval,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the surface
        print(f"cortex: unknown command {args.command!r}", file=sys.stderr)
        return 2
    try:
        return handler(args)
    except (ValueError, RuntimeError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"cortex {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
