#!/usr/bin/env python3
"""CPU smoke of the full cortex pipeline — charter §6 gate (A3c).

Runs the WHOLE library contour on programmatic synthetic data (no store,
no network, no fixtures) and proves four things at once:

1. SHAPE — every pipeline stage works end-to-end through the real CLI:
   synthetic dataset (records + labeled pairs + the frozen holdout split
   with physical isolation asserts) → cortex pretrain → train D → select
   → export-artifact → eval (exit 0) → eval AGAIN (exit 4: the
   single-shot run-log guard refuses).
2. DETERMINISM — the whole contour runs TWICE into separate directories;
   data files, model state, the selection report and the eval metrics are
   compared exactly (see _compare_runs for the equality set and the two
   documented exclusions).
3. BUDGET — total wall time is reported against a mode-dependent ceiling:
   120 s for the default D contour (charter §6: CPU smoke of the pipeline
   shape before any XPU run), 300 s with ``--with-n`` (the frozen N
   selection protocol alone is ~500 torch fits per run — see
   BUDGET_WITH_N_SECONDS).
4. MECHANISM (``--with-n``, train extra only) — the N ladder runs for
   real: corruption pretrain → N fit → N inside the frozen CV protocol →
   select_candidate verdict. N LOSING to D on synthetic data is a NORMAL
   smoke outcome — the mechanism is checked, not N's superiority.

Why Python and not bash: the smoke reuses tests/synth.py as its synthetic
data library (no duplication), drives the CLI through
cortex.cli.main:main — the same code path as the ``cortex`` console
script, in one process, without per-stage interpreter restarts — parses
each stage's JSON stdout and cross-compares two complete runs. All of
that is native Python; a bash wrapper would be a brittle bash/python
hybrid around it.

Output contract (machine-readable): one JSON line per stage on stdout —
``{"stage": ..., "status": "ok"|"fail", "elapsed_s": ..., <numbers>}``;
the final line has ``stage == "smoke"``. CLI chatter (stdout/stderr) is
captured, never echoed, so the report stays parseable line-by-line. The
full report is also written to ``<workdir>/smoke_report.json`` with an
environment provenance block (python/torch/xpu) — the XPU walkthrough
runbook archives exactly that file.

Exit codes: 0 — every stage green in BOTH runs and the runs match;
1 — a stage failed or determinism broke (the failing stage is named in
the final JSON line).

Usage:
    uv run python scripts/smoke_pipeline.py              # D-only contour
    uv run --extra train python scripts/smoke_pipeline.py --with-n
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# tests/synth.py is the shared synthetic-data library (tests + smoke, one
# home); src/ first so a repo checkout always shadows any installed copy.
for _entry in (REPO_ROOT / "src", REPO_ROOT / "tests"):
    sys.path.insert(0, str(_entry))

import synth  # noqa: E402 — src/tests sys.path shim above

from cortex.cli.main import RUN_REFUSED_EXIT  # noqa: E402 — sys.path shim above
from cortex.cli.main import main as cortex_main  # noqa: E402 — sys.path shim above
from cortex.data.fingerprints import (  # noqa: E402 — sys.path shim above
    canonical_json,
    pair_sha256,
)
from cortex.data.holdout import (  # noqa: E402 — sys.path shim above
    SplitPair,
    assert_labels_isolated,
    assert_no_pair_overlap,
    split_holdout,
)

#: Frozen smoke knobs — determinism is the point, nothing here is random
#: at run time (synth generators are internally seeded).
SMOKE_SEED = 5  # corruption seed for `cortex pretrain`
VECTOR_SEED = 7  # synthetic store-vector seed (synth default)
N_RECORDS = 24  # records for the corruption pretrain corpus
N_PAIRS = 72  # labeled pairs → frozen 30 % holdout split
#: Charter §6 budget: the DEFAULT (D-only) contour fits in 120 s CPU.
BUDGET_SECONDS = 120.0
#: The --with-n mode adds the FROZEN selection protocol (5 grid points ×
#: 20 seeds × 5 folds = 500 torch fits per run, ~100 s per run on the dev
#: box) — that cost belongs to the protocol, not to pipeline bloat, so the
#: ceiling for this mode is 300 s (an environment-regression tripwire,
#: e.g. thread thrash or a broken determinism restore), not the §6 budget.
BUDGET_WITH_N_SECONDS = 300.0

#: Clearly-synthetic embedder pin: the smoke never loads a real embedder;
#: the pin only has to survive the metadata round-trip (nano:sha256:<hex>).
SMOKE_EMBEDDER_PIN = "nano:sha256:" + "cd" * 32

#: Eval metric fields compared exactly between the two runs.
_EVAL_METRIC_KEYS = (
    "sensitivity",
    "specificity",
    "brier",
    "balanced_accuracy",
    "baseline_balanced_accuracy",
)

#: Files compared BYTE-for-BYTE between run1 and run2 (data, model state,
#: selection report — everything deterministic by construction).
_BYTE_COMPARE_FILES = (
    "corpus/records.jsonl",
    "corpus/train.jsonl",
    "holdout/pairs.jsonl",
    "holdout/labels.jsonl",
    "pretrain/pairs.jsonl",
    "pretrain/manifest.txt",
    "models/d-boost/booster.txt",
    "models/d-boost/meta.json",
    "select.json",
)


class SmokeFailure(RuntimeError):
    """A stage failed or determinism broke — the final line names it."""


# ── stage plumbing ────────────────────────────────────────────────────────────

_RECORDS: list[dict[str, Any]] = []


def _emit(stage: str, status: str, elapsed: float, **numbers: Any) -> None:
    record = {
        "stage": stage,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        **numbers,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    _RECORDS.append(record)


def _run_stage(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    start = time.monotonic()
    try:
        payload = fn()
    except Exception as exc:  # the smoke reports the failing stage, then dies
        _emit(name, "fail", time.monotonic() - start, error=str(exc)[:500])
        raise
    _emit(name, "ok", time.monotonic() - start, **payload)
    return payload


def _cli(argv: list[str], expect: tuple[int, ...] = (0,)) -> tuple[int, str, str]:
    """Run one cortex CLI command via its real main(); capture chatter."""
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cortex_main(argv)
    if code not in expect:
        raise SmokeFailure(
            f"cortex {argv[0]} exited {code} (expected {expect}): {err.getvalue().strip()[:400]}"
        )
    return code, out.getvalue(), err.getvalue()


def _parse_json_output(text: str) -> dict[str, Any]:
    """CLI stage output → dict: whole-text JSON (eval prints indented),
    falling back to the last JSON line (select prints report path after)."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise SmokeFailure(f"no JSON object in CLI output: {stripped[:200]!r}")


# ── the contour (one full pipeline run inside `root`) ────────────────────────


def _side_sha(side: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(side).encode("utf-8")).hexdigest()


def _stage_synth(root: Path) -> dict[str, Any]:
    """Synthetic dataset: records (with store vecs) + labeled pairs (with
    vec_a/vec_b) + the FROZEN holdout split with both isolation asserts."""
    corpus_dir = root / "corpus"  # the train tree
    holdout_dir = root / "holdout"  # OUTSIDE the train tree (physical isolation)

    records = synth.make_records(N_RECORDS)
    records_path = synth.write_records_jsonl(
        corpus_dir / "records.jsonl", records, seed=VECTOR_SEED
    )

    rows = synth.attach_store_vectors(synth.make_pair_rows(N_PAIRS), seed=VECTOR_SEED)
    split = split_holdout(
        [
            SplitPair(
                pair_id=row["pair_id"],
                stratum=row["stratum"],
                label=row["label"],
                sha_a=_side_sha(row["record"]),
                sha_b=_side_sha(row["candidate"]),
                pair_sha256=pair_sha256(
                    {
                        "record": row["record"],
                        "candidate": row["candidate"],
                        "similarity": row["similarity"],
                    }
                ),
            )
            for row in rows
        ]
    )
    assert_no_pair_overlap(split.train_pair_ids, split.holdout_pair_ids)

    by_id = {row["pair_id"]: row for row in rows}
    train_rows = [by_id[pair_id] for pair_id in sorted(split.train_pair_ids)]
    synth.write_manifest(corpus_dir / "train.jsonl", train_rows)

    holdout_rows = [by_id[pair_id] for pair_id in sorted(split.holdout_pair_ids)]
    holdout_dir.mkdir(parents=True, exist_ok=True)
    (holdout_dir / "pairs.jsonl").write_text(
        "\n".join(
            json.dumps(
                {k: v for k, v in row.items() if k != "label"}, ensure_ascii=False
            )
            for row in holdout_rows
        ),
        encoding="utf-8",
    )
    (holdout_dir / "labels.jsonl").write_text(
        "\n".join(
            json.dumps({"pair_id": row["pair_id"], "label": row["label"]})
            for row in holdout_rows
        ),
        encoding="utf-8",
    )
    assert_labels_isolated(corpus_dir, holdout_dir / "labels.jsonl")
    return {
        "records": N_RECORDS,
        "train_pairs": len(train_rows),
        "holdout_pairs": len(holdout_rows),
        "corpus_fingerprint": split.corpus_fingerprint,
        "records_path": str(records_path),
        "train_manifest": str(corpus_dir / "train.jsonl"),
        "holdout_dir": str(holdout_dir),
    }


def _stage_pretrain(root: Path, records_path: str) -> dict[str, Any]:
    _, out, _ = _cli(
        [
            "pretrain",
            "--corpus",
            records_path,
            "--seed",
            str(SMOKE_SEED),
            "--out",
            str(root / "pretrain" / "pairs.jsonl"),
        ]
    )
    summary = _parse_json_output(out)
    return {
        "pairs": summary["pairs"],
        "weak_positive": summary["weak_positive"],
        "hard_negative": summary["hard_negative"],
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "pretrain_corpus": summary["out"],
    }


def _stage_train_d(root: Path, train_manifest: str) -> dict[str, Any]:
    _cli(
        [
            "train",
            "--train-manifest",
            train_manifest,
            "--candidate",
            "d",
            "--out",
            str(root / "models"),
        ]
    )
    meta = json.loads(
        (root / "models" / "d-boost" / "meta.json").read_text(encoding="utf-8")
    )
    return {
        "config": meta["config"]["name"],
        "pairs": meta["n_pairs"],
        "model_dir": str(root / "models" / "d-boost"),
    }


def _stage_train_n(
    root: Path, train_manifest: str, pretrain_corpus: str
) -> dict[str, Any]:
    _, out, _ = _cli(
        [
            "train",
            "--train-manifest",
            train_manifest,
            "--candidate",
            "n",
            "--pretrain-corpus",
            pretrain_corpus,
            "--out",
            str(root / "models"),
        ]
    )
    summary = _parse_json_output(out)
    meta = json.loads(
        (root / "models" / "n-head" / "meta.json").read_text(encoding="utf-8")
    )
    return {
        "config": meta["config"]["name"],
        "pairs": summary["pairs"],
        "pretrained": meta["pretrained"],
        "model_dir": str(root / "models" / "n-head"),
    }


def _stage_select(root: Path, train_manifest: str, with_n: bool) -> dict[str, Any]:
    report_path = root / "select.json"
    _cli(
        [
            "select",
            "--train-manifest",
            train_manifest,
            "--candidate",
            "both" if with_n else "d",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    d = report["d_best"]
    payload: dict[str, Any] = {
        "winner": report["verdict"]["winner"],
        "margin_stds": report["verdict"]["margin_stds"],
        "d": {
            "config": d["config_name"],
            "ba_mean": d["balanced_accuracy_mean"],
            "brier_mean": d["brier_mean"],
        },
        "n": None,
    }
    if report["n_best"]:
        n = report["n_best"]
        payload["n"] = {
            "config": n["config_name"],
            "ba_mean": n["balanced_accuracy_mean"],
            "brier_mean": n["brier_mean"],
        }
    return payload


def _stage_export(root: Path, model_dir: str, winner: str) -> dict[str, Any]:
    # Always exports D: an n-head artifact needs vector sidecars in the eval
    # runner = A5 wiring (ADR 0001 V1 dependency); the smoke's single-shot
    # leg is D-shaped by contract. If select said n-head, that is stated
    # here explicitly — an honest note, never a silent swap.
    _, out, _ = _cli(
        [
            "export-artifact",
            "--model",
            model_dir,
            "--out",
            str(root / "model.onnx"),
            "--embedder-pin",
            SMOKE_EMBEDDER_PIN,
        ]
    )
    summary = _parse_json_output(out)
    return {
        "exported": "d-boost",
        "winner_note": "n-head won select; D exported for the eval leg (N eval = A5 wiring)"
        if winner == "n-head"
        else None,
        "sha256": summary[
            "sha256"
        ],  # per-run info; NOT determinism-compared (trained_at inside)
        "size_bytes": (root / "model.onnx").stat().st_size,
        "artifact": str(root / "model.onnx"),
    }


def _stage_eval(root: Path, artifact: str, holdout_dir: str) -> dict[str, Any]:
    run_log = root / "run_log.jsonl"
    argv = [
        "eval",
        "--artifact",
        artifact,
        "--holdout",
        holdout_dir,
        "--run-log",
        str(run_log),
    ]
    _, out, _ = _cli(argv)
    report = _parse_json_output(out)
    # Second eval over the SAME run log: the single-shot guard must refuse
    # (prereg v2 W5c) — exit 4 is the EXPECTED outcome here.
    code, _, err = _cli(argv, expect=(RUN_REFUSED_EXIT,))
    metrics = {key: report[key] for key in _EVAL_METRIC_KEYS}
    return {"second_eval_exit": code, "refusal": "single-shot", **metrics}


def _run_contour(root: Path, *, with_n: bool, run_tag: str) -> dict[str, Any]:
    """One full pipeline run; every stage emits its own JSON line."""
    synth_out = _run_stage(f"synth-{run_tag}", lambda: _stage_synth(root))
    pre_out = _run_stage(
        f"pretrain-{run_tag}",
        lambda: _stage_pretrain(root, synth_out["records_path"]),
    )
    train_d_out = _run_stage(
        f"train-d-{run_tag}",
        lambda: _stage_train_d(root, synth_out["train_manifest"]),
    )
    if with_n:
        # Stage runs for its side effects; its output is consumed downstream
        # via the run directory, not via this binding (lint: F841).
        _train_n_out = _run_stage(
            f"train-n-{run_tag}",
            lambda: _stage_train_n(
                root, synth_out["train_manifest"], pre_out["pretrain_corpus"]
            ),
        )
    else:
        _train_n_out = None
    select_out = _run_stage(
        f"select-{run_tag}",
        lambda: _stage_select(root, synth_out["train_manifest"], with_n),
    )
    export_out = _run_stage(
        f"export-{run_tag}",
        lambda: _stage_export(root, train_d_out["model_dir"], select_out["winner"]),
    )
    eval_out = _run_stage(
        f"eval-{run_tag}",
        lambda: _stage_eval(root, export_out["artifact"], synth_out["holdout_dir"]),
    )
    return {"root": root, "with_n": with_n, "select": select_out, "eval": eval_out}


# ── determinism comparison ────────────────────────────────────────────────────


def _compare_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Exact-equality gate over the two full runs.

    Compared byte-for-byte: corpus/holdout/pretrain data, model state
    (booster.txt, meta.json, n-head weights as ARRAYS — npz is a zip and
    zip timestamps are wall-clock noise), the selection report.

    Compared exactly (floats): the five eval metrics — same machine, same
    libs, deterministic fits (lightgbm deterministic=True n_jobs=1; torch
    manual_seed + single-thread + use_deterministic_algorithms).

    Deliberately NOT compared (documented exclusions, both are wall-clock
    provenance BY CONTRACT, inference-v1.md §4): model.onnx sha256 and
    trained_at — the artifact embeds an ISO-8601 timestamp, so its bytes
    differ across runs; artifact SIZE and the manifest minus
    sha256/trained_at ARE compared. Weight determinism itself is pinned
    by the booster/npz equality above (the ONNX graph is a pure function
    of the booster + Platt pair).
    """
    problems: list[str] = []
    a, b = Path(first["root"]), Path(second["root"])
    files = list(_BYTE_COMPARE_FILES)
    if first["with_n"]:
        files += ["models/n-head/meta.json"]
    for rel in files:
        if (a / rel).read_bytes() != (b / rel).read_bytes():
            problems.append(f"file mismatch: {rel}")
    if first["with_n"]:
        import numpy as np

        left = np.load(a / "models" / "n-head" / "weights.npz")
        right = np.load(b / "models" / "n-head" / "weights.npz")
        if sorted(left.files) != sorted(right.files):
            problems.append("n-head weights: different array sets")
        else:
            for name in left.files:
                if not np.array_equal(left[name], right[name]):
                    problems.append(f"n-head weights differ: {name}")

    for key in _EVAL_METRIC_KEYS:
        if first["eval"][key] != second["eval"][key]:
            problems.append(
                f"eval metric differs: {key} {first['eval'][key]!r} vs {second['eval'][key]!r}"
            )

    manifest_keys = (
        "name",
        "version",
        "embedder_pin",
        "corpus_fingerprint",
        "candidate",
        "features",
        "feature_set_sha256",
        "size_bytes",
    )
    left = json.loads((a / "model.manifest.json").read_text(encoding="utf-8"))
    right = json.loads((b / "model.manifest.json").read_text(encoding="utf-8"))
    for key in manifest_keys:
        if left[key] != right[key]:
            problems.append(f"artifact manifest differs: {key}")

    if problems:
        raise SmokeFailure("; ".join(problems))
    return {
        "files_compared": len(files) + (1 if first["with_n"] else 0),
        "metrics_compared": len(_EVAL_METRIC_KEYS),
        "excluded": [
            "model.onnx sha256 + trained_at (wall-clock provenance by contract, inference-v1 §4)",
            "n-head weights.npz bytes (zip mtime) — arrays compared instead",
        ],
    }


# ── provenance + entry point ─────────────────────────────────────────────────


def _environment() -> dict[str, Any]:
    import importlib.util

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "xpu_available": None,
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        env["torch"] = torch.__version__
        try:
            env["xpu_available"] = bool(torch.xpu.is_available())
        except Exception:  # noqa: BLE001 — provenance must never kill the smoke
            env["xpu_available"] = False
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="cortex CPU smoke — full pipeline contour on synthetic data"
    )
    parser.add_argument(
        "--with-n",
        action="store_true",
        help="also run the N ladder (needs the train extra: torch)",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="scratch directory (default: <repo>/.local/smoke)",
    )
    args = parser.parse_args(argv)

    workdir = Path(args.workdir) if args.workdir else REPO_ROOT / ".local" / "smoke"
    workdir.mkdir(parents=True, exist_ok=True)
    for run in ("run1", "run2"):
        shutil.rmtree(workdir / run, ignore_errors=True)

    started = time.monotonic()
    try:
        if args.with_n:
            import torch  # noqa: F401 — fail loud and early, not three stages in

        first = _run_contour(workdir / "run1", with_n=args.with_n, run_tag="1")
        second = _run_contour(workdir / "run2", with_n=args.with_n, run_tag="2")
        _run_stage("determinism", lambda: _compare_runs(first, second))
    except Exception as exc:  # noqa: BLE001 — the smoke's own error boundary
        elapsed = time.monotonic() - started
        _emit("smoke", "fail", elapsed, with_n=args.with_n, error=str(exc)[:500])
        _write_report(workdir, args.with_n, elapsed, ok=False)
        return 1

    elapsed = time.monotonic() - started
    budget_s = BUDGET_WITH_N_SECONDS if args.with_n else BUDGET_SECONDS
    _emit(
        "smoke",
        "ok",
        elapsed,
        with_n=args.with_n,
        runs=2,
        determinism="match",
        budget="ok" if elapsed <= budget_s else "over",
        budget_s=budget_s,
    )
    _write_report(workdir, args.with_n, elapsed, ok=True)
    return 0


def _write_report(workdir: Path, with_n: bool, elapsed: float, *, ok: bool) -> None:
    report = {
        "status": "ok" if ok else "fail",
        "with_n": with_n,
        "elapsed_s": round(elapsed, 3),
        "budget_s": BUDGET_WITH_N_SECONDS if with_n else BUDGET_SECONDS,
        "seeds": {
            "pretrain": SMOKE_SEED,
            "vectors": VECTOR_SEED,
            "pairs": N_PAIRS,
            "records": N_RECORDS,
        },
        "embedder_pin": SMOKE_EMBEDDER_PIN,
        "environment": _environment(),
        "stages": _RECORDS,
    }
    (workdir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
