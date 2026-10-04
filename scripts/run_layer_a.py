#!/usr/bin/env python3
"""Layer A eval-set runner — the frozen-set gate over a bundle (wave LA-1).

Loads a frozen eval set (re-verifying ``eval_set_sha256`` against its
``.meta.json`` — a mismatch voids the run), scores every probe through
``cortex.features.pair.features`` + the artifact graph, and prints the
report: BA, per-class breakdown, gate evaluations, verdict by role
(merge → invariants; release → invariants + corridors).

Usage (the proposed one-line CI step for the LA-3 eval-gate job):
    uv run python scripts/run_layer_a.py \
        --bundle models/vesma-cortex-v1 \
        --eval-set datasets/evalsets/merge-v1.jsonl
    uv run python scripts/run_layer_a.py --bundle <dir> --eval-set <jsonl> --json

Exit codes (mirror scripts/eval_sanity_suite.py):
    0 — verdict PASS; 1 — load error (bundle or set); 2 — gate FAILED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cortex.eval.sanity import SanityLoadError
from cortex.evalsets import EvalSetError, load_eval_set, report_to_dict, run_layer_a


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-layer-a",
        description="Layer A frozen eval-set runner (merge/release gate instrument)",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="bundle directory (model.onnx + manifest.json) or a direct .onnx path",
    )
    parser.add_argument(
        "--eval-set",
        required=True,
        help="frozen set JSONL (datasets/evalsets/<set-id>.jsonl); the sibling "
        "<set-id>.meta.json carries the pinned eval_set_sha256",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="override the set meta path (default: sibling <set-id>.meta.json)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="machine-readable JSON report on stdout",
    )
    args = parser.parse_args(argv)

    try:
        eval_set = load_eval_set(
            Path(args.eval_set),
            meta_path=Path(args.meta) if args.meta else None,
        )
    except EvalSetError as exc:
        print(f"run-layer-a: {exc}", file=sys.stderr)
        return 1

    try:
        report = run_layer_a(Path(args.bundle), eval_set)
    except SanityLoadError as exc:
        print(f"run-layer-a: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
    else:
        print(_human(report))
    return 0 if report.passed else 2


def _human(report) -> str:
    lines = [
        "Layer A eval-set run — vesma-cortex bundle",
        f"set: {report.set_id} (role {report.role}) — {report.n_pairs} pairs",
        f"eval_set_sha256: {report.eval_set_sha256}",
        f"weights_sha256: {report.weights_sha256}",
        f"balanced accuracy: {report.balanced_accuracy:.4f} (cut {report.threshold})",
        "",
        "per class:",
    ]
    for breakdown in report.per_class:
        sens = "—" if breakdown.sensitivity is None else f"{breakdown.sensitivity:.4f}"
        spec = "—" if breakdown.specificity is None else f"{breakdown.specificity:.4f}"
        lines.append(
            f"  {breakdown.probe_class:<22} n={breakdown.n:<4} "
            f"tp={breakdown.tp} fp={breakdown.fp} tn={breakdown.tn} "
            f"fn={breakdown.fn} sens={sens} spec={spec}"
        )
    lines.append("")
    for check in report.bundle_contract:
        lines.append(
            f"[{'PASS' if check.passed else 'FAIL'}] {check.name:<18} {check.detail}"
        )
    for gate in report.gate_evaluations:
        lines.append(
            f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name:<18} "
            f"({gate.gate_kind}) {gate.detail}"
        )
    lines.append("")
    if report.passed:
        lines.append(f"verdict: PASS — {report.role} gate green")
    else:
        failed = [check.name for check in report.bundle_contract if not check.passed]
        failed += [gate.name for gate in report.failed_gates]
        lines.append(f"verdict: FAIL — failing: {', '.join(failed)}")
        lines.append(
            "gate: merge blocks on invariants, release adds corridors "
            "(eval-methodology §4)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
