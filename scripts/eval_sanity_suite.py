#!/usr/bin/env python3
"""Adversarial sanity suite runner for a vesma-cortex bundle (#480 gate).

The mandatory pre-ADOPT instrument: loads a bundle (dir with model.onnx +
manifest.json, or a direct .onnx path) and runs the bundle-contract
preconditions plus the four adversarial checks — self-pair, near-boundary,
unrelated, monotonicity. All green BEFORE any other ADOPT metric may be
quoted (docs/experiments/calibration-b2-edge-neighborhood.md,
roadmap-v2.md §5).

Usage:
    uv run python scripts/eval_sanity_suite.py --bundle <dir-or-model.onnx>
    uv run python scripts/eval_sanity_suite.py --bundle <dir> --json

Exit codes: 0 — all checks green; 1 — bundle cannot be loaded (usage /
load error, details on stderr); 2 — at least one check FAILED (the ADOPT
gate verdict: no adopt on a red suite).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cortex.eval.sanity import SanityLoadError, run_sanity_suite


def _human(report) -> str:
    lines = [
        "adversarial sanity suite — vesma-cortex bundle",
        f"bundle: {report.bundle}",
        f"weights sha256: {report.weights_sha256}",
        "",
    ]
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"[{mark}] {check.name:<20} {check.detail}")
    lines.append("")
    if report.passed:
        lines.append("verdict: PASS — all checks green")
    else:
        names = ", ".join(check.name for check in report.failed)
        lines.append(
            f"verdict: FAIL — {len(report.failed)}/{len(report.checks)} checks failed: {names}"
        )
        lines.append("ADOPT gate: a red suite blocks adoption (roadmap-v2 §5.1)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval-sanity-suite",
        description="mandatory pre-ADOPT adversarial sanity suite for a vesma-cortex bundle (#480)",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="bundle directory (model.onnx + manifest.json) or a direct .onnx path",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="machine-readable JSON report on stdout",
    )
    args = parser.parse_args(argv)

    try:
        report = run_sanity_suite(Path(args.bundle))
    except SanityLoadError as exc:
        print(f"eval-sanity-suite: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = {
            "bundle": report.bundle,
            "weights_sha256": report.weights_sha256,
            "passed": report.passed,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in report.checks
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_human(report))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
