"""cortex CLI — pipeline entry points (argparse stubs in A3a).

Subcommands mirror the pipeline order: export-corpus (A2, prereg hygiene +
fingerprints) → pretrain (A3b, corruption pairs) → train (A3b, D/N) →
select (A3b, frozen CV) → export-artifact (A6, ONNX + metadata_props) →
eval (A5, single-shot runner).

A3a behaviour: every subcommand parses its arguments, then reports
"not implemented in A3a" on stderr and exits with code 2 (usage-class) —
a stub must not signal success. Exit code 2 is the stub contract until
the implementing slice lands.
"""

from __future__ import annotations

import argparse
import sys
from typing import Final

__all__ = ["main", "build_parser"]

#: Exit code for a not-yet-implemented subcommand (stub contract).
NOT_IMPLEMENTED_EXIT: Final[int] = 2

_STUB_MESSAGE = "not implemented in A3a — scheduled for A3b+ (see docs/specs/inference-v1.md)"


def _stub(command: str) -> int:
    print(f"cortex {command}: {_STUB_MESSAGE}", file=sys.stderr)
    return NOT_IMPLEMENTED_EXIT


def build_parser() -> argparse.ArgumentParser:
    """The frozen CLI surface: subcommand names are contract (runbooks and
    the charter's CPU-smoke gate reference them); options may grow later."""
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="mnema-cortex development pipeline (train epoch library)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-corpus", help="export eval/pretrain corpus from the store (prereg hygiene, fingerprints)")
    p.add_argument("--store-uri", required=True, help="read-only store URI: file:...?mode=ro")
    p.add_argument("--out", required=True, help="output directory under data/")

    p = sub.add_parser("pretrain", help="generate corruption pretrain pairs (candidate N)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="train D and N candidates on labeled train pairs")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--candidate", choices=["d", "n", "both"], default="both")

    p = sub.add_parser("select", help="frozen CV protocol: stratified 5-fold × 20 seeds")
    p.add_argument("--train-manifest", required=True)

    p = sub.add_parser("export-artifact", help="export the winning candidate as mnema-cortex-v1 ONNX")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("eval", help="single-shot holdout evaluation (prereg v2, append-only run log)")
    p.add_argument("--artifact", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--run-log", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _stub(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
