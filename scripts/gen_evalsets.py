#!/usr/bin/env python3
"""Regenerate the frozen Layer A eval sets (wave LA-1 procedure).

Deterministic: same code + same seed → byte-identical JSONL and the same
``eval_set_sha256`` (pinned by tests/test_evalsets.py). Regeneration is a
DELIBERATE act: any content change (including filling the LA-2 LLM slots)
produces a new set version with a new fingerprint and a manifest
addendum — never an in-place edit of a frozen set.

Usage:
    uv run python scripts/gen_evalsets.py                 # write v1 sets
    uv run python scripts/gen_evalsets.py --check         # verify only

Files per set (datasets/evalsets/):
    <set-id>.jsonl         — the probes (canonical JSON rows)
    <set-id>.manifest.txt  — data-contract §5 manifest (pair_id sha256)
    <set-id>.meta.json     — set_id/role/per-class counts/pinned sha
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from cortex.evalsets import (
    DEFAULT_SEED,
    MERGE_V1,
    RELEASE_V1,
    build_eval_set,
    eval_set_jsonl,
    eval_set_manifest_bytes,
)

#: Wave date (constant — regeneration must not churn the meta files).
CREATED: Final = "2026-10-04"

RECIPES: Final = (MERGE_V1, RELEASE_V1)


def meta_payload(set_id: str, role: str, sha256: str, per_class: dict, n: int) -> dict:
    return {
        "schema_version": 1,
        "set_id": set_id,
        "role": role,
        "eval_set_sha256": sha256,
        "n_pairs": n,
        "per_class": per_class,
        "seed": DEFAULT_SEED,
        "created": CREATED,
        "generator": "scripts/gen_evalsets.py (cortex.evalsets.build_eval_set)",
        "fingerprint_scheme": (
            "data-contract §5: pair_sha256 over canonical "
            "{record, candidate, similarity, label}; manifest 'pair_id <sha>' "
            "sorted by pair_id; BLAKE2b-256 over manifest bytes"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen-evalsets",
        description="regenerate the frozen Layer A eval sets (deterministic)",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/evalsets",
        help="output directory (default datasets/evalsets)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing files instead of writing (CI-mode byte check)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    for recipe in RECIPES:
        eval_set = build_eval_set(recipe, seed=DEFAULT_SEED)
        jsonl_text = eval_set_jsonl(eval_set.probes)
        manifest = eval_set_manifest_bytes(eval_set.probes)
        meta_text = (
            json.dumps(
                meta_payload(
                    eval_set.set_id,
                    eval_set.role,
                    eval_set.eval_set_sha256,
                    eval_set.per_class,
                    len(eval_set.probes),
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        jsonl_path = out_dir / f"{eval_set.set_id}.jsonl"
        manifest_path = out_dir / f"{eval_set.set_id}.manifest.txt"
        meta_path = out_dir / f"{eval_set.set_id}.meta.json"

        if args.check:
            for path, expected in (
                (jsonl_path, jsonl_text),
                (manifest_path, manifest.decode("utf-8")),
                (meta_path, meta_text),
            ):
                if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                    print(
                        f"gen-evalsets: --check FAILED for {path} "
                        "(missing or byte-divergent from the deterministic rebuild)",
                        file=sys.stderr,
                    )
                    return 1
            print(
                f"{eval_set.set_id}: check OK — {len(eval_set.probes)} pairs, "
                f"sha256={eval_set.eval_set_sha256}"
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(jsonl_text, encoding="utf-8")
        manifest_path.write_bytes(manifest)
        meta_path.write_text(meta_text, encoding="utf-8")
        print(
            f"{eval_set.set_id}: wrote {len(eval_set.probes)} pairs to "
            f"{jsonl_path} — eval_set_sha256={eval_set.eval_set_sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
