#!/usr/bin/env python3
"""Convert B2 rev 1.n corpus rows into the sidecar dataset-v2 input format.

``scripts/b2_graph_sidecar.py`` reads a dataset-v2 catalog (pairs-pos.jsonl
with original/candidate sides). This converter re-shapes v44 / v42-276 rows
into that surface — pair_id, stratum, label, original (= record),
candidate, source_id (= id_a when the row carries it, else absent) —
so the STORE-side join logic stays the one frozen script, byte-for-byte,
with the same read-only discipline. Nothing here touches the store.

Usage:
    python scripts/b2_rev1n_sidecar_input.py \
        --rows data/b2-graph-corpus/v44/train.jsonl --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="b2_rev1n_sidecar_input",
        description="corpus rows → sidecar dataset-v2 catalog (no store access)",
    )
    parser.add_argument(
        "--rows",
        required=True,
        nargs="+",
        help="corpus jsonl files (all rows land in the catalog)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output catalog dir (pairs-pos.jsonl + empty pairs-neg.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with (out_dir / "pairs-pos.jsonl").open("w", encoding="utf-8") as handle:
        for path in args.rows:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(
                    row.get("pair_id"), str
                ):
                    raise SystemExit(f"{path}: row without a string pair_id")
                entry = {
                    "pair_id": row["pair_id"],
                    "stratum": row.get("stratum"),
                    "label": row.get("label"),
                    "original": row["record"],
                    "candidate": row["candidate"],
                }
                if row.get("id_a") and row.get("id_b"):
                    entry["source_id"] = row["id_a"]
                    entry["source_id_variant"] = row["id_b"]
                handle.write(
                    json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
                )
                rows_written += 1
    (out_dir / "pairs-neg.jsonl").write_text("", encoding="utf-8")
    print(
        json.dumps({"catalog": str(out_dir), "rows": rows_written}),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
