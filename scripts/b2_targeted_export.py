#!/usr/bin/env python3
"""B2 targeted export — the edge-neighborhood pool for dataset v3.

Wave B2 of the v2 duty program: the A2 pretrain pool (794 records) holds
only 10 graph nodes and ZERO silver pairs, so dataset v3 gets no graph
signal from it. This exporter pulls EXACTLY the edge neighborhood out of
the live engine store:

1. seed ids — every node touched by ``memory_edges`` (generated here
   from the store read-only, or taken from ``--ids-file``);
2. 1-hop expansion over ``memory_edges`` (seeds ∪ edge endpoints);
3. the SAME frozen hygiene chain as the A2 pool export
   (``cortex.data.store_export``: engine secrets/danger detectors +
   ``no-federate`` + quarantine, per-reason counters, read-only URIs);
4. outputs under ``<out>/pretrain/<corpus-id>/`` (gitignored data/):
   ``records.jsonl`` (pool format, store vectors included),
   ``near_dup_candidates.jsonl`` (cosine band [0.85, 0.97)),
   ``silver_edges.jsonl`` (edges fully inside the pool, kind +
   provenance preserved), ``coverage.json`` (counters + fingerprints),
   and the §5 manifest files.

The store serves a live MCP server: both databases open strictly
read-only (``file:…?mode=ro`` + ``PRAGMA query_only``), open → read →
close, never written, never locked long.

Progress goes to stderr, a JSON summary to stdout. Thin wrapper — all
store/hygiene logic lives in ``cortex.data.store_export`` (tested there).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cortex.data.store_export import (
    DEFAULT_MAX_COSINE,
    DEFAULT_MIN_COSINE,
    StoreOpenError,
    _ro_connection,
    edge_node_ids,
    export_targeted_corpus,
    make_scanner,
    resolve_store_databases,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="b2_targeted_export",
        description="targeted edge-neighborhood export from the engine store (READ-ONLY)",
    )
    parser.add_argument(
        "--store-path",
        help="store data DIRECTORY holding mnemos.db + vectors.db",
    )
    parser.add_argument(
        "--store-uri",
        help="read-only URI of mnemos.db (file:...?mode=ro); vectors.db is its sibling",
    )
    parser.add_argument(
        "--out", required=True, help="output root under data/ (gitignored)"
    )
    parser.add_argument(
        "--corpus-id",
        default=None,
        help="corpus id (default: b2-edges-<UTC date>)",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="seed ids file, one store id per line (default: generated from memory_edges "
        "endpoints, written next to the corpus under <out>/ids/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="max records after hygiene (deterministic created_at/id trim; default 1200)",
    )
    parser.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument("--max-cosine", type=float, default=DEFAULT_MAX_COSINE)
    parser.add_argument(
        "--engine-src",
        default=None,
        help="engine source tree for the hygiene detectors (default: $CORTEX_ENGINE_SRC); "
        "absent → local fallback scanner, disclosed in provenance",
    )
    return parser.parse_args(argv)


def _progress(message: str) -> None:
    print(f"b2_targeted_export: {message}", file=sys.stderr)


def _generate_ids_file(
    store_path: str | None, store_uri: str | None, out: Path
) -> Path:
    """Seed ids = distinct memory_edges endpoints, written under <out>/ids/."""
    mnemos_path, _ = resolve_store_databases(store_path, store_uri)
    conn = _ro_connection(mnemos_path)
    try:
        ids = edge_node_ids(conn)
    finally:
        conn.close()
    path = out / "ids" / f"edge-node-ids-{time.strftime('%Y%m%d')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    _progress(f"seed ids generated from memory_edges: {len(ids)} → {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()

    if (args.store_path is None) == (args.store_uri is None):
        print(
            "b2_targeted_export: exactly one of --store-path / --store-uri is required",
            file=sys.stderr,
        )
        return 2
    if args.min_cosine >= args.max_cosine:
        print(
            f"b2_targeted_export: --min-cosine must be < --max-cosine, "
            f"got [{args.min_cosine}, {args.max_cosine})",
            file=sys.stderr,
        )
        return 2
    if args.limit <= 0:
        print(
            f"b2_targeted_export: --limit must be positive, got {args.limit}",
            file=sys.stderr,
        )
        return 2

    corpus_id = args.corpus_id or f"b2-edges-{time.strftime('%Y%m%d')}"
    out_dir = Path(args.out).expanduser().resolve()

    ids_file = (
        Path(args.ids_file).expanduser().resolve()
        if args.ids_file
        else _generate_ids_file(args.store_path, args.store_uri, out_dir)
    )
    if not ids_file.is_file():
        print(f"b2_targeted_export: ids file not found: {ids_file}", file=sys.stderr)
        return 2

    scanner = make_scanner(args.engine_src)
    if scanner.provenance != "engine":
        print(
            f"b2_targeted_export: WARNING scanner provenance is {scanner.provenance!r} — "
            "the prereg hygiene expects the ENGINE detectors (--engine-src / "
            "$CORTEX_ENGINE_SRC); any report built from this export must disclose the fallback",
            file=sys.stderr,
        )

    try:
        result = export_targeted_corpus(
            ids_file=ids_file,
            store_path=args.store_path,
            store_uri=args.store_uri,
            out_dir=out_dir,
            corpus_id=corpus_id,
            scanner=scanner,
            limit_pool=args.limit,
            min_cosine=args.min_cosine,
            max_cosine=args.max_cosine,
            progress=_progress,
        )
    except (StoreOpenError, ValueError) as exc:
        print(f"b2_targeted_export: {exc}", file=sys.stderr)
        return 2

    summary = result.summary()
    summary["ids_file"] = str(ids_file)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
