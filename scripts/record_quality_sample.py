#!/usr/bin/env python3
"""Record-quality phase-1 sample probe — read-only live-store census.

Pattern (data-contract §6 hygiene, mirror of the store-export probe
family): open the store with a ``mode=ro`` URI + ``query_only=ON``, take
an sqlite3 BACKUP into /tmp, close the live connection, then score the
COPY — the live server is never held or written.

Output discipline: counters, distributions and id prefixes ONLY. Titles,
content, metadata and even violation ``detail`` strings never print —
``detail`` is dropped at the aggregation boundary (it may quote record
text; the score needs only the code counts).

Usage:
    python3 scripts/record_quality_sample.py                  # newest 200
    python3 scripts/record_quality_sample.py --limit 500
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from cortex.quality import (  # noqa: E402
    aggregate,
    canon_scope,
    load_canon_validator,
    score_record,
)

DEFAULT_STORE = Path("/var/home/abyss/.vesma/data/vesma.db")
DEFAULT_ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE, help="path to vesma.db"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="sample size (newest rows)"
    )
    parser.add_argument("--engine-src", type=Path, default=DEFAULT_ENGINE_SRC)
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="backup path in /tmp (default: auto, timestamped)",
    )
    return parser.parse_args()


def backup_to_tmp(store: Path, backup: Path) -> None:
    """Read-only open of the live store, sqlite3 backup into /tmp, close."""
    if not store.is_file():
        raise SystemExit(f"store database not found: {store}")
    source = sqlite3.connect(f"file:{store.as_posix()}?mode=ro", uri=True, timeout=5.0)
    source.execute("PRAGMA query_only=ON")
    destination = sqlite3.connect(backup.as_posix())
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def parse_metadata(raw: object) -> tuple[object, str]:
    """Return (parsed-or-None, reason) — reason feeds a counter, not logs
    with content."""
    if raw is None or raw == "":
        return {}, "metadata_empty"
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return None, "metadata_unparsable"
    return parsed, "ok"


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup = args.backup or Path(f"/tmp/record-quality-phase1-sample-{stamp}.db")
    backup_to_tmp(args.store, backup)
    print(f"store:       {args.store}")
    print(f"backup copy: {backup} (read-only reuse, /tmp is scratch)")

    validator = load_canon_validator(args.engine_src)
    print(
        "validator:   vesma.canon_validate "
        f"(engine {validator.provenance['engine_version']}, "
        f"scorer {validator.provenance['scorer_revision']})"
    )

    conn = sqlite3.connect(f"file:{backup.as_posix()}?mode=ro", uri=True)
    total = int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
    rows = conn.execute(
        "SELECT id, title, content, metadata, memory_type, status "
        "FROM memories ORDER BY created_at DESC, id LIMIT ?",
        (args.limit,),
    ).fetchall()
    conn.close()

    results = []
    scope_reasons: Counter[str] = Counter()
    memory_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    violating_ids: list[tuple[str, float, str]] = []
    for record_id, title, content, metadata_raw, memory_type, status in rows:
        memory_types[str(memory_type)] += 1
        statuses[str(status)] += 1
        metadata, reason = parse_metadata(metadata_raw)
        if canon_scope(metadata):
            scope_reasons["canon_envelope"] += 1
        else:
            scope_reasons["no_canon_envelope" if reason == "ok" else reason] += 1
        quality = score_record(str(content), title, metadata, validator=validator)
        results.append(quality)
        if quality.in_scope and quality.violations:
            codes = ",".join(sorted(quality.components))
            violating_ids.append((str(record_id)[:8], quality.score, codes))  # type: ignore[arg-type]

    stats = aggregate(results)
    print(f"\nstore rows total:      {total}")
    print(f"sample (newest first): {stats.total}")
    print(f"in canon scope:        {stats.in_scope}")
    print(f"out of scope:          {stats.out_of_scope}  {dict(scope_reasons)}")
    in_scope_share = stats.in_scope / stats.total if stats.total else 0.0
    print(f"in-scope share:        {in_scope_share:.3f}")
    if stats.mean_penalty_hundredths is not None:
        mean_score = 1 - stats.mean_penalty_hundredths / 100
        print(f"mean score (in-scope): {mean_score:.3f}")
    print("score histogram (exact):")
    for value, count in sorted(stats.score_histogram.items(), reverse=True):
        print(f"  {value}: {count}")
    print("violation classes:")
    for code, count in sorted(
        stats.code_counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {code}: {count}")
    print(f"sample memory_type:    {dict(memory_types)}")
    print(f"sample status:         {dict(statuses)}")
    if violating_ids:
        print("violating id prefixes (for spot-check; no content):")
        for prefix, score, codes in violating_ids[:50]:
            print(f"  {prefix}  score={score:.2f}  {codes}")


if __name__ == "__main__":
    main()
