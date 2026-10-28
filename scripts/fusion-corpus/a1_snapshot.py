#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 1: frozen store snapshot (D0).

Pattern of the 2026-10-22 census / 2026-10-28 watchlist probe, verbatim:
the live store is opened read-only (mode=ro URI + query_only) and copied
CONSISTENTLY with the sqlite3 backup API into the worktree's gitignored
data dir. The live service is never touched. D0-fp = sha256 pair of the
two snapshot files + row counters, recorded in snapshot.json.

Output: data/a-retrial/snapshot/{vesma.db,vectors.db,snapshot.json}
Privacy: raw store content stays inside the gitignored data dir; only
counters / fingerprints ever leave it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
OUT_DIR = WORKTREE / "data" / "a-retrial" / "snapshot"
STORE = Path("/var/home/abyss/.vesma/data")
DBS = ("vesma.db", "vectors.db")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result: dict[str, object] = {"started_utc": started, "dbs": {}}
    for name in DBS:
        src_path = STORE / name
        dst_path = OUT_DIR / name
        src = sqlite3.connect(f"file:{src_path.as_posix()}?mode=ro", uri=True, timeout=10.0)
        src.execute("PRAGMA query_only=ON")
        dst = sqlite3.connect(dst_path.as_posix())
        src.backup(dst)
        dst.close()
        src.close()
        size = dst_path.stat().st_size
        digest = sha256_file(dst_path)
        result["dbs"][name] = {"bytes": size, "sha256": digest}
        print(f"{name}: {size} bytes sha256={digest}")

    # row counters straight from the frozen snapshot
    conn = sqlite3.connect(f"file:{(OUT_DIR / 'vesma.db').as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    counters = {
        "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "memories_published": conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='published'"
        ).fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0],
        "edge_kinds": {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT kind, COUNT(*) FROM memory_edges GROUP BY kind ORDER BY kind"
            )
        },
    }
    conn.close()
    conn = sqlite3.connect(f"file:{(OUT_DIR / 'vectors.db').as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    counters["vectors"] = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    result["counters"] = counters
    result["d0_utc"] = started
    (OUT_DIR / "snapshot.json").write_text(json.dumps(result, indent=1) + "\n")
    print("counters:", json.dumps(counters))
    print(f"D0 = {started}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
