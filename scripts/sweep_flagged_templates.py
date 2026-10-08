#!/usr/bin/env python3
"""Full teacher sweep for the cos-flagged template families; drops rows
whose nearest round-3 same-family cos >= 0.97 (advisory guard above the
ratified exact-hash assert; rows dropped are counted in the run-log)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/var/home/abyss/LABs/Projects/Project-Vesma/vesma")

from training.dataset.prepare_dataset import normalise  # noqa: E402
from filter_embed_r4 import Embedder, ROUND3_TRAIN, ROUND3_VAL, load_jsonl  # noqa: E402
from gen_embed_r4_texts import RUN_LOG, log_event  # noqa: E402

FLAGGED = {"science"}  # families whose sample max-cos crossed 0.97


def main() -> int:
    emb = Embedder(threads=8)
    r3: dict[str, list[str]] = {}
    for path in (ROUND3_TRAIN, ROUND3_VAL):
        for row in load_jsonl(path):
            src = str(row.get("source", ""))
            if src.startswith("synthetic-"):
                fam = src.removeprefix("synthetic-ru-").removeprefix("synthetic-en-")
                r3.setdefault(fam, []).append(normalise(row["text"]))
    path = REPO_ROOT / "data" / "embed-r4" / "gen" / "synth-template-mono.jsonl"
    rows = load_jsonl(path)
    drop: set[int] = set()
    report = {}
    for fam in FLAGGED:
        pool = r3.get(fam, [])
        if not pool:
            continue
        old = emb.embed(pool)
        fam_idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        new = emb.embed([normalise(rows[i]["text"]) for i in fam_idx])
        mx = (new @ old.T).max(dim=1).values
        for i, v in zip(fam_idx, mx.tolist()):
            if v >= 0.97:
                drop.add(i)
        report[fam] = {"checked": len(fam_idx), "dropped": sum(1 for v in mx.tolist() if v >= 0.97),
                       "worst": round(max(mx.tolist()), 4)}
    if drop:
        kept = [r for i, r in enumerate(rows) if i not in drop]
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_event("template-flagged-sweep", flagged=sorted(FLAGGED), dropped_rows=len(drop), per_family=report)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
