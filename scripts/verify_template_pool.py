#!/usr/bin/env python3
"""Non-verbatim verification for the template pool (TL condition, 2026-10-08).

Per family-lang: sample 30 template rows, embed vs up to 200 round-3
same-family rows, report max-cos (flag at 0.97). Exact-hash disjointness
was already verified at generation (overlap 0)."""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/var/home/abyss/LABs/Projects/Project-Vesma/vesma")

from training.dataset.prepare_dataset import normalise  # noqa: E402
from filter_embed_r4 import Embedder, ROUND3_TRAIN, ROUND3_VAL, load_jsonl  # noqa: E402
from gen_embed_r4_texts import log_event  # noqa: E402


def main() -> int:
    emb = Embedder(threads=8)
    r3: dict[str, list[str]] = {}
    for path in (ROUND3_TRAIN, ROUND3_VAL):
        for row in load_jsonl(path):
            src = str(row.get("source", ""))
            if src.startswith("synthetic-"):
                fam = src.removeprefix("synthetic-ru-").removeprefix("synthetic-en-")
                r3.setdefault(fam, []).append(normalise(row["text"]))
    rows = load_jsonl(
        REPO_ROOT / "data" / "embed-r4" / "gen" / "synth-template-mono.jsonl"
    )
    by: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        by.setdefault((r["family"], r["lang"]), []).append(normalise(r["text"]))
    rng = random.Random(43)
    report, worst = {}, 0.0
    for (fam, lang), texts in sorted(by.items()):
        new = rng.sample(texts, min(30, len(texts)))
        pool = r3.get(fam, [])
        if not pool:
            continue
        old = rng.sample(pool, min(200, len(pool)))
        a = emb.embed(new)
        b = emb.embed(old)
        mx = (a @ b.T).max().item()
        worst = max(worst, mx)
        report[f"{fam}-{lang}"] = round(mx, 4)
        print(f"{fam}-{lang}: max_cos={mx:.4f}", flush=True)
    log_event(
        "template-nonverbatim-sample",
        worst_max_cos=round(worst, 4),
        flag_at=0.97,
        per_family=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
