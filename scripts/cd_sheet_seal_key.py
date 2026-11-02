#!/usr/bin/env python3
"""Seal the gold-sheet key (owner opt-out fallback OQ-1, 02.11.2026).

The key holds the by-construction labels of the 48 scorable blocks.
It must NOT be opened by anyone before the committee synthesis: the
committee scores the blind render, THEN the key is unsealed for the
g_gold count (>= 45/50 + 0 systematic pattern on confusables).

The sampling logic is byte-identical to cmd_sheet (same rng stream) so
block Nxx maps 1:1 onto the blind render. Lives under data/ (gitignored);
the sha256 fingerprint is logged to the run-log — the key travels by
fingerprint, never by content.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402
from build_conflict_detector_corpus import (  # noqa: E402
    ORDER, SEALS_DIR, SEED_SHEET, SHEET_DIR, log,
)

TRAIN = REPO / "data" / "conflict-detector" / "corpus-v1" / "train.jsonl"
KEY = SHEET_DIR / "verdicts" / "KEY-cd-v1.sealed.json"
RENDER_FP_SRC = SHEET_DIR / "gold-sheet-v1.json"


def main() -> int:
    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l]
    a_rows = [r for r in rows if r["orientation"] == "a>b"]
    rng = random.Random(SEED_SHEET)
    by_stratum: dict[str, list] = {}
    for r in a_rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    picks = []
    for s in ORDER:
        cands = sorted(by_stratum[s], key=lambda r: r["pair_id"])
        picks += rng.sample(cands, min(3, len(cands)))
    cd1 = sorted(by_stratum["CD-1"], key=lambda r: r["pair_id"])
    nc2 = sorted(by_stratum["NC-2"], key=lambda r: r["pair_id"])
    nc3 = sorted(by_stratum["NC-3"], key=lambda r: r["pair_id"])
    picks += nc2[:4] + cd1[:4] + nc3[:4] + cd1[4:8]
    cd6 = sorted(by_stratum["CD-6"], key=lambda r: r["pair_id"])
    nc5 = sorted(by_stratum["NC-5"], key=lambda r: r["pair_id"])
    picks += cd6[:1] + nc5[:1]
    reserve = rng.sample([r for r in a_rows if r not in picks], 5)
    picks += reserve
    picks = picks[:48]
    order = list(range(len(picks)))
    rng.shuffle(order)

    blocks = []
    for i, idx in enumerate(order):
        r = picks[idx]
        blocks.append({
            "block": f"N{i+1:02d}",
            "pair_id": r["pair_id"],
            "stratum": r["stratum"],
            "label": r["label"],
        })
    labels = set(b["label"] for b in blocks)
    assert labels == {"contradicts", "not-contradicts"}, labels

    render_fp = lib.sha(RENDER_FP_SRC.read_text(encoding="utf-8")) if RENDER_FP_SRC.exists() else ""
    payload = {
        "kind": "gold-sheet-key",
        "sheet": "CD-v1",
        "ts": datetime.now(timezone.utc).isoformat(),
        "owner_opt_out": "02.11.2026 — fallback OQ-1: gold by committee (ADDENDUM-1), g_gold >= 45/50 + 0 systematic pattern",
        "rule": "SEALED — do not open before the committee synthesis",
        "render_fp_sha256": render_fp,
        "n": len(blocks),
        "blocks": blocks,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    fp = lib.sha(body)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(body + "\n", encoding="utf-8")
    log("key_sealed", sheet="CD-v1", n=len(blocks), fp=fp,
        render_fp=render_fp[:16], path=str(KEY.relative_to(REPO)))
    print(f"KEY SEALED: {len(blocks)} blocks, fp {fp[:16]}… -> {KEY}")
    print(f"render fp: {render_fp[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
