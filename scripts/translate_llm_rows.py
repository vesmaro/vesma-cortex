#!/usr/bin/env python3
"""Round-4 embedder corpus — translate the UNIQUE LLM mono rows.

Post-collapse repair (TL approved 2026-10-08): the freeform LLM pool
collapsed on uniqueness (23%); the surviving unique mono rows (LLM)
get teacher translations to their other language — these become
RU<->EN content twins of genuinely unique texts (their counterparts do
NOT exist elsewhere in the corpus), repairing the translated share.

Same contract as translate-real: LOCAL teacher, digit-preservation
validator, anti-copy gate, lang gate, checkpointed shards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT / "src"), "/var/home/abyss/LABs/Projects/Project-Vesma/vesma",
          str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gen_embed_r4_texts import (  # noqa: E402
    BATCH,
    GEN_DIR,
    RUN_LOG,
    _clean_translation,
    _is_copy,
    _translate_prompt,
    append_jsonl,
    detect_lang,
    load_jsonl,
    log_event,
    missing_digit_tokens,
    normalise,
    text_hash,
)


def unique_llm_units() -> list[tuple[str, str, str]]:
    """(key, target_lang, source_text) for every UNIQUE ok LLM mono row."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    mono_paths = sorted(GEN_DIR.glob("synth-mono.shard*.jsonl"))
    for path in mono_paths:
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            h = text_hash(row["text"])
            if h in seen:
                continue
            seen.add(h)
            target = "ru" if row["lang"] == "en" else "en"
            out.append((h, target, row["text"]))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args(argv)

    out_path = GEN_DIR / f"llm-translations.shard{args.shard}.jsonl"
    done_jobs = {row.get("job") for row in load_jsonl(out_path)}
    units = unique_llm_units()
    jobs = []
    for i in range(0, len(units), BATCH):
        jid = f"ll{i // BATCH:05d}"
        if jid in done_jobs or (i // BATCH) % args.num_shards != args.shard:
            continue
        jobs.append((jid, units[i : i + BATCH]))
    log_event("translate-llm-start", total_unique=len(units), jobs_todo=len(jobs))
    if not jobs:
        return 0

    from gen_embed_r4_texts import Generator

    gen = Generator(args.threads, ctx=3072)
    t0 = time.time()
    kept = bad = 0
    for n, (jid, chunk) in enumerate(jobs):
        prompts = [_translate_prompt(tgt, unit) for _k, tgt, unit in chunk]
        raws = gen.chat(prompts, max_new_tokens=560, seed=900000 + n)
        rows = []
        for (key, tgt, unit), raw in zip(chunk, raws):
            text = normalise(_clean_translation(raw))
            ok = (
                bool(text)
                and not missing_digit_tokens(unit, text)
                and detect_lang(text) == tgt
                and not _is_copy(unit, text)
                and 40 <= len(text) <= 4000
            )
            if ok:
                kept += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "ok", "text": text})
            else:
                bad += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "fail", "text": text})
        append_jsonl(out_path, rows)
        if n % 10 == 0:
            rate = (n + 1) / max(1.0, time.time() - t0)
            print(f"translate-llm job {n + 1}/{len(jobs)} rate={rate:.2f} eta={(len(jobs)-n-1)/max(0.05, rate)/60:.0f}min", flush=True)
    log_event("translate-llm-done", shard=args.shard, kept=kept, bad=bad, wall_sec=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
