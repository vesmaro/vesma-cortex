#!/usr/bin/env python3
"""Round-7 corpus — re-translate the llm class with the 1.7B teacher.

Prereg: docs/specs/embed-round7-prereg-llmquality.md §2 (ACTIVATED,
owner-ratified 2026-10-10). The single round-7 data variable: the 1617
`translated-llm-r4` pairs are RE-TRANSLATED by Qwen3-1.7B (official
GGUF, Q8_0, llama.cpp — same runtime and prompt contract as round-4's
Qwen3-0.6B). Generations REPLACE the 0.6B dst sides; they are never
mixed (no-generation-mixing rule).

Pair identity is preserved: units are re-derived from the SAME
synth-mono manifests via the round-4 machinery (unique_llm_units), and
the unit key set is asserted equal to the sealed round-4 ok manifest
(1617 keys, target langs included). The 68 gate-stratum keys are part
of the 1617 — re-translated for gate comparability, but the round-5
exclusion manifest still keeps them OUT of training.

Per-pair gates (round-4/5 contract, applied at generation): digit
preservation (Addendum-7, lost runs only), target-lang gate, anti-copy
(hash + jaccard >= 0.9), length 40..4000 chars and <= 256 approx
tokens. Teacher-cos >= 0.55 and the n-gram 8-13 zero-tolerance vs the
gate set + burned pairs run in the filter phase (filter_llm_r7.py).

Checkpointed append-only shards under data/embed-r7/gen/ (gitignored):
    llm-translations-r7.shard{N}.jsonl
rows: {key, target_lang, pass, status, text, [reason]}
Resume = (key, pass) pairs already present. Pass >= 2 re-attempts only
keys whose latest row is not ok.

Runs on its own venv (.venv-r7gen, llama-cpp-python==0.3.36, CPU).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
GEN_DIR_R4 = R4_WT / "data" / "embed-r4" / "gen"
OUT_DIR = R5_WT / "data" / "embed-r7" / "gen"
NUMBERS_DIR = R5_WT / "datasets" / "embed-r7"
RUN_LOG = NUMBERS_DIR / "run-log.jsonl"
MANIFESTS = sorted(GEN_DIR_R4.glob("llm-translations.shard*.jsonl"))

#: Round-7 teacher: Qwen3-1.7B, official GGUF repo, Q8_0 (hf download
#: 2026-10-10, blob 061b54da…, 1.83 GB). Same family/quant/runtime as the
#: round-4 0.6B teacher — the prereg's single data variable.
TEACHER_MODEL = "Qwen/Qwen3-1.7B-GGUF"
TEACHER_GGUF = (
    "/var/home/abyss/.distrobox/ubuntu/home/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-1.7B-GGUF/snapshots/"
    "90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf"
)

sys.path.insert(0, str(R4_WT / "scripts"))

from gen_embed_r4_texts import (  # noqa: E402
    BATCH,
    _clean_translation,
    _is_copy,
    _translate_prompt,
    append_jsonl,
    detect_lang,
    load_jsonl,
    log_event,
    missing_digit_tokens,
    normalise,
)
from translate_llm_rows import unique_llm_units  # noqa: E402

MAX_NEW_TOKENS = 560
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
SEED_BASE = 1_700_000  # distinct from round-4's 900000
#: max generation passes including the first (pass 1 + 2 retries)
MAX_PASS = 3


def log(msg_event: str, **fields: object) -> None:
    """Own append-only run-log (numbers only — no texts)."""
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "event": msg_event, **fields},
                ensure_ascii=False,
            ) + "\n"
        )
    print(f"[r7-run-log] {msg_event} {fields}", flush=True)


def r7_units() -> dict[str, tuple[str, str]]:
    """key -> (target_lang, source_text) for EXACTLY the 1617 sealed pairs."""
    ok_rows = [r for p in MANIFESTS for r in load_jsonl(p) if r.get("status") == "ok"]
    expected = {r["key"]: r["target_lang"] for r in ok_rows}
    assert len(expected) == 1617, f"expected 1617 sealed ok pairs, got {len(expected)}"
    derived_all = {k: (tgt, src) for k, tgt, src in unique_llm_units()}
    missing = set(expected) - set(derived_all)
    assert not missing, f"{len(missing)} sealed keys not re-derivable from synth-mono"
    derived = {k: derived_all[k] for k in expected}
    bad_tgt = [k for k in expected if derived[k][0] != expected[k]]
    assert not bad_tgt, f"target-lang drift on {len(bad_tgt)} keys"
    return derived


def load_state(shard_paths: list[Path]) -> tuple[dict[str, str], set[tuple[str, int]]]:
    """latest status per key + done (key, pass) pairs across shard files."""
    latest: dict[str, str] = {}
    done: set[tuple[str, int]] = set()
    for p in shard_paths:
        for r in load_jsonl(p):
            done.add((r["key"], r["pass"]))
            latest[r["key"]] = r["status"]
    return latest, done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--pass", dest="gen_pass", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args(argv)
    assert 1 <= args.gen_pass <= MAX_PASS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"llm-translations-r7.shard{args.shard}.jsonl"
    shard_paths = [OUT_DIR / f"llm-translations-r7.shard{s}.jsonl"
                   for s in range(args.num_shards)]

    units = r7_units()
    latest, done = load_state(shard_paths)

    if args.gen_pass == 1:
        todo_keys = [k for k in units if (k, 1) not in done]
    else:
        todo_keys = [k for k in units
                     if latest.get(k) != "ok" and (k, args.gen_pass) not in done]
    # deterministic shard split over the pass-1 unit order; retry passes
    # keep the same order filtered to failures (same shard owns a key)
    todo_set = set(todo_keys)
    jobs = []
    all_units = list(units.items())
    for i in range(0, len(all_units), BATCH):
        chunk = all_units[i : i + BATCH]
        if (i // BATCH) % args.num_shards != args.shard:
            continue
        sel = [(k, tgt, src) for k, (tgt, src) in chunk if k in todo_set]
        if sel:
            jobs.append((f"r7p{args.gen_pass}-{i // BATCH:05d}", sel))
    n_todo = sum(len(sel) for _j, sel in jobs)
    log(
        "retranslate-start",
        teacher=TEACHER_MODEL,
        quant="Q8_0",
        gen_pass=args.gen_pass,
        shard=args.shard,
        num_shards=args.num_shards,
        units_total=len(units),
        todo=n_todo,
    )
    if not jobs:
        log("retranslate-nothing-todo", gen_pass=args.gen_pass, shard=args.shard)
        return 0

    from llama_cpp import Llama

    llm = Llama(TEACHER_GGUF, n_ctx=3072, n_threads=args.threads,
                n_batch=512, verbose=False)

    def render(messages: list[dict]) -> str:
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n")
        return "".join(out)

    t0 = time.time()
    kept = bad = 0
    for n, (jid, chunk) in enumerate(jobs):
        prompts = [_translate_prompt(tgt, src) for _k, tgt, src in chunk]
        raws = []
        for pr in prompts:
            res = llm(
                render(pr),
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                seed=SEED_BASE + args.gen_pass * 1_000_000 + n,
                stop=["<|im_end|>"],
            )
            raws.append(res["choices"][0]["text"].strip())
        rows = []
        for (key, tgt, src), raw in zip(chunk, raws):
            text = normalise(_clean_translation(raw))
            reasons = []
            if not text:
                reasons.append("empty")
            if missing_digit_tokens(src, text):
                reasons.append("digit-loss:" + ",".join(missing_digit_tokens(src, text)[:3]))
            if detect_lang(text) != tgt:
                reasons.append(f"lang:{detect_lang(text)}!={tgt}")
            if _is_copy(src, text):
                reasons.append("copy")
            if not (40 <= len(text) <= 4000):
                reasons.append(f"len:{len(text)}")
            if reasons:
                bad += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt,
                             "pass": args.gen_pass, "status": "fail",
                             "text": text, "reason": ";".join(reasons)})
            else:
                kept += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt,
                             "pass": args.gen_pass, "status": "ok",
                             "text": text})
        append_jsonl(out_path, rows)
        if n % 5 == 0:
            rate = (n + 1) / max(1.0, time.time() - t0)
            print(
                f"r7 pass{args.gen_pass} shard{args.shard} job {n + 1}/{len(jobs)} "
                f"rate={rate:.3f} eta={(len(jobs) - n - 1) / max(0.001, rate) / 60:.0f}min",
                flush=True,
            )
    log(
        "retranslate-done",
        gen_pass=args.gen_pass,
        shard=args.shard,
        kept=kept,
        bad=bad,
        wall_sec=round(time.time() - t0, 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
