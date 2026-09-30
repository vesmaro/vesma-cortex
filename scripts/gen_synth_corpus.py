#!/usr/bin/env python3
"""Generate the stage-1 synthetic pair corpus via local ollama (A2s, П3).

CLI wrapper — lives OUTSIDE src/cortex on purpose: src/cortex is network-free
by contract (tests/test_skeleton.py AST tripwire), and this script is the only
place where the LLM callback meets a real HTTP backend.

Strategies and labels (ADR 0001 addendum П3, labels BY CONSTRUCTION):

    paraphrase        label 1  LLM rewrite, meaning preserved (+ light
                               procedural variations from corruption.py)
    near-topic        label 0  LLM new record, same topic, different fact
    broken-field      label 0  procedural (corruption hard negatives)
    trivial-negative  label 0  procedural (different topics; translation
                               twins refused by the topic-key guard)

Usage:

    # owner-validation sample (24 pairs → 8/8/4/4, ~5-10 min on CPU):
    uv run python scripts/gen_synth_corpus.py --sample 24 --out-dir data/synth-sample

    # full corpus after the owner validates the sample (defaults 300/strategy):
    uv run python scripts/gen_synth_corpus.py --pairs-per-strategy 300

Outputs (out-dir, default data/synth/ — gitignored, content NEVER committed):

    pairs.jsonl   one canonical JSON row per pair (pair_id, strategy, label,
                  record, candidate, seed)
    manifest.txt  ``pair_id <sha256>`` lines, sorted (data-contract §5 scheme)
    report.json   corpus fingerprint (BLAKE2b-256 over the manifest),
                  strategy/drop counters, full provenance (model, digest,
                  options, PROMPT_VERSION, seed scheme, wall time)

Determinism: procedural strategies are exactly seed-deterministic; LLM output
is pinned by temperature=0 + per-call seed (options.seed = master seed +
call index) — reproducible on the same machine/model, re-verified by the
corpus fingerprint after any re-run.

Exit codes: 0 ok · 2 usage error · 1 backend/transport failure (loud, never
a silent partial corpus).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.synth import (  # noqa: E402
    LLM_STRATEGIES,
    PROMPT_VERSION,
    STRATEGIES,
    SynthStats,
    TOPICS,
    corpus_fingerprint_of_pairs,
    generate_corpus,
    manifest_bytes_of_pairs,
    pair_row_json,
    sample_quotas,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SEED = 7
DEFAULT_NUM_PREDICT = 512
DEFAULT_TIMEOUT_S = 600.0
REQUEST_RETRIES = 2


def ollama_generate(
    prompt: str,
    *,
    host: str,
    model: str,
    seed: int,
    num_predict: int,
    timeout: float,
) -> str:
    """One non-streaming /api/generate call; raises on transport failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def ollama_model_info(host: str, model: str, timeout: float) -> dict[str, Any] | None:
    """Model entry from /api/tags (digest, size) or None when absent."""
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as response:
        tags = json.loads(response.read().decode("utf-8"))
    for entry in tags.get("models", []):
        if entry.get("model") == model or entry.get("name") == model:
            return entry
    return None


def build_llm_fn(
    *,
    host: str,
    model: str,
    master_seed: int,
    num_predict: int,
    timeout: float,
    stats: SynthStats,
) -> Callable[[str], str]:
    """The injected callback: prompt → text, with retries and progress lines.

    Per-call seed = master seed + call index (recorded in report.json as the
    seed scheme); transient transport errors are retried REQUEST_RETRIES
    times with linear backoff, then the failure propagates LOUDLY — the
    caller must never end up with a silently truncated corpus.
    """
    call_index = 0

    def llm_fn(prompt: str) -> str:
        nonlocal call_index
        index, call_index = call_index, call_index + 1
        last_error: Exception | None = None
        for attempt in range(REQUEST_RETRIES + 1):
            started = time.monotonic()
            try:
                text = ollama_generate(
                    prompt,
                    host=host,
                    model=model,
                    seed=master_seed + index,
                    num_predict=num_predict,
                    timeout=timeout,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < REQUEST_RETRIES:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break
            elapsed = time.monotonic() - started
            print(
                f"[llm {index + 1}] {len(text)} chars in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            return text
        raise RuntimeError(
            f"ollama backend failed after {REQUEST_RETRIES + 1} attempts "
            f"({host}, model {model!r}): {last_error}"
        )

    return llm_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the A2s synthetic pair corpus via local ollama.",
    )
    sampling = parser.add_mutually_exclusive_group(required=True)
    sampling.add_argument(
        "--pairs-per-strategy",
        type=int,
        metavar="N",
        help="uniform quota per strategy (full run; 300 → ~1200 pairs)",
    )
    sampling.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="owner-validation sample of N pairs (quotas: ceil(N/3) per LLM "
             "strategy, rest split procedurally; N=24 → 8/8/4/4)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"ollama base URL (default {DEFAULT_HOST})",
    )
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default data/synth/; use data/synth-sample/ "
             "for the owner-validation sample)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing corpus in --out-dir (default: refuse)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample is not None:
        if args.sample <= 0:
            parser.error("--sample must be a positive integer")
        quotas = sample_quotas(args.sample)
        pairs_per_strategy = 0  # base cycling aligns topics across strategies
        mode: dict[str, Any] = {"mode": "sample", "sample_n": args.sample}
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "synth-sample"
    else:
        if args.pairs_per_strategy <= 0:
            parser.error("--pairs-per-strategy must be a positive integer")
        quotas = {strategy: args.pairs_per_strategy for strategy in STRATEGIES}
        pairs_per_strategy = args.pairs_per_strategy
        mode = {"mode": "full", "pairs_per_strategy": args.pairs_per_strategy}
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "synth"

    pairs_file = out_dir / "pairs.jsonl"
    if pairs_file.exists() and not args.force:
        print(
            f"refusing to overwrite existing corpus {pairs_file} (use --force "
            "or another --out-dir; a fingerprinted corpus is never clobbered)",
            file=sys.stderr,
        )
        return 1

    need_llm = any(quotas[s] > 0 for s in LLM_STRATEGIES)
    provenance: dict[str, Any] = {
        "backend": "ollama" if need_llm else "procedural-only",
        "host": args.host,
        "model": args.model if need_llm else None,
        "options": {
            "temperature": 0,
            "seed_scheme": "master seed + call index",
            "master_seed": args.seed,
            "num_predict": args.num_predict,
        },
        "prompt_version": PROMPT_VERSION,
    }

    stats = SynthStats()
    llm_fn: Callable[[str], str] | None = None
    if need_llm:
        info = ollama_model_info(args.host, args.model, args.timeout)
        if info is None:
            print(
                f"model {args.model!r} not found on {args.host} — pull it first "
                "(ollama pull) or pass --model; aborting before any generation",
                file=sys.stderr,
            )
            return 1
        provenance["model_digest"] = info.get("digest")
        provenance["model_size_bytes"] = info.get("size")
        llm_fn = build_llm_fn(
            host=args.host,
            model=args.model,
            master_seed=args.seed,
            num_predict=args.num_predict,
            timeout=args.timeout,
            stats=stats,
        )

    started = time.monotonic()
    pairs = generate_corpus(
        TOPICS,
        pairs_per_strategy,
        llm_fn,
        args.seed,
        quotas=quotas,
        stats=stats,
    )
    elapsed = time.monotonic() - started

    fingerprint = corpus_fingerprint_of_pairs(pairs)
    by_strategy = {strategy: 0 for strategy in STRATEGIES}
    by_label = {"duplicate": 0, "not-duplicate": 0}
    for pair in pairs:
        by_strategy[pair.strategy] += 1
        by_label["duplicate" if pair.label == 1 else "not-duplicate"] += 1

    report = {
        "kind": "synth-corpus-report",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "cortex.synth.generate",
        **mode,
        "seed": args.seed,
        "quotas": quotas,
        "provenance": provenance,
        "counters": {**stats.as_dict(), "by_strategy": by_strategy, "by_label": by_label},
        "corpus_fingerprint": fingerprint,
        "elapsed_s": round(elapsed, 1),
    }

    if pairs:
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_file.write_text(
            "\n".join(pair_row_json(pair) for pair in pairs) + "\n", encoding="utf-8"
        )
        (out_dir / "manifest.txt").write_bytes(manifest_bytes_of_pairs(pairs))
    report["files"] = {
        "pairs": pairs_file.name if pairs else None,
        "manifest": "manifest.txt" if pairs else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({**report["counters"], "corpus_fingerprint": fingerprint,
                      "out_dir": str(out_dir), "elapsed_s": report["elapsed_s"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
