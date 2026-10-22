#!/usr/bin/env python3
"""Round-7 filter — teacher-cos gate + n-gram zero-tolerance on the new
llm-class translations.

Phase contract (round-4/5 machinery, one model at a time):
  1. cos gate: every status-ok r7 translation is embedded with the KD
     teacher Qwen/Qwen3-Embedding-0.6B (last-token pooling, bf16, the
     round-4 Embedder verbatim) together with its source; cos < 0.55
     flips the row to `cos-reject` (round-4/5 threshold, unchanged).
  2. n-gram 8-13 zero-tolerance vs the gate set (cb367733…/cb367577
     surface) + the 144 burned V43-TDUP pairs — the round-5 lesson
     applied to ALL new material: any ok row sharing at least one
     8-13-gram flips to `ngram-reject`.

Rows are annotated in place (append-only rewrite via a tmp file +
atomic rename). Failures feed the retry passes of retranslate_llm_r7;
whatever is still rejected after the last pass drops out of the corpus
at assembly (exact numbers in the report).

Runs on the repo .venv (torch CPU + transformers) — never alongside
the translation teacher in RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

R4_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
)
R5_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
)
REDO_WT = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo"
)
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
NUMBERS_DIR = R5_WT / "datasets" / "embed-r7"
RUN_LOG = NUMBERS_DIR / "run-log.jsonl"
GATE_SET = R4_WT / "data" / "embed-r4" / "r5_gate" / "gate_set.jsonl"
BURNED = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

MIN_COS = 0.55
GRAM_LO, GRAM_HI = 8, 13

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import load_jsonl, normalise  # noqa: E402
from translate_llm_rows import unique_llm_units  # noqa: E402


def log_event(event: str, **fields: object) -> None:
    """Own append-only run-log (numbers only) — NOT the r4 run-log."""
    import json as _json

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(
            _json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": event,
                    **fields,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"[r7-run-log] {event} {fields}", flush=True)


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    out: set[str] = set()
    for n in range(GRAM_LO, GRAM_HI + 1):
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def banned_surface() -> set[str]:
    grams: set[str] = set()
    for line in open(GATE_SET, encoding="utf-8"):
        g = json.loads(line)
        grams |= ngrams(g["text_src"])
        grams |= ngrams(g["text_dst"])
    n_burned = 0
    for line in open(BURNED, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        n_burned += 1
        for sk in ("record", "candidate"):
            side = r.get(sk) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    grams |= ngrams(v)
    log_event(
        "r7-banned-surface",
        gate_pairs=180,
        burned_pairs=n_burned,
        banned_grams=len(grams),
    )
    return grams


def rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--skip-cos",
        action="store_true",
        help="n-gram gate only (cos already annotated)",
    )
    args = p.parse_args(argv)

    units = {k: src for k, _tgt, src in unique_llm_units()}
    shard_paths = sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl"))
    assert shard_paths, "no r7 shard files found"

    banned = banned_surface()

    if args.skip_cos:
        emb = None
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer

        EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
        torch.set_num_threads(args.threads)

        class Embedder:
            def __init__(self):
                self.torch = torch
                self.tok = AutoTokenizer.from_pretrained(
                    EMBED_MODEL, padding_side="right"
                )
                self.model = AutoModel.from_pretrained(
                    EMBED_MODEL, dtype=torch.bfloat16
                ).eval()
                log_event("embedder-loaded", model=EMBED_MODEL, threads=args.threads)

            def embed(self, texts: list[str], batch: int = 32):
                out = []
                for i in range(0, len(texts), batch):
                    chunk = texts[i : i + batch]
                    enc = self.tok(
                        chunk,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    with torch.no_grad():
                        hidden = self.model(**enc).last_hidden_state
                    lens = enc["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[torch.arange(len(chunk)), lens]
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    out.append(pooled)
                    if (i // batch) % 20 == 0:
                        print(f"embed {i + len(chunk)}/{len(texts)}", flush=True)
                return self.torch.cat(out)

        emb = Embedder()

    t0 = time.time()
    for path in shard_paths:
        rows = load_jsonl(path)
        live = [r for r in rows if r.get("status") == "ok" and "cos" not in r]
        if live and not args.skip_cos:
            s = emb.embed([normalise(units[r["key"]]) for r in live])
            d = emb.embed([normalise(r["text"]) for r in live])
            cos = (s * d).sum(dim=1)
            rejected = 0
            for r, c in zip(live, cos.tolist()):
                r["cos"] = round(float(c), 4)
                if c < MIN_COS:
                    r["status"] = "cos-reject"
                    r["reason"] = f"cos:{r['cos']}<{MIN_COS}"
                    rejected += 1
            log_event(
                "r7-filter-cos", file=path.name, checked=len(live), rejected=rejected
            )
        # n-gram zero-tolerance (new material) — ok rows only
        live2 = [r for r in rows if r.get("status") == "ok"]
        ng_rejected = 0
        for r in live2:
            hit = ngrams(r["text"]) & banned
            if hit:
                r["status"] = "ngram-reject"
                r["reason"] = "ngram-8-13-hit"
                r["ngram_hits"] = len(hit)
                ng_rejected += 1
        log_event(
            "r7-filter-ngram", file=path.name, checked=len(live2), rejected=ng_rejected
        )
        rewrite(path, rows)
    log_event("r7-filter-done", wall_sec=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
