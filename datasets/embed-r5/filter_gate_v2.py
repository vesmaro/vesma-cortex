#!/usr/bin/env python3
"""Gate-set v2 filter — teacher-cos gate + FULL n-gram zero-tolerance.

Round-4/5/7 filter machinery (filter_llm_r7.py contract, unchanged):
  1. cos gate: every status-ok v2 translation/twin is embedded with the
     KD teacher Qwen/Qwen3-Embedding-0.6B (last-token pooling, bf16,
     the round-4 Embedder verbatim) together with its source;
     cos < 0.55 flips the row to `cos-reject` (threshold unchanged).
     For twins the pair is (ru, en) — same-content sides.
  2. n-gram 8-13 zero-tolerance: the ban set is the FULL training
     surface (13 surfaces, 5,265,881 grams — the survey/manifest ban
     set, asserted equal), not the r7-era gate+burned subset. Any ok
     row sharing at least one 8-13-gram flips to `ngram-reject`.

Rows are annotated in place (append-only rewrite via tmp + atomic
rename). Runs on the repo .venv (torch CPU + transformers) — never
alongside the translation teacher in RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
GEN_DIR = R5_WT / "data" / "embed-r4" / "r5_gate_v2" / "gen"
RUN_LOG = R5_WT / "datasets" / "embed-r5" / "run-log.jsonl"
GATE_SET = R4_WT / "data" / "embed-r4" / "r5_gate" / "gate_set.jsonl"
R4_CORPUS = R4_WT / "data" / "embed-r4" / "corpus"
R5_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R7_CORPUS = R5_WT / "data" / "embed-r7" / "corpus"
R4_GEN = R4_WT / "data" / "embed-r4" / "gen"
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

MIN_COS = 0.55
GRAM_LO, GRAM_HI = 8, 13
EXPECT_BAN_GRAMS = 5265881

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import load_jsonl, normalise  # noqa: E402


def log_event(event: str, **fields: object) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "event": f"gate_v2:{event}", **fields},
            ensure_ascii=False) + "\n")
    print(f"[v2:filter] {event} {fields}", flush=True)


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    return {" ".join(w[i : i + n]) for n in range(GRAM_LO, GRAM_HI + 1)
            for i in range(len(w) - n + 1)}


def full_ban_surface() -> set[str]:
    """The 13-surface ban set — byte-consistent with the survey/manifest
    numbers (asserted). Sources are consumed read-only."""
    grams: set[str] = set()

    def add(t: str) -> None:
        t = t.strip()
        if t:
            grams.update(ngrams(t))

    for fn in ("train.jsonl", "val.jsonl"):
        for cdir in (R4_CORPUS, R5_CORPUS, R7_CORPUS):
            for line in open(cdir / fn, encoding="utf-8"):
                add(json.loads(line)["text"])
    for line in open(R4_GEN / "synth-template-mono.jsonl", encoding="utf-8"):
        add(str(json.loads(line).get("text") or ""))
    for pat, flds in (("synth-mono.shard*.jsonl", ("text",)),
                      ("llm-translations.shard*.jsonl", ("text",)),
                      ("real-translations.shard*.jsonl", ("text",)),
                      ("synth-twins.shard*.jsonl", ("ru", "en"))):
        for path in sorted(R4_GEN.glob(pat)):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                for fld in flds:
                    add(str(r.get(fld) or ""))
    for path in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            add(str(json.loads(line).get("text") or ""))
    for path in sorted(SEED_DIR.glob("batch-translated-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            for v in json.loads(line).values():
                if isinstance(v, dict):
                    for fld in ("title", "body", "text"):
                        add(str(v.get(fld) or ""))
    for line in open(REAL_PART, encoding="utf-8"):
        side = json.loads(line).get("side") or {}
        add(str(side.get("title") or ""))
        add(str(side.get("body") or ""))
    for line in open(BURNED_PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = r.get(sk) or {}
            for fld in ("title", "body"):
                add(str(side.get(fld) or ""))
    for line in open(GATE_SET, encoding="utf-8"):
        g = json.loads(line)
        add(g["text_src"])
        add(g["text_dst"])
    assert len(grams) == EXPECT_BAN_GRAMS, f"ban set mismatch: {len(grams)}"
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
    p.add_argument("--skip-cos", action="store_true")
    args = p.parse_args(argv)

    shard_paths = sorted(GEN_DIR.glob("*-v2*.jsonl"))
    assert shard_paths, "no v2 shard files found"

    log_event("filter-ban-set-start", surfaces=13)
    banned = full_ban_surface()
    log_event("filter-ban-set", banned_grams=len(banned))

    src_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    for fn in ("sources.jsonl", "sources_extension1.jsonl"):
        path = R5_WT / "data" / "embed-r4" / "r5_gate_v2" / fn
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row["kind"] == "note":
                src = row["ru"] if row["direction"] == "ru->en" else row["en"]
                src_by_key[(row["pool"], row["key"])] = (normalise(src), normalise(row["en"] if src == row["ru"] else row["ru"]))
            elif row["kind"] == "real":
                src_by_key[("real", row["key"])] = (normalise(row["unit"]), "")

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
                self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="right")
                self.model = AutoModel.from_pretrained(
                    EMBED_MODEL, dtype=torch.bfloat16
                ).eval()
                log_event("embedder-loaded", model=EMBED_MODEL, threads=args.threads)

            def embed(self, texts: list[str], batch: int = 32):
                out = []
                for i in range(0, len(texts), batch):
                    chunk = texts[i : i + batch]
                    enc = self.tok(chunk, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
                    with torch.no_grad():
                        hidden = self.model(**enc).last_hidden_state
                    lens = enc["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[torch.arange(len(chunk)), lens]
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    out.append(pooled)
                return self.torch.cat(out)

        emb = Embedder()

    t0 = time.time()
    for path in shard_paths:
        rows = load_jsonl(path)
        if path.name.startswith("synth-twins"):
            live = [r for r in rows if r.get("status") == "ok" and "cos" not in r]
            if live and not args.skip_cos:
                a = emb.embed([normalise(r["ru"]) for r in live])
                b = emb.embed([normalise(r["en"]) for r in live])
                cos = (a * b).sum(dim=1)
                rejected = 0
                for r, c in zip(live, cos.tolist()):
                    r["cos"] = round(float(c), 4)
                    if c < MIN_COS:
                        r["status"] = "cos-reject"
                        r["reason"] = f"cos:{r['cos']}<{MIN_COS}"
                        rejected += 1
                log_event("filter-cos", file=path.name, checked=len(live), rejected=rejected)
        else:
            live = [r for r in rows if r.get("status") == "ok" and "cos" not in r]
            if live and not args.skip_cos:
                s = emb.embed([src_by_key[(r["stratum"], r["key"])][0] for r in live])
                d = emb.embed([normalise(r["text"]) for r in live])
                cos = (s * d).sum(dim=1)
                rejected = 0
                for r, c in zip(live, cos.tolist()):
                    r["cos"] = round(float(c), 4)
                    if c < MIN_COS:
                        r["status"] = "cos-reject"
                        r["reason"] = f"cos:{r['cos']}<{MIN_COS}"
                        rejected += 1
                log_event("filter-cos", file=path.name, checked=len(live), rejected=rejected)
        # n-gram zero-tolerance vs FULL training surface (ok rows only)
        live2 = [r for r in rows if r.get("status") == "ok"]
        ng = 0
        for r in live2:
            texts = (r["text"],) if "text" in r else (r["ru"], r["en"])
            hits = sum(len(ngrams(t) & banned) for t in texts)
            if hits:
                r["status"] = "ngram-reject"
                r["reason"] = "ngram-8-13-hit"
                r["ngram_hits"] = hits
                ng += 1
        log_event("filter-ngram", file=path.name, checked=len(live2), rejected=ng)
        rewrite(path, rows)
    log_event("filter-done", wall_sec=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
