#!/usr/bin/env python3
"""Round-4 embedder corpus — teacher-embedding filter phase.

Runs AFTER the generation chain (gen_embed_r4_texts.py) and BEFORE
assembly. Uses Qwen/Qwen3-Embedding-0.6B (the KD teacher, local) to:

1. cos-filter every translated pair (real-record translations + synthetic
   RU<->EN twins): cos(source, translation) >= --min-cos, else the row is
   marked rejected (assembly drops rejected rows; counted in run-log).
   A 0.6B generator mistranslates sometimes — the multilingual teacher
   catches content drift. Written pairs keep `status: ok` / get
   `status: cos-reject`.
2. Non-verbatim verification vs the round-3 pool (sample-based): per
   synthetic family, sample new rows vs round-3 rows of the SAME family;
   report max-cos. Exact reuse is already excluded by hash in assembly;
   this measures near-copy pressure (flag at >= 0.97).

The generator model is NOT loaded here (RAM discipline: phases run one
model at a time).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma")
for p in (str(REPO_ROOT / "src"), str(_ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.dataset.prepare_dataset import normalise  # noqa: E402
from gen_embed_r4_texts import (  # noqa: E402
    GEN_DIR,
    TRANSLATED_SEED_DIR,
    load_jsonl,
    log_event,
    real_units,
)
from translate_llm_rows import unique_llm_units  # noqa: E402

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ROUND3_TRAIN = _ENGINE / "training" / "probe" / "corpus" / "train.jsonl"
ROUND3_VAL = _ENGINE / "training" / "probe" / "corpus" / "val.jsonl"


class Embedder:
    def __init__(self, threads: int, max_len: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="right")
        self.model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.bfloat16).eval()
        self.max_len = max_len
        log_event("embedder-loaded", model=EMBED_MODEL, threads=threads)

    def embed(self, texts: list[str], batch: int = 32) -> list:
        import torch

        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            # last-token pooling (Qwen3-Embedding contract)
            lens = enc["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(len(chunk)), lens]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled)
            if (i // batch) % 20 == 0:
                print(f"embed {i + len(chunk)}/{len(texts)}", flush=True)
        return self.torch.cat(out)


def cos_matrix(a, b):
    return a @ b.T


def cmd_filter(min_cos: float, threads: int) -> int:
    emb = Embedder(threads)
    t0 = time.time()
    checked = 0
    rejections = 0

    def cos_gate(path: Path, get_pair, mark_label: str) -> None:
        nonlocal checked, rejections
        rows = load_jsonl(path)
        live = [r for r in rows if r.get("status") == "ok"]
        if not live or all("cos" in r for r in live):
            return  # nothing to gate / already gated
        s = emb.embed([normalise(get_pair(r)[0]) for r in live])
        d = emb.embed([normalise(get_pair(r)[1]) for r in live])
        cos = (s * d).sum(dim=1)
        rejected = 0
        for r, c in zip(live, cos.tolist()):
            checked += 1
            r["cos"] = round(c, 4)
            if c < min_cos:
                r["status"] = "cos-reject"
                rejected += 1
        rejections += rejected
        append_rows(path, rows)
        log_event(mark_label, file=path.name, checked=len(live), rejected=rejected)

    units = {}

    for key, _tgt, unit in real_units():
        units[key] = unit

    # 1. real-record translations: cos-gate per shard file --------------------
    for tr_path in sorted(GEN_DIR.glob("real-translations.shard*.jsonl")):
        cos_gate(
            tr_path, lambda r: (units[r["key"]], r["text"]), "filter-real-translations"
        )

    # twins: ru/en pairs per shard file
    for tw_path in sorted(GEN_DIR.glob("synth-twins.shard*.jsonl")):
        cos_gate(tw_path, lambda r: (r["ru"], r["en"]), "filter-twins")
    # llm-row translations: cos gate vs their source rows
    llm_units = {}
    for key, _tgt, unit in unique_llm_units():
        llm_units[key] = unit
    for path in sorted(GEN_DIR.glob("llm-translations.shard*.jsonl")):
        cos_gate(
            path, lambda r: (llm_units[r["key"]], r["text"]), "filter-llm-translations"
        )
    log_event(
        "filter-pairs-done",
        checked=checked,
        rejected=rejections,
        min_cos=min_cos,
        wall_sec=round(time.time() - t0, 1),
    )

    # 2. seed pairs: measure only (TL-validated content is not re-gated) ------
    seed_pairs: list[tuple[str, str]] = []
    for path in sorted(TRANSLATED_SEED_DIR.glob("batch-*.jsonl")):
        for row in load_jsonl(path):
            for a_key, b_key in (("record", "translation"), ("record", "sibling")):
                a, b = row.get(a_key), row.get(b_key)
                if isinstance(a, dict) and isinstance(b, dict):
                    ta = normalise(f"{a.get('title', '')}. {a.get('body', '')}").strip(
                        ". "
                    )
                    tb = normalise(f"{b.get('title', '')}. {b.get('body', '')}").strip(
                        ". "
                    )
                    if len(ta) >= 40 and len(tb) >= 40:
                        seed_pairs.append((ta, tb))
    if seed_pairs:
        s = emb.embed([p[0] for p in seed_pairs])
        d = emb.embed([p[1] for p in seed_pairs])
        vals = (s * d).sum(dim=1).tolist()
        below = sum(1 for c in vals if c < min_cos)
        log_event(
            "filter-seed-measured-only",
            pairs=len(seed_pairs),
            below_min_cos=below,
            min_cos=min_cos,
            cos_min=round(min(vals), 4),
            cos_p50=round(sorted(vals)[len(vals) // 2], 4),
        )

    # 3. non-verbatim sample check vs round-3 (same-family) -------------------
    r3 = load_jsonl(ROUND3_TRAIN) + load_jsonl(ROUND3_VAL)
    r3_by_family: dict[str, list[str]] = {}
    for row in r3:
        src = str(row.get("source", ""))
        if src.startswith("synthetic-"):
            fam = src.removeprefix("synthetic-ru-").removeprefix("synthetic-en-")
            r3_by_family.setdefault(fam, []).append(normalise(row["text"]))
    by_fam_lang: dict[tuple[str, str], list[str]] = {}
    for path in sorted(GEN_DIR.glob("synth-mono.shard*.jsonl")):
        for r in load_jsonl(path):
            if r.get("status") == "ok":
                by_fam_lang.setdefault((r["family"], r["lang"]), []).append(
                    normalise(r["text"])
                )
    rng = random.Random(42)
    report = {}
    worst = 0.0
    for (fam, lang), texts in sorted(by_fam_lang.items()):
        sample_new = rng.sample(texts, min(30, len(texts)))
        pool = r3_by_family.get(fam, [])
        if not pool or not sample_new:
            continue
        sample_old = rng.sample(pool, min(400, len(pool)))
        a = emb.embed(sample_new)
        b = emb.embed(sample_old)
        mx = cos_matrix(a, b).max().item()
        worst = max(worst, mx)
        report[f"{fam}-{lang}"] = round(mx, 4)
    log_event(
        "filter-nonverbatim-sample",
        worst_max_cos=round(worst, 4),
        flag_at=0.97,
        per_family=report,
    )
    log_event(
        "filter-done",
        total_checked=checked,
        total_rejected=rejections,
        wall_sec=round(time.time() - t0, 1),
    )
    return 0


def append_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="teacher-embedding filter for the round-4 pool"
    )
    p.add_argument("--min-cos", type=float, default=0.55)
    p.add_argument("--threads", type=int, default=10)
    args = p.parse_args(argv)
    return cmd_filter(args.min_cos, args.threads)


if __name__ == "__main__":
    raise SystemExit(main())
