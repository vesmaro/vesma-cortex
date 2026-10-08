#!/usr/bin/env python3
"""Round-4 embedder corpus — assembly, decontamination, asserts, sealing.

Canonical order (dataset-engineering contract): collect -> gates ->
dedup BEFORE splits -> decontaminate -> quota check -> stratified split
(rng 42, train/val 95/5, the round-3 ``prepare_dataset`` shape) ->
fingerprint. The output rows are ``{"text","lang","source"}`` jsonl —
the round-3 structure the training lane consumes (see
``training/probe/corpus`` for the shape reference).

Decontamination exclusions (hash-set over normalised lowercase text,
every string field >= 40 chars — the capacity-probe builder method):
  - the judged S1m corpus c2ce056d (golden entries + tech patterns +
    golden queries) — the frozen no-harm set (prereg §2);
  - every sealed cortex holdout + evalset surface (b2, dataset-v4,
    v4.3/f/g holdouts, ds1000 holdout, layer-A evalsets);
  - the round-3 SYNTHETIC pool (probe corpus train+val rows with
    source ``synthetic-*``) — synthetic rows must not be reused
    verbatim (mission constraint); the round-3 real-store paragraphs
    and the TL-authored seed batches re-enter by construction (same
    domain slice, same prepared batches).

MANDATORY ASSERTS (prereg §2, TL guardrail 2026-10-07):
  A1 train+val vs judged corpus intersection == 0
  A2 train+val vs sealed holdout/evalset intersection == 0
  A3 train+val vs round-3 pool verbatim intersection == 0
  A4 RU share >= 0.40 (prepare_dataset quota, ratified §2)
  A5 translated share within the ratified band (report numerator and
     denominator exactly; band floor 0.15)
  A6 every text passes the canonical gates (40..4000 chars, <= 256
     approx tokens)

Privacy: store-derived texts stay in gitignored data/embed-r4/corpus/;
the repo receives only this script, the run-log counters and the
README numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma")
_MAIN_REPO = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex")
_REDO_WT = _MAIN_REPO / "wt" / "corpus-v43-redo"
for p in (str(REPO_ROOT / "src"), str(_ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.dataset.prepare_dataset import (  # noqa: E402
    count_tokens,
    detect_lang,
    normalise,
)
from gen_embed_r4_texts import (  # noqa: E402
    GEN_DIR,
    MIN_CHARS,
    MAX_CHARS,
    REAL_PART,
    RUN_LOG,
    TRANSLATED_SEED_DIR,
    acceptable,
    load_jsonl,
    log_event,
    text_hash,
)

OUT_DIR = REPO_ROOT / "data" / "embed-r4" / "corpus"
ROUND3_FILES = [
    _ENGINE / "training" / "probe" / "corpus" / "train.jsonl",
    _ENGINE / "training" / "probe" / "corpus" / "val.jsonl",
]

#: sealed cortex eval surfaces (READ-ONLY reads; every string >= 40 chars).
HOLDOUT_GLOBS: list[Path] = [
    _MAIN_REPO / "data" / "stage2" / "b2",
    _MAIN_REPO / "data" / "stage2" / "dataset-v4-holdout",
    _MAIN_REPO / "data" / "stage2" / "v43-holdout",
    _MAIN_REPO / "data" / "stage2" / "ds1000-holdout-300.jsonl",
    _MAIN_REPO / "data" / "stage2" / "ds1000-holdout-ids.json",
    _MAIN_REPO / "data" / "stage2" / "ds1000-holdout-dir",
    _REDO_WT / "data" / "stage2" / "v43-g-holdout",
    _REDO_WT / "data" / "stage2" / "v43-g-recheck-holdout",
    _REDO_WT / "data" / "stage2" / "v43-f-holdout",
    _REDO_WT / "data" / "stage2" / "v43-f-recheck-holdout",
    _REDO_WT / "data" / "stage2" / "v43-recheck-holdout",
    _MAIN_REPO / "datasets" / "evalsets",
]


def iter_strings(obj: object):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def hashset_of_texts(texts) -> set[str]:
    return {text_hash(t) for t in texts}


def build_judged_hashes() -> set[str]:
    """Judged S1m corpus c2ce056d: golden entries + tech patterns + queries."""
    from benchmarks.corpus.danger_labels import LABELLED_ENTRIES
    from benchmarks.corpus.queries import GOLDEN_QUERIES

    texts: list[str] = []
    for e in LABELLED_ENTRIES:
        texts.append(getattr(e, "title", "") or "")
        body = getattr(e, "content", "") or ""
        texts.append(body)
        for para in re.split(r"\n\s*\n", body):
            texts.append(para)
    texts += [q.text for q in GOLDEN_QUERIES]
    hs = {text_hash(t) for t in texts if len(normalise(t)) >= MIN_CHARS}
    log_event("decontam-judged-loaded", entries=len(LABELLED_ENTRIES), queries=len(GOLDEN_QUERIES), hashes=len(hs))
    return hs


def build_holdout_hashes() -> set[str]:
    """Every string field (>= 40 chars) of the sealed holdout/evalset files."""
    banned: set[str] = []
    seen_files = 0

    def scan_path(path: Path) -> None:
        nonlocal seen_files
        if path.is_file():
            seen_files += 1
            if path.suffix == ".jsonl":
                for row in load_jsonl(path):
                    for s in iter_strings(row):
                        if len(normalise(s)) >= MIN_CHARS:
                            banned.append(s)
            elif path.suffix == ".json":
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return
                for s in iter_strings(obj):
                    if len(normalise(s)) >= MIN_CHARS:
                        banned.append(s)
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                scan_path(child)

    for base in HOLDOUT_GLOBS:
        if base.exists():
            scan_path(base)
        else:
            log_event("decontam-holdout-missing", path=str(base))
    hs = {text_hash(t) for t in banned}
    log_event("decontam-holdouts-loaded", files=seen_files, hashes=len(hs))
    return hs


def build_round3_hashes() -> set[str]:
    """Round-3 SYNTHETIC rows only (mission: 'do not reuse round-3
    synthetic outputs verbatim'). The round-3 pool also carries the same
    real-store paragraphs and the TL-authored seed — re-including those
    is by-construction (same domain slice, same prepared batches), not
    synthetic reuse."""
    texts: list[str] = []
    n_rows = n_synth = 0
    for path in ROUND3_FILES:
        for row in load_jsonl(path):
            n_rows += 1
            if not str(row.get("source", "")).startswith("synthetic"):
                continue
            n_synth += 1
            t = normalise(str(row.get("text", "")))
            if len(t) >= MIN_CHARS:
                texts.append(t)
    hs = hashset_of_texts(texts)
    log_event("decontam-round3-loaded", rows=n_rows, synthetic_rows=n_synth, hashes=len(hs))
    return hs


# ── collectors (canonical order; first-seen wins dedup) ─────────────────────


def collect_real() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for rec in load_jsonl(REAL_PART):
        body = (rec.get("side") or {}).get("body") or ""
        if not isinstance(body, str):
            continue
        for para in re.split(r"\n\s*\n", body):
            text = normalise(para)
            if len(text) < MIN_CHARS:
                continue
            out.append((text[:MAX_CHARS], detect_lang(text), "real-part-r4"))
    return out


def collect_seed() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(TRANSLATED_SEED_DIR.glob("batch-*.jsonl")):
        kind = "translated-dup" if "-dup-" in path.name else "translated-sibling"
        for row in load_jsonl(path):
            for side_key in ("record", "translation", "sibling"):
                side = row.get(side_key)
                if not isinstance(side, dict):
                    continue
                title = str(side.get("title") or "").strip()
                body = str(side.get("body") or "").strip()
                if not title and not body:
                    continue
                text = normalise(f"{title}. {body}" if title else body)
                lang = str(side.get("lang") or detect_lang(text))
                out.append((text[:MAX_CHARS], lang, kind))
    return out


def collect_real_translations() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(GEN_DIR.glob("real-translations.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            text = normalise(row["text"])
            if not acceptable(text):
                continue
            out.append((text, row["target_lang"], "translated-real-r4"))
    return out


def collect_twins() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(GEN_DIR.glob("synth-twins.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            for lang_key, lang in (("ru", "ru"), ("en", "en")):
                text = normalise(row[lang_key])
                if not acceptable(text):
                    continue
                out.append((text, lang, f"twin-{lang}-{row['family']}"))
    return out


def collect_template_mono() -> list[tuple[str, str, str]]:
    """Extended-vocabulary template cross-product pool (TL-approved repair
    2026-10-08; round-3 machinery, axis values replaced, hash-disjoint
    from round-3 — verified at generation)."""
    out: list[tuple[str, str, str]] = []
    for row in load_jsonl(GEN_DIR / "synth-template-mono.jsonl"):
        text = normalise(row["text"])
        if not acceptable(text):
            continue
        out.append((text, row["lang"], f"synthetic-{row['lang']}-{row['family']}"))
    return out


def collect_llm_translations() -> list[tuple[str, str, str]]:
    """Teacher translations of the unique LLM mono rows (content twins of
    texts whose counterpart does not exist elsewhere)."""
    out: list[tuple[str, str, str]] = []
    for path in sorted(GEN_DIR.glob("llm-translations.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            text = normalise(row["text"])
            if not acceptable(text):
                continue
            out.append((text, row["target_lang"], "translated-llm-r4"))
    return out


def collect_synth_mono() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(GEN_DIR.glob("synth-mono.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            text = normalise(row["text"])
            if not acceptable(text):
                continue
            out.append((text, row["lang"], f"synthetic-{row['lang']}-{row['family']}"))
    return out


# ── assembly ────────────────────────────────────────────────────────────────


def assemble(min_share: float = 0.15) -> dict:
    judged = build_judged_hashes()
    holdout = build_holdout_hashes()
    round3 = build_round3_hashes()

    collectors = [
        ("real", collect_real()),
        ("seed", collect_seed()),
        ("real-translations", collect_real_translations()),
        ("twins", collect_twins()),
        ("synthetic", collect_synth_mono()),
        ("llm-translations", collect_llm_translations()),
        ("template", collect_template_mono()),
    ]
    for name, rows in collectors:
        log_event(f"collected-{name}", rows=len(rows))

    counters: Counter[str] = Counter()
    seen: set[str] = set()
    kept: list[tuple[str, str, str]] = []
    is_translated = lambda src: src.startswith("translated-") or src.startswith("twin-")
    for name, rows in collectors:
        for text, lang, src in rows:
            if not acceptable(text):  # A6 gate
                counters[f"gate-fail:{name}"] += 1
                continue
            h = text_hash(text)
            if h in judged:
                counters[f"judged-hit:{name}"] += 1
                continue
            if h in holdout:
                counters[f"holdout-hit:{name}"] += 1
                continue
            if h in round3:
                counters[f"round3-hit:{name}"] += 1
                continue
            if h in seen:
                counters[f"dup:{name}"] += 1
                continue
            seen.add(h)
            kept.append((text, lang, src))

    total = len(kept)
    by_source = Counter(src for _t, _l, src in kept)
    ru = sum(1 for _t, l, _s in kept if l == "ru")
    translated_n = sum(1 for _t, _l, src in kept if is_translated(src))
    ru_share = ru / total
    tr_share = translated_n / total

    # split: stratified by source; ONE seeded rng; sorted label iteration
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for row in kept:
        groups.setdefault(row[2], []).append(row)
    rng = random.Random(42)
    train_rows: list[tuple[str, str, str]] = []
    val_rows: list[tuple[str, str, str]] = []
    for src in sorted(groups):
        g = list(groups[src])
        rng.shuffle(g)
        n_val = max(1, round(len(g) * 0.05)) if len(g) >= 5 else 0
        val_rows += g[:n_val]
        train_rows += g[n_val:]
    train_rows.sort(key=lambda r: (r[2], r[1]))
    val_rows.sort(key=lambda r: (r[2], r[1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    with open(train_path, "w", encoding="utf-8") as fh:
        for text, lang, src in train_rows:
            fh.write(json.dumps({"text": text, "lang": lang, "source": src}, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as fh:
        for text, lang, src in val_rows:
            fh.write(json.dumps({"text": text, "lang": lang, "source": src}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256()
    digest.update(train_path.read_bytes())
    digest.update(val_path.read_bytes())
    fp = digest.hexdigest()
    (OUT_DIR / "fingerprint.txt").write_text(fp + "\n", encoding="utf-8")

    # ── MANDATORY ASSERTS on the sealed artifact ────────────────────────────
    final_texts = train_rows + val_rows
    a1 = sum(1 for t, _l, _s in final_texts if text_hash(t) in judged)
    a2 = sum(1 for t, _l, _s in final_texts if text_hash(t) in holdout)
    a3 = sum(1 for t, _l, _s in final_texts if text_hash(t) in round3)
    a4 = ru_share >= 0.40
    a5 = tr_share >= min_share
    a6 = all(acceptable(t) for t, _l, _s in final_texts)
    asserts = {"A1_judged_intersection": a1 == 0, "A2_holdout_intersection": a2 == 0,
               "A3_round3_verbatim": a3 == 0, "A4_ru_share_ge_040": a4,
               "A5_translated_share_ge_015": a5, "A6_gates": a6}
    log_event(
        "assemble-done",
        total=total,
        train=len(train_rows),
        val=len(val_rows),
        fingerprint=fp,
        ru_share=round(ru_share, 4),
        translated_num=translated_n,
        translated_den=total,
        translated_share=round(tr_share, 4),
        by_source=dict(sorted(by_source.items())),
        counters=dict(counters),
        asserts=asserts,
    )
    report = {
        "kind": "dataset-embed-r4-report",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": fp,
        "counts": {"total": total, "train": len(train_rows), "val": len(val_rows)},
        "shares": {"ru": round(ru_share, 4), "translated_num": translated_n, "translated_den": total,
                    "translated": round(tr_share, 4)},
        "by_source": dict(sorted(by_source.items())),
        "exclusion_counters": dict(counters),
        "asserts": asserts,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if not all(asserts.values()):
        failed = [k for k, v in asserts.items() if not v]
        print(f"ASSERT FAILED: {failed}", flush=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="assemble the round-4 embedder corpus")
    p.add_argument("--min-share", type=float, default=0.15)
    p.add_argument("--verify", action="store_true", help="double assembly; assert byte-identical fingerprint")
    args = p.parse_args(argv)
    rc = assemble(args.min_share)
    if args.verify and rc == 0:
        fp1 = (OUT_DIR / "fingerprint.txt").read_text()
        b1 = (OUT_DIR / "train.jsonl").read_bytes(), (OUT_DIR / "val.jsonl").read_bytes()
        rc = assemble(args.min_share)
        fp2 = (OUT_DIR / "fingerprint.txt").read_text()
        b2 = (OUT_DIR / "train.jsonl").read_bytes(), (OUT_DIR / "val.jsonl").read_bytes()
        same = fp1 == fp2 and b1 == b2
        log_event("verify-double-assembly", byte_identical=same, fp1=fp1.strip()[:16], fp2=fp2.strip()[:16])
        if not same:
            return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
