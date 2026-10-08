#!/usr/bin/env python3
"""Phase-0.5 Job 2 — pilot pair pool + deterministic 25% subset.

Reconstructs the round-4 corpus translation-pair pool with the EXACT
pairing recipe of datasets/embed-r5/build_gate_set.py (real/llm via
recomputed units + gen key match; seed-dup inline via side_text_seed;
twin inline; pair_key = hash of both side hashes; construction-order
dedup, first occurrence wins) and selects the deterministic pilot
subset for the λ/temperature grid.

Pilot-specific rules (logged, prereg §1.1/§3 spirit):
  * both sides must sit in the sealed r4 TRAIN slice (the pilot trains on
    train rows only, mirroring the final round-5 training input);
  * a pair is dropped when EITHER side's text sha is in the gate manifest
    excluded_text_shas_all_sides (360 shas, gate cb367577) — "the
    exclusion is what matters" (TL mission); no other exclusions:
    burned V43-TDUP pairs stay in the training pool per prereg §3;
  * translated-sibling is never a positive (contrast pairs — gate
    builder note);
  * subset = pairs sorted by pair_key, every 4th kept (exact 25%, no
    RNG; seed-independent determinism, method logged).

Read-only inputs; writes training-pilot/results/pilot_subset.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
GEN_DIR = R4_WT / "data" / "embed-r4" / "gen"
CORPUS_DIR = R4_WT / "data" / "embed-r4" / "corpus"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
GATE_MANIFEST = R4_WT / "data" / "embed-r4" / "r5_gate" / "manifest.json"
OUT_JSON = Path(__file__).resolve().parent / "results" / "pilot_subset.json"

EXPECTED_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"
SUBSET_STEP = 4  # every 4th of the pair_key-sorted pool == 25%

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import (  # noqa: E402
    MAX_CHARS,
    UNIT_TOKEN_CAP,
    count_tokens,
    detect_lang,
    load_jsonl,
    normalise,
    text_hash,
)


def side_text_seed(side: dict) -> str:
    """Exact collect_seed composition (assemble_embed_r4.py)."""
    title = str(side.get("title") or "").strip()
    body = str(side.get("body") or "").strip()
    if not title and not body:
        return ""
    text = f"{title}. {body}" if title else body
    return normalise(text)[:MAX_CHARS]


def real_units() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for rec in load_jsonl(REAL_PART):
        body = (rec.get("side") or {}).get("body") or rec.get("content") or ""
        if not isinstance(body, str) or not body.strip():
            continue
        key = rec.get("content_hash") or text_hash(body)
        src_lang = (rec.get("side") or {}).get("language") or rec.get("language") or detect_lang(body)
        target = "en" if src_lang == "ru" else "ru"
        unit = body.strip()
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n\n" in unit:
            unit = unit.rsplit("\n\n", 1)[0].strip()
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n" in unit:
            unit = unit.rsplit("\n", 1)[0].strip()
        if count_tokens(unit) > UNIT_TOKEN_CAP:
            while count_tokens(unit) > 350 and len(unit) > 400:
                unit = unit[: int(len(unit) * 0.8)].rsplit(" ", 1)[0].strip()
        out[key] = (unit, target)
    return out


def llm_units() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    for path in sorted(GEN_DIR.glob("synth-mono.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            h = text_hash(row["text"])
            if h in seen:
                continue
            seen.add(h)
            target = "ru" if row["lang"] == "en" else "en"
            out[h] = (row["text"], target)
    return out


def load_corpus() -> dict[str, dict]:
    """text_hash -> {file, line, source, lang, text}; asserts sealed fp."""
    tp, vp = CORPUS_DIR / "train.jsonl", CORPUS_DIR / "val.jsonl"
    digest = hashlib.sha256()
    digest.update(tp.read_bytes())
    digest.update(vp.read_bytes())
    if digest.hexdigest() != EXPECTED_FP:
        raise SystemExit(f"FINGERPRINT MISMATCH: {digest.hexdigest()} != {EXPECTED_FP}")
    corpus: dict[str, dict] = {}
    for fn in ("train.jsonl", "val.jsonl"):
        with open(CORPUS_DIR / fn, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                row = json.loads(line)
                h = text_hash(row["text"])
                if h in corpus:
                    raise SystemExit(f"corpus hash collision at {fn}:{i}")
                corpus[h] = {"file": fn, "line": i, "source": row["source"],
                             "lang": row["lang"], "text": row["text"]}
    return corpus


def main() -> int:
    t0 = time.time()
    manifest = json.loads(GATE_MANIFEST.read_text())
    gate_shas = set(manifest["excluded_text_shas_all_sides"])
    assert manifest["integrity"]["n_text_shas"] == len(gate_shas)

    corpus = load_corpus()
    units_real, units_llm = real_units(), llm_units()
    print(f"corpus verified rows={len(corpus)}; real units={len(units_real)}; llm units={len(units_llm)}")

    candidates: list[dict] = []

    def add_candidate(cls, text_src, lang_src, text_dst, lang_dst, provenance):
        hs_src, hs_dst = text_hash(text_src), text_hash(text_dst)
        candidates.append({
            "class": cls,
            "sha_src": hs_src, "sha_dst": hs_dst,
            "text_src": corpus[hs_src]["text"] if hs_src in corpus else text_src,
            "lang_src": lang_src,
            "text_dst": corpus[hs_dst]["text"] if hs_dst in corpus else text_dst,
            "lang_dst": lang_dst,
            "row_src": corpus.get(hs_src),
            "row_dst": corpus.get(hs_dst),
            "pair_key": text_hash(f"{hs_src}\x00{hs_dst}"),
            "provenance": provenance,
        })

    # seed-dup (inline, verbatim recipe)
    n_seed = 0
    for path in sorted(SEED_DIR.glob("batch-translated-dup-*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            rec, tr = row.get("record"), row.get("translation")
            if not isinstance(rec, dict) or not isinstance(tr, dict):
                continue
            t_rec, t_tr = side_text_seed(rec), side_text_seed(tr)
            if not t_rec or not t_tr:
                continue
            lang_rec = str(rec.get("lang") or rec.get("language") or "")
            lang_tr = str(tr.get("lang") or tr.get("language") or "")
            add_candidate("seed-dup", t_rec, lang_rec, t_tr, lang_tr,
                          {"gen_file": path.name, "gen_line": ln})
            n_seed += 1

    # real (key match)
    n_real = 0
    for path in sorted(GEN_DIR.glob("real-translations.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            unit = units_real.get(row.get("key"))
            if unit is None:
                continue
            unit_text, target = unit
            tr_text = normalise(row["text"])
            if not tr_text:
                continue
            lang_src = "ru" if target == "en" else "en"
            add_candidate("real", unit_text, lang_src, tr_text, target,
                          {"gen_file": path.name, "gen_line": ln})
            n_real += 1

    # llm (key match)
    n_llm = 0
    for path in sorted(GEN_DIR.glob("llm-translations.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            unit = units_llm.get(row.get("key"))
            if unit is None:
                continue
            unit_text, target = unit
            tr_text = normalise(row["text"])
            if not tr_text:
                continue
            lang_src = "ru" if target == "en" else "en"
            add_candidate("llm", unit_text, lang_src, tr_text, target,
                          {"gen_file": path.name, "gen_line": ln})
            n_llm += 1

    # twin (inline)
    n_twin = 0
    for path in sorted(GEN_DIR.glob("synth-twins.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            ru_t, en_t = normalise(row["ru"]), normalise(row["en"])
            if not ru_t or not en_t:
                continue
            add_candidate("twin", ru_t, "ru", en_t, "en",
                          {"gen_file": path.name, "gen_line": ln})
            n_twin += 1

    print(f"candidates: seed-dup={n_seed} real={n_real} llm={n_llm} twin={n_twin}")

    # dedup by pair_key (first occurrence in construction order — gate rule)
    seen: set[str] = set()
    deduped = []
    for c in candidates:
        if c["pair_key"] in seen:
            continue
        seen.add(c["pair_key"])
        deduped.append(c)

    # pilot eligibility: both sides in TRAIN slice + expected classes + gate exclusion
    expected_dst = {"seed-dup": "translated-dup", "real": "translated-real-r4",
                    "llm": "translated-llm-r4", "twin": "twin-"}
    expected_src = {"seed-dup": "translated-dup", "twin": "twin-"}
    stats: dict[str, int] = {}
    pool: list[dict] = []
    used_texts: dict[str, str] = {}
    for c in deduped:
        cls = c["class"]
        rs, rd = c["row_src"], c["row_dst"]
        if rs is None or rd is None:
            stats[f"{cls}:not-both-sides-in-corpus"] = stats.get(f"{cls}:not-both-sides-in-corpus", 0) + 1
            continue
        if rs["file"] != "train.jsonl" or rd["file"] != "train.jsonl":
            stats[f"{cls}:side-in-val-slice"] = stats.get(f"{cls}:side-in-val-slice", 0) + 1
            continue
        if not rd["source"].startswith(expected_dst[cls]):
            stats[f"{cls}:dst-wrong-class"] = stats.get(f"{cls}:dst-wrong-class", 0) + 1
            continue
        if cls in expected_src and not rs["source"].startswith(expected_src[cls]):
            stats[f"{cls}:src-wrong-class"] = stats.get(f"{cls}:src-wrong-class", 0) + 1
            continue
        if {c["sha_src"], c["sha_dst"]} & gate_shas:
            stats[f"{cls}:gate-excluded"] = stats.get(f"{cls}:gate-excluded", 0) + 1
            continue
        # a corpus text may serve at most one positive pair (rows are unique)
        clash = None
        for h in (c["sha_src"], c["sha_dst"]):
            if h in used_texts:
                clash = used_texts[h]
        if clash:
            stats[f"{cls}:text-reused-in-{clash}"] = stats.get(f"{cls}:text-reused-in-{clash}", 0) + 1
            continue
        used_texts[c["sha_src"]] = cls
        used_texts[c["sha_dst"]] = cls
        pool.append(c)
        stats[f"{cls}:pool"] = stats.get(f"{cls}:pool", 0) + 1

    pool.sort(key=lambda c: c["pair_key"])
    subset = [c for i, c in enumerate(pool) if i % SUBSET_STEP == 0]

    out = {
        "kind": "embed-r5-phase05-pilot-subset",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": ["docs/specs/embed-round5-prereg.md §1.1-1.2, §3", "ADR-0005",
                      "TL mission phase-0.5 Job 2"],
        "subset_rule": f"pair pool sorted by pair_key, every {SUBSET_STEP}th kept "
                       f"(deterministic 25%, no RNG); pool = build_gate_set.py pairing "
                       f"recipe verbatim, both sides in sealed TRAIN slice, expected "
                       f"source classes, one-pair-per-text, gate manifest "
                       f"excluded_text_shas_all_sides applied to BOTH sides",
        "counts": {
            "candidates_raw": len(candidates),
            "after_pair_key_dedup": len(deduped),
            "pool": len(pool),
            "subset": len(subset),
            "subset_share": round(len(subset) / max(1, len(pool)), 4),
            "by_class_pool": {k.replace(":pool", ""): v for k, v in sorted(stats.items()) if k.endswith(":pool")},
            "by_class_subset": {},
            "eligibility_stats": dict(sorted(stats.items())),
        },
        "pool_key_list_sha256": hashlib.sha256(
            "\n".join(c["pair_key"] for c in pool).encode()).hexdigest(),
        "subset_key_list_sha256": hashlib.sha256(
            "\n".join(c["pair_key"] for c in subset).encode()).hexdigest(),
        "pairs": [
            {
                "pair_key": c["pair_key"], "class": c["class"],
                "text_src": c["text_src"], "lang_src": c["lang_src"],
                "text_dst": c["text_dst"], "lang_dst": c["lang_dst"],
                "sha_src": c["sha_src"], "sha_dst": c["sha_dst"],
                "row_src": {k: c["row_src"][k] for k in ("file", "line", "source")},
                "row_dst": {k: c["row_dst"][k] for k in ("file", "line", "source")},
            }
            for c in subset
        ],
    }
    by_cls: dict[str, int] = {}
    for c in subset:
        by_cls[c["class"]] = by_cls.get(c["class"], 0) + 1
    out["counts"]["by_class_subset"] = dict(sorted(by_cls.items()))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"pool={len(pool)} subset={len(subset)} ({out['counts']['subset_share']:.0%}) "
          f"by_class_subset={out['counts']['by_class_subset']}")
    print(f"-> {OUT_JSON} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
