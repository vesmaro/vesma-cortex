#!/usr/bin/env python3
"""Round-5 sealed cross-lingual gate set (ArchCom contract §3.3).

Cuts 150-200 translation pairs from the round-4 translated family
BEFORE splits, stratified by source class, deterministically (fixed
seed, sorted pair keys). READ-ONLY over the sealed r4 corpus
(fingerprint asserted before use), the gen manifests that carry the
pair mapping, and the burned 144-pair eval surface (zero-overlap
assert). Texts land in the gitignored r4 data dir; only this script,
the numbers-only manifest and the run-log are committed.

Strata (translation pairs only; translated-sibling is EXCLUDED — its
record/sibling sides are contrast pairs (not-duplicate), not
translations):
  seed-dup  batch-translated-dup-{en,ru}.jsonl   record <-> translation
  real      real-translations.shard*.jsonl       unit <-> translation (key = source hash)
  llm       llm-translations.shard*.jsonl        mono text <-> translation (key = source hash)
  twin      synth-twins.shard*.jsonl             ru <-> en (inline)

Eligibility (per class): pairing verified from the generation records
AND the translation side present in the sealed corpus under the
expected source class. Exclusion keys (text sha256 of BOTH sides) are
written to the manifest so the round-5 build can assert disjointness
(pair_id/text-hash train∩gate=∅, holdout∩gate=∅) and skip these texts
in arm-B re-translation.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
GEN_DIR = R4_WT / "data" / "embed-r4" / "gen"
CORPUS_DIR = R4_WT / "data" / "embed-r4" / "corpus"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"
OUT_DIR = R4_WT / "data" / "embed-r4" / "r5_gate"

EXPECTED_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"

SEED = 42
TARGET_TOTAL = 180
BAND = (150, 200)
FLOOR_PER_STRATUM = 20

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
    """Exact replication of gen_embed_r4_texts.real_units()."""
    out: dict[str, tuple[str, str]] = {}
    for rec in load_jsonl(REAL_PART):
        body = (rec.get("side") or {}).get("body") or rec.get("content") or ""
        if not isinstance(body, str) or not body.strip():
            continue
        key = rec.get("content_hash") or text_hash(body)
        src_lang = (
            (rec.get("side") or {}).get("language")
            or rec.get("language")
            or detect_lang(body)
        )
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
    """Exact replication of translate_llm_rows.unique_llm_units()."""
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
    """text_hash -> {file, line, source, text}; asserts sealed fingerprint."""
    tp, vp = CORPUS_DIR / "train.jsonl", CORPUS_DIR / "val.jsonl"
    digest = hashlib.sha256()
    digest.update(tp.read_bytes())
    digest.update(vp.read_bytes())
    fp = digest.hexdigest()
    if fp != EXPECTED_FP:
        raise SystemExit(f"FINGERPRINT MISMATCH: {fp} != {EXPECTED_FP}")
    corpus: dict[str, dict] = {}
    for fn in ("train.jsonl", "val.jsonl"):
        with open(CORPUS_DIR / fn, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                row = json.loads(line)
                h = text_hash(row["text"])
                if h in corpus:
                    raise SystemExit(f"corpus hash collision at {fn}:{i}")
                corpus[h] = {
                    "file": fn,
                    "line": i,
                    "source": row["source"],
                    "lang": row["lang"],
                }
    return corpus


def burned_hashes() -> set[str]:
    """Text hashes of every title/body side of the burned V43-TDUP pairs."""
    hs: set[str] = set()
    n_pairs = 0
    for line in open(BURNED_PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        n_pairs += 1
        for side_key in ("record", "candidate"):
            side = r.get(side_key) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    hs.add(text_hash(v))
    burned = {"pairs": n_pairs, "side_hashes": len(hs)}
    return hs, burned


def main() -> int:
    corpus = load_corpus()
    burned, burned_info = burned_hashes()
    log: list[str] = []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    emit(f"corpus verified fp={EXPECTED_FP[:16]} rows={len(corpus)}")
    emit(f"burned surface: {burned_info}")

    units_real = real_units()
    units_llm = llm_units()
    emit(f"real units recomputed: {len(units_real)}; llm units recomputed: {len(units_llm)}")

    candidates: list[dict] = []

    def add_candidate(
        cls: str,
        direction: str,
        text_src: str,
        lang_src: str,
        text_dst: str,
        lang_dst: str,
        provenance: dict,
    ) -> None:
        hs_src, hs_dst = text_hash(text_src), text_hash(text_dst)
        burned_hit = bool({hs_src, hs_dst} & burned)
        row_src, row_dst = corpus.get(hs_src), corpus.get(hs_dst)
        pair_key = text_hash(f"{hs_src}\x00{hs_dst}")
        candidates.append(
            {
                "class": cls,
                "direction": direction,
                "text_src": text_src,
                "lang_src": lang_src,
                "text_dst": text_dst,
                "lang_dst": lang_dst,
                "sha_src": hs_src,
                "sha_dst": hs_dst,
                "pair_key": pair_key,
                "row_src": row_src,
                "row_dst": row_dst,
                "burned_hit": burned_hit,
                "provenance": provenance,
            }
        )

    # ── seed-dup ────────────────────────────────────────────────────────────
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
            direction = f"{lang_rec}->{lang_tr}"
            add_candidate(
                "seed-dup",
                direction,
                t_rec,
                lang_rec,
                t_tr,
                lang_tr,
                {"gen_file": str(path.relative_to(R4_WT)), "gen_line": ln, "pair_key": row.get("key")},
            )
            n_seed += 1
    emit(f"seed-dup candidates: {n_seed}")

    # ── real ────────────────────────────────────────────────────────────────
    n_real = n_real_nomatch = 0
    for path in sorted(GEN_DIR.glob("real-translations.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            unit = units_real.get(row.get("key"))
            if unit is None:
                n_real_nomatch += 1
                continue
            unit_text, target = unit
            tr_text = normalise(row["text"])
            if not tr_text:
                continue
            direction = "ru->en" if target == "en" else "en->ru"
            lang_src = "ru" if target == "en" else "en"
            add_candidate(
                "real",
                direction,
                unit_text,
                lang_src,
                tr_text,
                target,
                {"gen_file": path.name, "gen_line": ln, "pair_key": row.get("key")},
            )
            n_real += 1
    emit(f"real candidates: {n_real} (unmatched keys: {n_real_nomatch})")

    # ── llm ─────────────────────────────────────────────────────────────────
    n_llm = n_llm_nomatch = 0
    for path in sorted(GEN_DIR.glob("llm-translations.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            unit = units_llm.get(row.get("key"))
            if unit is None:
                n_llm_nomatch += 1
                continue
            unit_text, target = unit
            tr_text = normalise(row["text"])
            if not tr_text:
                continue
            direction = "ru->en" if target == "en" else "en->ru"
            lang_src = "ru" if target == "en" else "en"
            add_candidate(
                "llm",
                direction,
                unit_text,
                lang_src,
                tr_text,
                target,
                {"gen_file": path.name, "gen_line": ln, "pair_key": row.get("key")},
            )
            n_llm += 1
    emit(f"llm candidates: {n_llm} (unmatched keys: {n_llm_nomatch})")

    # ── twin ────────────────────────────────────────────────────────────────
    n_twin = 0
    for path in sorted(GEN_DIR.glob("synth-twins.shard*.jsonl")):
        for ln, row in enumerate(load_jsonl(path), 1):
            if row.get("status") != "ok":
                continue
            ru_t, en_t = normalise(row["ru"]), normalise(row["en"])
            if not ru_t or not en_t:
                continue
            parity = int(text_hash(f"{ru_t}\x00{en_t}"), 16) % 2
            direction = "ru->en" if parity == 0 else "en->ru"
            if direction == "ru->en":
                text_src, lang_src, text_dst, lang_dst = ru_t, "ru", en_t, "en"
            else:
                text_src, lang_src, text_dst, lang_dst = en_t, "en", ru_t, "ru"
            add_candidate(
                "twin",
                direction,
                text_src,
                lang_src,
                text_dst,
                lang_dst,
                {
                    "gen_file": path.name,
                    "gen_line": ln,
                    "pair_key": f"{row.get('job')}:{row.get('family')}:{row.get('slot')}",
                },
            )
            n_twin += 1
    emit(f"twin candidates: {n_twin}")

    # ── eligibility ─────────────────────────────────────────────────────────
    # candidate-level dedup first (gen shards carry duplicate texts; every
    # duplicate maps to the same kept corpus row and would create two gate
    # pairs with identical surfaces). Deterministic: construction order,
    # first occurrence wins.
    stats: Counter = Counter()
    seen_pair_keys: set[str] = set()
    deduped: list[dict] = []
    for c in candidates:
        if c["pair_key"] in seen_pair_keys:
            stats[f"{c['class']}:dup-candidate"] += 1
            continue
        seen_pair_keys.add(c["pair_key"])
        deduped.append(c)

    # class requirements: seed-dup and twin have BOTH sides as corpus rows
    # of the family class; real/llm require the translation side only (the
    # source unit legitimately lives outside the corpus when it is a
    # truncated body prefix). Both sides must be absent from the burned set.
    expected_dst = {"seed-dup": "translated-dup", "real": "translated-real-r4",
                    "llm": "translated-llm-r4", "twin": "twin-"}
    needs_src = {"seed-dup": True, "twin": True, "real": False, "llm": False}
    expected_src = {"seed-dup": "translated-dup", "twin": "twin-"}
    eligible: list[dict] = []
    for c in deduped:
        dst_row = c["row_dst"]
        if dst_row is None:
            stats[f"{c['class']}:dst-not-in-corpus"] += 1
            continue
        exp = expected_dst[c["class"]]
        if not (dst_row["source"] == exp or exp.endswith("-") and dst_row["source"].startswith(exp)):
            stats[f"{c['class']}:dst-wrong-class:{dst_row['source']}"] += 1
            continue
        if needs_src[c["class"]]:
            src_row = c["row_src"]
            if src_row is None:
                stats[f"{c['class']}:src-not-in-corpus"] += 1
                continue
            exp_s = expected_src[c["class"]]
            if not (src_row["source"] == exp_s or exp_s.endswith("-") and src_row["source"].startswith(exp_s)):
                stats[f"{c['class']}:src-wrong-class:{src_row['source']}"] += 1
                continue
        if c["burned_hit"]:
            stats[f"{c['class']}:burned-overlap"] += 1
            continue
        if c["direction"] not in ("ru->en", "en->ru"):
            stats[f"{c['class']}:bad-direction:{c['direction']}"] += 1
            continue
        eligible.append(c)
        stats[f"{c['class']}:eligible"] += 1
    for c in deduped:
        src_in = c["row_src"] is not None
        stats[f"{c['class']}:src-side-in-corpus" if src_in else f"{c['class']}:src-side-not-in-corpus"] += 1
    emit("eligibility stats: " + json.dumps(dict(sorted(stats.items())), ensure_ascii=False))

    pool_by_cell: dict[tuple[str, str], list[dict]] = {}
    for c in eligible:
        pool_by_cell.setdefault((c["class"], c["direction"]), []).append(c)
    for key in pool_by_cell:
        pool_by_cell[key].sort(key=lambda c: c["pair_key"])

    pool_total = len(eligible)
    emit(f"eligible pool: {pool_total}")
    if pool_total < BAND[0]:
        emit(f"POOL TOO SMALL ({pool_total} < {BAND[0]}): STOP, no invented pairs")
        return 1

    # ── selection rule (pre-registered; executed verbatim) ──────────────────
    # quota: floor 20 per stratum (capped by pool), remainder proportional to
    # pool size (largest remainder), total 180. Direction split: half/half per
    # stratum (extra seat to ru->en when odd), capped by per-cell pool.
    strata = sorted({c["class"] for c in eligible})
    pool_per_stratum = {s: sum(len(v) for (cls, _d), v in pool_by_cell.items() if cls == s) for s in strata}
    quota = {s: min(FLOOR_PER_STRATUM, pool_per_stratum[s]) for s in strata}
    remaining = TARGET_TOTAL - sum(quota.values())
    elig_strata = [s for s in strata if pool_per_stratum[s] > quota[s]]
    free_cap = {s: pool_per_stratum[s] - quota[s] for s in elig_strata}
    total_cap = sum(free_cap.values())
    if remaining > total_cap:
        remaining = total_cap
    base = {s: remaining * free_cap[s] // total_cap if total_cap else 0 for s in elig_strata}
    leftover = remaining - sum(base.values())
    order = sorted(elig_strata, key=lambda s: (-(free_cap[s] - base[s]) / max(free_cap[s], 1), s))
    for s in order[:leftover]:
        base[s] += 1
    for s in elig_strata:
        quota[s] += base[s]

    rng = random.Random(SEED)
    selected: list[dict] = []
    for s in strata:
        n_ru_en = quota[s] // 2 + quota[s] % 2
        n_en_ru = quota[s] // 2
        for direction, k in (("en->ru", n_en_ru), ("ru->en", n_ru_en)):
            cell = pool_by_cell.get((s, direction), [])
            k = min(k, len(cell), max(0, quota[s] - sum(1 for x in selected if x["class"] == s)))
            if k:
                selected.extend(rng.sample(cell, k))
    emit(f"selected before direction repair: {len(selected)}; "
         f"directions: {dict(Counter(c['direction'] for c in selected))}")

    # top-up if a direction fell short of balance (cell pools exhausted)
    target_half = TARGET_TOTAL // 2
    for _ in range(3):
        cnt = Counter(c["direction"] for c in selected)
        if abs(cnt["ru->en"] - cnt["en->ru"]) <= 1 and len(selected) >= BAND[0]:
            break
        for direction in ("ru->en", "en->ru"):
            other = "en->ru" if direction == "ru->en" else "ru->en"
            if cnt[direction] < target_half:
                taken = {c["pair_key"] for c in selected}
                for s in strata:
                    cell = [c for c in pool_by_cell.get((s, direction), []) if c["pair_key"] not in taken]
                    need = min(target_half - cnt[direction], len(cell))
                    if need:
                        selected.extend(rng.sample(cell, need))
                        taken.update(c["pair_key"] for c in selected[:need])
                        cnt = Counter(c["direction"] for c in selected)
    # final trim to TARGET_TOTAL, keeping direction balance (drop from the
    # over-represented direction, sorted by pair_key, rng-free)
    over = Counter(c["direction"] for c in selected)
    while len(selected) > TARGET_TOTAL:
        d = max(over, key=lambda k: (over[k], k))
        victim = max((c for c in selected if c["direction"] == d), key=lambda c: c["pair_key"])
        selected.remove(victim)
        over[d] -= 1

    selected.sort(key=lambda c: c["pair_key"])
    for i, c in enumerate(selected, 1):
        c["pair_id"] = f"R5G-{i:04d}"

    dir_counts = Counter(c["direction"] for c in selected)
    class_counts = Counter(c["class"] for c in selected)
    class_dir_counts = Counter((c["class"], c["direction"]) for c in selected)
    emit(f"FINAL: {len(selected)} pairs; classes={dict(sorted(class_counts.items()))}; "
         f"directions={dict(sorted(dir_counts.items()))}")

    # ── outputs ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_path = OUT_DIR / "gate_set.jsonl"
    with open(gate_path, "w", encoding="utf-8") as fh:
        for c in selected:
            fh.write(
                json.dumps(
                    {
                        "pair_id": c["pair_id"],
                        "source_class": c["class"],
                        "direction": c["direction"],
                        "text_src": c["text_src"],
                        "lang_src": c["lang_src"],
                        "text_dst": c["text_dst"],
                        "lang_dst": c["lang_dst"],
                        "text_sha_src": c["sha_src"],
                        "text_sha_dst": c["sha_dst"],
                        "provenance": c["provenance"],
                        "corpus_row_src": c["row_src"],
                        "corpus_row_dst": c["row_dst"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    pair_ids = sorted(c["pair_id"] for c in selected)
    ids_sha = hashlib.sha256("\n".join(pair_ids).encode("utf-8")).hexdigest()
    id_text_shas = sorted({c["sha_src"] for c in selected} | {c["sha_dst"] for c in selected})
    text_shas_sha = hashlib.sha256("\n".join(id_text_shas).encode("utf-8")).hexdigest()

    excluded_rows = []
    for c in selected:
        for side in ("row_src", "row_dst"):
            row = c[side]
            if row is not None:
                excluded_rows.append(
                    {
                        "pair_id": c["pair_id"],
                        "side": "src" if side == "row_src" else "dst",
                        "text_sha": c["sha_src" if side == "row_src" else "sha_dst"],
                        "corpus_file": row["file"],
                        "corpus_line": row["line"],
                        "source": row["source"],
                    }
                )

    selection_rule = (
        "Candidate dedup by pair_key (gen shards carry duplicate texts; first "
        "occurrence in construction order wins). Eligible pool = pairing verified "
        "(key match for real/llm; inline for seed-dup/twin) AND translation side "
        "present in the sealed r4 corpus under the expected source class (AND, for "
        "seed-dup/twin, the source side too — both sides are family rows there) "
        "AND zero overlap with the burned V43-TDUP surface. "
        f"Quota: floor {FLOOR_PER_STRATUM}/stratum (pool-capped), remainder proportional to "
        f"stratum pool (largest remainder), total {TARGET_TOTAL}; direction split half/half "
        "per stratum (extra seat ru->en when odd, cell-pool capped, shortfall top-up from "
        "any stratum, final trim drops pair_key-max from the over-represented direction). "
        f"Within each (stratum, direction) cell: candidates sorted by pair_key, "
        f"random.Random({SEED}).sample over the sorted cell."
    )
    manifest = {
        "kind": "embed-r5-gate-set-manifest",
        "created": "2026-10-08",
        "contract": "2026-10-08-vesma-cortex-embed-round5-contract.md §3.3",
        "target": "150-200 translation pairs, stratified, direction-balanced, "
        "excluded from round-5 training and from arm-B re-translation",
        "inputs": {
            "corpus_fingerprint": EXPECTED_FP,
            "corpus_verified": True,
            "corpus_rows": len(corpus),
            "gen_dir": str(GEN_DIR),
            "seed_dir": str(SEED_DIR),
            "real_part": str(REAL_PART),
            "burned_pairs_file": str(BURNED_PAIRS),
            "burned_surface": burned_info,
        },
        "counts": {
            "total_pairs": len(selected),
            "by_class": dict(sorted(class_counts.items())),
            "by_direction": dict(sorted(dir_counts.items())),
            "by_class_x_direction": {
                f"{cls}:{d}": n for (cls, d), n in sorted(class_dir_counts.items())
            },
            "candidates_before_eligibility": len(candidates),
            "dup_candidates_dropped": len(candidates) - len(deduped),
            "eligible_pool": pool_total,
            "eligibility_stats": dict(sorted(stats.items())),
            "unmatched_real_keys": n_real_nomatch,
            "unmatched_llm_keys": n_llm_nomatch,
        },
        "selection_rule": selection_rule,
        "seed": SEED,
        "excluded_strata_note": "translated-sibling excluded: record/sibling sides are "
        "not-duplicate contrast pairs, not translation pairs (batch justifications); "
        "twin-* treated as one stratum with deterministic direction parity assignment",
        "integrity": {
            "gate_set_sha256": gate_sha,
            "sorted_pair_id_list_sha256": ids_sha,
            "sorted_text_sha_list_sha256": text_shas_sha,
            "n_text_shas": len(id_text_shas),
        },
        "excluded_corpus_rows": excluded_rows,
        "excluded_text_shas_all_sides": id_text_shas,
        "asserts": {
            "pair_count_in_band": BAND[0] <= len(selected) <= BAND[1],
            "direction_balance_abs_le_1": abs(dir_counts.get("ru->en", 0) - dir_counts.get("en->ru", 0)) <= 1,
            "burned_overlap_zero": all(not c["burned_hit"] for c in selected),
            "corpus_fingerprint_ok": True,
            "pair_ids_unique": len(set(pair_ids)) == len(pair_ids),
            "pair_text_sides_unique": len(id_text_shas) == 2 * len(selected)
            and len({c["pair_key"] for c in selected}) == len(selected),
            "dst_rows_all_in_corpus": all(c["row_dst"] is not None for c in selected),
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    emit(f"gate_set.jsonl sha256={gate_sha}")
    emit(f"sorted pair_id list sha256={ids_sha}")
    emit(f"asserts: {manifest['asserts']}")
    (R5_WT / "datasets" / "embed-r5" / "run-log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(R5_WT / "datasets" / "embed-r5" / "run-log.jsonl", "a", encoding="utf-8") as fh:
        for line in log:
            fh.write(json.dumps({"ts_step": "build_gate_set", "msg": line}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
