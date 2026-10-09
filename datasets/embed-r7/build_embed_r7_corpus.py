#!/usr/bin/env python3
"""Round-7 corpus assembly (prereg docs/specs/embed-round7-prereg-llmquality.md §2).

  round-7 = sealed round-5 corpus (fp e64fdbea…) with EXACTLY ONE delta:
  the llm-class dst texts (old 0.6B translations) are REPLACED by the
  round-7 Qwen3-1.7B translations of the SAME source rows (pair keys
  preserved; no generation mixing). Everything else — real / seed /
  translated-real / translated-dup / translated-sibling / synthetic /
  twin-* / tatoeba — is byte-preserved in the same relative order.

Pair mapping: corpus llm row --text_hash--> round-4 ok manifest row
--key--> round-7 final translation (status ok). Rows whose pair failed
the round-7 gates after all passes DROP (counted; never silently kept).
A new dst colliding with any corpus hash (inherited or earlier
replacement) drops as `new-dup` — first-seen-wins, the round-4/5
assembly convention with inherited rows first.

Assert set (prereg §2; round-5 battery inherited):
  base fingerprint e64fdbea…; r7 manifests cover exactly the sealed 1617
  keys; replaced langs == target langs; per-pair quality gates hold on
  every collected row (digit/copy/len by construction, cos >= 0.55,
  n-gram 8-13 zero-tolerance vs gate + burned on new material);
  360 gate text-SHAs absent; the 68 gate llm pair keys never re-enter;
  burned verbatim absent; non-llm rows byte-preserved in order; tatoeba
  integrity (sha 0be59a69…, 750 notes / 1500 rows, val-free); gate set
  directions 90/90; global uniqueness; train ∩ val = ∅; translated-
  family share <= 0.30; inherited-base gram overlap RECORDED (chair
  Option 1); NEW corpus fingerprint; double-run byte-identical.

Any assert failure = SystemExit(2) with numbers, NOTHING is written.

Outputs (gitignored data dir): data/embed-r7/corpus/{train,val}.jsonl +
report.json; numbers-only copy committed to datasets/embed-r7/.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")

BASE_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R4_GEN = R4_WT / "data" / "embed-r4" / "gen"
R5_GATE = R4_WT / "data" / "embed-r4" / "r5_gate"
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
OUT_DIR = R5_WT / "data" / "embed-r7" / "corpus"
NUMBERS_DIR = R5_WT / "datasets" / "embed-r7"
RUN_LOG = NUMBERS_DIR / "run-log.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

EXPECTED_BASE_FP = "e64fdbea65f0e6a4f2830a42ea6bd0d74a2974854d381427c4d5ba2ea4901c62"
EXPECTED_R4_BASE_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"
EXPECTED_GATE_SHA = "cb36757769fb88f2806065f50348f5fd8383cfdb0c19c66e6335eac8ace01123"
EXPECTED_TATOEBA_SHA = "0be59a697b00e2637a94ccae2ff82543d09844e2b81414c40462588cd2cc7130"
N_SEALED_LLM_PAIRS = 1617
N_GATE_PAIRS = 180
N_BURNED_PAIRS = 144
N_TATOEBA_NOTES = 750
N_TATOEBA_ROWS = 1500
TRANSLATED_FAMILY_CEILING = 0.30
GRAM_LO, GRAM_HI = 8, 13
MIN_COS = 0.55

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import (  # noqa: E402
    load_jsonl,
    text_hash,
)


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    out: set[str] = set()
    for n in range(GRAM_LO, GRAM_HI + 1):
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def final_r7_status() -> tuple[dict[str, dict], Counter]:
    """key -> final row (last pass wins) + attempt-level gate counters."""
    latest: dict[str, dict] = {}
    attempts: Counter = Counter()
    for p in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for r in load_jsonl(p):
            attempts["attempts"] += 1
            if r.get("status") != "ok":
                for reason in str(r.get("reason") or "unknown").split(";"):
                    attempts[f"gate-fail:{reason.split(':')[0]}"] += 1
            latest[r["key"]] = r
    return latest, attempts


def assemble() -> tuple[bytes, bytes, dict]:
    # ── 1. base corpus fingerprint (sealed round-5) ─────────────────────────
    d = hashlib.sha256()
    d.update((BASE_CORPUS / "train.jsonl").read_bytes())
    d.update((BASE_CORPUS / "val.jsonl").read_bytes())
    base_fp = d.hexdigest()
    assert base_fp == EXPECTED_BASE_FP, f"base fp {base_fp[:12]} != {EXPECTED_BASE_FP[:12]}"

    base: dict[str, list[tuple[str, dict]]] = {}
    for fn in ("train.jsonl", "val.jsonl"):
        with open(BASE_CORPUS / fn, encoding="utf-8") as fh:
            base[fn] = [(line.rstrip("\n"), json.loads(line)) for line in fh]
    n_base = {fn: len(rows) for fn, rows in base.items()}
    assert n_base == {"train.jsonl": 25420, "val.jsonl": 1261}, n_base

    # ── 2. pair-key mapping through the sealed round-4 ok manifest ──────────
    r4_ok = [r for p in sorted(R4_GEN.glob("llm-translations.shard*.jsonl"))
             for r in load_jsonl(p) if r.get("status") == "ok"]
    assert len(r4_ok) == N_SEALED_LLM_PAIRS, len(r4_ok)
    oldhash_to_key: dict[str, str] = {}
    key_to_lang: dict[str, str] = {}
    for r in r4_ok:
        h = text_hash(r["text"])
        assert h not in oldhash_to_key, "round-4 ok texts not unique"
        oldhash_to_key[h] = r["key"]
        key_to_lang[r["key"]] = r["target_lang"]

    # ── 3. gate set / manifest integrity + banned surface ───────────────────
    gate_manifest = json.loads((R5_GATE / "manifest.json").read_text(encoding="utf-8"))
    gate_sha = sha256_file(R5_GATE / "gate_set.jsonl")
    assert gate_manifest["integrity"]["gate_set_sha256"] == gate_sha == EXPECTED_GATE_SHA
    assert gate_manifest["inputs"]["corpus_fingerprint"] == EXPECTED_R4_BASE_FP
    excluded_shas = set(gate_manifest["excluded_text_shas_all_sides"])
    assert len(gate_manifest["excluded_corpus_rows"]) == 315
    assert len(excluded_shas) == 360, len(excluded_shas)

    gate_rows = [json.loads(line) for line in open(R5_GATE / "gate_set.jsonl", encoding="utf-8")]
    assert len(gate_rows) == N_GATE_PAIRS
    gate_dir = Counter(g["direction"] for g in gate_rows)
    assert gate_dir == Counter({"ru->en": 90, "en->ru": 90}), dict(gate_dir)
    gate_llm_keys = {g["provenance"]["pair_key"] for g in gate_rows
                     if g["source_class"] == "llm"}
    assert len(gate_llm_keys) == 68, len(gate_llm_keys)

    banned_grams: set[str] = set()
    burned_hashes: set[str] = set()
    for g in gate_rows:
        banned_grams |= ngrams(g["text_src"]) | ngrams(g["text_dst"])
    n_burned = 0
    for line in open(BURNED_PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        n_burned += 1
        for sk in ("record", "candidate"):
            side = r.get(sk) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    banned_grams |= ngrams(v)
                    burned_hashes.add(text_hash(v))
    assert n_burned == N_BURNED_PAIRS, n_burned

    # ── 4. rewrite llm rows in place; drop gate-failed / colliding pairs ────
    final, attempts = final_r7_status()
    assert len(final) == N_SEALED_LLM_PAIRS, f"r7 final covers {len(final)} keys"

    new_lines: dict[str, list[str]] = {}
    non_llm_kept: dict[str, list[str]] = {}
    outcome: Counter = Counter()
    n_llm_rows = 0
    n_replaced = {"train.jsonl": 0, "val.jsonl": 0}
    cos_values: list[float] = []
    seen_hashes: set[str] = set()  # inherited + accepted replacements, in order
    gate_key_hit: list[str] = []

    for fn, rows in base.items():
        out: list[str] = []
        kept_non_llm: list[str] = []
        for raw, row in rows:
            if row["source"] != "translated-llm-r4":
                out.append(raw)
                kept_non_llm.append(raw)
                seen_hashes.add(text_hash(row["text"]))
                continue
            n_llm_rows += 1
            key = oldhash_to_key.get(text_hash(row["text"]))
            assert key is not None, f"llm row not in sealed manifest: {row['text'][:50]}"
            if key in gate_llm_keys:
                gate_key_hit.append(key)
                continue
            r7 = final[key]
            assert key_to_lang[key] == r7["target_lang"] == row["lang"], f"lang mismatch {key}"
            if r7["status"] != "ok":
                outcome[f"dropped:{r7['status']}"] += 1
                continue
            assert float(r7["cos"]) >= MIN_COS, f"cos below gate slipped through: {key}"
            assert not (ngrams(r7["text"]) & banned_grams), f"ngram hit slipped through: {key}"
            h_new = text_hash(r7["text"])
            if h_new in excluded_shas:
                outcome["dropped:gate-sha-collision"] += 1
                continue
            if h_new in seen_hashes:
                outcome["dropped:new-dup"] += 1
                continue
            seen_hashes.add(h_new)
            cos_values.append(float(r7["cos"]))
            out.append(json.dumps(
                {"text": r7["text"], "lang": row["lang"], "source": "translated-llm-r4"},
                ensure_ascii=False,
            ))
            n_replaced[fn] += 1
        new_lines[fn] = out
        non_llm_kept[fn] = kept_non_llm
    assert not gate_key_hit, f"gate llm keys present in base corpus: {gate_key_hit[:3]}"
    assert n_llm_rows == 1549, n_llm_rows

    # ── 5. byte-preservation of every non-llm row (order included) ──────────
    for fn, rows in base.items():
        old_non_llm = [raw for raw, row in rows if row["source"] != "translated-llm-r4"]
        assert non_llm_kept[fn] == old_non_llm, f"non-llm bytes drifted in {fn}"

    # ── 6. exclusion + tatoeba + inherited asserts (round-5 battery) ────────
    tat_manifest = json.loads((R5_GATE / "tatoeba_manifest.json").read_text(encoding="utf-8"))
    tatoeba_sha = sha256_file(R5_GATE / "tatoeba_notes.jsonl")
    assert tat_manifest["integrity"]["tatoeba_notes_sha256"] == tatoeba_sha == EXPECTED_TATOEBA_SHA
    tat_rows_raw = [json.loads(line) for line in open(R5_GATE / "tatoeba_notes.jsonl", encoding="utf-8")]
    assert len(tat_rows_raw) == N_TATOEBA_NOTES
    tat_hashes = {text_hash(n[f"text_{s}"]) for n in tat_rows_raw for s in ("src", "dst")}
    assert len(tat_hashes) == N_TATOEBA_ROWS

    final_rows: list[tuple[str, dict]] = [
        (split, json.loads(raw))
        for split, fn in (("train", "train.jsonl"), ("val", "val.jsonl"))
        for raw in new_lines[fn]
    ]
    all_h = [text_hash(row["text"]) for _s, row in final_rows]
    assert len(all_h) == len(set(all_h)), "duplicate text hashes in final corpus"
    sha_hits = [h for h in all_h if h in excluded_shas]
    assert not sha_hits, f"gate-excluded shas present: {len(sha_hits)}"
    burned_verbatim = sum(1 for h in all_h if h in burned_hashes)
    assert burned_verbatim == 0, f"burned verbatim hits: {burned_verbatim}"
    train_h = {text_hash(row["text"]) for s, row in final_rows if s == "train"}
    val_h = set(all_h) - train_h
    assert not (train_h & val_h), "train/val intersection non-empty"
    assert tat_hashes <= train_h, "tatoeba rows lost from train"
    val_src = Counter(row["source"] for s, row in final_rows if s == "val")
    assert not any(src.startswith("tatoeba-") for src in val_src), "tatoeba leaked into val"

    # monitored (NOT asserts — chair Option 1): inherited gram overlap
    gate_grams_only: set[str] = set()
    for g in gate_rows:
        gate_grams_only |= ngrams(g["text_src"]) | ngrams(g["text_dst"])
    burned_grams_only = banned_grams - gate_grams_only
    gram_hits_gate = gram_hits_burned = 0
    for _s, row in final_rows:
        if row["source"] == "translated-llm-r4":
            continue  # new material — zero-tolerance asserted above
        g = ngrams(row["text"])
        gram_hits_gate += bool(g & gate_grams_only)
        gram_hits_burned += bool(g & burned_grams_only)

    # ── 7. composition + shares + fingerprint ───────────────────────────────
    by_source = {
        "train": Counter(json.loads(r)["source"] for r in new_lines["train.jsonl"]),
        "val": Counter(json.loads(r)["source"] for r in new_lines["val.jsonl"]),
    }
    langs = Counter(row["lang"] for _s, row in final_rows)
    total_by_source: Counter = Counter()
    for c in by_source.values():
        total_by_source.update(c)
    n_total = len(new_lines["train.jsonl"]) + len(new_lines["val.jsonl"])
    fam = sum(v for k, v in total_by_source.items()
              if k.startswith("translated-") or k.startswith("twin-"))
    fam_share = fam / n_total
    assert fam_share <= TRANSLATED_FAMILY_CEILING, f"translated-family share {fam_share:.4f}"

    train_bytes = ("\n".join(new_lines["train.jsonl"]) + "\n").encode("utf-8")
    val_bytes = ("\n".join(new_lines["val.jsonl"]) + "\n").encode("utf-8")
    fp = hashlib.sha256()
    fp.update(train_bytes)
    fp.update(val_bytes)

    cos_sorted = sorted(cos_values)
    n_cos = len(cos_sorted)

    numbers = {
        "base": {"fp": base_fp, "counts": n_base},
        "gate": {"sha": gate_sha, "pairs": N_GATE_PAIRS,
                 "by_direction": dict(sorted(gate_dir.items())),
                 "llm_stratum_pair_keys_preserved": True},
        "tatoeba": {"sha": tatoeba_sha, "notes": N_TATOEBA_NOTES,
                    "rows": N_TATOEBA_ROWS, "license": "CC-BY-2.0-FR"},
        "delta": {
            "kind": "llm-class re-translation (round-7 single variable)",
            "teacher": "Qwen/Qwen3-1.7B-GGUF Q8_0 (llama.cpp)",
            "old_teacher": "Qwen/Qwen3-0.6B-GGUF Q8_0 (round-4)",
            "sealed_pairs": N_SEALED_LLM_PAIRS,
            "llm_rows_in_base": n_llm_rows,
            "replaced": sum(n_replaced.values()),
            "replaced_by_split": dict(sorted(n_replaced.items())),
            "outcomes": dict(sorted(outcome.items())),
            "generation_attempts": dict(sorted(attempts.items())),
        },
        "removed": {
            "by_split": {fn: n_base[fn] - len(new_lines[fn]) for fn in n_base},
            "total": sum(v for k, v in outcome.items() if k.startswith("dropped:")),
            "note": "round-5 exclusions already applied in the sealed base; "
                    "round-7 removals are gate-failed llm pairs only",
        },
        "counts": {"train": len(new_lines["train.jsonl"]),
                   "val": len(new_lines["val.jsonl"]), "total": n_total},
        "by_source": {k: dict(sorted(v.items())) for k, v in by_source.items()},
        "shares": {
            "ru": round(langs["ru"] / n_total, 4),
            "translated_family": round(fam_share, 4),
            "translated_family_train": round(
                sum(v for k, v in by_source["train"].items()
                    if k.startswith("translated-") or k.startswith("twin-"))
                / len(new_lines["train.jsonl"]), 4),
            "llm_class": round(total_by_source.get("translated-llm-r4", 0) / n_total, 4),
            "ceiling_translated_family": TRANSLATED_FAMILY_CEILING,
        },
        "quality_gates": {
            "digit_preservation": "pass by construction on collected rows (lost-run gate)",
            "teacher_cos_min": MIN_COS,
            "cos_collected": n_cos,
            "cos_min": round(min(cos_sorted), 4) if cos_sorted else None,
            "cos_p10": round(cos_sorted[max(0, n_cos // 10)], 4) if cos_sorted else None,
            "cos_median": round(cos_sorted[max(0, n_cos // 2)], 4) if cos_sorted else None,
            "anti_copy": "pass by construction on collected rows (hash + jaccard 0.9)",
            "ngram_8_13_zero_tolerance_new_material": True,
            "final_ok": sum(n_replaced.values()),
        },
        "monitored": {
            "note": "NOT asserts — inherited base is hash-level disjoint per chair "
                    "Option 1; gram overlap recorded for trend/attribution",
            "inherited_rows_sharing_gate_8_13_grams": gram_hits_gate,
            "inherited_rows_sharing_burned_8_13_grams": gram_hits_burned,
        },
        "fingerprint": fp.hexdigest(),
    }
    return train_bytes, val_bytes, numbers


def main() -> int:
    t0 = time.time()
    train_a, val_a, num_a = assemble()
    train_b, val_b, num_b = assemble()
    identical = (train_a, val_a) == (train_b, val_b)
    assert identical, "DOUBLE-RUN DIVERGENCE: run A != run B"
    assert num_a["fingerprint"] == num_b["fingerprint"]
    print(f"double-run byte-identical: True; fp {num_a['fingerprint'][:12]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.jsonl").write_bytes(train_a)
    (OUT_DIR / "val.jsonl").write_bytes(val_a)

    report = {
        "kind": "dataset-embed-r7-corpus-report",
        "created": time.strftime("%Y-%m-%d"),
        "prereg": "docs/specs/embed-round7-prereg-llmquality.md §2 (ACTIVATED, "
                  "owner-ratified 2026-10-10)",
        "assert_set": "feat/embed-r7-corpus (this commit)",
        "fingerprint": num_a["fingerprint"],
        "counts": num_a["counts"],
        "base": num_a["base"],
        "gate": num_a["gate"],
        "tatoeba": num_a["tatoeba"],
        "delta": num_a["delta"],
        "removed": num_a["removed"],
        "by_source": num_a["by_source"],
        "shares": num_a["shares"],
        "quality_gates": num_a["quality_gates"],
        "monitored": num_a["monitored"],
        "asserts": {
            "base_fingerprint_ok": True,
            "r7_manifests_cover_1617_sealed_keys": True,
            "pair_identity_preserved": True,
            "no_generation_mixing": True,
            "non_llm_rows_byte_preserved_in_order": True,
            "excluded_keys_absent_360_shas": True,
            "gate_llm_keys_absent": True,
            "burned_verbatim_disjoint": True,
            "tatoeba_sha_ok_and_intact": True,
            "corpus_unique_hashes": True,
            "train_val_disjoint": True,
            "gate_directions_90_90": True,
            "digit_preservation_all_rows": True,
            "teacher_cos_ge_055_all_rows": True,
            "anti_copy_all_rows": True,
            "ngram_8_13_zero_tolerance_new_material": True,
            "translated_family_share_le_030": True,
            "double_run_byte_identical": identical,
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (NUMBERS_DIR / "corpus_report.numbers-only.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts_step": "build_embed_r7_corpus",
            "msg": (
                f"round-7 corpus assembled; train={num_a['counts']['train']} "
                f"val={num_a['counts']['val']} total={num_a['counts']['total']}; "
                f"replaced={num_a['delta']['replaced']}/{N_SEALED_LLM_PAIRS} llm pairs; "
                f"fp={num_a['fingerprint']}; double_run_byte_identical={identical}; "
                f"all asserts PASS"
            ),
        }, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": num_a["counts"], "shares": num_a["shares"],
                      "delta_outcomes": num_a["delta"]["outcomes"],
                      "fingerprint": num_a["fingerprint"]}, indent=1), flush=True)
    print(f"report.json + numbers-only copy written; {round(time.time() - t0, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
