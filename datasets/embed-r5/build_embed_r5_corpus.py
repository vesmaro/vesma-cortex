#!/usr/bin/env python3
"""Round-5 corpus assembly (prereg docs/specs/embed-round5-prereg.md §3).

  round-5 train = sealed round-4 train MINUS gate-set exclusions
                  PLUS Tatoeba stratum (1500 rows, both sides of 750 notes)
  round-5 val   = sealed round-4 val MINUS the same exclusions (mirror rule)

Row identity is preserved: surviving rows keep their original bytes and
order (round-4/round-5 corpora stay line-diffable); Tatoeba rows are
appended in note order (src row then dst row). Splits are NOT re-cut —
the 95/5 discipline is inherited from the sealed base.

Exclusion sources (ratified):
  manifest  r5_gate/manifest.json   gate set sha cb367577…
            315 corpus rows (file:line:source) + 360 text SHAs (both sides)
  tatoeba   r5_gate/tatoeba_notes.jsonl  sha 0be59a69…, CC-BY 2.0 FR,
            750 notes / 1500 rows, source classes tatoeba-*

Assert set (prereg §3 as amended; chair ratification 2026-10-09, Option 1):
  base fingerprint; gate/manifest/tatoeba SHA integrity; 315 exclusions
  verified row-by-row (file:line:source:sha); zero excluded keys remain;
  zero gate text-SHA matches; hash-level disjointness of the inherited
  base (gate text-SHA, burned verbatim hash); N-GRAM 8-13 ZERO-TOLERANCE
  ON NEW MATERIAL ONLY (every Tatoeba row vs gate set + burned pairs);
  Tatoeba stratum integrity, decontam and disjointness; directions; split
  discipline; uniqueness; translated-family share vs the ratified <=0.30
  ceiling; inherited-base gram overlap RECORDED as monitored numbers (not
  an assert — family relatives share store phrasing by construction);
  NEW corpus fingerprint; double-run byte-identical determinism.

Any assert failure = SystemExit(2) with numbers, NOTHING is written.

Outputs (gitignored data dir): data/embed-r5/corpus/{train,val}.jsonl +
report.json; numbers-only copy committed to datasets/embed-r5/.
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

BASE_CORPUS = R4_WT / "data" / "embed-r4" / "corpus"
R5_GATE = R4_WT / "data" / "embed-r4" / "r5_gate"
OUT_DIR = R5_WT / "data" / "embed-r5" / "corpus"
NUMBERS_DIR = R5_WT / "datasets" / "embed-r5"
RUN_LOG = NUMBERS_DIR / "run-log.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

EXPECTED_BASE_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"
EXPECTED_GATE_SHA = "cb36757769fb88f2806065f50348f5fd8383cfdb0c19c66e6335eac8ace01123"
EXPECTED_TATOEBA_SHA = "0be59a697b00e2637a94ccae2ff82543d09844e2b81414c40462588cd2cc7130"
N_EXCLUDED_ROWS = 315
N_EXCLUDED_SHAS = 360
N_GATE_PAIRS = 180
N_BURNED_PAIRS = 144
N_TATOEBA_NOTES = 750
N_TATOEBA_ROWS = 1500
TRANSLATED_FAMILY_CEILING = 0.30
GRAM_LO, GRAM_HI = 8, 13

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import text_hash  # noqa: E402  (round-3/4 canonical key)


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    out: set[str] = set()
    for n in range(GRAM_LO, GRAM_HI + 1):
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assemble() -> tuple[bytes, bytes, dict]:
    """One full deterministic pass: verify inputs, filter, merge, assert.

    Returns (train_bytes, val_bytes, numbers) — raises SystemExit(2) with
    the failing assert numbers if ANY check fails (no partial output).
    """
    # ── 1. base corpus fingerprint ──────────────────────────────────────────
    train_path, val_path = BASE_CORPUS / "train.jsonl", BASE_CORPUS / "val.jsonl"
    d = hashlib.sha256()
    d.update(train_path.read_bytes())
    d.update(val_path.read_bytes())
    base_fp = d.hexdigest()
    assert base_fp == EXPECTED_BASE_FP, f"base fp {base_fp[:12]} != {EXPECTED_BASE_FP[:12]}"

    base: dict[str, list[tuple[str, dict]]] = {"train.jsonl": [], "val.jsonl": []}
    for fn in base:
        with open(BASE_CORPUS / fn, encoding="utf-8") as fh:
            base[fn] = [(line.rstrip("\n"), json.loads(line)) for line in fh]
    n_base = {fn: len(rows) for fn, rows in base.items()}
    assert n_base == {"train.jsonl": 24221, "val.jsonl": 1275}, n_base

    # ── 2. gate manifest integrity ──────────────────────────────────────────
    manifest = json.loads((R5_GATE / "manifest.json").read_text(encoding="utf-8"))
    gate_sha = sha256_file(R5_GATE / "gate_set.jsonl")
    assert manifest["integrity"]["gate_set_sha256"] == gate_sha == EXPECTED_GATE_SHA, gate_sha[:12]
    assert manifest["inputs"]["corpus_fingerprint"] == EXPECTED_BASE_FP
    excluded_rows = manifest["excluded_corpus_rows"]
    excluded_shas = set(manifest["excluded_text_shas_all_sides"])
    assert len(excluded_rows) == N_EXCLUDED_ROWS, len(excluded_rows)
    assert len(excluded_shas) == N_EXCLUDED_SHAS, len(excluded_shas)

    # ── 3. gate set: shas + directions + n-gram surface ─────────────────────
    gate_rows = [json.loads(l) for l in open(R5_GATE / "gate_set.jsonl", encoding="utf-8")]
    assert len(gate_rows) == N_GATE_PAIRS, len(gate_rows)
    gate_shas: set[str] = set()
    gate_texts: list[str] = []
    gate_dir: Counter = Counter()
    for g in gate_rows:
        for side in ("src", "dst"):
            gate_shas.add(g[f"text_sha_{side}"])
            gate_texts.append(g[f"text_{side}"])
        gate_dir[g["direction"]] += 1
    assert gate_shas == excluded_shas, (
        f"gate sha set mismatch: recomputed {len(gate_shas)} vs manifest {len(excluded_shas)}"
    )
    gate_grams: set[str] = set()
    for t in gate_texts:
        gate_grams |= ngrams(t)

    # ── 4. burned 144 pairs surface ─────────────────────────────────────────
    burned_texts: list[str] = []
    for line in open(BURNED_PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = r.get(sk) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    burned_texts.append(v)
    n_burned = sum(
        1
        for l in open(BURNED_PAIRS, encoding="utf-8")
        if str(json.loads(l).get("pair_id", "")).startswith("V43-TDUP")
    )
    assert n_burned == N_BURNED_PAIRS, n_burned
    burned_grams: set[str] = set()
    burned_hashes: set[str] = set()
    for t in burned_texts:
        burned_grams |= ngrams(t)
        burned_hashes.add(text_hash(t))
    banned_grams = gate_grams | burned_grams

    # ── 5. verify all 315 exclusions row-by-row, then filter ────────────────
    excl_keys: set[tuple[str, int]] = set()
    excl_bad: list[str] = []
    for e in excluded_rows:
        key = (e["corpus_file"], e["corpus_line"])
        if key in excl_keys:
            excl_bad.append(f"duplicate key {key}")
            continue
        excl_keys.add(key)
        row = base[e["corpus_file"]][e["corpus_line"] - 1][1]
        if row["source"] != e["source"] or text_hash(row["text"]) != e["text_sha"]:
            excl_bad.append(f"mismatch {key}: {row['source']}/{e['source']}")
    assert not excl_bad, f"exclusion verification FAILED: {excl_bad[:5]}"
    assert len(excl_keys) == N_EXCLUDED_ROWS

    new_lines: dict[str, list[str]] = {}
    removed_by_source: Counter = Counter()
    for fn, rows in base.items():
        kept = []
        for i, (raw, row) in enumerate(rows, 1):
            if (fn, i) in excl_keys:
                removed_by_source[row["source"]] += 1
                continue
            kept.append(raw)
        new_lines[fn] = kept
    n_removed = {"train.jsonl": 24221 - len(new_lines["train.jsonl"]),
                 "val.jsonl": 1275 - len(new_lines["val.jsonl"])}

    # ── 6. Tatoeba stratum integrity + row emission ─────────────────────────
    tatoeba_sha = sha256_file(R5_GATE / "tatoeba_notes.jsonl")
    tat_manifest = json.loads((R5_GATE / "tatoeba_manifest.json").read_text(encoding="utf-8"))
    assert tat_manifest["integrity"]["tatoeba_notes_sha256"] == tatoeba_sha == EXPECTED_TATOEBA_SHA, (
        tatoeba_sha[:12]
    )
    # Provenance note (recorded, not a blocker): the stratum build decontaminated
    # against the pre-revision gate set 4a0ab11c (commit 20a52ba predates the
    # 178da43 revision to cb367577). This build RE-ASSERTS n-gram disjointness
    # of every final row (tatoeba included) against the ratified cb367577
    # surface with zero tolerance — see ngram asserts below.
    tat_manifest_gate_sha_used = tat_manifest["integrity"].get("gate_set_sha_used")
    tat_rows_raw = [json.loads(l) for l in open(R5_GATE / "tatoeba_notes.jsonl", encoding="utf-8")]
    assert len(tat_rows_raw) == N_TATOEBA_NOTES, len(tat_rows_raw)
    assert tat_manifest["counts"]["notes"] == N_TATOEBA_NOTES
    assert tat_manifest["license"]["name"].startswith("Creative Commons Attribution 2.0 France")

    tatoeba_rows: list[str] = []
    tat_dir: Counter = Counter()
    tat_hashes: set[str] = set()
    for n in tat_rows_raw:
        assert n["source_class_src"] == f"tatoeba-{n['lang_src']}", n["note_id"]
        assert n["source_class_dst"] == f"tatoeba-{n['lang_dst']}", n["note_id"]
        assert {n["lang_src"], n["lang_dst"]} == {"ru", "en"}, n["note_id"]
        assert n["direction"] in ("ru->en", "en->ru")
        if n["direction"] == "ru->en":
            assert n["lang_src"] == "ru" and n["lang_dst"] == "en"
        else:
            assert n["lang_src"] == "en" and n["lang_dst"] == "ru"
        assert n["text_sha_src"] == text_hash(n["text_src"])
        assert n["text_sha_dst"] == text_hash(n["text_dst"])
        assert n["license"] == "CC-BY-2.0-FR"
        tat_dir[n["direction"]] += 1
        for side in ("src", "dst"):
            h = text_hash(n[f"text_{side}"])
            assert h not in tat_hashes, f"tatoeba internal dup {n['note_id']}"
            tat_hashes.add(h)
            tatoeba_rows.append(
                json.dumps(
                    {"text": n[f"text_{side}"], "lang": n[f"lang_{side}"],
                     "source": n[f"source_class_{side}"]},
                    ensure_ascii=False,
                )
            )
    assert len(tatoeba_rows) == N_TATOEBA_ROWS, len(tatoeba_rows)
    assert sum(tat_dir.values()) == N_TATOEBA_NOTES

    # ── 7. ASSERTS on the filtered corpus (before merge) ────────────────────
    remaining: list[tuple[str, str, dict]] = []  # (split, hash, row)
    for fn, rows in base.items():
        for i, (_raw, row) in enumerate(rows, 1):
            if (fn, i) in excl_keys:
                continue
            remaining.append((fn.split(".")[0], text_hash(row["text"]), row))

    sha_hits = [(s, h) for s, h, _r in remaining if h in excluded_shas]
    assert not sha_hits, f"excluded keys remain: {len(sha_hits)} e.g. {sha_hits[:3]}"

    base_hashes = {h for _s, h, _r in remaining}
    tat_leak = tat_hashes & base_hashes
    assert not tat_leak, f"tatoeba verbatim hits vs corpus: {len(tat_leak)}"
    tat_banned_hits = sum(
        1
        for n in tat_rows_raw
        if ngrams(n["text_src"]) & banned_grams or ngrams(n["text_dst"]) & banned_grams
    )
    assert tat_banned_hits == 0, f"tatoeba n-gram hits vs gate+burned: {tat_banned_hits}"

    # Inherited base: hash-level disjointness (chair ratification Option 1,
    # 2026-10-09). Gram overlap is RECORDED as monitored numbers, not an
    # assert — family relatives share store phrasing by construction.
    gram_hits_gate = sum(1 for _s, _h, row in remaining if ngrams(row["text"]) & gate_grams)
    gram_hits_burned = sum(1 for _s, _h, row in remaining if ngrams(row["text"]) & burned_grams)

    burned_verbatim = sum(1 for _s, h, _r in remaining if h in burned_hashes)
    assert burned_verbatim == 0, f"burned verbatim hits: {burned_verbatim}"

    assert gate_dir == Counter({"ru->en": 90, "en->ru": 90}), dict(gate_dir)

    # ── 8. final assembly: train + tatoeba, val mirror ──────────────────────
    train_rows = new_lines["train.jsonl"] + tatoeba_rows
    val_rows = new_lines["val.jsonl"]
    train_bytes = ("\n".join(train_rows) + "\n").encode("utf-8")
    val_bytes = ("\n".join(val_rows) + "\n").encode("utf-8")

    # ── 9. post-merge asserts ───────────────────────────────────────────────
    final = [("train", h) for h in (text_hash(json.loads(l)["text"]) for l in train_rows)]
    final += [("val", h) for h in (text_hash(json.loads(l)["text"]) for l in val_rows)]
    all_h = [h for _s, h in final]
    assert len(all_h) == len(set(all_h)), "duplicate text hashes in final corpus"
    train_h = {h for s, h in final if s == "train"}
    val_h = {h for s, h in final if s == "val"}
    assert not (train_h & val_h), "train/val intersection non-empty"
    val_tat = sum(1 for l in val_rows if json.loads(l)["source"].startswith("tatoeba-"))
    assert val_tat == 0, f"tatoeba leaked into val: {val_tat} rows"
    assert len(train_rows) == 24221 - n_removed["train.jsonl"] + N_TATOEBA_ROWS
    assert len(val_rows) == 1275 - n_removed["val.jsonl"]

    # ── 10. composition ─────────────────────────────────────────────────────
    by_source = {"train": Counter(), "val": Counter()}
    for l in train_rows:
        by_source["train"][json.loads(l)["source"]] += 1
    for l in val_rows:
        by_source["val"][json.loads(l)["source"]] += 1
    total_by_source: Counter = Counter()
    for c in by_source.values():
        total_by_source.update(c)
    n_total = len(train_rows) + len(val_rows)
    fam = sum(v for k, v in total_by_source.items()
              if k.startswith("translated-") or k.startswith("twin-"))
    fam_train = sum(v for k, v in by_source["train"].items()
                    if k.startswith("translated-") or k.startswith("twin-"))
    xling = fam + total_by_source.get("tatoeba-ru", 0) + total_by_source.get("tatoeba-en", 0)
    # lang-based ru share (authoritative)
    ru_n = sum(1 for l in train_rows + val_rows if json.loads(l)["lang"] == "ru")
    fam_share = fam / n_total
    assert fam_share <= TRANSLATED_FAMILY_CEILING, f"translated-family share {fam_share:.4f} > ceiling"

    fp = hashlib.sha256()
    fp.update(train_bytes)
    fp.update(val_bytes)

    numbers = {
        "base": {"fp": base_fp, "counts": n_base},
        "gate": {"sha": gate_sha, "pairs": len(gate_rows), "by_direction": dict(sorted(gate_dir.items()))},
        "tatoeba": {
            "sha": tatoeba_sha,
            "notes": len(tat_rows_raw),
            "rows": len(tatoeba_rows),
            "by_direction": dict(sorted(tat_dir.items())),
            "license": "CC-BY-2.0-FR",
            "stratum_manifest_gate_sha_used": tat_manifest_gate_sha_used,
            "note": "stratum decontam ran against pre-revision gate 4a0ab11c (commit 20a52ba); "
            "this build re-asserted n-gram 8-13 zero-tolerance of all TATOEBA rows vs the "
            "ratified gate cb367577 + burned pairs (new-material scope, chair Option 1)",
        },
        "monitored": {
            "note": "NOT asserts — inherited base is hash-level disjoint per chair Option 1; "
            "gram overlap recorded for trend/attribution",
            "inherited_rows_sharing_gate_8_13_grams": gram_hits_gate,
            "inherited_rows_sharing_burned_8_13_grams": gram_hits_burned,
        },
        "removed": {"by_split": n_removed, "by_source": dict(sorted(removed_by_source.items())),
                    "total": sum(n_removed.values())},
        "counts": {"train": len(train_rows), "val": len(val_rows), "total": n_total},
        "by_source": {
            "train": dict(sorted(by_source["train"].items())),
            "val": dict(sorted(by_source["val"].items())),
            "total": dict(sorted(total_by_source.items())),
        },
        "shares": {
            "ru": round(ru_n / n_total, 4),
            "translated_family": round(fam_share, 4),
            "translated_family_train": round(fam_train / len(train_rows), 4),
            "cross_lingual_incl_tatoeba": round(xling / n_total, 4),
            "ceiling_translated_family": TRANSLATED_FAMILY_CEILING,
        },
        "fingerprint": fp.hexdigest(),
    }
    return train_bytes, val_bytes, numbers


def main() -> int:
    t0 = time.time()
    # Double-run determinism: two independent full passes, byte comparison.
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
        "kind": "dataset-embed-r5-corpus-report",
        "created": time.strftime("%Y-%m-%d"),
        "prereg": "docs/specs/embed-round5-prereg.md §3 (as amended: n-gram zero-tolerance "
        "on new material only; inherited base hash-level — chair ratification Option 1, 2026-10-09)",
        "assert_set": "feat/embed-r5-corpus (this commit)",
        "fingerprint": num_a["fingerprint"],
        "counts": num_a["counts"],
        "base": num_a["base"],
        "removed": num_a["removed"],
        "tatoeba": num_a["tatoeba"],
        "gate": num_a["gate"],
        "by_source": num_a["by_source"],
        "shares": num_a["shares"],
        "monitored": num_a["monitored"],
        "asserts": {
            "base_fingerprint_ok": True,
            "gate_set_sha_ok": True,
            "manifest_counts_ok": True,
            "exclusions_verified_315": True,
            "excluded_keys_absent": True,
            "gate_sha_set_consistent": True,
            "burned_pairs_144": True,
            "ngram_8_13_zero_tolerance_tatoeba_only": True,
            "inherited_base_hash_level_disjoint": True,
            "burned_verbatim_disjoint": True,
            "tatoeba_sha_ok": True,
            "tatoeba_rows_1500_notes_750": True,
            "tatoeba_decontam_zero_hit": True,
            "tatoeba_disjoint_base": True,
            "train_val_disjoint": True,
            "val_tatoeba_free": True,
            "corpus_unique_hashes": True,
            "directions_ok": True,
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
            "ts_step": "build_embed_r5_corpus",
            "msg": (
                f"round-5 corpus assembled; train={num_a['counts']['train']} "
                f"val={num_a['counts']['val']} total={num_a['counts']['total']}; "
                f"removed={num_a['removed']['total']} tatoeba=+{N_TATOEBA_ROWS}; "
                f"fp={num_a['fingerprint']}; double_run_byte_identical={identical}; "
                f"all asserts PASS"
            ),
        }, ensure_ascii=False) + "\n")

    print(json.dumps({"counts": num_a["counts"], "shares": num_a["shares"],
                      "fingerprint": num_a["fingerprint"]}, indent=1), flush=True)
    print(f"report.json + numbers-only copy written; {round(time.time() - t0, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
