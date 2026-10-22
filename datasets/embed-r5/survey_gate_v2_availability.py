#!/usr/bin/env python3
"""Phase-0a availability survey for gate-set v2 (ADDENDUM 1, TL 2026-10-22).

Recounts the clean Tatoeba pseudo-note pool for the v2 gate material
against EVERY training surface (expanded ban set), then excludes the 750
notes already trained inside the r5/r7 corpora. READ-ONLY over every
input; numbers + drop-reason counters only — no record text is printed.

Pipeline replication discipline:
  1. The original tatoeba builder (datasets/embed-r5/build_tatoeba_stratum.py)
     is replicated EXACTLY on its ORIGINAL ban set (burned V43-TDUP +
     sealed cb367577 gate set + r4 real-part-r4 corpus rows + full r4
     corpus) — asserts reproduce the committed manifest counts
     (817178 quality-pass pairs, 123534 notes after gates, 750 stratum).
  2. The EXPANDED ban set (all training surfaces) is then applied to
     notes[750:] (the trained 750 are the sorted-key head and are
     excluded by exact key match with the sealed tatoeba_notes.jsonl).

Expanded ban set (hash-level + n-gram 8-13 zero tolerance):
  r4 corpus train+val, r5 corpus train+val, r7 corpus train+val,
  r4 gen shards (synth-mono, synth-template-mono, llm-translations,
  real-translations, synth-twins), r7 gen llm-translations-r7 shards,
  v43 seed batch-translated-{dup,sibling}-{en,ru} (all side dicts),
  real-part.jsonl (title+body), burned V43-TDUP stage2 train
  (record/candidate title+body), sealed cb367577 gate set (360 sides;
  label-noise pairs R5G-0106/0141/0163 ride inside and are counted).

Output: counters only, to stdout (numbers go to the availability
manifest; no text leaves this script).
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")

DL_DIR = R5_WT / "data" / "embed-r5" / "tatoeba"
SENTENCES = DL_DIR / "sentences_detailed.tar.bz2"
LINKS = DL_DIR / "links.tar.bz2"
TRAINED_NOTES = R4_WT / "data" / "embed-r4" / "r5_gate" / "tatoeba_notes.jsonl"

GATE_SET = R4_WT / "data" / "embed-r4" / "r5_gate" / "gate_set.jsonl"
R4_CORPUS = R4_WT / "data" / "embed-r4" / "corpus"
R4_GEN = R4_WT / "data" / "embed-r4" / "gen"
R5_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R7_CORPUS = R5_WT / "data" / "embed-r7" / "corpus"
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import text_hash  # noqa: E402 — round-3 convention

GRAM_LO, GRAM_HI = 8, 13
NOTE_PAIRS_MIN, NOTE_PAIRS_MAX = 2, 4
SENT_MIN, SENT_MAX = 10, 300
MAX_TOKENS = 256
HEAD_N = 3000  # clean-head verification depth (usable-pool estimate)

_expect = {
    "quality_pass_pairs": 817178,
    "notes_after_gates": 123534,
    "stratum_notes": 750,
}


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    out: set[str] = set()
    for n in range(GRAM_LO, GRAM_HI + 1):
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def count_tokens(text: str) -> int:
    import re

    return len(re.findall(r"\w+", text)) + len(re.findall(r"[^\w\s]", text))


def sent_ok(text: str) -> bool:
    if not (SENT_MIN <= len(text.strip()) <= SENT_MAX):
        return False
    low = text.lower()
    if "http" in low or "www." in low or "@" in low or "<" in low:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 0.6 * len(text.strip()):
        return False
    return True


def note_ok(ru: str, en: str) -> bool:
    for t in (ru, en):
        if not (40 <= len(t) <= 4000):
            return False
        if count_tokens(t) > MAX_TOKENS:
            return False
    if text_hash(ru) == text_hash(en):
        return False
    return True


def read_bz2_lines(path: Path):
    import tarfile

    with tarfile.open(path, mode="r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            for raw in fh:
                yield raw.decode("utf-8").rstrip("\n")


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class Ban:
    """Hash-level + n-gram ban set with per-surface counters."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.grams: set[str] = set()
        self.sides: Counter = Counter()

    def add(self, text: object, surface: str) -> None:
        t = str(text or "").strip()
        if not t:
            return
        self.hashes.add(text_hash(t))
        self.grams |= ngrams(t)
        self.sides[surface] += 1

    def add_side_dict(self, side: object, surface: str) -> None:
        if not isinstance(side, dict):
            return
        for fld in ("title", "body", "text"):
            self.add(side.get(fld), surface)


def main() -> int:
    log: list[str] = []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    # ── 1. ORIGINAL ban set (exact replication of the tatoeba builder) ─────
    orig = Ban()
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            orig.add_side_dict(line.get(sk), "burned")
    for line in load_jsonl(GATE_SET):
        orig.add(line.get("text_src"), "gate")
        orig.add(line.get("text_dst"), "gate")
    r4_realpart_rows = 0
    for fn in ("train.jsonl", "val.jsonl"):
        for line in load_jsonl(R4_CORPUS / fn):
            if line["source"] == "real-part-r4":
                orig.add(line["text"], "r4-realpart-rows")
                r4_realpart_rows += 1
            orig.hashes.add(text_hash(line["text"]))  # corpus hash layer
    emit(f"original ban set: grams={len(orig.grams)} hashes={len(orig.hashes)} sides={dict(orig.sides)}")

    # ── 2. Tatoeba pipeline, exact replication ─────────────────────────────
    sent: dict[int, tuple[str, str, str]] = {}
    n_lines = 0
    for line_txt in read_bz2_lines(SENTENCES):
        parts = line_txt.split("\t")
        if len(parts) < 4:
            continue
        n_lines += 1
        sid, lang, text, user = parts[0], parts[1], parts[2], parts[3]
        if lang == "rus":
            lang = "ru"
        elif lang == "eng":
            lang = "en"
        else:
            continue
        sent[int(sid)] = (lang, text, user)
    emit(f"sentences lines={n_lines} ru+en={len(sent)}")

    pairs: set[tuple[int, int]] = set()
    for line_txt in read_bz2_lines(LINKS):
        parts = line_txt.split("\t")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a.isdigit() or not b.isdigit():
            continue
        ia, ib = int(a), int(b)
        sa, sb = sent.get(ia), sent.get(ib)
        if not sa or not sb or sa[0] == sb[0]:
            continue
        ru_id, en_id = (ia, ib) if sa[0] == "ru" else (ib, ia)
        pairs.add((ru_id, en_id))
    emit(f"direct ru-en pairs: {len(pairs)}")

    ok_pairs: list[tuple[int, int]] = []
    drop: Counter = Counter()
    seen_text: set[str] = set()
    for ru_id, en_id in sorted(pairs):
        ru_text, en_text = sent[ru_id][1], sent[en_id][1]
        if not sent_ok(ru_text):
            drop["ru:gate"] += 1
            continue
        if not sent_ok(en_text):
            drop["en:gate"] += 1
            continue
        if text_hash(ru_text) == text_hash(en_text):
            drop["copy"] += 1
            continue
        th = text_hash(ru_text + "\x00" + en_text)
        if th in seen_text:
            drop["dup-pair"] += 1
            continue
        if ngrams(ru_text) & orig.grams or ngrams(en_text) & orig.grams:
            drop["decontam-hit"] += 1
            continue
        if text_hash(ru_text) in orig.hashes or text_hash(en_text) in orig.hashes:
            drop["verbatim-corpus-hit"] += 1
            continue
        seen_text.add(th)
        ok_pairs.append((ru_id, en_id))
    emit(f"quality+decontam pass pairs: {len(ok_pairs)} drops={dict(sorted(drop.items()))}")
    assert len(ok_pairs) == _expect["quality_pass_pairs"], (
        f"REPLICATION FAIL: {len(ok_pairs)} != {_expect['quality_pass_pairs']}"
    )

    used: set[int] = set()
    notes: list[list[tuple[int, int]]] = []
    buf: list[tuple[int, int]] = []
    for ru_id, en_id in ok_pairs:
        if ru_id in used or en_id in used:
            if len(buf) >= NOTE_PAIRS_MIN:
                notes.append(buf)
            buf = []
            continue
        buf.append((ru_id, en_id))
        used.add(ru_id)
        used.add(en_id)
        if len(buf) == NOTE_PAIRS_MAX:
            notes.append(buf)
            buf = []
    if len(buf) >= NOTE_PAIRS_MIN:
        notes.append(buf)
    emit(f"notes assembled: {len(notes)}")

    final: list[dict] = []
    nfail: Counter = Counter()
    for members in notes:
        rows = [(ri, ei, sent[ri][1], sent[ei][1]) for ri, ei in members]
        ru_join = " ".join(r[2] for r in rows)
        en_join = " ".join(r[3] for r in rows)
        if not note_ok(ru_join, en_join):
            nfail["note-gate"] += 1
            continue
        if ngrams(ru_join) & orig.grams or ngrams(en_join) & orig.grams:
            nfail["note-decontam"] += 1
            continue
        key = text_hash(ru_join + "\x00" + en_join)
        final.append({"members": rows, "ru": ru_join, "en": en_join, "key": key})
    emit(f"notes after original gates: {len(final)} failures={dict(sorted(nfail.items()))}")
    assert len(final) == _expect["notes_after_gates"], (
        f"REPLICATION FAIL: {len(final)} != {_expect['notes_after_gates']}"
    )

    final.sort(key=lambda x: x["key"])
    stratum = final[: _expect["stratum_notes"]]
    # tatoeba_notes.jsonl carries sides, not the internal sort key —
    # recompute it (ru side is text_src iff direction == ru->en)
    trained_keys = set()
    for row in load_jsonl(TRAINED_NOTES):
        ru = row["text_src"] if row["direction"] == "ru->en" else row["text_dst"]
        en = row["text_dst"] if row["direction"] == "ru->en" else row["text_src"]
        trained_keys.add(text_hash(ru + "\x00" + en))
    stratum_keys = {n["key"] for n in stratum}
    assert stratum_keys == trained_keys, "REPLICATION FAIL: stratum keys != trained notes"
    emit("replication asserts: PASS (pairs, notes, stratum-750 keys byte-exact)")

    # ── 3. EXPANDED ban set over every training surface ────────────────────
    full = Ban()
    # corpora (r5/r7 include tatoeba-750, real-part-r4, gen families)
    for tag, cdir in (("r4", R4_CORPUS), ("r5", R5_CORPUS), ("r7", R7_CORPUS)):
        for fn in ("train.jsonl", "val.jsonl"):
            for line in load_jsonl(cdir / fn):
                full.add(line["text"], f"{tag}-corpus")
    # r4 gen shards
    for line in load_jsonl(R4_GEN / "synth-template-mono.jsonl"):
        full.add(line.get("text"), "r4-gen-template")
    for pat, flds in (
        ("synth-mono.shard*.jsonl", ("text",)),
        ("llm-translations.shard*.jsonl", ("text",)),
        ("real-translations.shard*.jsonl", ("text",)),
        ("synth-twins.shard*.jsonl", ("ru", "en")),
    ):
        for path in sorted(R4_GEN.glob(pat)):
            for line in load_jsonl(path):
                for fld in flds:
                    full.add(line.get(fld), "r4-gen-" + pat.split(".")[0])
    # r7 gen
    for path in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for line in load_jsonl(path):
            full.add(line.get("text"), "r7-gen")
    # v43 seed datasets (dup + sibling, every side dict)
    for path in sorted(SEED_DIR.glob("batch-translated-*.jsonl")):
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, dict):
                    full.add_side_dict(v, "v43-seed")
    # real part export
    for line in load_jsonl(REAL_PART):
        full.add_side_dict(line.get("side"), "real-part")
    # burned surface + gate set (idempotent adds, counted again)
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            full.add_side_dict(line.get(sk), "burned")
    label_noise_shas: set[str] = set()
    n_gate_sides = 0
    for line in load_jsonl(GATE_SET):
        full.add(line.get("text_src"), "gate-v1")
        full.add(line.get("text_dst"), "gate-v1")
        n_gate_sides += 2
        if line["pair_id"] in ("R5G-0106", "R5G-0141", "R5G-0163"):
            label_noise_shas.update((line["text_sha_src"], line["text_sha_dst"]))
    emit(f"expanded ban set: grams={len(full.grams)} hashes={len(full.hashes)} sides={dict(sorted(full.sides.items()))}")
    emit(f"gate-v1 sides counted={n_gate_sides} label-noise shas={sorted(label_noise_shas)}")

    # ── 4. expanded decontam over notes[750:] ──────────────────────────────
    pool = [n for n in final if n["key"] not in trained_keys]
    emit(f"pool before expanded decontam: {len(pool)}")
    clean: list[dict] = []
    drops2: Counter = Counter()
    for n in pool:
        # sentence-level (same layer the original applied per sentence)
        hit = False
        for _ri, _ei, ru_t, en_t in n["members"]:
            if ngrams(ru_t) & full.grams or ngrams(en_t) & full.grams:
                drops2["sent-ngram"] += 1
                hit = True
                break
            if text_hash(ru_t) in full.hashes or text_hash(en_t) in full.hashes:
                drops2["sent-hash"] += 1
                hit = True
                break
        if hit:
            continue
        if ngrams(n["ru"]) & full.grams or ngrams(n["en"]) & full.grams:
            drops2["note-ngram"] += 1
            continue
        if text_hash(n["ru"]) in full.hashes or text_hash(n["en"]) in full.hashes:
            drops2["note-hash"] += 1
            continue
        n["direction"] = "ru->en" if int(n["key"], 16) % 2 == 0 else "en->ru"
        clean.append(n)
    emit(f"CLEAN POOL: {len(clean)} notes; drops={dict(sorted(drops2.items()))}")
    dirs = Counter(n["direction"] for n in clean)
    emit(f"clean pool by direction: {dict(sorted(dirs.items()))}")

    head = clean[:HEAD_N]
    head_dirs = Counter(n["direction"] for n in head)
    sent_counts = Counter(len(n["members"]) for n in head)
    emit(f"clean head first {HEAD_N}: directions={dict(sorted(head_dirs.items()))} "
         f"sentence-groups={dict(sorted(sent_counts.items()))}")

    # allocation dry-run: llm head-cut then seed-dup next (key order)
    llm_take, seed_take = 180, 60
    llm_pool = head[:llm_take]
    seed_pool = head[llm_take : llm_take + seed_take]
    emit(f"allocation dry-run: llm_pool={len(llm_pool)} {dict(Counter(n['direction'] for n in llm_pool))}; "
         f"seed_pool={len(seed_pool)} {dict(Counter(n['direction'] for n in seed_pool))}")

    avail = {
        "kind": "embed-r5-gate-v2-availability-survey",
        "pool_total_clean": len(clean),
        "pool_drops_expanded": dict(sorted(drops2.items())),
        "pool_by_direction": dict(sorted(dirs.items())),
        "clean_head": {"n": HEAD_N, "directions": dict(sorted(head_dirs.items()))},
        "ban_set_sides": dict(sorted(full.sides.items())),
        "ban_set_grams": len(full.grams),
        "ban_set_hashes": len(full.hashes),
        "label_noise_shas": sorted(label_noise_shas),
    }
    print("AVAIL_JSON=" + json.dumps(avail, ensure_ascii=False))
    out = R5_WT / "datasets" / "embed-r5" / "run-log.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        for line in log:
            fh.write(json.dumps({"ts_step": "survey_gate_v2_availability", "msg": line}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
