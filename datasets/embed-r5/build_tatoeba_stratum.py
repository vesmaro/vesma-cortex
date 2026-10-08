#!/usr/bin/env python3
"""Round-5 Tatoeba stratum (contract §3.2 arm-B composition, owner-ratified
§6.3 = yes): license-clean CC-BY 2.0 FR RU-EN sentence pairs, grouped into
2-4 sentence pseudo-notes shaped like memory records, decontaminated
(n-gram 8-13, zero tolerance) against the burned 144 eval pairs, the sealed
r5 gate set, and store-derived texts (r4 real-part rows; the full r4 corpus
is checked as a stricter extra layer), plus verbatim-hash check.

Deterministic: no RNG at all — sorting only. Each Tatoeba sentence is used
in at most one note. Direction metadata (ru->en / en->ru) assigned by
sha256 parity of the note key.

Outputs (gitignored data dir):
  tatoeba_notes.jsonl   - the parallel pseudo-note pairs
  manifest is printed to stdout; numbers go to the committed manifest

License (recorded for the dataset card):
  Tatoeba sentence exports default license: Creative Commons Attribution
  2.0 France (CC-BY 2.0 FR) - https://tatoeba.org/en/terms-of-use.
  Attribution: cite Tatoeba (https://tatoeba.org/) and the per-sentence
  author usernames (carried per note and aggregated in the manifest).
"""

from __future__ import annotations

import bz2
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
DL_DIR = R5_WT / "data" / "embed-r5" / "tatoeba"
OUT_DIR = R4_WT / "data" / "embed-r4" / "r5_gate"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"
CORPUS_DIR = R4_WT / "data" / "embed-r4" / "corpus"
GATE_SET = OUT_DIR / "gate_set.jsonl"

SENTENCES = DL_DIR / "sentences_detailed.tar.bz2"
LINKS = DL_DIR / "links.tar.bz2"

TARGET_NOTES = 750
NOTE_PAIRS_MIN, NOTE_PAIRS_MAX = 2, 4
SENT_MIN, SENT_MAX = 10, 300
MAX_TOKENS = 256
GRAM_LO, GRAM_HI = 8, 13


def text_hash(text: str) -> str:
    n = unicodedata.normalize("NFC", text)
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


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


def main() -> int:
    log: list[str] = []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    # ── decontamination surfaces ────────────────────────────────────────────
    banned_grams: set[str] = set()
    burned_hashes: set[str] = set()
    for line in open(BURNED_PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if not str(r.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = r.get(sk) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    banned_grams |= ngrams(v)
                    burned_hashes.add(text_hash(v))
    emit(f"burned surface grams: {len(banned_grams)}")

    gate_hashes: set[str] = set()
    for line in open(GATE_SET, encoding="utf-8"):
        r = json.loads(line)
        for fld in ("text_src", "text_dst"):
            banned_grams |= ngrams(r[fld])
            gate_hashes.add(text_hash(r[fld]))
    emit(f"grams after gate set: {len(banned_grams)}")

    corpus_hashes: set[str] = set()
    corpus_rows = 0
    for fn in ("train.jsonl", "val.jsonl"):
        for line in open(CORPUS_DIR / fn, encoding="utf-8"):
            r = json.loads(line)
            corpus_rows += 1
            corpus_hashes.add(text_hash(r["text"]))
            if r["source"] == "real-part-r4":  # store-derived stratum
                banned_grams |= ngrams(r["text"])
    emit(f"grams after store-derived stratum: {len(banned_grams)} (corpus rows {corpus_rows})")

    # ── 1. sentences: ru + en only ──────────────────────────────────────────
    sent: dict[int, tuple[str, str, str]] = {}  # id -> (lang, text, username)
    n_lines = 0
    for line in read_bz2_lines(SENTENCES):
        parts = line.split("\t")
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
    emit(f"sentences file lines: {n_lines}; ru+en kept: {len(sent)}")

    # ── 2. direct ru-en links ───────────────────────────────────────────────
    pairs: set[tuple[int, int]] = set()
    for line in read_bz2_lines(LINKS):
        parts = line.split("\t")
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
    emit(f"direct ru-en link pairs: {len(pairs)}")

    # ── 3. quality filter + per-sentence decontam ───────────────────────────
    ok_pairs: list[tuple[int, int]] = []
    drop: Counter = Counter()
    seen_ru: set[int] = set()
    seen_en: set[int] = set()
    seen_text: set[str] = set()
    for ru_id, en_id in sorted(pairs):
        ru_lang, ru_text, ru_user = sent[ru_id]
        en_lang, en_text, en_user = sent[en_id]
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
        if ngrams(ru_text) & banned_grams or ngrams(en_text) & banned_grams:
            drop["decontam-hit"] += 1
            continue
        if text_hash(ru_text) in corpus_hashes or text_hash(en_text) in corpus_hashes:
            drop["verbatim-corpus-hit"] += 1
            continue
        seen_text.add(th)
        seen_ru.add(ru_id)
        seen_en.add(en_id)
        ok_pairs.append((ru_id, en_id))
    emit(f"quality+decontam pass pairs: {len(ok_pairs)}; drops: {dict(sorted(drop.items()))}")

    # ── 4. group into pseudo-notes (deterministic; each sentence once) ──────
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

    # ── 5. note-level gates + final decontam (concatenated text) ────────────
    final: list[dict] = []
    nfail: Counter = Counter()
    for members in notes:
        rows = []
        bad = False
        for ru_id, en_id in members:
            _l, ru_text, ru_user = sent[ru_id]
            _l2, en_text, en_user = sent[en_id]
            rows.append((ru_id, en_id, ru_text, en_text, ru_user, en_user))
        ru_join = " ".join(r[2] for r in rows)
        en_join = " ".join(r[3] for r in rows)
        if not note_ok(ru_join, en_join):
            nfail["note-gate"] += 1
            continue
        if ngrams(ru_join) & banned_grams or ngrams(en_join) & banned_grams:
            nfail["note-decontam"] += 1
            continue
        key = text_hash(ru_join + "\x00" + en_join)
        parity = int(key, 16) % 2
        final.append(
            {
                "members": rows,
                "ru": ru_join,
                "en": en_join,
                "key": key,
                "direction": "ru->en" if parity == 0 else "en->ru",
            }
        )
    emit(f"notes after note-level gates: {len(final)}; failures: {dict(sorted(nfail.items()))}")

    # ── 6. determinstic cut to target: sort by key, take first N ────────────
    final.sort(key=lambda x: x["key"])
    final = final[:TARGET_NOTES]
    emit(f"notes in stratum: {len(final)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "tatoeba_notes.jsonl"
    authors: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, n in enumerate(final, 1):
            note_id = f"R5T-{i:04d}"
            if n["direction"] == "ru->en":
                text_src, lang_src, text_dst, lang_dst = n["ru"], "ru", n["en"], "en"
            else:
                text_src, lang_src, text_dst, lang_dst = n["en"], "en", n["ru"], "ru"
            rec = {
                "note_id": note_id,
                "source_class_src": f"tatoeba-{lang_src}",
                "source_class_dst": f"tatoeba-{lang_dst}",
                "direction": n["direction"],
                "text_src": text_src,
                "lang_src": lang_src,
                "text_dst": text_dst,
                "lang_dst": lang_dst,
                "text_sha_src": text_hash(text_src),
                "text_sha_dst": text_hash(text_dst),
                "license": "CC-BY-2.0-FR",
                "tatoeba_sentence_ids": [r[0] for r in n["members"]] + [r[1] for r in n["members"]],
                "authors": sorted({r[4] for r in n["members"]} | {r[5] for r in n["members"]}),
                "provenance": "downloads.tatoeba.org/exports 2026-10-03 (sentences_detailed + links)",
            }
            for a in rec["authors"]:
                authors[a] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    dir_counts = Counter(n["direction"] for n in final)
    manifest = {
        "kind": "embed-r5-tatoeba-stratum-manifest",
        "created": "2026-10-08",
        "license": {
            "name": "Creative Commons Attribution 2.0 France (CC-BY 2.0 FR)",
            "statement": "Tatoeba text sentences default license per https://tatoeba.org/en/terms-of-use; "
            "attribution required: cite Tatoeba and the per-sentence author usernames.",
            "attribution_string": "Sentence data from Tatoeba (https://tatoeba.org), CC-BY 2.0 FR; "
            "authors: see per-note 'authors' fields and the aggregated username list in this manifest.",
            "exports_snapshot": "downloads.tatoeba.org/exports, files dated 2026-10-03",
            "audio_not_included": True,
        },
        "counts": {
            "notes": len(final),
            "rows_both_langs": len(final) * 2,
            "sentences": sum(len(n["members"]) for n in final) * 2,
            "by_direction": dict(sorted(dir_counts.items())),
            "distinct_authors": len(authors),
            "pipeline": {
                "direct_ru_en_link_pairs": len(pairs),
                "quality_decontam_pass_pairs": len(ok_pairs),
                "drops": dict(sorted(drop.items())),
                "notes_before_gates": len(notes),
                "note_gate_failures": dict(sorted(nfail.items())),
            },
        },
        "note_shape": "2-4 directly-linked ru-en sentence pairs, concatenated per side "
        "(parallel-concatenation unit, not narrative text); every Tatoeba sentence used at most once",
        "selection_rule": "deterministic, no RNG: pairs sorted by (ru_id,en_id); groups advanced in order; "
        "notes sorted by note key; first TARGET_NOTES kept",
        "decontamination": {
            "method": "n-gram 8-13 zero tolerance + verbatim hash, note-level and sentence-level",
            "surfaces": ["burned V43-TDUP 144 pairs", "r5 gate set 180 pairs",
                          "store-derived real-part-r4 rows", "full r4 corpus (stricter extra layer)"],
        },
        "integrity": {
            "tatoeba_notes_sha256": out_sha,
            "gate_set_sha_used": hashlib.sha256(GATE_SET.read_bytes()).hexdigest(),
        },
        "top_authors_by_notes": dict(authors.most_common(20)),
    }
    (OUT_DIR / "tatoeba_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    emit(f"tatoeba_notes.jsonl sha256={out_sha}")
    with open(R5_WT / "datasets" / "embed-r5" / "run-log.jsonl", "a", encoding="utf-8") as fh:
        for line in log:
            fh.write(json.dumps({"ts_step": "build_tatoeba_stratum", "msg": line}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
