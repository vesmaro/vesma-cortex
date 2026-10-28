#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 5a: PA-Q1 fresh-material sources.

Ratified recipe lineage: gate-set v2 (r7-confirm ADDENDUM 1, TL 2026-10-22)
constructors, adapted to the A-retrial ban set. Stages:

  census — (1) replicate the tatoeba pipeline EXACTLY (asserts 817178 /
           123534 / 750 byte-exact), apply the EXPANDED ban set + the
           A-retrial additions, count the clean llm-note pool;
           (2) census post-ceiling clean store records for the real
           stratum (created_at > real-part assembly ceiling, non-agent
           material, hash+n-gram clean, outside dup-territory / anchor-68 /
           masked-endpoint / frozen distractor texts);
  cut    — deterministically freeze sources.jsonl (llm notes, real records,
           twin job plan) with the declared A-retrial seed.

K1 verdict per stratum is computed from the census numbers against the
ratified STOP minimums (llm/real/twin >= 60 per set, sets alpha and beta
independent in PA-Q1 strata).

Privacy: record texts never leave the gitignored data dir; the console
carries counters only.
"""

from __future__ import annotations

import json
import pickle
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
B2_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/b2-graph-corpus")

sys.path.insert(0, str(R4_WT / "scripts"))
sys.path.insert(0, str(WORKTREE / "datasets" / "embed-r5"))
from gen_embed_r4_texts import text_hash  # noqa: E402  (round-3 convention)

OUT = WORKTREE / "data" / "a-retrial" / "gate"
HYG = WORKTREE / "data" / "a-retrial" / "hygiene"
CLS = WORKTREE / "data" / "a-retrial" / "classes"
MASK = WORKTREE / "data" / "a-retrial" / "mask"

DL_DIR = R5_WT / "data" / "embed-r5" / "tatoeba"
SENTENCES = DL_DIR / "sentences_detailed.tar.bz2"
LINKS = DL_DIR / "links.tar.bz2"
R4_CORPUS = R4_WT / "data" / "embed-r4" / "corpus"
R4_GEN = R4_WT / "data" / "embed-r4" / "gen"
R5_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R7_CORPUS = R5_WT / "data" / "embed-r7" / "corpus"
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"
GATE_V1 = R4_WT / "data" / "embed-r4" / "r5_gate" / "gate_set.jsonl"
TRAINED_NOTES = R4_WT / "data" / "embed-r4" / "r5_gate" / "tatoeba_notes.jsonl"
GATE_V2_DIR = R5_WT / "data" / "embed-r4" / "r5_gate_v2"
V44_DIR = B2_WT / "data" / "b2-graph-corpus" / "v44"

GRAM_LO, GRAM_HI = 8, 13
NOTE_PAIRS_MIN, NOTE_PAIRS_MAX = 2, 4
SENT_MIN, SENT_MAX = 10, 300
MAX_TOKENS = 256
REAL_CEILING = "2026-10-07T16:30:05Z"  # ratified real-part contamination cut
SEED_A_RETRIAL = 20261028  # declared: sources cut seed
LLM_TAKE = 240  # 120/set quota x2 failure margin
TWIN_JOBS = 30  # x TWIN_PER_JOB=6 -> 180 pairs capacity for 120 quota
TWIN_PER_JOB = 6
STOP_PER_SET = 60

_expect = {"quality_pass_pairs": 817178, "notes_after_gates": 123534, "stratum_notes": 750}


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    out: set[str] = set()
    for n in range(GRAM_LO, GRAM_HI + 1):
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def count_tokens(text: str) -> int:
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


def build_ban_registry() -> Ban:
    full = Ban()
    for tag, cdir in (("r4", R4_CORPUS), ("r5", R5_CORPUS), ("r7", R7_CORPUS)):
        for fn in ("train.jsonl", "val.jsonl"):
            for line in load_jsonl(cdir / fn):
                full.add(line["text"], f"{tag}-corpus")
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
    for path in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for line in load_jsonl(path):
            full.add(line.get("text"), "r7-gen")
    for path in sorted(SEED_DIR.glob("batch-translated-*.jsonl")):
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, dict):
                    full.add_side_dict(v, "v43-seed")
    for line in load_jsonl(REAL_PART):
        full.add_side_dict(line.get("side"), "real-part")
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            full.add_side_dict(line.get(sk), "burned")
    for line in load_jsonl(GATE_V1):
        full.add(line.get("text_src"), "gate-v1-cb367577")
        full.add(line.get("text_dst"), "gate-v1-cb367577")
    # burned gate-set v2 (r7-confirm): every text-like field, all files
    for fn in ("gate_set.jsonl", "gate_set.run-b.jsonl", "frozen_pool.jsonl", "sources.jsonl", "sources_extension1.jsonl"):
        path = GATE_V2_DIR / fn
        if not path.is_file():
            continue
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, str) and len(v) >= 20:
                    full.add(v, "gate-v2")
                elif isinstance(v, dict):
                    full.add_side_dict(v, "gate-v2")
    for path in sorted((GATE_V2_DIR / "gen").glob("*.jsonl")):
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, str) and len(v) >= 20:
                    full.add(v, "gate-v2-gen")
                elif isinstance(v, dict):
                    full.add_side_dict(v, "gate-v2-gen")
    # b2 calibration corpus v4.4 (training surface of the B2 line)
    for fn in ("train.jsonl", "holdout.jsonl", "labels-train.jsonl", "labels-holdout.jsonl"):
        path = V44_DIR / fn
        if not path.is_file():
            continue
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, str) and len(v) >= 20:
                    full.add(v, "v44")
                elif isinstance(v, dict):
                    full.add_side_dict(v, "v44")
    # TRAINED tatoeba notes (750)
    for row in load_jsonl(TRAINED_NOTES):
        full.add(row.get("text_src"), "tatoeba-trained")
        full.add(row.get("text_dst"), "tatoeba-trained")
    # ── A-retrial build-side additions (frozen pieces of THIS build) ──
    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    records = {o["id"]: o for o in (json.loads(l) for l in (HYG / "records.jsonl").read_text().splitlines())}
    dup = json.loads((CLS / "dup_pairs.json").read_text())
    dup_members = sorted({p["a"] for p in dup} | {p["b"] for p in dup})
    masked = json.loads((MASK / "masked_pairs.json").read_text())
    masked_ends = sorted({x for key in masked for x in key.rsplit("--", 1)})
    for rid in dup_members:
        r = records.get(rid)
        if r:
            full.add(r["title"], "build-dup-territory")
            full.add(r["body"], "build-dup-territory")
    for rid in masked_ends:
        r = records.get(rid)
        if r:
            full.add(r["title"], "build-masked-endpoint")
            full.add(r["body"], "build-masked-endpoint")
    # texts of store records that are sides of ANY post-hygiene edge OR FR-X/FR-Q pair members
    # are NOT banned here: train-side cuts happen AFTER the alpha/beta freeze and exclude
    # the frozen sources hash-level (В3 by-construction). Only EXISTING surfaces are banned.
    return full


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry_path = OUT / "ban_registry.pkl"

    # ── ban registry (cached: build once, reuse in later stages) ──
    if registry_path.is_file():
        full = pickle.loads(registry_path.read_bytes())
        print(f"ban registry loaded: hashes={len(full.hashes)} grams={len(full.grams)} sides={sum(full.sides.values())}")
    else:
        full = build_ban_registry()
        registry_path.write_bytes(pickle.dumps(full))
        print(f"ban registry built: hashes={len(full.hashes)} grams={len(full.grams)} sides={dict(sorted(full.sides.items()))}")

    # ── tatoeba pipeline: exact replication then expanded decontam ──
    sent: dict[int, tuple[str, str, str]] = {}
    for line_txt in read_bz2_lines(SENTENCES):
        parts = line_txt.split("\t")
        if len(parts) < 4:
            continue
        sid, lang, text = parts[0], parts[1], parts[2]
        if lang == "rus":
            lang = "ru"
        elif lang == "eng":
            lang = "en"
        else:
            continue
        sent[int(sid)] = (lang, text, parts[3])
    pairs: set[tuple[int, int]] = set()
    for line_txt in read_bz2_lines(LINKS):
        parts = line_txt.split("\t")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not (a.isdigit() and b.isdigit()):
            continue
        ia, ib = int(a), int(b)
        sa, sb = sent.get(ia), sent.get(ib)
        if not sa or not sb or sa[0] == sb[0]:
            continue
        ru_id, en_id = (ia, ib) if sa[0] == "ru" else (ib, ia)
        pairs.add((ru_id, en_id))
    ok_pairs: list[tuple[int, int]] = []
    seen_text: set[str] = set()
    for ru_id, en_id in sorted(pairs):
        ru_text, en_text = sent[ru_id][1], sent[en_id][1]
        if not sent_ok(ru_text) or not sent_ok(en_text):
            continue
        if text_hash(ru_text) == text_hash(en_text):
            continue
        th = text_hash(ru_text + "\x00" + en_text)
        if th in seen_text:
            continue
        seen_text.add(th)
        ok_pairs.append((ru_id, en_id))
    print(f"tatoeba replication: quality+decontam pass pairs = {len(ok_pairs)} (expect {_expect['quality_pass_pairs']})")
    assert len(ok_pairs) == _expect["quality_pass_pairs"], "REPLICATION FAIL: pairs"

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
    final: list[dict] = []
    for members in notes:
        rows = [(ri, ei, sent[ri][1], sent[ei][1]) for ri, ei in members]
        ru_join = " ".join(r[2] for r in rows)
        en_join = " ".join(r[3] for r in rows)
        if not note_ok(ru_join, en_join):
            continue
        key = text_hash(ru_join + "\x00" + en_join)
        final.append({"members": rows, "ru": ru_join, "en": en_join, "key": key})
    print(f"tatoeba replication: notes after original gates = {len(final)} (expect {_expect['notes_after_gates']})")
    assert len(final) == _expect["notes_after_gates"], "REPLICATION FAIL: notes"
    final.sort(key=lambda x: x["key"])
    trained_keys = set()
    for row in load_jsonl(TRAINED_NOTES):
        ru = row["text_src"] if row["direction"] == "ru->en" else row["text_dst"]
        en = row["text_dst"] if row["direction"] == "ru->en" else row["text_src"]
        trained_keys.add(text_hash(ru + "\x00" + en))
    assert {n["key"] for n in final[: _expect["stratum_notes"]]} == trained_keys, "REPLICATION FAIL: stratum-750"
    print("tatoeba replication asserts: PASS (pairs, notes, stratum-750 byte-exact)")

    pool = [n for n in final if n["key"] not in trained_keys]
    clean: list[dict] = []
    drops2: Counter = Counter()
    for n in pool:
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
    dirs = Counter(n["direction"] for n in clean)
    print(f"CLEAN llm-note pool (A-retrial bans): {len(clean)}; drops={dict(sorted(drops2.items()))}; directions={dict(sorted(dirs.items()))}")

    # ── store census for the real stratum ──
    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    records = {o["id"]: o for o in (json.loads(l) for l in (HYG / "records.jsonl").read_text().splitlines())}
    dup = json.loads((CLS / "dup_pairs.json").read_text())
    dup_members = {p["a"] for p in dup} | {p["b"] for p in dup}
    masked = json.loads((MASK / "masked_pairs.json").read_text())
    masked_ends = {x for key in masked for x in key.rsplit("--", 1)}
    real_census: Counter = Counter()
    real_ok: list[dict] = []
    for rid in ids:
        r = records[rid]
        if not r["created_at"] or r["created_at"] <= REAL_CEILING:
            real_census["pre-ceiling"] += 1
            continue
        if any(str(t).startswith("agent:") for t in r["tags"]):
            real_census["agent-service"] += 1
            continue
        if rid in dup_members:
            real_census["dup-territory"] += 1
            continue
        if rid in masked_ends:
            real_census["masked-endpoint"] += 1
            continue
        hit = None
        for fld in ("title", "body"):
            t = str(r.get(fld) or "").strip()
            if not t:
                continue
            if text_hash(t) in full.hashes:
                hit = f"{fld}-hash"
                break
            if ngrams(t) & full.grams:
                hit = f"{fld}-ngram"
                break
        if hit:
            real_census[hit] += 1
            continue
        lang = r.get("language")
        real_census["clean"] += 1
        real_ok.append({"id": rid, "lang": lang, "created_at": r["created_at"]})
    lang_hist = Counter(o["lang"] for o in real_ok)
    print(f"REAL stratum census: {dict(sorted(real_census.items()))}")
    print(f"REAL clean pool: {len(real_ok)}; languages={dict(sorted(lang_hist.items()))}")

    # ── K1 verdicts (ratified STOP minimums) ──
    verdicts = {
        "llm": {"pool": len(clean), "needed_per_set": STOP_PER_SET, "needed_total": 2 * STOP_PER_SET,
                "verdict": "PASS" if len(clean) >= 2 * STOP_PER_SET else "FAIL"},
        "real": {"pool": len(real_ok), "needed_per_set": STOP_PER_SET, "needed_total": 2 * STOP_PER_SET,
                 "verdict": "PASS" if len(real_ok) >= 2 * STOP_PER_SET else "FAIL"},
        "twin": {"pool": "synthetic-unlimited", "verdict": "PASS (generation-dependent)"},
    }
    print("K1 availability verdicts:", json.dumps(verdicts))

    (OUT / "census_sources.json").write_text(
        json.dumps(
            {
                "llm_pool": len(clean),
                "llm_pool_directions": dict(sorted(dirs.items())),
                "llm_pool_drops": dict(sorted(drops2.items())),
                "real_census": dict(sorted(real_census.items())),
                "real_pool": len(real_ok),
                "real_pool_languages": dict(sorted(lang_hist.items())),
                "ban_registry": {"hashes": len(full.hashes), "grams": len(full.grams), "sides": dict(sorted(full.sides.items()))},
                "k1_availability_verdicts": verdicts,
                "constants": {
                    "real_ceiling": REAL_CEILING,
                    "seed_sources": SEED_A_RETRIAL,
                    "llm_take": LLM_TAKE,
                    "twin_jobs": TWIN_JOBS,
                    "twin_per_job": TWIN_PER_JOB,
                    "stop_per_set": STOP_PER_SET,
                },
            },
            indent=1,
        ) + "\n"
    )
    with (OUT / "llm_clean_pool.jsonl").open("w", encoding="utf-8") as fh:
        for n in clean:
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")
    with (OUT / "real_clean_pool.jsonl").open("w", encoding="utf-8") as fh:
        for o in real_ok:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("pools written; run `a5_sources.py cut` to freeze sources.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
