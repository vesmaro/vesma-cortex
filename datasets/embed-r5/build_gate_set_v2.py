#!/usr/bin/env python3
"""Round-5 gate set v2 (ADDENDUM 1, TL decision 2026-10-22) — fresh-material
confirming gate for the round-7 ADOPT decision.

The original builder (datasets/embed-r5/build_gate_set.py) is UNTOUCHED and
its PA-Q1 population contract is preserved: strata llm/real/seed-dup/twin,
direction-balanced cells, same thresholds. The single change is the MATERIAL:
every v2 pair is built from sources that never entered ANY training surface
(zero-overlap proofs at hash level AND n-gram 8-13 zero tolerance enforce it).

Stages (subcommands):
  sources   — deterministically cut the v2 source lists from clean pools and
              freeze them to sources.jsonl (fingerprint asserted later):
                llm       notes[0:240]   of the clean tatoeba pool (key order)
                seed-dup  notes[240:330] (disjoint slice, assert)
                real      the 7 post-ceiling clean store records
                          (created_at > 2026-10-07T16:30:05Z, the real-part
                          assembly ceiling — the ratified contamination cut)
                twin      job plan 20 jobs x TWIN_PER_JOB=6, families
                          meeting/notes/rules/science/tech (r4 constructor)
  assemble  — build the FROZEN pool from the gated generation shards
              (filter_gate_v2.py annotations), assert disjointness, pool fp
  select    — deterministic selection from the frozen pool,
              random.Random(385918605); run twice, byte-identical assert;
              writes gate_set.jsonl (schema byte-identical to cb367577,
              pair_id prefix R5G2-)
  proofs    — zero-overlap proofs (hash-level + n-gram 8-13) of every
              selected side vs EVERY training surface, label-noise shas,
              cb367577; manifest numbers-only JSON; seal

Quotas (ratified 2026-10-22): llm 125 / seed-dup 40 / twin 52 / real 7 = 224
(direct proportions of cb367577 redistributed over the clean pool after the
real-stratum shortfall; honest deviations listed in the manifest).

Seeds: v2 selection random.Random(385918605); legacy 42 recorded alongside.

Read-only over every training surface; the live store is opened
mode=ro + query_only. Texts land ONLY in the gitignored data dir
(data/embed-r4/r5_gate_v2/); this script, the survey, the gen/filter
scripts, the numbers-only manifest and the run-log are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")

OUT_DIR = R5_WT / "data" / "embed-r4" / "r5_gate_v2"
GEN_DIR = OUT_DIR / "gen"
SOURCES_PATH = OUT_DIR / "sources.jsonl"
NUMBERS_DIR = R5_WT / "datasets" / "embed-r5"
RUN_LOG = NUMBERS_DIR / "run-log.jsonl"

GATE_SET_V1 = R4_WT / "data" / "embed-r4" / "r5_gate" / "gate_set.jsonl"
TRAINED_NOTES = R4_WT / "data" / "embed-r4" / "r5_gate" / "tatoeba_notes.jsonl"
DL_DIR = R5_WT / "data" / "embed-r5" / "tatoeba"
SENTENCES = DL_DIR / "sentences_detailed.tar.bz2"
LINKS = DL_DIR / "links.tar.bz2"
R4_CORPUS = R4_WT / "data" / "embed-r4" / "corpus"
R5_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R7_CORPUS = R5_WT / "data" / "embed-r7" / "corpus"
R4_GEN = R4_WT / "data" / "embed-r4" / "gen"
R7_GEN = R5_WT / "data" / "embed-r7" / "gen"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
REAL_PART = R4_WT / "datasets" / "v43" / "real-part.jsonl"
BURNED_PAIRS = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"
LIVE_STORE = Path("/var/home/abyss/.distrobox/vscode-box/home/.mnemos/data/mnemos.db")

sys.path.insert(0, str(R4_WT / "scripts"))
from gen_embed_r4_texts import (  # noqa: E402
    UNIT_TOKEN_CAP,
    count_tokens,
    detect_lang,
    load_jsonl,
    normalise,
    text_hash,
)

# ── ratified constants ───────────────────────────────────────────────────────
SEED_V2 = 385918605
SEED_LEGACY = 42
QUOTA = {"llm": 125, "seed-dup": 40, "twin": 52, "real": 7}
TARGET_TOTAL = sum(QUOTA.values())  # 224
LLM_NOTES = 240          # notes[0:240]
SEED_NOTES = 90          # notes[240:330]
TWIN_JOBS = 20           # 4 per family x 5 families, TWIN_PER_JOB pairs each
TWIN_PER_JOB = 6
TWIN_FAMILIES = ["meeting", "notes", "rules", "science", "tech"]
REAL_CUTOFF = "2026-10-07T16:30:06"  # > real-part assembly ceiling
REAL_MIN_CHARS = 120
GRAM_LO, GRAM_HI = 8, 13
LABEL_NOISE_PAIRS = ("R5G-0106", "R5G-0141", "R5G-0163")
POOL_EXPECT_BAN_GRAMS = 5265881   # cross-check vs survey_gate_v2_availability
POOL_EXPECT_BAN_HASHES = 35198

REAL_SQL = (
    "SELECT title, content, tags, memory_type, created_at, metadata FROM memories "
    "WHERE status IN ('published','raw') "
    "AND memory_type IN ('note','snippet','fact') "
    "AND length(content) >= 120 "
    "AND created_at > '2026-10-07T16:30:06'"
)
REAL_TAG_PREFIXES = ("task:", "telemetry")


def ngrams(text: str) -> set[str]:
    w = " ".join(unicodedata.normalize("NFC", text).lower().split()).split(" ")
    return {" ".join(w[i : i + n]) for n in range(GRAM_LO, GRAM_HI + 1)
            for i in range(len(w) - n + 1)}


def emit_run_log(stage: str, msg: str) -> None:
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "ts_step": f"gate_v2:{stage}", "msg": msg},
                            ensure_ascii=False) + "\n")


def say(stage: str, msg: str) -> None:
    print(f"[v2:{stage}] {msg}", flush=True)
    emit_run_log(stage, msg)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── clean tatoeba pool (exact survey replication) ────────────────────────────

def _read_bz2_lines(path: Path):
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


def _sent_ok(text: str) -> bool:
    if not (10 <= len(text.strip()) <= 300):
        return False
    low = text.lower()
    if "http" in low or "www." in low or "@" in low or "<" in low:
        return False
    if sum(1 for ch in text if ch.isalpha()) < 0.6 * len(text.strip()):
        return False
    return True


def _note_ok(ru: str, en: str) -> bool:
    for t in (ru, en):
        if not (40 <= len(t) <= 4000) or count_tokens(t) > 256:
            return False
    return text_hash(ru) != text_hash(en)


def clean_note_pool(log) -> list[dict]:
    """Exact replication of the tatoeba builder on its ORIGINAL ban set,
    then the expanded ban set on notes[750:] — identical to
    survey_gate_v2_availability.py (asserts included)."""
    orig_hashes: set[str] = set()
    orig_grams: set[str] = set()
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = line.get(sk) or {}
            for fld in ("title", "body"):
                v = str(side.get(fld) or "").strip()
                if v:
                    orig_hashes.add(text_hash(v))
                    orig_grams |= ngrams(v)
    for line in load_jsonl(GATE_SET_V1):
        for fld in ("text_src", "text_dst"):
            orig_hashes.add(text_hash(line[fld]))
            orig_grams |= ngrams(line[fld])
    for fn in ("train.jsonl", "val.jsonl"):
        for line in load_jsonl(R4_CORPUS / fn):
            orig_hashes.add(text_hash(line["text"]))
            if line["source"] == "real-part-r4":
                orig_grams |= ngrams(line["text"])

    sent: dict[int, tuple[str, str, str]] = {}
    for line_txt in _read_bz2_lines(SENTENCES):
        parts = line_txt.split("\t")
        if len(parts) < 4:
            continue
        sid, lang, text, user = parts[0], parts[1], parts[2], parts[3]
        if lang == "rus":
            lang = "ru"
        elif lang == "eng":
            lang = "en"
        else:
            continue
        sent[int(sid)] = (lang, text, user)
    pairs: set[tuple[int, int]] = set()
    for line_txt in _read_bz2_lines(LINKS):
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
        pairs.add((ia, ib) if sa[0] == "ru" else (ib, ia))

    ok_pairs = []
    seen_text = set()
    for ru_id, en_id in sorted(pairs):
        ru_text, en_text = sent[ru_id][1], sent[en_id][1]
        if not (_sent_ok(ru_text) and _sent_ok(en_text)):
            continue
        if text_hash(ru_text) == text_hash(en_text):
            continue
        th = text_hash(ru_text + "\x00" + en_text)
        if th in seen_text:
            continue
        if ngrams(ru_text) & orig_grams or ngrams(en_text) & orig_grams:
            continue
        if text_hash(ru_text) in orig_hashes or text_hash(en_text) in orig_hashes:
            continue
        seen_text.add(th)
        ok_pairs.append((ru_id, en_id))
    assert len(ok_pairs) == 817178, f"replication fail: {len(ok_pairs)} != 817178"

    used: set[int] = set()
    notes: list[list[tuple[int, int]]] = []
    buf: list[tuple[int, int]] = []
    for ru_id, en_id in ok_pairs:
        if ru_id in used or en_id in used:
            if len(buf) >= 2:
                notes.append(buf)
            buf = []
            continue
        buf.append((ru_id, en_id))
        used.add(ru_id)
        used.add(en_id)
        if len(buf) == 4:
            notes.append(buf)
            buf = []
    if len(buf) >= 2:
        notes.append(buf)

    final = []
    for members in notes:
        rows = [(ri, ei, sent[ri][1], sent[ei][1], sent[ri][2], sent[ei][2])
                for ri, ei in members]
        ru_join = " ".join(r[2] for r in rows)
        en_join = " ".join(r[3] for r in rows)
        if not _note_ok(ru_join, en_join):
            continue
        if ngrams(ru_join) & orig_grams or ngrams(en_join) & orig_grams:
            continue
        final.append({"members": rows, "ru": ru_join, "en": en_join,
                      "key": text_hash(ru_join + "\x00" + en_join)})
    assert len(final) == 123534, f"replication fail: {len(final)} != 123534"
    final.sort(key=lambda x: x["key"])
    stratum_keys = {n["key"] for n in final[:750]}
    trained_keys = set()
    for row in load_jsonl(TRAINED_NOTES):
        ru = row["text_src"] if row["direction"] == "ru->en" else row["text_dst"]
        en = row["text_dst"] if row["direction"] == "ru->en" else row["text_src"]
        trained_keys.add(text_hash(ru + "\x00" + en))
    assert stratum_keys == trained_keys, "replication fail: stratum-750 keys"
    log(f"replication asserts PASS: pairs=817178 notes=123534 stratum750-keys exact")

    expanded_hashes: set[str] = set(orig_hashes)
    expanded_grams: set[str] = set(orig_grams)
    for fn in ("train.jsonl", "val.jsonl"):
        for line in load_jsonl(R4_CORPUS / fn):
            expanded_hashes.add(text_hash(line["text"]))
            expanded_grams |= ngrams(line["text"])
        for ctag, cdir in (("r5", R5_CORPUS), ("r7", R7_CORPUS)):
            for line in load_jsonl(cdir / fn):
                expanded_hashes.add(text_hash(line["text"]))
                expanded_grams |= ngrams(line["text"])
    for line in load_jsonl(R4_GEN / "synth-template-mono.jsonl"):
        t = str(line.get("text") or "").strip()
        if t:
            expanded_hashes.add(text_hash(t))
            expanded_grams |= ngrams(t)
    for pat, flds in (("synth-mono.shard*.jsonl", ("text",)),
                      ("llm-translations.shard*.jsonl", ("text",)),
                      ("real-translations.shard*.jsonl", ("text",)),
                      ("synth-twins.shard*.jsonl", ("ru", "en"))):
        for path in sorted(R4_GEN.glob(pat)):
            for line in load_jsonl(path):
                for fld in flds:
                    t = str(line.get(fld) or "").strip()
                    if t:
                        expanded_hashes.add(text_hash(t))
                        expanded_grams |= ngrams(t)
    for path in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for line in load_jsonl(path):
            t = str(line.get("text") or "").strip()
            if t:
                expanded_hashes.add(text_hash(t))
                expanded_grams |= ngrams(t)
    for path in sorted(SEED_DIR.glob("batch-translated-*.jsonl")):
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, dict):
                    for fld in ("title", "body", "text"):
                        t = str(v.get(fld) or "").strip()
                        if t:
                            expanded_hashes.add(text_hash(t))
                            expanded_grams |= ngrams(t)
    for line in load_jsonl(REAL_PART):
        side = line.get("side") or {}
        for fld in ("title", "body"):
            t = str(side.get(fld) or "").strip()
            if t:
                expanded_hashes.add(text_hash(t))
                expanded_grams |= ngrams(t)
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = line.get(sk) or {}
            for fld in ("title", "body"):
                t = str(side.get(fld) or "").strip()
                if t:
                    expanded_hashes.add(text_hash(t))
                    expanded_grams |= ngrams(t)
    gate1_hashes: set[str] = set()
    for line in load_jsonl(GATE_SET_V1):
        for fld in ("text_src", "text_dst"):
            expanded_hashes.add(text_hash(line[fld]))
            expanded_grams |= ngrams(line[fld])
            gate1_hashes.add(text_hash(line[fld]))
    log(f"expanded ban set: grams={len(expanded_grams)} hashes={len(expanded_hashes)}")
    assert len(expanded_grams) == POOL_EXPECT_BAN_GRAMS, "ban-gram cross-check vs survey"
    assert len(expanded_hashes) == POOL_EXPECT_BAN_HASHES, "ban-hash cross-check vs survey"

    pool = []
    drops = Counter()
    for n in final:
        if n["key"] in trained_keys:
            continue
        hit = False
        for _ri, _ei, ru_t, en_t, _au, _ae in n["members"]:
            if ngrams(ru_t) & expanded_grams or ngrams(en_t) & expanded_grams:
                drops["sent-ngram"] += 1
                hit = True
                break
            if text_hash(ru_t) in expanded_hashes or text_hash(en_t) in expanded_hashes:
                drops["sent-hash"] += 1
                hit = True
                break
        if hit:
            continue
        if ngrams(n["ru"]) & expanded_grams or ngrams(n["en"]) & expanded_grams:
            drops["note-ngram"] += 1
            continue
        if n["ru"] and text_hash(n["ru"]) in expanded_hashes:
            drops["note-hash"] += 1
            continue
        n["direction"] = "ru->en" if int(n["key"], 16) % 2 == 0 else "en->ru"
        pool.append(n)
    log(f"clean pool: {len(pool)} notes; drops={dict(sorted(drops.items()))}")
    assert len(pool) == 122747, f"clean pool mismatch vs survey: {len(pool)}"
    return pool, expanded_hashes, expanded_grams, gate1_hashes


# ── real sources (store hygiene chain, verbatim assemble_real_part) ─────────

def real_sources(log) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(R4_WT / "scripts"))
    import assemble_real_part as arp

    conn = sqlite3.connect(f"file:{LIVE_STORE.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    scan, provenance = arp._scan_reasons(ENGINE_SRC)
    counters: Counter = Counter()
    rows = []
    for title, content, tags_json, mtype, created_at, metadata_json in conn.execute(REAL_SQL):
        counters["sql_candidates"] += 1
        tags = arp._parse_tags(tags_json)
        if any(isinstance(t, str) and t.startswith(p) for t in tags for p in REAL_TAG_PREFIXES):
            counters["excluded:privacy-tag"] += 1
            continue
        reasons = scan(title or "", content or "", tags)
        if reasons:
            for r in reasons:
                counters[f"excluded:{r}"] += 1
            continue
        rows.append({"title": title or "", "content": content or "",
                     "tags": tags, "memory_type": mtype,
                     "created_at": created_at or ""})
    conn.close()
    # dedup by engine content hash (real-part convention); keep post-ceiling
    by_hash: dict[str, dict] = {}
    for r in rows:
        h = arp.content_hash(r["content"])
        by_hash.setdefault(h, r)
    counters["dedup:unique_content_hashes"] = len(by_hash)

    out = []
    for r in sorted(by_hash.values(), key=lambda r: (r["created_at"], arp.content_hash(r["content"]))):
        cyr = arp._cyrillic_share(r["content"])
        src_lang = "ru" if cyr >= 0.2 else detect_lang(r["content"])
        if src_lang not in ("ru", "en"):
            counters["excluded:lang"] += 1
            continue
        unit = r["content"].strip()
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n\n" in unit:
            unit = unit.rsplit("\n\n", 1)[0].strip()
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n" in unit:
            unit = unit.rsplit("\n", 1)[0].strip()
        if count_tokens(unit) > UNIT_TOKEN_CAP:
            while count_tokens(unit) > 350 and len(unit) > 400:
                unit = unit[: int(len(unit) * 0.8)].rsplit(" ", 1)[0].strip()
        target = "en" if src_lang == "ru" else "ru"
        out.append({"kind": "real", "key": arp.content_hash(r["content"]),
                    "unit": unit, "src_lang": src_lang, "target": target,
                    "memory_type": r["memory_type"], "created_at": r["created_at"],
                    "content_cyrillic_share": round(cyr, 4),
                    "unit_tokens": count_tokens(unit)})
    log(f"real sources: {len(out)} (sql={counters['sql_candidates']}, "
        f"privacy={counters['excluded:privacy-tag']}, secrets={sum(v for k, v in counters.items() if ':secret' in k)}, "
        f"dedup={counters['dedup:unique_content_hashes']}, lang={counters['excluded:lang']})")
    info = {"sql": REAL_SQL, "tag_prefixes": list(REAL_TAG_PREFIXES),
            "scanner_provenance": provenance, "counters": dict(sorted(counters.items())),
            "cyrillic_ru_centric_threshold": 0.2, "min_content_chars": REAL_MIN_CHARS,
            "unit_token_cap": UNIT_TOKEN_CAP,
            "survivor_tag_names": sorted({t for r in rows for t in r["tags"]
                                          if not t.startswith(REAL_TAG_PREFIXES)})}
    return out, info


def stage_sources() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def logf(msg: str) -> None:
        say("sources", msg)
        log.append(msg)

    pool, _eh, _eg, _g1 = clean_note_pool(lambda m: logf(m))
    notes = []
    for rank, n in enumerate(pool[: LLM_NOTES + SEED_NOTES]):
        members = n["members"]
        notes.append({
            "kind": "note", "pool": "llm" if rank < LLM_NOTES else "seed-dup",
            "rank": rank, "key": n["key"], "direction": n["direction"],
            "ru": n["ru"], "en": n["en"],
            "tatoeba_sentence_ids": [r[0] for r in members] + [r[1] for r in members],
            "authors": sorted({r[4] for r in members} | {r[5] for r in members}),
            "license": "CC-BY-2.0-FR",
        })
    llm_n = [n for n in notes if n["pool"] == "llm"]
    seed_n = [n for n in notes if n["pool"] == "seed-dup"]
    logf(f"note allocation: llm={len(llm_n)} "
         f"{dict(Counter(n['direction'] for n in llm_n))}; seed={len(seed_n)} "
         f"{dict(Counter(n['direction'] for n in seed_n))}")

    reals, real_info = real_sources(logf)
    logf(f"real sources: {len(reals)}; directions="
         f"{dict(Counter('ru->en' if r['target'] == 'en' else 'en->ru' for r in reals))}")

    twins = [{"kind": "twin_job", "job": f"t2-{fam}-{idx:03d}", "family": fam,
              "npairs": TWIN_PER_JOB}
             for fam in TWIN_FAMILIES for idx in range(TWIN_JOBS // len(TWIN_FAMILIES))]
    logf(f"twin jobs: {len(twins)} x {TWIN_PER_JOB} pairs = {len(twins) * TWIN_PER_JOB} requested")

    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        for row in notes + reals + twins:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fp = sha256_file(SOURCES_PATH)
    meta = {"kind": "embed-r5-gate-v2-sources-manifest", "created": "2026-10-22",
            "sources_fp": fp,
            "counts": {"llm_notes": len(llm_n), "seed_notes": len(seed_n),
                       "real_records": len(reals), "twin_jobs": len(twins),
                       "twin_pairs_requested": len(twins) * TWIN_PER_JOB},
            "real_provenance": real_info}
    (OUT_DIR / "sources_manifest.numbers-only.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    logf(f"sources.jsonl fp={fp}")
    return 0


# ── proofs + manifest ────────────────────────────────────────────────────────

def expanded_ban() -> tuple[set[str], set[str], set[str], set[str]]:
    """(hashes, grams, gate1_shas, label_noise_shas) over every training
    surface — byte-consistent with survey_gate_v2_availability (asserted)."""
    hashes: set[str] = set()
    grams: set[str] = set()

    def add_text(t: str) -> None:
        t = t.strip()
        if not t:
            return
        hashes.add(text_hash(t))
        grams.update(ngrams(t))

    for fn in ("train.jsonl", "val.jsonl"):
        for cdir in (R4_CORPUS, R5_CORPUS, R7_CORPUS):
            for line in load_jsonl(cdir / fn):
                add_text(line["text"])
    for line in load_jsonl(R4_GEN / "synth-template-mono.jsonl"):
        add_text(str(line.get("text") or ""))
    for pat, flds in (("synth-mono.shard*.jsonl", ("text",)),
                      ("llm-translations.shard*.jsonl", ("text",)),
                      ("real-translations.shard*.jsonl", ("text",)),
                      ("synth-twins.shard*.jsonl", ("ru", "en"))):
        for path in sorted(R4_GEN.glob(pat)):
            for line in load_jsonl(path):
                for fld in flds:
                    add_text(str(line.get(fld) or ""))
    for path in sorted(R7_GEN.glob("llm-translations-r7.shard*.jsonl")):
        for line in load_jsonl(path):
            add_text(str(line.get("text") or ""))
    for path in sorted(SEED_DIR.glob("batch-translated-*.jsonl")):
        for line in load_jsonl(path):
            for v in line.values():
                if isinstance(v, dict):
                    for fld in ("title", "body", "text"):
                        add_text(str(v.get(fld) or ""))
    for line in load_jsonl(REAL_PART):
        side = line.get("side") or {}
        add_text(str(side.get("title") or ""))
        add_text(str(side.get("body") or ""))
    for line in load_jsonl(BURNED_PAIRS):
        if not str(line.get("pair_id", "")).startswith("V43-TDUP"):
            continue
        for sk in ("record", "candidate"):
            side = line.get(sk) or {}
            for fld in ("title", "body"):
                add_text(str(side.get(fld) or ""))
    gate1_shas: set[str] = set()
    for line in load_jsonl(GATE_SET_V1):
        add_text(line["text_src"])
        add_text(line["text_dst"])
        gate1_shas.update((text_hash(line["text_src"]), text_hash(line["text_dst"])))
    label_noise: set[str] = set()
    for line in load_jsonl(GATE_SET_V1):
        if line["pair_id"] in LABEL_NOISE_PAIRS:
            label_noise.update((line["text_sha_src"], line["text_sha_dst"]))
    assert len(grams) == POOL_EXPECT_BAN_GRAMS, "ban-gram cross-check vs survey"
    assert len(hashes) == POOL_EXPECT_BAN_HASHES, "ban-hash cross-check vs survey"
    assert len(gate1_shas) == 360, "gate v1 side count"
    assert len(label_noise) == 6, "label-noise sha count"
    return hashes, grams, gate1_shas, label_noise


def stage_proofs() -> int:
    gate_path = OUT_DIR / "gate_set.jsonl"
    rows = [json.loads(l) for l in open(gate_path, encoding="utf-8")]
    ban_hashes, ban_grams, gate1_shas, label_noise = expanded_ban()
    say("proofs", f"ban set rebuilt: grams={len(ban_grams)} hashes={len(ban_hashes)}")

    hash_hits = ngram_hit_sides = 0
    gate1_hits = noise_hits = 0
    for r in rows:
        for fld, sha in (("text_src", r["text_sha_src"]), ("text_dst", r["text_sha_dst"])):
            if sha in ban_hashes:
                hash_hits += 1
            if sha in gate1_shas:
                gate1_hits += 1
            if sha in label_noise:
                noise_hits += 1
            if ngrams(r[fld]) & ban_grams:
                ngram_hit_sides += 1
    say("proofs", f"zero-overlap: hash_hits_vs_all_surfaces={hash_hits} "
                  f"vs_cb367577={gate1_hits} vs_label_noise={noise_hits} "
                  f"ngram_hit_sides={ngram_hit_sides}")
    assert hash_hits == 0 and gate1_hits == 0 and noise_hits == 0 and ngram_hit_sides == 0, \
        "ZERO-OVERLAP PROOF FAILED"

    pair_ids = [r["pair_id"] for r in rows]
    assert pair_ids == [f"R5G2-{i:04d}" for i in range(1, len(rows) + 1)]
    all_shas = [r["text_sha_src"] for r in rows] + [r["text_sha_dst"] for r in rows]
    assert len(all_shas) == len(set(all_shas)) == 2 * len(rows)

    counts_class = Counter(r["source_class"] for r in rows)
    counts_dir = Counter(r["direction"] for r in rows)
    counts_cd = Counter((r["source_class"], r["direction"]) for r in rows)

    # generation reject statistics (honest numbers, as they fell)
    gen_stats: Counter = Counter()
    wall = []
    for line in open(RUN_LOG, encoding="utf-8"):
        e = json.loads(line)
        ev = str(e.get("event") or e.get("ts_step") or "")
        if ev == "gate_v2:gen-done":
            gen_stats[f"pass{e['gen_pass']}:kept"] += e["kept"]
            gen_stats[f"pass{e['gen_pass']}:bad"] += e["bad"]
            wall.append(e["wall_sec"])
    reject_reasons: Counter = Counter()
    for shard in sorted(GEN_DIR.glob("*-v2*.jsonl")):
        for row in load_jsonl(shard):
            if row.get("status") in ("fail", "gate-fail"):
                for reason in str(row.get("reason", "?")).split(";"):
                    reject_reasons[reason.split(":")[0]] += 1

    pool_fp = sha256_file(OUT_DIR / "frozen_pool.jsonl")
    ext_fp = sha256_file(OUT_DIR / "sources_extension1.jsonl")
    src_meta = json.loads((OUT_DIR / "sources_manifest.numbers-only.json").read_text())
    gate_sha = sha256_file(gate_path)
    run_b_sha = sha256_file(OUT_DIR / "gate_set.run-b.jsonl")
    ids_sha = hashlib.sha256("\n".join(sorted(pair_ids)).encode("utf-8")).hexdigest()
    shas_sha = hashlib.sha256("\n".join(sorted(set(all_shas))).encode("utf-8")).hexdigest()

    filter_stats: Counter = Counter()
    for line in open(RUN_LOG, encoding="utf-8"):
        e = json.loads(line)
        ev = str(e.get("event") or "")
        if ev in ("gate_v2:filter-cos", "gate_v2:filter-ngram"):
            filter_stats[f"{ev.split(':')[1]}:checked"] += e["checked"]
            filter_stats[f"{ev.split(':')[1]}:rejected"] += e["rejected"]

    selected_real = counts_class.get("real", 0)
    manifest = {
        "kind": "embed-r5-gate-v2-manifest",
        "created": "2026-10-22",
        "contract": "docs/specs/embed-round7-confirm-ADDENDUM-1.md §2.3; "
                    "TL decisions 2026-10-22 (OK 0b): real via real-part-ceiling cut, "
                    "redistribution llm125/seed-dup40/twin52/real7; realized "
                    "125/40/52/6 = 223 (real seat 7 -> 6, see deviations)",
        "pa_q1_formula": "UNCHANGED (owner pre-registration): strata llm/real/seed-dup/twin, "
                         "direction-balanced cells, thresholds G1..G5 unchanged",
        "teacher": {"model": "Qwen/Qwen3-1.7B-GGUF", "quant": "Q8_0",
                    "runtime": "llama.cpp (.venv-r7gen, llama-cpp-python 0.3.36, CPU, threads=8)",
                    "contracts": "translation prompt + gates = round-7 verbatim; "
                                 "twin prompt + gates = round-4 verbatim"},
        "seeds": {"selection_v2": SEED_V2, "legacy": SEED_LEGACY,
                  "generation": "385918605 + pass*1_000_000 + global_job_index"},
        "inputs": {
            "sources_fp": src_meta["sources_fp"],
            "sources_extension1_fp": ext_fp,
            "frozen_pool_fp": pool_fp,
            "ban_set": {"grams": len(ban_grams), "hashes": len(ban_hashes),
                        "surfaces": ["r4 corpus train+val", "r5 corpus train+val (incl. tatoeba-750)",
                                     "r7 corpus train+val", "r4 gen shards (mono/template/llm/real/twins)",
                                     "r7 gen llm-translations", "v43 seed batch-translated-{dup,sibling}",
                                     "real-part.jsonl", "burned V43-TDUP 288 sides",
                                     "cb367577 gate set 360 sides"],
                        "cross_check_vs_survey": "grams 5265881 and hashes 35198 asserted equal"},
            "pool_replication_asserts": {"quality_pass_pairs": 817178, "notes_after_gates": 123534,
                                         "stratum750_keys": "exact", "clean_pool": 122747},
        },
        "counts": {
            "total_pairs": len(rows),
            "by_class": dict(sorted(counts_class.items())),
            "by_direction": dict(sorted(counts_dir.items())),
            "by_class_x_direction": {f"{c}:{d}": n for (c, d), n in sorted(counts_cd.items())},
            "direction_abs_delta": abs(counts_dir.get("ru->en", 0) - counts_dir.get("en->ru", 0)),
        },
        "generation": {
            "discipline": "SINGLE-SHOT (ADDENDUM 1): pool frozen after filtering; "
                          "double-run determinism covers SELECTION only",
            "stats_by_pass": {k: v for k, v in sorted(gen_stats.items())},
            "reject_reasons": dict(sorted(reject_reasons.items())),
            "filter_gates": {k: v for k, v in sorted(filter_stats.items())},
            "pool_after_filter": {"llm": 215, "seed-dup": 86, "real": 6, "twin": 100,
                                  "total": 407},
            "wall_sec_total": round(sum(wall), 1),
            "shards": sorted(p.name for p in GEN_DIR.glob("*-v2*.jsonl")),
        },
        "selection_rule": (
            "Frozen pool sorted by (stratum, pair_key); strata alphabetical; "
            "per stratum quota seats: ru->en gets ceil(q/2), en->ru floor(q/2); "
            "real fills available cells only (en->ru pool empty, 6 of 7 seats "
            "realized); random.Random(385918605).sample over each sorted cell in "
            "fixed order (en->ru before ru->en); selected sorted by pair_key -> "
            "R5G2-%04d. Quotas: llm 125 / seed-dup 40 / twin 52 / real 7 = 224 "
            "seats, 223 realized (direct proportions of cb367577 68/22/29/61 "
            "redistributed after the real-stratum shortfall; TL-ratified 2026-10-22)."
        ),
        "double_run": {"runs": 2, "byte_identical": True,
                       "gate_set_sha256": gate_sha, "run_b_sha256": run_b_sha},
        "proofs": {
            "hash_overlap_vs_all_training_surfaces": hash_hits,
            "hash_overlap_vs_cb367577_360sides": gate1_hits,
            "hash_overlap_vs_label_noise_106_141_163": noise_hits,
            "ngram_8_13_hit_sides_vs_all_surfaces": ngram_hit_sides,
            "pair_ids_sequential_unique": True,
            "text_shas_unique": True,
        },
        "license": {
            "tatoeba": "CC-BY-2.0-FR; attribution: Tatoeba (tatoeba.org) + per-note authors "
                       "(authors and sentence ids ride the frozen pool rows of llm/seed-dup)",
            "real": "owner's own store records (private; local gitignored data only)",
            "llm_twin_dst_sides": "teacher output on clean inputs",
        },
        "real_sql_provenance": src_meta["real_provenance"],
        "deviations": [
            "real 82 -> 6 realized (7 seats): only 7 clean post-real-part-ceiling "
            "store records exist (created_at > 2026-10-07T16:30:05Z); 1 of the 7 is "
            "untranslatable under the ratified teacher contract (echo-class failure "
            "on all 3 passes, no clean replacement exists); strict post-freeze cut "
            "would yield 0; G2-real floor degenerates to zero-miss-on-6 (conservative "
            "falsifier)",
            "total 224 seats -> 223 realized (redistribution ratified by TL 2026-10-22)",
            "real direction cells 6/0 (en->ru empty) - publish as monitoring",
            "overall direction |delta|=7 (50/50 unattainable with real x/0)",
            "seed-dup material source: owner seed dataset is a burned training surface; "
            "v2 uses tatoeba-derived notes (record/translation construction preserved, no title)",
            "llm/seed-dup unit shape: tatoeba pseudo-note <=256 tokens/side (r4 unit cap was 100)",
            "generation is single-shot; double-run = selection only",
            "pair_id prefix R5G2- (schema unchanged)",
            "twin sampling seeds new (385918605-based), prompt/gates unchanged",
            "source extension-1 (pre-registered rule, fp in inputs): echo-class "
            "generation failures proved seed-independent (pass-2/3 recovery 6%/15%), "
            "cells closed with fresh clean notes llm ranks 330-389 / seed 430-459 / "
            "4 twin jobs, never by re-prompting burned or dirty material",
        ],
        "integrity": {
            "gate_set_sha256": gate_sha,
            "sorted_pair_id_list_sha256": ids_sha,
            "sorted_text_sha_list_sha256": shas_sha,
            "n_text_shas": len(set(all_shas)),
        },
    }
    out = NUMBERS_DIR / "gate_manifest_v2.numbers-only.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    say("proofs", f"manifest written: {out}")
    say("proofs", f"SEALED: gate_set sha256={gate_sha}")
    return 0

def load_sources() -> dict:
    notes, reals = {}, {}
    for fn in ("sources.jsonl", "sources_extension1.jsonl"):
        path = OUT_DIR / fn
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row["kind"] == "note":
                notes[(row["pool"], row["key"])] = row
            elif row["kind"] == "real":
                reals[row["key"]] = row
    return notes, reals


def stage_assemble() -> int:
    notes, reals = load_sources()
    pool: list[dict] = []
    stats: Counter = Counter()
    for shard in sorted(GEN_DIR.glob("*-v2*.jsonl")):
        for ln, row in enumerate(load_jsonl(shard), 1):
            if row.get("status") != "ok":
                stats[f"{row.get('stratum', 'twin')}:{row.get('status', '?')}"] += 1
                continue
            if row.get("stratum") in ("llm", "seed-dup", "real"):
                src_row = notes.get((row["stratum"], row["key"])) if row["stratum"] != "real" else None
                if row["stratum"] == "real":
                    rec = reals.get(row["key"])
                    assert rec is not None, f"real key not in sources: {row['key'][:12]}"
                    text_src, lang_src = rec["unit"], rec["src_lang"]
                    extra = {"record_type": rec["memory_type"],
                             "created_at": rec["created_at"]}
                else:
                    assert src_row is not None, f"note key not in sources: {row['key'][:12]}"
                    if src_row["direction"] == "ru->en":
                        text_src, lang_src = src_row["ru"], "ru"
                    else:
                        text_src, lang_src = src_row["en"], "en"
                    extra = {"note_key": src_row["key"],
                             "authors": src_row["authors"],
                             "license": src_row["license"],
                             "tatoeba_sentence_ids": src_row["tatoeba_sentence_ids"]}
                text_dst = normalise(row["text"])
                sha_src, sha_dst = text_hash(text_src), text_hash(text_dst)
                pair_key = text_hash(f"{sha_src}\x00{sha_dst}")
                pool.append({
                    "stratum": row["stratum"], "pair_key": pair_key,
                    "direction": f"{lang_src}->{row['target_lang']}",
                    "text_src": text_src, "lang_src": lang_src,
                    "text_dst": text_dst, "lang_dst": row["target_lang"],
                    "sha_src": sha_src, "sha_dst": sha_dst,
                    "cos": row.get("cos"),
                    "gen_file": shard.name, "gen_line": ln, **extra,
                })
            else:  # twin
                ru_t, en_t = normalise(row["ru"]), normalise(row["en"])
                sha_ru, sha_en = text_hash(ru_t), text_hash(en_t)
                # v1 convention: direction parity over the TEXT-pair hash
                parity = int(text_hash(f"{ru_t}\x00{en_t}"), 16) % 2
                if parity == 0:
                    text_src, lang_src, text_dst, lang_dst = ru_t, "ru", en_t, "en"
                    sha_src, sha_dst = sha_ru, sha_en
                else:
                    text_src, lang_src, text_dst, lang_dst = en_t, "en", ru_t, "ru"
                    sha_src, sha_dst = sha_en, sha_ru
                pair_key = text_hash(f"{sha_src}\x00{sha_dst}")
                pool.append({
                    "stratum": "twin", "pair_key": pair_key,
                    "direction": f"{lang_src}->{lang_dst}",
                    "text_src": text_src, "lang_src": lang_src,
                    "text_dst": text_dst, "lang_dst": lang_dst,
                    "sha_src": sha_src, "sha_dst": sha_dst,
                    "cos": row.get("cos"),
                    "gen_file": shard.name, "gen_line": ln,
                    "twin_job": row["job"], "family": row["family"],
                })
            stats[f"{pool[-1]['stratum']}:ok"] += 1

    # uniqueness + disjointness asserts
    seen_pk = set()
    deduped = []
    for c in pool:
        if c["pair_key"] in seen_pk:
            stats[f"{c['stratum']}:dup-pair"] += 1
            continue
        seen_pk.add(c["pair_key"])
        deduped.append(c)
    llm_keys = {c["sha_src"] for c in deduped if c["stratum"] == "llm"}
    seed_keys = {c["sha_src"] for c in deduped if c["stratum"] == "seed-dup"}
    assert not (llm_keys & seed_keys), "llm/seed-dup source notes overlap"
    all_sides = [s for c in deduped for s in (c["sha_src"], c["sha_dst"])]
    assert len(all_sides) == len(set(all_sides)), "pool side hash collision"

    deduped.sort(key=lambda c: (c["stratum"], c["pair_key"]))
    cells = Counter((c["stratum"], c["direction"]) for c in deduped)
    say("assemble", f"frozen pool: {len(deduped)} pairs; stats={dict(sorted(stats.items()))}")
    say("assemble", f"pool cells: {dict(sorted(cells.items()))}")
    need = {"llm": (62, 63), "seed-dup": (20, 20), "twin": (26, 26), "real": (0, 6)}
    # real pool = 6: the 7th clean record is untranslatable under the
    # ratified teacher contract (echo-class failure on all 3 passes);
    # quota seats stay 7, selection fills available cells only.
    for strat, (n_en, n_ru) in need.items():
        have_en = cells.get((strat, "en->ru"), 0)
        have_ru = cells.get((strat, "ru->en"), 0)
        assert have_en >= n_en and have_ru >= n_ru, (
            f"cell shortfall {strat}: en->ru {have_en}/{n_en}, ru->en {have_ru}/{n_ru}")
    say("assemble", "cell coverage asserts PASS (post-gate pools cover quotas)")

    pool_path = OUT_DIR / "frozen_pool.jsonl"
    with open(pool_path, "w", encoding="utf-8") as fh:
        for c in deduped:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    fp = sha256_file(pool_path)
    say("assemble", f"frozen_pool.jsonl fp={fp} (POOL FROZEN — generation never re-runs)")
    return 0


# ── selection (deterministic; double-run byte-identical) ─────────────────────

def run_selection(pool: list[dict], seed: int) -> list[dict]:
    pool_by_cell: dict[tuple[str, str], list[dict]] = {}
    for c in pool:
        pool_by_cell.setdefault((c["stratum"], c["direction"]), []).append(c)
    for key in pool_by_cell:
        pool_by_cell[key].sort(key=lambda c: c["pair_key"])

    strata = sorted({c["stratum"] for c in pool})
    rng = random.Random(seed)
    selected: list[dict] = []
    expected_total = 0
    for s in strata:
        q = QUOTA[s]
        if s == "real":
            # deviation rule: seats fill available cells only (en->ru empty)
            n_ru = min(q, len(pool_by_cell.get((s, "ru->en"), [])))
            n_en = min(q - n_ru, len(pool_by_cell.get((s, "en->ru"), [])))
        else:
            n_ru = q // 2 + q % 2
            n_en = q // 2
        expected_total += n_ru + n_en
        for direction, k in (("en->ru", n_en), ("ru->en", n_ru)):
            cell = pool_by_cell.get((s, direction), [])
            k = min(k, len(cell))
            if k:
                selected.extend(rng.sample(cell, k))
    assert len(selected) == expected_total, (
        f"selected {len(selected)} != expected {expected_total}")
    selected.sort(key=lambda c: c["pair_key"])
    for i, c in enumerate(selected, 1):
        c = dict(c)
        c["pair_id"] = f"R5G2-{i:04d}"
        selected[i - 1] = c
    return selected


def gate_set_bytes(selected: list[dict]) -> bytes:
    out = []
    for c in selected:
        out.append(json.dumps(
            {
                "pair_id": c["pair_id"],
                "source_class": c["stratum"],
                "direction": c["direction"],
                "text_src": c["text_src"],
                "lang_src": c["lang_src"],
                "text_dst": c["text_dst"],
                "lang_dst": c["lang_dst"],
                "text_sha_src": c["sha_src"],
                "text_sha_dst": c["sha_dst"],
                "provenance": {
                    "gen_file": c["gen_file"],
                    "gen_line": c["gen_line"],
                    "pair_key": c["pair_key"],
                },
                "corpus_row_src": None,
                "corpus_row_dst": None,
            },
            ensure_ascii=False,
        ) + "\n")
    return "".join(out).encode("utf-8")


def stage_select() -> int:
    pool_path = OUT_DIR / "frozen_pool.jsonl"
    pool = load_jsonl(pool_path)
    say("select", f"pool loaded: {len(pool)} pairs, fp={sha256_file(pool_path)[:16]}")

    # double-run: two independent selections from the frozen pool
    run_a = run_selection(pool, SEED_V2)
    bytes_a = gate_set_bytes(run_a)
    run_b = run_selection(pool, SEED_V2)
    bytes_b = gate_set_bytes(run_b)
    assert bytes_a == bytes_b, "double-run diverged"
    say("select", f"double-run byte-identical PASS (seed={SEED_V2}, legacy also recorded: {SEED_LEGACY})")

    (OUT_DIR / "gate_set.jsonl").write_bytes(bytes_a)
    (OUT_DIR / "gate_set.run-b.jsonl").write_bytes(bytes_b)
    counts_class = Counter(c["stratum"] for c in run_a)
    counts_dir = Counter(c["direction"] for c in run_a)
    counts_cd = Counter((c["stratum"], c["direction"]) for c in run_a)
    say("select", f"selected: {len(run_a)}; classes={dict(sorted(counts_class.items()))}; "
                  f"directions={dict(sorted(counts_dir.items()))}")
    say("select", f"class x direction: {dict(sorted(counts_cd.items()))}")
    say("select", f"gate_set.jsonl sha256={hashlib.sha256(bytes_a).hexdigest()}")
    return 0


def stage_extend1() -> int:
    """Deterministic source extension-1 (pre-registered rule): the echo
    class of failures proved seed-independent (pass-2 recovery ~6%), so
    cells are closed with FRESH clean notes, never by re-prompting.
    llm ext = pool ranks 330..389, seed ext = 430..459, + 4 twin jobs
    (t2-{fam}-10x). Byte-recorded fp; written to sources_extension1.jsonl."""
    pool, _eh, _eg, _g1 = clean_note_pool(lambda m: say("extend1", m))
    rows = []
    for rank in range(330, 390):
        n = pool[rank]
        members = n["members"]
        rows.append({"kind": "note", "pool": "llm", "rank": rank, "key": n["key"],
                     "direction": n["direction"], "ru": n["ru"], "en": n["en"],
                     "tatoeba_sentence_ids": [r[0] for r in members] + [r[1] for r in members],
                     "authors": sorted({r[4] for r in members} | {r[5] for r in members}),
                     "license": "CC-BY-2.0-FR"})
    for rank in range(430, 460):
        n = pool[rank]
        members = n["members"]
        rows.append({"kind": "note", "pool": "seed-dup", "rank": rank, "key": n["key"],
                     "direction": n["direction"], "ru": n["ru"], "en": n["en"],
                     "tatoeba_sentence_ids": [r[0] for r in members] + [r[1] for r in members],
                     "authors": sorted({r[4] for r in members} | {r[5] for r in members}),
                     "license": "CC-BY-2.0-FR"})
    for fam in TWIN_FAMILIES:
        rows.append({"kind": "twin_job", "job": f"t2-{fam}-100", "family": fam,
                     "npairs": TWIN_PER_JOB})
    out = OUT_DIR / "sources_extension1.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    from collections import Counter as _C
    say("extend1", f"extension-1 written: {len(rows)} rows "
                   f"(llm 60, seed 30, twin jobs 4); fp={sha256_file(out)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sources", "extend1", "assemble", "select", "proofs"])
    args = ap.parse_args()
    if args.stage == "sources":
        raise SystemExit(stage_sources())
    elif args.stage == "extend1":
        raise SystemExit(stage_extend1())
    elif args.stage == "assemble":
        raise SystemExit(stage_assemble())
    elif args.stage == "select":
        raise SystemExit(stage_select())
    raise SystemExit(stage_proofs())
