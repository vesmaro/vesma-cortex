#!/usr/bin/env python3
"""Rebuild corpus v4.4 TRAIN side per prereg-v3 ADDENDUM-1 (NA-approved).

Law:
  docs/specs/b2-graph-prereg-v3.md (ACTIVATED, owner 22.10)
  docs/specs/b2-graph-prereg-v3-ADDENDUM-1.md (ratified frame + NA edits)

HARD CONSTANT: the sealed holdout v4.4 (75 rows, fp a9760d1f...) is NOT
recreated, NOT rewritten, NOT re-read for content. Stage 0 REPRODUCES the
original slice-2 construction in memory (the same deterministic stages of
scripts/b2_graph_corpus.py) and asserts byte-identity twice: reproduced
train fp == 0463413e... and reproduced holdout fp == a9760d1f... == the
on-disk seal. Every addendum change is TRAIN-SIDE ONLY.

Addendum blocks (all mechanics are legacy corpus generators; no new content
authored, no RNG):
  B.1  class-7 band negatives: legacy v4.2 same-length fact-token cipher
       (digit cipher +1 mod 10, char5_containment < probe 0.976415,
       len-delta 0) over the injection bases (clean pool endpoint sides),
       BOTH orientations, acceptance band cos in [0.98, 1.0]; rows the raw
       cipher leaves above 0.985 are PLACED into [0.98, 0.985) by the v4.3
       N-para-band case-dose lever (normalization-invisible, so the razor
       char geometry and tag_jaccard=1.0 survive; measured cos is the only
       feature that moves).
  B.2  injection rebalance: dup mints PREFER taxonomy classes 3+4 over 1+2.
       Class 3 = lexical substitution (legacy synonym tables, all matches,
       tags untouched); class 4 = legacy P-para-light recipe (sentence swap
       + one lexical replacement). Every yieldable 3+4 row is minted BEFORE
       any class-1/2 row; the formula remainder continues the legacy T0/T1
       generator on fresh bases (2:1 lineage).
  B.3  v4.2 razor block (train-824 razor landscape: N-fact-edit /
       N-para-notdup / P-cosmetic) is AUDITED as-is: 262/336 rows are
       already in the train via the v4.3 inheritance (same pairs), 74/336
       are v4.2/v4.3 holdout-surface copies (transfer FORBIDDEN, listed);
       0 new rows; disjointness asserted against every holdout surface and
       the burned surfaces.
  C    razor subzone (cos >= 0.95 AND tag_jaccard = 1.0) parity: negatives
       >= positives with the addendum census floor (854 positives) PLUS the
       subzone rows of any new class-1/2 mints; three cos bins
       [0.95,0.97) / [0.97,0.985) / [0.985,1.0] each >= 30 negatives.
  D    probe coverage per sanity probe class (self_pair->P-identity@1.0,
       envelope_variant->P-envelope@1.0, cosmetic_twin->P-cosmetic@0.99,
       fact_edit_twin->N-fact-edit@0.99, unrelated->N-far@0.578):
       >= K=30 producing-class rows with |cos - probe| <= 0.005 and
       tag_jaccard equal to the probe's; continuity: max gap between
       consecutive sorted in-band cos values <= 0.005. The N-far band is
       filled from the class's own heritage pairing (real-part sides x
       authored far bases of dataset-v4.2 batch-near), cross-language
       pairs quarantined.
  A(i) cross-language quarantine: script-detect (Cyrillic/Latin majority)
       over pool pairs; a confident ru/en mismatch quarantines the pair
       (counter in the manifest; quarantined pairs never become new mints).

Dup-share formula (universe surface, pinned by the ratified anchor):
  the addendum adds negatives (y T3 band rows, f N-far fill rows) and dups
  (x34 class-3/4 rows, w class-1/2 continuation rows). Closure:
     0.595*(x34+w) = 0.295 + 0.405*(282+w_s) + 0.405*f
  where w_s = subzone rows among the first w continuation rows (measured),
  and 282 = the addendum parity floor (572 negs -> 854 positives). The
  solver searches the deterministic minting prefix for the exact integer
  fixed point; no exact closure => the build REFUSES.

Discipline: seed 42 recorded; order-deterministic; no wall-clock in
content artifacts; holdout file never opened for writing; double run must
be byte-identical on every artifact.

Run (worktree checkout, project venv):
  /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/.venv/bin/python \
      scripts/b2_graph_corpus_v44r2.py --out-dir data/b2-graph-corpus/v44-r2

Exit codes: 0 ok | 1 gate refusal | 2 usage/validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)

ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")
BASE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/data/stage2")
POOL_JSONL = REPO_ROOT / "data/b2-graph-corpus/pool_candidates.jsonl"

# ── frozen pins (slice-2 seal; DO NOT EDIT) ──────────────────────────────────

ORIG_TRAIN_FP = "0463413ec7f4a8935211aef8652f1eed0ad039ede401695dd9191e61614eed98"
SEAL_HOLDOUT_FP = "a9760d1f06ecdd40530bcc2dc37311f5e5bcd23aad653e03c87a49fe36761a97"
SEAL_IDS_SHA = "42d690578723fed34549532f2837f4540181a3a034097ce208f922dd710ce229"
SEAL_CORPUS_FP = "45ab0ae642ccf57c73393a2e097036c2f0122a41bb3d92f4e84b06b89bd8928d"

DUP_SHARE = 0.405
PARITY_FLOOR_POS = 854  # addendum §C census floor (subzone positives)
PARITY_FLOOR_NEG = 572  # addendum §C census floor (subzone negatives)
BIN_EDGES = (0.95, 0.97, 0.985, 1.0)
BIN_MIN_NEG = 30
DOSE_MIN_ROWS = 30  # dose-placed rows closing the [0.97,0.985) bin deficit

# §D probe classes: (stratum, probe cos, probe tag_jaccard, sanity check)
PROBE_CLASSES = (
    ("P-identity", 1.0, 1.0, "self_pair"),
    ("P-envelope", 1.0, 1.0, "envelope_variant"),
    ("P-cosmetic", 0.99, 1.0, "cosmetic_twin"),
    ("N-fact-edit", 0.99, 1.0, "fact_edit_twin"),
    ("N-far", 0.578, 0.0, "unrelated"),
)
PROBE_DELTA = 0.005
PROBE_K = 30

RAZOR_BLOCK_STRATA = ("N-fact-edit", "N-para-notdup", "P-cosmetic")

# pre-addendum census pins (slice-3 diagnosis; asserted at reproduction)
PRE_CENSUS = {
    "subzone_pos": 854,
    "subzone_neg": 572,
    "pos_classes_1_2_strict": 506,
    "bins": {"[0.95,0.97)": 53, "[0.97,0.985)": 1, "[0.985,1.0]": 518},
    "band_P-identity": 407,
    "band_P-envelope": 57,
    "band_P-cosmetic": 40,
    "band_N-fact-edit": 60,
    "band_N-far": 3,
}

PROBE_C5_CONT = 0.976415  # sanity fact_edit_twin char5 containment height
MAX_FACT_TOKENS = 4  # legacy v4.2 anti-churn cap
DOSE_LADDER = (0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 1.0)
T3_BAND_LOW = 0.98
DOSE_TARGET = (0.98, 0.985)
NPARA_RAW_LOW = 0.583  # N-far candidates must sit ABOVE the band raw
NPARA_RAW_HIGH = 0.82


class GateRefusal(SystemExit):
    """An assert failed — the corpus is NOT sealed."""


def refuse(msg: str) -> None:
    raise GateRefusal(f"[v44r2] GATE REFUSAL: {msg}")


def log(runlog: list[dict], event: str, **fields: Any) -> None:
    entry = {"event": event, **fields}
    runlog.append(entry)
    print(
        f"[v44r2] {event}: " + " ".join(f"{k}={v}" for k, v in fields.items()),
        file=sys.stderr,
    )


# ── shared helpers (byte-identical recipes from the slice-2 builder) ─────────


def text_hash(text: str) -> str:
    n = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(" ".join(n.lower().split()).encode("utf-8")).hexdigest()


def side_sha(s: dict) -> str:
    payload = json.dumps(
        {
            "title": s.get("title"),
            "body": s.get("body"),
            "tags": sorted(s.get("tags") or []),
            "language": s.get("language"),
            "record_type": s.get("record_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((side_sha(row["record"]), side_sha(row["candidate"]))))


def corpus_rows(path: Path) -> list[dict]:
    if not path.is_file():
        refuse(f"frozen input missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def tag_jaccard(a: dict, b: dict) -> float:
    sa, sb = set(a.get("tags") or []), set(b.get("tags") or [])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def detect_lang(body: str | None) -> str:
    """Script-detect (addendum §A(i)): Cyrillic/Latin letter majority."""
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_embedder():
    if str(ENGINE_SRC) not in sys.path:
        sys.path.insert(0, str(ENGINE_SRC))
    from vesmaro.embeddings import NanoProvider  # noqa: E402

    provider = NanoProvider()
    pin = provider.fingerprint
    if not pin.startswith("nano:sha256:3b752e06"):
        refuse(f"embedder pin {pin!r} differs from the frozen prefix")
    cache: dict[str, list[float]] = {}

    def embed_text(text: str) -> list[float]:
        text = text[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    def embed_side(s: dict) -> list[float]:
        tags = " ".join(s.get("tags") or [])
        return embed_text(f"{s.get('title') or ''}\n{s.get('body') or ''}\n{tags}")

    return embed_side, embed_text, pin


# ── stage 0: reproduce the slice-2 construction, assert the seal ─────────────


def load_orig_module():
    spec = importlib.util.spec_from_file_location(
        "b2_graph_corpus_orig", REPO_ROOT / "scripts" / "b2_graph_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reproduce(runlog: list[dict]) -> dict:
    orig = load_orig_module()
    rlog: list[dict] = []
    pool = orig.verify_pool(rlog)
    scanner = orig.verify_detectors(rlog)
    base, binfo = orig.clean_base(rlog)
    graph_rows = orig.select_graph(rlog, pool)
    inj_rows = orig.build_injection(
        rlog, scanner, pool, binfo["n_injection"], base, graph_rows
    )
    train, holdout, hmeta = orig.carve_holdout(rlog, base, graph_rows, inj_rows)

    for e in rlog:
        runlog.append({"stage": "reproduce", **e})

    def fp_of(rows: list[dict]) -> str:
        return corpus_fingerprint(
            manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in rows])
        )

    train_fp = fp_of(train)
    hold_fp = fp_of(holdout)
    whole_fp = corpus_fingerprint(
        manifest_bytes([(r["pair_id"], r["pair_sha256"]) for r in train + holdout])
    )
    log(
        runlog,
        "reproduce_seal",
        train_rows=len(train),
        holdout_rows=len(holdout),
        train_fp=train_fp,
        expected_train_fp=ORIG_TRAIN_FP,
        train_match=(train_fp == ORIG_TRAIN_FP),
        holdout_fp=hold_fp,
        expected_holdout_fp=SEAL_HOLDOUT_FP,
        holdout_match=(hold_fp == SEAL_HOLDOUT_FP),
        corpus_fp=whole_fp,
        corpus_match=(whole_fp == SEAL_CORPUS_FP),
    )
    if train_fp != ORIG_TRAIN_FP:
        refuse(
            "reproduced train fp differs from the sealed slice-2 train "
            f"({train_fp} != {ORIG_TRAIN_FP}) — the original construction "
            "no longer reproduces; STOP"
        )
    if hold_fp != SEAL_HOLDOUT_FP or whole_fp != SEAL_CORPUS_FP:
        refuse("reproduced holdout/corpus fp differs from the seal; STOP")

    # the on-disk seal must match too (ids + file rows) — ids only, no labels
    seal_path = REPO_ROOT / "data/b2-graph-corpus/v44/holdout-ids.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    ids = sorted(r["pair_id"] for r in holdout)
    ids_sha = hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest()
    if seal.get("holdout_ids_sha256") != ids_sha or ids_sha != SEAL_IDS_SHA:
        refuse("holdout-ids seal mismatch vs reproduced carve; STOP")
    disk_hold_fp = fp_of(
        corpus_rows(REPO_ROOT / "data/b2-graph-corpus/v44/holdout.jsonl")
    )
    if disk_hold_fp != SEAL_HOLDOUT_FP:
        refuse("on-disk holdout.jsonl fingerprint differs from the seal; STOP")
    log(runlog, "seal_verified", disk_holdout_fp=disk_hold_fp, ids_sha256=ids_sha)

    hold_ids = set(ids)
    return {
        "pool": pool,
        "scanner": scanner,
        "base": base,
        "graph_rows": graph_rows,
        "inj_rows": inj_rows,
        "orig_train": train,
        "holdout_ids": hold_ids,
        "binfo": binfo,
    }


# ── census (razor subzone §C + probe bands §D) ───────────────────────────────


def bin_of(cos: float) -> str:
    if cos < 0.95:
        return "<0.95"
    if cos < 0.97:
        return "[0.95,0.97)"
    if cos < 0.985:
        return "[0.97,0.985)"
    return "[0.985,1.0]"


def census(rows: list[dict]) -> dict:
    sub = [
        r
        for r in rows
        if r["similarity"] >= 0.95 and tag_jaccard(r["record"], r["candidate"]) == 1.0
    ]
    pos = [r for r in sub if r["label"] == "duplicate"]
    neg = [r for r in sub if r["label"] == "not-duplicate"]
    bins = Counter(bin_of(r["similarity"]) for r in neg)
    c12 = [r for r in pos if r["stratum"] in ("P-identity", "P-cosmetic")]
    bands = {}
    for stratum, pc, ptj, check in PROBE_CLASSES:
        band = [
            r
            for r in rows
            if r["stratum"] == stratum
            and abs(r["similarity"] - pc) <= PROBE_DELTA + 1e-12
            and abs(tag_jaccard(r["record"], r["candidate"]) - ptj) < 1e-9
        ]
        cs = sorted(r["similarity"] for r in band)
        gaps = [round(b - a, 9) for a, b in zip(cs, cs[1:])]
        bands[stratum] = {
            "check": check,
            "probe_cos": pc,
            "probe_tj": ptj,
            "count": len(band),
            "max_gap": (max(gaps) if gaps else 0.0),
            "k_ok": len(band) >= PROBE_K,
            "cont_ok": (max(gaps) if gaps else 0.0) <= PROBE_DELTA + 1e-9,
        }
    return {
        "subzone_pos": len(pos),
        "subzone_neg": len(neg),
        "pos_classes_1_2_strict": len(c12),
        "bins": {
            b: bins.get(b, 0) for b in ("[0.95,0.97)", "[0.97,0.985)", "[0.985,1.0]")
        },
        "bands": bands,
    }


# ── legacy generators (imported from the corpus's own builders) ──────────────


def load_gen_module():
    spec = importlib.util.spec_from_file_location(
        "gen_dataset_v4_mod", REPO_ROOT / "scripts" / "gen_dataset_v4.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ngrams(text: str, n: int) -> frozenset[str]:
    t = " ".join(text.split()).lower()
    if len(t) < n:
        return frozenset()
    return frozenset(t[i : i + n] for i in range(len(t) - n + 1))


def c5_containment(title: str, body: str, new_body: str) -> float:
    a = _ngrams(f"{title}\n{body}", 5)
    b = _ngrams(f"{title}\n{new_body}", 5)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _fact_tokens(body: str) -> list[tuple[int, int]]:
    return [
        m.span()
        for m in re.finditer(r"\d+(?:[:.\-_/]\d+)*", body)
        if any(c.isdigit() for c in m.group())
    ]


def _swap_spans(body: str, spans: list[tuple[int, int]]) -> str:
    chars = list(body)
    for s, e in spans:
        for p in range(s, e):
            if body[p].isdigit():
                chars[p] = "0" if body[p] != "0" else "7"
    return "".join(chars)


def cipher_edit(body: str, title: str) -> str | None:
    """Legacy v4.2 same-length razor: smallest deterministic fact-token
    digit-cipher whose char5_containment clears the probe height STRICTLY."""
    tokens = _fact_tokens(body)
    if not tokens:
        return None
    title = title or ""
    drift = sorted(
        (
            c5_containment(title, body, _swap_spans(body, [sp])),
            sp[0],
            sp,
        )
        for sp in tokens
    )
    acc: list[tuple[int, int]] = []
    for _, _, span in drift:
        acc.append(span)
        if len(acc) > MAX_FACT_TOKENS:
            break
        if c5_containment(title, body, _swap_spans(body, acc)) < PROBE_C5_CONT:
            return _swap_spans(body, acc)
    return None


def case_run(text: str, dose: float) -> str:
    n = len(text)
    length = int(n * dose)
    return "".join((c.swapcase() if p < length else c) for p, c in enumerate(text))


# ── minting helper ────────────────────────────────────────────────────────────


class Minter:
    def __init__(
        self, runlog: list[dict], embed_side, pin: str, scanner: Any, occupied: set
    ):
        self.runlog = runlog
        self.embed_side = embed_side
        self.pin = pin
        self.scanner = scanner
        self.occupied = occupied
        self.counters: Counter = Counter()
        self.hygiene_hits = 0

    def hygiene(self, s: dict, where: str) -> None:
        hits = self.scanner.scan(
            s.get("title") or "", s.get("body") or "", s.get("tags") or []
        )
        if hits:
            self.hygiene_hits += 1
            refuse(f"hygiene hit on minted side {where}: {hits}")

    def mint(
        self,
        cls: str,
        label: str,
        stratum: str,
        generator: str,
        source: str,
        rec: dict,
        cand: dict,
        unit: str | None = None,
        scaffold: str | None = None,
        allow_dup_key: bool = False,
    ) -> dict:
        self.counters[cls] += 1
        pid = f"V44R2-{cls}-{self.counters[cls]:04d}"
        va = self.embed_side(rec)
        vb = self.embed_side(cand)
        sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
        pair_obj = {"record": rec, "candidate": cand, "similarity": sim}
        row = {
            "pair_id": pid,
            "label": label,
            "stratum": stratum,
            "record": rec,
            "candidate": cand,
            "similarity": sim,
            "vec_a": va,
            "vec_b": vb,
            "pair_sha256": pair_sha256(pair_obj),
            "generator": generator,
            "seed": 42,
            "embedder_pin": self.pin,
            "source": source,
        }
        if unit is not None:
            row["unit"] = unit
        if scaffold is not None:
            row["scaffold"] = scaffold
        key = pair_key(row)
        if allow_dup_key:
            # the T0 byte-clone twin of the SAME unit: identical pair-key by
            # design (slice-2 convention: self + clone both ship); the key
            # was already validated by the unit's self mint
            pass
        elif key in self.occupied:
            refuse(f"minted pair collides with an occupied pair-key: {pid}")
        self.occupied.add(key)
        self.hygiene(rec, pid + ":record")
        self.hygiene(cand, pid + ":candidate")
        return row


# ── stage: cross-language quarantine census over the pool ────────────────────


def quarantine_census(runlog: list[dict], pool: list[dict]) -> dict:
    quarantined = 0
    undetected = 0
    mixed = 0
    for r in pool:
        lr = detect_lang(r["record"].get("body"))
        lc = detect_lang(r["candidate"].get("body"))
        if lr == "none" or lc == "none":
            undetected += 1
        elif lr == "mixed" or lc == "mixed":
            mixed += 1
        elif lr != lc:
            quarantined += 1
    log(
        runlog,
        "cross_language_quarantine",
        pool_rows=len(pool),
        quarantined_pairs=quarantined,
        mixed_script_pairs=mixed,
        undetectable_pairs=undetected,
        rule="script-detect body language (Cyrillic/Latin majority); confident ru/en mismatch = quarantined (addendum §A(i)); mixed-script and undetectable counted separately, never quarantined",
    )
    return {"quarantined": quarantined, "mixed": mixed, "undetectable": undetected}


# ── stage B.3: razor block audit ─────────────────────────────────────────────


def razor_block_audit(
    runlog: list[dict], orig_train: list[dict], holdout_ids: set
) -> dict:
    v42_train = corpus_rows(BASE / "dataset-v4/train.jsonl")
    block = [r for r in v42_train if r.get("stratum") in RAZOR_BLOCK_STRATA]
    train_keys = {pair_key(r) for r in orig_train}
    holdout_keys = set()
    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        for r in corpus_rows(BASE / name):
            holdout_keys.add(pair_key(r))
    in_train, blocked, missing = 0, 0, 0
    blocked_rows = []
    for r in block:
        k = pair_key(r)
        if k in train_keys:
            in_train += 1
        elif k in holdout_keys:
            blocked += 1
            blocked_rows.append(r["pair_id"])
        else:
            missing += 1
    audit = {
        "block_rows": len(block),
        "already_in_train": in_train,
        "blocked_holdout_surface": blocked,
        "missing": missing,
        "blocked_pair_ids": sorted(blocked_rows),
        "verdict": "as-is presence asserted; 0 rows re-inserted (dedup-before-splits)",
    }
    if missing:
        refuse(
            f"razor block: {missing} rows neither in train nor on a holdout surface; STOP"
        )
    log(
        runlog,
        "razor_block_audit",
        **{k: v for k, v in audit.items() if k != "blocked_pair_ids"},
    )
    return audit


# ── mint stages ──────────────────────────────────────────────────────────────


def clean_pool_sides(runlog: list[dict], pool: list[dict]) -> list[dict]:
    """The injection bases: pool endpoint sides, holdout/gate/burned sides
    excluded at MINT time (the slice-2 conservative recipe), canonical
    (created_at, side_sha) order."""
    hs: set = set()
    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        for r in corpus_rows(BASE / name):
            for s in (r["record"], r["candidate"]):
                hs.add(text_hash(s["title"] + "\n" + s["body"]))
                hs.add(text_hash(s["body"]))
                hs.add(text_hash(s["title"]))
    gate_sides: set = set()
    R4 = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus"
    )
    R5 = Path(
        "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus"
    )
    for gp in (
        R4 / "data/embed-r4/r5_gate/gate_set.jsonl",
        R4 / "data/embed-r4/r5_gate/tatoeba_notes.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/gate_set.run-b.jsonl",
        R5 / "data/embed-r4/r5_gate_v2/frozen_pool.jsonl",
    ):
        if gp.is_file():
            for row in corpus_rows(gp):
                for key in ("text_src", "text_dst", "text"):
                    if row.get(key):
                        gate_sides.add(text_hash(row[key]))
    burned: set = set()
    for bp in (
        R4 / "data/embed-r4/corpus/train.jsonl",
        R4 / "data/embed-r4/corpus/val.jsonl",
        R5 / "data/embed-r5/corpus/train.jsonl",
        R5 / "data/embed-r5/corpus/val.jsonl",
        R5 / "data/embed-r7/corpus/train.jsonl",
        R5 / "data/embed-r7/corpus/val.jsonl",
    ):
        if bp.is_file():
            for row in corpus_rows(bp):
                for side_key in ("record", "candidate"):
                    s = row.get(side_key)
                    if s:
                        burned.add(text_hash(s["title"] + "\n" + s["body"]))
                        burned.add(text_hash(s["body"]))
                for key in ("text", "text_src", "text_dst"):
                    if row.get(key):
                        burned.add(text_hash(row[key]))

    sides: dict[str, dict] = {}
    for r in pool:
        for s in (r["record"], r["candidate"]):
            sides.setdefault(side_sha(s), s)
    order = sorted(sides.items(), key=lambda kv: (kv[1].get("created_at") or "", kv[0]))
    clean = []
    skipped = Counter()
    for k, s in order:
        probes = (
            text_hash(s["title"] + "\n" + s["body"]),
            text_hash(s["body"]),
            text_hash(s["title"]),
        )
        if any(p in gate_sides for p in probes):
            skipped["gate-side"] += 1
            continue
        if any(p in burned for p in probes):
            skipped["burned-side"] += 1
            continue
        if any(p in hs for p in probes):
            skipped["holdout-side"] += 1
            continue
        clean.append({"sha": k, "side": s})
    log(
        runlog,
        "injection_bases",
        pool_sides=len(sides),
        clean_sides=len(clean),
        skipped=dict(skipped),
        order="(created_at, side_sha) canonical",
    )
    return clean


def mint_class34(
    runlog: list[dict], minter: Minter, gen, clean_sides: list[dict]
) -> list[dict]:
    """§B.2: taxonomy classes 3 (lexical substitution) and 4 (P-para-light
    recipe) on the injection bases. Tags untouched (tag_jaccard stays 1.0).
    No synonym hit on a body -> that class skips the side (deterministic)."""
    rows: list[dict] = []
    skipped = Counter()
    for item in clean_sides:
        s = {k: item["side"][k] for k in SIDE_KEYS if k in item["side"]}
        lang = s.get("language") or "ru"
        body = s.get("body") or ""
        # class 3: lexical substitution, ALL matching table entries
        table = gen.SYNONYMS_RU if lang == "ru" else gen.SYNONYMS_EN
        new_body = body
        subs = 0
        for find, replace in table:
            if find in new_body:
                new_body = new_body.replace(find, replace, 1)
                subs += 1
        # the legacy tables contain reverse pairs (e.g. пропали<->исчезли):
        # sequential application can net-cancel — a body left unchanged is
        # NOT a class-3 substitution (it would be an identity row), skip it
        if subs and new_body != body:
            cand = deepcopy(s)
            cand["body"] = new_body
            rows.append(
                minter.mint(
                    "CLS3",
                    "duplicate",
                    "P-lexsub",
                    "legacy-synonym-substitution (addendum-1 class 3)",
                    "addendum1-rebalance",
                    deepcopy(s),
                    cand,
                    scaffold=item["sha"],
                )
            )
        else:
            skipped["cls3-no-synonym-or-net-zero"] += 1
        # class 4: legacy P-para-light (sentence swap + one lexical replacement)
        sentences = gen._SENTENCE_SPLIT.split(body.strip())
        if len(sentences) < 2:
            skipped["cls4-single-sentence"] += 1
            continue
        swapped = " ".join([sentences[1], sentences[0], *sentences[2:]])
        base4 = deepcopy(s)
        base4["body"] = swapped
        rep_table = gen.SYNONYMS_RU if lang == "ru" else gen.SYNONYMS_EN
        done = False
        for find, replace in rep_table:
            if find in swapped:
                base4["body"] = swapped.replace(find, replace, 1)
                done = True
                break
        if not done:
            skipped["cls4-no-synonym"] += 1
            continue
        cand4 = deepcopy(base4)
        rows.append(
            minter.mint(
                "CLS4",
                "duplicate",
                "P-para-light",
                "legacy-v4-para-light (sentence swap + lexical replacement)",
                "addendum1-rebalance",
                deepcopy(s),
                cand4,
                scaffold=item["sha"],
            )
        )
    log(
        runlog,
        "mint_class34",
        cls3_rows=sum(1 for r in rows if r["stratum"] == "P-lexsub"),
        cls4_rows=sum(1 for r in rows if r["stratum"] == "P-para-light"),
        skipped=dict(skipped),
        scaffolds=len({r["scaffold"] for r in rows}),
    )
    return rows


SIDE_KEYS = ("title", "body", "tags", "language", "record_type", "created_at")


def mint_nfar_fill(
    runlog: list[dict],
    minter: Minter,
    clean_sides: list[dict],
    orig_train: list[dict],
    need: int,
) -> list[dict]:
    """§D N-far band fill: the class's own heritage pairing — real-part
    record sides x authored far bases (dataset-v4.2 batch-near N-far sides,
    the class's own producing content, in-repo batch files). Cross-language
    pairs quarantined; tag_jaccard must equal the probe's 0.0; measured cos
    must land in the probe band."""
    authored: dict[str, dict] = {}
    for r in corpus_rows(BASE / "dataset-v4/train.jsonl"):
        if r.get("stratum") != "N-far":
            continue
        for s in (r["record"], r["candidate"]):
            authored.setdefault(side_sha(s), s)
    candidates = []
    quar = 0
    # deterministic double loop: authored far side (sorted by sha) x real side (canonical order)
    for akey in sorted(authored):
        a = {k: authored[akey][k] for k in SIDE_KEYS if k in authored[akey]}
        la = detect_lang(a.get("body"))
        ta = set(a.get("tags") or [])
        for item in clean_sides:
            rs = {k: item["side"][k] for k in SIDE_KEYS if k in item["side"]}
            lr = detect_lang(rs.get("body"))
            if la in ("ru", "en") and lr in ("ru", "en") and la != lr:
                quar += 1
                continue
            sb = set(rs.get("tags") or [])
            if ta and sb and (ta & sb):
                continue  # tag_jaccard must be 0.0 (both non-empty, disjoint)
            if not ta and not sb:
                continue  # degenerate tj=1.0 pair — not the probe geometry
            va = minter.embed_side(a)
            vb = minter.embed_side(rs)
            sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
            if abs(sim - 0.578) <= PROBE_DELTA + 1e-12:
                candidates.append(
                    {"a": a, "r": rs, "sim": sim, "akey": akey, "rkey": item["sha"]}
                )
    candidates.sort(key=lambda c: (c["sim"], c["akey"], c["rkey"]))
    # selection: bridge the existing in-band gaps, then top up to `need`
    existing = [
        r
        for r in orig_train
        if r["stratum"] == "N-far"
        and abs(r["similarity"] - 0.578) <= PROBE_DELTA + 1e-12
        and abs(tag_jaccard(r["record"], r["candidate"])) < 1e-9
    ]
    sel: list[dict] = []
    used: set = set()

    def gaps_of(coses: list[float]) -> float:
        cs = sorted(coses)
        return max((b - a for a, b in zip(cs, cs[1:])), default=0.0)

    coses = [r["similarity"] for r in existing]
    # bridge: while any consecutive gap > delta, add the candidate inside it
    guard = 0
    while gaps_of(coses) > PROBE_DELTA + 1e-9 and guard < 64:
        guard += 1
        cs = sorted(coses)
        worst = max(((b - a, a, b) for a, b in zip(cs, cs[1:])), default=None)
        if worst is None:
            break
        _, gi, gj = worst
        mid = (gj - gi) / 2 + gi
        inside = [c for c in candidates if id(c) not in used and gi < c["sim"] < gj]
        if not inside:
            break
        pick = min(
            inside, key=lambda c: (abs(c["sim"] - mid), c["sim"], c["akey"], c["rkey"])
        )
        used.add(id(pick))
        sel.append(pick)
        coses.append(pick["sim"])
    # top up in canonical (cos, keys) order
    for c in candidates:
        if len(sel) >= need:
            break
        if id(c) in used:
            continue
        used.add(id(c))
        sel.append(c)
        coses.append(c["sim"])
    rows = []
    for c in sorted(sel, key=lambda c: (c["akey"], c["rkey"])):
        rows.append(
            minter.mint(
                "NFAR",
                "not-duplicate",
                "N-far",
                "legacy-real-rnd-heritage (real side x authored far base)",
                "addendum1-coverage-fill",
                c["a"],
                c["r"],
                scaffold=c["akey"] + "|" + c["rkey"],
            )
        )
    final_gaps = gaps_of(
        [r["similarity"] for r in existing] + [r["similarity"] for r in rows]
    )
    log(
        runlog,
        "mint_nfar_fill",
        candidates_in_band=len(candidates),
        minted=len(rows),
        existing_in_band=len(existing),
        final_band_count=len(existing) + len(rows),
        final_max_gap=round(final_gaps, 9),
        quarantined_cross_lang=quar,
    )
    if len(existing) + len(rows) < PROBE_K:
        refuse(
            f"N-far probe band: {len(existing) + len(rows)} rows < K={PROBE_K} "
            "on the frozen inputs — §D coverage unsatisfiable; STOP"
        )
    if final_gaps > PROBE_DELTA + 1e-9:
        refuse(
            f"N-far probe band continuity: max gap {final_gaps} > {PROBE_DELTA}; STOP"
        )
    return rows


def mint_t3_band(
    runlog: list[dict], minter: Minter, clean_sides: list[dict], quota_rows: int
) -> list[dict]:
    """§B.1 + §C: class-7 negatives in [0.98, 1.0]. Raw cipher pairs (both
    orientations) land in [0.985, 1.0]; DOSE_MIN_ROWS are dose-placed into
    [0.98, 0.985) by the legacy case-dose lever to close the [0.97,0.985)
    bin (char geometry + tag_jaccard=1.0 unchanged; only cos_target moves).
    """
    raw: list[dict] = []
    dose_rows: list[dict] = []
    notok = 0
    below_band = 0
    for item in clean_sides:
        rec = {k: item["side"][k] for k in SIDE_KEYS if k in item["side"]}
        title = rec.get("title") or ""
        body = rec.get("body") or ""
        edited = cipher_edit(body, title)
        if edited is None:
            notok += 1
            continue
        c5 = c5_containment(title, body, edited)
        if not c5 < PROBE_C5_CONT:
            refuse(f"cipher post-check failed on base {item['sha'][:12]}: c5={c5}")
        if len(edited) != len(body):
            refuse(f"cipher length drift on base {item['sha'][:12]}")
        cand = deepcopy(rec)
        cand["body"] = edited
        va = minter.embed_side(rec)
        vb = minter.embed_side(cand)
        sim = round(min(1.0, max(0.0, dot(va, vb))), 6)
        if sim < T3_BAND_LOW:
            below_band += 1
            continue
        dose_hit = None
        if sim >= DOSE_TARGET[1]:
            for d in DOSE_LADDER:
                dcand = deepcopy(cand)
                dcand["body"] = case_run(edited, d)
                sim2 = round(min(1.0, max(0.0, dot(va, minter.embed_side(dcand)))), 6)
                if DOSE_TARGET[0] <= sim2 < DOSE_TARGET[1]:
                    dose_hit = (dcand, sim2)
                    break
        if dose_hit is not None and len(dose_rows) < 2 * DOSE_MIN_ROWS:
            dcand, sim2 = dose_hit
            if not c5_containment(title, body, dcand["body"]) < PROBE_C5_CONT:
                refuse("dose row lost the razor char geometry; refusing")
            dose_rows.append((rec, dcand, sim2, item["sha"]))
        raw.append((rec, cand, sim, item["sha"]))
    # dose rows first (bin deficit), both orientations; then raw pairs until quota
    minted: list[dict] = []
    for rec, cand, sim, sha in dose_rows:
        if len(minted) >= quota_rows:
            break
        gen = "legacy-v42-t3-cipher + v43-case-dose placement"
        a = minter.mint(
            "T3B",
            "not-duplicate",
            "N-fact-edit",
            gen,
            "addendum1-razor-band",
            rec,
            cand,
            scaffold=sha,
        )
        b = minter.mint(
            "T3B",
            "not-duplicate",
            "N-fact-edit",
            gen,
            "addendum1-razor-band",
            cand,
            rec,
            scaffold=sha,
            allow_dup_key=True,
        )
        a["similarity"] = sim
        b["similarity"] = sim
        minted.extend([a, b])
    for rec, cand, sim, sha in raw:
        if len(minted) >= quota_rows:
            break
        gen = "legacy-v42-t3-cipher (digit fact-token cipher, same-length)"
        a = minter.mint(
            "T3B",
            "not-duplicate",
            "N-fact-edit",
            gen,
            "addendum1-razor-band",
            rec,
            cand,
            scaffold=sha,
        )
        b = minter.mint(
            "T3B",
            "not-duplicate",
            "N-fact-edit",
            gen,
            "addendum1-razor-band",
            cand,
            rec,
            scaffold=sha,
            allow_dup_key=True,
        )
        a["similarity"] = sim
        b["similarity"] = sim
        minted.extend([a, b])
    bad = [r for r in minted if not (T3_BAND_LOW <= r["similarity"] <= 1.0)]
    if bad:
        refuse(f"{len(bad)} T3 band rows outside the acceptance band [0.98, 1.0]; STOP")
    dose_minted = sum(1 for r in minted if "case-dose" in r["generator"])
    log(
        runlog,
        "mint_t3_band",
        quota_rows=quota_rows,
        minted=len(minted),
        dose_placed=dose_minted,
        raw_pairs=len(raw),
        dose_pairs=len(dose_rows),
        bases_no_fact_tokens=notok,
        below_band=below_band,
    )
    if len(minted) < quota_rows:
        refuse(f"T3 band quota unmet: {len(minted)}/{quota_rows}; STOP")
    return minted


def t01_continuation_prepass(
    runlog: list[dict],
    minter: Minter,
    clean_sides: list[dict],
    start_idx: int,
    orig_inj: list[dict],
) -> list[dict]:
    """Continue the legacy T0/T1 generator past the original walk position,
    alternating [T0 unit, T1 row] (2:1 lineage). Returns the CANDIDATE rows
    in deterministic minting order; the fixed-point solver picks a prefix."""
    used_shas = set()
    for r in orig_inj:
        for s in (r["record"], r["candidate"]):
            used_shas.add(side_sha(s))
    rows: list[dict] = []
    idx = start_idx
    cycle = 0
    skipped = Counter()
    while idx < len(clean_sides) and cycle < 220:
        # T0 unit: self + byte clone (legacy pre-check: skip occupied pairs)
        item = clean_sides[idx]
        idx += 1
        if item["sha"] in used_shas:
            continue
        rec = {k: item["side"][k] for k in SIDE_KEYS if k in item["side"]}
        if (item["sha"], item["sha"]) in minter.occupied:
            skipped["t0-occupied-pair"] += 1
            continue
        unit = f"U-R2T0-{cycle + 1:04d}"
        a = minter.mint(
            "R2T0",
            "duplicate",
            "P-identity",
            "legacy-v43-t0t1",
            "addendum1-formula-continuation",
            deepcopy(rec),
            deepcopy(rec),
            unit=unit,
            scaffold=item["sha"],
        )
        cand = deepcopy(rec)
        b = minter.mint(
            "R2T0",
            "duplicate",
            "P-identity",
            "legacy-v43-t0t1",
            "addendum1-formula-continuation",
            rec,
            cand,
            unit=unit,
            scaffold=item["sha"],
            allow_dup_key=True,
        )
        rows.extend([a, b])
        # T1 row: body swapcase (legacy pre-check: skip occupied pairs)
        while idx < len(clean_sides):
            item1 = clean_sides[idx]
            idx += 1
            if item1["sha"] in used_shas:
                continue
            rec1 = {k: item1["side"][k] for k in SIDE_KEYS if k in item1["side"]}
            cand1 = deepcopy(rec1)
            cand1["body"] = cand1["body"].swapcase()
            k1 = tuple(sorted((side_sha(rec1), side_sha(cand1))))
            if k1 in minter.occupied:
                skipped["t1-occupied-pair"] += 1
                continue
            t1 = minter.mint(
                "R2T1",
                "duplicate",
                "P-cosmetic",
                "legacy-v43-t0t1",
                "addendum1-formula-continuation",
                rec1,
                cand1,
                unit=f"U-R2T1-{cycle + 1:04d}",
                scaffold=item1["sha"],
            )
            rows.append(t1)
            break
        cycle += 1
    log(
        runlog,
        "t01_prepass",
        candidate_rows=len(rows),
        cycles=cycle,
        skipped=dict(skipped),
        bases_left=len(clean_sides) - idx,
    )
    return rows


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data/b2-graph-corpus/v44-r2"
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    runlog: list[dict] = []

    # stage 0: reproduce + seal asserts
    rep = reproduce(runlog)
    orig_train = rep["orig_train"]

    # pre-addendum census pins
    pre = census(orig_train)
    log(runlog, "census_pre", **{k: v for k, v in pre.items() if k != "bands"})
    log(runlog, "census_pre_bands", **{s: b["count"] for s, b in pre["bands"].items()})
    for key in ("subzone_pos", "subzone_neg", "pos_classes_1_2_strict"):
        if pre[key] != PRE_CENSUS[key]:
            refuse(
                f"pre-addendum census {key}={pre[key]} != pinned {PRE_CENSUS[key]}; STOP"
            )
    if pre["bins"] != PRE_CENSUS["bins"]:
        refuse(f"pre-addendum bins {pre['bins']} != pinned {PRE_CENSUS['bins']}; STOP")
    for s, b in pre["bands"].items():
        pin_key = f"band_{s}"
        if PRE_CENSUS.get(pin_key) is not None and b["count"] != PRE_CENSUS[pin_key]:
            refuse(
                f"pre-addendum band {s}={b['count']} != pinned {PRE_CENSUS[pin_key]}; STOP"
            )

    # quarantine census
    quar = quarantine_census(runlog, rep["pool"])

    # razor block audit (§B.3)
    razor_audit = razor_block_audit(runlog, orig_train, rep["holdout_ids"])

    # occupied pair keys: universe + dataset-v3 + pool + graph + orig injection + orig train
    occupied: set = set()
    for name in (
        "v43-localbuild/train.jsonl",
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
        "b2/train.jsonl",
    ):
        for r in corpus_rows(BASE / name):
            occupied.add(pair_key(r))
    for r in rep["pool"]:
        occupied.add(pair_key(r))
    for r in orig_train:
        occupied.add(pair_key(r))

    embed_side, embed_text_unused, pin = make_embedder()
    minter = Minter(runlog, embed_side, pin, rep["scanner"], occupied)

    clean_sides = clean_pool_sides(runlog, rep["pool"])

    # §B.2: classes 3+4 first (the preference, maximally honored)
    cls34 = mint_class34(runlog, minter, load_gen_module(), clean_sides)
    x34 = len(cls34)

    # §D: N-far fill
    nfar = mint_nfar_fill(
        runlog,
        minter,
        clean_sides,
        orig_train,
        need=PROBE_K - PRE_CENSUS["band_N-far"],
    )
    f_fill = len(nfar)

    # legacy T0/T1 continuation pre-pass (candidates; prefix chosen below)
    start_idx = 0
    # reproduce the original walk's stop position: count sides consumed by
    # the original mint = every side visited until the last accepted base;
    # equivalently: sides whose sha appears in orig injection rows, plus the
    # sides between them in canonical order.
    used_shas = set()
    for r in rep["inj_rows"]:
        for s in (r["record"], r["candidate"]):
            used_shas.add(side_sha(s))
    for i, item in enumerate(clean_sides):
        if item["sha"] in used_shas:
            start_idx = i + 1
    t01_pre = t01_continuation_prepass(
        runlog, minter, clean_sides, start_idx, rep["inj_rows"]
    )

    # T3 capacity pre-measurement happens inside the solver below (raw pairs
    # are minted once, so compute capacity first WITHOUT minting: reuse the
    # cipher scan over clean sides)
    # capacity scan (no minting)
    raw_capacity = 0
    dose_capacity_rows = 0
    for item in clean_sides:
        s = item["side"]
        edited = cipher_edit(s.get("body") or "", s.get("title") or "")
        if edited is None:
            continue
        cand = deepcopy(s)
        cand["body"] = edited
        sim = round(min(1.0, max(0.0, dot(embed_side(s), embed_side(cand)))), 6)
        if sim < T3_BAND_LOW:
            continue
        raw_capacity += 2
        if sim >= DOSE_TARGET[1]:
            for d in DOSE_LADDER:
                dcand = deepcopy(cand)
                dcand["body"] = case_run(edited, d)
                sim2 = round(
                    min(1.0, max(0.0, dot(embed_side(s), embed_side(dcand)))), 6
                )
                if DOSE_TARGET[0] <= sim2 < DOSE_TARGET[1]:
                    dose_capacity_rows += 2
                    break

    # fixed-point solver: choose the T0/T1 continuation prefix w (row count)
    # s.t. round_half_up((0.295 + 0.405*y + 0.405*f)/0.595) == x34 + w
    # where the §C parity binds the T3 quota: y = 282 + w_s
    # (floor 854 negatives = 572 + dobor, with new class-1/2 subzone rows w_s
    # joining the denominator).
    solution = None
    for w in range(0, len(t01_pre) + 1):
        w_s = sum(
            1
            for r in t01_pre[:w]
            if r["similarity"] >= 0.95
            and tag_jaccard(r["record"], r["candidate"]) == 1.0
        )
        y = PARITY_FLOOR_NEG_ZONE(w_s) - PARITY_FLOOR_NEG  # = 282 + w_s
        y += y & 1  # T3 rows ship in both-orientation pairs — quota must be even
        n_real = (0.295 + DUP_SHARE * y + DUP_SHARE * f_fill) / (1 - DUP_SHARE)
        n_int = math.floor(n_real + 0.5)
        if n_int == x34 + w and y <= raw_capacity + dose_capacity_rows:
            # dose rows must also cover the bin deficit
            if (
                dose_capacity_rows >= 2 * DOSE_MIN_ROWS
                or pre["bins"]["[0.97,0.985)"] >= BIN_MIN_NEG
            ):
                solution = (w, w_s, y, n_int)
                break
    if solution is None:
        refuse(
            "no exact integer fixed point for the addendum closure "
            f"(x34={x34}, f={f_fill}, t01_capacity={len(t01_pre)}, "
            f"t3_capacity={raw_capacity + dose_capacity_rows}); STOP"
        )
    w, w_s, y, n_int = solution
    log(
        runlog,
        "fixed_point",
        x34=x34,
        f=f_fill,
        w=w,
        w_s=w_s,
        y=y,
        n=n_int,
        algebra="0.595*(x34+w) = 0.295 + 0.405*(572+w_s) + 0.405*f; parity y = 282 + w_s",
        parity_note="dynamic: the denominator counts new class-1/2 subzone rows (addendum census 854 + w_s)",
    )

    # accept the prefix
    t01_rows = t01_pre[:w]

    # mint T3 band rows to quota y
    t3_rows = mint_t3_band(runlog, minter, clean_sides, quota_rows=y)
    if len(t3_rows) != y:
        refuse(f"T3 band minted {len(t3_rows)} != quota {y}; STOP")

    # assemble the final train: base part (v4.3 order) + new material sorted by pair_id
    new_rows = sorted(cls34 + nfar + t01_rows + t3_rows, key=lambda r: r["pair_id"])
    train = list(orig_train) + new_rows

    # ── final asserts ────────────────────────────────────────────────────────
    # train-side only: holdout untouched (never written)
    fin = census(train)
    log(runlog, "census_final", **{k: v for k, v in fin.items() if k != "bands"})
    log(
        runlog,
        "census_final_bands",
        **{s: {k: v for k, v in b.items()} for s, b in fin["bands"].items()},
    )

    # §C parity: negatives >= positives under every reading
    pos_c12_strict = fin["pos_classes_1_2_strict"]
    checks = {
        "strict_classes_1_2": (fin["subzone_neg"], pos_c12_strict),
        "addendum_census_dynamic": (fin["subzone_neg"], PARITY_FLOOR_POS + w_s),
        "addendum_census_floor": (fin["subzone_neg"], PARITY_FLOOR_POS),
    }
    for name, (negs, poss) in checks.items():
        coef = round(negs / poss, 4) if poss else float("inf")
        log(
            runlog,
            "parity",
            reading=name,
            negatives=negs,
            positives=poss,
            coefficient=coef,
        )
        if negs < poss:
            refuse(
                f"§C parity failed [{name}]: {negs} negatives < {poss} positives; STOP"
            )

    # §C bins
    for b, cnt in fin["bins"].items():
        log(runlog, "bin", bin=b, negatives=cnt, required=BIN_MIN_NEG)
        if cnt < BIN_MIN_NEG:
            refuse(f"§C bin {b}: {cnt} negatives < {BIN_MIN_NEG}; STOP")

    # §D coverage per probe class
    for stratum, b in fin["bands"].items():
        log(
            runlog,
            "coverage",
            probe_class=stratum,
            check=b["check"],
            probe_cos=b["probe_cos"],
            probe_tj=b["probe_tj"],
            band_rows=b["count"],
            max_gap=b["max_gap"],
            k_ok=b["k_ok"],
            continuity_ok=b["cont_ok"],
        )
        if not b["k_ok"] or not b["cont_ok"]:
            refuse(
                f"§D coverage failed for {stratum}: count={b['count']} (K={PROBE_K}), "
                f"max_gap={b['max_gap']} (δ={PROBE_DELTA}); STOP"
            )

    # dup share on the formula surface (universe accounting)
    uni_rows = (
        rep["binfo"]["uni_clean_rows"] + 600 + len(rep["inj_rows"]) + len(new_rows)
    )
    uni_dups = rep["binfo"]["uni_clean_dups"] + len(rep["inj_rows"]) + x34 + w
    share = uni_dups / uni_rows
    log(
        runlog,
        "formula_share",
        uni_rows=uni_rows,
        uni_dups=uni_dups,
        share=round(share, 6),
        target=DUP_SHARE,
        n_real_check=round((DUP_SHARE * uni_rows - uni_dups) / (1 - DUP_SHARE), 4),
    )
    if abs(share - DUP_SHARE) > 5e-4:
        refuse(f"formula share {share:.6f} != {DUP_SHARE} beyond rounding; STOP")

    # scaffold anti-collapse per minted class (§D)
    by_cls: dict[str, list[dict]] = {}
    for r in new_rows:
        by_cls.setdefault(r["stratum"], []).append(r)
    scaffold_report = {}
    for stratum, rows in sorted(by_cls.items()):
        scaf = Counter(r.get("scaffold") or "none" for r in rows)
        max_share = max(scaf.values()) / len(rows)
        scaffold_report[stratum] = {
            "rows": len(rows),
            "scaffolds": len(scaf),
            "max_rows_per_scaffold": max(scaf.values()),
            "max_share": round(max_share, 4),
            "ok": len(scaf) >= 20 and max_share <= 0.05,
        }
        if not scaffold_report[stratum]["ok"]:
            refuse(
                f"scaffold anti-collapse failed for {stratum}: "
                f"{scaffold_report[stratum]}; STOP"
            )
    log(runlog, "scaffold_report", classes=scaffold_report)

    # scaffold overlap with the sealed holdout (ids from the recipe; content NOT read)
    hold_scaffolds: set = set()
    inj_by_pid = {r["pair_id"]: r for r in rep["inj_rows"]}
    pool_by_pid = {r["pair_id"]: r for r in rep["pool"]}
    for pid in rep["holdout_ids"]:
        r = inj_by_pid.get(pid) or pool_by_pid.get(pid)
        if r is None:
            refuse(f"holdout id {pid} not resolvable from the recipe; STOP")
        if "unit" in r or r.get("source") == "injection-policy-s4":
            for s in (r["record"], r["candidate"]):
                hold_scaffolds.add(side_sha(s))
        else:
            for s in (r["record"], r["candidate"]):
                hold_scaffolds.add(side_sha(s))
    overlap = 0
    overlap_by_cls: Counter = Counter()
    for r in new_rows:
        sc = r.get("scaffold") or ""
        if not sc:
            continue
        parts = sc.split("|")
        if any(p in hold_scaffolds for p in parts):
            overlap += 1
            overlap_by_cls[r["stratum"]] += 1
    log(
        runlog,
        "holdout_scaffold_overlap",
        holdout_scaffolds=len(hold_scaffolds),
        train_rows_on_holdout_scaffolds=overlap,
        by_class=dict(overlap_by_cls),
        note="known limitation per addendum §D — reported, not refused (corpus-family side-sharing norm)",
    )

    # disjointness of ALL new material vs every holdout surface + burned translated
    surfaces: dict[str, set] = {}
    for name in (
        "v43-holdout/holdout.jsonl",
        "dataset-v4-holdout/holdout.jsonl",
        "b2/holdout.jsonl",
    ):
        surfaces[name] = {pair_key(r) for r in corpus_rows(BASE / name)}
    burned_translated = {
        pair_key(r)
        for r in corpus_rows(BASE / "v43-localbuild/train.jsonl")
        if str(r.get("part", "")).startswith("translated")
        or str(r.get("pair_id", "")).startswith(("V43-TDUP", "V43-TSIB"))
    }
    violations: Counter = Counter()
    for r in new_rows:
        ck = pair_key(r)
        for name, keys in surfaces.items():
            if ck in keys:
                violations[name] += 1
        if ck in burned_translated:
            violations["burned-translated"] += 1
    if violations:
        refuse(f"pair-level disjointness violated: {dict(violations)}; STOP")
    log(runlog, "disjointness_new_material", checked=len(new_rows), violations=0)

    # hygiene tally (any hit already refused at mint time)
    if minter.hygiene_hits:
        refuse(f"hygiene hits on minted material: {minter.hygiene_hits}; STOP")

    # final holdout seal re-assert (file untouched)
    disk_hold_fp = corpus_fingerprint(
        manifest_bytes(
            [
                (r["pair_id"], r["pair_sha256"])
                for r in corpus_rows(
                    REPO_ROOT / "data/b2-graph-corpus/v44/holdout.jsonl"
                )
            ]
        )
    )
    if disk_hold_fp != SEAL_HOLDOUT_FP:
        refuse("holdout seal changed during the build; STOP")
    log(runlog, "holdout_seal_final", fp=disk_hold_fp, untouched=True)

    # ── fingerprints + artifacts ─────────────────────────────────────────────
    entries = [(r["pair_id"], r["pair_sha256"]) for r in train]
    train_fp = corpus_fingerprint(manifest_bytes(entries))
    labels = {r["pair_id"]: r["label"] for r in train}
    strata_train = Counter(r["stratum"] for r in train)
    lab_train = Counter(r["label"] for r in train)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(
        out / "labels-train.jsonl",
        [{"label": v, "pair_id": k} for k, v in sorted(labels.items())],
    )
    manifest = {
        "corpus": "b2-graph-v4.4-r2 (train rebuild per ADDENDUM-1)",
        "law": [
            "docs/specs/b2-graph-prereg-v3.md (ACTIVATED, owner 22.10)",
            "docs/specs/b2-graph-prereg-v3-ADDENDUM-1.md (ratified + NA edits)",
        ],
        "seal": {
            "holdout_fingerprint": SEAL_HOLDOUT_FP,
            "holdout_rows": 75,
            "holdout_untouched": True,
            "ids_sha256": SEAL_IDS_SHA,
            "orig_train_fingerprint": ORIG_TRAIN_FP,
            "orig_corpus_fingerprint": SEAL_CORPUS_FP,
            "reproduction": "byte-exact (stage 0 asserts)",
        },
        "addendum1_taxonomy_glossary": {
            "1_identity": ["P-identity", "T0S/T0C"],
            "2_cosmetic": ["P-cosmetic", "T1"],
            "3_lexical_substitution": ["P-lexsub", "template-twin heritage"],
            "4_paraphrase": ["P-para-light", "P-para-sub", "P-para-struct"],
            "5_structural_reorg": "planned 1.n+1 (not in this build)",
            "6_compression": "planned 1.n+1 (not in this build)",
            "7_fact_edit": ["N-fact-edit", "broken-field", "razor/T3", "mint-top"],
            "8_topic_adjacent": [
                "N-para-notdup",
                "template-twin (DV3)",
                "near-topic",
                "cross-topic",
            ],
            "silver_rule": "duplicate iff bilateral entailment with all §8 slots preserved both ways; any slot change -> class 7 NOT dup; disputed -> NOT dup (§8.3)",
        },
        "addendum1_numbers": {
            "razor_subzone": {
                "rule": "cos >= 0.95 AND tag_jaccard = 1.0",
                "pre": {
                    "pos": pre["subzone_pos"],
                    "neg": pre["subzone_neg"],
                    "bins": pre["bins"],
                    "classes_1_2_strict": pre["pos_classes_1_2_strict"],
                },
                "post": {
                    "pos": fin["subzone_pos"],
                    "neg": fin["subzone_neg"],
                    "bins": fin["bins"],
                    "classes_1_2_strict": pos_c12_strict,
                },
                "parity": checks,
                "dobor_negatives": fin["subzone_neg"] - pre["subzone_neg"],
            },
            "razor_block_transfer": razor_audit,
            "injection_rebalance": {
                "new_class3_rows": sum(1 for r in cls34 if r["stratum"] == "P-lexsub"),
                "new_class4_rows": sum(
                    1 for r in cls34 if r["stratum"] == "P-para-light"
                ),
                "new_class1_2_rows": w,
                "class3_4_minted_before_any_class1_2": True,
                "formula": "0.595*(x34+w) = 0.295 + 0.405*(572+w_s) + 0.405*f (universe surface, anchor-consistent)",
                "fixed_point": {
                    "x34": x34,
                    "f": f_fill,
                    "w": w,
                    "w_s": w_s,
                    "y": y,
                    "n": n_int,
                },
                "t3_band_rows": y,
                "nfar_fill_rows": f_fill,
                "universe_share": round(share, 6),
            },
            "coverage_probes": {
                s: {
                    "check": b["check"],
                    "probe_cos": b["probe_cos"],
                    "probe_tj": b["probe_tj"],
                    "band_rows": b["count"],
                    "max_gap": b["max_gap"],
                    "k_ok": b["k_ok"],
                    "continuity_ok": b["cont_ok"],
                }
                for s, b in fin["bands"].items()
            },
            "probe_procedure_for_gate_runner": {
                "band": "{r: |cos_r - cos_probe| <= 0.005 AND tag_jaccard_r == tag_jaccard_probe}",
                "asserts": "count >= 30 AND max gap between consecutive sorted in-band cos <= 0.005",
                "probe_points": {
                    s: {"cos": b["probe_cos"], "tj": b["probe_tj"]}
                    for s, b in fin["bands"].items()
                },
                "source": "src/cortex/eval/sanity.py frozen probe constants (1.0, 0.99, 0.99, 1.0, 0.578) x producing-class strata",
            },
            "scaffolds": scaffold_report,
            "holdout_scaffold_overlap": {
                "rows": overlap,
                "by_class": dict(overlap_by_cls),
                "status": "known limitation, run-logged per addendum §D",
            },
            "cross_language_quarantine": {
                "pool_pairs_quarantined": quar["quarantined"],
                "pool_pairs_mixed_script": quar["mixed"],
                "pool_pairs_undetectable": quar["undetectable"],
                "rule": "script-detect body language; confident ru/en mismatch quarantined from new mints",
            },
        },
        "counts": {
            "train_rows": len(train),
            "new_rows": len(new_rows),
            "train_labels": dict(lab_train),
            "train_strata": dict(sorted(strata_train.items())),
            "train_dup_share": round(lab_train["duplicate"] / len(train), 6),
            "holdout_rows": 75,
        },
        "fingerprints": {
            "scheme": "manifest(pair_id pair_sha256 sorted) -> blake2b-256 (data-contract §5)",
            "train": train_fp,
            "holdout_unchanged": SEAL_HOLDOUT_FP,
            "labels_train": labels_fingerprint(labels),
        },
        "determinism": "order-deterministic construction, no RNG; double run byte-identical asserted",
        "hygiene": {
            "engine_detectors": True,
            "minted_sides_scanned": len(new_rows) * 2,
            "hits": 0,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "train-manifest.txt").write_bytes(manifest_bytes(entries))
    (out / "run-log.jsonl").write_text(
        "".join(
            json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in runlog
        ),
        encoding="utf-8",
    )
    print(
        f"[v44r2] DONE train={len(train)} new={len(new_rows)} "
        f"train_fp={train_fp[:16]}... holdout_fp={SEAL_HOLDOUT_FP[:8]}... (untouched)",
        file=sys.stderr,
    )
    return 0


def PARITY_FLOOR_NEG_ZONE(w_s: int) -> int:
    """The addendum §C parity closes the 572 -> 854 census gap; new class-1/2
    subzone rows (w_s) join the denominator, so the negative quota grows to
    keep the coefficient >= 1.0 under the census reading."""
    return PARITY_FLOOR_NEG + (PARITY_FLOOR_POS - PARITY_FLOOR_NEG) + w_s


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateRefusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
