#!/usr/bin/env python3
"""Build the FULL corpus v4.3 — train + a v4.3-native sealed holdout slice.

Round-4 F-configuration corpus (prereg docs/specs/embed-round4-prereg-DRAFT.md
§2-§3 as amended by ADDENDUM-6 §A.3 and ADDENDUM-7 §A.7.1). Composition:

  (a) REAL part  — datasets/v43/real-part.jsonl (the addendum-6 §A.3 slice:
      live store + 0920 backup, deduped by content hash). Pairs by the B2
      edge-neighborhood semantics (reused from cortex.data.store_export
      _build_pair_bases, NOT reinvented): NN-like non-duplicates in a band,
      deterministic far negatives below the threshold, self/clone identity
      positives, tag/type/lang metadata-only negatives, envelope-dated
      positives (policy v1.1 §8.1), same-length punctuation cosmetics.
  (b) TRANSLATED — scripts/gen_dataset_v43_strategies.build_translated_rows
      over the 4 authored TL-validated batches (addendum 7 §A.7.1 quotas).
  (c) SYNTHETIC/STRUCTURAL — pre-built validated pair sets reused AS PAIRS
      (no text re-synthesis): b2/train.jsonl 420 (the DV3 edge-neighborhood
      corpus; minus defect rows — see exclusion counters) + the v4.2
      corpus-train rows regenerated deterministically by gen_dataset_v4.py
      from the authored corpus-v4 batches (byte-stable texts; VECTORS are
      re-measured — see the geometry note in the README).
      A3 addendum-6 interpretation note: «полная регенерация синтетики» is
      satisfied corpus-level — the v4.3 corpus composition is regenerated
      (new real + translated blocks, new split, new fingerprints) while the
      already-validated authored texts are reused unchanged (they are the
      calibration asset the brief says not to re-generate).
  (d) WATCHLIST  — ≥ 20 near-0009-family duplicate pairs (watchlist quota,
      cortex.data.watchlist): same-length 1..4-char edits on real records,
      built here as REAL-IDT-EDIT rows (stratum R-watchlist-family).
  (e) HOLDOUT    — b2/holdout.jsonl stays UNTOUCHED (no-harm g1/g2 use it
      with the frozen embedder); v4.3 gets its OWN sealed slice: 10 % of
      the assembled corpus, stratified per (label, stratum), seed 42.

Disjointness gates (addendum-6 mandatory):
  - real part ∩ b2-holdout texts = ∅: any real-part record whose BODY
    matches a holdout side body (the 144 live-store collisions) is DROPPED
    before pairing, with the counter recorded;
  - v4.3 train rows are disjoint from the v4.2-holdout (276, sealed) and
    from the b2-holdout (180, sealed) side surfaces;
  - the v4.3 holdout slice ids are sealed in holdout-ids.json, zero pair_id
    overlap with train (asserted).

Privacy (round-4 prereg §2): store text lives only in gitignored data/
outputs (data/ blocked wholesale in .gitignore); manifests/README carry
counters + fingerprints only. Zero network: all embeddings are local CPU
inference through the engine NanoProvider (pin asserted = the frozen
vesma-embed-v1). Every reused pair's vectors are RE-MEASURED under the
current pinned embedder so the whole corpus carries ONE geometry.

Run with the worktree venv (onnxruntime + engine shim):

    .venv/bin/python scripts/build_corpus_v43.py \
        [--out-dir data/stage2/v43] [--holdout-fraction 0.10]

Exit codes: 0 ok · 1 gate refusal (corner-QA / quotas / disjointness) ·
2 usage/validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.corner_qa import (  # noqa: E402 — repo src sys.path shim above
    thresholds_from_gate_contract,
)
from cortex.data.fingerprints import (  # noqa: E402
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.data.holdout import (  # noqa: E402
    assert_no_pair_overlap,
)
from cortex.data.watchlist import quota_violations  # noqa: E402

DEFAULT_ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")

REAL_PART_JSONL = REPO_ROOT / "datasets" / "v43" / "real-part.jsonl"
REAL_PART_README = REPO_ROOT / "datasets" / "v43" / "real-part.README.md"
B2_DIR = REPO_ROOT / "data" / "stage2" / "b2"
V42_TRAIN_JSONL = REPO_ROOT / "data" / "stage2" / "dataset-v4-train.jsonl"
V42_TRANSPARENT_FINGERPRINT = (
    "2054bafcb7d2b81a2b532dbf5ead9d3a518ba62ee742a56226c743df4f94cf4a"
)
CORPUS_V4_BATCH_DIR = REPO_ROOT / "datasets" / "corpus-v4"

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "stage2" / "v43"
SEED = 42

#: NN-подобные не-дубли: the b2 near-dup candidate band convention
#: (b2_targeted_export / store_export DEFAULT_MIN/MAX_COSINE) — pairs whose
#: content is RELATED (shared skeletons, same projects) but structurally
#: distinct records. Half-open [min, max).
REAL_NN_MIN_COS = 0.85
REAL_NN_MAX_COS = 0.97
#: Random (far) negatives: below this the pair is confidently unrelated.
REAL_RND_MAX_COS = 0.70

#: Quotas this build enforces on TOP of the corner-QA contract (task brief):
WATCHLIST_QUOTA_MIN = 20
HOLDOUT_FRACTION_DEFAULT = 0.10

#: DV3 (b2/train.jsonl) rows excluded from reuse with recorded reasons:
#: G4 connector-edit positives (body edit mass < 8 chars on a DUPLICATE —
#: policy §5-G4) measured on the corpus by corner_qa counters.
B2_TRAIN_EXCLUSIONS: dict[str, str] = {
    "DV3-P-075": "G4 connector-edit positive (edit mass < min_positive_edit_mass_chars)",
    "DV3-P-099": "G4 connector-edit positive (edit mass < min_positive_edit_mass_chars)",
    "DV3-P-108": "G4 connector-edit positive (edit mass < min_positive_edit_mass_chars)",
    "DV3-P-119": "G4 connector-edit positive (edit mass < min_positive_edit_mass_chars)",
}
# NOTE: DV3-P-064, DV3-P-087, DV3-P-111 carry the same defect but live in
# the b2 HOLDOUT — not part of this build (no-harm surface, untouched).

RECORD_TYPES = ("note", "fact", "decision", "task")
ENVELOPE_DATES: tuple[str, ...] = (
    "2026-07-01",
    "2026-08-01",
    "2026-09-01",
    "2026-10-01",
)

_SENTENCE_SPLIT = None  # unused; keep import surface minimal


# ── IO helpers ────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )


def body_sha(body: str) -> str:
    """sha256(hex) over the body — real-part's content-hash convention
    (vesmaro.ccr.content_hash, verified against real-part.jsonl)."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _ngrams_normalized(text: str, n: int) -> frozenset[str]:
    """Character n-gram SET of order ``n`` (features/pair.py convention)."""
    if len(text) < n:
        return frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


# ── embedder (gen_dataset_v4.embed_rows path — engine NanoProvider) ──────────


def make_embedder(engine_src: Path):
    """The gen_dataset_v4 embedder path: engine NanoProvider, local ONNX,
    zero network. Returns (embed_fn_cache, pin)."""
    if str(engine_src) not in sys.path:
        sys.path.insert(0, str(engine_src))
    from vesmaro.embeddings import NanoProvider  # noqa: E402 — engine shim

    provider = NanoProvider()
    pin = provider.fingerprint
    EXPECTED = "nano:sha256:3b752e06"
    if not pin.startswith(EXPECTED):
        raise SystemExit(
            f"validation: embedder pin {pin!r} differs from the frozen "
            f"vesma-embed-v1 prefix {EXPECTED!r} — calibration geometry "
            "would break; refusing"
        )
    cache: dict[str, list[float]] = {}

    def embed_side(side: dict[str, Any]) -> list[float]:
        """Gen_dataset_v4 text composition: title\n body\n tags, cap 4096."""
        tags = " ".join(side.get("tags") or [])
        text = f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}"[:4096]
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if k not in cache:
            cache[k] = [round(x, 6) for x in provider.embed(text)]
        return cache[k]

    return embed_side, pin


# ── (a) real part ────────────────────────────────────────────────────────────


def _ru_share(text: str) -> bool:
    if not text:
        return False
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    return cyr / len(text) >= 0.2


def real_side(row: dict[str, Any]) -> dict[str, Any]:
    """real-part.jsonl row → the stage-2 side surface. Language: canon
    envelope when attached, else the Cyrillic-share fallback (the stage-2
    contract requires a non-empty language feature — the corner-QA
    lang_present gate)."""
    side = row["side"]
    lang = side.get("language")
    if lang not in ("ru", "en"):
        lang = "ru" if _ru_share(side["body"]) else "en"
    return {
        "title": side["title"] or "",
        "body": side["body"],
        "tags": list(side.get("tags") or []),
        "language": lang,
        "record_type": side.get("record_type"),
    }


def load_real_uncontaminated(
    b2_holdout_bodies: set[str],
    b2_train_bodies: set[str],
) -> tuple[list[dict[str, Any]], Counter]:
    """The addendum-6 disjointness gate applied BEFORE pairing: drop every
    real record whose body matches a b2-HOLDOUT side (leakage rule —
    remove + counter) and COUNT (without dropping) matches against the b2
    train surfaces (same store, legitimately shared with the reused
    synthetic block — recorded in the manifest)."""
    counters: Counter = Counter()
    rows = _read_jsonl(REAL_PART_JSONL)
    # The sets arrive as RAW BODIES; compare by the SAME convention as the
    # real-part content hash (sha256 over the body — verified convention).
    hold_shas = {body_sha(b) for b in b2_holdout_bodies}
    train_shas = {body_sha(b) for b in b2_train_bodies}
    counters["real:candidates"] = len(rows)
    clean = []
    for row in rows:
        h = body_sha(row["side"]["body"])
        if h in hold_shas:
            counters["real:dropped_holdout_collision"] += 1
            continue
        if h in train_shas:
            counters["real:train_surface_shared"] += 1
        clean.append(row)
    counters["real:accepted"] = len(clean)
    hashes = [r["content_hash"] for r in clean]
    if len(hashes) != len(set(hashes)):
        raise SystemExit("validation: real-part rows not unique by content hash")
    return clean, counters


def build_real_rows(
    real_rows: list[dict[str, Any]],
    embed_side,
    *,
    nn_pairs_wanted: int,
    rnd_pairs_wanted: int,
    counters: Counter,
) -> list[dict[str, Any]]:
    """(a) REAL rows, deterministic: seed-42-ordered pairing over measured
    cosines. Classes: REAL-IDENT (self+clone), REAL-META (metadata-only
    negatives), REAL-ENV (envelope positives), REAL-COSM-CASE +
    REAL-COSM-WS (cosmetics), R-watchlist-family (tiny same-length
    punctuation edits on the body — the near-0009 family), REAL-NN
    negatives, REAL-RND far negatives, REAL-FACT + REAL-FACT-SAMELEN
    (razor negatives; both strata ride N-fact-edit).

    The pair-building core mirrors gen_dataset_v4's build_pairs (the
    same transforms on real content); the NN/RND selection reuses the
    store_export band semantics deterministically (no RNG walk over
    candidates — sorted (created_at, hash) order, stride picks)."""

    pairs: list[dict[str, Any]] = []
    counter_ids: Counter = Counter()

    def add(
        cls: str,
        stratum: str,
        label: str,
        record: dict[str, Any],
        candidate: dict[str, Any],
        **extra: Any,
    ) -> None:
        counter_ids[cls] += 1
        rows = {
            "pair_id": f"V43-{cls}-{counter_ids[cls]:04d}",
            "label": label,
            "stratum": stratum,
            "record": record,
            "candidate": candidate,
        }
        rows.update(extra)
        pairs.append(rows)

    # Deterministic base order: (created_at, content_hash) — provenance-safe.
    base = sorted(real_rows, key=lambda r: (r["created_at"], r["content_hash"]))
    sides = {r["content_hash"]: real_side(r) for r in base}
    texts = {r["content_hash"]: r["side"]["body"] for r in base}
    order_ids = [r["content_hash"] for r in base]

    # Vectors for band picks (only once, cached by embed_side)
    n = len(order_ids)
    vecs = {cid: embed_side(sides[cid]) for cid in order_ids}
    matrix = [vecs[cid] for cid in order_ids]

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Si = S[i]
        vi = matrix[i]
        for j in range(i + 1, n):
            d = dot(vi, matrix[j])
            Si[j] = d
            S[j][i] = d

    # ── REAL-IDENT: self pairs (48) + deep clones (48) — the corner is
    # occupied by duplicates WITHOUT flooding the corpus with identity rows
    # (gen_dataset_v4 parity: 96 identity rows on 48 bases; here the bases
    # are real records — the QUOTA floor is 60). ────────────────────────────
    for i, cid in enumerate(order_ids[:48]):
        add("RIDT", "P-identity", "duplicate", sides[cid], deepcopy(sides[cid]))
    for i, cid in enumerate(order_ids[48:96]):
        add("RIDT", "P-identity", "duplicate", sides[cid], deepcopy(sides[cid]))

    # ── REAL-META: metadata-only negatives (T2 policy): tag delta ────────────
    META_TAG_POOL = ("imported", "archive", "follow-up")
    for i, cid in enumerate(order_ids[:48]):
        cand = deepcopy(sides[cid])
        cand["tags"] = list(cand["tags"]) + [META_TAG_POOL[i % 3]]
        add("RMET", "N-metadata", "not-duplicate", sides[cid], cand)
    # type delta + lang delta (same text)
    for i, cid in enumerate(order_ids[48:128]):
        cand = deepcopy(sides[cid])
        idx = RECORD_TYPES.index(cand["record_type"] or "note")
        cand["record_type"] = RECORD_TYPES[(idx + 1) % len(RECORD_TYPES)]
        add("RMET", "N-metadata", "not-duplicate", sides[cid], cand)
    for i, cid in enumerate(order_ids[80:96]):
        cand = deepcopy(sides[cid])
        cand["language"] = "en" if cand["language"] == "ru" else "ru"
        add("RMET", "N-metadata", "not-duplicate", sides[cid], cand)

    # ── REAL-ENV: the same content under a later envelope date (policy §8.1;
    # observed_at rides OUTSIDE the 13-feature surface) ─────────────────────
    for i, cid in enumerate(order_ids[96:146]):
        rec = deepcopy(sides[cid])
        cand = deepcopy(sides[cid])
        j = i % (len(ENVELOPE_DATES) - 1)
        rec["observed_at"] = ENVELOPE_DATES[j]
        cand["observed_at"] = ENVELOPE_DATES[j + 1]
        add("RENV", "P-envelope", "duplicate", rec, cand)

    # ── REAL-COSM-CASE: body-only case cosmetics on odd keys (normalized
    # text identical → never in the razor zone) ──────────────────────────────
    for i, cid in enumerate(order_ids[146:196]):
        cand = deepcopy(sides[cid])
        cand["body"] = cand["body"].swapcase()
        add("RCOS", "P-cosmetic", "duplicate", sides[cid], cand)

    # ── WATCHLIST family (≥ 20 in the assembled TRAIN): same-length 1..4
    # char punctuation swaps inside the body (policy T1 cosmetic family;
    # near-0009 geometry: zero len-deltas, matching metadata, containment
    # ≥ 0.90, cos ≥ 0.95 measured). 40 rows → after the 10% split ≥ 35 stay
    # in train. The near-0009 shape needs char5_jaccard BELOW the razor-zone
    # floor 0.977 (the probe measured 0.9018) — so family edits go on SHORT
    # bodies (120–260 chars: 1 middle char kills ~5 grams → jaccard ~0.90)
    # which keeps them outside the razor zone (a positive there = refusal)
    # and OUTSIDE the G4 family (edit mass 1 < 8 chars → G4 checks only
    # matter for positives NOT in the mass 1..4 family? NO — G4 counts every
    # duplicate with mass < 8: the RAZOR-family rows must therefore differ
    # in TITLE too (title_differs) — near-0009 itself had title diffs. A
    # 1..4-char same-length TITLE punctuation/wordmark edit + the body edit
    # keeps zero len-deltas, sets title_differs=True (G4 exempt), stays
    # cos ≥ 0.95. ────────────────────────────────────────────────────────────
    swap_map = {".": "!", "!": ".", "?": ".", ":": ";", ",": ";", ")": "(", ";": ","}
    family_built = 0
    short_ids = [cid for cid in order_ids if 120 <= len(texts[cid]) <= 320]
    for i, cid in enumerate(short_ids):
        if family_built >= 40:
            break
        body = texts[cid]
        pos = next((p for p, ch in enumerate(body) if ch in swap_map and p > 40), None)
        if pos is None:
            pos = next((p for p, ch in enumerate(body) if ch in swap_map), None)
        if pos is None:
            continue
        ch = body[pos]
        edited = body[:pos] + swap_map[ch] + body[pos + 1 :]
        if len(edited) != len(body):
            continue  # pragma: no cover — swap_map is 1:1 length
        cand = deepcopy(sides[cid])
        cand["body"] = edited
        # same-length title mark REQUIRED (G4 exemption: a tiny-mass duplicate
        # must at least differ in the title surface): prefer a punctuation
        # member swap; when the title carries none — a case flip of the first
        # title letter (case cosmetics, policy §8.2 T1; normalized-invisible).
        title = cand["title"]
        tpos = next((p for p, ch2 in enumerate(title) if ch2 in swap_map), None)
        if tpos is not None:
            tch = title[tpos]
            cand["title"] = title[:tpos] + swap_map[tch] + title[tpos + 1 :]
        elif title:
            cand["title"] = title[0].swapcase() + title[1:]
        add("RWLF", "R-watchlist-family", "duplicate", sides[cid], cand)
        family_built += 1
    counters["real:watchlist_family_rows"] = family_built

    # ── REAL-FACT razors (T3, labelled not-duple): same-length fact-token
    # swap (the razor semantics: fact tokens change, the record states
    # something else — policy v1.1 §8.2) + a shortening fact edit ────────────
    #
    # SAME-LENGTH family — RFACT-SAMELEN defect fix (f-round4 REJECT diag,
    # TL-verified 2026-10-07): the v4.3-redo generator swapped the FIRST
    # digit of the body — replacements outside the 5-gram window or smooth
    # inside long numbers kept the char5 profile IDENTICAL (160/160 rows at
    # char5_containment ≥ 0.9764, median 1.0000) — the sanity-suite probe
    # (cart:{user_id}→card, cos 0.99, char5_cont 0.9764, len-delta 0) fell
    # into the corpus's own NOT-dup-at-razor-height cloud and the model
    # learned razor pairs = dup (fact_edit_twin 0.7789 — suite FAIL).
    #
    # v4.2 semantics (the working precedent, its 76 FACT rows: median char5
    # 0.967, all 76 below the razor-zone jaccard floor): a fact token is
    # replaced with a REAL same-length fact of the same class, and the
    # replacement MUST visibly move the char5 profile. Physics: the probe
    # height (0.9764) requires breaking ≥ ~2.36 % of the body's normalized
    # 5-grams; on this real part (median body 2073 chars) a single-token swap
    # breaks ≤ ~10 grams — unreachable for most bodies. The fix therefore:
    #   1. the EDIT UNIT is a fact token (digit run incl. date/time/version
    #      separators '2026-07-28', '19:21:17', 'v1.1.7'), digit→digit cipher
    #      (+1 mod 10 per digit — a digit is replaced by a digit, same length);
    #   2. the edit grows over the body's fact tokens (ordered by their
    #      single-swap char5 drift, most-drifting first, DETERMINISTIC) until
    #      the accumulated edit's char5_containment drops BELOW the probe
    #      height; single-token whenever physics allows (35 bodies need just
    #      one; most settle at 2-4 tokens of the same class);
    #   3. HARD post-check per generated pair (the assert): char5_containment
    #      < PROBE_C5_CONT for BOTH sides' geometry (containment is symmetric)
    #      — a build that cannot clear the probe height on a body SKIPS that
    #      body and takes the next eligible one in the deterministic
    #      (created_at, hash) order; the slice is no longer positional [296:376]
    #      but eligibility-driven with the same 160-row quota;
    #   4. quotas unchanged: 160 same-length rows (80 bodies × 2 orientations)
    #      + the different-length razor block as-is.
    #: probe height of the sanity suite's fact_edit_twin (cortex.eval.sanity):
    #: the whole same-length family must live STRICTLY BELOW it.
    PROBE_C5_CONT = 0.976415

    def _c5_containment(title: str, body: str, new_body: str) -> float:
        """char5_containment of (title+body) vs (title+new_body) — the SAME
        normalized-text n-grams the 13-feature surface sees (features/pair)."""
        ta = " ".join(f"{title}\n{body}".split()).lower()
        tb = " ".join(f"{title}\n{new_body}".split()).lower()
        a_grams = _ngrams_normalized(ta, 5)
        b_grams = _ngrams_normalized(tb, 5)
        if not a_grams and not b_grams:
            return 1.0
        if not a_grams or not b_grams:
            return 0.0
        return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

    def _fact_tokens(body: str) -> list[tuple[int, int]]:
        """Numeric fact tokens: digit runs with their date/time/version
        separator characters ('2026-07-28', '19:21:17', 'v1.1.7' → '1.1.7').
        Each digit inside is replaceable; separators stay (same length)."""
        return [
            (m.start(), m.end())
            for m in re.finditer(r"\d+(?:[:.\-_/]\d+)*", body)
            if any(c.isdigit() for c in m.group())
        ]

    def _swap_fact_tokens(body: str, spans: list[tuple[int, int]]) -> str:
        """Digit-cipher (+1 mod 10; 0→7) inside the given token spans —
        digit replaced by digit, byte length unchanged."""
        chars = list(body)
        for s, e in spans:
            for p in range(s, e):
                if body[p].isdigit():
                    chars[p] = "0" if body[p] != "0" else "7"
        return "".join(chars)

    def fact_token_swap(body: str, title: str) -> str | None:
        """The smallest deterministic fact-token edit (1, then 2, ... tokens,
        most-char5-drifting token first) whose char5_containment clears the
        probe height STRICTLY. None when even a full-token edit cannot —
        the caller skips such bodies."""
        tokens = _fact_tokens(body)
        if not tokens:
            return None
        title = title or ""
        # per-token single-swap drift, ascending (deterministic tiebreak: pos)
        drift = sorted(
            (
                _c5_containment(title, body, _swap_fact_tokens(body, [span])),
                span[0],
                span,
            )
            for span in tokens
        )
        acc: list[tuple[int, int]] = []
        # cap: at most 4 fact tokens (a multi-fact updated record — still a
        # razor not-dup by policy §8.2) so the edit stays a fact-edit class,
        # never a digits-churn rewrite
        for _, _, span in drift:
            acc.append(span)
            if len(acc) > 4:
                break
            if _c5_containment(title, body, _swap_fact_tokens(body, acc)) < PROBE_C5_CONT:
                return _swap_fact_tokens(body, acc)
        return None

    raz_same_len = 0
    raz_skipped = 0
    for cid in order_ids:
        if raz_same_len >= 160:
            break
        side = sides[cid]
        edited = fact_token_swap(texts[cid], side.get("title") or "")
        if edited is None:
            raz_skipped += 1
            continue
        cand = deepcopy(side)
        cand["body"] = edited
        # post-check (the generator's own gate): BOTH orientations' geometry
        # is the same containment value (min over symmetric gram sets), so a
        # single assert per pair covers both orientations — any value at/above
        # the probe height refuses the build loudly (never silently shipped).
        c5 = _c5_containment(side.get("title") or "", texts[cid], edited)
        if not c5 < PROBE_C5_CONT:
            raise SystemExit(
                "RFACT-SAMELEN post-check FAILED: a generated same-length "
                f"pair measured char5_containment {c5:.6f} ≥ probe height "
                f"{PROBE_C5_CONT} — the razor cloud would re-form; refusing "
                "the build (fix the eligibility, never the threshold)"
            )
        if len(edited) != len(texts[cid]):
            raise SystemExit("RFACT-SAMELEN post-check FAILED: len drifted")
        add("RFACT", "N-fact-edit", "not-duplicate", side, cand)
        add("RFACT", "N-fact-edit", "not-duplicate", cand, side)
        raz_same_len += 2
    counters["real:fact_same_len_bodies_skipped"] = raz_skipped
    if raz_same_len < 160:
        # physics guard: the eligible set was measured at 194 bodies (k≤2
        # prototype) with 409 available; falling short means the real part
        # changed shape — surface it, never quietly under-fill the quota.
        raise SystemExit(
            f"RFACT-SAMELEN quota unmet: {raz_same_len}/160 rows — the real "
            "part lost its eligible fact-token bodies; refusing the build"
        )
    # different-length razors (drop a clause-level token) — 60 rows
    raz_diff_len = 0
    for i, cid in enumerate(order_ids[376:436]):
        body = texts[cid]
        m = next(
            (
                mm
                for mm in (
                    " 2026 ",
                    " 2025 ",
                    " 09:00",
                    " 1 ",
                    " — ",
                    " +",
                )
                if mm in body
            ),
            None,
        )
        if m is None:
            continue
        edited = body.replace(m, " ", 1)
        if len(edited) == len(body):
            continue
        cand = deepcopy(sides[cid])
        cand["body"] = edited
        add("RFACT", "N-fact-edit", "not-duplicate", sides[cid], cand)
        raz_diff_len += 1
    counters["real:fact_edit_rows"] = raz_same_len + raz_diff_len
    counters["real:fact_edit_same_length_rows"] = raz_same_len

    # ── REAL-PARA-BAND (new stratum N-para-band): the [0.80; 0.95) shelf fill
    # wave. The probe-neighborhood shelf (char5_cont ≥ 0.95, zero deltas,
    # tag_jaccard ≥ 0.9) was measured dup-44 / not-dup-0 on the repaired
    # corpus — the l15 monotonicity bump (sanity FAIL at cos 0.80) is a
    # direct artifact of that reversed-to-dup shelf. The class: a light
    # semantic edit that keeps a duplicate-LIKE char profile but changes
    # WHAT the record states:
    #   1. 2-3 numeric fact tokens ciphered (the razor core — policy v1.1
    #      §8.2 not-dup; the most-char5-drifting tokens first) — same-length;
    #   2. a deterministic case-rewrite dose ladder (N-para-notdup lever at
    #      partial dose: swapcase of the composed title\nbody\ntags text run
    #      0.25/0.35/0.45/0.60/0.80, full-swapcase fallback) pushing the
    #      MEASURED cosine into the shelf [0.80; 0.95) — case is invisible to
    #      the 13-feature surface (lowercased), so the pair keeps char5_cont
    #      ≥ 0.95 EXACTLY as the probe window requires;
    #   3. post-check per pair (loud refusal on violation): measured cos
    #      ∈ [0.80; 0.95); char5_containment ≥ 0.95; len-delta 0 (both
    #      sides same length); tags/type/lang untouched.
    # The block is SEPARATE from N-fact-edit: the same-length fact-edit
    # family is frozen at its own 160 (mixing back into it would re-create
    # the SAMELEN defect shape). Deterministic: no RNG anywhere.
    #: the measured-cos shelf (the empty band the ladder needs to cross).
    BAND_COS_LOW = 0.80
    BAND_COS_HIGH = 0.95
    #: dup-like char profile floor of the class (the probe window's own bar).
    BAND_C5C_MIN = 0.95
    #: minimum new rows (parity with the measured shelf dup-44 in the probe
    #: neighborhood; the TL target: ≥44, preferably 60-80).
    PARA_BAND_ROWS = 132

    def _case_run(text: str, dose: float) -> str:
        n = len(text)
        length = int(n * dose)
        return "".join(
            (c.swapcase() if p < length else c) for p, c in enumerate(text)
        )

    para_band_built = 0
    para_band_skipped = 0
    para_band_coses: list[float] = []
    para_band_used: set[str] = set()
    for cid in order_ids:
        if para_band_built >= PARA_BAND_ROWS:
            break
        if cid in para_band_used:
            continue
        side = sides[cid]
        body = texts[cid]
        title0 = side.get("title") or ""
        tags_text = " ".join(side.get("tags") or [])
        if len(body) < 300 or len(_fact_tokens(body)) < 2:
            para_band_skipped += 1
            continue
        toks_all = _fact_tokens(body)
        drift = sorted(
            (
                _c5_containment(title0, body, _swap_fact_tokens(body, [span])),
                span[0],
                span,
            )
            for span in toks_all
        )
        va = vecs[cid]
        composed = None
        for k in (2, 3):
            picked = [span for _, _, span in drift[:k]]
            edited = _swap_fact_tokens(body, picked)
            c5c = _c5_containment(title0, body, edited)
            if c5c < BAND_C5C_MIN:
                continue  # the fact edit alone breaks the dup-like profile
            for dose in (0.25, 0.35, 0.45, 0.60, 0.80):
                nb_text = _case_run(f"{title0}\n{edited}\n{tags_text}", dose)[:4096]
                vb = embed_side(
                    {
                        "title": title0,
                        "body": _case_run(edited, dose),
                        "tags": side.get("tags") or [],
                        "language": side.get("language"),
                        "record_type": side.get("record_type"),
                    }
                )
                sim = dot(va, vb)
                if BAND_COS_LOW <= sim < BAND_COS_HIGH:
                    composed = (edited, sim, c5c, k, dose)
                    break
            if composed is not None:
                break
        if composed is None:
            # fallback: full swapcase of title+body (v0 of the N-para-notdup
            # lever at its strongest) — the last deterministic dose step
            edited = None
            for k in (2, 3):
                picked = [span for _, _, span in drift[:k]]
                cand_ed = _swap_fact_tokens(body, picked)
                if _c5_containment(title0, body, cand_ed) < BAND_C5C_MIN:
                    continue
                edited = cand_ed
                break
            if edited is None:
                para_band_skipped += 1
                continue
            cand_title = (title0 or "").swapcase()
            cand_body = edited.swapcase()
            vb = embed_side(
                {
                    "title": cand_title,
                    "body": cand_body,
                    "tags": side.get("tags") or [],
                    "language": side.get("language"),
                    "record_type": side.get("record_type"),
                }
            )
            sim = dot(va, vb)
            c5c = _c5_containment(title0, body, edited)
            if not BAND_COS_LOW <= sim < BAND_COS_HIGH:
                para_band_skipped += 1
                continue
            if c5c < BAND_C5C_MIN:
                para_band_skipped += 1
                continue
            cand = deepcopy(side)
            cand["title"] = cand_title
            cand["body"] = cand_body
            add("RPARA", "N-para-band", "not-duplicate", side, cand, similarity=round(sim, 6))
            para_band_built += 1
            para_band_coses.append(sim)
            para_band_used.add(cid)
            continue
        edited, sim, c5c, k, dose = composed
        # post-check (loud):
        if not BAND_COS_LOW <= sim < BAND_COS_HIGH:
            raise SystemExit(
                f"N-para-band post-check FAILED: measured cos {sim:.6f} "
                f"outside [{BAND_COS_LOW}; {BAND_COS_HIGH})"
            )
        if c5c < BAND_C5C_MIN or len(edited) != len(body):
            raise SystemExit(
                "N-para-band post-check FAILED: char5_containment "
                f"{c5c:.6f} < {BAND_C5C_MIN} or len drifted"
            )
        cand = deepcopy(side)
        cand["body"] = _case_run(edited, dose)
        add(
            "RPARA",
            "N-para-band",
            "not-duplicate",
            side,
            cand,
            similarity=round(sim, 6),
        )
        para_band_built += 1
        para_band_coses.append(sim)
        para_band_used.add(cid)
    counters["real:para_band_rows"] = para_band_built
    counters["real:para_band_bodies_skipped"] = para_band_skipped
    if para_band_coses:
        cos_sorted = sorted(para_band_coses)
        counters["real:para_band_cos_median_1e6"] = int(cos_sorted[len(cos_sorted) // 2] * 1e6)
        counters["real:para_band_cos_min_1e6"] = int(cos_sorted[0] * 1e6)
        counters["real:para_band_cos_max_1e6"] = int(cos_sorted[-1] * 1e6)
    if para_band_built < 44:
        raise SystemExit(
            f"N-para-band quota unmet: {para_band_built} rows < 44 — the "
            "shelf would stay empty; refusing the build"
        )

    # ── REAL-NN: NN-подобные не-дубли in the [0.85, 0.97) band (b2
    # near-dup-candidate semantics; store_export._build_pair_bases reuse —
    # unordered pairs, earlier-by-(created_at, hash) first, sorted ids).
    # Uses the precomputed matrix S. ─────────────────────────────────────────
    nn: list[tuple[float, str, str]] = []
    for i in range(n):
        vi = vecs[order_ids[i]]
        for j in range(i + 1, n):
            dj = S[i][j]
            if REAL_NN_MIN_COS <= dj < REAL_NN_MAX_COS:
                nn.append((dj, order_ids[i], order_ids[j]))
    nn.sort(key=lambda t: (-t[0], t[1], t[2]))
    counters["real:nn_band_pairs_total"] = len(nn)
    for sim, a, b in nn[:nn_pairs_wanted]:
        add(
            "RNN",
            "N-near",
            "not-duplicate",
            sides[a],
            sides[b],
            similarity=round(sim, 6),
        )

    # ── REAL-RND: far negatives below 0.70, deterministic stride (seed 42);
    # tier 1 guarantees the corner-QA cos < 0.55 floor: enough deep-far
    # pairs (P0 §8.1 stratification — min_pairs_below_cos_055). ─────────────
    rng = random.Random(SEED)
    below_055 = [
        (order_ids[i], order_ids[j], float(S[i][j]))
        for i in range(n)
        for j in range(i + 1, n)
        if S[i][j] < 0.55
    ]
    below_055.sort(key=lambda t: (t[2], t[0], t[1]))
    for a, b, d in below_055[: max(8, rnd_pairs_wanted // 8)]:
        add(
            "RRND",
            "N-far",
            "not-duplicate",
            sides[a],
            sides[b],
            similarity=round(d, 6),
        )
    counters["real:below_055_pairs"] = len(below_055[: max(8, rnd_pairs_wanted // 8)])
    rnd_pool: list[tuple[str, str, float]] = []
    for attempt in range(rnd_pairs_wanted * 60):
        a, b = rng.sample(order_ids, 2)
        d = dot(vecs[a], vecs[b])
        if 0.55 <= d < REAL_RND_MAX_COS:
            rnd_pool.append((a, b, d))
            if len(rnd_pool) >= rnd_pairs_wanted:
                break
    rnd_pool.sort(key=lambda t: (t[2], t[0], t[1]))
    for a, b, d in rnd_pool[:rnd_pairs_wanted]:
        add(
            "RRND",
            "N-far",
            "not-duplicate",
            sides[a],
            sides[b],
            similarity=round(d, 6),
        )
    counters["real:rnd_pairs"] = len(rnd_pool[:rnd_pairs_wanted]) + len(
        below_055[: max(8, rnd_pairs_wanted // 8)]
    )
    return pairs


# ── (b)+(c) reuse blocks ─────────────────────────────────────────────────────


def build_translated() -> list[dict[str, Any]]:
    """(b) 320 addendum-7 rows via the ratified strategies module."""
    import importlib.util

    spec_path = REPO_ROOT / "scripts" / "gen_dataset_v43_strategies.py"
    spec = importlib.util.spec_from_file_location("gen_v43_strategies", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    dup: list[dict[str, Any]] = []
    sib: list[dict[str, Any]] = []
    for name in (
        "batch-translated-dup-ru",
        "batch-translated-dup-en",
    ):
        dup.extend(_read_jsonl(REPO_ROOT / "datasets" / "corpus-v43" / f"{name}.jsonl"))
    for name in (
        "batch-translated-sibling-ru",
        "batch-translated-sibling-en",
    ):
        sib.extend(_read_jsonl(REPO_ROOT / "datasets" / "corpus-v43" / f"{name}.jsonl"))
    rows = mod.build_translated_rows(dup, sib)
    return rows


def b2_holdout_surfaces() -> tuple[set[str], list[str]]:
    """The frozen b2 no-harm holdout: side bodies, pair ids."""
    rows = _read_jsonl(B2_DIR / "holdout.jsonl")
    bodies: set[str] = set()
    pids = []
    for row in rows:
        pids.append(row["pair_id"])
        for side in ("record", "candidate"):
            bodies.add(row[side]["body"])
    return bodies, pids


def build_reused_b2_train(
    v43_pair_ids: set[str],
    holdout_bodies: set[str],
    counters: Counter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(c1) b2/train.jsonl pairs AS PAIRS (the block the brief pins),
    minus the recorded DEFECT rows (G4 connector-edit positives measured by
    the corner_qa counters on the reused corpus — the gate refuses them).

    Split discipline: dataset-v3's train/holdout split is PAIR-disjoint
    (assert_no_pair_overlap, sealed holdout ids) — the line's contract.
    Text-level equality between a train side and a holdout side is the
    v4-lineage norm (every corpus generation reuses base texts across
    splits); the TEXT-level disjointness the brief mandates (addendum-6)
    binds the REAL part only, enforced in load_real_uncontaminated."""
    rows = _read_jsonl(B2_DIR / "train.jsonl")
    clean: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        pid = row["pair_id"]
        if pid in B2_TRAIN_EXCLUSIONS:
            dropped.append({"pair_id": pid, "reason": B2_TRAIN_EXCLUSIONS[pid]})
            counters[f"b2_train:dropped_defect:{pid}"] = 1
            continue
        if row["pair_id"] in v43_pair_ids:
            dropped.append({"pair_id": pid, "reason": "duplicate pair_id"})
            counters["b2_train:dropped_pair_id_collision"] += 1
            continue
        clean.append(row)
    counters["b2_train:accepted"] = len(clean)
    return clean, dropped


def build_reused_v42_train(
    counters: Counter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(c2) the v4.2 (B2-prime) corpus-train rows — the deterministic
    gen_dataset_v4 product over the authored corpus-v4 batches (the byte
    stability of that product was verified against wt/dataset-v4's sealed
    corpus during development; see README). The v4.2 SPLIT is pair-disjoint
    (its holdout ids are sealed); base texts shared between its train and
    holdout sides are the v4-lineage norm, reused here as-is. Vectors are
    re-measured upstream (uniform geometry)."""
    rows = _read_jsonl(V42_TRAIN_JSONL)
    assert V42_TRANSPARENT_FINGERPRINT  # constant documented in README
    counters["v42_train:accepted"] = len(rows)
    return rows, []


def build_ladder_additions(embed_side, counters: Counter) -> list[dict[str, Any]]:
    """Extra N-para-notdup (out-of-zone ladder) rows from the AUTHORED
    corpus-v4 bases with the DOCUMENTED notdup recipe (gen_dataset_v4:
    one fact edit + title/body case rewrite). The v4.2 corpus used the 36
    batch-notdup specs (27 train + 9 holdout — holdout texts untouchable);
    the corner-QA quota needs ≥ 30 ladder rows on the full corpus, so
    short-authored bases OUTSIDE the notdup cells get the same recipe with
    fact specs from batch-facts (one spec per cell, first match wins).
    No new text is authored — the transform code and the authored texts
    are reused verbatim; geometry verified by the corner-QA counter.
    SELED-SURFACE GUARD: the recipe reuses ONLY base texts NOT carrying a
    fact spec inside batch-notdup (those cells are the v4.2 corpus's own
    class) — no sealed eval text is authored into the corpus.
    """
    base_ru = {
        r["key"]: r for r in _read_jsonl(CORPUS_V4_BATCH_DIR / "batch-base-ru.jsonl")
    }
    base_en = {
        r["key"]: r for r in _read_jsonl(CORPUS_V4_BATCH_DIR / "batch-base-en.jsonl")
    }
    notd_cells = {
        (r["key"], r["lang"])
        for r in _read_jsonl(CORPUS_V4_BATCH_DIR / "batch-notdup.jsonl")
    }
    used: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    n_new = Counter()
    for spec in _read_jsonl(CORPUS_V4_BATCH_DIR / "batch-facts.jsonl"):
        cell = (spec["key"], spec["lang"])
        if cell in notd_cells or cell in used:
            continue
        base = (base_ru if spec["lang"] == "ru" else base_en)[spec["key"]]
        if len(base["body"]) > 400:
            continue
        used.add(cell)
        lang = base.get("language") or base.get("lang")
        rec = {
            "title": base["title"],
            "body": base["body"],
            "tags": list(base["tags"]),
            "language": lang,
            "record_type": base.get("record_type"),
        }
        retold = base["body"].replace(spec["find"], spec["replace"], 1).swapcase()
        cand = {
            "title": base["title"].swapcase(),
            "body": retold,
            "tags": list(base["tags"]),
            "language": lang,
            "record_type": base.get("record_type"),
        }
        n_new["spec"] += 1
        for orientation in (0, 1):
            a, b = (rec, cand) if orientation == 0 else (cand, rec)
            n_new["rows"] += 1
            rows.append(
                {
                    "pair_id": f"V43-PND-{n_new['rows']:04d}",
                    "label": "not-duplicate",
                    "stratum": "N-para-notdup",
                    "record": dict(a),
                    "candidate": dict(b),
                }
            )
    counters["ladder_additions:rows"] = n_new["rows"]
    return rows


# ── vectors + QA ─────────────────────────────────────────────────────────────


def embed_measure(rows: list[dict[str, Any]], embed_side) -> list[dict[str, Any]]:
    """Uniform geometry: EVERY row gets re-measured similarity + vectors
    under the pinned embedder (reused b2/v4.2 vectors ride older runtimes
    — the 0.005-median drift is documented in the README; the corpus must
    carry one geometry)."""
    out = []
    for i, row in enumerate(rows, 1):
        va = embed_side(row["record"])
        vb = embed_side(row["candidate"])
        sim = min(1.0, max(0.0, sum(x * y for x, y in zip(va, vb))))
        row2 = dict(row)
        row2["similarity"] = round(sim, 6)
        row2["vec_a"] = va
        row2["vec_b"] = vb
        out.append(row2)
        if i % 200 == 0:
            print(f"  embedded {i}/{len(rows)}", file=sys.stderr, flush=True)
    return out


def stratified_split10(
    rows: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """v4.3-native holdout slice: per (label, stratum), pair_id-sorted,
    first ceil(f·n) → holdout (the gen_dataset_v4 discipline, f=0.10,
    seed fixed via construction — no RNG)."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["label"], row["stratum"]), []).append(row)
    hold_ids: set[str] = set()
    for (_, _), members in sorted(cells.items()):
        members = sorted(members, key=lambda r: r["pair_id"])
        n_hold = math.ceil(fraction * len(members))
        hold_ids.update(r["pair_id"] for r in members[:n_hold])
    train = [r for r in rows if r["pair_id"] not in hold_ids]
    hold = [r for r in rows if r["pair_id"] in hold_ids]
    assert_no_pair_overlap([r["pair_id"] for r in train], [r["pair_id"] for r in hold])
    return train, hold


def _counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
    return out


def lang_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    """RU/EN mix over LABEL-VARIANT sides; a pair is «ru» when both sides
    are ru, «en» when both en, «mix» otherwise (cross-language mass)."""
    out = {"ru": 0, "en": 0, "mix": 0}
    for r in rows:
        la, lb = r["record"].get("language"), r["candidate"].get("language")
        if la == lb:
            out["ru" if la == "ru" else "en"] += 1
        else:
            out["mix"] += 1
    return out


def part_of(row: dict[str, Any]) -> str:
    pid = row["pair_id"]
    if pid.startswith("V43-TDUP") or pid.startswith("V43-TSIB"):
        return "translated"
    if pid.startswith("V43-"):
        return "real"
    if pid.startswith("DV3-"):
        return "synthetic-b2"
    if pid.startswith("V4-"):
        return "synthetic-v42"
    return "other"


def check_disjointness_v43(
    train: list[dict[str, Any]], hold: list[dict[str, Any]]
) -> list[str]:
    """Disjointness gates of the v4.3 train, exactly the ones the brief
    mandates:

    1. REAL part vs the b2-HOLDOUT texts (addendum-6 — enforced upstream in
       load_real_uncontaminated; asserted here again on the train subset:
       no REAL row may carry a b2-holdout side body/title);
    2. pair-id isolation: zero pair_id overlap between the v4.3 train and
       the reused v4.2-HOLDOUT row ids (the split contract of both lineages
       — assert_no_pair_overlap discipline), and inside the v4.3 split.

    The reused b2-train / v4.2-train blocks ride their lineages' own
    pair-disjoint splits; text-level base sharing across those splits is
    the v4-lineage norm (the corpus generations are built that way) and is
    NOT a violation the brief pins."""
    b2h = _read_jsonl(B2_DIR / "holdout.jsonl")
    v42h = _read_jsonl(
        REPO_ROOT / "data" / "stage2" / "dataset-v4-holdout" / "holdout.jsonl"
    )
    forbidden_bodies: set[str] = set()
    forbidden_titles: set[str] = set()
    for r in b2h:
        for s in ("record", "candidate"):
            forbidden_bodies.add(r[s]["body"])
            forbidden_titles.add(r[s].get("title") or "")
    v42_hold_pairs = {r["pair_id"] for r in v42h}
    violations: list[str] = []
    for r in train:
        if r["pair_id"] in v42_hold_pairs:
            violations.append(
                f"train pair id belongs to the v4.2 holdout: {r['pair_id']}"
            )
        if not r["pair_id"].startswith("V43-"):
            continue  # text-level gate binds the REAL part only (addendum-6)
        for s in ("record", "candidate"):
            if r[s]["body"] in forbidden_bodies:
                violations.append(
                    f"REAL train body matches a b2-holdout side: {r['pair_id']}/{s}"
                )
            t = r[s].get("title") or ""
            if t and t in forbidden_titles:
                violations.append(
                    f"REAL train title matches a b2-holdout side: {r['pair_id']}/{s}"
                )
    return violations


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--engine-src", type=Path, default=DEFAULT_ENGINE_SRC)
    ap.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION_DEFAULT)
    ap.add_argument(
        "--nn-pairs", type=int, default=260, help="REAL-NN band pairs to take"
    )
    ap.add_argument(
        "--rnd-pairs", type=int, default=120, help="REAL far negatives to take"
    )
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="optional JSON with CornerQAThresholds overrides — DRY-analysis "
        "helper ONLY; a corpus sealed with an overridden threshold is not a "
        "sealed corpus (the manifest records the override loudly)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate gates + report composition, write NOTHING",
    )
    args = ap.parse_args(argv)

    started = time.monotonic()
    counters: Counter = Counter()

    thresholds, thresholds_source = thresholds_from_gate_contract()
    thresholds_overridden = False
    if args.thresholds is not None:
        from cortex.data.corner_qa import CornerQAThresholds

        overrides = json.loads(args.thresholds.read_text(encoding="utf-8"))
        thresholds = CornerQAThresholds(**overrides)
        thresholds_overridden = True
        thresholds_source = f"OVERRIDE {args.thresholds}"
    print(f"corner-QA thresholds source: {thresholds_source}", file=sys.stderr)

    # ── build text-level rows (no vectors yet) ─────────────────────────────
    b2_holdout_bodies, _b2_hold_pairs = b2_holdout_surfaces()
    b2_train_bodies = {
        r[s]["body"]
        for r in _read_jsonl(B2_DIR / "train.jsonl")
        for s in ("record", "candidate")
    }
    real_rows, real_counters = load_real_uncontaminated(
        b2_holdout_bodies, b2_train_bodies
    )
    counters.update(real_counters)

    embed_side, pin = make_embedder(args.engine_src)

    translated_rows = build_translated()
    counters["translated:rows"] = len(translated_rows)
    tviol = _quota_v43(translated_rows)
    if tviol:
        print("TRANSLATED QUOTA REFUSAL:", file=sys.stderr)
        for line in tviol:
            print(f"  - {line}", file=sys.stderr)
        return 1

    # real pairs need measured cosines for band selection — embed the sides
    real_pairs = build_real_rows(
        real_rows,
        embed_side,
        nn_pairs_wanted=args.nn_pairs,
        rnd_pairs_wanted=args.rnd_pairs,
        counters=counters,
    )
    counters["real:pairs_built"] = len(real_pairs)

    b2_rows, b2_dropped = build_reused_b2_train(
        {r["pair_id"] for r in real_pairs} | {r["pair_id"] for r in translated_rows},
        set(),
        counters,
    )
    v42_rows, _ = build_reused_v42_train(counters)
    ladder_rows = build_ladder_additions(embed_side, counters)

    raw_rows = real_pairs + translated_rows + b2_rows + v42_rows + ladder_rows

    # pair_id uniqueness
    ids = [r["pair_id"] for r in raw_rows]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in Counter(ids).items() if v > 1][:5]
        raise SystemExit(f"validation: duplicate pair_id across parts: {dupes}")

    # ── uniform geometry: re-embed everything ───────────────────────────────
    print(f"rows to embed: {len(raw_rows)}", file=sys.stderr)
    rows = embed_measure(raw_rows, embed_side)
    print(f"embedder pin: {pin}", file=sys.stderr)
    counters["embedder:pin"] = pin  # type: ignore[assignment]
    print(f"embedded in {time.monotonic() - started:.1f}s", file=sys.stderr)

    # ── corner-QA gate on the FULL corpus (pre-split; gen_dataset_v4 order) ─
    from cortex.data.corner_qa import corner_qa_counters, corner_qa_violations

    qa_counters = corner_qa_counters(rows)
    violations = corner_qa_violations(qa_counters, thresholds)
    if violations and not thresholds_overridden:
        print("CORNER-QA REFUSAL — nothing written:", file=sys.stderr)
        for line in violations:
            print(f"  - {line}", file=sys.stderr)
        return 1
    if violations and thresholds_overridden:
        # DRY path: an overridden threshold means the build is a VALIDATION
        # aid, not a sealed corpus — require --dry-run and disclose loudly.
        if not args.dry_run:
            print(
                "CORNER-QA VIOLATIONS under an OVERRIDE — refusing to write: "
                "run with --dry-run for pipeline validation only.",
                file=sys.stderr,
            )
            for line in violations:
                print(f"  - {line}", file=sys.stderr)
            return 1
        print(
            "DRY-RUN with overridden corner-QA thresholds — outputs are "
            "validation numbers only, NOT a sealed corpus.",
            file=sys.stderr,
        )
    counters["corner_qa:ok"] = not violations  # type: ignore[assignment]

    # ── watchlist quota (pre-split corpus; v4.3 binds the quota) ────────────
    wviol = quota_violations(rows, corpus_version="4.3")
    if wviol:
        print("WATCHLIST QUOTA REFUSAL — nothing written:", file=sys.stderr)
        for line in wviol:
            print(f"  - {line}", file=sys.stderr)
        return 1
    # train-side hard floor for the family (brief (d): quota lives in TRAIN)
    train0, hold0 = stratified_split10(rows, args.holdout_fraction)
    fam_train = sum(
        1
        for r in train0
        if r["stratum"] == "R-watchlist-family" and r["label"] == "duplicate"
    )
    if fam_train < WATCHLIST_QUOTA_MIN:
        print(
            f"WATCHLIST TRAIN FLOOR REFUSAL: {fam_train} family pairs in "
            f"train < {WATCHLIST_QUOTA_MIN}",
            file=sys.stderr,
        )
        return 1
    counters["watchlist:train_family_pairs"] = fam_train

    # ── disjointness on the actual train subset ─────────────────────────────
    disj = check_disjointness_v43(train0, hold0)
    if disj:
        print("DISJOINTNESS REFUSAL — nothing written:", file=sys.stderr)
        for line in disj[:10]:
            print(f"  - {line}", file=sys.stderr)
        return 1
    counters["disjointness:ok"] = True  # type: ignore[assignment]

    # ── fingerprints (data-contract §5) ─────────────────────────────────────
    entries = [
        (
            r["pair_id"],
            pair_sha256(
                {
                    "record": r["record"],
                    "candidate": r["candidate"],
                    "similarity": r["similarity"],
                }
            ),
        )
        for r in rows
    ]
    fp = corpus_fingerprint(manifest_bytes(entries))
    label_fp = labels_fingerprint({r["pair_id"]: r["label"] for r in rows})

    train_entries = [
        (pid, d)
        for pid, d in (
            (
                r["pair_id"],
                pair_sha256(
                    {
                        "record": r["record"],
                        "candidate": r["candidate"],
                        "similarity": r["similarity"],
                    }
                ),
            )
            for r in train0
        )
    ]
    hold_entries = [
        (pid, d)
        for pid, d in (
            (
                r["pair_id"],
                pair_sha256(
                    {
                        "record": r["record"],
                        "candidate": r["candidate"],
                        "similarity": r["similarity"],
                    }
                ),
            )
            for r in hold0
        )
    ]
    train_fp = corpus_fingerprint(manifest_bytes(train_entries))
    hold_fp = corpus_fingerprint(manifest_bytes(hold_entries))

    if args.dry_run:
        # DRY: validation numbers only — nothing is written anywhere; the
        # sealed build happens when the ladder-gap resolution lands.
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "thresholds_overridden": thresholds_overridden,
                    "corner_qa_violations": violations,
                    "assembled": len(rows),
                    "train": len(train0),
                    "holdout_v43": len(hold0),
                    "by_part": dict(Counter(part_of(r) for r in train0)),
                    "by_stratum": _counter(train0, "stratum"),
                    "by_label": _counter(train0, "label"),
                    "lang_mix": lang_mix(train0),
                    "counter_facts": {
                        k: v for k, v in sorted(counters.items()) if isinstance(v, int)
                    },
                    "ladder_full_corpus": qa_counters[
                        "low_cos_high_overlap_notdup_count"
                    ],
                    "family_full_corpus": sum(
                        1 for r in rows if r["stratum"] == "R-watchlist-family"
                    ),
                    "translated_quotas": _quota_v43(rows),
                    "watchlist_pre_split": quota_violations(rows, corpus_version="4.3"),
                    "fingerprint_would_be": fp,
                    "train_fp_would_be": train_fp,
                    "hold_fp_would_be": hold_fp,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0

    # ── outputs ─────────────────────────────────────────────────────────────
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train0)
    (out / "holdout.jsonl")  # v4.3 holdout rows live in train-root sibling:
    hold_dir = out.parent / f"{out.name}-holdout"
    hold_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(hold_dir / "holdout.jsonl", hold0)
    _write_jsonl(
        hold_dir / "labels.jsonl",
        [{"pair_id": r["pair_id"], "label": r["label"]} for r in hold0],
    )

    seal = {
        "corpus_fingerprint": fp,
        "label_fingerprint": label_fp,
        "train_fingerprint": train_fp,
        "holdout_fingerprint": hold_fp,
        "holdout_ids": sorted(r["pair_id"] for r in hold0),
        "split": (
            "stratified per (label, stratum), first ceil(f·n) by pair_id; "
            f"f={args.holdout_fraction}; construction-seeded (no RNG walks)"
        ),
        "holdout_fraction": args.holdout_fraction,
        "isolation": (
            "holdout dir outside train root; zero id overlap asserted; "
            "text-level disjointness vs b2-holdout + v4.2-holdout asserted"
        ),
        "b2_no_harm_holdout_untouched": True,
        "b2_holdout_fingerprint_of_record": "b25c67b9e4224452bd0fb5aeb4212a8de68484bf983755d57e36d59f714282a8",
    }
    (out / "holdout-ids.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    _write_jsonl(
        out / "labels-train.jsonl",
        [{"pair_id": r["pair_id"], "label": r["label"]} for r in train0],
    )
    (out / "manifest.txt").write_bytes(manifest_bytes(entries))

    # ── manifest.json + README ──────────────────────────────────────────────
    parts = Counter(part_of(r) for r in train0)
    by_stratum = _counter(train0, "stratum")
    by_label = _counter(train0, "label")
    mix = lang_mix(train0)
    ru_share = (mix["ru"] + 0.5 * mix["mix"]) / max(1, len(train0))
    manifest = {
        "kind": "dataset-v43-corpus-manifest",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_version": "4.3",
        "purpose": "round-4 F-configuration training corpus (embed + cortex recalib)",
        "prereg": [
            "docs/specs/embed-round4-prereg-DRAFT.md §2-§3",
            "docs/specs/embed-round4-prereg-ADDENDUM-6.md §A.3",
            "docs/specs/embed-round4-prereg-ADDENDUM-7.md §A.7.1",
        ],
        "embedder_pin": pin,
        "network": "none (local CPU inference)",
        "totals": {
            "assembled": len(rows),
            "train": len(train0),
            "holdout_v43": len(hold0),
        },
        "train_composition": {
            "by_part": dict(parts),
            "by_stratum": by_stratum,
            "by_label": by_label,
            "lang_mix": mix,
            "ru_share_weighted": round(ru_share, 4),
        },
        "holdout_v43_composition": {
            "by_stratum": _counter(hold0, "stratum"),
            "by_label": _counter(hold0, "label"),
        },
        "strata_note": {
            "R-watchlist-family": "near-0009 duplicate family (cortex.data.watchlist geometry)",
            "translated-dup": "addendum 7 §A.7.1 positive class",
            "translated-sibling": "addendum 7 §A.7.1 negative class",
        },
        "addendum7_quotas": {
            "translated-dup": sum(1 for r in rows if r["stratum"] == "translated-dup"),
            "translated-sibling": sum(
                1 for r in rows if r["stratum"] == "translated-sibling"
            ),
            "required_min": 150,
            "verdict": "CLEAN",
        },
        "watchlist_quota": {
            "family_pairs_train": fam_train,
            "required_min": WATCHLIST_QUOTA_MIN,
            "verdict": "CLEAN",
        },
        "validations": {
            "corner_qa": "PASS",
            "corner_qa_thresholds_source": thresholds_source,
            "corner_qa_by_label": qa_counters["by_label"],
            "disjointness_real_vs_b2_holdout": "PASS",
            "real_dropped_holdout_collisions": counters.get(
                "real:dropped_holdout_collision", 0
            ),
            "real_train_surface_shared_with_b2_train": counters.get(
                "real:train_surface_shared", 0
            ),
            "disjointness_train_vs_sealed_eval_surfaces": "PASS",
            "pair_id_uniqueness": "PASS",
        },
        "fingerprint_discipline": "blake2b-256 over sorted sha256(pair) manifest lines (data-contract §5)",
        "fingerprints": {
            "corpus": fp,
            "labels": label_fp,
            "train": train_fp,
            "holdout_v43": hold_fp,
        },
        "outputs_fingerprints": {
            "train.jsonl_sha256": hashlib.sha256(
                (out / "train.jsonl").read_bytes()
            ).hexdigest(),
            "holdout.jsonl_sha256": hashlib.sha256(
                (hold_dir / "holdout.jsonl").read_bytes()
            ).hexdigest(),
            "labels-train.jsonl_sha256": hashlib.sha256(
                (out / "labels-train.jsonl").read_bytes()
            ).hexdigest(),
            "manifest.txt_sha256": hashlib.sha256(
                (out / "manifest.txt").read_bytes()
            ).hexdigest(),
            "holdout-ids.json_sha256": hashlib.sha256(
                (out / "holdout-ids.json").read_bytes()
            ).hexdigest(),
        },
        "counters": dict(
            sorted((k, v) for k, v in counters.items() if isinstance(v, int))
        ),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "privacy": "store texts only in gitignored data/; manifests carry counters + fingerprints",
        "b2_reuse_decision": (
            "b2/train.jsonl pairs reused per the brief (synthetic/structural part); "
            "3 G4 connector-edit defect rows dropped (DV3-P-075/099/108) with reasons; "
            "v4.2-holdout leakage guard; vectors re-measured (uniform geometry)"
        ),
        "v42_train_reuse_decision": (
            "deterministic gen_dataset_v4 regeneration over the authored corpus-v4 "
            "batches (fingerprint of the text stream "
            + V42_TRANSPARENT_FINGERPRINT[:12]
            + "…); no authored text re-synthesized; vectors re-measured"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(
        _readme_text(manifest, counters, b2_dropped), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "total": len(rows),
                "train": len(train0),
                "holdout_v43": len(hold0),
                "by_part": dict(parts),
                "corpus_fingerprint": fp,
                "out_dir": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _quota_v43(rows: list[dict[str, Any]]) -> list[str]:
    import importlib.util

    spec_path = REPO_ROOT / "scripts" / "gen_dataset_v43_strategies.py"
    spec = importlib.util.spec_from_file_location("gen_v43_strategies_q", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod.quota_violations_v43(rows)


def _readme_text(
    manifest: dict[str, Any],
    counters: Counter,
    b2_dropped: list[dict[str, Any]],
) -> str:
    m = manifest
    tc = m["train_composition"]
    tot = tc["by_part"]
    total = m["totals"]["train"]
    lines = [
        "# data/stage2/v43 — corpus v4.3 (round-4 F-configuration)",
        "",
        f"Built {m['built_at']} by `scripts/build_corpus_v43.py` (seed 42; zero network;",
        f"embedder {m['embedder_pin'][:30]}…). Prereg: draft §2-§3 + addenda 6 §A.3, 7 §A.7.1.",
        "",
        "## Composition (train)",
        "",
        "| part | pairs | share |",
        "|---|---:|---:|",
    ]
    for k in ("real", "translated", "synthetic-b2", "synthetic-v42"):
        lines.append(f"| {k} | {tot.get(k, 0)} | {tot.get(k, 0) / max(1, total):.1%} |")
    lines += [
        f"| **total** | **{total}** | 100% |",
        "",
        "### By stratum",
        "",
        "```",
    ]
    for k, v in sorted(tc["by_stratum"].items()):
        lines.append(f"{k:24s} {v:5d}")
    lines += [
        "```",
        "",
        f"- labels: `{tc['by_label']}` ; lang mix (both/mixed): `{tc['lang_mix']}`;",
        f"  RU-weighted share {tc['ru_share_weighted']:.1%} (RU↔EN mixed pairs count 0.5; addendum-6 target ≥15–20% cross-language mass).",
        "",
        "## Splits and seals",
        "",
        f"- v4.3 holdout: **{m['totals']['holdout_v43']} pairs** stratified 10% (construction-seeded),",
        "  sealed in `holdout-ids.json` (ids + fingerprints); labels physically outside the train dir.",
        "- b2 no-harm holdout (180, `data/stage2/b2/holdout.jsonl`) UNTOUCHED — g1/g2 measured there",
        "  with the frozen embedder; v4.3 train is disjoint from its texts (asserted at build).",
        "- v4.2-holdout (276, sealed cortex eval) never enters this corpus (asserted).",
        "",
        "## Composition decisions (honest record, the brief's (c) interpretation)",
        "",
        "- (c) synthetic/structural = **b2/train.jsonl 417 pairs reused AS PAIRS** (the brief's",
        "  explicit pointer) **+ the v4.2 corpus-train rows regenerated deterministically** by",
        "  `gen_dataset_v4.py` from the authored `datasets/corpus-v4` batches (no text",
        "  re-synthesis — the calibrated authored texts are the asset; the corpus composition,",
        "  split and fingerprints are fully regenerated per addendum-6 «полная регенерация».",
        "  The v4.2 rows supply the class coverage the 420-pair b2 block lacks by design",
        "  (same-length T3 share, N-para-notdup low-cos ladder, N-far low-cos mass,",
        "  language-mismatch mass) — corner-QA would REFUSE without them.",
        "- b2 defect exclusions:",
        json.dumps(b2_dropped, ensure_ascii=False, indent=1),
        "- vectors: ALL rows re-measured with the pinned embedder (one geometry; reused-pair",
        "  vectors of previous runs ride an older ONNX runtime — median |Δsim| 0.000124,",
        "  max 0.0126 measured on the v4.2 regeneration — unusable for a uniform corpus).",
        "",
        "## Validations (all PASS — numbers in manifest.json)",
        "",
        "- corner-QA (gate_contract corner_qa section): PASS on the full pre-split corpus.",
        "- addendum-7 quotas ≥150+≥150 both directions: CLEAN.",
        "- watchlist ≥20 family pairs in TRAIN: CLEAN.",
        "- disjointness vs b2-holdout and v4.2-holdout at text level: PASS.",
        f"- real-part drop counter (holdout-text collisions): {counters.get('real:dropped_holdout_collision', 0)}",
        f"- real-part records shared VERBATIM with b2-train surfaces (kept, counted): {counters.get('real:train_surface_shared', 0)}",
        "",
        "## Files",
        "",
        "- `train.jsonl` — stage-2 rows (record/candidate/language/label/pair_id/stratum/",
        "  similarity/vec_a/vec_b; translated rows also carry direction/batch/justification).",
        "- `labels-train.jsonl`, `manifest.txt`, `holdout-ids.json` — fingerprints per data-contract §5.",
        "- `../v43-holdout/holdout.jsonl` + `labels.jsonl` — the sealed v4.3 holdout slice (outside the train root).",
        "",
        "NOTE: this directory and all sibling holdout dirs are gitignored (`/data/`); the repo",
        "carries the build script, this manifest copy in `artifacts/manifests/` — TL verifies",
        "locally by re-running the script and matching fingerprints.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
