#!/usr/bin/env python3
"""Train the merge-arbiter Choice model ``vesma-cortex-merge-v1`` + gate runner.

Spec: docs/specs/merge-arbiter-corpus-prereg-DRAFT.md (v2 ACTIVATED, MLC
28.10 edits N1-N9) + ADDENDUM-1 (committee-silver, MLC 29.10). This is the
TRAIN-ONLY phase of the gate cycle: thresholds are fitted by train-only CV
(draft §2.2, edit N2) and frozen BEFORE any sealed read; the single-shot
gate phase is implemented (``gate`` subcommand) but refuses to run without
an explicit flag and is NOT executed in this phase.

Model (draft §2.2, §7 pos. 1):
  LightGBM binary at MEMBER level, symmetric listwise/eligibility
  parameterization. A member's features aggregate over the OTHER members
  (mean-over-others); the head outputs an eligibility score e_i in [0,1].
  Decision (ratified): keep = argmax_i e_i iff e_(1) >= tau_hi AND
  e_(1) - e_(2) >= tau_m, else ABSTAIN. The whole decision freedom is the
  two scalars (tau_hi, tau_m), fitted ONLY by K-fold CV over clusters
  (pairwise-disjoint folds) on the train corpus and frozen into the
  artifact before any holdout contact.

How abstention is achieved by thresholds (draft §2.2/§7 pos. 1):
  abstention is NOT a third class of the classifier. CH-2/CH-4 clusters
  enter training with ALL member rows labeled 0 (no member is eligible) —
  the head learns to score every member of such configurations low, so at
  decision time e_(1) falls below tau_hi and the rule abstains. CH-3 rows
  teach a score GAP (superset ~1, subset ~0) so the margin clears tau_m.
  CH-4 is deliberately in train (the abstain leg) because its predicate
  needs fact sets that production does not extract (§5.1) — the MODEL
  carries the CH-4 prohibition, and g_fold_exact measures the through
  pipeline. CH-0/CH-5 never reach the model leg (code tie-break, §5.1
  steps 1-2) and the real strata is gate-only (§3.1) — none of them is
  in train.

Features (A-frozen + A-null-pin; the numeric-vs-text choice, documented):
  BLOCK P (13) — the FROZEN pair contract ``cortex.features.pair.features``
  averaged over others. Import, zero reimplementation. ``cos_target``
  enters as the DECLARED constant 1.0 (v1 boundary: corpus v1 carries no
  measured vector leg — synthetic members have no store embedding; the
  column is inert). BLOCK D (3) — directional text relations on the
  frozen normalizer (norm1 substring direction, signed relative length):
  these break the k=2 feature degeneracy of purely symmetric pair
  features while staying permutation-invariant (a function of the member
  SET and the member, never of serialization order). BLOCK E (6) —
  edit-region statistics on the pinned lib's norm2 machinery: edit hull
  count, per-side divergent chars, relative divergence, digit fraction of
  each side's divergent segments — the channel that separates CH-3 (one
  fact-bearing side) from CH-4 (both sides fact-bearing) WITHOUT a
  production fact extractor; numeric surface statistics, not declared
  fact sets. BLOCK V (3) — envelope per A-null-pin: created_at ISO-8601
  UTC -> epoch seconds + separate null flag + signed mean delta; null
  path: flag=1, epoch=0, delta=0 (pin fixed in manifest.null_datetime_pins
  BEFORE corpus freeze, N8). Length enters only RELATIVELY (no raw
  own-length column): the model must survive the short-superset
  sub-strata structurally; sanity probe 4 asserts exactly that.

Order-invariance at the accumulation level: per-member terms are averaged
in the SORTED-mid order of the other members, so the feature row of a
member is BIT-identical under any presentation permutation (asserted by
a machine check, not assumed).

Discipline (ML-STD-0003 for every run; ML-STD-0002 for the tool):
  run-log append-only (artifacts/runs/run_log.jsonl); prologue order
  swap_check -> env -> host -> byte_tripwire -> activation -> frozen_inputs
  -> gate_start (gate phase) with EVERY eval computation (sanity suite,
  fixtures, no-harm, sealed reads) AFTER gate_start; fail-closed
  InstrumentAbort on any prologue mismatch (STOP class "instrument", not
  a gate red; sealed surfaces stay sealed); no seal of another surface is
  opened (holdout v4.5 / g4-spec-v1 / spot-sheet are NOT in the input
  list); resume = full repeated prologue; re-run after single-shot is
  refused by the runner (П8).

Determinism: LightGBM deterministic=True + force_row_wise=True, fixed
seed, fixed feature order; the trainer fits twice in-process and refuses
on booster byte-mismatch (P7-style evidence).

Resource declaration (P9, draft §9 light lane): LightGBM-class cycle,
precedent 57-78 s / ~640 MB; declared: prologue <= 2 s, feature build +
CV + final fit <= 180 s wall, peak RSS <= 1.5 GB. RAM overrun = stop
condition, not a surprise.

Usage:
  <venv-python> scripts/merge_arbiter_train.py train [--run-tag TAG]
  <venv-python> scripts/merge_arbiter_train.py sanity --artifact PATH
  <venv-python> scripts/merge_arbiter_train.py gate --artifact PATH \
      --noharm PATH.jsonl --execute-single-shot   # NOT run in this phase

Choice v2 (ADDENDUM-6, MLC 30.10 round 6; one cycle, no cuts):
  <venv-python> scripts/merge_arbiter_train.py train-v2     # 34-feature contract + zone-v1 contour + CV riders
  <venv-python> scripts/merge_arbiter_train.py floors-v2 --artifact ART --doc-out DOC   # ML-STD-0004 stage 2 (declared floor)
  <venv-python> scripts/merge_arbiter_train.py prelaunch --artifact ART --profile v2    # incl. zone-off probe + sheet prematch
All v1 subcommands keep their exact behaviour when the v2 files/ctx are absent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import merge_arbiter_lib as lib  # noqa: E402  (frozen build lib, sha-pinned)
import build_merge_arbiter_corpus as mb  # noqa: E402  (frozen recipe: gen_family)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402

# --- paths -------------------------------------------------------------------

CORPUS_DIR = REPO / "data" / "merge-arbiter" / "corpus-v1"
TRAIN_JSONL = CORPUS_DIR / "train.jsonl"
MANIFEST = CORPUS_DIR / "manifest.json"
HOLDOUT_JSONL = CORPUS_DIR / "holdout.jsonl"  # SEALED — never opened in this phase
NEG_PATH = CORPUS_DIR / "train-negatives.jsonl"  # v1.1 extension (ADDENDUM-3)
NULL_PATH = CORPUS_DIR / "null-envelope.json"  # null coverage (ADDENDUM-4)
SEAL_DIR = REPO / "data" / "merge-arbiter" / "seals"  # never opened in this phase
ART_DIR = REPO / "data" / "merge-arbiter" / "artifacts" / "merge-v1"
COMMITTED_DIR = REPO / "datasets" / "merge-arbiter"
COMMITTED_DOC = COMMITTED_DIR / "merge-v1-thresholds.frozen.json"
RUNLOG = REPO / "artifacts" / "runs" / "run_log.jsonl"
SPEC = REPO / "docs" / "specs" / "merge-arbiter-corpus-prereg-DRAFT.md"

SLICE_TRAIN = "merge-arbiter-merge-v1-train"
SLICE_GATE = "merge-arbiter-merge-v1-gate"
SLICE_FLOOR = "merge-arbiter-merge-v1-floor-seed"

#: Activation commits (P6): the corpus prereg v2 ACTIVATED commit and the
#: ADDENDUM-1 commit must both be ancestors of HEAD before any run.
ACTIVATION_COMMITS = {
    "prereg-v2-activated": "e3a0d1e",
    "addendum-1-ratified": "049064f",
}

# --- env pins (P2, fail-closed) ----------------------------------------------

#: The ratified light-lane env (ML-STD-0003 P2 reference combo: python
#: 3.12.13 / lightgbm 4.7.0) lives in the project venv. This trainer runs
#: on the PROJECT'S OWN venv (vesma-cortex/.venv — explicitly allowed;
#: no other product's venv is touched, nothing is installed into it).
EXPECTED_ENV = {
    "venv_suffix": "vesma-cortex/.venv/bin/python",
    "python": "3.12.13",
    "lightgbm": "4.7.0",
    "numpy": "2.5.3",
}
EXPECTED_HOST = "core-51"  # same host as the corpus build (run-log env event)

# --- training config (committed with the artifact; no drift) -----------------

SEED = 42
CV_K = 5
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "seed": SEED,
    "verbose": -1,
}
#: Threshold grid — small, fixed, train-only (draft §2.2).
TAU_HI_GRID = [round(0.50 + 0.05 * i, 2) for i in range(9)]   # 0.50..0.90
TAU_M_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

#: Train strata counts vs the ratified quota table (P4/П5 count assert).
#: v1.1 (ADDENDUM-3): the theme-near negative block (200) rides in
#: train-negatives.jsonl — the extension, not a re-emit; the 600 stay
#: byte-identical (the v1 train pin still enforced by the tripwire).
EXPECTED_TRAIN_COUNTS = {
    ("CH-3", 2): 240, ("CH-3", 3): 12,
    ("CH-0", 2): 90, ("CH-0", 3): 18,
    ("CH-2", 2): 90, ("CH-4", 2): 90, ("CH-5", 2): 60,
}
EXPECTED_NEG_COUNTS = {"cross-lang-sibling": 80, "template-twin": 60,
                       "near-topic": 60}
#: Manifest assert-ledger holdout strata (gate-time count assert, П5).
EXPECTED_HOLDOUT_STRATA = {"CH-3": 25, "CH-0": 11, "CH-2": 9, "CH-4": 9, "CH-5": 6}

#: CV-noregress tolerance — FIXED BEFORE RETRAIN (ADDENDUM-3 §2).
#: 0.02 = one acc quantum of a ~50-cluster CH-3 fold; the reference is
#: the v1 artifact's OOF CH-3 acc (aggregate 1.0 over cluster quanta ⇒
#: every fold 1.0). New-model per-fold OOF CH-3 acc must stay >= 0.98.
NOREGRESS_TOL = 0.02
#: CV-neg-abstain floor (ADDENDUM-3 §2): per-substratum OOF abstain rate.
NEG_ABSTAIN_MIN = 0.9

# --- delta_acc floor measurement (MLC 29.10 ratification, option (a)) --------
#
# Ratified procedure (TL dispatch 29.10, three unanimous positions, chairman-
# consolidated): floor_seed for delta_acc is measured at the GRANULARITY OF
# THE GATING STATISTIC (the CH-3 aggregate over the gate-leg composition:
# 22 k2 + 3 chain = the §3.2 CH-3 holdout quotas); the per-stratum
# floor_seed table in the build manifest is a characterization artifact and
# does NOT enter the formula. 10 composition draws, seeds 43..52, frozen
# recipe (the build generator, same code path), the FROZEN merge-v1 model
# (no retraining), delta_d = acc_choice(model, draw d) - max(baseline on d);
# floor_seed_aggregate := max pairwise |delta_d - delta_e| (max_abs_delta
# semantics); delta_acc := max(2 * floor_seed_aggregate; 0.10). Fixed BEFORE
# gate_start; re-measuring after the single-shot is FORBIDDEN.

FLOOR_SEEDS = list(range(43, 53))  # [43..52], pinned in run-log at measure start
#: The gate leg CH-3 composition (holdout §3.2 CH-3 quotas).
FLOOR_LEG_QUOTA = {"CH-3-k2": 22, "CH-3-chain": 3}
#: Zero-filled to the full build quota shape (gen_family reads every key).
_FLOOR_QUOTA_FULL = {
    "CH-3-k2": 22, "CH-3-chain": 3,
    "CH-0-k2": 0, "CH-0-triple": 0, "CH-2": 0, "CH-4": 0, "CH-5": 0,
}

# --- Choice v2 package (ADDENDUM-6, MLC 30.10 round 6; owner sanction 30.10) --
#
# ONE ratification, ONE retrain: zone leg 3.5 (zone-v1) + e_cos/zeps member
# features + the 4 negative strata (generated by the DE slice, commit 4ee4ec4).
# Everything here is ADDITIVE: with the v2 files absent and ctx=None the v1
# paths (train / sanity / floor-seed / floors-composition / prelaunch / gate)
# behave exactly as before — the iter-1..3 artifacts stay reproducible.

NEG2_PATH = CORPUS_DIR / "train-negatives-v2.jsonl"  # v2 extension (ADDENDUM-6 §2 pos 5)
NULL2_PATH = CORPUS_DIR / "null-envelope-v2.json"  # null coverage v2 (seed 9405)

#: v2 tripwire pins — the DE generation run (slice merge-arbiter-corpus-v2-negatives,
#: run-log events negatives_v2_emitted / null_coverage_v2_emitted, 2026-10-30).
NEG2_EXPECTED_SHA256 = "e601071a1542c4f4ae68542aed4315376534e371c61d1b9f6565355f117cf791"
NULL2_EXPECTED_SHA256 = "729582c6db56f6e84f23416b5fb9094ec66b03098f182bf6f36c507ce287b923"
#: corpus_fp_v2 = sha256(sha(train) : sha(neg1) : sha(neg2) : sha(null2)) —
#: recomputed by the v2 tripwire, never trusted from the sidecar alone.
CORPUS_FP_V2_EXPECTED = "c44c5491b5a49113c68e13e74b3ef1f335f9b68e94c89beda0b5175f4c3eacf9"

#: v2 negative strata row counts (pairs x 2 orientations, sidecar manifest
#: quota_rows; ADDENDUM-6 §7.1 execution record).
EXPECTED_NEG_V2_COUNTS = {"zone-boundary": 56, "cos-bins": 38,
                          "sibling-residue": 24, "digit-lookalike": 60}

#: Zone leg 3.5 (ADDENDUM-6 §2 pos 1). Constants are fixed BY RATIFICATION
#: (derived from the structural train-support voids train-only before seal;
#: owner directive 30.10: «константы уже фиксированы — не пересчитывай»).
#: One-sidedness (§1 leg 2): the leg emits ONLY abstain — the keep side of
#: both axes is read nowhere in the contour.
ZONE_V1 = {
    "version": "zone-v1",
    "rule": "(cos <= 0.85 AND char5 <= 0.05) OR (zeps >= 3 AND char5 <= 0.20)",
    "tau_cos": 0.85, "tau_lex": 0.05, "tau_lex2": 0.20, "zeps_min": 3, "eps": 0.01,
    "position": "leg 3.5 of §5.1: between the CH-2 refusal and the model leg",
    "pairness": "evaluates the PAIR cos of the candidate (nano, norm1); "
                "any unordered pair of the cluster inside the zone fires",
}

#: Known-uncovered pockets of zone-v1 (ADDENDUM-6 §2 errata/ML list) —
#: frozen verbatim into the v2 committed doc.
ZONE_V1_KNOWN_UNCOVERED = [
    "cos < 0.60 entirely (below the corpus coverage floor — no train support)",
    "char5 > 0.20 AND cos <= 0.85 AND zeps < 3 (residual band above the OR-leg)",
    "monolingual near-boundary pairs outside the TT/NT forms",
]

#: Residual risk accepted by ratification (ADDENDUM-6 §2 errata 2).
ZONE_V1_RESIDUAL_RISK = ("max char5 of the refused class 0.04947 -> margin to "
                         "tau_lex 0.05 is 0.0005 with zeps=1 (the OR-leg does "
                         "not insure it); accepted in frozen-doc zone-v1")

#: ML-STD-0004 (declared floor): stage-1 formula pin (committed BEFORE any v2
#: draw) and the ratified formula; stage 2 = >= 10 draws, distribution
#: (median, p5, p95, max-pairwise) published before gate_start, delta_acc
#: frozen from the DECLARED statistics.
STD0004_FORMULA_COMMIT = "5305c82"
DELTA_ACC_FORMULA = "delta_acc = max(p95 - p5; 0.10)"

#: The v1-iter3 frozen artifact (framing M5 pin): CV-noregress reference and
#: the zone-off probe base (reproduces keep = 24/171 exactly).
ITER3_ARTIFACT_SHA_PREFIX = "19bf073e"

#: v2 slice names (run-log discipline; the v1 slice names stay untouched).
SLICE_TRAIN_V2 = "merge-arbiter-merge-v2-train"
SLICE_FLOOR_V2 = "merge-arbiter-merge-v2-floors"
SLICE_GATE_V2 = "merge-arbiter-merge-v2-gate"

#: Persistent nano embedding cache (keyed by sha256 of the truncated text).
#: Derived data — lives under data/ (gitignored); phases reuse it so train /
#: floors / prelaunch never re-embed the same text.
EMBED_CACHE_PATH = (REPO / "data" / "merge-arbiter" / "artifacts" /
                    "embed-cache" / "nano-3b752e06.v1.jsonl")

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_SLICE = SLICE_TRAIN  # subcommands re-point this before their first log()


class InstrumentAbort(Exception):
    """Fail-closed abort, STOP class 'instrument' (П6 ML-STD-0002).

    Nothing downstream may run; no sealed surface is read; escalation to
    TL, not input regeneration.
    """


# --- run-log (append-only) ---------------------------------------------------

def log(event: str, **fields) -> None:
    row = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "slice": _SLICE,
        "event": event,
        **fields,
    }
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    with open(RUNLOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(f"[run-log] {event} {fields if fields else ''}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_doc_for(artifact_sha: str) -> dict:
    """Resolve the committed frozen doc whose durable artifact pin matches
    the artifact bytes (v1 doc or a v1.1-class successor)."""
    candidates = sorted(COMMITTED_DIR.glob("merge-v1-thresholds.frozen*.json"))
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if doc.get("durable_artifact", {}).get("sha256") == artifact_sha:
            doc["_doc_path"] = str(path)
            return doc
    raise InstrumentAbort(
        f"no committed frozen doc pins an artifact with sha {artifact_sha[:12]}… "
        f"(searched {candidates}) — the artifact has no ratified pin")


def v2_tripwire() -> dict:
    """Choice-v2 byte tripwire (ADDENDUM-6 §2 pos 5/6): the v2 extension
    files vs the DE generation pins; corpus_fp_v2 is RECOMPUTED over the
    four corpus files, never trusted from a sidecar."""
    neg2_manifest = json.loads((CORPUS_DIR / "train-negatives-v2.manifest.json")
                               .read_text(encoding="utf-8"))
    pins = {
        "neg2_sha256": sha256_file(NEG2_PATH),
        "null2_sha256": sha256_file(NULL2_PATH),
        "train_jsonl_sha256": sha256_file(TRAIN_JSONL),
        "neg1_sha256": sha256_file(NEG_PATH),
        "corpus_fp_v2": sha256_text(
            sha256_file(TRAIN_JSONL) + ":" + sha256_file(NEG_PATH) + ":" +
            sha256_file(NEG2_PATH) + ":" + sha256_file(NULL2_PATH)),
    }
    want = {
        "neg2_sha256": NEG2_EXPECTED_SHA256,
        "null2_sha256": NULL2_EXPECTED_SHA256,
        "train_jsonl_sha256": neg2_manifest["base_train_sha256"],
        "neg1_sha256": neg2_manifest["base_negatives_v1_sha256"],
        "corpus_fp_v2": CORPUS_FP_V2_EXPECTED,
    }
    mism = [k for k, v in want.items() if pins[k] != v]
    log("v2_byte_tripwire", pins=pins, expected=want, mismatch=mism,
        sidecar_manifest={"negatives_v2_sha256": neg2_manifest["negatives_v2_sha256"],
                          "seeds": neg2_manifest["seeds"],
                          "quota_rows": neg2_manifest["quota_rows"],
                          "cross_share": neg2_manifest["negative_mass"]["share_string"]})
    if mism:
        raise InstrumentAbort(f"v2 byte tripwire mismatch: {mism}")
    if neg2_manifest["negatives_v2_sha256"] != NEG2_EXPECTED_SHA256:
        raise InstrumentAbort("v2 sidecar manifest disagrees with the pinned "
                              "negatives-v2 sha")
    return pins


# --- prologue (ML-STD-0003 П1-П6, П9) ----------------------------------------

def prologue(expected_artifact: Path | None = None) -> dict:
    """Full machine prologue; order enforced by this runner, fail-closed."""
    # П9: first event = resource state of the machine (swap_check).
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    try:
        avail_mb = int(free.splitlines()[1].split()[6])
        mem_ok = avail_mb >= 2000  # light lane needs << 1.5 GB; margin
    except (IndexError, ValueError):
        avail_mb, mem_ok = -1, False
    log("swap_check", swapon_lines=swap.strip().splitlines(),
        free_mem_line=free.strip().splitlines()[1] if len(free.splitlines()) > 1 else "",
        available_mb=avail_mb, ram_overrun_stop=bool(not mem_ok))
    if not mem_ok:
        raise InstrumentAbort(f"RAM below declared floor (free -m): {avail_mb} MB")

    # П2: env identity — exact pins, fail-closed.
    exe = sys.executable
    env = {
        "executable": exe,
        "venv_suffix_ok": exe.endswith(EXPECTED_ENV["venv_suffix"]),
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
    }
    mism = [k for k in ("venv_suffix_ok",) if not env[k]] + [
        k for k in ("python", "lightgbm", "numpy") if env[k] != EXPECTED_ENV[k]
    ]
    log("env", **env, expected=EXPECTED_ENV, mismatch=mism)
    if mism:
        raise InstrumentAbort(f"env pin mismatch: {mism} (expected {EXPECTED_ENV})")

    # П3: host pin.
    host = platform.node()
    log("host", hostname=host, expectation=EXPECTED_HOST +
        " — same host as the corpus build; cross-host needs a TL decision BEFORE the run")
    if host != EXPECTED_HOST:
        raise InstrumentAbort(f"host changed vs expectation {EXPECTED_HOST}: {host}")

    # П4: byte tripwire against the ratified build pins.
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pins = {
        "merge_arbiter_lib_sha256": sha256_file(Path(lib.__file__)),
        "cortex_features_pair_sha256": sha256_file(
            REPO / "src" / "cortex" / "features" / "pair.py"),
        "feature_names_sha256": sha256_text(repr(lib.FEATURE_NAMES)),
        "train_jsonl_sha256": sha256_file(TRAIN_JSONL),
        "manifest_sha256": sha256_file(MANIFEST),
        "embedder_pin": lib.EMBEDDER_PIN,
    }
    want = {
        "merge_arbiter_lib_sha256": manifest["tool"]["lib_sha256"],
        "cortex_features_pair_sha256": manifest["env"]["cortex_features_pair_sha256"],
        "feature_names_sha256": manifest["env"]["feature_names_sha256"],
        "train_jsonl_sha256": manifest["synthetic_train"]["sha256"],
        "embedder_pin": manifest["env"]["embedder_pin"],
    }
    if expected_artifact is not None:
        # artifact tripwire: bytes vs the committed frozen doc, and the doc's
        # corpus pin vs the manifest (artifact <-> corpus binding chain).
        # Doc resolution: the doc whose durable_artifact.sha256 matches the
        # artifact bytes (v1 or a v1.1-class successor doc).
        pins["artifact_sha256"] = sha256_file(expected_artifact)
        doc = find_doc_for(pins["artifact_sha256"])
        want["artifact_sha256"] = doc["durable_artifact"]["sha256"]
        if doc["train_jsonl_sha256"] != manifest["synthetic_train"]["sha256"]:
            raise InstrumentAbort(
                "committed doc's train pin diverges from the manifest — "
                "artifact/corpus binding broken")
    trip_mism = [k for k, v in want.items() if pins[k] != v]
    log("byte_tripwire", pins=pins, expected=want, mismatch=trip_mism)
    if trip_mism:
        raise InstrumentAbort(f"byte tripwire mismatch: {trip_mism}")

    # П6: activation assert — ratified doc at HEAD, commits in ancestry.
    spec_text = SPEC.read_text(encoding="utf-8")
    activated = bool(re.search(
        r"\*\*Статус:\*\*\s*ACTIVATED v2 \(29\.10\)\s*\+\s*АДДЕНДУМ 1", spec_text))
    ancestry = {}
    for name, commit in ACTIVATION_COMMITS.items():
        r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                           cwd=REPO, capture_output=True)
        ancestry[name] = {"commit": commit, "is_ancestor": r.returncode == 0}
    ok = activated and all(v["is_ancestor"] for v in ancestry.values())
    log("activation", spec_sha256=sha256_text(spec_text),
        status_line_present=activated, commits=ancestry, ok=ok)
    if not ok:
        raise InstrumentAbort("activation assert failed: ratified spec not at HEAD")

    return {"manifest": manifest, "pins": pins}


def frozen_inputs(manifest: dict) -> list["lib.Cluster"]:
    """П5: sha pins + count asserts against the ratified table.

    Reads ONLY train.jsonl + manifest.json. The sealed holdout file and
    the seals directory are NOT in the train input list (A-seal-intact):
    no path under them is opened, hashed, or stat'ed in this phase.
    """
    clusters = load_train()
    counts: dict[tuple[str, int], int] = {}
    for c in clusters:
        counts[(c.stratum, len(c.members))] = counts.get((c.stratum, len(c.members)), 0) + 1
    count_mism = {f"{s}/k{k}": {"got": v, "want": EXPECTED_TRAIN_COUNTS[(s, k)]}
                  for (s, k), v in counts.items()
                  if (s, k) in EXPECTED_TRAIN_COUNTS
                  and EXPECTED_TRAIN_COUNTS[(s, k)] != v}
    extra = [f"{s}/k{k}" for (s, k) in counts
             if s not in {st for st, _k in EXPECTED_TRAIN_COUNTS}
             and s != "theme-near"]
    missing = [f"{s}/k{k}" for (s, k) in EXPECTED_TRAIN_COUNTS if (s, k) not in counts]
    # v1.1 extension counts (ADDENDUM-3): negatives ride the sidecar manifest
    neg_counts: dict[str, int] = {}
    if NEG_PATH.exists():
        neg_manifest = json.loads((CORPUS_DIR / "train-negatives.manifest.json")
                                  .read_text(encoding="utf-8"))
        neg_sha = sha256_file(NEG_PATH)
        if neg_sha != neg_manifest.get("negatives_sha256"):
            raise InstrumentAbort("train-negatives.jsonl diverges from its "
                                  "sidecar manifest pin")
        for c in clusters:
            if c.stratum == "theme-near" and c.substratum in EXPECTED_NEG_COUNTS:
                neg_counts[c.substratum] = neg_counts.get(c.substratum, 0) + 1
        if neg_counts != EXPECTED_NEG_COUNTS:
            count_mism["theme-near-v1"] = {"got": neg_counts, "want": EXPECTED_NEG_COUNTS}
    # Choice-v2 extension counts (ADDENDUM-6 §2 pos 5): own file, own sidecar
    # pins (the v2 byte tripwire runs separately in the v2 subcommands).
    neg2_counts: dict[str, int] = {}
    neg2_manifest = None
    if NEG2_PATH.exists():
        neg2_manifest = json.loads((CORPUS_DIR / "train-negatives-v2.manifest.json")
                                   .read_text(encoding="utf-8"))
        for c in clusters:
            if c.stratum == "theme-near" and c.substratum in EXPECTED_NEG_V2_COUNTS:
                neg2_counts[c.substratum] = neg2_counts.get(c.substratum, 0) + 1
        if neg2_counts != EXPECTED_NEG_V2_COUNTS:
            count_mism["theme-near-v2"] = {"got": neg2_counts,
                                           "want": EXPECTED_NEG_V2_COUNTS}
        unknown = sorted({c.substratum for c in clusters
                          if c.stratum == "theme-near"
                          and c.substratum not in EXPECTED_NEG_COUNTS
                          and c.substratum not in EXPECTED_NEG_V2_COUNTS})
        if unknown:
            count_mism["theme-near-unknown-substrata"] = unknown
    # null-coverage asserts on the EFFECTIVE corpus, PER ENVELOPE (v1:
    # ADDENDUM-4 quotas on the CH-* strata; v2: ADDENDUM-6 §7.6 quotas on the
    # new theme-near strata) — both-null semantics, 0 mixed-null, ~1/3 shares.
    null_info = None
    if NULL_PATH.exists():
        mixed: list[str] = []
        env_infos = []
        for env_path in ([NULL_PATH, NULL2_PATH] if NULL2_PATH.exists()
                         else [NULL_PATH]):
            env_doc = json.loads(env_path.read_text(encoding="utf-8"))
            env_ids = set(env_doc["cluster_ids"])
            per_stratum: dict[str, int] = {}
            per_key: dict[str, int] = {}
            for c in clusters:
                nulls = [m.created_at is None for m in c.members]
                if any(nulls) and not all(nulls):
                    mixed.append(c.cid)
                if c.cid in env_ids:
                    if not all(nulls):
                        mixed.append(c.cid + ":pin")
                    base = _stratum_of(c)
                    per_stratum[base] = per_stratum.get(base, 0) + 1
                    per_key[stratum_key(c)] = per_key.get(stratum_key(c), 0) + 1
            want_nulls = dict(env_doc["quota"])
            got_nulls = dict(per_stratum)
            null_mism = {k: {"got": got_nulls.get(k), "want": v}
                         for k, v in want_nulls.items() if got_nulls.get(k) != v}
            extra_nulls = {k: v for k, v in per_stratum.items() if k not in want_nulls}
            if null_mism or extra_nulls:
                raise InstrumentAbort(
                    f"null quota mismatch ({env_path.name}): {null_mism} "
                    f"extra={extra_nulls}")
            shares = {k: round(v / sum(1 for c in clusters
                                       if _stratum_of(c) == k), 4)
                      for k, v in per_stratum.items()}
            bad_share = {k: v for k, v in shares.items() if abs(v - 1 / 3) > 0.05}
            if bad_share:
                raise InstrumentAbort(f"A-null-indep RED ({env_path.name}): {bad_share}")
            env_infos.append({
                "sidecar": env_path.name,
                "sidecar_sha256": sha256_file(env_path),
                "n_clusters": len(env_ids), "per_stratum": per_stratum,
                "per_stratum_key": per_key, "shares": shares,
            })
        if mixed:
            raise InstrumentAbort(f"0-mixed-null RED: {sorted(mixed)[:5]}")
        null_info = {"envelopes": env_infos, "mixed_null": 0,
                     "pin": "null -> epoch 0 + flag 1 + delta 0 (A-null-pin strict)"}
        log("null_coverage_asserts", **null_info, ok=True)
    log("frozen_inputs",
        train_jsonl_sha256=sha256_file(TRAIN_JSONL),
        manifest_sha256=sha256_file(MANIFEST),
        negatives_sha256=sha256_file(NEG_PATH) if NEG_PATH.exists() else None,
        negatives_v2_sha256=(sha256_file(NEG2_PATH) if NEG2_PATH.exists() else None),
        corpus_fp_v1_1=(neg_manifest.get("corpus_fp_v1_1")
                        if NEG_PATH.exists() else None),
        corpus_fp_v2=(neg2_manifest.get("negatives_v2_sha256")
                      if neg2_manifest is not None else None),
        n_clusters=len(clusters),
        counts={f"{s}/k{k}": n for (s, k), n in sorted(counts.items())},
        theme_near_counts=neg_counts or None,
        theme_near_v2_counts=neg2_counts or None,
        mismatch=count_mism, missing=missing,
        sealed_inputs=["NONE — MA-holdout-choice, holdout v4.5, g4-spec-v1, "
                       "spot-sheet are NOT in the train input list"])
    if count_mism or missing or extra:
        raise InstrumentAbort(f"frozen_inputs count mismatch: {count_mism} "
                              f"missing={missing} extra={extra}")
    return clusters


# --- corpus loading ----------------------------------------------------------

def _cluster_from_row(row: dict) -> "lib.Cluster":
    members = [
        lib.Member(
            mid=m["id"], title=m["title"], body=m["body"],
            tags=tuple(m["tags"]), record_type=m["record_type"],
            language=m["language"], created_at=m["created_at"],
        )
        for m in row["members"]
    ]
    return lib.Cluster(
        cid=row["cluster_id"], stratum=row["stratum"],
        substratum=row["substratum"], members=members,
        label=row["label"], declaration=row["declaration"],
        scenario=row["scenario"], orientation=row["orientation"],
    )


def load_train() -> list["lib.Cluster"]:
    """Reconstruct frozen lib.Cluster objects from the emitted train.jsonl
    PLUS the v1.1 theme-near extension (train-negatives.jsonl, ADDENDUM-3)
    PLUS the Choice-v2 extension (train-negatives-v2.jsonl, ADDENDUM-6 §2
    pos 5) when present — extensions, never re-emits: the sealed pins of
    the earlier files are enforced by the byte tripwires.

    Loading through the lib types keeps every downstream predicate
    (classify / silver / baselines / correct_set_ok) on the pinned code
    path — zero reimplementation.
    """
    clusters: list[lib.Cluster] = []
    with open(TRAIN_JSONL, encoding="utf-8") as fh:
        for line in fh:
            clusters.append(_cluster_from_row(json.loads(line)))
    if NEG_PATH.exists():
        with open(NEG_PATH, encoding="utf-8") as fh:
            for line in fh:
                clusters.append(_cluster_from_row(json.loads(line)))
    if NEG2_PATH.exists():
        with open(NEG2_PATH, encoding="utf-8") as fh:
            for line in fh:
                clusters.append(_cluster_from_row(json.loads(line)))
    null_ids: set[str] = set()
    for env_path in (NULL_PATH, NULL2_PATH):
        if env_path.exists():
            null_ids |= set(json.loads(env_path.read_text(encoding="utf-8"))
                            .get("cluster_ids", []))
    if null_ids:
        for c in clusters:
            if c.cid in null_ids:
                for m in c.members:
                    m.created_at = None  # both-null pair semantics (ADDENDUM-4)
    return clusters


def _stratum_of(c: lib.Cluster) -> str:
    return c.substratum if c.stratum == "theme-near" else c.stratum


def stratum_key(c: lib.Cluster) -> str:
    if c.stratum == "CH-3":
        return "CH-3/chain" if len(c.members) == 3 else "CH-3/" + c.substratum
    if c.stratum == "theme-near":
        return "theme-near/" + c.substratum
    return c.stratum


# --- features (member-level, permutation-invariant) --------------------------

#: Feature order is part of the artifact contract (append-only, never
#: reorder — same rule as cortex.features.pair FEATURE_NAMES).
MEMBER_FEATURE_NAMES = (
    # BLOCK P: frozen pair contract, mean-over-others (13)
    *FEATURE_NAMES,
    # BLOCK D: directional text on the frozen normalizer (3)
    "d_other_in_mine", "d_mine_in_other", "d_rel_len",
    # BLOCK E: edit-region block on the pinned norm2 machinery (6)
    "e_n_clusters", "e_chars_mine", "e_chars_other",
    "e_rel_chars", "e_digit_mine", "e_digit_other",
    # BLOCK V: envelope per A-null-pin (3)
    "v_created_epoch", "v_created_null", "v_created_delta_mean",
)
N_PAIR_TERMS = len(MEMBER_FEATURE_NAMES) - 3  # 22 non-envelope per-other terms
IDX_EPOCH = N_PAIR_TERMS      # v_created_epoch (own value, not a mean)
IDX_NULL = N_PAIR_TERMS + 1   # v_created_null
IDX_DELTA = N_PAIR_TERMS + 2  # v_created_delta_mean (mean over others)

#: The declared constant passed as the frozen contract's measured-similarity
#: input (corpus v1 has no vector leg — see module docstring).
SIMILARITY_CONSTANT = 1.0

# --- Choice v2 feature block (ADDENDUM-6 §2 pos 2/3) -------------------------
#
# Appended AFTER the 25 v1 columns (append-only rule — never reorder):
#   e_cos        mean-over-others cos of the pinned nano on the norm1 text
#                (member aggregate of the SAME convention the zone uses);
#   zeps_* (7)   the framing §2.4 eps-flags (eps = 0.01, exact zero for the
#                discrete flags, digits flag on the empty intersection),
#                mean-over-others per member;
#   zeps_sum     the member aggregate of the pair zeps (sum of the 7 means).
# dir is FORBIDDEN as an inference feature (ADDENDUM-6 pos 4) — construction
# parameter of the digit-lookalike strata only; it appears in no column.

V2_FEATURE_NAMES = (
    *MEMBER_FEATURE_NAMES,
    "e_cos",
    "zeps_tok_body", "zeps_char5", "zeps_tok_title", "zeps_tags",
    "zeps_digits", "zeps_lang", "zeps_type", "zeps_sum",
)
N_V2_TERMS = len(V2_FEATURE_NAMES) - len(MEMBER_FEATURE_NAMES)  # 9
IDX_E_COS = len(MEMBER_FEATURE_NAMES)            # 25
IDX_ZEPS_FLAGS = IDX_E_COS + 1                   # 26..32
IDX_ZEPS_SUM = IDX_ZEPS_FLAGS + 7                # 33

_NEG2 = None  # merge_arbiter_negatives_v2 module (lazy — it imports this module back)


def neg2():
    """Lazy import of the DE generation module — the SINGLE source of the
    ratified v2 measurement conventions (n1 / char5_pair / zeps_pair /
    zone rule; nano pin asserted at its Embedder init). It imports this
    module back, so the import must stay lazy."""
    global _NEG2
    if _NEG2 is None:
        import merge_arbiter_negatives_v2 as _m
        _NEG2 = _m
    return _NEG2


class VecLeg:
    """Pinned nano embedder + persistent disk cache (ADDENDUM-6 §2 pos 6).

    ONE nano pin + ONE norm1 convention for BOTH the zone pair cos and the
    e_cos member feature (two different cos in the contour are forbidden).
    Embedding text: n1(title) + "\\n" + n1(body) + "\\n" + n1(" ".join(tags)),
    truncation 4096; the cache key is sha256 of the TRUNCATED text (the DE
    generation convention). The cache file lives under data/ (derived data,
    gitignored) and is append-only JSONL — train / floors / prelaunch phases
    reuse it instead of re-embedding.
    """

    def __init__(self) -> None:
        prov = neg2().emb()  # pin-asserted provider (InstrumentAbort on drift)
        self._provider = prov._provider
        self._cache: dict[str, list[float]] = {}
        if EMBED_CACHE_PATH.exists():
            with open(EMBED_CACHE_PATH, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn tail — re-embed that text
                    self._cache[row["sha256"]] = row["vec"]
        EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(EMBED_CACHE_PATH, "a", encoding="utf-8")
        log("embedder_leg_init", pin=lib.EMBEDDER_PIN,
            cached_texts=len(self._cache), cache=str(EMBED_CACHE_PATH),
            convention=neg2().EMBED_TEXT_DOC
            if hasattr(neg2(), "EMBED_TEXT_DOC") else
            "n1(title)\\nn1(body)\\nn1(space-joined tags), [:4096]")

    @staticmethod
    def member_text(m: "lib.Member") -> str:
        n1 = neg2().n1
        return f"{n1(m.title)}\n{n1(m.body)}\n{n1(' '.join(m.tags))}"[:4096]

    def embed_member(self, m: "lib.Member") -> list[float]:
        text = self.member_text(m)
        key = sha256_text(text)
        vec = self._cache.get(key)
        if vec is None:
            vec = self._provider.embed(text)
            self._cache[key] = vec
            self._fh.write(json.dumps({"sha256": key, "vec": vec},
                                      ensure_ascii=False) + "\n")
            self._fh.flush()
        return vec

    def cos(self, a: "lib.Member", b: "lib.Member") -> float:
        va, vb = self.embed_member(a), self.embed_member(b)
        return sum(x * y for x, y in zip(va, vb))


def _side_dict(m: "lib.Member") -> dict:
    return {"title": m.title, "body": m.body, "tags": list(m.tags),
            "language": m.language, "record_type": m.record_type}


def zeps_flags(sd_a: dict, sd_b: dict) -> tuple[int, ...]:
    """The 7 eps-flags of framing §2.4 (DE convention, neg2 constants).

    Equivalence guard: the sum MUST equal the generator's zeps_pair on the
    same sides — the zone constants and the A-zone asserts were computed
    with THAT implementation, so any drift aborts fail-closed.
    """
    n2 = neg2()
    eps = n2.EPS
    flags = (
        int(n2.jac(n2.tokset(sd_a["body"]), n2.tokset(sd_b["body"])) <= eps),
        int(n2.char5_pair(sd_a, sd_b) <= eps),
        int(len(n2.tokset(sd_a["title"]) & n2.tokset(sd_b["title"])) <= eps),
        int(n2.jac(frozenset(sd_a["tags"]), frozenset(sd_b["tags"])) <= eps),
        int(len(n2.digset(sd_a["body"]) & n2.digset(sd_b["body"])) == 0),
        int(sd_a["language"] != sd_b["language"]),
        int(sd_a["record_type"] != sd_b["record_type"]),
    )
    oracle = n2.zeps_pair(sd_a, sd_b)
    if sum(flags) != oracle:
        raise InstrumentAbort(
            f"zeps drift: local {sum(flags)} != generator oracle {oracle} "
            f"(flags={flags}) — the ratified convention diverged")
    return flags


def pair_zone_metrics(a: "lib.Member", b: "lib.Member", vec: VecLeg) -> dict:
    """The zone's pair measurements on the pinned convention: PAIR cos
    (nano/norm1) + char5 (frozen extractor) + zeps (7 flags, cross-checked)."""
    sd_a, sd_b = _side_dict(a), _side_dict(b)
    return {"cos": vec.cos(a, b),
            "char5": neg2().char5_pair(sd_a, sd_b),
            "zeps": sum(zeps_flags(sd_a, sd_b))}


def zone_hit_pair(m: dict, zone: dict = ZONE_V1) -> str | None:
    """The ratified one-sided rule; returns the firing leg name or None."""
    if m["cos"] <= zone["tau_cos"] and m["char5"] <= zone["tau_lex"]:
        return "cos"
    if m["zeps"] >= zone["zeps_min"] and m["char5"] <= zone["tau_lex2"]:
        return "zeps"
    return None


def zone_hit_cluster(members: list["lib.Member"], vec: VecLeg,
                     zone: dict = ZONE_V1) -> dict | None:
    """Any unordered pair of the cluster inside the zone fires the leg."""
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            m = pair_zone_metrics(members[i], members[j], vec)
            leg = zone_hit_pair(m, zone)
            if leg:
                return {"pair": [members[i].mid, members[j].mid],
                        "zone_leg": leg,
                        "cos": round(m["cos"], 6),
                        "char5": round(m["char5"], 6),
                        "zeps": int(m["zeps"])}
    return None


def _epoch(created_at: str | None) -> tuple[float, float]:
    """A-null-pin: (epoch_seconds, null_flag); strict ISO-8601 UTC."""
    if created_at is None:
        return 0.0, 1.0
    if not _ISO_RE.match(created_at):
        raise InstrumentAbort(f"created_at is not ISO-8601 UTC (Z): {created_at!r}")
    t = dt.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc).timestamp()
    return float(t), 0.0


def _digit_frac(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def _pair_member_terms(mi: lib.Member, mj: lib.Member) -> list[float]:
    """Per-pair terms of member i against member j (22 values)."""
    vec = features(
        RecordLike(title=mi.title, body=mi.body, tags=mi.tags,
                   language=mi.language, record_type=mi.record_type),
        RecordLike(title=mj.title, body=mj.body, tags=mj.tags,
                   language=mj.language, record_type=mj.record_type),
        similarity=SIMILARITY_CONSTANT,
    )
    terms = list(vec.values)

    bi, bj = lib.norm1_body(mi.body), lib.norm1_body(mj.body)
    terms.append(1.0 if bj in bi else 0.0)                       # d_other_in_mine
    terms.append(1.0 if bi in bj else 0.0)                       # d_mine_in_other
    li, lj = len(bi), len(bj)
    terms.append((li - lj) / (li + lj) if (li + lj) else 0.0)    # d_rel_len

    hulls = lib.norm2_edit_clusters(mi.body, mj.body)
    terms.append(float(len(hulls)))                              # e_n_clusters
    mine = other = 0
    mine_txt = other_txt = ""
    for (c1, c2, d1, d2) in hulls:
        mine += (c2 - c1)
        other += (d2 - d1)
        mine_txt += bi[c1:c2]
        other_txt += bj[d1:d2]
    terms.append(float(mine))                                    # e_chars_mine
    terms.append(float(other))                                   # e_chars_other
    terms.append((mine - other) / (mine + other + 1e-9))         # e_rel_chars
    terms.append(_digit_frac(mine_txt))                          # e_digit_mine
    terms.append(_digit_frac(other_txt))                         # e_digit_other
    return terms


def cluster_features(members: list[lib.Member],
                     vec: "VecLeg | None" = None) -> np.ndarray:
    """(k, 25) member feature matrix (v1, when ``vec`` is None — the
    bit-identical v1 path) or (k, 34) Choice-v2 matrix (e_cos + zeps block
    appended) — permutation-invariant by construction.

    Every row is a function of (the member SET, that member) only. Per-other
    terms are accumulated in SORTED-mid order of the others, so a member's
    row is BIT-identical under any presentation permutation (no float
    reassociation drift). No feature reads a serialization index (the
    'first' baseline carries no signal — A-order-indep at feature level).
    """
    k = len(members)
    width = len(MEMBER_FEATURE_NAMES) if vec is None else len(V2_FEATURE_NAMES)
    order = sorted(range(k), key=lambda p: members[p].mid)
    epochs = [0.0] * k
    nulls = [0.0] * k
    for p in range(k):
        epochs[p], nulls[p] = _epoch(members[p].created_at)
    sds = [_side_dict(m) for m in members] if vec is not None else None
    rows = np.zeros((k, width), dtype=np.float64)
    for i in range(k):
        acc = np.zeros(width, dtype=np.float64)
        for j in order:
            if j == i:
                continue
            acc[:N_PAIR_TERMS] += _pair_member_terms(members[i], members[j])
            acc[IDX_DELTA] += epochs[i] - epochs[j]
            if vec is not None:
                acc[IDX_E_COS] += vec.cos(members[i], members[j])
                flags = zeps_flags(sds[i], sds[j])
                for f, v in enumerate(flags):
                    acc[IDX_ZEPS_FLAGS + f] += float(v)
                acc[IDX_ZEPS_SUM] += float(sum(flags))
        n_others = k - 1
        acc[:N_PAIR_TERMS] /= n_others
        acc[IDX_DELTA] /= n_others
        acc[IDX_EPOCH] = epochs[i]
        acc[IDX_NULL] = nulls[i]
        if vec is not None:
            acc[IDX_E_COS] /= n_others
            acc[IDX_ZEPS_FLAGS:IDX_ZEPS_FLAGS + 7] /= n_others
            acc[IDX_ZEPS_SUM] /= n_others
        rows[i] = acc
    return rows


def assert_permutation_invariance(clusters: list[lib.Cluster], n_check: int = 40,
                                  vec: "VecLeg | None" = None) -> None:
    """Machine check: reversing the member order changes no member's row."""
    rng = random.Random(SEED)
    sample = rng.sample(clusters, min(n_check, len(clusters)))
    for c in sample:
        x = cluster_features(c.members, vec=vec)
        rev = list(reversed(c.members))
        xr = cluster_features(rev, vec=vec)
        idx = {m.mid: p for p, m in enumerate(c.members)}
        idxr = {m.mid: p for p, m in enumerate(rev)}
        for mid, p in idx.items():
            if not np.array_equal(x[p], xr[idxr[mid]]):
                raise InstrumentAbort(
                    f"permutation invariance RED on {c.cid}/{mid}: features "
                    "depend on serialization order")
    log("feature_permutation_invariance", checked=len(sample), ok=True)


# --- member-level training table ---------------------------------------------

def member_rows(clusters: list[lib.Cluster],
                vec: "VecLeg | None" = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build the member-level table (see module docstring for the labeling).

    CH-3 k2: positive = the keep member (superset), negative = the other.
    CH-3 chain: positive = the unique ⊆-maximum, negatives = the rest.
    CH-2/CH-4: ALL rows negative — the abstain leg. CH-0/CH-5/real: not
    in train (code legs / gate-only strata). Choice v2: EVERY member of a
    theme-near cluster (v1.1 strata AND the four v2 strata) is negative —
    the label type is abstain by construction, so ``pos`` stays empty.
    """
    xs, ys, meta = [], [], []
    for c in clusters:
        if c.stratum in ("CH-0", "CH-5"):
            continue
        x = cluster_features(c.members, vec=vec)
        pos = {c.label["member"]} if c.stratum == "CH-3" else set()
        for p, m in enumerate(c.members):
            xs.append(x[p])
            ys.append(1.0 if m.mid in pos else 0.0)
            meta.append({"cluster_id": c.cid, "mid": m.mid,
                         "stratum": stratum_key(c)})
    return np.array(xs), np.array(ys), meta


def fit_booster(x: np.ndarray, y: np.ndarray) -> lgb.Booster:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(x, y)
    return model.booster_


def predict_scores(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    """Binary objective -> P(positive) = eligibility score e_i."""
    return booster.predict(x)


# --- decision rule (draft §2.2) ----------------------------------------------

def decide(scores: dict[str, float], created: dict[str, float],
           tau_hi: float, tau_m: float) -> dict:
    """keep = argmax e_i iff e_(1) >= tau_hi and e_(1) - e_(2) >= tau_m.

    Order-invariant tie-break among exact score ties: keep-oldest (min
    epoch), then min member id — never the serialization index.
    e_(2) = runner-up score among the NON-argmax members.
    """
    order = sorted(scores, key=lambda m: (-scores[m], created[m], m))
    top = order[0]
    second = order[1] if len(order) > 1 else order[0]
    e1, e2 = scores[top], scores[second]
    if e1 >= tau_hi and (e1 - e2) >= tau_m:
        return {"verdict": "keep", "member": top, "e1": round(e1, 6),
                "e2": round(e2, 6)}
    return {"verdict": "abstain", "e1": round(e1, 6), "e2": round(e2, 6)}


# --- CV over clusters (train-only threshold fit, draft §2.2) ------------------

def make_folds(clusters: list[lib.Cluster]) -> list[set[str]]:
    """K-fold BY CLUSTER, stratified by stratum, pairwise-disjoint folds.

    Round-robin over shuffled per-stratum lists; seed fixed (SEED) and
    committed — fold assignment is part of the frozen config.
    """
    rng = random.Random(SEED)
    by_stratum: dict[str, list[str]] = {}
    for c in clusters:
        by_stratum.setdefault(stratum_key(c), []).append(c.cid)
    folds: list[set[str]] = [set() for _ in range(CV_K)]
    for key in sorted(by_stratum):
        ids = sorted(by_stratum[key])
        rng.shuffle(ids)
        for p, cid in enumerate(ids):
            folds[p % CV_K].add(cid)
    for a in range(CV_K):
        for b in range(a + 1, CV_K):
            if folds[a] & folds[b]:
                raise InstrumentAbort(f"fold {a} and fold {b} overlap — "
                                      "folds must be pairwise disjoint")
    return folds


def cv_thresholds(clusters: list[lib.Cluster],
                  vec: "VecLeg | None" = None) -> dict:
    """Fit (tau_hi, tau_m) by out-of-fold cluster decisions on train only.

    Selection metric (brief/§2.2): correct-set acc + abstain rate on CH-3,
    jointly: score = acc − abstain; ties: acc desc, tau_hi desc, tau_m
    desc (more conservative wins ties). NO sealed surface participates in
    any choice here (N2). The zone leg does NOT participate in the CV
    decisions either: the readers measure the MODEL's own behaviour —
    the model defect must be fixed by the retrain (ADDENDUM-6 §1 leg 3),
    the zone must not be what makes the riders green.
    """
    # folds over the TRAIN-participating clusters only (CH-0/CH-5 are code
    # legs — they carry no member rows and no threshold decision)
    train_clusters = [c for c in clusters if c.stratum not in ("CH-0", "CH-5")]
    xs, ys, meta = member_rows(train_clusters, vec=vec)
    folds = make_folds(train_clusters)
    by_cid = {c.cid: c for c in train_clusters}
    row_idx: dict[str, list[int]] = {}
    for p, m in enumerate(meta):
        row_idx.setdefault(m["cluster_id"], []).append(p)

    oof: dict[str, dict[str, float]] = {}
    for f, fold in enumerate(folds):
        train_ids = set(by_cid) - fold
        tr_rows = [p for cid in sorted(train_ids) for p in row_idx[cid]]
        m_f = fit_booster(xs[tr_rows], ys[tr_rows])
        for cid in sorted(fold):
            rows = row_idx[cid]
            sc = predict_scores(m_f, xs[rows])
            oof[cid] = {meta[p]["mid"]: float(s) for p, s in zip(rows, sc)}
        log("cv_fold", fold=f, n_train_clusters=len(train_ids),
            n_eval_clusters=len(fold))

    created_by_cid = {
        cid: {m.mid: _epoch(m.created_at)[0] for m in c.members}
        for cid, c in by_cid.items()
    }
    ch3_ids = sorted(cid for cid, c in by_cid.items() if c.stratum == "CH-3")

    grid = []
    for tau_hi in TAU_HI_GRID:
        for tau_m in TAU_M_GRID:
            ok_n = abst_n = 0
            for cid in ch3_ids:
                c = by_cid[cid]
                d = decide(oof[cid], created_by_cid[cid], tau_hi, tau_m)
                if d["verdict"] == "abstain":
                    abst_n += 1
                elif d["member"] == c.label["member"]:
                    ok_n += 1
            acc = ok_n / len(ch3_ids)
            abst = abst_n / len(ch3_ids)
            grid.append({"tau_hi": tau_hi, "tau_m": tau_m,
                         "acc_ch3": round(acc, 4),
                         "abstain_rate_ch3": round(abst, 4),
                         "score": round(acc - abst, 4)})
    grid.sort(key=lambda g: (-g["score"], -g["acc_ch3"], -g["tau_hi"], -g["tau_m"]))
    best = grid[0]
    log("cv_threshold_selected", **best,
        selection="max(acc_ch3 - abstain_rate_ch3); ties: acc, tau_hi, tau_m (desc)",
        grid_points=len(grid), ch3_n=len(ch3_ids),
        note="train-only; the ratified q = max(0.20; p90+2*floor_q) gate "
             "constant is NOT used for any choice here")
    tau_hi, tau_m = best["tau_hi"], best["tau_m"]

    # CV-neg-abstain (ADDENDUM-3 §2; ADDENDUM-6 §1 leg 3 for v2): OOF abstain
    # >= 0.9 per substratum on ALL theme-near substrata — the three v1.1
    # strata AND (when the v2 extension is loaded) each of the four new
    # strata; a missing substratum is fail-closed, not silently skipped.
    neg_by_sub: dict[str, list[int]] = {}
    for cid, c in by_cid.items():
        if c.stratum == "theme-near":
            neg_by_sub.setdefault(c.substratum, []).append(cid)
    expected_subs = set(EXPECTED_NEG_COUNTS)
    if NEG2_PATH.exists():
        expected_subs |= set(EXPECTED_NEG_V2_COUNTS)
    absent_subs = sorted(expected_subs - set(neg_by_sub))
    if absent_subs:
        raise InstrumentAbort(f"CV-neg-abstain: substrata absent from the "
                              f"effective corpus: {absent_subs}")
    neg_rates = {}
    for sub, ids in sorted(neg_by_sub.items()):
        abst = sum(1 for cid in ids
                   if decide(oof[cid], created_by_cid[cid], tau_hi, tau_m)
                   ["verdict"] == "abstain") / len(ids)
        neg_rates[sub] = round(abst, 4)
    red = {s: r for s, r in neg_rates.items() if r < NEG_ABSTAIN_MIN}
    log("cv_neg_abstain", per_substratum=neg_rates, floor=NEG_ABSTAIN_MIN, red=red)
    if red:
        raise InstrumentAbort(f"CV-neg-abstain RED (< {NEG_ABSTAIN_MIN}): {red}")

    # CV-noregress (ADDENDUM-3 §2): per-fold OOF CH-3 acc >= 1.0 - tol
    # (reference: the v1 artifact's OOF CH-3 acc — aggregate 1.0 over
    # cluster quanta ⇒ every fold 1.0). Tolerance pinned NOREGRESS_TOL.
    per_fold_acc = {}
    for f, fold in enumerate(folds):
        fold_ch3 = sorted(cid for cid in fold if cid in by_cid
                          and by_cid[cid].stratum == "CH-3")
        if not fold_ch3:
            continue
        ok = 0
        for cid in fold_ch3:
            d = decide(oof[cid], created_by_cid[cid], tau_hi, tau_m)
            if d["verdict"] == "keep" and d.get("member") == by_cid[cid].label["member"]:
                ok += 1
        per_fold_acc[f] = round(ok / len(fold_ch3), 4)
    noreg_red = {f: a for f, a in per_fold_acc.items() if a < 1.0 - NOREGRESS_TOL}
    log("cv_noregress", per_fold_ch3_acc=per_fold_acc, reference=1.0,
        reference_artifact="merge_arbiter_v1-1-iter3.json (sha256 19bf073e…, "
                           "aggregate OOF CH-3 acc 1.0 over cluster quanta)",
        tolerance=NOREGRESS_TOL,
        tolerance_pin="0.02 fixed BEFORE the v2 retrain (ADDENDUM-3 §2; "
                      "carried over from the v1 cycle unchanged)",
        floor=round(1.0 - NOREGRESS_TOL, 4), red=noreg_red)
    if noreg_red:
        raise InstrumentAbort(f"CV-noregress RED: fold accs {noreg_red} below "
                              f"{1.0 - NOREGRESS_TOL}")

    return {"best": best, "grid_top": grid[:10],
            "ch3_n": len(ch3_ids),
            "neg_abstain": neg_rates,
            "noregress": {"per_fold": per_fold_acc, "reference": 1.0,
                          "tol": NOREGRESS_TOL},
            "folds": [{"fold": f, "n": len(fold)} for f, fold in enumerate(folds)]}


def oof_decided_member(scores: dict[str, float], created: dict[str, float],
                       tau_hi: float, tau_m: float) -> str | None:
    """The keep member of the decision rule, or None on abstain."""
    d = decide(scores, created, tau_hi, tau_m)
    return d.get("member") if d["verdict"] == "keep" else None


# --- baselines (draft §5; compare vs manifest characterization) ---------------

NULL_EPOCH_SENTINEL = "1970-01-01T00:00:00Z"  # A-null-pin: null -> epoch 0


def _sentinel_members(c: lib.Cluster) -> list[lib.Member]:
    """Members with null created_at mapped to the A-null-pin epoch-0
    sentinel so the FROZEN baseline functions stay usable; on both-null
    clusters keep-oldest is tie-ambiguous (all epoch-0) — the formal
    argmin E(r) is degenerate there, disclosed in the artifact."""
    return [lib.Member(m.mid, m.title, m.body, m.tags, m.record_type,
                       m.language,
                       NULL_EPOCH_SENTINEL if m.created_at is None
                       else m.created_at)
            for m in c.members]


def baselines_per_stratum(clusters: list[lib.Cluster],
                          sentinel: bool = False) -> dict:
    table: dict[str, dict[str, list[int]]] = {}
    for c in clusters:
        key = stratum_key(c)
        row = table.setdefault(key, {b: [0, 0] for b in lib.BASELINES})
        members = _sentinel_members(c) if sentinel else c.members
        for bname, bfn in lib.BASELINES.items():
            row[bname][1] += 1
            row[bname][0] += int(lib.correct_set_ok(c, bfn(members)))
    return {key: {b: round(n[0] / n[1], 4) if n[1] else None for b, n in row.items()}
            for key, row in table.items()}


def check_baselines_vs_manifest(manifest: dict, mine: dict) -> None:
    want = manifest["characterization"]["baselines_per_stratum"]
    diff = {}
    for key, row in want.items():
        for b, v in row.items():
            got = mine.get(key, {}).get(b)
            if got is None or abs(got - v) > 1e-9:
                diff[f"{key}/{b}"] = {"got": got, "want": v}
    log("baseline_reproduction", mismatch=diff,
        note="same train material + same frozen baseline code path -> exact")
    if diff:
        raise InstrumentAbort(f"baseline table diverges from the build manifest: {diff}")


# --- delta_acc floor measurement (ratified MLC 29.10, see constants block) ----

def _train_scenario_keys() -> set[str]:
    """CH-3/CH-4 scenario keys of train.jsonl — the exclusion set for the
    draws (holdout-style independence: the draws share train's exclusion
    set, exactly as the build shared it with the holdout emission)."""
    keys: set[str] = set()
    with open(TRAIN_JSONL, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["stratum"] not in ("CH-3", "CH-4"):
                continue
            scenario = row["scenario"]
            if row["k"] == 3:
                scenario = scenario.rsplit("+", 1)[0]  # strip the +v2 tail
            keys.add(scenario)
    return keys


def _holdout_id_manifest() -> set[str]:
    """ID-ONLY read of the sealed holdout (ratified disjointness leg).

    Precedent ADDENDUM-1 §2 pre-check leg 3: the intersection of draw ids
    with the MA-holdout-choice id-manifest is checked machine-internally
    WITHOUT reading content — this loader extracts ONLY cluster_id and
    member ids and immediately discards every other field (no text, no
    label, no envelope value is kept or logged). The FACT of the id-manifest
    read is logged in run-log. The single-shot content read stays sealed
    behind gate_start.
    """
    ids: set[str] = set()
    with open(HOLDOUT_JSONL, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ids.add(row["cluster_id"])
            ids.update(m["id"] for m in row["members"])
    return ids


def _pooled_acc(clusters: list[lib.Cluster], scorer) -> float:
    hits = sum(int(scorer(c)) for c in clusters)
    return hits / len(clusters) if clusters else 0.0


def _parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def cmd_floor_seed(args: argparse.Namespace) -> int:
    """Ratified delta_acc floor measurement (train-side, no sealed content).

    Executes the MLC 29.10 procedure verbatim: 10 composition draws
    (seeds 43..52, frozen build recipe via mb.gen_family), the FROZEN
    merge-v1 model (tripwired against the committed doc), CH-3 aggregate
    of the gate-leg composition (22 k2 + 3 chain), max-pairwise-abs-delta
    floor semantics, delta_acc = max(2*floor; 0.10). Disjointness asserts:
    draw ids vs train ids = 0, draw ids vs the holdout ID-manifest = 0
    (id-only read), draw CH-3 scenarios vs train CH-3/CH-4 scenarios = 0.
    Updates ONLY the delta_acc_measure section of the committed doc; the
    frozen thresholds and the durable payload artifact are NOT rewritten.
    """
    global _SLICE
    _SLICE = SLICE_FLOOR
    started = dt.datetime.now(dt.timezone.utc)
    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=60,
        expected_peak_ram_gb=1.0, lane="light (draft §9)")
    pro = prologue(expected_artifact=Path(args.artifact))
    model, tau_hi, tau_m = load_frozen(Path(args.artifact))
    model_sha = pro["pins"]["artifact_sha256"]
    seeds = _parse_seeds(args.seeds)
    doc_path = Path(args.doc_out)
    doc = json.loads(doc_path.read_text(encoding="utf-8"))

    log("floor_seed_measure_start",
        seeds=seeds, n_draws=len(seeds),
        gate_leg_quota=FLOOR_LEG_QUOTA,
        recipe="mb.gen_family (frozen build recipe, same code path)",
        model_sha256=model_sha,
        statistic="acc_choice (hits/total, abstain=miss) - max pooled baseline",
        floor_semantics="max pairwise |delta_d - delta_e| over all draw pairs",
        delta_acc_formula="max(2 * floor_seed_aggregate; 0.10)",
        ratification="MLC 29.10, option (a), unanimous; chairman-consolidated",
        note="model is NOT retrained; fixed BEFORE gate_start")

    train_clusters = load_train()
    train_ids = {m.mid for c in train_clusters for m in c.members}
    train_ids |= {c.cid for c in train_clusters}
    train_scen = _train_scenario_keys()
    holdout_ids = _holdout_id_manifest()
    log("holdout_id_manifest_read", n_ids=len(holdout_ids),
        content_read=False,
        note="id-only read for the ratified disjointness leg "
             "(ADDENDUM-1 §2 pre-check precedent); no content kept")

    rows = []
    for seed in seeds:
        used = set(train_scen)  # holdout-style exclusion (fresh set per draw)
        fam = mb.gen_family(seed, _FLOOR_QUOTA_FULL, used,
                            f"ma-fs-{seed}", f"ma-fseed{seed}")
        # count assert: the draw is exactly the gate-leg composition
        got: dict[tuple[str, int], int] = {}
        for c in fam:
            got[(c.stratum, len(c.members))] = got.get((c.stratum, len(c.members)), 0) + 1
        if got != {("CH-3", 2): 22, ("CH-3", 3): 3} or len(fam) != 25:
            raise InstrumentAbort(f"draw seed={seed} composition mismatch: {got}")
        # disjointness asserts (fail-closed, id + scenario level)
        draw_ids = {m.mid for c in fam for m in c.members} | {c.cid for c in fam}
        id_train_hits = draw_ids & train_ids
        id_holdout_hits = draw_ids & holdout_ids
        draw_scen = {c.scenario for c in fam}
        scen_hits = draw_scen & train_scen
        log("floor_seed_draw_disjointness", seed=seed,
            ids_train=sorted(id_train_hits), ids_holdout=sorted(id_holdout_hits),
            scenarios_train=sorted(scen_hits))
        if id_train_hits or id_holdout_hits or scen_hits:
            raise InstrumentAbort(
                f"draw seed={seed} intersects existing material: "
                f"train_ids={sorted(id_train_hits)} holdout_ids={sorted(id_holdout_hits)} "
                f"train_scenarios={sorted(scen_hits)}")

        acc_model = _pooled_acc(fam, lambda c: (
            (d := arbitrate_production(c, model, tau_hi, tau_m))["verdict"] == "keep"
            and d.get("member") == c.label["member"]))
        acc_base = max(
            _pooled_acc(fam, lambda c, b=bfn: lib.correct_set_ok(c, b(c.members)))
            for bfn in lib.BASELINES.values())
        delta = acc_model - acc_base
        rows.append({"seed": seed, "n_clusters": len(fam),
                     "acc_model": round(acc_model, 4),
                     "acc_baseline_max": round(acc_base, 4),
                     "delta": round(delta, 4)})
        log("floor_seed_draw", **rows[-1])

    deltas = [r["delta"] for r in rows]
    floor = max(abs(a - b) for i, a in enumerate(deltas) for b in deltas[i + 1:])
    delta_acc = max(2 * floor, 0.10)
    log("floor_seed_measure_end",
        per_seed=rows, floor_seed_aggregate=round(floor, 4),
        delta_acc=round(delta_acc, 4),
        pairs_compared=len(seeds) * (len(seeds) - 1) // 2,
        model_sha256=model_sha, artifact=args.artifact,
        wall_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
        remeasure_after_single_shot="FORBIDDEN")

    # committed doc: append the delta_acc_measure section ONLY (thresholds,
    # cv, baselines and the durable artifact stay byte-untouched)
    if "delta_acc_measure" in doc:
        raise InstrumentAbort("delta_acc_measure already present in "
                              f"{doc_path} — re-measuring is forbidden (fixed "
                              "BEFORE gate_start; rewrite would be "
                              "threshold-fitting)")
    doc["delta_acc_measure"] = {
        "ratification": "MLC 29.10, option (a), unanimous, chairman-consolidated",
        "granularity": "floor_seed for delta_acc is measured at the granularity "
                       "of the gating statistic (CH-3 aggregate, gate-leg "
                       "composition 22 k2 + 3 chain); the per-stratum table in "
                       "the build manifest is a characterization artifact and "
                       "does NOT enter the formula",
        "seeds": FLOOR_SEEDS,
        "per_seed": rows,
        "floor_seed_aggregate": round(floor, 4),
        "delta_acc": round(delta_acc, 4),
        "model_sha256": model_sha,
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "remeasure_after_single_shot": "FORBIDDEN",
    }
    doc.pop("committed_doc_sha256", None)
    doc["committed_doc_sha256"] = sha256_text(
        json.dumps(doc, ensure_ascii=False, sort_keys=True))
    doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    log("committed_doc_updated", path=str(doc_path),
        committed_doc_sha256=doc["committed_doc_sha256"],
        payload_artifact_untouched=True,
        thresholds_untouched=True)
    return 0


# --- production contour §5.1 (code legs + model leg) --------------------------

def arbitrate_production(cluster: lib.Cluster, booster: lgb.Booster,
                         tau_hi: float, tau_m: float,
                         ctx: "dict | None" = None) -> dict:
    """Through pipeline: classify (NO declarations — production has none)
    -> CH-0/CH-5 tie-break -> CH-2 refusal -> [Choice v2: zone leg 3.5
    (zone-v1, refusal-only)] -> model leg for the rest.

    ``ctx`` carries the v2 legs: {"vec": VecLeg, "zone": ZONE_V1-like dict}.
    ctx=None keeps the EXACT v1 path (no embedder, no zone) — the iter-1..3
    artifacts stay reproducible. Fail-closed: the model's feature contract
    must match what this contour can build (25 features without the vec
    leg, the artifact's full width with it), and the zone cannot run
    without the vec leg (it evaluates the pair cos).
    """
    ctx = ctx or {}
    vec = ctx.get("vec")
    zone = ctx.get("zone")
    cls = lib.classify(cluster.members, declarations=None)
    if cls in ("CH-0", "CH-5"):
        keep = min(cluster.members,
                   key=lambda m: (_epoch(m.created_at)[0], m.mid))
        return {"verdict": "keep", "member": keep.mid, "leg": "tie-break",
                "class": cls}
    if cls == "CH-2":
        return {"verdict": "abstain", "leg": "refusal", "class": cls}
    if zone is not None:
        if vec is None:
            raise InstrumentAbort("zone leg configured without the vec leg — "
                                  "the zone evaluates the PAIR nano cos")
        hit = zone_hit_cluster(cluster.members, vec, zone)
        if hit:
            # ONE-SIDEDNESS (ADDENDUM-6 §1 leg 2): the leg emits ONLY abstain;
            # the keep side of cos/zeps is read nowhere in the contour.
            return {"verdict": "abstain", "leg": "zone-v1", "class": cls,
                    "zone": hit}
    x = cluster_features(cluster.members, vec=vec)
    if x.shape[1] != booster.num_feature():
        raise InstrumentAbort(
            f"feature contract mismatch: contour builds {x.shape[1]} columns, "
            f"the frozen booster expects {booster.num_feature()}")
    sc = predict_scores(booster, x)
    scores = {m.mid: float(s) for m, s in zip(cluster.members, sc)}
    created = {m.mid: _epoch(m.created_at)[0] for m in cluster.members}
    d = decide(scores, created, tau_hi, tau_m)
    d["leg"] = "model"
    d["class"] = cls
    d["scores"] = {k: round(v, 6) for k, v in scores.items()}
    return d


# --- sanity suite §6.1 through the production contour -------------------------

SANITY_EXPECT_MODEL = {
    # §6.1 expectations; probes 3/4 ride the MODEL leg with the frozen
    # model, the rest stay code legs — the suite runs the WHOLE contour (N3).
    1: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    2: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    3: lambda r: r["verdict"] == "abstain" and r["leg"] == "model",
    4: lambda r: r["verdict"] == "keep" and r.get("member") == "fx-4a"
                  and r["leg"] == "model",
    5: lambda r: r["verdict"] == "abstain" and r["leg"] == "refusal",
    6: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    7: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
}


def run_model_sanity_suite(booster: lgb.Booster, tau_hi: float, tau_m: float,
                           log_it: bool = True,
                           ctx: "dict | None" = None) -> list[dict]:
    """7 probes through the production contour with the FROZEN model.

    Every probe is additionally executed with the REVERSED member order:
    the decision must be identical (pipeline-level order-invariance).
    With the v2 ctx the probes ride the WHOLE v2 contour — a zone fire on
    a CH-3/CH-4 fixture would surface as leg zone-v1 and RED the probe
    (the expectations pin probes 3/4 to the model leg).
    Red = STOP class 'instrument' — holdout is not read (§6.1).
    """
    results = []
    all_ok = True
    for cell in lib.build_fixture_cells():
        res = arbitrate_production(cell["cluster"], booster, tau_hi, tau_m, ctx=ctx)
        rev = list(reversed(cell["cluster"].members))
        res_rev = arbitrate_production(
            lib.Cluster(cid=cell["cluster"].cid, stratum=cell["cluster"].stratum,
                        substratum=cell["cluster"].substratum, members=rev,
                        label=cell["cluster"].label,
                        declaration=cell["cluster"].declaration),
            booster, tau_hi, tau_m, ctx=ctx)
        ok = bool(SANITY_EXPECT_MODEL[cell["probe"]](res))
        inv = (res["verdict"] == res_rev["verdict"]
               and res.get("member") == res_rev.get("member"))
        all_ok = all_ok and ok and inv
        row = {"probe": cell["probe"], "name": cell["name"], "ok": ok,
               "order_invariant": inv, "result": res}
        results.append(row)
        if log_it:
            log("sanity_probe", probe=cell["probe"], name=cell["name"], ok=ok,
                order_invariant=inv, verdict=res.get("verdict"),
                member=res.get("member"), leg=res.get("leg"),
                zone=res.get("zone"),
                e1=res.get("e1"), e2=res.get("e2"), scores=res.get("scores"))
    if log_it:
        log("sanity_suite_done", ok=all_ok, probes=len(results),
            expectation="7/7 + order invariance; red => holdout NOT read")
    if not all_ok:
        raise InstrumentAbort("sanity suite RED (§6.1) — sealed surfaces stay sealed")
    return results


# --- artifact freeze ----------------------------------------------------------

#: Active output naming for the train subcommand (v1 defaults preserved;
#: the v1.1 refreeze passes explicit values).
ART_NAME = "merge_arbiter_v1.json"
DOC_OUT = COMMITTED_DOC


def artifact_paths() -> tuple[Path, Path]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    COMMITTED_DIR.mkdir(parents=True, exist_ok=True)
    return ART_DIR / ART_NAME, DOC_OUT


def freeze_artifact(booster: lgb.Booster, tau_hi: float, tau_m: float,
                    cv: dict, baselines: dict, pins: dict) -> tuple[Path, Path, str]:
    durable, committed = artifact_paths()
    neg_manifest = None
    if NEG_PATH.exists():
        neg_manifest = json.loads((CORPUS_DIR / "train-negatives.manifest.json")
                                  .read_text(encoding="utf-8"))
    art = {
        "artifact": ("vesma-cortex-merge-v1" if neg_manifest is None
                     else "vesma-cortex-merge-v1 (corpus v1.1, ADDENDUM-3)"),
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "thresholds": {"tau_hi": tau_hi, "tau_m": tau_m,
                       "rule": "keep=argmax e_i iff e1>=tau_hi and e1-e2>=tau_m; "
                               "else ABSTAIN (draft §2.2)",
                       "fit": "train-only 5-fold CV by clusters, pairwise-disjoint "
                              "folds, seed 42; NO sealed surface used for any "
                              "choice (N2)"},
        "feature_names": list(MEMBER_FEATURE_NAMES),
        "similarity_constant": SIMILARITY_CONSTANT,
        "lgb_params": dict(LGB_PARAMS),
        "booster_model_string": booster.model_to_string(),
        "cv": {"best": cv["best"], "top10": cv["grid_top"],
               "ch3_n": cv["ch3_n"], "folds": cv["folds"],
               "neg_abstain": cv.get("neg_abstain"),
               "noregress": cv.get("noregress")},
        "baselines_per_stratum_train": baselines,
        "corpus": (None if neg_manifest is None else {
            "recipe": "corpus v1.1 = v1 (600, byte-identical) + theme-near "
                      "negative block (200, ADDENDUM-3)",
            "train_jsonl_sha256": pins["train_jsonl_sha256"],
            "negatives_sha256": neg_manifest["negatives_sha256"],
            "corpus_fp_v1_1": neg_manifest["corpus_fp_v1_1"],
            "seeds": neg_manifest["seeds"],
        }),
        "pins": dict(pins),
    }
    durable.write_text(json.dumps(art, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    art_sha = sha256_file(durable)

    committed_doc = {
        "artifact": art["artifact"],
        "frozen_at": art["frozen_at"],
        "thresholds": art["thresholds"],
        "cv_ch3_oof": {"acc": cv["best"]["acc_ch3"],
                       "abstain_rate": cv["best"]["abstain_rate_ch3"],
                       "score": cv["best"]["score"],
                       "ch3_n": cv["ch3_n"]},
        "cv_neg_abstain": cv.get("neg_abstain"),
        "cv_noregress": cv.get("noregress"),
        "cv_grid_top10": cv["grid_top"],
        "baselines_per_stratum_train": baselines,
        "n_features": len(MEMBER_FEATURE_NAMES),
        "feature_names_sha256": sha256_text(json.dumps(list(MEMBER_FEATURE_NAMES))),
        "durable_artifact": {"path": str(durable), "sha256": art_sha},
        "train_jsonl_sha256": pins["train_jsonl_sha256"],
        "manifest_sha256": pins["manifest_sha256"],
        "merge_arbiter_lib_sha256": pins["merge_arbiter_lib_sha256"],
        "cortex_features_pair_sha256": pins["cortex_features_pair_sha256"],
        "feature_names_pin_sha256": pins["feature_names_sha256"],
        "embedder_pin": pins["embedder_pin"],
        "env": {"python": EXPECTED_ENV["python"],
                "lightgbm": EXPECTED_ENV["lightgbm"],
                "numpy": EXPECTED_ENV["numpy"],
                "venv_suffix": EXPECTED_ENV["venv_suffix"]},
        "seed": SEED,
        "boundary_v1": "cos_target enters as the declared constant "
                       f"{SIMILARITY_CONSTANT} (no measured vector leg in "
                       "corpus v1); the vector leg joins the Choice contract "
                       "in a later engine wave",
    }
    if neg_manifest is not None:
        committed_doc["corpus_v1_1"] = {
            "recipe": art["corpus"]["recipe"],
            "negatives_sha256": neg_manifest["negatives_sha256"],
            "corpus_fp_v1_1": neg_manifest["corpus_fp_v1_1"],
            "seeds": neg_manifest["seeds"],
            "addendum": "ADDENDUM-3 (MLC 29.10, option б), commit e953eec",
        }
        # the noharm surface is UNCHANGED (§4) — carry its frozen section
        # from the v1 doc so the gate keeps a single doc contract
        if DOC_OUT != COMMITTED_DOC and COMMITTED_DOC.exists():
            v1 = json.loads(COMMITTED_DOC.read_text(encoding="utf-8"))
            if "noharm_surface" in v1:
                committed_doc["noharm_surface"] = v1["noharm_surface"]
    committed_doc["committed_doc_sha256"] = sha256_text(
        json.dumps(committed_doc, ensure_ascii=False, sort_keys=True))
    committed.write_text(json.dumps(committed_doc, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    log("artifact_frozen", durable=str(durable), durable_sha256=art_sha,
        committed=str(committed), thresholds=art["thresholds"])
    return durable, committed, art_sha


def load_frozen(path: Path) -> tuple[lgb.Booster, float, float]:
    art = json.loads(path.read_text(encoding="utf-8"))
    t = art["thresholds"]
    return lgb.Booster(model_str=art["booster_model_string"]), \
        float(t["tau_hi"]), float(t["tau_m"])


# --- subcommand: train --------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    global _SLICE, ART_NAME, DOC_OUT
    _SLICE = SLICE_TRAIN
    ART_NAME = args.artifact_name
    DOC_OUT = Path(args.doc_out)
    if DOC_OUT.exists():
        raise InstrumentAbort(f"committed doc already exists: {DOC_OUT} — "
                              "re-freeze over an existing doc is forbidden")
    durable_path = ART_DIR / ART_NAME
    if durable_path.exists():
        raise InstrumentAbort(f"durable artifact already exists: {durable_path} "
                              "— refreeze over frozen bytes is forbidden")
    started = dt.datetime.now(dt.timezone.utc)
    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=180,
        expected_peak_ram_gb=1.5, run_tag=args.run_tag,
        lane="light (draft §9): LightGBM cycle, direct sequential run allowed")
    pro = prologue()
    clusters = frozen_inputs(pro["manifest"])

    assert_permutation_invariance(clusters)

    # train-side baselines: the manifest check runs on the ORIGINAL 600
    # (train.jsonl only — the characterization was computed exactly there;
    # no negatives, no null overrides) for exact reproduction; the recorded
    # table runs on the EFFECTIVE corpus (with the ADDENDUM-4 null
    # coverage, sentinel-wrapped)
    v1_clusters = [_cluster_from_row(json.loads(line))
                   for line in open(TRAIN_JSONL, encoding="utf-8")]
    check_baselines_vs_manifest(pro["manifest"],
                                baselines_per_stratum(v1_clusters))
    baselines = baselines_per_stratum(clusters, sentinel=True)

    cv = cv_thresholds(clusters)
    tau_hi, tau_m = cv["best"]["tau_hi"], cv["best"]["tau_m"]

    # final refit on ALL train rows + in-process determinism evidence (P7)
    xs, ys, _meta = member_rows(clusters)
    booster = fit_booster(xs, ys)
    booster2 = fit_booster(xs, ys)
    s1 = booster.model_to_string()
    s2 = booster2.model_to_string()
    log("determinism_selfcheck", boosters_byte_equal=bool(s1 == s2),
        n_rows=int(len(ys)), positives=int(ys.sum()))
    if s1 != s2:
        raise InstrumentAbort("booster not byte-deterministic across two fits")

    durable, _committed, art_sha = freeze_artifact(
        booster, tau_hi, tau_m, cv, baselines, pro["pins"])

    # sanity suite on the FROZEN artifact (proves the load path too)
    fm, f_hi, f_m = load_frozen(durable)
    run_model_sanity_suite(fm, f_hi, f_m)

    # in-sample diagnostic on CH-3 (the honest threshold number is the CV OOF)
    ch3 = [c for c in clusters if c.stratum == "CH-3"]
    ok_n = abst_n = 0
    for c in ch3:
        d = arbitrate_production(c, fm, f_hi, f_m)
        if d["verdict"] == "abstain":
            abst_n += 1
        elif d.get("member") == c.label["member"]:
            ok_n += 1
    log("train_ch3_insampling", n=len(ch3), acc=round(ok_n / len(ch3), 4),
        abstain_rate=round(abst_n / len(ch3), 4),
        note="in-sample diagnostic; the selection number is the CV OOF row")

    wall = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    log("train_done", status="GREEN", wall_s=round(wall, 1),
        tau_hi=f_hi, tau_m=f_m, artifact_sha256=art_sha, cv_best=cv["best"],
        note="thresholds FROZEN before any sealed read (N2); the single-shot "
             "gate run stays unexecuted in this phase")
    return 0


# --- Choice v2: zone asserts, contour ctx, v2 freeze, train-v2 ----------------

def assert_zone_ch3_zero_v2(clusters: list[lib.Cluster], vec: VecLeg) -> None:
    """A-zone-ch3-zero (ADDENDUM-6 §2): 0 CH-3 pairs inside the zone over
    ALL pairs of the EFFECTIVE corpus, incl. the transitive chain pairs
    (276 = 240 k2 + 12 chains x 3 on the v2 corpus). The zone must never
    catch a keep-stratum pair — precondition of the v2 contour."""
    pairs = hits = 0
    worst = {"cos": 1.0, "char5": 1.0, "zeps": 0}
    for c in clusters:
        if c.stratum != "CH-3":
            continue
        ms = c.members
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                m = pair_zone_metrics(ms[i], ms[j], vec)
                pairs += 1
                worst["cos"] = min(worst["cos"], m["cos"])
                worst["char5"] = min(worst["char5"], m["char5"])
                worst["zeps"] = max(worst["zeps"], int(m["zeps"]))
                if zone_hit_pair(m):
                    hits += 1
    log("a_zone_ch3_zero", pairs=pairs, hits=hits,
        worst={"cos": round(worst["cos"], 6),
               "char5": round(worst["char5"], 6), "zeps": worst["zeps"]},
        zone=ZONE_V1["rule"],
        note="incl. transitive chain pairs (stricter than the author-side 264)")
    if hits:
        raise InstrumentAbort(f"A-zone-ch3-zero RED: {hits} CH-3 pairs in zone")


def resolve_ctx(artifact_path: Path,
                with_zone: bool = True) -> tuple[lgb.Booster, float, float,
                                                 "dict | None"]:
    """Load a frozen artifact and build its contour ctx (fail-closed).

    A 25-feature artifact -> ctx=None (the exact v1 contour). A Choice-v2
    (34-feature) artifact REQUIRES the vec leg; the zone rides whenever the
    caller asks for the production contour (with_zone) — a zone-less v2
    contour exists ONLY as the explicit zone-off probe, never by accident.
    """
    art = json.loads(artifact_path.read_text(encoding="utf-8"))
    booster, tau_hi, tau_m = load_frozen(artifact_path)
    names = art.get("feature_names") or MEMBER_FEATURE_NAMES
    if len(names) == len(MEMBER_FEATURE_NAMES):
        if with_zone and art.get("zone_v1"):
            raise InstrumentAbort("v1-width artifact declares zone_v1 — "
                                  "inconsistent artifact")
        return booster, tau_hi, tau_m, None
    if with_zone and not art.get("zone_v1"):
        raise InstrumentAbort("Choice-v2 artifact without the zone_v1 section — "
                              "refusing an accidental zone-less contour")
    zone = art.get("zone_v1", {}).get("constants") if with_zone else None
    return booster, tau_hi, tau_m, {"vec": VecLeg(), "zone": zone}


def freeze_artifact_v2(booster: lgb.Booster, tau_hi: float, tau_m: float,
                       cv: dict, baselines: dict, pins: dict,
                       v2pins: dict) -> tuple[Path, Path, str]:
    """Freeze the Choice-v2 durable artifact + committed doc (ADDENDUM-6 §2):
    34-feature contract, the zone_v1 section (constants + provenance +
    known-uncovered pockets + residual risk), the v2 corpus pins, the
    convention pins. The noharm_surface section is carried from the v1
    committed doc (single doc contract; the surface is untouched)."""
    durable = ART_DIR / ART_NAME
    committed = DOC_OUT
    neg_manifest = json.loads((CORPUS_DIR / "train-negatives.manifest.json")
                              .read_text(encoding="utf-8"))
    neg2_manifest = json.loads((CORPUS_DIR / "train-negatives-v2.manifest.json")
                               .read_text(encoding="utf-8"))
    zone_sec = {
        "constants": dict(ZONE_V1),
        "provenance": "ADDENDUM-6 §2 pos 1 (MLC 30.10 round 6, 3/3; commit "
                      "f2b51bd): constants derived from the structural "
                      "train-support voids train-only before seal and fixed "
                      "by ratification (owner directive 30.10: constants "
                      "already fixed — not re-derived); §1 legality: "
                      "structural + one-sided (refusal-only) + indivisible "
                      "package with the train extension + zone-off probe",
        "residual_risk": ZONE_V1_RESIDUAL_RISK,
        "known_uncovered": ZONE_V1_KNOWN_UNCOVERED,
        "pairness": "evaluates the PAIR cos of the candidate (nano + norm1, "
                    "the SAME convention as e_cos); any unordered pair of "
                    "the cluster inside the zone fires",
        "build_asserts": {"A-zone-coverage": "GREEN at generation "
                          "(edge_gap 0.0018, max_gap 0.0196, 132 points; "
                          "sidecar manifest coverage_after_generation)",
                          "A-zone-ch3-zero": "0/276 at generation (DE) and "
                          "re-asserted on the effective corpus at train time"},
    }
    art = {
        "artifact": "vesma-cortex-merge-v2 (Choice v2, ADDENDUM-6)",
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "thresholds": {"tau_hi": tau_hi, "tau_m": tau_m,
                       "rule": "keep=argmax e_i iff e1>=tau_hi and e1-e2>=tau_m; "
                               "else ABSTAIN (draft §2.2); the zone-v1 leg "
                               "precedes the model leg and refuses only",
                       "fit": "train-only 5-fold CV by clusters, pairwise-disjoint "
                              "folds, seed 42; NO sealed surface used for any "
                              "choice (N2); CV readers measure the MODEL leg "
                              "only (the zone does not make the riders green)"},
        "feature_names": list(V2_FEATURE_NAMES),
        "similarity_constant": SIMILARITY_CONSTANT,
        "lgb_params": dict(LGB_PARAMS),
        "booster_model_string": booster.model_to_string(),
        "cv": {"best": cv["best"], "top10": cv["grid_top"],
               "ch3_n": cv["ch3_n"], "folds": cv["folds"],
               "neg_abstain": cv.get("neg_abstain"),
               "noregress": cv.get("noregress")},
        "zone_v1": zone_sec,
        "convention": {
            "embedder_pin": lib.EMBEDDER_PIN,
            "embed_text": "n1(title)\\nn1(body)\\nn1(space-joined tags), "
                          "[:4096]; n1 = lower + whitespace-collapse "
                          "(release/6.6.0, 458fffc; shim vesmaro->vesma "
                          "unconditional — empty namespace husk)",
            "char5": "frozen cortex.features.pair (its n-gram text is norm1; "
                     "the similarity input feeds cos_target only)",
            "zeps": "7 eps-flags (eps=0.01, framing §2.4), generator-oracle "
                    "cross-checked on every pair (fail-closed)",
            "dir": "construction-only parameter of the digit-lookalike "
                   "strata (ADDENDUM-6 pos 4); NOT an inference feature",
        },
        "baselines_per_stratum_train": baselines,
        "corpus": {
            "recipe": "Choice v2 = corpus v1.1 (600 + 200, byte-identical) "
                      "+ 4 theme-near v2 strata (178, ADDENDUM-6 §2 pos 5)",
            "train_jsonl_sha256": pins["train_jsonl_sha256"],
            "negatives_sha256": neg_manifest["negatives_sha256"],
            "negatives_v2_sha256": NEG2_EXPECTED_SHA256,
            "null_envelope_v2_sha256": NULL2_EXPECTED_SHA256,
            "corpus_fp_v2": v2pins["corpus_fp_v2"],
            "seeds": neg2_manifest["seeds"],
            "negative_mass": neg2_manifest["negative_mass"],
        },
        "pins": {**pins, "neg2_sha256": v2pins["neg2_sha256"],
                 "null2_sha256": v2pins["null2_sha256"],
                 "corpus_fp_v2": v2pins["corpus_fp_v2"]},
    }
    art_sha = ""
    durable.write_text(json.dumps(art, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    art_sha = sha256_file(durable)

    committed_doc = {
        "artifact": art["artifact"],
        "frozen_at": art["frozen_at"],
        "thresholds": art["thresholds"],
        "zone_v1": zone_sec,
        "convention": art["convention"],
        "ratification": "ADDENDUM-6 (MLC 30.10 round 6, commit f2b51bd) + "
                        "ML-STD-0004 (declared floor); owner sanction 30.10 "
                        "«фич-работа v2»; one cycle §3 ADDENDUM-3, no cuts",
        "cv_ch3_oof": {"acc": cv["best"]["acc_ch3"],
                       "abstain_rate": cv["best"]["abstain_rate_ch3"],
                       "score": cv["best"]["score"],
                       "ch3_n": cv["ch3_n"]},
        "cv_neg_abstain": cv.get("neg_abstain"),
        "cv_noregress": cv.get("noregress"),
        "cv_grid_top10": cv["grid_top"],
        "baselines_per_stratum_train": baselines,
        "n_features": len(V2_FEATURE_NAMES),
        "feature_names": list(V2_FEATURE_NAMES),
        "feature_names_sha256": sha256_text(json.dumps(list(V2_FEATURE_NAMES))),
        "durable_artifact": {"path": str(durable), "sha256": art_sha},
        "train_jsonl_sha256": pins["train_jsonl_sha256"],
        "manifest_sha256": pins["manifest_sha256"],
        "merge_arbiter_lib_sha256": pins["merge_arbiter_lib_sha256"],
        "cortex_features_pair_sha256": pins["cortex_features_pair_sha256"],
        "feature_names_pin_sha256": pins["feature_names_sha256"],
        "embedder_pin": pins["embedder_pin"],
        "negatives_v2_sha256": NEG2_EXPECTED_SHA256,
        "null_envelope_v2_sha256": NULL2_EXPECTED_SHA256,
        "corpus_fp_v2": v2pins["corpus_fp_v2"],
        "env": {"python": EXPECTED_ENV["python"],
                "lightgbm": EXPECTED_ENV["lightgbm"],
                "numpy": EXPECTED_ENV["numpy"],
                "venv_suffix": EXPECTED_ENV["venv_suffix"]},
        "seed": SEED,
        "boundary_v2": "e_cos is MEASURED (pinned nano on norm1; the v1 "
                       "declared-constant boundary is retired for v2); the "
                       "zone-v1 leg precedes the model leg and refuses only",
    }
    if COMMITTED_DOC.exists():
        v1_doc = json.loads(COMMITTED_DOC.read_text(encoding="utf-8"))
        if "noharm_surface" in v1_doc:
            committed_doc["noharm_surface"] = v1_doc["noharm_surface"]
    committed_doc["committed_doc_sha256"] = sha256_text(
        json.dumps(committed_doc, ensure_ascii=False, sort_keys=True))
    committed.write_text(json.dumps(committed_doc, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    log("artifact_frozen_v2", durable=str(durable), durable_sha256=art_sha,
        committed=str(committed), thresholds=art["thresholds"],
        n_features=len(V2_FEATURE_NAMES), zone=ZONE_V1["rule"])
    return durable, committed, art_sha


def cmd_train_v2(args: argparse.Namespace) -> int:
    """Choice-v2 retrain (ADDENDUM-6 §6 step 2, ML leg): v2 tripwires ->
    effective corpus (600 + 200 + 178) -> A-zone-ch3-zero -> CV (model-leg
    riders) -> refit -> freeze v2 artifact -> sanity 7/7 through the v2
    contour. Train-only: no sealed surface is touched."""
    global _SLICE, ART_NAME, DOC_OUT
    _SLICE = SLICE_TRAIN_V2
    ART_NAME = args.artifact_name
    DOC_OUT = Path(args.doc_out)
    if DOC_OUT.exists():
        raise InstrumentAbort(f"committed doc already exists: {DOC_OUT} — "
                              "re-freeze over an existing doc is forbidden")
    durable_path = ART_DIR / ART_NAME
    if durable_path.exists():
        raise InstrumentAbort(f"durable artifact already exists: {durable_path} "
                              "— refreeze over frozen bytes is forbidden")
    started = dt.datetime.now(dt.timezone.utc)
    log("cost_declaration", expected_prolog_s=3, expected_cycle_s=1500,
        expected_peak_ram_gb=2.5, run_tag=args.run_tag,
        lane="v2 light+: LightGBM cycle + pinned nano embedder (disk-cached); "
             "sequential, single lane")
    pro = prologue()
    v2pins = v2_tripwire()
    clusters = frozen_inputs(pro["manifest"])

    vec = VecLeg()
    assert_zone_ch3_zero_v2(clusters, vec)
    assert_permutation_invariance(clusters, vec=vec)

    # train-side baselines: the manifest check runs on the ORIGINAL 600
    # (exact reproduction of the build characterization, unchanged); the
    # recorded table runs on the EFFECTIVE v2 corpus (sentinel-wrapped)
    v1_clusters = [_cluster_from_row(json.loads(line))
                   for line in open(TRAIN_JSONL, encoding="utf-8")]
    check_baselines_vs_manifest(pro["manifest"],
                                baselines_per_stratum(v1_clusters))
    baselines = baselines_per_stratum(clusters, sentinel=True)

    cv = cv_thresholds(clusters, vec=vec)
    tau_hi, tau_m = cv["best"]["tau_hi"], cv["best"]["tau_m"]

    # final refit on ALL train rows + in-process determinism evidence (P7)
    xs, ys, _meta = member_rows(clusters, vec=vec)
    booster = fit_booster(xs, ys)
    booster2 = fit_booster(xs, ys)
    s1 = booster.model_to_string()
    s2 = booster2.model_to_string()
    log("determinism_selfcheck", boosters_byte_equal=bool(s1 == s2),
        n_rows=int(len(ys)), positives=int(ys.sum()))
    if s1 != s2:
        raise InstrumentAbort("booster not byte-deterministic across two fits")

    durable, _committed, art_sha = freeze_artifact_v2(
        booster, tau_hi, tau_m, cv, baselines, pro["pins"], v2pins)

    # sanity suite on the FROZEN artifact through the FULL v2 contour
    fbooster, f_hi, f_m, fctx = resolve_ctx(durable, with_zone=True)
    run_model_sanity_suite(fbooster, f_hi, f_m, ctx=fctx)

    # in-sample diagnostic on CH-3 (the honest threshold number is the CV OOF)
    ch3 = [c for c in clusters if c.stratum == "CH-3"]
    ok_n = abst_n = zone_n = 0
    for c in ch3:
        d = arbitrate_production(c, fbooster, f_hi, f_m, ctx=fctx)
        if d["leg"] == "zone-v1":
            zone_n += 1
        if d["verdict"] == "abstain":
            abst_n += 1
        elif d.get("member") == c.label["member"]:
            ok_n += 1
    log("train_ch3_insampling", n=len(ch3), acc=round(ok_n / len(ch3), 4),
        abstain_rate=round(abst_n / len(ch3), 4), zone_fired=zone_n,
        note="in-sample diagnostic; the selection number is the CV OOF row; "
             "zone_fired must be 0 (A-zone-ch3-zero)")
    if zone_n:
        raise InstrumentAbort("zone fired on train CH-3 in-sample — "
                              "A-zone-ch3-zero drift")

    wall = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    log("train_v2_done", status="GREEN", wall_s=round(wall, 1),
        tau_hi=f_hi, tau_m=f_m, artifact_sha256=art_sha, cv_best=cv["best"],
        n_features=len(V2_FEATURE_NAMES),
        note="thresholds FROZEN before any sealed read (N2); the single-shot "
             "gate run stays unexecuted in this phase")
    return 0




def _gate_leg_draw(seed: int) -> list[lib.Cluster]:
    """CH-3 gate-leg composition (22 k2 + 3 chain), frozen build recipe."""
    return mb.gen_family(seed, dict(_FLOOR_QUOTA_FULL), set(),
                         f"ma-fl-{seed}", f"ma-flseed{seed}")


def cmd_floors_composition(args: argparse.Namespace) -> int:
    """Re-derive ALL composition-dependent floors for the NEW artifact
    (ADDENDUM-3 §3.3): floor_q (10 composition draws, CH-3 abstain rate,
    q = max(0.20; p90_char + 2*floor)), floor_inv (3 cosmetic-variation
    draws, max disagreement), floor_perm (3 permutations — structurally 0
    by the bit-identical feature construction). Writes floors_v2 into the
    doc; fixed BEFORE gate_start."""
    global _SLICE
    _SLICE = SLICE_FLOOR
    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=120,
        expected_peak_ram_gb=1.0, lane="light (draft §9)")
    pro = prologue(expected_artifact=Path(args.artifact))
    model, tau_hi, tau_m = load_frozen(Path(args.artifact))
    doc_path = Path(args.doc_out)
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if "floors_v2" in doc:
        raise InstrumentAbort("floors_v2 already present — re-derivation "
                              "forbidden (fixed BEFORE gate_start)")

    q_seeds = _parse_seeds(args.q_seeds)
    q_rates = []
    for seed in q_seeds:
        fam = _gate_leg_draw(seed)
        abst = sum(1 for c in fam
                   if arbitrate_production(c, model, tau_hi, tau_m)
                   ["verdict"] == "abstain") / len(fam)
        q_rates.append(round(abst, 4))
    floor_q = max(abs(a - b) for i, a in enumerate(q_rates)
                  for b in q_rates[i + 1:]) if len(q_rates) > 1 else 0.0
    p90_char = sorted(q_rates)[int(round(0.9 * (len(q_rates) - 1)))] if q_rates else 0.0
    q = max(0.20, p90_char + 2 * floor_q)
    log("floor_q_measure", seeds=q_seeds, rates=q_rates, p90_char=p90_char,
        floor_q=round(floor_q, 4), q=round(q, 4),
        formula="q := max(0.20; p90_char + 2*floor_q)")

    # floor_inv: cosmetic variation on train fold-eligible clusters, 3 draws
    eligible = [c for c in load_train() if c.stratum in ("CH-3", "CH-5")]
    inv_disagreements = []
    for draw_seed in _parse_seeds(args.inv_seeds):
        rng = random.Random(draw_seed)
        sample = rng.sample(eligible, min(80, len(eligible)))
        dis = 0
        for c in sample:
            base = arbitrate_production(c, model, tau_hi, tau_m)
            pert = lib.Cluster(
                cid=c.cid, stratum=c.stratum, substratum=c.substratum,
                members=[lib.Member(m.mid, m.title,
                                    lib.cosmetic_variant(m.body, rng),
                                    m.tags, m.record_type, m.language,
                                    m.created_at)
                         for m in c.members],
                label=c.label, declaration=c.declaration)
            alt = arbitrate_production(pert, model, tau_hi, tau_m)
            dis += int(base["verdict"] != alt["verdict"]
                       or base.get("member") != alt.get("member"))
        inv_disagreements.append(round(dis / len(sample), 4))
    floor_inv = max(inv_disagreements) if inv_disagreements else 0.0
    log("floor_inv_measure", draws=args.inv_seeds,
        disagreement_rates=inv_disagreements, floor_inv=floor_inv,
        delta_inv_formula="delta_inv := 2*floor_inv")

    # floor_perm: order permutation on a train sample, 3 permutations
    pairs = [c for c in eligible if len(c.members) == 2]
    sample = random.Random(78).sample(pairs, min(80, len(pairs)))
    perm_deltas = []
    for perm_seed in _parse_seeds(args.perm_seeds):
        delta = 0
        for c in sample:
            base = arbitrate_production(c, model, tau_hi, tau_m)
            rev = lib.Cluster(cid=c.cid, stratum=c.stratum,
                              substratum=c.substratum,
                              members=list(reversed(c.members)),
                              label=c.label, declaration=c.declaration)
            alt = arbitrate_production(rev, model, tau_hi, tau_m)
            delta += int(base["verdict"] != alt["verdict"]
                         or base.get("member") != alt.get("member"))
        perm_deltas.append(round(delta / len(sample), 4))
    floor_perm = max(perm_deltas) if perm_deltas else 0.0
    log("floor_perm_measure", perms=args.perm_seeds, delta_rates=perm_deltas,
        floor_perm=floor_perm,
        note="structurally 0: member features are bit-identical under "
             "presentation permutation (machine-asserted)")

    doc["floors_v2"] = {
        "artifact_sha256": pro["pins"]["artifact_sha256"],
        "floor_q": {"seeds": q_seeds, "rates": q_rates,
                    "p90_char": p90_char, "floor": round(floor_q, 4)},
        "q": round(q, 4),
        "floor_inv": {"draws": args.inv_seeds,
                      "disagreement_rates": inv_disagreements,
                      "floor": floor_inv},
        "delta_inv": round(2 * floor_inv, 4),
        "floor_perm": {"perms": args.perm_seeds, "delta_rates": perm_deltas,
                       "floor": floor_perm},
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"),
        "remeasure_after_single_shot": "FORBIDDEN",
    }
    doc.pop("committed_doc_sha256", None)
    doc["committed_doc_sha256"] = sha256_text(
        json.dumps(doc, ensure_ascii=False, sort_keys=True))
    doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    log("committed_doc_updated", path=str(doc_path),
        floors=["floor_q", "floor_inv", "floor_perm"])
    return 0


# --- Choice v2 floors (ML-STD-0004 declared floor, stage 2) -------------------

def _dist(values: list[float]) -> dict:
    """The ML-STD-0004 published distribution: median, p5, p95,
    max-pairwise (all four mandatory; only p5/p95 enter the formula)."""
    if not values:
        raise InstrumentAbort("distribution of an empty draw list")
    med = float(np.median(values))
    p5 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    mp = max((abs(a - b) for i, a in enumerate(values)
              for b in values[i + 1:]), default=0.0)
    return {"n": len(values),
            "median": round(med, 6), "p5": round(p5, 6), "p95": round(p95, 6),
            "max_pairwise": round(mp, 6),
            "method": "numpy.percentile linear interpolation (numpy "
                      f"{np.__version__})"}


def cmd_floors_v2(args: argparse.Namespace) -> int:
    """ALL Choice-v2 floors on the NEW artifact sha (ADDENDUM-6 §6 step 2;
    ML-STD-0004 stage 2): floor_seed (>= 10 composition draws seeds 43..52,
    distribution median/p5/p95/max-pairwise published BEFORE gate_start,
    delta_acc = max(p95 - p5; 0.10) frozen from the DECLARED statistics) +
    floor_q / floor_inv / floor_perm distributions in the SAME section (П4;
    their gate thresholds keep the ratified formulas). The v1
    «delta_acc_measure» section is NOT reused — a NEW v2 section. Fixed
    BEFORE gate_start; re-measuring after the single-shot is FORBIDDEN (П6)."""
    global _SLICE
    _SLICE = SLICE_FLOOR_V2
    started = dt.datetime.now(dt.timezone.utc)
    log("cost_declaration", expected_prolog_s=3, expected_cycle_s=1800,
        expected_peak_ram_gb=2.5,
        lane="v2 light+: 10 seed draws + 10 q draws through the v2 contour "
             "(embedder disk-cached); sequential, single lane")
    pro = prologue(expected_artifact=Path(args.artifact))
    booster, tau_hi, tau_m, ctx = resolve_ctx(Path(args.artifact), with_zone=True)
    model_sha = pro["pins"]["artifact_sha256"]
    doc_path = Path(args.doc_out)
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if "delta_acc_measure_v2" in doc:
        raise InstrumentAbort("delta_acc_measure_v2 already present — "
                              "re-derivation is forbidden (ML-STD-0004 П3/П6; "
                              "the v1 delta_acc_measure section is NOT reused)")

    # П1 order assert: stage 1 (formula committed) STRICTLY BEFORE stage 2
    # (first draw) — enforced by the runner, not by agreement.
    r = subprocess.run(["git", "merge-base", "--is-ancestor",
                        STD0004_FORMULA_COMMIT, "HEAD"], cwd=REPO,
                       capture_output=True)
    log("std0004_order_assert",
        formula_commit=STD0004_FORMULA_COMMIT, is_ancestor=r.returncode == 0,
        formula=DELTA_ACC_FORMULA,
        stage1="ML-STD-0004 §1/П2: formula fixed BEFORE any v2 draw "
               "(commit 5305c82, 2026-10-30, MLC round 6)",
        stage2="this run: draws -> published distribution -> frozen delta_acc "
               "-> gate_start",
        ok=r.returncode == 0)
    if r.returncode != 0:
        raise InstrumentAbort("ML-STD-0004 П1 order assert RED: the stage-1 "
                              f"formula commit {STD0004_FORMULA_COMMIT} is not "
                              "an ancestor of HEAD")

    log("floor_seed_v2_measure_start",
        seeds=FLOOR_SEEDS, n_draws=len(FLOOR_SEEDS),
        gate_leg_quota=FLOOR_LEG_QUOTA,
        recipe="mb.gen_family (frozen build recipe, same code path)",
        model_sha256=model_sha,
        contour="production §5.1 WITH the zone-v1 leg (the gating statistic "
                "is the COMPOSITION zone+model; ADDENDUM-6 §3)",
        statistic="acc_choice(model, draw) - max pooled baseline, CH-3 "
                  "gate-leg composition",
        delta_acc_formula=DELTA_ACC_FORMULA,
        standard="ML-STD-0004 (declared floor), stage 2",
        note="the FROZEN v2 model is NOT retrained; fixed BEFORE gate_start")

    train_clusters = load_train()
    train_ids = {m.mid for c in train_clusters for m in c.members}
    train_ids |= {c.cid for c in train_clusters}
    train_scen = _train_scenario_keys()
    holdout_ids = _holdout_id_manifest()
    log("holdout_id_manifest_read", n_ids=len(holdout_ids),
        content_read=False,
        note="id-only read for the ratified disjointness leg; no content kept")

    rows = []
    for seed in _parse_seeds(args.seeds):
        used = set(train_scen)  # holdout-style exclusion (fresh per draw)
        fam = mb.gen_family(seed, dict(_FLOOR_QUOTA_FULL), used,
                            f"ma-fs-{seed}", f"ma-fseed{seed}")
        got: dict[tuple[str, int], int] = {}
        for c in fam:
            got[(c.stratum, len(c.members))] = got.get((c.stratum, len(c.members)), 0) + 1
        if got != {("CH-3", 2): 22, ("CH-3", 3): 3} or len(fam) != 25:
            raise InstrumentAbort(f"draw seed={seed} composition mismatch: {got}")
        draw_ids = {m.mid for c in fam for m in c.members} | {c.cid for c in fam}
        id_train_hits = draw_ids & train_ids
        id_holdout_hits = draw_ids & holdout_ids
        draw_scen = {c.scenario for c in fam}
        scen_hits = draw_scen & train_scen
        log("floor_seed_draw_disjointness", seed=seed,
            ids_train=sorted(id_train_hits), ids_holdout=sorted(id_holdout_hits),
            scenarios_train=sorted(scen_hits))
        if id_train_hits or id_holdout_hits or scen_hits:
            raise InstrumentAbort(
                f"draw seed={seed} intersects existing material: "
                f"train_ids={sorted(id_train_hits)} "
                f"holdout_ids={sorted(id_holdout_hits)} "
                f"train_scenarios={sorted(scen_hits)}")

        hits = zone_fired = 0
        for c in fam:
            d = arbitrate_production(c, booster, tau_hi, tau_m, ctx=ctx)
            zone_fired += int(d.get("leg") == "zone-v1")
            hits += int(d["verdict"] == "keep"
                        and d.get("member") == c.label["member"])
        if zone_fired:
            raise InstrumentAbort(
                f"draw seed={seed}: the zone-v1 leg fired on fresh CH-3 "
                f"({zone_fired} pairs) — composition poisoned, floors are "
                "not measurable (report to TL, not an input fix)")
        acc_model = hits / len(fam)
        acc_base = max(
            _pooled_acc(fam, lambda c, b=bfn: lib.correct_set_ok(c, b(c.members)))
            for bfn in lib.BASELINES.values())
        rows.append({"seed": seed, "n_clusters": len(fam),
                     "acc_model": round(acc_model, 4),
                     "acc_baseline_max": round(acc_base, 4),
                     "delta": round(acc_model - acc_base, 4)})
        log("floor_seed_draw", **rows[-1], zone_fired=0)

    deltas = [r_["delta"] for r_ in rows]
    dist = _dist(deltas)
    delta_acc = max(dist["p95"] - dist["p5"], 0.10)
    log("floor_seed_v2_measure_end",
        per_seed=rows, floor_seed_distribution=dist,
        delta_acc_declared=round(delta_acc, 4),
        formula=DELTA_ACC_FORMULA,
        pairs_compared=len(deltas) * (len(deltas) - 1) // 2,
        model_sha256=model_sha, artifact=args.artifact,
        wall_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
        remeasure_after_single_shot="FORBIDDEN")

    # --- floor_q (gate g_abstain_sens): 10 draws, the ratified formula
    # q = max(0.20; p90_char + 2*floor_q) is NOT touched (ML-STD-0004 П4) —
    # only its distribution is published now.
    q_seeds = _parse_seeds(args.q_seeds)
    q_rates = []
    for seed in q_seeds:
        fam = _gate_leg_draw(seed)
        abst = 0
        zone_fired = 0
        for c in fam:
            d = arbitrate_production(c, booster, tau_hi, tau_m, ctx=ctx)
            zone_fired += int(d.get("leg") == "zone-v1")
            abst += int(d["verdict"] == "abstain")
        if zone_fired:
            raise InstrumentAbort(f"floor_q draw seed={seed}: zone fired on "
                                  "fresh CH-3 — composition poisoned")
        q_rates.append(round(abst / len(fam), 4))
    dist_q = _dist([float(x) for x in q_rates])
    floor_q = dist_q["max_pairwise"]
    p90_char = sorted(q_rates)[int(round(0.9 * (len(q_rates) - 1)))] if q_rates else 0.0
    q = max(0.20, p90_char + 2 * floor_q)
    log("floor_q_v2_measure", seeds=q_seeds, rates=q_rates, distribution=dist_q,
        p90_char=p90_char, floor_q=floor_q, q=round(q, 4),
        formula="q := max(0.20; p90_char + 2*floor_q) — unchanged (П4)",
        note="only the distribution is new; the threshold formula is not touched")

    # --- floor_inv: cosmetic variation on train fold-eligible clusters
    eligible = [c for c in train_clusters if c.stratum in ("CH-3", "CH-5")]
    inv_seeds = _parse_seeds(args.inv_seeds)
    inv_disagreements = []
    for draw_seed in inv_seeds:
        rng = random.Random(draw_seed)
        sample = rng.sample(eligible, min(80, len(eligible)))
        dis = 0
        for c in sample:
            base = arbitrate_production(c, booster, tau_hi, tau_m, ctx=ctx)
            pert = lib.Cluster(
                cid=c.cid, stratum=c.stratum, substratum=c.substratum,
                members=[lib.Member(m.mid, m.title,
                                    lib.cosmetic_variant(m.body, rng),
                                    m.tags, m.record_type, m.language,
                                    m.created_at)
                         for m in c.members],
                label=c.label, declaration=c.declaration)
            alt = arbitrate_production(pert, booster, tau_hi, tau_m, ctx=ctx)
            dis += int(base["verdict"] != alt["verdict"]
                       or base.get("member") != alt.get("member"))
        inv_disagreements.append(round(dis / len(sample), 4))
    dist_inv = _dist([float(x) for x in inv_disagreements])
    floor_inv = max(inv_disagreements) if inv_disagreements else 0.0
    log("floor_inv_v2_measure", draws=args.inv_seeds,
        disagreement_rates=inv_disagreements, distribution=dist_inv,
        floor_inv=floor_inv, delta_inv=round(2 * floor_inv, 4),
        delta_inv_formula="delta_inv := 2*floor_inv — unchanged (П4)")

    # --- floor_perm: order permutation on a train sample
    pairs = [c for c in eligible if len(c.members) == 2]
    sample = random.Random(78).sample(pairs, min(80, len(pairs)))
    perm_seeds = _parse_seeds(args.perm_seeds)
    perm_deltas = []
    for perm_seed in perm_seeds:
        delta = 0
        for c in sample:
            base = arbitrate_production(c, booster, tau_hi, tau_m, ctx=ctx)
            rev = lib.Cluster(cid=c.cid, stratum=c.stratum,
                              substratum=c.substratum,
                              members=list(reversed(c.members)),
                              label=c.label, declaration=c.declaration)
            alt = arbitrate_production(rev, booster, tau_hi, tau_m, ctx=ctx)
            delta += int(base["verdict"] != alt["verdict"]
                         or base.get("member") != alt.get("member"))
        perm_deltas.append(round(delta / len(sample), 4))
    dist_perm = _dist([float(x) for x in perm_deltas])
    floor_perm = max(perm_deltas) if perm_deltas else 0.0
    log("floor_perm_v2_measure", perms=args.perm_seeds,
        delta_rates=perm_deltas, distribution=dist_perm, floor_perm=floor_perm,
        note="structurally 0: member features are bit-identical under "
             "presentation permutation (machine-asserted)")

    doc["delta_acc_measure_v2"] = {
        "standard": "ML-STD-0004 (declared floor), stage 2",
        "formula": DELTA_ACC_FORMULA,
        "stage1_pin": {"commit": STD0004_FORMULA_COMMIT,
                       "assert_event": "std0004_order_assert (above in run-log)",
                       "note": "formula fixed 2026-10-30 BEFORE any v2 draw"},
        "seeds": FLOOR_SEEDS,
        "per_seed": rows,
        "floor_seed_distribution": dist,
        "delta_acc_declared": round(delta_acc, 4),
        "delta_acc_recheck": round(max(dist["p95"] - dist["p5"], 0.10), 4),
        "floor_q": {"seeds": q_seeds, "rates": q_rates,
                    "distribution": dist_q, "floor": floor_q,
                    "p90_char": p90_char, "q": round(q, 4),
                    "formula": "q := max(0.20; p90_char + 2*floor_q)"},
        "floor_inv": {"draws": inv_seeds,
                      "disagreement_rates": inv_disagreements,
                      "distribution": dist_inv, "floor": floor_inv,
                      "delta_inv": round(2 * floor_inv, 4),
                      "formula": "delta_inv := 2*floor_inv"},
        "floor_perm": {"perms": perm_seeds, "delta_rates": perm_deltas,
                       "distribution": dist_perm, "floor": floor_perm},
        "model_sha256": model_sha,
        "artifact": args.artifact,
        "contour": "production §5.1 WITH zone-v1 (composition statistic)",
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"),
        "remeasure_after_single_shot": "FORBIDDEN",
    }
    if doc["delta_acc_measure_v2"]["delta_acc_declared"] != \
            doc["delta_acc_measure_v2"]["delta_acc_recheck"]:
        raise InstrumentAbort("declared delta_acc does not reproduce from the "
                              "published p5/p95 under the ML-STD-0004 formula")
    doc.pop("committed_doc_sha256", None)
    doc["committed_doc_sha256"] = sha256_text(
        json.dumps(doc, ensure_ascii=False, sort_keys=True))
    doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    log("committed_doc_updated", path=str(doc_path),
        section="delta_acc_measure_v2",
        floors=["floor_seed", "floor_q", "floor_inv", "floor_perm"],
        delta_acc_declared=round(delta_acc, 4))
    return 0


def cmd_prelaunch(args: argparse.Namespace) -> int:
    """ADDENDUM-3 §3.6 pre-launch checklist on the new sha (train-side; NO
    sealed read): committee-scored ∉ inputs, sanity 7/7, noharm QA
    (expect 0/171 + 0/27), sheet pre-check (member_ids ⊆ synthetic_train,
    id level), floors completeness. ``--profile v2`` runs the Choice-v2
    checklist instead (ADDENDUM-6 §6 step 2 tail): the v2 contour with the
    zone leg, the ML-STD-0004 prereg event, the zone-off honesty probe and
    the sheet prematch subcommand."""
    global _SLICE
    if getattr(args, "profile", "v1") == "v2":
        return _prelaunch_v2(args)
    _SLICE = SLICE_FLOOR
    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=120,
        expected_peak_ram_gb=1.0, lane="light (draft §9)")
    pro = prologue(expected_artifact=Path(args.artifact))
    model, tau_hi, tau_m = load_frozen(Path(args.artifact))
    doc = find_doc_for(pro["pins"]["artifact_sha256"])

    # committee-scored-v1.json NOT in the inputs (machine assert)
    forbidden = "committee-scored-v1.json"
    input_list = ["corpus manifest", "train.jsonl", "train-negatives.jsonl",
                  "negatives sidecar manifest", "frozen artifact + committed doc",
                  "sheet-selection.json (prelaunch id check only)"]
    in_inputs = any(forbidden in p for p in input_list)
    log("committee_scored_not_in_inputs", forbidden=forbidden,
        input_list=input_list, in_inputs=in_inputs, ok=not in_inputs,
        note="existence of the file on disk is NOT an input; the assert is "
             "on the declared input list")
    if in_inputs:
        raise InstrumentAbort("committee-scored-v1.json found in the input list")

    # П6 pre-registration, VERBATIM per ADDENDUM-4 §4 — written to the
    # run-log BEFORE any gate_start of the upcoming single-shot cycle
    PREREG = ("ратифицированная процедура на материал-чувствительной модели "
              "даёт δ_acc = 0.48; свежие дельты 0.24–0.48 (медиана 0.44). "
              "Если g_acc красный: интерпретация — \"модель v1 не "
              "сертифицируемо превосходит базлайны под собственной "
              "ратифицированной процедурой допуска\"; вердикт цикла DECLINE "
              "окончателен, повтор с теми же числами запрещён, возврат "
              "только через новый fresh-seal цикл с канон-вопросом о "
              "семантике пола в ArchCom. Послемёрная аналитика (легальна): "
              "дельта holdout сравнивается с распределением 10 розыгрышей — "
              "внутри диапазона → \"нестабильность на свежем материале\" "
              "(дефект покрытия/рецепта), ниже хвоста → \"реальная слабость "
              "на holdout\"; оба исхода = DECLINE, порог не пересматривается")
    log("prereg_delta_acc", prereg=PREREG, verbatim="ADDENDUM-4 §4",
         delta_acc=doc.get("delta_acc_measure", {}).get("delta_acc"),
         note="written BEFORE gate_start; the gate asserts this event exists "
              "in the run-log prior to gate_start (П6 order)")

    # sanity 7/7 on the NEW model (probes 3/4 = CH-3 quality guards)
    run_model_sanity_suite(model, tau_hi, tau_m)

    # noharm QA: expect 0/171 + 0/27 (surface untouched, §4)
    surface = Path(doc["noharm_surface"]["path"])
    rows_total, proposals_total = _noharm_proposals(surface, model, tau_hi, tau_m)
    notdup_rows = [json.loads(line) for line in open(surface, encoding="utf-8")
                   if json.loads(line)["pair_id"].startswith("V4-PND")]
    tmp = ART_DIR / "_qa_notdup.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in notdup_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    n_notdup, p_notdup = _noharm_proposals(tmp, model, tau_hi, tau_m)
    tmp.unlink()
    qa_ok = proposals_total == 0 and p_notdup == 0
    log("noharm_qa", rows=rows_total, proposals=proposals_total,
        notdup_rows=n_notdup, notdup_proposals=p_notdup,
        expectation="0/171 + 0/27", ok=qa_ok,
        note="red => разбор рецепта; поверхность не трогается (§3.5)")
    if not qa_ok:
        raise InstrumentAbort(f"noharm QA RED: {proposals_total}/{rows_total} "
                              f"(notdup {p_notdup}/{n_notdup})")

    # sheet pre-check: member_ids ⊆ synthetic_train (id level only)
    sheet_doc = json.loads((REPO / "data" / "merge-arbiter" / "spot-sheet" /
                            "sheet-selection.json").read_text(encoding="utf-8"))
    sheet_ids = set()
    for row in sheet_doc.get("synthetic", []):
        sheet_ids.update(row.get("member_ids", []))
    train_ids = {m.mid for c in load_train() for m in c.members}
    missing = sorted(sheet_ids - train_ids)
    log("sheet_precheck", sheet_member_ids=len(sheet_ids),
        missing_from_train=missing[:5], ok=not missing,
        note="id-level read only; the sheet fp e45e72a4… is not touched")
    if missing:
        raise InstrumentAbort(f"sheet pre-check RED: {len(missing)} member_ids "
                              "outside synthetic_train")


    absent = [k for k in ("delta_acc_measure", "floors_v2") if k not in doc]
    log("prelaunch_done", artifact=args.artifact,
        doc=doc.get("_doc_path", "n/a"), missing_sections=absent,
        status="GREEN" if not absent else "INCOMPLETE",
        note="single-shot stays unexecuted; return to it on a separate "
             "command after MLC scoring validation")
    if absent:
        print(f"PRELAUNCH INCOMPLETE — missing doc sections: {absent}",
              file=sys.stderr)
        return 1
    return 0


# --- Choice v2 pre-launch (ADDENDUM-6 §6 step 2 tail) -------------------------

def _zone_off_probe(doc: dict, booster: lgb.Booster, tau_hi: float,
                    tau_m: float, vec: VecLeg) -> dict:
    """ADDENDUM-6 §1 leg 4: the zone-off honesty probe (NOT a gate).

    Reproduce the refused class (keep = exactly 24/171) with the FROZEN
    v1-iter3 artifact (sha 19bf073e…), then run the RETRAINED model through
    the same contour WITHOUT the zone on those 24 rows: how many does the
    model itself still refuse? Published in the run-log BEFORE gate_start
    as the honesty number — it gates nothing."""
    v1_path = None
    for p in sorted(ART_DIR.glob("*.json")):
        try:
            if sha256_file(p).startswith(ITER3_ARTIFACT_SHA_PREFIX):
                v1_path = p
                break
        except OSError:
            continue
    if v1_path is None:
        raise InstrumentAbort(f"v1-iter3 artifact (sha {ITER3_ARTIFACT_SHA_PREFIX}…) "
                              f"not found under {ART_DIR} — the zone-off probe "
                              "needs the v1 reference")
    v1_booster, v1_hi, v1_m = load_frozen(v1_path)
    surface = Path(doc["noharm_surface"]["path"])
    rows = [json.loads(line) for line in open(surface, encoding="utf-8")]
    v1_kept = []
    for idx, r in enumerate(rows):
        d = arbitrate_production(_noharm_cluster(r, idx), v1_booster, v1_hi, v1_m)
        if d["verdict"] == "keep":
            v1_kept.append(r)
    if len(v1_kept) != 24:
        raise InstrumentAbort(f"v1 reproduction on the 171-surface diverges: "
                              f"keep = {len(v1_kept)} (framing §1.1: exactly "
                              "24) — surface or reference artifact drifted")
    nozone_ctx = {"vec": vec}  # zone OFF, vec ON (the v2 model needs its features)
    keeps_of_24 = 0
    for r in v1_kept:
        d = arbitrate_production(_noharm_cluster(r, 0), booster, tau_hi,
                                 tau_m, ctx=nozone_ctx)
        keeps_of_24 += int(d["verdict"] == "keep")
    total_proposals = 0
    for idx, r in enumerate(rows):
        d = arbitrate_production(_noharm_cluster(r, idx), booster, tau_hi,
                                 tau_m, ctx=nozone_ctx)
        total_proposals += int(d["verdict"] == "keep")
    probe = {
        "v1_reference": str(v1_path),
        "v1_reference_sha256": ITER3_ARTIFACT_SHA_PREFIX,
        "v1_keep_reproduced": len(v1_kept),
        "v2_zone_off_keep_of_24": keeps_of_24,
        "v2_zone_off_refuses_of_24": 24 - keeps_of_24,
        "surface_rows": len(rows),
        "surface_proposals_zone_off": total_proposals,
        "not_a_gate": True,
        "honesty": "how many of the 24 the RETRAINED model rejects WITHOUT "
                   "the zone (ADDENDUM-6 §1 leg 4; published before gate_start)",
    }
    log("zone_off_probe", **probe)
    return probe


def _prelaunch_v2(args: argparse.Namespace) -> int:
    """Choice-v2 pre-launch checklist (ADDENDUM-6 §6 step 2 tail; NO sealed
    read): committee-scored ∉ inputs, ML-STD-0004 prereg event, sanity 7/7
    through the FULL v2 contour, noharm QA 0/171 + 0/27 (composition
    zone+model; ANY red = STOP + report to TL — hard terminality ADDENDUM-5
    §2, NOT an iteration), zone-off honesty probe, sheet id pre-check and
    the prematch subcommand."""
    global _SLICE
    _SLICE = SLICE_FLOOR_V2
    log("cost_declaration", expected_prolog_s=3, expected_cycle_s=1200,
        expected_peak_ram_gb=2.5,
        lane="v2 light+: contour probes over the surface + floors reads "
             "(embedder disk-cached); sequential, single lane")
    pro = prologue(expected_artifact=Path(args.artifact))
    booster, tau_hi, tau_m, ctx = resolve_ctx(Path(args.artifact), with_zone=True)
    vec = ctx["vec"]
    doc = find_doc_for(pro["pins"]["artifact_sha256"])

    # committee-scored surfaces are NOT in the inputs (machine assert,
    # both scoring generations)
    forbidden = ("committee-scored-v1.json", "committee-scored-v2.json")
    input_list = ["corpus manifest", "train.jsonl", "train-negatives.jsonl",
                  "train-negatives-v2.jsonl", "null-envelope.json",
                  "null-envelope-v2.json", "frozen artifact + committed doc",
                  "sheet-selection.json (prelaunch id check only)"]
    in_inputs = [f for f in forbidden if any(f in p for p in input_list)]
    log("committee_scored_not_in_inputs", forbidden=list(forbidden),
        input_list=input_list, in_inputs=in_inputs, ok=not in_inputs,
        note="existence of the files on disk is NOT an input; the assert is "
             "on the declared input list")
    if in_inputs:
        raise InstrumentAbort(f"committee-scored file found in the input "
                              f"list: {in_inputs}")

    # ML-STD-0004 prereg (П1/П6): the declared floor is written to the
    # run-log BEFORE any gate_start of the upcoming single-shot cycle
    measure = doc.get("delta_acc_measure_v2")
    if not measure or "floor_seed_distribution" not in measure:
        raise InstrumentAbort("committed doc has no delta_acc_measure_v2 — "
                              "run floors-v2 BEFORE prelaunch (the declared "
                              "floor must exist before the prereg event)")
    dist = measure["floor_seed_distribution"]
    PREREG = (
        "Цикл Choice v2 (ADDENDUM-6): допуск g_acc объявлен по ML-STD-0004 "
        "ДО gate_start. Формула (ступень 1, коммит "
        f"{STD0004_FORMULA_COMMIT}): {DELTA_ACC_FORMULA}. Ступень 2: "
        f"{dist['n']} розыгрышей (seeds {measure['seeds']}), распределение "
        f"дельт: медиана {dist['median']}, p5 {dist['p5']}, p95 {dist['p95']}, "
        f"max-pairwise {dist['max_pairwise']}; объявленный "
        f"δ_acc = {measure['delta_acc_declared']} заморожен до sealed-чтения; "
        "перемер пола после single-shot запрещён (П6); послемёрная "
        "аналитика легальна и вердикт не меняет")
    log("prereg_delta_acc_v2", prereg=PREREG, formula=DELTA_ACC_FORMULA,
        distribution=dist, delta_acc_declared=measure["delta_acc_declared"],
        artifact_sha256=measure.get("model_sha256"),
        note="written BEFORE gate_start; the gate asserts this event exists "
             "prior to gate_start (П1/П6)")

    # sanity 7/7 through the FULL v2 contour (a zone fire on a CH-3/CH-4
    # fixture would surface as leg zone-v1 and RED the probe)
    run_model_sanity_suite(booster, tau_hi, tau_m, ctx=ctx)

    # noharm QA through the v2 contour: expect 0/171 + 0/27 — the
    # composition zone+model is what is certified (ADDENDUM-6 §3)
    surface = Path(doc["noharm_surface"]["path"])
    rows_total, proposals_total = _noharm_proposals(surface, booster,
                                                    tau_hi, tau_m, ctx=ctx)
    notdup_rows = [json.loads(line) for line in open(surface, encoding="utf-8")
                   if json.loads(line)["pair_id"].startswith("V4-PND")]
    tmp = ART_DIR / "_qa_notdup.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in notdup_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    n_notdup, p_notdup = _noharm_proposals(tmp, booster, tau_hi, tau_m, ctx=ctx)
    tmp.unlink()
    qa_ok = proposals_total == 0 and p_notdup == 0
    log("noharm_qa_v2", rows=rows_total, proposals=proposals_total,
        notdup_rows=n_notdup, notdup_proposals=p_notdup,
        expectation="0/171 + 0/27", ok=qa_ok,
        contour="COMPOSITION zone-v1 + model (ADDENDUM-6 §3; the wording "
                "«модель зелёная на noharm» is forbidden)",
        note="any red => СТОП и отчёт TL (хард-терминальность ADDENDUM-5 §2), "
             "НЕ итерация; поверхность не трогается")
    if not qa_ok:
        raise InstrumentAbort(f"noharm QA RED: {proposals_total}/{rows_total} "
                              f"(notdup {p_notdup}/{n_notdup}) — STOP, report "
                              "to TL; do NOT iterate")

    # zone-off honesty probe (published, NOT a gate)
    _zone_off_probe(doc, booster, tau_hi, tau_m, vec)

    # sheet pre-check: member_ids ⊆ synthetic_train (id level only)
    sheet_doc = json.loads((REPO / "data" / "merge-arbiter" / "spot-sheet" /
                            "sheet-selection.json").read_text(encoding="utf-8"))
    sheet_ids = set()
    for row in sheet_doc.get("synthetic", []):
        sheet_ids.update(row.get("member_ids", []))
    train_ids = {m.mid for c in load_train() for m in c.members}
    missing = sorted(sheet_ids - train_ids)
    log("sheet_precheck", sheet_member_ids=len(sheet_ids),
        missing_from_train=missing[:5], ok=not missing,
        note="id-level read only; the sheet fp e45e72a4… is not touched")
    if missing:
        raise InstrumentAbort(f"sheet pre-check RED: {len(missing)} member_ids "
                              "outside synthetic_train")

    # prematch: the ONLY permitted spot-sheet subcommand (render/verdicts/
    # internal stay untouched — detached scoring in progress)
    r = subprocess.run([sys.executable,
                        str(REPO / "scripts" / "merge_arbiter_sheet_score.py"),
                        "prematch"], cwd=REPO, capture_output=True, text=True)
    try:
        payload = json.loads(r.stdout)
    except json.JSONDecodeError:
        payload = {}
    prematch_ok = r.returncode == 0 and bool(payload.get("prematch_ok"))
    log("sheet_prematch", ok=prematch_ok,
        prematch_ok=payload.get("prematch_ok"),
        sheet_rows=payload.get("sheet_rows"),
        syn_members_subset_train=payload.get("syn_members_subset_train"),
        syn_members_intersect_holdout=payload.get("syn_members_intersect_holdout"),
        sheet_class_counts=payload.get("sheet_class_counts"),
        stderr_tail=r.stderr.strip().splitlines()[-3:],
        note="scripts/merge_arbiter_sheet_score.py prematch ONLY")
    if not prematch_ok:
        raise InstrumentAbort("sheet prematch RED — the pre-launch checklist "
                              "is not green")

    absent = [k for k in ("delta_acc_measure_v2",) if k not in doc]
    log("prelaunch_v2_done", artifact=args.artifact,
        doc=doc.get("_doc_path", "n/a"), missing_sections=absent,
        contour="zone-v1 + model (composition)",
        status="GREEN" if not absent else "INCOMPLETE",
        note="the single-shot stays UNEXECUTED — it runs only on a separate, "
             "explicit owner command (ADDENDUM-6 §6 step 3)")
    if absent:
        print(f"PRELAUNCH V2 INCOMPLETE — missing doc sections: {absent}",
              file=sys.stderr)
        return 1
    return 0


# --- subcommand: sanity (frozen artifact, no training) ------------------------

def cmd_sanity(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_TRAIN
    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=30,
        expected_peak_ram_gb=0.5, lane="light (draft §9)")
    pro = prologue(expected_artifact=Path(args.artifact))
    _ = pro  # pins asserted in the prologue
    model, tau_hi, tau_m = load_frozen(Path(args.artifact))
    run_model_sanity_suite(model, tau_hi, tau_m)
    log("sanity_cmd_done", status="GREEN", artifact=args.artifact)
    return 0


# --- subcommand: gate (single-shot; implemented, NOT executed in this phase) --

SEALED_EVENTS = ("sealed_read_once", "gate_done")


def prior_single_shot(slice_name: str) -> bool:
    """П8: a re-run after the spent single-shot is refused by the runner."""
    if not RUNLOG.exists():
        return False
    for line in open(RUNLOG, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a concurrent appender — not evidence
        if row.get("slice") == slice_name and row.get("event") in SEALED_EVENTS:
            return True
    return False


def _noharm_cluster(row: dict, idx: int) -> lib.Cluster:
    a = lib.Member(mid=row.get("id_a", "a"), title=row["title_a"],
                   body=row["body_a"], tags=tuple(row.get("tags_a", ())),
                   record_type=row.get("type_a"), language=row.get("lang_a"),
                   created_at=row.get("created_a"))
    b = lib.Member(mid=row.get("id_b", "b"), title=row["title_b"],
                   body=row["body_b"], tags=tuple(row.get("tags_b", ())),
                   record_type=row.get("type_b"), language=row.get("lang_b"),
                   created_at=row.get("created_b"))
    return lib.Cluster(cid=row.get("pair_id", f"noharm-{idx}"),
                       stratum="NOT-DUP", substratum="v42",
                       members=[a, b], label={"type": "abstain"},
                       declaration={})


def _noharm_proposals(noharm: Path, model: lgb.Booster,
                      tau_hi: float, tau_m: float,
                      ctx: "dict | None" = None,
                      rows_out: "list | None" = None) -> tuple[int, int]:
    """g_noharm_v42: frozen NOT-dup surface (committed v42/v4.3 train
    strata; holdouts and seals are NOT read). A keep verdict on any row =
    a fold proposal. Expected: exact 0.

    SURFACE CONTRACT (ratified TL dispatch 29.10; surface accepted with
    27 train-side rows, pins ece065de / 404d43f2 / 526522bf / 36ced21d /
    6ab5b5fa): the surface is built from TRAIN-side material ONLY.
    THE TRAP: the source pool the surface is drawn from also contains
    9 rows of holdout origin — a naive builder that sweeps the source
    file wholesale pulls those 9 rows in and BURNS g_noharm on sealed
    material (a holdout read outside the single-shot: run-log shows
    eval content reads before sealed_read_once, surface fp carries
    holdout bytes). The builder must filter to the 27 train rows by the
    pinned id list and ASSERT row count == 27 and 0 id overlap with the
    holdout id-manifest before emission. The gate runner additionally
    checks the surface sha against the ratified pins at load time.
    """
    proposals = rows = 0
    with open(noharm, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cluster = _noharm_cluster(r, rows)
            d = arbitrate_production(cluster, model, tau_hi, tau_m, ctx=ctx)
            rows += 1
            if rows_out is not None:
                rows_out.append({"pair_id": cluster.cid, "verdict": d})
            proposals += int(d["verdict"] == "keep")
    return rows, proposals


def cmd_gate(args: argparse.Namespace) -> int:
    global _SLICE
    artifact = Path(args.artifact)
    noharm = Path(args.noharm)

    # sniff the artifact generation for the slice name (the v1 slice/event
    # history stays untouched; the v2 cycle has its own single-shot slot)
    art_sniff = json.loads(artifact.read_text(encoding="utf-8"))
    is_v2 = len(art_sniff.get("feature_names", MEMBER_FEATURE_NAMES)) > \
        len(MEMBER_FEATURE_NAMES)
    slice_name = SLICE_GATE_V2 if is_v2 else SLICE_GATE
    _SLICE = slice_name

    if prior_single_shot(slice_name):
        log("refuse", reason="single-shot already spent for this slice "
                             "(sealed_read_once/gate_done in run-log) — П8")
        print("REFUSED: single-shot already spent (П8).", file=sys.stderr)
        return 2
    if not args.execute_single_shot:
        log("refuse", reason="gate phase implemented but NOT executed: "
                             "--execute-single-shot absent; no sealed read",
            note="this phase stays train-only; the single-shot is a separate, "
                 "explicitly flagged run")
        print("REFUSED: gate phase requires --execute-single-shot. "
              "No sealed surface was touched.", file=sys.stderr)
        return 2

    log("cost_declaration", expected_prolog_s=2, expected_cycle_s=90,
        expected_peak_ram_gb=1.0, lane="light (draft §9)")
    pro = prologue(expected_artifact=artifact)
    doc = find_doc_for(sha256_file(artifact))
    # noharm surface pin: fixed BEFORE gate_start (committed doc section
    # noharm_surface); divergence = STOP class instrument, no sealed read
    surf_sec = doc.get("noharm_surface")
    if not surf_sec or surf_sec.get("sha256") != sha256_file(noharm):
        raise InstrumentAbort(
            "noharm surface missing or its sha256 diverges from the "
            "committed noharm_surface pin — the gate never runs on an "
            "unpinned surface")
    log("noharm_surface_pin", path=str(noharm),
        sha256=surf_sec["sha256"], counts=surf_sec.get("counts"), ok=True)
    model, tau_hi, tau_m = load_frozen(artifact)
    # Choice v2: the gate runs the FULL v2 contour (zone-v1 + model) — the
    # noharm certificate and the gating statistic are COMPOSITION legs.
    if is_v2:
        _, _, _, ctx = resolve_ctx(artifact, with_zone=True)
    else:
        ctx = None

    # П1+П6: the ratified prereg event must exist BEFORE gate_start
    # (v1: ADDENDUM-4 §4 verbatim; v2: the ML-STD-0004 declared floor)
    prereg_seen = False
    prereg_event = None
    if RUNLOG.exists():
        for line in open(RUNLOG, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") in ("prereg_delta_acc", "prereg_delta_acc_v2"):
                prereg_seen = True
                prereg_event = row.get("event")
                break
    if not prereg_seen:
        raise InstrumentAbort("prereg_delta_acc[_v2] event not found in the "
                              "run-log — the ratified pre-registration is a "
                              "precondition of gate_start (П6)")
    log("prereg_order_assert", prereg_seen=True, prereg_event=prereg_event, ok=True)

    # П1: gate_start BEFORE any eval computation (sanity suite, fixtures,
    # no-harm, sealed reads are all eval reads — they start only now).
    log("gate_start",
        note="full prologue green above; every eval read starts after this event")

    # pre-sealed: sanity §6.1 with the frozen model (red => abort, no holdout)
    run_model_sanity_suite(model, tau_hi, tau_m, ctx=ctx)

    # g_corner leg 1 (identity corner) — fixture-based, no sealed read
    cells = {cell["probe"]: cell for cell in lib.build_fixture_cells()}
    corner = arbitrate_production(cells[1]["cluster"], model, tau_hi, tau_m,
                                  ctx=ctx)
    corner_ok = (corner["verdict"] == "keep"
                 and corner.get("member") in
                 {m.mid for m in cells[1]["cluster"].members})
    log("g_corner_leg1", ok=corner_ok, result=corner, threshold="exact")

    rows, proposals = _noharm_proposals(noharm, model, tau_hi, tau_m, ctx=ctx)
    g_noharm_ok = proposals == 0
    log("g_noharm_v42", rows=rows, proposals=proposals, ok=g_noharm_ok,
        threshold="exact 0", surface_sha256=sha256_file(noharm),
        contour="COMPOSITION zone-v1 + model" if is_v2 else "model leg")

    if not (corner_ok and g_noharm_ok):
        log("abort_pregate_red", corner_ok=corner_ok, noharm_ok=g_noharm_ok,
            note="pre-gate red -> holdout NOT read (ML-STD-0002 П6)")
        return 3

    # ============ SEALED READ — SINGLE-SHOT (not executed in this phase) =====
    holdout = _read_holdout_sealed()          # the single legal read
    log("sealed_read_once", n_clusters=holdout["n"], sha256=holdout["sha256"])
    gates = _eval_gates_on_holdout(model, tau_hi, tau_m, holdout,
                                   pro["manifest"], doc, ctx)
    for name, g in gates.items():
        log("gate_result", gate=name, **g)
    verdict = "ADOPT" if all(g["ok"] for g in gates.values()) else "DECLINE"
    log("gate_done", verdict=verdict,
        red=[k for k, g in gates.items() if not g["ok"]],
        discipline="holdout read exactly once; no re-run (П8)")
    print(f"VERDICT: {verdict}")
    return 0


def _read_holdout_sealed() -> dict:
    """THE single legal read of MA-holdout-choice (single-shot).

    Caller contract: full prologue green + gate_start logged + pre-gate
    legs green. Any prologue mismatch raises InstrumentAbort BEFORE this
    point, so an aborted cycle never reaches the sealed surface. The seal
    file itself is not opened — the holdout bytes are pinned by the BUILD
    manifest (holdout.sha256), which this runner verifies (П4-style).
    """
    sha = sha256_file(HOLDOUT_JSONL)
    clusters = []
    with open(HOLDOUT_JSONL, encoding="utf-8") as fh:
        for line in fh:
            clusters.append(_cluster_from_row(json.loads(line)))
    strata: dict[str, int] = {}
    for c in clusters:
        strata[c.stratum] = strata.get(c.stratum, 0) + 1
    mism = {k: {"got": v, "want": EXPECTED_HOLDOUT_STRATA[k]}
            for k, v in strata.items() if EXPECTED_HOLDOUT_STRATA.get(k) != v}
    if mism or set(strata) != set(EXPECTED_HOLDOUT_STRATA):
        raise InstrumentAbort(f"holdout strata count mismatch: {mism}")
    return {"sha256": sha, "clusters": clusters, "n": len(clusters),
            "strata": strata}


def _eval_gates_on_holdout(model: lgb.Booster, tau_hi: float, tau_m: float,
                           holdout: dict, manifest: dict, doc: dict,
                           ctx: "dict | None" = None) -> dict:
    """Gate formulas §5 — thresholds are FROZEN constants, never revisited.

    δ_acc resolution (fail-closed; the gate never runs on an unfixed floor):
    Choice v2 reads the ML-STD-0004 section delta_acc_measure_v2 and
    RECOMPUTES the declared value from the published p5/p95 under the
    frozen formula (П2) — a mismatch is a process incident, not a silent
    fix. The v1 section delta_acc_measure stays the v1 path (П5: the v1
    letter 0.48 is not revisited). q: v2 reads it from the published
    floor_q derivation; v1 carries the ratified 0.20.
    """
    m2 = doc.get("delta_acc_measure_v2")
    if m2:
        dist = m2.get("floor_seed_distribution") or {}
        declared = float(m2["delta_acc_declared"])
        recheck = max(float(dist["p95"]) - float(dist["p5"]), 0.10)
        if abs(declared - recheck) > 1e-9:
            raise InstrumentAbort(
                "ML-STD-0004 П2: the declared delta_acc does not reproduce "
                f"from the published p5/p95 (declared {declared}, recheck "
                f"{recheck}) — процессный инцидент, эскалация MLC/владельцу")
        delta_acc = declared
        floor_seed_aggregate = float(dist["max_pairwise"])
        q = float(m2.get("floor_q", {}).get("q", 0.20))
        floor_source = "ML-STD-0004 delta_acc_measure_v2 (declared floor), " \
                       "fixed before gate_start"
    else:
        measure = doc.get("delta_acc_measure")
        if not measure or "delta_acc" not in measure or \
                "floor_seed_aggregate" not in measure:
            raise InstrumentAbort(
                "committed doc has no ratified delta_acc section — δ_acc is "
                "unfixed; run the floor measurement BEFORE the gate cycle")
        delta_acc = float(measure["delta_acc"])
        floor_seed_aggregate = float(measure["floor_seed_aggregate"])
        q = 0.20
        floor_source = "ratified delta_acc_measure (MLC 29.10, option a), " \
                       "fixed before gate_start"

    clusters: list[lib.Cluster] = holdout["clusters"]
    ch3 = [c for c in clusters if c.stratum == "CH-3"]
    ch24 = [c for c in clusters if c.stratum in ("CH-2", "CH-4")]

    d3 = {c.cid: arbitrate_production(c, model, tau_hi, tau_m, ctx=ctx)
          for c in ch3}
    hits = sum(1 for c in ch3 if d3[c.cid]["verdict"] == "keep"
               and d3[c.cid].get("member") == c.label["member"])
    abst3 = sum(1 for c in ch3 if d3[c.cid]["verdict"] == "abstain")
    zone3 = sum(1 for c in ch3 if d3[c.cid].get("leg") == "zone-v1")
    acc_model = hits / len(ch3) if ch3 else 0.0
    abst_rate3 = abst3 / len(ch3) if ch3 else 0.0

    # baselines on the SAME CH-3 material, pooled (the g_acc statistic is
    # the CH-3 aggregate, not per-sub-stratum)
    def pooled_acc(bfn) -> float:
        return sum(int(lib.correct_set_ok(c, bfn(c.members))) for c in ch3) \
            / len(ch3) if ch3 else 0.0

    acc_baseline = max(pooled_acc(bfn) for bfn in lib.BASELINES.values())

    # g_inv: cosmetic invariance on CH-5 (δ_inv := 2·floor_inv; floor 0.0)
    inv_pairs = inv_ok = 0
    for c in [c for c in clusters if c.stratum == "CH-5"]:
        base = arbitrate_production(c, model, tau_hi, tau_m, ctx=ctx)
        pert = lib.Cluster(
            cid=c.cid, stratum=c.stratum, substratum=c.substratum,
            members=[lib.Member(m.mid, m.title,
                                lib.cosmetic_variant(m.body, random.Random(7)),
                                m.tags, m.record_type, m.language, m.created_at)
                     for m in c.members],
            label=c.label, declaration=c.declaration)
        alt = arbitrate_production(pert, model, tau_hi, tau_m, ctx=ctx)
        inv_pairs += 1
        inv_ok += int(base["verdict"] == alt["verdict"]
                      and base.get("member") == alt.get("member"))
    inv_rate = inv_ok / inv_pairs if inv_pairs else 1.0

    prop24 = sum(1 for c in ch24
                 if arbitrate_production(c, model, tau_hi, tau_m,
                                         ctx=ctx)["verdict"] == "keep")

    return {
        "g_acc": {"ok": bool((acc_model - acc_baseline) >= delta_acc),
                  "acc_model": round(acc_model, 4),
                  "acc_best_baseline": round(acc_baseline, 4),
                  "delta": round(acc_model - acc_baseline, 4),
                  "delta_acc": round(delta_acc, 4),
                  "floor_seed_aggregate": floor_seed_aggregate,
                  "source": floor_source},
        "g_fold_exact": {"ok": prop24 == 0, "proposals": prop24,
                         "threshold": "exact 0"},
        "g_abstain_sens": {"ok": bool(abst_rate3 <= q),
                           "abstain_rate": round(abst_rate3, 4), "q": q,
                           "zone_fired_ch3": zone3},
        "g_inv": {"ok": bool(inv_rate >= 1.0),
                  "agreement": round(inv_rate, 4), "pairs": inv_pairs,
                  "delta_inv": 0.0},
    }


# --- CLI ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train-only CV, freeze artifact, sanity")
    p_train.add_argument("--run-tag", default="attempt1")
    p_train.add_argument("--artifact-name", default="merge_arbiter_v1.json")
    p_train.add_argument("--doc-out", default=str(COMMITTED_DOC))
    p_train.set_defaults(fn=cmd_train)
    p_train2 = sub.add_parser(
        "train-v2",
        help="Choice-v2 retrain (ADDENDUM-6): 34-feature contract + zone-v1 "
             "contour + CV riders; train-only, no sealed read")
    p_train2.add_argument("--run-tag", default="v2-attempt1")
    p_train2.add_argument("--artifact-name", default="merge_arbiter_v2.json")
    p_train2.add_argument("--doc-out",
                          default=str(COMMITTED_DIR /
                                      "merge-v1-thresholds.frozen-choice-v2.json"))
    p_train2.set_defaults(fn=cmd_train_v2)
    p_floors = sub.add_parser("floors-composition",
                              help="floor_q / floor_inv / floor_perm (§3.3)")
    p_floors.add_argument("--artifact", required=True)
    p_floors.add_argument("--doc-out", required=True)
    p_floors.add_argument("--q-seeds", default="63-72")
    p_floors.add_argument("--inv-seeds", default="73-75")
    p_floors.add_argument("--perm-seeds", default="76-78")
    p_floors.set_defaults(fn=cmd_floors_composition)
    p_floors2 = sub.add_parser(
        "floors-v2",
        help="Choice-v2 floors per ML-STD-0004 (stage 2): floor_seed "
             "distribution + declared delta_acc + floor_q/inv/perm in one "
             "delta_acc_measure_v2 section")
    p_floors2.add_argument("--artifact", required=True)
    p_floors2.add_argument("--doc-out", required=True)
    p_floors2.add_argument("--seeds", default="43-52",
                           help="floor_seed composition draws (>= 10, П3)")
    p_floors2.add_argument("--q-seeds", default="63-72")
    p_floors2.add_argument("--inv-seeds", default="73-75")
    p_floors2.add_argument("--perm-seeds", default="76-78")
    p_floors2.set_defaults(fn=cmd_floors_v2)
    p_pre = sub.add_parser("prelaunch",
                           help="ADDENDUM-3 §3.6 checklist (no sealed read)")
    p_pre.add_argument("--artifact", required=True)
    p_pre.add_argument("--profile", default="v1", choices=["v1", "v2"],
                       help="v2 = Choice-v2 checklist (zone contour, ML-STD-0004 "
                            "prereg, zone-off probe, sheet prematch)")
    p_pre.set_defaults(fn=cmd_prelaunch)

    p_san = sub.add_parser("sanity", help="7-probe suite on the frozen artifact")
    p_san.add_argument("--artifact", required=True)
    p_san.set_defaults(fn=cmd_sanity)

    p_floor = sub.add_parser(
        "floor-seed",
        help="ratified delta_acc floor measurement (MLC 29.10, option a)")
    p_floor.add_argument("--artifact", required=True,
                         help="frozen merge-v1 artifact (model NOT retrained)")
    p_floor.add_argument("--seeds", default="43-52",
                         help="inclusive range or comma list, e.g. 53-62")
    p_floor.add_argument("--doc-out", default=str(COMMITTED_DOC))
    p_floor.set_defaults(fn=cmd_floor_seed)

    p_gate = sub.add_parser("gate", help="single-shot gate cycle (sealed read)")
    p_gate.add_argument("--artifact", required=True)
    p_gate.add_argument("--noharm", required=True,
                        help="committed NOT-dup v42/v4.3 surface (jsonl), not a seal")
    p_gate.add_argument("--execute-single-shot", action="store_true",
                        help="EXPLICIT flag required for any sealed read")
    p_gate.set_defaults(fn=cmd_gate)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except InstrumentAbort as exc:
        log("abort_instrument", error=str(exc),
            note="STOP class instrument; sealed surfaces not read")
        print(f"INSTRUMENT ABORT: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
