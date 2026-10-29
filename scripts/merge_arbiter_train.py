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
                  if EXPECTED_TRAIN_COUNTS.get((s, k)) != v}
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
            if c.stratum == "theme-near":
                neg_counts[c.substratum] = neg_counts.get(c.substratum, 0) + 1
        if neg_counts != EXPECTED_NEG_COUNTS:
            count_mism["theme-near"] = {"got": neg_counts, "want": EXPECTED_NEG_COUNTS}
    log("frozen_inputs",
        train_jsonl_sha256=sha256_file(TRAIN_JSONL),
        manifest_sha256=sha256_file(MANIFEST),
        negatives_sha256=sha256_file(NEG_PATH) if NEG_PATH.exists() else None,
        corpus_fp_v1_1=(neg_manifest.get("corpus_fp_v1_1")
                        if NEG_PATH.exists() else None),
        n_clusters=len(clusters),
        counts={f"{s}/k{k}": n for (s, k), n in sorted(counts.items())},
        theme_near_counts=neg_counts or None,
        mismatch=count_mism, missing=missing,
        sealed_inputs=["NONE — MA-holdout-choice, holdout v4.5, g4-spec-v1, "
                       "spot-sheet are NOT in the train input list"])
    if count_mism or missing:
        raise InstrumentAbort(f"frozen_inputs count mismatch: {count_mism} missing={missing}")
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
    when present — the extension, never a re-emit: train.jsonl stays
    byte-identical (its sealed pin is enforced by the byte tripwire).

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
    return clusters


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


def cluster_features(members: list[lib.Member]) -> np.ndarray:
    """(k, 25) member feature matrix — permutation-invariant by construction.

    Every row is a function of (the member SET, that member) only. Per-other
    terms are accumulated in SORTED-mid order of the others, so a member's
    row is BIT-identical under any presentation permutation (no float
    reassociation drift). No feature reads a serialization index (the
    'first' baseline carries no signal — A-order-indep at feature level).
    """
    k = len(members)
    order = sorted(range(k), key=lambda p: members[p].mid)
    epochs = [0.0] * k
    nulls = [0.0] * k
    for p in range(k):
        epochs[p], nulls[p] = _epoch(members[p].created_at)
    rows = np.zeros((k, len(MEMBER_FEATURE_NAMES)), dtype=np.float64)
    for i in range(k):
        acc = np.zeros(len(MEMBER_FEATURE_NAMES), dtype=np.float64)
        for j in order:
            if j == i:
                continue
            acc[:N_PAIR_TERMS] += _pair_member_terms(members[i], members[j])
            acc[IDX_DELTA] += epochs[i] - epochs[j]
        n_others = k - 1
        acc[:N_PAIR_TERMS] /= n_others
        acc[IDX_DELTA] /= n_others
        acc[IDX_EPOCH] = epochs[i]
        acc[IDX_NULL] = nulls[i]
        rows[i] = acc
    return rows


def assert_permutation_invariance(clusters: list[lib.Cluster], n_check: int = 40) -> None:
    """Machine check: reversing the member order changes no member's row."""
    rng = random.Random(SEED)
    sample = rng.sample(clusters, min(n_check, len(clusters)))
    for c in sample:
        x = cluster_features(c.members)
        rev = list(reversed(c.members))
        xr = cluster_features(rev)
        idx = {m.mid: p for p, m in enumerate(c.members)}
        idxr = {m.mid: p for p, m in enumerate(rev)}
        for mid, p in idx.items():
            if not np.array_equal(x[p], xr[idxr[mid]]):
                raise InstrumentAbort(
                    f"permutation invariance RED on {c.cid}/{mid}: features "
                    "depend on serialization order")
    log("feature_permutation_invariance", checked=len(sample), ok=True)


# --- member-level training table ---------------------------------------------

def member_rows(clusters: list[lib.Cluster]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build the member-level table (see module docstring for the labeling).

    CH-3 k2: positive = the keep member (superset), negative = the other.
    CH-3 chain: positive = the unique ⊆-maximum, negatives = the rest.
    CH-2/CH-4: ALL rows negative — the abstain leg. CH-0/CH-5/real: not
    in train (code legs / gate-only strata).
    """
    xs, ys, meta = [], [], []
    for c in clusters:
        if c.stratum in ("CH-0", "CH-5"):
            continue
        x = cluster_features(c.members)
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


def cv_thresholds(clusters: list[lib.Cluster]) -> dict:
    """Fit (tau_hi, tau_m) by out-of-fold cluster decisions on train only.

    Selection metric (brief/§2.2): correct-set acc + abstain rate on CH-3,
    jointly: score = acc − abstain; ties: acc desc, tau_hi desc, tau_m
    desc (more conservative wins ties). NO sealed surface participates in
    any choice here (N2).
    """
    # folds over the TRAIN-participating clusters only (CH-0/CH-5 are code
    # legs — they carry no member rows and no threshold decision)
    train_clusters = [c for c in clusters if c.stratum not in ("CH-0", "CH-5")]
    xs, ys, meta = member_rows(train_clusters)
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

    # CV-neg-abstain (ADDENDUM-3 §2): per-substratum OOF abstain >= 0.9
    neg_by_sub: dict[str, list[int]] = {}
    for cid, c in by_cid.items():
        if c.stratum == "theme-near":
            neg_by_sub.setdefault(c.substratum, []).append(cid)
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
        tolerance=NOREGRESS_TOL, floor=round(1.0 - NOREGRESS_TOL, 4), red=noreg_red)
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

def baselines_per_stratum(clusters: list[lib.Cluster]) -> dict:
    table: dict[str, dict[str, list[int]]] = {}
    for c in clusters:
        key = stratum_key(c)
        row = table.setdefault(key, {b: [0, 0] for b in lib.BASELINES})
        for bname, bfn in lib.BASELINES.items():
            row[bname][1] += 1
            row[bname][0] += int(lib.correct_set_ok(c, bfn(c.members)))
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
    for seed in FLOOR_SEEDS:
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
                         tau_hi: float, tau_m: float) -> dict:
    """Through pipeline: classify (NO declarations — production has none)
    -> CH-0/CH-5 tie-break -> CH-2 refusal -> model leg for the rest."""
    cls = lib.classify(cluster.members, declarations=None)
    if cls in ("CH-0", "CH-5"):
        keep = min(cluster.members,
                   key=lambda m: (_epoch(m.created_at)[0], m.mid))
        return {"verdict": "keep", "member": keep.mid, "leg": "tie-break",
                "class": cls}
    if cls == "CH-2":
        return {"verdict": "abstain", "leg": "refusal", "class": cls}
    x = cluster_features(cluster.members)
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
                           log_it: bool = True) -> list[dict]:
    """7 probes through the production contour with the FROZEN model.

    Every probe is additionally executed with the REVERSED member order:
    the decision must be identical (pipeline-level order-invariance).
    Red = STOP class 'instrument' — holdout is not read (§6.1).
    """
    results = []
    all_ok = True
    for cell in lib.build_fixture_cells():
        res = arbitrate_production(cell["cluster"], booster, tau_hi, tau_m)
        rev = list(reversed(cell["cluster"].members))
        res_rev = arbitrate_production(
            lib.Cluster(cid=cell["cluster"].cid, stratum=cell["cluster"].stratum,
                        substratum=cell["cluster"].substratum, members=rev,
                        label=cell["cluster"].label,
                        declaration=cell["cluster"].declaration),
            booster, tau_hi, tau_m)
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

    # train-side baselines + exact reproduction of the build characterization
    baselines = baselines_per_stratum(clusters)
    check_baselines_vs_manifest(pro["manifest"], baselines)

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


# --- subcommands: floors-composition, prelaunch -------------------------------

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


def cmd_prelaunch(args: argparse.Namespace) -> int:
    """ADDENDUM-3 §3.6 pre-launch checklist on the new sha (train-side; NO
    sealed read): committee-scored ∉ inputs, sanity 7/7, noharm QA
    (expect 0/171 + 0/27), sheet pre-check (member_ids ⊆ synthetic_train,
    id level), floors completeness."""
    global _SLICE
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


def _noharm_proposals(noharm: Path, model: lgb.Booster,
                      tau_hi: float, tau_m: float) -> tuple[int, int]:
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
            a = lib.Member(mid=r.get("id_a", "a"), title=r["title_a"],
                           body=r["body_a"], tags=tuple(r.get("tags_a", ())),
                           record_type=r.get("type_a"), language=r.get("lang_a"),
                           created_at=r.get("created_a"))
            b = lib.Member(mid=r.get("id_b", "b"), title=r["title_b"],
                           body=r["body_b"], tags=tuple(r.get("tags_b", ())),
                           record_type=r.get("type_b"), language=r.get("lang_b"),
                           created_at=r.get("created_b"))
            cluster = lib.Cluster(cid=r.get("pair_id", f"noharm-{rows}"),
                                  stratum="NOT-DUP", substratum="v42",
                                  members=[a, b], label={"type": "abstain"},
                                  declaration={})
            d = arbitrate_production(cluster, model, tau_hi, tau_m)
            rows += 1
            proposals += int(d["verdict"] == "keep")
    return rows, proposals


def cmd_gate(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_GATE
    artifact = Path(args.artifact)
    noharm = Path(args.noharm)

    if prior_single_shot(SLICE_GATE):
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

    # П1: gate_start BEFORE any eval computation (sanity suite, fixtures,
    # no-harm, sealed reads are all eval reads — they start only now).
    log("gate_start",
        note="full prologue green above; every eval read starts after this event")

    # pre-sealed: sanity §6.1 with the frozen model (red => abort, no holdout)
    run_model_sanity_suite(model, tau_hi, tau_m)

    # g_corner leg 1 (identity corner) — fixture-based, no sealed read
    cells = {cell["probe"]: cell for cell in lib.build_fixture_cells()}
    corner = arbitrate_production(cells[1]["cluster"], model, tau_hi, tau_m)
    corner_ok = (corner["verdict"] == "keep"
                 and corner.get("member") in
                 {m.mid for m in cells[1]["cluster"].members})
    log("g_corner_leg1", ok=corner_ok, result=corner, threshold="exact")

    rows, proposals = _noharm_proposals(noharm, model, tau_hi, tau_m)
    g_noharm_ok = proposals == 0
    log("g_noharm_v42", rows=rows, proposals=proposals, ok=g_noharm_ok,
        threshold="exact 0", surface_sha256=sha256_file(noharm))

    if not (corner_ok and g_noharm_ok):
        log("abort_pregate_red", corner_ok=corner_ok, noharm_ok=g_noharm_ok,
            note="pre-gate red -> holdout NOT read (ML-STD-0002 П6)")
        return 3

    # ============ SEALED READ — SINGLE-SHOT (not executed in this phase) =====
    holdout = _read_holdout_sealed()          # the single legal read
    log("sealed_read_once", n_clusters=holdout["n"], sha256=holdout["sha256"])
    gates = _eval_gates_on_holdout(model, tau_hi, tau_m, holdout, pro["manifest"], doc)
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
                           holdout: dict, manifest: dict, doc: dict) -> dict:
    """Gate formulas §5 — thresholds are FROZEN constants, never revisited.

    δ_acc comes from the RATIFIED measurement (MLC 29.10, option (a)):
    doc["delta_acc_measure"] — floor_seed at the granularity of the gating
    statistic (CH-3 aggregate over the gate-leg composition), frozen in
    the committed doc BEFORE gate_start; missing section = instrument
    abort (the gate never runs with an unfixed δ_acc). The per-stratum
    floor table of the build manifest does NOT enter the formula.
    """
    measure = doc.get("delta_acc_measure")
    if not measure or "delta_acc" not in measure or "floor_seed_aggregate" not in measure:
        raise InstrumentAbort(
            "committed doc has no ratified delta_acc_measure — δ_acc is "
            "unfixed; run the floor-seed measurement BEFORE the gate cycle")
    delta_acc = float(measure["delta_acc"])
    floor_seed_aggregate = float(measure["floor_seed_aggregate"])

    clusters: list[lib.Cluster] = holdout["clusters"]
    ch3 = [c for c in clusters if c.stratum == "CH-3"]
    ch24 = [c for c in clusters if c.stratum in ("CH-2", "CH-4")]

    d3 = {c.cid: arbitrate_production(c, model, tau_hi, tau_m) for c in ch3}
    hits = sum(1 for c in ch3 if d3[c.cid]["verdict"] == "keep"
               and d3[c.cid].get("member") == c.label["member"])
    abst3 = sum(1 for c in ch3 if d3[c.cid]["verdict"] == "abstain")
    acc_model = hits / len(ch3) if ch3 else 0.0
    abst_rate3 = abst3 / len(ch3) if ch3 else 0.0

    # baselines on the SAME CH-3 material, pooled (the g_acc statistic is
    # the CH-3 aggregate, not per-sub-stratum)
    def pooled_acc(bfn) -> float:
        return sum(int(lib.correct_set_ok(c, bfn(c.members))) for c in ch3) \
            / len(ch3) if ch3 else 0.0

    acc_baseline = max(pooled_acc(bfn) for bfn in lib.BASELINES.values())

    # g_abstain_sens: q := max(0.20; p90_char + 2·floor_q); the model-side
    # p90_char/floor_q draws are a gate-time mechanism (N2/N4); v1 carries
    # the ratified 0.20 floor until those draws are recorded.
    q = 0.20

    # g_inv: cosmetic invariance on CH-5 (δ_inv := 2·floor_inv; floor 0.0)
    inv_pairs = inv_ok = 0
    for c in [c for c in clusters if c.stratum == "CH-5"]:
        base = arbitrate_production(c, model, tau_hi, tau_m)
        pert = lib.Cluster(
            cid=c.cid, stratum=c.stratum, substratum=c.substratum,
            members=[lib.Member(m.mid, m.title,
                                lib.cosmetic_variant(m.body, random.Random(7)),
                                m.tags, m.record_type, m.language, m.created_at)
                     for m in c.members],
            label=c.label, declaration=c.declaration)
        alt = arbitrate_production(pert, model, tau_hi, tau_m)
        inv_pairs += 1
        inv_ok += int(base["verdict"] == alt["verdict"]
                      and base.get("member") == alt.get("member"))
    inv_rate = inv_ok / inv_pairs if inv_pairs else 1.0

    prop24 = sum(1 for c in ch24
                 if arbitrate_production(c, model, tau_hi, tau_m)["verdict"] == "keep")

    return {
        "g_acc": {"ok": bool((acc_model - acc_baseline) >= delta_acc),
                  "acc_model": round(acc_model, 4),
                  "acc_best_baseline": round(acc_baseline, 4),
                  "delta": round(acc_model - acc_baseline, 4),
                  "delta_acc": round(delta_acc, 4),
                  "floor_seed_aggregate": floor_seed_aggregate,
                  "source": "ratified delta_acc_measure (MLC 29.10, option a), "
                            "fixed before gate_start"},
        "g_fold_exact": {"ok": prop24 == 0, "proposals": prop24,
                         "threshold": "exact 0"},
        "g_abstain_sens": {"ok": bool(abst_rate3 <= q),
                           "abstain_rate": round(abst_rate3, 4), "q": q},
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
    p_floors = sub.add_parser("floors-composition",
                              help="floor_q / floor_inv / floor_perm (§3.3)")
    p_floors.add_argument("--artifact", required=True)
    p_floors.add_argument("--doc-out", required=True)
    p_floors.add_argument("--q-seeds", default="63-72")
    p_floors.add_argument("--inv-seeds", default="73-75")
    p_floors.add_argument("--perm-seeds", default="76-78")
    p_floors.set_defaults(fn=cmd_floors_composition)
    p_pre = sub.add_parser("prelaunch",
                           help="ADDENDUM-3 §3.6 checklist (no sealed read)")
    p_pre.add_argument("--artifact", required=True)
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
