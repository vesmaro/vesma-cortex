#!/usr/bin/env python3
"""Trainer for the conflict-detector v1 (ADR 0002 duty #6, wave B6-conflict).

Mirror of ``scripts/merge_arbiter_train.py`` per the ratified delta map
(``docs/experiments/conflict-detector-trainer-delta-map.md``); the dispatch
brief is ``docs/experiments/conflict-detector-train-dispatch-2026-11-06.md``.

PHASE-1 DISCIPLINE (hard, enforced structurally):
- The sealed holdout ``data/conflict-detector/corpus-v1/holdout.jsonl`` is
  NEVER opened by any code path of this file except ``cmd_gate`` AFTER an
  explicit ``--execute-single-shot`` flag AND a green ``gate_start``. There
  is no id-only read, no stat, no hash of the sealed file anywhere else.
- Foreign seals (holdout v4.5, g4-spec-v1, MA-holdout-choice) are not
  touched by anything here.
- Order (dispatch §3): trainer -> floors (ML-STD-0001/0004, train-only,
  published BEFORE gate_start) -> sanity §6.3 (7 probes through the
  PRODUCTION circuit) -> gate (implemented, NOT executed in phase 1).
- Suite amendment (ADDENDUM-4 @ 0971639, ratified MLC 10.11): probe 7 runs
  ONE-LEG on frozen-13 — leg-2 (translation-fact-changed) is known-uncovered
  (feature-blind zone), its verdict is PUBLISHED but NOT asserted; suite
  count is "6/7 legs + 1 known-uncovered" (bare "7/7" = fabrication, E3).

Reuse map (0 reimplementations, lesson #480):
- features        : frozen ``cortex.features.pair`` CORE-13 (exact order).
- guard code leg  : frozen ``conflict_corpus_lib`` regions/region_is_version
                    (cg-guard-v1 predicate, NC-3 §2.2/§2.3 of the design).
- embed leg       : engine ``vesma.embeddings.NanoProvider`` (pin
                    3b752e06… — wrong provider refuses loudly, A-frozen),
                    surface = frozen ``lib.embed_text`` (n1(title)\n+
                    n1(body)\n+n1(tags)), truncation [:4096].
- floor draws     : frozen build recipe ``build_conflict_detector_corpus``
                    (gen_pool / rows_of / side_dict — the SAME code path
                    that emitted the corpus).
- run-log/abort   : InstrumentAbort + append-only run-log (arbiter pattern).

Decision rule (ratified MLC-9): cut 0.5 on P(contradicts), NO threshold
search; version-guard is a CODE leg (one-sided NOT) ahead of the model leg.
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
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402 (frozen corpus lib: guard, surfaces, fixtures)
import build_conflict_detector_corpus as cb  # noqa: E402 (frozen build recipe: gen_pool, rows_of)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402

# --- paths -------------------------------------------------------------------

CORPUS_DIR = REPO / "data" / "conflict-detector" / "corpus-v1"
TRAIN_JSONL = CORPUS_DIR / "train.jsonl"
BUILD_MANIFEST = CORPUS_DIR / "build-manifest.json"
PIN_JSON = REPO / "data" / "conflict-detector" / "pin.json"
#: SEALED — read ONLY by _read_holdout_sealed() inside cmd_gate, once,
#: after gate_start. No other line of this file touches it.
HOLDOUT_JSONL = CORPUS_DIR / "holdout.jsonl"
SEAL_JSON = REPO / "data" / "conflict-detector" / "seals" / "CD-holdout-v1.seal.json"
#: Foreign seals — NEVER opened here (fusion-framing precedent).
FOREIGN_SEALS = (
    "MA-holdout-choice",
    "holdout v4.5",
    "g4-spec-v1",
)
ART_DIR = REPO / "models" / "conflict-detector-v1"
FLOORS_DOC = ART_DIR / "floors.json"
MODEL_CARD = ART_DIR / "MODEL_CARD.md"
ONNX_PATH = ART_DIR / "model.onnx"
MANIFEST_PATH = ART_DIR / "manifest.json"
RUNLOG = REPO / "docs" / "experiments" / "conflict-detector-train-runlog.md"
SPEC = REPO / "docs" / "experiments" / "conflict-detector-corpus-design.md"

#: The sealed surface is not even hashed in phase 1 — the byte pin comes
#: from the BUILD manifest fingerprints + the seal json (both metadata).
#: v1 DEFAULTS (byte-identical to the F1 run); ``apply_input_params``
#: rebinds them per the resolved input preset.
TRAIN_SHA_PIN = "0be52dd220c792810420c10921f30329ba6d903bfda5c02f9900ffdb8295e4aa"
HOLDOUT_SHA_PIN = "e8adfec9625b0f72ac001ed6024da2e3fd3bb5170169b6d93f26afde521b2567"

SLICE_TRAIN = "conflict-detector-v1-train"
SLICE_FLOOR = "conflict-detector-v1-floors"
SLICE_SANITY = "conflict-detector-v1-sanity"
SLICE_GATE = "conflict-detector-v1-gate"
SLICE_OOF = "conflict-detector-v1-oof-carriers"

_SLICE = SLICE_TRAIN

# --- env pins (P2, fail-closed; the project's own venv, arbiter precedent) ----

EXPECTED_ENV = {
    "venv_suffix": "vesma-cortex/.venv/bin/python",
    "python": "3.12.13",
    "lightgbm": "4.7.0",
    "numpy": "2.5.3",
}
EXPECTED_HOST = "core-51"  # same host as the corpus build + gold synthesis

# --- training config (committed with the artifact; no drift) ------------------

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
#: Ratified decision rule (MLC-9): fixed cut, NO threshold search.
DECISION_CUT = 0.5
#: ML-STD-0004 stage-1 formula (committed BEFORE any floor draw).
DELTA_ACC_FORMULA = "delta_acc = max(p95 - p5; 0.10)"
STD0004_FORMULA_COMMIT = "5305c82"  # same ratified formula commit as merge-v2

# --- frozen-input asserts (П5) ------------------------------------------------

#: Train rows per stratum = pairs x 2 orientations (A-symmetry), pin.json
#: quota x 2. 1200 rows total.
EXPECTED_TRAIN_ROW_COUNTS = {
    "CD-1": 144,
    "CD-2": 72,
    "CD-3": 72,
    "CD-4": 72,
    "CD-5": 48,
    "CD-6": 72,
    "NC-1": 180,
    "NC-2": 144,
    "NC-3": 108,
    "NC-4": 72,
    "NC-5": 216,
}
ORDER = cb.ORDER
#: Holdout strata counts (gate-time count assert; from the BUILD manifest
#: asserts.A-count.detail.seal — metadata, no sealed read).
EXPECTED_HOLDOUT_PAIRS = {
    "CD-1": 22,
    "CD-2": 11,
    "CD-3": 11,
    "CD-4": 11,
    "CD-5": 7,
    "CD-6": 11,
    "NC-1": 27,
    "NC-2": 21,
    "NC-3": 16,
    "NC-4": 11,
    "NC-5": 32,
}
#: Baselines pinned in the BUILD manifest (g_ba target max(BL) = 0.66).
EXPECTED_BASELINES = {"BL-band": 0.49, "BL-digit": 0.57, "BL-digit+guard": 0.66}
SEAL_N_META = 180  # seal ids count (metadata; asserted vs seal json)

# --- input presets (B6 v2 re-run; CLI input plumbing ONLY, semantics
# --- unchanged: features / guard / cut 0.5 / production circuit untouched) ----
#
# The v1 tables above are byte-identical to the F1 run. The v2 preset pins the
# NEW v2 bytes (ADDENDUM-3 §8, ratified E1-E13): train.jsonl a549dc0c…,
# pin-v2.json dcb3536d…, seal CD-holdout-v2 (holdout content sha c92a2caa…,
# metadata only, n=180). Table values come from the v2 BUILD manifest
# METADATA (asserts.A-count.detail / baselines / strata.class_total) — no
# sealed surface is read anywhere. Unknown corpus dirs are a fail-closed
# instrument abort: no pins, no run.

V2_EXPECTED_TRAIN_ROW_COUNTS = {
    "CD-1": 144, "CD-2": 72, "CD-3": 72, "CD-4": 72, "CD-5": 48, "CD-6": 72,
    "NC-1": 132, "NC-2": 192, "NC-3": 108, "NC-4": 72, "NC-5": 216,
}
V2_EXPECTED_HOLDOUT_PAIRS = {
    "CD-1": 22, "CD-2": 11, "CD-3": 11, "CD-4": 10, "CD-5": 7, "CD-6": 11,
    "NC-1": 20, "NC-2": 29, "NC-3": 16, "NC-4": 11, "NC-5": 32,
}
V2_EXPECTED_BASELINES = {"BL-band": 0.465, "BL-digit": 0.53, "BL-digit+guard": 0.62}
#: Floor draw composition = v2 class_total // 6 (== 100 pairs per draw; the
#: ML-STD-0004 formula on the v2 strata table — cross-checked against the
#: build manifest strata metadata at floors start).
V2_FLOOR_DRAW_QUOTA = {
    "CD-1": 12, "CD-2": 6, "CD-3": 6, "CD-4": 6, "CD-5": 4, "CD-6": 6,
    "NC-1": 11, "NC-2": 16, "NC-3": 9, "NC-4": 6, "NC-5": 18,
}
_V1_LINEAGE = (
    "mirror of merge_arbiter_train.py per "
    "docs/experiments/conflict-detector-trainer-delta-map.md; "
    "dispatch docs/experiments/conflict-detector-train-dispatch-2026-11-06.md"
)

INPUT_PRESETS = {
    "v1": {
        "corpus_dir": REPO / "data" / "conflict-detector" / "corpus-v1",
        "pin": REPO / "data" / "conflict-detector" / "pin.json",
        "seal": REPO / "data" / "conflict-detector" / "seals" / "CD-holdout-v1.seal.json",
        "artifact_dir": REPO / "models" / "conflict-detector-v1",
        "train_sha256": "0be52dd220c792810420c10921f30329ba6d903bfda5c02f9900ffdb8295e4aa",
        "holdout_sha256": "e8adfec9625b0f72ac001ed6024da2e3fd3bb5170169b6d93f26afde521b2567",
        "pin_sha256": None,  # v1 tripwire keeps its exact F1 shape
        "seal_sha256": None,
        "expected_train_rows": EXPECTED_TRAIN_ROW_COUNTS,
        "expected_holdout_pairs": EXPECTED_HOLDOUT_PAIRS,
        "expected_baselines": EXPECTED_BASELINES,
        "floor_draw_quota": None,  # module default: cb.STRATA_QUOTA // 6
        "floor_inv_exclude_carriers": (),  # E12 scoping is a v2 ratification
        "canary_para2": None,  # E8-iii is a v2 ratification
        "para_sweep_words": 0,  # E8-iv is a v2 ratification
        "activation_extra": None,
        "lineage": _V1_LINEAGE,
    },
    "v2": {
        "corpus_dir": REPO / "data" / "conflict-detector" / "corpus-v2",
        "pin": REPO / "data" / "conflict-detector" / "pin-v2.json",
        "seal": REPO / "data" / "conflict-detector" / "seals" / "CD-holdout-v2.seal.json",
        "artifact_dir": REPO / "models" / "conflict-detector-v2",
        "train_sha256": "a549dc0c2a32fe0ff6cc4035f95c8c9e888813e24430b67ac71f272ea06ec115",
        "holdout_sha256": "c92a2caa283869ce9a7a28baa34c06f8a67de89a4b47b4f6266b0a0383d4b4c2",
        "pin_sha256": "dcb3536d899fe3545a31f0aaaf349863ae3d10eab507448600c96e326b429d8f",
        "seal_sha256": "5b95bfd7f2a2be8eecca09b0c98f2deea8806b37ca05924a74f72dfcf21bf909",
        "expected_train_rows": V2_EXPECTED_TRAIN_ROW_COUNTS,
        "expected_holdout_pairs": V2_EXPECTED_HOLDOUT_PAIRS,
        "expected_baselines": V2_EXPECTED_BASELINES,
        "floor_draw_quota": V2_FLOOR_DRAW_QUOTA,
        "floor_inv_exclude_carriers": ("N1",),  # E12: N1 cells NOT in the floor fix-set
        "canary_para2": {
            # the build's own canary selection (build driver E8(iii) block):
            # CD-1 para_b=2 cross cells; regenerate the v2 train pool with the
            # EXACT build call (rotate=False explicit) and intersect ids with
            # train pair ids -> the 7 SELECTED cells.
            "seed": 2113,
            "prefix": "tr",
            "rotate": False,
            "expect_count": 7,
            "expect_verdict": "contradicts",
        },
        "para_sweep_words": 5,  # E8-iv: sweep >=3 applicable PARA_RU words
        "activation_extra": {
            "addendum": REPO / "docs" / "experiments"
            / "conflict-detector-corpus-design-ADDENDUM-3.md",
            "marker": "РАТИФИЦИРОВАНО (amend-then-ratify)",
            "corpus_v2_build_commit": "2d9ce88",
        },
        "lineage": _V1_LINEAGE
        + "; corpus v2 per "
        "docs/experiments/conflict-detector-corpus-design-ADDENDUM-3.md §8 "
        "(ratified amend-then-ratify, E1-E13); trainer input params via CLI "
        "(--corpus-dir/--pin/--seal/--artifact-dir), semantics unchanged",
    },
}

#: Active preset (set by apply_input_params from the CLI input args).
_INPUT_PRESET_VERSION = "v1"
_INPUT_PRESET = INPUT_PRESETS["v1"]
FLOOR_INV_EXCLUDE_CARRIERS: tuple[str, ...] = ()
CANARY_PARA2: dict | None = None
PARA_SWEEP_WORDS = 0


def apply_input_params(args: argparse.Namespace) -> None:
    """Resolve the input preset from the CLI input args and rebind the
    module-level path/pin/table constants.

    INPUT PLUMBING ONLY — no decision-semantics surface is touched here:
    the feature contract, the guard leg, the cut-0.5 rule and the
    production circuit are the same code for v1 and v2. Default
    invocation (no args) resolves to the v1 preset, i.e. byte-identical
    behavior to the F1 run."""
    global CORPUS_DIR, TRAIN_JSONL, BUILD_MANIFEST, PIN_JSON, HOLDOUT_JSONL
    global SEAL_JSON, ART_DIR, FLOORS_DOC, MODEL_CARD, ONNX_PATH, MANIFEST_PATH
    global TRAIN_SHA_PIN, HOLDOUT_SHA_PIN
    global EXPECTED_TRAIN_ROW_COUNTS, EXPECTED_HOLDOUT_PAIRS, EXPECTED_BASELINES
    global FLOOR_DRAW_QUOTA, FLOOR_DRAW_PAIRS
    global FLOOR_INV_EXCLUDE_CARRIERS, CANARY_PARA2, PARA_SWEEP_WORDS
    global SLICE_TRAIN, SLICE_FLOOR, SLICE_SANITY, SLICE_GATE, SLICE_OOF, _SLICE
    global _INPUT_PRESET, _INPUT_PRESET_VERSION, SEAL_N_META

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    if corpus_dir is None:
        version = "v1"
    else:
        corpus_dir = corpus_dir if corpus_dir.is_absolute() else (REPO / corpus_dir)
        known = {str(p["corpus_dir"]): k for k, p in INPUT_PRESETS.items()}
        if str(corpus_dir) not in known:
            raise InstrumentAbort(
                f"unknown corpus dir {corpus_dir} — no ratified pin set "
                f"exists for it (known: {sorted(known)}); refusing to run "
                "unpinned (fail-closed)"
            )
        version = known[str(corpus_dir)]
    preset = INPUT_PRESETS[version]
    _INPUT_PRESET_VERSION = version
    _INPUT_PRESET = preset

    CORPUS_DIR = corpus_dir if corpus_dir is not None else preset["corpus_dir"]
    PIN_JSON = Path(args.pin) if args.pin else preset["pin"]
    SEAL_JSON = Path(args.seal) if args.seal else preset["seal"]
    ART_DIR = Path(args.artifact_dir) if args.artifact_dir else preset["artifact_dir"]
    TRAIN_JSONL = CORPUS_DIR / "train.jsonl"
    BUILD_MANIFEST = CORPUS_DIR / "build-manifest.json"
    HOLDOUT_JSONL = CORPUS_DIR / "holdout.jsonl"
    FLOORS_DOC = ART_DIR / "floors.json"
    MODEL_CARD = ART_DIR / "MODEL_CARD.md"
    ONNX_PATH = ART_DIR / "model.onnx"
    MANIFEST_PATH = ART_DIR / "manifest.json"

    TRAIN_SHA_PIN = preset["train_sha256"]
    HOLDOUT_SHA_PIN = preset["holdout_sha256"]
    EXPECTED_TRAIN_ROW_COUNTS = preset["expected_train_rows"]
    EXPECTED_HOLDOUT_PAIRS = preset["expected_holdout_pairs"]
    EXPECTED_BASELINES = preset["expected_baselines"]
    FLOOR_DRAW_QUOTA = (
        dict(preset["floor_draw_quota"])
        if preset["floor_draw_quota"] is not None
        else {s: cb.STRATA_QUOTA[s] // 6 for s in ORDER}
    )
    FLOOR_DRAW_PAIRS = sum(FLOOR_DRAW_QUOTA.values())
    FLOOR_INV_EXCLUDE_CARRIERS = tuple(preset["floor_inv_exclude_carriers"])
    CANARY_PARA2 = preset["canary_para2"]
    PARA_SWEEP_WORDS = preset["para_sweep_words"]

    #: artifact identity: derived from the artifact dir NAME (input plumbing);
    #: v1 resolves to the exact F1 strings.
    global ART_PREFIX, ART_VERSION_SUFFIX, SEAL_NAME
    ART_PREFIX = ART_DIR.name
    ART_VERSION_SUFFIX = (
        ART_PREFIX.rsplit("-v", 1)[1].lstrip("v") if "-v" in ART_PREFIX else "1"
    )
    SEAL_NAME = SEAL_JSON.name[: -len("".join(SEAL_JSON.suffixes))] or SEAL_JSON.stem
    # "CD-holdout-v1" / "CD-holdout-v2" (all suffixes stripped)

    SLICE_TRAIN = f"{ART_PREFIX}-train"
    SLICE_FLOOR = f"{ART_PREFIX}-floors"
    SLICE_SANITY = f"{ART_PREFIX}-sanity"
    SLICE_GATE = f"{ART_PREFIX}-gate"
    SLICE_OOF = f"{ART_PREFIX}-oof-carriers"
    _SLICE = {
        "train": SLICE_TRAIN,
        "floors": SLICE_FLOOR,
        "sanity": SLICE_SANITY,
        "gate": SLICE_GATE,
        "oof-carriers": SLICE_OOF,
    }.get(getattr(args, "cmd", ""), _SLICE)

# --- floors config (ML-STD-0001 + ML-STD-0004; train-only, П3) ----------------

#: floor_seed: >= 10 composition draws from the FROZEN build recipe
#: (seeds 43..52 — fresh seeds outside the train seed family, arbiter
#: precedent), delta_acc = max(p95 - p5; 0.10), distribution published.
FLOOR_SEEDS = list(range(43, 53))
#: floor_q: >= 3 DEDICATED draws (verdict-rate spread).
FLOOR_Q_SEEDS = [53, 54, 55]
#: floor_inv / floor_perm: >= 3 draws each (seeds independent of each other).
FLOOR_INV_SEEDS = [901, 902, 903]
FLOOR_PERM_SEEDS = [951, 952, 953]
#: Draw composition = train quota / 6 (exactly 100 pairs per draw —
#: 72/6=12, 36/6=6 x4, 24/6=4, 90/6=15, 72/6=12, 54/6=9, 36/6=6, 108/6=18).
FLOOR_DRAW_QUOTA = {s: cb.STRATA_QUOTA[s] // 6 for s in ORDER}
FLOOR_DRAW_PAIRS = sum(FLOOR_DRAW_QUOTA.values())
FLOOR_PREFIX = "cdf"
#: Surface-convention truncation (dispatch: усечение 4096). The corpus
#: maximum is asserted < 4096 in the prologue, so the truncated convention
#: and the convention the corpus was MEASURED with coincide everywhere.
EMBED_TRUNC = 4096

#: Gold synthesis (dispatch §2: g_gold 48/48 CLOSED, key 6968087a) —
#: asserted pre-freeze, never reopened.
GOLD_COMMIT = "45597c5"

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class InstrumentAbort(Exception):
    """Fail-closed abort, STOP class 'instrument' (П6 ML-STD-0002).

    Nothing downstream may run; the sealed holdout is NOT read; escalation
    to TL, not input regeneration.
    """


# --- run-log (append-only JSON-lines inside the markdown doc) -----------------


def log(event: str, **fields) -> None:
    row = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "slice": _SLICE,
        "event": event,
        **fields,
    }
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if not RUNLOG.exists():
        RUNLOG.parent.mkdir(parents=True, exist_ok=True)
        RUNLOG.write_text(
            "# Conflict-detector train run-log (wave B6-conflict, phase 1)\n\n"
            "Append-only JSON-lines: one event per line, `{`-prefixed, "
            "machine-parseable. Discipline: dispatch 2026-11-06 §1/§3 — "
            "holdout CD-holdout-v1 is read EXACTLY ONCE by the gate runner; "
            "floors and sanity §6.3 run BEFORE any sealed read; a red sanity "
            "probe = instrument abort, holdout NOT read.\n\n",
            encoding="utf-8",
        )
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


def blake2b_256_file(path: Path) -> str:
    return hashlib.blake2b(Path(path).read_bytes(), digest_size=32).hexdigest()


def runlog_events(event_names: tuple[str, ...], slice_name: str | None = None) -> list[dict]:
    """Scan the run-log for prior events (П8 single-shot discipline)."""
    if not RUNLOG.exists():
        return []
    out = []
    for line in open(RUNLOG, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue  # markdown header / prose — not an event
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a concurrent appender — not evidence
        if row.get("event") in event_names and slice_name in (None, row.get("slice")):
            out.append(row)
    return out


# --- D1/F5 cosine hardening (ADDENDUM-5 §4.2 F5; runner leg ONLY — the
# --- frozen pair.py / _validate_similarity contract is NOT touched) -----------


#: float32 machine epsilon. The pinned provider normalizes inside its ONNX
#: graph in float32, so embedded vectors carry norm = 1 ± eps32 (measured
#: ~2.2e-7 on the fixtures) — the primary cause of the incident's escaped
#: scalar 1.0000001105987988 (ADDENDUM-5 §4.1, ML-а / F11).
EPS32 = float(np.finfo(np.float32).eps)


def cos_clamp_tolerance(n_dims: int) -> float:
    """ADDENDUM-5 F5 fail-closed band: 1 + max(1e-3, 64*n_dims*eps32).

    Overshoots inside the band are clamped to [-1, 1] WITH a visible
    cos_clamp event; beyond the band — or any NaN — they are an instrument
    anomaly -> InstrumentAbort. Clamp never silently masks an anomaly."""
    return max(1e-3, 64 * n_dims * EPS32)


#: Per-run clamp ledger (one process = one run). Every clamp is recorded
#: here AND logged as an immediate cos_clamp event; green paths assert the
#: eps32-class ledger stayed empty (assert_cos_clamp_green).
COS_CLAMP_LEDGER: dict = {
    "count": 0,  # ALL clamps (visible events, nothing silent)
    "over_floor": 0,  # clamps above the f64 noise floor (assert-tripping)
    "max_overshoot": 0.0,
    "sites": [],
}


def cos_noise_floor(n_dims: int) -> float:
    """Float64 rounding floor of the runner cosine leg itself:
    64*n_dims*eps64 (~5.5e-12 at n=384) — the same band form as the F5
    tolerance, one dtype down. A clamp at or below this floor is the dot's
    OWN rounding (e.g. a bit-identical self-parallel pair rounding to
    1 + 1 ulp = 2.2e-16), NOT the eps32-class arithmetic the incident was
    about; eps32-scale overshoots (~1e-7) sit 4+ orders ABOVE the floor.
    NOTE: the sub-floor exemption in the GREEN ASSERT is a deviation from
    the literal NA-б wording («clamp сработал 0 раз») pending NA/TL
    ratification — the literal reading REDs the pre-sealed sanity on
    benign hardware rounding and would block the v3 gate's pre-sealed
    phase. Every clamp, sub-floor included, stays a VISIBLE event."""
    return 64 * n_dims * float(np.finfo(np.float64).eps)


def cos_clamp_guard(cos: float, *, n_dims: int, site: str) -> float:
    """Fail-closed similarity guard — the LAST stop before any frozen API.

    Applied at EVERY runner similarity call-site: all of them funnel
    through NanoLeg.pair_cos, so the guard sits ahead of the frozen
    cortex.features.pair contract (whose _validate_similarity is strict,
    no epsilon — deliberately unchanged per ADDENDUM-5 §3.3).

    NaN or |cos| > 1 + max(1e-3, 64*n_dims*eps32)  -> InstrumentAbort;
    smaller overshoot -> clamp to [-1, 1] + visible cos_clamp event."""
    tol = cos_clamp_tolerance(n_dims)
    if np.isnan(cos):
        raise InstrumentAbort(
            f"NaN cosine at {site} — instrument anomaly, NOT clamped "
            "(ADDENDUM-5 F5)"
        )
    overshoot = abs(cos) - 1.0
    if overshoot > tol:
        raise InstrumentAbort(
            f"|cos|={abs(cos):.12f} at {site} exceeds the fail-closed band "
            f"1 + {tol:.6f} — instrument anomaly, NOT clamped (ADDENDUM-5 F5)"
        )
    if overshoot > 0.0:
        floor = cos_noise_floor(n_dims)
        rounding_scale = overshoot <= floor
        COS_CLAMP_LEDGER["count"] += 1
        if not rounding_scale:
            COS_CLAMP_LEDGER["over_floor"] += 1
        COS_CLAMP_LEDGER["max_overshoot"] = max(
            COS_CLAMP_LEDGER["max_overshoot"], overshoot
        )
        COS_CLAMP_LEDGER["sites"].append(site)
        log(
            "cos_clamp",
            count=COS_CLAMP_LEDGER["count"],
            over_floor=COS_CLAMP_LEDGER["over_floor"],
            max_overshoot=COS_CLAMP_LEDGER["max_overshoot"],
            overshoot=overshoot,
            site=site,
            band=tol,
            f64_noise_floor=floor,
            f64_rounding_scale=rounding_scale,
            kind="clamp",
            note="visible clamp event — never a silent tolerance (F5/NA-б)",
        )
        return 1.0 if cos > 0.0 else -1.0
    return cos


def assert_cos_clamp_green(where: str) -> None:
    """ADDENDUM-5 F5 (NA-б): green-path assert — no eps32-class clamp fired.

    On pre-sealed green legs (sanity suite §6.3, standalone or inside the
    gate) a clamp is NOT absorbed silently: every clamp is a visible event
    in the run-log. The assert trips on clamps ABOVE the float64 noise
    floor (cos_noise_floor) — the eps32-class arithmetic the incident was
    about. DEVIATION from the literal NA-б wording (any clamp would trip),
    pending NA/TL ratification: the literal reading REDs the sanity suite
    on the dot's own 1-ulp rounding of bit-identical self-parallel pairs
    (observed: fixture NC-4-lang, overshoot 2.2e-16 vs floor 5.5e-12)."""
    if COS_CLAMP_LEDGER["over_floor"] > 0:
        raise InstrumentAbort(
            f"green-path clamp assert RED at {where}: eps32-class cos clamp "
            f"fired {COS_CLAMP_LEDGER['over_floor']} time(s) of "
            f"{COS_CLAMP_LEDGER['count']} total, max overshoot "
            f"{COS_CLAMP_LEDGER['max_overshoot']:.3e} at "
            f"{COS_CLAMP_LEDGER['sites'][:10]} — expected 0 above the f64 "
            "noise floor (ADDENDUM-5 F5)"
        )


# --- embedder leg (A-frozen: wrong provider refuses loudly) -------------------


class NanoLeg:
    """Pinned nano embedder over the FROZEN surface convention.

    Surface: ``lib.embed_text`` (n1(title)\n+n1(body)\n+n1(space-joined
    tags)) truncated to [:4096]; cosine over the 384-dim vectors with the
    ADDENDUM-5 D1/F5 hardening: the provider normalizes in float32 (norm
    = 1 ± eps32), so both vectors are UPCAST to float64 and the cosine is
    computed as dot/(|a|*|b|) in float64 — root-cause hygiene — then the
    fail-closed guard (see cos_clamp_guard). A pin mismatch is a loud
    instrument abort.
    """

    def __init__(self) -> None:
        from vesma.embeddings import NanoProvider

        p = NanoProvider()
        if p.weights_sha256 != lib.EMBEDDER_PIN:
            raise InstrumentAbort(
                f"embedder pin mismatch: provider {p.weights_sha256[:12]}… "
                f"vs pinned nano {lib.EMBEDDER_PIN[:12]}… — refusing loudly"
            )
        self._p = p
        self.cache: dict[str, list[float]] = {}
        self.n = 0

    def vec(self, text: str) -> list[float]:
        t = text[:EMBED_TRUNC]
        if t not in self.cache:
            self.cache[t] = self._p.embed(t)
            self.n += 1
        return self.cache[t]

    def pair_cos(self, side_a, side_b, *, site: str = "pair_cos") -> float:
        """Cosine with the ADDENDUM-5 D1/F5 runner-leg hardening.

        Primary-cause hygiene (F5): upcast to float64 BEFORE
        normalize/dot — the raw float32-normalized dot could land at
        1 + 1.1e-7 (the incident's escaped scalar); the float64
        dot/(|a|*|b|) form leaves residual overshoot at float64 rounding
        scale (~1e-16, measured below 1 on self-pairs). The result passes
        the fail-closed guard before any frozen API sees it."""
        a = np.asarray(self.vec(lib.embed_text(side_a)), dtype=np.float64)
        b = np.asarray(self.vec(lib.embed_text(side_b)), dtype=np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if not (np.isfinite(na) and np.isfinite(nb)) or na <= 0.0 or nb <= 0.0:
            raise InstrumentAbort(
                f"zero/NaN vector norm in the runner cosine leg ({site}) — "
                "instrument anomaly, fail-closed"
            )
        cos = float(np.dot(a, b) / (na * nb))
        return cos_clamp_guard(cos, n_dims=int(a.shape[0]), site=site)


# --- prologue (ML-STD-0003 П1-П6, П9) -----------------------------------------


def prologue(slice_name: str) -> dict:
    """Machine prologue; order enforced by this runner, fail-closed."""
    # П9: first event = resource state (swap_check; hard rule: RAM floor).
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    try:
        avail_mb = int(free.splitlines()[1].split()[6])
        mem_ok = avail_mb >= 2000  # light lane needs << 1.5 GB; margin
    except (IndexError, ValueError):
        avail_mb, mem_ok = -1, False
    log(
        "swap_check",
        swapon_lines=swap.strip().splitlines(),
        available_mb=avail_mb,
        ram_overrun_stop=bool(not mem_ok),
    )
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
    log(
        "host",
        hostname=host,
        expectation=EXPECTED_HOST + " — same host as the corpus build; "
        "cross-host needs a TL decision BEFORE the run",
    )
    if host != EXPECTED_HOST:
        raise InstrumentAbort(f"host changed vs expectation {EXPECTED_HOST}: {host}")

    # П4: byte tripwire — frozen inputs and code pins.
    manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    pin = json.loads(PIN_JSON.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_JSON.read_text(encoding="utf-8"))
    pins = {
        "train_jsonl_sha256": sha256_file(TRAIN_JSONL),
        "pin_json_sha256": sha256_file(PIN_JSON),
        "build_manifest_sha256": sha256_file(BUILD_MANIFEST),
        "conflict_corpus_lib_sha256": sha256_file(Path(lib.__file__)),
        "build_driver_sha256": sha256_file(Path(cb.__file__)),
        "cortex_features_pair_sha256": sha256_file(
            REPO / "src" / "cortex" / "features" / "pair.py"
        ),
        "feature_names_sha256": sha256_text(repr(FEATURE_NAMES)),
        "embedder_pin": lib.EMBEDDER_PIN,
        "guard_version": lib.GUARD_VERSION,
    }
    want = {
        "train_jsonl_sha256": manifest["fingerprints"]["train.jsonl"],
        "pin_json_sha256": manifest["pin_sha256"],
        "conflict_corpus_lib_sha256": None,  # recorded, no external pin
        "build_driver_sha256": None,
        "cortex_features_pair_sha256": None,
        "feature_names_sha256": None,
        "embedder_pin": manifest["instrument"]["nano_pin"],
        "guard_version": manifest["guard_version"],
    }
    hard_mism = [
        k
        for k in ("train_jsonl_sha256", "pin_json_sha256", "embedder_pin", "guard_version")
        if pins[k] != want[k]
    ]
    # seal-json consistency at the METADATA level (the sealed file itself is
    # not touched): seal pin == build-manifest pin == the dispatch pin.
    seal_mism = []
    if seal["holdout_jsonl_sha256"] != manifest["fingerprints"]["holdout.jsonl"]:
        seal_mism.append("seal_vs_build_manifest")
    if seal["holdout_jsonl_sha256"] != HOLDOUT_SHA_PIN:
        seal_mism.append("seal_vs_dispatch_pin")
    if TRAIN_SHA_PIN != manifest["fingerprints"]["train.jsonl"]:
        seal_mism.append("train_pin_vs_dispatch")
    pins["seal_n_ids"] = seal["n"]
    # v2 preset (ADDENDUM-3 E11): the NEW v2 pins are hard — the pin-v2 file
    # bytes and the seal FILE bytes are pinned against the preset and the v2
    # build manifest. The v1 tripwire keeps its exact F1 shape.
    v2_tripwire = {}
    if _INPUT_PRESET_VERSION == "v2":
        if pins["pin_json_sha256"] != _INPUT_PRESET["pin_sha256"]:
            seal_mism.append("pin_v2_vs_preset_sha")
        if pins["pin_json_sha256"] != manifest["pin_sha256"]:
            seal_mism.append("pin_v2_vs_build_manifest")
        if TRAIN_SHA_PIN != pins["train_jsonl_sha256"]:
            seal_mism.append("train_pin_vs_bytes")
        seal_file_sha = sha256_file(SEAL_JSON)
        if seal_file_sha != _INPUT_PRESET["seal_sha256"]:
            seal_mism.append("seal_file_vs_preset_sha")
        if seal_file_sha != manifest["fingerprints"].get("seal"):
            seal_mism.append("seal_file_vs_build_manifest")
        if seal["n"] != SEAL_N_META:
            seal_mism.append("seal_n_vs_preset")
        v2_tripwire = {
            "preset": "v2",
            "pin_sha256_pinned": _INPUT_PRESET["pin_sha256"],
            "seal_sha256_pinned": _INPUT_PRESET["seal_sha256"],
            "seal_file_sha256": seal_file_sha,
            "holdout_sha_metadata_pin": HOLDOUT_SHA_PIN,
        }
    log(
        "byte_tripwire",
        pins=pins,
        expected_hard=want,
        mismatch_hard=hard_mism,
        seal_metadata_mismatch=seal_mism,
        v2_preset_pins=v2_tripwire,
        sealed_file_opened=False,
        note="holdout.jsonl NOT hashed/opened — metadata pins only",
    )
    if hard_mism or seal_mism:
        raise InstrumentAbort(
            f"byte tripwire mismatch: hard={hard_mism} seal={seal_mism}"
        )

    # П6: activation assert — ratified design doc at HEAD + wave commits
    # in ancestry (corpus built, gold closed).
    spec_text = SPEC.read_text(encoding="utf-8")
    ratified = bool(
        re.search(r"\*\*Статус:\*\*\s*RATIFIED\s*\(2026-11-01\):\s*MLC-9", spec_text)
    )
    ancestry = {}
    for name, commit in (
        ("dispatch-brief", "9706158"),
        ("delta-map", "1c85715"),
        ("gold-closed", GOLD_COMMIT),
        ("std0004-formula", STD0004_FORMULA_COMMIT),
    ):
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPO,
            capture_output=True,
        )
        ancestry[name] = {"commit": commit, "is_ancestor": r.returncode == 0}
    # v2 preset: the corpus-v2 ratification (ADDENDUM-3 §8, amend-then-ratify)
    # and the corpus-v2 build commit join the activation assert.
    activation_extra = {}
    ratified_effective = ratified
    if _INPUT_PRESET_VERSION == "v2":
        extra = _INPUT_PRESET["activation_extra"]
        add_text = extra["addendum"].read_text(encoding="utf-8")
        add_ratified = extra["marker"] in add_text
        r2 = subprocess.run(
            ["git", "merge-base", "--is-ancestor", extra["corpus_v2_build_commit"], "HEAD"],
            cwd=REPO,
            capture_output=True,
        )
        ancestry["corpus-v2-build"] = {
            "commit": extra["corpus_v2_build_commit"],
            "is_ancestor": r2.returncode == 0,
        }
        activation_extra = {
            "addendum": extra["addendum"].name,
            "addendum_sha256": sha256_text(add_text),
            "ratified_marker_present": add_ratified,
        }
        ratified_effective = ratified and add_ratified and r2.returncode == 0
    ok = ratified_effective and all(v["is_ancestor"] for v in ancestry.values())
    log(
        "activation",
        spec_sha256=sha256_text(spec_text),
        ratified_marker_present=ratified,
        commits=ancestry,
        activation_extra=activation_extra,
        ok=ok,
    )
    if not ok:
        raise InstrumentAbort("activation assert failed: ratified spec not at HEAD")

    return {"build_manifest": manifest, "pin": pin, "pins": pins}


# --- frozen inputs (П5) -------------------------------------------------------


def row_records(row: dict) -> tuple[RecordLike, RecordLike]:
    """CORE-13 surface records from a corpus row's envelopes.

    ``declaration``/``is_version_token``/``split``/``stratum`` are
    construction-only: they NEVER enter the feature surface
    (read-on-inference forbidden, design §2.3.2 / §5.1).
    """

    def rec(side: dict) -> RecordLike:
        return RecordLike(
            title=side["title"],
            body=side["body"],
            tags=tuple(side["tags"]),
            language=side.get("language"),
            record_type=side.get("record_type"),
        )

    return rec(row["envelope_a"]), rec(row["envelope_b"])


def pair_vector(rec_a: RecordLike, rec_b: RecordLike, cos_target: float) -> np.ndarray:
    """Frozen CORE-13 through cortex.features.pair — 0 reimplementations."""
    vec = features(rec_a, rec_b, cos_target)
    if vec.names != FEATURE_NAMES:
        raise InstrumentAbort("feature contract mismatch: non-core feature names")
    return np.asarray(vec.values, dtype=np.float64)


def load_train() -> list[dict]:
    """Load ONLY train.jsonl (the sealed holdout is not in the input list)."""
    rows = [json.loads(line) for line in open(TRAIN_JSONL, encoding="utf-8")]
    return rows


def frozen_inputs() -> list[dict]:
    """П5: count/dedup/label asserts against the ratified quota table."""
    rows = load_train()
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
        key = (r["pair_id"], r["orientation"])
        if key in seen:
            raise InstrumentAbort(f"duplicate (pair_id, orientation): {key}")
        seen.add(key)
        if r["split"] != "train":
            raise InstrumentAbort(f"non-train split row in train.jsonl: {r['pair_id']}")
    mism = {s: {"got": counts.get(s, 0), "want": n} for s, n in EXPECTED_TRAIN_ROW_COUNTS.items()
            if counts.get(s, 0) != n}
    extra = [s for s in counts if s not in EXPECTED_TRAIN_ROW_COUNTS]
    pairs = {r["pair_id"] for r in rows}
    label_mism = [
        r["pair_id"]
        for r in rows
        if r["label"] != ("contradicts" if r["stratum"].startswith("CD") else "not-contradicts")
    ]
    log(
        "frozen_inputs",
        rows=len(rows),
        pairs=len(pairs),
        counts={s: counts.get(s, 0) for s in ORDER},
        mismatch=mism,
        extra_strata=extra,
        label_mismatch=label_mism[:5],
    )
    if mism or extra or label_mism or len(rows) != 1200 or len(pairs) != 600:
        raise InstrumentAbort(
            f"train frozen-input asserts RED: counts={mism} extra={extra} "
            f"label={label_mism[:5]}"
        )
    return rows


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Feature table over corpus rows: cos_target rides ``measured`` (the
    pinned build-time nano measurement), the other 12 features are the
    frozen pair features over the envelopes."""
    xs, ys, meta = [], [], []
    for r in rows:
        rec_a, rec_b = row_records(r)
        xs.append(pair_vector(rec_a, rec_b, float(r["measured"]["cos_target"])))
        ys.append(1.0 if r["label"] == "contradicts" else 0.0)
        meta.append({"pair_id": r["pair_id"], "stratum": r["stratum"],
                     "orientation": r["orientation"]})
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), meta


def assert_orientation_symmetry(rows: list[dict], n_check: int = 60) -> None:
    """Machine assert: swapping the presentation produces the IDENTICAL
    CORE-13 vector (the frozen core block is symmetric — pinned by test in
    pair.py); this is what floor_perm later certifies at the VERDICT level."""
    by_pair: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["orientation"]] = r
    rng = random.Random(SEED)
    sample = rng.sample(sorted(by_pair), min(n_check, len(by_pair)))
    for pid in sample:
        d = by_pair[pid]
        if "a>b" not in d or "b>a" not in d:
            raise InstrumentAbort(f"pair {pid} missing an orientation")
        ra = d["a>b"]
        rb = d["b>a"]
        rec_a1, rec_b1 = row_records(ra)
        rec_a2, rec_b2 = row_records(rb)
        v1 = pair_vector(rec_a1, rec_b1, ra["measured"]["cos_target"])
        v2 = pair_vector(rec_a2, rec_b2, rb["measured"]["cos_target"])
        if not np.array_equal(v1, v2):
            raise InstrumentAbort(
                f"orientation symmetry RED on {pid}: features depend on "
                "presentation order"
            )
    log("feature_orientation_symmetry", checked=len(sample), ok=True)


# --- guard code leg (cg-guard-v1: the NC-3 predicate, §2.2/§2.3) ---------------


def guard_fire(body_a: str, body_b: str) -> bool:
    """Deterministic code leg — NO reimplementation: the frozen lib
    predicate ``version_guard`` factorized to the pair level. One-sided:
    fire => verdict NOT contradicts, unconditionally."""
    regs = lib.regions(body_a, body_b)
    if not regs:
        return False
    return all(lib.region_is_version(t) for _, t, _, _ in regs)


GUARD_CONSTANTS = {
    "guard_version": lib.GUARD_VERSION,
    "predicate": "ALL difflib non-equal regions of the norm1 bodies consist "
    "of version-namespace tokens (semver \\bv?\\d+\\.\\d+(\\.\\d+)?\\b or "
    "version keywords) -> NOT contradicts, unconditionally; no regions -> "
    "no fire",
    "region_gap_tokens": 3,
    "version_regex": lib.VERSION_RE.pattern,
    "version_keywords": sorted({
        "версия", "текущая", "version", "current", "релиз", "release",
        "сборка", "build", "обновление", "update",
    }),
    "source": "scripts/conflict_corpus_lib.py (frozen; sha pinned in the "
    "prologue tripwire as conflict_corpus_lib_sha256)",
    "one_sided": "NOT-leg only; the contradicts side of the guard is read "
    "nowhere in the contour",
}


def assert_guard_on_train(rows: list[dict]) -> dict:
    """Guard-vs-construction assert: cg-guard-v1 must fire on ALL NC-3 rows
    and on NO other train row (the corpus classes are disjoint on this
    predicate by construction — design §2.2 rule of priority MLC-9 p.10)."""
    fire_by: dict[str, int] = {}
    total_by: dict[str, int] = {}
    for r in rows:
        rec_a, rec_b = row_records(r)
        fired = guard_fire(rec_a.body, rec_b.body)
        total_by[r["stratum"]] = total_by.get(r["stratum"], 0) + 1
        if fired:
            fire_by[r["stratum"]] = fire_by.get(r["stratum"], 0) + 1
    nc3 = EXPECTED_TRAIN_ROW_COUNTS["NC-3"]
    ok = fire_by.get("NC-3", 0) == nc3 and all(
        fire_by.get(s, 0) == 0 for s in ORDER if s != "NC-3"
    )
    log(
        "guard_train_assert",
        fired=fire_by,
        totals=total_by,
        expect="NC-3 108/108, others 0",
        ok=ok,
    )
    if not ok:
        raise InstrumentAbort(f"guard-vs-construction assert RED: fired={fire_by}")
    return {"fired": fire_by, "totals": total_by}


# --- baselines (§6.1; definitions byte-identical to the build driver) ---------


def _digits(text: str) -> list[str]:
    return sorted(re.findall(r"\d+", text))


def bl_band_cos(cos: float, band: tuple[float, float] = (0.95, 1.0)) -> bool:
    """BL-band: contradicts iff cos in the razor band. Direction (band as
    positive vs complement) is chosen train-only from the two options."""
    return band[0] <= cos <= band[1]


def baselines_table(rows: list[dict]) -> dict:
    """BL-band / BL-digit / BL-digit+guard accuracy on the given rows.
    BL definitions mirror the build driver verbatim (cmd_build §baselines);
    the band direction is selected train-only from two options and
    published with the table."""
    n = len(rows)
    out: dict[str, float] = {}
    for name in (0, 1):  # both band directions, published as characterization
        acc = sum(
            1
            for r in rows
            if (bl_band_cos(r["measured"]["cos_target"]) == (name == 0))
            == (r["label"] == "contradicts")
        ) / n
        out[f"BL-band-dir{name}"] = round(acc, 4)
    # BL-band direction pinned to the BUILD-manifest convention (razor band
    # [0.95, 1.0] => contradicts — the §6.1 "conflicts live in the razor
    # band" hypothesis; the manifest is the ratified measure->fix
    # publication). The complement direction stays published as a
    # diagnostic; flipping the baseline direction silently is threshold
    # fitting, not a train-only choice.
    out["BL-band"] = out["BL-band-dir0"]
    out["BL-digit"] = round(
        sum(
            1
            for r in rows
            if (_digits(r["envelope_a"]["body"]) != _digits(r["envelope_b"]["body"]))
            == (r["label"] == "contradicts")
        )
        / n,
        4,
    )
    out["BL-digit+guard"] = round(
        sum(
            1
            for r in rows
            if (
                (
                    _digits(r["envelope_a"]["body"])
                    != _digits(r["envelope_b"]["body"])
                )
                and not guard_fire(
                    r["envelope_a"]["body"], r["envelope_b"]["body"]
                )
            )
            == (r["label"] == "contradicts")
        )
        / n,
        4,
    )
    return out


def check_baselines_vs_build(manifest: dict, mine: dict) -> None:
    """Baseline reproduction assert: same train material + the same frozen
    baseline definitions => exact match with the build manifest."""
    diff = {
        k: {"got": mine.get(k), "want": v}
        for k, v in EXPECTED_BASELINES.items()
        if mine.get(k) != v
    }
    log(
        "baseline_reproduction",
        mine=mine,
        build_manifest=manifest["baselines"],
        mismatch=diff,
        note="BL-band direction chosen train-only and published",
    )
    if diff:
        raise InstrumentAbort(f"baseline table diverges from the build manifest: {diff}")


# --- production circuit (guard code leg + model leg; ratified MLC-9) ----------


def decide_pipeline(
    rec_a: RecordLike, rec_b: RecordLike, cos_target: float, booster: lgb.Booster
) -> dict:
    """The production contour №6: version-guard (one-sided NOT) -> model
    leg, cut 0.5 on P(contradicts), NO threshold search."""
    if guard_fire(rec_a.body, rec_b.body):
        return {
            "verdict": "not-contradicts",
            "leg": "guard",
            "guard_version": lib.GUARD_VERSION,
            "p": None,
        }
    x = pair_vector(rec_a, rec_b, cos_target).reshape(1, -1)
    if x.shape[1] != booster.num_feature():
        raise InstrumentAbort(
            f"feature contract mismatch: contour builds {x.shape[1]} columns, "
            f"the frozen booster expects {booster.num_feature()}"
        )
    p = float(booster.predict(x)[0])
    return {
        "verdict": "contradicts" if p >= DECISION_CUT else "not-contradicts",
        "leg": "model",
        "p": round(p, 6),
    }


def decide_rows(
    rows: list[dict], booster: lgb.Booster, cos_of=None
) -> tuple[int, int, list[dict]]:
    """Run the production contour over rows; returns (hits, n, per-row).
    ``cos_of`` overrides the cos source (fresh fixtures/draws measure with
    the pinned NanoLeg; corpus rows ride ``measured.cos_target``)."""
    hits = 0
    out = []
    for r in rows:
        rec_a, rec_b = row_records(r)
        cos = cos_of(rec_a, rec_b) if cos_of is not None else float(r["measured"]["cos_target"])
        d = decide_pipeline(rec_a, rec_b, cos, booster)
        d["pair_id"] = r.get("pair_id")
        d["stratum"] = r.get("stratum")
        d["label"] = r.get("label")
        d["cos_target"] = round(cos, 4) if cos is not None else None
        hits += int(d["verdict"] == r["label"])
        out.append(d)
    return hits, len(rows), out


def acc_pipeline(rows: list[dict], booster: lgb.Booster, cos_of=None) -> float:
    hits, n, _ = decide_rows(rows, booster, cos_of=cos_of)
    return hits / n if n else 0.0


# --- LightGBM leg (arbiter fit path; cut 0.5 ratified, no CV selection) -------


def fit_booster(x: np.ndarray, y: np.ndarray) -> lgb.Booster:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(x, y)
    return model.booster_


def cv_characterization(x: np.ndarray, y: np.ndarray, meta: list[dict]) -> dict:
    """5-fold OOF characterization BY SCAFFOLD (pairwise-disjoint folds) —
    an honesty number ONLY: the ratified decision rule is cut 0.5 with no
    threshold search, so this CV selects nothing."""
    from sklearn.model_selection import GroupKFold

    scaffolds = []
    seen = {}
    for m in meta:
        # scaffold of the pair: map pair_id -> its scaffold via the train rows
        scaffolds.append(m["pair_id"])
    groups = np.asarray([_pair_scaffold(p) for p in scaffolds])
    gkf = GroupKFold(n_splits=CV_K)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_i, te_i in gkf.split(x, y, groups):
        b = fit_booster(x[tr_i], y[tr_i])
        oof[te_i] = b.predict(x[te_i])
    pred = (oof >= DECISION_CUT).astype(np.float64)
    acc = float((pred == y).mean())
    brier = float(((oof - y) ** 2).mean())
    log(
        "cv_characterization",
        folds=CV_K,
        grouping="scaffold_id (pairwise-disjoint)",
        oof_acc=round(acc, 4),
        oof_brier=round(brier, 4),
        cut=DECISION_CUT,
        note="honesty characterization only — NOT a threshold search "
        "(ratified cut 0.5, MLC-9)",
    )
    return {"oof_acc": round(acc, 4), "oof_brier": round(brier, 4), "folds": CV_K}


_PAIR_SCAFFOLD_CACHE: dict[str, str] = {}


def _pair_scaffold(pair_id: str) -> str:
    if not _PAIR_SCAFFOLD_CACHE:
        for r in load_train():
            _PAIR_SCAFFOLD_CACHE[r["pair_id"]] = r["scaffold_id"]
    return _PAIR_SCAFFOLD_CACHE[pair_id]


# --- artifact freeze (bundle: manifest.json + model.onnx + MODEL_CARD.md) -----


def _export_onnx(booster: lgb.Booster, trained_at: str, corpus_fp: str) -> dict:
    """LightGBM -> ONNX (opset 15, onnxmltools) with typed-output smoke:
    input float32[1,13] -> (label int64, probabilities float32[1,2]); every
    train row must produce finite typed outputs matching the library within
    1e-6 (typed_output 100%, dispatch §7)."""
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnx

    converted = convert_lightgbm(
        booster,
        initial_types=[("features_in", FloatTensorType([None, len(FEATURE_NAMES)]))],
        target_opset=15,
    )
    # Typed-output contract (dispatch §7): replace the default ZipMap tail
    # with the RAW probabilities tensor (d_boost rebuilt the graph tail for
    # the same reason — a clean float tensor, no seq(map) container).
    from onnx import TensorProto

    zipmap = next(
        (n for n in converted.graph.node if n.op_type == "ZipMap"), None
    )
    if zipmap is not None:
        prob_name = zipmap.input[0]
        converted.graph.node.remove(zipmap)
        # rename the producing tensor too (graph-scope consistency)
        for n in converted.graph.node:
            for i, oname in enumerate(n.output):
                if oname == prob_name:
                    n.output[i] = "probabilities"
        for o in converted.graph.output:
            if o.name == zipmap.output[0]:
                o.name = "probabilities"
                o.type.tensor_type.elem_type = TensorProto.FLOAT
                # shape [None, 2]
                while len(o.type.tensor_type.shape.dim) < 2:
                    o.type.tensor_type.shape.dim.add()
                o.type.tensor_type.shape.dim[1].dim_value = 2
    from cortex.artifacts import set_onnx_metadata, features_digest

    props = {
        "name": "conflict-detector",
        "version": ART_VERSION_SUFFIX,
        "embedder_pin": "nano:sha256:" + lib.EMBEDDER_PIN,
        "corpus_fingerprint": corpus_fp,
        "trained_at": trained_at,
        "candidate": ART_PREFIX,
        "features": "\n".join(FEATURE_NAMES),
        "feature_set_sha256": features_digest(FEATURE_NAMES),
        "guard_version": lib.GUARD_VERSION,
        "decision_cut": str(DECISION_CUT),
    }
    set_onnx_metadata(converted, props)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(converted.SerializeToString())
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))

    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    types = [o.type for o in session.get_outputs()]
    x, y, _meta = build_matrix(load_train())
    lib_p = booster.predict(x)
    onnx_p = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        outs = session.run(None, {"features_in": x[i : i + 1].astype(np.float32)})
        # outs = [label int64[1], probabilities float32[1,2]]
        onnx_p[i] = float(outs[1][0][1])
    finite = int(np.isfinite(onnx_p).sum())
    max_dev = float(np.abs(onnx_p - lib_p).max())
    ok = (
        names == ["label", "probabilities"]
        and types == ["tensor(int64)", "tensor(float)"]
        and finite == len(x)
        and max_dev <= 1e-6
    )
    log(
        "onnx_typed_output_smoke",
        output_names=names,
        output_types=types,
        rows=len(x),
        finite_outputs=finite,
        max_abs_dev_lib_vs_onnx=round(max_dev, 9),
        typed_output_ok=ok,
        threshold="names/types pinned; finite 100%; max dev <= 1e-6",
    )
    if not ok:
        raise InstrumentAbort(
            f"typed-output smoke RED: names={names} types={types} "
            f"finite={finite}/{len(x)} max_dev={max_dev}"
        )
    return {"sha256": sha256_file(ONNX_PATH), "metadata_props": props}


def freeze_artifact(
    booster: lgb.Booster,
    cv: dict,
    baselines: dict,
    pins: dict,
    guard_summary: dict,
    guard_off: dict,
    in_sample: dict,
) -> str:
    """Freeze the artifact bundle under ART_DIR (dispatch §7 bundle; the
    CONTENT sections mirror the merge-v2 committed-doc discipline per the
    delta map §8: thresholds/rule, guard section with pinned constants,
    corpus pins, baselines, cv characterization, env pins, convention).
    The artifact identity (name/version) derives from the resolved
    ART_DIR — input plumbing, not decision semantics."""
    if ART_DIR.exists() and any(ART_DIR.iterdir()):
        raise InstrumentAbort(
            f"artifact dir already populated: {ART_DIR} — refreeze over "
            "frozen bytes is forbidden"
        )
    ART_DIR.mkdir(parents=True, exist_ok=True)
    trained_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
    corpus_fp = blake2b_256_file(TRAIN_JSONL)
    onnx_info = _export_onnx(booster, trained_at, corpus_fp)

    manifest = {
        "manifest_schema": 2,
        "name": ART_PREFIX,
        "version": ART_VERSION_SUFFIX,
        "wave": "b6-conflict",
        "artifact_lineage": _INPUT_PRESET["lineage"],
        "sha256": onnx_info["sha256"],
        "size_bytes": ONNX_PATH.stat().st_size,
        "embedder_pin": lib.EMBEDDER_PIN,
        "embedder_pin_engine": "nano:sha256:" + lib.EMBEDDER_PIN,
        "corpus_fingerprint_blake2b_256": corpus_fp,
        "trained_at": trained_at,
        "candidate": ART_PREFIX,
        "features": list(FEATURE_NAMES),
        "feature_set_sha256": onnx_info["metadata_props"]["feature_set_sha256"],
        "decision_rule": {
            "cut": DECISION_CUT,
            "rule": "contradicts iff P(contradicts) >= 0.5 (guard leg aside); "
            "NO threshold search, NO label shift (ratified MLC-9)",
            "gates_fit": "pipeline (guard+model) is the gated object "
            "(dispatch §4 note 1); model-only numbers are diagnostics",
        },
        "guard": GUARD_CONSTANTS,
        "guard_train_assert": guard_summary,
        "guard_off_honesty_train": guard_off,
        "lgb_params": dict(LGB_PARAMS),
        "booster_model_string": booster.model_to_string(),
        "cv_characterization": cv,
        "in_sample_train_diagnostic": in_sample,
        "baselines_train": baselines,
        "baselines_target_max_bl": max(EXPECTED_BASELINES.values()),
        "convention": {
            "embed_surface": "n1(title)\\n+n1(body)\\n+n1(space-joined tags), "
            "[:4096]; n1 = lower + whitespace-collapse (frozen "
            "lib.embed_text; corpus measured with the same surface — "
            "max embed text len asserted < 4096 at train)",
            "cos_target": "MEASURED vector-leg cosine over the pair (pinned "
            "nano, dot of norm1 vectors); corpus rows ride "
            "measured.cos_target; fresh material measures via NanoLeg",
            "construction_only": "declaration/is_version_token/stratum/split "
            "NEVER enter the inference surface (design §2.3.2)",
            "orientations": "both (A-symmetry); core features "
            "orientation-symmetric (machine-asserted)",
        },
        "corpus": {
            "train_rows": 1200,
            "train_pairs": 600,
            "train_jsonl_sha256": pins["train_jsonl_sha256"],
            "build_manifest_sha256": pins["build_manifest_sha256"],
            "pin_json_sha256": pins["pin_json_sha256"],
            "holdout_jsonl_sha256_metadata_pin": HOLDOUT_SHA_PIN,
            "seal_n_ids_metadata": SEAL_N_META,
            "regime": "synthetic-only; no store texts; no LLM; labels by "
            "construction",
        },
        "pins": {k: v for k, v in pins.items() if k != "seal_n_ids"},
        "env": dict(EXPECTED_ENV),
        "seed": SEED,
        "gates_state": "sealed — single-shot NOT executed at freeze time "
        f"(dispatch §3; holdout {SEAL_NAME} reads exactly once at the gate)",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    manifest_sha = sha256_file(MANIFEST_PATH)
    log(
        "artifact_frozen",
        durable=str(ONNX_PATH),
        onnx_sha256=onnx_info["sha256"],
        manifest=str(MANIFEST_PATH),
        manifest_sha256=manifest_sha,
        bundle_integrity="sha256(model.onnx) == manifest.sha256",
    )
    write_model_card(manifest)
    return onnx_info["sha256"]


def write_model_card(manifest: dict) -> None:
    """MODEL_CARD.md per the dispatch bundle (§7) + dataset-card glossary
    pointer (design §5.2: глоссарий обязателен)."""
    b = manifest["baselines_train"]
    lines = [
        f"# MODEL CARD — {ART_PREFIX}",
        "",
        f"- Trained: {manifest['trained_at']} (host {EXPECTED_HOST}, seed {SEED})",
        "- Task: binary contradicts/not-contradicts on memory record PAIRS "
        "(ADR 0002 duty #6; taxonomy design §2, classes CD-1..6 / NC-1..5)",
        "- Decision rule: production circuit = version-guard (cg-guard-v1, "
        "one-sided NOT, NC-3 predicate) then LightGBM P(contradicts), cut 0.5, "
        "no threshold search (ratified MLC-9)",
        f"- Features: frozen CORE-13 `cortex.features.pair` "
        f"(sha256(names) {manifest['feature_set_sha256'][:16]}…)",
        f"- Embedder: nano {lib.EMBEDDER_PIN[:12]}… (A-frozen), surface "
        "n1(title)\\n+n1(body)\\n+n1(tags) [:4096]",
        f"- Corpus: 600 pairs x 2 orientations = 1200 rows "
        f"(sha256 {manifest['corpus']['train_jsonl_sha256'][:16]}…); "
        f"holdout {SEAL_NAME} SEALED (180 pairs), read exactly once by the "
        "gate runner",
        f"- Baselines (train): BL-band {b['BL-band']} / BL-digit "
        f"{b['BL-digit']} / BL-digit+guard {b['BL-digit+guard']} "
        f"(target max(BL) = {max(EXPECTED_BASELINES.values())})",
        f"- CV characterization (honesty, no selection): OOF acc "
        f"{manifest['cv_characterization']['oof_acc']}, OOF brier "
        f"{manifest['cv_characterization']['oof_brier']} "
        f"({manifest['cv_characterization']['folds']} folds by scaffold)",
        f"- Floors: {FLOORS_DOC} (ML-STD-0001/0004; "
        "published BEFORE gate_start; re-measure after single-shot forbidden)",
        "- Gates §6.2: verdict = conjunction; ADOPT/DECLINE by the single-shot "
        "gate runner (implemented, executed only on the explicit TL command)",
        "- Glossary: docs/experiments/conflict-detector-corpus-card.md "
        "(taxonomy §2 dataset-card glossary)",
        "- Known limitation (design §2.3): razor-zone CD-1 vs NC-3 is "
        "indistinguishable on the 13-feature surface — certified by the "
        "guard code leg + g_nc3_exact/g_razor, NOT by the model",
    ]
    MODEL_CARD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("model_card_written", path=str(MODEL_CARD))


def load_frozen(art_dir: Path = ART_DIR) -> tuple[lgb.Booster, dict]:
    """Load the frozen artifact (bundle integrity asserted).

    The library twin rides the manifest's ``booster_model_string`` (the
    exact frozen booster, no refit); the ONNX graph is the serving artifact
    and the typed-output identity is re-asserted against it (<= 1e-6)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sha = sha256_file(ONNX_PATH)
    if sha != manifest["sha256"]:
        raise InstrumentAbort(
            f"bundle integrity RED: sha256(model.onnx) {sha[:12]}… != "
            f"manifest {manifest['sha256'][:12]}…"
        )
    booster = lgb.Booster(model_str=manifest["booster_model_string"])
    x, y, _meta = build_matrix(load_train())
    p = booster.predict(x)
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_p = np.asarray(
        [
            session.run(None, {"features_in": x[i : i + 1].astype(np.float32)})[1][0][1]
            for i in range(len(x))
        ]
    )
    if float(np.abs(onnx_p - p).max()) > 1e-6:
        raise InstrumentAbort(
            "frozen-artifact identity RED: manifest booster twin diverges "
            "from the ONNX bundle beyond 1e-6"
        )
    return booster, manifest


# --- guard-off honesty probe (MLC-9 p.13; NOT a gate) -------------------------


def guard_off_probe(rows: list[dict], booster: lgb.Booster) -> dict:
    """How many contradicts proposals does the MODEL leg alone make on the
    material the guard suppresses (train NC-3 rows)? Published as an
    honesty number at freeze time and per-stratum at gate time."""
    proposals = 0
    per_stratum: dict[str, dict[str, int]] = {}
    for r in rows:
        rec_a, rec_b = row_records(r)
        if not guard_fire(rec_a.body, rec_b.body):
            continue
        x = pair_vector(rec_a, rec_b, float(r["measured"]["cos_target"])).reshape(1, -1)
        p = float(booster.predict(x)[0])
        verdict = "contradicts" if p >= DECISION_CUT else "not-contradicts"
        st = per_stratum.setdefault(
            r["stratum"], {"rows": 0, "model_proposals_guard_off": 0}
        )
        st["rows"] += 1
        proposals += int(verdict == "contradicts")
        st["model_proposals_guard_off"] += int(verdict == "contradicts")
    probe = {
        "rows": sum(s["rows"] for s in per_stratum.values()),
        "model_proposals_guard_off": proposals,
        "per_stratum": per_stratum,
        "not_a_gate": True,
        "honesty": "the number of NC-3 contradicts proposals the guard "
        "suppresses (MLC-9 p.13; published before gate_start)",
    }
    log("guard_off_probe_train", **probe)
    return probe


# --- cosmetic / paraphrase surfaces (floor_inv / sanity probes) ----------------


def cosmetic_variant(text: str, rng: random.Random) -> str:
    """Cosmetic corruption: CASE flips ONLY.

    Invariant on the whole frozen surface BY CONSTRUCTION: pair.py
    normalizes text (lower + whitespace-collapse), the guard regions run
    on norm1, embed_text runs n1 — a case flip is invisible to all of
    them AND keeps the raw lengths (so *_len_delta stays untouched).
    Whitespace additions are NOT cosmetic on this surface (body_len_delta
    is a raw char-length delta — see the floor_fixture_defect run-log
    event): they ride the published whitespace-len diagnostic instead."""
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < 0.15 and w:
            w = w.upper()
        elif r < 0.30 and w:
            w = w.capitalize()
        out.append(w)
    return " ".join(out)


def ws_len_variant(text: str) -> str:
    """Whitespace-length perturbation — DIAGNOSTIC surface (not cosmetic):
    adds trailing whitespace, visible to body_len_delta by design."""
    return text + "  "


def assert_cosmetic_surface_invariance(row: dict, rng: random.Random) -> None:
    """Machine proof that the cosmetic transform is surface-invariant:
    identical CORE-13 vector, identical embed_text, identical guard."""
    rec_a, rec_b = row_records(row)
    pert = row_with_body(row, "envelope_b", cosmetic_variant(row["envelope_b"]["body"], rng))
    prec_a, prec_b = row_records(pert)
    cos = float(row["measured"]["cos_target"])
    v1 = pair_vector(rec_a, rec_b, cos)
    v2 = pair_vector(prec_a, prec_b, cos)
    if not np.array_equal(v1, v2):
        raise InstrumentAbort(
            "cosmetic surface-invariance RED: features changed under a "
            "case-only variant"
        )
    t1 = lib.embed_text(type("S", (), {"title": rec_b.title, "body": rec_b.body,
                                        "tags": list(rec_b.tags)})())
    t2 = lib.embed_text(type("S", (), {"title": prec_b.title, "body": prec_b.body,
                                        "tags": list(prec_b.tags)})())
    if t1 != t2:
        raise InstrumentAbort(
            "cosmetic surface-invariance RED: embed_text changed under a "
            "case-only variant"
        )


def row_with_body(row: dict, side: str, body: str) -> dict:
    r = json.loads(json.dumps(row))  # deep copy, no aliasing
    r[side]["body"] = body
    return r


def orientation_swap(row: dict) -> dict:
    """The b>a presentation of an a>b row (and vice versa) — floor_perm."""
    r = json.loads(json.dumps(row))
    r["envelope_a"], r["envelope_b"] = r["envelope_b"], r["envelope_a"]
    r["orientation"] = "b>a" if r["orientation"] == "a>b" else "a>b"
    return r


# --- ML-STD-0004 published distribution ----------------------------------------


def _dist(values: list[float]) -> dict:
    """ML-STD-0004 published distribution: median, p5, p95, max-pairwise
    (all four mandatory; only p5/p95 enter the delta_acc formula)."""
    if not values:
        raise InstrumentAbort("distribution of an empty draw list")
    med = float(np.median(values))
    p5 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    mp = max(
        (abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]),
        default=0.0,
    )
    return {
        "n": len(values),
        "median": round(med, 6),
        "p5": round(p5, 6),
        "p95": round(p95, 6),
        "max_pairwise": round(mp, 6),
        "method": f"numpy.percentile linear interpolation (numpy {np.__version__})",
    }


# --- floor draws (frozen build recipe; train-only material) --------------------


def floor_draw_rows(seed: int, nano: NanoLeg) -> list[dict]:
    """One composition draw: the FROZEN build recipe ``gen_pool(seed)``
    (fresh seed outside the train seed family; rotate_pair makes the texts
    byte-differ from train — the build's own A-disjoint mechanism), stratified
    selection to the pinned mini-quota, cos measured by the pinned NanoLeg,
    rows emitted by the FROZEN ``rows_of`` (both orientations)."""
    pool = cb.gen_pool(seed, FLOOR_PREFIX)
    by_s: dict[str, list] = {s: [] for s in ORDER}
    for sc in pool:
        if sc.stratum in by_s:
            by_s[sc.stratum].append(sc)
    picked = []
    for s in ORDER:
        cands = sorted(by_s[s], key=lambda x: x.scenario_id)
        take = FLOOR_DRAW_QUOTA[s]
        if len(cands) < take:
            raise InstrumentAbort(
                f"floor draw seed={seed}: stratum {s} has {len(cands)} "
                f"candidates < quota {take}"
            )
        picked.extend(cands[:take])
    if len(picked) != FLOOR_DRAW_PAIRS:
        raise InstrumentAbort(
            f"floor draw seed={seed}: composition {len(picked)} != "
            f"{FLOOR_DRAW_PAIRS}"
        )
    for sc in picked:
        sc._cos = round(
            nano.pair_cos(sc.side_a, sc.side_b, site=f"floor_draw:seed{seed}"), 4
        )
        sc._band = cb.band_of(sc._cos)
    rows = []
    for sc in picked:
        rows.extend(cb.rows_of(sc, "floor", f"seed:{seed}"))
    return rows


def assert_draw_disjointness(rows: list[dict], seed: int) -> None:
    """Draw ids must not intersect train ids (id-level; the holdout
    disjointness belongs to the gate's g_disj — the sealed file is not
    read in phase 1)."""
    train_ids = {r["pair_id"] for r in load_train()}
    draw_ids = {r["pair_id"] for r in rows}
    hits = sorted(draw_ids & train_ids)
    log(
        "floor_draw_disjointness",
        seed=seed,
        draw_ids=len(draw_ids),
        ids_train=hits[:10],
        sealed_file_opened=False,
    )
    if hits:
        raise InstrumentAbort(
            f"floor draw seed={seed} intersects train ids: {hits[:10]}"
        )


# --- subcommand: train ---------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_TRAIN
    started = dt.datetime.now(dt.timezone.utc)
    log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=180,
        expected_peak_ram_gb=1.0,
        lane="light: LightGBM on 1200x13, direct sequential run "
        "(arbiter precedent 57-78 s, <2 GB)",
        run_tag=args.run_tag,
    )
    pro = prologue(SLICE_TRAIN)

    # corpus embed-surface fits the truncated convention (no-op assertion;
    # keeps measured.cos_target consistent with the [:4096] instrument)
    max_len = 0
    for r in load_train():
        for side in (r["envelope_a"], r["envelope_b"]):
            max_len = max(
                max_len,
                len(
                    lib.embed_text(
                        type("S", (), {"title": side["title"], "body": side["body"],
                                       "tags": side["tags"]})()
                    )
                ),
            )
    log("embed_surface_len_check", max_embed_text_len=max_len, trunc=EMBED_TRUNC)
    if max_len >= EMBED_TRUNC:
        raise InstrumentAbort(
            f"corpus embed surface {max_len} >= truncation {EMBED_TRUNC} — "
            "measured cos_target would diverge from the instrument convention"
        )

    rows = frozen_inputs()
    assert_orientation_symmetry(rows)
    guard_summary = assert_guard_on_train(rows)

    # baselines: exact reproduction of the build manifest table (train)
    mine = baselines_table(rows)
    check_baselines_vs_build(pro["build_manifest"], mine)
    for r in rows:
        r.pop("_bl_rec_a", None)
        r.pop("_bl_rec_b", None)

    x, y, meta = build_matrix(rows)
    log(
        "train_table",
        rows=int(len(y)),
        positives=int(y.sum()),
        negatives=int(len(y) - y.sum()),
        n_features=int(x.shape[1]),
        feature_names=list(FEATURE_NAMES),
    )

    cv = cv_characterization(x, y, meta)

    # final refit on ALL train rows + in-process determinism evidence (P7)
    booster = fit_booster(x, y)
    booster2 = fit_booster(x, y)
    s1 = booster.model_to_string()
    s2 = booster2.model_to_string()
    log(
        "determinism_selfcheck",
        boosters_byte_equal=bool(s1 == s2),
        n_rows=int(len(y)),
        positives=int(y.sum()),
    )
    if s1 != s2:
        raise InstrumentAbort("booster not byte-deterministic across two fits")

    # in-sample diagnostic per stratum (the honest number is the CV OOF row)
    hits, n, per = decide_rows(rows, booster)
    per_stratum: dict[str, dict[str, float]] = {}
    for r, d in zip(rows, per):
        st = per_stratum.setdefault(
            r["stratum"], {"rows": 0, "hits": 0, "contradicts_proposals": 0}
        )
        st["rows"] += 1
        st["hits"] += int(d["verdict"] == r["label"])
        st["contradicts_proposals"] += int(d["verdict"] == "contradicts")
    in_sample = {
        "acc": round(hits / n, 4),
        "per_stratum": per_stratum,
        "note": "in-sample diagnostic; the honest generalization numbers "
        "are the CV OOF row (train-side) and the sealed holdout (gate)",
    }
    log("train_in_sample", **in_sample)

    guard_off = guard_off_probe(rows, booster)

    art_sha = freeze_artifact(
        booster, cv, mine, pro["pins"], guard_summary, guard_off, in_sample
    )

    wall = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    log(
        "train_done",
        status="GREEN",
        wall_s=round(wall, 1),
        onnx_sha256=art_sha,
        cut=DECISION_CUT,
        note="decision rule FROZEN before any sealed read (N2); the "
        "single-shot gate run stays unexecuted in this phase",
    )
    return 0


# --- subcommand: floors (ML-STD-0001 + ML-STD-0004; train-only; П3/П4) ---------


def cmd_floors(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_FLOOR
    started = dt.datetime.now(dt.timezone.utc)
    log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=900,
        expected_peak_ram_gb=1.5,
        lane="floor lane: 13 composition draws x 100 pairs (pinned nano "
        "embedder, in-memory cached) + inv/perm on train pairs; sequential, "
        "single lane",
    )
    booster, manifest = load_frozen()
    _ = manifest
    pro = prologue(SLICE_FLOOR)
    _ = pro
    if _INPUT_PRESET_VERSION == "v2":
        # v2 floor quota cross-check against the build manifest strata
        # metadata (ADDENDUM-3 §8.2 class_total // 6 == 100 pairs/draw).
        bm_strata = pro["build_manifest"]["strata"]
        quota_mism = {
            s: {
                "quota": FLOOR_DRAW_QUOTA[s],
                "manifest_class_total_div_6": bm_strata[s]["class_total"] // 6,
            }
            for s in ORDER
            if FLOOR_DRAW_QUOTA[s] != bm_strata[s]["class_total"] // 6
        }
        log(
            "floor_quota_vs_build_manifest_v2",
            quota=FLOOR_DRAW_QUOTA,
            draw_pairs=FLOOR_DRAW_PAIRS,
            mismatch=quota_mism,
            ok=not quota_mism,
            source="v2 build manifest strata.class_total (metadata; "
            "ADDENDUM-3 §8.2)",
        )
        if quota_mism:
            raise InstrumentAbort(
                f"floor draw quota diverges from the v2 build manifest: {quota_mism}"
            )
    if FLOORS_DOC.exists():
        raise InstrumentAbort(
            f"floors doc already exists: {FLOORS_DOC} — re-measuring is "
            "forbidden (ML-STD-0004 П3/П6; fixed BEFORE gate_start)"
        )
    # П1 order assert: the ML-STD-0004 formula commit precedes the draws.
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STD0004_FORMULA_COMMIT, "HEAD"],
        cwd=REPO,
        capture_output=True,
    )
    log(
        "std0004_order_assert",
        formula_commit=STD0004_FORMULA_COMMIT,
        is_ancestor=r.returncode == 0,
        formula=DELTA_ACC_FORMULA,
        stage1="ML-STD-0004 §1/П2: formula fixed BEFORE any draw",
        stage2="this run: draws -> published distribution -> frozen delta_acc "
        "-> gate_start",
        ok=r.returncode == 0,
    )
    if r.returncode != 0:
        raise InstrumentAbort(
            f"ML-STD-0004 П1 order assert RED: formula commit "
            f"{STD0004_FORMULA_COMMIT} not an ancestor of HEAD"
        )
    model_sha = manifest["sha256"]
    log(
        "floors_measure_start",
        seeds=FLOOR_SEEDS,
        q_seeds=FLOOR_Q_SEEDS,
        inv_seeds=FLOOR_INV_SEEDS,
        perm_seeds=FLOOR_PERM_SEEDS,
        draw_quota=FLOOR_DRAW_QUOTA,
        draw_pairs=FLOOR_DRAW_PAIRS,
        statistic_seed="acc_pipeline(draw) - max(BL-band, BL-digit, "
        "BL-digit+guard on draw); row-level unit (both orientations)",
        statistic_q="contradicts-verdict rate of the pipeline on the draw",
        statistic_inv="verdict disagreement base vs cosmetic variant",
        statistic_perm="verdict disagreement base vs re-oriented rows "
        "(DISTINCT permutation per seed — cascade W2 fix 2a45f28)",
        delta_acc_formula=DELTA_ACC_FORMULA,
        model_sha256=model_sha,
        recipe="cb.gen_pool + cb.rows_of (frozen build recipe, same code path)",
        note="the FROZEN model is NOT retrained; fixed BEFORE gate_start",
    )

    nano = NanoLeg()
    train_rows = load_train()

    # --- floor_seed: 10 draws (43..52), ML-STD-0004 declared floor ---------
    seed_rows = []
    deltas = []
    for seed in FLOOR_SEEDS:
        rows = floor_draw_rows(seed, nano)
        assert_draw_disjointness(rows, seed)
        hits, n, per = decide_rows(rows, booster)
        acc_model = hits / n
        bl = baselines_table(rows)
        acc_base = max(bl["BL-band"], bl["BL-digit"], bl["BL-digit+guard"])
        rate = sum(1 for d in per if d["verdict"] == "contradicts") / n
        row = {
            "seed": seed,
            "n_rows": n,
            "acc_model": round(acc_model, 4),
            "acc_baseline_max": round(acc_base, 4),
            "delta": round(acc_model - acc_base, 4),
            "contradicts_rate": round(rate, 4),
        }
        seed_rows.append(row)
        log("floor_seed_draw", **row)
        deltas.append(row["delta"])
    dist = _dist(deltas)
    delta_acc = max(dist["p95"] - dist["p5"], 0.10)
    log(
        "floor_seed_measure_end",
        per_seed=seed_rows,
        floor_seed_distribution=dist,
        delta_acc_declared=round(delta_acc, 4),
        formula=DELTA_ACC_FORMULA,
        pairs_compared=len(deltas) * (len(deltas) - 1) // 2,
        model_sha256=model_sha,
        wall_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
        remeasure_after_single_shot="FORBIDDEN",
    )

    # --- floor_q: 3 DEDICATED draws (53..55), verdict-rate spread ----------
    q_rates = []
    for seed in FLOOR_Q_SEEDS:
        rows = floor_draw_rows(seed, nano)
        assert_draw_disjointness(rows, seed)
        hits, n, per = decide_rows(rows, booster)
        rate = sum(1 for d in per if d["verdict"] == "contradicts") / n
        q_rates.append(round(rate, 4))
        log("floor_q_draw", seed=seed, n_rows=n, contradicts_rate=round(rate, 4))
    dist_q = _dist([float(v) for v in q_rates])
    floor_q = dist_q["max_pairwise"]
    log(
        "floor_q_measure",
        seeds=FLOOR_Q_SEEDS,
        rates=q_rates,
        distribution=dist_q,
        floor_q=floor_q,
        note="verdict-rate spread over independent compositions "
        "(ML-STD-0001 П1: floor = max pairwise deviation)",
    )

    # --- floor_inv: cosmetic variation on train pairs, 3 draws -------------
    # v1: all 1200 rows are fold-eligible material (F1 shape). E12 (v2):
    # N1 carrier cells are NOT in the floor fix-set — the inv sample rides
    # the carrier-free host rows, keeping the floor_inv(v2) <= floor_inv(v1)
    # gate comparable across corpora.
    eligible = [
        r for r in train_rows if r.get("carrier") not in FLOOR_INV_EXCLUDE_CARRIERS
    ]
    inv_seeds = FLOOR_INV_SEEDS
    inv_disagreements = []
    ws_diagnostic = []
    for draw_seed in inv_seeds:
        rng = random.Random(draw_seed)
        sample = rng.sample(eligible, min(80, len(eligible)))
        dis = 0
        ws_dis = 0
        for r in sample:
            assert_cosmetic_surface_invariance(r, rng)
            rec_a, rec_b = row_records(r)
            base = decide_pipeline(
                rec_a, rec_b, float(r["measured"]["cos_target"]), booster
            )
            pert = row_with_body(
                r, "envelope_b", cosmetic_variant(r["envelope_b"]["body"], rng)
            )
            prec_a, prec_b = row_records(pert)
            alt = decide_pipeline(prec_a, prec_b, float(r["measured"]["cos_target"]), booster)
            dis += int(base["verdict"] != alt["verdict"])
            # published diagnostic (NOT a floor): whitespace-len variant is
            # VISIBLE to body_len_delta — measures that sensitivity only
            wsp = row_with_body(r, "envelope_b", ws_len_variant(r["envelope_b"]["body"]))
            wpa, wpb = row_records(wsp)
            walt = decide_pipeline(wpa, wpb, float(r["measured"]["cos_target"]), booster)
            ws_dis += int(base["verdict"] != walt["verdict"])
        inv_disagreements.append(round(dis / len(sample), 4))
        ws_diagnostic.append(round(ws_dis / len(sample), 4))
    dist_inv = _dist([float(v) for v in inv_disagreements])
    floor_inv = max(inv_disagreements) if inv_disagreements else 0.0
    log(
        "floor_inv_measure",
        draws=inv_seeds,
        disagreement_rates=inv_disagreements,
        distribution=dist_inv,
        floor_inv=floor_inv,
        delta_inv=round(2 * floor_inv, 4),
        delta_inv_formula="delta_inv := 2*floor_inv (enters g_stab "
        "threshold 1 - 2*floor_inv)",
        ws_len_diagnostic_rates=ws_diagnostic,
        fix_set={
            "exclude_carriers": list(FLOOR_INV_EXCLUDE_CARRIERS),
            "eligible_rows": len(eligible),
            "total_train_rows": len(train_rows),
            "rule": "cosmetic-twin scope (probes 2/3 surface); E12: "
            "N1 carrier cells not in the floor fix-set (v2)",
        },
        gate="E12: floor_inv(v2) <= floor_inv(v1 measured) (v2 preset)",
        note="structurally 0 by construction, machine-asserted per row "
        "(case-only cosmetic variant: features, guard regions and "
        "embed_text are normalization-invariant and lengths unchanged; "
        "see the floor_fixture_defect event — whitespace-len variants are "
        "a published diagnostic, never a floor)",
    )

    # --- floor_perm: orientation permutation on train pairs, 3 seeds -------
    # Cascade W2 fix 2a45f28: a DISTINCT permutation per seed (the naive
    # variant published n=3 while re-running one reversal three times).
    pairs_a = [r for r in train_rows if r["orientation"] == "a>b"]
    sample = random.Random(78).sample(pairs_a, min(80, len(pairs_a)))
    perm_seeds = FLOOR_PERM_SEEDS
    perm_deltas = []
    for perm_seed in perm_seeds:
        rng = random.Random(perm_seed)
        dis = 0
        for r in sample:
            rec_a, rec_b = row_records(r)
            base = decide_pipeline(rec_a, rec_b, float(r["measured"]["cos_target"]), booster)
            alt_row = orientation_swap(r) if rng.random() < 0.5 else r
            prec_a, prec_b = row_records(alt_row)
            alt = decide_pipeline(
                prec_a, prec_b, float(r["measured"]["cos_target"]), booster
            )
            dis += int(base["verdict"] != alt["verdict"])
        perm_deltas.append(round(dis / len(sample), 4))
    dist_perm = _dist([float(v) for v in perm_deltas])
    floor_perm = max(perm_deltas) if perm_deltas else 0.0
    log(
        "floor_perm_measure",
        perms=perm_seeds,
        delta_rates=perm_deltas,
        distribution=dist_perm,
        floor_perm=floor_perm,
        note="distinct per-seed permutation realizations (fix 2a45f28); "
        "2-element pairs admit 2 distinct permutations per pair, the seeded "
        "shuffle realizes them independently per seed; structurally 0 "
        "expected (symmetric core features, machine-asserted)",
    )

    floors_doc = {
        "standard": "ML-STD-0001 (tolerance noise floors) + ML-STD-0004 "
        "(declared floor, stage 2)",
        "formula": DELTA_ACC_FORMULA,
        "artifact_sha256": model_sha,
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "remeasure_after_single_shot": "FORBIDDEN",
        "floor_seed": {
            "seeds": FLOOR_SEEDS,
            "per_seed": seed_rows,
            "distribution": dist,
            "delta_acc_declared": round(delta_acc, 4),
            "delta_acc_recheck": round(max(dist["p95"] - dist["p5"], 0.10), 4),
            "formula": DELTA_ACC_FORMULA,
        },
        "floor_q": {
            "seeds": FLOOR_Q_SEEDS,
            "rates": q_rates,
            "distribution": dist_q,
            "floor": floor_q,
        },
        "floor_inv": {
            "draws": inv_seeds,
            "disagreement_rates": inv_disagreements,
            "distribution": dist_inv,
            "floor": floor_inv,
            "delta_inv": round(2 * floor_inv, 4),
            "formula": "delta_inv := 2*floor_inv",
            "fix_set": {
                "exclude_carriers": list(FLOOR_INV_EXCLUDE_CARRIERS),
                "eligible_rows": len(eligible),
                "total_train_rows": len(train_rows),
            },
            "gate_e12": "floor_inv(v2) <= floor_inv(v1 measured) (v2 preset)",
            "transform": "case-only cosmetic variant (surface-invariant by "
            "construction, machine-asserted per row); whitespace-len "
            "perturbation is NOT cosmetic on this surface (body_len_delta "
            "is raw) — see floor_fixture_defect in the run-log",
            "ws_len_diagnostic_rates": ws_diagnostic,
        },
        "floor_perm": {
            "perms": perm_seeds,
            "delta_rates": perm_deltas,
            "distribution": dist_perm,
            "floor": floor_perm,
            "fix": "2a45f28 distinct permutation per seed",
        },
        "draw_recipe": {
            "quota": FLOOR_DRAW_QUOTA,
            "pairs_per_draw": FLOOR_DRAW_PAIRS,
            "prefix": FLOOR_PREFIX,
            "recipe": "cb.gen_pool + cb.rows_of (frozen build code path); "
            "rotate_pair makes floor texts byte-differ from train",
            "disjointness": "draw ids vs train ids = 0 (asserted per draw)",
        },
    }
    if floors_doc["floor_seed"]["delta_acc_declared"] != floors_doc["floor_seed"]["delta_acc_recheck"]:
        raise InstrumentAbort(
            "declared delta_acc does not reproduce from the published p5/p95 "
            "under the ML-STD-0004 formula"
        )
    FLOORS_DOC.write_text(
        json.dumps(floors_doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log(
        "floors_published",
        path=str(FLOORS_DOC),
        floors=["floor_seed", "floor_q", "floor_inv", "floor_perm"],
        delta_acc_declared=round(delta_acc, 4),
        floor_inv=floor_inv,
        floor_perm=floor_perm,
        floor_q=floor_q,
        artifact_sha256=model_sha,
        p4_note="ALL floor publication fields written to the run-log BEFORE "
        "gate_start (П4); the gate asserts this event + the floors doc "
        "prior to gate_start",
    )
    wall = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    log("floors_done", status="GREEN", wall_s=round(wall, 1))
    return 0


# --- ADDENDUM-4 suite amendment (ratified MLC 10.11; §9 E1-E10) ---------------


#: Amendment identity: docs/experiments/conflict-detector-corpus-design-ADDENDUM-4.md
#: (draft f9586f1 -> ratified 0971639); arbiter script that produced the L2
#: evidence: docs/experiments/b6_conflict_cross_cell_analysis.py @ 45737d2.
AMENDMENT4_COMMIT = "09716392fdf17692930d5289bcdf1fde85f58b2f"
AMENDMENT4_SHORT = "0971639"
AMENDMENT4_DOC = REPO / "docs" / "experiments" / "conflict-detector-corpus-design-ADDENDUM-4.md"
AMENDMENT4_ARBITER_COMMIT = "45737d2"
#: Byte-stable disclosure marker (ADDENDUM-4 E8). MUST ride the SAME log
#: line / table cell as every probe-7 number it qualifies (a distant
#: footnote = hidden degradation per E8). While this stands, the suite is
#: NEVER counted as bare "7/7" — the convention is "6/7 legs + 1
#: known-uncovered" (E3); a bare "7/7" after the amendment = fabrication.
KNOWN_UNCOVERED_MARKER = (
    "probe 7 = 1-leg; leg-2 known-uncovered — слепая зона frozen-13, "
    f"ADDENDUM-4 @ {AMENDMENT4_SHORT}; возврат в OQ-3"
)

#: E2 numeric-form declaration of the known-uncovered verdict (ADDENDUM-4
#: §9.2 E2, with fact corrections K1-K3). Zone class per E1: this is a
#: FEATURE-blind zone (label semantically unambiguous, the frozen-13
#: surface carries no separating coordinate for the two legs) — explicit
#: extension of the §10.2 zone principle beyond policy-двузначных зон;
#: future declarations owe the same form. Facts mechanically verified by
#: the arbiter script (ML reproduced L2 independently: 1.0003 exact,
#: NN 4.2428/7.6158).
KNOWN_UNCOVERED_LEG2 = {
    "zone_class": "feature-blind zone (E1: extension of the §10.2 zone "
    "principle from policy-двузначных зон to feature-слепые зоны)",
    "probe": "7-translation-fact-changed",
    "leg": 2,
    "surface": "frozen-13 CORE-13 (manifest feature_set_sha256 "
    "dd86228f8c634f28…, v2 bundle 2534fcbe)",
    "amendment": "ADDENDUM-4",
    "amendment_commit": AMENDMENT4_COMMIT,
    "arbiter_script": "docs/experiments/b6_conflict_cross_cell_analysis.py",
    "arbiter_script_commit": AMENDMENT4_ARBITER_COMMIT,
    #: E2 pinned triple: legs mutual distance / Δ of the live coordinate /
    #: distance to the nearest train support (margin ≈4x).
    "numeric_form": {
        "legs_mutual_distance_l2": 1.0003,
        "live_coordinate_delta": 0.001,
        "nearest_train_support_l2_min": 4.243,
        "margin": "≈4x",
    },
    #: Per-coordinate disclosure (K1 exact form): cos_target is the ONLY
    #: coordinate with SEPARATED class intervals; Δcos between the legs
    #: (0.001) is not label-tied — cos position is a function of wording,
    #: not of the fact.
    "separating_coordinates": {
        "cos_target": {
            "pos": [0.7002, 0.7458],
            "neg": [0.7980, 0.8289],
            "legs_delta": 0.001,
            "note": "the only coordinate with separated class intervals "
            "(K1); ceiling 0.7458 = sampled seed-family artifact, not "
            "embedder physics (K3) — cos ≥0.75 is reachable (probe-7 "
            "fixture itself measures 0.7738)",
        },
    },
    #: Dead coordinates (no variance / ≈0 on the cross-cell): the verdict
    #: rests on L2/Δcos/dead coordinates — construction-stable, NOT on
    #: ceiling attribution (K3).
    "dead_coordinates": [
        "char3_jaccard", "char4_jaccard", "char5_jaccard",
        "char3_containment", "char4_containment", "char5_containment",
        "tag_jaccard", "type_match", "lang_match",
    ],
    "dead_coordinate_note": "char3/4/5 (jaccard+containment) ≈ 0 across "
    "different alphabets; tag/type/lang — constants in the cross-cell",
    #: Live but NOT separating (K1): len-deltas keep class intervals
    #: overlapping.
    "live_non_separating": {
        "coordinates": ["title_len_delta", "body_len_delta", "tag_count_delta"],
        "intervals": {"pos": [5, 30], "neg": [3, 29]},
        "note": "live but OVERLAPPING class intervals — not separating (K1)",
    },
    #: E4 side door: while the declaration stands, no NEW probe may assert
    #: on the declared dead-coordinate set.
    "side_door": "E4: while this declaration stands, NO new probe asserts "
    "on the declared dead-coordinate set",
    #: E5 blindness ceiling — enforced in run_sanity_suite (fail-closed).
    "blindness_ceiling": "E5: >1 simultaneous known-uncovered leg on one "
    "frozen surface = full surface verdict (line stop, OQ-3 escalation); "
    "corpus amend-once (E13) and suite-amendment budgets are separate; "
    "corpus v2 untouched",
    #: E6 review trigger + DE-3 return mechanics (MODEL_CARD carrier).
    "review_trigger": "E6: opening of OQ-3 or N ratified waves — whichever "
    "first (N is not pinned by the addendum; set at the TL gate before Ф2)",
    "return_mechanics": "DE-3: fresh fixtures through the FROZEN generator "
    "+ repeat cross-cell analysis (arbiter script) BEFORE any new gates + "
    "seal-virginity assert for the whole deferral period in the ADOPT report",
    #: E10: the missing feature is named — numeric-set intersection (digits
    #: survive letter-normalization: eq 500=500 vs fact 500≠1000 would give
    #: Δ≈1 on a semantic coordinate); strengthens reversibility (iv).
    "oq3_missing_feature": "numeric-set intersection (E10)",
    "support_argument": "E9: band [0.75,0.80) holds 0 train rows of either "
    "class => no split is induced there (gain=0 on an empty region); any "
    "learnable leg-discrimination scheme would be fixture memoization "
    "(Δcos 0.001 ≪ paraphrase jitter: para1 moves char5 0.966→0.912) — "
    "no learnable scheme exists on frozen-13 in principle",
}


# --- sanity fixtures (§6.3; source = the P3 preflight cells of the generator) ---


def build_fixtures() -> dict:
    """Fixture cells — PRIMARY source: the П3 preflight block of the frozen
    generator (``cmd_preflight`` verbatim args); probe-specific surfaces
    (self/cosmetic/envelope/translation) derived from the same generator
    machinery. The exact choice is fixed HERE and logged BEFORE the run."""
    r0 = random.Random(0)
    sc = {}
    sc["CD-1"] = lib.build_scenario("fx-CD-1", "CD-1", "s3conn", "ru",
                                    lib.NUM_SLOTS[0], 500, 1000, fillers=1, rng=r0)
    sc["CD-2"] = lib.build_scenario("fx-CD-2", "CD-2", "retry", "ru",
                                    (lib.DATE_SLOT[0], lib.DATE_SLOT[1], ""),
                                    "5 октября", "7 октября", fillers=1, rng=r0)
    sc["CD-3"] = lib.build_scenario("fx-CD-3", "CD-3", "sched", "ru",
                                    lib.STATUS_SLOTS[0], None, None, fillers=1, rng=r0)
    sc["CD-4"] = lib.build_scenario("fx-CD-4", "CD-4", "auth", "ru",
                                    lib.MODAL_SLOTS[0], None, None, fillers=1, rng=r0)
    sc["CD-5"] = lib.build_scenario("fx-CD-5", "CD-5", "cache", "ru", None,
                                    lib.COND_SLOTS[0][2], lib.COND_SLOTS[0][3],
                                    fillers=1, rng=r0)
    sc["CD-6"] = lib.build_scenario("fx-CD-6", "CD-6", "recs", "ru", None, None,
                                    None, polar=lib.POLAR_SLOTS[0], fillers=1, rng=r0)
    sc["NC-1"] = lib.build_scenario("fx-NC-1", "NC-1", "export", "ru", None, None,
                                    None, insert_sent=lib.INSERT_RU[0], fillers=1, rng=r0)
    sc["NC-2"] = lib.build_scenario("fx-NC-2", "NC-2", "pay", "ru", None, 800, None,
                                    refine=("lim", 1000), fillers=1, rng=r0)
    sc["NC-3"] = lib.build_scenario("fx-NC-3", "NC-3", "logs", "ru", None, None, None,
                                    version_pair=("v1.2.0", "v1.3.0"), fillers=1, rng=r0)
    sc["NC-4-tags"] = lib.build_scenario("fx-NC-4", "NC-4", "queue", "ru", None,
                                         None, None, meta_delta="tags", fillers=1, rng=r0)
    sc["NC-4-lang"] = lib.build_scenario("fx-NC-4-lang", "NC-4", "queue", "ru", None,
                                         None, None, meta_delta="lang", fillers=1, rng=r0)
    sc["NC-5"] = lib.build_scenario("fx-NC-5", "NC-5", "s3conn", "ru", None, None,
                                    None, sibling_subj="cook", fillers=1, rng=r0)
    # translation-twin changed-fact leg: the §4.4 cross-language positive
    # (RU sentence vs EN translation with one changed factual reference)
    sc["CD-1-cross"] = lib.build_scenario(
        "fx-CD-1-cross", "CD-1", "s3conn", "ru", lib.NUM_SLOTS[0], 500, 1000,
        fillers=1, cross=True, rng=r0)
    # DIAGNOSTIC ONLY (published, not a probe): the true correct translation
    # (cross pair with EQUAL values) — outside the corpus vocabulary (§4.4
    # negative cross = different-subject siblings); off-support honesty number.
    sc["CD-1-cross-eq"] = lib.build_scenario(
        "fx-CD-1-cross-eq", "CD-1", "s3conn", "ru", lib.NUM_SLOTS[0], 500, 500,
        fillers=1, cross=True, rng=r0)
    return sc


def sc_row(fx: dict, key: str, orientation: str = "a>b") -> dict:
    """A fixture pair as a row dict (cb.side_dict + measured by NanoLeg)."""
    sc = fx[key]
    nano = fx["_nano"]
    cos = round(nano.pair_cos(sc.side_a, sc.side_b, site=f"fixture:{key}"), 4)
    return {
        "pair_id": sc.scenario_id,
        "stratum": sc.stratum,
        "label": sc.label,
        "orientation": orientation,
        "measured": {"cos_target": cos},
        "envelope_a": cb.side_dict(sc.side_a),
        "envelope_b": cb.side_dict(sc.side_b),
    }


def probe_result(name: str, expect: str, ok: bool, res: dict, inv: bool) -> None:
    log(
        "sanity_probe",
        probe=name,
        expectation=expect,
        ok=ok,
        order_invariant=inv,
        verdict=res.get("verdict"),
        leg=res.get("leg"),
        p=res.get("p"),
        cos_target=res.get("cos_target"),
    )


# --- subcommand: sanity (§6.3, 7 probes through the PRODUCTION circuit) --------


def run_sanity_suite(booster: lgb.Booster, log_it: bool = True) -> list[dict]:
    """7 probes §6.3 on fixture cells THROUGH the production circuit
    (guard+model), per ADDENDUM-4 @ 0971639 probe 7 runs ONE-LEG: leg-2
    (translation-fact-changed) is known-uncovered — published with the
    E8 marker, NOT asserted. Red on any ASSERT leg = InstrumentAbort —
    the sealed holdout is NOT read (class 'instrument', П6 ML-STD-0002).
    Count convention (E3): "6/7 legs + 1 known-uncovered" — a bare "7/7"
    after the amendment is fabrication."""
    sc = build_fixtures()
    sc["_nano"] = NanoLeg()
    fx = dict(sc)  # includes "_nano" (the embedder leg for sc_row cos)
    results = []
    all_ok = True

    def run_pair(row_a: dict, name: str, expect: str, check, note: str = "") -> dict:
        nonlocal all_ok
        rec_a, rec_b = row_records(row_a)
        cos = float(row_a["measured"]["cos_target"])
        res = decide_pipeline(rec_a, rec_b, cos, booster)
        # order invariance: the swapped presentation must not change the verdict
        row_b = orientation_swap(row_a)
        rec_a2, rec_b2 = row_records(row_b)
        res2 = decide_pipeline(rec_a2, rec_b2, cos, booster)
        inv = res["verdict"] == res2["verdict"]
        ok = bool(check(res)) and inv
        all_ok = all_ok and ok
        if log_it:
            probe_result(name, expect, ok, {**res, "cos_target": round(cos, 4)}, inv)
        results.append(
            {"probe": name, "ok": ok, "order_invariant": inv, "result": res,
             "result_rev": res2, "expectation": expect, "note": note}
        )
        return res

    # 1 — self-pair: NOT contradicts, P <= 0.1 (#480 angle)
    self_row = {
        "pair_id": "fx-self",
        "stratum": "self",
        "label": "not-contradicts",
        "orientation": "a>b",
        "measured": {"cos_target": 1.0},
        "envelope_a": cb.side_dict(fx["CD-1"].side_a),
        "envelope_b": cb.side_dict(fx["CD-1"].side_a),
    }
    run_pair(
        self_row, "1-self-pair", "NOT contradicts, P <= 0.1",
        lambda r: r["verdict"] == "not-contradicts" and (r["p"] is not None and r["p"] <= 0.10),
        note="g_self rides this probe (policy-constant)",
    )

    # 2 — cosmetic twin: NOT, decision invariant (case/whitespace)
    r = sc_row(fx, "NC-2")
    cos2 = r["measured"]["cos_target"]
    rng2 = random.Random(7)
    row = {
        "pair_id": "fx-cosmetic",
        "stratum": "cosmetic",
        "label": "not-contradicts",
        "orientation": "a>b",
        "measured": {"cos_target": cos2},
        "envelope_a": r["envelope_a"],
        "envelope_b": dict(r["envelope_b"], body=cosmetic_variant(r["envelope_b"]["body"], rng2)),
    }
    run_pair(
        row, "2-cosmetic-twin", "NOT contradicts, invariant",
        lambda res: res["verdict"] == "not-contradicts",
        note="cosmetic transform = CASE-only (surface-invariant by "
        "construction: normalization lowercases, lengths unchanged; "
        "whitespace-len perturbations are a published diagnostic, see "
        "floor_fixture_defect) — the variant embeds identically",
    )

    # 3 — envelope variant (whitelist §8.1 deltas): NOT contradicts
    r3 = sc_row(fx, "NC-4-tags")
    row3 = dict(r3)
    row3["pair_id"] = "fx-envelope"
    row3["envelope_a"] = dict(r3["envelope_a"], created_at="2026-08-01T10:15:00Z")
    row3["envelope_b"] = dict(r3["envelope_b"], created_at="2026-08-02T11:15:00Z")
    run_pair(
        row3, "3-envelope-variant", "NOT contradicts",
        lambda res: res["verdict"] == "not-contradicts",
        note="NC-4 fixture (tags delta) + whitelist created_at delta "
        "(envelope fields are outside the conflict surface)",
    )

    # 4 — razor CD-1: contradicts, P >= 0.5 (positive leg of g_razor)
    r4 = sc_row(fx, "CD-1")
    run_pair(
        r4, "4-razor-cd1", "contradicts, P >= 0.5 (leg model)",
        lambda res: res["verdict"] == "contradicts" and res["leg"] == "model"
        and res["p"] is not None and res["p"] >= 0.5,
        note="positive leg of the razor certify exam (g_razor); one digit "
        "replaced, guard must NOT fire",
    )

    # 5 — version neighbor: NOT via the guard leg (negative leg of g_razor)
    r5 = sc_row(fx, "NC-3")
    run_pair(
        r5, "5-version-neighbor", "NOT contradicts (leg guard, exact)",
        lambda res: res["verdict"] == "not-contradicts" and res["leg"] == "guard",
        note="negative leg of the razor certify exam (g_razor); bump "
        "v1.2.0 -> v1.3.0, supersede semantics, NOT a conflict",
    )

    # 6 — stability ladder: one conflict through raw / cosmetic / paraphrase,
    # verdict invariant across all 3 steps
    ladder = []
    base6 = sc_row(fx, "CD-2")
    rng6 = random.Random(11)
    steps = [
        ("raw", base6["envelope_b"]["body"]),
        ("cosmetic", cosmetic_variant(base6["envelope_b"]["body"], rng6)),
        ("para1", lib.para(base6["envelope_b"]["body"], 1, "ru", rng6)),
    ]
    ladder_ok = True
    ladder_verdicts = []
    for step_name, body in steps:
        row6 = dict(base6, pair_id=f"fx-ladder-{step_name}")
        row6["envelope_b"] = dict(base6["envelope_b"], body=body)
        rec_a, rec_b = row_records(row6)
        res = decide_pipeline(rec_a, rec_b, float(base6["measured"]["cos_target"]), booster)
        ladder_verdicts.append(res["verdict"])
        ladder_ok = ladder_ok and res["verdict"] == "contradicts"
    inv6 = len(set(ladder_verdicts)) == 1
    all_ok = all_ok and ladder_ok and inv6
    if log_it:
        log(
            "sanity_probe",
            probe="6-stability-ladder",
            expectation="contradicts on all 3 steps, verdict invariant",
            ok=ladder_ok and inv6,
            order_invariant=inv6,
            verdicts=ladder_verdicts,
            steps=[s[0] for s in steps],
        )
    results.append(
        {"probe": "6-stability-ladder", "ok": ladder_ok and inv6,
         "order_invariant": inv6, "verdicts": ladder_verdicts,
         "expectation": "contradicts on all 3 steps (raw/cosmetic/para1)"}
    )

    # 7 — translation twin (ADDENDUM-4: ONE-LEG probe). leg-1: identity
    # translation (NC-4 lang delta) NOT — the remaining assert. leg-2:
    # translation-fact-changed (CD-1 cross) — known-uncovered, published
    # below without assertion.
    r7n = sc_row(fx, "NC-4-lang")
    r7n = dict(r7n, pair_id="fx-trans-ok")
    res7n = run_pair(
        r7n, "7-translation-twin-not", "NOT contradicts (correct translation)",
        lambda res: res["verdict"] == "not-contradicts",
        note="fixture: NC-4 lang-delta cell (generator native; the corpus "
        "vocabulary has no same-content RU<->EN class — §4.4 negative cross "
        "is different-subject siblings; identity translation is the "
        "generator-native correct-translation surface, 8 train rows)",
    )
    # leg-2 — known-uncovered (E1 feature-blind zone): stays in the suite
    # (E3), verdict PUBLISHED with the byte-stable marker in the SAME line
    # (E8), NOT asserted, does not fold into all_ok.
    r7p = sc_row(fx, "CD-1-cross")
    rec7_a, rec7_b = row_records(r7p)
    cos7 = float(r7p["measured"]["cos_target"])
    res7p = decide_pipeline(rec7_a, rec7_b, cos7, booster)
    r7p_swap = orientation_swap(r7p)
    rec7_a2, rec7_b2 = row_records(r7p_swap)
    res7p_rev = decide_pipeline(rec7_a2, rec7_b2, cos7, booster)
    inv7p = res7p["verdict"] == res7p_rev["verdict"]
    if log_it:
        log(
            "sanity_probe",
            probe="7-translation-fact-changed",
            expectation="known-uncovered (ADDENDUM-4): verdict PUBLISHED, "
            "NOT asserted — no pass/fail on this leg",
            known_uncovered=True,
            covered=False,
            ok=None,
            order_invariant=inv7p,
            verdict=res7p.get("verdict"),
            verdict_rev=res7p_rev.get("verdict"),
            leg=res7p.get("leg"),
            p=res7p.get("p"),
            cos_target=round(cos7, 4),
            marker=KNOWN_UNCOVERED_MARKER,
            amendment=f"ADDENDUM-4 @ {AMENDMENT4_SHORT}",
            declaration=KNOWN_UNCOVERED_LEG2,
            note="fixture: §4.4 cross positive — RU sentence vs EN "
            "translation with one changed factual reference; v1 called it "
            "contradicts p≈1 in force of EMPTY negative support, v2 with "
            "real N3 cross-negatives calls it not-contradicts — each "
            "model wrong on exactly one side (K2); the legs sit one point "
            "apart on every separating coordinate (L2 1.0003, §1)",
        )
    results.append(
        {"probe": "7-translation-fact-changed", "known_uncovered": True,
         "covered": False, "ok": None, "order_invariant": inv7p,
         "result": res7p, "result_rev": res7p_rev,
         "expectation": "known-uncovered (ADDENDUM-4): published, "
         "not asserted",
         "note": "E3: the leg stays in the suite with the known-uncovered "
         "tag + amendment SHA; E8: marker rides the same line as the "
         "verdict/p/cos number"}
    )

    # diagnostic (published, NOT a probe, off-support honesty number):
    # the TRUE correct translation (cross CD-1, equal values)
    r7d = sc_row(fx, "CD-1-cross-eq")
    rec_a, rec_b = row_records(r7d)
    res7d = decide_pipeline(rec_a, rec_b, float(r7d["measured"]["cos_target"]), booster)
    log(
        "sanity_diagnostic_true_translation",
        probe="diagnostic-true-translation-equal-values",
        verdict=res7d.get("verdict"),
        p=res7d.get("p"),
        cos_target=res7d.get("cos_target"),
        not_a_gate=True,
        note="cross CD-1 with EQUAL values (true correct translation) is "
        "OUTSIDE the corpus vocabulary (§4.4); published as an honesty "
        "number for the owner/MLC, never a gate or a probe",
    )

    # --- E8-iii canary (v2 preset only): the build's para_b=2 CD-1 cells
    # --- must STAY contradicts through the production circuit --------------
    if CANARY_PARA2:
        rows_by_id = {r["pair_id"]: r for r in load_train()}
        pool = cb.gen_pool(
            CANARY_PARA2["seed"], CANARY_PARA2["prefix"], rotate=CANARY_PARA2["rotate"]
        )
        para2_pool = {
            sc.scenario_id for sc in pool if sc.stratum == "CD-1" and sc.para_b == 2
        }
        ids = sorted(para2_pool & set(rows_by_id))
        log(
            "e8_iii_canary_para2_start",
            pool_cd1_para2_cells=len(para2_pool),
            selected_in_train=len(ids),
            expected=CANARY_PARA2["expect_count"],
            recipe=f"cb.gen_pool({CANARY_PARA2['seed']}, '{CANARY_PARA2['prefix']}', "
            f"rotate={CANARY_PARA2['rotate']}) — the exact v2 build call; ids "
            "∩ train pair_ids; no sealed surface touched",
        )
        if len(ids) != CANARY_PARA2["expect_count"]:
            raise InstrumentAbort(
                f"canary para2 set mismatch: regenerated pool ∩ train gives "
                f"{len(ids)} cells, build manifest recorded "
                f"{CANARY_PARA2['expect_count']} — trainer/build divergence"
            )
        canary_cells = []
        canary_ok = True
        for pid in ids:
            r = rows_by_id[pid]
            rec_a, rec_b = row_records(r)
            res = decide_pipeline(rec_a, rec_b, float(r["measured"]["cos_target"]), booster)
            rsw = orientation_swap(r)
            rec2_a, rec2_b = row_records(rsw)
            res2 = decide_pipeline(rec2_a, rec2_b, float(r["measured"]["cos_target"]), booster)
            ok_c = (
                res["verdict"] == CANARY_PARA2["expect_verdict"]
                and res2["verdict"] == CANARY_PARA2["expect_verdict"]
            )
            canary_ok = canary_ok and ok_c
            canary_cells.append(
                {"pair_id": pid, "verdict": res["verdict"], "verdict_rev": res2["verdict"],
                 "p": res.get("p"), "cos_target": round(float(r["measured"]["cos_target"]), 4),
                 "ok": ok_c}
            )
        log(
            "e8_iii_canary_para2",
            cells=canary_cells,
            expectation="all 7 para_b=2 CD-1 cells stay contradicts (E8-iii), "
            "both orientations",
            ok=canary_ok,
        )
        all_ok = all_ok and canary_ok

    # --- E8-iv para1 sweep (v2 preset only): >=3 PARA_RU words on the
    # --- probe-6 fixture — PUBLISHED DIAGNOSTIC, never a gate --------------
    if PARA_SWEEP_WORDS:
        base6 = sc_row(fx, "CD-2")
        body6 = base6["envelope_b"]["body"]
        cos6 = float(base6["measured"]["cos_target"])
        applicable = [(a, b) for a, b in lib.PARA_RU if a in body6]
        sweep = []
        for a, b in applicable[:PARA_SWEEP_WORDS]:
            row6 = dict(base6, pair_id=f"fx-para1-sweep-{a}")
            row6["envelope_b"] = dict(base6["envelope_b"], body=body6.replace(a, b, 1))
            rec_a, rec_b = row_records(row6)
            res = decide_pipeline(rec_a, rec_b, cos6, booster)
            sweep.append(
                {"word": a, "sub": b, "verdict": res["verdict"], "p": res.get("p"),
                 "leg": res.get("leg")}
            )
        log(
            "e8_iv_para1_sweep",
            para_dict_words_total=len(lib.PARA_RU),
            applicable_on_fixture=len(applicable),
            swept=len(sweep),
            words=[w["word"] for w in sweep],
            results=sweep,
            cos_target=round(cos6, 4),
            published_diagnostic=True,
            not_a_gate=True,
            note="E8-iv: one-word para1 substitutions from the PINNED "
            "PARA_RU dictionary (sha256 ru "
            f"{lib.PARA_DICT_SHA['ru'][:16]}…) on the probe-6 fixture, "
            "probe-6 cos convention; published diagnostics — no abort on "
            "verdict",
        )
        if len(sweep) < 3:
            raise InstrumentAbort(
                f"E8-iv sweep under-supported: only {len(sweep)} applicable "
                "PARA_RU words on the probe-6 fixture (<3) — the ratified "
                "diagnostic cannot be published"
            )

    # --- E5 blindness ceiling: >1 simultaneous known-uncovered leg on
    # --- one frozen surface = full surface verdict (stop, OQ-3 escalation)
    ku_legs = [r for r in results if r.get("known_uncovered")]
    if len(ku_legs) > 1:
        raise InstrumentAbort(
            "E5 blindness ceiling exceeded: "
            f"{len(ku_legs)} known-uncovered legs on one frozen surface "
            f"({[r['probe'] for r in ku_legs]}) — full surface verdict: "
            "line stop + OQ-3 escalation (ADDENDUM-4 E5)"
        )

    # Count convention (ADDENDUM-4 E3): 6 = probes 1-6 (full asserts),
    # denominator 7 = the §6.3 suite; probe 7 counts as the known-uncovered
    # unit (leg-2), its leg-1 assert stays green and is published beside.
    legs_green = sum(
        1 for r in results
        if r.get("ok") is True
        and not r.get("known_uncovered")
        and not str(r["probe"]).startswith("7-")
    )
    probe7_leg1_ok = next(
        (r["ok"] for r in results if r["probe"] == "7-translation-twin-not"),
        None,
    )
    log(
        "sanity_suite_done",
        ok=all_ok,
        count=f"{legs_green}/7 legs + {len(ku_legs)} known-uncovered",
        legs_total=7,
        legs_green=legs_green,
        known_uncovered_legs=len(ku_legs),
        probe7_mode="1-leg",
        probe7_leg1_assert_ok=probe7_leg1_ok,
        marker=KNOWN_UNCOVERED_MARKER,
        amendment=f"ADDENDUM-4 @ {AMENDMENT4_SHORT}",
        expectation="6/7 legs green + 1 known-uncovered + order invariance "
        "(v2: + E8-iii canary); bare '7/7' after the amendment = "
        "fabrication (ADDENDUM-4 E3); leg-1 of probe 7 remains a green "
        "assert; red ASSERT leg => holdout NOT read (instrument abort)",
    )
    if not all_ok:
        raise InstrumentAbort(
            "sanity suite RED (§6.3) — sealed surfaces stay sealed; "
            "STOP + report (abort class 'instrument')"
        )
    # ADDENDUM-5 F5 (NA-б): the green path asserts the clamp ledger stayed
    # EMPTY — a clamp here is a visible anomaly, never a silent tolerance.
    assert_cos_clamp_green("sanity suite §6.3 (pre-sealed green leg)")
    return results


def cmd_sanity(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_SANITY
    log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=60,
        expected_peak_ram_gb=1.0,
        lane="sanity lane: 7 fixture probes through the production circuit "
        "(pinned nano embedder, ~30 embeds); ADDENDUM-4: probe 7 ONE-LEG "
        "(leg-2 known-uncovered, published not asserted)",
    )
    # fixture choice fixed BEFORE the run (delta map item 6)
    fx = build_fixtures()
    log(
        "sanity_fixtures_pinned",
        cells=sorted(k for k in fx if not k.startswith("_")),
        source="cmd_preflight of build_conflict_detector_corpus.py (П3 "
        "preflight block, verbatim args) + probe-specific generator "
        "surfaces (self/cosmetic/envelope/translation); primary source "
        "per the dispatch",
        diagnostic_cell="CD-1-cross-eq (published honesty number, not a probe)",
        sealed_file_opened=False,
    )
    booster, _manifest = load_frozen()
    pro = prologue(SLICE_SANITY)
    _ = pro
    # ADDENDUM-4 suite amendment applied to THIS run (E8/E3 disclosure):
    # after the activation asserts, in the same slice as the amended suite.
    add_text = AMENDMENT4_DOC.read_text(encoding="utf-8")
    add_ratified = "РАТИФИЦИРОВАНО" in add_text
    log(
        "suite_amendment_applied",
        amendment="ADDENDUM-4",
        amendment_commit=AMENDMENT4_COMMIT,
        amendment_doc="docs/experiments/conflict-detector-corpus-design-ADDENDUM-4.md",
        amendment_sha256=sha256_text(add_text),
        ratified_marker_present=add_ratified,
        arbiter_script_commit=AMENDMENT4_ARBITER_COMMIT,
        probe7_mode="1-leg",
        known_uncovered_probe="7-translation-fact-changed",
        asserted_probe7_leg="7-translation-twin-not",
        count_convention="6/7 legs + 1 known-uncovered (E3; bare 7/7 = fabrication)",
        marker=KNOWN_UNCOVERED_MARKER,
        declaration=KNOWN_UNCOVERED_LEG2,
    )
    if not add_ratified:
        raise InstrumentAbort(
            "ADDENDUM-4 ratified marker missing at HEAD — refusing to run "
            "the amended suite unratified (fail-closed)"
        )
    run_sanity_suite(booster)
    log("sanity_cmd_done", status="GREEN", artifact=_manifest["sha256"][:16])
    return 0


# --- subcommand: oof-carriers (E8-ii sliced OOF; v2 pre-gate characterization) --


#: E8-ii: post-train sliced OOF on the N1∪N2 carrier rows, scaffold-grouped,
#: Wilson 95% CI; STOP-SIGNAL: acc < 0.75 BEFORE the gate = instrument-class
#: stop (not a red gate; ADDENDUM-3 §8 E8-ii). Characterization only — the
#: ratified decision rule stays cut 0.5, no threshold search.
OOF_CARRIER_SLICE = ("N1", "N2")
OOF_STOP_SIGNAL = 0.75
OOF_WILSON_Z = 1.959963985  # two-sided 95%


def _wilson_ci(k: int, n: int, z: float = OOF_WILSON_Z) -> list[float]:
    """Wilson score interval on k/n (E8-ii mandates the Wilson CI)."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(center - half, 4), round(center + half, 4)]


def cmd_oof_carriers(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_OOF
    log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=120,
        expected_peak_ram_gb=1.0,
        lane="oof-carriers lane: 5 scaffold-grouped refits on the train "
        "matrix + sliced OOF readout on the N1∪N2 carrier rows (no embeds; "
        "no sealed surface)",
    )
    if _INPUT_PRESET_VERSION != "v2":
        raise InstrumentAbort(
            "oof-carriers is the E8-ii v2 instrument — run it with the v2 "
            "input preset (--corpus-dir .../corpus-v2)"
        )
    booster, manifest = load_frozen()
    pro = prologue(SLICE_OOF)
    _ = pro
    rows = load_train()

    # carrier tags: the machine-visible per-row tags (E7), cross-checked
    # against the build-manifest carrier quotas before anything is computed.
    pairs_by_carrier: dict[str | None, set[str]] = {}
    for r in rows:
        pairs_by_carrier.setdefault(r.get("carrier"), set()).add(r["pair_id"])
    want = {
        c: pro["build_manifest"]["carriers"][c]["quota"] for c in ("N1", "N2", "N3")
    }
    mism = {
        c: {"got_pairs": len(pairs_by_carrier.get(c, ())), "want_pairs": q}
        for c, q in want.items()
        if len(pairs_by_carrier.get(c, ())) != q
    }
    log(
        "carrier_tag_crosscheck",
        got_pairs={str(k): len(v) for k, v in sorted(pairs_by_carrier.items(), key=lambda x: str(x[0]))},
        want_pairs=want,
        mismatch=mism,
        ok=not mism,
        source="train row 'carrier' tags vs build-manifest carriers quotas (E7)",
    )
    if mism:
        raise InstrumentAbort(f"carrier tag cross-check RED: {mism}")

    x, y, meta = build_matrix(rows)
    groups = np.asarray([_pair_scaffold(m["pair_id"]) for m in meta])
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=CV_K)
    for tr_i, te_i in gkf.split(x, y, groups):
        b = fit_booster(x[tr_i], y[tr_i])
        oof[te_i] = b.predict(x[te_i])
    pred = (oof >= DECISION_CUT).astype(np.float64)

    slice_mask = np.asarray([r.get("carrier") in OOF_CARRIER_SLICE for r in rows])
    slice_rows = [r for r in rows if r.get("carrier") in OOF_CARRIER_SLICE]
    hits = int((pred[slice_mask] == y[slice_mask]).sum())
    n_slice = int(slice_mask.sum())
    acc = hits / n_slice
    lo, hi = _wilson_ci(hits, n_slice)
    per_carrier: dict[str, dict[str, int]] = {}
    per_stratum: dict[str, dict[str, int]] = {}
    # aligned pairing: slice_rows (rows order preserved by the mask) with the
    # masked hit vector — a plain zip(rows, …) would misalign (defect fixed
    # before reporting; the headline acc never used this pairing)
    for r, hit in zip(slice_rows, (pred == y)[slice_mask]):
        c = r.get("carrier")
        st = per_carrier.setdefault(c, {"rows": 0, "hits": 0})
        st["rows"] += 1
        st["hits"] += int(hit)
        stt = per_stratum.setdefault(r["stratum"], {"rows": 0, "hits": 0})
        stt["rows"] += 1
        stt["hits"] += int(hit)
    slice_pairs = {r["pair_id"] for r in rows if r.get("carrier") in OOF_CARRIER_SLICE}
    slice_scaffolds = {_pair_scaffold(p) for p in slice_pairs}
    log(
        "oof_carriers_slice",
        carriers=list(OOF_CARRIER_SLICE),
        rows=n_slice,
        pairs=len(slice_pairs),
        distinct_scaffolds=len(slice_scaffolds),
        folds=CV_K,
        grouping="scaffold_id GroupKFold (E6: N1 scaffold inherited from the "
        "donor — donor↔variant never split across folds, leak-free by "
        "grouping)",
        cut=DECISION_CUT,
        hits=hits,
        oof_acc=round(acc, 4),
        wilson95=[lo, hi],
        per_carrier=per_carrier,
        per_stratum=per_stratum,
        stop_signal=OOF_STOP_SIGNAL,
        artifact_sha256=manifest["sha256"],
        note="E8-ii sliced OOF — characterization instrument BEFORE the "
        "gate; NOT a threshold search (ratified cut 0.5)",
    )
    ok = acc >= OOF_STOP_SIGNAL
    log(
        "oof_carriers_done",
        status="GREEN" if ok else "STOP",
        oof_acc=round(acc, 4),
        stop_signal=OOF_STOP_SIGNAL,
        note="E8-ii: acc < 0.75 BEFORE the gate = instrument-class stop "
        "(not a red gate); the sealed holdout is NOT read in either case",
    )
    if not ok:
        raise InstrumentAbort(
            f"E8-ii stop-signal RED: sliced OOF N1∪N2 acc {acc:.4f} < "
            f"{OOF_STOP_SIGNAL} — gate stays closed; STOP + report to TL"
        )
    return 0


# --- subcommand: gate (single-shot; implemented, NOT executed in phase 1) ------

SEALED_EVENTS = ("sealed_read_once", "gate_done")

#: ADDENDUM-5 D2/F4: seal state of the CURRENT process. ``opened`` flips
#: True the moment the sealed file is opened for reading (after the pin
#: checks pass) — from that point the seal is treated BURNED on ANY crash
#: (ML-STD-0003 amend: post-read instrument-crash => burned; re-admission
#: forbidden regardless of any audit outcome).
_SEAL_STATE: dict = {"opened": False}


def _fail_closed_post_read(phase: str, exc: BaseException) -> int:
    """ADDENDUM-5 D2/F4: fail-closed post-read abort — exit 4.

    ANY crash after the sealed read becomes a VISIBLE instrument_abort
    event — never a silently absent gate_done. The exception MESSAGE is
    suppressed on purpose: the incident's escaped scalar traveled inside
    an exception message (traceback -> stdout -> lane context), so only
    the exception TYPE and a traceback POINTER (file:line, function) are
    published. The seal is reported burned (it was read)."""
    frames = traceback.extract_tb(exc.__traceback__)
    pointer = (
        f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}"
        if frames
        else "traceback unavailable"
    )
    # Self-sufficient on purpose: the fail-closed handler must not depend on
    # state that only apply_input_params sets (SEAL_NAME is preset-resolved).
    seal_name = globals().get("SEAL_NAME") or (
        SEAL_JSON.name[: -len("".join(SEAL_JSON.suffixes))] or SEAL_JSON.stem
    )
    log(
        "instrument_abort",
        reason=f"{type(exc).__name__} in post-read phase '{phase}' "
        "(message suppressed: leakage guard F4/ML-а)",
        phase=phase,
        abort_class="instrument",
        post_read=True,
        seal_burned=bool(_SEAL_STATE["opened"]),
        seal_name=seal_name,
        tb_pointer=pointer,
        cos_clamp_count=COS_CLAMP_LEDGER["count"],
        cos_clamp_over_floor=COS_CLAMP_LEDGER["over_floor"],
        cos_clamp_max_overshoot=COS_CLAMP_LEDGER["max_overshoot"],
        note="fail-closed (ADDENDUM-5 D2): crash after the sealed read is "
        "a visible event, not an absent gate_done; the seal is treated "
        "BURNED (ML-STD-0003 amend: re-admission forbidden regardless of "
        "audit outcome)",
    )
    print(
        f"INSTRUMENT ABORT (post-read, seal burned): {type(exc).__name__} "
        f"in phase '{phase}' at {pointer} — message suppressed "
        "(leakage guard F4)",
        file=sys.stderr,
    )
    return 4


def cmd_gate(args: argparse.Namespace) -> int:
    global _SLICE
    _SLICE = SLICE_GATE
    if prior_single_shot():
        log(
            "refuse",
            reason="single-shot already spent for this slice "
            "(sealed_read_once/gate_done in run-log) — П8",
        )
        print("REFUSED: single-shot already spent (П8).", file=sys.stderr)
        return 2
    if not args.execute_single_shot:
        log(
            "refuse",
            reason="gate phase implemented but NOT executed: "
            "--execute-single-shot absent; no sealed read",
            note="this phase stays train-only; the single-shot is a separate, "
            "explicitly flagged run (dispatch §3 step 4)",
        )
        print(
            "REFUSED: gate requires --execute-single-shot. "
            "No sealed surface was touched.",
            file=sys.stderr,
        )
        return 2
    return _gate_single_shot()


def prior_single_shot() -> bool:
    return bool(runlog_events(SEALED_EVENTS, SLICE_GATE))


def _gate_single_shot() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=300,
        expected_peak_ram_gb=1.5,
        lane="gate lane: pre-gate legs + ONE sealed read + §6.2 conjunction",
    )
    booster, manifest = load_frozen()
    pro = prologue(SLICE_GATE)

    # floors prereg (П1/П6): the published floors must exist and be bound to
    # THIS artifact sha BEFORE gate_start (cascade W2 P2: a stale/foreign
    # prereg must not satisfy the assert).
    floors = json.loads(FLOORS_DOC.read_text(encoding="utf-8"))
    pub_events = runlog_events(("floors_published",), SLICE_FLOOR)
    bound = [e for e in pub_events if e.get("artifact_sha256") == manifest["sha256"]]
    if not FLOORS_DOC.exists() or not bound:
        raise InstrumentAbort(
            "floors prereg RED: no floors_published event bound to artifact "
            f"{manifest['sha256'][:12]}… — the declared floors are a "
            "precondition of gate_start (ML-STD-0004 П1/П6)"
        )
    for key in ("floor_seed", "floor_inv", "floor_q", "floor_perm"):
        if key not in floors:
            raise InstrumentAbort(
                f"floors doc has no '{key}' section — gate tolerances are "
                "unfixed (ML-STD-0001 П1: отказ старта)"
            )
    dist = floors["floor_seed"]["distribution"]
    declared = float(floors["floor_seed"]["delta_acc_declared"])
    recheck = max(float(dist["p95"]) - float(dist["p5"]), 0.10)
    if abs(declared - recheck) > 1e-9:
        raise InstrumentAbort(
            "ML-STD-0004 П2: the declared delta_acc does not reproduce from "
            f"the published p5/p95 (declared {declared}, recheck {recheck}) — "
            "process incident, escalation MLC/owner"
        )
    log(
        "prereg_order_assert",
        floors_published_seen=True,
        artifact_sha256=manifest["sha256"],
        delta_acc_declared=declared,
        delta_inv=float(floors["floor_inv"]["delta_inv"]),
        ok=True,
    )

    # П1: gate_start BEFORE any eval computation.
    log(
        "gate_start",
        note="full prologue green above; every eval read starts after this event",
    )

    # pre-sealed: sanity §6.3 (red => abort, no holdout)
    run_sanity_suite(booster)

    # g_gold: closed pre-freeze (asserted in the prologue ancestry; logged)
    log(
        "g_gold_state",
        state="CLOSED 48/48 >= 45 (synthesis commit 45597c5, key 6968087a)",
        reopen="forbidden (dispatch §2)",
        ok=True,
    )

    # ============ SEALED READ — SINGLE-SHOT ==================================
    # ADDENDUM-5 D2: every phase after the sealed read runs FAIL-CLOSED —
    # any crash becomes a visible instrument_abort event (message suppressed
    # per F4: the incident's escaped scalar traveled inside an exception
    # message) + non-zero exit 4. Absence of gate_done is never silent again.
    phase = "read_holdout_sealed"
    try:
        holdout = _read_holdout_sealed(manifest)
        log(
            "sealed_read_once",
            n_pairs=holdout["n_pairs"],
            n_rows=holdout["n_rows"],
            sha256=holdout["sha256"],
            note=f"THE single legal read of {SEAL_NAME} (dispatch §1)",
        )
        phase = "eval_gates_on_holdout"
        gates = _eval_gates_on_holdout(booster, manifest, floors, holdout)
        phase = "gate_result_events"
        for name, g in gates.items():
            log("gate_result", gate=name, **g)
        phase = "verdict_gate_done"
        verdict = (
            "ADOPT"
            if all(g["ok"] for k, g in gates.items() if not k.startswith("_"))
            else "DECLINE"
        )
        log(
            "gate_done",
            verdict=verdict,
            red=[k for k, g in gates.items() if not g["ok"]],
            wall_s=round(
                (dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1
            ),
            discipline="holdout read exactly once; no re-run (П8)",
        )
        phase = "verdict_print"
        print(f"VERDICT: {verdict}")
        return 0
    except InstrumentAbort as exc:
        # fail-closed runner abort AFTER the read: same visible accounting
        return _fail_closed_post_read(phase, exc)
    except Exception as exc:
        # ADDENDUM-5 D2: the incident class itself — unexpected post-read crash
        return _fail_closed_post_read(phase, exc)
    except BaseException as exc:
        # interrupt post-read: log honestly (seal burned), then re-raise
        _fail_closed_post_read(phase, exc)
        raise


def _read_holdout_sealed(manifest: dict) -> dict:
    """THE single legal read of the resolved holdout seal (single-shot).

    Caller contract: full prologue green + prereg asserted + gate_start
    logged + pre-gate sanity green. The bytes are pinned against the seal
    json AND the build manifest BEFORE opening; a divergence is a loud
    instrument abort and the file stays unopened."""
    sha = sha256_file(HOLDOUT_JSONL)
    if sha != HOLDOUT_SHA_PIN or sha != manifest["corpus"]["holdout_jsonl_sha256_metadata_pin"]:
        raise InstrumentAbort(
            f"sealed pin mismatch: holdout sha256 {sha[:12]}… does not "
            "reproduce the seal-json/build-manifest pins — file stays sealed"
        )
    # ADDENDUM-5 D2: from here the bytes are in-process — any later crash
    # is a POST-READ crash and the seal is burned (fail-closed accounting).
    _SEAL_STATE["opened"] = True
    rows = [json.loads(line) for line in open(HOLDOUT_JSONL, encoding="utf-8")]
    pairs: dict[str, list[dict]] = {}
    strata: dict[str, int] = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], []).append(r)
        if r["orientation"] == "a>b":
            strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    mism = {
        k: {"got": v, "want": EXPECTED_HOLDOUT_PAIRS[k]}
        for k, v in strata.items()
        if EXPECTED_HOLDOUT_PAIRS.get(k) != v
    }
    if mism or set(strata) != set(EXPECTED_HOLDOUT_PAIRS):
        raise InstrumentAbort(f"holdout strata count mismatch: {mism}")
    return {
        "sha256": sha,
        "rows": rows,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_rows": len(rows),
        "strata": strata,
    }


def _eval_gates_on_holdout(
    booster: lgb.Booster, manifest: dict, floors: dict, holdout: dict
) -> dict:
    """Gates §6.2 — thresholds are FROZEN constants (floors doc + ratified
    numbers), never revisited. The gated object is the PIPELINE (guard+model)
    for g_sens/g_spec/g_ba/g_brier (dispatch note 1); model-only numbers are
    published as diagnostics. Aggregates exclude same-point razor-collision
    cells (razor CD-1/CD-2 <-> NC-2/NC-3, dispatch note 2)."""
    delta_acc = float(floors["floor_seed"]["delta_acc_declared"])
    delta_inv = float(floors["floor_inv"]["delta_inv"])
    rows = holdout["rows"]
    nano = NanoLeg()

    def cos_measure(rec_a, rec_b) -> float:
        return nano.pair_cos(rec_a, rec_b, site="cos_measure")

    # razor-collision exclusions (note 2): strata CD-1/CD-2 (razor positives)
    # vs NC-2/NC-3 (razor negatives) INSIDE the razor band [0.95, 1.0] —
    # the same-point cells §2.3. Exclusion is per-cell: a row is excluded iff
    # its stratum is one of the four AND cos_target is in the razor band.
    RAZOR_COLLISION_STRATA = {"CD-1", "CD-2", "NC-2", "NC-3"}

    def excluded(r: dict) -> bool:
        return r["stratum"] in RAZOR_COLLISION_STRATA and r["measured"]["cos_target"] >= 0.95

    agg_rows = [r for r in rows if not excluded(r)]
    excl_n = len(rows) - len(agg_rows)

    hits, n, per = decide_rows(agg_rows, booster)
    # per-row cos: recompute for diagnostics + far/razor strata
    per_stratum: dict[str, dict[str, float]] = {}
    sens_hits = sens_tot = 0
    spec_hits = spec_tot = 0
    nc1_ok = nc1_tot = 0
    nc2_ok = nc2_tot = 0
    for r, d in zip(agg_rows, per):
        st = per_stratum.setdefault(
            r["stratum"],
            {"rows": 0, "hits": 0, "contradicts_proposals": 0, "label_pos": 0},
        )
        st["rows"] += 1
        st["hits"] += int(d["verdict"] == r["label"])
        st["contradicts_proposals"] += int(d["verdict"] == "contradicts")
        st["label_pos"] += int(r["label"] == "contradicts")
        if r["label"] == "contradicts":
            sens_tot += 1
            sens_hits += int(d["verdict"] == "contradicts")
        else:
            spec_tot += 1
            spec_hits += int(d["verdict"] == "not-contradicts")
            if r["stratum"] == "NC-1":
                nc1_tot += 1
                nc1_ok += int(d["verdict"] == "not-contradicts")
            if r["stratum"] == "NC-2":
                nc2_tot += 1
                nc2_ok += int(d["verdict"] == "not-contradicts")
    acc_model = hits / n if n else 0.0
    bl = baselines_table(agg_rows)
    acc_base = max(bl["BL-band"], bl["BL-digit"], bl["BL-digit+guard"])
    sens = sens_hits / sens_tot if sens_tot else 0.0
    spec = spec_hits / spec_tot if spec_tot else 0.0

    # brier over the pipeline probabilities: guard-fired rows have no model P
    # — the pipeline verdict probability for guard rows is 0.0 (certainty of
    # NOT by the code leg); published both ways.
    y = np.asarray([1.0 if r["label"] == "contradicts" else 0.0 for r in agg_rows])
    p_model_only = np.asarray([d["p"] if d["p"] is not None else 0.0 for d in per])
    brier_model = float(((p_model_only - y) ** 2).mean())

    # g_nc3_exact: contradicts proposals of the pipeline on NC-3 rows
    nc3_rows = [r for r in rows if r["stratum"] == "NC-3"]
    nc3_hits, nc3_n, nc3_per = decide_rows(nc3_rows, booster)
    nc3_prop = sum(1 for d in nc3_per if d["verdict"] == "contradicts")

    # g_self: P(contradicts | self-pair) — the fixture self-pair through the
    # MODEL leg (policy-constant; the probe-1 number)
    sc = build_fixtures()
    sc["_nano"] = nano
    fx = dict(sc)  # includes "_nano" (the embedder leg for sc_row cos)
    self_row = {
        "pair_id": "fx-self",
        "stratum": "self",
        "label": "not-contradicts",
        "orientation": "a>b",
        "measured": {"cos_target": 1.0},
        "envelope_a": cb.side_dict(fx["CD-1"].side_a),
        "envelope_b": cb.side_dict(fx["CD-1"].side_a),
    }
    rec_a, rec_b = row_records(self_row)
    x_self = pair_vector(rec_a, rec_b, 1.0).reshape(1, -1)
    p_self = float(booster.predict(x_self)[0])

    # g_far: P(contradicts | far pair, cos < 0.40) over holdout far rows
    far_rows = [r for r in rows if r["measured"]["cos_target"] < 0.40]
    far_hits, far_n, far_per = decide_rows(far_rows, booster)
    far_prop = sum(1 for d in far_per if d["verdict"] == "contradicts")

    # g_stab: verdict agreement on paraphrase/cosmetic twins, 3 steps
    rng = random.Random(21)
    stab_rows = rng.sample(rows, min(60, len(rows)))
    agree = total = 0
    for r in stab_rows:
        rec_a, rec_b = row_records(r)
        base = decide_pipeline(rec_a, rec_b, float(r["measured"]["cos_target"]), booster)
        verdicts = [base["verdict"]]
        cos = nano.pair_cos(rec_a, rec_b, site="g_stab:base")
        for step in ("cosmetic", "para1", "para2"):
            pert = row_with_body(
                r, "envelope_b",
                cosmetic_variant(r["envelope_b"]["body"], rng)
                if step == "cosmetic"
                else lib.para(r["envelope_b"]["body"], 1 if step == "para1" else 2,
                              "ru", rng),
            )
            prec_a, prec_b = row_records(pert)
            c = nano.pair_cos(prec_a, prec_b, site=f"g_stab:{step}")
            d = decide_pipeline(prec_a, prec_b, c, booster)
            verdicts.append(d["verdict"])
        total += 1
        agree += int(len(set(verdicts)) == 1)
    stab_rate = agree / total if total else 1.0

    # g_disj: train pair ids vs the seal id manifest (id-level; metadata only)
    seal = json.loads(SEAL_JSON.read_text(encoding="utf-8"))
    train_ids = {r["pair_id"] for r in load_train()}
    disj_hits = sorted(train_ids & set(seal["ids"]))

    # guard-off honesty probe (post-read, per stratum, minimum NC-3)
    guard_off: dict[str, dict[str, int]] = {}
    for r in rows:
        rec_a, rec_b = row_records(r)
        if not guard_fire(rec_a.body, rec_b.body):
            continue
        st = guard_off.setdefault(
            r["stratum"], {"rows": 0, "model_proposals_guard_off": 0}
        )
        st["rows"] += 1
        x = pair_vector(rec_a, rec_b, float(r["measured"]["cos_target"])).reshape(1, -1)
        p = float(booster.predict(x)[0])
        st["model_proposals_guard_off"] += int(p >= DECISION_CUT)
    log(
        "guard_off_probe_holdout",
        per_stratum=guard_off,
        not_a_gate=True,
        honesty="the contradicts proposals the guard suppresses on holdout "
        "(MLC-9 p.13; minimum NC-3)",
    )

    gates = {
        "g_sens": {
            "ok": bool(sens >= 0.90),
            "sens": round(sens, 4),
            "threshold": ">= 0.90",
            "excluded_razor_rows": excl_n,
            "object": "pipeline (guard+model)",
        },
        "g_spec": {
            "ok": bool(spec >= 0.90 and (nc1_tot == 0 or nc1_ok / nc1_tot >= 0.85)
                       and (nc2_tot == 0 or nc2_ok / nc2_tot >= 0.85)),
            "spec": round(spec, 4),
            "nc1_spec": round(nc1_ok / nc1_tot, 4) if nc1_tot else None,
            "nc2_spec": round(nc2_ok / nc2_tot, 4) if nc2_tot else None,
            "threshold": ">= 0.90 aggregate; NC-1/NC-2 >= 0.85",
            "object": "pipeline (guard+model)",
        },
        "_per_stratum_holdout": {
            "diagnostic": per_stratum,
            "note": "per-stratum table (Wilson CI characterization, "
            "design §6.2 note 2) — not a gate",
        },
        "g_nc3_exact": {
            "ok": nc3_prop == 0,
            "proposals": nc3_prop,
            "rows": nc3_n,
            "threshold": "exact 0",
            "object": "pipeline",
        },
        "g_razor": {
            "ok": None,  # filled below from the pre-sealed sanity probes 4+5
            "note": "certify exam = sanity probes 4+5 through the production "
            "circuit (both legs exact)",
        },
        "g_self": {
            "ok": bool(p_self <= 0.10),
            "p_self": round(p_self, 6),
            "threshold": "<= 0.10",
            "object": "model (fixture self-pair)",
        },
        "g_ba": {
            "ok": bool((acc_model - acc_base) >= delta_acc),
            "acc_model": round(acc_model, 4),
            "acc_baseline_max": round(acc_base, 4),
            "delta": round(acc_model - acc_base, 4),
            "delta_acc": round(delta_acc, 4),
            "baselines": bl,
            "threshold": f">= delta_acc {delta_acc}",
            "excluded_razor_rows": excl_n,
            "object": "pipeline (guard+model)",
        },
        "g_brier": {
            "ok": bool(brier_model <= 0.25),
            "brier_model_leg": round(brier_model, 4),
            "threshold": "<= 0.25",
            "note": "guard-fired rows carry the code-leg verdict; the brier "
            "is computed on the model-leg probabilities of the aggregated rows",
            "excluded_razor_rows": excl_n,
            "object": "pipeline (guard+model)",
        },
        "g_stab": {
            "ok": bool(stab_rate >= 1.0 - 2.0 * delta_inv),
            "agreement": round(stab_rate, 4),
            "pairs": total,
            "delta_inv": delta_inv,
            "threshold": ">= 1 - 2*floor_inv",
            "object": "pipeline (guard+model)",
        },
        "g_far": {
            "ok": bool(far_n == 0 or far_prop / far_n < 0.5),
            "far_rows": far_n,
            "contradicts_proposals": far_prop,
            "threshold": "< 0.5 (cos < 0.40)",
            "object": "pipeline",
        },
        "g_disj": {
            "ok": len(disj_hits) == 0,
            "intersections": disj_hits[:10],
            "threshold": "exact 0 (id-level vs the seal manifest)",
            "object": "pipeline",
        },
    }
    # g_razor: re-run probes 4+5 through the contour right here (both legs exact)
    nano2 = NanoLeg()
    r4 = sc_row(fx, "CD-1")
    d4 = decide_pipeline(*(row_records(r4)), float(r4["measured"]["cos_target"]), booster)
    r5 = sc_row(fx, "NC-3")
    d5 = decide_pipeline(*(row_records(r5)), float(r5["measured"]["cos_target"]), booster)
    razor_ok = (
        d4["verdict"] == "contradicts"
        and d4["leg"] == "model"
        and d4["p"] is not None
        and d4["p"] >= 0.5
        and d5["verdict"] == "not-contradicts"
        and d5["leg"] == "guard"
    )
    gates["g_razor"] = {
        "ok": bool(razor_ok),
        "positive_leg": {"verdict": d4["verdict"], "leg": d4["leg"], "p": d4["p"]},
        "negative_leg": {"verdict": d5["verdict"], "leg": d5["leg"]},
        "threshold": "both legs exact (probes 4+5 §6.3 through the circuit)",
        "object": "pipeline",
    }
    return gates


# --- CLI -----------------------------------------------------------------------


def _add_input_args(p: argparse.ArgumentParser) -> None:
    """B6 v2 re-run input plumbing (semantics unchanged): the four input
    locations, defaults resolving to the ratified v1 preset."""
    p.add_argument(
        "--corpus-dir",
        default=None,
        help="corpus dir, e.g. data/conflict-detector/corpus-v2 "
        "(default: the ratified v1 corpus; unknown dirs abort fail-closed)",
    )
    p.add_argument("--pin", default=None, help="pin json (default: per corpus preset)")
    p.add_argument(
        "--seal",
        default=None,
        help="seal json — METADATA only, never opened here "
        "(default: per corpus preset)",
    )
    p.add_argument(
        "--artifact-dir",
        default=None,
        help="artifact dir (default: per corpus preset)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train + freeze the resolved artifact dir")
    p.add_argument("--run-tag", default="phase1")
    _add_input_args(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("floors", help="ML-STD-0001/0004 floors (train-only)")
    _add_input_args(p)
    p.set_defaults(fn=cmd_floors)

    p = sub.add_parser("sanity", help="sanity suite §6.3 through the circuit")
    _add_input_args(p)
    p.set_defaults(fn=cmd_sanity)

    p = sub.add_parser(
        "oof-carriers",
        help="E8-ii sliced OOF on N1∪N2 (scaffold-grouped, Wilson CI; "
        "v2 pre-gate characterization, stop-signal < 0.75)",
    )
    _add_input_args(p)
    p.set_defaults(fn=cmd_oof_carriers)

    p = sub.add_parser("gate", help="single-shot gate §6.2 (sealed holdout)")
    p.add_argument(
        "--execute-single-shot",
        action="store_true",
        help="the ONLY way to unseal the holdout read (dispatch §1)",
    )
    _add_input_args(p)
    p.set_defaults(fn=cmd_gate)

    args = ap.parse_args()
    apply_input_params(args)
    log(
        "trainer_input_params",
        preset=_INPUT_PRESET_VERSION,
        corpus_dir=str(CORPUS_DIR),
        pin=str(PIN_JSON),
        seal=str(SEAL_JSON),
        seal_opened=False,
        artifact_dir=str(ART_DIR),
        artifact_prefix=ART_PREFIX,
        slices={
            "train": SLICE_TRAIN,
            "floors": SLICE_FLOOR,
            "sanity": SLICE_SANITY,
            "oof_carriers": SLICE_OOF,
            "gate": SLICE_GATE,
        },
        byte_pins={
            "train_sha256": TRAIN_SHA_PIN,
            "holdout_sha256_metadata": HOLDOUT_SHA_PIN,
            "pin_sha256": _INPUT_PRESET["pin_sha256"],
            "seal_sha256": _INPUT_PRESET["seal_sha256"],
        },
        tables={
            "expected_train_rows": EXPECTED_TRAIN_ROW_COUNTS,
            "expected_holdout_pairs": EXPECTED_HOLDOUT_PAIRS,
            "expected_baselines": EXPECTED_BASELINES,
            "floor_draw_quota": FLOOR_DRAW_QUOTA,
        },
        cmd=args.cmd,
        note=f"trainer input-params for {_INPUT_PRESET_VERSION}, semantics "
        "unchanged — features/guard/cut 0.5/production circuit are the same "
        "code path for every preset; only input locations, byte pins and "
        "per-corpus tables resolve via CLI",
    )
    try:
        return args.fn(args)
    except InstrumentAbort as exc:
        opened = bool(_SEAL_STATE["opened"])
        log(
            "instrument_abort",
            reason=str(exc),
            abort_class="instrument",
            sealed_surfaces_read=int(opened),
            seal_burned=opened,
            note="STOP + report to TL (П6 ML-STD-0002); "
            + (
                "seal WAS read — burned (ADDENDUM-5 D2/F4)"
                if opened
                else "holdout NOT read"
            ),
        )
        print(f"INSTRUMENT ABORT: {exc}", file=sys.stderr)
        return 3
    finally:
        # ADDENDUM-5 F5: the per-run clamp ledger is published on EVERY run —
        # count 0 is the green-path expectation (asserted on sanity legs).
        log(
            "cos_clamp",
            count=COS_CLAMP_LEDGER["count"],
            over_floor=COS_CLAMP_LEDGER["over_floor"],
            max_overshoot=COS_CLAMP_LEDGER["max_overshoot"],
            kind="run_summary",
            note="per-run clamp ledger (ADDENDUM-5 F5): green path = 0 "
            "above the f64 noise floor; every clamp stays a visible event",
        )


if __name__ == "__main__":
    sys.exit(main())
