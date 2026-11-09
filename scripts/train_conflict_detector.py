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
TRAIN_SHA_PIN = "0be52dd220c792810420c10921f30329ba6d903bfda5c02f9900ffdb8295e4aa"
HOLDOUT_SHA_PIN = "e8adfec9625b0f72ac001ed6024da2e3fd3bb5170169b6d93f26afde521b2567"

SLICE_TRAIN = "conflict-detector-v1-train"
SLICE_FLOOR = "conflict-detector-v1-floors"
SLICE_SANITY = "conflict-detector-v1-sanity"
SLICE_GATE = "conflict-detector-v1-gate"

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


# --- embedder leg (A-frozen: wrong provider refuses loudly) -------------------


class NanoLeg:
    """Pinned nano embedder over the FROZEN surface convention.

    Surface: ``lib.embed_text`` (n1(title)\n+n1(body)\n+n1(space-joined
    tags)) truncated to [:4096]; cosine = dot product of the normalized
    384-dim vectors — the SAME computation as the corpus build's ``Cos``
    (the trainer imports the frozen lib functions instead of re-deriving).
    A pin mismatch is a loud instrument abort.
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

    def pair_cos(self, side_a, side_b) -> float:
        a = np.asarray(self.vec(lib.embed_text(side_a)), dtype=np.float64)
        b = np.asarray(self.vec(lib.embed_text(side_b)), dtype=np.float64)
        return float(np.dot(a, b))


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
    log(
        "byte_tripwire",
        pins=pins,
        expected_hard=want,
        mismatch_hard=hard_mism,
        seal_metadata_mismatch=seal_mism,
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
    ratified = bool(re.search(r"RATIFIED,\s*MLC-9", spec_text))
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
    ok = ratified and all(v["is_ancestor"] for v in ancestry.values())
    log(
        "activation",
        spec_sha256=sha256_text(spec_text),
        ratified_marker_present=ratified,
        commits=ancestry,
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
    for name in (0, 1):  # two band directions
        acc = sum(
            1
            for r in rows
            if (bl_band_cos(r["measured"]["cos_target"]) == (name == 0))
            == (r["label"] == "contradicts")
        ) / n
        out[f"BL-band-dir{name}"] = round(acc, 4)
    band_dir = max((0, 1), key=lambda d: out[f"BL-band-dir{d}"])
    out["BL-band"] = out[f"BL-band-dir{band_dir}"]
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
    from cortex.artifacts import set_onnx_metadata, features_digest

    props = {
        "name": "conflict-detector",
        "version": "1",
        "embedder_pin": "nano:sha256:" + lib.EMBEDDER_PIN,
        "corpus_fingerprint": corpus_fp,
        "trained_at": trained_at,
        "candidate": "conflict-detector-v1",
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
    """Freeze models/conflict-detector-v1/ (dispatch §7 bundle; the CONTENT
    sections mirror the merge-v2 committed-doc discipline per the delta
    map §8: thresholds/rule, guard section with pinned constants, corpus
    pins, baselines, cv characterization, env pins, convention)."""
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
        "name": "conflict-detector-v1",
        "version": "1",
        "wave": "b6-conflict",
        "artifact_lineage": "mirror of merge_arbiter_train.py per "
        "docs/experiments/conflict-detector-trainer-delta-map.md; "
        "dispatch docs/experiments/conflict-detector-train-dispatch-2026-11-06.md",
        "sha256": onnx_info["sha256"],
        "size_bytes": ONNX_PATH.stat().st_size,
        "embedder_pin": lib.EMBEDDER_PIN,
        "embedder_pin_engine": "nano:sha256:" + lib.EMBEDDER_PIN,
        "corpus_fingerprint_blake2b_256": corpus_fp,
        "trained_at": trained_at,
        "candidate": "conflict-detector-v1",
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
            "seal_n_ids_metadata": 180,
            "regime": "synthetic-only; no store texts; no LLM; labels by "
            "construction",
        },
        "pins": {k: v for k, v in pins.items() if k != "seal_n_ids"},
        "env": dict(EXPECTED_ENV),
        "seed": SEED,
        "gates_state": "sealed — single-shot NOT executed at freeze time "
        "(dispatch §3; holdout v1 reads exactly once at the gate)",
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
        "# MODEL CARD — conflict-detector-v1",
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
        "holdout CD-holdout-v1 SEALED (180 pairs), read exactly once by the "
        "gate runner",
        f"- Baselines (train): BL-band {b['BL-band']} / BL-digit "
        f"{b['BL-digit']} / BL-digit+guard {b['BL-digit+guard']} "
        "(target max(BL) = 0.66)",
        f"- CV characterization (honesty, no selection): OOF acc "
        f"{manifest['cv_characterization']['oof_acc']}, OOF brier "
        f"{manifest['cv_characterization']['oof_brier']} "
        f"({manifest['cv_characterization']['folds']} folds by scaffold)",
        "- Floors: models/conflict-detector-v1/floors.json (ML-STD-0001/0004; "
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
    """Cosmetic corruption (case + whitespace only): invisible to the whole
    frozen surface (norm1 lower+collapse everywhere: features, guard
    regions, embed_text) — the П1 'заведомо-нулевое условие' for floor_inv."""
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < 0.15 and w:
            w = w.upper()
        elif r < 0.30 and w:
            w = w.capitalize()
        out.append(w)
    s = " ".join(out)
    if rng.random() < 0.5:
        s = s + "  "
    if rng.random() < 0.5:
        s = " " + s
    return s


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
        sc._cos = round(nano.pair_cos(sc.side_a, sc.side_b), 4)
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
    eligible = train_rows  # all 1200 rows are fold-eligible material
    inv_seeds = FLOOR_INV_SEEDS
    inv_disagreements = []
    for draw_seed in inv_seeds:
        rng = random.Random(draw_seed)
        sample = rng.sample(eligible, min(80, len(eligible)))
        dis = 0
        for r in sample:
            rec_a, rec_b = row_records(r)
            base = decide_pipeline(
                rec_a, rec_b, float(r["measured"]["cos_target"]), booster
            )
            pert = row_with_body(
                r, "envelope_b", cosmetic_variant(r["envelope_b"]["body"], rng)
            )
            prec_a, prec_b = row_records(pert)
            # cosmetic variants keep the corpus cos (the surface is
            # normalization-invariant; asserted by the identity of the
            # embedded text below via the same convention string)
            alt = decide_pipeline(prec_a, prec_b, float(r["measured"]["cos_target"]), booster)
            dis += int(base["verdict"] != alt["verdict"])
        inv_disagreements.append(round(dis / len(sample), 4))
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
        note="structurally 0 expected: the whole frozen surface "
        "(features, guard regions, embed_text) is normalization-invariant",
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


def sc_row(sc, orientation: str = "a>b") -> dict:
    """A fixture pair as a row dict (cb.side_dict + measured by NanoLeg)."""
    nano = sc["_nano"]
    cos = round(nano.pair_cos(sc.side_a, sc.side_b), 4)
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
    (guard+model). Red on ANY probe = InstrumentAbort — the sealed holdout
    is NOT read (class 'instrument', П6 ML-STD-0002)."""
    sc = build_fixtures()
    sc["_nano"] = NanoLeg()
    fx = {k: v for k, v in sc.items() if not k.startswith("_")}
    results = []
    all_ok = True

    def run_pair(row_a: dict, name: str, expect: str, check, note: str = "") -> dict:
        nonlocal all_ok
        rec_a, rec_b = row_records(row_a)
        res = decide_pipeline(rec_a, rec_b, float(row_a["measured"]["cos_target"]), booster)
        # order invariance: the swapped presentation must not change the verdict
        row_b = orientation_swap(row_a)
        rec_a2, rec_b2 = row_records(row_b)
        res2 = decide_pipeline(rec_a2, rec_b2, float(row_a["measured"]["cos_target"]), booster)
        inv = res["verdict"] == res2["verdict"]
        ok = bool(check(res)) and inv
        all_ok = all_ok and ok
        if log_it:
            probe_result(name, expect, ok, res, inv)
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
    r = sc_row(fx["NC-2"])
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
        note="cos rides the corpus-measured convention (the surface is "
        "normalization-invariant, so the variant embeds identically)",
    )

    # 3 — envelope variant (whitelist §8.1 deltas): NOT contradicts
    r3 = sc_row(fx["NC-4-tags"])
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
    r4 = sc_row(fx["CD-1"])
    run_pair(
        r4, "4-razor-cd1", "contradicts, P >= 0.5 (leg model)",
        lambda res: res["verdict"] == "contradicts" and res["leg"] == "model"
        and res["p"] is not None and res["p"] >= 0.5,
        note="positive leg of the razor certify exam (g_razor); one digit "
        "replaced, guard must NOT fire",
    )

    # 5 — version neighbor: NOT via the guard leg (negative leg of g_razor)
    r5 = sc_row(fx["NC-3"])
    run_pair(
        r5, "5-version-neighbor", "NOT contradicts (leg guard, exact)",
        lambda res: res["verdict"] == "not-contradicts" and res["leg"] == "guard",
        note="negative leg of the razor certify exam (g_razor); bump "
        "v1.2.0 -> v1.3.0, supersede semantics, NOT a conflict",
    )

    # 6 — stability ladder: one conflict through raw / cosmetic / paraphrase,
    # verdict invariant across all 3 steps
    ladder = []
    base6 = sc_row(fx["CD-2"])
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

    # 7 — translation twin: identity translation (NC-4 lang delta) NOT +
    # translation with changed fact (CD-1 cross) contradicts
    r7n = sc_row(fx["NC-4-lang"])
    r7n = dict(r7n, pair_id="fx-trans-ok")
    res7n = run_pair(
        r7n, "7-translation-twin-not", "NOT contradicts (correct translation)",
        lambda res: res["verdict"] == "not-contradicts",
        note="fixture: NC-4 lang-delta cell (generator native; the corpus "
        "vocabulary has no same-content RU<->EN class — §4.4 negative cross "
        "is different-subject siblings; identity translation is the "
        "generator-native correct-translation surface, 8 train rows)",
    )
    r7p = sc_row(fx["CD-1-cross"])
    run_pair(
        r7p, "7-translation-fact-changed", "contradicts (changed-fact translation)",
        lambda res: res["verdict"] == "contradicts",
        note="fixture: §4.4 cross positive — RU sentence vs EN translation "
        "with one changed factual reference (12 train rows)",
    )

    # diagnostic (published, NOT a probe, off-support honesty number):
    # the TRUE correct translation (cross CD-1, equal values)
    r7d = sc_row(fx["CD-1-cross-eq"])
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

    log(
        "sanity_suite_done",
        ok=all_ok,
        probes=len(results),
        expectation="7/7 probes green + order invariance; red => holdout "
        "NOT read (instrument abort)",
    )
    if not all_ok:
        raise InstrumentAbort(
            "sanity suite RED (§6.3) — sealed surfaces stay sealed; "
            "STOP + report (abort class 'instrument')"
        )
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
        "(pinned nano embedder, ~30 embeds)",
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
    run_sanity_suite(booster)
    log("sanity_cmd_done", status="GREEN", artifact=manifest["sha256"][:16])
    return 0


# --- subcommand: gate (single-shot; implemented, NOT executed in phase 1) ------

SEALED_EVENTS = ("sealed_read_once", "gate_done")


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
    holdout = _read_holdout_sealed(manifest)
    log(
        "sealed_read_once",
        n_pairs=holdout["n_pairs"],
        n_rows=holdout["n_rows"],
        sha256=holdout["sha256"],
        note="THE single legal read of CD-holdout-v1 (dispatch §1)",
    )

    gates = _eval_gates_on_holdout(booster, manifest, floors, holdout)
    for name, g in gates.items():
        log("gate_result", gate=name, **g)
    verdict = "ADOPT" if all(g["ok"] for k, g in gates.items() if not k.startswith("_")) else "DECLINE"
    log(
        "gate_done",
        verdict=verdict,
        red=[k for k, g in gates.items() if not g["ok"]],
        wall_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
        discipline="holdout read exactly once; no re-run (П8)",
    )
    print(f"VERDICT: {verdict}")
    return 0


def _read_holdout_sealed(manifest: dict) -> dict:
    """THE single legal read of CD-holdout-v1 (single-shot).

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
        return nano.pair_cos(rec_a, rec_b)

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
    fx = {k: v for k, v in sc.items() if not k.startswith("_")}
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
        cos = nano.pair_cos(rec_a, rec_b)
        for step in ("cosmetic", "para1", "para2"):
            pert = row_with_body(
                r, "envelope_b",
                cosmetic_variant(r["envelope_b"]["body"], rng)
                if step == "cosmetic"
                else lib.para(r["envelope_b"]["body"], 1 if step == "para1" else 2,
                              "ru", rng),
            )
            prec_a, prec_b = row_records(pert)
            c = nano.pair_cos(prec_a, prec_b)
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
    r4 = sc_row(fx["CD-1"])
    d4 = decide_pipeline(*(row_records(r4)), float(r4["measured"]["cos_target"]), booster)
    r5 = sc_row(fx["NC-3"])
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train + freeze models/conflict-detector-v1")
    p.add_argument("--run-tag", default="phase1")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("floors", help="ML-STD-0001/0004 floors (train-only)")
    p.set_defaults(fn=cmd_floors)

    p = sub.add_parser("sanity", help="sanity suite §6.3 through the circuit")
    p.set_defaults(fn=cmd_sanity)

    p = sub.add_parser("gate", help="single-shot gate §6.2 (sealed holdout)")
    p.add_argument(
        "--execute-single-shot",
        action="store_true",
        help="the ONLY way to unseal the holdout read (dispatch §1)",
    )
    p.set_defaults(fn=cmd_gate)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except InstrumentAbort as exc:
        log(
            "instrument_abort",
            reason=str(exc),
            abort_class="instrument",
            sealed_surfaces_read=0,
            note="STOP + report to TL (П6 ML-STD-0002); holdout NOT read",
        )
        print(f"INSTRUMENT ABORT: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
