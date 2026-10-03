#!/usr/bin/env python3
"""Stage-2 orchestrator — the final-sprint glue (labels.csv → artifact).

Runs the frozen prereg v2 pipeline end to end over the OWNER-LABELED
corpus: ingest labels → join + holdout split → train D/N → frozen-CV
select → export artifact. The single-shot holdout EVAL is deliberately
NOT part of the chain: it is a separate explicit subcommand guarded by
``--i-know-this-is-single-shot`` (prereg W5c: the holdout is touched
ONCE; default ``run`` stops after the artifact export).

Frozen discipline reused (never reimplemented):

- split: ``cortex.data.holdout.split_holdout`` (per stratum, pair_id
  sorted, first ⌈0.3·n⌉ → holdout) + ``assert_no_pair_overlap`` +
  ``assert_labels_isolated``;
- fingerprints: ``cortex.data.fingerprints`` (labels fingerprint per
  data-contract §5.5) + the canon-data corpus scheme (sha256 of the
  FULL canonical row — pair_id included — manifest-sorted, BLAKE2b-256;
  cross-checked against the frozen W5b value ``bcdc6e31…`` before the
  run: a mismatch voids the run, prereg «запрет пост-фактум правок»);
- training/selection/export/eval: the frozen ``cortex`` CLI surface,
  invoked in-process (no subprocess drift).

Authoritative stops (the run is FORBIDDEN, not warned):

- LABELS-INCOMPLETE / NO-DATA (exit 5) — labeling floors violated
  (duplicate ≥ 40, not-duplicate ≥ 40, disputed ≤ 40 over the full
  corpus; holdout class ≥ 20) or the labeling pass is unfinished;
- CORPUS-MISMATCH / CORPUS-STRATA / CORPUS-COSINES (exit 2) — the
  corpus is not the frozen prereg corpus or is unreadable as such;
- VECTORS-NEEDED (exit 2) — candidate N requested/selected without
  store vectors; the stop message carries the docompute recipe;
- EVAL-GUARD (exit 2) — eval without the single-shot acknowledgment.

Physical holdout isolation (ADR 0001 V1): the train tree IS the train
manifest file — the training stage receives only that path; the holdout
lives in its own subtree ``<root>/holdout/`` (pairs.jsonl WITHOUT
labels + labels.jsonl + the id manifest). The load-bearing invariant is
``assert_no_pair_overlap`` over the written manifests (pinned by tests);
``assert_labels_isolated`` is additionally checked at split time.

Dry-run (``--dry-run``): materializes a canon-shaped stand-in corpus +
labels.csv from the in-repo synthetic pairs (strata from the strategy
column) under ``data/stage2-dry/`` and runs stages 1–5 through the SAME
code path. Canon-data is NEVER read in dry mode; the real run log is
NEVER written (dry eval uses ``<dry-root>/run_log.jsonl``).

Exit codes: 0 ok · 2 contract/usage (incl. guards) · 3 train-env missing
(passthrough) · 4 single-shot refused (passthrough) · 5 NO-DATA.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from cortex.data.fingerprints import (
    canonical_json,
    corpus_fingerprint,
    labels_fingerprint,
    manifest_bytes,
)
from cortex.data.holdout import (
    LABEL_DUPLICATE,
    LABEL_NOT_DUPLICATE,
    SplitPair,
    assert_labels_isolated,
    assert_no_pair_overlap,
    split_holdout,
)

__all__ = [
    "EXPECTED_CORPUS_FINGERPRINT",
    "Stage2Stop",
    "build_dry_standin",
    "build_parser",
    "canon_labels_fingerprint",
    "canon_pair_digest",
    "ingest_labels",
    "join_and_split",
    "load_corpus",
    "load_cosines",
    "load_strata",
    "main",
]

# ── frozen constants ──────────────────────────────────────────────────────────

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Sources (read-only). Canon-data is read ONLY outside dry mode.
CANON_DATA_ROOT: Final[Path] = REPO_ROOT.parent / "vesmaro-canon-data"
DEFAULT_CORPUS_DIR: Final[Path] = CANON_DATA_ROOT / "corpus"
DEFAULT_LABELS_CSV: Final[Path] = CANON_DATA_ROOT / "labeling" / "labels.csv"
DEFAULT_DRY_SYNTH: Final[Path] = REPO_ROOT / "data" / "synth" / "pairs.sim.jsonl"
#: Corruption pretrain corpus of candidate N (stage-1 handshake).
DEFAULT_PRETRAIN_CORPUS: Final[Path] = (
    REPO_ROOT / "data" / "runs" / "stage1" / "pretrain-store-pairs.jsonl"
)

#: Frozen prereg corpus identity (canon-data W5b build_stats.json).
#: Recomputed from corpus/pairs.jsonl and cross-checked before every run.
EXPECTED_CORPUS_FINGERPRINT: Final[str] = (
    "bcdc6e31518f72edcff164abc055a6908ba34ca9d011d703344622b647f0561e"
)

#: Engine embedder pin of the prereg corpus (canon-data build_stats.json) —
#: the artifact metadata default; override with --embedder-pin.
DEFAULT_EMBEDDER_PIN: Final[str] = (
    "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
)

#: Prereg W5b label floors — over the FULL labeled corpus, before the split.
FLOOR_DUPLICATE_MIN: Final[int] = 40
FLOOR_NOT_DUPLICATE_MIN: Final[int] = 40
FLOOR_DISPUTED_MAX: Final[int] = 40
#: Prereg W5c holdout floor — per class INSIDE the holdout.
FLOOR_HOLDOUT_CLASS_MIN: Final[int] = 20

REAL_ROOT: Final[str] = "data/stage2"
DRY_ROOT: Final[str] = "data/stage2-dry"

#: Dry-run stand-in strata: strategy → P*/N* stratum. The REAL corpus
#: carries strata in corpus/strata.csv and pair_id prefixes — this map is
#: dry-run-only glue for the synthetic stand-in.
STRATEGY_STRATUM: Final[dict[str, str]] = {
    "paraphrase": "P1",
    "near-topic": "N2",
    "broken-field": "N3",
    "trivial-negative": "N1",
}

EXIT_OK: Final[int] = 0
EXIT_CONTRACT: Final[int] = 2
EXIT_ENV: Final[int] = 3
EXIT_REFUSED: Final[int] = 4
EXIT_NO_DATA: Final[int] = 5

#: Stratum prefix inside a pair_id (canon scheme: ``N1-001``). Optional —
#: strata.csv is authoritative when present.
_STRATUM_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^([PN]\d{1,2})-")

#: The canon codebook spells the negative class ``not_duplicate``; the cortex
#: internals (holdout.py, eval runner, CLI) use ``not-duplicate``. Both are
#: accepted on ingest and normalized to the internal spelling.
_LABEL_ALIASES: Final[dict[str, str]] = {
    "duplicate": LABEL_DUPLICATE,
    "not_duplicate": LABEL_NOT_DUPLICATE,
    "not-duplicate": LABEL_NOT_DUPLICATE,
    "disputed": "disputed",
}


def normalize_label(raw: str) -> str | None:
    """Codebook label → internal spelling (``not_duplicate`` → ``not-duplicate``);
    ``None`` for anything outside the codebook."""
    return _LABEL_ALIASES.get(raw.strip())


_DOCOMPUTE_RECIPE: Final[str] = (
    "vectors recipe: docompute store vectors with the engine embedder OUTSIDE "
    "this repo (s1_synth_similarity.py pattern): PYTHONPATH=<engine-src> "
    "<engine-venv>/bin/python <docompute script> -> sidecar jsonl rows "
    "{'pair_id', 'vec_a', 'vec_b'} (a2_field_cosines.py sidecar discipline), "
    "then re-run stage-2 with --vectors <sidecar>"
)


class Stage2Stop(RuntimeError):
    """Authoritative stop — the run is FORBIDDEN, never a warning."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"[{kind}] {message}")
        self.kind = kind

    @property
    def exit_code(self) -> int:
        if self.kind in ("NO-DATA", "LABELS-INCOMPLETE"):
            return EXIT_NO_DATA
        return EXIT_CONTRACT


def _log(message: str) -> None:
    print(f"stage2: {message}", file=sys.stderr)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── canon-data corpus scheme (frozen W5b) ─────────────────────────────────────


def canon_pair_digest(row: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of the FULL corpus row (pair_id INCLUDED).

    This is the canon-data W5b scheme (pipeline/verify_artifacts.py) — the
    fingerprint the prereg froze (``bcdc6e31…``). It is deliberately NOT the
    cortex data-contract §3 scheme (``cortex.data.fingerprints.pair_sha256``
    fingerprints ``{"record","candidate","similarity"}`` and excludes
    pair_id); the two schemes answer different questions and both are
    recorded in the report.
    """
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def load_corpus(corpus_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Load corpus/pairs.jsonl, recompute the canon fingerprint, cross-check
    the sibling manifest.txt when present. Returns (rows, digests, fingerprint)."""
    corpus_dir = Path(corpus_dir)
    pairs_path = corpus_dir / "pairs.jsonl"
    if not pairs_path.exists():
        raise Stage2Stop(
            "CORPUS-MISSING",
            f"{pairs_path} does not exist — point --corpus at the canon-data corpus dir",
        )
    rows = _read_jsonl(pairs_path)
    if not rows:
        raise Stage2Stop("CORPUS-MISSING", f"{pairs_path} is empty")
    seen: set[str] = set()
    for row in rows:
        pid = row.get("pair_id")
        if not pid or not isinstance(pid, str):
            raise Stage2Stop("CORPUS-MISSING", "corpus row without a string pair_id")
        if pid in seen:
            raise Stage2Stop(
                "CORPUS-MISSING", f"duplicate pair_id {pid!r} in the corpus"
            )
        seen.add(pid)
        for side in ("a", "b"):
            if not isinstance(row.get(side), dict):
                raise Stage2Stop(
                    "CORPUS-MISSING",
                    f"pair {pid}: side {side!r} missing (canon rows carry 'a'/'b')",
                )
    digests = {row["pair_id"]: canon_pair_digest(row) for row in rows}
    fingerprint = corpus_fingerprint(manifest_bytes(digests.items()))

    manifest_path = corpus_dir / "manifest.txt"
    if manifest_path.exists():
        mismatches = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pid, _, digest = line.partition(" ")
            if digests.get(pid) != digest:
                mismatches.append(pid)
        if mismatches:
            raise Stage2Stop(
                "CORPUS-MANIFEST",
                f"manifest.txt disagrees with pairs.jsonl for {len(mismatches)} pairs "
                f"(first: {mismatches[:3]}) — the corpus was edited after fingerprinting",
            )
    return rows, digests, fingerprint


def load_strata(corpus_dir: Path, rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Stratum per pair: strata.csv is authoritative when present, the P*/N*
    pair_id prefix otherwise; a prefix that contradicts strata.csv is a stop."""
    corpus_dir = Path(corpus_dir)
    pair_ids = [row["pair_id"] for row in rows]
    strata_csv = corpus_dir / "strata.csv"
    mapping: dict[str, str] = {}
    if strata_csv.exists():
        for r in csv.DictReader(strata_csv.open(encoding="utf-8")):
            mapping[r["pair_id"].strip()] = r["stratum"].strip()
        missing = [pid for pid in pair_ids if pid not in mapping]
        if missing:
            raise Stage2Stop(
                "CORPUS-STRATA",
                f"strata.csv misses {len(missing)} corpus pairs (first: {missing[:3]}) — files disagree",
            )
    else:
        for pid in pair_ids:
            m = _STRATUM_PREFIX_RE.match(pid)
            if not m:
                raise Stage2Stop(
                    "CORPUS-STRATA",
                    f"pair {pid!r}: no strata.csv and no P*/N* pair_id prefix — stratum undecidable",
                )
            mapping[pid] = m.group(1)
    for pid, stratum in mapping.items():
        m = _STRATUM_PREFIX_RE.match(pid)
        if m and m.group(1) != stratum:
            raise Stage2Stop(
                "CORPUS-STRATA",
                f"pair {pid!r}: strata.csv says {stratum!r} but the pair_id prefix says {m.group(1)!r}",
            )
    return mapping


def load_cosines(corpus_dir: Path, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Measured pair cosine per pair (cosines.csv) — the frozen `similarity`
    feature input; train/eval manifest rows cannot exist without it."""
    cosines_csv = Path(corpus_dir) / "cosines.csv"
    if not cosines_csv.exists():
        raise Stage2Stop(
            "CORPUS-COSINES",
            f"{cosines_csv} does not exist — similarity is a frozen feature input",
        )
    mapping = {
        r["pair_id"].strip(): float(r["cosine"])
        for r in csv.DictReader(cosines_csv.open(encoding="utf-8"))
    }
    missing = [row["pair_id"] for row in rows if row["pair_id"] not in mapping]
    if missing:
        raise Stage2Stop(
            "CORPUS-COSINES",
            f"cosines.csv misses {len(missing)} corpus pairs (first: {missing[:3]})",
        )
    return mapping


def load_vector_sidecar(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    """Optional vector sidecar: rows ``{"pair_id", "vec_a", "vec_b"}``
    (a2_field_cosines.py sidecar discipline)."""
    vectors: dict[str, tuple[list[float], list[float]]] = {}
    for row in _read_jsonl(Path(path)):
        pid = row.get("pair_id")
        if not pid or "vec_a" not in row or "vec_b" not in row:
            raise Stage2Stop(
                "VECTORS-NEEDED", f"sidecar row for {pid!r} misses pair_id/vec_a/vec_b"
            )
        vectors[pid] = (list(row["vec_a"]), list(row["vec_b"]))
    return vectors


# ── stage 1: ingest labels ────────────────────────────────────────────────────


def canon_labels_fingerprint(raw_rows: Iterable[dict[str, str]]) -> str:
    """Replica of the canon-data labels fingerprint (pipeline/labels_fingerprint.py):
    per-row canonical JSON {pair_id, label(RAW), note} → sha256; manifest
    sorted by pair_id; BLAKE2b-256. The value canon STATUS.md will record.
    NOTE: the cortex scheme (``labels_fingerprint`` over the normalized
    pair_id→label dict, note excluded) is a DIFFERENT number used by the
    eval run log — both are published in the report."""
    manifest = []
    for row in raw_rows:
        obj = {
            "pair_id": row["pair_id"],
            "label": row["label"],
            "note": row.get("note", ""),
        }
        cj = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        manifest.append(
            f"{row['pair_id']} {hashlib.sha256(cj.encode('utf-8')).hexdigest()}"
        )
    manifest.sort(key=lambda line: line.split(" ", 1)[0])
    return hashlib.blake2b(
        ("\n".join(manifest) + "\n").encode("utf-8"), digest_size=32
    ).hexdigest()


@dataclass(frozen=True)
class LabelsIngest:
    """Result of stage 1 — normalized labels + both fingerprint schemes."""

    labels: dict[str, str]  # pair_id → normalized label (corpus ids only)
    counts: dict[str, int]  # normalized label counts
    fingerprint_cortex: str  # data-contract §5.5 scheme (eval run log)
    fingerprint_canon: str  # canon pipeline scheme (raw csv + note; STATUS.md)


def ingest_labels(labels_csv: Path, corpus_ids: set[str]) -> LabelsIngest:
    """Read the owner's labels.csv, validate the codebook and the prereg
    floors; any violation is an authoritative stop (exit 5, run forbidden)."""
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        raise Stage2Stop(
            "LABELS-INCOMPLETE",
            f"{labels_csv} does not exist — labeling (W5b) has not started",
        )
    with labels_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for required in ("pair_id", "label"):
            if required not in columns:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE",
                    f"{labels_csv}: column {required!r} missing (header: {columns})",
                )
        raw_rows: list[dict[str, str]] = []
        labels: dict[str, str] = {}
        for row in reader:
            pid = (row.get("pair_id") or "").strip()
            raw_label = (row.get("label") or "").strip()
            note = (row.get("note") or "").strip()
            if not pid:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE", "labels.csv has a row without pair_id"
                )
            if not raw_label:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE",
                    f"pair {pid}: label is EMPTY — labeling (W5b) is unfinished; "
                    "every corpus pair must be labeled before the run (run forbidden)",
                )
            normalized = normalize_label(raw_label)
            if normalized is None:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE",
                    f"pair {pid}: unknown label {raw_label!r} — codebook: duplicate / not_duplicate / disputed",
                )
            if normalized == "disputed" and not note:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE",
                    f"pair {pid}: disputed requires a one-line reason in note (codebook)",
                )
            if normalized != "disputed" and note:
                raise Stage2Stop(
                    "LABELS-INCOMPLETE",
                    f"pair {pid}: note is only allowed for disputed rows",
                )
            labels[pid] = normalized
            raw_rows.append({"pair_id": pid, "label": raw_label, "note": note})

    extra = sorted(set(labels) - corpus_ids)
    if extra:
        raise Stage2Stop(
            "LABELS-MISMATCH",
            f"labels.csv covers {len(extra)} pairs absent from the corpus (first: {extra[:3]}) — "
            "corpus and labels must be rebuilt together (prereg: no post-hoc edits)",
        )
    missing = sorted(corpus_ids - set(labels))
    if missing:
        raise Stage2Stop(
            "LABELS-INCOMPLETE",
            f"{len(missing)} corpus pairs carry no label (first: {missing[:3]}) — "
            "labeling (W5b) is unfinished; the run is forbidden",
        )

    counts = dict(Counter(labels.values()))
    violated = []
    if counts.get(LABEL_DUPLICATE, 0) < FLOOR_DUPLICATE_MIN:
        violated.append(
            f"duplicate {counts.get(LABEL_DUPLICATE, 0)}/{FLOOR_DUPLICATE_MIN} min"
        )
    if counts.get(LABEL_NOT_DUPLICATE, 0) < FLOOR_NOT_DUPLICATE_MIN:
        violated.append(
            f"not-duplicate {counts.get(LABEL_NOT_DUPLICATE, 0)}/{FLOOR_NOT_DUPLICATE_MIN} min"
        )
    if counts.get("disputed", 0) > FLOOR_DISPUTED_MAX:
        violated.append(
            f"disputed {counts.get('disputed', 0)}/{FLOOR_DISPUTED_MAX} max"
        )
    if violated:
        raise Stage2Stop(
            "NO-DATA",
            f"prereg label floors violated ({'; '.join(violated)}) — NO-DATA: the corpus is not "
            "interpretable and must be rebuilt per the frozen procedure (prereg v2 W5b); the run is forbidden",
        )
    return LabelsIngest(
        labels=labels,
        counts=counts,
        fingerprint_cortex=labels_fingerprint(labels),
        fingerprint_canon=canon_labels_fingerprint(raw_rows),
    )


# ── stage 2: join + frozen split ──────────────────────────────────────────────


@dataclass(frozen=True)
class SplitOutcome:
    """Result of stage 2 — written manifests + published counters."""

    train_rows: int
    holdout_rows: int
    disputed_excluded: int
    per_stratum: dict[str, dict[str, int]]
    holdout_class_counts: dict[str, int]
    split_fingerprint: str
    train_manifest: Path
    holdout_dir: Path
    vectors_attached: bool


def join_and_split(
    *,
    corpus_rows: Sequence[dict[str, Any]],
    digests: dict[str, str],
    strata: dict[str, str],
    cosines: dict[str, float],
    labels: dict[str, str],
    vectors: dict[str, tuple[list[float], list[float]]] | None,
    out_root: Path,
) -> SplitOutcome:
    """Join labels with the corpus, exclude disputed (count published),
    run the frozen split, enforce the holdout class floor, write the
    train manifest + the physically separated holdout dir."""
    out_root = Path(out_root)
    rows_by_id = {row["pair_id"]: row for row in corpus_rows}
    items: list[SplitPair] = []
    disputed_excluded = 0
    for row in corpus_rows:
        pid = row["pair_id"]
        label = labels[pid]
        if label == "disputed":
            disputed_excluded += 1
            continue
        items.append(
            SplitPair(
                pair_id=pid,
                stratum=strata[pid],
                label=label,
                sha_a=hashlib.sha256(
                    canonical_json(row["a"]).encode("utf-8")
                ).hexdigest(),
                sha_b=hashlib.sha256(
                    canonical_json(row["b"]).encode("utf-8")
                ).hexdigest(),
                pair_sha256=digests[pid],
            )
        )

    split = split_holdout(items)  # frozen prereg split; disputed already excluded
    assert_no_pair_overlap(split.train_pair_ids, split.holdout_pair_ids)

    holdout_classes = Counter(labels[pid] for pid in split.holdout_pair_ids)
    violated = []
    if holdout_classes.get(LABEL_DUPLICATE, 0) < FLOOR_HOLDOUT_CLASS_MIN:
        violated.append(
            f"duplicate {holdout_classes.get(LABEL_DUPLICATE, 0)}/{FLOOR_HOLDOUT_CLASS_MIN} min"
        )
    if holdout_classes.get(LABEL_NOT_DUPLICATE, 0) < FLOOR_HOLDOUT_CLASS_MIN:
        violated.append(
            f"not-duplicate {holdout_classes.get(LABEL_NOT_DUPLICATE, 0)}/{FLOOR_HOLDOUT_CLASS_MIN} min"
        )
    if violated:
        raise Stage2Stop(
            "NO-DATA",
            f"holdout class floor violated ({'; '.join(violated)}) — NO-DATA (prereg v2 W5c): "
            "the holdout cannot carry a decision number; the run is forbidden",
        )

    def manifest_row(pid: str, with_label: bool) -> dict[str, Any]:
        row = rows_by_id[pid]
        record: dict[str, Any] = {
            "pair_id": pid,
            "stratum": strata[pid],
            "similarity": cosines[pid],
            "record": dict(row["a"]),
            "candidate": dict(row["b"]),
        }
        if with_label:
            record["label"] = labels[pid]
        if vectors is not None and pid in vectors:
            record["vec_a"], record["vec_b"] = vectors[pid]
        return record

    train_manifest = out_root / "train-manifest.jsonl"
    holdout_dir = out_root / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        manifest_row(pid, with_label=True) for pid in sorted(split.train_pair_ids)
    ]
    holdout_rows = [
        manifest_row(pid, with_label=False) for pid in sorted(split.holdout_pair_ids)
    ]
    with train_manifest.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (holdout_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in holdout_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (holdout_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for pid in sorted(split.holdout_pair_ids):
            handle.write(
                json.dumps(
                    {"pair_id": pid, "label": labels[pid]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    # holdout-id manifest — the split side carries its own manifest lines
    (holdout_dir / "manifest.txt").write_bytes(
        manifest_bytes((pid, digests[pid]) for pid in split.holdout_pair_ids)
    )

    # Physical isolation (ADR 0001 V1): the train tree IS the train-manifest
    # file — training receives only that path; the holdout labels must not
    # resolve inside it. The load-bearing invariant is the pair-id overlap
    # assert above (pinned by tests).
    assert_labels_isolated(train_manifest, holdout_dir / "labels.jsonl")

    per_stratum: dict[str, dict[str, int]] = {}
    for pid in split.train_pair_ids:
        per_stratum.setdefault(strata[pid], {"train": 0, "holdout": 0})["train"] += 1
    for pid in split.holdout_pair_ids:
        per_stratum.setdefault(strata[pid], {"train": 0, "holdout": 0})["holdout"] += 1

    return SplitOutcome(
        train_rows=len(train_rows),
        holdout_rows=len(holdout_rows),
        disputed_excluded=disputed_excluded,
        per_stratum=dict(sorted(per_stratum.items())),
        holdout_class_counts={k: holdout_classes[k] for k in sorted(holdout_classes)},
        split_fingerprint=split.corpus_fingerprint,
        train_manifest=train_manifest,
        holdout_dir=holdout_dir,
        vectors_attached=bool(vectors),
    )


# ── stages 3–5: the frozen CLI surface, in-process ────────────────────────────


def _cli(argv: list[str]) -> tuple[int, str]:
    """Run a cortex CLI command in-process; return (exit code, stdout)."""
    from cortex.cli.main import main as cli_main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli_main(argv)
    return code, buffer.getvalue()


def _manifest_vectors_complete(manifest: Path) -> bool:
    rows = _read_jsonl(manifest)
    return bool(rows) and all("vec_a" in row and "vec_b" in row for row in rows)


def stage_train(
    root: Path,
    candidate: str,
    pretrain_corpus: Path | None,
) -> dict[str, Any]:
    """Stage 3: train the ladder on the train manifest.

    ``candidate='auto'`` → both when every train row carries vec_a/vec_b,
    D-only otherwise (loudly disclosed in the report — never silent).
    An explicit ``n``/``both`` without vectors is a hard stop: the ladder
    comparison must not silently degrade."""
    root = Path(root)
    train_manifest = root / "train-manifest.jsonl"
    if not train_manifest.exists():
        raise Stage2Stop(
            "REUSE-MISSING",
            f"{train_manifest} does not exist — run the split stage first",
        )
    vectors_complete = _manifest_vectors_complete(train_manifest)
    mode = candidate
    if mode == "auto":
        mode = "both" if vectors_complete else "d"
    if mode in ("n", "both") and not vectors_complete:
        raise Stage2Stop(
            "VECTORS-NEEDED",
            f"candidate {mode} needs vec_a/vec_b on every train row and the corpus carries none; "
            + _DOCOMPUTE_RECIPE,
        )

    argv = [
        "train",
        "--train-manifest",
        str(train_manifest),
        "--candidate",
        mode,
        "--out",
        str(root / "models"),
    ]
    if pretrain_corpus is not None:
        argv += ["--pretrain-corpus", str(pretrain_corpus)]
    code, out = _cli(argv)

    result: dict[str, Any] = {"mode": mode, "vectors_complete": vectors_complete}
    if code == EXIT_ENV:
        # torch absent (train extra) — N skipped honestly, D proceeds.
        if not (root / "models" / "d-boost" / "meta.json").exists():
            raise Stage2Stop(
                "TRAIN-FAILED",
                f"cortex train exited {code} and D did not train: {out.strip()[:300]}",
            )
        result["n_status"] = "skipped-env-missing"
        result["mode"] = "d"
        _log(
            "candidate N skipped: torch absent (train extra) — disclosed, D-only ladder"
        )
    elif code != 0:
        raise Stage2Stop(
            "TRAIN-FAILED", f"cortex train exited {code}: {out.strip()[:300]}"
        )
    else:
        result["n_status"] = "trained" if mode in ("n", "both") else "not-trained"
    return result


def stage_select(root: Path, mode: str) -> dict[str, Any]:
    """Stage 4: the frozen CV protocol over the train manifest only."""
    root = Path(root)
    argv = [
        "select",
        "--train-manifest",
        str(root / "train-manifest.jsonl"),
        "--candidate",
        "d" if mode == "d" else mode,
        "--out",
        str(root / "selection.json"),
    ]
    code, out = _cli(argv)
    if code != 0:
        raise Stage2Stop(
            "SELECT-FAILED", f"cortex select exited {code}: {out.strip()[:300]}"
        )
    payload = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    verdict = payload["verdict"]
    return {
        "winner": verdict["winner"],
        "margin_stds": verdict["margin_stds"],
        "d_best": payload["d_best"],
        "n_best": payload["n_best"],
        "rule": "ADR 0001 V1: N wins only beyond 1 std repeat-spread; on doubt — D",
    }


def stage_export(
    root: Path, winner: str, embedder_pin: str, corpus_fingerprint_value: str
) -> dict[str, Any]:
    """Stage 5: export the winner as the vesma-cortex ONNX artifact (+ size gate)."""
    root = Path(root)
    model_dir = root / "models" / winner
    if not (model_dir / "meta.json").exists():
        raise Stage2Stop(
            "REUSE-MISSING",
            f"{model_dir}/meta.json does not exist — train the winner first",
        )
    artifact = root / "artifact" / "model.onnx"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "export-artifact",
        "--model",
        str(model_dir),
        "--out",
        str(artifact),
        "--embedder-pin",
        embedder_pin,
        "--corpus-fingerprint",
        corpus_fingerprint_value,
    ]
    code, out = _cli(argv)
    if code != 0:
        raise Stage2Stop(
            "EXPORT-FAILED",
            f"cortex export-artifact exited {code}: {out.strip()[:300]}",
        )
    from cortex.artifacts import assert_artifact_size

    assert_artifact_size(artifact)  # ≤5 MB gate, belt and braces
    manifest = json.loads(
        artifact.with_name("model.manifest.json").read_text(encoding="utf-8")
    )
    return {
        "artifact": str(artifact),
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
        "candidate": manifest["candidate"],
    }


# ── dry-run stand-in ──────────────────────────────────────────────────────────


def build_dry_standin(synth_path: Path, root: Path) -> Path:
    """Materialize the synthetic stand-in as a canon-shaped corpus +
    labels.csv under ``<root>`` so stages 1–5 run through the SAME code
    path. Reads ONLY the in-repo synth jsonl — canon-data is never touched."""
    synth_path = Path(synth_path)
    if not synth_path.exists():
        raise Stage2Stop(
            "DRY-SOURCE-MISSING",
            f"{synth_path} does not exist — the dry-run stand-in source is required",
        )
    rows = _read_jsonl(synth_path)
    corpus_rows: list[dict[str, Any]] = []
    cosine_lines = ["pair_id,cosine"]
    strata_lines = ["pair_id,stratum,band,fallback"]
    label_lines = ["pair_id,label,note"]
    vector_rows: list[dict[str, Any]] = []

    def side(record: dict[str, Any]) -> dict[str, Any]:
        keys = ("title", "body", "tags", "language", "record_type", "created_at")
        return {k: record[k] for k in keys if k in record}

    for row in rows:
        pid = row["pair_id"]
        strategy = row.get("strategy")
        stratum = STRATEGY_STRATUM.get(strategy)
        if stratum is None:
            raise Stage2Stop(
                "DRY-SOURCE-BAD",
                f"pair {pid!r}: unknown strategy {strategy!r} — extend STRATEGY_STRATUM (dry-run glue)",
            )
        corpus_rows.append(
            {"pair_id": pid, "a": side(row["record"]), "b": side(row["candidate"])}
        )
        cosine_lines.append(f"{pid},{row['similarity']}")
        strata_lines.append(f"{pid},{stratum},,no")
        label_lines.append(f"{pid},{row.get('label', '')},")
        if "vec_a" in row and "vec_b" in row:
            vector_rows.append(
                {"pair_id": pid, "vec_a": row["vec_a"], "vec_b": row["vec_b"]}
            )

    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in corpus_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digests = {row["pair_id"]: canon_pair_digest(row) for row in corpus_rows}
    (corpus_dir / "manifest.txt").write_bytes(manifest_bytes(digests.items()))
    (corpus_dir / "strata.csv").write_text(
        "\n".join(strata_lines) + "\n", encoding="utf-8"
    )
    (corpus_dir / "cosines.csv").write_text(
        "\n".join(cosine_lines) + "\n", encoding="utf-8"
    )
    labels_dir = root / "labeling"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "labels.csv").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )
    if vector_rows:
        with (root / "vectors.jsonl").open("w", encoding="utf-8") as handle:
            for row in vector_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return root


# ── report ────────────────────────────────────────────────────────────────────


def _load_prev_report(root: Path) -> dict[str, Any] | None:
    path = Path(root) / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(root: Path, report: dict[str, Any]) -> Path:
    path = Path(root) / "report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    dry = args.dry_run
    root = (
        Path(args.root) if args.root else REPO_ROOT / (DRY_ROOT if dry else REAL_ROOT)
    )
    report: dict[str, Any] = {
        "kind": "stage2-report",
        "dry_run": dry,
        "generated_at": _utc_now(),
        "root": str(root),
        "stages": {},
    }

    if dry:
        standin = build_dry_standin(Path(args.synth_source or DEFAULT_DRY_SYNTH), root)
        corpus_dir = standin / "corpus"
        labels_csv = standin / "labeling" / "labels.csv"
        expected_fp: str | None = None  # no prereg identity for the stand-in
        report["stages"]["standin"] = {"built": str(standin)}
        _log(f"dry-run stand-in built at {standin} (canon-data NOT touched)")
    else:
        corpus_dir = Path(args.corpus) if args.corpus else DEFAULT_CORPUS_DIR
        labels_csv = Path(args.labels) if args.labels else DEFAULT_LABELS_CSV
        if args.expect_corpus_fingerprint is not None:
            # empty string explicitly skips the frozen-fingerprint check
            expected_fp = args.expect_corpus_fingerprint or None
        else:
            expected_fp = EXPECTED_CORPUS_FINGERPRINT

    if args.skip_ingest and not args.skip_split:
        raise Stage2Stop(
            "REUSE-MISSING",
            "--skip-ingest requires --skip-split (the split re-derives from labels)",
        )

    # stage 1: ingest labels (+ corpus identity check)
    ingest: LabelsIngest | None = None
    corpus_rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    corpus_fp = ""
    if not args.skip_ingest:
        corpus_rows, digests, corpus_fp = load_corpus(corpus_dir)
        if expected_fp is not None and corpus_fp != expected_fp:
            raise Stage2Stop(
                "CORPUS-MISMATCH",
                f"corpus fingerprint {corpus_fp} != frozen prereg {expected_fp} — this is not the W5b corpus; "
                "the run is forbidden (a rebuilt corpus means full re-labeling, prereg v2)",
            )
        strata = load_strata(corpus_dir, corpus_rows)
        cosines = load_cosines(corpus_dir, corpus_rows)
        ingest = ingest_labels(labels_csv, {row["pair_id"] for row in corpus_rows})
        report["corpus"] = {
            "path": str(corpus_dir),
            "size": len(corpus_rows),
            "fingerprint_expected": expected_fp,
            "fingerprint_recomputed": corpus_fp,
            "fingerprint_match": expected_fp is None or corpus_fp == expected_fp,
        }
        report["labels"] = {
            "path": str(labels_csv),
            "counts": ingest.counts,
            "floors": {
                "duplicate_min": FLOOR_DUPLICATE_MIN,
                "not_duplicate_min": FLOOR_NOT_DUPLICATE_MIN,
                "disputed_max": FLOOR_DISPUTED_MAX,
                "pass": True,
            },
            "fingerprint_cortex_scheme": ingest.fingerprint_cortex,
            "fingerprint_canon_scheme": ingest.fingerprint_canon,
            "note": "cortex scheme (normalized labels, no note) keys the eval run log; "
            "canon scheme (raw csv values + note) is the value canon STATUS.md records",
        }
        report["stages"]["ingest"] = "ran"
        _log(f"labels ingested: {ingest.counts}, corpus fingerprint verified")
    else:
        prev = _load_prev_report(root)
        if prev is None or "labels" not in prev or "corpus" not in prev:
            raise Stage2Stop(
                "REUSE-MISSING",
                "--skip-ingest needs a previous report.json with labels/corpus sections",
            )
        report["labels"] = prev["labels"]
        report["corpus"] = prev["corpus"]
        corpus_fp = prev["corpus"]["fingerprint_recomputed"]
        report["stages"]["ingest"] = "skipped"

    # stage 2: join + frozen split (skip_ingest ⇒ skip_split was validated
    # above, so inside this block the corpus and labels are always loaded)
    if not args.skip_split:
        vectors = None
        vectors_path = args.vectors or (
            (root / "vectors.jsonl")
            if dry and (root / "vectors.jsonl").exists()
            else None
        )
        if vectors_path:
            vectors = load_vector_sidecar(Path(vectors_path))
        strata = load_strata(corpus_dir, corpus_rows)
        cosines = load_cosines(corpus_dir, corpus_rows)
        outcome = join_and_split(
            corpus_rows=corpus_rows,
            digests=digests,
            strata=strata,
            cosines=cosines,
            labels=ingest.labels,
            vectors=vectors,
            out_root=root,
        )
        report["split"] = {
            "train_rows": outcome.train_rows,
            "holdout_rows": outcome.holdout_rows,
            "disputed_excluded": outcome.disputed_excluded,
            "per_stratum": outcome.per_stratum,
            "holdout_class_counts": outcome.holdout_class_counts,
            "holdout_floor_min": FLOOR_HOLDOUT_CLASS_MIN,
            "split_fingerprint": outcome.split_fingerprint,
            "train_manifest": str(outcome.train_manifest),
            "holdout_dir": str(outcome.holdout_dir),
            "vectors_attached": outcome.vectors_attached,
        }
        report["stages"]["split"] = "ran"
        _log(
            f"split: {outcome.train_rows} train / {outcome.holdout_rows} holdout, disputed excluded {outcome.disputed_excluded}"
        )
    else:
        prev = _load_prev_report(root)
        if prev is None or "split" not in prev:
            raise Stage2Stop(
                "REUSE-MISSING",
                "--skip-split needs a previous report.json with a split section",
            )
        report["split"] = prev["split"]
        report["stages"]["split"] = "skipped"

    train_mode = "auto"
    if not args.skip_train:
        train = stage_train(
            root,
            args.candidate,
            Path(args.pretrain_corpus) if args.pretrain_corpus else None,
        )
        report["train"] = {
            **train,
            "pretrain_corpus": args.pretrain_corpus or str(DEFAULT_PRETRAIN_CORPUS),
        }
        report["stages"]["train"] = "ran"
        _log(f"train: mode={train['mode']} n_status={train.get('n_status')}")
    else:
        prev = _load_prev_report(root)
        if prev is None or "train" not in prev:
            raise Stage2Stop(
                "REUSE-MISSING",
                "--skip-train needs a previous report.json with a train section",
            )
        report["train"] = prev["train"]
        train_mode = prev["train"]["mode"]
        report["stages"]["train"] = "skipped"

    if not args.skip_select:
        mode = report.get("train", {}).get("mode", train_mode)
        selection = stage_select(root, mode)
        report["select"] = selection
        report["stages"]["select"] = "ran"
        _log(
            f"select: winner={selection['winner']} margin_stds={selection['margin_stds']}"
        )
    else:
        prev = _load_prev_report(root)
        if prev is None or "select" not in prev:
            raise Stage2Stop(
                "REUSE-MISSING",
                "--skip-select needs a previous report.json with a select section",
            )
        report["select"] = prev["select"]
        report["stages"]["select"] = "skipped"

    winner = report["select"]["winner"]
    if not args.skip_export:
        export = stage_export(
            root, winner, args.embedder_pin or DEFAULT_EMBEDDER_PIN, corpus_fp
        )
        report["artifact"] = export
        report["stages"]["export"] = "ran"
        _log(
            f"export: {export['artifact']} sha256={export['sha256'][:12]}… size={export['size_bytes']}"
        )
    else:
        prev = _load_prev_report(root)
        if prev is None or "artifact" not in prev:
            raise Stage2Stop(
                "REUSE-MISSING",
                "--skip-export needs a previous report.json with an artifact section",
            )
        report["artifact"] = prev["artifact"]
        report["stages"]["export"] = "skipped"

    report["config"] = {
        "dry_run": dry,
        "candidate": args.candidate,
        "embedder_pin": args.embedder_pin or DEFAULT_EMBEDDER_PIN,
        "pretrain_corpus": args.pretrain_corpus or str(DEFAULT_PRETRAIN_CORPUS),
        "next_step": "single-shot eval is a SEPARATE explicit command: "
        "python scripts/stage2_run.py eval --i-know-this-is-single-shot",
    }
    path = _write_report(root, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    _log(f"report written: {path}")
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """The ONLY path to the holdout — explicit, guarded, single-shot."""
    if not args.i_know_this_is_single_shot:
        raise Stage2Stop(
            "EVAL-GUARD",
            "refusing: the single-shot holdout eval requires --i-know-this-is-single-shot "
            "(prereg W5c: the holdout is touched ONCE per trained artifact)",
        )
    root = (
        Path(args.root)
        if args.root
        else REPO_ROOT / (DRY_ROOT if args.dry_run else REAL_ROOT)
    )
    artifact = (
        Path(args.artifact) if args.artifact else root / "artifact" / "model.onnx"
    )
    if not artifact.exists():
        raise Stage2Stop(
            "REUSE-MISSING",
            f"{artifact} does not exist — run the chain (stages 1–5) first",
        )
    holdout_dir = Path(args.holdout) if args.holdout else root / "holdout"
    if not (holdout_dir / "pairs.jsonl").exists():
        raise Stage2Stop(
            "REUSE-MISSING",
            f"{holdout_dir}/pairs.jsonl does not exist — run the split stage first",
        )

    report = _load_prev_report(root)
    corpus_fp = args.corpus_fingerprint
    if corpus_fp is None:
        if report is None or "corpus" not in report:
            raise Stage2Stop(
                "REUSE-MISSING",
                "no corpus fingerprint: pass --corpus-fingerprint or run the chain first (report.json)",
            )
        corpus_fp = report["corpus"]["fingerprint_recomputed"]

    # Winner pre-check: the current cortex eval refuses n-head artifacts
    # (A5 runner wiring, ADR 0001 V1 dependency) — stop BEFORE the run log
    # with the actionable recipe instead of a bare refusal.
    manifest_path = artifact.with_name("model.manifest.json")
    if manifest_path.exists():
        candidate = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "candidate"
        )
        if candidate == "n-head":
            rows = _read_jsonl(holdout_dir / "pairs.jsonl")
            missing = [
                row["pair_id"]
                for row in rows
                if "vec_a" not in row or "vec_b" not in row
            ]
            detail = (
                f"{len(missing)} holdout rows lack vec_a/vec_b"
                if missing
                else "vectors are present but the A5 runner wiring for N is still missing"
            )
            raise Stage2Stop(
                "VECTORS-NEEDED",
                f"winner is n-head and the single-shot eval for N is not wired yet ({detail}); "
                + _DOCOMPUTE_RECIPE
                + "; the holdout was NOT touched",
            )

    # Run-log placement: the COMMITTED repo run log (artifacts/runs/) is used
    # only for the default real root — the prereg W5c run. A dry run or an
    # explicitly overridden root carries its own run log and never touches
    # the committed one (tests and stand-ins must not pollute tamper-evidence).
    if args.run_log:
        run_log = Path(args.run_log)
    elif args.dry_run or root != REPO_ROOT / REAL_ROOT:
        run_log = root / "run_log.jsonl"
    else:
        run_log = REPO_ROOT / "artifacts" / "runs" / "run_log.jsonl"
    code, out = _cli(
        [
            "eval",
            "--artifact",
            str(artifact),
            "--holdout",
            str(holdout_dir),
            "--run-log",
            str(run_log),
            "--corpus-fingerprint",
            corpus_fp,
        ]
    )
    if out.strip():
        print(out.strip())
    if code == EXIT_REFUSED:
        _log(
            "single-shot refused: this corpus fingerprint was already evaluated (append-only run log)"
        )
    return code


# ── argparse surface ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage2_run",
        description="stage-2 orchestrator: labels.csv → train → select → artifact (eval is separate)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "run",
        help="stages 1–5 (ingest → split → train → select → export); eval NOT included",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic stand-in end to end; canon-data never read, root data/stage2-dry",
    )
    p.add_argument(
        "--root",
        default=None,
        help="output root (default data/stage2, dry: data/stage2-dry)",
    )
    p.add_argument(
        "--corpus",
        default=None,
        help="canon-data corpus dir (default ../vesmaro-canon-data/corpus)",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="labels.csv (default ../vesmaro-canon-data/labeling/labels.csv)",
    )
    p.add_argument(
        "--synth-source", default=None, help="dry-run only: stand-in synth pairs jsonl"
    )
    p.add_argument(
        "--expect-corpus-fingerprint",
        default=None,
        help=f"default {EXPECTED_CORPUS_FINGERPRINT[:12]}… (frozen W5b); pass '' to skip the check",
    )
    p.add_argument(
        "--vectors",
        default=None,
        help="optional vector sidecar jsonl ({pair_id, vec_a, vec_b}) for candidate N",
    )
    p.add_argument(
        "--candidate",
        choices=["auto", "d", "n", "both"],
        default="auto",
        help="auto: both when vectors complete, else D-only (disclosed)",
    )
    p.add_argument(
        "--pretrain-corpus",
        default=None,
        help=f"candidate N corruption pairs (default {DEFAULT_PRETRAIN_CORPUS.name})",
    )
    p.add_argument(
        "--embedder-pin", default=None, help=f"default {DEFAULT_EMBEDDER_PIN[:16]}…"
    )
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="reuse labels/corpus sections from report.json",
    )
    p.add_argument(
        "--skip-split",
        action="store_true",
        help="reuse existing train-manifest + holdout dir",
    )
    p.add_argument("--skip-train", action="store_true", help="reuse existing models/")
    p.add_argument("--skip-select", action="store_true", help="reuse selection.json")
    p.add_argument(
        "--skip-export", action="store_true", help="reuse existing artifact/"
    )

    p = sub.add_parser(
        "eval", help="single-shot holdout eval — EXPLICIT, guarded, not part of run"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="eval the dry-run stand-in artifact (own run log)",
    )
    p.add_argument(
        "--root",
        default=None,
        help="stage-2 root (default data/stage2, dry: data/stage2-dry)",
    )
    p.add_argument(
        "--artifact", default=None, help="default <root>/artifact/model.onnx"
    )
    p.add_argument("--holdout", default=None, help="default <root>/holdout")
    p.add_argument(
        "--run-log",
        default=None,
        help="default artifacts/runs/run_log.jsonl (dry: <root>/run_log.jsonl)",
    )
    p.add_argument(
        "--corpus-fingerprint", default=None, help="default: from report.json"
    )
    p.add_argument(
        "--i-know-this-is-single-shot",
        action="store_true",
        help="required acknowledgment: the holdout is touched ONCE (prereg W5c)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cmd_run if args.command == "run" else cmd_eval
    try:
        return handler(args)
    except Stage2Stop as exc:
        print(f"stage2: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
