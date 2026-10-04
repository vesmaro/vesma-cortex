#!/usr/bin/env python3
"""Assemble the v2 Layer A eval sets (wave LA-2) — v1 recipes + the LLM batch.

The LA-2 LLM batch (``datasets/evalsets/llm-batches/la2-llm-batch-1.jsonl``)
fills the three honest empty slots of LA-1: ``translation-twins`` (RU/EN
language twins), ``llm-paraphrase`` (same thought restructured) and
``llm-near-topic`` (topically close, another fact — the hard negative).

Assembly contract (eval-methodology §3 + generate.py module docstring):

- a frozen set never changes in place — the batch produces NEW set
  versions (``merge-v2`` / ``release-v2``) with new fingerprints and a
  manifest addendum; the v1 files stay byte-frozen;
- merge-v2 rides the SAME procedural base as merge-v1 plus a DETERMINISTIC
  SUBSET of the batch: the first ``MERGE_SLOTS_PER_CLASS`` rows of each
  slot class in batch order (mirrors the v1 prefix convention where the
  merge set is a subset of the release set);
- release-v2 = the release-v1 procedural base + the FULL batch;
- every batch row is validated BEFORE assembly: schema (through the
  builder's slot contract), per-class counts, label semantics, language
  contract, text dedup (within the batch AND against every v1 probe
  text), topic disjointness from the train seed list and the eval
  anchors, and a ``cortex.features.pair.features`` pass on every pair.

Usage:
    uv run python scripts/gen_evalsets_v2.py          # write v2 sets
    uv run python scripts/gen_evalsets_v2.py --check  # verify only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Final

from cortex.eval.sanity import PROBE_RECORD, UNRELATED_RECORD
from cortex.evalsets import (
    MERGE_V1,
    RELEASE_V1,
    DEFAULT_SEED,
    EvalSetError,
    build_eval_set,
    eval_set_jsonl,
    eval_set_manifest_bytes,
    load_eval_set,
    load_llm_slots,
)
from cortex.evalsets.taxonomy import (
    LLM_SLOT_PROBE_CLASSES,
    PROBE_CLASS_LLM_NEAR_TOPIC,
    PROBE_CLASS_TRANSLATION_TWINS,
    SOURCE_LLM_LA2,
)
from cortex.evalsets.topics import EVAL_TOPICS
from cortex.features.pair import FEATURE_NAMES, features
from cortex.pretrain.corruption import record_key
from cortex.synth.generate import TOPICS

#: Wave date (constant — regeneration must not churn the meta files).
CREATED: Final = "2026-10-04"

GENERATOR: Final = "la2-llm-batch-1"
BATCH_PATH: Final = Path("datasets/evalsets/llm-batches/la2-llm-batch-1.jsonl")

#: Rows per slot class in the committed batch.
BATCH_ROWS_PER_CLASS: Final[int] = 12

#: merge-v2 takes the FIRST rows per class, batch order (v1 prefix convention).
MERGE_SLOTS_PER_CLASS: Final[int] = 5

MERGE_V2: Final = dataclasses.replace(MERGE_V1, set_id="merge-v2")
RELEASE_V2: Final = dataclasses.replace(RELEASE_V1, set_id="release-v2")

RECIPES: Final = (MERGE_V2, RELEASE_V2)


def select_merge_slots(rows: list[dict]) -> list[dict]:
    """First MERGE_SLOTS_PER_CLASS rows per slot class, batch order."""
    counts: dict[str, int] = {name: 0 for name in LLM_SLOT_PROBE_CLASSES}
    selected = []
    for row in rows:
        name = str(row["class"])
        if counts[name] < MERGE_SLOTS_PER_CLASS:
            selected.append(row)
            counts[name] += 1
    if any(c != MERGE_SLOTS_PER_CLASS for c in counts.values()):
        raise EvalSetError(
            f"merge slot selection failed: per-class counts {counts} "
            f"(need {MERGE_SLOTS_PER_CLASS} each) — the batch is too small"
        )
    return selected


def _row_sides(row: dict) -> tuple[dict, dict]:
    return row["record"], row["candidate"]


def _as_record(side: dict):
    from cortex.features.pair import PairRecord

    return PairRecord(
        title=str(side["title"]),
        body=str(side["body"]),
        tags=tuple(str(t) for t in side.get("tags", ())),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def validate_batch(rows: list[dict], evalsets_dir: Path) -> None:
    """The full LA-2 admission check — fail loud, never warn through."""
    if len(rows) != BATCH_ROWS_PER_CLASS * len(LLM_SLOT_PROBE_CLASSES):
        raise EvalSetError(
            f"batch must carry {BATCH_ROWS_PER_CLASS} rows per slot class "
            f"({BATCH_ROWS_PER_CLASS * len(LLM_SLOT_PROBE_CLASSES)} total), got "
            f"{len(rows)}"
        )

    # per-class counts, label semantics, language contract, feature pass
    counts = {name: 0 for name in LLM_SLOT_PROBE_CLASSES}
    for n, row in enumerate(rows):
        name = str(row["class"])
        if name not in counts:
            raise EvalSetError(f"batch[{n}]: class {name!r} is not an LLM slot")
        counts[name] += 1
        label = str(row["label"])
        if name == PROBE_CLASS_LLM_NEAR_TOPIC:
            if label != "not-duplicate":
                raise EvalSetError(
                    f"batch[{n}]: llm-near-topic is the hard-negative class, "
                    f"label must be not-duplicate, got {label!r}"
                )
        elif label != "duplicate":
            raise EvalSetError(
                f"batch[{n}]: {name} probes are duplicate-labeled twins, got {label!r}"
            )
        if str(row["source"]) != SOURCE_LLM_LA2:
            raise EvalSetError(f"batch[{n}]: source must be {SOURCE_LLM_LA2!r}")
        rec, cand = _row_sides(row)
        if name == PROBE_CLASS_TRANSLATION_TWINS:
            if rec["language"] == cand["language"]:
                raise EvalSetError(
                    f"batch[{n}]: translation twin sides must differ in language"
                )
        elif rec["language"] != cand["language"]:
            raise EvalSetError(f"batch[{n}]: {name} sides must share the language")
        # the #480 rule: features through the frozen contract on EVERY pair
        vector = features(_as_record(rec), _as_record(cand), float(row["similarity"]))
        if vector.names != FEATURE_NAMES or len(vector.values) != len(FEATURE_NAMES):
            raise EvalSetError(f"batch[{n}]: feature contract violated")

    for name, count in counts.items():
        if count != BATCH_ROWS_PER_CLASS:
            raise EvalSetError(
                f"class {name}: {count} rows, need {BATCH_ROWS_PER_CLASS}"
            )

    # text dedup: no two identical texts anywhere — within the batch, and
    # against every v1 probe text (both frozen sets)
    seen_keys: set[str] = set()
    seen_texts: set[tuple[str, str, str]] = set()
    for set_id in ("merge-v1", "release-v1"):
        frozen = load_eval_set(evalsets_dir / f"{set_id}.jsonl")
        for probe in frozen.probes:
            for side in (probe.record, probe.candidate):
                seen_keys.add(record_key(side))
                seen_texts.add((side.title, side.body, side.language or ""))
    for n, row in enumerate(rows):
        for side_raw in _row_sides(row):
            side = _as_record(side_raw)
            key = record_key(side)
            if (
                key in seen_keys
                or (side.title, side.body, side.language or "") in seen_texts
            ):
                raise EvalSetError(
                    f"batch[{n}]: text duplicates an existing probe record "
                    f"(record_key {key!r}) — dedup violated"
                )
            seen_keys.add(key)
            seen_texts.add((side.title, side.body, side.language or ""))

    # topic disjointness: batch texts never train-topic or eval-topic themselves
    train_titles = {topic.title for topic in TOPICS}
    train_keys = {record_key(topic.as_record()) for topic in TOPICS}
    eval_titles = {topic.title for topic in EVAL_TOPICS} | {
        PROBE_RECORD.title,
        UNRELATED_RECORD.title,
    }
    eval_keys = {record_key(r) for r in (PROBE_RECORD, UNRELATED_RECORD)} | {
        record_key(topic.as_record()) for topic in EVAL_TOPICS
    }
    for n, row in enumerate(rows):
        for side_raw in _row_sides(row):
            side = _as_record(side_raw)
            if side.title in train_titles or side.title in eval_titles:
                raise EvalSetError(
                    f"batch[{n}]: title {side.title!r} collides with a train/eval "
                    "topic — the eval surface must not train-topic itself"
                )
            key = record_key(side)
            if key in train_keys or key in eval_keys:
                raise EvalSetError(
                    f"batch[{n}]: record_key {key!r} collides with a train/eval record"
                )


def meta_payload(eval_set, llm_rows: int) -> dict:
    return {
        "schema_version": 1,
        "set_id": eval_set.set_id,
        "role": eval_set.role,
        "eval_set_sha256": eval_set.eval_set_sha256,
        "n_pairs": len(eval_set.probes),
        "per_class": eval_set.per_class,
        "seed": DEFAULT_SEED,
        "created": CREATED,
        "generator": (
            f"scripts/gen_evalsets_v2.py (cortex.evalsets.build_eval_set + {GENERATOR})"
        ),
        "llm_batch": str(BATCH_PATH),
        "llm_batch_rows": llm_rows,
        "llm_generator": GENERATOR,
        "fingerprint_scheme": (
            "data-contract §5: pair_sha256 over canonical "
            "{record, candidate, similarity, label}; manifest 'pair_id <sha>' "
            "sorted by pair_id; BLAKE2b-256 over manifest bytes"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen-evalsets-v2",
        description="assemble the v2 Layer A eval sets (v1 recipes + LA-2 LLM batch)",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/evalsets",
        help="output directory (default datasets/evalsets)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing files instead of writing (CI-mode byte check)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    rows = load_llm_slots(BATCH_PATH)
    validate_batch(rows, out_dir)
    merge_rows = select_merge_slots(rows)
    plans = (
        (MERGE_V2, merge_rows),
        (RELEASE_V2, rows),
    )

    for recipe, slots in plans:
        eval_set = build_eval_set(recipe, seed=DEFAULT_SEED, llm_slots=slots)
        jsonl_text = eval_set_jsonl(eval_set.probes)
        manifest = eval_set_manifest_bytes(eval_set.probes)
        meta_text = (
            json.dumps(
                meta_payload(eval_set, len(slots)),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        jsonl_path = out_dir / f"{eval_set.set_id}.jsonl"
        manifest_path = out_dir / f"{eval_set.set_id}.manifest.txt"
        meta_path = out_dir / f"{eval_set.set_id}.meta.json"

        if args.check:
            for path, expected in (
                (jsonl_path, jsonl_text),
                (manifest_path, manifest.decode("utf-8")),
                (meta_path, meta_text),
            ):
                if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                    print(
                        f"gen-evalsets-v2: --check FAILED for {path} "
                        "(missing or byte-divergent from the deterministic rebuild)",
                        file=sys.stderr,
                    )
                    return 1
            print(
                f"{eval_set.set_id}: check OK — {len(eval_set.probes)} pairs, "
                f"sha256={eval_set.eval_set_sha256}"
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(jsonl_text, encoding="utf-8")
        manifest_path.write_bytes(manifest)
        meta_path.write_text(meta_text, encoding="utf-8")
        print(
            f"{eval_set.set_id}: wrote {len(eval_set.probes)} pairs to "
            f"{jsonl_path} — eval_set_sha256={eval_set.eval_set_sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
