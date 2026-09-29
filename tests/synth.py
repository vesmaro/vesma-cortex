"""Programmatic synthetic data for A3b tests — no store, no fixtures.

Labels are BY CONSTRUCTION: a "duplicate" candidate is a mechanical
perturbation of the base record (W4c-positive family), a "not-duplicate"
candidate swaps the topical anchor while keeping the body scaffold and a
HIGH measured similarity — the hard zone the ladder must learn to split
(char divergence under global closeness).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from cortex.features.pair import PairRecord, features

_TOPICS = (
    ("заметка о кофе", "заметка о чае"),
    ("поход в горы", "отчёт о походе"),
    ("рецепт борща", "рецепт щей"),
    ("python tips", "golang tips"),
    ("план на неделю", "план на месяц"),
    ("книга о море", "книга о горе"),
)


def make_records(n: int, seed: int = 0) -> list[PairRecord]:
    rng = random.Random(seed)
    records: list[PairRecord] = []
    for i in range(n):
        topic = _TOPICS[i % len(_TOPICS)][0]
        records.append(
            PairRecord(
                title=topic,
                body=f"{topic} — тело записи номер {i}",
                tags=("memory", "draft", "ru")[: 1 + (i % 3)],
                language="ru" if i % 2 == 0 else "en",
                record_type="note" if i % 3 else "fact",
            )
        )
    rng.shuffle(records)
    return records


def make_pair_rows(n: int, seed: int = 1) -> list[dict]:
    """Labeled train-manifest rows (CLI format), balanced classes."""
    if n % 2:
        n += 1
    rows: list[dict] = []
    for i in range(n):
        duplicate = i % 2 == 0
        topic_pair = _TOPICS[i % len(_TOPICS)]
        anchor, twin, other = topic_pair[0], topic_pair[0] + "!", topic_pair[1]
        record = {
            "title": anchor,
            "body": f"{anchor} — общее тело пары {i % (len(_TOPICS))}",
            "tags": ["memory"],
            "language": "ru",
            "record_type": "note",
        }
        if duplicate:
            candidate = dict(record, title=twin)
            similarity = 0.96
        else:
            candidate = dict(record, title=other)
            similarity = 0.93
        rows.append(
            {
                "pair_id": f"syn-{i:04d}--syn-{i + 500:04d}",
                "record": record,
                "candidate": candidate,
                "similarity": similarity,
                "label": "duplicate" if duplicate else "not-duplicate",
                "stratum": "P1" if duplicate else "N1",
            }
        )
    return rows


def rows_to_vectors(rows: list[dict]):
    from cortex.cli.main import _record_like  # same projection the pipeline uses

    return [
        features(_record_like(r["record"]), _record_like(r["candidate"]), float(r["similarity"]))
        for r in rows
    ]


def rows_labels(rows: list[dict]) -> list[int]:
    return [1 if r["label"] == "duplicate" else 0 for r in rows]


def make_vectors(n: int, seed: int = 1):
    rows = make_pair_rows(n, seed)
    return rows_to_vectors(rows), rows_labels(rows), rows


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path


def unit_vec(rng: np.random.RandomState, dim: int = 384) -> np.ndarray:
    vec = rng.normal(size=dim).astype(np.float32)
    return vec / np.linalg.norm(vec)
