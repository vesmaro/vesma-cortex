#!/usr/bin/env python3
"""A2 field-cosine docompute sidecar — runs in the ENGINE environment.

Deliberately OUTSIDE src/cortex: it imports the engine's NanoProvider
(bundle mnema-embed-v1, CPU, zero network) and needs onnxruntime +
``tokenizers`` — deps of the engine venv, not of this library. Run it as:

    PYTHONPATH=<engine-src> <engine-venv>/bin/python scripts/a2_field_cosines.py \
        --records data/pretrain/<corpus-id>/records.jsonl \
        --pairs  data/pretrain/<corpus-id>/near_dup_candidates.jsonl \
        --out-npz data/vectors/<corpus-id>/field_vecs.npz

Privacy invariant (frozen): the input is the EXPORTED records.jsonl —
every row already passed the prereg hygiene chain. This script never
opens the store.

Outputs:
- field_vecs.npz — per-record field vectors (ids, title/body/tags) +
  embedder provenance; reader lives in cortex.data.field_cosines;
- near_dup_field_cosines.jsonl (with --pairs) — pair_id → cos_title /
  cos_body / cos_tags, attach-ready for cortex.features.pair.

Progress goes to stderr, a JSON summary to stdout. No cortex imports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="a2_field_cosines",
        description="docompute title/body/tags field vectors + pair cosines (mnema-embed-v1, CPU)",
    )
    parser.add_argument(
        "--records", required=True, help="exported records.jsonl (hygiene-passed)"
    )
    parser.add_argument(
        "--pairs", default=None, help="near_dup_candidates.jsonl (optional)"
    )
    parser.add_argument("--out-npz", required=True, help="output field_vecs.npz path")
    parser.add_argument(
        "--out-cosines",
        default=None,
        help="output pair-cosines jsonl (default: next to npz)",
    )
    parser.add_argument(
        "--engine-src", required=True, help="engine source tree (vesmaro package root)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4096,
        help="per-field text cap (engine EMBEDDING_TEXT_MAX_CHARS convention)",
    )
    return parser.parse_args(argv)


def _load_engine_provider(engine_src: str):
    src = Path(engine_src).resolve()
    if not (src / "vesmaro" / "embeddings" / "__init__.py").is_file():
        raise SystemExit(f"engine source tree has no vesmaro.embeddings under {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from vesmaro.embeddings import (
        MNEMA_EMBED_MODEL,
        NanoProvider,
    )

    provider = NanoProvider()  # eager init: bundle load + smoke-inference
    return provider, MNEMA_EMBED_MODEL


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    args = _parse_args(argv)
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"a2_field_cosines: {message}", file=sys.stderr)

    provider, model_name = _load_engine_provider(args.engine_src)
    progress(
        f"embedder ready: {model_name} fingerprint={provider.fingerprint} dim={provider.dimension}"
    )

    ids: list[str] = []
    titles: list[str] = []
    bodies: list[str] = []
    tags_texts: list[str] = []
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(str(row["id"]))
        titles.append(str(row.get("title", ""))[: args.max_chars])
        bodies.append(str(row.get("body", ""))[: args.max_chars])
        tags = row.get("tags", [])
        tags_texts.append(" ".join(str(tag) for tag in tags)[: args.max_chars])
    if not ids:
        raise SystemExit(f"{args.records}: no records")
    progress(f"records loaded: {len(ids)}")

    def embed_block(texts: list[str]) -> np.ndarray:
        return np.asarray(provider.embed_batch(texts), dtype=np.float32)

    t0 = time.perf_counter()
    title_vecs = embed_block(titles)
    progress(f"title vectors done in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    body_vecs = embed_block(bodies)
    progress(f"body vectors done in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    tags_vecs = embed_block(tags_texts)
    progress(f"tags vectors done in {time.perf_counter() - t0:.1f}s")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        ids=np.asarray(ids),
        title_vecs=title_vecs,
        body_vecs=body_vecs,
        tags_vecs=tags_vecs,
        embedder_fingerprint=np.asarray(provider.fingerprint),
        embedder_model=np.asarray(model_name),
    )
    progress(f"wrote {out_npz}")

    pairs_written = 0
    if args.pairs:
        out_cosines = (
            Path(args.out_cosines)
            if args.out_cosines
            else out_npz.parent / "near_dup_field_cosines.jsonl"
        )
        index = {record_id: i for i, record_id in enumerate(ids)}

        def cosine(block: np.ndarray, i: int, j: int) -> float:
            denominator = float(np.linalg.norm(block[i]) * np.linalg.norm(block[j]))
            if denominator == 0.0:
                return 1.0
            # float32 overshoot clamp — keeps the file attach-ready
            # (cortex.features.pair.attach_field_cosines validates [-1, 1]).
            return min(1.0, max(-1.0, float(np.dot(block[i], block[j]) / denominator)))

        out_cosines.parent.mkdir(parents=True, exist_ok=True)
        with out_cosines.open("w", encoding="utf-8") as handle:
            for line in Path(args.pairs).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                pair = json.loads(line)
                i, j = index[pair["id_a"]], index[pair["id_b"]]
                handle.write(
                    json.dumps(
                        {
                            "pair_id": pair["pair_id"],
                            "cos_title": cosine(title_vecs, i, j),
                            "cos_body": cosine(body_vecs, i, j),
                            "cos_tags": cosine(tags_vecs, i, j),
                        }
                    )
                    + "\n"
                )
                pairs_written += 1
        progress(f"wrote {out_cosines} ({pairs_written} pairs)")

    summary = {
        "records": len(ids),
        "pairs_cosines": pairs_written,
        "dimension": int(provider.dimension),
        "embedder_fingerprint": provider.fingerprint,
        "embedder_model": model_name,
        "out_npz": str(out_npz),
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
