#!/usr/bin/env python3
"""kNN train-manifold sidecar for the ADDENDUM-4 characterization block (§E).

Builds ONCE, BEFORE the attempt-4 gate cycle, the train-side manifold the
gate runner reads for the kNN characterization (non-gating):

  - manifold = BOTH sides of every row of the v4.5 train corpus, rendered by
    the FROZEN builder's own ``embed_side`` convention
    (``title\nbody\ntags`` — the exact text the frozen embedder sees,
    including its 4096-char truncation);
  - dedup key = sha256 of the truncated embed text (two sides that embed
    identically are ONE manifold point — keeps the leave-one-out zero
    honest);
  - embedder = the frozen ``make_embedder`` (pin assert inside:
    ``nano:sha256:3b752e06…``), 384-dim, L2-normalized by the ONNX graph;
  - the engine 6.4.0 package rename (vesmaro -> vesma) is bridged by the
    same alias shim as ``b2_g1b_probes.py``;
  - output: ONE .npz (keys, float32 vecs, meta json). The runner pins the
    file's sha256 as a prolog constant (fail-closed, "наравне с бустерами").

This script computes NO distances and NO quantiles: the train-to-train
null distribution and the probe distances live in ONE code path — the gate
runner's (ADDENDUM-4 §E / MLC Р5). No RNG. Never touches eval surfaces.

Usage:
    python scripts/b2_knn_train_sidecar.py \
        --corpus-dir data/b2-graph-corpus/v45 \
        --out data/b2-graph-corpus/v45/attempt-4/knn-train-sidecar.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# APPEND, not insert(0) — same reason as b2_g1b_probes.py: cortex lives in
# src/, vesma/vesmaro/mnemos must resolve via ENGINE_SRC (prepended by the
# builder at make_embedder time).
sys.path.append(str(REPO_ROOT / "src"))

from b2_graph_corpus_v45 import (  # noqa: E402
    ENGINE_SRC,
    make_embedder,
)
from b2_rev1n_lib import load_jsonl  # noqa: E402
from cortex.data.fingerprints import (  # noqa: E402
    corpus_fingerprint,
    manifest_bytes,
)

TRAIN_FP_PIN = "3bf7ed53e1dbf9092ec1e557c539b6fedd202ffd8fdbe9798643e2f6fc828a70"
EMBEDDER_PIN_PREFIX = "nano:sha256:3b752e06"
EMBEDDER_DIM = 384


def _ensure_vesmaro_alias() -> None:
    """Engine 6.4.0 renamed the package vesmaro -> vesma (engine commit
    5e0e4b4); the frozen builder still imports ``vesmaro.embeddings``.
    Same shim as b2_g1b_probes.py (pattern carried over verbatim)."""
    import importlib
    import types

    try:
        importlib.import_module("vesmaro.embeddings")
        return
    except ImportError:
        pass
    if ENGINE_SRC not in sys.path:
        sys.path.insert(0, ENGINE_SRC)
    vesmaro = types.ModuleType("vesmaro")
    vesmaro.__path__ = [str(Path(ENGINE_SRC) / "vesma")]
    sys.modules["vesmaro"] = vesmaro
    emb = importlib.import_module("vesma.embeddings")
    sys.modules["vesmaro.embeddings"] = emb
    vesmaro.embeddings = emb


_ensure_vesmaro_alias()


def refuse(msg: str) -> None:
    raise SystemExit(f"[knn-sidecar] REFUSE: {msg}; STOP")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b2-knn-train-sidecar")
    parser.add_argument("--corpus-dir", default="data/b2-graph-corpus/v45")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    train_path = corpus_dir / "train.jsonl"

    # ── corpus fingerprint assert (frozen input, ADDENDUM-4 §C) ───────────
    rows = load_jsonl(train_path)
    fp = corpus_fingerprint(
        manifest_bytes((r["pair_id"], r["pair_sha256"]) for r in rows)
    )
    if fp != TRAIN_FP_PIN:
        refuse(f"train corpus fingerprint {fp} != pinned {TRAIN_FP_PIN}")

    # ── collect the manifold texts (both sides, builder convention) ───────
    _embed_side, embed_text, pin = make_embedder()
    if not pin.startswith(EMBEDDER_PIN_PREFIX):
        refuse(f"embedder pin {pin!r} differs from the frozen prefix")

    texts: dict[str, str] = {}  # dedup key (sha of truncated text) -> text
    sides_seen = 0
    for row in rows:
        for side in (row["record"], row["candidate"]):
            sides_seen += 1
            # EXACTLY what embed_side embeds — truncation included in the
            # dedup key so two texts differing only past 4096 chars are one
            # manifold point.
            tags = " ".join(side.get("tags") or [])
            text = (f"{side.get('title') or ''}\n{side.get('body') or ''}\n{tags}")[
                :4096
            ]
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            texts.setdefault(key, text)

    n = len(texts)
    print(
        f"train rows={len(rows)} sides={sides_seen} unique_manifold_points={n}",
        flush=True,
    )

    # ── one pass over the unique texts (deterministic key order) ─────────
    keys = sorted(texts)
    vecs = np.empty((n, EMBEDDER_DIM), dtype=np.float32)
    t0 = time.monotonic()
    for i, key in enumerate(keys):
        # the collected text IS the exact embed_side layout
        # (f"{title}\n{body}\n{tags}") — embed it through embed_text
        # directly, no re-rendering.
        arr = np.asarray(embed_text(texts[key]), dtype=np.float32)
        if arr.shape != (EMBEDDER_DIM,):
            refuse(f"embedder returned dim {arr.shape}, expected {EMBEDDER_DIM}")
        vecs[i] = arr
        if (i + 1) % 500 == 0:
            print(f"  embedded {i + 1}/{n} ({time.monotonic() - t0:.0f}s)", flush=True)

    norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        refuse(
            f"embedder output not L2-normalized (norm range "
            f"{norms.min():.6f}..{norms.max():.6f})"
        )

    meta = {
        "artifact": "knn-train-sidecar (ADDENDUM-4 §E characterization, non-gating)",
        "attempt": 4,
        "corpus_dir": str(corpus_dir),
        "train_fingerprint": fp,
        "train_rows": len(rows),
        "sides_seen": sides_seen,
        "manifold_points": n,
        "embedder_pin": pin,
        "embedder_dim": EMBEDDER_DIM,
        "dedup_key": "sha256 of the truncated embed text (title\\nbody\\ntags, 4096)",
        "storage": "float32; distances computed float64 by the gate runner",
        "builder_script": "scripts/b2_knn_train_sidecar.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "frozen input of the attempt-4 gate cycle; sha256 pinned in "
        "scripts/b2_gate_rev1n1.py prolog (fail-closed). Rebuilds are "
        "NOT promised byte-identical (npz container timestamps).",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        keys=np.asarray(keys, dtype=np.str_),
        vecs=vecs,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=np.str_),
    )
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print("meta:", json.dumps(meta, ensure_ascii=False))
    print(f"sidecar: {out}")
    print(f"sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
