#!/usr/bin/env python3
"""One-off: reconcile fp32-torch student path vs stored int8-export floors.

Stored candidate floors (r5_phase0_characterization.json, int8 ONNX + ORT):
  proxy overall median 0.9216 / S_cal 0.7889
  mono overall median_best_distractor_cos 0.9113
Compares:
  A) fp32 torch ckpt (epoch10 + projector) via StudentEncoder (pilot path)
  B) int8 ONNX export via NanoHarness (gate harness, THIS run's ort)
on the same gate+pool texts (pool identity npz-asserted in Job 1).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus/training-pilot",
)
sys.path.insert(
    0, "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/wt/embed-r4-train/training"
)

from pilot_contrastive import GATE_DIR, load_student, StudentEncoder  # noqa: E402
from audit_kd_teachers import NanoHarness, build_pool, sha256_text, evaluate_space  # noqa: E402

CAND_DIR = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/wt/embed-r4-train/training/runs/embed-r4/export"
)
CORPUS_TRAIN = Path(
    "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus/data/embed-r4/corpus/train.jsonl"
)


def main() -> int:
    rows = [
        json.loads(line)
        for line in open(GATE_DIR / "gate_set.jsonl", encoding="utf-8")
        if line.strip()
    ]
    pair_sides = [
        {
            "src": (sha256_text(r["text_src"]), r["lang_src"]),
            "dst": (sha256_text(r["text_dst"]), r["lang_dst"]),
        }
        for r in rows
    ]
    uniq = {}
    for r in rows:
        for side in ("src", "dst"):
            uniq.setdefault(sha256_text(r["text_" + side]), r["text_" + side])
    gate_order = sorted(uniq)
    gate_texts = [uniq[s] for s in gate_order]
    corpus_rows = [
        json.loads(line)
        for line in open(CORPUS_TRAIN, encoding="utf-8")
        if line.strip()
    ]
    pool = build_pool(rows, corpus_rows)
    pool_texts = {
        "en": [r["text"] for r in pool["en"]],
        "ru": [r["text"] for r in pool["ru"]],
    }
    texts = gate_texts + pool_texts["en"] + pool_texts["ru"]
    n_gate = len(gate_texts)

    # A) fp32 torch path
    import torch

    torch.set_num_threads(8)
    model, tok, proj = load_student("cpu")
    enc = StudentEncoder(model, proj, tok, "cpu")
    vec_fp32 = enc.embed_all(texts)

    # B) int8 export via gate harness
    harness = NanoHarness(CAND_DIR, "candidate-int8")
    vec_int8 = harness.embed_all(texts)

    for name, vec in (("fp32-torch", vec_fp32), ("int8-onnx", vec_int8)):
        out = evaluate_space(
            vec,
            gate_order,
            rows,
            pair_sides,
            pool_texts,
            {"en": vec[n_gate:], "ru": vec[n_gate:]},
        )
        p = out["proxy"]["overall"]
        m = out["retrieval"]["mono"]["overall"]
        print(
            f"{name}: proxy median={p['M_cal_median']} S_cal={p['S_cal_share_ge_085']} "
            f"mono_median={m['median_best_distractor_cos']} "
            f"cross_R@5={out['retrieval']['cross']['overall']['R@5']}"
        )

    per_text = np.array(
        [float(np.dot(vec_fp32[i], vec_int8[i])) for i in range(len(texts))]
    )
    print(
        f"per-text cos(fp32,int8): median={np.median(per_text):.4f} "
        f"min={per_text.min():.4f} p10={np.percentile(per_text, 10):.4f}"
    )
    gm = per_text[:n_gate]
    print(f"gate texts only: median={np.median(gm):.4f} min={gm.min():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
