#!/usr/bin/env python3
"""Round-4 build-gate calibration baseline (contract §3.3 gX1 calibration).

Cross-lingual similarity of the round-4 translated-family pairs under the
FROZEN production pin vesma-embed-v1 (nano, weights sha 3b752e06…) — the
round-4 corpus rows carry no pre-materialized similarities (the mission
premise 'embed_measure already in corpus rows' does not hold for the sealed
r4 artifact; embed_measure() exists in build_corpus_v43.py for the v43 pair
surface). This script measures with the exact engine NanoProvider recipe
(the /tmp/vesma-r5-embeddiag/measure.py harness, prod sha asserted).

Population = every reconstructed translation pair whose sides belong to the
sealed round-4 corpus (same reconstruction machinery as build_gate_set.py,
WITHOUT the round-5 gate exclusions — this is the as-trained round-4 family):
  real 1451 / llm 1617 / seed-dup 80 / twin 330 pairs
  sibling 80 pairs — reported SEPARATELY: contrast (not-duplicate) pairs,
  not translation positives; the chair decides whether they feed the rule.

Output per stratum + aggregate: n, median, mean, P10, share >= 0.85,
share >= 0.80, min/max. Deterministic; single-threaded embedding loop with
4 ONNX intra-op threads.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
GEN_DIR = R4_WT / "data" / "embed-r4" / "gen"
SEED_DIR = R4_WT / "datasets" / "corpus-v43"
OUT_DIR = R4_WT / "data" / "embed-r4" / "r5_gate"
PROD_DIR = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src/vesmaro/models/vesma-embed-v1")
EXPECTED_PROD_SHA = "3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
MAX_SEQ = 256

sys.path.insert(0, str(R5_WT / "datasets" / "embed-r5"))
from build_gate_set import (  # noqa: E402
    GEN_DIR as _G,  # noqa: F401  (consistency check only)
    SEED_DIR as _S,  # noqa: F401
    llm_units,
    load_corpus,
    real_units,
    side_text_seed,
)
from gen_embed_r4_texts import load_jsonl, normalise  # noqa: E402


class NanoHarness:
    """Exact NanoProvider recipe (measure.py), prod artifact."""

    def __init__(self, artifact_dir: Path):
        sha = hashlib.sha256((artifact_dir / "model.onnx").read_bytes()).hexdigest()
        if sha != EXPECTED_PROD_SHA:
            raise SystemExit(f"PROD SHA MISMATCH: {sha}")
        self._tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ)
        self._tokenizer.enable_padding(length=MAX_SEQ)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(artifact_dir / "model.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def embed(self, text: str) -> np.ndarray:
        e = self._tokenizer.encode_batch([text])[0]
        ids = np.array([e.ids], dtype=np.int64)
        mask = np.array([e.attention_mask], dtype=np.int64)
        inputs = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            inputs["token_type_ids"] = np.zeros_like(ids)
        return self._session.run(None, inputs)[0][0]


def stats_of(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "median": round(float(np.median(x)), 4),
        "mean": round(float(np.mean(x)), 4),
        "p10": round(float(np.percentile(x, 10)), 4),
        "p25": round(float(np.percentile(x, 25)), 4),
        "p75": round(float(np.percentile(x, 75)), 4),
        "min": round(float(np.min(x)), 4),
        "max": round(float(np.max(x)), 4),
        "share_ge_085": round(float((x >= 0.85).mean()), 4),
        "share_ge_080": round(float((x >= 0.80).mean()), 4),
        "count_ge_085": int((x >= 0.85).sum()),
    }


def main() -> int:
    corpus = load_corpus()
    print(f"corpus verified rows={len(corpus)}", flush=True)
    model = NanoHarness(PROD_DIR)
    print("prod nano pin verified + session ready", flush=True)

    units_real = real_units()
    units_llm = llm_units()

    strata: dict[str, list[tuple[str, str]]] = {
        "real": [],
        "llm": [],
        "seed-dup": [],
        "twin": [],
        "sibling": [],
    }

    seen: set[str] = set()

    def cls_ok(row, exp: str) -> bool:
        return row is not None and (
            row["source"] == exp or (exp.endswith("-") and row["source"].startswith(exp))
        )

    def try_add(cls: str, a: str, b: str, need_src: bool, exp_a: str, exp_b: str) -> None:
        key = hashlib.sha256(
            (hashlib.sha256(a.encode()).hexdigest() + "\x00"
             + hashlib.sha256(b.encode()).hexdigest()).encode()
        ).hexdigest()
        if key in seen:
            return
        ha = normalise(a).lower()
        hb = normalise(b).lower()
        row_a = corpus.get(hashlib.sha256(ha.encode()).hexdigest())
        row_b = corpus.get(hashlib.sha256(hb.encode()).hexdigest())
        if not cls_ok(row_b, exp_b):
            return
        if need_src and not cls_ok(row_a, exp_a):
            return
        seen.add(key)
        strata[cls].append((a, b))

    for path in sorted(SEED_DIR.glob("batch-translated-dup-*.jsonl")):
        for row in load_jsonl(path):
            rec, tr = row.get("record"), row.get("translation")
            if not isinstance(rec, dict) or not isinstance(tr, dict):
                continue
            t_rec, t_tr = side_text_seed(rec), side_text_seed(tr)
            if t_rec and t_tr:
                try_add("seed-dup", t_rec, t_tr, True, "translated-dup", "translated-dup")
    for path in sorted(GEN_DIR.glob("real-translations.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            unit = units_real.get(row.get("key"))
            if unit is None:
                continue
            tr_text = normalise(row["text"])
            if tr_text:
                try_add("real", unit[0], tr_text, False, "real-part-r4", "translated-real-r4")
    for path in sorted(GEN_DIR.glob("llm-translations.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            unit = units_llm.get(row.get("key"))
            if unit is None:
                continue
            tr_text = normalise(row["text"])
            if tr_text:
                try_add("llm", unit[0], tr_text, False, "", "translated-llm-r4")
    for path in sorted(GEN_DIR.glob("synth-twins.shard*.jsonl")):
        for row in load_jsonl(path):
            if row.get("status") != "ok":
                continue
            ru_t, en_t = normalise(row["ru"]), normalise(row["en"])
            if ru_t and en_t:
                try_add("twin", ru_t, en_t, True, "twin-", "twin-")
    for path in sorted(SEED_DIR.glob("batch-translated-sibling-*.jsonl")):
        for row in load_jsonl(path):
            rec, sib = row.get("record"), row.get("sibling")
            if not isinstance(rec, dict) or not isinstance(sib, dict):
                continue
            t_rec, t_sib = side_text_seed(rec), side_text_seed(sib)
            if t_rec and t_sib:
                try_add("sibling", t_rec, t_sib, True, "translated-sibling", "translated-sibling")

    for k, v in strata.items():
        print(f"stratum {k}: {len(v)} pairs", flush=True)

    results: dict[str, dict] = {}
    sims_all: list[float] = []
    for cls, pairs in strata.items():
        sims: list[float] = []
        for a, b in pairs:
            va, vb = model.embed(a), model.embed(b)
            sims.append(min(1.0, max(0.0, float(np.dot(va, vb)))))
        arr = np.array(sims)
        results[cls] = stats_of(arr)
        print(f"{cls}: {json.dumps(results[cls])}", flush=True)
        if cls != "sibling":
            sims_all.extend(sims)

    results["aggregate_translation_strata"] = stats_of(np.array(sims_all))
    print(f"aggregate(excl sibling): {json.dumps(results['aggregate_translation_strata'])}", flush=True)

    out = {
        "kind": "embed-r5-build-gate-baseline-round4",
        "created": "2026-10-08",
        "pin": {
            "model": "vesma-embed-v1 (nano)",
            "weights_sha256": EXPECTED_PROD_SHA,
            "verified": True,
            "recipe": "engine NanoProvider (tokenizer trunc/pad 256, onnxruntime CPU, in-graph L2 norm), "
            "cos = dot of full-384 vectors, clipped [0,1] (embed_measure convention)",
        },
        "population": {k: len(v) for k, v in strata.items()},
        "note": "sibling = contrast (not-duplicate) pairs, excluded from the translation aggregate; "
        "round-4 corpus rows carry no pre-materialized similarities — measured here with the frozen pin",
        "results": results,
    }
    out_path = OUT_DIR / "build_gate_baseline.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"sha256 {hashlib.sha256(out_path.read_bytes()).hexdigest()}", flush=True)

    with open(R5_WT / "datasets" / "embed-r5" / "run-log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts_step": "build_gate_baseline",
            "msg": f"baseline complete; population={out['population']}; "
            f"aggregate={results['aggregate_translation_strata']}",
        }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
