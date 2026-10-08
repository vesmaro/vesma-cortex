#!/usr/bin/env python3
"""Phase-0.5 Job 1 — KD-teacher audit for round-5 (project vesma-cortex).

Prereg embed-round5-prereg.md §2 «Учитель» arm + ADR-0005: audit light
multilingual teachers (e5-base / LaBSE / MiniLM-L12-v2) BEFORE any round-5
training. Audit criteria (prereg'd, numbers only here):
  (a) median translated-pair cosine >= 0.90 on the 144 burned pairs
      (corpus-v43-redo stage2 v43 train.jsonl, stratum=translated-dup,
      label=duplicate — read-only);
  (b) monolingual probe intact (gX0 harness): |Delta| of the gate-set mono
      control median (median_best_distractor_cos) vs production 3b752e06
      <= 0.005 — same control protocol as /tmp/r5-charact/run_phase0.py.
Also measured (feeds the arm decision, NOT gate verdicts): per candidate on
the sealed gate set cb367577 — proxy S_cal (share >= 0.85) / median / cross
R@5 with the exact run_phase0 harness (same 240/lang distractor pool,
seed 42).

Measured numbers only. Single deterministic pass per model.
Read-only inputs; writes only under training-pilot/results/.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

# ---------------- paths and expected fingerprints ----------------
R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
GATE_DIR = R4_WT / "data" / "embed-r4" / "r5_gate"
GATE_FILE = GATE_DIR / "gate_set.jsonl"
GATE_MANIFEST = GATE_DIR / "manifest.json"
EXPECTED_GATE_SHA = "cb36757769fb88f2806065f50348f5fd8383cfdb0c19c66e6335eac8ace01123"

CORPUS_DIR = R4_WT / "data" / "embed-r4" / "corpus"
CORPUS_TRAIN = CORPUS_DIR / "train.jsonl"
EXPECTED_CORPUS_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"

PROD_DIR = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src/vesmaro/models/vesma-embed-v1")
EXPECTED_PROD_SHA = "3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"

BURNED_FILE = REDO_WT / "data" / "stage2" / "v43" / "train.jsonl"

CHARACT_JSON = Path("/tmp/r5-charact/r5_phase0_characterization.json")
CHARACT_NPZ = Path("/tmp/r5-charact/r5_phase0_embeddings.npz")

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = OUT_DIR / "audit_kd_teachers.json"

MAX_SEQ = 256
POOL_PER_LANG = 240   # identical to run_phase0.py
POOL_SEED = 42        # identical to run_phase0.py
RANK_K = 5

E5_PREFIX = "query: "  # intfloat model card: symmetric tasks -> "query: " on both sides

CANDIDATES = [
    {
        "id": "intfloat/multilingual-e5-base",
        "short": "e5-base",
        "params": "278M",
        "license": "MIT",
        "kind": "hf-mean",
        "prefix": E5_PREFIX,
        "dim": 768,
    },
    {
        "id": "sentence-transformers/LaBSE",
        "short": "LaBSE",
        "params": "471M",
        "license": "Apache-2.0",
        "kind": "st",
        "prefix": "",
        "dim": 768,
    },
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short": "MiniLM-L12-v2",
        "params": "118M",
        "license": "Apache-2.0",
        "kind": "st",
        "prefix": "",
        "dim": 384,
    },
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------- NanoHarness: exact NanoProvider recipe (run_phase0.py) ----------------


class NanoHarness:
    def __init__(self, artifact_dir: Path, name: str):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.name = name
        self.weights_sha256 = sha256_file(artifact_dir / "model.onnx")
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
        self._model_inputs = {inp.name for inp in self._session.get_inputs()}

    def embed_all(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            e = self._tokenizer.encode_batch([t])[0]
            ids = np.array([e.ids], dtype=np.int64)
            mask = np.array([e.attention_mask], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._model_inputs:
                inputs["token_type_ids"] = np.zeros_like(ids)
            out = self._session.run(None, inputs)
            v = out[0][0]
            n = float(np.linalg.norm(v))
            if abs(n - 1.0) > 1e-3:
                v = v / n
            vecs.append(v)
        return np.vstack(vecs)


# ---------------- HF encoders ----------------


class HFMeanEncoder:
    """HF AutoModel + (optional prefix) + mask-aware mean pooling + L2."""

    def __init__(self, model_id: str, prefix: str, max_length: int = MAX_SEQ, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.prefix = prefix
        self.batch_size = batch_size
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        mpe = getattr(self._model.config, "max_position_embeddings", max_length)
        self.max_length = min(max_length, int(mpe))
        torch.set_grad_enabled(False)

    def embed_all(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [self.prefix + t for t in texts[start : start + self.batch_size]]
            enc = self._tok(
                chunk, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            hs = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            emb = (hs * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float64))
        return np.vstack(out)


class STEncoder:
    """sentence-transformers loader (pooling config comes from the repo)."""

    def __init__(self, model_id: str, max_length: int = MAX_SEQ, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.batch_size = batch_size
        self._st = SentenceTransformer(model_id)
        self._st.max_seq_length = max_length
        pool = self._st[0]
        self.pooling_mode = getattr(pool, "get_pooling_config_str", lambda: "?")()

    def embed_all(self, texts: list[str]) -> np.ndarray:
        return (
            self._st.encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float64)
        )


# ---------------- harness statistics (run_phase0 protocol) ----------------


def proxy_stats(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "S_cal_share_ge_085": round(float((x >= 0.85).mean()), 4),
        "count_ge_085": int((x >= 0.85).sum()),
        "M_cal_median": round(float(np.median(x)), 4),
        "P10": round(float(np.percentile(x, 10)), 4),
        "mean": round(float(np.mean(x)), 4),
        "min": round(float(np.min(x)), 4),
        "max": round(float(np.max(x)), 4),
        "P90": round(float(np.percentile(x, 90)), 4),
    }


def subset_stats(x: np.ndarray, mask) -> dict:
    m = np.array(mask, dtype=bool)
    return proxy_stats(x[m]) if m.any() else {"n": 0}


def rank_stats(ranks: np.ndarray) -> dict:
    return {
        "n": int(ranks.size),
        "R@5": round(float((ranks <= RANK_K).mean()), 4),
        "R@1": round(float((ranks == 1).mean()), 4),
        "median_rank": float(np.median(ranks)),
        "mean_rank": round(float(np.mean(ranks)), 2),
        "max_rank": int(ranks.max()),
    }


def build_pool(rows: list[dict], corpus_rows: list[dict]):
    """Verbatim run_phase0.py pool construction (seed 42, 240/lang,
    family-stratified largest-remainder, sha-sorted canonical order)."""
    side_families: dict[str, dict[str, int]] = {"en": {}, "ru": {}}
    for r in rows:
        for side in ("src", "dst"):
            ref = r.get("corpus_row_" + side)
            if ref and ref.get("source"):
                lang = r["lang_" + side]
                side_families[lang][ref["source"]] = side_families[lang].get(ref["source"], 0) + 1
    gate_texts_en = {r["text_src"] if r["lang_src"] == "en" else r["text_dst"] for r in rows}
    gate_texts_ru = {r["text_src"] if r["lang_src"] == "ru" else r["text_dst"] for r in rows}

    seen_pool_sha = set()
    deduped: list[dict] = []
    for rec in corpus_rows:
        rec["_sha"] = sha256_text(rec["text"])
        if rec["_sha"] in seen_pool_sha:
            continue
        seen_pool_sha.add(rec["_sha"])
        deduped.append(rec)
    corpus_rows = deduped

    uniq_texts: dict[str, str] = {}
    for r in rows:
        for side in ("src", "dst"):
            t = r["text_" + side]
            uniq_texts.setdefault(sha256_text(t), t)
    text_of_sha = dict(uniq_texts)

    pool: dict[str, list[dict]] = {"en": [], "ru": []}
    rng = random.Random(POOL_SEED)
    for lang in ("en", "ru"):
        fams = sorted(side_families[lang])
        by_fam: dict[str, list[dict]] = {f: [] for f in fams}
        for rec in corpus_rows:
            if rec["lang"] != lang or rec["source"] not in by_fam:
                continue
            if rec["text"] in gate_text_set_check(rows) or rec["_sha"] in text_of_sha:
                continue
            by_fam[rec["source"]].append(rec)
        for f in fams:
            by_fam[f].sort(key=lambda r: r["_sha"])
        avail = {f: len(by_fam[f]) for f in fams if by_fam[f]}
        total_avail = sum(avail.values())
        assert total_avail >= POOL_PER_LANG, f"pool too small for {lang}: {total_avail}"
        exact = {f: POOL_PER_LANG * n / total_avail for f, n in avail.items()}
        alloc = {f: int(np.floor(exact[f])) for f in avail}
        alloc = {f: max(2, alloc[f]) for f in alloc}
        rem = POOL_PER_LANG - sum(alloc.values())
        for f in sorted(avail, key=lambda f: (-(exact[f] - np.floor(exact[f])), f)):
            if rem <= 0:
                break
            alloc[f] += 1
            rem -= 1
        for f in sorted(avail):
            take = min(alloc[f], avail[f])
            pool[lang].extend(rng.sample(by_fam[f], take))
        pool[lang].sort(key=lambda r: r["_sha"])
    return pool


def gate_text_set_check(rows: list[dict]) -> set:
    s = set()
    for r in rows:
        s.add(r["text_src"])
        s.add(r["text_dst"])
    return s


def evaluate_space(vec: np.ndarray, gate_order: list[str], rows: list[dict],
                   pair_sides: list[dict], pool_texts: dict, pool_vec: dict) -> dict:
    """Proxy + retrieval blocks exactly as run_phase0.py."""
    n_gate = len(gate_order)
    gv = vec[:n_gate]
    sha_row = {s: i for i, s in enumerate(gate_order)}

    cos = np.array([
        float(np.dot(gv[sha_row[a]], gv[sha_row[b]]))
        for a, b in ((s["src"][0], s["dst"][0]) for s in pair_sides)
    ])
    dir_en2ru = [r["direction"] == "en->ru" for r in rows]
    cls_masks = {c: [r["source_class"] == c for r in rows]
                 for c in ("twin", "llm", "real", "seed-dup")}
    proxy = {
        "overall": proxy_stats(cos),
        "by_direction": {
            "en->ru": subset_stats(cos, dir_en2ru),
            "ru->en": subset_stats(cos, [not d for d in dir_en2ru]),
        },
        "by_class": {c: subset_stats(cos, m) for c, m in cls_masks.items()},
    }

    cross_ranks, mono_rows = [], []
    for i, s in enumerate(pair_sides):
        for qside, cside in (("src", "dst"), ("dst", "src")):
            q_sha = s[qside][0]
            a_sha, a_lang = s[cside]
            qv = gv[sha_row[q_sha]]
            av = gv[sha_row[a_sha]]
            cands = np.vstack([pool_vec[a_lang], av[None, :]])
            sims = cands @ qv
            order = np.argsort(-sims, kind="stable")
            rank = int(np.where(order == len(cands) - 1)[0][0]) + 1
            cross_ranks.append((i, qside, rank))
        for side in ("src", "dst"):
            q_sha, q_lang = s[side]
            qv = gv[sha_row[q_sha]]
            sims = pool_vec[q_lang] @ qv
            mono_rows.append((i, q_lang, float(sims.max())))

    self_rows_count = len(mono_rows)
    margins = np.array([m[2] for m in mono_rows])
    qlangs = np.array([m[1] for m in mono_rows])

    def mono_split(mask: np.ndarray) -> dict:
        return {
            "n": int(mask.sum()),
            "median_best_distractor_cos": round(float(np.median(margins[mask])), 4),
            "P10_best_distractor_cos": round(float(np.percentile(margins[mask], 10)), 4),
            "max_best_distractor_cos": round(float(margins[mask].max()), 4),
            "share_best_ge_085": round(float((margins[mask] >= 0.85).mean()), 4),
        }

    cross_by_class = {c: [] for c in ("twin", "llm", "real", "seed-dup")}
    cross_by_qlang = {"en": [], "ru": []}
    for k, (i, orient, rank) in enumerate(cross_ranks):
        s = pair_sides[i]
        qlang = s["src"][1] if orient == "src" else s["dst"][1]
        cross_by_qlang[qlang].append(rank)
        cross_by_class[rows[i]["source_class"]].append(rank)

    retrieval = {
        "cross": {
            "note": "candidates = 240 same-language distractors + 1 correct; rank of correct",
            "overall": rank_stats(np.array([r[2] for r in cross_ranks])),
            "by_query_lang": {q: rank_stats(np.array(v)) for q, v in cross_by_qlang.items()},
            "by_class": {c: rank_stats(np.array(v)) for c, v in cross_by_class.items()},
        },
        "mono": {
            "note": "margin = best same-language distractor cosine vs the verbatim side text",
            "overall": mono_split(np.ones(self_rows_count, dtype=bool)),
            "en_sides": mono_split(qlangs == "en"),
            "ru_sides": mono_split(qlangs == "ru"),
        },
    }
    return {"proxy": proxy, "retrieval": retrieval, "pair_cosines": cos.tolist()}


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify: dict = {}

    # 0. input verification (hard stop on mismatch)
    gate_sha = sha256_file(GATE_FILE)
    verify["gate_set"] = {"sha256": gate_sha, "expected": EXPECTED_GATE_SHA,
                          "ok": gate_sha == EXPECTED_GATE_SHA}
    assert verify["gate_set"]["ok"], "gate_set sha mismatch - STOP"

    prod_sha = sha256_file(PROD_DIR / "model.onnx")
    verify["model_prod"] = {"model_sha256": prod_sha, "expected": EXPECTED_PROD_SHA,
                            "ok": prod_sha == EXPECTED_PROD_SHA}
    assert verify["model_prod"]["ok"], "prod model sha mismatch - STOP"

    fp_file = (CORPUS_DIR / "fingerprint.txt").read_text().strip()
    verify["corpus"] = {"fingerprint_txt": fp_file, "expected": EXPECTED_CORPUS_FP,
                        "ok": fp_file == EXPECTED_CORPUS_FP}
    assert verify["corpus"]["ok"], "corpus fingerprint mismatch - STOP"

    burned_rows = [json.loads(l) for l in open(BURNED_FILE, encoding="utf-8") if l.strip()]
    burned_sel = [r for r in burned_rows
                  if r.get("stratum") == "translated-dup" and r.get("label") == "duplicate"]
    verify["burned_pairs"] = {"file_rows": len(burned_rows), "selected": len(burned_sel)}
    assert len(burned_sel) == 144, f"expected 144 burned translated-dup pairs, got {len(burned_sel)}"

    # 1. gate set + pair sides
    rows = [json.loads(l) for l in open(GATE_FILE, encoding="utf-8") if l.strip()]
    assert len(rows) == 180
    pair_sides = []
    for r in rows:
        pair_sides.append({
            "src": (sha256_text(r["text_src"]), r["lang_src"]),
            "dst": (sha256_text(r["text_dst"]), r["lang_dst"]),
        })
    uniq_texts: dict[str, str] = {}
    for r in rows:
        for side in ("src", "dst"):
            t = r["text_" + side]
            uniq_texts.setdefault(sha256_text(t), t)
    gate_order = sorted(uniq_texts)
    gate_text_list = [uniq_texts[s] for s in gate_order]

    # burned pair texts (dedup by string)
    burned_texts: list[str] = []
    burned_index: dict[str, int] = {}
    for r in burned_sel:
        for k in ("record", "candidate"):
            body = str(r[k].get("body") or "").strip()
            if body and body not in burned_index:
                burned_index[body] = len(burned_texts)
                burned_texts.append(body)

    # 2. distractor pool — verbatim run_phase0 construction, cross-checked vs npz
    corpus_rows = [json.loads(l) for l in open(CORPUS_TRAIN, encoding="utf-8") if l.strip()]
    pool = build_pool(rows, corpus_rows)
    pool_texts = {lang: [r["text"] for r in pool[lang]] for lang in ("en", "ru")}
    if CHARACT_NPZ.exists():
        npz = np.load(CHARACT_NPZ, allow_pickle=True)
        assert list(npz["pool_en_sha"]) == [r["_sha"] for r in pool["en"]], "en pool mismatch vs npz"
        assert list(npz["pool_ru_sha"]) == [r["_sha"] for r in pool["ru"]], "ru pool mismatch vs npz"
        verify["pool_vs_npz"] = "identical (pool_en/pool_ru sha lists equal)"
    else:
        verify["pool_vs_npz"] = "npz not found - pool reconstructed from verbatim recipe"

    all_texts = gate_text_list + burned_texts + pool_texts["en"] + pool_texts["ru"]
    n_gate, n_burned = len(gate_text_list), len(burned_texts)
    n_en, n_ru = len(pool_texts["en"]), len(pool_texts["ru"])
    print(f"[texts] gate={n_gate} burned={n_burned} pool_en={n_en} pool_ru={n_ru} total={len(all_texts)}")

    # 3. production floor (plumbing validation vs stored characterization)
    prod_res = NanoHarness(PROD_DIR, "production-3b752e06")
    prod_vec = prod_res.embed_all(all_texts)
    prod_out = evaluate_space(prod_vec, gate_order, rows, pair_sides, pool_texts,
                              {"en": prod_vec[n_gate + n_burned:n_gate + n_burned + n_en],
                               "ru": prod_vec[n_gate + n_burned + n_en:]})
    burned_vec = prod_vec[n_gate:n_gate + n_burned]
    prod_burned_median = burned_median(burned_sel, burned_index, burned_vec)
    stored = json.loads(CHARACT_JSON.read_text()) if CHARACT_JSON.exists() else None
    plumbing = {"measured": {
        "proxy_median": prod_out["proxy"]["overall"]["M_cal_median"],
        "proxy_S_cal": prod_out["proxy"]["overall"]["S_cal_share_ge_085"],
        "mono_median": prod_out["retrieval"]["mono"]["overall"]["median_best_distractor_cos"],
        "cross_R@5": prod_out["retrieval"]["cross"]["overall"]["R@5"],
        "burned144_median": prod_burned_median,
    }}
    if stored:
        plumbing["stored_phase0"] = {
            "proxy_median": stored["results"]["proxy"]["production"]["overall"]["M_cal_median"],
            "proxy_S_cal": stored["results"]["proxy"]["production"]["overall"]["S_cal_share_ge_085"],
            "mono_median": stored["results"]["retrieval"]["production"]["mono"]["overall"]["median_best_distractor_cos"],
            "cross_R@5": stored["results"]["retrieval"]["production"]["cross"]["overall"]["R@5"],
        }
    verify["production_plumbing"] = plumbing
    prod_mono_median = prod_out["retrieval"]["mono"]["overall"]["median_best_distractor_cos"]

    results: dict = {"production": {
        "sha256": prod_sha,
        "burned144_median": prod_burned_median,
        "mono_median_best_distractor_cos": prod_mono_median,
        "gate": {k: v for k, v in prod_out.items() if k != "pair_cosines"},
    }}

    # 4. candidates, one at a time (RAM discipline), fp32 torch reference pass
    for cand in CANDIDATES:
        tc = time.time()
        print(f"[cand] {cand['short']} loading...")
        if cand["kind"] == "hf-mean":
            enc = HFMeanEncoder(cand["id"], cand["prefix"])
        else:
            enc = STEncoder(cand["id"])
        vec = enc.embed_all(all_texts)
        assert vec.shape[1] == cand["dim"], f"dim mismatch {vec.shape} vs {cand['dim']}"
        out = evaluate_space(vec, gate_order, rows, pair_sides, pool_texts,
                             {"en": vec[n_gate + n_burned:n_gate + n_burned + n_en],
                              "ru": vec[n_gate + n_burned + n_en:]})
        burned_vec = vec[n_gate:n_gate + n_burned]
        bmed = burned_median(burned_sel, burned_index, burned_vec)
        mono_med = out["retrieval"]["mono"]["overall"]["median_best_distractor_cos"]
        d_mono = round(mono_med - prod_mono_median, 4)
        verdict_a = bmed >= 0.90
        verdict_b = abs(d_mono) <= 0.005
        results[cand["short"]] = {
            "model_id": cand["id"], "params": cand["params"], "license": cand["license"],
            "dim": cand["dim"], "prefix": cand["prefix"] or "(none)",
            "pooling": getattr(enc, "pooling_mode", "mean"),
            "burned144_median": bmed,
            "criterion_a_median_ge_090": {"value": bmed, "pass": bool(verdict_a)},
            "mono_median_best_distractor_cos": mono_med,
            "criterion_b_mono_delta_le_0005": {"delta_vs_prod": d_mono,
                                               "prod": prod_mono_median, "pass": bool(verdict_b)},
            "audit_pass": bool(verdict_a and verdict_b),
            "gate": {k: v for k, v in out.items() if k != "pair_cosines"},
            "wall_seconds": round(time.time() - tc, 1),
        }
        print(f"[cand] {cand['short']}: burned144={bmed:.4f} mono={mono_med:.4f} "
              f"dMono={d_mono:+.4f} S_cal={out['proxy']['overall']['S_cal_share_ge_085']} "
              f"median={out['proxy']['overall']['M_cal_median']} "
              f"R@5={out['retrieval']['cross']['overall']['R@5']} "
              f"PASS={verdict_a and verdict_b} ({time.time()-tc:.0f}s)")
        del enc, vec, out
        gc.collect()

    wall = time.time() - t0
    out_doc = {
        "kind": "embed-r5-phase05-kd-teacher-audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": ["docs/specs/embed-round5-prereg.md §2", "ADR-0005"],
        "scope": "measured audit numbers per prereg'd criteria; NO round-5 gate verdicts",
        "criteria": {
            "a": "median translated-pair cosine >= 0.90 on 144 burned pairs (translated-dup/duplicate)",
            "b": "|delta| gate-set mono control median (median_best_distractor_cos) vs prod 3b752e06 <= 0.005",
        },
        "harness": {
            "recipe": "run_phase0.py verbatim (NanoProvider ORT for prod; HF fp32 torch for "
                      "candidates; max_seq 256; mean+L2; e5 'query: ' prefix per model card)",
            "pool": {"per_language": POOL_PER_LANG, "seed": POOL_SEED, "identity": verify.get("pool_vs_npz")},
            "runtime": {"wall_seconds": round(wall, 1)},
        },
        "inputs_verification": {k: v for k, v in verify.items()},
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out_doc, indent=1, ensure_ascii=False))
    print(f"[done] wall={wall:.0f}s -> {OUT_JSON}")
    return 0


def burned_pairs_ok(sel: list[dict]) -> int:
    return len(sel)


def burned_median(burned_sel: list[dict], burned_index: dict, burned_vec: np.ndarray) -> float:
    cos = []
    for r in burned_sel:
        ia = burned_index[str(r["record"].get("body") or "").strip()]
        ib = burned_index[str(r["candidate"].get("body") or "").strip()]
        cos.append(float(np.dot(burned_vec[ia], burned_vec[ib])))
    return round(float(np.median(np.array(cos))), 4)


if __name__ == "__main__":
    raise SystemExit(main())
