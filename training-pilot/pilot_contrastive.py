#!/usr/bin/env python3
"""Phase-0.5 Job 2 — translation-contrastive pilot grid (λ × temperature).

Prereg embed-round5-prereg.md §1.1-1.2: pilot the translation-contrastive
term on ~25% of the round-4 corpus pairs BEFORE the decisive round-5 run;
mechanics (in-batch InfoNCE vs paired-cos margin) decided here. Batch 32
(16 pairs) is available on CPU, so the primary mechanics is in-batch
InfoNCE (positive = the translation pair, negatives = the other 30 texts
in batch) — the richer form; the paired-cos margin form stays un-piloted
(only needed at batch < 32, which does not occur).

Grid: lambda in {0.2, 0.5, 1.0} x temperature in {0.05, 0.1, 0.2}, plus a
lambda=0 CONTROL arm (KD only) for loss attribution. Total loss =
mrl_kd_loss(4,2,1,1 weights, round-5 base config) + lambda * InfoNCE(tau)
on the full 384-d runtime embedding.

Pilot design (logged):
  * warm-start from the round-4 selected checkpoint
    (runs/embed-r4/a-30m/epoch10, source of the exported REJECT candidate
    f70672aa) — the pilot measures the MARGINAL effect of the contrastive
    term on top of the round-4 KD state; from-scratch 1-2 epochs on a 25%
    subset leaves the proxies in noise (round-4 evidence: 1 full epoch
    from scratch -> val R@5 0.0786). The decisive round-5 training runs
    fresh on the round-5 corpus; the pilot informs ONLY the loss term.
  * teacher vectors from the frozen round-4 lockstep npz (no teacher
    forward; texts assert gives the lockstep).
  * selection metric (prereg §1.2 wording): seed-dup + llm proxy cosine
    improvement WITHOUT monolingual control drift (gX0 spirit, |Δ| ≤
    0.005) — drift measured BOTH vs the pilot baseline (attribution) and
    vs the production floor 3b752e06 (gX0 spirit).

Measured numbers only — no round-5 gate verdicts (chair computes those).
CPU-only; deterministic: seed 42, identical batch ORDER across configs
(only lambda/tau differ); fp32 torch reference pass.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PILOT_DIR = Path(__file__).resolve().parent
ENGINE_TRAIN = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/wt/embed-r4-train/training")
R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
REDO_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/corpus-v43-redo")
CORPUS_DIR = R4_WT / "data" / "embed-r4" / "corpus"
GATE_DIR = R4_WT / "data" / "embed-r4" / "r5_gate"
NPZ = ENGINE_TRAIN / "runs" / "embed-r4" / "teacher_vectors.npz"
CKPT = ENGINE_TRAIN / "runs" / "embed-r4" / "a-30m" / "epoch10"
SUBSET_JSON = PILOT_DIR / "results" / "pilot_subset.json"
OUT_JSON = PILOT_DIR / "results" / "pilot_grid.json"
CHARACT_JSON = Path("/tmp/r5-charact/r5_phase0_characterization.json")

EXPECTED_FP = "7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645"
EXPECTED_GATE_SHA = "cb36757769fb88f2806065f50348f5fd8383cfdb0c19c66e6335eac8ace01123"

SEED = 42
EPOCHS = 2
PAIRS_PER_BATCH = 16            # 32 texts per batch -> in-batch negatives = 30
LR = 2e-5                       # round-4 config
KD_TEMPERATURE = 0.05           # round-4 config
MRL_WEIGHTS_SPEC = "4,2,1,1"    # round-5 base config (prereg §1.3)
GRID_LAMBDA = [0.2, 0.5, 1.0]
GRID_TAU = [0.05, 0.1, 0.2]
MAX_SEQ = 256
MONO_DRIFT_LIMIT = 0.005        # gX0 spirit (selection-side, not a gate verdict)

sys.path.insert(0, str(ENGINE_TRAIN))
sys.path.insert(0, str(PILOT_DIR))

from distill import (  # noqa: E402  (engine trainer, reused read-only)
    mean_pool,
    mrl_kd_loss,
    parse_mrl_weights,
    set_determinism,
)
from audit_kd_teachers import (  # noqa: E402  (protocol identity with Job 1)
    build_pool,
    burned_median,
    evaluate_space,
    sha256_file,
    sha256_text,
)


class StudentEncoder:
    """Pilot student embedding: HF encoder + mean pool + projector + L2 (384d)."""

    def __init__(self, model, projector, tokenizer, device, batch_size=32):
        import torch

        self.torch = torch
        self.model, self.projector, self.tok = model, projector, tokenizer
        self.device, self.batch_size = device, batch_size
        self.pooling_mode = "mean"

    def embed_all(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        was_training = self.model.training
        self.model.eval()
        out = []
        for start in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[start : start + self.batch_size], padding=True,
                           truncation=True, max_length=MAX_SEQ, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hs = self.model(**enc).last_hidden_state
            emb = self.projector(mean_pool(hs, enc["attention_mask"]))
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.detach().cpu().numpy().astype(np.float64))
        if was_training:
            self.model.train()
        return np.vstack(out)


def load_student(device: str):
    """Fresh warm-start state: round-4 epoch10 checkpoint + its projector."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(str(CKPT)).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    proj_sd = torch.load(CKPT / "projector.pt", map_location="cpu")
    out_dim, in_dim = proj_sd["weight"].shape
    proj = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    proj.load_state_dict(proj_sd)
    proj.eval()
    return model, tok, proj


def contrastive_infonce(emb: "object", tau: float) -> "object":
    """Symmetric in-batch InfoNCE over a pair-adjacent batch.

    Batch layout [s1, d1, s2, d2, ...]: partner of i is i^1. Cosine
    logits scaled by 1/tau; cross-entropy toward the partner.
    """
    import torch

    logits = (emb @ emb.T) / tau
    labels = torch.arange(logits.shape[0], device=logits.device) ^ 1
    return torch.nn.functional.cross_entropy(logits, labels)


def main() -> int:
    t0 = time.time()
    results: dict = {}

    # ---- verification ----
    assert sha256_file(GATE_DIR / "gate_set.jsonl") == EXPECTED_GATE_SHA, "gate sha mismatch"
    digest = hashlib.sha256()
    digest.update((CORPUS_DIR / "train.jsonl").read_bytes())
    digest.update((CORPUS_DIR / "val.jsonl").read_bytes())
    assert digest.hexdigest() == EXPECTED_FP, "corpus fingerprint mismatch"
    assert (CKPT / "model.safetensors").exists() or (CKPT / "pytorch_model.bin").exists(), \
        "warm-start checkpoint missing"
    # round-4 a-30m checkpoints carry NO mrl_dims.json (probe-only artifact;
    # runlog 2026-10-08) — take the MRL spec from the COMMITTED run config.
    r4_cfg = json.loads((ENGINE_TRAIN / "embed_r4" / "run_config.json").read_text())
    mrl_dims: list[int] = r4_cfg["student"]["mrl_dims"]
    embed_dim: int = int(mrl_dims[-1])
    assert mrl_dims == [64, 128, 256, 384] and embed_dim == 384, "unexpected r4 MRL spec"
    mrl_weights = parse_mrl_weights(MRL_WEIGHTS_SPEC, len(mrl_dims))
    print(f"[verify] corpus+gate OK; ckpt={CKPT.name} embed_dim={embed_dim} mrl={mrl_dims}")

    subset = json.loads(SUBSET_JSON.read_text())
    pairs = subset["pairs"]
    print(f"[subset] {len(pairs)} pairs ({subset['counts']['subset_share']:.0%} of pool "
          f"{subset['counts']['pool']}) by_class={subset['counts']['by_class_subset']}")

    # ---- teacher vectors (lockstep npz: train rows then val rows) ----
    train_rows = [json.loads(l) for l in open(CORPUS_DIR / "train.jsonl", encoding="utf-8") if l.strip()]
    val_rows = [json.loads(l) for l in open(CORPUS_DIR / "val.jsonl", encoding="utf-8") if l.strip()]
    npz = np.load(NPZ, allow_pickle=True)
    npz_texts = [str(x) for x in npz["texts"]]
    corpus_texts = [r["text"] for r in train_rows] + [r["text"] for r in val_rows]
    assert npz_texts == corpus_texts, "npz lockstep violated (texts differ)"
    vec_full = npz["vectors"]
    print(f"[npz] lockstep OK {vec_full.shape}")

    train_texts_index = {}
    for i, r in enumerate(train_rows):
        train_texts_index[r["text"]] = i
    train_items = []  # (src_text, dst_text, i_src, i_dst)
    for p in pairs:
        ts, td = p["text_src"], p["text_dst"]
        assert ts in train_texts_index and td in train_texts_index, "subset text not in train slice"
        train_items.append((ts, td, train_texts_index[ts], train_texts_index[td]))
    # assert subset.json texts are byte-exact corpus rows (provenance intact;
    # subset shas are gen_embed_r4_texts.text_hash values, NOT byte sha256)
    sys.path.insert(0, str(R4_WT / "scripts"))
    from gen_embed_r4_texts import text_hash as provenance_hash

    for p, (ts, td, _, _) in zip(pairs, train_items):
        assert provenance_hash(ts) == p["sha_src"] and provenance_hash(td) == p["sha_dst"]

    # ---- eval surfaces (Job-1 protocol identity) ----
    gate_rows = [json.loads(l) for l in open(GATE_DIR / "gate_set.jsonl", encoding="utf-8") if l.strip()]
    pair_sides = [{"src": (sha256_text(r["text_src"]), r["lang_src"]),
                   "dst": (sha256_text(r["text_dst"]), r["lang_dst"])} for r in gate_rows]
    uniq_texts: dict[str, str] = {}
    for r in gate_rows:
        for side in ("src", "dst"):
            uniq_texts.setdefault(sha256_text(r["text_" + side]), r["text_" + side])
    gate_order = sorted(uniq_texts)
    gate_text_list = [uniq_texts[s] for s in gate_order]

    burned_rows = [json.loads(l) for l in open(REDO_WT / "data" / "stage2" / "v43" / "train.jsonl",
                                               encoding="utf-8") if l.strip()]
    burned_sel = [r for r in burned_rows
                  if r.get("stratum") == "translated-dup" and r.get("label") == "duplicate"]
    assert len(burned_sel) == 144
    burned_texts, burned_index = [], {}
    for r in burned_sel:
        for k in ("record", "candidate"):
            body = str(r[k].get("body") or "").strip()
            if body and body not in burned_index:
                burned_index[body] = len(burned_texts)
                burned_texts.append(body)

    corpus_rows_all = train_rows  # pool source (run_phase0 used train.jsonl)
    pool = build_pool(gate_rows, [dict(r) for r in corpus_rows_all])
    pool_texts = {"en": [r["text"] for r in pool["en"]], "ru": [r["text"] for r in pool["ru"]]}
    npz_pool_check = None  # pool identity vs npz cross-checked in Job 1 run

    subset_src_list = [it[0] for it in train_items]
    subset_dst_list = [it[1] for it in train_items]

    eval_texts = gate_text_list + burned_texts + pool_texts["en"] + pool_texts["ru"]
    n_gate, n_burned = len(gate_text_list), len(burned_texts)
    n_en, n_ru = len(pool_texts["en"]), len(pool_texts["ru"])

    def evaluate(model, tok, proj, device) -> dict:
        enc = StudentEncoder(model, proj, tok, device)
        vec = enc.embed_all(eval_texts)
        out = evaluate_space(vec, gate_order, gate_rows, pair_sides, pool_texts,
                             {"en": vec[n_gate + n_burned:n_gate + n_burned + n_en],
                              "ru": vec[n_gate + n_burned + n_en:]})
        bmed = burned_median(burned_sel, burned_index, vec[n_gate:n_gate + n_burned])
        sv = enc.embed_all(subset_src_list + subset_dst_list)
        s, d = sv[: len(subset_src_list)], sv[len(subset_src_list):]
        pair_cos = np.array([float(np.dot(s[i], d[i])) for i in range(len(subset_src_list))])
        out["burned144_median"] = bmed
        out["subset_pair_cos_median"] = round(float(np.median(pair_cos)), 4)
        out["subset_pair_cos_share_ge_085"] = round(float((pair_cos >= 0.85).mean()), 4)
        return out

    # ---- production floor for the gX0-spirit reference ----
    prod_mono = None
    if CHARACT_JSON.exists():
        stored = json.loads(CHARACT_JSON.read_text())
        prod_mono = stored["results"]["retrieval"]["production"]["mono"]["overall"]["median_best_distractor_cos"]

    import torch

    device = "cpu"
    threads = 8
    torch.set_num_threads(threads)
    set_determinism(SEED)

    # ---- baseline: warm-start checkpoint evaluated as-is ----
    model, tok, proj = load_student(device)
    baseline = evaluate(model, tok, proj, device)
    base_sel = {
        "seed_dup_median": baseline["proxy"]["by_class"]["seed-dup"]["M_cal_median"],
        "llm_median": baseline["proxy"]["by_class"]["llm"]["M_cal_median"],
        "mono_median": baseline["retrieval"]["mono"]["overall"]["median_best_distractor_cos"],
        "burned144_median": baseline["burned144_median"],
        "subset_pair_cos_median": baseline["subset_pair_cos_median"],
    }
    results["baseline_epoch10_fp32"] = {"selection_metrics": base_sel,
                                        "full": {k: v for k, v in baseline.items()}}
    print(f"[baseline] {json.dumps(base_sel)}")
    del model, proj
    gc_collect = __import__("gc").collect

    # ---- training texts (pair-adjacent layout) ----
    flat_texts: list[str] = []
    flat_vec_idx: list[int] = []
    for ts, td, i_s, i_d in train_items:
        flat_texts.extend([ts, td])
        flat_vec_idx.extend([i_s, i_d])
    flat_targets = torch.tensor(
        np.stack([vec_full[i][..., :embed_dim] for i in flat_vec_idx]), dtype=torch.float32
    )
    print(f"[train] {len(flat_texts)} texts ({len(train_items)} pairs) -> "
          f"{len(flat_texts) // (2 * PAIRS_PER_BATCH)} batches/epoch")

    def run_config(lmbda: float, tau: float, tag: str) -> dict:
        set_determinism(SEED)
        model, tok, proj = load_student(device)
        model.train()
        proj.train()
        params = list(model.parameters()) + list(proj.parameters())
        opt = torch.optim.AdamW(params, lr=LR)
        cfg_out = {"lambda": lmbda, "tau": tau, "epochs": {}}
        rng = random.Random(SEED)
        order = list(range(len(train_items)))
        for epoch in range(1, EPOCHS + 1):
            rng.shuffle(order)
            ep_kd, ep_ct, nb = 0.0, 0.0, 0
            for start in range(0, len(order), PAIRS_PER_BATCH):
                idx = order[start : start + PAIRS_PER_BATCH]
                text_chunk = [t for i in idx for t in (train_items[i][0], train_items[i][1])]
                vec_chunk = flat_targets[[j for i in idx for j in (2 * i, 2 * i + 1)]]
                enc = tok(text_chunk, padding=True, truncation=True,
                          max_length=MAX_SEQ, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                hs = model(**enc).last_hidden_state
                pooled = mean_pool(hs, enc["attention_mask"])
                kd = mrl_kd_loss(pooled, vec_chunk, mrl_dims, KD_TEMPERATURE,
                                 projector=proj, weights=mrl_weights)
                if lmbda > 0.0:
                    emb = torch.nn.functional.normalize(proj(pooled), p=2, dim=-1)
                    ct = contrastive_infonce(emb, tau)
                    loss = kd + lmbda * ct
                else:
                    ct = torch.zeros((), device=device)
                    loss = kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_kd += float(kd.detach())
                ep_ct += float(ct.detach())
                nb += 1
            model.eval()
            proj.eval()
            ev = evaluate(model, tok, proj, device)
            sel = {
                "seed_dup_median": ev["proxy"]["by_class"]["seed-dup"]["M_cal_median"],
                "llm_median": ev["proxy"]["by_class"]["llm"]["M_cal_median"],
                "mono_median": ev["retrieval"]["mono"]["overall"]["median_best_distractor_cos"],
                "burned144_median": ev["burned144_median"],
                "subset_pair_cos_median": ev["subset_pair_cos_median"],
                "subset_pair_share_ge_085": ev["subset_pair_cos_share_ge_085"],
            }
            cfg_out["epochs"][str(epoch)] = {
                "avg_kd_loss": round(ep_kd / max(1, nb), 6),
                "avg_contrastive_loss": round(ep_ct / max(1, nb), 6),
                "selection_metrics": sel,
            }
            print(f"[{tag}] ep{epoch}: kd={ep_kd/max(1,nb):.4f} ct={ep_ct/max(1,nb):.4f} "
                  f"seed_dup={sel['seed_dup_median']} llm={sel['llm_median']} "
                  f"mono={sel['mono_median']} burned={sel['burned144_median']} "
                  f"subset_cos={sel['subset_pair_cos_median']}")
        del model, proj
        gc_collect()
        return cfg_out

    # ---- grid (lambda=0 control first, then lambda x tau) ----
    configs = [(0.0, None, "control-lambda0")] + [
        (l, t, f"l{l}_t{t}") for l in GRID_LAMBDA for t in GRID_TAU
    ]
    for lmbda, tau, tag in configs:
        tc = time.time()
        print(f"[grid] === {tag} (lambda={lmbda}, tau={tau}) ===")
        results[tag] = run_config(lmbda, tau, tag)
        results[tag]["wall_seconds"] = round(time.time() - tc, 1)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:  # incremental persist
            json.dump({"kind": "embed-r5-phase05-contrastive-pilot-grid",
                       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "config": {"seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
                                  "lr": LR, "kd_temperature": KD_TEMPERATURE,
                                  "mrl_dims": mrl_dims, "mrl_weights_spec": MRL_WEIGHTS_SPEC,
                                  "warm_start": str(CKPT), "subset": str(SUBSET_JSON.name),
                                  "subset_pairs": len(train_items),
                                  "mechanics": "in-batch InfoNCE on full 384-d embedding, "
                                               "pair-adjacent batch layout, partner = i^1",
                                  "mono_drift_limit": MONO_DRIFT_LIMIT,
                                  "prod_mono_floor_reference": prod_mono,
                                  "pool_identity_note": npz_pool_check},
                       "results": results}, fh, indent=1, ensure_ascii=False)

    print(f"[done] wall={time.time()-t0:.0f}s -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
