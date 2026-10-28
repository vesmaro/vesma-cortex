#!/usr/bin/env python3
"""A-retrial B-leg circularity QA gate — cosine-as-score AUC vs edge label on frozen corpus v5-fr.

Question (MLC member proposal 28.10, threshold 0.98 NOT ratified — measurement only):
does the embedding cosine predict WHICH pairs the auto-dedupe min-stream minted as
edges? If the label is circular w.r.t. the score (mint = cosine threshold), AUC -> 1.

Data (frozen D0 = 2026-10-28T20:39:47Z, gitignored data/a-retrial/):
  positives: classes/edges_post_hygiene.jsonl  (post-hygiene edges, 1648)
  negatives: classes/zone_pairs.npz            (census zone: ALL no-edge pairs cos>=0.92, 989191)
  vectors:   snapshot/vectors.db               (embeddings, 4942 x 384, prod pin nano:sha256:3b752e06…)

Method (honest, representative):
  1. ALL cosines (positives AND negatives) recomputed in float64 from the frozen
     snapshot vectors — one computation path for both classes; stored cos fields
     (edges jsonl / npz) used only as agreement checks.
  2. Negatives: the COMPLETE census zone >=0.92 (full enumeration, no subsampling
   noise); per-band breakdown [0.92,0.94)/[0.94,0.96)/[0.96,0.97)/[0.97,1.0]
   (positive side binned by its own recomputed cosine). No <0.92 material mixed in;
   sub-0.92 positives counted and excluded from the zone AUC (trivial stratum).
  3. AUC = Mann-Whitney P(cos_pos > cos_neg), exact average-rank tie handling via
   searchsorted left/right. Bootstrap 95% CI over positives (B=1000, seed 20261028),
   negative side kept as the full population.
  4. Extra: share of edges where the edge is the top-1 cosine neighbor of at least
   one endpoint (top-1 over all 4942 published vectors = what auto-dedupe saw;
   secondary variant: top-1 within the 3442 hygiene-passing population).

Interpreter: /usr/bin/python3 (numpy only). Scoring venv (.venv, ort 1.30) NOT used —
no re-embedding, vectors are frozen. Read-only over data/a-retrial/. Prints numbers only;
no raw texts touched. Budget: light lane, ~1 min CPU, < 1 GB RAM.
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
DATA = WT / "data" / "a-retrial"
SEED = 20261028
BOOTSTRAP_B = 1000
PIN_PREFIX = "nano:sha256:3b752e06"
BANDS = [("[0.92,0.94)", 0.92, 0.94), ("[0.94,0.96)", 0.94, 0.96),
         ("[0.96,0.97)", 0.96, 0.97), ("[0.97,1.0]", 0.97, 2.0)]  # float32 artifacts >=1.0 live here

MANIFEST_FP = {  # from docs/experiments/a-retrial-phase1-corpus-manifest.json
    "vectors.db": "0e97cc096f198caaa5ed19c9266c7e8b3e40b060dd5b18caca36e47df3d18278",
    "edges_post_hygiene.jsonl": "2ddcebd94e6933ccdb3d86c19ad588dd029941378b864cd9529419eb6d4254a5",
    "zone_pairs.npz": "4c7acf2f4872ab87869eccbc2e3ba6b0833adb35189e470f18c3c87aa9804461",
    "sidecar_G.jsonl": "d36f3f53d63dab603b6ec69281ea15859a66779f3ce643c6b258466d11f94636",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc_mw(pos: np.ndarray, neg: np.ndarray) -> float:
    """Exact Mann-Whitney AUC P(pos > neg) + 0.5 P(pos == neg), tie-safe."""
    neg = np.sort(neg)
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(np.mean((left + 0.5 * (right - left)) / neg.size))


def bootstrap_ci(pos: np.ndarray, neg_sorted: np.ndarray, rng: np.random.Generator,
                 b: int = BOOTSTRAP_B) -> tuple[float, float]:
    n = pos.size
    idx = rng.integers(0, n, size=(b, n))
    left = np.searchsorted(neg_sorted, pos[idx], side="left")
    right = np.searchsorted(neg_sorted, pos[idx], side="right")
    aucs = np.mean((left + 0.5 * (right - left)) / neg_sorted.size, axis=1)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    report: dict = {}

    # --- 0. frozen-artifact binding: sha256 must match the phase-1 manifest ---
    targets = {
        "snapshot/vectors.db": MANIFEST_FP["vectors.db"],
        "classes/edges_post_hygiene.jsonl": MANIFEST_FP["edges_post_hygiene.jsonl"],
        "classes/zone_pairs.npz": MANIFEST_FP["zone_pairs.npz"],
        "mask/sidecar_G.jsonl": MANIFEST_FP["sidecar_G.jsonl"],
    }
    fps_ok = True
    for rel, want in targets.items():
        got = sha256_file(DATA / rel)
        ok = got == want
        fps_ok &= ok
        print(f"[fp] {rel}: {'PASS' if ok else 'MISMATCH got ' + got}")
    assert fps_ok, "frozen artifact fingerprint mismatch — refusing to measure"

    # --- 1. snapshot vectors + prod-pin assert ---
    con = sqlite3.connect(DATA / "snapshot" / "vectors.db")
    ids, vecs, pin_ok = [], [], 0
    for id_, blob, mdj in con.execute("SELECT id, vector, metadata FROM embeddings ORDER BY id"):
        md = json.loads(mdj) if mdj else {}
        pin_ok += int(str(md.get("model_fingerprint", "")).startswith(PIN_PREFIX))
        ids.append(id_)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
    con.close()
    V32 = np.stack(vecs)
    ids_arr = np.array(ids)
    n_vec = len(ids)
    print(f"[vectors] rows={n_vec} dims={V32.shape[1]} pin_assert={pin_ok}/{n_vec}")
    assert pin_ok == n_vec, "prod-pin assert failed"
    norms = np.linalg.norm(V32.astype(np.float64), axis=1)
    print(f"[vectors] unit-norm: min={norms.min():.8f} max={norms.max():.8f}")
    V = V32.astype(np.float64) / norms[:, None]
    id2row = {id_: r for r, id_ in enumerate(ids)}

    # --- 2. positives: post-hygiene edges, cosine recomputed ---
    edges = [json.loads(l) for l in open(DATA / "classes" / "edges_post_hygiene.jsonl")]
    pa = np.array([id2row[e["a"]] for e in edges])
    pb = np.array([id2row[e["b"]] for e in edges])
    cos_pos = np.sum(V[pa] * V[pb], axis=1)
    stored_pos = np.array([e["cos"] for e in edges], dtype=np.float64)
    d_pos = np.abs(cos_pos - stored_pos)
    print(f"[positives] n={len(edges)} | recomputed-vs-stored(cos) max|d|={d_pos.max():.3e} mean={d_pos.mean():.3e}")
    print(f"[positives] provenance histogram: " +
          ", ".join(f"{k}={v}" for k, v in sorted(
              {p: sum(1 for e in edges if e["provenance"] == p) for p in set(e["provenance"] for e in edges)}.items())))
    sub092 = int((cos_pos < 0.92).sum())
    pos_band_hist = {}
    for name, lo, hi in BANDS:
        pos_band_hist[name] = int(((cos_pos >= lo) & (cos_pos < hi)).sum())
    print(f"[positives] recomputed band hist: {pos_band_hist} + <0.92: {sub092}")
    report["positives"] = {"n": len(edges), "band_hist": pos_band_hist, "sub_092": sub092,
                           "recomp_vs_stored_max_absdiff": float(d_pos.max())}

    # --- 3. negatives: full census zone >=0.92, cosine recomputed in chunks ---
    z = np.load(DATA / "classes" / "zone_pairs.npz")
    zi, zj, zc = z["i"], z["j"], z["cos"].astype(np.float64)
    meta = json.load(open(DATA / "classes" / "zone_pairs_meta.json"))
    zids = meta["ids"]
    zrow = np.array([id2row[t] for t in zids])
    assert zi.min() >= 0 and zj.min() >= 0 and zi.max() < len(zids) and zj.max() < len(zids)
    assert (zi != zj).all()
    cos_neg = np.empty(zi.size, dtype=np.float64)
    CH = 100_000
    for s in range(0, zi.size, CH):
        e = min(s + CH, zi.size)
        cos_neg[s:e] = np.sum(V[zrow[zi[s:e]]] * V[zrow[zj[s:e]]], axis=1)
    d_neg = np.abs(cos_neg - zc)
    print(f"[negatives] census rows={zi.size} | recomputed-vs-npz(cos) max|d|={d_neg.max():.3e} mean={d_neg.mean():.3e}")
    band_neg, neg_total = {}, 0
    for name, lo, hi in BANDS:
        m = (cos_neg >= lo) & (cos_neg < hi)
        band_neg[name] = m
        neg_total += int(m.sum())
    band_counts = {k: int(v.sum()) for k, v in band_neg.items()}
    stragglers = zi.size - sum(band_counts.values())  # float64 score < 0.92 while frozen npz float32 said >=0.92
    print(f"[negatives] band census (float64 score bins): {band_counts} boundary-straddlers={stragglers} total={zi.size}")
    # Census binding is by the FROZEN file membership (float32 arithmetic, manifest numbers):
    f32_counts = {name: int(((zc >= lo) & (zc < hi)).sum()) for name, lo, hi in BANDS}
    print(f"[negatives] band census (frozen npz float32, manifest arithmetic): {f32_counts} total={zi.size}")
    assert f32_counts["[0.97,1.0]"] == 44100 and sum(f32_counts.values()) == 989191, \
        "frozen float32 census != manifest (989191 / 44100)"
    # float64 scoring bins may shift 0-2 boundary pairs; report, never silently drop.
    # Stragglers stay IN the overall zone AUC (membership is frozen-file-defined).

    # --- 4. AUC overall + per band, bootstrap CI (seed fixed) ---
    rng = np.random.default_rng(SEED)
    pos_in_zone = cos_pos >= 0.92
    pz = cos_pos[pos_in_zone]
    print(f"[auc] positives in zone: {pz.size} (sub-0.92 excluded: {sub092})")
    auc_all = auc_mw(pz, cos_neg)
    lo, hi = bootstrap_ci(pz, np.sort(cos_neg), rng)
    print(f"[auc] OVERALL zone>=0.92: AUC={auc_all:.6f}  CI95=[{lo:.6f},{hi:.6f}]  (pos n={pz.size}, neg n={cos_neg.size})")
    report["auc_overall"] = {"auc": auc_all, "ci95": [lo, hi], "pos_n": int(pz.size), "neg_n": int(cos_neg.size)}

    report["auc_per_band"] = {}
    print("[auc] per band (positives binned by recomputed cos):")
    for name, blo, bhi in BANDS:
        pb_pos = pz[(pz >= blo) & (pz < bhi)]
        nb = cos_neg[band_neg[name]]
        a = auc_mw(pb_pos, nb)
        clo, chi = bootstrap_ci(pb_pos, np.sort(nb), rng)
        print(f"  {name}: pos n={pb_pos.size:4d} neg n={nb.size:6d}  AUC={a:.6f}  CI95=[{clo:.6f},{chi:.6f}]")
        report["auc_per_band"][name] = {"auc": a, "ci95": [clo, chi], "pos_n": int(pb_pos.size), "neg_n": int(nb.size)}

    # --- 4b. composition-only AUC (band ordinal as score) + mint rate per band ---
    band_ord = np.zeros(cos_neg.size, dtype=np.float64)
    for k, (name, _lo, _hi) in enumerate(BANDS):
        band_ord[band_neg[name]] = k
    pos_ord = np.full(pz.size, -1.0)
    for k, (name, lo, hi) in enumerate(BANDS):
        pos_ord[(pz >= lo) & (pz < hi)] = k
    auc_comp = auc_mw(pos_ord, band_ord)
    print(f"[auc] composition-only (band ordinal as score, within-band ties kept): AUC={auc_comp:.6f}"
          f" — the gap to the overall AUC is the within-band cosine contribution")
    report["auc_composition_only"] = auc_comp
    print("[mint_rate] positives per 1000 census pairs of the band:")
    mint_rate = {}
    for name, _lo, _hi in BANDS:
        r = pos_band_hist[name] / band_counts[name] * 1000
        mint_rate[name] = r
        print(f"  {name}: {pos_band_hist[name]}/{band_counts[name]} = {r:.3f}")
    report["mint_rate_per_1000"] = mint_rate

    # --- 5. top-1 neighbor share (edge == top-1 of at least one endpoint) ---
    def top1_matrix(rows: np.ndarray) -> np.ndarray:
        base = rows
        nn = np.empty(base.size, dtype=np.int64)
        for s in range(0, base.size, 512):
            e = min(s + 512, base.size)
            sim = V[base[s:e]] @ V[base].T
            for r in range(e - s):
                sim[r, s + r] = -2.0  # exclude self
            nn[s:e] = np.argmax(sim, axis=1)
        return nn

    all_rows = np.arange(n_vec)
    nn_all = top1_matrix(all_rows)
    hyg_mask = np.zeros(n_vec, dtype=bool)
    for t in zids:
        hyg_mask[id2row[t]] = True
    hyg_rows = np.where(hyg_mask)[0]
    nn_hyg_local = top1_matrix(hyg_rows)
    nn_hyg = np.empty(n_vec, dtype=np.int64)      # global row -> top-1 within hygiene population (global row)
    nn_hyg[hyg_rows] = hyg_rows[nn_hyg_local]
    hit_all = hit_hyg = 0
    for r in range(len(edges)):
        a, b = pa[r], pb[r]
        if nn_all[a] == b or nn_all[b] == a:
            hit_all += 1
        if nn_hyg[a] == b or nn_hyg[b] == a:
            hit_hyg += 1
    print(f"[top1] edge is top-1 neighbor of >=1 endpoint: store-wide(4942) {hit_all}/{len(edges)} = {hit_all/len(edges):.4f}"
          f" | hygiene-population(3442) {hit_hyg}/{len(edges)} = {hit_hyg/len(edges):.4f}")
    report["top1"] = {"store_wide": {"hits": hit_all, "share": hit_all / len(edges)},
                      "hygiene_population": {"hits": hit_hyg, "share": hit_hyg / len(edges)}}

    # --- 6. verdict ---
    verdict = "YES" if auc_all >= 0.98 else "NO"
    print(f"[verdict] labels circular w.r.t. cosine at threshold 0.98 (proposal, NOT ratified): {verdict} "
          f"(overall zone AUC {auc_all:.6f})")
    report["verdict"] = {"threshold": 0.98, "ratified": False, "circular": verdict, "auc_overall": auc_all}
    print("[meta] seed={} bootstrap_B={} interpreter=python3+numpy(float64) scoring_venv=not-used(no re-embedding) "
          "negatives=full-enumeration".format(SEED, BOOTSTRAP_B))
    print("[meta] inputs bound to manifest fingerprints: PASS x4")
    print("[report_json] " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
