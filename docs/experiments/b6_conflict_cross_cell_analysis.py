#!/usr/bin/env python3
"""B6-conflict wave: cross-language cell trainability on frozen-13 (train-side).

MLC question (sanity probe 7 red on v2): are "translation with a changed fact"
and "true translation" separable on the frozen-13 feature surface in the
fixture cos band, or is the cell one point (P0-E4 analog, design package
2.3)?

Train-side characterization ONLY:
  - reads corpus-v1/train.jsonl and corpus-v2/train.jsonl and nothing else
    from data/conflict-detector/ (sealed holdout v1/v2 is NOT opened);
  - features ONLY via cortex.features.pair (CORE-13), the exact trainer
    recipe (cos_target rides measured, the build-time pinned nano number);
  - embedder: pinned nano 3b752e06 (NanoLeg pin check, loud abort);
  - prints ids/numbers/counts only, never raw record texts.

Selection (per corpus):
  positives = cross CD-1  (stratum CD-1, languages differ; contradicts)
  negatives = v2: carrier == "N3" (true-translation subclass, NC-5)
              v1: no N3 class exists (added in v2, ADDENDUM-3) -> the
              generator-native true-translation carrier NC-4-lang
              (stratum NC-4, languages differ) is used, declared explicitly.
  reference = NC-4-lang cross pairs of BOTH corpora (probe-7 negative-leg
              family, cos ~1.0) kept for fixture NN context.

Env (dispatch convention):
  PYTHONPATH APPEND /var/home/abyss/LABs/Projects/Project-Vesma/vesma/src
  PYTHONDONTWRITEBYTECODE=1, taskset -c 0-12, python = vesma-cortex/.venv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402 (frozen corpus lib, EMBEDDER_PIN)
from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402

CORPUS = {
    "v1": REPO / "data" / "conflict-detector" / "corpus-v1" / "train.jsonl",
    "v2": REPO / "data" / "conflict-detector" / "corpus-v2" / "train.jsonl",
}

#: cos bands (ratified dissection grid, ADDENDUM-1 lineage)
BAND_EDGES = [0.66, 0.70, 0.75, 0.80, 0.85, 0.95, 1.0001]
BAND_LABELS = [
    "[0.66,0.70)", "[0.70,0.75)", "[0.75,0.80)",
    "[0.80,0.85)", "[0.85,0.95)", "[0.95,1.0]",
]


def band_of(cos: float) -> str | None:
    for lab, lo, hi in zip(BAND_LABELS, BAND_EDGES[:-1], BAND_EDGES[1:]):
        if lo <= cos < hi or (lab == BAND_LABELS[-1] and abs(cos - 1.0) < 1e-9):
            return lab
    return None


# --- corpus loading (train files only; orientation dedup) ---------------------


def load_pairs(version: str) -> list[dict]:
    """One row per pair_id; both orientations must share measured.cos_target."""
    rows = [json.loads(l) for l in open(CORPUS[version], encoding="utf-8")]
    assert all(r["split"] == "train" for r in rows), "non-train row leaked"
    by_pair: dict[str, dict] = {}
    for r in rows:
        seen = by_pair.setdefault(r["pair_id"], r)
        if abs(seen["measured"]["cos_target"] - r["measured"]["cos_target"]) > 1e-9:
            raise SystemExit(f"orientation cos mismatch on {r['pair_id']}")
    return list(by_pair.values())


def langs(r: dict) -> tuple[str | None, str | None]:
    return r["envelope_a"].get("language"), r["envelope_b"].get("language")


def is_cross(r: dict) -> bool:
    a, b = langs(r)
    return a is not None and b is not None and a != b


def select(version: str, pairs: list[dict]) -> dict[str, list[dict]]:
    pos = [r for r in pairs if r["stratum"] == "CD-1" and is_cross(r)]
    if version == "v2":
        neg = [r for r in pairs if r.get("carrier") == "N3"]
    else:
        neg = [r for r in pairs if r["stratum"] == "NC-4" and is_cross(r)]
    ref = [r for r in pairs if r["stratum"] == "NC-4" and is_cross(r)]
    for r in pos:
        assert r["label"] == "contradicts", r["pair_id"]
    for r in neg:
        assert r["label"] == "not-contradicts", r["pair_id"]
    return {"pos": pos, "neg": neg, "ref_nc4": ref}


# --- frozen-13 (exact trainer recipe) ------------------------------------------


def rec(side: dict) -> RecordLike:
    return RecordLike(
        title=side["title"],
        body=side["body"],
        tags=tuple(side["tags"]),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def vector13(row: dict) -> np.ndarray:
    vec = features(rec(row["envelope_a"]), rec(row["envelope_b"]),
                   float(row["measured"]["cos_target"]))
    if vec.names != FEATURE_NAMES:
        raise SystemExit("feature contract mismatch")
    return np.asarray(vec.values, dtype=np.float64)


# --- separability primitives ---------------------------------------------------


def feature_gaps(pos_v: np.ndarray, neg_v: np.ndarray) -> list[dict]:
    """Per feature: class intervals and the L-inf style gap/overlap."""
    out = []
    for j, name in enumerate(FEATURE_NAMES):
        p, n = pos_v[:, j], neg_v[:, j]
        lo_p, hi_p, lo_n, hi_n = p.min(), p.max(), n.min(), n.max()
        if hi_n < lo_p:
            gap = lo_p - hi_n
            direction = "neg<pos"
        elif hi_p < lo_n:
            gap = lo_n - hi_p
            direction = "pos<neg"
        else:
            gap = 0.0
            direction = "overlap"
        out.append({"feature": name, "pos": (lo_p, hi_p), "neg": (lo_n, hi_n),
                    "gap": gap, "direction": direction})
    return out


def knn_loo(X: np.ndarray, y: np.ndarray) -> float:
    """Crude 1-NN leave-one-out accuracy, L2 on the raw frozen-13 vector."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    pred = y[d2.argmin(axis=1)]
    return float((pred == y).mean())


def knn_loo_z(X: np.ndarray, y: np.ndarray) -> float:
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return knn_loo((X - mu) / sd, y)


def nn_distances(q: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """(min L2, max feature-cosine) from query to the X set."""
    l2 = np.sqrt(((X - q) ** 2).sum(1))
    qn, Xn = q / np.linalg.norm(q), X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = Xn @ qn
    return float(l2.min()), float(cos.max())


# --- fixtures (probe 7 legs through the frozen generator, seed 0) --------------


def fixture_rows() -> dict[str, dict]:
    import train_conflict_detector as tr

    fx = tr.build_fixtures()
    fx["_nano"] = tr.NanoLeg()
    return {k: tr.sc_row(fx, k) for k in ("CD-1-cross", "CD-1-cross-eq", "NC-4-lang")}


def main() -> int:
    print("== B6-conflict cross-cell frozen-13 train-side analysis ==")
    print("inputs:", {k: str(p.name) for k, p in CORPUS.items()}, "(train files only; holdout NOT opened)")

    data = {}
    for version in ("v1", "v2"):
        pairs = load_pairs(version)
        sel = select(version, pairs)
        print(f"\n-- corpus {version}: {len(pairs)} unique pairs (of {len(pairs) * 2} rows)")
        v2_only_carriers = {c: sum(1 for r in pairs if r.get("carrier") == c)
                            for c in ("N1", "N2", "N3")} if version == "v2" else None
        if v2_only_carriers:
            print("   carrier tags present:", v2_only_carriers)
        else:
            print("   carrier tags present: none (v1 predates the donor carriers)")
        for k in ("pos", "neg", "ref_nc4"):
            rows = sel[k]
            cos = sorted(r["measured"]["cos_target"] for r in rows)
            print(f"   {k}: n={len(rows)} cos=[{cos[0]:.4f}, {cos[-1]:.4f}]" if rows
                  else f"   {k}: n=0")
        data[version] = {
            "sel": sel,
            "pos_v": np.stack([vector13(r) for r in sel["pos"]]) if sel["pos"] else None,
            "neg_v": np.stack([vector13(r) for r in sel["neg"]]) if sel["neg"] else None,
            "ref_v": np.stack([vector13(r) for r in sel["ref_nc4"]]) if sel["ref_nc4"] else None,
        }

    # (3) cos-band table
    print("\n== cos bands: (n_pos, n_neg) unique pairs ==")
    header = "band        | v2 pos | v2 neg | v1 pos | v1 neg"
    print(header + "   (v1 neg = NC-4-lang: no N3 class in v1)")
    for lab in BAND_LABELS:
        row = [lab]
        for version in ("v2", "v1"):
            for k in ("pos", "neg"):
                n = sum(1 for r in data[version]["sel"][k]
                        if band_of(r["measured"]["cos_target"]) == lab)
                row.append(str(n))
        print(" | ".join(x.ljust(11) if i == 0 else x.rjust(6 if x.isdigit() else 7)
                         for i, x in enumerate(row)))
    for version in ("v2", "v1"):
        pos_cos = [r["measured"]["cos_target"] for r in data[version]["sel"]["pos"]]
        above = [c for c in pos_cos if c > 0.75]
        print(f"cross-positives with cos>0.75 ({version}): {len(above)}"
              f" (max {max(pos_cos):.4f})" if pos_cos else f"{version}: no positives")

    # (4) separability
    print("\n== per-feature class intervals (full cross cell) ==")
    for version in ("v2", "v1"):
        pos_v, neg_v = data[version]["pos_v"], data[version]["neg_v"]
        print(f"-- {version}: pos n={len(pos_v)}, neg n={len(neg_v)}"
              f" (neg = {'carrier N3' if version == 'v2' else 'NC-4-lang'})")
        for g in feature_gaps(pos_v, neg_v):
            print(f"   {g['feature']:<18} pos[{g['pos'][0]:8.4f},{g['pos'][1]:8.4f}] "
                  f"neg[{g['neg'][0]:8.4f},{g['neg'][1]:8.4f}] {g['direction']}"
                  + (f" gap={g['gap']:.4f}" if g["direction"] != "overlap" else ""))

    print("\n== bands containing BOTH classes: per-feature gap check ==")
    any_both = False
    for version in ("v2", "v1"):
        for lab in BAND_LABELS:
            p = [vector13(r) for r in data[version]["sel"]["pos"]
                 if band_of(r["measured"]["cos_target"]) == lab]
            n = [vector13(r) for r in data[version]["sel"]["neg"]
                 if band_of(r["measured"]["cos_target"]) == lab]
            if p and n:
                any_both = True
                gaps = feature_gaps(np.stack(p), np.stack(n))
                sep = [g for g in gaps if g["direction"] != "overlap"]
                print(f"   {version} {lab}: pos={len(p)} neg={len(n)} "
                      f"separating features: "
                      + (", ".join(f"{g['feature']}({g['direction']} gap={g['gap']:.4f})"
                                   for g in sep) or "NONE (classes overlap in every feature)"))
    if not any_both:
        print("   NONE: no cos band contains both a cross-positive and a true-translation"
              " negative in either corpus — the classes never share a cos band; per-band"
              " occupancy is read from the table above.")

    print("\n== 1-NN leave-one-out on the cross cell (frozen-13, L2) ==")
    for version in ("v2", "v1"):
        X = np.concatenate([data[version]["pos_v"], data[version]["neg_v"]])
        y = np.concatenate([np.ones(len(data[version]["pos_v"])),
                            np.zeros(len(data[version]["neg_v"]))])
        acc = knn_loo(X, y)
        accz = knn_loo_z(X, y)
        print(f"   {version}: n={len(X)} raw-acc={acc:.4f} z-scored-acc={accz:.4f}")

    # (5) probe-7 fixture position
    print("\n== probe-7 fixture position (frozen-13 NN in train cross cell) ==")
    fx_rows = fixture_rows()
    v2pos, v2neg = data["v2"]["pos_v"], data["v2"]["neg_v"]
    v2ref = data["v2"]["ref_v"]
    for name, row in fx_rows.items():
        cos = float(row["measured"]["cos_target"])
        q = vector13(row)
        l2p, cp = nn_distances(q, v2pos)
        l2n, cn = nn_distances(q, v2neg)
        l2r, cr = nn_distances(q, v2ref)
        nearest = min((l2p, "pos"), (l2n, "neg"), (l2r, "ref(nc4-lang)"))
        print(f"   fixture {name} cos={cos:.4f}: NN_pos L2={l2p:.4f}/cosfeat={cp:.4f} | "
              f"NN_neg(N3) L2={l2n:.4f}/cosfeat={cn:.4f} | NN_ref(nc4) L2={l2r:.4f}/cosfeat={cr:.4f}"
              f" -> nearest family: {nearest[1]}")
    target = abs(float(fx_rows["CD-1-cross"]["measured"]["cos_target"]) - 0.7738)
    print(f"   fixture cos reproduction vs runlog 0.7738: delta={target:.5f} "
          f"({'OK' if target < 5e-4 else 'MISMATCH'})")

    # the two probe-7 legs against EACH OTHER: are they one point on frozen-13?
    q_pos = vector13(fx_rows["CD-1-cross"])
    q_eq = vector13(fx_rows["CD-1-cross-eq"])
    l2_self = float(np.linalg.norm(q_pos - q_eq))
    print(f"   probe-7 legs pairwise: fact-changed vs true-eq L2={l2_self:.4f} "
          f"(vs NN_pos {min(nn_distances(q_pos, v2pos)[0], nn_distances(q_eq, v2pos)[0]):.4f}, "
          f"vs NN_neg {min(nn_distances(q_pos, v2neg)[0], nn_distances(q_eq, v2neg)[0]):.4f}) "
          f"-> legs closer to each other than to ANY train row: {l2_self < 4.0}")
    print("   probe-7 fact-changed vector:",
          {n: round(v, 4) for n, v in zip(FEATURE_NAMES, q_pos) if abs(v) > 1e-9})
    print("   probe-7 true-eq      vector:",
          {n: round(v, 4) for n, v in zip(FEATURE_NAMES, q_eq) if abs(v) > 1e-9})
    print("   differing coordinates (>1e-4):",
          [n for n, a, b in zip(FEATURE_NAMES, q_pos, q_eq) if abs(a - b) > 1e-4])

    # (6) declarative v1 -> v2 cross-landscape check
    print("\n== declarative v1 -> v2 cross-family check ==")
    v1p = {r["pair_id"] for r in data["v1"]["sel"]["pos"]}
    v2p = {r["pair_id"] for r in data["v2"]["sel"]["pos"]}
    print(f"   cross CD-1 pairs: v1={len(v1p)} -> v2={len(v2p)} (shared ids: {len(v1p & v2p)}; "
          f"fresh seed families expected)")
    for version in ("v1", "v2"):
        cos = [r["measured"]["cos_target"] for r in data[version]["sel"]["pos"]]
        print(f"   {version} cross CD-1 cos range: [{min(cos):.4f}, {max(cos):.4f}]")
    print(f"   N3 true-translation class: v1 absent -> v2 {len(data['v2']['sel']['neg'])} pairs "
          f"(ADDENDUM-3 donor, declarative: not a cut of the cross-positive family)")
    extra = {}
    for version in ("v1", "v2"):
        pairs = load_pairs(version)
        extra[version] = {
            "CD-2": sum(1 for r in pairs if r["stratum"] == "CD-2" and is_cross(r)),
            "CD-6": sum(1 for r in pairs if r["stratum"] == "CD-6" and is_cross(r)),
            "NC-4": sum(1 for r in pairs if r["stratum"] == "NC-4" and is_cross(r)),
        }
    print(f"   other cross rows (pairs) v1 vs v2: {extra['v1']} vs {extra['v2']} "
          f"(out of scope for this cell, census only)")

    print("\nverdict inputs collected; interpretation per the MLC taxonomy is in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
