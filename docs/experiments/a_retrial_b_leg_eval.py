"""B-leg eval context: frozen funnel features + gate metrics + bootstrap.

Shared by the characterize (halves) and the single-shot gate runner so both
stages consume byte-identical feature surfaces.

Interpreter: vesma-cortex/.venv; PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a_retrial_b_leg_common import (  # noqa: E402
    ABLATED_FEATURES,
    BLEG,
    FUSION_FEATURES,
    build_gprime,
    cos64,
    feature_row,
    load_edges_direction,
    load_manifest,
    assert_fingerprints,
    load_published_meta,
    load_sidecar,
    load_snapshot_vectors,
    pair_graph_features,
    pair_key,
)
from cortex.features.graph import GraphEvidence  # noqa: E402


class EvalContext:
    """Frozen funnel + gspec surfaces, features computed once."""

    def __init__(self) -> None:
        man = load_manifest()
        assert_fingerprints(man)
        self.snap_ids, self.snap_vec, pin_ok = load_snapshot_vectors()
        assert pin_ok == 4942
        self.snap_idx = {r: i for i, r in enumerate(self.snap_ids)}
        self.meta = load_published_meta()
        self.sidecar = load_sidecar()
        self.adj = build_gprime(self.snap_ids, self.snap_idx, self.sidecar)
        self.hyg_ids: list[str] = json.loads(
            (BLEG.parents[0] / "hygiene" / "ids.json").read_text())

        funnel = json.loads((BLEG / "funnel.json").read_text())
        self.pairs = funnel["pairs"]          # [{pair_id, q, pos}] in masked order
        self.pools: dict[str, list[str]] = funnel["pools"]
        self.pos_by_query: dict[str, list[str]] = {}
        for p in self.pairs:
            self.pos_by_query.setdefault(p["q"], []).append(p["pos"])
        self.queries = sorted(self.pos_by_query)

        self.pool_ids: list[str] = []
        self.pool_query_of_row: list[str] = []
        self.pool_slices: dict[str, slice] = {}
        for q in self.queries:
            start = len(self.pool_ids)
            self.pool_ids.extend(self.pools[q])
            self.pool_query_of_row.extend([q] * len(self.pools[q]))
            self.pool_slices[q] = slice(start, len(self.pool_ids))
        self.X_fusion, self.X_abl, self.pool_cos = self._features(self.pool_ids,
                                                                  self.pool_query_of_row)

        gspec = json.loads((BLEG / "gspec_pairs.json").read_text())
        g_ids = [p["c"] for p in gspec["positives"]] + [p["c"] for p in gspec["hard"]]
        g_qs = [p["q"] for p in gspec["positives"]] + [p["q"] for p in gspec["hard"]]
        self.gspec_y = np.array([1] * len(gspec["positives"]) + [0] * len(gspec["hard"]),
                                dtype=np.int32)
        gx = []
        for q, c in zip(g_qs, g_ids):
            gx.append(self._pair_row(q, c))
        gx = np.array(gx, dtype=np.float64)
        self.gspec_X = gx
        self.gspec_X_abl = gx[:, [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]]
        self.gspec_base_rate = float(self.gspec_y.mean())

    def _features(self, ids: list[str], queries: list[str]):
        """Feature matrices for pairs (queries[k], ids[k]) in the given order."""
        ablated_idx = [FUSION_FEATURES.index(f) for f in ABLATED_FEATURES]
        X, Xa, cos = [], [], []
        ci = FUSION_FEATURES.index("cos_ab")
        for q, rid in zip(queries, ids):
            row = self._pair_row(q, rid)
            X.append(row)
            Xa.append(row[ablated_idx])
            cos.append(row[ci])
        return (np.array(X, dtype=np.float64),
                np.array(Xa, dtype=np.float64),
                np.array(cos, dtype=np.float64))

    def _pair_row(self, q: str, c: str) -> np.ndarray:
        i, j = self.snap_idx[q], self.snap_idx[c]
        c64 = cos64(self.snap_vec[i], self.snap_vec[j])
        cn, nj, ctcm = pair_graph_features(q, c, self.adj, self.snap_idx,
                                           self.snap_vec, self.snap_ids)
        ev = GraphEvidence(
            edge_ab=(c in self.adj.get(q, set())),
            edge_kind_supersedes=False,  # all snapshot edges are relates_to
            common_neighbors=cn,
            neighbor_jaccard=nj,
            cos_to_common_max=ctcm if cn else 0.0,
        )
        return np.array(feature_row(c64, ev, self.meta[q], self.meta[c]),
                        dtype=np.float64)

    # ── metrics ────────────────────────────────────────────────────────
    def query_metrics(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per-query recall@5 and mrr@10 for a scoring of every pool row.

        scores: mapping name -> array aligned with self.pool_ids.
        Returns {name: (recall5_by_query, mrr10_by_query)} aligned with
        self.queries; plus "cosine" from the frozen cos column.
        """
        all_scores = dict(scores)
        all_scores["cosine"] = self.pool_cos
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, sc in all_scores.items():
            rec = np.zeros(len(self.queries))
            mrr = np.zeros(len(self.queries))
            for qi, q in enumerate(self.queries):
                s = self.pool_slices[q]
                order = np.argsort(-sc[s])
                ids_sorted = self.pool_ids[s.start:s.stop]
                pos = set(self.pos_by_query[q])
                top5_ids = [ids_sorted[t] for t in order[:5].tolist()]
                rec[qi] = len(pos & set(top5_ids)) / len(pos)
                mrr[qi] = 0.0
                for rank, t in enumerate(order[:10].tolist(), start=1):
                    if ids_sorted[t] in pos:
                        mrr[qi] = 1.0 / rank
                        break
            out[name] = (rec, mrr)
        return out

    def nharm_mask(self) -> np.ndarray:
        """Queries with zero visible degree on G' (no-evidence stratum)."""
        return np.array([len(self.adj.get(q, ())) == 0 for q in self.queries])

    def competitor_edge_share(self) -> float:
        """Share of pool rows with a visible G' edge to their query
        (mechanism context for the P6 forecast)."""
        n_edge = 0
        n_all = 0
        for q in self.queries:
            s = self.pool_slices[q]
            for k in range(s.start, s.stop):
                c = self.pool_ids[k]
                n_all += 1
                if c in self.adj.get(q, ()):
                    n_edge += 1
        return n_edge / n_all if n_all else 0.0


def bootstrap_ci_delta(per_query_delta: np.ndarray, b: int = 10000,
                       seed: int = 20261101) -> dict:
    """Paired bootstrap over queries: mean delta + CI95 + halfwidth."""
    rng = np.random.default_rng(seed)
    n = len(per_query_delta)
    idx = rng.integers(0, n, size=(b, n))
    means = per_query_delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(per_query_delta.mean()),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "halfwidth": float((hi - lo) / 2.0)}
