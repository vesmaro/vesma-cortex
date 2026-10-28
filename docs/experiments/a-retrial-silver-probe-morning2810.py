#!/usr/bin/env python3
"""A-retrial silver probe — morning-2810 snapshot: sub-0.92 relations vs old mint policy.

Question (TL, 2026-10-29): are the sub-0.92 relations of the morning snapshot real
semantic relations or mint-policy garbage (auto-dedupe duty removed them during the
day — semantics or threshold?).

Preregistered expectations this script CHECKS (not assumes):
  E1: morning snapshot (10:08) holds ~289 sub-0.92 edges  -> stratified sample of 50.
  E2: D0 snapshot (20:39) holds 4 sub-0.92 edges.

Findings recorded in docs/experiments/a-retrial-runlog.md. E1 was REFUTED by direct
measurement: the true sub-0.92 edge mass of the morning snapshot is 4. The likely
origin of the "289" figure is probe_c3b_fresh.py's hygiene-filtered edge-zone count
(1844 total - 1555 hygiene-passed in-zone = 289 edges with >=1 hygiene-failing
endpoint), not a sub-0.92 mass. The script therefore adapts: it labels ALL 4 true
sub-0.92 edges (the full population), a boundary stratum [0.92,0.925) (10 pairs —
does semantic quality cliff at the threshold?), and control edges >=0.92 that
survived in D0. The direct duty diff (morning vs D0 edge sets) is measured here too.

Interpreter: /usr/bin/python3 + numpy (float64 cosine, single code path).
Raw pair texts live ONLY in gitignored data/a-retrial/; holdout v4.5 / g4-spec-v1
are never read; no training, no store mutation (all DBs opened read-only).
"""

from __future__ import annotations

import json
import random
import sqlite3
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]  # wt/fusion-framing
DATA = WT / "data" / "a-retrial"
MORNING = DATA / "snapshot-morning-2810"
D0 = DATA / "snapshot"
OUT_NUMBERS = (
    WT / "docs" / "experiments" / "a-retrial-silver-probe-morning2810-summary.json"
)

MORNING_SHA = {
    "vesma.db": "1fa0a7983675aade803ff1206d1cdfe847b8ea4b38068b6869c35c37bca30736",
    "vectors.db": "ba137d8cae88df4c13c39592d0dbd4feeda25f913202ccb98509c028f25ce389",
}
D0_SHA = {
    "vesma.db": "252b9d5a78b9c9b0b444509048c696b687ee14981824c57ce8ce5957068403bf",
    "vectors.db": "0e97cc096f198caaa5ed19c9266c7e8b3e40b060dd5b18caca36e47df3d18278",
}
PIN = "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
SEED = 20261031
N_BOUNDARY = 10
BOUNDARY_HI = 0.925
N_CONTROL = 10
BANDS = [(0.92, 0.94), (0.94, 0.96), (0.96, 0.97), (0.97, 1.01)]  # [a, b)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ro(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=10.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def band_of(c: float) -> str:
    if c < 0.92:
        return "<0.92"
    for a, b in BANDS:
        if a <= c < b:
            return f"[{a:.2f},{min(b, 1.0):.2f})" if b <= 1.0 else "[0.97,1.0]"
    return "nan"


def load_vectors(db: Path):
    """Return (ids, float64 unit-normalized matrix, pin_stats). Single code path."""
    conn = ro(db)
    ids, rows = [], []
    pin_ok = pin_bad = 0
    for mid, blob, meta_json in conn.execute(
        "SELECT id, vector, metadata FROM embeddings"
    ):
        meta = json.loads(meta_json or "{}")
        if str(meta.get("model_fingerprint", "")).startswith(PIN[:21]):
            pin_ok += 1
        else:
            pin_bad += 1
        assert isinstance(blob, bytes) and len(blob) == 384 * 4, (mid, len(blob))
        ids.append(mid)
        rows.append(np.frombuffer(blob, dtype=np.float32))
    conn.close()
    mat = np.stack(rows).astype(np.float64)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return ids, mat, {"pin_ok": pin_ok, "pin_bad": pin_bad, "total": len(ids)}


def edge_cosines(vesma: Path, vectors: Path):
    """All relates_to edges of a snapshot with float64 cosine, one code path."""
    ids, mat, pins = load_vectors(vectors)
    idx = {m: i for i, m in enumerate(ids)}
    conn = ro(vesma)
    edges = conn.execute(
        "SELECT from_memory_id, to_memory_id FROM memory_edges WHERE kind='relates_to'"
    ).fetchall()
    mem_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    conn.close()
    missing = [e for e in edges if e[0] not in idx or e[1] not in idx]
    assert not missing, f"edge endpoints missing from vectors: {len(missing)}"
    cos = np.array([float(mat[idx[a]] @ mat[idx[b]]) for a, b in edges])
    assert np.all(np.isfinite(cos)) and cos.min() >= -1.0 and cos.max() <= 1.0
    return edges, cos, pins, mem_count


def main() -> int:
    out = {"seed": SEED}

    # A. integrity of both snapshots
    for name, expect in MORNING_SHA.items():
        got = sha256_file(MORNING / name)
        assert got == expect, f"morning {name}: {got}"
    for name, expect in D0_SHA.items():
        got = sha256_file(D0 / name)
        assert got == expect, f"D0 {name}: {got}"
    out["sha_verified"] = {"morning": list(MORNING_SHA.values()), "d0": list(D0_SHA.values())}

    # B. morning edge mass, float64
    m_edges, m_cos, m_pins, m_mem = edge_cosines(MORNING / "vesma.db", MORNING / "vectors.db")
    out["morning"] = {
        "memories": m_mem,
        "vectors": m_pins,
        "edges": len(m_edges),
        "cos_min": round(float(m_cos.min()), 6),
        "cos_mean": round(float(m_cos.mean()), 6),
        "cos_max": round(float(m_cos.max()), 6),
        "band_hist": dict(Counter(band_of(c) for c in m_cos)),
        "sub092_edges": int((m_cos < 0.92).sum()),
        "sub092_detail": sorted(round(float(c), 6) for c in m_cos[m_cos < 0.92]),
    }

    # C. D0 edge mass + duty diff morning -> D0
    d_edges, d_cos, d_pins, d_mem = edge_cosines(D0 / "vesma.db", D0 / "vectors.db")
    out["d0"] = {
        "memories": d_mem,
        "vectors": d_pins,
        "edges": len(d_edges),
        "band_hist": dict(Counter(band_of(c) for c in d_cos)),
        "sub092_edges": int((d_cos < 0.92).sum()),
    }
    out["d0"]["sub092_same_as_morning"] = sorted(
        f"{a[:8]}--{b[:8]}" for (a, b), c in zip(d_edges, d_cos) if c < 0.92
    ) == sorted(
        f"{a[:8]}--{b[:8]}" for (a, b), c in zip(m_edges, m_cos) if c < 0.92
    )
    m_set = {tuple(e): c for e, c in zip(m_edges, m_cos)}
    d_set = {tuple(e) for e in d_edges}
    removed = sorted(p for p in m_set if p not in d_set)
    added = sorted(p for p in d_set if p not in m_set)
    survived = sorted(p for p in m_set if p in d_set)
    r_cos = np.array([m_set[p] for p in removed]) if removed else np.array([])
    a_idx = {tuple(e): c for e, c in zip(d_edges, d_cos)}
    a_cos = np.array([a_idx[p] for p in added]) if added else np.array([])
    # duty target: records vs edges
    d_conn = ro(D0 / "vesma.db")
    d_mems = {r[0] for r in d_conn.execute("SELECT id FROM memories")}
    d_conn.close()
    r_ep_gone = sum(1 for a, b in removed if a not in d_mems or b not in d_mems)
    out["duty_diff"] = {
        "removed_edges": len(removed),
        "removed_band_hist": dict(Counter(band_of(c) for c in r_cos)),
        "removed_sub092": int((r_cos < 0.92).sum()) if len(r_cos) else 0,
        "removed_min_cos": round(float(r_cos.min()), 6) if len(r_cos) else None,
        "removed_max_cos": round(float(r_cos.max()), 6) if len(r_cos) else None,
        "removed_endpoint_records_gone": r_ep_gone,
        "added_edges": len(added),
        "added_band_hist": dict(Counter(band_of(c) for c in a_cos)),
        "added_sub092": int((a_cos < 0.92).sum()) if len(a_cos) else 0,
        "survived_edges": len(survived),
    }

    # D. silver sample
    rng = random.Random(SEED)
    sample = []
    sub_pairs = [p for p, c in zip(m_edges, m_cos) if c < 0.92]
    for p in sub_pairs:  # ALL true sub-0.92 edges, none skipped
        sample.append({"pair": p, "group": "sub092", "cos": round(m_set[tuple(p)], 6)})
    boundary = sorted(p for p, c in m_set.items() if 0.92 <= c < BOUNDARY_HI)
    rng.shuffle(boundary)
    for p in boundary[:N_BOUNDARY]:
        sample.append({"pair": p, "group": "boundary", "cos": round(m_set[p], 6)})
    ctrl_pool = sorted(p for p in survived if m_set[p] >= BOUNDARY_HI)
    rng.shuffle(ctrl_pool)
    for p in ctrl_pool[:N_CONTROL]:
        sample.append({"pair": p, "group": "control", "cos": round(m_set[p], 6)})

    # E. texts (gitignored data only)
    m_conn = ro(MORNING / "vesma.db")
    texts = {
        r[0]: {"title": r[1] or "", "content": r[2] or "", "tags": r[3] or "[]",
               "memory_type": r[4], "created_at": r[5]}
        for r in m_conn.execute(
            "SELECT id, title, content, tags, memory_type, created_at FROM memories")
    }
    m_conn.close()
    sample_path = MORNING / "silver_sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for s in sample:
            a, b = s["pair"]
            row = {
                "from_id": a, "to_id": b, "group": s["group"], "cos": s["cos"],
                "band": band_of(s["cos"]),
                "from": texts[a], "to": texts[b],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # F. committable numbers-only summary (short ids)
    out["sample"] = {
        "path_gitignored": str(sample_path.relative_to(WT)),
        "n_total": len(sample),
        "n_sub092": sum(1 for s in sample if s["group"] == "sub092"),
        "n_boundary": sum(1 for s in sample if s["group"] == "boundary"),
        "n_control": sum(1 for s in sample if s["group"] == "control"),
        "boundary_pool_size": len(boundary),
        "pairs_short": [
            {"g": s["group"], "cos": s["cos"], "a": s["pair"][0][:8], "b": s["pair"][1][:8]}
            for s in sample
        ],
    }
    OUT_NUMBERS.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out["morning"], ensure_ascii=False))
    print(json.dumps(out["d0"], ensure_ascii=False))
    print(json.dumps(out["duty_diff"], ensure_ascii=False))
    print("boundary_pool:", len(boundary))
    print(f"sample rows written: {len(sample)} -> {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
