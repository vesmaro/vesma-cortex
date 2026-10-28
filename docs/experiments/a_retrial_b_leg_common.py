"""B-leg fusion-ranker common helpers (shared by the four B-leg recipes).

Frozen Phase-1 corpus (a-retrial, D0 snapshot) loaders + the frozen graph
feature contract (``cortex.features.graph``) applied ON TOP of the feature
graph G' = G - (masked + knockon), exactly as ratified by the framing spec
§6.2 / A-retrial prereg §5.1.

Privacy: raw store content is read only to compute id-level features
(language detection over body text, in-memory); nothing textual ever leaves
the process. Repo/report outputs carry ids and numbers only.

Interpreter: vesma-cortex/.venv (lightgbm 4.7.0 / onnxruntime 1.30.0).
PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]  # wt/fusion-framing
DATA = WT / "data" / "a-retrial"
SRC = WT / "src"
sys.path.insert(0, str(SRC))

from cortex.features.graph import GraphEvidence, GraphSidecarError  # noqa: E402

SNAP = DATA / "snapshot"
ROLES = DATA / "roles"
MASK = DATA / "mask"
CLASSES = DATA / "classes"
HYG = DATA / "hygiene"
GATE = DATA / "gate"
BLEG = DATA / "b-leg"

MANIFEST_JSON = WT / "docs" / "experiments" / "a-retrial-phase1-corpus-manifest.json"

EMBED_PIN = "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
EMBED_DIM = 384

#: Feature order of the fusion artifact (canon of THIS duty family, v1).
FUSION_FEATURES = (
    "cos_ab",
    "edge_ab",
    "edge_kind_supersedes",
    "common_neighbors",
    "neighbor_jaccard",
    "cos_to_common_max",
    "tag_jaccard",
    "same_record_type",
    "same_language",
)
#: Ablated leg = same recipe WITHOUT the graph five (g_abl, retrain-inclusive).
ABLATED_FEATURES = (
    "cos_ab",
    "tag_jaccard",
    "same_record_type",
    "same_language",
)

#: Positive cos bands of the corpus (manifest band_hist boundaries).
BANDS = ((0.0, 0.92), (0.92, 0.94), (0.94, 0.96), (0.96, 0.97), (0.97, 1.01))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


def assert_fingerprints(man: dict) -> dict[str, str]:
    """Recompute sha256 of every frozen input file pinned in the manifest."""
    fps = man["fingerprints"]
    pairs = {
        "hygiene_records_jsonl": HYG / "records.jsonl",
        "hygiene_vectors_npy": HYG / "vectors.npy",
        "edges_post_hygiene_jsonl": CLASSES / "edges_post_hygiene.jsonl",
        "masked_pairs_json": MASK / "masked_pairs.json",
        "knockon_pairs_json": MASK / "knockon_pairs.json",
        "sidecar_G_jsonl": MASK / "sidecar_G.jsonl",
        "train_pairs_jsonl": ROLES / "train_pairs.jsonl",
        "val_pairs_jsonl": ROLES / "val_pairs.jsonl",
        "replay_probe_jsonl": ROLES / "replay_probe_2500.jsonl",
    }
    actual: dict[str, str] = {}
    for key, path in pairs.items():
        want = fps[key]
        got = sha256_file(path)
        if got != want:
            raise AssertionError(f"fp mismatch {key}: manifest {want} vs actual {got}")
        actual[key] = got
    for key, name in (
        ("snapshot_vesma_db", "vesma.db"),
        ("snapshot_vectors_db", "vectors.db"),
    ):
        got = sha256_file(SNAP / name)
        if got != fps[key]:
            raise AssertionError(f"fp mismatch {key}")
        actual[key] = got
    return actual


def pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a < b else (b, a)
    return f"{x}--{y}"


def load_snapshot_vectors() -> tuple[list[str], np.ndarray, int]:
    """All published snapshot vectors (4942 x 384 float32) + pin-assert count."""
    con = sqlite3.connect(f"file:{SNAP / 'vectors.db'}?mode=ro", uri=True)
    rows = con.execute("SELECT id, vector, metadata FROM embeddings ORDER BY id").fetchall()
    con.close()
    ids = [r[0] for r in rows]
    mat = np.zeros((len(rows), EMBED_DIM), dtype=np.float32)
    pin_ok = 0
    for k, (_id, blob, meta) in enumerate(rows):
        mat[k] = np.frombuffer(blob, dtype="<f4", count=EMBED_DIM)
        md = json.loads(meta) if meta else {}
        if md.get("model_fingerprint") == EMBED_PIN:
            pin_ok += 1
    return ids, mat, pin_ok


def load_published_meta() -> dict[str, dict]:
    """id -> {memory_type, tags(list), language} for all published records.

    language via the pinned census detect_lang recipe (verbatim copy from
    scripts/fusion-corpus/a3_classes.py — no reimplementation).
    """
    con = sqlite3.connect(f"file:{SNAP / 'vesma.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, tags, memory_type, content FROM memories WHERE status='published'"
    ).fetchall()
    con.close()
    meta: dict[str, dict] = {}
    for rid, tags, mtype, content in rows:
        try:
            tag_list = json.loads(tags) if tags else []
            if not isinstance(tag_list, list):
                tag_list = []
        except (json.JSONDecodeError, TypeError):
            tag_list = []
        meta[rid] = {
            "memory_type": mtype or "",
            "tags": sorted(str(t) for t in tag_list),
            "language": detect_lang(content or ""),
        }
    return meta


def detect_lang(body: str) -> str:
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def load_sidecar() -> dict[str, GraphEvidence]:
    """Frozen sidecar rows validated through the frozen contract reader."""
    out: dict[str, GraphEvidence] = {}
    with open(MASK / "sidecar_G.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            ev = GraphEvidence.from_row(row)
            key = row["pair_id"]
            if key in out:
                raise GraphSidecarError(f"duplicate sidecar pair_id {key}")
            out[key] = ev
    return out


def load_edges_direction() -> dict[str, tuple[str, str]]:
    """pair_id -> (a=from, b=to) from frozen post-hygiene edges."""
    out: dict[str, tuple[str, str]] = {}
    with open(CLASSES / "edges_post_hygiene.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["pair_id"]] = (row["a"], row["b"])
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_gprime(snap_ids: list[str], snap_idx: dict[str, int],
                 sidecar: dict[str, GraphEvidence]) -> dict[str, set[str]]:
    """Undirected adjacency of the FEATURE graph G' (edge_ab=true rows only)."""
    adj: dict[str, set[str]] = {}
    n_gprime = 0
    for key, ev in sidecar.items():
        if not ev.edge_ab:
            continue
        a, b = key.split("--")
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        n_gprime += 1
    known = set(snap_idx)
    missing = [x for k, v in adj.items() for x in (k, *v) if x not in known]
    if missing:
        raise AssertionError(f"G' endpoints missing from snapshot vectors: {len(missing)}")
    return adj


def cos_b2(u: np.ndarray, v: np.ndarray) -> float:
    """The B2 sidecar cosine recipe: float32 dot / norms, clipped."""
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return min(1.0, max(-1.0, float(np.dot(u, v) / denom)))


def cos64(u: np.ndarray, v: np.ndarray) -> float:
    """float64 recomputation of cosine from frozen float32 vectors."""
    a = u.astype(np.float64)
    b = v.astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return min(1.0, max(-1.0, float(np.dot(a, b) / denom)))


def pair_graph_features(a: str, b: str, adj: dict[str, set[str]],
                        snap_idx: dict[str, int], vec: np.ndarray,
                        ids: list[str]) -> tuple[int, float, float]:
    """(common_neighbors, neighbor_jaccard, cos_to_common_max) on G'.

    B2 recipe verbatim: nj = |∩|/|∪| (0 on empty union); cos_to_common_max =
    max over ALL (side, neighbor) values of the B2 cosine.
    """
    na = adj.get(a, set())
    nb = adj.get(b, set())
    common = na & nb
    union = na | nb
    nj = (len(common) / len(union)) if union else 0.0
    ctcm = 0.0
    if common:
        vals = []
        for side in (a, b):
            si = snap_idx[side]
            for n in sorted(common):
                vals.append(cos_b2(vec[si], vec[snap_idx[n]]))
        ctcm = max(vals)
    return len(common), nj, ctcm


def band_of(cos: float) -> str:
    for lo, hi in BANDS:
        if lo <= cos < hi:
            label = f"[{lo},{hi})" if hi <= 1.0 else "<0.92" if lo == 0.0 else f"[{lo},1.0]"
            return label
    return "?"


def meta_features(ma: dict, mb: dict) -> tuple[float, float, float]:
    """(tag_jaccard, same_record_type, same_language)."""
    ta, tb = set(ma["tags"]), set(mb["tags"])
    union = ta | tb
    tj = (len(ta & tb) / len(union)) if union else 0.0
    st = 1.0 if ma["memory_type"] == mb["memory_type"] and ma["memory_type"] else 0.0
    sl = 1.0 if ma["language"] == mb["language"] and ma["language"] != "none" else 0.0
    return tj, st, sl


def feature_row(cos: float, ev: GraphEvidence, ma: dict, mb: dict,
                ablated: bool = False) -> list[float]:
    tj, st, sl = meta_features(ma, mb)
    if ablated:
        return [cos, tj, st, sl]
    return [
        cos,
        1.0 if ev.edge_ab else 0.0,
        1.0 if ev.edge_kind_supersedes else 0.0,
        float(ev.common_neighbors),
        ev.neighbor_jaccard,
        ev.cos_to_common_max if ev.cos_to_common_max is not None else 0.0,
        tj,
        st,
        sl,
    ]
