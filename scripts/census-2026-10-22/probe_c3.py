#!/usr/bin/env python3
"""C3 probe — read-only live-store census of real-real dup candidates with graph evidence.

Mirror of the pool-build hygiene chain (_compose_record order), strictly RO
(mode=ro + query_only). Raw content stays in memory / /tmp only; output is
aggregates + <=1-line anonymized skeletons.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/b2-graph-corpus")
sys.path.insert(0, str(REPO / "src"))
from cortex.data.store_export import HygieneCounters, make_scanner  # noqa: E402

STORE = Path("/var/home/abyss/.vesma/data")
VESMA_DB = STORE / "vesma.db"
VECTORS_DB = STORE / "vectors.db"
ENGINE_SRC = "/tmp/b2_engine_detectors"
NO_FEDERATE_TAG = "mnemos:no-federate"
EMBED_DIM, BLOB, UNIT_TOL = 384, 384 * 4, 1e-3
PIN_PREFIX = "nano:sha256:3b752e06"
MINT_COS = 0.92
BAND_CAP = 0.97


def ro(db: Path):
    conn = __import__("sqlite3").connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def norm1(s):
    return " ".join((s or "").lower().split())


def norm2(s):
    s = re.sub(r"[^\w\s]", " ", (s or "").lower(), flags=re.UNICODE)
    return " ".join(s.split())


def detect_lang(body):
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def token_razor_numeric(ba, bb):
    ta, tb = norm1(ba).split(), norm1(bb).split()
    if len(ta) != len(tb):
        return False
    diff = 0
    for xa, xb in zip(ta, tb):
        if xa == xb:
            continue
        diff += 1
        if not (any(ch.isdigit() for ch in xa) and any(ch.isdigit() for ch in xb)):
            return False
    return diff > 0


def anon(body, n=60):
    letters = [c for c in body if c.isalpha()]
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    script = "cyr" if cyr > len(letters) / 2 else "lat"
    shape = "".join("d" if ch.isdigit() else ("w" if ch.isalpha() else ".") for ch in body[:n])
    return f"<{script} len={len(body)} shape={shape}...>"


def main() -> int:
    scanner = make_scanner(ENGINE_SRC)
    assert scanner.provenance == "engine", scanner.provenance

    conn = ro(VESMA_DB)
    mem = {}
    for row in conn.execute(
        "SELECT id, title, content, tags, memory_type, created_at, metadata,"
        " quarantine_reason, status FROM memories"
    ):
        mem[row[0]] = row
    edges = list(
        conn.execute(
            "SELECT from_memory_id, to_memory_id, kind, provenance, weight FROM memory_edges"
        )
    )
    conn.close()

    conn = ro(VECTORS_DB)
    embeddings = {}
    for mid, blob, meta_json in conn.execute("SELECT id, vector, metadata FROM embeddings"):
        if isinstance(blob, bytes) and len(blob) == BLOB:
            embeddings[mid] = (blob, meta_json or "{}")
    conn.close()

    counters = HygieneCounters()
    counters.candidates_sql = len(mem)
    passed: dict[str, dict] = {}
    for mid, row in mem.items():
        (rid, title, content, tags_json, mtype, created, meta_json, quar, status) = row
        if status != "published":
            counters.exclude("sql-gate:status")
            continue
        if mtype not in ("note", "snippet", "fact"):
            counters.exclude("sql-gate:memory_type")
            continue
        if len(content or "") < 80:
            counters.exclude("sql-gate:length<80")
            continue
        tags = tuple(str(t) for t in json.loads(tags_json or "[]"))
        if NO_FEDERATE_TAG in tags:
            counters.exclude("no-federate")
            continue
        if quar:
            counters.exclude("quarantine")
            continue
        emb = embeddings.get(rid)
        if emb is None:
            counters.exclude("missing-embedding")
            continue
        blob, meta_j = emb
        meta = json.loads(meta_j)
        pin = meta.get("model_fingerprint")
        if not pin:
            counters.exclude("unpinned-embedding")
            continue
        if not str(pin).startswith(PIN_PREFIX):
            counters.exclude("pin-mismatch")
            continue
        if not meta.get("content_hash"):
            counters.exclude("missing-content-hash")
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        nrm = float(np.linalg.norm(vec))
        if not np.isfinite(nrm) or abs(nrm - 1.0) > UNIT_TOL:
            counters.exclude("bad-vector")
            continue
        # language from memories.metadata (same source as pool: memories.metadata.canon.language)
        mmeta = json.loads(meta_json or "{}")
        canon = mmeta.get("canon") if isinstance(mmeta.get("canon"), dict) else {}
        language = canon.get("language")
        language = language if isinstance(language, str) else None
        reasons = scanner.scan(title or "", content or "", tags)
        if reasons:
            for r in reasons:
                counters.exclude(r)
            continue
        passed[rid] = {
            "title": title or "",
            "body": content or "",
            "tags": tags,
            "language": language,
            "record_type": mtype,
            "created_at": created,
            "vec": vec / nrm,
        }

    print("hygiene candidates:", counters.candidates_sql)
    print("hygiene excluded:", counters.reasons, "total:", counters.excluded_total)
    print("passed:", len(passed))

    # drift: pool endpoints still passing
    pool_ids = set()
    with (REPO / "datasets/b2-graph/pool_manifest.txt").open() as fh:
        for line in fh:
            a, b = line.split()[0].split("--")
            pool_ids.add(a)
            pool_ids.add(b)
    still = sum(1 for p in pool_ids if p in passed)
    print(f"pool endpoints: {len(pool_ids)}, still hygiene-passing now: {still}")

    # adjacency + components
    adjacency: dict[str, set] = defaultdict(set)
    edge_lookup: dict[frozenset, tuple] = {}
    for s, d, kind, prov, w in edges:
        if s == d:
            continue
        adjacency[s].add(d)
        adjacency[d].add(s)
        edge_lookup[frozenset((s, d))] = (kind, prov, w)

    ids = sorted(passed)
    mat = np.stack([passed[i]["vec"] for i in ids])
    print("matrix:", mat.shape)

    hits = []
    B = 512
    for i0 in range(0, len(ids), B):
        block = mat[i0 : i0 + B] @ mat.T
        for li, row in enumerate(block):
            i = i0 + li
            row[i] = -1.0  # no self
            js = np.nonzero(row >= MINT_COS)[0]
            for j in js:
                if j <= i:
                    continue
                hits.append((i, int(j), float(row[j])))
    print(f"pairs cos>={MINT_COS}: {len(hits)}")

    # silver classification
    cls_counter = Counter()
    dup_rows = []
    band_counter = Counter()
    ev_counter = Counter()
    for i, j, cos in hits:
        ia, ib = ids[i], ids[j]
        A, Brec = passed[ia], passed[ib]
        la, lb = detect_lang(A["body"]), detect_lang(Brec["body"])
        if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
            cls_counter["quarantine"] += 1
            continue
        body_eq = norm1(A["body"]) == norm1(Brec["body"])
        title_eq = norm1(A["title"]) == norm1(Brec["title"])
        tags_eq = set(A["tags"]) == set(Brec["tags"])
        type_eq = A["record_type"] == Brec["record_type"]
        lang_eq = A["language"] == Brec["language"]
        s_dup = body_eq and title_eq and tags_eq and type_eq and lang_eq
        if s_dup:
            cls = "silver-dup"
        elif body_eq and title_eq:
            cls = "notdup-metadata-T2"
        elif (not body_eq) and token_razor_numeric(A["body"], Brec["body"]):
            cls = "notdup-razor-numeric-T3"
        else:
            cls = "disputed"
        cls_counter[cls] += 1
        band = "[0.92,0.97)" if cos < BAND_CAP else "[0.97,1.0]"
        band_counter[(cls, band)] += 1
        if cls == "silver-dup":
            fro = frozenset((ia, ib))
            edge = edge_lookup.get(fro)
            cn = len(adjacency[ia] & adjacency[ib])
            union = len(adjacency[ia] | adjacency[ib])
            nj = (cn / union) if union else 0.0
            cmax = 0.0
            for n in adjacency[ia] & adjacency[ib]:
                if n in passed:
                    cmax = max(cmax, float(passed[ia]["vec"] @ passed[n]["vec"]),
                               float(passed[ib]["vec"] @ passed[n]["vec"]))
            ev = bool(edge) or cn > 0
            ev_counter[("silver-dup", "evidence" if ev else "zero-evidence")] += 1
            ev_counter[("silver-dup", "edge" if edge else "no-edge")] += 1
            ev_counter[("silver-dup", band)] += 1
            dup_rows.append({
                "pair": f"{ia[:8]}..--{ib[:8]}..", "cos": round(cos, 4),
                "band": band, "edge": 1 if edge else 0,
                "edge_kind": (edge or (None,))[0], "cn": cn, "nj": round(nj, 4),
                "cos_to_common_max": round(cmax, 4),
                "tags_eq": tags_eq, "type_eq": type_eq, "lang_eq": lang_eq,
                "created_delta": A["created_at"] != Brec["created_at"],
                "a": anon(A["body"]), "b": anon(Brec["body"]),
            })

    print("\nsilver classes on cos>=0.92 pairs:", dict(cls_counter))
    print("band x class:", {f"{k[0]}|{k[1]}": v for k, v in sorted(band_counter.items())})
    print("silver-dup evidence split:", {f"{k[0]}|{k[1]}": v for k, v in sorted(ev_counter.items())})

    # continuity on silver-dup with evidence
    dwe = sorted(r["cos"] for r in dup_rows if r["edge"] or r["cn"] > 0)
    if dwe:
        gaps = [b - a for a, b in zip(dwe, dwe[1:])]
        print(f"dup-with-evidence n={len(dwe)} cos range [{dwe[0]}, {dwe[-1]}] max-gap {max(gaps):.4f}")
    dwe_band = [c for c in dwe if c >= BAND_CAP]
    if dwe_band:
        gaps = [b - a for a, b in zip(dwe_band, dwe_band[1:])]
        print(f"dup-with-evidence at cos>=0.97: n={len(dwe_band)} range [{dwe_band[0]},{dwe_band[-1]}] max-gap {max(gaps):.4f}" if len(dwe_band) > 1 else f"dup-with-evidence at cos>=0.97: n={len(dwe_band)}")

    # all-pairs cos>=0.97 context
    n97 = sum(1 for _, _, c in hits if c >= BAND_CAP)
    print(f"\ncontext: all pairs cos>=0.97: {n97}")

    # floor check: >=30 with K=30 + continuity <=0.005
    print("\nC3 verdict inputs:")
    print("  silver-dup total:", cls_counter["silver-dup"])
    print("  silver-dup with graph evidence:", len(dwe))

    for r in dup_rows[:20]:
        print("  DUP", r["pair"], r["cos"], r["band"], "edge", r["edge"], r["edge_kind"],
              "cn", r["cn"], "nj", r["nj"], "cmax", r["cos_to_common_max"],
              "env-delta", r["created_delta"], r["a"])

    Path("/tmp/b2_census/c3_numbers.json").write_text(json.dumps({
        "store_snapshot": {"memories": len(mem), "edges": len(edges), "vectors": len(embeddings)},
        "hygiene_passed": len(passed),
        "pool_endpoints_still_passing": still,
        "pairs_ge_092": len(hits),
        "pairs_ge_097": n97,
        "silver_classes": dict(cls_counter),
        "band_x_class": {f"{k[0]}|{k[1]}": v for k, v in band_counter.items()},
        "dup_evidence_split": {f"{k[0]}|{k[1]}": v for k, v in ev_counter.items()},
        "dup_rows": dup_rows,
    }, indent=1))
    print("\nwrote /tmp/b2_census/c3_numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
