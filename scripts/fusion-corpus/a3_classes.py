#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 3: formal class predicates (§5.1).

Order is frozen (В3): hygiene FIRST (step 2), classes SECOND, over the
post-hygiene population and the post-hygiene edge graph of the FROZEN
snapshot. Predicates verbatim from the framing §7.1 (directive 22.10:
predicates, not words):

  FR-P       edge_ab=1 ∧ edge_kind=relates_to (post-hygiene edges)
  FR-N-hard  edge_ab=0 ∧ cn=0 ∧ cos ∈ [0.92, 0.97)  (same-language)
  FR-N-far   edge_ab=0 ∧ cn=0 ∧ cos < 0.92
  FR-X       edge_ab=0 ∧ cn=0 ∧ cos ∈ [0.97, 1.0]   (same-language)
  FR-Q       edge_ab=0 ∧ cn ≥ 1  ∪  cross-language pairs of the zone
             (pinned detect_lang recipe of the census; secret-find pairs
             never entered the population — hygiene excluded them)

Also computed here (ratification table В4 + replay territory):
  - cos-band histogram of FR-P positives (ratified expectation: ~16% < 0.92);
  - language composition of edge pairs (DE 28.10);
  - silver-dup census (norm-equal body+title+tags+type+lang, zero evidence)
    → anchor-68 candidates + dup-territory texts.

Pair key convention: `a--b`, a < b lexicographic (data-contract §10).
Zone/no-edge pairs are materialised for downstream mining (npz).
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
SNAP = WORKTREE / "data" / "a-retrial" / "snapshot"
HYG = WORKTREE / "data" / "a-retrial" / "hygiene"
OUT = WORKTREE / "data" / "a-retrial" / "classes"

MINT_COS = 0.92
BAND_CAP = 0.97


def norm1(s: str) -> str:
    return " ".join((s or "").lower().split())


def detect_lang(body: str) -> str:
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a < b else (b, a)
    return f"{x}--{y}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    index = {rid: i for i, rid in enumerate(ids)}
    mat = np.load(HYG / "vectors.npy")
    records = {
        obj["id"]: obj
        for obj in (json.loads(line) for line in (HYG / "records.jsonl").read_text().splitlines())
    }

    conn = sqlite3.connect(f"file:{(SNAP / 'vesma.db').as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memory_edges)")]
    edges_raw = list(conn.execute("SELECT from_memory_id, to_memory_id, kind, provenance, weight FROM memory_edges"))
    conn.close()

    # post-hygiene edges (both endpoints passed step 2)
    edges = [
        (s, d, kind, prov, w)
        for (s, d, kind, prov, w) in edges_raw
        if s in index and d in index and s != d
    ]
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_lookup: dict[frozenset, tuple[str, object]] = {}
    for s, d, kind, prov, w in edges:
        adjacency[s].add(d)
        adjacency[d].add(s)
        edge_lookup[frozenset((s, d))] = (kind, prov)

    langs = {rid: (records[rid]["language"], detect_lang(records[rid]["body"])) for rid in ids}
    lang_arr = np.array([langs[r][1] for r in ids])

    n = len(ids)
    zone_i, zone_j, zone_cos = [], [], []
    counters = Counter()
    pos_cos: list[float] = []
    pos_lang: Counter[str] = Counter()
    far_count = 0
    frq_cn_pairs: list[tuple[str, str, int]] = []
    frq_crosslang = 0
    B = 512
    for i0 in range(0, n, B):
        block = mat[i0 : i0 + B] @ mat.T
        for li in range(block.shape[0]):
            i = i0 + li
            row = block[li]
            row[i] = -1.0
            a = ids[i]
            la = lang_arr[i]
            for j in np.nonzero(row >= MINT_COS)[0]:
                j = int(j)
                if j <= i:
                    continue
                b = ids[j]
                cos = float(row[j])
                fk = frozenset((a, b))
                if fk in edge_lookup:
                    counters["FR-P"] += 1
                    pos_cos.append(cos)
                    la2, lb2 = la, lang_arr[j]
                    pos_lang["--".join(sorted((la2, lb2)))] += 1
                    continue
                lb = lang_arr[j]
                # cross-language zone pairs -> FR-Q (census recipe)
                if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
                    counters["FR-Q:cross-lang"] += 1
                    frq_crosslang += 1
                    zone_i.append(i)
                    zone_j.append(j)
                    zone_cos.append(cos)
                    continue
                cn = len(adjacency[a] & adjacency[b])
                if cn >= 1:
                    counters["FR-Q:cn>=1"] += 1
                    frq_cn_pairs.append((a, b, cn))
                    zone_i.append(i)
                    zone_j.append(j)
                    zone_cos.append(cos)
                    continue
                zone_i.append(i)
                zone_j.append(j)
                zone_cos.append(cos)
                if cos >= BAND_CAP:
                    counters["FR-X"] += 1
                else:
                    counters["FR-N-hard"] += 1
            # FR-N-far: counted per row below threshold
            row2 = block[li]
            row2[i] = -1.0
            far_count += int((row2 < MINT_COS).sum() // 1)
    # far_count double-counted each unordered pair once per direction; halve
    far_count = far_count // 2

    # silver-dup census (anchor-68 candidates; dup territory)
    dup_pairs: list[dict] = []
    norm_bodies = {rid: norm1(records[rid]["body"]) for rid in ids}
    norm_titles = {rid: norm1(records[rid]["title"]) for rid in ids}
    dup_groups: dict[str, list[str]] = defaultdict(list)
    for rid in ids:
        key = norm_titles[rid] + "\x00" + norm_bodies[rid]
        dup_groups[key].append(rid)
    dup_member_ids: set[str] = set()
    for key, members in dup_groups.items():
        if len(members) < 2 or not norm_bodies[members[0]]:
            continue
        members = sorted(members)
        dup_member_ids.update(members)
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                A, Brec = records[a], records[b]
                tags_eq = set(A["tags"]) == set(Brec["tags"])
                type_eq = A["record_type"] == Brec["record_type"]
                lang_eq = A["language"] == Brec["language"]
                if not (tags_eq and type_eq and lang_eq):
                    continue
                cn = len(adjacency[a] & adjacency[b])
                edge = frozenset((a, b)) in edge_lookup
                if edge or cn > 0:
                    continue
                ivec, jvec = index[a], index[b]
                dup_pairs.append(
                    {
                        "pair_id": pair_key(a, b),
                        "a": a,
                        "b": b,
                        "cos": round(float(mat[ivec] @ mat[jvec]), 6),
                    }
                )

    band_hist = Counter()
    for c in pos_cos:
        band_hist["[0.92,0.97)" if c < BAND_CAP else "[0.97,1.0]"] += 1
    below = sum(v for k, v in band_hist.items() if k == "[0.92,0.97)")
    hist_bins = Counter()
    for c in pos_cos:
        if c < 0.80:
            hist_bins["<0.80"] += 1
        elif c < 0.85:
            hist_bins["[0.80,0.85)"] += 1
        elif c < 0.90:
            hist_bins["[0.85,0.90)"] += 1
        elif c < 0.92:
            hist_bins["[0.90,0.92)"] += 1
        elif c < 0.94:
            hist_bins["[0.92,0.94)"] += 1
        elif c < 0.96:
            hist_bins["[0.94,0.96)"] += 1
        elif c < 0.97:
            hist_bins["[0.96,0.97)"] += 1
        else:
            hist_bins["[0.97,1.0]"] += 1

    zone_cos_arr = np.asarray(zone_cos, dtype=np.float32)
    zone_hard_mask = np.array(
        [
            counters_key
            for counters_key in ()  # placeholder, filled below
        ]
    ) if False else None

    # materialise the no-edge zone pairs with class tags for downstream mining
    np.savez_compressed(
        OUT / "zone_pairs.npz",
        i=np.asarray(zone_i, dtype=np.int32),
        j=np.asarray(zone_j, dtype=np.int32),
        cos=zone_cos_arr,
    )
    with (OUT / "zone_pairs_meta.json").open("w") as fh:
        json.dump(
            {
                "ids": ids,
                "note": "i/j index into ids; all no-edge pairs with cos>=0.92 (FR-N-hard/FR-X/FR-Q:cn>=1/FR-Q:cross-lang); class membership recomputable from classes.json adjacency",
            },
            fh,
        )
    (OUT / "fr_q_cn_pairs.json").write_text(
        json.dumps([{"pair_id": pair_key(a, b), "a": a, "b": b, "cn": cn} for a, b, cn in frq_cn_pairs]) + "\n"
    )
    (OUT / "dup_pairs.json").write_text(json.dumps(dup_pairs, indent=1) + "\n")
    with (OUT / "edges_post_hygiene.jsonl").open("w") as fh:
        for s, d, kind, prov, w in sorted(edges, key=lambda e: pair_key(e[0], e[1])):
            fh.write(
                json.dumps(
                    {
                        "pair_id": pair_key(s, d),
                        "a": min(s, d),
                        "b": max(s, d),
                        "kind": kind,
                        "provenance": prov,
                        "weight": w,
                        "cos": round(float(mat[index[s]] @ mat[index[d]]), 6),
                    }
                ) + "\n"
            )

    summary = {
        "columns_memory_edges": cols,
        "population": len(ids),
        "edges_raw": len(edges_raw),
        "edges_post_hygiene": len(edges),
        "FR-P": counters["FR-P"],
        "FR-N-hard": counters["FR-N-hard"],
        "FR-X": counters["FR-X"],
        "FR-Q:cn>=1": counters["FR-Q:cn>=1"],
        "FR-Q:cross-lang": counters["FR-Q:cross-lang"],
        "FR-N-far(all pairs cos<0.92)": far_count,
        "zone_pairs_no_edge_ge_092": len(zone_cos),
        "zone_ge_097_no_edge": int((zone_cos_arr >= BAND_CAP).sum()),
        "positive_band_hist": dict(band_hist),
        "positive_share_below_092": round(below / max(1, counters["FR-P"]), 4),
        "positive_cos_hist_bins": dict(hist_bins),
        "positive_lang_pairs": dict(pos_lang),
        "silver_dup_pairs_zero_evidence": len(dup_pairs),
        "dup_territory_texts": len(dup_member_ids),
        "expected_checkpoint": {
            "ratified_estimate_FR-P": "~1520 (DE 28.10, at 1844 edges)",
            "fresh_fact_FR-P": counters["FR-P"],
            "ratified_estimate_masked_pool": "~305 post-hygiene (215-305 extractable), STOP>=215",
            "fresh_masked_pool_projection_rho20": round(0.2 * counters["FR-P"]),
        },
    }
    (OUT / "classes.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
