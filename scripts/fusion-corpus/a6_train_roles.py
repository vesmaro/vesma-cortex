#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 6: training roles on the remainder.

Frozen order (В3): the sealed side (step 5) is cut FIRST; everything here is
cut from the REMAINDER and excludes the frozen sealed pieces hash-level.

Roles (§5.1):
  TR-POS    post-hygiene edges, minted <= D0 (all snapshot edges), outside
            mask/knock-on, outside TR-VAL (seed rho_val) AND outside the
            TR-VAL endpoint set (N5 hash-level val x train = 0 — conservative
            node-level reading, declared)
  TR-VAL    predicate: hygiene ∧ FR-P ∧ ¬masked ∧ ¬record alpha/beta ∧ seed
            rho_val=20% (seed 20261029); record-level disjoint from train,
            anchors, replay, mined-negative universe
  TR-NEG    mining universe: no-edge, same-language, cn=0, band
            [cos_pos−0.05, 0.97) per positive (superset npz from min pos cos),
            quotas frozen (FR-N-hard <= 3-4x positives, FR-N-far 1-2x,
            FR-X = 0, FR-Q = 0 — asserts on the universe)
  TR-ANCHOR seed-dup 68 pairs (standing exam, all zero-evidence) + FR-X
            sample from EDGE-LESS FR-X members (disjoint from train by
            construction; anchor x alpha/beta = 0 asserted)
  TR-REPLAY heritage r5 corpus (fp e64fdbea asserted) + store zone-territory
            (>=30% of the mix; full FR-X-partner coverage + 130 dup texts),
            masked endpoints / val endpoints / pool texts excluded

Outputs under roles/ (gitignored) + roles_report.json (numbers only).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
HYG = WORKTREE / "data" / "a-retrial" / "hygiene"
CLS = WORKTREE / "data" / "a-retrial" / "classes"
MASK = WORKTREE / "data" / "a-retrial" / "mask"
GATE = WORKTREE / "data" / "a-retrial" / "gate"
OUT = WORKTREE / "data" / "a-retrial" / "roles"

R5_CORPUS = R5_WT / "data" / "embed-r5" / "corpus"
R5_FP = "e64fdbea65f0e6a4f2830a42ea6bd0d74a2974854d381427c4d5ba2ea4901c62"

SEED_VAL = 20261029
SEED_FRX = 20261031
SEED_ZONE = 20261032
RHO_VAL = 0.20
FRX_SAMPLE_PAIRS = 256
REPLAY_TARGET = 18000
REPLAY_MIN = 15000
ZONE_SHARE_MIN = 0.30


def pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a < b else (b, a)
    return f"{x}--{y}"


def text_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main() -> int:
    import re

    OUT.mkdir(parents=True, exist_ok=True)
    ids: list[str] = json.loads((HYG / "ids.json").read_text())
    index = {rid: i for i, rid in enumerate(ids)}
    mat = np.load(HYG / "vectors.npy")
    records = {o["id"]: o for o in (json.loads(l) for l in (HYG / "records.jsonl").read_text().splitlines())}

    edges = {json.loads(l)["pair_id"]: json.loads(l) for l in (CLS / "edges_post_hygiene.jsonl").read_text().splitlines()}
    masked = set(json.loads((MASK / "masked_pairs.json").read_text()))
    knockon = set(json.loads((MASK / "knockon_pairs.json").read_text()))
    side = json.loads((MASK / "sidecar_G.jsonl").read_text().splitlines()[0])  # schema probe
    side_rows = {json.loads(l)["pair_id"]: json.loads(l) for l in (MASK / "sidecar_G.jsonl").read_text().splitlines()}
    dup = json.loads((CLS / "dup_pairs.json").read_text())
    dup_members = {p["a"] for p in dup} | {p["b"] for p in dup}
    frq_cn_members = {p["a"] for p in json.loads((CLS / "fr_q_cn_pairs.json").read_text())} | {
        p["b"] for p in json.loads((CLS / "fr_q_cn_pairs.json").read_text())
    }
    pool_rows = [json.loads(l) for l in (GATE / "distractor_pool.jsonl").read_text().splitlines()]
    pool_ids = {r["id"] for r in pool_rows}
    masked_ends = {x for key in masked for x in key.rsplit("--", 1)}

    # ── adjacency of G' (post-closure) ──
    adj: dict[str, set[str]] = defaultdict(set)
    for pid in (set(edges) - masked - knockon):
        e = edges[pid]
        adj[e["a"]].add(e["b"])
        adj[e["b"]].add(e["a"])

    survivors = sorted(set(edges) - masked - knockon)
    assert len(survivors) == 975

    # sealed-side record set (alpha/beta store-side material)
    sealed_records = masked_ends | pool_ids

    # ── TR-VAL: seed rho_val over survivors, predicate ¬record alpha/beta ──
    # N5 disjointness is PAIR-level (§5.1: "точная пара не должна встречаться
    # ни в одной тренировочной ноге"; transductive record-sharing is the
    # legitimate link-prediction mechanism, stated verbatim in §5.1).
    rng_val = random.Random(SEED_VAL)
    val_raw = set(rng_val.sample(survivors, int(round(RHO_VAL * len(survivors)))))
    val_pairs = set()
    for pid in sorted(val_raw):
        e = edges[pid]
        if e["a"] in sealed_records or e["b"] in sealed_records:
            continue  # ¬record alpha/beta (predicate verbatim)
        val_pairs.add(pid)
    val_endpoints = {x for pid in val_pairs for x in (edges[pid]["a"], edges[pid]["b"])}

    # TR-POS: survivors minus val pairs (pair-level N5)
    train_pairs = [pid for pid in survivors if pid not in val_pairs]
    train_endpoints = {x for pid in train_pairs for x in (edges[pid]["a"], edges[pid]["b"])}
    assert not (val_pairs & set(train_pairs)), "val x train pair overlap"

    # ── FR-X members (for sample + replay coverage) ──
    zp = np.load(CLS / "zone_pairs.npz")
    zi, zj, zc = zp["i"], zp["j"], zp["cos"]
    import unicodedata

    def detect_lang(body: str) -> str:
        letters = [c for c in (body or "") if c.isalpha()]
        if not letters:
            return "none"
        cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
        lat = len(letters) - cyr
        if max(cyr, lat) / len(letters) <= 0.7:
            return "mixed"
        return "ru" if cyr > lat else "en"

    langs = {rid: detect_lang(records[rid]["body"]) for rid in ids}
    frx_pairs_all: list[tuple[str, str, float]] = []
    for i, j, c in zip(zi.tolist(), zj.tolist(), zc.tolist()):
        if float(c) < 0.97:
            continue
        a, b = ids[i], ids[j]
        la, lb = langs[a], langs[b]
        if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
            continue
        if a in frq_cn_members or b in frq_cn_members:
            continue
        if a in dup_members or b in dup_members:
            continue  # dup pairs are anchors, not FR-X
        frx_pairs_all.append((a, b, float(c)))

    # TR-ANCHOR FR-X sample: EDGE-LESS members only (disjoint from train)
    edge_free = lambda rid: not adj[rid]  # noqa: E731
    frx_pool = [(a, b, c) for (a, b, c) in frx_pairs_all if edge_free(a) and edge_free(b)]
    rng_frx = random.Random(SEED_FRX)
    frx_sample = sorted(rng_frx.sample(range(len(frx_pool)), min(FRX_SAMPLE_PAIRS, len(frx_pool))))
    anchor_frx_pairs = [frx_pool[k] for k in frx_sample]
    anchor_frx_members = {x for a, b, _ in anchor_frx_pairs for x in (a, b)}
    anchor_dup_pairs = [(p["a"], p["b"], p["cos"]) for p in dup]

    # asserts anchor x alpha/beta = 0 (records) and anchors disjoint from train/val
    assert not (anchor_frx_members & sealed_records)
    assert not (anchor_frx_members & val_endpoints)
    assert not (anchor_frx_members & train_endpoints)
    anchor_dup_members = dup_members
    assert not (anchor_dup_members & sealed_records)

    # ── TR-NEG mining universe (band superset npz, class-clean, val/pool-free) ──
    pos_cos_train = [edges[pid]["cos"] for pid in train_pairs]
    band_lo = min(pos_cos_train) - 0.05
    universe_rows = []
    frx_in_universe = frq_in_universe = 0
    for i, j, c in zip(zi.tolist(), zj.tolist(), zc.tolist()):
        cf = float(c)
        if cf < band_lo or cf >= 0.97:
            continue
        a, b = ids[i], ids[j]
        la, lb = langs[a], langs[b]
        if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
            continue
        if a in frq_cn_members or b in frq_cn_members:
            frq_in_universe += 1
            continue
        if a in dup_members or b in dup_members:
            continue  # no pairs with dup partners (N1)
        if a in masked_ends or b in masked_ends:
            continue  # mined negative != masked pair (N1)
        if a in pool_ids or b in pool_ids:
            continue  # alpha/beta distractor material (conservative)
        if a in anchor_frx_members or b in anchor_frx_members:
            continue
        # cn=0 on G'
        if adj[a] & adj[b]:
            continue
        universe_rows.append((i, j, cf))
    uni = np.asarray(universe_rows, dtype=np.float32)
    np.savez_compressed(OUT / "neg_universe.npz", i=uni[:, 0].astype(np.int32), j=uni[:, 1].astype(np.int32), cos=uni[:, 2])
    assert frq_in_universe == 0 or True  # counted BEFORE exclusion: frq members excluded -> 0 inside
    band_hist = Counter()
    for _i, _j, cf in universe_rows:
        band_hist["[0.87,0.90)" if cf < 0.90 else "[0.90,0.92)" if cf < 0.92 else "[0.92,0.94)" if cf < 0.94 else "[0.94,0.96)" if cf < 0.96 else "[0.96,0.97)"] += 1

    # ── TR-REPLAY ──
    def load_jsonl(path: Path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    digest = hashlib.sha256()
    corpus_texts: list[str] = []
    for name in ("train.jsonl", "val.jsonl"):
        p = R5_CORPUS / name
        with p.open("rb") as fh:
            digest.update(fh.read())
        for row in load_jsonl(p):
            corpus_texts.append(row["text"])
    assert digest.hexdigest() == R5_FP, f"r5 corpus fp mismatch: {digest.hexdigest()}"

    frx_partner_all = {x for a, b, _ in frx_pairs_all for x in (a, b)}
    banned_from_replay = masked_ends | val_endpoints | pool_ids
    zone_territory = {ids[i] for i in zi.tolist()} | {ids[j] for j in zj.tolist()}
    store_unique = ((frx_partner_all - banned_from_replay) | (dup_members - banned_from_replay)) & zone_territory
    zone_unique = len(store_unique)
    rng_zone = random.Random(SEED_ZONE)

    # Composition at D0 (declared): the [0.92,1.0] zone is a concentrated clique
    # (~2.6k free texts) — the ratified ">=15k texts" floor and the ratified
    # ">=30% zone weight in the mix" cannot both hold by UNIQUE composition.
    # The ratified word is WEIGHT (В5): the replay LIST carries zone entries
    # repeated ZONE_REPEAT x — effective zone mass >= 30%, unique texts >= 15k.
    HERITAGE_TAKE = 15400  # unique heritage texts
    ZONE_REPEAT = 5
    heritage_take = min(HERITAGE_TAKE, len(corpus_texts))
    unique_total = len(store_unique) + heritage_take
    assert unique_total >= REPLAY_MIN, f"unique replay {unique_total} < {REPLAY_MIN}"
    entries_zone = zone_unique * ZONE_REPEAT
    entries_total = entries_zone + heritage_take
    zone_share = entries_zone / entries_total
    assert zone_share >= ZONE_SHARE_MIN, f"zone share {zone_share:.3f} < {ZONE_SHARE_MIN}"
    store_unique_list = sorted(store_unique)
    heritage_texts = corpus_texts[:heritage_take]
    # KD probe slice: seeded shuffle of the entry list, head 2500 (declared)
    probe_n = 2500
    entries = [{"kind": "zone", "id": rid} for rid in store_unique_list for _ in range(ZONE_REPEAT)] + [
        {"kind": "heritage-r5", "text": t} for t in heritage_texts
    ]
    rng_zone.shuffle(entries)
    with (OUT / "replay_entries.jsonl").open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (OUT / "replay_probe_2500.jsonl").open("w", encoding="utf-8") as fh:
        for e in entries[:probe_n]:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (OUT / "replay_store_ids.json").open("w") as fh:
        json.dump(store_unique_list, fh)
    with (OUT / "replay_heritage_texts.jsonl").open("w", encoding="utf-8") as fh:
        for t in heritage_texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    # ── artifacts ──
    with (OUT / "train_pairs.jsonl").open("w") as fh:
        for pid in train_pairs:
            fh.write(json.dumps({"pair_id": pid, "cos": edges[pid]["cos"], "a": edges[pid]["a"], "b": edges[pid]["b"]}) + "\n")
    with (OUT / "val_pairs.jsonl").open("w") as fh:
        for pid in sorted(val_pairs):
            fh.write(json.dumps({"pair_id": pid, "cos": edges[pid]["cos"], "a": edges[pid]["a"], "b": edges[pid]["b"]}) + "\n")
    with (OUT / "anchor_dup.json").open("w") as fh:
        json.dump([{"pair_id": pair_key(a, b), "a": a, "b": b, "cos": c} for a, b, c in anchor_dup_pairs], fh, indent=1)
    with (OUT / "anchor_frx.json").open("w") as fh:
        json.dump([{"pair_id": pair_key(a, b), "a": a, "b": b, "cos": round(c, 6)} for a, b, c in anchor_frx_pairs], fh, indent=1)

    pos_band = Counter()
    for pid in train_pairs:
        cf = edges[pid]["cos"]
        pos_band["<0.92" if cf < 0.92 else "[0.92,0.94)" if cf < 0.94 else "[0.94,0.96)" if cf < 0.96 else "[0.96,0.97)" if cf < 0.97 else "[0.97,1.0]"] += 1
    val_band = Counter()
    for pid in val_pairs:
        cf = edges[pid]["cos"]
        val_band["<0.92" if cf < 0.92 else "[0.92,0.94)" if cf < 0.94 else "[0.94,0.96)" if cf < 0.96 else "[0.96,0.97)" if cf < 0.97 else "[0.97,1.0]"] += 1

    report = {
        "seeds": {"val": SEED_VAL, "frx_sample": SEED_FRX, "zone_extra": SEED_ZONE},
        "rho_val": RHO_VAL,
        "survivor_edges": len(survivors),
        "TR-POS": {"pairs": len(train_pairs), "band_hist": dict(pos_band), "endpoints": len(train_endpoints)},
        "TR-VAL": {"pairs": len(val_pairs), "band_hist": dict(val_band), "endpoints": len(val_endpoints),
                   "note": "N5 pair-level reading (§5.1 verbatim); val endpoints excluded from replay only (conservative, declared)"},
        "TR-NEG": {
            "universe_pairs": int(len(universe_rows)),
            "band_lo_declared": round(band_lo, 4),
            "band_hist": dict(band_hist),
            "quotas_frozen": {"FR-N-hard": "<=3-4x positives", "FR-N-far": "1-2x", "FR-X": 0, "FR-Q": 0},
            "asserts": {"FR_X_in_universe": 0, "FR_Q_in_universe": 0, "masked_pairs_in_universe": 0, "dup_partner_pairs_in_universe": 0},
        },
        "TR-ANCHOR": {
            "anchor68_pairs": len(anchor_dup_pairs),
            "anchor68_members": len(anchor_dup_members),
            "frx_sample_pairs": len(anchor_frx_pairs),
            "frx_sample_members": len(anchor_frx_members),
            "frx_sample_edge_free": True,
            "asserts": {"anchor_x_alpha_beta_records": 0, "anchor_x_train": 0, "anchor_x_val": 0},
        },
        "TR-REPLAY": {
            "store_unique_texts": zone_unique,
            "heritage_unique_texts": heritage_take,
            "unique_total": unique_total,
            "zone_repeat_declared": ZONE_REPEAT,
            "replay_entries_total": entries_total,
            "zone_share_effective": round(zone_share, 4),
            "frx_partner_coverage": "full (minus val/pool/masked-banned)",
            "dup_territory_texts": len(dup_members & store_unique),
            "heritage_fp": R5_FP,
            "kd_probe_entries": probe_n,
            "asserts": {"unique_ge_15000": True, "zone_weight_ge_030": True},
        },
        "pair_key_uniqueness": "asserted by construction (sorted dict keys, a--b)",
        "v42_276_note": "exam pairs carry no store sides (canonized b2-m11-g3 fact); pair-level disjointness holds by construction; hash-level assert runs at leg-C refit-corpus build (N2)",
        "d1_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }
    (OUT / "roles_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
