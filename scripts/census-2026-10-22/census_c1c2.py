#!/usr/bin/env python3
"""B2 M1.1 evidence census C1/C2 — deterministic silver over the sealed pool snapshot.

Read-only: pool_candidates.jsonl (gitignored data/, fp-verified 87b3d4c2).
No store access. No LLM. Output: aggregates only (+ <=1-line anonymized examples).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

POOL = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/b2-graph-corpus/data/b2-graph-corpus/pool_candidates.jsonl")

EVIDENCE_STRATA = ("edge-direct", "edge-common", "mint-top")


def norm1(s: str | None) -> str:
    """Pinned silver norm: lower + collapse whitespace."""
    return " ".join((s or "").lower().split())


def norm2(s: str | None) -> str:
    """Sensitivity norm: additionally folds punctuation (policy §8.2 cosmetic)."""
    s = re.sub(r"[^\w\s]", " ", (s or "").lower(), flags=re.UNICODE)
    return " ".join(s.split())


def detect_lang(body: str | None) -> str:
    """Replica of b2_graph_corpus_v44r2.detect_lang (addendum §A(i))."""
    letters = [c for c in (body or "") if c.isalpha()]
    if not letters:
        return "none"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    lat = len(letters) - cyr
    if max(cyr, lat) / len(letters) <= 0.7:
        return "mixed"
    return "ru" if cyr > lat else "en"


def token_razor_numeric(ba: str, bb: str) -> bool:
    """Deterministic T3 subset: token seqs equal except positions where BOTH
    sides carry digit-bearing tokens (numeric key-fact edit proxy)."""
    ta = norm1(ba).split()
    tb = norm1(bb).split()
    if len(ta) != len(ta):
        return False
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


def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def anon(body: str, n: int = 60) -> str:
    """Anonymized skeleton: keep only script shape + digits pattern + length."""
    letters = [c for c in body if c.isalpha()]
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    script = "cyr" if cyr > len(letters) / 2 else "lat"
    digits = "".join("d" if ch.isdigit() else ("w" if ch.isalpha() else ".") for ch in body[:n])
    return f"<{script} len={len(body)} shape={digits}...>"


def classify(r: dict) -> dict:
    rec, cand = r["record"], r["candidate"]
    ba, bb = rec.get("body") or "", cand.get("body") or ""
    ta, tb = rec.get("title") or "", cand.get("title") or ""
    tags_a, tags_b = set(rec.get("tags") or []), set(cand.get("tags") or [])
    type_a, type_b = rec.get("record_type"), cand.get("record_type")
    lang_a, lang_b = rec.get("language"), cand.get("language")
    created_a, created_b = rec.get("created_at"), cand.get("created_at")

    out: dict = {}
    out["pair_id"] = r["pair_id"]
    out["stratum"] = r["stratum"]
    out["similarity"] = r["similarity"]
    out["sifter_verdict"] = (r.get("sifter") or {}).get("verdict")
    out["sifter_crosslang"] = (r.get("sifter") or {}).get("crosslang")
    out["sens_body_only"] = False
    out["sens_norm2"] = False
    out["punct_cosmetic"] = False
    out["body_eq1"] = False
    out["envelope_delta"] = False

    # quarantine (§A(i))
    la, lb = detect_lang(ba), detect_lang(bb)
    out["lang_a"], out["lang_b"] = la, lb
    if la in ("ru", "en") and lb in ("ru", "en") and la != lb:
        out["cls"] = "quarantine"
        return out
    out["undetectable"] = la == "none" or lb == "none"
    out["mixed_script"] = la == "mixed" or lb == "mixed"

    body_eq1 = norm1(ba) == norm1(bb)
    title_eq1 = norm1(ta) == norm1(tb)
    body_eq2 = norm2(ba) == norm2(bb)
    title_eq2 = norm2(ta) == norm2(tb)
    tags_eq = tags_a == tags_b
    type_eq = type_a == type_b
    lang_eq = lang_a == lang_b

    out["body_eq1"] = body_eq1
    out["title_eq1"] = title_eq1
    out["body_eq2"] = body_eq2
    out["tags_eq"] = tags_eq
    out["type_eq"] = type_eq
    out["lang_eq"] = lang_eq
    out["envelope_delta"] = created_a != created_b

    # primary silver-dup: body+title norm equality, metadata equal, envelope free
    s_dup = body_eq1 and title_eq1 and tags_eq and type_eq and lang_eq
    # sensitivities
    s_dup_body_only = body_eq1
    s_dup_norm2 = body_eq2 and title_eq2 and tags_eq and type_eq and lang_eq
    # T2 metadata-only (policy v1.1): identical body+title, metadata differs
    t2 = body_eq1 and title_eq1 and not (tags_eq and type_eq and lang_eq)
    # T3 numeric razor subset
    t3_num = (not body_eq1) and token_razor_numeric(ba, bb)

    if s_dup:
        out["cls"] = "silver-dup"
    elif t2:
        out["cls"] = "notdup-metadata-T2"
    elif t3_num:
        out["cls"] = "notdup-razor-numeric-T3"
    else:
        out["cls"] = "disputed"
    out["sens_body_only"] = s_dup_body_only and not s_dup and out["cls"] != "quarantine"
    out["sens_norm2"] = s_dup_norm2 and not s_dup and out["cls"] != "quarantine"
    # punctuation-only cosmetic candidates (norm2 dup, norm1 not)
    out["punct_cosmetic"] = (not body_eq1) and body_eq2
    return out


def graph_feats(r: dict) -> dict:
    edge = r.get("edge")
    return {
        "edge_ab": 1 if edge else 0,
        "edge_kind": (edge or {}).get("kind"),
        "edge_prov": (edge or {}).get("provenance"),
        "edge_kind_supersedes": 1 if (edge or {}).get("kind") == "supersedes" else 0,
        "common_neighbors": r.get("common_neighbors") or 0,
        "neighbor_jaccard": r.get("neighbor_jaccard") or 0.0,
        # cos_to_common_max: not in snapshot; 0 is PROVEN when cn==0
        "cos_to_common_max_known": (r.get("common_neighbors") or 0) == 0,
    }


def main() -> None:
    rows = [json.loads(line) for line in POOL.open()]
    assert len(rows) == 8654
    strata = Counter(r["stratum"] for r in rows)
    print("pool strata:", dict(strata))

    ev = [r for r in rows if r["stratum"] in EVIDENCE_STRATA]
    ctrl = [r for r in rows if r["stratum"] == "control"]
    print(f"evidence population on disk: {len(ev)} (edge-direct {strata['edge-direct']} / "
          f"edge-common {strata['edge-common']} / mint-top {strata['mint-top']})")

    # sanity: stratum vs edge field consistency
    mism = sum(1 for r in ev if bool(r.get("edge")) != (r["stratum"] == "edge-direct"))
    print("stratum-vs-edge-field mismatches:", mism)

    results = [classify(r) for r in ev]
    ctrl_results = [classify(r) for r in ctrl]

    def tally(rs: list[dict]) -> dict:
        return dict(Counter(x["cls"] for x in rs))

    print("\nC1 evidence-strata silver:", tally(results))
    print("C1 control (context):", tally(ctrl_results))

    # quarantine + undetectable/mixed breakdown
    print("quarantine crosslang:", sum(1 for x in results if x["cls"] == "quarantine"),
          "undetectable:", sum(1 for x in results if x.get("undetectable")),
          "mixed:", sum(1 for x in results if x.get("mixed_script")))
    print("sifter.crosslang flags:", sum(1 for x in results if x["sifter_crosslang"]))
    # cross-lang agreement with sifter flags
    myq = {x["pair_id"] for x in results if x["cls"] == "quarantine"}
    sfq = {x["pair_id"] for x in results if x["sifter_crosslang"]}
    print("my-q vs sifter-q: only-mine", len(myq - sfq), "only-sifter", len(sfq - myq))

    per_stratum: dict[str, Counter] = defaultdict(Counter)
    for x in results:
        per_stratum[x["stratum"]][x["cls"]] += 1
    for s in EVIDENCE_STRATA:
        print(f"  {s}: {dict(per_stratum[s])}")

    # sensitivities
    print("sens: body-only dup (norm1, any title/meta):",
          sum(1 for x in results if x["sens_body_only"]))
    print("sens: norm2 dup (punct-folded):", sum(1 for x in results if x["sens_norm2"]))
    print("punct-cosmetic candidates (norm2 eq, norm1 neq):",
          sum(1 for x in results if x["punct_cosmetic"]))
    print("envelope-delta among body_eq1:", sum(1 for x in results if x["body_eq1"] and x["envelope_delta"]))

    # C2: graph features on silver-dup + overall evidence
    dups = [x for x in results if x["cls"] == "silver-dup"]
    row_by_id = {r["pair_id"]: r for r in ev}
    print("\nC2 silver-dup pairs:", len(dups))
    for x in dups[:10]:
        gf = graph_feats(row_by_id[x["pair_id"]])
        print("  DUP", x["pair_id"][:13], x["stratum"], gf["edge_ab"], gf["common_neighbors"],
              gf["neighbor_jaccard"], x["similarity"], anon(row_by_id[x["pair_id"]]["record"]["body"]))

    # graph activity over all evidence rows (context for C2/C4)
    act = {"D2_frozen_nonzero": 0, "D3_edge_only": 0}
    dist_cn = Counter()
    dist_nj_pos = 0
    for r in ev:
        gf = graph_feats(r)
        dist_cn[gf["common_neighbors"]] += 1
        if gf["neighbor_jaccard"] > 0:
            dist_nj_pos += 1
        nonz = gf["edge_ab"] or gf["edge_kind_supersedes"] or gf["common_neighbors"] > 0 or gf["neighbor_jaccard"] > 0
        if nonz:
            act["D2_frozen_nonzero"] += 1
        if gf["edge_ab"]:
            act["D3_edge_only"] += 1
    print("\ngraph-activity over evidence strata:", act, "(cos_to_common_max: proven-0 for cn==0 rows)")
    print("common_neighbors dist (top):", dist_cn.most_common(8))
    print("neighbor_jaccard>0 rows:", dist_nj_pos)

    # per-stratum D2
    for s in EVIDENCE_STRATA:
        rs = [r for r in ev if r["stratum"] == s]
        d2 = sum(
            1
            for r in rs
            if graph_feats(r)["edge_ab"]
            or graph_feats(r)["edge_kind_supersedes"]
            or (r.get("common_neighbors") or 0) > 0
            or (r.get("neighbor_jaccard") or 0) > 0
        )
        cn_pos = sum(1 for r in rs if (r.get("common_neighbors") or 0) > 0)
        nj_pos = sum(1 for r in rs if (r.get("neighbor_jaccard") or 0) > 0)
        print(f"  {s}: n={len(rs)} D2_nonzero={d2} cn>0={cn_pos} nj>0={nj_pos}")

    # sifter cross-tab vs my silver
    ct = Counter((x["sifter_verdict"], x["cls"]) for x in results)
    print("\nsifter x silver cross-tab:")
    for k, v in sorted(ct.items(), key=lambda kv: -kv[1]):
        print("  ", k, v)

    # similarity ranges per stratum
    for s in EVIDENCE_STRATA:
        sims = sorted(r["similarity"] for r in ev if r["stratum"] == s)
        print(f"  cos {s}: min={sims[0]} med={sims[len(sims)//2]} max={sims[-1]}")

    # anonymized examples of disputed razor-numeric + T2 for the doc
    for cls in ("notdup-metadata-T2", "notdup-razor-numeric-T3"):
        ex = [x for x in results if x["cls"] == cls][:3]
        print(f"\nexamples {cls}: {len([x for x in results if x['cls']==cls])} total")
        for x in ex:
            r = row_by_id[x["pair_id"]]
            print("   ", x["stratum"], x["similarity"],
                  "tags_eq", x["tags_eq"], "type_eq", x["type_eq"], "lang_eq", x["lang_eq"],
                  anon(r["record"]["body"]), "||", anon(r["candidate"]["body"]))

    # dump full per-pair census (numbers/flags only, no content) for the doc
    out = {
        "population": {"pool_total": len(rows), "evidence": len(ev), "control": len(ctrl),
                       "strata": dict(strata)},
        "c1_evidence": tally(results),
        "c1_control": tally(ctrl_results),
        "per_stratum": {s: dict(per_stratum[s]) for s in EVIDENCE_STRATA},
        "sens_body_only_dup": sum(1 for x in results if x["sens_body_only"]),
        "sens_norm2_dup": sum(1 for x in results if x["sens_norm2"]),
        "punct_cosmetic": sum(1 for x in results if x["punct_cosmetic"]),
        "quarantine": sum(1 for x in results if x["cls"] == "quarantine"),
        "graph_activity": act,
        "common_neighbors_dist": dict(dist_cn),
        "neighbor_jaccard_pos": dist_nj_pos,
    }
    Path("/tmp/b2_census/c1c2_numbers.json").write_text(json.dumps(out, indent=1))
    print("\nwrote /tmp/b2_census/c1c2_numbers.json")


if __name__ == "__main__":
    sys.exit(main())
