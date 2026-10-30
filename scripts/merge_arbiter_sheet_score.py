#!/usr/bin/env python3
"""Merge-arbiter spot-sheet scoring infrastructure (ADDENDUM-1, MLC 2026-10-29).

Subcommands:
  prematch     machine pre-checks; emits booleans/counts/fingerprints ONLY
  render       build the blind verdictor render (neutral row ids, surface
               fields only) + TL-internal mapping; pins sha256 of both
  consolidate  (after the three verdict files exist) join verdicts, run
               mechanical arbitration, consensus, vs-construction QA ladder;
               write committee-scored-v1.json + report

Seal discipline: holdout.jsonl is opened ONLY here, machine-internal; the
script never prints holdout or real-record content; outputs are booleans,
counts and fingerprints (data-contract §1/§6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
DATA = WT / "data" / "merge-arbiter"
SHEET_DIR = DATA / "spot-sheet"
CORPUS = DATA / "corpus-v1"
INTERNAL = SHEET_DIR / "internal"

SEALED_SHEET_FP = "e45e72a4a52587087bca2ae25ae34a666bcb340a635ac076b0a38a22a63b6824"
EXPECTED_STRATA = {"real": 20, "keep_CH-3": 8, "abstain_CH-2_CH-4": 12}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sheet_rows() -> list[dict]:
    """Parse the frozen owner sheet table into row dicts."""
    lines = (
        (SHEET_DIR / "owner-spot-sheet-v1.md").read_text(encoding="utf-8").splitlines()
    )
    rows = []
    for ln in lines:
        if ln.startswith("| ") and not ln.startswith("| #"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            # separator rows have no digit in the first cell; skeletons may
            # legitimately contain '---' (markdown frontmatter of real records)
            if len(cells) >= 6 and cells[0].isdigit():
                rows.append(
                    {
                        "row": int(cells[0]),
                        "cluster": cells[1],
                        "a": cells[2],
                        "b": cells[3],
                        "c": cells[4],
                    }
                )
    return rows


def selection() -> dict:
    return json.loads((SHEET_DIR / "sheet-selection.json").read_text(encoding="utf-8"))


def surface(rec: dict) -> dict:
    return {
        "title": rec.get("title") or "",
        "body": rec.get("body") or "",
        "tags": list(rec.get("tags") or []),
        "record_type": rec.get("record_type"),
        "language": rec.get("language"),
    }


def cmd_premarket() -> int:
    checks: dict[str, object] = {}
    checks["sheet_fp_match"] = (
        sha256_file(SHEET_DIR / "owner-spot-sheet-v1.md") == SEALED_SHEET_FP
    )
    checks["selection_sha256"] = sha256_file(SHEET_DIR / "sheet-selection.json")

    rows = sheet_rows()
    sel = selection()
    checks["sheet_rows"] = len(rows)
    checks["selection_real"] = len(sel["real"])
    checks["selection_synthetic"] = len(sel["synthetic"])
    checks["stratum_counts_early_ok"] = (
        len(sel["real"]) == EXPECTED_STRATA["real"] and len(sel["synthetic"]) == 20
    )

    train = load_jsonl(
        CORPUS / "train.jsonl"
    )  # cluster rows: members + label + declaration
    holdout = load_jsonl(CORPUS / "holdout.jsonl")  # machine-internal only
    train_by_id = {m["id"]: m for cl in train for m in cl["members"]}
    train_ids = set(train_by_id)
    holdout_ids = {r["id"] for cl in holdout for r in cl["members"]}
    syn_members = [m for item in sel["synthetic"] for m in item["member_ids"]]
    checks["syn_members"] = len(syn_members)
    checks["syn_members_subset_train"] = set(syn_members) <= train_ids
    checks["syn_members_intersect_holdout"] = len(
        set(syn_members) & holdout_ids
    )  # must be 0

    # stratum composition of sheet synthetic clusters — aggregate counts ONLY
    syn_by_cluster = {
        item["sheet_cluster"]: item["member_ids"] for item in sel["synthetic"]
    }
    class_counts: dict[str, int] = {}
    for cl in train:
        if cl["cluster_id"] in syn_by_cluster:
            class_counts[cl["label"]["type"]] = (
                class_counts.get(cl["label"]["type"], 0) + 1
            )
    checks["sheet_class_counts"] = (
        class_counts  # expect {keep: 8, abstain: 12} = CH-3 8 + CH-2/CH-4 12
    )
    checks["stratum_counts_ok"] = (
        len(sel["real"]) == EXPECTED_STRATA["real"]
        and class_counts
        == {
            "keep": EXPECTED_STRATA["keep_CH-3"],
            "abstain": EXPECTED_STRATA["abstain_CH-2_CH-4"],
        }
        and len(syn_by_cluster) == 20
    )

    anchor_ids = _real_record_ids()
    store_ids = [s for item in sel["real"] for s in item["store_ids"]]
    checks["real_store_ids"] = len(store_ids)
    checks["real_store_ids_subset_anchor68"] = set(store_ids) <= anchor_ids

    # predicate completeness: surface fields present for every sheet candidate
    incomplete = 0
    for item in sel["synthetic"]:
        for mid in item["member_ids"]:
            rec = train_by_id.get(mid)
            if rec is None or not (rec.get("body") or "").strip():
                incomplete += 1
    checks["syn_surface_incomplete"] = incomplete
    checks["real_snapshot_available"] = _snapshot_ok()

    ok = (
        checks["sheet_fp_match"]
        and checks["stratum_counts_ok"]
        and checks["syn_members_subset_train"]
        and checks["syn_members_intersect_holdout"] == 0
        and checks["syn_surface_incomplete"] == 0
        and checks["real_snapshot_available"]
        and checks["real_store_ids_subset_anchor68"]
    )
    print(json.dumps({"prematch_ok": ok, **checks}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


_REAL_CACHE: dict | None = None


def _snapshot_ok() -> bool:
    snap = DATA / "snapshot" / "probe-20261028"
    if not snap.exists():
        return False
    return (
        any(snap.glob("*.db"))
        or any(snap.glob("*.sqlite*"))
        or any(p.is_file() for p in snap.iterdir())
    )


def _real_record_ids() -> set[str]:
    """Record ids of the pinned real strata (anchor-68 census; ids only)."""
    global _REAL_CACHE
    if _REAL_CACHE is not None:
        return _REAL_CACHE
    rs = json.loads((CORPUS / "real-strata.json").read_text(encoding="utf-8"))
    ids: set[str] = set()
    for cl in rs.get("clusters", []):
        ids.update(cl.get("ids", []))
    _REAL_CACHE = ids
    return ids


def cmd_render() -> int:
    INTERNAL.mkdir(parents=True, exist_ok=True)
    rows = sheet_rows()
    sel = selection()
    train_by_id = {
        m["id"]: m for cl in load_jsonl(CORPUS / "train.jsonl") for m in cl["members"]
    }
    real = _load_real_records({s for item in sel["real"] for s in item["store_ids"]})

    mapping = {}
    render_parts = [
        "# Слепой рендер листа — скоринг комитета-серебра (ADDENDUM-1)",
        "",
        "Инструкция: для каждого кластера выберите, КАКАЯ запись должна остаться:",
        "A / B / C — или NO-SAFE (схлопывать нельзя).",
        "Часть кластеров — реальные записи стора (попарно байт-идентичны, отличаются",
        "только служебными полями); часть — синтетические. Тип каждой строки не",
        "раскрывается. На реальных парах валиден любой ответ A/B (ничего не теряется);",
        "NO-SAFE на них = стоп-сигнал. На синтетических выбирайте по содержанию",
        "(какая запись несёт больше ключевых фактов: числа, даты, версии, имена,",
        "суммы — либо схлопывать нельзя).",
        "",
        "Ответ — по одной строке на кластер: `N##: выбор (A|B|C|NO-SAFE) — основание`.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        nid = f"N{i:02d}"
        cid = row["cluster"]
        item = next((x for x in sel["synthetic"] if x["sheet_cluster"] == cid), None)
        if item:
            cands = [surface(train_by_id[mid]) for mid in item["member_ids"]]
            mapping[nid] = {
                "kind_ref": "synthetic",
                "cluster": cid,
                "member_ids": item["member_ids"],
            }
        else:
            ritem = next((x for x in sel["real"] if x["sheet_cluster"] == cid), None)
            cands = [real[sid] for sid in ritem["store_ids"]]
            mapping[nid] = {
                "kind_ref": "real",
                "cluster": cid,
                "store_ids": ritem["store_ids"],
            }
        render_parts.append(f"## {nid}")
        for label, cand in zip("ABC", cands):
            render_parts.append(f"### Кандидат {label}")
            render_parts.append(f"- title: {cand['title']}")
            render_parts.append(f"- tags: {cand['tags']}")
            render_parts.append(
                f"- record_type: {cand['record_type']} · language: {cand['language']}"
            )
            render_parts.append(f"- body:\n\n{cand['body']}")
            render_parts.append("")
    render_path = SHEET_DIR / "blind-render-v1.md"
    render_path.write_text("\n".join(render_parts), encoding="utf-8")
    (INTERNAL / "render-mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # 1:1 assert vs frozen sheet: row count, candidate counts, skeleton prefixes
    sheet_ok = len(rows) == 40 and len(mapping) == 40
    print(
        json.dumps(
            {
                "render": str(render_path),
                "rows": len(rows),
                "assert_1_to_1_row_count": sheet_ok,
                "render_sha256": sha256_file(render_path),
                "mapping_sha256": sha256_file(INTERNAL / "render-mapping.json"),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0 if sheet_ok else 1


def _load_real_records(store_ids: set[str]) -> dict[str, dict]:
    import sqlite3

    db = DATA / "snapshot" / "probe-20261028" / "vesma.db"
    out: dict[str, dict] = {}
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.execute("PRAGMA query_only=ON")
        try:
            q = (
                "SELECT id, title, content, tags, memory_type, metadata "
                "FROM memories WHERE id IN (%s)" % ",".join("?" * len(store_ids))
            )
            for rid, title, content, tags, mtype, meta in con.execute(
                q, list(store_ids)
            ):
                lang = None
                try:
                    lang = (json.loads(meta) or {}).get("canon", {}).get("language")
                except Exception:
                    pass
                out[rid] = surface(
                    {
                        "title": title,
                        "body": content,
                        "tags": tags or [],
                        "record_type": mtype,
                        "language": lang,
                    }
                )
        finally:
            con.close()
    missing = store_ids - set(out)
    if missing:
        print(
            f"WARN real records missing from snapshot: {len(missing)}", file=sys.stderr
        )
    return out


def cmd_consolidate() -> int:
    """Executor-only (TL): three verdict files must already exist.

    Reveals by-construction labels AFTER the three scorings are fixed
    (ADDENDUM-1 §4). Emits committee-scored-v1.json + QA ladder report.
    """
    vdir = SHEET_DIR / "verdicts"
    files = {
        name: json.loads((vdir / f"{name}-v1.json").read_text(encoding="utf-8"))
        for name in ("llm-assistant", "ml-engineer", "tech-lead")
    }
    mapping = json.loads((INTERNAL / "render-mapping.json").read_text(encoding="utf-8"))
    render_sha = sha256_file(SHEET_DIR / "blind-render-v1.md")
    for name, v in files.items():
        assert v["render_sha256"] == render_sha, f"{name}: render sha mismatch"
        assert len(v["rows"]) == 40, f"{name}: rows != 40"

    train = load_jsonl(CORPUS / "train.jsonl")
    cl_by_id = {cl["cluster_id"]: cl for cl in train}
    rows = sheet_rows()

    # by-construction: CH-3 rows -> keep member id; CH-2/CH-4 -> forbid;
    # real rows -> degenerate (any member)
    truth = {}
    for row in rows:
        nid = f"N{row['row']:02d}"
        cid = row["cluster"]
        if nid not in mapping:
            truth[nid] = {"class": "unknown", "row": row["row"]}
            continue
        m = mapping[nid]
        if m["kind_ref"] == "real":
            truth[nid] = {"class": "REAL-CH0", "row": row["row"], "acceptable": "any"}
        else:
            cl = cl_by_id[cid]
            lab = cl["label"]
            if lab["type"] == "keep":
                member_ids = m["member_ids"]
                keep_idx = member_ids.index(lab["member"])
                truth[nid] = {
                    "class": "CH-3",
                    "row": row["row"],
                    "keep_letter": "ABC"[keep_idx],
                    "keep_member": lab["member"],
                    "member_ids": member_ids,
                    "cluster_id": cid,
                }
            else:
                truth[nid] = {
                    "class": "abstain",
                    "row": row["row"],
                    "acceptable": "abstain",
                }

    order = [f"N{i:02d}" for i in range(1, 41)]
    per_row = []
    ind_hits = {n: 0 for n in files}
    ch3 = [n for n in order if truth[n]["class"] == "CH-3"]
    abst = [n for n in order if truth[n]["class"] == "abstain"]
    real = [n for n in order if truth[n]["class"] == "REAL-CH0"]
    # declarations for the mechanical arbitrator (ADDENDUM-1 §2: predicate
    # arbitration PRECEDES majority; declaration ⊆-check is the ratified
    # mechanical form for synthetic CH-3, A-silver §2.1)
    decl = {}
    for cl in train:
        decl[cl["cluster_id"]] = cl.get("declaration", {}).get("facts", {})
    for nid in order:
        verdicts = {
            name: next(r for r in files[name]["rows"] if r["row"] == nid)
            for name in files
        }
        t = truth[nid]
        rec = {"row": nid, "truth_class": t["class"], "verdicts": {}}
        for name, v in verdicts.items():
            rec["verdicts"][name] = {"choice": v["choice"], "basis": v["basis"][:200]}
        if t["class"] == "CH-3":
            for name, v in verdicts.items():
                if v["choice"] == t["keep_letter"]:
                    ind_hits[name] += 1
            raw_agree = all(v["choice"] == t["keep_letter"] for v in verdicts.values())
            cons = t["keep_letter"] if raw_agree else None
            if not raw_agree:
                counts: dict[str, int] = {}
                for v in verdicts.values():
                    counts[v["choice"]] = counts.get(v["choice"], 0) + 1
                top = max(counts, key=lambda c: counts[c])
                # mechanical arbitration FIRST (ADDENDUM-1 §2): declaration
                # ⊆-check on the generator's declared fact slots; resolves
                # only when strictly one member carries extra declared facts
                facts = decl.get(t["cluster_id"], {})
                mids = t["member_ids"]
                fsets = [set(facts.get(mid, [])) for mid in mids]
                sup_i = None
                for i, fi in enumerate(fsets):
                    if all(fi >= fj for j, fj in enumerate(fsets) if j != i) and all(
                        len(fi) > len(fj) for j, fj in enumerate(fsets) if j != i
                    ):
                        sup_i = i
                if sup_i is not None:
                    arb_letter = "ABC"[sup_i]
                    rec["arbitration"] = {
                        "resolved_by": "declaration-subset-check",
                        "letter": arb_letter,
                        "matches_construction": arb_letter == t["keep_letter"],
                    }
                    cons = arb_letter
                elif counts[top] >= 2:
                    cons = top
                    rec["majority_resolved"] = True
            rec["consensus"] = cons
            rec["raw_unanimous_vs_truth"] = raw_agree
        else:
            safes = [name for name, v in verdicts.items() if v["choice"] == "NO-SAFE"]
            rec["no_safe_members"] = len(safes)
            rec["consensus"] = "NO-SAFE" if len(safes) >= 2 else "FOLD-ELIGIBLE"
            if t["class"] == "abstain":
                rec["raw_unanimous_vs_truth"] = len(safes) == 3
            else:
                rec["raw_unanimous_vs_truth"] = len(safes) == 0
        per_row.append(rec)

    # QA ladder (ADDENDUM-1 §4)
    ladder = {
        "individual_CH3_hits": {
            n: {"hits": h, "of": len(ch3)} for n, h in ind_hits.items()
        },
        "individual_pass": all(h >= 7 for h in ind_hits.values()),
        "consensus_raw_unanimous_CH3": sum(
            1
            for n in ch3
            if next(r for r in per_row if r["row"] == n)["raw_unanimous_vs_truth"]
        ),
        "abstain_unanimous": sum(
            1
            for n in abst
            if next(r for r in per_row if r["row"] == n)["raw_unanimous_vs_truth"]
        ),
        "abstain_of": len(abst),
        "real_consensus_nosafe": sum(
            1
            for n in real
            if next(r for r in per_row if r["row"] == n)["no_safe_members"] >= 2
        ),
        "real_single_nosafe": [
            n
            for n in real
            if next(r for r in per_row if r["row"] == n)["no_safe_members"] == 1
        ],
        "ch3_majority_resolved": [
            n
            for n in ch3
            if next(r for r in per_row if r["row"] == n).get("majority_resolved")
        ],
        "tight_path": None,
    }
    cons_hits = ladder["consensus_raw_unanimous_CH3"]
    ladder["ladder_verdict"] = (
        "PASS"
        if ladder["individual_pass"]
        and cons_hits == 8
        and ladder["abstain_unanimous"] == ladder["abstain_of"]
        and ladder["real_consensus_nosafe"] == 0
        else "REVIEW"
        if ladder["individual_pass"]
        and cons_hits == 7
        and ladder["abstain_unanimous"] == ladder["abstain_of"]
        and ladder["real_consensus_nosafe"] == 0
        else "STOP"
    )
    if cons_hits == 7:
        ladder["tight_path"] = (
            "model must take 7/7 remaining CH-3 cells; any second miss = DECLINE"
        )

    out = {
        "sheet_fp": SEALED_SHEET_FP,
        "render_sha256": render_sha,
        "selection_sha256": checks_sel_sha(),
        "verdict_files": {n: str(vdir / f"{n}-v1.json") for n in files},
        "rows": per_row,
        "qa_ladder": ladder,
    }
    outp = SHEET_DIR / "committee-scored-v1.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                "committee_scored": str(outp),
                "sha256": sha256_file(outp),
                "qa_ladder": ladder,
                "truth_classes": {
                    "CH-3": len(ch3),
                    "abstain": len(abst),
                    "REAL-CH0": len(real),
                },
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


def checks_sel_sha() -> str:
    return sha256_file(SHEET_DIR / "sheet-selection.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prematch")
    sub.add_parser("render")
    sub.add_parser("consolidate")
    args = ap.parse_args()
    return {
        "prematch": cmd_premarket,
        "render": cmd_render,
        "consolidate": cmd_consolidate,
    }[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
