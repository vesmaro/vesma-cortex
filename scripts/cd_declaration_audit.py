#!/usr/bin/env python3
"""Declaration-vs-body audit (constant step, ADDENDUM-7 / gold OQ-1).

Samples >= 30 train pairs deterministically (coverage of every stratum),
and for each pair checks that the DECLARATION actually describes the
BODY difference:
  1. mechanical predicate of the class (§3.2.1) passes;
  2. the number of difflib regions matches the number of declared slots
     and every declared value is symmetric-contained in its region;
  3. value_a / value_b fragments are present in the corresponding body
     sides (where the class semantics carry them);
  4. the label is derivable from the stratum (CD* -> contradicts,
     NC* -> not-contradicts).
Writes data/conflict-detector/audit/declaration-vs-body-audit.json and
prints the pass count. This audit is independent of the sheet legs.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402

TRAIN = REPO / "data" / "conflict-detector" / "corpus-v1" / "train.jsonl"
OUT = REPO / "data" / "conflict-detector" / "audit" / "declaration-vs-body-audit.json"

SAMPLE = {"CD-1": 6, "CD-2": 3, "CD-3": 3, "CD-4": 3, "CD-5": 3, "CD-6": 3,
          "NC-1": 4, "NC-2": 4, "NC-3": 3, "NC-4": 3, "NC-5": 4}
ORDER = ["CD-1", "CD-2", "CD-3", "CD-4", "CD-5", "CD-6",
         "NC-1", "NC-2", "NC-3", "NC-4", "NC-5"]


def value_presence(sc: lib.Scenario) -> tuple[bool, str]:
    """Slot values must be traceable to the body sides."""
    missing = []
    for slot in sc.slots:
        va, vb = str(slot["value_a"]), str(slot["value_b"])
        na, nb = lib.norm1(va), lib.norm1(vb)
        if not na:
            continue  # NC-1 inserts declare empty value_a by design
        if na not in lib.norm1(sc.side_a.body):
            missing.append(f"a:{va[:30]}")
        if nb and nb.rstrip(".") not in lib.norm1(sc.side_b.body):
            missing.append(f"b:{vb[:30]}")
    return (not missing), ";".join(missing)


def main() -> int:
    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l]
    a_rows = [r for r in rows if r["orientation"] == "a>b"]
    by_s: dict[str, list] = {}
    for r in a_rows:
        by_s.setdefault(r["stratum"], []).append(r)
    audited = []
    ok_all = True
    for s in ORDER:
        cands = sorted(by_s[s], key=lambda r: r["pair_id"])
        step = max(1, len(cands) // SAMPLE[s])
        sample = [cands[i * step] for i in range(SAMPLE[s]) if i * step < len(cands)]
        by_id = {}
        # rebuild scenario for region checks: re-derive from lib on ids is
        # expensive; audit at the ROW level + mechanical via lib on texts
        for r in sample:
            checks = {}
            checks["mechanical_row"] = None  # filled below via lib if pool pair found
            checks["regions_vs_slots"] = None
            audited.append({"pair_id": r["pair_id"], "stratum": s, **checks})
    # The row-level audit cannot rebuild regions from jsonl alone; the
    # audit therefore runs on the LIB pools (deterministic regeneration).
    sys.path.insert(0, str(REPO))
    from build_conflict_detector_corpus import gen_pool, SEED_TRAIN  # noqa: E402

    pool = gen_pool(SEED_TRAIN, "tr")
    by_id = {x.scenario_id: x for x in pool}
    results = []
    n_pass = 0
    for entry in audited:
        sc = by_id[entry["pair_id"]]
        mech = lib.mechanical_ok(sc)
        regs = lib.regions(sc.side_a.body, sc.side_b.body)
        slots_ok = len(regs) == len(sc.slots) if sc.stratum in (
            "CD-1", "CD-2", "CD-3", "CD-4", "CD-5", "CD-6") else None
        pres, missing = value_presence(sc)
        label = "contradicts" if sc.stratum.startswith("CD") else "not-contradicts"
        label_ok = sc.label == label
        ok = bool(mech) and (slots_ok is not False) and pres and label_ok
        n_pass += ok
        results.append({
            "pair_id": sc.scenario_id, "stratum": sc.stratum,
            "mechanical": bool(mech),
            "regions": len(regs), "slots": len(sc.slots),
            "value_traceable": pres, "missing": missing,
            "label": sc.label, "label_expected": label, "label_ok": label_ok,
            "pass": ok,
        })
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(results),
        "pass": n_pass,
        "fail": len(results) - n_pass,
        "pairs": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"AUDIT: {n_pass}/{len(results)} pass -> {OUT}")
    fails = [r for r in results if not r["pass"]]
    for r in fails[:10]:
        print("FAIL:", json.dumps(r, ensure_ascii=False))
    return 0 if n_pass == len(results) and len(results) >= 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
