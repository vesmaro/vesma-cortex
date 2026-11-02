#!/usr/bin/env python3
"""Committee synthesis for the gold fallback OQ-1 (owner opt-out 02.11).

Unseals KEY-cd-v1.sealed.json AFTER all three legs are frozen, computes
per-leg agreement and the majority committee verdict, and applies the
ratified g_gold gate: >= 45/50 AND no systematic disagreement pattern on
the confusible twins (razor CD-1 vs NC-2/NC-3, same-point pairs).

RAW legs only: ml-leg.json (ML), tl-leg.json (TL), llm-leg.json (LLM,
resumed from llm-leg.partial.jsonl). Mechanical-first is honoured by
construction: judges never saw the key; the mechanical predicate is not
a judge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SHEET = Path(__file__).resolve().parent.parent / "data/conflict-detector/sheet"
VERDICTS = SHEET / "verdicts"


def load_leg(name: str) -> dict[str, str]:
    rows = json.load(open(VERDICTS / name, encoding="utf-8"))
    return {r["block"]: r["verdict"] for r in rows}


def main() -> int:
    key = json.load(open(VERDICTS / "KEY-cd-v1.sealed.json", encoding="utf-8"))
    truth = {b["block"]: b["label"] for b in key["blocks"]}
    strata = {b["block"]: b["stratum"] for b in key["blocks"]}
    label_of = lambda v: "contradicts" if v == "CONTRADICTS" else "not-contradicts"

    legs = {"ML": load_leg("ml-leg.json"), "TL": load_leg("tl-leg.json")}
    llm_path = VERDICTS / "llm-leg.json"
    if llm_path.exists():
        env = json.load(open(llm_path, encoding="utf-8"))
        legs["LLM"] = {r["block"]: r["verdict"] for r in env.get("blocks", [])}
    else:
        print("LLM leg NOT frozen — synthesis requires all three legs")
        return 2

    print(f"legs: { {k: len(v) for k, v in legs.items()} } | key blocks: {len(truth)}")
    for name, leg in legs.items():
        agree = sum(1 for b, t in truth.items() if b in leg and label_of(leg[b]) == t)
        print(f"agreement {name}: {agree}/48 = {agree / len(truth):.3f}")

    # majority committee verdict (2 of 3; unanimous where all agree)
    committee, disputes = {}, []
    for b, t in truth.items():
        votes = [label_of(leg[b]) for name, leg in legs.items() if b in leg]
        c = "contradicts" if votes.count("contradicts") >= 2 else "not-contradicts"
        committee[b] = c
        if c != t:
            disputes.append((b, strata[b], t, votes))
    print(f"\ncommittee vs key: {48 - len(disputes)}/48  (g_gold needs >= 45)")
    for b, st, t, votes in disputes:
        print(f"  DISPUTE {b} [{st}] key={t} votes={votes}")

    # systematic-pattern check on confusible twins: razor CD-1/CD-2 vs NC-2/NC-3
    razor_cd = {b for b, s in strata.items() if s in ("CD-1", "CD-2")}
    nc_knife = {b for b, s in strata.items() if s in ("NC-2", "NC-3")}
    leg_dis = {n: {b for b, t in truth.items() if b in leg and label_of(leg[b]) != t} for n, leg in legs.items()}
    for n, dis in leg_dis.items():
        on_knife = dis & (razor_cd | nc_knife)
        print(f"{n} disagreements: {sorted(dis)} | on confusible axis: {sorted(on_knife)}")
    sys_pattern = all(len(dis & (razor_cd | nc_knife)) >= 3 for dis in leg_dis.values())
    ok = (48 - len(disputes)) >= 45 and not sys_pattern
    print(f"\nsystematic confusible pattern: {'YES' if sys_pattern else 'NO'}")
    print(f"g_gold verdict: {'PASS' if ok else 'FAIL'} ({48 - len(disputes)}/48)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
