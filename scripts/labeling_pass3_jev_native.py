"""Labeling pass 3 — PURE Jev (~typesafe/jev-latest) via native decisions API.
Owner directive: pass 2 was the rotating router (comparison only); this is the
real Jev model. Emits labels-C.csv (verdict) + labels-C-notes.jsonl (probability)."""
import json, time, urllib.request, sys, os

CARDS = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-canon-data/corpus/pairs.jsonl"
OUT = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma-canon-data/labeling"
API = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"
KEY = os.environ["OPENROUTER_API_KEY"]

INSTR = ("duplicate = both records are ONE memory (same fact/decision/setting/conclusion); "
         "merging them would lose no information. not_duplicate = differences change the meaning, "
         "or the records are about different entities — even if the texts are highly similar "
         "(shared session-checkpoint templates, same topic, same project). Judge semantics, not "
         "surface similarity; note dates and which record adds unique information. "
         "Answer with a calibrated probability that the two records are duplicates.")

rows = [json.loads(l) for l in open(CARDS)]
results = {}
cost_total = 0.0
models = set()
for i, r in enumerate(rows, 1):
    state = {
        "record_a": {"title": (r["a"].get("title") or "")[:200],
                     "tags": r["a"].get("tags") or [],
                     "body": (r["a"].get("body") or "")[:2500]},
        "record_b": {"title": (r["b"].get("title") or "")[:200],
                     "tags": r["b"].get("tags") or [],
                     "body": (r["b"].get("body") or "")[:2500]},
    }
    payload = {"model": MODEL, "state": state, "questions": {
        "is_duplicate": {"type": "noul", "question": "Are record_a and record_b the SAME memory (duplicates)?",
                         "instructions": INSTR}}}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.load(resp)
            p = d["answers"]["is_duplicate"]["noul"]
            results[r["pair_id"]] = p
            cost_total += d.get("usage", {}).get("cost", 0.0)
            models.add(d.get("model", "?"))
            if i % 20 == 0:
                print(f"{i}/{len(rows)} cost=${cost_total:.4f}", file=sys.stderr, flush=True)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * 2**attempt); continue
            print(f"FATAL {r['pair_id']}: HTTP {e.code} {e.read()[:200]}", file=sys.stderr)
            results[r["pair_id"]] = None
            break
        except Exception as e:
            print(f"  retry {r['pair_id']} a{attempt}: {type(e).__name__} {str(e)[:100]}", file=sys.stderr, flush=True)
            time.sleep(5 * 2**attempt)
    else:
        results[r["pair_id"]] = None

def label(p):
    if p is None: return ""
    return "duplicate" if p >= 0.5 else "not_duplicate"

with open(f"{OUT}/labels-C.csv", "w") as fh:
    fh.write("pair_id,label,note\n")
    for r in rows:
        p = results.get(r["pair_id"])
        note = f"noul={p:.3f}" if p is not None else ""
        fh.write(f"{r['pair_id']},{label(p)},{note}\n")
with open(f"{OUT}/labels-C-notes.jsonl", "w") as fh:
    for r in rows:
        p = results.get(r["pair_id"])
        fh.write(json.dumps({"pair_id": r["pair_id"], "noul": p, "label": label(p)}, ensure_ascii=False) + "\n")
import collections
c = collections.Counter(label(p) for p in results.values())
print(json.dumps({"labeled": sum(1 for p in results.values() if p is not None), "of": len(rows),
                  "by_label": dict(c), "cost_usd": round(cost_total, 4), "models": sorted(models)}))
