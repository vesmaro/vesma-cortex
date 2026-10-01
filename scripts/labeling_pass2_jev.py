"""Labeling pass 2 — typesafe/jev-router via OpenRouter (owner directive).
Reads canon labeling cards? No — reads canon corpus pairs.jsonl (same pairs,
machine-readable) + codebook, emits labels-B.csv + labels-B-notes.jsonl.
One-off operational script; batched JSON verdicts, temperature 0."""
import json, time, urllib.request, sys, os

CARDS = "/var/home/abyss/LABs/Projects/Project-Mnemos/vesmaro-canon-data/corpus/pairs.jsonl"
COSINES = "/var/home/abyss/LABs/Projects/Project-Mnemos/vesmaro-canon-data/corpus/cosines.csv"
OUT = "/var/home/abyss/LABs/Projects/Project-Mnemos/vesmaro-canon-data/labeling"
API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "typesafe/jev-router"
KEY = os.environ["OPENROUTER_API_KEY"]

codebook = """You are labeling memory-record PAIRS for a duplicate detector's ground truth (an independent second opinion pass; another annotator did pass 1 independently).
Codebook (frozen):
- "duplicate": both records are ONE memory (same fact/decision/setting/conclusion); merging them would lose no information.
- "not_duplicate": differences change the meaning, or the records are about different entities — even if the texts are highly similar (shared templates, same topic, same project).
- "disputed": you cannot decide per the codebook (then give a one-line reason).
Judge SEMANTICS, not surface similarity: session-checkpoint records share a template ("# Session checkpoint / Goals / Completed") — two checkpoints of DIFFERENT sessions are NOT duplicates. Records re-ingesting the same document with different content are NOT duplicates. Pay attention to dates, actual facts, and which record adds unique information.
Answer STRICTLY as JSON: {"verdicts": [{"pair_id": "...", "label": "duplicate|not_duplicate|disputed", "reason": "<=15 words"}]}"""

rows = [json.loads(l) for l in open(CARDS)]
cos = {}
for line in open(COSINES):
    if line.startswith("pair_id"): continue
    parts = line.strip().split(",")
    cos[parts[0]] = parts[1] if len(parts) > 1 else "?"

def body(side):
    t = side.get("title") or ""
    b = side.get("body") or ""
    return f"[TITLE] {t}\n[BODY] {b[:2500]}"

BATCH = 6
verdicts = {}
calls = 0
for i in range(0, len(rows), BATCH):
    chunk = rows[i:i+BATCH]
    payload = {"model": MODEL, "temperature": 0,
        "messages": [
            {"role": "system", "content": codebook},
            {"role": "user", "content": "Label these pairs:\n" + "\n\n".join(
                f"pair_id={r['pair_id']} (vector cosine {cos.get(r['pair_id'],'?')}):\n--- A ---\n{body(r['a'])}\n--- B ---\n{body(r['b'])}" for r in chunk
            )}]}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.load(resp)
            text = data["choices"][0]["message"]["content"]
            js = text[text.find("{"): text.rfind("}")+1]
            parsed = json.loads(js)
            for v in parsed["verdicts"]:
                verdicts[v["pair_id"]] = v
            calls += 1
            print(f"batch {i//BATCH+1}/{(len(rows)+BATCH-1)//BATCH}: +{len(parsed['verdicts'])} (total {len(verdicts)})", file=sys.stderr, flush=True)
            break
        except Exception as e:
            print(f"  retry {attempt+1}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(5 * 2**attempt)
    else:
        print(f"BATCH FAILED at offset {i}", file=sys.stderr)
        break

with open(f"{OUT}/labels-B.csv", "w") as fh:
    fh.write("pair_id,label,note\n")
    for r in rows:
        v = verdicts.get(r["pair_id"], {})
        note = (v.get("reason") or "").replace(",", ";")[:80] if v.get("label") == "disputed" else ""
        fh.write(f"{r['pair_id']},{v.get('label','')},{note}\n")
with open(f"{OUT}/labels-B-notes.jsonl", "w") as fh:
    for r in rows:
        v = verdicts.get(r["pair_id"], {})
        fh.write(json.dumps({"pair_id": r["pair_id"], "label": v.get("label"),
                             "rationale": v.get("reason", "")}, ensure_ascii=False) + "\n")
import collections
c = collections.Counter(v.get("label") for v in verdicts.values())
print(json.dumps({"labeled": len(verdicts), "of": len(rows), "calls": calls, "by_label": dict(c)}))
