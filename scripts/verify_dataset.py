#!/usr/bin/env python3
"""Dataset verification pass — judge every pair honestly (no expected labels).

Independent verifier for three-pass dataset validation (B1 pipeline):
pass 1 = creator subagent, passes 2-3 = this script with `router` (rotating,
comparison only) and `native` (pure Jev decisions API) backends.

Usage:
    export OPENROUTER_API_KEY=...
    python3 scripts/verify_dataset.py <mode: router|native> <in-dir> <out.jsonl>

<in-dir> holds pairs-pos.jsonl (rows: pair_id, original{title,body,tags}, variant{...})
and pairs-neg.jsonl (rows: pair_id, a{...}, b{...}). Emits out.jsonl rows
{pair_id, noul} — the calibrated P(duplicate) per pair (None on API miss).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

mode, IN_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# Sandboxed runners may sanitize env — read the key from the secrets file
# directly (value stays in memory only; never printed/logged).
KEY = (
    os.environ.get("OPENROUTER_API_KEY")
    or open("/var/home/abyss/.secrets/openrouter_api_key", encoding="utf-8")
    .read()
    .strip()
)
if not KEY:
    raise SystemExit("no OpenRouter key: env empty and secrets file unreadable")
if mode == "native":
    API, MODEL = "https://openrouter.ai/api/alpha/decisions", "~typesafe/jev-latest"
else:
    API, MODEL = "https://openrouter.ai/api/v1/chat/completions", "typesafe/jev-router"

INSTR = (
    "duplicate = both records are ONE memory (same fact/decision/setting/conclusion); "
    "merging them would lose no information. not_duplicate = differences change the meaning, "
    "or the records are about different entities — even if texts are highly similar "
    "(shared templates, same topic, same project). Judge semantics, not surface similarity. "
    "Calibrated probability that these are duplicates."
)

rows = []
for line in open(f"{IN_DIR}/pairs-pos.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows.append((r["pair_id"], r["original"], r.get("variant") or r.get("candidate")))
for line in open(f"{IN_DIR}/pairs-neg.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows.append((r["pair_id"], r["a"], r["b"]))


def side(s):
    return {
        "title": (s.get("title") or "")[:200],
        "tags": s.get("tags") or [],
        "body": (s.get("body") or "")[:2500],
    }


def headers():
    return {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


results = {}
t0 = time.time()
if mode == "native":
    for i, (pid, a, b) in enumerate(rows, 1):
        payload = {
            "model": MODEL,
            "state": {"record_a": side(a), "record_b": side(b)},
            "questions": {
                "is_duplicate": {
                    "type": "noul",
                    "question": "Are record_a and record_b the SAME memory (duplicates)?",
                    "instructions": INSTR,
                }
            },
        }
        req = urllib.request.Request(
            API, data=json.dumps(payload).encode(), headers=headers()
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    d = json.load(resp)
                results[pid] = round(d["answers"]["is_duplicate"]["noul"], 4)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(5 * 2**attempt)
                    continue
                print(f"FATAL {pid}: HTTP {e.code}", file=sys.stderr)
                results[pid] = None
                break
            except Exception as e:
                print(
                    f"retry {pid} a{attempt}: {type(e).__name__} {str(e)[:80]}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5 * 2**attempt)
        else:
            results[pid] = None
        if i % 50 == 0:
            print(
                f"{mode}: {i}/{len(rows)} elapsed {time.time() - t0:.0f}s",
                file=sys.stderr,
                flush=True,
            )
else:
    B = 6
    for i in range(0, len(rows), B):
        chunk = rows[i : i + B]
        content = "\n\n".join(
            f"pair_id={pid}\n--- A ---\n{json.dumps(side(a), ensure_ascii=False)}\n--- B ---\n{json.dumps(side(b), ensure_ascii=False)}"
            for pid, a, b in chunk
        )
        payload = {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You label memory-record pairs. "
                    + INSTR
                    + ' Respond STRICTLY as JSON: {"verdicts": [{"pair_id": "...", "label": "duplicate|not_duplicate|disputed"}]}',
                },
                {"role": "user", "content": content},
            ],
        }
        req = urllib.request.Request(
            API, data=json.dumps(payload).encode(), headers=headers()
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    d = json.load(resp)
                text = d["choices"][0]["message"]["content"]
                js = text[text.find("{") : text.rfind("}") + 1]
                for v in json.loads(js)["verdicts"]:
                    results[v["pair_id"]] = {
                        "duplicate": 0.95,
                        "disputed": 0.5,
                        "not_duplicate": 0.05,
                    }.get(v["label"])
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(5 * 2**attempt)
                    continue
                break
            except Exception as e:
                print(
                    f"retry batch {i} a{attempt}: {type(e).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5 * 2**attempt)
        if (i // B) % 10 == 0:
            print(
                f"{mode}: batch {i // B + 1}/{(len(rows) + B - 1) // B} elapsed {time.time() - t0:.0f}s",
                file=sys.stderr,
                flush=True,
            )

with open(OUT, "w", encoding="utf-8") as fh:
    for pid, _, _ in rows:
        fh.write(
            json.dumps({"pair_id": pid, "noul": results.get(pid)}, ensure_ascii=False)
            + "\n"
        )
ok = [v for v in results.values() if v is not None]
print(
    json.dumps(
        {
            "mode": mode,
            "labeled": len(ok),
            "of": len(rows),
            "dup(>=0.5)": sum(1 for v in ok if v >= 0.5),
            "not_dup(<0.5)": sum(1 for v in ok if v < 0.5),
        }
    )
)
