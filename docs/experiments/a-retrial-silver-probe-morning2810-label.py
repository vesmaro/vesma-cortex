#!/usr/bin/env python3
"""A-retrial silver probe — LLM labeling stage (local ollama assistant).

Labels the 24 silver-sample pairs (4 sub-0.92 full population + 10 boundary
[0.92,0.925) + 10 controls >=0.925) with the rubric fixed by the TL brief:
"related?" = one topic/project/object/artifact, refinement or development of
each other. The LLM is an ASSISTANT: verdicts go to the TL, no auto-status.

Model: qwen2.5:7b-instruct (local ollama, temperature 0). Resumable: pairs
already present in the labels file are skipped. Texts are truncated to 500
chars per side for CPU prompt-eval economy; truncated labeling of topic
relatedness is the declared probe method (not a fingerprint benchmark).
Raw texts stay in gitignored data/a-retrial/ only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
SAMPLE = WT / "data" / "a-retrial" / "snapshot-morning-2810" / "silver_sample.jsonl"
LABELS = WT / "data" / "a-retrial" / "snapshot-morning-2810" / "silver_labels.jsonl"
MODEL = "qwen2.5:7b-instruct"
CHARS = 300


def side(rec: dict) -> str:
    t = (rec.get("title") or "").strip()
    c = " ".join((rec.get("content") or "").split())
    return (f"[заголовок: {t}] " if t else "") + c[:CHARS]


def prompt_for(row: dict) -> str:
    return f"""Ты — ассистент разметки связей между заметками базы знаний.

Запись A: {side(row['from'])}

Запись B: {side(row['to'])}

Вопрос: связаны ли эти две записи по существу (одна тема/проект/объект/артефакт, уточнение или развитие друг друга)?

Ответь СТРОГО в формате (без лишнего текста):
VERDICT: related|not-related
WHY: одно предложение основания"""


def call_ollama(prompt: str) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 60, "num_thread": 8},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return resp["message"]["content"]


def parse(text: str):
    m = re.search(r"verdict\s*:\s*([a-zA-Z-]+)", text, flags=re.IGNORECASE)
    if not m:
        return None, text
    v = m.group(1).lower().strip()
    if v.startswith("not"):
        return "not-related", text
    if v == "related":
        return "related", text
    return None, text


def main() -> int:
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    done = set()
    if LABELS.exists():
        for l in open(LABELS, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["from_id"], r["to_id"]))
    tally = {"related": 0, "not-related": 0, "unparsed": 0}
    by_group = {}
    with open(LABELS, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            key = (row["from_id"], row["to_id"])
            if key in done:
                continue
            t0 = time.time()
            verdict, raw = parse(call_ollama(prompt_for(row)))
            if verdict is None:  # one strict retry
                verdict, raw = parse(call_ollama(prompt_for(row) + "\nВерни ТОЛЬКО две строки VERDICT и WHY."))
            tally[verdict or "unparsed"] += 1
            g = row["group"]
            by_group.setdefault(g, {"related": 0, "n": 0})
            by_group[g]["n"] += 1
            if verdict == "related":
                by_group[g]["related"] += 1
            f.write(json.dumps(
                {
                    "from_id": row["from_id"], "to_id": row["to_id"],
                    "group": g, "cos": row["cos"], "band": row["band"],
                    "verdict": verdict, "raw": raw,
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(rows)}] {g} cos={row['cos']:.4f} -> {verdict} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("TALLY:", json.dumps(tally), "BY_GROUP:", json.dumps(by_group))
    return 0


if __name__ == "__main__":
    sys.exit(main())
