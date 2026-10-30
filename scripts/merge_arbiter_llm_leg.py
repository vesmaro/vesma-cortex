#!/usr/bin/env python3
"""LLM-assistant leg of the committee-silver sheet scoring (ADDENDUM-1 §1).

Pinned per MLC 29.10: local open-weights model (real store texts never go
to hosted APIs), frozen prompt, per-row scoring without cross-row context,
ONE pass — transport failures become NOT SCORED, never retried.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
RENDER = WT / "data" / "merge-arbiter" / "spot-sheet" / "blind-render-v1.md"
OUT = (
    WT / "data" / "merge-arbiter" / "spot-sheet" / "verdicts" / "llm-assistant-v1.json"
)

MODEL_PIN = "qwen2.5:7b-instruct"
ENDPOINT = "http://127.0.0.1:11434/api/generate"

PROMPT_TEMPLATE = """Ты независимый вердиктор спот-чека консолидации дубликатов. Для одного кластера из 2–3 кандидатов выбери, КАКАЯ запись должна остаться: A / B / C — или NO-SAFE (схлопывать нельзя).

Правила суждения:
1. Если кандидаты попарно байт-идентичны (или различие стирается нормализацией регистра/пробелов) — выбор вырожден, ничего не теряется: ответь A (первый кандидат) и укажи в basis, что тексты идентичны.
2. Если у кандидатов одинаковое тело и заголовок, но РАЗНЫЕ tags / record_type / language — схлопывать нельзя (метаданные управляют видимостью): NO-SAFE.
3. Если тела различаются и различие — противоречащие ключевые факты (разные числа, даты, версии, имена, суммы при почти одинаковом тексте) — схлопывать нельзя: NO-SAFE.
4. Если одно тело является расширением другого, добавляющим ключевые факты (числа, даты, версии, имена, суммы), без противоречий — оставь члена-носителя большего факт-множества.
5. Не используй длину текста, позицию или порядок как аргумент — только факт-слоты. Спорное — NO-SAFE.

Ответ — СТРОГО один JSON-объект безmarkdown, вида {{"choice": "A", "basis": "обоснование ≤2 предложения со ссылкой на факт-слоты или на идентичность текстов"}}.

Кластер {row}:
{body}
"""


def parse_rows(text: str) -> list[tuple[str, str]]:
    rows = []
    parts = re.split(r"\n## (N\d+)\n", text)
    for i in range(1, len(parts), 2):
        rows.append((parts[i], parts[i + 1]))
    return rows


def main() -> int:
    render_sha = hashlib.sha256(RENDER.read_bytes()).hexdigest()
    rows = parse_rows(RENDER.read_text(encoding="utf-8"))
    assert len(rows) == 40, f"expected 40 rows, got {len(rows)}"
    results = []
    for row_id, row_body in rows:
        prompt = PROMPT_TEMPLATE.format(row=row_id, body=row_body.strip())
        payload = json.dumps(
            {
                "model": MODEL_PIN,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 2048},
            }
        ).encode()
        choice, basis, note = "NOT SCORED", "", ""
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
                ),
                timeout=600,
            ) as resp:
                answer = json.loads(resp.read())["response"]
            parsed = json.loads(answer)
            choice = str(parsed.get("choice", "NOT SCORED")).strip().upper()
            basis = str(parsed.get("basis", ""))[:500]
            if choice not in {"A", "B", "C", "NO-SAFE"}:
                choice, note = "NOT SCORED", f"invalid choice: {choice!r}"
        except Exception as exc:  # transport/parse failure: one pass, no retry
            note = f"{type(exc).__name__}: {exc}"[:200]
        results.append({"row": row_id, "choice": choice, "basis": basis, "note": note})
        print(f"{row_id}: {choice}", flush=True)
    scored = [r for r in results if r["choice"] != "NOT SCORED"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "verdictor": "llm-assistant",
                "model_pin": MODEL_PIN,
                "temperature": 0,
                "passes": 1,
                "retries": 0,
                "render_sha256": render_sha,
                "rows": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"scored": len(scored), "not_scored": 40 - len(scored), "out": str(OUT)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
