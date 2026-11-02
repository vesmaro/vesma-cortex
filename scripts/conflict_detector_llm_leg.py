#!/usr/bin/env python3
"""LLM-assistant leg of the gold fallback OQ-1 (owner opt-out 02.11.2026).

Scores the BLIND sheet render with the pinned local model (same policy
as merge_arbiter_llm_leg.py): real corpus texts never leave the host,
frozen prompt, per-block scoring without cross-block context, ONE cold
pass per block at temperature 0 — transport/parse failures become
NOT SCORED, never retried. Verdicts are written RAW: agreement with the
construction labels is NOT computed here (the committee synthesis does
that after unsealing the key).

Provenance before the run: model pin + prompt sha256 + render sha256
are printed and logged (run-log event llm_leg_started).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import conflict_corpus_lib as lib  # noqa: E402
from build_conflict_detector_corpus import RUNLOG, log  # noqa: E402

RENDER = REPO / "data" / "conflict-detector" / "sheet" / "gold-sheet-render-v1.txt"
PARTIAL = REPO / "data" / "conflict-detector" / "sheet" / "verdicts" / "llm-leg.partial.jsonl"
OUT = REPO / "data" / "conflict-detector" / "sheet" / "verdicts" / "llm-leg.json"

MODEL_PIN = "qwen2.5:7b-instruct"
ENDPOINT = "http://127.0.0.1:11434/api/generate"

PROMPT_TEMPLATE = """Ты независимый вердиктор спот-чека корпуса противоречий. Ниже два текста (A и B) об одном или разных предметах. Определи, противоречит ли один текст другому.

Правила суждения:
1. Противоречие — несовместимые факты об одном предмете: разные числа/лимиты одного слота, разные даты одного события, взаимоисключающие статусы, противоположная модальность (разрешено vs запрещено), безусловное отрицание условного утверждения, противоположные качественные оценки одного эффекта.
2. НЕ противоречие: добавка новых фактов без отмены сказанного; уточнение, укладывающееся в заявленную границу; различие только в номере версии; тождественные тела при различии только тегов/типа записи; разные предметы.
3. Не используй длину текста, позицию или оформление как аргумент — только факт-слоты. Спорное трактуй как NO-CONFLICT только если факты совместимы; иначе CONTRADICTS.

Ответ — СТРОГО один JSON-объект без markdown, вида {{"verdict": "CONTRADICTS", "basis": "обоснование ≤2 предложения со ссылкой на конкретные факт-слоты (числа, даты, статусы, версии, модальности) или на их отсутствие"}}. Допустимый verdict: "CONTRADICTS" или "NO-CONFLICT".

Блок {block}:
{body}"""


def parse_blocks(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n(?=N\d\d\n)", text)
    out = []
    for p in parts:
        m = re.match(r"(N\d\d)\n(.*)", p, flags=re.S)
        if m and "verdict: ______" in p:
            out.append((m.group(1), m.group(2).strip()))
    return out


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    render_bytes = RENDER.read_bytes()
    render_sha = hashlib.sha256(render_bytes).hexdigest()
    prompt_sha = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    blocks = parse_blocks(render_bytes.decode("utf-8"))
    print(f"model pin: {MODEL_PIN}")
    print(f"prompt sha256: {prompt_sha}")
    print(f"render sha256: {render_sha}")
    print(f"blocks: {len(blocks)} (one cold pass, temperature 0)")
    if len(blocks) != 48:
        die(f"expected 48 scorable blocks, got {len(blocks)}")

    done: dict[str, dict] = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["block"]] = r
    fresh = [b for b in blocks if b[0] not in done]
    log("llm_leg_started", model_pin=MODEL_PIN, prompt_sha256=prompt_sha,
        render_sha256=render_sha, blocks=len(blocks), already_done=len(done))
    print(f"resuming: {len(done)} done, {len(fresh)} to score", flush=True)

    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    with PARTIAL.open("a", encoding="utf-8") as fh:
        for block_id, body in fresh:
            prompt = PROMPT_TEMPLATE.format(block=block_id, body=body)
            payload = json.dumps({
                "model": MODEL_PIN,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 4096},
            }).encode()
            verdict, basis, note = "NOT SCORED", "", ""
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        ENDPOINT, data=payload,
                        headers={"Content-Type": "application/json"}),
                    timeout=1800,
                ) as resp:
                    answer = json.loads(resp.read())["response"]
                parsed = json.loads(answer)
                verdict = str(parsed.get("verdict", "")).strip().upper()
                basis = str(parsed.get("basis", ""))[:500]
                if verdict not in {"CONTRADICTS", "NO-CONFLICT"}:
                    verdict, note = "NOT SCORED", f"invalid verdict: {verdict!r}"
                elif len(basis) < 10:
                    verdict, note = "NOT SCORED", "basis too short (no slot reference)"
            except Exception as exc:  # transport/parse failure: one pass, no retry
                note = f"{type(exc).__name__}: {exc}"[:200]
            rec = {"block": block_id, "verdict": verdict, "basis": basis, "note": note}
            done[block_id] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{block_id}: {verdict}", flush=True)

    missing = [b for b, _ in blocks if b not in done]
    if missing:
        print(f"INCOMPLETE: {len(missing)} blocks unscored: {missing[:5]}…; "
              f"re-run to resume (scored blocks are not re-sent)")
        return 1
    results = [done[b] for b, _ in blocks]
    scored = [r for r in results if r["verdict"] != "NOT SCORED"]
    OUT.write_text(json.dumps({
        "verdictor": "llm-assistant",
        "sheet": "CD-v1",
        "ts": now(),
        "model_pin": MODEL_PIN,
        "prompt_sha256": prompt_sha,
        "render_sha256": render_sha,
        "temperature": 0,
        "passes": 1,
        "retries": 0,
        "agreement_with_labels": None,  # computed at synthesis, not here
        "blocks": [{"block": r["block"], "verdict": r["verdict"], "basis": r["basis"]}
                   for r in results],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log("llm_leg_done", model_pin=MODEL_PIN, scored=len(scored),
        not_scored=len(results) - len(scored), out=str(OUT.relative_to(REPO)))
    print(f"LLM LEG DONE: scored {len(scored)}/48, not scored {48 - len(scored)}; -> {OUT}")
    return 0


def die(msg: str) -> None:
    print(f"LLM-LEG STOP: {msg}", file=sys.stderr)
    log("llm_leg_stop", reason=msg)
    sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
