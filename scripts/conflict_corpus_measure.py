#!/usr/bin/env python3
"""Measure step for the conflict-detector corpus build (duty #6, phase D).

Ratified design: docs/experiments/conflict-detector-corpus-design.md
(RATIFIED, MLC-9, commit 1acd88f). The build order is measure -> fix ->
build; this script is the DRY pool measurement (no emission): it probes
the authored construction space against the ratified cosine ladder
(design 4.3) and verdicts each floor BEFORE any quota is pinned.

Instrument: the pinned nano embedder, direct ``vesma.embeddings``
import (B4 precedent, merge_arbiter_negatives_v2.py) -- 384-dim,
int8 ONNX shipped inside the engine package; the pin
``nano:sha256:3b752e06...`` is verified at runtime and a mismatch is a
loud refusal (wrong-provider check, A-frozen).

Stop rule (TL lane order): a red floor at measure = STOP, report, do
not weaken. This script emits the numbers; it never emits corpus rows.

Run: <engine-venv-python> scripts/conflict_corpus_measure.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "conflict-detector" / "measure"

#: Ratified pin (design 4.3, A-frozen).
NANO_PIN = "3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"

#: The ratified ladder floors that the measure step must verify as
#: fillable from the authored construction space (design 4.3):
#: (1) far band "<0.40, 18 pairs, NC-5 only";
#: (2) hard zone "[0.40,0.55), positives >= 30 (CD-6 >= 20, CD-1 >= 10)".
#: Everything between 0.55 and 1.0 is covered by construction families
#: already measured on this line's history (policy 3 bands table).
FAR_BAND_LIMIT = 0.40
HARD_ZONE_LIMIT = 0.55

#: Authored probe pairs (all synthetic, ADR-0003-R2 publication class).
#: Each entry: (family, text_a, text_b, is_positive_candidate).
PROBES: list[tuple[str, str, str, bool]] = [
    # --- far-band candidates (any class; the band wants NC-5) ---
    ("far", "Модуль экспорта формирует отчёты в PDF и отправляет их в очередь уведомлений",
            "Boil the beetroot with cabbage, add potatoes and simmer the soup for thirty minutes", False),
    ("far", "Воркер ретраев повторяет неудачные запросы с экспоненциальной задержкой",
            "The striker scored twice in the second half of the championship final", False),
    ("far", "Таймаут 30 секунд, ретраев 5, батч 200, порт 8443",
            "Autumn in the mountains brings early snow and long quiet evenings", False),
    ("far", "порт 8443, таймаут 30, ретраи 5",
            "варить свёклу 40 минут, затем добавить капусту и лавровый лист", False),
    ("far", "deploy failed: exit code 137, pod evicted",
            "Silently the winter covers the sleeping garden beds with snow", False),
    ("far", "retry queue depth 42, worker pid 118",
            "the violin section tuned before the overture began", False),
    ("far", "Порт 8443, таймаут 30 секунд, ретраев пять",
            "Simmer the soup gently for half an hour and season with herbs", False),
    ("far", "ретраи 5, таймаут 30, батч 200",
            "калорийность 350, белки 12, жиры 8, углеводы 45", False),
    ("far", "Модуль экспорта отчётов", "Кулинарный рецепт борща", False),
    # --- hard-zone positive candidates (CD-6 / cross-lang CD-6) ---
    ("hard-pos", "Подход с кэшированием дал заметный прирост скорости ответов",
                 "Оптимизация кэша не ускорила систему, задержки остались прежними", True),
    ("hard-pos", "Кэш ускорил ответы", "Кэш не помог, задержки на месте", True),
    ("hard-pos", "Кэширование ускорило ответы API",
                 "Caching did NOT improve the API latency at all", True),
    ("hard-pos", "Модуль рекомендаций увеличил конверсию на 12%",
                 "The recommendations module reduced conversion, users ignored it", True),
    ("hard-pos", "Пайплайн обучения моделей переобучается еженедельно на свежих данных и показывает рост качества",
                 "Обучение деградирует", True),
    ("hard-pos", "Кэш ускорил ответы", "The cache made responses slower", True),
    ("hard-pos", "Экспорт работает", "Export is broken", True),
    # --- hard-zone negative candidates (NC-5 cross-lang sibling) ---
    ("hard-neg", "Планировщик задач запускает ночные джобы по расписанию",
                 "The retry worker repeats failed requests with exponential backoff", False),
    # --- razor sanity (band floor evidence, policy T3 band) ---
    ("razor", "батч-лимит 500 запросов в минуту", "батч-лимит 1000 запросов в минуту", True),
    ("razor", "релиз коннектора v1.2.0: обновлён драйвер",
              "релиз коннектора v1.3.0: обновлён драйвер", False),
]


def main() -> int:
    from vesma.embeddings import NanoProvider

    provider = NanoProvider()
    sha = getattr(provider, "weights_sha256", "")
    if sha != NANO_PIN:
        print(f"REFUSE: embedder pin mismatch: got {sha!r}, want {NANO_PIN!r} (A-frozen)",
              file=sys.stderr)
        return 2
    print(f"instrument: nano pin OK ({sha[:16]}…), dim {len(provider.embed('x'))}")

    rows = []
    t0 = time.monotonic()
    for family, a, b, positive in PROBES:
        va, vb = provider.embed(a), provider.embed(b)
        cos = sum(x * y for x, y in zip(va, vb))
        rows.append({"family": family, "cos": round(cos, 4), "positive": positive,
                     "text_a": a, "text_b": b})
    elapsed = time.monotonic() - t0

    far_rows = [r for r in rows if r["family"] == "far"]
    pos_hard = [r for r in rows if r["family"] == "hard-pos"]
    neg_hard = [r for r in rows if r["family"] == "hard-neg"]

    #: Fillability = the CLOSEST a construction family gets to the band
    #: (for the far band that is the MINIMUM cos over candidates).
    far_best = min(r["cos"] for r in far_rows)
    pos_hard_best_in_zone = max((r["cos"] for r in pos_hard if r["cos"] < HARD_ZONE_LIMIT),
                                default=None)
    pos_hard_min_overall = min(r["cos"] for r in pos_hard)
    neg_hard_in_zone = sum(1 for r in neg_hard if FAR_BAND_LIMIT <= r["cos"] < HARD_ZONE_LIMIT)

    verdict_far = "FILLABLE" if far_best < FAR_BAND_LIMIT else "RED: unfillable"
    verdict_pos = ("FILLABLE" if pos_hard_best_in_zone is not None
                   else f"RED: unfillable (closest overall {pos_hard_min_overall})")

    print("\n| cos | family | positive | pair (a vs b, first 60 chars) |")
    print("|---|---|---|---|")
    for r in rows:
        a = r["text_a"][:60].replace("|", "/")
        print(f"| {r['cos']:.4f} | {r['family']} | {r['positive']} | {a}… |")

    print(f"\nfar-band (<{FAR_BAND_LIMIT}): closest candidate {far_best} -> {verdict_far}")
    print(f"hard-zone positives (<{HARD_ZONE_LIMIT}): best in zone "
          f"{pos_hard_best_in_zone} -> {verdict_pos}")
    print(f"hard-zone negatives in [0.40,0.55): {neg_hard_in_zone}/"
          f"{len(neg_hard)} candidates land in zone (NC-5 fillable)")

    stop = verdict_far.startswith("RED") or verdict_pos.startswith("RED")
    print(f"\nVERDICT: {'STOP - do not emit; report floors to the chair' if stop else 'PROCEED to fix'}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "instrument": {"provider": "vesma.embeddings.NanoProvider", "nano_pin": NANO_PIN},
        "rows": rows,
        "verdicts": {
            "far_band": {"limit": FAR_BAND_LIMIT, "closest_candidate": far_best,
                         "verdict": verdict_far},
            "hard_zone_positives": {"limit": HARD_ZONE_LIMIT,
                                    "best_in_zone": pos_hard_best_in_zone,
                                    "closest_overall": pos_hard_min_overall,
                                    "verdict": verdict_pos},
            "hard_zone_negatives_in_zone": neg_hard_in_zone,
        },
        "stop": stop,
        "elapsed_s": round(elapsed, 1),
    }
    out_json = OUT / "measure-probes.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evidence: {out_json}")
    print(f"elapsed: {elapsed:.1f}s")
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
