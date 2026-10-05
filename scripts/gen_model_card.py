#!/usr/bin/env python3
"""Generate MODEL_CARD.md from eval-results.json (model-registry.md #4).

The card is GENERATED, never hand-edited: fix the report, re-run the
generator. Deterministic: python3 stdlib only, no network, no clock reads —
same eval-results.json -> byte-identical card.

Usage:
    uv run python scripts/gen_model_card.py
    uv run python scripts/gen_model_card.py --eval-results <path> --out <path>

Defaults: reads models/vesma-cortex-v1/eval-results.json and writes
MODEL_CARD.md next to it (the registry layout, model-registry.md #1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REGISTRY_DIR = Path("models/vesma-cortex-v1")
EVAL_RESULTS_NAME = "eval-results.json"
CARD_NAME = "MODEL_CARD.md"

#: Short weights fingerprint (first 8 hex) — the report-level id convention
#: (ADR 0003 R3: short ids are sha256 prefixes).


def _fp8(weights_sha256: str) -> str:
    return weights_sha256[:8]


def _calibration_status(entry: dict) -> str:
    """Human status line for one calibration_history row."""
    status = str(entry.get("status", "NO-DATA"))
    fp8 = _fp8(str(entry.get("weights_sha256", "")))
    if status == "RECALLED":
        return (
            f"ADOPT отозван {entry.get('recalled_on', 'NO-DATA')} — "
            f"{entry.get('recall_reason', 'см. отчёт')}"
        )
    if status == "ADOPT" and "in production" in str(entry.get("note", "")):
        return f"ADOPT, в проде ({fp8})"
    if status == "ADOPT":
        return f"ADOPT ({fp8})"
    return status


def _render_card(report: dict) -> str:
    artifact = report["artifact"]
    revision = artifact["revision"]
    layers = report["layers"]
    sanity = layers["A_sanity"]
    calib = layers["B_calibration"]
    provenance = report["provenance"]

    checks = sanity["checks"]
    lines: list[str] = []
    add = lines.append

    add(f"# Model Card: vesma-cortex (ревизия {revision})")
    add("")
    add("> Сгенерировано `scripts/gen_model_card.py` из `eval-results.json`;")
    add("> ручные правки карты запрещены — правится отчёт, карта")
    add("> перегенерируется ([model-registry](../../docs/specs/model-registry.md) §4).")
    add("")
    add("## Что делает модель")
    add("")
    add(
        "Пара записей памяти → P(duplicate) [0, 1] + типизированный вердикт "
        "is-duplicate. Роль — советчик дедупликации (guard-подсказка), не "
        "автономное слияние. Не LLM: деревянный бустинг (кандидат d-boost) "
        "над 13 абстрактными фичами пары — модель не хранит текст. Инференс "
        "локальный, офлайн, ноль сети; весит ~98 КБ (ONNX, self-contained)."
    )
    add("")
    add("## Инференс-контракт")
    add("")
    add(
        "[docs/specs/inference-v1.md](../../docs/specs/inference-v1.md): "
        "вход — canon-состояние пары (две записи + измеренный косинус), "
        "выход — Noul + Score; коды ошибок `CORTEX-E-*`; загрузка — паттерн "
        "NanoProvider, гейты размера/метаданных/embedder_pin."
    )
    add("")
    add("## Калибровки (holdout single-shot, BA модель / baseline)")
    add("")
    add("| Корпус | Модель | Baseline | Статус |")
    add("|---|---|---|---|")
    for entry in report.get("calibration_history", []):
        label = str(entry.get("label", entry.get("revision", "NO-DATA")))
        add(
            f"| {label} | {entry['ba']} | {entry['baseline_ba']} "
            f"| {_calibration_status(entry)} |"
        )
    add("")
    add(
        f"Прогрессирующая трудность корпусов: baseline-эвристика (косинус 0.92) "
        f"проседает с 0.81 до 0.628, модель держит ≥ 0.98. Текущая ревизия "
        f"{revision}: {calib['ba']} / {calib['baseline_ba']} "
        f"(sens {calib['sensitivity']}, spec {calib['specificity']}, "
        f"Brier {calib['brier']}) — {calib.get('report_ref', '')}."
    )
    add("")
    add("## Проверки артефакта (Layer A sanity, #480-гейт)")
    add("")
    add(
        f"Статус: **{sanity['status']}** "
        f"(прогон {sanity['suite'].get('run_at', 'NO-DATA')}, "
        f"`cortex.eval.sanity`). Само-пара P(dup|запись против себя) = "
        f"{checks['self_pair']} при пороге ≥ 0.9; cosmetic-твин = "
        f"{checks.get('cosmetic_twin', checks.get('near_boundary', 'n/a'))} "
        f"@ cos 0.99 (порог ≥ 0.5; v2, экс-near_boundary); unrelated = "
        f"{checks['unrelated']} @ cos 0.578 (потолок < 0.5); монотонность по "
        f"лестнице {{1.0, 0.99, 0.95, 0.8, 0.5}} — {checks['monotonicity']} "
        f"(v2 зонная: razor (0.95; 1.0) двузначна); fact-edit твин = "
        f"{checks.get('fact_edit_twin', 'n/a')} (< 0.5); envelope-вариант = "
        f"{checks.get('envelope_variant', 'n/a')} (≥ 0.5) — sanity v2."
    )
    add("")
    add("## Данные обучения")
    add("")
    add(
        f"Обучена ТОЛЬКО на train-половине корпуса: "
        f"{calib.get('train_corpus', calib.get('corpus', 'NO-DATA'))}. "
        "Корпуса — пары store-происхождения (гигиеничный экспорт: сырые строки "
        "стора в репу и в корпус не попадают) + синтетика на шаблонах; метки — "
        "по построению и трёхпроходной валидации с TL-арбитражем. Фингерпринт "
        f"train-корпуса: `{provenance['train_corpus_fingerprint'][:12]}…` "
        f"(blake2b-256, [data-contract](../../docs/specs/data-contract.md) §5). "
        "Вклад претрейна в финальную ревизию — нулевой (лестница отбора, "
        "отчёт B1)."
    )
    add("")
    add("## Ограничения")
    add("")
    add(
        "- **Классы проб:** сюит покрывает identity/self, near-identity, "
        "монотонность, far-negative; хвосты — симметрия пары P(x,y)=P(y,x), "
        "переводные близнецы, type/lang рассогласования, дегенераты "
        "([eval-methodology](../../docs/specs/eval-methodology.md) §7)."
    )
    add(
        "- **Урок #480 (слепое пятно):** holdout-метрики слепы к дефектам, "
        "которых нет на поверхности оценки — B2 прошёл ADOPT с sens 0.989 и "
        "при этом инвертировал self-пару (P=0.0108). С 2026-10-03 sanity-сьют "
        "обязателен ДО цитирования любых метрик."
    )
    add(
        f"- **embedder_pin-зависимость:** модель валидна только на геометрии "
        f"vesma-embed-v1 (`{provenance['embedder_pin'][:24]}…`); несовпадение "
        "пина = громкий отказ `CORTEX-E-PIN` (событие перекалибровки)."
    )
    add(
        "- **Доменный микс 74/26** в корпусе B1 — пополнение пула "
        "не-инженерными памятями остаётся опцией (ограничение отчёта B1)."
    )
    add(
        "- **Baseline-коридоры:** коридор ловит обрывы, не ранжирует близкие "
        "ревизии; verdict-метрики информационные, решение — по замороженному "
        "decision rule."
    )
    add("")
    add("## Провенанс")
    add("")
    add(
        f"- trained_at: `{provenance['trained_at']}`\n"
        f"- train_corpus_fingerprint: `{provenance['train_corpus_fingerprint']}`\n"
        f"- embedder_pin: `{provenance['embedder_pin']}`\n"
        f"- package_version: `{provenance['package_version']}`\n"
        f"- gate_contract_sha256: `{report['gate_contract_sha256']}` "
        "(пороги гейтов — gate_contract.json в корне репы, пин — "
        "[eval-methodology](../../docs/specs/eval-methodology.md) §6)\n"
        f"- weights_sha256: `{artifact['weights_sha256']}`\n"
        f"- manifest_sha256: `{artifact['manifest_sha256']}`"
    )
    add("")
    add("## Лицензия")
    add("")
    add(
        "Apache-2.0 (решение владельца 2026-09-29: патентная защита + "
        "семейный дефолт — движок и канон Apache-2.0). Размещение: "
        "github.com/vesmaro/vesma-cortex, публичная репа; веса — артефакт, "
        "сырой контент стора в репу не входит "
        "([ADR 0003](../../docs/decisions/0003-model-artifacts-release-and-eval.md) R2)."
    )
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen-model-card",
        description="generate MODEL_CARD.md from eval-results.json (deterministic, stdlib only)",
    )
    parser.add_argument(
        "--eval-results",
        default=str(REGISTRY_DIR / EVAL_RESULTS_NAME),
        help="path to eval-results.json (default: models/vesma-cortex-v1/eval-results.json)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path (default: MODEL_CARD.md next to eval-results.json)",
    )
    args = parser.parse_args(argv)

    results_path = Path(args.eval_results)
    try:
        report = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"gen-model-card: cannot read {results_path} — {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else results_path.parent / CARD_NAME
    out_path.write_text(_render_card(report), encoding="utf-8")
    print(f"gen-model-card: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
