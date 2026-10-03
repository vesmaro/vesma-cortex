# Модельный реестр vesma-cortex

- **Статус:** Accepted (решение АрхКома MR0 —
  [ADR 0003](../decisions/0003-model-artifacts-release-and-eval.md); волна
  MR-1 — наполнение реестра)
- **Дата:** 2026-10-03
- **Область:** раскладка и состав артефакта в реестре, схемы
  `eval-results.json` и `MODEL_CARD.md`, процедуры вендоринга в движок и
  отзыва. Контракты инференса и данных определяются соседними спеками —
  [inference-v1.md](inference-v1.md), [data-contract.md](data-contract.md).

## 1. Раскладка

```
models/
  vesma-cortex-v1/          # ОДИН канонический путь на main = текущий прод
    model.onnx              # веса, self-contained, ≤ 5 МБ (~98 КБ; LFS не нужен)
    manifest.json           # машинные метаданные, schema 2 (ADR 0003 R4)
    MODEL_CARD.md           # человекочитаемая карта — ГЕНЕРИРУЕТСЯ (§4)
    eval-results.json       # машинный отчёт оценки + вердикт (§3)
```

- `main` держит ровно одну ревизию; директории по фингерпринтам не копятся.
- История ревизий — git-теги `model-v1-<fp8>` + GitHub Releases с ассетами
  `model.onnx`, `model.manifest.json`, `SHA256SUMS.txt`.
- Тренировочные веса (`artifacts/weights/`, `data/`) остаются gitignored —
  реестр трекает только артефактные ревизии.

## 2. Состав артефакта

| Файл | Контракт | Проверяется |
|---|---|---|
| `model.onnx` | [inference-v1](inference-v1.md) §4: opset 15, metadata_props, ≤ 5 МБ | sanity-сьют, verify-джоба релиза |
| `manifest.json` | schema 2: `manifest_schema: 2`, `blake2b_256` (+ переходный `sha256` до миграции движка), без `weights_path` | sanity-сьют (bundle_integrity) |
| `MODEL_CARD.md` | генерируется из `eval-results.json` (§4) | ревью PR с весами |
| `eval-results.json` | схема §3 | verify-джоба релиза |

## 3. Схема `eval-results.json`

Машинный отчёт одной артефактной ревизии. Слои без данных несут явное
`"NO-DATA"` — поле не опускается и не заполняется догадками.

```json
{
  "schema_version": 1,
  "artifact": {
    "name": "vesma-cortex",
    "revision": "B1",
    "weights_sha256": "<sha256 байтов model.onnx>",
    "feature_set_sha256": "<дайджест ordered-списка фич>",
    "manifest_sha256": "<sha256 manifest.json>",
    "candidate": "d-boost"
  },
  "gate_contract_sha256": "<sha256 gate_contract.json, по которому гонялся гейт>",
  "layers": {
    "A_sanity":     {"status": "PASS", "checks": {"self_pair": 1.0, "near_boundary": 0.0, "unrelated": 0.0, "monotonicity": "PASS"}},
    "A_regression": {"status": "PASS", "eval_set_sha256": "…", "ba": 0.0, "prev_adopt_ba": 0.0, "corridor": "PASS"},
    "B_calibration": {"status": "ADOPT", "corpus": "…", "sensitivity": 0.0, "specificity": 0.0, "brier": 0.0, "ba": 0.0, "baseline_ba": 0.0, "typed": 1.0},
    "B0_field":     {"status": "NO-DATA", "window": null, "delta_rate": null, "no_harm_band_ratio": null}
  },
  "decision": "ADOPT",
  "provenance": {
    "trained_at": "<ISO-8601 UTC>",
    "train_corpus_fingerprint": "<blake2b-256>",
    "embedder_pin": "<nano:sha256:…>",
    "package_version": "<semver pyproject>",
    "run_log": "<artifacts/runs/….jsonl>"
  }
}
```

- `decision` ∈ `ADOPT | DECLINE | NO-DATA | RECALLED` — отозванная ревизия
  помечается `RECALLED` задним числом (история правок — в протоколе отзыва).
- `gate_contract_sha256` связывает отчёт с конкретной версией порогов —
  анти-HARKing ([eval-methodology](eval-methodology.md) §6).

## 4. `MODEL_CARD.md`: генерируется, не пишется

Правило: карта **генерируется из `eval-results.json`** (генератор — хвост
волны MR-1); ручные правки карты запрещены — правится отчёт, карта
перегенерируется. Шаблон секций:

```markdown
# Model Card: vesma-cortex (ревизия <revision>)

## Что делает модель
Пара записей памяти → P(duplicate) [0, 1] + типизированный вердикт
is-duplicate. Роль — советчик дедупликации (guard-подсказка), не
автономное слияние. Не LLM: деревянный бустинг над 13 абстрактными
фичами пары.

## Инференс-контракт
[docs/specs/inference-v1.md](../../docs/specs/inference-v1.md):
вход — canon-состояние пары, выход — Noul + Score; коды ошибок CORTEX-E-*.

## Калибровки (holdout single-shot, BA модель / baseline)
| Корпус | Модель | Baseline | Статус |
|---|---|---|---|
| ds1000 (инженерный) | 0.99 | 0.81 | ADOPT (281bd0fd) |
| B1 (доменное разнообразие) | 0.987 | 0.72 | ADOPT, в проде (beb0a65d) |
| B2 (рёберная окрестность) | 0.983 | 0.628 | ADOPT отозван 2026-10-03 — инверсия self-pair (#480) |

## Данные обучения
<корпус + фингерпринт из provenance; синтетика/реальные доли>

## Ограничения
<из таблицы покрытия поверхности: непокрытые классы проб;
 baseline-коридоры; embedder_pin-зависимость>

## Провенанс
<trained_at, corpus_fingerprint, embedder_pin, package_version,
 gate_contract_sha256 — из eval-results.json>
```

Метрики трёх калибровок и пометка отзыва B2 — обязательные строки шаблона;
числа подтягиваются из `layers.B_calibration` всех записанных ревизий.

## 5. Вендоринг в движок

1. Ревизия в реестре на `main` имеет зелёный сюит и `decision: ADOPT`.
2. Релиз движка берёт артефакт по фингерпринту (прецедент W5d): копия в
   `src/vesmaro/models/vesma-cortex-v1/` (движок грузит из своей раскладки).
3. CI движка верифицирует вендоренную копию: дайджест манифеста == хеш
   байтов (тест — межрепо-хвост [ADR 0003](../decisions/0003-model-artifacts-release-and-eval.md) R1; до его появления байт-верификация —
   в `release-model.yml` этой репы).
4. Версия движка, вендорившая ревизию, фиксируется в `provenance` отчёта.

## 6. Отзыв (recall)

Процедура — по прецеденту 2026-10-03
([b0-escalation.md](../experiments/b0-escalation.md)):

1. Триггер: красный sanity-сьют или подтверждённый вред в поле → решение
   TL/владельца (компетенция по постскриптуму калибровки B2).
2. Замена во **всех** точках деплоя на последний sanity-PASS артефакт
   (пин последних 2 хороших бандлов — [ADR 0003](../decisions/0003-model-artifacts-release-and-eval.md) R5), бэкапы заменённых файлов, рестарт.
3. Sanity-сьют на заменённом бандле — PASS до закрытия инцидента.
4. `decision` отозванной ревизии в реестре → `RECALLED`; `MODEL_CARD.md`
   перегенерируется; протокол — в `docs/experiments/`.
5. Ограничение, известное из прецедента: уже запущенные сессии держат
   старые веса в памяти до рестарта — фиксируется в протоколе отзыва.

## 7. Ссылки

- [ADR 0003](../decisions/0003-model-artifacts-release-and-eval.md) —
  реестр, релизный поезд, freeze тегов.
- [eval-methodology.md](eval-methodology.md) — слои оценки, гейты,
  классы проб.
- [inference-v1.md](inference-v1.md) / [data-contract.md](data-contract.md)
  — инференс-контракт и фингерпринты.
- [Протокол отзыва 2026-10-03](../experiments/b0-escalation.md) —
  прецедент recall; [постскриптум B2](../experiments/calibration-b2-edge-neighborhood.md).
