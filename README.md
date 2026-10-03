# vesma-cortex

[![CI](https://github.com/vesmaro/vesma-cortex/actions/workflows/ci.yml/badge.svg)](https://github.com/vesmaro/vesma-cortex/actions/workflows/ci.yml)
[![nightly](https://github.com/vesmaro/vesma-cortex/actions/workflows/nightly.yml/badge.svg)](https://github.com/vesmaro/vesma-cortex/actions/workflows/nightly.yml)

Локальная модель решений **is-duplicate** для памяти vesmaro: пара записей →
вердикт + калиброванная вероятность дубликата. Не LLM: self-contained ONNX
~98 КБ — деревянный бустинг над 13 абстрактными фичами пары; офлайн, ноль
сети. Имя — нейро-парадигма: «кора» = ассоциативные и исполнительные функции
над памятью, а не хранение.

## Статус

| Что | Состояние |
|---|---|
| Ревизия в проде | **vesma-cortex-v1 (B1)**, sha256 `beb0a65d…` — sanity-PASS, ADOPT |
| Подключение в движке | за флагом `mnemos.decision_provider: vesma` (default `deterministic`) |
| Калибровки | ds1000 ADOPT 0.99/0.81 · B1 ADOPT, в проде 0.987/0.72 · B2 **отозван** 0.983/0.628 |
| Отзыв B2 | 2026-10-03: инверсия self-pair (#480) — прод переведён на B1, [протокол](docs/experiments/b0-escalation.md) |
| B0-телеметрия | идёт; окно сегментировано точкой отзыва, вердикт — до 2026-10-16 ([план](docs/experiments/b0-telemetry-plan.md)) |
| Реестр артефактов | [models/vesma-cortex-v1/](models/vesma-cortex-v1/) — веса, манифест (schema 2), [MODEL_CARD](models/vesma-cortex-v1/MODEL_CARD.md), [eval-results](models/vesma-cortex-v1/eval-results.json) ([ADR 0003](docs/decisions/0003-model-artifacts-release-and-eval.md)) |

## Quickstart

Пакет `vesma-cortex` публикуется на PyPI — публикация готовится (релизный
поезд: полный сюит CI → wheel с весами, ADR 0003 R5).

**Веса из wheel** — артефакт едет внутри пакета (`cortex/models/`), загрузка
с проверкой целостности:

```python
import hashlib, json
from importlib import resources
import onnxruntime as ort
bundle = resources.files("cortex") / "models" / "vesma-cortex-v1"
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
assert hashlib.sha256((bundle / "model.onnx").read_bytes()).hexdigest() == manifest["sha256"], "weights digest mismatch"
session = onnxruntime.InferenceSession((bundle / "model.onnx").read_bytes(), providers=["CPUExecutionProvider"])
```

Дальше — по инференс-контракту: фичи пары считаются на вызывающей стороне,
выход — `Noul` + `Score`, коды ошибок `CORTEX-E-*`
([docs/specs/inference-v1.md](docs/specs/inference-v1.md)).

**Из репы** — разработка и переобучение:

```bash
git clone https://github.com/vesmaro/vesma-cortex && cd vesma-cortex
uv sync                                   # дефолтная среда (без torch)
make test                                 # uv run pytest tests/ -q (N-тесты скипаются честно)
make smoke                                # CPU-смоук полного контура (~5 с)

uv sync --extra train                     # + CPU-torch: нейроголова N
make smoke-n                              # полный смоук с N-лестницей (~90 с)
```

## Интеграция в движок vesma

- Флаг конфига `mnemos.decision_provider: vesma` переводит решения дедупликации
  с baseline-эвристики (косинус 0.92) на модель; любой отказ `CORTEX-E-*` —
  fail-open на `deterministic` с громким warn-кодом (семантика —
  [inference-v1 §7](docs/specs/inference-v1.md)).
- Движок вендорит артефакт **по фингерпринту** при своём релизе
  (`src/vesmaro/models/vesma-cortex-v1/`, прецедент W5d) и верифицирует байты
  по манифесту — процедуры вендоринга и отзыва:
  [docs/specs/model-registry.md](docs/specs/model-registry.md).

## Методология оценки — в трёх строках

1. **Препрег single-shot**: калибровка по замороженному протоколу, holdout
   сжигается одним прогоном, ADOPT/DECLINE — по фиксированному decision rule.
2. **Adversarial sanity-сьют** (урок #480): self-pair, near-boundary,
   unrelated, монотонность — обязателен ДО цитирования любых метрик
   (`cortex.eval.sanity`); пороги заморожены в `gate_contract.json`.
3. **B0-телеметрия**: delta-rate и no-harm-коридор в поле — источник правды
   о ценности, не гейт качества.

Подробно: [docs/specs/eval-methodology.md](docs/specs/eval-methodology.md).

## Документы

| Документ | Что содержит |
|---|---|
| [docs/charter.md](docs/charter.md) | чартер: строим / не строим, критерии успеха, порядок волн |
| [docs/roadmap-v2.md](docs/roadmap-v2.md) | дорожная карта v2: расширение обязанностей модели (фазы R/D/T, волны B0–B5) |
| [docs/status.md](docs/status.md) | живая доска состояния |
| [docs/decisions/](docs/decisions/) | ADR репозитория (архитектура, реестр артефактов, релизный цикл) |
| [docs/specs/inference-v1.md](docs/specs/inference-v1.md) | инференс-контракт: вход/выход, ONNX-артефакт, загрузка, коды `CORTEX-E-*` |
| [docs/specs/model-registry.md](docs/specs/model-registry.md) | реестр: раскладка, схемы отчёта и карты, вендоринг, отзыв |
| [docs/specs/eval-methodology.md](docs/specs/eval-methodology.md) | методология оценки: метрики, слои, гейты, классы проб |
| [docs/specs/data-contract.md](docs/specs/data-contract.md) | контракт данных: pair-манифесты, фингерпринты, гигиена экспорта |
| [docs/experiments/](docs/experiments/) | отчёты калибровок (ds1000, B1, B2), телеметрия B0, протокол отзыва |
| [docs/runbooks/](docs/runbooks/) | операторский протокол, XPU-walkthrough |

## Лицензия и размещение

- **Лицензия:** Apache-2.0 (решение владельца 2026-09-29: патентная защита +
  семейный дефолт — движок и канон Apache-2.0).
- **Размещение:** github.com/vesmaro/vesma-cortex — публичная репа; пакет
  `vesma-cortex` публикуется на PyPI (публикация готовится).
- **Дисциплина данных:** ни одной сырой строки стора в репе; веса — артефакт,
  а не данные стора ([ADR 0003 R2](docs/decisions/0003-model-artifacts-release-and-eval.md));
  секрет-скан и `no-federate` до экспорта любых данных.
