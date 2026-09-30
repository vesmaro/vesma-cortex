# vesmaro-cortex

Локальные модели решений для экосистемы памяти vesmaro. Имя — нейро-парадигма:
«кора» = ассоциативные и исполнительные функции над памятью, а не хранение.

**Стартовый скоуп (волна A0, [charter.md](docs/charter.md)):** обучение с нуля
локальной модели решений **mnema-cortex** (решение владельца 2026-09-29;
историческое имя трека в прозе canon — «mnema-роутер») — пара записей памяти →
типизированный вердикт `is-duplicate(a, b)` + скор уверенности [0, 1]. Не LLM,
не SFT готовой модели, не генерация текста. Инференс — ноль сети; подключение
в движок — `MnemaProvider` (волна W5d vesmaro-canon) по интерфейсу
[ADR 0004 decision-provider](https://github.com/vesmaro/vesmaro-canon).

Оценка модели — только по замороженной
[препегистрации v2](https://github.com/vesmaro/vesmaro-canon) калибровки
(7 стратов / 200 пар, holdout 30 % single-shot, baseline = косинус 0.92);
DECLINE — честный исход, прецедент W4c.

| Документ | Что содержит |
|---|---|
| [docs/charter.md](docs/charter.md) | чартер A0: строим / не строим, критерии успеха, порядок волн |
| [docs/status.md](docs/status.md) | живая доска состояния |
| [docs/decisions/](docs/decisions/) | ADR репозитория |
| [docs/specs/inference-v1.md](docs/specs/inference-v1.md) | инференс-контракт `mnema-cortex-v1`: вход/выход, ONNX-артефакт, загрузка (паттерн NanoProvider), коды `CORTEX-E-*` |
| [docs/specs/data-contract.md](docs/specs/data-contract.md) | контракт данных: pair-манифесты, фингерпринты (BLAKE2b-256), гигиена экспорта, каталог data/ + artifacts/ |

- **Размещение:** github.com/vesmaro/vesmaro-cortex — публичная репа
  (решение владельца 2026-09-29: пакет будет публиковаться на PyPI, приватность
  смысла не имеет); публикация пакета `vesmaro-cortex` на PyPI — после вердикта
  оценки A5.
- **Лицензия:** Apache-2.0 (решение владельца 2026-09-29; выбор из
  Apache-2.0/MIT делегирован TL — Apache-2.0 за патентную защиту и единый
  семейный дефолт).
- **Дисциплина данных:** ни одной сырой строки стора в репе; секрет-скан и
  `no-federate` до экспорта любых данных (гигиена препрега v2).

## Quickstart

```bash
git clone https://github.com/vesmaro/vesmaro-cortex && cd vesmaro-cortex
uv sync                                   # дефолтная среда (без torch)
uv run pytest tests/ -q                   # контрактные тесты (N-тесты скипаются)
uv run python scripts/smoke_pipeline.py   # CPU-смоук полного контура (~5 с)

uv sync --extra train                     # + CPU-torch: нейроголова N
uv run --extra train pytest tests/ -q     # все тесты, включая N
uv run --extra train python scripts/smoke_pipeline.py --with-n   # полный смоук (~90 с)
```

XPU-прогон — опциональный walkthrough для владельца:
[docs/runbooks/xpu-walkthrough.md](docs/runbooks/xpu-walkthrough.md)
(оценка A5 от него не зависит, ADR 0001 V6).
