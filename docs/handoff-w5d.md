# Бриф W5d: подключение vesma-cortex-v1 в движок vesma (VesmaProvider)

> От TL-сессии vesma-cortex к исполнителю W5d (движок vesma / канон).
> Дата: 2026-10-01. Вердикт калибровки: **ADOPT** (все 6 условий замороженного
> правила, single-shot по запечатанному holdout 300; отчёт:
> `docs/experiments/calibration-ds1000-a5.md`). Артефакт готов к бандлингу.
> Этот документ — ЕДИНСТВЕННЫЙ бриф на подключение; инференс-спека —
> технический контракт.

## 0. Состояние сторон (пост-ребрендинг, проверено 2026-10-01)

| Сторона | Где | Состояние |
|---|---|---|
| Модель-репа | `github.com/vesmaro/vesma-cortex` (публичная, Apache-2.0; локально `Project-Vesma/vesma-cortex`), main `b29f045` | A0–A6 закрыты, ADOPT |
| Движок | `Project-Vesma/vesma`, main `b03ae74` | обновлён владельцем; python-пакет — `src/vesmaro/` (import НЕ переименовывался) |
| Артефакт | `vesma-cortex/docs/../data/stage2/artifact-ds1000/` — `model.onnx` + `model.manifest.json` (локальные, в репу не коммитятся) | sha256 `281bd0fd39bf9935c86a8b32fed68191fa9b7da95a83b53eb10844a5fd100ac7`, 97 738 байт |
| Канон | `vesmaro/vesma-canon`, аддендум 2 = `8b24212` | пороги/правило не менялись |

Именование: артефакт и обёртка — **vesma-cortex-v1 / VesmaProvider**
(переименовано владельцем из mnema-*; рефакторы `45b4297`, `c341698`).
Пакет cortex переименован в `vesma-cortex` (PyPI-публикация — после
решения владельца). Пакет движка (`vesmaro`) и стор (`~/.mnemos/data`) —
переименование запланировано отдельной волной движка (окно 6.0, бриф у TL;
пока ссылки спеки на `src/vesmaro/...` валидны, при переименовании —
синхронизировать три скрипта cortex: a2_field_cosines, s1_synth_similarity,
store_export).

## 1. Что подключаем

`vesma-cortex-v1` — градиентный бустинг (кандидат D, конфиг `d-l7-lr010`)
над 13 замороженными признаками пары: вход — канонический JSON
(`{record, candidate, similarity}` — два `CanonRecordView`-среза +
измеренный косинус), выход — `probability ∈ [0,1]` (P-duplicate).
Вариант N (нейроголова с векторами) НЕ подключается: победил D — аддендум
ADR 0004 о передаче векторов не требуется. Полный контракт:
`vesma-cortex/docs/specs/inference-v1.md`.

## 2. Шаги подключения (движок)

1. **Бандл:** `src/vesmaro/models/vesma-cortex-v1/{model.onnx, manifest.json}`
   — скопировать артефакт из cortex-репы (см. §0), manifest по образцу
   бандла `vesma-embed-v1`; fingerprint весов = sha256 файла.
2. **`VesmaProvider`** (рядом с `DeterministicProvider` в
   `src/vesmaro/decision_provider.py`) — загрузка по спеке §6, шаги 1–8:
   eager init; валидация `metadata_props` (name=`vesma-cortex`,
   version-мажор 1, `feature_set_sha256`); гейт ≤5 МБ; sha256 в телеметрию;
   ассерт `embedder_pin` (`nano:sha256:3b752e06…`) против живого fingerprint
   эмбеддера — несовпадение = `CORTEX-E-PIN`, громкий отказ (событие
   перекалибровки, не штатная деградация); ORT CPU-сессия
   (`VESMARO_ORT_THREADS`); smoke-inference на старте.
3. **Сборка признаков** в обёртке: 13 core-фич по замороженному
   `FEATURE_NAMES` (пин в metadata артефакта), детерминированный python,
   ноль сети; полевые косинусы — gated OFF (не реализовывать).
4. **Включение за флагом:** `decision_provider` получает реализацию
   `vesma`; дефолт остаётся `deterministic` — флип флага = решение
   владельца после опытной обкатки.
5. **Fail-open (спека §7):** любой `CORTEX-E-*` → деградация на
   `DeterministicProvider` + машино-парсируемый warn; ingest не
   блокируется. Порог применения вердикта — политика W5d (конфиг), не
   артефакт.
6. **Тесты:** AST-изоляция сети (паттерн `test_mcp_core_isolation`);
   unit на загрузку/валидацию metadata/код ошибок; интеграционный
   smoke-вердикт на реальных записях; полный сьют движка зелёный; docs
   (EN/RU) синхронно.

## 3. Справочные числа (holdout 300, single-shot)

Sensitivity 1.0 · Specificity 0.98 · Brier 0.0053 · Balanced accuracy 0.99
против baseline 0.81 (косинус 0.92 тем же раннером) · типизированность и
record-quality 1.0. Прруф-отчёты: `docs/reports/vesma-cortex-model-report.md`
(модель), `docs/reports/vesma-embed-model-report.md` (эмбеддер).

## 4. Не требуется / отложено

- Векторный аддендум ADR 0004 (нужен только варианту N) — снято с гейтов;
  при будущем N — бандлить вместе с graph-evidence.
- `Noul.confidence` — рекомендация: не расширять (probability достаточно).
- record-quality — плейсхолдер 0.5/0.5 (NO-DATA-дисциплина препрега).
- Публикация пакета `vesma-cortex` на PyPI — после отдельного решения
  владельца; на подключение не влияет.

## 5. Приёмка W5d

Артефакт грузится (пин совпал) → smoke-вердикт осмыслен → fail-open
проверен (битый артефакт → DeterministicProvider + `CORTEX-E-LOAD`) →
полный сьют зелёный → docs EN/RU синхронны → флаг выключен по умолчанию,
владелец решает момент включения.
