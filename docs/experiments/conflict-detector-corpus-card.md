# Dataset card — conflict-detector corpus v1 (CD-holdout-v1)

- **Линия:** ADR 0002 duty №6 (conflict-detector, Noul), фаза D.
- **Дизайн:** RATIFIED MLC-9 (`1acd88f`) + ADDENDUM-1 (`0917181`,
  перекройка лестницы по измеренному пределу прибора) + ADDENDUM-2
  (`8c756a0`, трудное окно [0.70, 0.75) как ограничение населения банда,
  конвенционные замеры) + measure-отчёты (`87a248e`, `deb440b`).
  Манифест: `data/conflict-detector/corpus-v1/build-manifest.json`
  (sha256 `0be52dd220c79281…` — см. fingerprints ниже).
- **Прибор:** vesma.embeddings.NanoProvider, пин nano `3b752e0671a50da5…`;
  guard `cg-guard-v1`. known-uncovered: cos < 0.527
  (измеренный floor корпуса; кросс-ссылка B4 ADDENDUM-6 §2 errata).

## Глоссарий таксономии (обязательная секция)

| Класс | Определение (предикат) |
|---|---|
| CD-1 | несовместимые значения одной числовой оси слота |
| CD-2 | несовместимые даты/сроки одного событийного слота тела |
| CD-3 | несовместимые состояния статусного словаря |
| CD-4 | полярность модальности над одной способностью |
| CD-5 | безусловное отрицание условного клейма на домене условия |
| CD-6 | качественное несовместимое предикат-пара (полярный оператор) |
| NC-1 | вставка слота-надмножества, конъюнкция выполнима |
| NC-2 | совместимое уточнение v |= c |
| NC-3 | version-namespace соседи — supersede-семантика, не конфликт |
| NC-4 | дельта метаданных при тождественном содержании |
| NC-5 | разные предметы (sibling/far), конфликтировать нечему |

## Состав (пары; строки = пары × 2 ориентации)

| Страта | train | seal (CD-holdout-v1) |
|---|---|---|
| CD-1 | 72 | 22 |
| CD-2 | 36 | 11 |
| CD-3 | 36 | 11 |
| CD-4 | 36 | 11 |
| CD-5 | 24 | 7 |
| CD-6 | 36 | 11 |
| NC-1 | 90 | 27 |
| NC-2 | 72 | 21 |
| NC-3 | 54 | 16 |
| NC-4 | 36 | 11 |
| NC-5 | 108 | 32 |

Банды train: {'razor': 448, 'b70': 35, 'b85': 53, 'b55': 62, 'b40': 2}. Базлайны (train): {'BL-band': 0.49, 'BL-digit': 0.57, 'BL-digit+guard': 0.66}.
Ассерты: 15/15 PASS.

## Провенанс и дисциплина

- 100 % authored-синтетика (ADR 0003 R2), 0 текстов стора, 0 LLM;
  метки by construction (A-labels), механическая карта §3.2.1 на каждой
  паре (A-silver); guard `cg-guard-v1`: 100 % NC-3 / 0 % CD
  (A-guard, construction-side only).
- Holdout: pool 600 → seal 180 (CD-holdout-v1), остаток
  не используется (A-disjoint); seal fp `1f676613de08fcdd…`.
- Null-конверт ≈1/3 both-null; документированное исключение: NC-4 corner
  twins (NC-4 corner twins are both-null exempt (label semantics requ…).
- Золотой лист: fp `3cf995028de31a30…` — статус: TL format gate, владельцу
  не выдавался.

## Fingerprints (sha256)

- `train.jsonl`: `0be52dd220c792810420c10921f30329ba6d903bfda5c02f9900ffdb8295e4aa`
- `holdout.jsonl`: `e8adfec9625b0f72ac001ed6024da2e3fd3bb5170169b6d93f26afde521b2567`
- `holdout-pool.jsonl`: `6982b66b33b632c770603b461e8901aa64fcc66c1fff4168009e4958564f0c87`
- `seal`: `1f676613de08fcdd6234c2119e311d2d9323605cdac3ceaeaa02d1e4bb5261d1`
