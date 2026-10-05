# Манифест корпуса dataset-v4 (B2-prime) — v4.2, волна D4-4 (2026-10-05)

Корпус собран и запечатан по ратифицированному плану
[dataset-v4-plan](../../docs/experiments/dataset-v4-plan.md), политике
[labeling-policy-b2](../../docs/specs/labeling-policy-b2.md) и препрегу
[calibration-b2p-prereg](../../docs/experiments/calibration-b2p-prereg.md)
§8 (Аддендум 5c ФИНАЛ). Метки — **по построению**: генератор знает, что
именно изменил, и ставит метку по тиру T0–T3. v4.2 = C1-коррекция по
ратифицированному диагнозу
[b2p-v41-sanity-fail-diagnosis](../../docs/experiments/b2p-v41-sanity-fail-diagnosis.md).

- **Статус:** sealed. Corner-QA гейт зелёный (счётчики ниже), holdout
  запечатан ДО обучения, лейблы holdout физически вне train-дерева.
- **Верификация:** `verify_dataset.py qa` — `ok: true`, 0 нарушений,
  exit 0; двойной прогон генератора байт-в-байт по всем 8 контент-файлам;
  `uv run pytest tests/ -q` — 383 passed / 3 skipped (honest skips);
  make lint — exit 0. Обучение НЕ запускалось (порядок §5 препрега:
  следующее действие — TL).
- **Гигиена:** ноль сырых строк стора — весь контент авторский
  синтетический (ADR 0003 R2); дизъюнктность с synth `TOPICS`,
  evalsets `EVAL_TOPICS`, sanity-якорями и LA-2-батчем закреплена
  тестом и проверкой на сборке; сеть не использовалась (эмбеддинг —
  локальный CPU-инференс, pin ниже).

## 0. Эволюция корпуса

| Версия | Волна | Пар | Что изменилось | Корпус-fingerprint |
|---|---|---|---|---|
| v4.0 | D4-1 | 868 | стартовый корпус (история B2-prime) | `ff2fb8f9…09b3` |
| v4.1 | D4-3 | 1036 | razor-ребаланс 192:192 + envelope-класс (препрег 5b) | `afa74585…aefc1` |
| **v4.2** | **D4-4** | **1100** | **+T3 same-length (28), +N-para-notdup (36); остальное v4.1 байт-стабильно** | `2edb8f71…d57e` |

## 1. Числа

| Пул | Пар | duplicate | not-duplicate |
|---|---|---|---|
| train | 824 | 431 | 393 |
| holdout | 276 | 144 | 132 |
| **всего** | **1100** | **575** | **525** |

Сплит — стратифицированный по (label × stratum): в каждой ячейке первые
`⌈0.25·n⌉` по сортировке `pair_id` уходят в holdout. Holdout читается
один раз — single-shot.

### По стратам (класс построения → тиры политики)

| Страта | Тир/класс | Метка | Пар | Holdout |
|---|---|---|---|---|
| P-identity | T0 self + байт-клоны | duplicate | 96 | 24 |
| P-cosmetic | T1 регистр/пробелы/пунктуация (192-баланс v4.1) | duplicate | 192 | 48 |
| P-envelope | whitelist §8.1 (конверт-дата, вне 13 фич — OQ-2) | duplicate | 48 | 12 |
| P-para-light | T4 light (своп + лексическая замена) | duplicate | 96 | 24 |
| P-para-sub | T4 substantial | duplicate | 48 | 12 |
| P-para-struct | T4 structural | duplicate | 35 | 9 |
| P-trans | переводные близнецы RU/EN | duplicate | 60 | 15 |
| N-metadata | T2 только метаданные | not-duplicate | 112 | 28 |
| N-fact-edit | T3 факт-токен, обе ориентации: 192 разной длины (v4.1) + **28 same-length (v4.2)** | not-duplicate | 220 | 55 |
| **N-para-notdup** | **v4.2: пересказ-НЕ-дубликат (одна same-length факт-правка + case-rewrite; косинус 0.4–0.7)** | **not-duplicate** | **36** | **9** |
| N-near | близкая тема, другой факт (hard-negative) | not-duplicate | 49 | 13 |
| N-far | далёкая тема (half кросс-язык) | not-duplicate | 108 | 27 |

### Два класса волны D4-4 (лечат красный сьют B2-v41)

1. **T3 same-length** — одно-символьные факт-правки с нулевой
   len-дельтой: **124/220 пар (56.4 %)** страты (порог gate ≥ 30 %);
   из них **16 пар на высоте сьютовской пробы** (`char5_containment ≥
   0.9764`, автосчётчик; в v4.1 было 2/192). Диагноз, корень (а):
   проба геометрически сидела в T1-облаке — теперь высоты заняты и
   негативами.
2. **N-para-notdup** — 36 пар «высокий char-оверлак (jac ≥ 0.9,
   containment ≥ 0.9) И низкий косинус»: кандидат = та же запись с
   одной same-length факт-правкой и case-rewrite'ом тайтла и тела.
   Регистр невидим нормализации фич (char-профиль остаётся
   дубликатным) но видим замороженному эмбеддеру: измеренный косинус
   **0.5365–0.6986** (2 пары в [0.40; 0.55), 34 в [0.55; 0.70)).
   Диагноз, корень (б): лестница монотонности получает обучаемый спад
   на дубликатном char-профиле.

### Распределение измеренного косинуса (vesma-embed-v1, CPU)

| Полоса | Пар |
|---|---|
| [0.40, 0.55) | 8 |
| [0.55, 0.70) | 155 |
| [0.70, 0.85) | 123 |
| [0.85, 0.95) | 108 |
| [0.95, 1.0) | 533 |
| 1.0 (точный угол) | 173 |

min = 0.450, max = 1.0; ниже пола B2 0.836 — 272 пары.

### Вариативность контрактных фич

- `record_type` / `language` заполнены на 100 % сторон; mismatch-пары:
  type 13.6 %, lang 13.3 % (≥ 3 % / ≥ 5 %).
- Ни одна контрактная фича не константа (все 13 имеют ≥ 2 значения).

## 2. Corner-QA отчёт (гейт ДО обучения)

| Гейт | Порог | Факт | Вердикт |
|---|---|---|---|
| Клоны-негативы (P0 §8.1, 0-толерантность) | = 0 | **0** | PASS |
| Corner-DUP (char4>0.99 ∧ body_len_delta=0) | ≥ 10 | **240** | PASS |
| Identity-класс позитивов | ≥ 60 | **240** | PASS |
| Metadata-only негативы | ≥ 40 | **112** | PASS |
| Fact-edit негативы, char5 ∈ [0.85;1) | ≥ 75 | **240** | PASS |
| **T3 same-length доля (v4.2, diagnosis C1-1)** | ≥ 0.30 | **0.5636 (124/220)** | PASS |
| **Low-cos high-overlap not-dup (v4.2, diagnosis C1-2)** | ≥ 30 | **36** | PASS |
| Razor-зона без позитивов (§5-G3) | = 0 | **0** | PASS |
| G1 фиче-потолок max(нег) ≤ max(поз) | все 7 | 1.0 ≤ 1.0 | PASS |
| G4 масса правки позитивов | 0 наруш. | **0** | PASS |
| record_type / language presence | ≥ 0.85 | 1.0 / 1.0 | PASS |
| type/lang mismatch доли | ≥ 3 % / ≥ 5 % | 13.6 % / 13.3 % | PASS |
| cos < 0.55 / < 0.70 / < 0.84 | ≥ 5 / ≥ 60 / ≥ 180 | 8 / 163 / 272 | PASS |
| Константные фичи | none | **none** | PASS |

Гейт живёт в `src/cortex/data/corner_qa.py`; пороги — секция
`corner_qa` в `gate_contract.json`. В v4.2 секция дополнена ДВУМЯ
presence-счётчиками (`min_t3_same_length_share`, 
`min_low_cos_high_overlap_notdup`) — пороговые значения НЕ
ревизировались, authority — диагноз b2p-v41; изменение уехало
отдельным коммитом вагона D4-4 (правило разделения). Лоадер
`thresholds_from_gate_contract` с fallback на замороженные константы;
когерентность пиннута тестом. Локальный прогон:
`scripts/verify_dataset.py qa data/stage2/dataset-v4` — `ok: true`.

## 3. Фингерпринты (data-contract §5)

| Объект | BLAKE2b-256 |
|---|---|
| **Корпус** (1100 пар, manifest.txt) | `2edb8f71ec2aabc7123b1bb897b5db141019b369615493219e05fa7e477cd57e` |
| Лейблы (pair_id → label) | `888b9b65d62e0edc81f3423a46a619ba52cc2a18e3af87ed03fa2c3b6da27639` |
| Train-подмножество (824) | `b45f385345526caeb3c11073d6e7aec410da89cc6e6bc43555737e43978af860` |
| Holdout-подмножество (276) | `66afeae65053924241240108204e4e7a050860e42d4b88e6f72359c657aefc48` |

Фингерпринтуемый объект пары: `{"record", "candidate", "similarity"}`;
`similarity` — измеренный косинус NanoProvider, pin
`nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba`.

## 4. Источники и воспроизводимость

Батчи авторского контента (sha256 на момент сборки; правка = новая
версия батча + пересборка + новое событие):

| Батч | sha256 |
|---|---|
| batch-base-ru.jsonl (48 тем RU) | `404d43f2…d89ba1482` |
| batch-base-en.jsonl (48 тем EN) | `526522bf…798ab4c293` |
| batch-near.jsonl (49 hard-neg) | `7c03701f…3545f6e7b` |
| batch-paras-1.jsonl (48 para-sub) | `12491501…527c795eea` |
| batch-paras-2.jsonl (35 para-struct + 12 trans2) | `842d6d24…6bd7af6af` |
| batch-facts.jsonl (158 спецификаций, 96 в работе) | `a1429d6f…3eef89` |
| **batch-facts-same-length.jsonl (14, new v4.2)** | `d5766d78…26f89b` |
| **batch-notdup.jsonl (36, new v4.2)** | `ece065de…c859d55` |

Оба новых батча несут same-length контракт `len(find) == len(replace)`
— валидируется на загрузке (loud refusal). Сборка:
`scripts/gen_dataset_v4.py` (детерминизм структурный, без RNG; двойной
прогон байт-в-байт по всем 8 контент-файлам — проверено). Все 1036 пар
v4.1 сохранили pair_id и содержимое (новые блоки добавлены после
N-far). Локальные артефакты (gitignored):

- `data/stage2/dataset-v4/` — train.jsonl, labels-train.jsonl,
  manifest.txt, holdout-ids.json (запечатан), corpus-report.json;
- `data/stage2/dataset-v4-holdout/` — holdout.jsonl, labels.jsonl,
  seal.json (вне train-корня; `assert_labels_isolated` +
  `assert_no_pair_overlap` — ноль пересечений id);
- `data/runs/b2p-v42/train-manifest.jsonl` — join train.jsonl +
  labels-train.jsonl для обучения, 824 строки, sha256
  `c37a8dde87f03b01b6bae06d62f01fbc02047e3c5c35f19051fc1ff5d1dbdf6f`.

Тесты: `tests/test_corner_qa.py` (включая два новых gate-теста D4-4),
`tests/test_dataset_v4_batch.py` (квоты v4.2, дизъюнктность, both
ориентации T3 на 220 парах, высотное покрытие, геометрия notdup,
байт-детерминизм, сплит 824/276).

## 5. Границы и что дальше

Обучение не входило в волну: следующий шаг за TL — обучение на
`data/runs/b2p-v42/train-manifest.jsonl` по порядку §5 препрега
(Аддендум 5c, commit-order зафиксирован там же), затем сьют sanity v2
и single-shot holdout (276 пар). Фиче-контракт, зонды sanity (код v2) и
пороги `sanity_v2` не менялись.
