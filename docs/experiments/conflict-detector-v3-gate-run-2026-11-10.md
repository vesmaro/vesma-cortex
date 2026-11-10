# B6-conflict — SINGLE-SHOT гейт §6.2 на CD-holdout-v3: ВЕРДИКТ DECLINE (крас g_stab), F1-финал волны

**Дата/время (UTC):** 2026-11-10T02:50:05.681Z (gate_start) → 02:50:11.063Z (gate_done), wall 6.1 s
**Ветка/HEAD:** `feat/conflict-detector-corpus-design` @ `23fd9b1` (чистое дерево до прогона; пресет v3 = коммит 23fd9b1, seal3 HEAD 110a699 в анцестрии)
**Раннер:** `scripts/train_conflict_detector.py gate --corpus-dir data/conflict-detector/corpus-v3 --execute-single-shot` (пресет v3: seal CD-holdout-v3 `8e90729c…`, pin-v3 `6a91d73c…`, artifact `models/conflict-detector-v2` НЕИЗМЕННЫЙ (F3), пороги §6.2 без правок), host `core-51`, venv `python 3.12.13 / lightgbm 4.7.0 / numpy 2.5.3`, `taskset -c 0-12`, PYTHONPATH append `vesma/src`
**Run-log:** `docs/experiments/conflict-detector-train-runlog.md`, slice `conflict-detector-v2-gate-v3` (свежий П8; v2-слайс `conflict-detector-v2-gate` остаётся потраченным с 2026-11-10)
**Машинное состояние:** `data/conflict-detector/measure/gate-v3-single-shot-2026-11-10.json` (46 событий, numbers-only копия)

## 0. Подготовка полосы (до прогона, метаданные)

- Пресет v3 добавлен в `INPUT_PRESETS` (коммит 23fd9b1) — ввод-пайпинг по образцу v2: путь поезда (corpus-v2 `train.jsonl`, F7 pool-only), слайс гейта `-v3` (floors-слайс не тронут — binding `floors_published` сохранён), обобщение трипваера/активации на pinned-пресеты, sealed-read отвечает ТЕКУЩЕМУ пин-набору + событие `seal_verified`, публикация G3-сплита `_g3_split` (не гейт). Семантика (фичи/guard/cut 0.5/контур/пороги) не тронута.
- Preflight ДО прогона: 43/43 чека зелёные (метаданные; запечатанные поверхности не открывались).
- Фальстарт запуска (честность): первая попытка старта упала на `taskset`: относительный путь `.venv/bin/python` не существует в worktree (venv живёт в корне vesma-cortex) — python НЕ стартовал, run-log не писан, запечатанные поверхности не тронуты (0). Перезапуск с абсолютным путём venv — успешный; единственное `sealed_read_once` в истории v3 — из успешного прогона.
- Run-log событие `input_preset_v3_added` (02:49:08.713Z) — ДО gate_start.

## 1. Пролог П1-П9 — весь зелёный, строго до gate_start

| Событие | Факт |
|---|---|
| swap_check | available 14425 MB, ram_overrun_stop=false |
| env | venv_suffix ok, python 3.12.13, lightgbm 4.7.0, numpy 2.5.3 — mismatch: [] |
| host | core-51 == expectation |
| byte_tripwire | train `a549dc0c…` == пресет == pin-v3 `train_v2_untouched`; pin-v3 `6a91d73c…` == bytes == build-manifest; seal-файл `8e90729c…` == пресет == build-manifest fp.seal; holdout metadata pin `548eae2a…` == seal json == build manifest == пресет; seal n=180 ids=180; embedder `3b752e06…`, guard `cg-guard-v1`; mismatch_hard=[] seal_metadata_mismatch=[]; sealed_file_opened=false |
| activation | spec RATIFIED-маркер + ADDENDUM-6 маркер `ADDENDUM-6 РАТИФИЦИРОВАН` (sha текста cd46f292…); анцестрия: dispatch 9706158, delta-map 1c85715, gold-closed 45597c5, std0004-formula 5305c82, corpus-v3-build 110a699 — ok=true |
| prereg_order_assert | floors_published (conflict-detector-v2-floors @ 2026-11-09T22:43:27Z) привязан к sha артефакта `2534fcbe…`; delta_acc_declared 0.10 воспроизводится из p5/p95; delta_inv 0.0 |
| prior_single_shot | слайс `conflict-detector-v2-gate-v3` — 0 sealed-событий до прогона (свежий) |
| gate_start | 2026-11-10T02:50:05.681Z |

## 2. Пре-sealed фазы — зелёные

- Sanity §6.3: **6/7 legs + 1 known-uncovered** (ADDENDUM-4 @ 0971639). Проба 7 leg-2 опубликована, не ассертится: verdict not-contradicts, p=0.008105 (cos 0.7738). Диагностика true-translation (honesty, вне корпуса): not-contradicts, p=0.012081.
- Проба 4 (razor CD-1): contradicts, leg model, p=0.998386. Проба 5 (version-сосед): NOT, leg guard exact — пара 4+5 = certify-экзамен зелёный (→ g_razor).
- E8-iii canary para2: 7/7 selected-ячеек остались contradicts (p=0.9995–0.9997).
- E8-iv para1 sweep: 3/3 слов — contradicts (p=0.9946–0.9997, диагностика).
- g_gold_state: CLOSED 48/48 ≥ 45 (синтез 45597c5) — не пересчитывался.
- cos_clamp в пре-sealed: 1 событие, overshoot 2.22e-16 (f64- rounding scale), over_floor=0 — зелёный ассерт пройден.

## 3. Sealed чтение — состоялось (single-shot ПОТРАЧЕН)

- `seal_verified` @ 02:50:06.240Z: sha256 файла `548eae2a9655c57f…` == dispatch-пин == build-manifest fingerprint (текущий пин-набор). Диагностика: artifact freeze-time pin `c92a2caa…` (предок v2) matches=false — легально при ротации (ADDENDUM-5 §4 F1-F12: тот же артефакт, новое поколение печати).
- `sealed_read_once` @ 02:50:06.245Z: **n_pairs=180, n_rows=360, sha256 `548eae2a…`** — совпал с seal json == build manifest == пресет (сверка до открытия). Единственное чтение CD-holdout-v3; печати v1 (девственна) и v2 (сожжена) не тронуты.

## 4. ТАБЛИЦА 11 гейтов §6.2 (конъюнкция; пороги не менялись)

| Гейт | Норматив | Факт | Вердикт |
|---|---|---|---|
| g_sens (CD-агрегат, эксклюзия razor-коллизий) | ≥ 0.90 | 1.0 (39/39 пар; Wilson95 [0.9103, 1]) | OK |
| g_spec (NC-агрегат, та же эксклюзия) | ≥ 0.90 | 1.0 (63/63 пар; Wilson95 [0.9425, 1]) | OK |
| — нож NC-1 | ≥ 0.85 | 1.0 (20/20 пар, transfer-only) | OK |
| — нож NC-2 | ≥ 0.85 | ножа НЕ СУЩЕСТВУЕТ на агрегате: все 29 пар NC-2 — razor-коллизионные ячейки (cos ≥ 0.95), эксклюированы; certify-зона: g_razor + g_nc3_exact + g_gold (§6.2 прим. 2) | disclose (vacuous) |
| g_nc3_exact | exact 0 | 0 contradicts-предложений на 32 строках NC-3 (16 пар; CI [0, 0.1936]) | OK |
| g_razor (обе ноги exact, пробы 4+5 через контур) | exact | positive: contradicts, leg model, p=0.998386; negative: NOT, leg guard exact | OK |
| g_self | ≤ 0.10 | 0.000227 | OK |
| g_ba vs max(BL) на holdout | ≥ δ_acc=0.10 | acc_model 1.0 vs max(BL) 0.6176 (BL-band 0.5196 / BL-digit 0.6176 / BL-digit+guard 0.6176 — пересчитаны раннером на 102 парах агрегата); delta 0.3824 | OK |
| g_brier | ≤ 0.25 | 0.0 (model-leg вероятности на агрегате; guard-строки несут код-вердикт) | OK |
| g_stab (3 ступени cosmetic/para1/para2) | ≥ 1 − 2·floor_inv = 1.0 | **0.5833 (35/60 пар инвариантны)** | **RED** |
| g_far (cos < 0.40, разные темы) | < 0.5 | far-строк 0 (b40-полоса пуста по построению резки) — vacuously true | OK (vacuous, disclose) |
| g_disj (train↔holdout v3, id-level vs seal ids) | exact 0 | 0 пересечений | OK |
| g_gold | CLOSED | CLOSED 48/48 (синтез 45597c5) | OK |

**gate_done @ 02:50:11.063Z: verdict = DECLINE, red = [g_stab].** Конъюнкция провалена одной ногой. Модель-одиночная диагностика: guard-off NC-3 (ниже) — без guard модель дала бы 18 contradicts-предложений на 32 NC-3 строках (p ≥ 0.5 при cut 0.5); g_self/g_brier — модельные числа опубликованы в таблице.

## 5. Эксклюзия razor-коллизий (§6.2 прим. 2)

- Исключено **156 строк = 78 пар** из агрегата: CD-1 22 + CD-2 11 + NC-2 29 + NC-3 16 — все строки этих страт в razor-бэнде cos ≥ 0.95 (same-point ячейки razor CD-1/CD-2 ↔ NC-2/NC-3).
- Агрегат гейтов: **102 пары** (CD-3 11, CD-4 10, CD-5 7, CD-6 11, NC-1 20, NC-4 11, NC-5 32). Razor-зона сертифицируется пайплайном: g_razor (обе ноги exact) + g_nc3_exact (0) + g_gold (CLOSED) — все зелёные.

## 6. G3-сплит (ADDENDUM-6 §5 G3/ML-E1; pin-v3 transfer_split) — обязательная публикация, НЕ гейт

Маркировка transfer: pair id prefix `ho3-` (новые банки gen_pool_v3). Ноги наследуют эксклюзию агрегата (NC-2/NC-3 — целиком razor-коллизионные, в сплит не попадают; их сертификация — g_nc3_exact/g_razor/g_gold).

| Нога | spec | NC-пар (rows) | По стратам |
|---|---|---|---|
| g_spec_indist | 1.0 (52 rows NC) | NC-4 11 (22), NC-5 15 (30) | обе 1.0 |
| g_spec_transfer | 1.0 (74 rows NC) | NC-1 20 (40), NC-5 17 (34) | обе 1.0 |

- **NC-1 — transfer-only: in-dist ноги НЕ существует (disclose, G3/ML-E1).**
- NC-3: сплит-ноги на поверхности агрегата отсутствуют (6 transfer / 10 indist пар — все в razor-эксклюзии); exact-0 пайплайна на ВСЕХ 16 парах подтверждён g_nc3_exact.
- NC-4: целиком из СТАРОГО банка (G7, classic indist), fallback 0 — disclosed.
- Интерпретация: transfer-поверхность новых банков НЕ является дефектом — spec transfer 1.0 (74/74 rows), в т.ч. unseen-scaffold ряды расширенных классов (ADDENDUM-6 §2.4 disclosure). Крас гейта — НЕ на transfer-ногах.

## 7. Per-strata таблицы (агрегат 102 пар; пары = ориентации/2; Wilson95 95%)

| Страта | n пар | hits | доля | Wilson95 | примечание |
|---|---|---|---|---|---|
| CD-3 | 11 | 11 | 1.0 | [0.7412, 1] | sens |
| CD-4 | 10 | 10 | 1.0 | [0.7225, 1] | sens |
| CD-5 | 7 | 7 | 1.0 | [0.6457, 1] | sens; **n<10 → CI, descriptive-only (F9)** |
| CD-6 | 11 | 11 | 1.0 | [0.7412, 1] | sens |
| NC-1 | 20 | 20 | 1.0 | [0.8389, 1] | spec; transfer-only (G3) |
| NC-4 | 11 | 11 | 1.0 | [0.7412, 1] | spec; indist (old bank, G7) |
| NC-5 | 32 | 32 | 1.0 | [0.8928, 1] | spec; transfer 17 / indist 15 |

Ни один вердикт не держится на ячейке n<10 (F9): CD-5 — descriptive с CI; нож NC-1 (n=20) и агрегаты (39/63) выше ценза.

## 8. Guard-off проба (MLC-9 п.13 — число честности, не гейт)

- На holdout guard-fire строки: **только NC-3: 32 строки, модель без guard предложила бы contradicts на 18** (model_proposals_guard_off=18) — guard (cg-guard-v1) подавляет их все; в производственном контуре NC-3 = 0 contradicts (g_nc3_exact подтверждает). Другие страты: guard не firing (0 строк).

## 9. cos_clamp ledger (ADDENDUM-5 F5/D1)

- **5 клампов за прогон, все sub-floor:** overshoot 2.220446049250313e-16 ≤ f64 noise floor 5.457e-12 (бAND 1+max(1e-3, 64·384·eps32) = 0.00293). over_floor = **0** — eps32-класс арифметики инцидента v2 НЕ воспроизвёлся. Сайты: fixture:NC-4-lang (1, пре-sealed sanity — известный benign 1-ulp self-parallel rounding), g_stab:base ×2, g_stab:cosmetic ×2. Каждое событие видно в run-log (nothing silent). Нан-ов нет, fail-closed банд не нарушен ни разу.
- Пост-read try/except (D2) в боевом режиме не понадобился: краёв не было; `gate_done` записан.

## 10. Вердикт волны и интерпретация (якорь G4)

- **ВЕРДИКТ §6.2 (конъюнкция): DECLINE** — крас единственной ноги g_stab: agreement 0.5833 (35/60 пар) при нормативе 1.0 (floor_inv=0, порог 1 − 2·0.0). Пары сэмплированы по всем 180 (seed 21); инвариантность вердикта base/cosmetic/para1/para2 нарушена на 25 парах из 60.
- **Интерпретация по якорю G4 (ADDENDUM-6 G4, char2 a9ac15d): characterization GREEN + гейт RED = дефект модели.** Уточнение якоря: крас не на transfer-ногах (специфичность на transfer 1.0 — новые банки/ unseen-scaffold модель читает корректно), а на СТАБИЛЬНОСТИ модельного вердикта поверх реальной holdout-поверхности: фикстурные лестницы (char2, sanity проба 6) инвариантны, а на holdout-строках парафраз/косметика envelope_b переводят p через cut 0.5 на 25 из 60 пар сэмпла. Класс дефекта: хрупкость модельной ноги под поверхностными возмущениями (переуверенные p ≈ 0/1 при cut 0.5 + чувствительность фичей к para-возмущениям) — вопрос MLC (калибровка/фичевая поверхность), НЕ корпуса и НЕ банков.
- **F1: красный v3 = ФИНАЛ волны.** Single-shot потрачен (слайс `conflict-detector-v2-gate-v3`), повторов/перекалибровок/подгонок не будет; v4 на том же артефакте+порогах невозможна (ADDENDUM-5 F1). Следующий ход — TL/MLC.
- **MINТ НЕ ЗАПУСКАЕТСЯ** (ADOPT не достигнут — монетный шаг санcionирован только зелёной конъюнкцией + TL-верификацией).

## 11. Что НЕ делалось (честность)

- Повторных чтений holdout v3 — ноль; одного `sealed_read_once` в run-log, single-shot spent.
- Печати v1 (девственна) и v2 (сожжена) не открывались; holdout-контент никуда не копировался.
- Артефакт `models/conflict-detector-v2` (onnx `2534fcbe…`), floors, pin-v3, seal-файлы — не изменялись (F3 проверена).
- Пороги §6.2 не пересматривались; никакие числа не «улучшались» — red опубликован как измерен.
- Фальстарт запуска (taskset/относительный venv-путь) — до старта python; зафиксирован в §0.

## Пути

- Run-log (раннер-писанный): `docs/experiments/conflict-detector-train-runlog.md` (slice `conflict-detector-v2-gate-v3`, 45 событий от trainer_input_params до cos_clamp run_summary)
- Машинное состояние цикла (numbers-only копия, gitignored): `data/conflict-detector/measure/gate-v3-single-shot-2026-11-10.json`
- Stdout (volatile): `/tmp/b6_gate_v3_stdout.log`
- Отчёт: `docs/experiments/conflict-detector-v3-gate-run-2026-11-10.md`
- HEAD прогона: `23fd9b1` (родитель 110a699 — seal3)
- Итог: **VERDICT DECLINE — крас g_stab 0.5833 < 1.0; F1-финал волны; эскалация TL/MLC**
