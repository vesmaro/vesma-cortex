# Дельта-карта тренера conflict-detector (аддендум к диспатч-брифу 06.11)

- **Назначение:** карта «что переиспользовать из `merge_arbiter_train.py`
  (4164 строк), что менять» — снижает риск импровизации ML-ланы 06.11.
  Это СПЕКА-КАРТА, не код: реализация остаётся за `gcw-ml-engineer`
  (ML-ONLY); TL верифицирует.
- **Ратифицированные решения MLC-9, которые карта учитывает:** cut 0.5 без
  сдвига меток; version-guard = кодовая нога (P0 E4, легальность «нога-
  эшелон» §8.5 пакета); мерин = пайплайн guard+model; merge-v2 НЕ метчик.
- **Проверено 04.11:** фикс каскада W2 `2a45f28` (floor_perm — различные
  перестановки на сид) уже в анцестрии ветки — машина полов в
  `merge_arbiter_train.py` на этой ветке уже содержит его.

## 1. Переиспользуется почти дословно (ссылки на строки merge_arbiter_train.py)

| Блок | Строки | Что взять |
|---|---|---|
| Paths + env pins (P2 fail-closed) | 119–166 | пин эмбеддера nano `3b752e06…` уже в pin.json; wrong provider refuses loudly (A-frozen) |
| InstrumentAbort + run-log append-only | 341–395 | log/sha256_file/sha256_text/find_doc_for — как есть; run-log: `docs/experiments/conflict-detector-train-runlog.md` |
| Prologue (ML-STD-0003 П1–П6, П9) | 446–565 | дисциплина одноразового holdout + seal-сверка перед чтением + событие чтения в run-log |
| Машина полов | 1462–1686 + 2618–2993 | cmd_floor_seed 1515 + **полы v2 (ML-STD-0004 declared floor): cmd_floors_v2 2642**, floor_perm-механика 2548–2572 — фикс `2a45f28` (РАЗЛИЧНЫЕ перестановки на сид) в этих блоках, подтверждено grep+git show 04.11 |
| fit_booster / predict_scores | 1176–1187 | LightGBM-нога как есть (цикл 57–78 c, <2 ГБ — прецедент B4) |
| Проверка базлайнов против манифеста | 1404–1462 | паттерн check_baselines_vs_manifest — переадаптировать на BL-band 0.49 / BL-digit 0.57 / BL-digit+guard 0.66 |
| Производственный контур (концепт) | 1686–1743 | arbitrate_production = «кодовые ноги + модельная нога»; для №6: ЕДИНСТВЕННАЯ кодовая нога = version-guard |
| Sanity через производственный контур | 1743–1840 | run_model_sanity_suite — зеркалить на 7 проб §6.3 (ниже) |
| Freeze артефакта | 1835–1986 (freeze_artifact_v2 2156) | bundle-формат зеркалить на vesma-cortex-merge-v2 (60d1fe73), не на onnx-бандл v1 |
| **Sealed holdout + single-shot гейт** | 3430–4057: cmd_gate 3430, **_read_holdout_sealed 3740**, _eval_gates_on_holdout 3904 | машина одноразового чтения (seal-сверка, событие в run-log) и конъюнкции гейтов — зеркало почти дословно; prior_single_shot 3435 |
| Guard-off образец | _zone_off_probe 3140 (в _prelaunch_v2 3207) | зеркалить как guard-off пробу постратно (минимум NC-3), число честности |
| Sanity-сабкоманд | cmd_sanity 3412 | раннер по замороженному артефакту — держать тот же интерфейс |

## 2. Меняется (дельты arbiter → conflict)

1. **Загрузчик корпуса:** кластерные ряды (729–803) → парные ряды §5.1
   (`pair_id, stratum CD-1..6/NC-1..5, label contradicts|not-contradicts,
   envelope_a/b, measured, declaration, orientation, split`). Train =
   1200 строк = 600 пар × 2 ориентации (A-symmetry), дедуп по
   (pair_id, orientation). **Тренер holdout НЕ загружает** — holdout
   открывает только раннер гейтов (одно чтение).
2. **Фичи:** Choice v2-блок (835–1147: VecLeg, e_cos/zeps, zone metrics)
   **ВЫКИНУТЬ** — это прибор другой задачи. Фичи = замороженные **CORE-13**
   `cortex.features.pair` (gate_contract FEATURE_NAMES, включая cos_target;
   подтверждено импортом 04.11: ровно 13 имен, порядок == gate_contract).
   **FIELD_COSINE_FEATURES — ablation-расширение pair.py, в контур №6 НЕ
   входит** (pair.py:85: core ИЛИ core+field-cosine — берём core только).
   0 реимплементаций (#480); конвенция поверхности
   `n1(title)+\n+n1(body)+\n+n1(tags)`, усечение 4096.
   `is_version_token` и `declaration` — construction-only, в модель не
   едут (read-on-inference запрещён, §2.3.2 пакета).
3. **Кодовая нога контура:** арбитрские code-legs → **version-guard**
   (cg-guard-v1, предикат NC-3 §2.2/§2.3 пакета: все non-equal регионы
   из version-namespace токенов → NOT contradicts безусловно).
   Односторонний (только NOT), константы пинуются с артефактом,
   `guard_version` в metadata (MLC-9 п.14).
4. **Правило решения:** НЕ cv_thresholds (1213–1396) — ратифицирован
   **cut 0.5 без сдвига меток**, без zone-эшелона (зона — прибор Choice).
   CV-механику можно оставить как characterization (число честности),
   не как подбор порога.
5. **Базлайны:** BL-band / BL-digit / BL-digit+guard (§6.1 пакета) —
   три простых правила, новых ~60 строк; НЕ арбитрские per-stratum.
   max(BL) = 0.66 — цель g_ba.
6. **Sanity §6.3 (7 проб, до holdout):** self-пара ≤0.1 · cosmetic-twin
   NOT · envelope-вариант NOT · razor CD-1 → contradicts P≥0.5 ·
   version-сосед NOT (проба 4+5 = двусторонний certify) · stability-
   лестница (3 ступени) инвариантна · translation-твин — оба вердикта.
   Каждая — через ПРОИЗВОДСТВЕННЫЙ контур (guard+model). Источник
   фикстур — ПЕРВИЧНО префлайт-режим генератора
   (`build_conflict_detector_corpus.py`, «П3 preflight: one fixture cell
   per class», ~стр. 589): по ячейке на каждый класс. `measure-probes.json`
   (семейства far/hard-neg/hard-pos/razor) — только контекст лестницы cos:
   **translation-семейства там НЕТ** (проверено 04.11) — пробу 7 брать из
   кросс-языковой страты генератора. Точный выбор фиксирует run-log ДО
   прогона.
7. **Гейты §6.2:** гейтуют ПАЙПЛАЙН (примечание 1 брифа); агрегаты с
   эксклюзией razor-коллизионных ячеек (примечание 2); policy-константные
   g_nc3_exact/g_razor/g_self/g_far/g_disj без пола; guard-off проба
   постратно (минимум NC-3) — число честности в отчёте.
8. **Артефакт:** `models/conflict-detector-v1/` по образцу bundle
   merge-v2 (60d1fe73): manifest.json (sha256 весов, feature_set_sha256,
   guard_version, pins, полы) + веса + MODEL_CARD.md; typed_output 100%.

## 3. Не входит в тренер (чужие поверхности)

Минт-поверхность (`manager.mint_contradicts_edge`, conflict_minting.py) —
движок, post-ADOPT шаг диспатча (§5 брифа). Кластерные приборы Choice
(ZONE_V1, VecLeg) — не переиспользуются.
