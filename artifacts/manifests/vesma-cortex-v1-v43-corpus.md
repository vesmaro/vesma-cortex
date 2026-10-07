# Корпус v4.3 (round-4 поезд) — рукопожатие

- **Статус:** корпус пересобран (RFACT-SAMELEN ремонт, f-round4 REJECT диагноз) и запечатан; внутренний sanity после retrain — PASS; holdout v43 по-прежнему НЕ ЧИТАН (гейт TL)
- **Дата:** 2026-10-07 (фикс того же дня — wt/corpus-v43-redo)
- **Fingerprint корпуса:** `3537df85802d569326e5ab451c958aad9013f9fe12e4f9b8ea4991f79e15f08e` (labels `cd38c346…`, train `b4f01070…`, holdout_v43 `ced469ff…` — в manifest.json)
- **Эмбеддер-пин:** nano:sha256:3b752e06… — вся геометрия единая (переиспользованные пары re-embedded)
- **Состав:** 2675 пар → train 2399 + v4.3-holdout 276 (запечатан; ids в gitignored holdout-ids.json)
- **by_part (train):** real 1111 · translated 288 (320 построено: 288 train + 32 в v43-holdout по 10% страт-сплиту) · synthetic-b2 371 · synthetic-v42 629
- **RFACT-SAMELEN ремонт:** same-length семья собрана факт-токенными заменами (цифра↔цифра внутри дата/время/версия-токена; 1..4 токена по дрейфу char5) с жёстким пост-чеком `char5_containment < 0.976415` (высота сьютовской пробы) на каждой паре; селекция тел — по пригодности в детерминистском (created_at, hash) порядке (180 тел пропущено как физически неспособные дотянуть профиль до высоты пробы). Квоты не изменены: 160 same-length (RFACT) + 220 diff + 76 v4.2-FACT.
- **Числа после ремонта (замер features/pair, train):** same-length ALL 236 строк — медиана char5_cont 0.9725, max 0.9873, на высоте ≥0.9764 — 16/236 (уровень v4.2-эталона 16/236; все 16 — legacy v4.2-строки по дизайну); RFACT новые 0/160 ≥0.9764 (медиана 0.9733, было 160/160, медиана 1.0000).
- **Внутреннее санью (dataset-engineer, до сдачи TL):** retrain d-l7-lr005 (seed 1, frozen grid, train fp `b4f01070…`) → экспорт ONNX `fd37dbd1…` → sanity-сьют v2: **PASS 9/9** — fact_edit_twin 0.1526 (< 0.5; было 0.7789), monotonicity PASS (0.9960/0.9981/0.9989/0.9989/0.8386 — вне зоны не-возрастание), self_pair 0.9980, cosmetic_twin 0.9981, envelope_variant 0.9980, unrelated 0.0002.
- **Гейты:** corner-QA violations 0; translated квоты CLEAN; watchlist-family в train 36; дизъюнктность train↔holdouts OK (144 real-коллизий с B2-holdout дропнуто, счётчик в манифесте)
- **Данные:** data/stage2/v43-f/ + ../v43-f-holdout/ — gitignored (тексты стора); в репо — этот манифест + скрипт
- **Детерминизм:** scripts/build_corpus_v43.py; dry-run, прогон-1 и прогон-2 (v43-f-recheck) дали БАЙТ-одинаковые train.jsonl + все отпечатки (train-байты сравнены sha256; 2026-10-07)
- **Prereg:** docs/specs/embed-round4-prereg-* (ACTIVATED); g-T1/T2/T3 (0.90/−0.02/0.90) PROPOSED — ратифицировать ДО eval