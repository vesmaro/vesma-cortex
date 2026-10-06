# Корпус v4.3 (round-4 поезд) — рукопожатие

- **Статус:** корпус собран и запечатан; ОБУЧЕНИЕ НЕ ЗАПУСКАЛОСЬ (prereg ACTIVATED, PR #22; пороги g-T — на ратификации)
- **Дата:** 2026-10-07
- **Fingerprint корпуса:** `` (train/hold — в manifest.json fingerprints)
- **Эмбеддер-пин:** nano:sha256:3b752e06… — вся геометрия единая (переиспользованные пары re-embedded)
- **Состав:** 2675 пар → train 2399 + v4.3-holdout 276 (запечатан; ids в gitignored holdout-ids.json)
- **by_part (train):** real None · translated None (320 построено: 288 train + 32 в v43-holdout по 10% страт-сплиту) · synthetic-b2 None · synthetic-v42 None
- **Гейты:** corner-QA violations 0; translated квоты CLEAN; watchlist-family в train None; дизъюнктность train↔holdouts OK (144 real-коллизий с B2-holdout дропнуто, счётчик в манифесте)
- **Данные:** data/stage2/v43/ + ../v43-holdout/ — gitignored (тексты стора); в репо — этот манифест + скрипт
- **Детерминизм:** scripts/build_corpus_v43.py; dry-run и полный прогон дали байт-одинаковый состав и отпечатки (TL-прогон лично 2026-10-07)
- **Prereg:** docs/specs/embed-round4-prereg-* (ACTIVATED); g-T1/T2/T3 (0.90/−0.02/0.90) PROPOSED — ратифицировать ДО eval
