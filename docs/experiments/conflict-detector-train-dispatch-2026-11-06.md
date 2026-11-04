# Диспатч 06.11 ≥06:00 — обучение conflict-detector → single-shot → ADOPT (волна B6-conflict)

- **Статус волны (на 04.11):** корпус + золото + миграция ЗАВЕРШЕНЫ
  (main движка `4c417d2`, PR #536). Осталось: обучение → sanity →
  single-shot гейт §6.2 → ADOPT → авто-минт → каскад. Этот документ —
  бриф диспатча (долговечный носитель; /tmp-брифы не переживают ребут).
- **Точка входа памяти:** авто-память `b6-conflict-detector-wave.md`
  (итоговый блок) + чекпоинты task:b6-conflict.
- **Роли:** диспатчер = Tech Lead. ML-шаги — только `gcw-ml-engineer`
  (ML-ONLY); TL лично верифицирует числа агентским раннером на агентских
  данных; вердикт гейтов — конъюнкция §6.2.

## 0. Идемпотентность (первый шаг всегда)

Поиск по памяти (`vesma_search`: «conflict-detector ADOPT training»,
«conflict-detector train runlog»), `ls wt/conflict-detector-draft/models/`
(артефакт conflict-detector), `docs/experiments/conflict-detector-train-*`.
Если тренинг/ADOPT уже исполнены другой сессией — одна строка и стоп,
ничего не дублировать.

## 1. Дисциплина чтений

- Holdout **CD-holdout-v1 читается РОВНО ОДИН раз раннером**. До чтения:
  полы ML-STD-0001 (§3.2 брифа) + sanity §6.3. Событие чтения пишется в
  run-log ДО вычисления гейтов.
- Чужие печати не открывать: holdout v4.5, g4-spec-v1, MA-holdout-choice
  (другая label-поверхность — прецедент fusion-framing).
- Смена весов после чтения holdout = перекалибровка (eval-methodology §6).
  **FAIL любого гейта → DECLINE + разбор + STOP + чекпоинт. НЕ
  импровизировать** — следующий ход TL/MLC, holdout v1 сожжён.

## 2. Входы (пре-флайт TL пройден 04.11)

| Артефакт | Путь (от `wt/conflict-detector-draft/`) | Контроль |
|---|---|---|
| Train 600 пар | `data/conflict-detector/corpus-v1/train.jsonl` | sha256 `0be52dd2…e4aa`, 1200 строк, конверты `envelope_a/b` на месте (реконструкция не нужна) |
| Holdout 180 пар | `data/conflict-detector/corpus-v1/holdout.jsonl` | sha256 `e8adfec9…2567` == `seals/CD-holdout-v1.seal.json` (ts 01.11T23:28Z, «content NOT read after this event») |
| Пул-остаток 420 | `corpus-v1/holdout-pool.jsonl` | в train/eval не входит (A-disjoint) |
| Пин прибора | `data/conflict-detector/pin.json` | nano `3b752e06…17ba`, guard `cg-guard-v1`, floor 0.527, сиды 42/777/4242, hard window [0.7;0.75], razor_pos_quota 191 |
| Золото | `data/conflict-detector/sheet/verdicts/` (ml/tl/llm-leg.json) | g_gold **48/48 ≥ 45 ЗАКРЫТ** (синтез, ключ 6968087a) — не reopen |
| Дизайн-пакет | `docs/experiments/conflict-detector-corpus-design.md` | гейты §6.2, sanity §6.3, базлайны §6.1 |
| Зеркало тренера | `scripts/merge_arbiter_train.py` (4164 строки) | образец для нового `scripts/train_conflict_detector.py` |

## 3. Порядок работ (строго)

1. **Тренер-зеркало:** `scripts/train_conflict_detector.py` по образцу
   `merge_arbiter_train.py`. **Дельта-карта (что переиспользовать/что
   менять): `docs/experiments/conflict-detector-trainer-delta-map.md` —
   обязательна к прочтению до реализации.** Фичи ТОЛЬКО через `cortex.features.pair`
   (0 реимплементаций, урок #480); конвенция поверхности
   `n1(title)+\n+n1(body)+\n+n1(tags)`, усечение 4096; пин эмбеддера nano
   `3b752e06…` (A-frozen: wrong provider refuses loudly);
   `is_version_token` — construction-only, read-on-inference запрещён
   (§2.3.2 пакета); обе ориентации (A-symmetry).
2. **Полы ДО holdout** (ML-STD-0001): floor_seed ≥ 10 розыгрышей (П3),
   floor_inv / floor_perm / floor_q ≥ 3, оценка max-pairwise;
   floor_perm — РАЗЛИЧНЫЕ перестановки на сид (фикс каскада W2, `2a45f28`);
   поля публикации полов — в run-log ДО gate_start (П4).
3. **Sanity §6.3 — 7 проб** на фикстурных ячейках через производственный
   контур, ДО любого чтения holdout (таблица проб — §6.3 пакета:
   self-пара ≤0.1; cosmetic-twin NOT; envelope-вариант NOT; razor CD-1 →
   contradicts P≥0.5; version-сосед NOT; stability-лестница инвариантна;
   translation-твин — оба ответа). Крас любой = аборта класса
   «инструмент», holdout НЕ читается, чекпоинт, стоп.
4. **ОДНО чтение holdout** раннером: перед чтением сверяет
   sha256(holdout.jsonl) с seal-json; событие чтения в run-log.
5. **Гейты §6.2** (§4 ниже) — вердикт = конъюнкция.
6. PASS → ADOPT (run-log + отчёт) → авто-минт (§5) → каскад (§6).
   FAIL → DECLINE + STOP + чекпоинт.

## 4. Гейты ADOPT §6.2 (пороги ратифицированы MLC-9 amend-then-ratify)

| Гейт | Формула | Порог |
|---|---|---|
| g_sens | sens(holdout, CD-агрегат, эксклюзия*) | ≥ 0.90 |
| g_spec | spec(holdout, NC-агрегат, эксклюзия*); нож-страты отдельно | ≥ 0.90; NC-1/NC-2 ≥ 0.85 |
| g_nc3_exact | contradicts-предложения пайплайна на NC-3 holdout | exact 0 |
| g_razor | certify-экзамен razor-зоны (пробы 4+5 через производственный контур) | обе ноги exact |
| g_self | P(contradicts \| self-пара) | ≤ 0.10 |
| g_ba | Δ acc против max(BL) (эксклюзия*) | ≥ δ_acc = max(p95−p5; 0.10) |
| g_brier | brier_score_loss на holdout (эксклюзия*) | ≤ 0.25 |
| g_stab | agreement на парафраз/косметик-твинах, 3 ступени | ≥ 1 − 2·floor_inv |
| g_far | P(contradicts \| far-пара, cos < 0.40) | < 0.5 |
| g_gold | согласие золотого листа | **48/48 ЗАКРЫТ до freeze** |
| g_disj | дизъюнктность train↔holdout (A-disjoint) | exact 0 |

\* **Примечание 1:** g_sens/g_spec/g_ba/g_brier гейтуют ПАЙПЛАЙН
(guard+model); модель-одиночные числа публикуются рядом как ДИАГНОСТИКА.
**Примечание 2:** агрегаты без same-point razor-коллизионных ячеек
(razor CD-1/CD-2 ↔ NC-2/NC-3); razor-зона сертифицируется
g_razor + g_nc3_exact + g_gold. Policy-константные (без пола):
g_nc3_exact, g_razor, g_self, g_far, g_disj. **Guard-off проба:**
постратно, минимум NC-3 — число честности в отчёте (не гейт).
Базлайны: BL-band 0.49 / BL-digit 0.57 / BL-digit+guard 0.66 →
max(BL) = 0.66.

## 5. После ADOPT — авто-минт contradicts (advisory; санкционирован, без владельца)

1. Локальный чекаут движка (проверено 04.11: HEAD = `release/6.6.0`,
   локальных коммитов 0, позади origin/main на 94): `git switch main` →
   `git merge --ff-only origin/main` (цель `4c417d2+`). Крас `switch`
   (tracked-изменения) или ff-only → стоп + чекпоинт. Untracked-хвосты
   движка не трогать.
2. Раннер минта через `manager.mint_contradicts_edge` (движок):
   **сначала DRY-RUN** — число предложений по стратам, счёт
   NC-3-suppression; числа в отчёт. Аномалии → стоп ДО записи.
3. Минт: advisory, rank-only вес 1.0, provenance `conflict-detector`,
   NC-3 suppression обе стороны не-опционально, `guard_version`
   `cg-guard-v1` в metadata_props, CONFLICT-MINT телеметрия.
4. Прод-стор `~/.vesma/data/vesma.db` — писать только после чистого dry-run.

## 6. Каскад волны (волновым диффом)

Аудиторам (`gcw-code-reviewer`: security-нога + quality-нога, раздельно,
каждый в СВОЁМ temp-worktree — никогда в общем/живом): дифф волны =
ветка `feat/conflict-detector-corpus-design` (корпус+тренер+вердикты) +
движковый PR #536 (`4c417d2`) + раннер минта. Findings → fix-slice в той
же волне → re-run → вердикт accept / accept-after-fix / reject. Споры →
АрхКом; открытые позиции — владельцу, не прятать.

## 7. Окружение и ресурсы

- python ТОЛЬКО `/usr/bin/python3` (голый `python3` сломан); скоринг/тренинг —
  `vesma-cortex/.venv` (ort 1.30.0, tokenizers 0.22.2); `PYTHONPATH=<движок>/src`
  — APPEND (не prepend, изоляция venv — CR-008); шим vesmaro→vesma при
  импортах (запись b0f0d2f3).
- ДО диспатча: `swapon --show` + `free -h` (железное правило); тренинг под
  taskset с высвобождением 2–3 ядер; следить RAM available (OOM не должен
  задеть zcode).
- Не трогать: локальный main корневого чекаута vesma-cortex (чужой
  незапушенный `0d7d758` — их ход); чужие ветки; наружу ничего не
  публиковать; секреты маскировать.
- Run-log: `docs/experiments/conflict-detector-train-runlog.md`
  (append-only, инкрементальные коммиты). Артефакт:
  `models/conflict-detector-v1/` (manifest.json + model.onnx +
  MODEL_CARD.md; bundle_integrity: sha256(onnx)==manifest; typed_output
  100%).
- Чекпоинт `vesma_save_context(project=vesma-cortex, task=b6-conflict)` на
  каждом переходе этапов; отчёт владельцу конвертом checkpoint-board,
  persist в карточку волны (`vesma_add`, task:b6-conflict).

## 8. Что НЕ входит в этот диспатч

- MLC-повестка (FYI аддендумы 1+2, банд-доли razor 74.7% фактом, Q1
  in_cache_prefix-proxy, S-2) — отдельный контакт после волны.
- W0-фриз ~07.11 — вручную при следующей сессии (харнес-лимит: один
  автоматический крон на сессию; слот ушёл на крон диспатча 06.11,
  `automation-33b2d2bd`).
- record-quality фаза 2 — припаркована на №8 (≥1000 used-событий).
