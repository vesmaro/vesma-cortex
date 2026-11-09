# B6-conflict — SINGLE-SHOT гейт §6.2 на CD-holdout-v2: инструмент-аборт ПОСЛЕ чтения, вердикт НЕ ДОСТИГНУТ

**Дата/время (UTC):** 2026-11-09T23:41:18Z (gate_start) → крах ~23:41:19Z
**Ветка/HEAD:** `feat/conflict-detector-corpus-design` @ `b8b2085` (чистое дерево до прогона)
**Раннер:** `scripts/train_conflict_detector.py gate --corpus-dir data/conflict-detector/corpus-v2 --execute-single-shot` (preset v2: `--pin/--seal/--artifact-dir` разрешены пресетом), host `core-51`, venv `python 3.12.13 / lightgbm 4.7.0 / numpy 2.5.3`, `taskset -c 0-12`
**Run-log:** `docs/experiments/conflict-detector-train-runlog.md`, slice `conflict-detector-v2-gate`

## 1. Пролог П1-П9 — весь зелёный, строго до gate_start

| Событие | Факт |
|---|---|
| swap_check | available 13844 MB, ram_overrun_stop=false |
| env | venv_suffix ok, python 3.12.13, lightgbm 4.7.0, numpy 2.5.3 — mismatch: [] |
| host | core-51 == expectation |
| byte_tripwire | train `a549dc0c…` == build-manifest == пресет; pin-v2 `dcb3536d…` == bytes == manifest; seal-файл `5b95bfd7…` == пресет == manifest fp.seal; holdout metadata pin `c92a2caa…` == seal json == build manifest; seal n=180 ids=180; embedder `3b752e06…`, guard `cg-guard-v1`; mismatch_hard=[] seal_metadata_mismatch=[]; sealed_file_opened=false |
| activation | spec RATIFIED-маркер + ADDENDUM-3 маркер; анцестрия: dispatch 9706158, delta-map 1c85715, gold-closed 45597c5, std0004-formula 5305c82, corpus-v2-build 2d9ce88 — ok=true |
| prereg_order_assert | floors_published (conflict-detector-v2-floors @ 2026-11-09T22:43:27Z) привязан к sha артефакта `2534fcbe…`; delta_acc_declared 0.10 воспроизводится из p5/p95; delta_inv 0.0 |
| gate_start | 2026-11-09T23:41:18.703Z |

## 2. Пре-sealed фазы — зелёные

- Sanity §6.3: **6/7 legs + 1 known-uncovered** (ADDENDUM-4 @ 0971639; probe-7 leg-2 опубликован, не ассертится: verdict not-contradicts, p=0.008105).
- E8-iii canary para2: 7/7 selected-ячеек остались contradicts (p≈0.9995–0.9997).
- E8-iv para1 sweep: 3/3 applicable слов — contradicts (диагностика).
- g_gold_state: CLOSED 48/48 ≥ 45 (синтез 45597c5); v2-delta gold 12/12 единогласно (событие gold_v2_delta_synthesis, holdout_reads=0) — не пересчитывалось.

## 3. Sealed чтение — состоялось (single-shot ПОТРАЧЕН)

`sealed_read_once` @ 2026-11-09T23:41:19.281Z: **n_pairs=180, n_rows=360, sha256 `c92a2caa283869ce…` — совпал с seal json == build-manifest == dispatch-пином** (сверка раннера до открытия). Это единственное `sealed_read_once` во всей истории проекта (v1-печать CD-holdout-v1 осталась девственной, не тронута).

## 4. Крах — инструмент-класс, ПОСЛЕ чтения, ДО любых gate_result

```
File "scripts/train_conflict_detector.py", line 2984, in _eval_gates_on_holdout   (фаза g_stab)
  → decide_pipeline (line 961) → pair_vector (line 709)
  → src/cortex/features/pair.py:181 _validate_similarity
ValueError: similarity must be within [-1, 1] and not NaN, got 1.0000001105987988
```

**Причина:** `NanoLeg.pair_cos` (косинус свежепарафразенной пары в стаб-лесенке cosmetic/para1/para2) вернул 1.0 + 1.1e-7 — классический float-эпсилон нормированного скалярного произведения; фрозен фиче-контракт валидирует [-1,1] строго и поднимает ValueError; раннер перехватывает только InstrumentAbort → необработанный traceback.

**Следствие:** ни одного `gate_result`, ни `gate_done`, ни `guard_off_probe_holdout` в run-log; промежуточные числа, вычисленные в памяти до краха, не залогированы и не существуют как измерения. Вердикт ADOPT/DECLINE **не достигнут** — конъюнкция не состоялась. Holdout v2 сожжён; повтор запрещён (П8).

## 5. Таблица 11 гейтов §6.2

| Гейт | Норматив | Факт | Вердикт |
|---|---|---|---|
| g_sens ≥0.90 (CD-агрегат, эксклюзия razor-коллизий) | ≥ 0.90 | не вычислен | — |
| g_spec ≥0.90 + NC-1/NC-2 ≥0.85 | ≥ 0.90 / 0.85 | не вычислен | — |
| g_nc3_exact = 0 contradicts | exact 0 | не вычислен | — |
| g_razor (обе ноги exact, пробы 4+5) | exact | не вычислен (в гейт-фазе; пре-sealed пробы 4+5 зелёные) | — |
| g_self ≤0.10 | ≤ 0.10 | не вычислен | — |
| g_ba ≥ δ_acc=0.10 vs max(BL) | ≥ +0.10 | не вычислен (v2 build-пин max(BL)=0.62; бриеф-число 0.66 — v1) | — |
| g_brier ≤0.25 | ≤ 0.25 | не вычислен | — |
| g_stab ≥ 1−2·floor_inv = 1.0 | ≥ 1.0 | **точка краха** (para-петля) | — |
| g_far <0.5 | < 0.5 | не вычислен | — |
| g_disj exact 0 (train↔seal ids) | exact 0 | не вычислен | — |
| g_gold | CLOSED | CLOSED (48/48 v1 + 12/12 v2-delta) | OK |

Эксклюзия razor-коллизий: не подсчитана (агрегаты не вычислены). Guard-off проба (честность, не гейт): не вычислена. Per-strata таблицы: не вычислены. **Чисел гейтов на holdout не существует** — ничего не публикуется и не восстанавливается.

## 6. П8 — отказ повтора зафиксирован машиной

Повторный запуск той же команды (2026-11-09T23:47Z, сразу после краха): run-log событие `refuse {reason: "single-shot already spent for this slice (sealed_read_once/gate_done in run-log) — П8"}`, exit 2, sealed-поверхности не тронуты (sealed_file_opened=false).

## 7. Классификация и следующий ход

- Класс: **инструмент-аборт после чтения** (дефект инструмента, не модели и не входов; не гейт-красный). Стандарт ML-STD-0003 определяет STOP «печати не читаются» только для абортов до чтения — этот шов (чтение состоялось, вычисление упало) стандартом явно не покрыт; зафиксирован как есть, без ретроспективной легализации.
- Вердикт линии: **не вынесен**. Никаких повторов, фиксировок, перекалибровок не предпринято и не будет в этом цикле. Следующий ход — **TL/MLC** (судьба волны при сожжённом holdout v2; corpus amend-once бюджет E13 v2 не трогал).
- CD-holdout-v1: девственна (0 чтений за историю) — остаётся единственной незажжённой печатью линии; её судьба — решение TL/MLC, не раннера.

## 8. Дефекты инструмента для issue (не фиксятся в этом цикле)

1. `cortex.features.pair._validate_similarity` — строгая [-1,1]-валидация без эпсилон-допуска против float-косинусов (значение 1+1.1e-7 физически легально для нормированных векторов). Решение о допуске (или clamp в NanoLeg.pair_cos) — изменение фрозен-контракта → MLC.
2. Раннер: пост-read фазы не обёрнуты в try/except → crash без run-log события (последнее событие = sealed_read_once, крах виден только как отсутствие gate_done); нужен `instrument_abort`-лог в spirit П1/П8.
3. Косметика: note строки `sealed_read_once` жёстко говорят «CD-holdout-v1» (v2-прогон идентифицируется slice/sha — текст не смысловой, но вводит в заблуждение).

## 9. Что НЕ делалось (честность)

- Повторных чтений holdout — ноль (П8 refuse — единственное прикосновение раннера после краха).
- Holdout-контент никуда не копировался; числа из памяти после краха не восстанавливались и нигде не публикуются.
- Артефакт `models/conflict-detector-v2` (onnx `2534fcbe…`), floors, пины, печати — не изменялись.
- MINТ не запускался (и не мог: ADOPT не достигнут).

## Пути

- Run-log (раннер-писанный): `docs/experiments/conflict-detector-train-runlog.md` (+26 строк, slice `conflict-detector-v2-gate`)
- Машинное состояние цикла (gitignored): `data/conflict-detector/measure/gate-v2-single-shot-2026-11-10.json`
- Stdout+traceback (volatile): `/tmp/b6_gate_single_shot_stdout.log`
- Итог гейта: **VERDICT NOT REACHED — instrument abort after sealed read; escalation TL/MLC**
