# B6-conflict E2 PRE-CUT CHARACTERIZATION (ADDENDUM-6 §5 G4) — numbers-only отчёт

**Вердикт characterization: GREEN (PASS против пре-регистрированной полосы).**
Пробы §6.3 на new-template фикстурах: GREEN (3/3).

- **Дата/время (UTC):** 2026-11-10, one-shot `char2_pre_cut_start` → `char2_pre_cut_done`, elapsed 22.6 s
- **Ветка/HEAD:** `feat/conflict-detector-corpus-design` @ `c5efe0e` (bank freeze 7e91f50, banks commit 066eea6, banks file sha 9df6550c…, content sha 31861e0d… — не менялись)
- **Артефакт:** `models/conflict-detector-v2`, manifest sha256 `2534fcbec2b7872a…` == dispatch-пин 2534fcbe; CORE-13 frozen; cut 0.5; guard cg-guard-v1 ahead of model leg
- **Run-log:** slice `conflict-detector-v2-char2` (events: cost_declaration → char2_preregister → char2_pre_cut_start → char2_model_loaded → char2_embed_pass → char2_clean_reproduced ×3 → char2_stratum_verdict ×3 → char2_probe_63 ×3 → char2_pre_cut_done); инструмент `scripts/char2_pre_cut.py` (selfcheck PASS до прогона, событий не писал)
- **Legality:** pool-only F7 — pool-ряды + sha-сеты печатей (seal-манифесты + pool-origin, `sealed_generation_sha_set`); sealed holdout.jsonl v1/v2 НЕ открыты (`sealed_files_opened=false`); holdout-числа не читались; веса не менялись; ничего не резано, не запечатано, не напечатано

## 1. Пре-регистрированная полоса (пин ратификации G4, не менялся)

per-stratum Δ = P(contradicts | новый банк) − P(contradicts | старый банк) на match-ячейках stratum×band×len (обе стороны имеют clean-ряды); Wilson 95% на каждую долю + Newcombe hybrid-score интервал для Δ; гейт: **|Δ| ≤ 0.15 И Δ-CI накрывает 0**. Strata без старого паритета — честный «нет матча». Scope: NC-1, NC-3, NC-5 (банки, из которых v3 реально режет). NC-4 — out-of-scope по G7 (fallback-маржа, из нового банка не режется; тот же критерий, что исключил её из проб) — disclosed, не skipped silently.

**Популяция:** G10 census clean-супплай (mechanical_ok → F10-cull train-v2/v1-seal/v2-seal → side-text dedup), classic-префикс `ho-` vs v3-префикс `ho3-`. Репродукция census — ТОЧНАЯ по всем трём стратам (pool/attrition/culls/dups/clean/clean_v3/clean_classic все совпали): NC-1 672→clean 157 (v3 156 / classic 1); NC-3 180→103 (72 / 31); NC-5 1128→57 (46 / 11). Эмбеддинг: pinned nano (cb.Cos, census-инструмент), 1258 текстов / 1980 ячеек / 20.7 s.

## 2. Per-stratum Δ (match-ячейки) — гейт characterization

| Stratum | Δ | Δ CI95 (Newcombe) | Wilson95 new | Wilson95 old | new k/n | old k/n | |Δ|≤0.15 | CI∋0 | Гейт |
|---|---|---|---|---|---|---|---|---|---|
| NC-1 | +0.0000 | [−0.7935, +0.2588] | [0.0000, 0.2588] | [0.0000, 0.7935] | 0/11 | 0/1 | да | да | **GREEN** |
| NC-3 | +0.0000 | [−0.1103, +0.0507] | [0.0000, 0.0507] | [0.0000, 0.1103] | 0/72 | 0/31 | да | да | **GREEN** |
| NC-5 | +0.0000 | [−0.2588, +0.0803] | [0.0000, 0.0803] | [0.0000, 0.2588] | 0/44 | 0/11 | да | да | **GREEN** |

Ноль contradicts-предложений контура на ОБЕИХ сторонах каждой страты: новый банк не сдвигает P(contradicts) относительно старого (0 vs 0). Полоса пройдена во всех трёх стратах с матчем.

### Match-ячейки (band/len) и покров

| Stratum | Match-ячейки (new k/n, old k/n) | v3-only ячейки (disclosed, вне гейта) | classic-only |
|---|---|---|---|
| NC-1 | razor/short (0/11, 0/1) | b85/long 0/24, b85/medium 0/17, b85/short 0/37, razor/medium 0/67 | — |
| NC-3 | razor/long (0/24, 0/12), razor/medium (0/24, 0/10), razor/short (0/24, 0/9) | — | — |
| NC-5 | b55/short (0/13, 0/1), b70/short (0/19, 0/4), razor/long (0/12, 0/6) | b40/short 0/2 | — |

Ноги контура: NC-1 — model leg 156/156 v3 + 1/1 classic; NC-3 — guard leg 72/72 + 31/31 (version-only regions, односторонний NOT — policy-константа, p=None; производство как спроектировано); NC-5 — model leg 46/46 + 11/11.

### Честность малых n (disclosed, не влияет на вердикт — вердикт по пину)

- NC-1: старая сторона = 1 clean-ряд во всей страте (шаблонная вселенная исчерпана — сам триггер ADDENDUM-6); match = единственная ячейка этой строки (razor/short). CI [−0.79, +0.26] — широкий, накрывает 0. Читается как «нет сигнала», не как «доказанный паритет».
- NC-5: старая сторона 11 рядов; CI [−0.26, +0.08].
- NC-3: полная ячеечная паритетность 3×3 (razor × long/medium/short), самый узкий CI [−0.11, +0.05].
- «Нет матча»-страт нет: все три страты имеют ≥1 match-ячейку.

## 3. Пробы §6.3 на new-template фикстурах (производственный контур)

Фикстуры — те же декларированные preflight-ячейки (идентичные билдеры + name-derived seeds; selfcheck подтвердил mechanical_ok + label). Лестница raw/cosmetic/para1 (проба 6), cosmetic-твин (проба 2), swap-инвариантность, machine-proof косметической поверхностной инвариантности (case-flip → идентичный CORE-13 вектор + идентичный embed_text). Ladder RNG seed 21 (g_stab-константа).

| Stratum | Фикстура | raw | cosmetic | para1 | Инвариантность | checks |
|---|---|---|---|---|---|---|
| NC-1 | nc1-single | NOT, model, p=0.000205 | NOT, model, p=0.000205, cos 0.9402 | NOT, model, p=0.000698, cos 0.9183 (para применён: body изменился) | verdict инвариантен; swap-инвариант | 4/4 |
| NC-3 | nc3-fam0-fl0 | NOT, guard (p=None) | NOT, guard, cos 0.9889 | NOT, guard, cos 0.9889 | verdict инвариантен; swap-инвариант | 4/4 |
| NC-5 | nc5-garden-cross | NOT, model, p=0.000108 | NOT, model, p=0.000108, cos 0.5654 | NOT, model, p=0.000108, cos 0.5654 | verdict инвариантен; swap-инвариант | 4/4 |

- Ожидания §6.3 выполнены: cosmetic-твин → NOT contradicts (проба 2), лестница инвариантна (проба 6), raw = NOT contradicts (NC-класс, false-positive ось).
- **Disclosure (para1 no-op):** на nc3-fam0-fl0 и nc5-garden-cross para-подстановка не нашла слова из таблицы в теле — rung прошёл с неизменённым телом (верdict-инвариантность на этих ступенях тривиальна). На nc1-single para применён реально (body изменился, p сместился 0.000205→0.000698, вердикт удержался). Практика E8-iv: неприменимые слова disclose, не ассерт.
- **Disclosure (cosmetic identity):** на всех трёх фикстурах cosmetic rung дал p, бит-в-бит равный raw (case-flip невидим поверхности — machine-proof прошёл; на NC-5 и NC-3 cos тоже неизменен).

## 4. Вердикт и якорная формула для отчёта v3

- **Characterization: GREEN** — гейт |Δ| ≤ 0.15 И CI∋0 выполнен во всех трёх стратах (Δ = 0.0 всюду); пробы §6.3 GREEN 3/3.
- **Якорь (G4, дословно):** characterization green + v3-гейт red = **дефект модели**; characterization red + v3-гейт red = **дрейф банка**. Для грядущего v3-гейта: при red на transfer-ногах (g_spec_indist / g_spec_transfer сплит по G3) с зелёной characterization — дефект модели v3, эскалация MLC; шаблоны/банки не правятся (заморожены G1).
- Цепочка G2: BuildStop → MLC → commit банков (066eea6) → freeze (7e91f50) → **E2 characterization (этот отчёт, GREEN)** → DE ценз+резка+seal v3 → TL-гейт → v3 single-shot.

## 5. ONE-SHOT дисциплина

Одно событие `char2_pre_cut_start`, одно `char2_pre_cut_done`, exit 0. Все числа этого отчёта — из этого единственного прогона (артефакт `artifacts/runs/char2_pre_cut_2026-11-10.json`). Повтор прогона refuses на run-log guard (П8-класс). Selfcheck до прогона событий не писал и one-shot не тратил.

## 6. Отклонения от брифа

1. NC-4 не в characterization-scope — бриф задал пробы только на NC-1/NC-3/NC-5 («NC-4 fallback не участвует — не режется из нового банка»); тот же критерий применён к Δ-полосе (G7). Fixed pre-run в char2_preregister.
2. CI для Δ реализован как Newcombe hybrid-score от двух Wilson-интервалов — стандартный метод разности долей; пин ратификации («Wilson 95% CI», «CI, накрывающим 0») соблюдён, интерпретация не менялась.
3. Других отклонений нет: пре-регистрированная полоса не менялась, банки/шаблоны/веса/печати не тронуты, re-roll'ов не было.
