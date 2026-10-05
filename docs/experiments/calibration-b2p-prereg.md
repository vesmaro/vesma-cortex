# Препрег-аддендум v2 для перекалибровки B2′ (волна D4-1/3)

- **Статус: препрег.** Пороги ратифицированы: corner-QA перенесены в
  `gate_contract.json` (секция `corner_qa`, отдельный контракт-коммит —
  правило разделения), sanity- и ADOPT-критерии ниже зафиксированы
  арбитражем TL 2026-10-05 (открытые вопросы волны D4-1 закрыты: (1)
  полоса [0.40; 0.55) — ограничение замороженного embedder принято,
  (2) tier-spec тематических негативов — агрегат near+far, (3) пороги
  в контракте — исполнено). Корпус dataset-v4 собран и запечатан
  (манифест:
  [artifacts/manifests/dataset-v4.md](../../artifacts/manifests/dataset-v4.md)).
  **Препрег коммитится в canon main ДО прогона — commit-order и есть
  доказательство предрегистрации.** Прогон НЕ запущен.
- **Дата:** 2026-10-05
- **Основание:** [dataset-v4-plan](dataset-v4-plan.md) (ратифицирован),
  [labeling-policy-b2](../specs/labeling-policy-b2.md) §6 (форма ADOPT),
  [p0-export-desync-investigation](p0-export-desync-investigation.md) §8.1,
  находки Layer A v2 ([layer-a-evalsets](../../artifacts/manifests/layer-a-evalsets.md):
  translation-twins sens 0.00, paraphrase sens 0.17).

## 1. Замороженные входы

| Объект | Значение |
|---|---|
| corpus_id | `dataset-v4` |
| corpus fingerprint (BLAKE2b-256) | `ff2fb8f9b06b392c09c82c94028ca558d037587bbdb2373a3abd112f1e9c09b3` |
| label fingerprint | `6e94f6a6eade75fc357f17dd208cb3b5f7d7611e1f1b4792105fc6bf33f2616c` |
| train fingerprint | `17961167afcc64d8b6a40b7c7a19e70e82094944692aa525e7e7b7c6d8e8aad8` |
| holdout fingerprint | `e04bd598eb2a03db6d49e9392f126c21fcdb606f00c93012b91c37e91d8f6cab` |
| embedder pin | `nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba` |
| всего / train / holdout | 868 / 650 / 218 |
| сплит | стратифицированный по (label × stratum), `⌈0.25·n⌉` по pair_id, holdout запечатан до обучения, single-shot |

Holdout-расклад по классам (минимумы для tier-оценок):

| Класс | Пар | Holdout |
|---|---|---|
| P-identity (T0) | 96 | 24 |
| P-cosmetic (T1) | 72 | 18 |
| P-para-light | 96 | 24 |
| P-para-sub | 48 | 12 |
| P-para-struct | 35 | 9 |
| P-trans (переводные близнецы) | 60 | 15 |
| N-metadata (T2) | 112 | 28 |
| N-fact-edit (T3 razor) | 192 | 48 |
| N-near | 49 | 13 |
| N-far | 108 | 27 |

## 2. Corner-QA пороги (источник: `gate_contract.json`, секция `corner_qa`)

Клоны-негативы = 0 · corner-DUP ≥ 10 · identity-позитивы ≥ 60 ·
metadata-негативы ≥ 40 · fact-edit-негативы ≥ 75 · razor-зона без
позитивов · G1-потолок по 7 монотонным фичам · G4 масса ≥ 8 симв. ·
type/lang presence ≥ 0.85 · type/lang mismatch ≥ 3 % / ≥ 5 % ·
cos < 0.55 / < 0.70 / < 0.84: ≥ 5 / ≥ 60 / ≥ 180 · ни одной константной
фичи из 13. Ревизия контракта: sha256
`f94f0c907e0ae6e450cb20bebf016b5b1ae3582e4c7c6e21896cab26d65f350f`
(пин — [eval-methodology](../specs/eval-methodology.md) §6). Лоадер —
`cortex.data.corner_qa.thresholds_from_gate_contract` (fallback на
замороженные константы вне репы; когерентность констант и контракта
пиннута тестом). Факт исполнения на запечатанном корпусе — манифест §2
(все PASS). Изменение порогов = аддендум ДО прогона + пересборка корпуса
по той же процедуре с раскрытием события (запрет пост-фактум правок).

Калибровочная заметка (арбитраж TL 2026-10-05, вопрос 1 — ПРИНЯТО):
полоса [0.40; 0.55) достижима замороженным vesma-embed-v1 лишь краем
(несвязные темы ≈ 0.51) — фиксируем как **ограничение замороженного
embedder**, не как дефект корпуса; несущая линия §8.1 — **188 пар ниже
пола B2 0.836**: cos_target впервые обучаем (пол корпуса 0.450).
Требование смены полосы = смена эмбеддера, отдельное решение.

## 3. Sanity-сьют (расширенный) — обязателен ДО цитирования holdout-чисел

Зонды `cortex.eval.sanity` + аддендумы §6 политики:

| # | Проба | Требование |
|---|---|---|
| S1 | self-pair (cos 1.0) | P(dup) ≥ 0.9 |
| S2 | cosmetic-твин (регистр/пробелы, char5 = 1.0) | P(dup) ≥ 0.5 (замена near_boundary-зонда: «title + !» — это T3-класс, кодировал опровергнутую #480 аксиому) |
| S3 | razor-зонд: замена одной цифры, cos ≈ 0.999 | P(dup) < 0.5 (новый) |
| S4 | metadata-зонд: дельта тегов при тождественном теле | P(dup) < 0.5 (новый) |
| S5 | лестница монотонности {1.0, 0.99, 0.95, 0.8, 0.5} | нестрогое невозрастание, tol 1e-6 |
| S6 | unrelated | P(dup) < 0.5 |

Вердикт сьюта: PASS = exit 0; любой RED → стоп, holdout-числа не цитируются.

## 4. Критерии ADOPT препрега v2 (все одновременно)

| # | Критерий | Порог |
|---|---|---|
| 1 | Sanity-сьют (§3) | PASS, exit 0 |
| 2 | Layer A v2 (merge-v2 + release-v2): инварианты self_pair / монотонность / симметрия | PASS |
| 3 | Layer A v2 коридоры к B1: near-identity, far-negative | в пределах max(0.02; CI95) от B1 |
| 4 | Layer B holdout dataset-v4: typedness / sens / spec / Brier | 100 % / ≥ 0.90 / ≥ 0.90 / ≤ 0.25 |
| 5 | BA строго выше косинус-базлайна 0.92 (тот же раннер) | > |
| 6 | Коридор prev_adopt: BA ≥ BA(B1 `beb0a65d`) − max(0.02; CI95); B1 на release-v2 = 0.7079 (like-for-like) | ≥ |
| 7 | **translation-twins sens** (ключевая способность, лечение 0.00) | **≥ 0.5** |
| 8 | **paraphrase sens** (P-para-light/sub/struct, лечение 0.17) | **≥ 0.5** |
| 9 | Контрактные фичи живы: важности type / lang / tag > 0 в отчёте важностей | все |
| 10 | tier-sens: T0/T1; T4 (light/substantial/structural) | ≥ 0.95; ≥ 0.90 |
| 11 | tier-spec: T3 fact-edit; T2 metadata; тематические негативы (N-near + N-far, агрегат — holdout 40; арбитраж TL 2026-10-05, вопрос 2 — УТВЕРЖДЕНО: отдельная N-near-оценка на 13 парах нестатистична) | ≥ 0.90; ≥ 0.90; ≥ 0.90 |
| 12 | cos-blind razor: P(dup \| косметика, cos ≈ 0.99) ≥ 0.5 ∧ P(dup \| замена факта, cos ≈ 0.999) < 0.5 | оба |
| 13 | family-бэнд B1: P(dup) в полосе cos ≈ 0.98 на B1-семейных пробах | < cut (прецедент B1: 0.016) |
| 14 | Отчёт важностей: ни одна контрактная фича не константа/ноль | все 13 живы |

Правило решения: ADOPT = все критерии одновременно; любой провал =
DECLINE + разбор. Негативная дельта против B1 — расследование, никогда не
merge. Прогон пишется в append-only run-лог (data-contract §8): повторный
`corpus_fingerprint` = отказ. Отчёт открывается таблицей покрытия
поверхности (eval-methodology §8) — числа без неё не цитируются.

## 5. Порядок волны (после ратификации)

1. TL: пороги §3–4 → `gate_contract.json` отдельным PR (правило разделения).
2. Обучение на train (CPU ~10–20 мин, RAM < 2 ГБ) — окну B0 не гейтится.
3. Sanity-сьют ОБЯЗАТЕЛЕН → Layer A v2 → отчёт важностей (§4 #9, #14).
4. Single-shot holdout (218 пар, классы §1) — один прогон.
5. Отчёт по форме eval-methodology §8 → вердикт ADOPT/DECLINE.

## 6. Границы

Веса и обучение не входят в этот PR. Движок vesma не трогается. Сеть не
используется. Holdout-лейблы физически вне train-дерева
(`data/stage2/dataset-v4-holdout/`); изоляция ассерчена кодом и тестами.
