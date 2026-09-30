# A2 store-корпус претрейна (экспорт из стора движка) — рукопожатие

> Контент корпуса в репу НЕ входит (гигиена данных, data-contract §6):
> здесь зафиксированы счётчики и фингерпринты. Корпус живёт локально в
> `data/pretrain/pretrain-20260930/` и `data/vectors/pretrain-20260930/`
> (gitignored); валидность = пересчёт фингерпринтов из локальных файлов
> совпал с этой записью.

- **Дата экспорта:** 2026-09-30 · ветка `feat/a2-export-corpus`
- **Источник:** стор движка vesmaro (mnemos.db + vectors.db), доступ
  строго read-only (`file:…?mode=ro` + `PRAGMA query_only`), стор живой
  (MCP) — не тронут
- **Гигиена (порядок препрега, заморожен):** SQL-отбор → сканер секретов
  движка + danger-детекторы + `no-federate` + `quarantine` ДО любого
  вывода → минимальный состав (title, body, tags, language, record_type,
  created_at, content_hash, vector). Исключения — только счётчики по
  причинам, без содержания.
- **Сканер:** провенанс `engine` — детекторы движка
  (`vesmaro.secrets_detector` + `vesmaro.danger_detectors`) исполнены из
  read-only worktree движка (`wt/a2-engine-readonly`, main `0046526`),
  primary checkout движка не тронут. Fallback-сканер не использовался.

## Пул записей (базы corruption-претрейна)

- **SQL-кандидаты:** 2918 (published · note/snippet/fact · length ≥ 80 ·
  пин эмбеддера есть); лимит пула 800
- **Пул после гигиены:** 794 записи; исключено 6:
  `no-federate` ×4, `prompt-injection:system-prefix` ×2
- **Фингерпринт пула** (BLAKE2b-256 манифеста
  `records_manifest.txt`, схема data-contract §5 по составу
  title/body/tags/language/record_type/created_at/content_hash/vec_sha256):
  `e2449dd66e97bdb8ef3c781b8570fa523b4edb75851093db0f2db4e24074061a`
  (воспроизведён двумя независимыми прогонами байт-в-байт)

## Пары-базы near-duplicate (неразмеченные, [0.85, 0.97))

- **194 915 пар** по косинусу сторовских векторов (полуоткрытый бэнд);
  стор-вектор цели пары: mean 0.9114, p50 0.9147
- Строки компактные: `pair_id`, `id_a`, `id_b`, `similarity`,
  `pair_sha256`; стороны §3 соединяемы по id из `records.jsonl`,
  `pair_sha256` пересчитывается из join (тампер-эвиденс сохранён)
- **Фингерпринт корпуса пар** (BLAKE2b-256 `near_dup_manifest.txt`):
  `9cf1c9f2af5a9690d03a50c1a6259e13fe16de90689e256747d8efcd030b7ff6`

## Сайдкар полевых косинусов (мнема-embed-v1, CPU, ноль сети)

- **Пин эмбеддера:** `nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba`
  — совпадает с пином сторовских векторов (ре-эмбеда не было)
- 794 записи × 3 поля (title/body/tags) → `field_vecs.npz`
  (id → три полевых вектора 384-dim; косинусы пары выводятся
  детерминированно — совместимо с `attach_field_cosines`), готовые
  косинусы пар-баз → `near_dup_field_cosines.jsonl`
- Распределения по пар-базам: cos_title mean 0.8714 / p50 0.9852;
  cos_body mean 0.9018 / p50 0.9030; cos_tags mean 0.9075 / p50 0.9239;
  все значения в [-1, 1] (float32-выбросы клипуются)
- Полевые векторы считаны ТОЛЬКО по записям, прошедшим гигиену
  (вход сайдкара — сам `records.jsonl`)
- Тайминг: экспорт 7.3 с, сайдкар 48.6 с

## Цепочка §4 (проверка консьюмости)

`cortex pretrain --corpus records.jsonl --seed 7` из пула строит
§4-манифест: **1588 пар** (794 weak-positive + 794 hard-negative),
векторы едут в `vec_a`/`vec_b` для кандидата N;
фингерпринт: `c4c175ab4ba01fee21c00a8540928efb007acb6383cf74da9407d3f31567b1ae`

## Решения инстанциации (не меняют контракт)

- `record_type` — колонка `memory_type` (note/snippet/fact), НЕ
  `canon.type`: на живом сторе `canon.type` несёт значения пайплайна
  (`checkpoint`), а цикл corruption-трансформаций заморожен над
  note/snippet/fact. `language` — `metadata.canon.language` (зеркало
  `CanonRecordView.from_memory`).
- §4 `pairs.jsonl` (source/transform) генерируется ПОТОКОМ ниже —
  `cortex pretrain` из пула; экспортёр отдаёт пул + пары-базы
  (diagnostics/базы), что и разрешено формулировкой среза.

## Статус

Пакет собран и само-согласован (фингерпринты воспроизведены, цепочка до
§4-манифеста прошла). Использование в обучении кандидата N — срез A3b.
