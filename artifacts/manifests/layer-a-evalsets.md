# Layer A evalsets (волна LA-1) — рукопожатие

Замороженные наборы проб Layer A по
[eval-methodology](../../docs/specs/eval-methodology.md) §3/§7: контент
коммитится (синтетика, ADR 0003 R2), фингерпринт — `eval_set_sha256` по
схеме data-contract §5, пробы считаются ТОЛЬКО через
`cortex.features.pair.features`.

## Наборы

| Набор | Роль | Пар | eval_set_sha256 (BLAKE2b-256) |
|---|---|---|---|
| `datasets/evalsets/merge-v1.jsonl` | merge-гейт (быстрый, каждый PR) | 60 | `bab69df3d6e827d0784de6323d9db2803ababea37161d6f99e45c195fb53d67d` |
| `datasets/evalsets/release-v1.jsonl` | release-гейт (стратифицированный) | 201 | `fc84730a73102d3262799b2731af472680055c95ec66b1025d9806087e2d87ff` |

Рядом с каждым набором: `<set-id>.manifest.txt` (манифест §5:
`pair_id <pair_sha256>`, сортировка по pair_id) и `<set-id>.meta.json`
(роль, счётчики по классам, запиненный sha). Лоадер пересчитывает
фингерпринт при каждом прогоне — расхождение = прогон НЕВАЛИДЕН
(дисциплина §5.6).

**Фингерпринтуемый объект** — eval-надмножество §3:
`{record, candidate, similarity, label}` — у пробы назначенный косинус и
метка «по построению» являются СЕМАНТИКОЙ пробы (в отличие от train-пар,
где метка живёт в манифесте). `pair_id`/`class`/`source`/`group` —
назначение, в фингерпринт не входят.

## Таблица покрытия (eval-methodology §8 — числа без неё не цитируются)

| Класс | merge-v1 | release-v1 | Источник | Гейт |
|---|---|---|---|---|
| identity/self | 8 | 18 | процедурный (якоря sanity + 16 тем evalsets) | инвариант (self_pair ≥ 0.9) |
| near-identity/twin | 8 | 18 | процедурный (W4c-positive пертурбации, cos 0.99) | коридор (≥ 0.5) |
| pair-symmetry | 10 | 28 | процедурный (каждая база в обе стороны) | инвариант (\|ΔP\| ≤ 1e-6; хвост §7 закрыт) |
| monotonicity-ladder | 10 | 35 | процедурный (COS_LADDER 1.0→0.5, метки: cos ≥ 0.95 → dup) | инвариант (невозрастание, tol 1e-6) |
| far-negative | 10 | 60 | процедурный (разные темы, cos 0.52–0.60) | коридор (< 0.5) |
| translation-twins | 0 | 0 | **LLM-batch, фаза LA-2** (слот) | breakdown |
| type/lang mismatch | 8 | 36 | процедурный (тот же текст, битое note/fact или ru/en, cos 1.0 → not-dup) | breakdown |
| degenerate | 6 | 6 | процедурный (пустое тело / 1 токен / очень длинное × self+far) | breakdown |
| llm-paraphrase | 0 | 0 | **LLM-batch, фаза LA-2** (слот) | breakdown |
| llm-near-topic | 0 | 0 | **LLM-batch, фаза LA-2** (слот) | breakdown |

Темы проб — 16 НОВЫХ тем (8 RU + 8 EN-двойников,
`src/cortex/evalsets/topics.py`) плюс якоря `PROBE_RECORD`/`UNRELATED_RECORD`
из sanity-сьюта; дизъюнктность с train-темами (`TOPICS` синт-модуля)
закреплена ТЕСТОМ (`tests/test_evalsets.py`), не везением.

## Генерация и детерминизм

    uv run python scripts/gen_evalsets.py          # пересобрать оба набора
    uv run python scripts/gen_evalsets.py --check  # байт-проверка заморозки

Детерминизм структурный (скани без RNG) + фиксированный сид
`DEFAULT_SEED=20261004`; двойной прогон байт-идентичен — pinned тестом.
Правка контента замороженного набора запрещена: новое содержание (в т.ч.
наполнение LA-2-слотов) = НОВАЯ версия набора с новым sha и аддендумом
этого манифеста.

## Прогон на прод-ревизии B1 (`beb0a65d…`, 2026-10-04, read-only)

    uv run python scripts/run_layer_a.py --bundle models/vesma-cortex-v1 \
        --eval-set datasets/evalsets/merge-v1.jsonl   # verdict PASS
    uv run python scripts/run_layer_a.py --bundle models/vesma-cortex-v1 \
        --eval-set datasets/evalsets/release-v1.jsonl # verdict PASS

- merge-v1: BA 0.7931 · release-v1: BA 0.8031 (cut 0.5; prev_adopt-коридор
  для BA — волна LA-3, сравнение like-for-like по этому же набору);
- инварианты зелёные на обоих: self_pair 1.0 (26/26), монотонность
  (9 групп), симметрия 0.00e+00 (19 групп); коридоры зелёные на релизе:
  near 1.0 (18/18), far-negative max 0.0000 (60/60);
- **находка breakdown-класса type/lang mismatch (информационная):**
  B1 ставит P(dup) ≈ 1.0 на «тот же текст + битое поле» (36/36 на
  релизном) — модель перевешивает глобальную близость против
  `type_match`/`lang_match`; ровно хвост, который §7 держала видимым;
- **находка ladder-меток:** на текст-идентичных парах P не опускается
  ниже cut при cos 0.8/0.5 (spec 0.0 на классе при зелёной
  монотонности) — модель игнорирует измеренный косинус на идентичном
  тексте; кандидат на разбор в перекалибровку, не блокер (класс
  breakdown, инвариант монотонности зелёный).

## Границы волны

- `gate_contract.json` НЕ менялся (sha
  `6dd67367601bd6a4fa0d9a067591742a1ac05f58c6c36825596d748d781fef32` —
  сверён); раннер — новый ПОТРЕБИТЕЛЬ контракта;
- ноль сырых строк стора: контент проб — только шаблоны этого репо;
- сети нет (AST-tripwire сюит зелёный), прод-веса не тронуты;
- LA-2: LLM-батчи в слоты (`translation-twins`, `llm-paraphrase`,
  `llm-near-topic`) → набор v2;
- LA-3: шаг CI eval-gate — предложен в описании волны (одна команда,
  exit ≠ 0 = провал).
