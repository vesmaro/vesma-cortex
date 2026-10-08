# Dataset card — vesma-embed round-4 F-corpus (TEXTS)

- **Artifact:** round-4 embedder distillation corpus (texts only; teacher
  vectors ride with the training lane). Round-3 structure:
  `{"text","lang","source"}` jsonl; train/val 95/5 (seed 42, stratified by
  source, val = ceil(5%) per source with >=5 rows); corpus fingerprint =
  sha256 over train+val bytes (`fingerprint.txt`).
- **Spec:** docs/specs/embed-round4-prereg-DRAFT.md §2 (ACTIVATED) +
  addenda 6 (§A.3 corpus rescale), 7 (translated classes), 8 (slice rule).
  TL resolutions 2026-10-07: translated share filled with NEW corpus-stage
  pairs to the ratified band; composition approved.
- **Corpus date:** 2026-10-07 (A.8.2 slice rule).

## Composition (final numbers in README.md / report.json)

| class | source labels | construction |
|---|---|---|
| real | `real-part-r4` | 2418 deduped store records (content_hash, 4 read-only sources) → canonical paragraph split (>=40 chars, <=4000 chars, <=256 approx tokens) |
| translated seed | `translated-dup`, `translated-sibling` | 320 texts from the TL-validated v4.3 batches (datasets/corpus-v43) |
| translated new | `translated-real-r4` | one teacher translation per real record, other language |
| translated twins | `twin-{ru,en}-{family}` | RU<->EN twin pairs inside prose synthetic families (digit-preserving) |
| synthetic | `synthetic-{ru,en}-{family}` | FULL regeneration, 14 round-3 families x 2 languages |

## Gates (assembly asserts — all must be TRUE)

- A1: train+val ∩ judged corpus `c2ce056d` (89 entries + 192 queries) = 0
- A2: train+val ∩ sealed cortex holdouts/evalsets = 0
  (b2, dataset-v4, v4.3/v43-g/f + recheck holdouts, ds1000, layer-A evalsets)
- A3: train+val ∩ round-3 SYNTHETIC pool (verbatim) = 0
  (round-3 real-store paragraphs and the TL-authored seed re-enter by
  construction — same domain slice, same prepared batches)
- A4: RU share >= 0.40 (prepare_dataset quota)
- A5: translated share >= 0.15 (ratified band 15-20%; numerator/denominator
  in the report)
- A6: every row passes canonical gates (40..4000 chars, <=256 approx tokens)

## Provenance

- generator: Qwen/Qwen3-0.6B (Apache-2.0), llama.cpp Q8_0 GGUF, LOCAL ONLY
  (store-derived content never leaves the machine); seeded sampling —
  generation is not byte-reproducible, the corpus fingerprint is the seal.
- pair filter: Qwen/Qwen3-Embedding-0.6B cosine gate on every translated
  pair (threshold in run-log), digit-preservation validator (addendum-7
  rule) on every pair, anti-copy gate on translations.
- engine swap note: HF-transformers CPU decode projected ~50h for the pool;
  llama.cpp runtime used for the same model+prompts (run-log 2026-10-07).
- base: main@3b099ac; branch feat/embed-r4-corpus; run-log:
  data/embed-r4/run-log.jsonl (append-only).
- known issue carried (mission note): PR #29 red lint was ruff F841 dead
  `nb_text/tags_text` leftovers + formatting in build_corpus_v43.py —
  fixed pre-merge (3f0c209, bfd86c3); noted, not acted on.
