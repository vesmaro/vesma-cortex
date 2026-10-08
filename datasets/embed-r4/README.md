# datasets/embed-r4 — round-4 EMBEDDER distillation corpus (TEXTS) — FINAL

Sealed 2026-10-08. Numbers only (store-derived texts live in gitignored
`data/embed-r4/`). Per-class composition: numerator / denominator exact.

## Seal

- **corpus fingerprint (sha256 over train+val bytes):**
  `7a3bf00b22951df59e3cb3e638c16e1c05f09a51223da8e19ef246ed29f8b645`
- **totals:** 25496 texts (train 24221 / val 1275; split stratified by source,
  rng seed 42, val = ceil(5%) per source with >=5 rows — round-3
  `prepare_dataset` shape)
- **double assembly:** byte-identical (PASS, run-log `verify-double-assembly`)
- **branch:** feat/embed-r4-corpus (base main@3b099ac)

## Composition (final)

| class | source labels | texts | share |
|---|---|---:|---:|
| real store slice | `real-part-r4` | 12903 | 50.6% |
| translated seed (TL-validated v4.3 batches) | `translated-dup`, `translated-sibling` | 320 | 1.3% |
| translated real records (teacher, local) | `translated-real-r4` | 1451 | 5.7% |
| translated unique LLM rows (teacher, local) | `translated-llm-r4` | 1617 | 6.3% |
| synthetic RU<->EN twins (teacher, local) | `twin-{ru,en}-{family}` | 650 | 2.5% |
| synthetic mono (template 6583 + LLM unique 1972) | `synthetic-{ru,en}-{family}` | 8555 | 33.6% |
| **TOTAL** | 32 source labels | **25496** | 100% |

- **translated share: 4038 / 25496 = 15.84%** — inside the ratified [15;20] band.
- **RU share: 42.08%** (>= 40% quota).
- **mono wall (TL verdict B, composition decision 2026-10-08):** mono
  generation stopped at 11170 generated rows (10186 gate-ok, 849 jobs short
  of full scale — honest counters in the run-log `mono-wall` event). Corpus
  is mono-wall@12000: the training lane owns epoch/hyperparameter
  compensation; the gates arbitrate.

## Gates (assembly asserts — all TRUE on the sealed artifact)

| assert | result |
|---|---|
| A1 train+val ∩ judged corpus c2ce056d (89 entries + 192 queries, 119 hashes) | 0 PASS |
| A2 train+val ∩ sealed holdouts/evalsets (b2, dataset-v4, v4.3 f/g + recheck, ds1000, layer-A evalsets; 42 files, 2454 hashes) | 0 PASS |
| A3 train+val ∩ round-3 SYNTHETIC pool (24949 hashes, verbatim) | 0 PASS |
| A4 RU share >= 0.40 | 0.4208 PASS |
| A5 translated share >= 0.15 | 0.1584 PASS |
| A6 canonical gates (40..4000 chars, <=256 approx tokens) | all rows PASS |

## Teacher quality gates (per family / per pool)

- digit-preservation (addendum-7 rule): enforced on every translated pair
  at generation; failures dropped or retried (counters `translate-real-*`,
  `translate-llm-done` in the run-log).
- anti-copy gate: enforced on translations (counter `dropped_copy`).
- teacher-cos gate (Qwen3-Embedding-0.6B, min-cos 0.55): 5573 pairs checked,
  170 rejected (2.9%) — `filter-pairs-done`.
- non-verbatim vs round-3 (teacher sample, flag 0.97): LLM pool worst
  0.8711 — clean; template pool flagged science (0.9844 sample) ->
  FULL family sweep, 77/300 rows dropped (worst 0.9922) —
  `template-nonverbatim-sample`, `template-flagged-sweep`. Exact-hash
  disjointness of the template pool vs round-3: overlap 0 (verified at
  generation; A3 re-asserts on the sealed artifact).

## Collapse + repair provenance (honest trail)

1. LLM freeform synthetic pool collapsed on uniqueness (~23% unique; 8270
   mono + 3594 twin dup drops) — caught by A4/A5 at assembly, never sealed.
2. Repair (TL approved 2026-10-08): mono bulk = round-3 template
   cross-product machinery with every vocabulary axis REPLACED (6660 rows,
   0 hash overlap with round-3); 2288+ genuinely-unique LLM rows kept;
   translated share repaired via teacher translations of the unique LLM
   rows (1717 kept) + twins.
3. Generator swap (TL approved): HF-CPU decode ~50h projected -> llama.cpp
   Q8_0 of the SAME ratified Qwen/Qwen3-0.6B; swaps/loads in the run-log.

## Provenance

- real part: `scripts/assemble_real_part.py`, 4 read-only sources
  (live + backups 0920/0921/0924), 2418 unique content hashes,
  fp `d06be1e3…` (evening slice 2026-10-07; A.8.2 corpus-date rule).
- generator: Qwen/Qwen3-0.6B (Apache-2.0), llama.cpp Q8_0, LOCAL ONLY for
  store-derived content; seeded sampling — not byte-reproducible; the
  corpus fingerprint is the seal (assembly itself is deterministic:
  double-run PASS).
- filter teacher: Qwen/Qwen3-Embedding-0.6B (cos gates + non-verbatim).
- run-log (append-only): full copy at `datasets/embed-r4/run-log.jsonl`.
- known issue carried (mission note): PR #29 red lint was ruff F841 dead
  `nb_text/tags_text` leftovers + formatting in build_corpus_v43.py,
  fixed pre-merge (3f0c209, bfd86c3) — noted, not acted on.
