# datasets/embed-r4 — round-4 EMBEDDER distillation corpus (texts), assembly numbers

Numbers only — no record text, titles or ids (privacy gates, addendum 6 §A.3;
store-derived rows live only in gitignored `data/embed-r4/`).

- Product: round-4 F-corpus TEXTS for vesma-embed KD (round-3 structure:
  `{"text","lang","source"}` jsonl, train/val 95/5, rng seed 42, fingerprint
  sha256 over train+val bytes — the `training/probe/corpus` shape).
- Spec: docs/specs/embed-round4-prereg-DRAFT.md §2 + addenda 6-8 (ACTIVATED).
- Corpus date: 2026-10-07 (A.8.2 slice rule).

## Composition plan (ratified + TL-approved resolution 2026-10-07)

| class | source | target |
|---|---|---|
| real | real-part.jsonl paragraphs (canonical collector shape) | 2418 records (evening slice, fp d06be1e3…) |
| translated seed | datasets/corpus-v43 batches (TL-validated) | 320 texts (160 dup + 160 sibling pairs) |
| translated new | real-record translations, local teacher | ~2418 (one per record) |
| synthetic | FULL regeneration, 14 round-3 families x {ru,en} | ~23000 rows incl. ~2000 RU↔EN twins |

Translated share target: 15-20% of FINAL corpus (prereg §2 mechanism: new
corpus-stage translated pairs; TL verdict: seed + fill to ~17%).

## Provenance

- base: main@3b099ac (PR #29 merged; SAMELEN/RPARA machinery not used —
  unlabeled KD corpus).
- teacher: Qwen/Qwen3-0.6B (generative, Apache-2.0), LOCAL ONLY for
  store-derived content; filter teacher: Qwen/Qwen3-Embedding-0.6B.
- round-3 synthetic outputs: NOT reused (exact-hash exclusion asserted).
