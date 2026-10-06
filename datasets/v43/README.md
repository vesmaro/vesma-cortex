# datasets/v43 — corpus v4.3 (translated classes + real part)

Wave of 2026-10-06 (prereg addenda 6 and 7, ratified by the owner;
docs/specs/embed-round4-prereg-ADDENDUM-6.md,
docs/specs/embed-round4-prereg-ADDENDUM-7.md). The frozen v4.2 code is
NOT modified — v4.3 rides a sibling strategies module and a separate
assembler; pair ids of v4.1/v4.2 stay byte-stable.

## What the wave carries

| Piece | Where | Status |
|---|---|---|
| translated-dup strategy + validator | `scripts/gen_dataset_v43_strategies.py` | code done — spec batches are the NEXT wave |
| translated-sibling strategy + validator | same module | code done — spec batches are the NEXT wave |
| quota gate (≥150 pairs per class, both directions) | `quota_violations_v43()` in the same module | done |
| real-part assembler (live store + 0920/0921/0924 backups) | `scripts/assemble_real_part.py` | done, RUN against real stores (numbers in `real-part.README.md`) |
| tests | `tests/test_gen_dataset_v43.py` | 22 green |
| real-part data | `real-part.jsonl` (LOCAL — gitignored) | 2402 unique records |

## Pre-registered quotas (addendum 7 §A.7.1, PROPOSED numbers)

- `translated-dup` — ≥ 150 pairs, is-dup = 1, digits byte-for-byte
  preserved (dates/sums/versions; the validator refuses a lost digit
  run); g-T1 sensitivity ≥ 0.90 on the sealed eval.
- `translated-sibling` — ≥ 150 pairs, is-dup = 0, semantic difference
  must carry a non-empty `justification` field; digit divergence is
  ALLOWED; g-T3 specificity ≥ 0.90.
- g-T2 (no-harm on non-translated strata vs B2-v42, −0.02 band) —
  an eval-time gate, not a corpus property.
- Both directions RU→EN and EN→RU for both classes; each spec builds
  BOTH orientations (a/b and b/a).

## Real part (addendum 6 §A.3)

`real-part.README.md` (next to this file) pins the assembly numbers:
source × candidates, accepted, dedup counters, RU share, memory types,
scanner provenance and the manifest fingerprint (blake2b-256,
data-contract §5 scheme). Record TEXT is never committed — the jsonl
lives locally under the `/datasets/*` gitignore block.

DEFAULT tag filter excludes `task:`, `telemetry` prefixes; the literal
brief list (`task:,telemetry,mnemos:`) is reproducible with
`--brief-tag-filter` (cut is documented in real-part.README.md — the
`mnemos:*` tags are the storage data-contract markers, not telemetry).
Telemetry lives in `session_context`/`conversation` memory types and is
excluded by the domain type gate (note/snippet/fact).

## What runs AFTER this wave (not this wave's work)

1. **LLM batch generation** — authored `translated-dup` /
   `translated-sibling` spec files (quota-scale, ≥75 originals each
   direction), sibling of `datasets/corpus-v4/batch-*.jsonl`; the
   subagent-generation charter is the next wave (dataset-engineer owns
   strategies + validation only).
2. **Assembly** — the v4.3 corpus build (v4.2 strategies + v4.3 blocks +
   real part) with corner-QA, watchdog quotas and the near-miss
   watchlist quota (v4.3 binds `WATCHLIST_QUOTA` = 20 near-0009-family
   pairs).
3. **Three-pass validation** — creator pass + 2 verifier passes over
   the assembled pairs (local mode without a key runs the corner-QA
   gate: `scripts/verify_dataset.py qa <corpus-dir>`).
4. **Fingerprint + seal** — corpus/labels fingerprints, sealed eval
   split per the operator discipline; g-T1/T2/T3 measured at the round-4
   eval stage (TL-exclusive: training and calibration runs).