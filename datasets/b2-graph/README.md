# datasets/b2-graph — v4.4 candidate pool, graph stratum (slice-1, numbers only)

Status (v4.4-r2 update, 2026-10-22): the TRAIN side of corpus v4.4 is
REBUILT per prereg-v3 ADDENDUM-1 (NA-approved); see `v44-r2/` below. The
sealed holdout v4.4 is untouched. Slice-1/slice-2 records are kept below as
history.

## v4.4-r2 — train rebuild per ADDENDUM-1 (razor fix, taxonomy rebalance)

- Law: `docs/specs/b2-graph-prereg-v3.md` (ACTIVATED) +
  `docs/specs/b2-graph-prereg-v3-ADDENDUM-1.md` (ratified + NA edits 1-5).
  Builder `scripts/b2_graph_corpus_v44r2.py`; artifacts under gitignored
  `data/b2-graph-corpus/v44-r2/`, numbers-only mirrors here in `v44-r2/`.
- **Holdout seal (hard constant):** the 75-row holdout v4.4 is NOT
  recreated, NOT rewritten, NOT read for content. Stage 0 reproduces the
  slice-2 construction in memory and asserts byte-identity: train
  `0463413e…`, holdout `a9760d1f…`, corpus `45ab0ae6…` — all three match
  the seal; `holdout.jsonl` re-verified unchanged at exit.
- **§A taxonomy glossary (mandatory card section):** 1 = identity
  (P-identity/T0), 2 = cosmetic (P-cosmetic/T1), 3 = lexical substitution
  (P-lexsub, template-twin heritage), 4 = paraphrase (P-para-*), 5/6 =
  structural/compression — planned 1.n+1, NOT in this build, 7 = fact edit
  (N-fact-edit, broken-field, mint-top — razor/T3), 8 = topic-adjacent
  (N-para-notdup, DV3 template-twin, near/cross-topic). Silver rule: dup
  iff bilateral entailment with all §8 slots preserved both ways; any slot
  change → class 7 NOT dup; disputed → NOT dup (§8.3).
- **§B.3 razor-block transfer (v4.2 train-824):** the block is 336 rows
  (N-fact-edit 165 + N-para-notdup 27 + P-cosmetic 144); audit: 262/336
  already in the train via the v4.3 inheritance (same pairs), 74/336 are
  holdout-surface copies (transfer FORBIDDEN, listed in the manifest),
  0 new rows; disjointness vs all holdout surfaces + burned = 0 violations.
- **§B.1 class-7 band negatives:** legacy v4.2 same-length fact-token
  cipher (digit cipher, char5_containment < 0.976415, len-delta 0, BOTH
  orientations) on the injection bases; acceptance band cos ∈ [0.98, 1.0];
  rows the raw cipher leaves above 0.985 are dose-placed into
  [0.98, 0.985) by the legacy v4.3 case-dose lever (char geometry +
  tag_jaccard=1.0 unchanged). Minted: 428 rows (120 dose-placed in
  [0.98, 0.985)).
- **§B.2 injection rebalance:** dup mints PREFER classes 3+4:
  P-lexsub 53 (legacy synonym tables, net-zero guard) + P-para-light 55
  (sentence swap + lexical replacement) minted BEFORE any class-1/2 row;
  the formula remainder continues the legacy T0/T1 generator on fresh
  bases: 202 rows (T0 135 / T1 67, 2:1 lineage).
- **§C razor subzone (cos ≥ 0.95 ∧ tag_jaccard = 1.0):**
  pre 854 pos / 572 neg, bins 53/1/518 → post 1000 neg vs 1000 pos —
  parity coefficient exactly 1.0 on the dynamic census reading
  (854 floor + 146 new class-1/2 subzone rows); strict classes-1-2 reading
  1.534; static floor reading 1.171 — ALL readings ≥ 1.0. Bins
  [0.95,0.97) 53 / [0.97,0.985) 121 / [0.985,1.0] 826 — all ≥ 30.
  Добор негативов подзоны: +428.
- **§D probe coverage (sanity v2, band ±0.005 + equal tag_jaccard +
  continuity ≤ 0.005):** self_pair→P-identity@1.0: 542 rows; envelope→
  P-envelope@1.0: 57; cosmetic_twin→P-cosmetic@0.99: 40; fact_edit_twin→
  N-fact-edit@0.99: 76 (max gap 0.0024); unrelated→N-far@0.578 (tj=0.0):
  30 (max gap 0.0028; the fill is the class's own heritage pairing:
  real-part sides × authored far bases of dataset-v4.2 batch-near,
  cross-language pairs quarantined). Gate-runner procedure: count rows of
  the producing stratum with |cos − probe| ≤ 0.005 ∧ tj equal, assert
  count ≥ 30 AND max consecutive in-band gap ≤ 0.005.
- **Scaffold anti-collapse:** every minted class ≥ 20 scaffolds and ≤ 5%
  of its rows per scaffold (max observed: N-far 3.7%); N-fact-edit 187
  scaffolds on 428 rows. Train-injection scaffolds overlapping the sealed
  holdout: 80 rows (ids from the recipe, holdout content NOT read) —
  run-logged as the known limitation per §D.
- **Cross-language quarantine (§A(i)):** script-detect over 8654 pool
  pairs: 370 confident ru/en mismatches quarantined (never new mints);
  6871 mixed-script pairs counted separately, never quarantined.
- **Formula closure (universe surface, anchor-consistent):**
  0.595·(x₃₄+w) = 0.295 + 0.405·(282+w_s) + 0.405·f with the measured
  subzone mass of the continuation; exact integer fixed point
  x₃₄=108, f=27, w=202 (w_s=146), y=428, n=310; universe dup share
  0.404976 vs the 0.405 target. Legacy anchor re-asserted: the same
  formula on the pre-addendum numbers reproduces 429 (and 410 on the
  legacy pre-cleaning numbers).
- **Shipped:** train 4097 (dup 1599 / not-dup 2498; dup share 0.3903),
  765 new rows; hygiene: engine detectors over all 1530 minted sides,
  0 hits. New train fp `bb1ca943…`; holdout fp unchanged `a9760d1f…`.
- **Determinism:** double run byte-identical on all artifacts (verified
  twice, including after ruff formatting of the builder).

## v4.4 — corpus built and sealed (slice-2, prereg v3 §1-§2)

- Builder `scripts/b2_graph_corpus.py`, verifier
  `scripts/b2_graph_corpus_verify.py` (independent re-check from disk).
- Artifacts under gitignored `data/b2-graph-corpus/v44/` (train/holdout jsonl,
  labels, holdout-ids seal, manifest, run-log); numbers-only mirrors here in
  `v44/` (manifest.json, run-log.jsonl, train-manifest.txt, holdout-ids.json).
- Base: v4.3 universe minus the 21 train∩holdout pair-copies (pair-key join
  at build time; 18 v43-holdout + 3 v42-holdout; 20 duplicate + 1
  not-duplicate train-side rows removed, holdouts untouched; 2399 → 2378).
- Graph stratum 600: edge-direct 240 / edge-common 150 / mint-top 150 /
  control 60; per-component cap 90 (=15% of 600), max observed 90; 46
  components available in pool, 30 covered at selection; ALL not-duplicate
  per prereg §2 semantics (edge-direct = HARD NEG).
- Injection 429 by the ratied formula n = (0.405×(uni_clean_rows+600) −
  uni_clean_dups)/0.595 = 429.4958 → 429 (round-half-up; on legacy
  pre-cleaning numbers the same formula gives 410.175 → the prereg's «≈410»).
  T0 143 self + 143 clone units, T1 143 swapcase (v4.3 real-part lineage
  ratio ≈2:1); bases = pool endpoint sides, holdout/gate/burned-side records
  excluded at MINT time; engine-detector hygiene over every minted side:
  0 hits; vectors re-measured under the frozen embedder pin
  `nano:sha256:3b752e06…`.
- Holdout: sealed 75 rows = 10% (ceil) per (label, stratum) over the
  EVAL-ELIGIBLE new material (707 of 1029 rows; 322 graph rows carry
  side-overlap flags and stay train-only per the slice-1 pre-gate recipe).
  T0 units atomic (self+clone never split). Pair-level 0 vs v4.4 train and
  vs every historical surface (v43-holdout, v42-276, b2-180, v4.3 universe,
  dataset-v3, burned translated); 0 gate-side hits.
- Shipped: train 3332 (dup 1289 / not-dup 2043), holdout 75 (45/30).
  Dup shares: train 0.3869, files 0.3915, formula surface (universe
  accounting) 0.4049 vs 0.405 target.
- Fingerprints (data-contract §5): corpus v4.4
  `45ab0ae642ccf57c73393a2e097036c2f0122a41bb3d92f4e84b06b89bd8928d`,
  train `0463413ec7f4a8935211aef8652f1eed0ad039ede401695dd9191e61614eed98`,
  holdout `a9760d1f06ecdd40530bcc2dc37311f5e5bcd23aad653e03c87a49fe36761a97`.
- Determinism: double run byte-identical on ALL 8 artifacts (no RNG draws —
  construction is order-deterministic; seed 42 recorded per prereg); the
  verifier passes 47/47 checks on both runs.



- Branch: `feat/b2-graph-corpus` (from `origin/main` @ `d01d28f`).
- Builder: `scripts/b2_graph_pool.py` (seed 42; deterministic; double-run
  byte-identical — see "Determinism").
- Full populations + sides: `data/b2-graph-corpus/pool_candidates.jsonl`
  (8654 rows) + `pool_manifest.txt`; `coverage.json` / `run-log.jsonl` /
  `quotas_proposal.json` / `silver-edge-common-30.jsonl` mirrored here.

## Store snapshot and provenance

| Item | Value |
|---|---|
| Store | `/var/home/abyss/.vesma/data/vesma.db` + `vectors.db`, strictly `mode=ro` + `PRAGMA query_only=ON` |
| Snapshot | 4335 memories, 1479 edges, 4307 vectors (live store; grows between runs; both final runs saw an identical pool-relevant snapshot) |
| Edges | 1476 × `relates_to`, provenance `auto-dedupe`, weight 2.0 (100 %) |
| Connected components | 48 (matches TL slice-0 verification; top sizes 261/108/90/54/35/32/25/20/20/19; no singletons) |
| Hygiene scanner | ENGINE detectors (`make_scanner` → provenance `engine`), copied read-only from vesma `6fce6df` (5.6.4) `src/vesmaro/` |
| Detector sha256 | `secrets_detector.py` `4ff7fcc655f16f1307ce986f58d381daa0c091601439214a9d88e0212863ad95`; `danger_detectors.py` `c338af80483dfcd95509a00ec0272278e92fc56243a207c8cb5fb58c27dcef48` |
| Band | [0.85, 0.97) (dataset-v3 construction); mint threshold 0.92; mint-top-K = 3 |
| Embedder pin (store vectors) | nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba |

## Hygiene (frozen engine chain, per b2_targeted_export precedent)

Endpoint pool: 883 edge endpoints → **736 passed**, 147 excluded —
`sql-gate:memory_type` 135 (session_context/conversation endpoints),
`sql-gate:length<80` 12. Zero exclusions for: no-federate, quarantine,
secrets/danger scanner, missing/unpinned embedding, vector size/norm
(matches the b2-edges-20261002 precedent: 0 hygiene exclusions).

Final pool numbers are counted AFTER hygiene.

## Populations (post-hygiene, pre-dedup → final)

| Stratum | Band pairs (pre-dedup) | Final in pool |
|---|---:|---:|
| edge-direct (direct relates_to in band) | 1140 | 1139 |
| edge-common (≥1 common neighbour, no edge, cos < 0.92) | 801 | 801 |
| mint-top (cos ≥ 0.92, no edge, top-3 per record) | 68874 → 1786 after top-K | 1714 |
| control (no graph evidence) | 155854 | 5000 of 155838 (file cap, deterministic round-robin over components) |
| **total in pool file** | | **8654** |

Dedup by pair-hash (3 layers: artifact-side projection, raw-text pair,
store-id pair): 89 dropped as `occupied-text-key` (already present in the
v4.3 universe / dataset-v3); 0 artifact-key / store-id / burned-translated /
within-pool collisions. 3377 corpus rows did not resolve to pool records
(synthetic rows — expected, disclosed).

Probe-vs-build delta: slice-0 numbers (1351 edge / 3730 common / 322870 band)
were counted pre-hygiene over 882 endpoints; this build counts post-hygiene
over 736 passed endpoints (band 226669). Same construction, stricter gate.

## Strata semantics (pre-gate v3 pins; this pool proposes)

- **edge-direct → not-duplicate, HARD NEGATIVE.** Silver basis: edge
  provenance `auto-dedupe` (100 % of edges) + wave-B2 verdict (relates_to =
  connectivity link, median body-ratio 0.04). TL decision, pinned by pre-gate.
- **edge-common → not-duplicate (silver, labeling-policy-b2 §8 policy v1.1).**
  Double-check sample of 30 (below) confirms the negative label mechanically.
- **mint-top → not-duplicate razor-zone T3-class material.** EMPIRICAL
  FINDING: 0 dup-silver pairs in the whole graph stratum (mechanical §8
  sifter). The edge layer already absorbed the duplicate mass at mint time;
  true duplicates sit at cos ≥ 0.97 — ABOVE the band cap by construction
  (dataset-v3 semantics: the band is the "related but distinct" zone; v3-era
  positives were synthesized paraphrases, not raw store pairs).
- **control → not-duplicate (no graph evidence), ablation control.**

## Silver double-check sample — 30 random EDGE-COMMON pairs (seed 42)

All 30 mechanically labeled `not-dup-razor` per §8 (policy v1.1); 0 disputed,
0 cross-language. Feature distribution (token counts, no content):

| Feature | min | median | max |
|---|---:|---:|---:|
| razor (key-fact) tokens | 6 | 38 | 84 |
| envelope (whitelist) tokens | 0 | 0 | 7 |
| other changed tokens | 124 | 293 | 816 |

28/30 pairs differ in record metadata (tags/type/language). common
neighbours: 1 (25 pairs), 2 (5 pairs). similarity 0.862–0.9199.
Per-pair metadata: `silver-edge-common-30.jsonl` (pair ids, hashes, counts).

## Sifter self-test (mechanical §8 sifter, 300 labeled v4.3 rows, seed 42)

Monolingual agreement 0.8108; dup-silver precision 0.79; cross-language rows
excluded from mechanical verdicts (translation twins reformat facts).
Reading: the sifter is silver-grade for NEGATIVE confirmation only; its dup
verdicts never mint positives in this pool. Confusion highlights:
N-para-notdup → dup-silver 21 (known mechanical limit), translated-dup →
not-dup-razor 15 (cross-language reformats), N-fact-edit → disputed 6.

## Disjointness

Pair-level hard asserts — **0 violations**: no pool pair equals a pair of
v42-holdout-276, b2-holdout-180, v43-holdout, the v4.3 universe, dataset-v3
600, gate sets (cb367577, r5_gate_v2 ×3 files) or burned V43-TDUP/TSIB 288.

Side-hash joins (REPORTED as per-pair flags in the pool file, NOT excluded —
record pools are shared by corpus-family design; the base v4.3 train itself
shares sides with b2-holdout: 127/308 holdout sides):

| Surface | Pool pairs flagged | Pool sides hit |
|---|---:|---:|
| b2-holdout-180 sides | 65960 (population) | — |
| v43-holdout sides | 20667 | — |
| v42-holdout-276 sides | 0 | — |
| gate sets (cb367577 3, r5_gate_v2 2, run-b 2, frozen_pool 2) | 2352 | 9 |
| burned embed r4/r5/r7 corpora | 27741 | 167 |

Pre-gate recipe: when carving any future eval/holdout from this pool, exclude
pairs carrying `holdout_side_overlap` / `gate_side_overlap` /
`burned_side_overlap` flags (columns are in every pool row).

## Quota proposals (variants; NOT pinned)

Negative split: edge-direct 40 %, edge-common 25 %, mint-top 25 %, control
10 %; per-component cap 15 % of the variant total (anti-collapse, baseline
0.628 lesson); selection deterministic over pair_id order.

| Variant | edge-direct | edge-common | mint-top | control | components | feasible | pool dup share after | dups needed for 46/54 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 400 | 160 | 100 | 100 | 40 | 24/48 | yes | 0.3588 | 357 |
| 600 | 240 | 150 | 150 | 60 | 28/48 | yes | 0.3396 | 449 |
| 800 | 320 | 200 | 200 | 80 | 30/48 | yes | 0.3223 | 541 |

Class balance: the graph stratum is ALL-NEGATIVE (see empirical finding);
pool-level dup share falls from 0.4046 (inherited universe 1267/3131) toward
0.32–0.36. Reaching the v4.3-style ~46/54 requires positive injection —
T0/T1 synthesis per labeling-policy §4 or an LLM §8 pass over mint-top —
a pre-gate v3 / owner decision, out of this slice's scope.

Direction balance: NOT APPLICABLE — store pairs are undirected objects with
canonical ordering (earlier `created_at` first, tie: smaller id), unlike
synthetic generators with free pair orientation.

Project-tag spread (pool file, side A): 46/48 components represented; top
projects: project:vesma 1808, mnemos-mesh 743, ollama-cloud-provider 667,
mnemos-eyes 584, vesmaro-eyes 420, githubcopilotworkflow 355 (full counter
in `coverage.json`).

## Determinism

Seed 42 everywhere; no wall-clock in content artifacts. Double run on the
same pool-relevant store snapshot: `pool_candidates.jsonl`,
`pool_manifest.txt`, `silver-edge-common-30.jsonl`, `quotas_proposal.json` —
byte-identical (sha256 match). `run-log.jsonl` differs at most in the
`store_snapshot` counter line (the live store appends non-endpoint records
between runs — verified: the only C-vs-D delta was memories 4334 → 4335,
pool untouched). `coverage.json` additionally carries `generated_at_utc` /
`seconds` by design.

## Pool fingerprint

`pool_fingerprint = 87b3d4c22a43befbfd81dc70c83a2cf7fdfce972378a9b550cc100922050cb1e`
(manifest of `pair_id pair_sha256` sorted → BLAKE2b-256; data-contract §5).

## Honest deviations

1. The task's "dup/not-dup как в v4.3: ~46/54" does not reproduce on disk:
   v4.3 universe measures 40.5/59.5 (train 38.6/61.4; v4.2 train 52/48).
   Treated 46/54 as the target; consequences computed per variant above.
2. "Top-1..3 минт = возможные дубли кандидатов": implemented as top-3
   mint-zone neighbours per record; empirically 0 of them are mechanical
   duplicates (all razor-different) — reported as the central finding
   instead of silently rebalancing.
3. Side-hash overlaps with holdouts/gates/burned are FLAGS, not exclusions:
   excluding them would have cut 29 k pairs (incl. a third of edge strata)
   without a leakage argument for a TRAINING pool; the strict side-isolation
   recipe applies when the pre-gate carves eval sets (flags provided).
4. The inherited base carries a mild pre-existing pair-level train/holdout
   overlap (NOT built by this slice): 18/276 v43-holdout pairs and 3/276
   v42-holdout pairs also appear in v43-train with identical labels
   (20 × P-identity + 1 × template-twin; 0 label contradictions). Base is
   inherited as-is per task; flagged for pre-gate v3 / TL.
5. r5 tatoeba-750 jsonl no longer exists on disk (raw archives only); its
   gate-side coverage is approximated by `r5_gate/tatoeba_notes.jsonl`
   (1500 sides, 0 hits) + gate_set text hashes.
6. Live-store drift: the store grew during the session (4329 → 4332
   memories); all reported numbers come from ONE final snapshot; double-run
   determinism verified within that snapshot.
