# MODEL CARD — conflict-detector-v1

- Trained: 2026-11-09T17:28:05.128+00:00 (host core-51, seed 42)
- Task: binary contradicts/not-contradicts on memory record PAIRS (ADR 0002 duty #6; taxonomy design §2, classes CD-1..6 / NC-1..5)
- Decision rule: production circuit = version-guard (cg-guard-v1, one-sided NOT, NC-3 predicate) then LightGBM P(contradicts), cut 0.5, no threshold search (ratified MLC-9)
- Features: frozen CORE-13 `cortex.features.pair` (sha256(names) dd86228f8c634f28…)
- Embedder: nano 3b752e0671a5… (A-frozen), surface n1(title)\n+n1(body)\n+n1(tags) [:4096]
- Corpus: 600 pairs x 2 orientations = 1200 rows (sha256 0be52dd220c79281…); holdout CD-holdout-v1 SEALED (180 pairs), read exactly once by the gate runner
- Baselines (train): BL-band 0.49 / BL-digit 0.57 / BL-digit+guard 0.66 (target max(BL) = 0.66)
- CV characterization (honesty, no selection): OOF acc 0.9733, OOF brier 0.0241 (5 folds by scaffold)
- Floors: models/conflict-detector-v1/floors.json (ML-STD-0001/0004; published BEFORE gate_start; re-measure after single-shot forbidden)
- Gates §6.2: verdict = conjunction; ADOPT/DECLINE by the single-shot gate runner (implemented, executed only on the explicit TL command)
- Glossary: docs/experiments/conflict-detector-corpus-card.md (taxonomy §2 dataset-card glossary)
- Known limitation (design §2.3): razor-zone CD-1 vs NC-3 is indistinguishable on the 13-feature surface — certified by the guard code leg + g_nc3_exact/g_razor, NOT by the model
