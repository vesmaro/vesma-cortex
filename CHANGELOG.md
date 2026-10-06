# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/#semantic-versioning).

## [Unreleased]

## [0.2.1] - 2026-10-06

### Added

- **Zoned monotonicity v2 in the evalsets gate** (PR #19, la4): the runner
  now applies the ratified zoned rule (gate_contract v3 `monotonicity_v2`) —
  the razor zone (0.95, 1.0) is two-valued, steps starting strictly above its
  low edge are not asserted, outside the zone the non-increase holds.
  Single source for the edge constant (`RAZOR_ZONE_COS_LOW` shared with the
  sanity suite); `REPORT_SCHEMA_VERSION` 1 -> 2 (monotonicity detail
  semantics changed).
- **Near-0009 watchlist module** (`cortex.data.watchlist`, PR #19, la4):
  near-miss observation quota per the calibration-b2p prereg §9 (SCOLET) and
  the evalsets manifest; documented in the la4 zone-rule addenda.

### Fixed

- **sdist privacy gate pruned recursively** (PR #18): hatchling's bare
  directory patterns (`/datasets`) match the directory entry but do not
  prune its children — the 0.2.0 sdist leaked `datasets/` (22 files, the
  training corpus). All four exclude patterns now carry explicit `/**`
  forms. Root cause is the same as the earlier `data/stage2-dry` leak.
  (ADR 0003 R2: training-side data never ship.)

### Packaging

- 0.2.0 shipped on PyPI as a **wheel-only** release: the sdist was caught by
  the release-engineer privacy check (datasets leak above) and was never
  published. 0.2.1 restores the full wheel + sdist pair from the clean
  build with the recursive gate in place.

## [0.2.0] - 2026-10-05

Initial PyPI release (wheel-only; see the sdist note under 0.2.1).
Decision-model development library: pair features, D/N training, selection,
ONNX export, single-shot eval, machine bundle registry
(`cortex/models/vesma-cortex-v1/`).

[Unreleased]: https://github.com/vesmaro/vesma-cortex/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/vesmaro/vesma-cortex/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vesmaro/vesma-cortex/releases/tag/v0.2.0