# datasets/v43/real-part — ASSEMBLY NUMBERS (no content, sanction 2026-10-06)

Generated: 2026-10-07T18:07:47Z by `scripts/assemble_real_part.py`.

## Sources (read-only, deduped by content hash — addendum 6 §A.3)

| source | SQL candidates | accepted |
|---|---:|---:|
| live | 3146 | 2924 |
| backup-20260920 | 22 | 22 |
| backup-20260921 | 2528 | 2478 |
| backup-20260924 | 2622 | 2562 |

- unique content hashes after dedup: **2418**
- duplicate rows dropped: 5568
- RU-centred share (Cyrillic ≥ 0.2 of content): 1326 (54.8%), non-RU 1092 (addendum-7 reference ~69%)
- memory types: `{'note': 2205, 'snippet': 148, 'fact': 65}`
- scanner provenance: **engine**
- exclusion counters: `{'excluded:high-entropy:backup-20260921': 4, 'excluded:high-entropy:backup-20260924': 4, 'excluded:high-entropy:live': 4, 'excluded:privacy-tag:backup-20260921': 46, 'excluded:privacy-tag:backup-20260924': 56, 'excluded:privacy-tag:live': 218}`
- fingerprint (blake2b-256 of the sorted content_hash manifest): `d06be1e351e2f9fb3ea77c82538d6c68630890d441f44e813cdbdf67f0990e74`

## Dedup / fingerprint logic (data-contract §5 reuse)

- content hash = sha256(hex) over the record content — the engine's
  `vesmaro.ccr.content_hash` convention (single-source rule).
- manifest = sorted `content_hash content_hash` lines,
  corpus fingerprint = blake2b-256 (`cortex.data.fingerprints`).
- first-seen wins (live store first; backups fill what it no
  longer holds); provenance keeps every first-seen source.

## Privacy

- record text lives ONLY in `datasets/v43/real-part.jsonl` —
  gitignored (`/datasets/*` block, no negation for v43/).
- every store opened via `file:…?mode=ro` + `PRAGMA query_only=ON`.
- tags filtered by prefixes: task:, telemetry

NOTE (brief vs addendum on the `mnemos:` prefix): the literal brief
default would cut the corpus to 34 rows — the `mnemos:*` tags are the
storage data-contract markers riding 3,324 of 3,647 live domain rows.
Addendum 6 §A.3 («весь доступный доменный срез ≤2.7k») wins: the
DEFAULT keeps `mnemos:` records; `--brief-tag-filter` reproduces the
literal-brief cut for the TL gate to compare. Telemetry lives in
memory_type session_context/conversation — excluded by the domain
type gate (note/snippet/fact).
