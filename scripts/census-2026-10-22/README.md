# B2 M1.1 evidence census — scripts snapshot (2026-10-22)

Copied verbatim from /tmp/b2_census (perishable) into the repo for
reproducibility — canonized per docs/experiments/b2-m11-evidence-census.md §8.
No content dumps: the JSONs carry counters and anonymized side skeletons only
(sha-free shape strings), the store was read strictly read-only.

- census_c1c2.py  — C1/C2 over the sealed pool snapshot (pool_candidates.jsonl)
- probe_c3.py     — live-store probe: silver + edge mass (read-only, §5.1)
- probe_c3b.py    — follow-up: edge bands, dup clusters, pool linkage
- *_numbers.json  — census numbers as canonized in the document
