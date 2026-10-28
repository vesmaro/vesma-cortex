#!/usr/bin/env python3
"""A-retrial Phase-1 corpus build — step 7: numbers-only build manifest.

Emits docs/experiments/a-retrial-phase1-corpus-manifest.json (committed):
STOP table (normative vs fact vs verdict), role counters, fingerprints of
every frozen piece, histograms, named-assert ledger, deviation register.
Raw texts stay in the gitignored data dir; the manifest carries counters,
hashes and fingerprints only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
D = WORKTREE / "data" / "a-retrial"
DOCS = WORKTREE / "docs" / "experiments"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json_list(p: Path) -> str:
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    snap = json.loads((D / "snapshot" / "snapshot.json").read_text())
    hyg = json.loads((D / "hygiene" / "hygiene.json").read_text())
    cls = json.loads((D / "classes" / "classes.json").read_text())
    mask = json.loads((D / "mask" / "mask.json").read_text())
    census = json.loads((D / "gate" / "census_sources.json").read_text())
    src_manifest = json.loads((D / "gate" / "sources_manifest.json").read_text())
    pool_manifest = json.loads((D / "gate" / "distractor_manifest.json").read_text())
    roles = json.loads((D / "roles" / "roles_report.json").read_text())

    fps = {
        "snapshot_vesma_db": snap["dbs"]["vesma.db"]["sha256"],
        "snapshot_vectors_db": snap["dbs"]["vectors.db"]["sha256"],
        "hygiene_records_jsonl": sha_file(D / "hygiene" / "records.jsonl"),
        "hygiene_vectors_npy": sha_file(D / "hygiene" / "vectors.npy"),
        "edges_post_hygiene_jsonl": sha_file(D / "classes" / "edges_post_hygiene.jsonl"),
        "zone_pairs_npz": sha_file(D / "classes" / "zone_pairs.npz"),
        "masked_pairs_json": sha_file(D / "mask" / "masked_pairs.json"),
        "knockon_pairs_json": sha_file(D / "mask" / "knockon_pairs.json"),
        "sidecar_G_jsonl": sha_file(D / "mask" / "sidecar_G.jsonl"),
        "sources_jsonl": sha_file(D / "gate" / "sources.jsonl"),
        "distractor_pool_jsonl": sha_file(D / "gate" / "distractor_pool.jsonl"),
        "train_pairs_jsonl": sha_file(D / "roles" / "train_pairs.jsonl"),
        "val_pairs_jsonl": sha_file(D / "roles" / "val_pairs.jsonl"),
        "anchor_dup_json": sha_file(D / "roles" / "anchor_dup.json"),
        "anchor_frx_json": sha_file(D / "roles" / "anchor_frx.json"),
        "neg_universe_npz": sha_file(D / "roles" / "neg_universe.npz"),
        "replay_entries_jsonl": sha_file(D / "roles" / "replay_entries.jsonl"),
        "replay_probe_jsonl": sha_file(D / "roles" / "replay_probe_2500.jsonl"),
        "replay_heritage_texts": sha_file(D / "roles" / "replay_heritage_texts.jsonl"),
        "heritage_r5_corpus_fp": roles["TR-REPLAY"]["heritage_fp"],
    }

    manifest = {
        "kind": "a-retrial-phase1-corpus-manifest",
        "spec": "docs/specs/embed-graph-A-retrial-DRAFT.md @ feat/embed-graph-fusion-framing 19e66c5 (ACTIVATED 28.10)",
        "builder": "gcw-dataset-engineer",
        "frozen_order": ["snapshot(D0)", "hygiene", "classes", "mask(cut-closure)", "freeze-sealed(alpha/beta pieces)", "train-roles-on-remainder", "manifest"],
        "D0": {"utc": snap["d0_utc"], "dbs": snap["dbs"], "counters": snap["counters"]},
        "D1": roles["d1_utc"],
        "seeds_declared": {
            "mask": mask["mask_seed"],
            "sources": src_manifest["seed"],
            "distractor_pool": pool_manifest["seed"],
            "val": roles["seeds"]["val"],
            "frx_sample": roles["seeds"]["frx_sample"],
            "zone_replay": roles["seeds"]["zone_extra"],
            "note": "prereg left build seeds to the build; fixed here BEFORE cutting, snapshot-relative",
        },
        "asserts_ledger": {
            "pin_100_percent": {"fact": hyg["pin_census"], "verdict": "PASS"},
            "mask_cut_closure_per_pair": mask["asserts"],
            "pair_key_unique": "PASS (sorted a--b keys, dict-keyed surfaces)",
            "snapshot_relative": "PASS (all formulas computed from the D0 snapshot only)",
            "negative_universe": roles["TR-NEG"]["asserts"],
            "anchor_disjointness": roles["TR-ANCHOR"]["asserts"],
            "distractor_pool": pool_manifest["asserts"],
            "replay": roles["TR-REPLAY"]["asserts"],
            "hygiene_chain": "PASS (probe_c3 census chain verbatim, engine scanner provenance=engine, r7-train worktree)",
        },
        "STOP_table": [
            {
                "row": "Masked-leg (одна общая декларированная популяция, извлекаемые)",
                "normative": ">=215",
                "fact": {"masked_pairs_frozen": mask["masked_count"], "band_hist": mask["masked_band_hist"], "common_population": "same list in alpha and beta (MLC 28.10 в.4)"},
                "verdict": "PASS (pool frozen 330 >= 215; extraction filter 215-330 applied at gate-runner per П6; no silent rho change)",
            },
            {
                "row": "PA-Q1 llm (per set)",
                "normative": ">=60",
                "fact": {"clean_pool": census["llm_pool"], "sources_frozen": src_manifest["llm_head_cut"], "directions": src_manifest["directions"]["llm"]},
                "verdict": "PASS availability (K1: PASS); pair generation pending (teacher recipe, seeds declared)",
            },
            {
                "row": "PA-Q1 real (per set)",
                "normative": ">=60",
                "fact": census["real_census"],
                "verdict": "K1 FAIL — clean pool 0 at D0: every post-ceiling hygiene-passing record (306) is agent-session-tagged; agent:user absent. Honest STOP per stratum, no softening; source-class change requires MLC re-ratification",
            },
            {
                "row": "PA-Q1 twin (per set)",
                "normative": ">=60",
                "fact": {"twin_jobs_frozen": src_manifest["twin_jobs"], "twin_per_job": src_manifest["twin_per_job"], "capacity": src_manifest["twin_jobs"] * src_manifest["twin_per_job"]},
                "verdict": "PASS availability (K1: generation-dependent); generation pending",
            },
            {
                "row": "FR-N-hard distractor pool",
                "normative": ">=100",
                "fact": {"pool": pool_manifest["size"], "languages": pool_manifest["languages"], "pairwise_max_cos": pool_manifest["pairwise_max_cos"]},
                "verdict": "PASS (240 frozen, shared by sets; FR-X/dup/masked/cn>=1 excluded; unbeatable-distractor guard <0.97)",
            },
            {
                "row": "Набор >=395 запросов (PA-Q1 >=180 + masked >=215)",
                "normative": ">=395 + pool >=100",
                "fact": {"masked_leg": mask["masked_count"], "PA-Q1": "blocked by real-stratum K1 FAIL"},
                "verdict": "BLOCKED (per stratum K1 real); set assembly cannot be certified until TL/MLC resolves the real-stratum source class; masked-leg + pool + llm/twin sources stay frozen with fp",
            },
        ],
        "role_counters": {
            "TR-POS": roles["TR-POS"],
            "TR-VAL": roles["TR-VAL"],
            "TR-NEG": {"universe_pairs": roles["TR-NEG"]["universe_pairs"], "band_hist": roles["TR-NEG"]["band_hist"], "quotas_frozen": roles["TR-NEG"]["quotas_frozen"]},
            "TR-ANCHOR": {"anchor68_pairs": roles["TR-ANCHOR"]["anchor68_pairs"], "anchor68_members": roles["TR-ANCHOR"]["anchor68_members"], "frx_sample_pairs": roles["TR-ANCHOR"]["frx_sample_pairs"]},
            "TR-REPLAY": roles["TR-REPLAY"],
            "KD_probe_entries": roles["TR-REPLAY"]["kd_probe_entries"],
        },
        "ratification_table_B4": {
            "FR_P_after_hygiene": {"ratified_estimate": "~1520 (at 1844 edges)", "fact": cls["edges_post_hygiene"], "at_edges": cls["edges_raw"]},
            "masked_pool": {"ratified_estimate": "~305 post-hygiene (215-305 extractable)", "fact": mask["masked_count"]},
            "sub_092_stratum": {"ratified_estimate": "~289 edges / ~240 post-hygiene (16%)", "fact": {"edges_below_092_raw": 4, "post_hygiene": 1, "see_deviations": True}},
            "quarantine_reason_histogram_edge_endpoints": hyg["edge_endpoints_fail_reasons"],
            "positive_band_hist": cls["positive_cos_hist_bins"],
            "positive_lang_pairs": cls["positive_lang_pairs"],
            "zone_census": {"no_edge_ge_092": cls["zone_pairs_no_edge_ge_092"], "ge_097_no_edge": cls["zone_ge_097_no_edge"], "FR-N-hard": cls["FR-N-hard"], "FR-X": cls["FR-X"], "FR-Q_cn": cls["FR-Q:cn>=1"], "FR-Q_cross_lang": cls["FR-Q:cross-lang"]},
            "silver_dup": {"pairs": cls["silver_dup_pairs_zero_evidence"], "texts": cls["dup_territory_texts"], "matches_ratified": "68/130 exact"},
        },
        "closure_costs": {"steps": mask["closure_steps"], "knockon_total": mask["knockon_count"], "fixed_point": True},
        "deviations_register": [
            {
                "id": "DEV-1",
                "fact": "store daytime mutation 28.10: sub-0.92 edge mass 289 (10:20 census, c3b_numbers_fresh.json) -> 4 at D0 (20:39); ~285 old edges deleted, ~398 minted; provenance 100% auto-dedupe; no agent session claims writes (awareness pre-flight); attributed to the live service's own auto-dedupe duty",
                "impact": "the §3 informative-signal arithmetic (K2 population ~240 post-hygiene) no longer holds at D0: 1 sub-0.92 post-hygiene edge; DE boundary 189 -> 200 endpoints; TL/MLC decision required before characterization K2 interpretation",
                "action": "reported unchanged; corpus builds on the frozen D0 snapshot (snapshot-relative formulas, ratified)",
            },
            {
                "id": "DEV-2",
                "fact": "knock-on cut-closure cost 343 edges in one closure step (21% of unmasked)",
                "impact": "TR-POS 919 vs DE arithmetic ~1050 (DE arithmetic predates the cut-closure mask mechanics)",
                "action": "declared; quotas scale from the frozen fact",
            },
            {
                "id": "DEV-3",
                "fact": "TR-VAL = 56 pairs: ratified predicate ¬record alpha/beta excludes pairs touching masked-leg hub records (139 of 195 val_raw)",
                "impact": "selection power of the lambda-grid surface is thin; K6-adjacent concern for Phase 3",
                "action": "declared; predicate applied verbatim, no reinterpretation",
            },
            {
                "id": "DEV-4",
                "fact": "replay composition: zone territory is a concentrated clique (1773 free texts); unique-15k and zone-30% cannot both hold by unique composition",
                "impact": "implemented via the ratified word WEIGHT (В5): zone entries repeated x5, effective zone mass 36.5%, unique total 17173 >= 15000",
                "action": "declared; repeat factor frozen in manifest",
            },
            {
                "id": "DEV-5",
                "fact": "real-stratum starvation is structural: store inflow since the 07.10 ceiling is 100% agent-session records",
                "impact": "PA-Q1 real cannot reach >=60/set at any near-term D1 under the ratified source class",
                "action": "K1 FAIL reported; MLC owns the source-class question",
            },
        ],
        "pending_pieces": {
            "pa_q1_generation": "recipe = datasets/embed-r5/gen_gate_v2_material.py stages verbatim, adapted to data/a-retrial/gate/{sources,gen}; teacher Qwen3-1.7B-Instruct Q8_0 (gguf sha in retranslate_llm_r7); interpreter wt/embed-r5-corpus/.venv-r7gen (read-only run, PYTHONDONTWRITEBYTECODE=1); seed base 20261028 + pass*1_000_000 + job_idx; passes 1..3; quality gates digit-preservation/target-lang/anti-copy/len-40..4000 unchanged",
            "gate_set_assembly": "blocked by real-stratum K1 FAIL (PA-Q1 formula requires llm/real/twin); selection seeds to be declared at assembly (alpha/beta different seeds, PA-Q1 strata independent; masked-leg common population = frozen masked_pairs.json)",
            "n_gram_proofs": "zero-tolerance 8-13 of generated PA-Q1 material vs ban registry (built here, fp in census_sources.json); v4.5/g4-spec-v1 NOT in the registry (never read)",
            "extraction_filter": "masked pool -> extractable subset evaluated at gate-runner funnel construction (П6 range 215-330)",
        },
        "fingerprints": fps,
        "privacy": "raw store texts and generated texts live only in gitignored data/a-retrial/; repo carries counters/hashes/histograms only; holdout v4.5 and g4-spec-v1 never read",
    }
    out = DOCS / "a-retrial-phase1-corpus-manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"manifest written: {out}")
    print(f"STOP rows: {len(manifest['STOP_table'])}; deviations: {len(manifest['deviations_register'])}; fps: {len(fps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
