"""Near-miss watchlist tests (card la4-layer-a-findings).

Pins, in order:

1. the selection criterion — pure, no ONNX: a probe shaped EXACTLY like
   the near-0009 finding (release-v2, the geometry reproduced here) is
   admitted with its measured facts; healthy probes (score ≥ cut) are
   NOT; off-zone / wrong-class / metadata-delta probes are NOT;
2. the registry: row schema round-trip, append-only uniqueness, missing
   file = honest empty registry, malformed rows fail loud;
3. the quota contract: v4.3+ refuse without the family, v4.2 exempt,
   resolved registry no-op, quota satisfied passes;
4. the runner integration: build_eval_set probes flow through
   near_miss_evaluations with hand-assigned probabilities;
5. near-0009 lives VERBATIM in the committed registry with the
   documented key facts (calibration-b2p.md Layer A finding 2) and
   status=open — the registry must not silently lose the case;
6. the runner monotonicity gate is ZONED (eval-methodology §10.2): the
   0.99 → 0.95 zone-crossing rise is not a violation, the 0.8 → 0.5
   out-of-zone rise still is (the probe row for both is the same fixed
   ladder group — the la4 instrument-debt fix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cortex.data import watchlist as wl  # noqa: E402 — repo sys.path shim above
from cortex.evalsets.generate import (  # noqa: E402
    DEFAULT_SEED,
    MERGE_V1,
    build_eval_set,
    probe_row,
)

# ── fixtures: the near-0009 shape (release-v2 geometry, synthetic text) ───────


def _row(pair_id: str, body: str, cls: str = "near-identity-twin", **kw) -> dict:
    return {
        "pair_id": pair_id,
        "class": cls,
        "label": kw.get("label", "duplicate"),
        "similarity": kw.get("similarity", 0.99),
        "record": {
            "title": "Полив комнатных растений",
            "body": body,
            "language": "ru",
            "record_type": "fact",
            "tags": ["plants", "care"],
        },
        "candidate": {
            "title": "Полив комнатных растений",
            "body": kw.get("cand_body", body.replace(" — ", " - ")),
            "language": "ru",
            "record_type": "fact",
            "tags": ["plants", "care"],
        },
    }


NEAR9_BODY = (
    "Суккуленты — раз в две недели, папоротник — раз в три дня, "
    "остальное по сухому верхнему слою. Вода отстоянная, лишнее из "
    "поддона сливается."
)


# ── 1. the selection criterion ────────────────────────────────────────────────


def test_criterion_admits_the_near_0009_shape_and_carries_its_facts() -> None:
    row = _row("near-0009", NEAR9_BODY)
    hits = wl.near_miss_evaluations([row], [0.009977])
    assert len(hits) == 1
    hit = hits[0]
    assert hit["pair_id"] == "near-0009"
    # the documented geometry of the finding (calibration-b2p.md):
    assert hit["key_facts"]["char5_jaccard"] == pytest.approx(0.9018, abs=1e-3)
    assert hit["key_facts"]["char5_containment"] == pytest.approx(0.9484, abs=1e-3)
    assert hit["key_facts"]["tag_jaccard"] == 1.0
    assert hit["key_facts"]["len_deltas_zero"] is True
    assert hit["key_facts"]["body_edit_mass_chars"] == 2
    assert hit["probability"] == pytest.approx(0.009977)


def test_criterion_refuses_healthy_offzone_and_metadata_cases() -> None:
    healthy = _row("near-0001", NEAR9_BODY)
    offzone = _row("near-0002", NEAR9_BODY, similarity=0.94)
    meta_delta = {
        **_row("near-0003", NEAR9_BODY),
        "candidate": {**_row("near-0003", NEAR9_BODY)["candidate"], "tags": ["plants"]},
    }
    wrong_class = _row("near-0004", NEAR9_BODY, cls="monotonicity-ladder")
    # healthy score (≥ cut): geometry matches, but it is NOT a near-miss
    assert (
        wl.near_miss_evaluations(
            [healthy, offzone, meta_delta, wrong_class], [0.9, 0.9, 0.9, 0.9]
        )
        == []
    )
    # red scores: the off-zone / metadata-delta / wrong-class rows stay
    # refused by the GEOMETRY — only the healthy-geometry row is admitted
    hits = wl.near_miss_evaluations(
        [healthy, offzone, meta_delta, wrong_class], [0.01, 0.01, 0.01, 0.01]
    )
    assert [h["pair_id"] for h in hits] == ["near-0001"]


def test_criterion_length_mismatch_fails_loud() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        wl.near_miss_evaluations([_row("x", NEAR9_BODY)], [0.1, 0.2])


# ── 2. the registry ───────────────────────────────────────────────────────────


def test_registry_row_round_trip_and_schema(tmp_path: Path) -> None:
    hit = wl.near_miss_evaluations([_row("near-0009", NEAR9_BODY)], [0.009977])[0]
    row = wl.registry_row(
        hit,
        weights_sha256="00fc6b71c62dbfa8268db358b39c0a4d9f30148805a4a18986487bc70e8346b2",
        added="2026-10-05",
        evidence="test evidence line",
    )
    assert row["status"] == "open"
    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = wl.load_registry(path)
    assert loaded == [row]


def test_registry_missing_file_is_empty(tmp_path: Path) -> None:
    assert wl.load_registry(tmp_path / "absent.jsonl") == []


def test_registry_malformed_rows_fail_loud(tmp_path: Path) -> None:
    bad_status = tmp_path / "bad-status.jsonl"
    bad_status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "near_miss_id": "x",
                "key_facts": {},
                "weights_sha256": "w" * 64,
                "evidence": "e",
                "status": "closed",
                "added": "2026-10-05",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status must be open\\|resolved"):
        wl.load_registry(bad_status)

    dup = tmp_path / "dup.jsonl"
    row = {
        "schema_version": 1,
        "near_miss_id": "x",
        "key_facts": {},
        "weights_sha256": "w" * 64,
        "evidence": "e",
        "status": "open",
        "added": "2026-10-05",
    }
    dup.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate near_miss_id"):
        wl.load_registry(dup)

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        wl.load_registry(broken)


# ── 3. the quota contract ─────────────────────────────────────────────────────


def _family_rows(n: int) -> list[dict]:
    """Pairs of the EXACT near-0009 family: dash-swap body twins (mass 2,
    zero len-deltas), full matching metadata, high measured cosine — the
    family geometry the quota demands (char5 containment ≥ 0.90 is met by
    construction: 3 em-dash replacements inside a long body keep the
    char profile dense)."""
    rows = []
    base = (
        "Суккуленты — раз в две недели, папоротник — раз в три дня, "
        "остальное по сухому верхнему слою. Вода отстоянная, лишнее из "
        "поддона сливается. Запись {i}: вариант с вариацией темы и "
        "небольшим дополнением в конце."
    )
    for i in range(n):
        body = base.format(i=i)
        rows.append(
            {
                "pair_id": f"W-{i:04d}",
                "label": "duplicate",
                "stratum": "P-cosmetic",
                "record": {
                    "title": f"Полив комнатных растений {i}",
                    "body": body,
                    "language": "ru",
                    "record_type": "fact",
                    "tags": ["plants", "care"],
                },
                "candidate": {
                    "title": f"Полив комнатных растений {i}",
                    "body": body.replace(" — ", " - "),
                    "language": "ru",
                    "record_type": "fact",
                    "tags": ["plants", "care"],
                },
                "similarity": 0.99,
            }
        )
    return rows


def _open_registry() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "near_miss_id": "near-0009",
            "key_facts": {},
            "weights_sha256": "0" * 64,
            "evidence": "e",
            "status": "open",
            "added": "2026-10-05",
        }
    ]


def test_quota_refuses_v43_without_the_family() -> None:
    rows = [{**p, "similarity": 0.6} for p in _family_rows(0)]  # no family at all
    rows += [
        {
            "pair_id": "X-1",
            "label": "not-duplicate",
            "stratum": "N-far",
            "record": {"title": "a", "body": "b"},
            "candidate": {"title": "a", "body": "c"},
            "similarity": 0.5,
        }
    ]
    refusals = wl.quota_violations(
        rows, corpus_version="4.3", registry=_open_registry()
    )
    assert len(refusals) == 1
    assert "near-0009" in refusals[0]
    assert "< 20" in refusals[0]


def test_quota_v42_exempt_and_resolved_registry_noop() -> None:
    refusals_42 = wl.quota_violations(
        [], corpus_version="4.2", registry=_open_registry()
    )
    assert refusals_42 == []
    resolved = [dict(_open_registry()[0], status="resolved")]
    assert wl.quota_violations([], corpus_version="4.3", registry=resolved) == []
    assert wl.quota_violations([], corpus_version="4.3", registry=[]) == []


def test_quota_passes_with_family_mass() -> None:
    rows = _family_rows(int(wl.WATCHLIST_QUOTA) + 5)
    assert (
        wl.quota_violations(rows, corpus_version="4.3", registry=_open_registry()) == []
    )


def test_quota_family_band_excludes_identity_corner_and_heavy_edits() -> None:
    # mass 0 (identity corner) must NOT count toward the family quota
    corner_only = [
        {
            "pair_id": "C-1",
            "label": "duplicate",
            "stratum": "P-identity",
            "record": {"title": "t", "body": "same body"},
            "candidate": {"title": "t", "body": "same body"},
            "similarity": 0.99,
        }
    ]
    assert wl.quota_violations(
        corner_only, corpus_version="4.3", registry=_open_registry()
    )
    # a heavy edit (mass > 4) is not the near-0009 family either
    heavy = [
        {
            "pair_id": "H-1",
            "label": "duplicate",
            "stratum": "P-para-sub",
            "record": {"title": "t", "body": NEAR9_BODY},
            "candidate": {
                "title": "t",
                "body": "Совершенно другой текст про велосипеды и горы, дальние поездки.",
            },
            "similarity": 0.99,
        }
    ]
    assert wl.quota_violations(heavy, corpus_version="4.3", registry=_open_registry())


# ── 4. runner integration ─────────────────────────────────────────────────────


def test_eval_set_probes_flow_through_the_selector() -> None:
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    rows = [probe_row(p) for p in eval_set.probes]
    probs = [0.99 if p.label == "duplicate" else 0.01 for p in eval_set.probes]
    # healthy scores → no near-misses (even though the geometry matches)
    assert wl.near_miss_evaluations(rows, probs) == []
    # zero out ONLY the zero-len-delta near-identity probes → exactly those
    # are admitted: the criterion is the near-0009 family (zero len-deltas —
    # the idiom twins near-0001/0006 carry a whitespace-variant with a
    # non-zero len-delta and stay OUT of the watchlist by design)
    family_ids = {
        p.pair_id
        for p in eval_set.probes
        if p.probe_class == "near-identity-twin"
        and wl.near_miss_evaluations([probe_row(p)], [0.0]) != []
    }
    assert family_ids, "the merge-v1 recipe must carry zero-len-delta twins"
    probs2 = [
        0.0 if p.pair_id in family_ids else pr for p, pr in zip(eval_set.probes, probs)
    ]
    hits = wl.near_miss_evaluations(rows, probs2)
    assert {h["pair_id"] for h in hits} == family_ids


# ── 5. the committed registry carries near-0009 verbatim ─────────────────────


def test_committed_registry_carries_near_0009_open() -> None:
    rows = wl.load_registry()  # default path: datasets/watchlist/near-miss.jsonl
    near9 = [r for r in rows if r["near_miss_id"] == "near-0009"]
    assert len(near9) == 1
    row = near9[0]
    assert row["status"] == "open"
    assert row["probe_class"] == "near-identity-twin"
    assert row["probability"] == pytest.approx(0.009977, abs=1e-4)
    assert row["key_facts"]["char5_jaccard"] == pytest.approx(0.90184)
    assert row["key_facts"]["char5_containment"] == pytest.approx(0.948387)
    assert row["key_facts"]["len_deltas_zero"] is True
    assert row["weights_sha256"].startswith("00fc6b71")
    # the quota contract is bound to versions ≥ 4.3
    assert wl.MIN_CORPUS_VERSION_CARRYING_QUOTA == (4, 3)
    assert wl.WATCHLIST_QUOTA == 20


def test_zone_rule_constants_shared_with_sanity_suite() -> None:
    """The zoned rule single-source: the runner must consult the SAME
    constant as the sanity suite and the gate contract (coherence with
    test_eval_sanity — mirrored here for the runner side)."""
    from cortex.eval.sanity import RAZOR_ZONE_COS_LOW
    import cortex.evalsets.runner as runner

    assert runner.RAZOR_ZONE_COS_LOW is RAZOR_ZONE_COS_LOW
    assert RAZOR_ZONE_COS_LOW == 0.95
    assert runner.REPORT_SCHEMA_VERSION == 2
