"""dataset-v4 batch + construction tests (wave D4-1).

Pins, in the wave order:

1. batch structural validation (load_batches): RU/EN 1:1, roles, fact
   anchors occur exactly once;
2. DISJOINTNESS of corpus themes from synth TOPICS / evalsets / the LA-2
   batch / sanity anchors — a test, not luck (evalsets discipline);
3. construction quotas: exact per-stratum counts, label balance, unique
   pair ids, ZERO cross-pair content duplicates, contract-surface sides;
4. labels BY CONSTRUCTION: identity/cosmetic/trans/para → duplicate;
   metadata/fact/near/far → not-duplicate; T3 carries BOTH orientations;
5. transform mechanics: light = sentence swap + lexical replacement with
   edit mass and BELOW the razor zone signature; metadata deltas change
   exactly one metadata surface; fact edits change exactly one fact token;
6. BYTE determinism: double build_pairs → identical canonical JSON;
7. stratified 75/25 split: per (label, stratum) holdout, zero overlap
   (assert_no_pair_overlap), holdout labels never enter train rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_dataset_v4 as gen  # noqa: E402 — repo scripts sys.path shim above

from cortex.data.fingerprints import canonical_json  # noqa: E402
from cortex.data.holdout import assert_no_pair_overlap  # noqa: E402

EXPECTED_STRATA = {
    "P-identity": 96,
    "P-cosmetic": 72,
    "P-para-light": 96,
    "P-para-sub": 48,
    "P-para-struct": 35,
    "P-trans": 60,
    "N-metadata": 112,
    "N-fact-edit": 192,
    "N-near": 49,
    "N-far": 108,
}


@pytest.fixture(scope="module")
def loaded():
    return gen.load_batches()


@pytest.fixture(scope="module")
def pairs(loaded):
    base_ru, base_en, near_rows, paras, facts = loaded
    return gen.build_pairs(base_ru, base_en, near_rows, paras, facts)


def test_batches_validate(loaded) -> None:
    base_ru, base_en, near_rows, paras, facts = loaded
    assert len(base_ru) == len(base_en) == 48
    assert set(base_ru) == set(base_en)
    assert near_rows, "near-topic negatives must exist"
    assert facts, "fact specs must exist"


def test_batch_themes_are_disjoint_from_eval_and_train_surfaces(loaded) -> None:
    base_ru, base_en, near_rows, paras, facts = loaded
    para_flat = [row for rows in paras.values() for row in rows]
    gen.check_disjointness(base_ru, base_en, near_rows, para_flat)


def test_construction_quotas(pairs) -> None:
    by_stratum: dict[str, int] = {}
    for p in pairs:
        by_stratum[p["stratum"]] = by_stratum.get(p["stratum"], 0) + 1
    assert by_stratum == EXPECTED_STRATA
    labels = {"duplicate": 0, "not-duplicate": 0}
    for p in pairs:
        labels[p["label"]] += 1
    assert labels == {"duplicate": 407, "not-duplicate": 461}
    ids = [p["pair_id"] for p in pairs]
    assert len(ids) == len(set(ids)) == 868


def test_sides_carry_the_contract_surface_only(pairs) -> None:
    for p in pairs:
        for side in (p["record"], p["candidate"]):
            assert set(side.keys()) == {
                "title",
                "body",
                "tags",
                "language",
                "record_type",
            }
            assert side["language"] in ("ru", "en")
            assert side["record_type"] in gen.RECORD_TYPES
            assert isinstance(side["tags"], list) and side["tags"]


def test_no_cross_pair_content_duplicates(pairs) -> None:
    seen: dict[str, str] = {}
    for p in pairs:
        key = canonical_json([p["record"], p["candidate"]])
        assert key not in seen, f"{p['pair_id']} duplicates {seen[key]}"
        seen[key] = p["pair_id"]


def test_labels_by_construction(pairs) -> None:
    for p in pairs:
        if p["stratum"].startswith("P-"):
            assert p["label"] == "duplicate", p["pair_id"]
        else:
            assert p["label"] == "not-duplicate", p["pair_id"]


def test_t3_carries_both_orientations(pairs) -> None:
    """Every T3 pair has its mirror in the corpus: the unordered pair set is
    closed under swap (symmetry tail, eval-methodology §7)."""
    facts = [p for p in pairs if p["stratum"] == "N-fact-edit"]
    forward = {
        (canonical_json(p["record"]), canonical_json(p["candidate"])) for p in facts
    }
    reverse = {
        (canonical_json(p["candidate"]), canonical_json(p["record"])) for p in facts
    }
    assert len(facts) == 192
    assert forward == reverse
    assert len(forward) == len(facts)  # every pair distinct, mirror present


def test_light_transform_mass_and_signature(pairs) -> None:
    """Light positives: sentence swap + lexical replacement must move real
    mass (G4) and must NOT carry the razor-zone char signature (G3)."""
    import difflib

    from cortex.data.corner_qa import RAZOR_ZONE, corners_of_pair

    lights = [p for p in pairs if p["stratum"] == "P-para-light"]
    assert len(lights) == 96
    for p in lights:
        c = corners_of_pair({**p, "similarity": 0.98})
        a = p["record"]["body"]
        b = p["candidate"]["body"]
        mass = sum(
            max(len(a[i1:i2]), len(b[j1:j2]))
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
            if tag != "equal"
        )
        assert mass >= 8, p["pair_id"]
        assert c["char5_jaccard"] < RAZOR_ZONE["char5_jaccard_min"], p["pair_id"]
        assert p["record"]["tags"] == p["candidate"]["tags"]
        assert p["record"]["title"] == p["candidate"]["title"]


def test_metadata_transform_changes_exactly_one_surface(pairs) -> None:
    metas = [p for p in pairs if p["stratum"] == "N-metadata"]
    assert len(metas) == 112
    for p in metas:
        a, b = p["record"], p["candidate"]
        assert a["title"] == b["title"]
        assert a["body"] == b["body"]
        deltas = sum(
            (
                a["tags"] != b["tags"],
                a["record_type"] != b["record_type"],
                a["language"] != b["language"],
            )
        )
        assert 1 <= deltas <= 2, p["pair_id"]


def test_fact_edit_changes_exactly_one_token(pairs) -> None:
    for p in pairs:
        if p["stratum"] != "N-fact-edit":
            continue
        a, b = p["record"], p["candidate"]
        assert a["title"] == b["title"]
        assert a["tags"] == b["tags"]
        assert a["language"] == b["language"]
        assert a["body"] != b["body"]


def test_trans_twins_are_cross_language_same_type(pairs) -> None:
    for p in pairs:
        if p["stratum"] != "P-trans":
            continue
        assert p["record"]["language"] == "ru"
        assert p["candidate"]["language"] == "en"
        assert p["record"]["record_type"] == p["candidate"]["record_type"]


def test_identity_positives_pin_the_corner(pairs) -> None:
    """All 96 identity pairs are content-identical sides (48 self-pairs +
    48 byte-clones — the two flavors differ by construction intent, not by
    content: both pin the exact 1.0 corner as duplicates)."""
    idents = [p for p in pairs if p["stratum"] == "P-identity"]
    assert len(idents) == 96
    for p in idents:
        assert canonical_json(p["record"]) == canonical_json(p["candidate"])
    assert len({p["pair_id"] for p in idents}) == 96


def test_double_build_is_byte_identical(loaded) -> None:
    base_ru, base_en, near_rows, paras, facts = loaded
    a = gen.build_pairs(base_ru, base_en, near_rows, paras, facts)
    b = gen.build_pairs(base_ru, base_en, near_rows, paras, facts)
    assert canonical_json(a) == canonical_json(b)


def test_first_spec_per_cell_is_deterministic_quota(loaded) -> None:
    _, _, _, _, facts = loaded
    picked = gen._first_spec_per_cell(facts)
    cells = [(s["key"], s["lang"]) for s in picked]
    assert len(cells) == len(set(cells)) == 96
    # file-order firsts: re-running picks the same specs
    assert [canonical_json(s) for s in picked] == [
        canonical_json(s) for s in gen._first_spec_per_cell(facts)
    ]


def test_split_seventyfive_twentyfive_no_overlap(pairs) -> None:
    train, hold = gen.stratified_split(pairs, gen.HOLDOUT_FRACTION)
    assert len(train) + len(hold) == len(pairs)
    assert len(hold) == 218 and len(train) == 650
    assert_no_pair_overlap([r["pair_id"] for r in train], [r["pair_id"] for r in hold])
    # per (label, stratum) cell the holdout share is exactly ceil(0.25·n)
    cells: dict[tuple[str, str], int] = {}
    hold_cells: dict[tuple[str, str], int] = {}
    for p in pairs:
        cells[(p["label"], p["stratum"])] = cells.get((p["label"], p["stratum"]), 0) + 1
    for p in hold:
        key = (p["label"], p["stratum"])
        hold_cells[key] = hold_cells.get(key, 0) + 1
    import math

    for key, n in cells.items():
        assert hold_cells.get(key, 0) == math.ceil(gen.HOLDOUT_FRACTION * n), key


def test_holdout_labels_never_in_train_rows(pairs) -> None:
    train, hold = gen.stratified_split(pairs, gen.HOLDOUT_FRACTION)
    hold_ids = {r["pair_id"] for r in hold}
    for r in train:
        assert r["pair_id"] not in hold_ids
