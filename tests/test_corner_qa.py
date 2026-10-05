"""Corner-QA gate tests (P0 §8.1 + labeling-policy §5, pre-train gate).

The #480 lesson in test form: each gate must CATCH the defect it was
written for, on a hand-built mini corpus, and pass a healthy one. All
rows carry hand-assigned similarity (no embedder here) — the gate judges
the corpus, it never re-measures cosines.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cortex.data.corner_qa import (
    DEFAULT_THRESHOLDS,
    CornerQAThresholds,
    corner_qa_counters,
    corner_qa_violations,
    run_corner_qa,
    thresholds_from_gate_contract,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def rec(
    title: str,
    body: str,
    *,
    tags: tuple[str, ...] = ("home",),
    lang: str = "ru",
    rtype: str = "note",
) -> dict:
    return {
        "title": title,
        "body": body,
        "tags": list(tags),
        "language": lang,
        "record_type": rtype,
    }


def row(pid: str, label: str, a: dict, b: dict, sim: float, stratum: str = "X") -> dict:
    return {
        "pair_id": pid,
        "label": label,
        "stratum": stratum,
        "record": a,
        "candidate": b,
        "similarity": sim,
    }


BASE_BODY = (
    "Фильтр насоса чищу раз в три месяца, последний раз 12 марта. "
    "После чистки барабан перестал гудеть на отжиме. "
    "Машинка стоит на кухне, вода уходит в сифон без протечек, "
    "сливной шланг закреплён на высоте 65 см и не дёргается при отжиме."
) * 2


def healthy_rows() -> list[dict]:
    """A miniature but gate-complete corpus: every class the policy needs."""
    base = rec("Фильтр насоса", BASE_BODY, tags=("home", "appliance"))
    paraphrase = rec(
        "Фильтр насоса",
        "После чистки барабан перестал гудеть на отжиме. "
        "Фильтр насоса чищу каждые три месяца, последний раз 12 марта.",
    )
    fact_edit = rec(
        "Фильтр насоса",
        BASE_BODY.replace("12 марта", "12 апреля"),
    )
    meta = rec("Фильтр насоса", BASE_BODY, tags=("home", "appliance", "archive"))
    other = rec(
        "Кормушка для птиц",
        "Кормушка висит в двух метрах от куста. Наполняю раз в два дня.",
    )
    return [
        row("p-ident-1", "duplicate", base, base, 1.0, "P-identity"),
        row("p-ident-2", "duplicate", other, other, 1.0, "P-identity"),
        row(
            "p-cosm-1",
            "duplicate",
            base,
            rec("ФИЛЬТР НАСОСА", BASE_BODY.upper(), tags=("home", "appliance")),
            1.0,
            "P-cosmetic",
        ),
        row(
            "p-cosm-2",
            "duplicate",
            other,
            rec(
                "Кормушка для птиц",
                "Кормушка висит в двух метрах от куста.  Наполняю раз в два дня.",
                tags=("garden",),
            ),
            1.0,
            "P-cosmetic",
        ),
        row("p-light-1", "duplicate", base, paraphrase, 0.93, "P-para-light"),
        row("n-fact-1", "not-duplicate", base, fact_edit, 0.99, "N-fact-edit"),
        row("n-fact-2", "not-duplicate", fact_edit, base, 0.99, "N-fact-edit"),
        row("n-meta-1", "not-duplicate", base, meta, 1.0, "N-metadata"),
        row(
            "n-near-1",
            "not-duplicate",
            base,
            rec(
                "Фильтр заливного клапана",
                "Клапан заливки изношен, заменили 3 апреля. Машинка перестала недоливать воду.",
            ),
            0.85,
            "N-near",
        ),
        row(
            "n-far-1",
            "not-duplicate",
            base,
            rec(
                "Bird feeder",
                "The feeder hangs two meters from the bush. I refill it every other day.",
                lang="en",
            ),
            0.50,
            "N-far",
        ),
        row(
            "n-far-2",
            "not-duplicate",
            other,
            rec(
                "Паспорт: сроки",
                "Заявление подали 12 мая, готовность пришла 27 мая.",
                rtype="fact",
            ),
            0.48,
            "N-far",
        ),
    ]


LOOSE = CornerQAThresholds(
    min_corner_dup_positives=2,
    min_identity_class_positives=2,
    min_metadata_negatives=1,
    min_fact_edit_negatives=2,
    min_type_mismatch_fraction=0.0,
    min_lang_mismatch_fraction=0.0,
    min_pairs_below_cos_055=1,
    min_pairs_below_cos_070=1,
    min_pairs_below_cos_084=2,
)


def test_healthy_mini_corpus_passes() -> None:
    ok, report = run_corner_qa(healthy_rows(), LOOSE)
    assert ok, report["violations"]
    assert report["clone_negatives_count"] == 0
    assert report["corner_dup_positives_count"] == 3  # 2 identity + case twin
    assert report["identity_class_positives_count"] == 3
    assert report["metadata_negatives_count"] == 1
    assert report["razor_zone_positives_count"] == 0


def test_clone_negative_is_refused_with_zero_tolerance() -> None:
    rows = healthy_rows()
    rows.append(
        row(
            "clone-evil",
            "not-duplicate",
            healthy_rows()[0]["record"],
            healthy_rows()[0]["candidate"],
            1.0,
            "evil",
        )
    )
    ok, report = run_corner_qa(rows, LOOSE)
    assert not ok
    assert any("clone_negatives=1" in v for v in report["violations"])
    assert report["clone_negatives"] == ["clone-evil"]


def test_t2_metadata_negative_is_not_a_clone() -> None:
    """The full-surface rule: a text-identical negative with a metadata
    delta is the REQUIRED T2 class, never counted as a clone."""
    ok, report = run_corner_qa(healthy_rows(), LOOSE)
    assert ok
    assert report["clone_negatives_count"] == 0
    assert report["metadata_negatives"] == ["n-meta-1"]


def test_corner_dup_quota_is_enforced() -> None:
    rows = [
        r for r in healthy_rows() if r["stratum"] not in ("P-identity", "P-cosmetic")
    ]
    ok, report = run_corner_qa(rows, LOOSE)
    assert not ok
    assert any("corner_dup_positives=0" in v for v in report["violations"])


def test_constant_feature_is_refused() -> None:
    rows = healthy_rows()
    # make every pair same-language, same-type: lang_match/type_match die
    for r in rows:
        for side in (r["record"], r["candidate"]):
            side["language"] = "ru"
            side["record_type"] = "note"
    counters = corner_qa_counters(rows)
    assert "lang_match" in counters["constant_features"]
    assert "type_match" in counters["constant_features"]
    assert corner_qa_violations(counters, LOOSE)


def test_g1_ceiling_catches_the_480_inversion() -> None:
    """The v3 shape: without corner positives, a broken-field negative sits
    ABOVE the positive char5 maximum (v3: 0.9979 > 0.9967) — the G1 ceiling
    must name the feature and refuse. The negative body is long so a single
    substitution keeps the razor signature (char5 ≥ 0.977)."""
    long_body = BASE_BODY * 10
    rows = [
        r for r in healthy_rows() if r["stratum"] not in ("P-identity", "P-cosmetic")
    ]
    rows.append(
        row(
            "n-broken-high",
            "not-duplicate",
            rec("Фильтр насоса", long_body),
            rec("Фильтр насоса", long_body.replace("чистки", "чистка", 1)),
            0.999,
            "evil",
        )
    )
    counters = corner_qa_counters(rows)
    pos_max = counters["g1_ceiling"]["char5_jaccard"]["max_positive"]
    neg_max = counters["g1_ceiling"]["char5_jaccard"]["max_negative"]
    assert neg_max > pos_max, "fixture must reproduce the v3 inversion shape"
    assert neg_max >= 0.977, "fixture must sit in the razor band like v3"
    violations = corner_qa_violations(counters, LOOSE)
    assert any("G1 ceiling violated for char5_jaccard" in v for v in violations), (
        violations
    )
    # with the corner populated (full healthy corpus + the long positive
    # twin) the ceiling holds: identity pins 1.0 above every negative
    full_rows = healthy_rows()
    full_rows.append(
        row(
            "p-ident-long",
            "duplicate",
            rec("Фильтр насоса", long_body),
            rec("Фильтр насоса", long_body),
            1.0,
            "P-identity",
        )
    )
    full = corner_qa_counters(full_rows)
    assert full["g1_ceiling"]["char5_jaccard"]["max_negative"] <= (
        full["g1_ceiling"]["char5_jaccard"]["max_positive"] + 1e-9
    )


def test_g1_ceiling_violation_is_reported() -> None:
    counters = corner_qa_counters(healthy_rows())
    counters["g1_ceiling"]["char5_jaccard"] = {
        "max_positive": 0.95,
        "max_negative": 0.99,
    }
    violations = corner_qa_violations(counters, LOOSE)
    assert any("G1 ceiling violated for char5_jaccard" in v for v in violations)


def test_razor_zone_positive_is_refused() -> None:
    """A positive carrying the T3 char-signature (open razor zone, policy
    §5-G3) is generation noise — refused. The body is long enough that a
    single same-length substitution keeps char5 ≥ 0.977 with full len
    deltas and cos ≥ 0.995."""
    long_body = BASE_BODY * 10
    assert long_body.count("чистки") == 20
    rows = healthy_rows()
    rows.append(
        row(
            "p-razor-evil",
            "duplicate",
            rec("Фильтр насоса", long_body),
            rec("Фильтр насоса", long_body.replace("чистки", "чистка", 1)),
            0.999,
            "evil",
        )
    )
    ok, report = run_corner_qa(rows, LOOSE)
    assert not ok
    assert any("razor_zone_positives=1" in v for v in report["violations"])


def test_g4_connector_edit_positive_is_refused() -> None:
    """A 3-char connector edit on the body with untouched title/tags is the
    v3 light-noise class (policy §5-G4) — refused."""
    rows = healthy_rows()
    base_tags = healthy_rows()[0]["record"]["tags"]
    rows.append(
        row(
            "p-g4-evil",
            "duplicate",
            healthy_rows()[0]["record"],
            rec(
                "Фильтр насоса",
                BASE_BODY.replace(" чистки", " чистки!"),
                tags=tuple(base_tags),
            ),
            0.97,
            "P-para-light",
        )
    )
    ok, report = run_corner_qa(rows, LOOSE)
    assert not ok
    assert any("G4 connector-edit positives=1" in v for v in report["violations"])


def test_g4_exempts_corner_positives() -> None:
    """Identity/cosmetic positives have zero edit mass BY DESIGN — the mass
    rule applies only to non-corner positives (char5_jaccard < 1.0)."""
    ok, _ = run_corner_qa(healthy_rows(), LOOSE)
    assert ok


def test_type_lang_presence_floors() -> None:
    rows = healthy_rows()
    for r in rows:
        for side in (r["record"], r["candidate"]):
            side["record_type"] = None
            side["language"] = None
    counters = corner_qa_counters(rows)
    assert counters["type_present_fraction"] == 0.0
    assert counters["lang_present_fraction"] == 0.0
    assert any(
        "type_present_fraction=0.0" in v for v in corner_qa_violations(counters, LOOSE)
    )
    assert any(
        "lang_present_fraction=0.0" in v for v in corner_qa_violations(counters, LOOSE)
    )


def test_cos_stratification_floors() -> None:
    rows = healthy_rows()
    for r in rows:
        r["similarity"] = max(r["similarity"], 0.9)
    counters = corner_qa_counters(rows)
    assert counters["cos_below_055"] == 0
    violations = corner_qa_violations(counters, LOOSE)
    assert any("pairs_below_cos_0.55=0" in v for v in violations)
    assert any("pairs_below_cos_0.84" in v for v in violations)


def test_counters_are_deterministic() -> None:
    a = corner_qa_counters(healthy_rows())
    b = corner_qa_counters(list(reversed(healthy_rows())))
    assert a["corner_dup_positives"] == b["corner_dup_positives"]
    assert a["identity_class_positives"] == b["identity_class_positives"]
    assert a["g4_violations"] == b["g4_violations"]
    assert a["by_stratum"] == b["by_stratum"]


def test_default_thresholds_match_the_prereg_draft() -> None:
    """The shipped defaults are the pre-train lines pinned in the corpus
    prereg draft — a silent change here is a silent gate change."""
    t = DEFAULT_THRESHOLDS
    assert t.max_clone_negatives == 0
    assert t.min_corner_dup_positives == 10
    assert t.min_identity_class_positives == 60
    assert t.min_metadata_negatives == 40
    assert t.min_fact_edit_negatives == 75
    assert t.max_razor_zone_positives == 0
    assert t.min_type_present_fraction == 0.85
    assert t.min_lang_present_fraction == 0.85
    assert t.min_pairs_below_cos_055 == 5
    assert t.min_pairs_below_cos_070 == 60
    assert t.min_pairs_below_cos_084 == 180
    assert t.min_positive_edit_mass_chars == 8


def test_verify_dataset_qa_mode_green_and_red(tmp_path: Path) -> None:
    import json as _json

    corpus = tmp_path / "ds"
    corpus.mkdir()

    def write(name: str, rows: list[dict]) -> None:
        (corpus / name).write_text(
            "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        _json.dumps(
            {
                "min_corner_dup_positives": 2,
                "min_identity_class_positives": 2,
                "min_metadata_negatives": 1,
                "min_fact_edit_negatives": 2,
                "min_type_mismatch_fraction": 0.0,
                "min_lang_mismatch_fraction": 0.0,
                "min_pairs_below_cos_055": 1,
                "min_pairs_below_cos_070": 1,
                "min_pairs_below_cos_084": 2,
            }
        ),
        encoding="utf-8",
    )
    script = str(REPO_ROOT / "scripts" / "verify_dataset.py")

    write("train.jsonl", healthy_rows())
    out = tmp_path / "qa.json"
    proc = subprocess.run(
        [sys.executable, script, "qa", str(corpus), str(out), str(thresholds)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"ok": true' in out.read_text(encoding="utf-8")

    # inject a clone negative → refusal, exit 1
    rows = healthy_rows()
    rows.append(
        row(
            "clone-evil",
            "not-duplicate",
            healthy_rows()[0]["record"],
            healthy_rows()[0]["candidate"],
            1.0,
        )
    )
    write("train.jsonl", rows)
    proc = subprocess.run(
        [
            sys.executable,
            script,
            "qa",
            str(corpus),
            str(tmp_path / "qa2.json"),
            str(thresholds),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "clone_negatives=1" in proc.stderr


# ── gate-contract loader (eval-methodology §6: thresholds live in the file) ──


def test_loader_reads_the_repo_contract_and_matches_frozen_fallback() -> None:
    """The coherence pin: the committed corner_qa section of
    gate_contract.json and the frozen code constants must be EQUAL — a
    threshold change that touches only one side fails here."""
    thresholds, source = thresholds_from_gate_contract()
    assert "gate_contract" in source
    assert "fallback" not in source
    assert thresholds == DEFAULT_THRESHOLDS


def test_loader_falls_back_when_file_is_missing(tmp_path) -> None:
    thresholds, source = thresholds_from_gate_contract(tmp_path / "nope.json")
    assert thresholds == DEFAULT_THRESHOLDS
    assert "fallback" in source


def test_loader_falls_back_when_section_is_absent(tmp_path) -> None:
    contract = tmp_path / "gate_contract.json"
    contract.write_text('{"schema_version": 1}\n', encoding="utf-8")
    thresholds, source = thresholds_from_gate_contract(contract)
    assert thresholds == DEFAULT_THRESHOLDS
    assert "no 'corner_qa' section" in source


def test_loader_applies_contract_overrides(tmp_path) -> None:
    contract = tmp_path / "gate_contract.json"
    section = {
        "min_corner_dup_positives": 12,
        "min_pairs_below_cos_055": 7,
    }
    contract.write_text(
        json.dumps({"schema_version": 1, "corner_qa": section}), encoding="utf-8"
    )
    thresholds, source = thresholds_from_gate_contract(contract)
    assert "gate_contract" in source
    assert thresholds.min_corner_dup_positives == 12
    assert thresholds.min_pairs_below_cos_055 == 7
    # untouched fields keep the frozen values
    assert thresholds.max_clone_negatives == DEFAULT_THRESHOLDS.max_clone_negatives
    # types are coerced per the dataclass annotation
    assert isinstance(thresholds.min_corner_dup_positives, int)
    assert isinstance(thresholds.min_pairs_below_cos_055, int)


def test_loader_warns_on_unknown_keys(tmp_path, capsys) -> None:
    contract = tmp_path / "gate_contract.json"
    contract.write_text(
        json.dumps({"corner_qa": {"min_corner_dup_positives": 10, "mystery": 1}}),
        encoding="utf-8",
    )
    thresholds, _ = thresholds_from_gate_contract(contract)
    assert thresholds == DEFAULT_THRESHOLDS
    assert "unknown" in capsys.readouterr().err
