"""Corner-QA — pre-train dataset gates (P0 §8.1 + labeling-policy-b2 §5).

The #480 lesson: holdout metrics are BLIND to corpus defects — the B2
inversion (self-pair P(dup)=0.0108) passed a healthy-looking holdout. These
gates judge the CORPUS before any training run:

- clone negatives are forbidden with zero tolerance (P0 §8.1: 26–29 such
  rows in v3-era corpora taught the model "almost identical = not a
  duplicate", which extrapolated onto self-pairs);
- the exact corner must be OCCUPIED by duplicates (T0 identity + T1
  cosmetic positives) — the policy's cure for the inversion;
- T2 metadata / T3 fact-edit negatives are REQUIRED classes (policy §4
  quotas), while the open razor zone (policy §5-G3) must stay
  positive-free — a positive there is generation noise by §3 finding 2;
- no contract feature may be constant (B1 shipped record_type=None on
  100 % of rows — inert features, zero importance);
- measured cosine must stratify DOWN to 0.4–0.55 (P0 §8.1: B2 lived in
  [0.836; 1.0], cos_target was untrainable);
- G1 feature ceiling (policy §5): for monotone-duplicative features
  ``max(negatives) <= max(positives)`` — a negative above the positive
  ceiling is the inversion, caught before training.

Pure stdlib + ``cortex.features.pair`` math: no embedder, no store, no
network (src/ is network-free by the AST tripwire). ``similarity`` rides
the row (measured upstream, data-contract §3) — this module never
re-measures it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Final

from cortex.features.pair import FEATURE_NAMES, RecordLike, features

__all__ = [
    "COS_BANDS",
    "CornerQAThresholds",
    "DEFAULT_THRESHOLDS",
    "G1_CEILING_FEATURES",
    "corners_of_pair",
    "corner_qa_counters",
    "corner_qa_violations",
    "run_corner_qa",
]

#: G1 ceiling family (policy §5-G1): monotone-duplicative features — a
#: NEGATIVE may never sit above the positive maximum on any of these.
G1_CEILING_FEATURES: Final[tuple[str, ...]] = (
    "cos_target",
    "char3_jaccard",
    "char4_jaccard",
    "char5_jaccard",
    "char3_containment",
    "char4_containment",
    "char5_containment",
)

#: Razor zone (policy §5-G3, constants fixed by the policy, not code whim):
#: an open zone under the identical-text point that T3 negatives legitimately
#: occupy and POSITIVES must never enter (unlearnable char-signature noise).
RAZOR_ZONE: Final[dict[str, float]] = {
    "char5_jaccard_min": 0.977,
    "char5_containment_min": 0.995,
    "cos_min": 0.995,
}

#: Cosine stratification bands for the manifest histogram (P0 §8.1: the
#: corpus must reach down to 0.4–0.55; B2 lived in [0.836; 1.0]).
COS_BANDS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 0.40),
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 0.95),
    (0.95, 1.0),
    (1.0, 1.0 + 1e-9),
)

TOL: Final[float] = 1e-9


@dataclass(frozen=True)
class CornerQAThresholds:
    """Frozen gate thresholds — the pre-train refusal lines.

    Draft values live here and in the corpus prereg draft
    (docs/experiments/calibration-b2p-prereg-draft.md); ratification rides
    wave 3 (TL, commit-order rule). Foundations: P0 §8.1 quotas, labeling
    policy §4 generator targets and §5 corner gates.
    """

    #: P0 §8.1: clone negatives — zero tolerance.
    max_clone_negatives: int = 0
    #: P0 §8.1 corner quota: ≥ 10 duplicate rows with char4_jaccard > 0.99
    #: and body_len_delta == 0.
    min_corner_dup_positives: int = 10
    #: Policy §4 G-identity target: ≥ 60 pairs (≥ 10 % of positives) at the
    #: identical-text point (T0 identity + T1 cosmetic).
    min_identity_class_positives: int = 60
    #: Policy §4 G-metadata target: ≥ 40 metadata-only negatives.
    min_metadata_negatives: int = 40
    #: Policy §4 G-fact-edit target: v3's 50 + ≥ 25 new razor negatives.
    min_fact_edit_negatives: int = 75
    #: Policy §5-G3: zero positives in the open razor zone.
    max_razor_zone_positives: int = 0
    #: P0 §8.1 variability: record_type / language present (non-null) on
    #: ≥ 85 % of record sides (B1: record_type was None everywhere).
    min_type_present_fraction: float = 0.85
    min_lang_present_fraction: float = 0.85
    #: Type/lang mismatch pairs must exist with real mass (owner codebook
    #: classes, not ghost features): ≥ 3 % / ≥ 5 % of pairs.
    min_type_mismatch_fraction: float = 0.03
    min_lang_mismatch_fraction: float = 0.05
    #: P0 §8.1: cos stratification down to 0.4–0.5, calibrated to the frozen
    #: vesma-embed-v1 geometry: fully unrelated topics measure ≈ 0.51+, so the
    #: gate demands (a) the corpus REACHES the 0.4–0.55 band, (b) real mass
    #: below 0.70, (c) mass below the B2 floor of 0.836 (B2 lived in
    #: [0.836; 1.0] — everything below that floor is new trainable signal).
    min_pairs_below_cos_055: int = 5
    min_pairs_below_cos_070: int = 60
    min_pairs_below_cos_084: int = 180
    #: Policy §5-G4: every NON-corner positive carries ≥ 8 chars of edit
    #: mass on the body, or an edited title/tags (connector micro-edits are
    #: the v3 noise class). Corner positives (char5_jaccard == 1.0) are
    #: exempt — they are gated by the identity counters instead.
    min_positive_edit_mass_chars: int = 8


DEFAULT_THRESHOLDS: Final[CornerQAThresholds] = CornerQAThresholds()


def _side(record: dict[str, Any]) -> RecordLike:
    return RecordLike(
        title=str(record.get("title") or ""),
        body=str(record.get("body") or ""),
        tags=tuple(record.get("tags") or ()),
        language=record.get("language"),
        record_type=record.get("record_type"),
    )


def _feature_map(row: dict[str, Any]) -> dict[str, float]:
    """The frozen 13-feature vector as a name→value map (no reorder)."""
    vec = features(
        _side(row["record"]),
        _side(row["candidate"]),
        float(row["similarity"]),
    )
    return dict(zip(vec.names, vec.values))


def corners_of_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Corner-relevant derived facts for ONE pair (counted, not judged)."""
    fm = _feature_map(row)
    rec, cand = row["record"], row["candidate"]
    title_delta = abs(
        len(str(rec.get("title") or "")) - len(str(cand.get("title") or ""))
    )
    body_delta = abs(len(str(rec.get("body") or "")) - len(str(cand.get("body") or "")))
    len_deltas_zero = (
        title_delta == 0 and body_delta == 0 and fm["tag_count_delta"] == 0.0
    )
    text_identical = fm["char5_jaccard"] >= 1.0 - TOL
    metadata_differs = (
        fm["tag_jaccard"] < 1.0 - TOL
        or fm["type_match"] == 0.0
        or fm["lang_match"] == 0.0
    )
    razor_zone = (
        RAZOR_ZONE["char5_jaccard_min"] <= fm["char5_jaccard"] < 1.0 - TOL
        and fm["char5_containment"] >= RAZOR_ZONE["char5_containment_min"]
        and fm["cos_target"] >= RAZOR_ZONE["cos_min"]
        and len_deltas_zero
    )
    # difflib edit mass of the body (inserted+deleted+replaced chars) —
    # the policy §5-G4 measure of a real rewording vs connector noise.
    a = str(rec.get("body") or "")
    b = str(cand.get("body") or "")
    mass = sum(
        max(len(a[i1:i2]), len(b[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
        if tag != "equal"
    )
    title_differs = str(rec.get("title") or "") != str(cand.get("title") or "")
    tags_differ = list(rec.get("tags") or []) != list(cand.get("tags") or ())
    return {
        "char4_jaccard": fm["char4_jaccard"],
        "char5_jaccard": fm["char5_jaccard"],
        "body_len_delta_zero": body_delta == 0,
        "len_deltas_zero": len_deltas_zero,
        "text_identical": text_identical,
        "metadata_differs": metadata_differs,
        "razor_zone": razor_zone,
        "body_edit_mass_chars": mass,
        "title_differs": title_differs,
        "tags_differ": tags_differ,
        "features": fm,
    }


def corner_qa_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counters + distributions for a labeled corpus (train + holdout).

    A row is ``{"pair_id", "label", "record", "candidate", "similarity",
    "stratum"}`` — the stage2 train-manifest shape. Labels: duplicate /
    not-duplicate (disputed must be filtered upstream).
    """
    counters: dict[str, Any] = {
        "pairs": len(rows),
        "by_label": {"duplicate": 0, "not-duplicate": 0},
        "by_stratum": {},
        "clone_negatives": [],
        "corner_dup_positives": [],
        "identity_class_positives": [],
        "metadata_negatives": [],
        "fact_edit_negatives": [],
        "razor_zone_positives": [],
        "razor_zone_negatives": 0,
        "g4_violations": [],
        "lang_mismatch_pairs": 0,
        "type_mismatch_pairs": 0,
    }
    positive_max: dict[str, float] = {f: float("-inf") for f in G1_CEILING_FEATURES}
    negative_max: dict[str, float] = {f: float("-inf") for f in G1_CEILING_FEATURES}
    feature_values: dict[str, set[float]] = {f: set() for f in FEATURE_NAMES}
    type_present = 0
    lang_present = 0
    sides = 0
    cos_below_055 = 0
    cos_below_070 = 0
    cos_below_084 = 0
    cos_hist = {f"[{lo}, {hi})" if hi <= 1.0 else "[1.0]": 0 for lo, hi in COS_BANDS}

    for row in rows:
        label = row["label"]
        counters["by_label"][label] = counters["by_label"].get(label, 0) + 1
        stratum = row.get("stratum") or "unspecified"
        counters["by_stratum"][stratum] = counters["by_stratum"].get(stratum, 0) + 1

        c = corners_of_pair(row)
        fm = c["features"]
        pid = row["pair_id"]
        is_pos = label == "duplicate"

        for f in FEATURE_NAMES:
            feature_values[f].add(fm[f])
        for f in G1_CEILING_FEATURES:
            if is_pos:
                positive_max[f] = max(positive_max[f], fm[f])
            else:
                negative_max[f] = max(negative_max[f], fm[f])

        for side in (row["record"], row["candidate"]):
            sides += 1
            if side.get("record_type") is not None:
                type_present += 1
            if side.get("language") is not None:
                lang_present += 1
        if fm["lang_match"] == 0.0:
            counters["lang_mismatch_pairs"] += 1
        if fm["type_match"] == 0.0:
            counters["type_mismatch_pairs"] += 1

        cos = float(row["similarity"])
        if cos < 0.55:
            cos_below_055 += 1
        if cos < 0.70:
            cos_below_070 += 1
        if cos < 0.84:
            cos_below_084 += 1
        for lo, hi in COS_BANDS:
            if lo <= cos < hi or (hi > 1.0 and abs(cos - 1.0) <= TOL):
                key = f"[{lo}, {hi})" if hi <= 1.0 else "[1.0]"
                cos_hist[key] += 1
                break

        if not is_pos and c["text_identical"] and not c["metadata_differs"]:
            counters["clone_negatives"].append(pid)
        if is_pos and c["char4_jaccard"] > 0.99 and c["body_len_delta_zero"]:
            counters["corner_dup_positives"].append(pid)
        if is_pos and c["text_identical"] and c["len_deltas_zero"]:
            counters["identity_class_positives"].append(pid)
        if (
            not is_pos
            and c["text_identical"]
            and not c["title_differs"]
            and c["body_len_delta_zero"]
            and c["metadata_differs"]
        ):
            counters["metadata_negatives"].append(pid)
        if not is_pos and 0.85 <= c["char5_jaccard"] < 1.0 - TOL:
            counters["fact_edit_negatives"].append(pid)
        if c["razor_zone"]:
            if is_pos:
                counters["razor_zone_positives"].append(pid)
            else:
                counters["razor_zone_negatives"] += 1
        if (
            is_pos
            and not c["text_identical"]
            and c["body_edit_mass_chars"]
            < DEFAULT_THRESHOLDS.min_positive_edit_mass_chars
            and not c["title_differs"]
            and not c["tags_differ"]
        ):
            counters["g4_violations"].append(pid)

    counters["clone_negatives_count"] = len(counters["clone_negatives"])
    counters["corner_dup_positives_count"] = len(counters["corner_dup_positives"])
    counters["identity_class_positives_count"] = len(
        counters["identity_class_positives"]
    )
    counters["metadata_negatives_count"] = len(counters["metadata_negatives"])
    counters["fact_edit_negatives_count"] = len(counters["fact_edit_negatives"])
    counters["razor_zone_positives_count"] = len(counters["razor_zone_positives"])
    counters["g4_violations_count"] = len(counters["g4_violations"])
    counters["type_present_fraction"] = round(type_present / sides, 4) if sides else 0.0
    counters["lang_present_fraction"] = round(lang_present / sides, 4) if sides else 0.0
    n = len(rows)
    counters["type_mismatch_fraction"] = (
        round(counters["type_mismatch_pairs"] / n, 4) if n else 0.0
    )
    counters["lang_mismatch_fraction"] = (
        round(counters["lang_mismatch_pairs"] / n, 4) if n else 0.0
    )
    counters["cos_below_055"] = cos_below_055
    counters["cos_below_070"] = cos_below_070
    counters["cos_below_084"] = cos_below_084
    counters["cos_histogram"] = cos_hist
    counters["feature_n_distinct"] = {f: len(feature_values[f]) for f in FEATURE_NAMES}
    counters["constant_features"] = [
        f for f in FEATURE_NAMES if len(feature_values[f]) < 2
    ]
    counters["g1_ceiling"] = {
        f: {
            "max_positive": round(positive_max[f], 6),
            "max_negative": round(negative_max[f], 6),
        }
        for f in G1_CEILING_FEATURES
        if positive_max[f] > float("-inf") and negative_max[f] > float("-inf")
    }
    # deterministic id lists (sorted) — the report is a committed artifact
    for key in (
        "clone_negatives",
        "corner_dup_positives",
        "identity_class_positives",
        "metadata_negatives",
        "fact_edit_negatives",
        "razor_zone_positives",
        "g4_violations",
    ):
        counters[key] = sorted(counters[key])
    return counters


def corner_qa_violations(
    counters: dict[str, Any],
    thresholds: CornerQAThresholds = DEFAULT_THRESHOLDS,
) -> list[str]:
    """Refusal lines for a counter report — empty list means the corpus may
    be exported and fingerprinted (policy §5 sanction order)."""
    bad: list[str] = []
    if counters["clone_negatives_count"] > thresholds.max_clone_negatives:
        bad.append(
            f"clone_negatives={counters['clone_negatives_count']} > "
            f"{thresholds.max_clone_negatives} (zero tolerance, P0 §8.1): "
            f"{counters['clone_negatives'][:5]}"
        )
    if counters["corner_dup_positives_count"] < thresholds.min_corner_dup_positives:
        bad.append(
            f"corner_dup_positives={counters['corner_dup_positives_count']} < "
            f"{thresholds.min_corner_dup_positives} (P0 §8.1 corner quota)"
        )
    if (
        counters["identity_class_positives_count"]
        < thresholds.min_identity_class_positives
    ):
        bad.append(
            f"identity_class_positives={counters['identity_class_positives_count']} < "
            f"{thresholds.min_identity_class_positives} (policy §4 G-identity)"
        )
    if counters["metadata_negatives_count"] < thresholds.min_metadata_negatives:
        bad.append(
            f"metadata_negatives={counters['metadata_negatives_count']} < "
            f"{thresholds.min_metadata_negatives} (policy §4 G-metadata)"
        )
    if counters["fact_edit_negatives_count"] < thresholds.min_fact_edit_negatives:
        bad.append(
            f"fact_edit_negatives={counters['fact_edit_negatives_count']} < "
            f"{thresholds.min_fact_edit_negatives} (policy §4 G-fact-edit)"
        )
    if counters["razor_zone_positives_count"] > thresholds.max_razor_zone_positives:
        bad.append(
            f"razor_zone_positives={counters['razor_zone_positives_count']} > 0 "
            f"(policy §5-G3: {counters['razor_zone_positives'][:5]})"
        )
    if counters["type_present_fraction"] < thresholds.min_type_present_fraction:
        bad.append(
            f"type_present_fraction={counters['type_present_fraction']} < "
            f"{thresholds.min_type_present_fraction} (P0 §8.1: B1 had record_type=None everywhere)"
        )
    if counters["lang_present_fraction"] < thresholds.min_lang_present_fraction:
        bad.append(
            f"lang_present_fraction={counters['lang_present_fraction']} < "
            f"{thresholds.min_lang_present_fraction}"
        )
    if counters["type_mismatch_fraction"] < thresholds.min_type_mismatch_fraction:
        bad.append(
            f"type_mismatch_fraction={counters['type_mismatch_fraction']} < "
            f"{thresholds.min_type_mismatch_fraction}"
        )
    if counters["lang_mismatch_fraction"] < thresholds.min_lang_mismatch_fraction:
        bad.append(
            f"lang_mismatch_fraction={counters['lang_mismatch_fraction']} < "
            f"{thresholds.min_lang_mismatch_fraction}"
        )
    if counters["cos_below_055"] < thresholds.min_pairs_below_cos_055:
        bad.append(
            f"pairs_below_cos_0.55={counters['cos_below_055']} < "
            f"{thresholds.min_pairs_below_cos_055} (P0 §8.1 stratification)"
        )
    if counters["cos_below_070"] < thresholds.min_pairs_below_cos_070:
        bad.append(
            f"pairs_below_cos_0.70={counters['cos_below_070']} < "
            f"{thresholds.min_pairs_below_cos_070}"
        )
    if counters["cos_below_084"] < thresholds.min_pairs_below_cos_084:
        bad.append(
            f"pairs_below_cos_0.84={counters['cos_below_084']} < "
            f"{thresholds.min_pairs_below_cos_084} (B2 floor: new trainable signal)"
        )
    if counters["constant_features"]:
        bad.append(
            f"constant contract features={counters['constant_features']} "
            "(no feature may be inert — P0 §8.1)"
        )
    for f, m in counters["g1_ceiling"].items():
        if m["max_negative"] > m["max_positive"] + TOL:
            bad.append(
                f"G1 ceiling violated for {f}: max_negative "
                f"{m['max_negative']} > max_positive {m['max_positive']} (policy §5-G1)"
            )
    if counters["g4_violations_count"] > 0:
        bad.append(
            f"G4 connector-edit positives={counters['g4_violations_count']} "
            f"(policy §5-G4): {counters['g4_violations'][:5]}"
        )
    return bad


def run_corner_qa(
    rows: list[dict[str, Any]],
    thresholds: CornerQAThresholds = DEFAULT_THRESHOLDS,
) -> tuple[bool, dict[str, Any]]:
    """Counters + verdict in one call: (ok, report with 'violations')."""
    counters = corner_qa_counters(rows)
    violations = corner_qa_violations(counters, thresholds)
    return (not violations), {**counters, "violations": violations}
