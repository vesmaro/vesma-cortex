#!/usr/bin/env python3
"""dataset-v4.3 translated classes — strategies + validator (addendum 7).

Addendum 7 (docs/specs/embed-round4-prereg-ADDENDUM-7.md, ratified
2026-10-06): the cortex feature contract carries ``lang_match`` as an
ANTI-duplicate signal (1.0 only when both non-empty language labels
match), so a cross-language duplicate is uncatchable today: the embedder
is blind to translations (translation-twins sens 0.00), cortex votes
anti, policy v1.1 §8.1 keeps language outside the envelope whitelist.
The cure is corpus-level: two new mandatory classes ≥ 150 pairs each.

    translated-dup     (record, translation of it)      label=duplicate
                       is-dup = 1, g-T1 sensitivity ≥ 0.90.
                       The number-preservation rule: dates, sums,
                       versions in the CONTENT survive byte-for-byte —
                       every digit token of the original side must be
                       present in the translated side (validated; the
                       addendum calls a missing digit-token a labeling
                       DEFECT of the pair, and the corpus keeps such
                       pairs out).
    translated-sibling (record, NEW statement on the same topic with a
                       DIFFERENT meaning)               label=not-duplicate
                       is-dup = 0, g-T3 specificity ≥ 0.90. Digit
                       tokens are NOT required to match; the semantic
                       difference is written into a justification field.

Both directions RU→EN and EN→RU are built; every row carries the
``stratum`` field (``translated-dup`` /
``translated-sibling``) as the class marker for the pre-registered
quotas, plus ``batch`` with the same value (the brief's field-marker
name — batch and stratum carry the class, either projection works).

The strategies are PURE data-builders over authored batch rows: no RNG,
no store lines, no network — the same determinism contract as
gen_dataset_v4 (a double build is byte-identical). The embedder pass,
split and sealing stay in the v4/v4.3 assembly flow (frozen v4.2 code
is NOT modified).

g-T2 (no-harm on non-translated strata) needs no code here: it is an
eval-holdout delta against B2-v42, not a corpus property.
"""

from __future__ import annotations

import difflib
import re

from cortex.data.fingerprints import pair_sha256

__all__ = [
    "STRATUM_TRANSLATED_DUP",
    "STRATUM_TRANSLATED_SIBLING",
    "TRANSLATED_DUP_QUOTA",
    "TRANSLATED_SIBLING_QUOTA",
    "TRANSLATION_DIRECTIONS",
    "DigitTokenContractError",
    "SiblingJustificationError",
    "build_translated_rows",
    "digit_tokens",
    "missing_digit_tokens",
    "quota_violations_v43",
    "translated_pair_sha256",
]

#: Addendum 7 class markers (the ``stratum`` value in corpus rows).
STRATUM_TRANSLATED_DUP: str = "translated-dup"
STRATUM_TRANSLATED_SIBLING: str = "translated-sibling"

#: Pre-registered quotas (addendum 7 §A.7.1, PROPOSED numbers): ≥ 150
#: pairs per class. Both directions RU→EN and EN→RU count toward the
#: class quota; 75+75 splits both quotas evenly by direction.
TRANSLATED_DUP_QUOTA: int = 150
TRANSLATED_SIBLING_QUOTA: int = 150

#: Directions both classes must cover (addendum: «направление RU→EN и
#: EN→RU обоих»). ``dir`` is the translation direction of the pair:
#: the RECORD side is the source language, the CANDIDATE side the target.
TRANSLATION_DIRECTIONS: tuple[str, ...] = ("ru->en", "en->ru")

#: Any digit run (ASCII or full-width) — the number-preservation surface:
#: dates (2026-10-06, 5 октября), sums (12 500), versions (v4.3, 0.2.1),
#: ports (8443).
_DIGIT_RUN: re.Pattern[str] = re.compile(r"\d+")


def digit_tokens(text: str) -> list[str]:
    """All digit runs of a text, in order (the number-preservation surface).

    A byte-for-byte preservation rule over RUNS: ``12 500`` carries two
    tokens (``12``, ``500``); every token of the original side must be
    a standalone digit run in the translated side (see
    :func:`missing_digit_tokens` for the adjacency rule).
    """
    return _DIGIT_RUN.findall(text)


def missing_digit_tokens(original: str, translated: str) -> list[str]:
    """Digit tokens of ``original`` absent from ``translated`` (empty = ok).

    Byte-for-byte ADDENDUM-7 rule: every digit run of the original must
    occur in the translation AS A RUN — standalone, with no adjacent
    digit (a bare ``5`` is NOT satisfied by the ``5`` inside ``500``;
    ``192.168`` keeps both runs since the dot breaks adjacency). A
    translation may introduce its own numbers (month names, a duplicated
    run — fine); only LOST runs are reported.
    """
    lost: list[str] = []
    for tok in digit_tokens(original):
        pattern = re.compile(rf"(?<!\d){re.escape(tok)}(?!\d)")
        if not pattern.search(translated):
            lost.append(tok)
    return lost


class DigitTokenContractError(ValueError):
    """A translated-dup pair lost a digit token — a labeling defect per
    addendum 7 §A.7.1 («число в переводе отсутствует = разметочный
    дефект пары, отбор корпуса это отсекает»)."""


class SiblingJustificationError(ValueError):
    """A translated-sibling pair without a non-empty justification of the
    semantic difference — the class definition demands it («содержания
    различаются по смыслу... семантическая разница описана в поле-
    обосновании»)."""


# ── pair construction (pure, deterministic) ─────────────────────────────────


def _class_marker_row(
    cls: str,
    direction: str,
    record: dict,
    candidate: dict,
    *,
    label: str,
    pair_id: str,
    justification: str | None = None,
) -> dict:
    """One corpus row of a v4.3 translated class.

    Shape mirrors the stage-2 pair (``record``/``candidate`` sides, stage2
    feature surface fields only) with the class fields on top: ``stratum``
    AND ``batch`` both carry the class marker (either projection works for
    the quota check), ``direction`` pins which side is source, and for the
    sibling class ``justification`` holds the semantic-difference note.
    Labels follow addendum 7: dup=1 ↔ ``duplicate``, sibling ↔
    ``not-duplicate`` (the stage-2 label vocabulary).
    """
    if label not in ("duplicate", "not-duplicate"):
        raise ValueError(f"bad label {label!r}")
    row = {
        "pair_id": pair_id,
        "label": label,
        "stratum": cls,
        "batch": cls,
        "direction": direction,
        "record": {
            "title": record["title"],
            "body": record["body"],
            "tags": list(record["tags"]),
            "language": record["language"],
            "record_type": record.get("record_type"),
        },
        "candidate": {
            "title": candidate["title"],
            "body": candidate["body"],
            "tags": list(candidate["tags"]),
            "language": candidate["language"],
            "record_type": candidate.get("record_type"),
        },
    }
    if justification is not None:
        row["justification"] = justification
    return row


def _side_lang(side: dict) -> str:
    lang = side.get("language") or side.get("lang")
    if lang not in ("ru", "en"):
        raise ValueError(f"side language must be ru|en, got {lang!r}")
    return str(lang)


def _direction_of(record: dict, candidate: dict) -> str:
    return f"{_side_lang(record)}->{_side_lang(candidate)}"


def build_translated_rows(
    dup_specs: list[dict],
    sibling_specs: list[dict],
) -> list[dict]:
    """Build ALL v4.3 translated-class rows from authored spec batches.

    Spec shapes (authored batch files, deterministic scans, no RNG):

    - ``dup_specs`` → translated-dup. Each spec: ``{"key", "record"`,
      ``"translation"}`` — ``record`` is the original side, ``translation``
      a full translation of it as an English-or-Russian statement; sides
      MUST differ in language; every digit token of the record side must
      be present in the translation side (byte-for-byte rule; also
      validated on the TITLE when the translation carries its own title).
      The generator builds BOTH orientations from one spec (record↔
      translation and translation↔record) with the SAME ``direction``
      (ru->en or en->ru — the direction of the translation itself), because
      g-T1 measures the class in both orderings.
    - ``sibling_specs`` → translated-sibling. Each spec:
      ``{"key", "record", "sibling", "justification"}`` — ``sibling`` is a
      NEW statement on the same topic with a DIFFERENT meaning; the
      justification field must be non-empty (class contract) and MUST
      NOT be a digit-preservation claim (numbers are allowed to differ);
      language of the sibling side must differ from the record side
      (fresh cross-language formulation — addendum: свежая английская/
      русская формулировка). Both orientations, same as dup.

    pair_ids: ``V43-TDUP-NNNN`` / ``V43-TSIB-NNNN`` — deterministic,
    construction-order based (spec order + orientation), stable across
    double builds like the v4 ids.

    Raises:
        DigitTokenContractError: a dup spec lost a digit token.
        SiblingJustificationError: empty/absent sibling justification.
    """
    rows: list[dict] = []
    counters = {STRATUM_TRANSLATED_DUP: 0, STRATUM_TRANSLATED_SIBLING: 0}

    for spec in dup_specs:
        record = _validate_side(spec.get("record"), "dup.record", spec)
        translation = _validate_side(spec.get("translation"), "dup.translation", spec)
        r_lang, t_lang = _side_lang(record), _side_lang(translation)
        if r_lang == t_lang:
            raise DigitTokenContractError(
                f"dup spec {spec.get('key')!r}: translation language equals "
                f"record language ({r_lang!r}) — not a translation"
            )
        direction = _direction_of(record, translation)
        if direction not in TRANSLATION_DIRECTIONS:  # pragma: no cover
            raise DigitTokenContractError(f"bad direction {direction}")
        missing = missing_digit_tokens(
            f"{record['title']}\n{record['body']}",
            f"{translation['title']}\n{translation['body']}",
        )
        if missing:
            raise DigitTokenContractError(
                f"dup spec {spec.get('key')!r}: digit tokens {missing!r} of the "
                "original are absent from the translation (addendum-7 "
                "number-preservation rule — fix the authored pair)"
            )
        for orientation in (0, 1):
            counters[STRATUM_TRANSLATED_DUP] += 1
            a, b = (record, translation) if orientation == 0 else (translation, record)
            rows.append(
                _class_marker_row(
                    STRATUM_TRANSLATED_DUP,
                    direction,
                    a,
                    b,
                    label="duplicate",
                    pair_id=(f"V43-TDUP-{counters[STRATUM_TRANSLATED_DUP]:04d}"),
                )
            )

    for spec in sibling_specs:
        record = _validate_side(spec.get("record"), "sibling.record", spec)
        sibling = _validate_side(spec.get("sibling"), "sibling.sibling", spec)
        justification = spec.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            raise SiblingJustificationError(
                f"sibling spec {spec.get('key')!r}: justification of the "
                "semantic difference is required and must be non-empty"
            )
        r_lang, s_lang = _side_lang(record), _side_lang(sibling)
        if r_lang == s_lang:
            raise SiblingJustificationError(
                f"sibling spec {spec.get('key')!r}: sibling language equals "
                f"record language ({r_lang!r}) — the class is a FRESH "
                "cross-language formulation on the same topic"
            )
        direction = _direction_of(record, sibling)
        for orientation in (0, 1):
            counters[STRATUM_TRANSLATED_SIBLING] += 1
            a, b = (record, sibling) if orientation == 0 else (sibling, record)
            rows.append(
                _class_marker_row(
                    STRATUM_TRANSLATED_SIBLING,
                    direction,
                    a,
                    b,
                    label="not-duplicate",
                    pair_id=(f"V43-TSIB-{counters[STRATUM_TRANSLATED_SIBLING]:04d}"),
                    justification=justification,
                )
            )
    return rows


def _validate_side(side, what: str, spec: dict) -> dict:
    """Structural check of one side (title/body/tags/language present)."""
    if not isinstance(side, dict):
        raise ValueError(f"{what} in spec {spec.get('key')!r}: must be an object")
    for field in ("title", "body"):
        value = side.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{what} in spec {spec.get('key')!r}: empty {field}")
    if not isinstance(side.get("tags"), list) or not side["tags"]:
        raise ValueError(f"{what} in spec {spec.get('key')!r}: tags non-empty list")
    return side


# ── quota gate (mirrors the watchlist refusal discipline) ───────────────────


def quota_violations_v43(rows: list[dict]) -> list[str]:
    """Refusal lines for the addendum-7 quotas over built corpus rows.

    Same shape as ``cortex.data.watchlist.quota_violations``: an empty
    list means the build may proceed; otherwise each line names the
    deficit. A class counts by the ``stratum`` marker (``translated-dup`` /
    ``translated-sibling``); every direction of
    :data:`TRANSLATION_DIRECTIONS` must appear in the class too — the
    quota alone is not sufficient (addendum: «направление RU→EN и
    EN→RU обоих»).
    """
    violations: list[str] = []
    for stratum, quota in (
        (STRATUM_TRANSLATED_DUP, TRANSLATED_DUP_QUOTA),
        (STRATUM_TRANSLATED_SIBLING, TRANSLATED_SIBLING_QUOTA),
    ):
        members = [r for r in rows if r.get("stratum") == stratum]
        if len(members) < quota:
            violations.append(
                f"v43 quota: {len(members)} {stratum} pairs < {quota} "
                "(prereg addendum 7 §A.7.1)"
            )
            continue
        directions = {r.get("direction") for r in members}
        for needed in TRANSLATION_DIRECTIONS:
            if needed not in directions:
                violations.append(
                    f"v43 quota: {stratum} lacks the {needed} direction "
                    f"(present: {sorted(d for d in directions if d)})"
                )
    return violations


def translated_pair_sha256(row: dict) -> str:
    """Canonical pair digest for a v4.3 row — the corpus scheme
    (:func:`cortex.data.fingerprints.pair_sha256` over record/candidate/
    similarity) EXTENDED with the class surface (stratum, direction,
    justification): the class fields are content of the v4.3 corpus pair
    (they drive g-T1/T3 stratified reads), so they ride the digest."""
    payload = {
        "record": row["record"],
        "candidate": row["candidate"],
        "similarity": row.get("similarity", 0.0),
        "stratum": row["stratum"],
        "direction": row.get("direction"),
        **({"justification": row["justification"]} if "justification" in row else {}),
    }
    return pair_sha256(payload)


# ── self-check helpers (used by tests and the assembler's neighbor pass) ────


def body_edit_mass_chars(a: str, b: str) -> int:
    """difflib edit-mass between two bodies (the corner_qa measure) —
    exposed here so the sibling class can assert it is NOT in the
    near-identity family (a sibling is a DIFFERENT statement)."""
    return sum(
        max(len(a[i1:i2]), len(b[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
        if tag != "equal"
    )
