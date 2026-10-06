"""Near-miss watchlist — the registry and the corpus quota mechanism.

The near-0009 lesson (Layer A on release-v2, calibration B2-prime
findings, docs/experiments/calibration-b2p.md «Layer A на замороженных
наборах»): a near-identity-twin probe (cos 0.99, char5_jaccard 0.9018,
tag_jaccard 1.0, zero len-deltas) scored P(dup)=0.0100 on B2-v42 against
its duplicate label (floor ≥ 0.5; 17/18 of the class passed; B1 was
18/18). Read as a C1-rebalance side effect: same-length T3 fact-edit
negatives sharpened the high-overlap boundary into the near-identity
cloud. The corridor to B1 held (CI95 for n=18) — ratified criteria were
not breached, so the probe is a WATCHED case, not a gate: the answer is
exposure in the NEXT corpus revision, not a threshold change (no
post-hoc threshold edits — eval-methodology §6).

What lives here:

- the SELECTION CRITERION as code: an eval-set probe whose observed
  geometry matches ``NEAR_MISS_CRITERION`` — a duplicate-labeled
  near-identity twin on the razor band's positive side (cos ≥ 0.95,
  char5 containment ≥ 0.90, zero len-deltas, tag/type/lang matching)
  that scored BELOW the duplicate cut — plus the measured facts the
  finding carried (feature values, P(dup), the scoring bundle);
- the REGISTRY: ``datasets/watchlist/near-miss.jsonl`` — one JSONL row
  per near-miss (append-only; a case leaves the registry only through
  ``status: resolved``, never by deletion);
- the QUOTA CONTRACT: a next-corpus build plan (dataset-v4.3+) must
  inject ``WATCHLIST_QUOTA`` near-0009-family pairs into the razor
  band's duplicate side — the same geometry family (see
  :func:`quota_violations`) — verified BEFORE that corpus is sealed
  (mirrors the corner-QA refusal discipline, policy §5 sanction order).

Geometry reuses the frozen feature pass (:func:`cortex.features.pair.features`,
the #480 single-source rule) plus the same difflib edit-mass measure as
``cortex.data.corner_qa``. Threshold constants are FROZEN here (the
registry rows carry the evidence; the criteria must not drift under it);
no gate_contract section is added in this wave — a contract revision is
the separate-car procedure the watchlist explicitly does NOT rush.

Pure stdlib: no embedder, no store, no network (src/ is network-free by
the AST tripwire). Registry rows carry probe-side content summaries only
(synthetic eval-surface texts, ADR 0003 R2 — no store lines).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cortex.features.pair import PairRecord, features

__all__ = [
    "MIN_FAMILY_EDIT_MASS_CHARS",
    "MAX_FAMILY_EDIT_MASS_CHARS",
    "MIN_CORPUS_VERSION_CARRYING_QUOTA",
    "NEAR_MISS_CLASS",
    "NEAR_MISS_ID",
    "REGISTRY_FILENAME",
    "REGISTRY_SCHEMA_VERSION",
    "WATCHLIST_DIRNAME",
    "WATCHLIST_QUOTA",
    "NearMissCriterion",
    "NEAR_MISS_CRITERION",
    "default_registry_path",
    "load_registry",
    "near_miss_evaluations",
    "quota_violations",
    "registry_row",
]

# ── the case (calibration-b2p.md Layer A finding 2 — the evidence row) ────────

#: The probe that opened the watchlist (release-v2 near-identity class).
NEAR_MISS_ID: Final[str] = "near-0009"

#: Its probe class in the Layer A taxonomy.
NEAR_MISS_CLASS: Final[str] = "near-identity-twin"


# ── selection criterion (frozen — the zone geometry + the red score) ─────────


@dataclass(frozen=True)
class NearMissCriterion:
    """Near-miss admission criterion (the near-0009 shape, frozen).

    An eval-set probe enters the watchlist when ALL hold:

    - its class carries a DUPLICATE label (the floor classes);
    - its assigned cosine sits on the razor band's positive side
      (cos ≥ min_cos — the zone (0.95; 1.0) is two-valued by policy
      v1.1, this criterion watches ITS positive side);
    - the near-identity char profile holds (char5 containment ≥ the
      bound; near-0009 measured 0.9484);
    - zero length deltas and matching tag/type/lang surfaces — the
      same-length family signature;
    - it scored below the duplicate cut — the miss itself.
    """

    #: floor classes the criterion watches (label = duplicate by taxonomy).
    probe_classes: tuple[str, ...]
    min_cos: float
    min_char5_containment: float
    require_zero_len_deltas: bool
    require_matching_metadata: bool
    #: the miss: P(dup) strictly below the runner's scoring cut.
    max_miss_probability: float

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable criterion (rides the registry header row)."""
        return {
            "probe_classes": list(self.probe_classes),
            "min_cos": self.min_cos,
            "min_char5_containment": self.min_char5_containment,
            "require_zero_len_deltas": self.require_zero_len_deltas,
            "require_matching_metadata": self.require_matching_metadata,
            "max_miss_probability": self.max_miss_probability,
        }


#: The near-0009 criterion: near-identity twins (the only floor class with
#: razor-band near-identity geometry in the current eval sets) at
#: cos ≥ 0.95, char5 containment ≥ 0.90, zero len-deltas, matched
#: metadata, scored P(dup) < 0.5 (the runner cut).
NEAR_MISS_CRITERION: Final[NearMissCriterion] = NearMissCriterion(
    probe_classes=(NEAR_MISS_CLASS,),
    min_cos=0.95,
    min_char5_containment=0.90,
    require_zero_len_deltas=True,
    require_matching_metadata=True,
    max_miss_probability=0.5,
)


# ── registry constants ────────────────────────────────────────────────────────

#: The registry lives under datasets/ (content-bearing, like corpus-v4 /
#: evalsets) — no new hierarchy invented.
WATCHLIST_DIRNAME: Final[str] = "watchlist"
REGISTRY_FILENAME: Final[str] = "near-miss.jsonl"

#: Schema version stamped into every row.
REGISTRY_SCHEMA_VERSION: Final[int] = 1

#: The quota the NEXT corpus build plan (dataset-v4.3+) must carry while
#: the registry holds open cases: inject at least this many near-0009-
#: family duplicate pairs into the razor band's positive side — the same
#: C1 lever that fixed the 192:72 dominance, pointed at the near-identity
#: boundary (the zone stays under-sampled on THIS boundary otherwise).
WATCHLIST_QUOTA: Final[int] = 20

#: The watchlist-family geometry band: a corpus pair counts toward the
#: quota when its body carries a real but tiny edit — difflib mass ≥ 1
#: (near-0009 measured 2; mass 0 = the identity corner, watched by the
#: corner-DUP counters, not here) and ≤ this ceiling (the pairs must live
#: ON the high-overlap boundary, not merely somewhere in the band).
MAX_FAMILY_EDIT_MASS_CHARS: Final[int] = 4
MIN_FAMILY_EDIT_MASS_CHARS: Final[int] = 1

#: First corpus revision the quota binds (the card's fixpoint: v4.2 is
#: sealed history; the quota rides the NEXT build plan, v4.3+).
MIN_CORPUS_VERSION_CARRYING_QUOTA: Final[tuple[int, int]] = (4, 3)


# ── geometry (one feature pass — the #480 single-source rule) ────────────────


def _side(record: dict[str, Any]) -> PairRecord:
    return PairRecord(
        title=str(record.get("title") or ""),
        body=str(record.get("body") or ""),
        tags=tuple(str(tag) for tag in (record.get("tags") or ())),
        language=record.get("language"),
        record_type=record.get("record_type"),
    )


def _geometry(row: dict[str, Any]) -> dict[str, Any]:
    """The frozen-13 feature map + boundary facts of ONE pair row
    (``{record, candidate, similarity}`` — the stage-2 manifest shape).
    The difflib edit-mass measure mirrors ``cortex.data.corner_qa``."""
    vec = features(
        _side(row["record"]), _side(row["candidate"]), float(row["similarity"])
    )
    fm = dict(zip(vec.names, vec.values))
    title_delta = abs(
        len(str(row["record"].get("title") or ""))
        - len(str(row["candidate"].get("title") or ""))
    )
    body_delta = abs(
        len(str(row["record"].get("body") or ""))
        - len(str(row["candidate"].get("body") or ""))
    )
    a = str(row["record"].get("body") or "")
    b = str(row["candidate"].get("body") or "")
    mass = sum(
        max(len(a[i1:i2]), len(b[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
        if tag != "equal"
    )
    return {
        "features": fm,
        "len_deltas_zero": (
            title_delta == 0 and body_delta == 0 and fm["tag_count_delta"] == 0.0
        ),
        "metadata_matches": (
            fm["tag_jaccard"] == 1.0
            and fm["type_match"] == 1.0
            and fm["lang_match"] == 1.0
        ),
        "body_edit_mass_chars": mass,
    }


def near_miss_evaluations(
    probes: list[dict[str, Any]],
    probabilities: list[float],
    spec: NearMissCriterion = NEAR_MISS_CRITERION,
) -> list[dict[str, Any]]:
    """Selection pass over scored eval-set probes (pure, unit-testable).

    ``probes`` are probe_row JSONL dicts (``pair_id``/``class``/``label``/
    ``record``/``candidate``/``similarity``); ``probabilities`` are the
    runner's P(dup) values in the same order. Returns one admission dict
    per near-miss ready for :func:`registry_row` assembly — an EMPTY list
    for healthy probes: the criterion is the miss, not the geometry.
    """
    if len(probes) != len(probabilities):
        raise ValueError(
            f"probes/probabilities length mismatch: {len(probes)} vs {len(probabilities)}"
        )
    hits: list[dict[str, Any]] = []
    for row, probability in zip(probes, probabilities):
        if str(row.get("class")) not in spec.probe_classes:
            continue
        g = _geometry(row)
        fm = g["features"]
        if float(row["similarity"]) < spec.min_cos:
            continue
        if fm["char5_containment"] < spec.min_char5_containment:
            continue
        if spec.require_zero_len_deltas and not g["len_deltas_zero"]:
            continue
        if spec.require_matching_metadata and not g["metadata_matches"]:
            continue
        if probability >= spec.max_miss_probability:
            continue  # the probe is healthy — not a near-miss
        hits.append(
            {
                "pair_id": str(row["pair_id"]),
                "probe_class": str(row["class"]),
                "similarity": float(row["similarity"]),
                "key_facts": {
                    "char5_jaccard": round(fm["char5_jaccard"], 6),
                    "char5_containment": round(fm["char5_containment"], 6),
                    "tag_jaccard": round(fm["tag_jaccard"], 6),
                    "cos_target": fm["cos_target"],
                    "len_deltas_zero": g["len_deltas_zero"],
                    "body_edit_mass_chars": g["body_edit_mass_chars"],
                },
                "probability": round(probability, 6),
            }
        )
    return hits


def registry_row(
    hit: dict[str, Any],
    *,
    weights_sha256: str,
    added: str,
    evidence: str,
    status: str = "open",
) -> dict[str, Any]:
    """One append-only registry row. ``status`` is ``open`` until the
    corpus revision carrying the quota passes Layer A on it; the exit is
    ``resolved`` (with ``resolved_at``), never a row deletion."""
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "near_miss_id": str(hit["pair_id"]),
        "probe_class": str(hit["probe_class"]),
        "similarity": float(hit["similarity"]),
        "probability": float(hit["probability"]),
        "key_facts": hit["key_facts"],
        "weights_sha256": weights_sha256,
        "evidence": evidence,
        "status": status,
        "added": added,
    }


# ── registry I/O (append-only; load refuses to silently truncate) ────────────


def default_registry_path() -> Path:
    """datasets/watchlist/near-miss.jsonl under the repo root."""
    return (
        Path(__file__).resolve().parents[3]
        / "datasets"
        / WATCHLIST_DIRNAME
        / REGISTRY_FILENAME
    )


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse the registry JSONL. A missing file is an EMPTY registry (the
    honest precondition, same as load_llm_slots); an existing file must
    parse and carry the row schema — anything else fails loud."""
    path = default_registry_path() if path is None else Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}[{n}]: not valid JSON — {exc}") from exc
        for key in (
            "schema_version",
            "near_miss_id",
            "key_facts",
            "weights_sha256",
            "evidence",
            "status",
            "added",
        ):
            if key not in row:
                raise ValueError(f"{path}[{n}]: registry row missing {key!r}")
        if row["status"] not in ("open", "resolved"):
            raise ValueError(
                f"{path}[{n}]: status must be open|resolved, got {row['status']!r}"
            )
        rows.append(row)
    seen: set[str] = set()
    for row in rows:
        if row["near_miss_id"] in seen:
            raise ValueError(
                f"{path}: duplicate near_miss_id {row['near_miss_id']!r} — "
                "append-only registry, same id = same row"
            )
        seen.add(row["near_miss_id"])
    return rows


# ── the quota gate (build-plan check, mirrors the corner-QA refusal) ─────────


def quota_violations(
    rows: list[dict[str, Any]],
    *,
    corpus_version: str,
    registry: list[dict[str, Any]] | None = None,
    quota: int = WATCHLIST_QUOTA,
) -> list[str]:
    """Refusal lines for a corpus build plan (the gen_dataset_v4 gate,
    policy §5 sanction order). Rules:

    - a registry with NO open cases → no quota (nothing is watched);
    - a corpus revision below :data:`MIN_CORPUS_VERSION_CARRYING_QUOTA`
      is exempt (sealed history — v4.2 predates the mechanism);
    - otherwise the build must contain ≥ ``quota`` near-0009-FAMILY
      pairs labeled duplicate: tiny-but-real body edit (difflib mass in
      [MIN_FAMILY_EDIT_MASS_CHARS; MAX_FAMILY_EDIT_MASS_CHARS] — mass 0
      is the identity corner, gated elsewhere), zero len-deltas,
      matching tag/type/lang surfaces, char5 containment ≥ the criterion
      bound, and cos ≥ the criterion floor (the family geometry; the
      measured cosine of injected pairs rides the corpus embedder pass
      as usual — a family pair that measures below the floor is caught
      HERE as a quota miss, which is the honest check).
    """
    active = load_registry() if registry is None else registry
    open_ids = sorted(r["near_miss_id"] for r in active if r["status"] == "open")
    if not open_ids:
        return []
    digits = [int(part) for part in corpus_version.split(".") if part.isdigit()]
    if len(digits) >= 2 and tuple(digits[:2]) < MIN_CORPUS_VERSION_CARRYING_QUOTA:
        return []  # sealed-history versions predate the mechanism

    family = 0
    for row in rows:
        if row.get("label") != "duplicate":
            continue
        g = _geometry(row)
        fm = g["features"]
        if not g["len_deltas_zero"] or not g["metadata_matches"]:
            continue
        if not (
            MIN_FAMILY_EDIT_MASS_CHARS
            <= g["body_edit_mass_chars"]
            <= MAX_FAMILY_EDIT_MASS_CHARS
        ):
            continue
        if fm["char5_containment"] < NEAR_MISS_CRITERION.min_char5_containment:
            continue
        if float(row["similarity"]) < NEAR_MISS_CRITERION.min_cos:
            continue
        family += 1
    if family < quota:
        return [
            f"watchlist quota: {family} near-0009-family duplicate pairs < {quota} "
            f"(open near-miss ids: {', '.join(open_ids)}); inject the dash-swap "
            "twin family (zero len-deltas, edit mass in "
            f"[{MIN_FAMILY_EDIT_MASS_CHARS}; {MAX_FAMILY_EDIT_MASS_CHARS}]) into "
            "the razor band's duplicate side per prereg addendum §9 "
            "(calibration-b2p-prereg.md)"
        ]
    return []
