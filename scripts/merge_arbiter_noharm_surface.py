#!/usr/bin/env python3
"""Materialize the g_noharm_v42 surface ``noharm-surface-v1`` (TL dispatch).

Sources (committed, sha256 pins ratified by the committee verdictor; every
pin re-verified HERE — trust no one, including this file's own constants):

- datasets/corpus-v4/batch-notdup.jsonl  (36 specs) ece065de5a74…
- datasets/corpus-v4/batch-base-ru.jsonl (48)       404d43f28d38…
- datasets/corpus-v4/batch-base-en.jsonl (48)       526522bf4e74…
- datasets/corpus-v43/batch-translated-sibling-ru.jsonl (40) 6ab5b5faf7aa…
- datasets/corpus-v43/batch-translated-sibling-en.jsonl (40) 36ced21dc2c6…

NOTE ON COUNTS: the verdictor's "39+39" (and 105 total) came from `wc -l`,
which counts NEWLINES — both sibling files end without a trailing newline,
so each hides one record. Actual: 40+40 specs. Recorded as the measured
fact with this explanation (brief: "если число другое — разбери и
зафиксируй фактическое").

Render — frozen generators only, zero reimplementations:

- notdup (gen_dataset_v4 recipe): rec = base[spec.key]; retold =
  _swapcase_record(_fact_edited(rec, spec), title=True, body=True);
  single orientation (the style asymmetry IS the class); pair_id
  V4-PND-%04d in spec order; split by the FROZEN gen_dataset_v4
  stratified_split (fraction 0.25): the first ceil(0.25*36)=9 sorted
  pair_ids are the SEALED v4.2-holdout side. THE TRAP (verdictor): those
  9 rows' texts are sealed v4.2-holdout material — a wholesale sweep of
  the 36-spec pool would burn g_noharm on sealed content (A-seal-intact
  class). The split is replayed through the frozen function and the
  surface takes ONLY the 27 train ids; "exactly 27" and "0 emitted ids
  ∩ 9 holdout ids" are fail-closed asserts. The 9 ids (IDS ONLY — no
  text) are pinned in the committed doc for the TL's machine check.
- sibling (gen_dataset_v43_strategies.build_translated_rows): not-dup by
  construction, fresh cross-language formulation, justification required
  non-empty per spec (the builder itself refuses otherwise). BOTH
  orientations per spec (recipe: "Both orientations, same as dup") →
  80 specs → 160 V43-TSIB pairs (ru block first, then en — the v4.3
  build's exact consumption order, ids are construction-order based).
  The v4.3-native split (build_corpus_v43.stratified_split10, f=0.10)
  moves the first ceil(0.10*160)=16 sorted pair_ids into the v4.3-holdout
  slice — they are EXCLUDED (same sealed-material trap); the surface
  takes the 144 train-side pairs, "0 emitted ids ∩ 16 slice ids"
  fail-closed, the 16 ids pinned (ids only).

Row schema = the g_noharm loader (_noharm_proposals): pair_id,
title_a/body_a/tags_a/type_a/lang_a/created_a + _b. created_* = null
(v4/v4.3 records carry no envelope date; the loader's A-null-pin null
path applies: flag=1, epoch=0, delta=0).

Prohibition honored: data/merge-arbiter/spot-sheet/ and the
MA-holdout-choice content are NOT read here; the text-hash disjointness
of the surface vs the sheet/holdout is the TL's machine leg after this
commit. Seal files v4.2-holdout/v4.3-holdout are never opened — the two
holdout id sets are REPLAYED from the frozen split functions over the
committed batches (deterministic, construction-seeded), not read from
any sealed dir.

Product:
- data/merge-arbiter/artifacts/merge-v1/noharm-surface-v1.jsonl (durable,
  gitignored) + run-log event with sha256 and counts;
- committed doc section noharm_surface (pins, counts, id pins) in
  datasets/merge-arbiter/merge-v1-thresholds.frozen.json — fixed BEFORE
  gate_start; thresholds and the model payload untouched.

Usage:
  <venv-python> scripts/merge_arbiter_noharm_surface.py \
      --artifact data/merge-arbiter/artifacts/merge-v1/merge_arbiter_v1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import merge_arbiter_train as mt  # noqa: E402  (helpers: log/sha/abort/env)

import build_corpus_v43 as b43  # noqa: E402  (frozen v4.3 split + loader)
import gen_dataset_v4 as g4  # noqa: E402  (frozen v4 notdup recipe + split)
from gen_dataset_v43_strategies import build_translated_rows  # noqa: E402

SURFACE_PATH = mt.ART_DIR / "noharm-surface-v1.jsonl"
SLICE = "merge-arbiter-merge-v1-noharm-surface"

#: Ratified source pins (committee verdictor; re-verified in the prologue).
SOURCE_PINS = {
    "datasets/corpus-v4/batch-notdup.jsonl": "ece065de5a74ccba1ca96401e298a9dd445ba3fef7d6702463cef83efc859d55",
    "datasets/corpus-v4/batch-base-ru.jsonl": "404d43f28d384d0d78e834aed02d59e4bd00fc0d134236a867d20c4d89ba1482",
    "datasets/corpus-v4/batch-base-en.jsonl": "526522bf4e749baf96bdd501e7f46a7a5d5f406aaac3d2c17ef793798ab4c293",
    "datasets/corpus-v43/batch-translated-sibling-ru.jsonl": "6ab5b5faf7aa207cb3ce321a50ea42f89932f6901af014a437788e3c2377da54",
    "datasets/corpus-v43/batch-translated-sibling-en.jsonl": "36ced21dc2c63480bca8482aba9fbdd1c641394e32a5dc9dc8471741c1ca9f0f",
}


def _prologue() -> None:
    """Light-lane prologue with the BUILDER's input list (ML-STD-0003)."""
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    try:
        avail_mb = int(free.splitlines()[1].split()[6])
    except (IndexError, ValueError):
        avail_mb = -1
    mt.log(
        "swap_check",
        swapon_lines=swap.strip().splitlines(),
        free_mem_line=free.strip().splitlines()[1]
        if len(free.splitlines()) > 1
        else "",
        available_mb=avail_mb,
        ram_overrun_stop=bool(avail_mb < 2000),
    )
    if avail_mb < 2000:
        raise mt.InstrumentAbort(f"RAM below declared floor: {avail_mb} MB")
    exe = sys.executable
    env = {
        "executable": exe,
        "venv_suffix_ok": exe.endswith(mt.EXPECTED_ENV["venv_suffix"]),
        "python": sys.version.split()[0],
        "lightgbm": mt.lgb.__version__,
        "numpy": mt.np.__version__,
        "machine": platform.machine(),
    }
    mism = [k for k in ("venv_suffix_ok",) if not env[k]] + [
        k for k in ("python", "lightgbm", "numpy") if env[k] != mt.EXPECTED_ENV[k]
    ]
    mt.log("env", **env, expected=mt.EXPECTED_ENV, mismatch=mism)
    if mism:
        raise mt.InstrumentAbort(f"env pin mismatch: {mism}")
    host = platform.node()
    mt.log("host", hostname=host, expectation=mt.EXPECTED_HOST)
    if host != mt.EXPECTED_HOST:
        raise mt.InstrumentAbort(f"host changed vs {mt.EXPECTED_HOST}: {host}")

    pins = {name: mt.sha256_file(REPO / name) for name in SOURCE_PINS}
    pins.update(
        {
            "merge_arbiter_lib_sha256": mt.sha256_file(Path(mt.lib.__file__)),
            "cortex_features_pair_sha256": mt.sha256_file(
                REPO / "src" / "cortex" / "features" / "pair.py"
            ),
            "artifact_sha256": mt.sha256_file(Path(args_artifact[0])),
        }
    )
    want = dict(SOURCE_PINS)
    doc = json.loads(mt.COMMITTED_DOC.read_text(encoding="utf-8"))
    want["artifact_sha256"] = doc["durable_artifact"]["sha256"]
    want["merge_arbiter_lib_sha256"] = doc["merge_arbiter_lib_sha256"]
    want["cortex_features_pair_sha256"] = doc["cortex_features_pair_sha256"]
    trip = [k for k, v in want.items() if pins[k] != v]
    mt.log("byte_tripwire", pins=pins, expected=want, mismatch=trip)
    if trip:
        raise mt.InstrumentAbort(f"byte tripwire mismatch: {trip}")

    spec_text = mt.SPEC.read_text(encoding="utf-8")
    activated = bool(
        re.search(
            r"\*\*Статус:\*\*\s*ACTIVATED v2 \(29\.10\)\s*\+\s*АДДЕНДУМ 1", spec_text
        )
    )
    ancestry = {}
    for name, commit in mt.ACTIVATION_COMMITS.items():
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPO,
            capture_output=True,
        )
        ancestry[name] = {"commit": commit, "is_ancestor": r.returncode == 0}
    ok = activated and all(v["is_ancestor"] for v in ancestry.values())
    mt.log("activation", status_line_present=activated, commits=ancestry, ok=ok)
    if not ok:
        raise mt.InstrumentAbort("activation assert failed")


#: single-argument carrier for the artifact path (prologue needs it before
#: main() parses into locals — set from cmd)
args_artifact: list[str] = []


def surface_row(pair_id: str, a: dict, b: dict) -> dict:
    """Loader schema row; created_* = null (A-null-pin null path)."""
    return {
        "pair_id": pair_id,
        "title_a": a["title"],
        "body_a": a["body"],
        "tags_a": list(a["tags"]),
        "type_a": a.get("record_type"),
        "lang_a": a.get("language") or a.get("lang"),
        "created_a": None,
        "title_b": b["title"],
        "body_b": b["body"],
        "tags_b": list(b["tags"]),
        "type_b": b.get("record_type"),
        "lang_b": b.get("language") or b.get("lang"),
        "created_b": None,
    }


def notdup_leg() -> tuple[list[dict], list[str], list[str]]:
    """36 specs -> frozen render -> frozen v4.2 split -> 27 train rows.

    Returns (surface_rows, train_ids, holdout9_ids). Holdout TEXTS live
    only in process memory for the split replay and are never emitted.
    """
    specs = g4._read_jsonl(g4.NOTDUP_FILE)
    bases = {
        "ru": {
            r["key"]: r for r in g4._read_jsonl(g4.BATCH_DIR / "batch-base-ru.jsonl")
        },
        "en": {
            r["key"]: r for r in g4._read_jsonl(g4.BATCH_DIR / "batch-base-en.jsonl")
        },
    }
    rows = []
    counters: dict[str, int] = {}
    for spec in specs:
        base = bases[spec["lang"]]
        rec = base[spec["key"]]
        retold = g4._swapcase_record(g4._fact_edited(rec, spec), title=True, body=True)
        counters["PND"] = counters.get("PND", 0) + 1
        rows.append(
            {
                "pair_id": f"V4-PND-{counters['PND']:04d}",
                "label": "not-duplicate",
                "stratum": "N-para-notdup",
                "record": g4._side_of(rec),
                "candidate": g4._side_of(retold),
            }
        )
    if len(rows) != 36:
        raise mt.InstrumentAbort(f"notdup render: {len(rows)} rows, want 36")
    # the FROZEN v4.2 split (first ceil(0.25*n) sorted pair_ids -> holdout)
    train_rows, hold_rows = g4.stratified_split(rows, g4.HOLDOUT_FRACTION)
    hold_ids = sorted(r["pair_id"] for r in hold_rows)
    train_ids = sorted(r["pair_id"] for r in train_rows)
    if len(hold_ids) != 9 or len(train_ids) != 27:
        raise mt.InstrumentAbort(
            f"v4.2 split replay: {len(train_ids)} train / {len(hold_ids)} holdout, "
            "want 27/9"
        )
    mt.log(
        "noharm_notdup_leg",
        specs=len(specs),
        pairs=len(rows),
        train=len(train_ids),
        v42_holdout_excluded=len(hold_ids),
        v42_holdout_ids=hold_ids,
        note="ids only; holdout texts never emitted",
    )
    out = [surface_row(r["pair_id"], r["record"], r["candidate"]) for r in train_rows]
    if len(out) != 27 or ({r["pair_id"] for r in out} & set(hold_ids)):
        raise mt.InstrumentAbort("notdup surface leg: exactly-27 / no-holdout RED")
    return out, train_ids, hold_ids


def sibling_leg() -> tuple[list[dict], list[str], list[str]]:
    """80 specs -> build_translated_rows (both orientations, 160 pairs)
    -> frozen v4.3 split10 -> 144 train-side rows (16 slice ids excluded).
    """
    sib_specs: list[dict] = []
    for name in ("batch-translated-sibling-ru", "batch-translated-sibling-en"):
        sib_specs.extend(
            b43._read_jsonl(REPO / "datasets" / "corpus-v43" / f"{name}.jsonl")
        )
    if len(sib_specs) != 80:
        raise mt.InstrumentAbort(
            f"sibling specs: {len(sib_specs)}, want 80 "
            "(40+40; wc -l undercounts the unterminated tails)"
        )
    # dup_specs=[] — the TSIB counter is per-stratum, ids unaffected
    rows = build_translated_rows([], sib_specs)
    if len(rows) != 160:
        raise mt.InstrumentAbort(
            f"sibling render: {len(rows)} rows, want 160 (80 specs x 2 orientations)"
        )
    ids = [r["pair_id"] for r in rows]
    if len(set(ids)) != 160 or ids[0] != "V43-TSIB-0001" or ids[-1] != "V43-TSIB-0160":
        raise mt.InstrumentAbort(
            "sibling pair_id assignment diverged from the construction-order contract"
        )
    if any(r["label"] != "not-duplicate" for r in rows):
        raise mt.InstrumentAbort("sibling rows must be not-duplicate by construction")
    # the FROZEN v4.3-native split (first ceil(0.10*n) sorted pair_ids -> hold)
    train_rows, hold_rows = b43.stratified_split10(rows, b43.HOLDOUT_FRACTION_DEFAULT)
    hold_ids = sorted(r["pair_id"] for r in hold_rows)
    train_ids = sorted(r["pair_id"] for r in train_rows)
    if len(hold_ids) != 16 or len(train_ids) != 144:
        raise mt.InstrumentAbort(
            f"v4.3 split replay: {len(train_ids)} train / {len(hold_ids)} holdout, "
            "want 144/16"
        )
    mt.log(
        "noharm_sibling_leg",
        specs=len(sib_specs),
        pairs=len(rows),
        train=len(train_ids),
        v43_holdout_slice_excluded=len(hold_ids),
        v43_holdout_slice_ids=hold_ids,
        note="ids only; slice texts never emitted",
    )
    out = [surface_row(r["pair_id"], r["record"], r["candidate"]) for r in train_rows]
    if len(out) != 144 or ({r["pair_id"] for r in out} & set(hold_ids)):
        raise mt.InstrumentAbort("sibling surface leg: exactly-144 / no-slice RED")
    return out, train_ids, hold_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--artifact",
        required=True,
        help="frozen merge-v1 artifact (for the self-test + tripwire)",
    )
    args = ap.parse_args(argv)
    args_artifact.append(args.artifact)

    mt._SLICE = SLICE
    started = dt.datetime.now(dt.timezone.utc)
    mt.log(
        "cost_declaration",
        expected_prolog_s=2,
        expected_cycle_s=60,
        expected_peak_ram_gb=1.0,
        lane="light (draft §9)",
    )
    _prologue()

    notdup_rows, notdup_train, v42_hold9 = notdup_leg()
    sib_rows, sib_train, v43_hold16 = sibling_leg()

    surface = notdup_rows + sib_rows
    pids = [r["pair_id"] for r in surface]
    if len(surface) != 171 or len(set(pids)) != 171:
        raise mt.InstrumentAbort(
            f"surface total {len(surface)} (unique "
            f"{len(set(pids))}), want 171 = 27 + 144"
        )
    for r in surface:
        for side in ("a", "b"):
            if not r[f"title_{side}"].strip() or not r[f"body_{side}"].strip():
                raise mt.InstrumentAbort(f"row {r['pair_id']}: empty {side}-side text")
            if r[f"tags_{side}"] is None:
                raise mt.InstrumentAbort(f"row {r['pair_id']}: null tags")
    # text-hash cross-leg disjointness inside the surface (ids already clean)
    hashes = {mt.sha256_text(r[f"body_{side}"]) for r in surface for side in ("a", "b")}
    mt.log(
        "noharm_surface_assembled",
        notdup=len(notdup_rows),
        sibling=len(sib_rows),
        total=len(surface),
        unique_body_hashes=len(hashes),
    )

    mt.ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(SURFACE_PATH, "w", encoding="utf-8") as fh:
        for r in surface:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    surface_sha = mt.sha256_file(SURFACE_PATH)

    # emission QA on committed material (NOT the gate): the through-pipeline
    # must propose nothing on this surface; diagnostic here, verdict at cycle
    model, tau_hi, tau_m = mt.load_frozen(Path(args.artifact))
    rows_seen, proposals = mt._noharm_proposals(SURFACE_PATH, model, tau_hi, tau_m)
    mt.log(
        "noharm_surface_selftest",
        rows=rows_seen,
        proposals=proposals,
        note="emission QA on committed train-side material; the gate "
        "re-executes this at cycle time behind gate_start",
    )

    doc = json.loads(mt.COMMITTED_DOC.read_text(encoding="utf-8"))
    if "noharm_surface" in doc:
        raise mt.InstrumentAbort(
            "noharm_surface section already present — surface re-emission is forbidden"
        )
    doc["noharm_surface"] = {
        "path": str(SURFACE_PATH),
        "sha256": surface_sha,
        "counts": {
            "notdup": len(notdup_rows),
            "sibling": len(sib_rows),
            "total": len(surface),
        },
        "count_note": "verdictor's 39+39/105 came from wc -l (both sibling "
        "files lack the trailing newline: actual 40+40 specs); "
        "both-orientation sibling recipe gives 160 V43-TSIB "
        "pairs, minus 16 v4.3-holdout-slice ids = 144",
        "sources": SOURCE_PINS,
        "split_pins": {
            "v42_holdout_excluded_ids": v42_hold9,
            "v42_train_ids": notdup_train,
            "v43_holdout_slice_excluded_ids": v43_hold16,
            "v43_train_ids_count": len(sib_train),
        },
        "trap_note": "the 9 v4.2-holdout and 16 v4.3-holdout-slice rows are "
        "sealed material: filtered by the replayed frozen split "
        "id lists, emitted set asserted disjoint (exact 0); "
        "texts never emitted, ids pinned here for the TL check",
        "tl_machine_leg": "surface-vs-sheet and surface-vs-MA-holdout "
        "text-hash disjointness stays with the TL after "
        "this commit",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    doc.pop("committed_doc_sha256", None)
    doc["committed_doc_sha256"] = mt.sha256_text(
        json.dumps(doc, ensure_ascii=False, sort_keys=True)
    )
    mt.COMMITTED_DOC.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    mt.log(
        "noharm_surface_emitted",
        path=str(SURFACE_PATH),
        sha256=surface_sha,
        committed_doc_sha256=doc["committed_doc_sha256"],
        thresholds_untouched=True,
        payload_artifact_untouched=True,
        wall_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
