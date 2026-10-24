#!/usr/bin/env python3
"""g1b flora probes — unseen-scaffold probes from the builder's own recipe
(prereg v3 ADDENDUM-2 §A2.6; alignment §D: class → generator-parameters →
probes, ONE source).

This script NEVER touches the sealed holdout / g4-spec surface. It:

1. loads the SCAFFOLDS library + render/ladder machinery from the frozen
   v4.5 builder module (``b2_graph_corpus_v45.py``) — the same generator
   recipe that minted the train injection;
2. computes the TRAIN-USED scaffold ids from the v4.5 train rows (the
   ``scaffold`` column, Sxx ids) and asserts the probe scaffold set is
   DISJOINT from it (the A2.6 scaffold-disjointness assert);
3. renders four probe classes (g1b table, thresholds frozen):
   - ``paradup``    same-language paraphrase dup (slot verbalization T vs V
                    + order-only reorder lever, taxonomy classes 3/4/5) —
                    P(dup) ≥ 0.95 required;
   - ``factedit``   single-slot fact edit on the dup template (taxonomy
                    class 7, razor shape) — P(dup) < 0.5 required;
   - ``xlangdup``   cross-language dup, A2.2 recipe (T verbatim + V
                    verbalized subfamilies) — P(dup) ≥ 0.5 required;
   - ``xlangsib``   cross-language sibling (dup-family × sib-family,
                    disjoint slot tokens, cloned tags, lm=0) — P(dup) < 0.5.
4. scores similarity with the FROZEN embedder via the builder's own
   ``make_embedder`` (pin assert inside) and the builder's corpus formula
   ``round(min(1, max(0, dot)), 6)``;
5. asserts every probe pair-key is disjoint from the train pair keys.

Probes are synthetic, store-external by construction → the legitimate
graph-empty evidence (zeros + has_graph_evidence=false), exactly what the
production provider would build for such sides (sanity-suite convention).

Outputs: ``g1b-probes.jsonl`` + ``g1b-probes-summary.json``. Scoring is the
gate runner's job (through the exported ONNX artifacts).

Usage:
    python scripts/b2_g1b_probes.py --corpus-dir data/b2-graph-corpus/v45 \
        --out-dir <gate dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# APPEND, not insert(0): src/ here carries cortex (not in the engine src),
# while vesmaro/vesma/mnemos must resolve via ENGINE_SRC, which the builder
# prepends at make_embedder time — inserting src/ first shadows the engine
# package with the local namespace stub (embeddings import breaks).
sys.path.append(str(REPO_ROOT / "src"))

from b2_graph_corpus_v45 import (  # noqa: E402
    DOSE_LADDER,
    ENGINE_SRC,
    SCAFFOLDS,
    SIB2,
    TAILS,
    body_len_delta,
    case_run,
    detect_lang,
    dot,
    ladder_combos,
    make_embedder,
    render_template,
    swap_sentences,
)
from b2_rev1n_lib import load_jsonl  # noqa: E402
from cortex.features.pair import PairRecord, features  # noqa: E402


def _ensure_vesmaro_alias() -> None:
    """Engine 6.4.0 renamed the package vesmaro -> vesma (engine commit
    5e0e4b4); the frozen builder still imports ``vesmaro.embeddings``.
    Alias the renamed home under the old name BEFORE make_embedder runs.
    The builder's pin assert (nano:sha256:3b752e06...) proves the frozen
    embedder either way — a wrong provider refuses loudly."""
    import importlib
    import types

    try:
        importlib.import_module("vesmaro.embeddings")
        return
    except ImportError:
        pass
    if ENGINE_SRC not in sys.path:
        sys.path.insert(0, ENGINE_SRC)
    vesmaro = types.ModuleType("vesmaro")
    vesmaro.__path__ = [str(Path(ENGINE_SRC) / "vesma")]
    sys.modules["vesmaro"] = vesmaro
    emb = importlib.import_module("vesma.embeddings")
    sys.modules["vesmaro.embeddings"] = emb
    vesmaro.embeddings = emb


_ensure_vesmaro_alias()

COMBO_CAP = 8          # deterministic slot combos per family (ladder_combos)
SIB_TEMPLATE_CAP = 8   # sibling-side combos per template


def refuse(msg: str) -> None:
    raise SystemExit(f"[g1b-probes] REFUSE: {msg}; STOP")


def pair_key(record: dict, candidate: dict) -> tuple[str, str]:
    import hashlib

    def side_sha(s: dict) -> str:
        obj = {k: s.get(k) for k in ("title", "body", "tags", "language",
                                     "record_type", "created_at")}
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")).hexdigest()

    return side_sha(record), side_sha(candidate)


def make_side(scaf: dict, lang: str, body: str) -> dict:
    return {
        "title": scaf["title"][lang],
        "body": body,
        "tags": list(scaf["tags"]),
        "language": lang,
        "record_type": scaf["type"],
    }


SHADOW_KEYS = ("ch3c", "bld", "tld")


def _pair_record(side: dict) -> PairRecord:
    return PairRecord(
        title=str(side.get("title", "")),
        body=str(side.get("body", "")),
        tags=tuple(str(t) for t in side.get("tags") or []),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def probe_shadow(rec: dict, cand: dict, sim: float) -> dict[str, float]:
    """§D shadow of a pair through the FROZEN 13-feature contract."""
    fv = features(_pair_record(rec), _pair_record(cand), sim)
    by_name = dict(zip(fv.names, fv.values))
    return {
        "ch3c": float(by_name["char3_containment"]),
        "bld": float(by_name["body_len_delta"]),
        "tld": float(by_name["title_len_delta"]),
    }


def train_shadow_ranges(rows: list[dict]) -> dict[str, dict]:
    """Per-stratum [p5..p95] of the §D shadow over the train corpus
    (NA audit 23.10): xlangdup ↔ the ``translated-dup`` train family in the
    [0.75,0.86] band; xlangsib ↔ the ``translated-sibling`` family in the
    same band; paradup/factedit (report-only this cycle) ↔ the P-* positive
    mass and the ``N-fact-edit`` razor population respectively. A stratum
    under 20 rows is a hard refuse (no silent widening)."""
    strata: dict[str, list[dict]] = {"xlangdup": [], "xlangsib": [],
                                     "paradup_report_only": [],
                                     "factedit_report_only": []}
    for r in rows:
        sim = float(r.get("similarity") or 0.0)
        sh = probe_shadow(r["record"], r["candidate"], sim)
        st = str(r.get("stratum", ""))
        if st == "translated-dup" and 0.75 <= sim <= 0.86:
            strata["xlangdup"].append(sh)
        elif st == "translated-sibling" and 0.75 <= sim <= 0.86:
            strata["xlangsib"].append(sh)
        elif st.startswith("P-"):
            strata["paradup_report_only"].append(sh)
        elif st == "N-fact-edit":
            strata["factedit_report_only"].append(sh)

    def ranges(samples: list[dict]) -> dict:
        if len(samples) < 20:
            refuse(f"§D shadow stratum too thin: {len(samples)} rows (<20)")
        out: dict = {}
        for key in SHADOW_KEYS:
            vals = sorted(s[key] for s in samples)
            n = len(vals)
            out[key] = (vals[max(0, int(round(0.05 * (n - 1))))],
                        vals[min(n - 1, int(round(0.95 * (n - 1))))])
        out["_n"] = n
        return out

    return {name: ranges(samples) for name, samples in strata.items()}


def build_probes(corpus_dir: Path) -> tuple[list[dict], dict]:
    rows = load_jsonl(corpus_dir / "train.jsonl")
    used: dict[str, set[str]] = {}
    train_keys: set[tuple[str, str]] = set()
    for r in rows:
        sid = r.get("scaffold")
        if sid and len(str(sid)) == 3 and str(sid).startswith("S"):
            used.setdefault(str(sid), set()).add(str(r.get("stratum", "")))
        train_keys.add(pair_key(r["record"], r["candidate"]))

    shadow_ranges = train_shadow_ranges(rows)
    dup_cell_sim: dict[tuple[str, str, int], float] = {}
    sib_pending: list[dict] = []

    unused = [s for s in SCAFFOLDS if s["id"] not in used]
    if not unused:
        refuse("no unseen scaffolds left in the frozen SCAFFOLDS library "
               f"(train used {len(used)}/{len(SCAFFOLDS)}); g1b undefined")
    probe_scaffold_ids = {s["id"] for s in unused}
    if probe_scaffold_ids & set(used):
        refuse("scaffold disjointness assert failed")

    embed_side, _embed_text, pin = make_embedder()

    def sim_of(rec: dict, cand: dict) -> float:
        return round(min(1.0, max(0.0, dot(embed_side(rec), embed_side(cand)))), 6)

    probes: list[dict] = []
    counters = {"paradup": 0, "factedit": 0, "xlangdup": 0, "xlangsib": 0}

    def emit(cls: str, scaffold: str, rec: dict, cand: dict, note: str) -> None:
        key = pair_key(rec, cand)
        if key in train_keys:
            refuse(f"probe {cls}#{counters[cls]} collides with a train pair key")
        counters[cls] += 1
        probes.append({
            "pair_id": f"G1B-{cls.upper()}-{counters[cls]:04d}",
            "probe_class": cls,
            "scaffold": scaffold,
            "label_rule": "dup by construction (slot map equal / order-only lever)"
            if cls in ("paradup", "xlangdup")
            else "not-dup by construction (slot token edited / sib family)",
            "record": rec,
            "candidate": cand,
            "similarity": sim_of(rec, cand),
            "lang_match": 1.0 if detect_lang(rec["body"]) == detect_lang(cand["body"]) else 0.0,
            "note": note,
        })

    for scaf in unused:
        sid = scaf["id"]
        combos = ladder_combos(scaf["slots"], COMBO_CAP)
        sib_combos = ladder_combos(scaf["sib"]["slots"], SIB_TEMPLATE_CAP)

        # ── paradup: same slot map, wording/order differs (classes 3/4/5) ──
        for values in combos:
            for lang in ("ru", "en"):
                base = make_side(scaf, lang, render_template(scaf["dup"][lang], values, lang, "T"))
                verbal = make_side(scaf, lang, render_template(scaf["dup"][lang], values, lang, "V"))
                emit("paradup", sid, base, verbal, f"verbalization paraphrase ({lang})")
                emit("paradup", sid, base,
                     make_side(scaf, lang, swap_sentences(verbal["body"])),
                     f"verbalization + order-only lever ({lang})")

        # ── factedit: exactly ONE slot token stepped on the ladder (class 7)
        for values in combos:
            for slot in sorted(scaf["slots"]):
                ladder = scaf["slots"][slot]
                if len(ladder) < 2:
                    continue
                edited = dict(values)
                cur = values[slot]
                idx = next(i for i, v in enumerate(ladder) if v["t"] == cur["t"])
                edited[slot] = ladder[(idx + 1) % len(ladder)]
                rec = make_side(scaf, "ru", render_template(scaf["dup"]["ru"], values, "ru", "T"))
                cand = make_side(scaf, "ru", render_template(scaf["dup"]["ru"], edited, "ru", "T"))
                base_tokens = {v["t"] for v in values.values()}
                cand_tokens = {v["t"] for v in edited.values()}
                diff = base_tokens ^ cand_tokens
                if len(diff) != 2:  # exactly one token out + one token in
                    refuse(f"factedit probe on {sid}/{slot}: token diff {diff!r} != single edit")
                emit("factedit", sid, rec, cand, f"single-slot edit {slot}")

        # ── xlangdup / xlangsib: A2.2 recipe — JOINT placement (NA fix
        # 2026-10-23). Per cell: the sib template variants' reachable sim
        # sets (tails × dose grid, bld within the gating §D shadow window)
        # are built FIRST; a dup candidate is acceptable only if BOTH sib
        # templates of the same cell can pair within ±0.005 of its sim —
        # the A2.2 pairing holds by construction, not by post-hoc fitting.
        # Dup rows: one per cell, greedy ±0.005 spacing, §D-shadow-aware. ──
        XL_TAILS = ("none", "a", "b", "c", "d")
        XL_BAND = (0.75, 0.86)
        SIB_TOL = 0.005
        sib_bld_lo, sib_bld_hi = shadow_ranges["xlangsib"]["bld"]

        def dose_sweep(rec: dict, cand_body: str):
            """All reachable (dose, sim) for one rendered pair, dose grid."""
            for k in range(0, 20):
                d = round(0.05 * k, 2)
                body = case_run(cand_body, d) if d else cand_body
                cand = make_side(scaf, "en", body)
                yield d, sim_of(rec, cand)

        sib_pool: dict[int, dict[str, list[dict]]] = {}
        for ci, (_values, svals) in enumerate(zip(combos, sib_combos)):
            sib_pool[ci] = {}
            for tmpl_name, tmpl in (("sib", scaf["sib"]["en"]),
                                    ("sib2", SIB2[sid]["en"])):
                base_en = render_template(tmpl, svals, "en", "T")
                pool = []
                for cand_t, rec_t in (("a", "none"), ("a2", "none"),
                                      ("a3", "none"), ("none", "a"),
                                      ("none", "a2"), ("none", "a3"),
                                      ("a", "a2"), ("a3", "a"),
                                      ("none", "none")):
                    rec = make_side(scaf, "ru",
                                    render_template(scaf["dup"]["ru"],
                                                    combos[ci], "ru", "T")
                                    + TAILS[rec_t]["ru"])
                    cand_body = base_en + TAILS[cand_t]["en"]
                    cand = make_side(scaf, "en", cand_body)
                    bld = float(body_len_delta(rec, cand))
                    if not (sib_bld_lo <= bld <= sib_bld_hi):
                        continue
                    for d, sim in dose_sweep(rec, cand_body):
                        pool.append({"rec": rec, "cand": cand, "bld": bld,
                                     "cand_t": cand_t, "rec_t": rec_t,
                                     "dose": d if d else None, "sim": sim})
                if not pool:
                    refuse(f"xlangsib {sid}/{tmpl_name} cell {ci}: no variant "
                           f"with bld in ({sib_bld_lo:.0f},{sib_bld_hi:.0f}); "
                           f"STOP")
                sib_pool[ci][tmpl_name] = pool

        def sib_feasible(ci: int, sim: float) -> bool:
            if ci not in sib_pool:
                return True  # combo has no sib family — unconstrained cell
            return all(any(abs(e["sim"] - sim) <= SIB_TOL for e in pool)
                       for pool in sib_pool[ci].values())

        def sib_pick(ci: int, sim: float, used_ids: set[int]) -> dict:
            for tmpl_name in ("sib", "sib2"):
                cands = [e for e in sib_pool[ci][tmpl_name]
                         if id(e) not in used_ids
                         and abs(e["sim"] - sim) <= SIB_TOL]
                if not cands:
                    refuse(f"xlangsib {sid}/{tmpl_name} cell {ci}: pairing "
                           f"pool exhausted at sim={sim}; STOP")
                best = min(cands, key=lambda e: (abs(e["sim"] - sim),
                                                 e["dose"] is not None,
                                                 e["cand_t"]))
                used_ids.add(id(best))
                return {"tmpl_name": tmpl_name, **best}
            refuse("unreachable")

        ranges = shadow_ranges["xlangdup"]

        def in_range(sh: dict, _r: dict = ranges) -> bool:
            return all(_r[k][0] <= sh[k] <= _r[k][1] for k in SHADOW_KEYS)

        dup_cell_sim: dict[tuple[str, int], float] = {}
        for family in ("T", "V"):
            grid: list[dict] = []
            for ci, values in enumerate(combos):
                base_ru = render_template(scaf["dup"]["ru"], values, "ru", family)
                base_en = render_template(scaf["dup"]["en"], values, "en", family)
                for tail_id in XL_TAILS:
                    rec_body = base_ru + TAILS[tail_id]["ru"]
                    cand_body = base_en + TAILS[tail_id]["en"]
                    cand = make_side(scaf, "en", cand_body)
                    # dose modulation on the record side (mirror of the
                    # tver dose rows: normalization-invisible — the §D
                    # shadow is invariant under case_run, only sim moves)
                    for d in (None,) + tuple(round(0.05 * k, 2)
                                             for k in range(1, 20)):
                        rec = make_side(scaf, "ru",
                                        case_run(rec_body, d) if d else rec_body)
                        sim = sim_of(rec, cand)
                        if not (XL_BAND[0] <= sim <= XL_BAND[1]):
                            continue
                        grid.append({"ci": ci, "sim": sim, "tail": tail_id,
                                     "dose": d, "rec": rec, "cand": cand,
                                     "shadow": probe_shadow(rec, cand, sim)})
            if len(grid) < len(combos):
                refuse(f"xlangdup {sid}/{family}: tail/dose grid {len(grid)} < "
                       f"{len(combos)} combos inside band {XL_BAND}; STOP")
            # Per-cell best-row pick: sib-paired cells take ONLY sib-feasible
            # rows (hard — the A2.2 ±0.005 pairing is by construction);
            # spacing between chosen rows is not a recipe assert for probes
            # and is not enforced.
            n_sib_cells = len(sib_pool)
            picked: list[dict] = []
            for ci in range(len(combos)):
                cands = [g for g in grid if g["ci"] == ci]
                if ci < n_sib_cells:
                    cands = [g for g in cands if sib_feasible(ci, g["sim"])]
                if not cands:
                    refuse(f"xlangdup {sid}/{family} cell {ci}: no feasible "
                           f"candidate under joint pairing; STOP")
                picked.append(min(cands, key=lambda g: (
                    not in_range(g["shadow"]), g["sim"], g["tail"],
                    g.get("dose") is not None)))
            for g in sorted(picked, key=lambda x: x["sim"]):
                dose_note = f", dose={g['dose']}" if g.get("dose") else ""
                emit("xlangdup", sid, g["rec"], g["cand"],
                     f"A2.2 subfamily {family} (tails r/e={g['tail']}"
                     f"{dose_note}, "
                     f"shadow {'in' if in_range(g['shadow']) else 'OUT'})")
                dup_cell_sim[(family, g["ci"])] = g["sim"]

        used_sib: set[int] = set()
        n_sib_cells = len(sib_pool)
        for (family, ci), sim in sorted(dup_cell_sim.items(), key=lambda x: x[1]):
            if ci >= n_sib_cells:
                continue  # unconstrained cell — no sib family to pair
            pick = sib_pick(ci, sim, used_sib)
            emit("xlangsib", sid, pick["rec"], pick["cand"],
                 f"cross-lang sibling ({pick['tmpl_name']}; tails c/r="
                 f"{pick['cand_t']}/{pick['rec_t']}; bld={pick['bld']:.0f}; "
                 f"band ±{SIB_TOL} of dup {family}-partner cell {ci}"
                 + (f", dose={pick['dose']}" if pick.get("dose") else "") + ")")

    # class sanity: declared language pairs
    lm_bad = [p["pair_id"] for p in probes
              if (p["probe_class"] in ("paradup", "factedit")) != (p["lang_match"] == 1.0)]
    if lm_bad:
        refuse(f"lang_match contract violated on {len(lm_bad)} probes, e.g. {lm_bad[:3]}")

    # §D shadow assert (NA fix): xlang legs GATING — every probe's shadow
    # must sit inside [p5..p95] of its train stratum; legacy legs
    # (paradup/factedit, geometry untouched per NA) report-only, disclosed.
    shadow_report: dict = {}
    for cls in ("xlangdup", "xlangsib"):
        ranges = shadow_ranges[cls]
        bad = []
        for p in probes:
            if p["probe_class"] != cls:
                continue
            sh = probe_shadow(p["record"], p["candidate"], p["similarity"])
            ok = all(ranges[k][0] <= sh[k] <= ranges[k][1] for k in SHADOW_KEYS)
            p["shadow"] = {**{k: round(sh[k], 6) for k in SHADOW_KEYS},
                           "in_range": ok}
            if not ok:
                bad.append(p["pair_id"])
        if bad:
            refuse(f"{cls} §D shadow assert FAILED: {len(bad)} probes outside "
                   f"[p5..p95] of the train stratum, e.g. {bad[:4]}; STOP")
        shadow_report[cls] = {
            "stratum_n": ranges["_n"],
            "ranges": {k: list(ranges[k]) for k in SHADOW_KEYS},
            "n_probes": sum(1 for p in probes if p["probe_class"] == cls),
            "status": "all in range (gating)",
        }
    for cls in ("paradup", "factedit"):
        ranges = shadow_ranges[f"{cls}_report_only"]
        flags = []
        for p in probes:
            if p["probe_class"] != cls:
                continue
            sh = probe_shadow(p["record"], p["candidate"], p["similarity"])
            ok = all(ranges[k][0] <= sh[k] <= ranges[k][1] for k in SHADOW_KEYS)
            p["shadow"] = {**{k: round(sh[k], 6) for k in SHADOW_KEYS},
                           "in_range": ok}
            flags.append(ok)
        shadow_report[cls] = {
            "stratum_n": ranges["_n"],
            "ranges": {k: list(ranges[k]) for k in SHADOW_KEYS},
            "n_probes": len(flags),
            "status": f"{sum(flags)}/{len(flags)} in range (report-only)",
        }

    summary = {
        "probe_scaffolds": sorted(probe_scaffold_ids),
        "train_used_scaffolds": sorted(used),
        "scaffold_disjointness": "asserted (probe scaffolds ∩ train-used = ∅)",
        "scaffold_source": "SCAFFOLDS (frozen builder library), same generator recipe",
        "embedder_pin": pin,
        "counts": counters,
        "total": len(probes),
        "similarity_stats": {
            cls: {
                "n": counters[cls],
                "cos_min": min(p["similarity"] for p in probes if p["probe_class"] == cls),
                "cos_max": max(p["similarity"] for p in probes if p["probe_class"] == cls),
                "cos_mean": round(statistics.fmean(
                    p["similarity"] for p in probes if p["probe_class"] == cls), 6),
            }
            for cls in counters if counters[cls]
        },
        "regime": ("NA fix 2026-10-23: full A2.2 machinery on xlang legs "
                   "(tails both sides / band-paired sib), §D shadow gates "
                   "xlang legs"),
        "shadow_assert": shadow_report,
    }
    return probes, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b2-g1b-probes")
    parser.add_argument("--corpus-dir", default="data/b2-graph-corpus/v45")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes, summary = build_probes(Path(args.corpus_dir))
    with (out_dir / "g1b-probes.jsonl").open("w", encoding="utf-8") as fh:
        for p in probes:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out_dir / "g1b-probes-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
