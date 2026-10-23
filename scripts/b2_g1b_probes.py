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

from b2_graph_corpus_v45 import (  # noqa: E402
    SCAFFOLDS,
    SIB2,
    detect_lang,
    dot,
    ladder_combos,
    make_embedder,
    render_template,
    swap_sentences,
)
from b2_rev1n_lib import load_jsonl  # noqa: E402

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


def build_probes(corpus_dir: Path) -> tuple[list[dict], dict]:
    rows = load_jsonl(corpus_dir / "train.jsonl")
    used: dict[str, set[str]] = {}
    train_keys: set[tuple[str, str]] = set()
    for r in rows:
        sid = r.get("scaffold")
        if sid and len(str(sid)) == 3 and str(sid).startswith("S"):
            used.setdefault(str(sid), set()).add(str(r.get("stratum", "")))
        train_keys.add(pair_key(r["record"], r["candidate"]))

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

        # ── xlangdup: A2.2 recipe, T and V subfamilies, ru→en ─────────────
        for values in combos:
            rec = make_side(scaf, "ru", render_template(scaf["dup"]["ru"], values, "ru", "T"))
            emit("xlangdup", sid, rec,
                 make_side(scaf, "en", render_template(scaf["dup"]["en"], values, "en", "T")),
                 "A2.2 subfamily T (verbatim tokens)")
            emit("xlangdup", sid, rec,
                 make_side(scaf, "en", render_template(scaf["dup"]["en"], values, "en", "V")),
                 "A2.2 subfamily V (verbalized slots)")

        # ── xlangsib: dup-family × sib-family, same scaffold, lm=0 ────────
        for values, svals in zip(combos, sib_combos):
            rec = make_side(scaf, "ru", render_template(scaf["dup"]["ru"], values, "ru", "T"))
            for tmpl_name, tmpl in (("sib", scaf["sib"]["en"]), ("sib2", SIB2[sid]["en"])):
                emit("xlangsib", sid, rec,
                     make_side(scaf, "en", render_template(tmpl, svals, "en", "T")),
                     f"cross-lang sibling ({tmpl_name} template)")

    # class sanity: declared language pairs
    lm_bad = [p["pair_id"] for p in probes
              if (p["probe_class"] in ("paradup", "factedit")) != (p["lang_match"] == 1.0)]
    if lm_bad:
        refuse(f"lang_match contract violated on {len(lm_bad)} probes, e.g. {lm_bad[:3]}")

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
