#!/usr/bin/env python3
"""Gate-set v2 — fresh material generation with the round-7 teacher.

Teacher: Qwen3-1.7B-Instruct Q8_0 (official GGUF, llama.cpp) — the SAME
runtime, prompt contracts and per-pair generation gates as round-7
(retranslate_llm_r7.py machinery, imported, not copied): digit
preservation, target-lang gate, anti-copy (hash + jaccard >= 0.9),
length 40..4000. Twins reuse the round-4 twin contract (TWIN_SYSTEM,
TWIN_PER_JOB, digit-preservation both directions, acceptable, lang).

Generation is SINGLE-SHOT by design (ADDENDUM 1): the pool is frozen
after filtering (pool fp) and the double-run determinism claim covers
the SELECTION stage only. Sampling seeds are derived from the ratified
v2 seed: seed = 385918605 + pass*1_000_000 + global_job_index.

Checkpointed append-only shards under data/embed-r4/r5_gate_v2/gen/
(gitignored); resume = (stratum, key|job, pass) already present; pass
>= 2 re-attempts only keys whose latest row is not ok.

Runs on .venv-r7gen (llama-cpp-python==0.3.36, CPU, threads capped).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

R4_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r4-corpus")
R5_WT = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/wt/embed-r5-corpus")
OUT_DIR = R5_WT / "data" / "embed-r4" / "r5_gate_v2"
GEN_DIR = OUT_DIR / "gen"
SOURCES_PATH = OUT_DIR / "sources.jsonl"
RUN_LOG = R5_WT / "datasets" / "embed-r5" / "run-log.jsonl"

sys.path.insert(0, str(R4_WT / "scripts"))
sys.path.insert(0, str(R5_WT / "datasets" / "embed-r7"))

from gen_embed_r4_texts import (  # noqa: E402
    FAMILY_RU_PROJECTS,
    FAMILY_SPECS,
    TWIN_PER_JOB,
    TWIN_SYSTEM,
    _clean_translation,
    _is_copy,
    _translate_prompt,
    acceptable,
    append_jsonl,
    detect_lang,
    extract_json_array,
    load_jsonl,
    missing_digit_tokens,
    normalise,
)
from retranslate_llm_r7 import MAX_NEW_TOKENS as TRANS_MAX_NEW_TOKENS
from retranslate_llm_r7 import TEACHER_GGUF, TEACHER_MODEL

TRANS_CLASSES = ("llm", "seed-dup", "real")
MAX_NEW_TOKENS_SYNTH = 720  # r4 twin contract
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
SEED_V2 = 385918605
MAX_PASS = 3


def log(event: str, **fields: object) -> None:
    import json

    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "event": f"gate_v2:{event}", **fields},
                            ensure_ascii=False) + "\n")
    print(f"[v2:gen] {event} {fields}", flush=True)


def shard_paths() -> dict[str, Path]:
    return {
        "llm": GEN_DIR / "translations-v2.llm.jsonl",
        "seed-dup": GEN_DIR / "translations-v2.seed-dup.jsonl",
        "real": GEN_DIR / "translations-v2.real.jsonl",
        "twin": GEN_DIR / "synth-twins-v2.jsonl",
    }


def load_jobs() -> tuple[list[dict], list[dict]]:
    """(translation_jobs, twin_jobs) in frozen sources order.

    translation job: {"stratum", "key", "src", "target", "ext"}
      llm/seed-dup: src = the note side per its direction; target = other lang
      real: src = the trimmed unit; target from sources
    twin job: {"job", "family", "npairs", "ext"}
    Base = sources.jsonl; extension-1 = sources_extension1.jsonl (if present).
    Passes 1..3 process base keys; passes >= 4 process extension keys only
    (base echo failures proved seed-independent — closed by fresh notes).
    """
    trans, twins = [], []
    for is_ext, path in ((False, SOURCES_PATH),
                         (True, SOURCES_PATH.parent / "sources_extension1.jsonl")):
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row["kind"] == "note":
                src, tgt = (row["ru"], "en") if row["direction"] == "ru->en" else (row["en"], "ru")
                trans.append({"stratum": row["pool"], "key": row["key"], "src": src,
                              "target": tgt, "ext": is_ext})
            elif row["kind"] == "real":
                trans.append({"stratum": "real", "key": row["key"], "src": row["unit"],
                              "target": row["target"], "ext": is_ext})
            elif row["kind"] == "twin_job":
                twins.append({"job": row["job"], "family": row["family"],
                              "npairs": row["npairs"], "ext": is_ext})
    assert trans and twins, "sources incomplete"
    return trans, twins


def load_state(paths: dict[str, Path]) -> tuple[dict[tuple[str, str], str], set]:
    latest: dict[tuple[str, str], str] = {}
    done: set = set()
    for cls, p in paths.items():
        if not p.exists():
            continue
        for r in load_jsonl(p):
            kid = (cls, r.get("key") or r.get("job"))
            done.add(kid + (r["pass"],))
            latest[kid] = r["status"]
    return latest, done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--pass", dest="gen_pass", type=int, required=True)
    args = p.parse_args(argv)
    assert 1 <= args.gen_pass <= 5

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    paths = shard_paths()
    trans_jobs, twin_jobs = load_jobs()
    latest, done = load_state(paths)

    plan: list[tuple[str, dict, list[dict]]] = []  # (kind, meta, prompts-data)
    seed_counter = 0
    trans_todo = []
    for j in trans_jobs:
        kid = (j["stratum"], j["key"])
        if args.gen_pass == 1:
            if kid + (1,) not in done:
                trans_todo.append(j)
        elif latest.get(kid) != "ok" and kid + (args.gen_pass,) not in done:
            trans_todo.append(j)
    twin_todo = []
    for j in twin_jobs:
        kid = ("twin", j["job"])
        if args.gen_pass == 1:
            if kid + (1,) not in done:
                twin_todo.append(j)
        elif latest.get(kid) != "ok" and kid + (args.gen_pass,) not in done:
            twin_todo.append(j)

    for j in trans_todo:
        plan.append(("trans", {"seed": SEED_V2 + args.gen_pass * 1_000_000 + seed_counter}, j))
        seed_counter += 1
    for j in twin_todo:
        plan.append(("twin", {"seed": SEED_V2 + args.gen_pass * 1_000_000 + seed_counter}, j))
        seed_counter += 1
    if args.gen_pass >= 4:  # extension passes: fresh notes only, no base retries
        plan = [p_ for p_ in plan if p_[2].get("ext")]

    log("gen-start", teacher=TEACHER_MODEL, quant="Q8_0", gen_pass=args.gen_pass,
        trans_todo=len(trans_todo), twin_todo=len(twin_todo),
        seeds="385918605+pass*1e6+job_idx")

    def render(messages: list[dict]) -> str:
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n")
        return "".join(out)

    if not plan:
        log("gen-nothing-todo", gen_pass=args.gen_pass)
        return 0

    from llama_cpp import Llama

    llm = Llama(TEACHER_GGUF, n_ctx=3072, n_threads=args.threads,
                n_batch=512, verbose=False)

    out_paths = {cls: paths[cls].open("a", encoding="utf-8") for cls in paths}
    t0 = time.time()
    kept = bad = 0
    try:
        for n, (kind, meta, job) in enumerate(plan):
            seed = meta["seed"]
            if kind == "trans":
                prompt = _translate_prompt(job["target"], job["src"])
                raw = llm(render(prompt), max_tokens=TRANS_MAX_NEW_TOKENS,
                          temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                          seed=seed, stop=["<|im_end|>"])["choices"][0]["text"]
                text = normalise(_clean_translation(raw))
                reasons = []
                if not text:
                    reasons.append("empty")
                if missing_digit_tokens(job["src"], text):
                    reasons.append("digit-loss:" + ",".join(missing_digit_tokens(job["src"], text)[:3]))
                if detect_lang(text) != job["target"]:
                    reasons.append(f"lang:{detect_lang(text)}!={job['target']}")
                if _is_copy(job["src"], text):
                    reasons.append("copy")
                if not (40 <= len(text) <= 4000):
                    reasons.append(f"len:{len(text)}")
                row = {"stratum": job["stratum"], "key": job["key"],
                       "target_lang": job["target"], "pass": args.gen_pass,
                       "status": "fail" if reasons else "ok", "text": text,
                       "job": f"g2-{job['stratum'][:4]}-{seed_counter:05d}"}
                if reasons:
                    row["reason"] = ";".join(reasons)
                    bad += 1
                else:
                    kept += 1
                out_paths[job["stratum"]].write(
                    json.dumps(row, ensure_ascii=False) + "\n")
                out_paths[job["stratum"]].flush()
            else:
                prompt = [
                    {"role": "system", "content": TWIN_SYSTEM.format(n=job["npairs"])},
                    {"role": "user",
                     "content": f"Family: {FAMILY_SPECS[job['family']]['ru']} "
                                f"Topics: {', '.join(FAMILY_RU_PROJECTS[:6])}."},
                ]
                raw = llm(render(prompt), max_tokens=MAX_NEW_TOKENS_SYNTH,
                          temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                          seed=seed, stop=["<|im_end|>"])["choices"][0]["text"]
                arr = extract_json_array(raw) or []
                pairs = [q for q in arr
                         if isinstance(q, dict)
                         and isinstance(q.get("ru"), str)
                         and isinstance(q.get("en"), str)][: job["npairs"]]
                if not pairs:
                    out_paths["twin"].write(json.dumps(
                        {"stratum": "twin", "job": job["job"], "family": job["family"],
                         "pass": args.gen_pass, "status": "fail", "reason": "parse-empty"},
                        ensure_ascii=False) + "\n")
                    bad += 1
                    continue
                job_ok = True
                for k, q in enumerate(pairs):
                    ru_t, en_t = normalise(q["ru"]), normalise(q["en"])
                    lost = missing_digit_tokens(ru_t, en_t) + missing_digit_tokens(en_t, ru_t)
                    reasons = []
                    if lost:
                        reasons.append("digit-loss")
                    if not (acceptable(ru_t) and acceptable(en_t)):
                        reasons.append("len-or-quality")
                    if detect_lang(ru_t) != "ru" or detect_lang(en_t) != "en":
                        reasons.append("lang")
                    row = {"stratum": "twin", "job": job["job"], "family": job["family"],
                           "slot": k, "pass": args.gen_pass,
                           "status": "fail" if reasons else "ok", "ru": ru_t, "en": en_t}
                    if reasons:
                        row["reason"] = ";".join(reasons)
                        job_ok = False
                        bad += 1
                    else:
                        kept += 1
                    out_paths["twin"].write(json.dumps(row, ensure_ascii=False) + "\n")
                if job_ok:
                    out_paths["twin"].flush()
            if (n + 1) % 10 == 0:
                rate = (n + 1) / max(1.0, time.time() - t0)
                print(f"v2 gen pass{args.gen_pass} {n + 1}/{len(plan)} "
                      f"rate={rate:.3f}/s eta={(len(plan) - n - 1) / max(0.001, rate) / 60:.0f}min",
                      flush=True)
    finally:
        for fh in out_paths.values():
            fh.close()
    log("gen-done", gen_pass=args.gen_pass, kept=kept, bad=bad,
        wall_sec=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    import json  # noqa: E402 — used inside main loop writes

    raise SystemExit(main())
