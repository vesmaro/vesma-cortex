#!/usr/bin/env python3
"""Round-4 embedder corpus — teacher generation of the NEW text pool.

Prereg round-4 §2 + addendum 6 §A.3 (ACTIVATED): the synthetic part is
fully regenerated for the round-4 F-corpus and the cross-lingual share
is filled with NEW corpus-stage RU<->EN translated pairs (TL verdict
2026-10-07: prepared v4.3 batches = seed; fill to 15-20% of the FINAL
denominator). Round-3 synthetic outputs are NOT reused.

Subcommands (checkpoint-safe: outputs append to data/embed-r4/gen/,
job-level done-keys are skipped on re-run; a top-up pass backfills
gates-lost rows):
    translate-real  translate every real-part record to its other
                    language (LOCAL teacher only — store-derived content
                    never leaves the machine, privacy gates §A.3)
    synth           regenerate the synthetic pool: 14 round-3 families
                    x {ru,en}, incl. RU<->EN twin rows (digit-preserving)

Companion scripts: filter_embed_r4.py (teacher-cos pair filter),
assemble_embed_r4.py (gates, decontamination, asserts, split, fp).

Determinism: assembly/split/fingerprint are deterministic given the
pools (double assembly = byte-identical fp); generation is seeded per
call but LLM sampling is not byte-reproducible across environments —
the corpus fingerprint is the seal, the seed scheme is provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma")
for p in (str(REPO_ROOT / "src"), str(_ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.dataset.prepare_dataset import (  # noqa: E402
    count_tokens,
    detect_lang,
    normalise,
)

DATA_DIR = REPO_ROOT / "data" / "embed-r4"
GEN_DIR = DATA_DIR / "gen"
REAL_PART = REPO_ROOT / "datasets" / "v43" / "real-part.jsonl"
TRANSLATED_SEED_DIR = REPO_ROOT / "datasets" / "corpus-v43"
RUN_LOG = DATA_DIR / "run-log.jsonl"

GEN_MODEL = "Qwen/Qwen3-0.6B"
#: GGUF Q8_0 of the SAME ratified open-weights model (official Qwen repo);
#: llama.cpp runtime — HF-transformers CPU decode projected ~50h for the
#: pool, llama.cpp ~4-5x faster per job (run-log 2026-10-07 engine swap).
GEN_GGUF = (
    "/var/home/abyss/.distrobox/ubuntu/home/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-0.6B-GGUF/snapshots/"
    "23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
)

MIN_CHARS = 40
MAX_CHARS = 4000
MAX_TOKENS = 256

BATCH = 8           # rows requested per one completion
TWIN_PER_JOB = 6    # pairs requested per one twin completion
JOB_BATCH = 8       # job prompts per generate() call (left-padded batch)
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
MAX_NEW_TOKENS_SYNTH = 720
MAX_NEW_TOKENS_TRANSLATE = 560
UNIT_TOKEN_CAP = 100  # source unit cap so the translation fits the gate

_DIGIT_RUN = re.compile(r"\d+")


def log_event(event: str, **fields: object) -> None:
    """Append-only run-log (data-contract §8 discipline)."""
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields},
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"[run-log] {event} {fields}", flush=True)


def text_hash(text: str) -> str:
    """Canonical dedup/exclusion key (round-3 convention)."""
    return hashlib.sha256(normalise(text).lower().encode("utf-8")).hexdigest()


def digit_tokens(text: str) -> list[str]:
    return _DIGIT_RUN.findall(text)


def missing_digit_tokens(original: str, translated: str) -> list[str]:
    """Addendum-7 rule: every digit run must occur standalone (no
    adjacent digits); only LOST runs are defects."""
    lost: list[str] = []
    for tok in digit_tokens(original):
        if not re.search(rf"(?<!\d){re.escape(tok)}(?!\d)", translated):
            lost.append(tok)
    return lost


def acceptable(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS and count_tokens(text) <= MAX_TOKENS


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class Generator:
    """Local Qwen3-0.6B chat generator (llama.cpp, Q8_0 GGUF, no network).

    Prompts are rendered in the Qwen3 chat format directly, with the
    official empty-``<think>`` block (non-thinking mode).
    """

    def __init__(self, threads: int, ctx: int = 3072):
        from llama_cpp import Llama

        self.llm = Llama(GEN_GGUF, n_ctx=ctx, n_threads=threads, n_batch=512, verbose=False)
        log_event("generator-loaded", model=GEN_MODEL, runtime="llama.cpp", quant="Q8_0", ctx=ctx, threads=threads)

    @staticmethod
    def render(messages: list[dict]) -> str:
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        out.append("<|im_start|>assistant\n<think>\n\n</think>\n")
        return "".join(out)

    def chat(self, prompts: list[list[dict]], max_new_tokens: int, seed: int) -> list[str]:
        outs = []
        for p in prompts:
            res = self.llm(
                self.render(p),
                max_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                seed=seed,
                stop=["<|im_end|>"],
            )
            outs.append(res["choices"][0]["text"].strip())
        return outs


def extract_json_array(raw: str) -> list | None:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except json.JSONDecodeError:
        return None


def extract_json_obj(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ── phase C: real-record translations ────────────────────────────────────────

TRANSLATE_SYSTEM_EN = (
    "You translate personal work notes into English. Rules: "
    "(1) translate the WHOLE source text; "
    "(2) copy every number, date, timestamp, version and amount UNCHANGED, digit for digit, "
    "including full ISO timestamps like 2026-08-07T10:25:29.989686+00:00; "
    "(3) write natural plain prose, no markdown headers, no explanations, no quotes around the result; "
    "(4) output ONLY the translation."
)
TRANSLATE_SYSTEM_RU = (
    "Ты переводишь личные рабочие записи на русский язык. Правила: "
    "(1) переводи ВЕСЬ исходный текст; "
    "(2) переписывай все числа, даты, временные метки, версии и суммы БЕЗ ИЗМЕНЕНИЙ, цифра в цифру, "
    "включая полные ISO-метки вроде 2026-08-07T10:25:29.989686+00:00; "
    "(3) пиши естественным обычным языком, без markdown-заголовков, без пояснений, без кавычек вокруг результата; "
    "(4) выведи ТОЛЬКО перевод."
)
TRANSLATE_SHOT_EN = (
    "Source:\nОтчёт 05.10.2026: деплой v2.3.1 на k3s прошёл в 14:32, ошибок нет.\n\n"
    "Translation:\nReport 05.10.2026: the v2.3.1 deploy to k3s completed at 14:32, no errors."
)
TRANSLATE_SHOT_RU = (
    "Source:\nReport Oct 5, 2026: the v2.3.1 deploy to k3s finished at 14:32, no errors.\n\n"
    "Translation:\nОтчёт 5 октября 2026: деплой v2.3.1 на k3s завершился в 14:32, ошибок нет."
)


def _translate_prompt(target: str, unit: str) -> list[dict]:
    if target == "en":
        return [
            {"role": "system", "content": TRANSLATE_SYSTEM_EN},
            {"role": "user", "content": f"Example.\n{TRANSLATE_SHOT_EN}\n\nNow translate.\nSource:\n{unit}\n\nTranslation:"},
        ]
    return [
        {"role": "system", "content": TRANSLATE_SYSTEM_RU},
        {"role": "user", "content": f"Пример.\n{TRANSLATE_SHOT_RU}\n\nТеперь переведи.\nSource:\n{unit}\n\nTranslation:"},
    ]


def _clean_translation(raw: str) -> str:
    """Plain-text output hygiene: strip echo labels, wrapping quotes, fences."""
    t = raw.strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^(Translation|Перевод)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^<translation>|</translation>$", "", t.strip(), flags=re.IGNORECASE)
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'«»“”":
        t = t[1:-1]
    return t.strip()


def _is_copy(source: str, translation: str) -> bool:
    """Near-copy gate: mixed-language sources sometimes come back
    unrendered; such rows would fake translated-pair share."""
    if not translation:
        return False
    if text_hash(source) == text_hash(translation):
        return True
    a = set(re.findall(r"\w+", source.lower()))
    b = set(re.findall(r"\w+", translation.lower()))
    if not a:
        return False
    return len(a & b) / len(a | b) >= 0.9


def real_units() -> list[tuple[str, str, str]]:
    """(key, target_lang, source_unit) for every real-part record."""
    out: list[tuple[str, str, str]] = []
    for rec in load_jsonl(REAL_PART):
        body = (rec.get("side") or {}).get("body") or rec.get("content") or ""
        if not isinstance(body, str) or not body.strip():
            continue
        key = rec.get("content_hash") or text_hash(body)
        src_lang = (rec.get("side") or {}).get("language") or rec.get("language") or detect_lang(body)
        target = "en" if src_lang == "ru" else "ru"
        unit = body.strip()
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n\n" in unit:
            unit = unit.rsplit("\n\n", 1)[0].strip()
        # records without blank lines: drop trailing lines until inside cap
        while count_tokens(unit) > UNIT_TOKEN_CAP and "\n" in unit:
            unit = unit.rsplit("\n", 1)[0].strip()
        if count_tokens(unit) > UNIT_TOKEN_CAP:
            # single giant paragraph: hard cap — prompt+completion must fit
            # the llama.cpp context even for dense punctuation-heavy text
            while count_tokens(unit) > 350 and len(unit) > 400:
                unit = unit[: int(len(unit) * 0.8)].rsplit(" ", 1)[0].strip()
        out.append((key, target, unit))
    return out


def cmd_translate_real(threads: int, shard: int = 0, num_shards: int = 1) -> int:
    out_path = GEN_DIR / f"real-translations.shard{shard}.jsonl"
    done_jobs = {row.get("job") for row in load_jsonl(out_path)}
    units = real_units()
    jobs: list[tuple[str, list[tuple[str, str, str]]]] = []
    for i in range(0, len(units), BATCH):
        chunk = units[i : i + BATCH]
        jid = f"tr-{i // BATCH:05d}"
        if jid in done_jobs or (i // BATCH) % num_shards != shard:
            continue
        jobs.append((jid, chunk))
    log_event(
        "translate-real-start",
        total_records=len(units),
        jobs_done=len(done_jobs),
        jobs_todo=len(jobs),
    )
    bad_rows = [r for r in load_jsonl(out_path) if r.get("status") != "ok"]
    if not jobs and not bad_rows:
        return 0  # nothing to generate and nothing to retry
    gen = Generator(threads)
    t0 = time.time()
    kept = dropped_digits = dropped_json = dropped_lang = dropped_copy = 0
    for n, (jid, chunk) in enumerate(jobs):
        prompts = [_translate_prompt(tgt, unit) for _key, tgt, unit in chunk]
        raws = gen.chat(prompts, max_new_tokens=MAX_NEW_TOKENS_TRANSLATE, seed=100000 + n)
        rows = []
        for (key, tgt, unit), raw in zip(chunk, raws):
            text = normalise(_clean_translation(raw))
            if not text:
                dropped_json += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "json-fail", "text": ""})
                continue
            lost = missing_digit_tokens(unit, text)
            if lost:
                dropped_digits += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "digit-fail", "lost": lost[:8], "text": text})
                continue
            if detect_lang(text) != tgt:
                dropped_lang += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "lang-fail", "text": text})
                continue
            if _is_copy(unit, text):
                dropped_copy += 1
                rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "copy-fail", "text": text})
                continue
            kept += 1
            rows.append({"job": jid, "key": key, "target_lang": tgt, "status": "ok", "text": text})
        append_jsonl(out_path, rows)
        if n % 10 == 0:
            rate = (n + 1) / max(1.0, time.time() - t0)
            print(
                f"translate-real job {n + 1}/{len(jobs)} rate={rate:.2f} jobs/s "
                f"eta={(len(jobs) - n - 1) / max(0.05, rate) / 60:.0f}min",
                flush=True,
            )
    log_event(
        "translate-real-pass1",
        kept=kept,
        dropped_json=dropped_json,
        dropped_digits=dropped_digits,
        dropped_lang=dropped_lang,
        dropped_copy=dropped_copy,
        wall_sec=round(time.time() - t0, 1),
    )
    return _top_up_translations(gen, out_path, shard, num_shards)


TOPUP_SYSTEM_EXTRA = (
    " NOTE: the source is mixed-language or already partly in the target language. "
    "Render ALL of its content in the target language: keep target-language parts, "
    "translate the rest. Never return the source unchanged."
)


def _top_up_translations(gen: "Generator", out_path: Path, shard: int, num_shards: int) -> int:
    """Retry pass for this shard's non-ok rows (json/lang/digit/copy)."""
    rows = load_jsonl(out_path)
    bad = [r for r in rows if r.get("status") != "ok"]
    if not bad:
        return 0
    all_units = real_units()
    shard_keys: set[str] = set()
    for ordinal in range(0, len(all_units), BATCH):
        if (ordinal // BATCH) % num_shards == shard:
            for key, _tgt, _unit in all_units[ordinal : ordinal + BATCH]:
                shard_keys.add(key)
    units = {key: (tgt, unit) for key, tgt, unit in all_units if key in shard_keys}
    redo: list[tuple[str, str, str]] = []
    seen_redo: set[str] = set()
    for r in bad:
        if r["key"] in units and r["key"] not in seen_redo:
            seen_redo.add(r["key"])
            tgt, unit = units[r["key"]]
            redo.append((r["key"], tgt, unit))
    if not redo:
        log_event("translate-real-topup-skip", shard=shard, nothing_to_redo=len(bad))
        return 0
    log_event("translate-real-topup-start", shard=shard, todo=len(redo))
    kept = still_bad = 0
    for i in range(0, len(redo), BATCH):
        chunk = redo[i : i + BATCH]
        prompts = []
        for _key, tgt, unit in chunk:
            msgs = _translate_prompt(tgt, unit)
            prompts.append([{"role": "system", "content": msgs[0]["content"] + TOPUP_SYSTEM_EXTRA}, msgs[1]])
        raws = gen.chat(prompts, max_new_tokens=MAX_NEW_TOKENS_TRANSLATE, seed=777000 + i)
        out_rows = []
        for k, ((key, tgt, unit), raw) in enumerate(zip(chunk, raws)):
            text = normalise(_clean_translation(raw)) if raw.strip() else ""
            if (
                text
                and acceptable(text)
                and not missing_digit_tokens(unit, text)
                and detect_lang(text) == tgt
                and not _is_copy(unit, text)
            ):
                kept += 1
                out_rows.append({"job": f"topup-{i:05d}", "key": key, "target_lang": tgt, "status": "ok", "text": text})
            else:
                still_bad += 1
        append_jsonl(out_path, out_rows)
    log_event("translate-real-topup-done", shard=shard, kept=kept, still_bad=still_bad)
    return 0


# ── phase D: synthetic regeneration ──────────────────────────────────────────

#: 14 round-3 families, per-family totals (probe-corpus distribution
#: normalised to S=23000). Labels = the round-3 source-label set.
SYNTH_TARGETS: dict[str, int] = {
    "notes": 5967, "chat": 3274, "commits": 2310, "tech": 1979,
    "agentchat": 1868, "config": 1290, "science": 1108, "rules": 1076,
    "docstrings": 1076, "bugs": 1006, "mixed": 590, "meeting": 533,
    "code": 464, "snippets": 459,
}
#: RU<->EN twin rows inside prose families (translate-well shapes);
#: counted TWICE toward the family total (one row per language).
#: TL verdict B (2026-10-08): mono wall at ~12000 rows; twins 1500 pairs
#: — composition decision, keeps the translated share inside [15;20].
TWIN_TARGETS: dict[str, int] = {"notes": 600, "tech": 300, "science": 225, "rules": 225, "meeting": 150}

FAMILY_SPECS: dict[str, dict[str, str]] = {
    "notes": {
        "ru": "короткая личная заметка-решение или статус: что решили/сделали, по какому проекту или модулю, с деталью (дата, версия, число). 1-3 предложения, 40-300 символов, без приветствий",
        "en": "a short personal decision/status memory note: what was decided or done, for which project or module, with a concrete detail (date, version, number). 1-3 sentences, 40-300 chars, no greetings",
    },
    "chat": {
        "ru": "фрагмент рабочего чата человека с ассистентом: реплика-вопрос и короткий ответ, рабочая тема (деплой, баг, ревью, доступы). 2-4 реплики, 60-350 символов, формат 'Человек: ... Ассистент: ...'",
        "en": "a work chat excerpt between a human and an assistant: question and short answer on a work topic (deploy, bug, review, access). 2-4 turns, 60-350 chars, format 'Human: ... Assistant: ...'",
    },
    "commits": {
        "ru": "сообщение коммита: строка 'type(scope): суть' плюс 1-2 строки тела с деталью (число файлов, имя модуля, причина). 40-250 символов",
        "en": "a commit message: 'type(scope): summary' header plus 1-2 body lines with a detail (files touched, module name, reason). 40-250 chars",
    },
    "tech": {
        "ru": "техническая заметка о версии: 'проект vX.Y: тема — факт/решение/ограничение', с числом или версией. 1-3 предложения, 60-300 символов",
        "en": "a versioned tech note: 'project vX.Y: topic — fact/decision/constraint', with a number or version. 1-3 sentences, 60-300 chars",
    },
    "agentchat": {
        "ru": "диалог агента с человеком о задаче: постановка, уточнение, договорённость о следующем шаге. 2-4 реплики, 80-400 символов, формат 'Пользователь: ... Агент: ...'",
        "en": "an agent-human task dialogue: task statement, clarification, agreed next step. 2-4 turns, 80-400 chars, format 'User: ... Agent: ...'",
    },
    "config": {
        "ru": "фрагмент конфигурации одной строкой или двумя (yaml/env/ini) с коротким комментарием-пояснением зачем. без markdown-заборов, 40-300 символов",
        "en": "a one-or-two-line config fragment (yaml/env/ini) with a short why-comment. no markdown fences, 40-300 chars",
    },
    "science": {
        "ru": "конспект научной или технической статьи: тема, метод, главный результат с числом. 1-3 предложения, 60-320 символов",
        "en": "a paper/tech article digest: topic, method, key result with a number. 1-3 sentences, 60-320 chars",
    },
    "rules": {
        "ru": "правило или конвенция команды: 'конвенция <область>: <правило и причина>'. 1-2 предложения, 40-250 символов",
        "en": "a team rule or convention: 'convention <area>: <rule and reason>'. 1-2 sentences, 40-250 chars",
    },
    "docstrings": {
        "ru": "докстринг функции или метода одной-двумя строками: что делает, аргументы, возвращает. имя функции латиницей, 40-260 символов",
        "en": "a one-or-two-line function/method docstring: what it does, args, returns. function name in latin script, 40-260 chars",
    },
    "bugs": {
        "ru": "отчёт о баге: где воспроизводится, что ожидалось vs что произошло, серьёзность. 1-3 предложения, 60-320 символов",
        "en": "a bug report: where it reproduces, expected vs actual, severity. 1-3 sentences, 60-320 chars",
    },
    "mixed": {
        "ru": "русская заметка с встроенными латинскими идентификаторами кода (имена функций, переменных, флагов) — как реальная заметка разработчика. 40-300 символов",
        "en": "an english note with inline code identifiers (function names, variables, flags) — like a real developer note. 40-300 chars",
    },
    "meeting": {
        "ru": "заметка со созвона: дата или день, участники по ролям, одно решение и один next-step. 1-3 предложения, 50-280 символов",
        "en": "a meeting note: date or weekday, participants by role, one decision and one next step. 1-3 sentences, 50-280 chars",
    },
    "code": {
        "ru": "однострочный заголовок кода с пояснением: 'ИмяКласса.метод() — что делает', или маленькая функция одной строкой. 30-200 символов",
        "en": "a one-line code headline with a gloss: 'ClassName.method() — what it does', or a tiny one-line function. 30-200 chars",
    },
    "snippets": {
        "ru": "крошечный сниппет кода одной строкой (python-подобный) с коротким пояснением применения. 30-220 символов, без markdown-заборов",
        "en": "a tiny one-line code snippet (python-like) with a short usage note. 30-220 chars, no markdown fences",
    },
}
FAMILY_RU_PROJECTS = ["аурора-апи", "волт-юи", "мнемос-ядро", "атлас-парсер", "гелиос-индекс", "орбис-шлюз", "нова-поиск", "тесса-воркер", "сифра-деплой", "минерва-пакет"]
FAMILY_EN_PROJECTS = ["aurora-api", "vault-ui", "mnemos-core", "atlas-parser", "helios-index", "orbis-gateway", "nova-search", "tessa-worker", "cipher-deploy", "minerva-pack"]

SYNTH_SYSTEM = (
    "You generate rows for a private memory-search corpus of {lang_name} work notes. "
    "Rows must sound like real personal engineer/analyst memories: concrete, dry, varied in phrasing. "
    "Vary projects, modules, topics, numbers, dates and sentence shapes across rows — never repeat a row. "
    "No markdown fences, no numbering, no quotes around rows. "
    "Answer with a strict JSON array of {n} strings and nothing else."
)

TWIN_SYSTEM = (
    "You produce translation-twin pairs for a memory-search corpus: the SAME personal work note "
    "written in Russian AND in English, same meaning, same detail, every number/date/version identical in both. "
    "Vary topics, projects and phrasing across pairs. "
    "Answer with a strict JSON array of {n} objects {{\"ru\": \"...\", \"en\": \"...\"}} and nothing else."
)


def _mono_jobs() -> list[tuple[str, str, int, int]]:
    """(family, lang, job_base_idx, n_rows) monolingual generation jobs,
    minus twin seats."""
    jobs: list[tuple[str, str, int, int]] = []
    for fam, target in SYNTH_TARGETS.items():
        t = TWIN_TARGETS.get(fam, 0)
        n_mono = target - 2 * t
        base = 0
        for lang in ("ru", "en"):
            remaining = n_mono // 2 + (1 if (lang == "ru" and n_mono % 2) else 0)
            while remaining > 0:
                n = min(BATCH, remaining)
                jobs.append((fam, lang, base, n))
                base += n
                remaining -= n
    return jobs


def _twin_jobs() -> list[tuple[str, int, int]]:
    """(family, job_idx, n_pairs) twin jobs."""
    jobs: list[tuple[str, int, int]] = []
    for fam, target in TWIN_TARGETS.items():
        remaining = target
        idx = 0
        while remaining > 0:
            n = min(TWIN_PER_JOB, remaining)
            jobs.append((fam, idx, n))
            idx += 1
            remaining -= n
    return jobs


def _stable_seed(*parts: object) -> int:
    """Process-stable seed (Python str hash is salted per process)."""
    return 2000000 + zlib.crc32("|".join(str(p) for p in parts).encode()) % 700000


def _run_mono_pass(gen: "Generator", out_path: Path, only_missing: bool, shard: int = 0, num_shards: int = 1) -> None:
    done_jobs = {row.get("job") for row in load_jsonl(out_path)}
    jobs = [j for o, j in enumerate(_mono_jobs()) if o % num_shards == shard and f"m-{j[0]}-{j[1]}-{j[2]:05d}" not in done_jobs]
    if only_missing:
        # top-up mode: FRESH jobs for the shortfall (done jobs never
        # re-run — gate-fail rows inside them are backfilled by new
        # job ids beyond the plan range)
        kept: dict[tuple[str, str], int] = {}
        for row in load_jsonl(out_path):
            if row.get("status") == "ok":
                kept[(row["family"], row["lang"])] = kept.get((row["family"], row["lang"]), 0) + 1
        max_base: dict[tuple[str, str], int] = {}
        for fam, lang, base, _n in jobs:
            max_base[(fam, lang)] = max(max_base.get((fam, lang), 0), base)
        jobs = []
        for (fam, lang), n_kept in sorted(kept.items()):
            target = SYNTH_TARGETS[fam] - 2 * TWIN_TARGETS.get(fam, 0)
            per_lang = target // 2 + (1 if (lang == "ru" and target % 2) else 0)
            shortfall = per_lang - n_kept
            base = max_base.get((fam, lang), 0) + 100000
            while shortfall > 0:
                jobs.append((fam, lang, base, min(BATCH, shortfall)))
                base += 1
                shortfall -= BATCH
    log_event("synth-mono-pass", jobs=len(jobs), mode="topup" if only_missing else "main")
    if not jobs:
        return
    t0 = time.time()
    kept = bad = 0
    for n in range(0, len(jobs), JOB_BATCH):
        chunk = jobs[n : n + JOB_BATCH]
        prompts = []
        for fam, lang, base, nrows in chunk:
            spec = FAMILY_SPECS[fam][lang]
            if lang == "ru":
                hint = f"Темы: {', '.join(FAMILY_RU_PROJECTS)} и рабочие темы (деплой, миграции, кэш, ревью, отчёты)."
                ask = f"Сгенерируй {nrows} разных строк."
            else:
                hint = f"Topics: {', '.join(FAMILY_EN_PROJECTS)} and work themes (deploys, migrations, caching, reviews, reports)."
                ask = f"Generate {nrows} distinct rows."
            prompts.append([
                {"role": "system", "content": SYNTH_SYSTEM.format(lang_name="Russian" if lang == "ru" else "English", n=nrows)},
                {"role": "user", "content": f"{spec}\n{hint}\n{ask}"},
            ])
        seed = _stable_seed("mono-batch", chunk[0][0], chunk[0][1], chunk[0][2])
        raws = gen.chat(prompts, MAX_NEW_TOKENS_SYNTH, seed)
        rows = []
        for (fam, lang, base, nrows), raw in zip(chunk, raws):
            jid = f"m-{fam}-{lang}-{base:05d}"
            arr = extract_json_array(raw) or []
            texts = [normalise(str(x)) for x in arr if isinstance(x, str)][:nrows]
            for k, txt in enumerate(texts):
                ok = bool(txt) and acceptable(txt) and detect_lang(txt) == lang
                if ok:
                    kept += 1
                    rows.append({"job": jid, "family": fam, "lang": lang, "idx": base + k, "status": "ok", "text": txt})
                else:
                    bad += 1
                    rows.append({"job": jid, "family": fam, "lang": lang, "idx": base + k, "status": "gate-fail", "text": txt})
        append_jsonl(out_path, rows)
        done_now = n + len(chunk)
        if (done_now // JOB_BATCH) % 10 == 0:
            rate = done_now / max(1.0, time.time() - t0)
            print(f"synth-mono job {done_now}/{len(jobs)} rate={rate:.2f} jobs/s eta={(len(jobs) - done_now) / max(0.05, rate) / 60:.0f}min", flush=True)
    log_event("synth-mono-pass-done", kept=kept, bad=bad, wall_sec=round(time.time() - t0, 1))


def _run_twin_pass(gen: "Generator", out_path: Path, only_missing: bool, shard: int = 0, num_shards: int = 1) -> None:
    done_jobs = {row.get("job") for row in load_jsonl(out_path)}
    jobs = [j for o, j in enumerate(_twin_jobs()) if o % num_shards == shard and f"t-{j[0]}-{j[1]:03d}" not in done_jobs]
    if only_missing:
        kept: dict[str, int] = {}
        for row in load_jsonl(out_path):
            if row.get("status") == "ok":
                kept[row["family"]] = kept.get(row["family"], 0) + 1
        max_idx = {fam: max((j[1] for j in jobs if j[0] == fam), default=0) for fam in TWIN_TARGETS}
        jobs = []
        for fam in sorted(TWIN_TARGETS):
            shortfall = TWIN_TARGETS[fam] - kept.get(fam, 0)
            idx = max_idx.get(fam, 0) + 5000
            while shortfall > 0:
                jobs.append((fam, idx, min(TWIN_PER_JOB, shortfall)))
                idx += 1
                shortfall -= TWIN_PER_JOB
    log_event("synth-twin-pass", jobs=len(jobs), mode="topup" if only_missing else "main")
    if not jobs:
        return
    t0 = time.time()
    kept = bad = 0
    for n in range(0, len(jobs), JOB_BATCH):
        chunk = jobs[n : n + JOB_BATCH]
        prompts = []
        for fam, idx, npairs in chunk:
            prompts.append([
                {"role": "system", "content": TWIN_SYSTEM.format(n=npairs)},
                {"role": "user", "content": f"Family: {FAMILY_SPECS[fam]['ru']} Topics: {', '.join(FAMILY_RU_PROJECTS[:6])}."},
            ])
        seed = _stable_seed("twin-batch", chunk[0][0], chunk[0][1])
        raws = gen.chat(prompts, MAX_NEW_TOKENS_SYNTH, seed)
        rows = []
        for (fam, idx, npairs), raw in zip(chunk, raws):
            arr = extract_json_array(raw) or []
            pairs = [p for p in arr if isinstance(p, dict) and isinstance(p.get("ru"), str) and isinstance(p.get("en"), str)][:npairs]
            for k, p in enumerate(pairs):
                ru_t, en_t = normalise(p["ru"]), normalise(p["en"])
                lost = missing_digit_tokens(ru_t, en_t) + missing_digit_tokens(en_t, ru_t)
                ok = (
                    not lost
                    and acceptable(ru_t)
                    and acceptable(en_t)
                    and detect_lang(ru_t) == "ru"
                    and detect_lang(en_t) == "en"
                )
                if ok:
                    kept += 2
                    rows.append({"job": f"t-{fam}-{idx:03d}", "family": fam, "slot": k, "status": "ok", "ru": ru_t, "en": en_t})
                else:
                    bad += 2
                    rows.append({"job": f"t-{fam}-{idx:03d}", "family": fam, "slot": k, "status": "gate-fail", "ru": ru_t, "en": en_t})
        append_jsonl(out_path, rows)
        done_now = n + len(chunk)
        if (done_now // JOB_BATCH) % 10 == 0:
            rate = done_now / max(1.0, time.time() - t0)
            print(f"synth-twin job {done_now}/{len(jobs)} rate={rate:.2f} jobs/s eta={(len(jobs) - done_now) / max(0.05, rate) / 60:.0f}min", flush=True)
    log_event("synth-twin-pass-done", kept=kept, bad=bad, wall_sec=round(time.time() - t0, 1))


def cmd_synth(threads: int, shard: int = 0, num_shards: int = 1, topup_rounds: int = 2,
              twins_only: bool = False) -> int:
    mono_path = GEN_DIR / f"synth-mono.shard{shard}.jsonl"
    twin_path = GEN_DIR / f"synth-twins.shard{shard}.jsonl"
    gen = Generator(threads, ctx=1280)  # synth prompts are small; lean KV
    if not twins_only:
        _run_mono_pass(gen, mono_path, only_missing=False, shard=shard, num_shards=num_shards)
    _run_twin_pass(gen, twin_path, only_missing=False, shard=shard, num_shards=num_shards)
    for r in range(topup_rounds):
        log_event("synth-topup-round", shard=shard, round=r + 1)
        if not twins_only:
            _run_mono_pass(gen, mono_path, only_missing=True, shard=shard, num_shards=num_shards)
        _run_twin_pass(gen, twin_path, only_missing=True, shard=shard, num_shards=num_shards)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="round-4 embedder corpus generation")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate-real")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--shard", type=int, default=0)
    t.add_argument("--num-shards", type=int, default=1)
    s = sub.add_parser("synth")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--twins-only", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "translate-real":
        return cmd_translate_real(args.threads, args.shard, args.num_shards)
    if args.cmd == "synth":
        return cmd_synth(args.threads, args.shard, args.num_shards, twins_only=args.twins_only)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
