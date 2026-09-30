#!/usr/bin/env python3
"""Generate the stage-1 synthetic pair corpus (A2s, П3) — multi-provider.

CLI wrapper — lives OUTSIDE src/cortex on purpose: src/cortex is network-free
by contract (tests/test_skeleton.py AST tripwire), and this script is the only
place where the LLM callback meets a real HTTP backend.

Backends:

    openrouter  https://openrouter.ai/api/v1   z-ai/glm-5.3-flash   (default)
    groq        https://api.groq.com/openai/v1 qwen/qwen3.8-27b
    ollama      http://127.0.0.1:11434         qwen2.5:7b-instruct  (offline)

OpenRouter/Groq are OpenAI-compatible chat/completions; records are generated
in BATCHES (default 8 per call, JSON-array contract) with malformed-JSON
retries. Repeated provider failures (429/5xx/transport/garbage output) first
back off exponentially (up to --rate-retries), then switch to the fallback
provider (--fallback-provider, default groq for openrouter) — the switch is
recorded in report.json provenance. ollama keeps the sequential native
/api/generate path. API keys are read ONLY from environment variables
(--env-var, per-provider default) and are never printed, logged or written.

Strategies and labels (ADR 0001 addendum П3, labels BY CONSTRUCTION):

    paraphrase        label 1  LLM rewrite, meaning preserved (+ light
                               procedural variations from corruption.py)
    near-topic        label 0  LLM new record, same topic, different fact
    broken-field      label 0  procedural (corruption hard negatives)
    trivial-negative  label 0  procedural (different topics; translation
                               twins refused by the topic-key guard)

Protocol-marker cleanup: some models decorate lines with
«Заголовок:/Тело:/Теги:» (and EN twins) instead of the bare title+body
contract. cortex.synth.strip_protocol_markers removes them deterministically
BEFORE parsing; every strip is counted into report.json (llm_strips) —
cleanup is counted protocol hygiene, never silent content repair.

Usage:

    # owner-validation sample (24 pairs → 8/8/4/4) via the cloud provider:
    uv run python scripts/gen_synth_corpus.py --sample 24 --out-dir data/synth-sample

    # full corpus via the cloud (defaults 300/strategy):
    uv run python scripts/gen_synth_corpus.py --pairs-per-strategy 300

    # offline, local ollama (sequential, as before the cloud backend):
    uv run python scripts/gen_synth_corpus.py --provider ollama --pairs-per-strategy 300

    # owner REVIEW sample rendered from an existing corpus:
    uv run python scripts/gen_synth_corpus.py --render-review data/synth/pairs.jsonl

Outputs (out-dir, default data/synth/ — gitignored, content NEVER committed):

    pairs.jsonl   one canonical JSON row per pair (pair_id, strategy, label,
                  record, candidate, seed)
    manifest.txt  ``pair_id <sha256>`` lines, sorted (data-contract §5 scheme)
    report.json   corpus fingerprint (BLAKE2b-256 over the manifest),
                  strategy/drop/strip counters, full provenance (provider,
                  model id, PROMPT_VERSION, seed scheme, fallback events,
                  wall time). An interrupted run writes the partial corpus
                  with ``"status": "incomplete"`` and exits 3 — never a
                  silently truncated "complete" corpus.

Determinism: procedural strategies are exactly seed-deterministic; LLM output
is pinned by temperature=0 + per-batch seed (master seed + batch index) where
the provider honors it — cloud output is reproducible only insofar as the
provider honors temperature/seed; the corpus fingerprint after any re-run is
the arbiter.

Exit codes: 0 ok · 2 usage error / missing key env · 1 backend failure ·
3 deadline reached, partial corpus written (see --deadline-min).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402
    corpus_fingerprint,
    pair_sha256,
)
from cortex.synth import (  # noqa: E402
    LLM_STRATEGIES,
    PROMPT_VERSION,
    STRATEGIES,
    SynthPair,
    SynthStats,
    TOPICS,
    corpus_fingerprint_of_pairs,
    generate_corpus,
    manifest_bytes_of_pairs,
    pair_row_json,
    sample_quotas,
)

DEFAULT_SEED = 7
DEFAULT_NUM_PREDICT = 512
DEFAULT_TIMEOUT_S = 600.0
OLLAMA_REQUEST_RETRIES = 2  # sequential ollama transport retries (unchanged)
DEFAULT_BATCH_SIZE = 8
DEFAULT_JSON_RETRIES = 3
DEFAULT_RATE_RETRIES = 5
DEFAULT_DEADLINE_MIN = 90.0
#: Per-call output budget bounds (tokens). Reasoning models (glm-5.3-flash
#: via openrouter burns ~800 reasoning tokens PER RECORD before the answer)
#: need ~1024 tokens per record; a batch of 8 gets the full 8192 cap.
MAX_TOKENS_CAP = 8192
MIN_TOKENS_BUDGET = 2048
TOKENS_PER_RECORD = 1024

PROVIDER_DEFAULTS: Final[dict[str, dict[str, str]]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "z-ai/glm-5.3-flash",
        "env_var": "OPENROUTER_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.8-27b",
        "env_var": "GROQ_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:7b-instruct",
        "env_var": "",
    },
}


class ProviderDown(RuntimeError):
    """A cloud provider failed its retry budget — pool switches or aborts."""


class DeadlineReached(RuntimeError):
    """--deadline-min exhausted mid-run; the partial corpus is the result."""


# ── ollama sequential backend (offline path, unchanged behavior) ──────────────


def ollama_generate(
    prompt: str,
    *,
    host: str,
    model: str,
    seed: int,
    num_predict: int,
    timeout: float,
) -> str:
    """One non-streaming /api/generate call; raises on transport failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def ollama_model_info(host: str, model: str, timeout: float) -> dict[str, Any] | None:
    """Model entry from /api/tags (digest, size) or None when absent."""
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as response:
        tags = json.loads(response.read().decode("utf-8"))
    for entry in tags.get("models", []):
        if entry.get("model") == model or entry.get("name") == model:
            return entry
    return None


def build_ollama_llm_fn(
    *,
    host: str,
    model: str,
    master_seed: int,
    num_predict: int,
    timeout: float,
) -> Callable[[str], str]:
    """Sequential callback: prompt → text, retries, progress lines.

    Per-call seed = master seed + call index (recorded in report.json as the
    seed scheme); transient transport errors are retried
    OLLAMA_REQUEST_RETRIES times with linear backoff, then the failure
    propagates LOUDLY — the caller must never end up with a silently
    truncated corpus.
    """
    call_index = 0

    def llm_fn(prompt: str) -> str:
        nonlocal call_index
        index, call_index = call_index, call_index + 1
        last_error: Exception | None = None
        for attempt in range(OLLAMA_REQUEST_RETRIES + 1):
            started = time.monotonic()
            try:
                text = ollama_generate(
                    prompt,
                    host=host,
                    model=model,
                    seed=master_seed + index,
                    num_predict=num_predict,
                    timeout=timeout,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < OLLAMA_REQUEST_RETRIES:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break
            elapsed = time.monotonic() - started
            print(
                f"[llm {index + 1}] {len(text)} chars in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            return text
        raise RuntimeError(
            f"ollama backend failed after {OLLAMA_REQUEST_RETRIES + 1} attempts "
            f"({host}, model {model!r}): {last_error}"
        )

    return llm_fn


# ── cloud chat/completions backend (OpenAI-compatible) ────────────────────────


class _WallDeadlineExceeded(TimeoutError):
    """Hard wall-clock cap of one HTTP attempt (watcher-thread based)."""


def _post_for_body(request: urllib.request.Request, wall_cap: float, socket_timeout: float) -> dict[str, Any]:
    """One urlopen under a WATCHER THREAD with a hard wall cap.

    urllib's socket timeout is an INACTIVITY timeout — gateways that keep
    the connection warm while the model generates defeat it (observed: an
    openrouter call hung 10+ minutes with timeout=120; a SIGALRM cap also
    failed to interrupt the ssl poll loop). ``threading.Thread.join`` has
    real timeout semantics. A timed-out worker leaks as a daemon holding
    its socket until process exit — bounded by the retry budget, fine for
    a CLI run.
    """
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            with urllib.request.urlopen(request, timeout=socket_timeout) as response:
                outcome["body"] = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # relayed verbatim to the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_run, daemon=True, name="chat-completion")
    worker.start()
    worker.join(wall_cap)
    if worker.is_alive():
        raise _WallDeadlineExceeded(f"wall cap of {wall_cap:.0f}s exceeded")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["body"]


def chat_completion(
    prompt: str,
    *,
    name: str,
    base_url: str,
    model: str,
    api_key: str,
    seed: int,
    max_tokens: int,
    timeout: float,
    rate_retries: int,
) -> dict[str, Any]:
    """One chat/completions call; returns the assistant MESSAGE dict.

    429/5xx/transport failures back off (2·2^k s, honoring Retry-After) and
    retry up to ``rate_retries`` times, then raise ProviderDown. Any other
    HTTP error (bad key, unknown model) is fatal immediately — retrying
    cannot fix it. The message dict (content, reasoning, refusal...) is
    returned raw: reasoning models may put the answer in either field.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # urllib's default UA (Python-urllib/x) is Cloudflare-banned by some
        # gateways (groq answers HTTP 403 error 1010 to it).
        "User-Agent": "vesmaro-cortex-synth/0.1",
    }
    last_error: Exception | None = None
    budget = max_tokens
    for attempt in range(rate_retries + 1):
        payload["max_tokens"] = budget
        started = time.monotonic()
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            body = _post_for_body(request, timeout, timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            if exc.fp is not None:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            if (
                exc.code == 400
                and "max_tokens" in detail.lower()
                and budget > MIN_TOKENS_BUDGET
            ):
                # this provider caps output lower than asked — halve and retry
                budget = max(budget // 2, MIN_TOKENS_BUDGET)
                last_error = ProviderDown(f"max_tokens above provider cap: {detail}")
                continue
            if exc.code == 429 or exc.code >= 500:
                last_error = ProviderDown(f"HTTP {exc.code} from {name}: {detail}")
                pause = 0.0
                try:
                    pause = min(float(exc.headers.get("Retry-After") or 0), 120.0)
                except ValueError:
                    pass
                if attempt < rate_retries:
                    time.sleep(pause if pause else 2.0 * 2**attempt)
                    continue
                break
            raise ProviderDown(
                f"HTTP {exc.code} from {name} (fatal, no retry): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = ProviderDown(
                f"transport failure ({name}) after "
                f"{time.monotonic() - started:.0f}s: {exc}"
            )
            if attempt < rate_retries:
                time.sleep(2.0 * 2**attempt)
                continue
            break
        choices = body.get("choices") or []
        if not choices:
            last_error = ProviderDown(f"empty choices from {name}")
            if attempt < rate_retries:
                time.sleep(2.0 * 2**attempt)
                continue
            break
        return choices[0].get("message") or {}
    raise ProviderDown(
        f"{name} down after {rate_retries + 1} attempts ({base_url}): {last_error}"
    )


BATCH_WRAPPER_HEAD: Final[str] = (
    "Ниже {n} независимых заданий одной и той же формы. Выполни КАЖДОЕ задание.\n"
    "Формат ответа на каждое задание указан внутри самого задания (строка 1 —\n"
    "заголовок, остальные строки — тело записи); язык ответа — язык задания.\n"
)
BATCH_TASK_SEP: Final[str] = "\n=== ЗАДАНИЕ {k} ===\n"
BATCH_WRAPPER_TAIL: Final[str] = (
    "\n\nВЕРНИ СТРОГО ОДИН JSON-МАССИВ из {n} объектов вида "
    '{{"title": "<заголовок задания>", "body": "<тело задания>"}} — по одному\n'
    "объекту на задание, В ПОРЯДКЕ заданий. Никакого текста, кроме JSON-массива:\n"
    "без markdown-заборов, без пояснений. Переводы строк внутри строк JSON\n"
    "кодируй как \\n."
)


def _message_text(value: Any) -> str:
    """Flatten a message field: plain string or typed parts list → text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") for part in value if isinstance(part, dict)
        )
    return ""


def build_batch_prompt(prompts: Sequence[str]) -> str:
    """Wrap N record prompts into one chat prompt with a JSON-array contract."""
    parts: list[str] = [BATCH_WRAPPER_HEAD.format(n=len(prompts))]
    for k, prompt in enumerate(prompts, start=1):
        parts.append(BATCH_TASK_SEP.format(k=k))
        parts.append(prompt)
    parts.append(BATCH_WRAPPER_TAIL.format(n=len(prompts)))
    return "".join(parts)


def parse_batch_json(text: str, expected: int) -> list[str]:
    """Parse the JSON-array batch answer into per-prompt raw texts.

    Uses a ``raw_decode`` scan from every ``[`` occurrence — the first array
    that parses WINS, so markdown fences and trailing model commentary
    (observed from glm-5.3-flash) are tolerated. Each item maps to
    ``"title\\nbody"`` — the same title+body protocol the sequential path
    speaks. A structurally broken array or a wrong item COUNT raises
    ValueError (the caller retries the whole batch); an item that is not an
    object with string title/body maps to "" and dies later at validation,
    counted as a drop — never silently repaired into content.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:])
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    decoder = json.JSONDecoder()
    data: Any = None
    idx = stripped.find("[")
    while idx != -1:
        try:
            data, _ = decoder.raw_decode(stripped[idx:])
            break
        except json.JSONDecodeError:
            idx = stripped.find("[", idx + 1)
    if data is None:
        raise ValueError("no JSON array in model output")
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON array")
    if len(data) != expected:
        raise ValueError(f"JSON array has {len(data)} items, expected {expected}")
    outputs: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            outputs.append("")
            continue
        title, body = item.get("title"), item.get("body")
        if isinstance(title, str) and isinstance(body, str):
            outputs.append(f"{title}\n{body}")
        else:
            outputs.append("")
    return outputs


class CloudProvider:
    """One OpenAI-compatible provider with batch/JSON retry discipline."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        stats: SynthStats,
        num_predict: int,
        batch_size: int,
        json_retries: int,
        rate_retries: int,
        timeout: float,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.stats = stats
        self.num_predict = num_predict
        self.batch_size = batch_size
        self.json_retries = json_retries
        self.rate_retries = rate_retries
        self.timeout = timeout
        self.wall_s = 0.0

    def generate_batch(self, prompts: list[str], *, seed: int) -> list[str]:
        """Prompts → raw texts; malformed JSON retries the whole batch.

        ``llm_batches`` is counted by the generator per batched-callback
        invocation (1:1 with this call); this layer owns ``llm_json_retries``.
        """
        wrapper = build_batch_prompt(prompts)
        max_tokens = min(
            MAX_TOKENS_CAP,
            max(MIN_TOKENS_BUDGET, TOKENS_PER_RECORD * len(prompts)),
        )
        last_error: Exception | None = None
        for attempt in range(self.json_retries + 1):
            started = time.monotonic()
            try:
                message = chat_completion(
                    wrapper,
                    name=self.name,
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    seed=seed + attempt,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    rate_retries=self.rate_retries,
                )
            finally:
                self.wall_s += time.monotonic() - started
            # content first (clean answer); reasoning models sometimes leave
            # it empty with the draft living in the reasoning field.
            candidates = [
                text
                for text in (
                    _message_text(message.get("content")),
                    _message_text(message.get("reasoning")),
                )
                if text
            ]
            refusal = _message_text(message.get("refusal"))
            if refusal:
                candidates.append(refusal)  # unparsable — but the snippet diagnoses
            last_error = None
            for text in candidates:
                try:
                    return parse_batch_json(text, len(prompts))
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
            self.stats.llm_json_retries += 1
            snippet = (candidates[0] if candidates else refusal or "<empty>").strip()
            print(
                f"[{self.name}] malformed batch JSON "
                f"(attempt {attempt + 1}/{self.json_retries + 1}): {last_error} "
                f"| head: {snippet[:200]!r}",
                file=sys.stderr,
                flush=True,
            )
        raise ProviderDown(
            f"{self.name}: malformed batch JSON after {self.json_retries + 1} "
            f"attempts: {last_error}"
        )


class CloudPool:
    """Primary provider + optional sticky fallback; deadline + provenance.

    A ProviderDown from the active provider switches the pool to the fallback
    (recorded in ``events``) and retries the batch there ONCE — after that
    the fallback is sticky for the rest of the run. Without a fallback the
    failure propagates (loud abort, never a silent partial corpus).
    """

    def __init__(
        self,
        primary: CloudProvider,
        fallback: CloudProvider | None,
        *,
        deadline_s: float,
        master_seed: int,
    ) -> None:
        self.active = primary
        self.fallback = fallback
        self.events: list[dict[str, Any]] = []
        self.chain: list[str] = [primary.name]
        self.deadline_s = deadline_s
        self.master_seed = master_seed
        self.started = time.monotonic()
        self.batch_index = 0

    def _next_seed(self) -> int:
        seed = self.master_seed + self.batch_index
        self.batch_index += 1
        return seed

    def _check_deadline(self) -> None:
        if self.deadline_s > 0 and time.monotonic() - self.started > self.deadline_s:
            raise DeadlineReached(
                f"deadline {self.deadline_s:.0f}s exhausted after "
                f"{self.batch_index} batch calls"
            )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self._check_deadline()
        try:
            return self.active.generate_batch(prompts, seed=self._next_seed())
        except ProviderDown as exc:
            if self.fallback is None or self.active is self.fallback:
                raise
            self.events.append(
                {
                    "from": self.active.name,
                    "to": self.fallback.name,
                    "reason": str(exc)[:200],
                    "after_batch": self.batch_index,
                }
            )
            print(
                f"[pool] {self.active.name} down → switching to "
                f"{self.fallback.name} (sticky)",
                file=sys.stderr,
                flush=True,
            )
            self.active = self.fallback
            self.chain.append(self.fallback.name)
            self._check_deadline()
            return self.active.generate_batch(prompts, seed=self._next_seed())


# ── owner REVIEW renderer (human validation sample from an existing corpus) ───

LABEL_RU: Final[dict[str, str]] = {
    "duplicate": "дубликат",
    "not-duplicate": "НЕ дубликат",
}


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Recompute the BLAKE2b-256 corpus fingerprint from pairs.jsonl rows."""
    entries = [
        (
            row["pair_id"],
            pair_sha256(
                {
                    "record": row["record"],
                    "candidate": row["candidate"],
                    "label": row["label"],
                }
            ),
        )
        for row in rows
    ]
    lines = [f"{pair_id} {digest}" for pair_id, digest in sorted(entries)]
    return corpus_fingerprint(("\n".join(lines) + "\n").encode("utf-8"))


def render_review(pairs_path: Path, out_path: Path, n: int) -> None:
    """Render the owner-validation sample: n pairs spread across strategies.

    Quotas reuse :func:`cortex.synth.sample_quotas` (n=30 → 10/10/5/5);
    within a strategy the picks are evenly strided over corpus order — a
    deterministic walk across the whole strategy range, not its first n.
    """
    if n <= 0:
        raise ValueError(f"review sample must be positive, got {n}")
    rows = [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quotas = sample_quotas(n)
    picked: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        want = quotas[strategy]
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        take = min(want, len(group))
        picked.extend(group[(j * len(group)) // take] for j in range(take))

    fingerprint = _rows_fingerprint(rows)
    header = [
        "# Сэмпл синтетического корпуса — валидация владельцем",
        f"{len(picked)} пар из корпуса {pairs_path} "
        f"(фингерпринт {fingerprint[:12]}…); квоты: "
        + ", ".join(f"{s}={quotas[s]}" for s in STRATEGIES if quotas[s])
        + ".",
        "Вопрос по кодбуку: являются ли две записи ОДНОЙ памятью "
        "(слияние не теряет информации)?",
        "Отметьте: OK — метка верна; либо номер пары + верная метка.",
        "",
    ]
    body: list[str] = []
    for k, row in enumerate(picked, start=1):
        record, candidate = row["record"], row["candidate"]
        body.append("---")
        body.append(
            f"## Пара {k} [{row['strategy']}] — метка: {LABEL_RU[row['label']]}"
        )
        body.append(f"**A. {record['title']}**")
        body.append(record["body"])
        body.append(f"*теги: {', '.join(record['tags'])}*")
        body.append("")
        body.append(f"**B. {candidate['title']}**")
        body.append(candidate["body"])
        body.append(f"*теги: {', '.join(candidate['tags'])}*")
        body.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    print(
        f"[review] {len(picked)} pairs from {pairs_path} → {out_path} "
        f"(fingerprint {fingerprint[:12]}…)",
        file=sys.stderr,
        flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the A2s synthetic pair corpus via an OpenAI-compatible "
            "cloud provider (default openrouter) or local ollama."
        ),
    )
    sampling = parser.add_mutually_exclusive_group(required=True)
    sampling.add_argument(
        "--pairs-per-strategy",
        type=int,
        metavar="N",
        help="uniform quota per strategy (full run; 300 → ~1200 pairs)",
    )
    sampling.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="owner-validation sample of N pairs (quotas: ceil(N/3) per LLM "
             "strategy, rest split procedurally; N=24 → 8/8/4/4)",
    )
    sampling.add_argument(
        "--render-review",
        metavar="PAIRS_JSONL",
        help="no generation: render a human-readable owner REVIEW sample "
             "from an existing corpus (see --review-n/--review-out)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_DEFAULTS),
        default="openrouter",
        help="LLM backend (default openrouter; ollama = offline sequential)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: provider default — openrouter "
             "z-ai/glm-5.3-flash, groq qwen/qwen3.8-27b, ollama "
             "qwen2.5:7b-instruct)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="base URL override (default: provider default endpoint)",
    )
    parser.add_argument(
        "--env-var",
        default=None,
        help="name of the env var holding the API key (default: provider "
             "default, e.g. OPENROUTER_API_KEY); the VALUE is never printed",
    )
    parser.add_argument(
        "--fallback-provider",
        choices=[*sorted(PROVIDER_DEFAULTS), "none"],
        default=None,
        help="sticky fallback after repeated provider failures "
             "(default: groq for openrouter, none otherwise)",
    )
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--json-retries", type=int, default=DEFAULT_JSON_RETRIES)
    parser.add_argument("--rate-retries", type=int, default=DEFAULT_RATE_RETRIES)
    parser.add_argument(
        "--deadline-min",
        type=float,
        default=DEFAULT_DEADLINE_MIN,
        help="stop with a marked partial corpus after this many minutes "
             "(0 disables; default 90)",
    )
    parser.add_argument("--review-n", type=int, default=30)
    parser.add_argument(
        "--review-out",
        default=None,
        help="REVIEW output path (default data/synth-sample/REVIEW.md)",
    )
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-attempt cap (seconds): socket inactivity AND a hard "
             "wall-clock cap via SIGALRM (cloud gateways keep the socket "
             "warm while the model generates)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default data/synth/; use data/synth-sample/ "
             "for the owner-validation sample)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing corpus in --out-dir (default: refuse)",
    )
    return parser


def _resolve_fallback(args: argparse.Namespace) -> tuple[str, str] | None:
    """(provider, model) of the fallback, or None when disabled/unresolvable.

    A missing fallback KEY is a loud stderr warning + run without the
    fallback — the fallback is resilience, not a precondition.
    """
    name = args.fallback_provider
    if name is None:
        name = "groq" if args.provider == "openrouter" else "none"
    if name == "none" or name == args.provider:
        return None
    defaults = PROVIDER_DEFAULTS[name]
    if not defaults["env_var"]:
        # ollama as a cloud-run fallback makes no sense: no key scheme, and a
        # local daemon is not guaranteed on the generating machine.
        print(
            f"[fallback] {name} has no key scheme — fallback disabled",
            file=sys.stderr,
        )
        return None
    api_key = os.environ.get(defaults["env_var"])
    if not api_key:
        print(
            f"[fallback] env {defaults['env_var']} is empty — running WITHOUT "
            f"fallback ({name}); set it to enable the switch",
            file=sys.stderr,
        )
        return None
    model = args.fallback_model or defaults["model"]
    return name, model  # type: ignore[return-value]


def _make_cloud_fn(
    args: argparse.Namespace,
    stats: SynthStats,
) -> tuple[Callable[[list[str]], list[str]], dict[str, Any]]:
    """Build the batched callback + its provenance fragment (no secrets)."""
    primary_cfg = PROVIDER_DEFAULTS[args.provider]
    primary_env = args.env_var or primary_cfg["env_var"]
    api_key = os.environ.get(primary_env)
    if not api_key:
        print(
            f"env {primary_env} is not set — export it (the value is never "
            "printed or logged) or pass --provider ollama / --env-var",
            file=sys.stderr,
        )
        raise SystemExit(2)
    primary = CloudProvider(
        name=args.provider,
        base_url=args.host or primary_cfg["base_url"],
        model=args.model or primary_cfg["model"],
        api_key=api_key,
        stats=stats,
        num_predict=args.num_predict,
        batch_size=args.batch_size,
        json_retries=args.json_retries,
        rate_retries=args.rate_retries,
        timeout=args.timeout,
    )
    fallback: CloudProvider | None = None
    fb = _resolve_fallback(args)
    if fb is not None:
        fb_name, fb_model = fb
        fb_cfg = PROVIDER_DEFAULTS[fb_name]
        fallback = CloudProvider(
            name=fb_name,
            base_url=fb_cfg["base_url"],
            model=fb_model,
            api_key=os.environ[fb_cfg["env_var"]],
            stats=stats,
            num_predict=args.num_predict,
            batch_size=args.batch_size,
            json_retries=args.json_retries,
            rate_retries=args.rate_retries,
            timeout=args.timeout,
        )
    pool = CloudPool(
        primary,
        fallback,
        deadline_s=args.deadline_min * 60.0,
        master_seed=args.seed,
    )
    provenance = {
        "provider": primary.name,
        "base_url": primary.base_url,
        "model": primary.model,
        "api_key_source": primary_env,
        "prompt_version": PROMPT_VERSION,
        "fallback_provider": fallback.name if fallback else "none",
        "fallback_model": fallback.model if fallback else None,
        "providers_used": pool.chain,
        "fallback_events": pool.events,
        "options": {
            "temperature": 0,
            "seed_scheme": "master seed + batch index",
            "master_seed": args.seed,
            "num_predict": args.num_predict,
            "batch_size": args.batch_size,
            "max_tokens_per_call": min(
                MAX_TOKENS_CAP,
                max(MIN_TOKENS_BUDGET, TOKENS_PER_RECORD * args.batch_size),
            ),
        },
    }
    # live provenance objects (mutated during the run):
    provenance["_pool"] = pool
    provenance["_providers"] = [p for p in (primary, fallback) if p]
    return pool.generate_batch, provenance


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.render_review is not None:
        if args.render_review.strip() == "":
            parser.error("--render-review needs a pairs.jsonl path")
        if args.batch_size < 1 or args.json_retries < 0 or args.rate_retries < 0:
            parser.error("--batch-size/--json-retries/--rate-retries must be >= 1/0/0")
        review_out = (
            Path(args.review_out)
            if args.review_out
            else REPO_ROOT / "data" / "synth-sample" / "REVIEW.md"
        )
        render_review(Path(args.render_review), review_out, args.review_n)
        return 0

    if args.sample is not None:
        if args.sample <= 0:
            parser.error("--sample must be a positive integer")
        quotas = sample_quotas(args.sample)
        pairs_per_strategy = 0  # base cycling aligns topics across strategies
        mode: dict[str, Any] = {"mode": "sample", "sample_n": args.sample}
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "synth-sample"
    else:
        if args.pairs_per_strategy <= 0:
            parser.error("--pairs-per-strategy must be a positive integer")
        quotas = {strategy: args.pairs_per_strategy for strategy in STRATEGIES}
        pairs_per_strategy = args.pairs_per_strategy
        mode = {"mode": "full", "pairs_per_strategy": args.pairs_per_strategy}
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "synth"
    if args.batch_size < 1 or args.json_retries < 0 or args.rate_retries < 0:
        parser.error("--batch-size/--json-retries/--rate-retries must be >= 1/0/0")
    if args.deadline_min < 0:
        parser.error("--deadline-min must be >= 0")

    pairs_file = out_dir / "pairs.jsonl"
    if pairs_file.exists() and not args.force:
        print(
            f"refusing to overwrite existing corpus {pairs_file} (use --force "
            "or another --out-dir; a fingerprinted corpus is never clobbered)",
            file=sys.stderr,
        )
        return 1

    need_llm = any(quotas[s] > 0 for s in LLM_STRATEGIES)
    stats = SynthStats()
    llm_fn: Callable[[str], str] | None = None
    llm_batch_fn: Callable[[list[str]], list[str]] | None = None
    cloud_provenance: dict[str, Any] | None = None

    if need_llm and args.provider == "ollama":
        host = args.host or PROVIDER_DEFAULTS["ollama"]["base_url"]
        model = args.model or PROVIDER_DEFAULTS["ollama"]["model"]
        info = ollama_model_info(host, model, args.timeout)
        if info is None:
            print(
                f"model {model!r} not found on {host} — pull it first "
                "(ollama pull) or pass --model; aborting before any generation",
                file=sys.stderr,
            )
            return 1
        cloud_provenance = {
            "backend": "ollama",
            "provider": "ollama",
            "host": host,
            "model": model,
            "model_digest": info.get("digest"),
            "model_size_bytes": info.get("size"),
            "prompt_version": PROMPT_VERSION,
            "fallback_provider": "none",
            "fallback_events": [],
            "options": {
                "temperature": 0,
                "seed_scheme": "master seed + call index",
                "master_seed": args.seed,
                "num_predict": args.num_predict,
            },
        }
        llm_fn = build_ollama_llm_fn(
            host=host,
            model=model,
            master_seed=args.seed,
            num_predict=args.num_predict,
            timeout=args.timeout,
        )
    elif need_llm:
        llm_batch_fn, cloud_provenance = _make_cloud_fn(args, stats)

    started = time.monotonic()
    cache: list[SynthPair] = []
    journal: Path | None = None
    journal_fh = None
    if llm_batch_fn is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        journal = out_dir / "pairs.partial.jsonl"
        journal_fh = journal.open("w", encoding="utf-8")

        def on_emit(pair: SynthPair) -> None:
            cache.append(pair)
            journal_fh.write(pair_row_json(pair) + "\n")  # type: ignore[union-attr]

    status = "complete"
    deadline_error: str | None = None
    pairs: list[SynthPair] = []
    try:
        pairs = generate_corpus(
            TOPICS,
            pairs_per_strategy,
            llm_fn,
            args.seed,
            quotas=quotas,
            stats=stats,
            llm_batch_fn=llm_batch_fn,
            batch_size=args.batch_size,
            emit_callback=on_emit if journal_fh is not None else None,
        )
    except DeadlineReached as exc:
        status = "incomplete"
        deadline_error = str(exc)
        pairs = list(cache)
        print(
            f"[deadline] {deadline_error} — writing the marked PARTIAL corpus "
            f"({len(pairs)} pairs) and exiting 3",
            file=sys.stderr,
            flush=True,
        )
    finally:
        if journal_fh is not None:
            journal_fh.close()

    elapsed = time.monotonic() - started

    fingerprint = corpus_fingerprint_of_pairs(pairs)
    by_strategy = {strategy: 0 for strategy in STRATEGIES}
    by_label = {"duplicate": 0, "not-duplicate": 0}
    for pair in pairs:
        by_strategy[pair.strategy] += 1
        by_label["duplicate" if pair.label == 1 else "not-duplicate"] += 1
    remaining = {s: quotas[s] - by_strategy[s] for s in STRATEGIES}

    counters = stats.as_dict()
    counters["by_strategy"] = by_strategy
    counters["by_label"] = by_label
    pool = cloud_provenance.pop("_pool", None) if cloud_provenance else None
    providers = cloud_provenance.pop("_providers", []) if cloud_provenance else []
    report: dict[str, Any] = {
        "kind": "synth-corpus-report",
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "cortex.synth.generate",
        **mode,
        "seed": args.seed,
        "quotas": quotas,
        "provenance": cloud_provenance,
        "counters": counters,
        "corpus_fingerprint": fingerprint,
        "elapsed_s": round(elapsed, 1),
    }
    if status == "incomplete":
        report["incomplete"] = {
            "reason": deadline_error,
            "remaining_by_strategy": remaining,
        }

    if pairs:
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_file.write_text(
            "\n".join(pair_row_json(pair) for pair in pairs) + "\n", encoding="utf-8"
        )
        (out_dir / "manifest.txt").write_bytes(manifest_bytes_of_pairs(pairs))
    report["files"] = {
        "pairs": pairs_file.name if pairs else None,
        "manifest": "manifest.txt" if pairs else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if journal is not None:
        journal.unlink(missing_ok=True)

    provider_timings = {
        p.name: round(p.wall_s, 1) for p in providers if isinstance(p, CloudProvider)
    }
    print(
        json.dumps(
            {
                "status": status,
                **report["counters"],
                "corpus_fingerprint": fingerprint,
                "out_dir": str(out_dir),
                "elapsed_s": report["elapsed_s"],
                "providers_used": (pool.chain if pool else None),
                "provider_wall_s": provider_timings or None,
                "fallback_events": len(report["provenance"]["fallback_events"])
                if report["provenance"]
                else 0,
            },
            ensure_ascii=False,
        )
    )
    return 3 if status == "incomplete" else 0


if __name__ == "__main__":
    sys.exit(main())
