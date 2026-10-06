#!/usr/bin/env python3
"""Assemble the REAL part of the round-4 corpus (addendum 6 §A.3).

Sources (all READ-ONLY, ``file:…?mode=ro`` + ``PRAGMA query_only=ON``):
the live store plus the 2026-09-2x backup snapshots — the domain slice
only (memory_type note/snippet/fact), deduplicated by content hash.

Dedup uses the ENGINE's content-hash convention, not a new one:
``vesmaro.ccr.content_hash`` = sha256(hex) of the UTF-8 text (the
single-source rule — found, not reinvented). The corpus-level manifest
fingerprint then goes through the repo's
:mod:`cortex.data.fingerprints` scheme (blake2b-256 over sorted
``pair_id sha256`` lines, data-contract §5).

Privacy (addendum 6 §A.3, owner sanction 2026-10-06):
- output rows land ONLY in a local gitignored jsonl
  (``datasets/v43/real-part.jsonl`` — ``/datasets/*`` is blocked in
  .gitignore and ``datasets/v43`` gets NO negation);
- stdout/README carry counters and distributions ONLY — never record
  text, titles or ids;
- a secrets pass (engine scanner when available, stdlib fallback
  otherwise) excludes records with reason names only — the same hygiene
  chain as the A2 exporter (store_export.py, data-contract §6 order:
  selection → per-record hygiene → composition → fingerprints).

Defaults per the assembly brief:
- ``--exclude-tag-prefix task:,telemetry,mnemos:`` (any tag starting
  with one of the prefixes excludes the record);
- ``--min-content-chars 120``;
- status filter: ``published`` (the prod stores) plus ``raw`` — the
  0920 snapshot's 22 prod records are status ``raw`` (verified against
  the store: 0920 carries 22 rows, all raw, none privacy-tagged), the
  addendum inventarisation counted them for the live store of 06.10.
- memory_type gate: note/snippet/fact — session_context/conversation
  are live instrumentation (не доменные), excluded.

NOTE (v4.3 assembly brief vs ratified addendum, resolved in favor of the
ratified doc): the brief's default exclude list includes ``mnemos:``,
but on the live store the ``mnemos:*`` data-contract tags ride almost
every DOMAIN memory (3,324 of 3,647 published) — applying it yields a
34-row corpus, which contradicts addendum 6 §A.3 («реальная часть — весь
доступный доменный срез … ожидаемый потолок ≤2.7k», verified union
2,605). The DEFAULT here excludes only ``task:``/``telemetry``; pass
``--exclude-tag-prefix mnemos:`` to reproduce the literal-brief 34-row
cut. The README records both counter sets; the TL gate picks the
corpus-feeding cut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.data.fingerprints import (  # noqa: E402 — sys.path shim above
    corpus_fingerprint,
    manifest_bytes,
)

#: Engine source for the secrets scanner (read-only reuse, ADR 0001 V2
#: hygiene; the store_export fallback keeps the build running when the
#: tree is absent — provenance is then disclosed LOUDLY).
DEFAULT_ENGINE_SRC = Path("/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src")

#: The four ratified sources (addendum 6 §A.3 table + the live store).
DEFAULT_SOURCES: tuple[tuple[str, Path], ...] = (
    ("live", Path("/var/home/abyss/.distrobox/vscode-box/home/.mnemos/data/mnemos.db")),
    (
        "backup-20260920",
        Path("/var/home/abyss/backups/mnemos-laptop-prod-20260920/mnemos.db"),
    ),
    (
        "backup-20260921",
        Path("/var/home/abyss/backups/mnemos-laptop-prod-20260921/mnemos.db"),
    ),
    (
        "backup-20260924",
        Path("/var/home/abyss/backups/mnemos-laptop-prod-20260924/mnemos.db"),
    ),
)

#: Domain slice: session_context/conversation are live instrumentation.
DOMAIN_MEMORY_TYPES: tuple[str, ...] = ("note", "snippet", "fact")
DOMAIN_STATUS: tuple[str, ...] = ("published", "raw")

#: Brief defaults. ``mnemos:`` deliberately NOT in the default list (see
#: the module docstring NOTE) — addendum 6 §A.3 wins over the brief.
DEFAULT_EXCLUDE_TAG_PREFIXES: tuple[str, ...] = ("task:", "telemetry")
BRIEF_EXCLUDE_TAG_PREFIXES: tuple[str, ...] = ("task:", "telemetry", "mnemos:")
DEFAULT_MIN_CONTENT_CHARS: int = 120

OUTPUT_JSONL = REPO_ROOT / "datasets" / "v43" / "real-part.jsonl"
OUTPUT_README = REPO_ROOT / "datasets" / "v43" / "real-part.README.md"

#: Engine fallback scanner import path — stdlib-only module, loaded only
#: when the engine tree is missing (store_export.FallbackScanner needs
#: no dependency but lives in src/cortex/data/store_export.py which
#: imports numpy; the local pattern set here avoids that import chain).
_FALLBACK_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    (name, pattern)
    for name, pattern in (
        ("aws-key", r"AKIA[0-9A-Z]{16}"),
        ("github-token", r"gh[pousr]_[A-Za-z0-9]{36,}"),
        ("gitlab-token", r"glpat-[A-Za-z0-9_\-]{20,}"),
        ("slack-token", r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
        ("openai-key", r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
        ("anthropic-key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
        ("google-api-key", r"AIza[0-9A-Za-z_\-]{35}"),
        ("telegram-bot-token", r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b"),
        ("private-key-block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        (
            "jwt",
            r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
        ),
        (
            "generic-secret-assignment",
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}",
        ),
    )
)

_FALLBACK_COMPILED = tuple(
    (name, re.compile(p, re.IGNORECASE)) for name, p in _FALLBACK_PATTERNS
)


def _fallback_reasons(title: str, body: str, tags: tuple[str, ...]) -> tuple[str, ...]:
    text = "\n".join([title or "", body or "", " ".join(tags or ())])
    return tuple(
        f"secret:{name}" for name, pat in _FALLBACK_COMPILED if pat.search(text)
    )


# ── content hash: the ENGINE convention (ccr.content_hash = sha256 hex) ──────


def content_hash(text: str) -> str:
    """sha256(hex) of the UTF-8 text — the engine's ``ccr.content_hash``
    convention, reimplemented here as a pinned literal (the engine tree
    is a foreign checkout; importing it here would violate the repo's
    network-free src/ discipline — scripts import it lazily instead)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cyrillic_share(text: str) -> float:
    """Share of Cyrillic characters in a non-empty text (0.0 for empty)."""
    if not text:
        return 0.0
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    return cyr / len(text)


def _ru_share_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    """RU-centred counter for the README: a record is RU-centred when its
    Cyrillic share ≥ 0.2 (the addendum-7 fact mixes on Cyrillic CONTENT:
    «записей с кириллическим контентом»). Counts only."""
    ru = sum(1 for r in rows if _cyrillic_share(r["side"]["body"]) >= 0.2)
    return {"ru_centric": ru, "not_ru_centric": len(rows) - ru}


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open one store database strictly read-only (mode=ro + query_only)."""
    if not db_path.is_file():
        raise FileNotFoundError(f"store database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _scan_reasons(engine_src: Path | None) -> tuple[Any, str]:
    """Scanner factory: engine detectors when the tree loads, else the
    local fallback. Returns (scan_fn, provenance)."""
    if (
        engine_src is not None
        and (engine_src / "vesmaro" / "secrets_detector.py").is_file()
    ):
        if str(engine_src) not in sys.path:
            sys.path.insert(0, str(engine_src))
        try:
            from vesmaro.secrets_detector import detect_secrets, findings_by_pattern

            def scan(title: str, body: str, tags: tuple[str, ...]) -> tuple[str, ...]:
                return tuple(
                    findings_by_pattern(
                        detect_secrets("\n".join([title or "", body or ""]))
                    )
                )

            return scan, "engine"
        except ImportError:
            pass
    return _fallback_reasons, "fallback scanner, not engine"


SQL_CANDIDATES = (
    "SELECT id, title, content, tags, memory_type, created_at "
    "FROM memories "
    "WHERE status IN ('published','raw') "
    "AND memory_type IN ('note','snippet','fact') "
    "AND length(content) >= :min_chars"
)


def collect_source(
    name: str,
    db_path: Path,
    *,
    min_chars: int,
    exclude_prefixes: tuple[str, ...],
    scan,
    counters: Counter,
) -> list[dict[str, Any]]:
    """One source's hygiene pass → rows (id-free composition + provenance).

    Order mirrors data-contract §6: SQL selection → per-record hygiene
    (privacy tag prefixes, secrets) → composition. Exclusions count by
    reason only; store ids NEVER leave this function.
    """
    conn = _open_ro(db_path)
    try:
        counters[f"candidates_sql:{name}"] = conn.execute(
            "SELECT count(*) FROM memories WHERE status IN ('published','raw') "
            "AND memory_type IN ('note','snippet','fact') "
            "AND length(content) >= :min_chars",
            {"min_chars": min_chars},
        ).fetchone()[0]
        rows: list[dict[str, Any]] = []
        for (
            record_id,
            title,
            content,
            tags_json,
            memory_type,
            created_at,
        ) in conn.execute(SQL_CANDIDATES, {"min_chars": min_chars}):
            tags = _parse_tags(tags_json)
            if any(
                isinstance(t, str) and any(t.startswith(p) for p in exclude_prefixes)
                for t in tags
            ):
                counters[f"excluded:privacy-tag:{name}"] += 1
                continue
            if reasons := scan(title or "", content or "", tags):
                for reason in reasons:
                    counters[f"excluded:{reason}:{name}"] += 1
                continue
            rows.append(
                {
                    "content_hash": content_hash(content or ""),
                    "title_len": len(title or ""),
                    "content_len": len(content or ""),
                    "memory_type": memory_type,
                    "created_at": created_at,
                    "cyrillic_share": round(_cyrillic_share(content or ""), 4),
                    "source": name,
                    # side composition (the minimal export shape); the id
                    # stays behind (privacy: ids never leave this module).
                    # language fills in later from the canon envelope
                    # (_attach_languages — metadata is not in this SELECT).
                    "side": {
                        "title": title or "",
                        "body": content or "",
                        "tags": tags,
                        "language": None,
                        "record_type": memory_type,
                    },
                }
            )
    finally:
        conn.close()
    return rows


def _parse_tags(tags_json: Any) -> tuple[str, ...]:
    """Tags list of a row (defensive: malformed JSON → empty tuple)."""
    try:
        raw = json.loads(tags_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(t) for t in raw if isinstance(t, str))


def _canon_language(metadata_json: Any) -> str | None:
    """canon.language from metadata (the engine's envelope projection)."""
    try:
        meta = json.loads(metadata_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    canon = meta.get("canon") if isinstance(meta.get("canon"), dict) else {}
    lang = canon.get("language")
    return lang if isinstance(lang, str) else None


def assemble(
    sources: tuple[tuple[str, Path], ...],
    *,
    min_chars: int,
    exclude_prefixes: tuple[str, ...],
    engine_src: Path | None,
) -> tuple[list[dict[str, Any]], Counter, str, str]:
    """The full pass over all sources. Returns (deduped rows newest-side,
    counters, manifest fingerprint, scanner provenance).

    Dedup discipline: first source wins (`live` first — the actual store
    is the authority; backups only fill what it no longer holds), order
    inside a source deterministic (created_at, id). The deduped pool is
    sorted by content_hash for a byte-stable fingerprint.
    """
    scan, provenance = _scan_reasons(engine_src)
    counters: Counter = Counter()

    all_rows: list[dict[str, Any]] = []
    for name, db_path in sources:
        rows = collect_source(
            name,
            db_path,
            min_chars=min_chars,
            exclude_prefixes=exclude_prefixes,
            scan=scan,
            counters=counters,
        )
        counters[f"accepted:{name}"] = len(rows)
        all_rows.extend(rows)

    # dedup by content_hash — the addendum 6 §A.3 rule. First-seen wins
    # (source order: live first), provenance keeps ALL first-seen sources.
    by_hash: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        by_hash.setdefault(row["content_hash"], row)

    deduped = sorted(by_hash.values(), key=lambda r: r["content_hash"])
    counters["dedup:unique_content_hashes"] = len(by_hash)
    counters["dedup:dropped_duplicates"] = len(all_rows) - len(by_hash)

    # canon.language enrichment (envelope metadata only) — one extra RO
    # open per source, ids matched in-memory.
    _attach_languages(sources, by_hash, counters)

    # manifest fingerprint: content_hash lines (the real-part manifest,
    # §5 scheme — sorted pair_id <hash> rows where pair_id = content_hash).
    entries = [(r["content_hash"], r["content_hash"]) for r in deduped]
    fp = corpus_fingerprint(manifest_bytes(entries))
    return deduped, counters, fp, provenance


def _attach_languages(
    sources: tuple[tuple[str, Path], ...],
    by_hash: dict[str, dict[str, Any]],
    counters: Counter,
) -> None:
    """Fill ``side.language`` from each source's canon envelope (RO).

    The composition step left ``language`` unset (collect_source does not
    SELECT metadata); languages ride the metadata envelope — canon.language
    was verified 'ru' on 583 live rows. One extra read-only open per
    source; matching by hash of content (not id — ids never cross).
    """
    for name, db_path in sources:
        conn = _open_ro(db_path)
        try:
            for metadata_json, content in conn.execute(
                "SELECT metadata, content FROM memories WHERE status IN "
                "('published','raw') AND memory_type IN ('note','snippet','fact')"
            ):
                h = content_hash(content or "")
                row = by_hash.get(h)
                lang = _canon_language(metadata_json) if row is not None else None
                if (
                    row is not None
                    and row["source"] == name
                    and row["side"]["language"] is None
                    and lang is not None
                ):
                    row["side"]["language"] = lang
                    counters[f"language_attached:{name}"] += 1
        finally:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=OUTPUT_JSONL,
        help=f"local gitignored output (default {OUTPUT_JSONL})",
    )
    ap.add_argument(
        "--readme",
        type=Path,
        default=OUTPUT_README,
        help=f"repo README with counters only (default {OUTPUT_README})",
    )
    ap.add_argument(
        "--exclude-tag-prefix",
        action="append",
        default=None,
        help="exclude records having any tag with this prefix; repeatable. "
        f"Default: {','.join(DEFAULT_EXCLUDE_TAG_PREFIXES)} (the literal "
        "assembly-brief list incl. mnemos: is available via "
        "--brief-tag-filter)",
    )
    ap.add_argument(
        "--brief-tag-filter",
        action="store_true",
        help=f"use the brief's literal default prefixes "
        f"({','.join(BRIEF_EXCLUDE_TAG_PREFIXES)}) — produces the 34-row "
        "cut documented in the README; the ratified addendum §A.3 slice "
        "is the DEFAULT",
    )
    ap.add_argument(
        "--min-content-chars",
        type=int,
        default=DEFAULT_MIN_CONTENT_CHARS,
        help=f"min content length (default {DEFAULT_MIN_CONTENT_CHARS})",
    )
    ap.add_argument(
        "--engine-src",
        type=Path,
        default=DEFAULT_ENGINE_SRC,
        help="engine source tree for the secrets scanner (fallback otherwise)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="counters + fingerprint only, write nothing",
    )
    args = ap.parse_args(argv)

    exclude_prefixes = (
        BRIEF_EXCLUDE_TAG_PREFIXES
        if args.brief_tag_filter
        else (
            tuple(args.exclude_tag_prefix)
            if args.exclude_tag_prefix
            else DEFAULT_EXCLUDE_TAG_PREFIXES
        )
    )

    started = time.monotonic()
    rows, counters, fp, provenance = assemble(
        DEFAULT_SOURCES,
        min_chars=args.min_content_chars,
        exclude_prefixes=exclude_prefixes,
        engine_src=args.engine_src,
    )

    report = {
        "kind": "dataset-v43-real-part-report",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [name for name, _ in DEFAULT_SOURCES],
        "exclude_tag_prefixes": list(exclude_prefixes),
        "min_content_chars": args.min_content_chars,
        "scanner_provenance": provenance,
        "real_part_fingerprint": fp,
        "counters": dict(sorted(counters.items())),
        "ru_share": _ru_share_counter(rows),
        "memory_types": dict(Counter(r["memory_type"] for r in rows)),
        "by_source_first_seen": dict(Counter(r["source"] for r in rows)),
        "elapsed_sec": round(time.monotonic() - started, 2),
        "privacy": "no record text, titles or ids in the report; rows land "
        "only in the gitignore-protected local jsonl",
    }

    if not args.dry_run:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(
                    json.dumps(
                        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
        args.readme.parent.mkdir(parents=True, exist_ok=True)
        args.readme.write_text(_readme_text(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def _readme_text(report: dict[str, Any]) -> str:
    """README body: numbers only (counters, distributions, schema notes)."""
    counters = report["counters"]
    ru = report["ru_share"]
    lines = [
        "# datasets/v43/real-part — ASSEMBLY NUMBERS (no content, sanction 2026-10-06)",
        "",
        f"Generated: {report['built_at']} by `scripts/assemble_real_part.py`.",
        "",
        "## Sources (read-only, deduped by content hash — addendum 6 §A.3)",
        "",
        "| source | SQL candidates | accepted |",
        "|---|---:|---:|",
    ]
    for name, _ in DEFAULT_SOURCES:
        cand = counters.get(f"candidates_sql:{name}", 0)
        acc = counters.get(f"accepted:{name}", 0)
        lines.append(f"| {name} | {cand} | {acc} |")
    lines += [
        "",
        f"- unique content hashes after dedup: **{counters.get('dedup:unique_content_hashes', 0)}**",
        f"- duplicate rows dropped: {counters.get('dedup:dropped_duplicates', 0)}",
        f"- RU-centred share (Cyrillic ≥ 0.2 of content): {ru['ru_centric']} "
        f"({ru['ru_centric'] / max(1, ru['ru_centric'] + ru['not_ru_centric']):.1%}), "
        f"non-RU {ru['not_ru_centric']} (addendum-7 reference ~69%)",
        f"- memory types: `{report['memory_types']}`",
        f"- scanner provenance: **{report['scanner_provenance']}**",
        f"- exclusion counters: `{ {k: v for k, v in counters.items() if k.startswith('excluded:')} }`",
        f"- fingerprint (blake2b-256 of the sorted content_hash manifest): "
        f"`{report['real_part_fingerprint']}`",
        "",
        "## Dedup / fingerprint logic (data-contract §5 reuse)",
        "",
        "- content hash = sha256(hex) over the record content — the engine's",
        "  `vesmaro.ccr.content_hash` convention (single-source rule).",
        "- manifest = sorted `content_hash content_hash` lines,",
        "  corpus fingerprint = blake2b-256 (`cortex.data.fingerprints`).",
        "- first-seen wins (live store first; backups fill what it no",
        "  longer holds); provenance keeps every first-seen source.",
        "",
        "## Privacy",
        "",
        "- record text lives ONLY in `datasets/v43/real-part.jsonl` —",
        "  gitignored (`/datasets/*` block, no negation for v43/).",
        "- every store opened via `file:…?mode=ro` + `PRAGMA query_only=ON`.",
        "- tags filtered by prefixes: " + ", ".join(report["exclude_tag_prefixes"]),
        "",
        "NOTE (brief vs addendum on the `mnemos:` prefix): the literal brief",
        "default would cut the corpus to 34 rows — the `mnemos:*` tags are the",
        "storage data-contract markers riding 3,324 of 3,647 live domain rows.",
        "Addendum 6 §A.3 («весь доступный доменный срез ≤2.7k») wins: the",
        "DEFAULT keeps `mnemos:` records; `--brief-tag-filter` reproduces the",
        "literal-brief cut for the TL gate to compare. Telemetry lives in",
        "memory_type session_context/conversation — excluded by the domain",
        "type gate (note/snippet/fact).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
