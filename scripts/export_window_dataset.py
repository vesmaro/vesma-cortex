#!/usr/bin/env python3
"""W0 window-composition dataset exporter (Plane A, ML-only).

Spec: docs/decisions/0007-window-telemetry-two-planes.md (ADR 0007) +
docs/specs/window-dataset-prereg.md (§6 = this script's acceptance).

What it does
------------
Reads the vitals sidecar (``assemble_metrics`` → ``injection_blocks`` →
``usage_reports``) and the main memory store (``memories``) STRICTLY
read-only (``file:…?mode=ro`` URI — the only legal Plane A access per
ADR 0007 §4.1; ``immutable=1`` is BANNED: the DB is live, WAL), computes
the derived fields of ADR 0007 §2, applies the conservative secret scan
at the freeze boundary (prereg §5: hit → row blocked + no-federate tag),
and freezes a weekly dataset + manifest + fingerprints under
``data/stage3/window/``.

Privacy contract (hard)
-----------------------
- No memory text is ever written anywhere: at most ``memory_id`` +
  ``dup_group = sha256(norm(content))`` (prereg §5).
- The HMAC key (``*.hkey``) is NEVER read; de-anonymization is not
  attempted.
- Sidecar and memory store open via ``mode=ro`` URI + ``PRAGMA
  query_only`` — a write handle is a firewall violation.

Token-unit contract (ADR 0007 §2)
---------------------------------
Every exported block row carries ``token_unit = "engine_estimate"``
(the engine's ``estimate_tokens``, W1–W4 classes). The harness W0 class
(estimator ``external``) is NOT exportable from Plane A — it is declared
absent in the manifest's class map, so units can never mix in one sum.

Derived fields (ADR 0007 §2), per block row:
- ``ccr_expanded`` — block's ``ccr_origin`` carries CCR hashes;
- ``redactions_gt0`` — parent assembly had ``redactions > 0``
  (the engine persists redactions per ASSEMBLY, not per block — the
  flag is inherited; recorded in the manifest method map);
- ``dup_group`` — sha256 of normalized memory content;
- ``age_days`` — assembly ts minus memory ``created_at``;
- ``in_cache_prefix`` — PROXY (see ``IN_CACHE_METHOD``): the sidecar
  keeps no per-block cache-position signal, so the flag derives from
  recurrence stability across consecutive rebuilds; the manifest and
  the freeze report must carry this caveat;
- ``content_type`` — the W0–W4 class of the event, mapped from the
  engine's per-block ``content_type``/``source`` vocabulary; every
  artifact is classified exactly once (no W2c/W4 double counting:
  W2c tails live OUTSIDE budget blocks and never appear in
  ``injection_blocks``).

Labels (prereg §2), per block row — separate columns, composed
post-hoc over the GLOBAL assembly stream (ts, id order):
- ``l0_dead`` — no usage signal within the deadness window
  T = 3 subsequent assemblies or 30 minutes (whichever first);
  the scope of deadness is the window, never a single assembly;
  ``null`` when the window is censored at the data edge;
- ``l1_survival`` — the block's memory was re-injected in >= 3
  subsequent assemblies (free default label);
- ``re_requested`` — same memory re-requested (re-injected) within
  30 minutes after the deadness window closed;
- ``l3_quote`` / ``l2_action`` — NOT collectable from Plane A (the
  harness does not feed model output back); declared in the manifest
  dataset map, columns stay null.

CLI is deterministic and cron-ready: explicit flags, fixed exit codes
(0 ok / 3 source unreadable / 4 schema mismatch / 5 output failure),
no implicit state. Scheduling itself is NOT configured here.

Stdlib only (sqlite3/json/hashlib/argparse/re/datetime/pathlib/unicodedata).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

__version__ = "0.1.0"

SPEC_ADR = "ADR-0007 v1 (MLC round 8, ratified 2026-10-31)"
SPEC_PREREG = "window-dataset-prereg v0 (MLC round 8, ratified 2026-10-31)"

#: Deadness window T (prereg §2): 3 subsequent assemblies OR 30 minutes,
#: whichever first. The scope of deadness is always the window.
DEAD_ASSEMBLIES = 3
DEAD_SECONDS = 30 * 60

#: W0–W4 class map (ADR 0007 §2). Engine per-block ``content_type``
#: vocabulary seen so far: {'prose'} (default 'memory'). Rules-lane and
#: beacon blocks would surface through the same column; W2c tail
#: compositions and the harness classes W0/W3 never appear in
#: ``injection_blocks`` (Plane B / non-budget tails) — declared absent.
RULE_SOURCES = frozenset({"rule", "rules", "pinned_rule", "pinned_rules"})
BEACON_SOURCES = frozenset({"beacon", "codegraph_beacon"})
CLASS_MAP_NOTE = {
    "W0": "harness_system — Plane B, estimator external, NOT collected here",
    "W1": "pinned_rules — injection_blocks source in rules vocabulary",
    "W2": "memory_recall — any other injected memory block (default)",
    "W2c": "tail compositions live OUTSIDE budget blocks — not in injection_blocks",
    "W3": "tool_dialog — harness side, Plane B, NOT collected here",
    "W4": "beacon — injection_blocks source in beacon vocabulary",
}

#: in_cache_prefix derivation (see module docstring): the block's memory
#: also appeared in the previous assembly of the same session at an
#: ordinal <= the current one — recurrence-stable across rebuilds.
IN_CACHE_METHOD = (
    "proxy: memory re-injected in previous assembly of the same session "
    "at ordinal <= current; sidecar keeps no per-block cache-position "
    "signal (align stats are assembly-level aggregates)"
)

#: Conservative secret patterns (freeze-boundary scan, prereg §5).
#: Reused scanner not present in scripts/ — minimal fail-safe set:
#: provider/API tokens, Bearer headers, private-key blocks,
#: credentials-in-URL.
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),              # GitHub classic PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),      # GitHub fine-grained PAT
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),        # GitHub app/user tokens
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),         # sk- style API keys
    re.compile(r"AKIA[0-9A-Z]{12,}"),                 # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),            # Google API key
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{8,}"),       # Slack tokens
    re.compile(r"glpat-[A-Za-z0-9_-]{12,}"),          # GitLab PAT
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.I),  # auth headers
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # PEM key blocks
    re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),  # creds URL
)


class ExportError(Exception):
    """Fatal export condition carrying the intended process exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ── helpers ───────────────────────────────────────────────────────────────────


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_content(text: str) -> str:
    """Canonical normalization for ``dup_group``: NFC, casefold, all
    whitespace runs collapsed to one space, stripped. Deterministic."""
    collapsed = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text))
    return collapsed.strip().casefold()


def iso_to_epoch(value: str) -> float | None:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def percentile(values: list[int], q: float) -> int | None:
    """Nearest-rank percentile over a sorted copy — deterministic."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(q * len(ordered) + 0.5))
    return ordered[min(rank, len(ordered)) - 1]


def open_ro(path: Path) -> sqlite3.Connection:
    """The ONLY legal open for this exporter: read-only URI + query_only.

    ``immutable=1`` is deliberately absent — the databases are live WAL
    files; pretending they are immutable is both false and banned.
    """
    if not path.is_file():
        raise ExportError(f"source database missing: {path}", 3)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
        conn.execute("PRAGMA query_only = 1")
    except sqlite3.Error as exc:
        raise ExportError(f"cannot open read-only: {path}: {exc}", 3) from exc
    return conn


def require_tables(conn: sqlite3.Connection, path: Path, table_columns: dict[str, set[str]]) -> None:
    """Fail fast on schema drift — never improvise around a foreign schema."""
    present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, columns in table_columns.items():
        if table not in present:
            raise ExportError(f"schema mismatch in {path}: table '{table}' absent", 4)
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}  # nosec B608 - static table name
        missing = columns - actual
        if missing:
            raise ExportError(
                f"schema mismatch in {path}: {table} lacks columns {sorted(missing)}", 4
            )


def classify(source: str) -> str:
    """Engine per-block content_type → W0–W4 class, exactly once."""
    if source in RULE_SOURCES:
        return "W1"
    if source in BEACON_SOURCES:
        return "W4"
    return "W2"


def scan_secrets(serialized: str) -> bool:
    return any(pattern.search(serialized) for pattern in SECRET_PATTERNS)


# ── extraction ────────────────────────────────────────────────────────────────


def load_assemblies(conn: sqlite3.Connection, where: str, params: list) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session, project, agent, ts, mode, budget, tokens_estimated,"
        " blocks_count, blocks_refused, redactions, ccr_expanded, query_source,"
        " file_stem, stage_stats_json, fingerprint"
        f" FROM assemble_metrics{where} ORDER BY ts, id",  # nosec B608 - static filter
        params,
    ).fetchall()
    return [
        {
            "metrics_id": r[0], "session": r[1], "project": r[2], "agent": r[3],
            "ts": r[4], "mode": r[5], "budget": r[6], "tokens_estimated": r[7],
            "blocks_count": r[8], "blocks_refused": r[9], "redactions": r[10],
            "ccr_expanded": r[11], "query_source": r[12], "file_stem": r[13],
            "stage_stats_json": r[14], "fingerprint": r[15],
        }
        for r in rows
    ]


def load_blocks(conn: sqlite3.Connection, ids: list[int]) -> dict[int, list[dict]]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT metrics_id, block_id, memory_id, source, score, tokens, ccr_origin"
        f" FROM injection_blocks WHERE metrics_id IN ({marks}) ORDER BY metrics_id, id",
        ids,
    ).fetchall()
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r[0], []).append(
            {
                "block_id": r[1], "memory_id": r[2], "source": r[3], "score": r[4],
                "tokens": r[5], "ccr_origin": r[6],
            }
        )
    return out


def load_usage(conn: sqlite3.Connection, ids: list[int]) -> tuple[dict[int, dict], dict[int, set[str]]]:
    """usage_reports keyed by metrics_id + per-assembly touched block ids.

    A usage report names blocks of ITS OWN assembly (ordinals
    ``<metrics_id>:<i>``), so the touched set is per assembly — the
    label composer propagates a signal to the same MEMORY re-injected
    later (usage_reports can only reference their own assembly's ids).
    """
    if not ids:
        return {}, {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT metrics_id, block_ids_touched_json, tokens_out, wrong_tool_flag"
        f" FROM usage_reports WHERE metrics_id IN ({marks}) ORDER BY id",
        ids,
    ).fetchall()
    per_assembly: dict[int, dict] = {}
    touched_by_assembly: dict[int, set[str]] = {}
    for metrics_id, touched_json, tokens_out, wrong_flag in rows:
        slot = per_assembly.setdefault(metrics_id, {"reports": 0, "tokens_out": 0, "wrong_tool": False})
        slot["reports"] += 1
        slot["tokens_out"] += int(tokens_out or 0)
        slot["wrong_tool"] = slot["wrong_tool"] or bool(wrong_flag)
        if touched_json:
            try:
                entries = json.loads(touched_json)
            except (json.JSONDecodeError, TypeError):
                entries = []
            target = touched_by_assembly.setdefault(metrics_id, set())
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, str):
                    target.add(entry)
    return per_assembly, touched_by_assembly


def load_memory_facts(store: sqlite3.Connection, memory_ids: set[str]) -> dict[str, dict]:
    """``memories`` join by memory_id: dup_group + created_at (epochs).
    Content is read to hash and NEVER persisted (prereg §1/§5)."""
    facts: dict[str, dict] = {}
    for memory_id in sorted(memory_ids):
        row = store.execute(
            "SELECT content, created_at FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            facts[memory_id] = {"resolved": False, "dup_group": None, "created_epoch": None}
            continue
        content, created_at = row
        facts[memory_id] = {
            "resolved": True,
            "dup_group": sha256_text(normalize_content(content or "")),
            "created_epoch": iso_to_epoch(created_at) if created_at else None,
        }
    return facts


# ── label composition ─────────────────────────────────────────────────────────


def deadness_window(session_seq: list[dict], index: int) -> tuple[list[dict], float, bool]:
    """The T window after assembly ``index`` of a session sequence:
    up to DEAD_ASSEMBLIES subsequent assemblies or DEAD_SECONDS, whichever
    first. Returns (window assemblies, window end ts, censored flag).

    Censored (undecidable) only when the session ENDS inside an
    unfinished window — fewer than DEAD_ASSEMBLIES subsequent assemblies
    all within DEAD_SECONDS. A window closed by the time budget is always
    decidable, even with zero assemblies inside it.
    """
    origin_ts = session_seq[index]["ts"]
    following = session_seq[index + 1 :]
    if not following:
        return [], origin_ts, True  # nothing after — window never observed
    window: list[dict] = []
    closed_by_time = False
    for k, nxt in enumerate(following):
        if k >= DEAD_ASSEMBLIES:
            break
        if nxt["ts"] - origin_ts > DEAD_SECONDS:
            closed_by_time = True
            break
        window.append(nxt)
    if closed_by_time:
        return window, following[k]["ts"], False
    if len(following) < DEAD_ASSEMBLIES:
        # session ended while the window was still open — undecided yet
        return window, window[-1]["ts"] if window else origin_ts, True
    return window, window[-1]["ts"], False


def compose_labels(session_seq: list[dict], blocks_by_assembly,
                   touched_by_assembly: dict[int, set[str]]) -> dict[str, dict]:
    """Per-block labels over the deadness window (prereg §2).

    The sequence is the GLOBAL assembly stream (ts, id order): T = the
    next 3 assemblies or 30 minutes, whichever first — the prereg scopes
    deadness by the window, never by a single assembly, and a
    session-local window would censor every block of a session that
    happened to contain one assembly (a session artifact, not deadness).
    A signal is usage_report contact with the block's id, propagated to
    the same MEMORY inside the window (a report can only name blocks of
    its own assembly; re-injected copies carry the entity forward — the
    alternative would call a cited memory dead whenever the citation hit
    a later copy). The origin assembly's own report counts: it arrives
    seconds after injection, inside T by construction.
    """
    # memory → touched, per assembly (from that assembly's own reports)
    touched_mem_by_assembly: dict[int, dict[str, bool]] = {}
    for mid, blocks in blocks_by_assembly.items():
        report_ids = touched_by_assembly.get(mid, set())
        slot: dict[str, bool] = {}
        for block in blocks:
            hit = block["block_id"] in report_ids
            slot[block["memory_id"]] = slot.get(block["memory_id"], False) or hit
        touched_mem_by_assembly[mid] = slot

    labels: dict[str, dict] = {}
    memory_positions: dict[str, list[int]] = {}
    for index, assembly in enumerate(session_seq):
        for block in blocks_by_assembly[assembly["metrics_id"]]:
            memory_positions.setdefault(block["memory_id"], []).append(index)

    for index, assembly in enumerate(session_seq):
        window, window_end, censored = deadness_window(session_seq, index)
        window_last_index = index + len(window)  # last assembly index inside the window
        mid = assembly["metrics_id"]
        for order, block in enumerate(blocks_by_assembly[mid]):
            memory = block["memory_id"]
            signal = touched_mem_by_assembly[mid].get(memory, False)  # own report
            for window_assembly in window:
                if touched_mem_by_assembly.get(window_assembly["metrics_id"], {}).get(memory, False):
                    signal = True
                    break
            future = [i for i in memory_positions.get(memory, []) if i > index]
            survival = None
            if len(session_seq) - 1 - index >= 3:
                survival = len(future) >= 3
            re_requested = None
            if not censored and not signal and not survival:
                # same memory re-injected AFTER leaving the window, within T
                for later in future:
                    if later <= window_last_index:
                        continue
                    later_ts = session_seq[later]["ts"]
                    if window_end <= later_ts <= window_end + DEAD_SECONDS:
                        re_requested = True
                        break
                else:
                    re_requested = False
            labels[block["block_id"]] = {
                "touched_in_window": signal,
                "l0_dead": None if censored else not signal,
                "window_censored": censored,
                "window_assemblies": len(window),
                "l1_survival": survival,
                "re_requested": re_requested,
                "ordinal": order,
            }
    return labels


def in_cache_proxy(session_seq: list[dict], blocks_by_assembly) -> dict[str, bool]:
    """Recurrence-stability proxy for ``in_cache_prefix`` (IN_CACHE_METHOD)."""
    flags: dict[str, bool] = {}
    previous_memory_ordinals: dict[str, int] | None = None
    for assembly in session_seq:
        current: dict[str, int] = {}
        for order, block in enumerate(blocks_by_assembly[assembly["metrics_id"]]):
            current[block["memory_id"]] = order
            prev_order = (previous_memory_ordinals or {}).get(block["memory_id"])
            flags[block["block_id"]] = prev_order is not None and prev_order <= order
        previous_memory_ordinals = current
    return flags


# ── export ────────────────────────────────────────────────────────────────────


def build_rows(assemblies, blocks_by_assembly, usage_by_assembly, memory_facts,
               session_sequences, touched_by_assembly) -> tuple[list[dict], list[dict]]:
    block_rows, assembly_rows = [], []
    # Deadness/survival windows run over the GLOBAL stream (ts, id);
    # the cache-prefix proxy stays session-scoped (KV cache is a
    # per-conversation concept).
    labels = compose_labels(assemblies, blocks_by_assembly, touched_by_assembly)
    for session, sequence in session_sequences.items():
        cache_flags = in_cache_proxy(sequence, blocks_by_assembly)
        for assembly in sequence:
            mid = assembly["metrics_id"]
            usage = usage_by_assembly.get(mid, {"reports": 0, "tokens_out": 0, "wrong_tool": False})
            assembly_rows.append({
                "metrics_id": mid,
                "ts_utc": datetime.fromtimestamp(assembly["ts"], timezone.utc).isoformat(),
                "session": assembly["session"], "project": assembly["project"],
                "agent": assembly["agent"], "mode": assembly["mode"],
                "budget": assembly["budget"],
                "tokens_estimated_engine": assembly["tokens_estimated"],
                "blocks_count": assembly["blocks_count"],
                "blocks_refused": assembly["blocks_refused"],
                "redactions": assembly["redactions"],
                "ccr_expanded": assembly["ccr_expanded"],
                "query_source": assembly["query_source"],
                "fingerprint": assembly["fingerprint"],
                "usage_reported": usage["reports"] > 0,
                "usage_tokens_out": usage["tokens_out"],
                "wrong_tool_flag": usage["wrong_tool"],
            })
            for block in blocks_by_assembly[mid]:
                fact = memory_facts.get(block["memory_id"], {})
                label = labels[block["block_id"]]
                ccr_hashes = []
                if block["ccr_origin"]:
                    try:
                        parsed = json.loads(block["ccr_origin"])
                        ccr_hashes = parsed if isinstance(parsed, list) else []
                    except (json.JSONDecodeError, TypeError):
                        ccr_hashes = []
                block_rows.append({
                    "metrics_id": mid,
                    "block_id": block["block_id"],
                    "ordinal": label["ordinal"],
                    "ts_utc": datetime.fromtimestamp(assembly["ts"], timezone.utc).isoformat(),
                    "session": assembly["session"], "project": assembly["project"],
                    "agent": assembly["agent"], "mode": assembly["mode"],
                    "content_type_class": classify(block["source"]),
                    "engine_content_type": block["source"],
                    "memory_id": block["memory_id"],
                    "memory_resolved": bool(fact.get("resolved")),
                    "score": block["score"],
                    "tokens_engine": block["tokens"],
                    "token_unit": "engine_estimate",
                    "ccr_expanded": len(ccr_hashes) > 0,
                    "ccr_origin_count": len(ccr_hashes),
                    "assembly_redactions": assembly["redactions"],
                    "redactions_gt0": assembly["redactions"] > 0,
                    "dup_group": fact.get("dup_group"),
                    "age_days": (
                        round((assembly["ts"] - fact["created_epoch"]) / 86400.0, 2)
                        if fact.get("created_epoch") is not None else None
                    ),
                    "in_cache_prefix": cache_flags.get(block["block_id"], False),
                    "touched_in_window": label["touched_in_window"],
                    "l0_dead": label["l0_dead"],
                    "window_censored": label["window_censored"],
                    "window_assemblies": label["window_assemblies"],
                    "l1_survival": label["l1_survival"],
                    "re_requested": label["re_requested"],
                    "wrong_tool_flag": usage["wrong_tool"],
                })
    return block_rows, assembly_rows


def summarize(block_rows, assembly_rows) -> dict:
    """Freeze aggregates — the numbers of prereg §2–§4.1, computed once.
    Token percentiles are PER ASSEMBLY (sum of the class's tokens within
    one assembly), per ADR 0007 §2 / prereg §4.1 — never per-block and
    never across units (each class carries exactly one token_unit)."""
    decided = [r for r in block_rows if r["l0_dead"] is not None]
    dead = [r for r in decided if r["l0_dead"]]
    per_assembly: dict[str, dict[int, int]] = {}
    for row in block_rows:
        slot = per_assembly.setdefault(row["content_type_class"], {})
        slot[row["metrics_id"]] = slot.get(row["metrics_id"], 0) + row["tokens_engine"]
    dup_within = 0
    by_assembly: dict[int, list[str]] = {}
    for row in block_rows:
        if row["content_type_class"] == "W2" and row["dup_group"]:
            by_assembly.setdefault(row["metrics_id"], []).append(row["dup_group"])
    for groups in by_assembly.values():
        if len(groups) > len(set(groups)):
            dup_within += 1
    w2_rows = [r for r in block_rows if r["content_type_class"] == "W2"]
    all_groups = [r["dup_group"] for r in w2_rows if r["dup_group"]]
    cross_dup = len(all_groups) - len(set(all_groups))
    tokens_by_class = {}
    for cls, assembly_sums in sorted(per_assembly.items()):
        values = list(assembly_sums.values())
        tokens_by_class[cls] = {
            "token_unit": "engine_estimate",
            "n_blocks": sum(
                1 for r in block_rows if r["content_type_class"] == cls
            ),
            "sum_tokens": sum(values),
            "p50_per_assembly": percentile(values, 0.50),
            "p90_per_assembly": percentile(values, 0.90),
        }
    return {
        "assemblies": len(assembly_rows),
        "blocks": len(block_rows),
        "usage_coverage": (
            round(sum(1 for a in assembly_rows if a["usage_reported"]) / len(assembly_rows), 4)
            if assembly_rows else None
        ),
        "dead_share_decided": round(len(dead) / len(decided), 4) if decided else None,
        "dead_decided_blocks": len(decided),
        "dead_censored_blocks": len(block_rows) - len(decided),
        "re_requested_true": sum(1 for r in block_rows if r["re_requested"]),
        "survival_true": sum(1 for r in block_rows if r["l1_survival"]),
        "tokens_by_class_per_assembly": tokens_by_class,
        "dup_within_assembly_blocks": dup_within,
        "dup_within_assembly_share_w2": round(dup_within / len(w2_rows), 4) if w2_rows else None,
        "dup_cross_assembly_w2": cross_dup,
        "memory_join_resolved": sum(1 for r in block_rows if r["memory_resolved"]),
        "window_ts_utc": [
            min((a["ts_utc"] for a in assembly_rows), default=None),
            max((a["ts_utc"] for a in assembly_rows), default=None),
        ],
    }


def write_freeze(out_dir: Path, block_rows, assembly_rows, summary, args, sources) -> dict:
    """Serialize + secret-scan at the freeze boundary + manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counters = {"rows_scanned": 0, "rows_blocked": 0}
    blocks_path = out_dir / "blocks.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        for row in block_rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            counters["rows_scanned"] += 1
            if scan_secrets(line):
                counters["rows_blocked"] += 1
                tomb = {
                    "block_id": row["block_id"], "blocked": True,
                    "reason": "secret_scan_hit", "no_federate": True,
                }
                fh.write(json.dumps(tomb, sort_keys=True) + "\n")
            else:
                fh.write(line + "\n")
    assemblies_path = out_dir / "assemblies.jsonl"
    with assemblies_path.open("w", encoding="utf-8") as fh:
        for row in assembly_rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            counters["rows_scanned"] += 1
            if scan_secrets(line):
                counters["rows_blocked"] += 1
                tomb = {
                    "metrics_id": row["metrics_id"], "blocked": True,
                    "reason": "secret_scan_hit", "no_federate": True,
                }
                fh.write(json.dumps(tomb, sort_keys=True) + "\n")
            else:
                fh.write(line + "\n")
    no_federate = counters["rows_blocked"] > 0
    manifest = {
        "schema": "cortex-window-dataset/0.1",
        "exporter_version": __version__,
        "spec_adr": SPEC_ADR,
        "spec_prereg": SPEC_PREREG,
        "spec_commit": getattr(args, "spec_commit", None),
        "freeze_ts_utc": utcnow_iso(),
        "window_ts_utc": summary["window_ts_utc"],
        "cadence": "weekly",
        "filters": {"projects": args.project or "all", "sessions": args.session or "all",
                     "window_start": args.window_start, "window_end": args.window_end},
        "sources": sources,
        "row_counts": {
            "blocks": len(block_rows), "assemblies": len(assembly_rows),
            "blocked_rows": counters["rows_blocked"],
        },
        "fingerprints": {
            "blocks_jsonl_sha256": sha256_file(blocks_path),
            "assemblies_jsonl_sha256": sha256_file(assemblies_path),
        },
        "secret_scan": {
            "patterns": len(SECRET_PATTERNS), "rows_scanned": counters["rows_scanned"],
            "rows_blocked": counters["rows_blocked"],
            "policy": "hit → row replaced by tombstone + no_federate",
        },
        "no_federate": no_federate,
        "class_map": CLASS_MAP_NOTE,
        "token_units": {
            "W1": "engine_estimate", "W2": "engine_estimate", "W4": "engine_estimate",
            "W0": "external (harness) — not exportable from Plane A, declared absent",
            "W3": "n/a — Plane B", "W2c": "n/a — outside budget blocks",
            "mixing_rule": "one unit per row (token_unit column); no cross-unit sums",
        },
        "methods": {
            "dup_group": "sha256(NFC + casefold + whitespace-collapsed content)",
            "age_days": "assembly_ts - memory.created_at (days, 2 decimals)",
            "in_cache_prefix": IN_CACHE_METHOD,
            "redactions_gt0": "inherited from parent assembly (engine persists per-assembly)",
            "wrong_tool_flag": "anti-proxy, inherited from the assembly's usage_reports",
            "l0_dead": "no usage signal within T=3 subsequent assemblies / 30 min "
                       "over the global assembly stream (ts, id); null = censored "
                       "at data edge",
            "l1_survival": "memory re-injected in >=3 subsequent assemblies of the "
                           "global stream; null when fewer than 3 subsequent exist",
            "re_requested": "memory re-injected within 30 min after window close; "
                            "entity proxy = memory_id (slot-level not persisted)",
            "l3_quote": "not collected (harness does not feed model output back)",
            "l2_action": "not collectable from Plane A (tool-call/entity link absent)",
        },
        "summary": summary,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the W0 window-composition dataset (Plane A, read-only)."
    )
    parser.add_argument("--sidecar", type=Path, required=True,
                        help="path to the vitals metrics.sqlite (opened mode=ro)")
    parser.add_argument("--store", type=Path, required=True,
                        help="path to the main vesma.db with memories (opened mode=ro)")
    parser.add_argument("--out", type=Path, required=True,
                        help="freeze directory, e.g. data/stage3/window/freeze-YYYY-MM-DD")
    parser.add_argument("--project", action="append", default=[],
                        help="restrict to project slug (repeatable; default all)")
    parser.add_argument("--session", action="append", default=[],
                        help="restrict to session id (repeatable; default all)")
    parser.add_argument("--window-start", type=str, default=None,
                        help="ISO UTC lower bound on assembly ts")
    parser.add_argument("--window-end", type=str, default=None,
                        help="ISO UTC upper bound on assembly ts")
    parser.add_argument("--spec-commit", type=str, default=None,
                        help="git SHA of the ratified spec the freeze is made under")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        sidecar = open_ro(args.sidecar)
        require_tables(sidecar, args.sidecar, {
            "assemble_metrics": {"id", "session", "project", "agent", "ts", "mode",
                                 "budget", "tokens_estimated", "blocks_count",
                                 "blocks_refused", "redactions", "ccr_expanded",
                                 "query_source", "file_stem", "stage_stats_json",
                                 "fingerprint"},
            "injection_blocks": {"id", "metrics_id", "block_id", "memory_id",
                                 "source", "score", "tokens", "ccr_origin"},
            "usage_reports": {"id", "metrics_id", "block_ids_touched_json",
                              "tokens_out", "wrong_tool_flag"},
        })
        store = open_ro(args.store)
        require_tables(store, args.store, {
            "memories": {"id", "content", "created_at"},
        })

        where, params = [], []
        if args.project:
            where.append(f"project IN ({','.join('?' * len(args.project))})")
            params.extend(args.project)
        if args.session:
            where.append(f"session IN ({','.join('?' * len(args.session))})")
            params.extend(args.session)
        start = iso_to_epoch(args.window_start) if args.window_start else None
        end = iso_to_epoch(args.window_end) if args.window_end else None
        if args.window_start and start is None:
            raise ExportError(f"unparseable --window-start: {args.window_start}", 4)
        if args.window_end and end is None:
            raise ExportError(f"unparseable --window-end: {args.window_end}", 4)
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts <= ?")
            params.append(end)
        clause = f" WHERE {' AND '.join(where)}" if where else ""

        assemblies = load_assemblies(sidecar, clause, params)
        if not assemblies:
            print("export_window_dataset: no assemblies in scope — nothing to freeze",
                  file=sys.stderr)
            return 2
        ids = [a["metrics_id"] for a in assemblies]
        blocks_by_assembly = load_blocks(sidecar, ids)
        usage_by_assembly, touched_by_assembly = load_usage(sidecar, ids)

        memory_ids = {
            b["memory_id"]
            for blocks in blocks_by_assembly.values()
            for b in blocks
            if b["memory_id"] != "unknown"
        }
        memory_facts = load_memory_facts(store, memory_ids)

        session_sequences: dict[str, list[dict]] = {}
        for assembly in assemblies:  # already ORDER BY ts, id
            session_sequences.setdefault(assembly["session"] or f"__anon_{assembly['metrics_id']}",
                                         []).append(assembly)

        block_rows, assembly_rows = build_rows(
            assemblies, blocks_by_assembly, usage_by_assembly, memory_facts,
            session_sequences, touched_by_assembly,
        )
        summary = summarize(block_rows, assembly_rows)
        sources = {
            "sidecar": {"path": str(args.sidecar), "sha256": sha256_file(args.sidecar),
                        "size_bytes": args.sidecar.stat().st_size,
                        "wal_size_bytes": (
                            (args.sidecar.with_name(args.sidecar.name + "-wal")).stat().st_size
                            if (args.sidecar.with_name(args.sidecar.name + "-wal")).exists() else 0
                        )},
            "memory_store": {"path": str(args.store), "sha256": sha256_file(args.store),
                             "size_bytes": args.store.stat().st_size},
        }
        manifest = write_freeze(args.out, block_rows, assembly_rows, summary, args, sources)
    except ExportError as exc:
        print(f"export_window_dataset: FAIL[{exc.exit_code}]: {exc}", file=sys.stderr)
        return exc.exit_code
    except (sqlite3.Error, OSError) as exc:
        print(f"export_window_dataset: FAIL[5]: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5

    print(json.dumps({
        "frozen_to": str(args.out),
        "assemblies": summary["assemblies"], "blocks": summary["blocks"],
        "usage_coverage": summary["usage_coverage"],
        "dead_share_decided": summary["dead_share_decided"],
        "secret_rows_blocked": manifest["secret_scan"]["rows_blocked"],
        "no_federate": manifest["no_federate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
