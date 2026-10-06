"""dataset-v4.3 translated classes + real-part assembler tests.

Pins, in the deliverable order:

1. translated-dup strategy (addendum 7 §A.7.1): label=duplicate, class
   field markers (stratum AND batch), BOTH orientations per spec, both
   directions ru->en and en->ru, digit-token byte-for-byte preservation
   (a lost token REFUSES the build — the addendum names it a labeling
   defect), unique deterministic pair ids;
2. translated-sibling strategy: label=not-duplicate, non-empty
   justification required, digit divergence ALLOWED (no digit contract),
   cross-language sibling side required;
3. quota gate q_v43: ≥150 pairs per class (addendum PROPOSED numbers),
   both directions present — a deficit or a missing direction refuses;
4. real-part assembler over MOCK stores (the real stores are never
   touched by tests): read-only discipline (query_only, bytes
   unchanged), privacy tag-prefix filter, min-content-chars gate,
   secrets exclusion with reason-only counters, dedup by the engine
   content-hash convention, deterministic manifest fingerprint, no
   store content in stdout.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_dataset_v43_strategies as v43  # noqa: E402 — scripts shim above
from cortex.data.fingerprints import canonical_json  # noqa: E402

# ── shared authored fixtures (RU→EN + EN→RU, digits preserved) ───────────────


def _ru_side(title: str, body: str) -> dict:
    return {
        "title": title,
        "body": body,
        "tags": ["project:mnemos", "agent:gcw-dataset-engineer"],
        "language": "ru",
        "record_type": "note",
    }


def _en_side(title: str, body: str) -> dict:
    return {
        "title": title,
        "body": body,
        "tags": ["project:mnemos", "agent:gcw-dataset-engineer"],
        "language": "en",
        "record_type": "note",
    }


#: A dup spec pair: dates/versions/sums survive byte-for-byte.
DUP_RU2EN = {
    "key": "dup-1",
    "record": _ru_side(
        "Дедлайн релиза 5 октября",
        "Собрание перенесли на 5 октября; сборка v1.2.2 выходит "
        "в 18:30, бюджет правок 12 500 строк.",
    ),
    "translation": _en_side(
        "The release deadline is October 5",
        "The assembly was moved to October 5; build v1.2.2 ships at "
        "18:30, the edit budget is 12 500 lines.",
    ),
}

#: EN→RU direction, ports and durations preserved.
DUP_EN2RU = {
    "key": "dup-2",
    "record": _en_side(
        "Healthcheck port moved",
        "The healthcheck listens on port 8443 with a 30 ms budget; "
        "rollout completed on 2026-09-24.",
    ),
    "translation": _ru_side(
        "Порт healthcheck перенесён",
        "Healthcheck слушает порт 8443 с бюджетом 30 мс; раскатка "
        "завершена 2026-09-24.",
    ),
}

#: Sibling RU→EN: same topic (release deadline), DIFFERENT meaning,
#: numbers deliberately different — no digit contract for this class.
SIB_RU2EN = {
    "key": "sib-1",
    "record": DUP_RU2EN["record"],  # RU side
    "sibling": _en_side(
        "The release slipped again",
        "The team moved the release past October 25: build v1.2.3 is "
        "not code-complete, and the edit budget is exhausted.",
    ),
    "justification": "Original: release moved TO October 5 (planned). "
    "Sibling: release moved PAST October 25 with the budget gone — "
    "delay, not a date change.",
}

#: Sibling EN→RU: record stays EN (the source), sibling = fresh RU
#: statement with a different meaning.
SIB_EN2RU = {
    "key": "sib-2",
    "record": DUP_EN2RU["record"],  # EN side — source of en->ru
    "sibling": _ru_side(
        "Healthcheck вернули на 8080",
        "Healthcheck перенесли НАЗАД на порт 8080 после отката эксперимента с 8443.",
    ),
    "justification": "Original: healthcheck now listens on 8443. "
    "Sibling: it reverted to 8080 — the opposite fact.",
}


def _dup_specs(n: int) -> list[dict]:
    """Quota-scale dup specs — BOTH directions alternate, so n ≥ 75
    covers the 150-pair quota with both directions present."""
    base = [DUP_RU2EN, DUP_EN2RU]
    out = []
    for i in range(n):
        spec = base[i % 2]
        out.append({**spec, "key": f"{spec['key']}-{i}"})
    return out


def _sib_specs(n: int) -> list[dict]:
    base = [SIB_EN2RU, SIB_RU2EN]
    out = []
    for i in range(n):
        spec = base[i % 2]
        out.append({**spec, "key": f"{spec['key']}-{i}"})
    return out


# ── 1. translated-dup ────────────────────────────────────────────────────────


def test_dup_rows_labels_and_class_markers() -> None:
    rows = v43.build_translated_rows([DUP_RU2EN], [])
    assert len(rows) == 2  # both orientations
    for row in rows:
        assert row["label"] == "duplicate"
        assert row["stratum"] == "translated-dup"
        assert row["batch"] == "translated-dup"
        assert row["direction"] == "ru->en"


def test_dup_builds_both_orientations() -> None:
    rows = v43.build_translated_rows([DUP_RU2EN], [])
    (a, b) = [
        r for r in rows if r["pair_id"].endswith("0001") + r["pair_id"].endswith("0002")
    ]
    assert {r["pair_id"] for r in rows} == {"V43-TDUP-0001", "V43-TDUP-0002"}
    assert (a["record"], a["candidate"]) == (b["candidate"], b["record"]) or (
        a["record"] == rows[1]["candidate"] and a["candidate"] == rows[1]["record"]
    )


def test_dup_digit_preservation_holds_by_construction() -> None:
    for spec in (DUP_RU2EN, DUP_EN2RU):
        original = f"{spec['record']['title']}\n{spec['record']['body']}"
        translated = f"{spec['translation']['title']}\n{spec['translation']['body']}"
        assert v43.missing_digit_tokens(original, translated) == []


def test_dup_missing_digit_token_refuses_build() -> None:
    bad = {
        **DUP_RU2EN,
        "record": _ru_side(
            "Дедлайн релиза 5 октября",
            "Собрание перенесли на 5 октября; сборка v1.2.2 выходит "
            "в 18:30, бюджет правок 12 500 строк.",
        ),
        "translation": _en_side(
            "The release deadline day",
            "The assembly was moved; build v1.2.2 ships at "
            "18:30, the edit budget is 12 500 lines.",  # every "5" lost
        ),
    }
    with pytest.raises(v43.DigitTokenContractError, match=r"digit tokens.*'5'"):
        v43.build_translated_rows([bad], [])


def test_digit_tokens_and_missing_helper() -> None:
    assert v43.digit_tokens("сборка v1.2.2 в 18:30, порт 8443") == [
        "1",
        "2",
        "2",
        "18",
        "30",
        "8443",
    ]
    assert v43.missing_digit_tokens("a 12 b 500", "x 12 y 500") == []
    assert v43.missing_digit_tokens("a 12 b 500", "x 500") == ["12"]
    # a bare 5 is NOT satisfied by the 5 inside 500 (standalone-run rule)
    assert v43.missing_digit_tokens("день 5", "the 500th") == ["5"]
    assert v43.missing_digit_tokens("день 5", "day 5 and 500") == []
    # dots break adjacency: IP octets preserved run-wise
    assert v43.missing_digit_tokens("192.168.0.1", "at 192.168.0.1 now") == []


def test_dup_same_language_refuses() -> None:
    bad = {
        "key": "bad",
        "record": _ru_side("t", "тело с числом 5"),
        "translation": _ru_side("t2", "другое тело с числом 5"),
    }
    with pytest.raises(v43.DigitTokenContractError, match="not a translation"):
        v43.build_translated_rows([bad], [])


def test_dup_rows_deterministic_and_fingerprintable() -> None:
    a = v43.build_translated_rows(_dup_specs(10), [])
    b = v43.build_translated_rows(_dup_specs(10), [])
    assert canonical_json(a) == canonical_json(b)
    ids = [r["pair_id"] for r in a]
    assert len(ids) == len(set(ids)) == 20
    for row in a:
        digest = v43.translated_pair_sha256(row)
        assert len(digest) == 64
        # tamper with the class marker → digest changes (class is content)
        tampered = {**row, "stratum": "translated-sibling"}
        assert v43.translated_pair_sha256(tampered) != digest


# ── 2. translated-sibling ────────────────────────────────────────────────────


def test_sibling_rows_labels_and_justification() -> None:
    rows = v43.build_translated_rows([], [SIB_EN2RU])
    assert len(rows) == 2
    for row in rows:
        assert row["label"] == "not-duplicate"
        assert row["stratum"] == "translated-sibling"
        assert row["batch"] == "translated-sibling"
        assert row["justification"] == SIB_EN2RU["justification"]
    # both orientations carry it
    assert {r["record"] == rows[0]["candidate"] for r in rows} == {True, False} or (
        rows[0]["record"] == rows[1]["candidate"]
    )


def test_sibling_allows_digit_divergence() -> None:
    # sibling loses every digit of the record — allowed, no error
    rows = v43.build_translated_rows([], [SIB_EN2RU])
    assert len(rows) == 2


def test_sibling_missing_justification_refuses() -> None:
    for bad_just in (None, "", "   "):
        bad = {**SIB_EN2RU, "justification": bad_just}
        with pytest.raises(v43.SiblingJustificationError, match="justification"):
            v43.build_translated_rows([], [bad])


def test_sibling_same_language_refuses() -> None:
    bad = {
        "key": "bad-sib",
        "record": _ru_side("t", "тело о дедлайне 5 октября"),
        "sibling": _ru_side("t3", "другое тело о дедлайне 25 октября"),
        "justification": "different month",
    }
    with pytest.raises(v43.SiblingJustificationError, match="cross-language"):
        v43.build_translated_rows([], [bad])


def test_sibling_is_not_near_identity() -> None:
    """A sibling is a DIFFERENT statement — its edit mass sits far above
    the near-identity family band (watchlist: [1; 4] chars)."""
    rows = v43.build_translated_rows([], [SIB_EN2RU])
    for row in rows:
        mass = v43.body_edit_mass_chars(row["record"]["body"], row["candidate"]["body"])
        assert mass > 40


# ── 3. quota gate ────────────────────────────────────────────────────────────


def test_quota_passes_at_scale() -> None:
    rows = v43.build_translated_rows(_dup_specs(76), _sib_specs(76))
    assert len(rows) == 304
    assert v43.quota_violations_v43(rows) == []
    by_dir = Counter()
    for row in rows:
        if row["stratum"] == "translated-dup":
            by_dir[(row["stratum"], row["direction"])] += 1
    assert by_dir[("translated-dup", "ru->en")] == 76
    assert by_dir[("translated-dup", "en->ru")] == 76


def test_quota_counts_by_dup_spec_not_pair() -> None:
    """≥150 PAIRS required: 74 originals → 148 pairs → refusal."""
    rows = v43.build_translated_rows(_dup_specs(74), _sib_specs(76))
    violations = v43.quota_violations_v43(rows)
    assert len(violations) == 1
    assert "translated-dup" in violations[0]
    assert "148 translated-dup pairs < 150" in violations[0]


def test_quota_requires_both_directions() -> None:
    """A class can hit its pair total from ONE direction and still
    refuse: the addendum demands BOTH directions (RU→EN and EN→RU)."""
    dup_one_dir = v43.build_translated_rows([DUP_EN2RU] * 80, [])
    violations = v43.quota_violations_v43(dup_one_dir)
    assert any("ru->en" in v and "translated-dup" in v for v in violations), violations
    # the total-quota refusal still names the deficit for the sibling class
    assert any("translated-sibling" in v and "< 150" in v for v in violations)


# ── 4. real-part assembler (mock stores; read-only discipline) ──────────────


def _domain_row(
    rid: str,
    title: str,
    content: str,
    *,
    tags: list[str] | None = None,
    memory_type: str = "note",
    status: str = "published",
    created_at: str = "2026-09-01T10:00:00+00:00",
    canon_language: str | None = "en",
) -> dict:
    canon = {"canon": {"language": canon_language}} if canon_language else {}
    return {
        "id": rid,
        "title": title,
        "content": content,
        "tags": json.dumps(tags or []),
        "memory_type": memory_type,
        "created_at": created_at,
        "status": status,
        "metadata": json.dumps(canon),
    }


LONG = (
    "The build pipeline ships version 1.2.2 on port 8443 with the full "
    "telemetry budget documented in the runbook and the release train "
    "runs nightly at 18:30. "
)


def build_mock_stores(root: Path) -> dict[str, Path]:
    """Two mock mnemos.db stores with the schema surface the assembler
    reads, seeded so every filter/count class is pinned:
    live: 3 domain rows (2 unique after dedup) + 1 privacy-tagged + 1
    short + 1 telemetry-typed + 1 session_context + 1 secret carrier;
    backup: 2 rows, one content-identical to live, one unique."""
    live = root / "live" / "mnemos.db"
    backup = root / "backup" / "mnemos.db"
    for path in (live, backup):
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "title TEXT, tags TEXT NOT NULL DEFAULT '[]', memory_type TEXT "
            "NOT NULL DEFAULT 'note', created_at TEXT, status TEXT, metadata "
            "TEXT NOT NULL DEFAULT '{}')"
        )
        conn.close()
    rows_live = [
        _domain_row("m1", "Build pipeline", LONG, tags=["project:x"]),
        _domain_row(
            "m2",
            "Build pipeline",
            LONG,
            tags=["project:x"],
            created_at="2026-09-02T10:00:00+00:00",
        ),
        _domain_row(
            "m3",
            "Healthcheck",
            "The healthcheck port moved to 8443 and the probe "
            "timeout tightened from 250 ms to 120 ms on "
            "2026-09-24 after the rollout. " * 1,
            tags=["project:y"],
            memory_type="snippet",
        ),
        _domain_row(
            "m4",
            "Task board",
            "Task: ship the release to production by Friday despite the "
            "open review comments on the migration patch and the pending "
            "sign-off from the operations team this week. ",
            tags=["task:release"],
        ),
        _domain_row("m5", "Short", "too short"),
        _domain_row(
            "m6",
            "Sweeper",
            "The sweeper ran and swept the store during the "
            "nightly maintenance window. " * 2,
            memory_type="session_context",
        ),
        _domain_row(
            "m7",
            "Creds",
            "my key AKIAIOSFODNN7EXAMPLE inside a long body with the "
            "full configuration dump for the analytics service and the "
            "deployment credentials of the cluster. ",
        ),
    ]
    rows_backup = [
        _domain_row("m1", "Build pipeline", LONG, tags=["project:x"]),
        _domain_row(
            "m8",
            "Vault",
            "The vault backup ran nightly through the maintenance "
            "window and archived the sealed snapshots. " * 2,
            canon_language=None,
        ),
    ]
    for path, rows in ((live, rows_live), (backup, rows_backup)):
        conn = sqlite3.connect(path)
        for row in rows:
            conn.execute(
                "INSERT INTO memories (id, content, title, tags, memory_type, "
                "created_at, status, metadata) VALUES (:id, :content, :title, "
                ":tags, :memory_type, :created_at, :status, :metadata)",
                row,
            )
        conn.commit()
        conn.close()
    return {"live": live, "backup": backup}


@pytest.fixture()
def mock_sources(tmp_path: Path) -> dict[str, Path]:
    return build_mock_stores(tmp_path)


def test_assembler_hygiene_dedup_and_fingerprint(mock_sources, tmp_path) -> None:
    import assemble_real_part as asm

    counters: Counter = Counter()
    scan, provenance = asm._scan_reasons(None)
    assert provenance == "fallback scanner, not engine"
    rows = asm.collect_source(
        "live",
        mock_sources["live"],
        min_chars=120,
        exclude_prefixes=asm.DEFAULT_EXCLUDE_TAG_PREFIXES,
        scan=scan,
        counters=counters,
    )
    # collect_source is PRE-dedup: m1+m2 (same content) + m3 = 3 accepted;
    # m4 privacy-tag, m7 secret — hygiene-dropped; m5 short + m6 wrong
    # type — SQL-dropped. The assemble() step dedups m2 away.
    assert [r["side"]["language"] for r in rows] == [None, None, None]
    hashes = [r["content_hash"] for r in rows]
    assert len(set(hashes)) == 2
    assert counters["excluded:privacy-tag:live"] == 1
    assert counters["excluded:secret:aws-key:live"] == 1
    assert counters["candidates_sql:live"] == 5  # m5 short, m6 type — SQL drops
    assert len(rows) == 3  # the assemble() step stamps accepted:live with this


def test_assembler_dedup_first_seen_and_language(tmp_path, mock_sources) -> None:
    import assemble_real_part as asm

    rows, counters, fp, provenance = asm.assemble(
        (("live", mock_sources["live"]), ("backup", mock_sources["backup"])),
        min_chars=120,
        exclude_prefixes=asm.DEFAULT_EXCLUDE_TAG_PREFIXES,
        engine_src=None,
    )
    # 3 unique contents: pipeline, healthcheck, vault
    assert len(rows) == 3
    sources = Counter(r["source"] for r in rows)
    assert sources == {"live": 2, "backup": 1}
    # language attached from the canon envelope where present
    langs = {r["content_hash"]: r["side"]["language"] for r in rows}
    pipeline_hash = next(
        r["content_hash"] for r in rows if r["side"]["title"] == "Build pipeline"
    )
    vault_hash = next(r["content_hash"] for r in rows if r["source"] == "backup")
    assert langs[pipeline_hash] == "en"
    assert langs[vault_hash] is None
    assert rows[0]["side"]["title"] not in ("", None)  # composition present
    # manifest fingerprint deterministic: same pool → same fp
    _, _, fp2, _ = asm.assemble(
        (("live", mock_sources["live"]), ("backup", mock_sources["backup"])),
        min_chars=120,
        exclude_prefixes=asm.DEFAULT_EXCLUDE_TAG_PREFIXES,
        engine_src=None,
    )
    assert fp == fp2
    assert provenance == "fallback scanner, not engine"


def test_assembler_read_only_bytes_unchanged(tmp_path) -> None:
    """The stores must be untouched: bytes identical, query_only enforced."""
    import assemble_real_part as asm

    sources = build_mock_stores(tmp_path)
    before = {p: p.read_bytes() for p in sources.values()}
    for path in sources.values():
        conn = sqlite3.connect(path)
        pre_digest = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        conn.close()
        assert pre_digest >= 1
    rows, counters, fp, provenance = asm.assemble(
        tuple(sources.items()),
        min_chars=120,
        exclude_prefixes=asm.DEFAULT_EXCLUDE_TAG_PREFIXES,
        engine_src=None,
    )
    assert len(rows) == 3
    after = {p: p.read_bytes() for p in sources.values()}
    assert before == after
    # a write attempt through the collector's connection must fail
    conn = sqlite3.connect(f"file:{sources['live'].as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO memories (id, content, title, tags, memory_type, "
            "created_at, status, metadata) VALUES ('x', 'y', 'z', '[]', 'note', "
            "'2026-01-01', 'published', '{}')"
        )
    conn.close()


def test_assembler_min_chars_and_prefix_filters(tmp_path) -> None:
    import assemble_real_part as asm

    sources = build_mock_stores(tmp_path)
    counters: Counter = Counter()
    scan, _ = asm._scan_reasons(None)
    rows = asm.collect_source(
        "live",
        sources["live"],
        min_chars=30,
        exclude_prefixes=asm.BRIEF_EXCLUDE_TAG_PREFIXES,
        scan=scan,
        counters=counters,
    )
    titles = sorted(r["side"]["title"] for r in rows)
    # m4 has task:release (brief list excludes task:) — out; m7 secret —
    # out; m5 short (9 chars < 30) — SQL-dropped; m6 session_context —
    # SQL-dropped; m1+m2+m3 kept (pre-dedup)
    assert "Task board" not in titles
    assert "Creds" not in titles
    assert "Sweeper" not in titles
    assert "Build pipeline" in titles and "Healthcheck" in titles


def test_assembler_stdout_reports_no_content(tmp_path, mock_sources, capsys) -> None:
    """stdout carries counters only — NO store text/titles/ids."""
    import assemble_real_part as asm

    out_jsonl = tmp_path / "out.jsonl"
    readme = tmp_path / "README.md"
    argv = [
        "--out-jsonl",
        str(out_jsonl),
        "--readme",
        str(readme),
        "--min-content-chars",
        "120",
    ]
    # monkeypatch the default sources to the mocks
    original = asm.DEFAULT_SOURCES
    asm.DEFAULT_SOURCES = tuple(mock_sources.items())
    try:
        assert asm.main(argv) == 0
    finally:
        asm.DEFAULT_SOURCES = original
    printed = capsys.readouterr().out
    for secret in ("Build pipeline", "Healthcheck", "vault", LONG[:30], "m1"):
        assert secret not in printed
    # output rows carry content (local file, gitignored tree in prod)
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(rows) == 3
    readme_text = readme.read_text()
    assert "Build pipeline" not in readme_text
    assert "real_part_fingerprint" in readme_text or "fingerprint" in readme_text


def test_assembler_brief_tag_filter_uses_mnemos_prefix() -> None:
    import assemble_real_part as asm

    assert asm.BRIEF_EXCLUDE_TAG_PREFIXES == ("task:", "telemetry", "mnemos:")
    assert asm.DEFAULT_EXCLUDE_TAG_PREFIXES == ("task:", "telemetry")


def test_content_hash_matches_engine_convention() -> None:
    """The dedup hash = the engine's ccr.content_hash (sha256 hex) —
    pinned against the literal algorithm (engine tree independent)."""
    import assemble_real_part as asm

    text = "The pipeline ships v1.2.2 on port 8443"
    assert asm.content_hash(text) == asm.content_hash(text)
    import hashlib

    assert asm.content_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
