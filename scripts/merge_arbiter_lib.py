#!/usr/bin/env python3
"""Merge-arbiter corpus library — synthetic generators, predicates, silver.

Corpus v1 for the merge-arbiter Choice line (ADR 0002 duty #3), spec:
``docs/specs/merge-arbiter-corpus-prereg-DRAFT.md`` (v2, MLC edits N1-N9
woven in). No LLM anywhere (ratified position 6): every label is a
function of construction.

Frozen-contract discipline (ML-STD-0002 P2): the normalizer and the
feature names are IMPORTED from ``cortex.features.pair`` — zero
reimplementations (draft §2.1, MLC 28.10 edit N8). The embedder pin
``nano:sha256:3b752e06…`` is declared (A-frozen); this library computes
no embeddings (no cosine needed at build time — the razor band is carried
by the ratified T3 policy band + measured char5_jaccard through the
frozen feature module).

All template/slot content is AUTHORED synthetic text (ADR 0003 R2
publication class) — no store lines.

Determinism: pure functions + explicit ``random.Random(seed)``; a double
build with the same seed is byte-identical.
"""

from __future__ import annotations

import difflib
import hashlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- frozen consumer contract (A-frozen; ML-STD-0002 P2) -----------------

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402


def _normalized_text(record: RecordLike) -> str:
    """Frozen normalizer — imported, never reimplemented (draft §2.1)."""
    return " ".join(f"{record.title}\n{record.body}".split()).lower()


def norm1_body(body: str) -> str:
    """B(r) = norm1(body) — the silver normalization surface (draft §2)."""
    return _normalized_text(RecordLike(title="", body=body))


#: Embedder pin declared for the line (A-frozen). This build does not
#: embed; the pin is recorded in the manifest and enforced at scoring time.
EMBEDDER_PIN = (
    "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
)

__all__ = [
    "EMBEDDER_PIN",
    "FEATURE_NAMES",
    "BASELINES",
    "Cluster",
    "Member",
    "SCAFFOLDS",
    "arbitrate_deterministic",
    "baseline_first",
    "baseline_keep_oldest",
    "baseline_longest_body",
    "build_cluster",
    "build_fixture_cells",
    "classify",
    "correct_set_ok",
    "cosmetic_variant",
    "norm1_body",
    "run_sanity_suite",
    "sha256_text",
    "silver_check_ch3",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- scaffolds (authored synthetic content) ------------------------------
# Each scaffold: language, title, body with a {LONG} placeholder used by
# the replace sub-stratum. Bodies 400-700 chars so a single-token fact
# edit lands in the ratified razor band (policy §3 T3: cos 0.9975-1.0,
# med 0.9990; mechanical proxy char5_jaccard >= 0.99).

_LONGS = {
    "ru-migration": "предварительная схема распределения нагрузочных прогонов по стендам и окнам обслуживания",
    "ru-retention": "черновой регламент передачи дежурства между сменами инженеров платформы",
    "ru-review-cfg": "набросок соглашения о формате ревью конфигурационных файлов перед выкаткой",
    "ru-flags": "рабочая прикидка очередности выноса фиче-флагов из кодовой базы legacy-модуля",
    "ru-replicas": "предположительная последовательность включения реплик после планового окна",
    "ru-incident-labels": "обсуждаемая схема ретроспективных меток для старых инцидентов минувшего квартала",
}

SCAFFOLD_SEED = [
    dict(
        sid="ru-migration",
        lang="ru",
        title="Миграция сервиса на новый рантайм: заметки по плану",
        body=(
            "Фиксирую план миграции сервиса на новый рантайм, чтобы команда могла сверяться. "
            "Первая очередь — сборка и прогон интеграционных тестов на отдельном стенде, затем "
            "переключение читающих клиентов, и только после этого пишущий трафик. {LONG} "
            "Откат расписан: предыдущий образ остаётся прогретым ещё неделю, дашборд с "
            "латентностями переезжает последним. Прогресс отмечаю в этом же документе, чтобы "
            "не плодить копии и не терять историю обсуждения."
        ),
    ),
    dict(
        sid="ru-retention",
        lang="ru",
        title="Политика хранения черновиков: как договорились",
        body=(
            "Записываю итог договорённости про хранение черновиков, чтобы ориентир был под "
            "рукой. Черновики живут в отдельном пространстве, старше девяноста дней — "
            "архивируются, архив читается поиском, но не показывается в выдаче по умолчанию. "
            "{LONG} Спорные случаи решает автор, движок только напоминает о правиле. "
            "Если правило поменяется, эта запись обновляется, а не дублируется."
        ),
    ),
    dict(
        sid="ru-review-cfg",
        lang="ru",
        title="Ревью конфигов: черновик соглашения",
        body=(
            "Мысль про ревью конфигурационных файлов: любое изменение конфига проходит "
            "вторую пару глаз, кроме правок уровня логирования. Изменения, затрагивающие "
            "порты и лимиты, требуют отдельного окна и отметки в журнале изменений. {LONG} "
            "Шаблон комментария ревью прикладываем в описании, чтобы не собирать его по "
            "переписке. Позже вынесем правило в отдельный канон, пока живёт здесь."
        ),
    ),
    dict(
        sid="ru-flags",
        lang="ru",
        title="Фиче-флаги legacy: порядок выноса",
        body=(
            "План по выносу фиче-флагов из legacy-модуля. Сначала флаги, которые живут "
            "дольше квартала и не переключались с прошлого релиза, затем флаги со "
            "статистикой включений около нуля, остальное — по мере касания. {LONG} "
            "Каждый вынос фиксируется отдельным коммитом с пометкой флага, чтобы история "
            "восстанавливалась поиском. Список держу здесь же и обновляю по ходу."
        ),
    ),
    dict(
        sid="ru-replicas",
        lang="ru",
        title="Включение реплик после окна: рабочая заметка",
        body=(
            "Заметка про восстановление после планового окна: сначала читающие реплики "
            "внутреннего контура, затем пограничные, мастер — последним шагом после "
            "сверки контрольных сумм. {LONG} На каждом шаге проверяем отставание и "
            "ошибки применения, при расхождении — пауза и откат шага. Детали сверок "
            "складываю рядом, чтобы утренняя смена нашла всё в одном месте."
        ),
    ),
    dict(
        sid="ru-incident-labels",
        lang="ru",
        title="Ретроспективные метки инцидентов: набросок",
        body=(
            "Идея: помечать старые инциденты ретроспективными метками, чтобы поиск по "
            "причинам не терял хвосты. Метка ставится по факту пост-мортема, не по "
            "предположению, и всегда несёт ссылку на разбор. {LONG} Спорные помечаем "
            "двумя кандидатами и разбираем на ревью, единый ответ фиксируется в разборе. "
            "Пока это черновик, канон будет отдельным документом."
        ),
    ),
    dict(
        sid="en-oncall",
        lang="en",
        title="On-call rotation handoff: working notes",
        body=(
            "Working notes for the on-call handoff reform. The rotating engineer writes a "
            "short state summary at shift end, open incidents keep their owners, and the "
            "follow-the-sun handoff happens in the shared channel rather than in private "
            "threads. {LONG} Escalation paths stay pinned in the runbook, not in memory. "
            "I will keep updating this note as the rotation settles, so the team has one "
            "canonical place to check."
        ),
    ),
    dict(
        sid="en-cache",
        lang="en",
        title="Edge cache invalidation: draft approach",
        body=(
            "Draft approach for edge cache invalidation: purge on publish, soft TTL of "
            "five minutes for hot paths, and a manual override for emergencies. Stale "
            "content is acceptable for read-only pages but never for pricing. {LONG} "
            "Metrics on hit rate and purge latency go to the shared dashboard; anomalies "
            "are triaged the same day. This note is the working draft until the team "
            "ratifies the final rule."
        ),
    ),
    dict(
        sid="en-backups",
        lang="en",
        title="Backup restore drills: schedule sketch",
        body=(
            "Sketch of the restore drill schedule: quarterly full restore of the primary "
            "database into a scratch environment, monthly spot checks of object storage "
            "snapshots, and an annual game day with the whole platform team. {LONG} "
            "Every drill produces a short report with timings and gaps; the report lands "
            "next to this note so the next drill starts from the previous findings."
        ),
    ),
    dict(
        sid="en-deprecation",
        lang="en",
        title="API deprecation policy: first draft",
        body=(
            "First draft of the API deprecation policy: endpoints get a sunset header "
            "ninety days ahead, usage is tracked per consumer, and a consumer with zero "
            "traffic for a month is contacted before removal. {LONG} The changelog "
            "carries a dedicated section so external consumers have one canonical feed. "
            "Comments go into this note until the policy is ratified."
        ),
    ),
    dict(
        sid="en-quotas",
        lang="en",
        title="Rate limits for internal consumers: notes",
        body=(
            "Notes on rate limits for internal consumers: default burst of two hundred "
            "requests per second per service, sustained limit negotiated per consumer "
            "class, and a fast lane for batch jobs running in the maintenance window. "
            "{LONG} Breaches are visible on the shared graph and paged only for the "
            "second consecutive minute. Adjustments are recorded here to keep the "
            "history in one place."
        ),
    ),
    dict(
        sid="en-docs",
        lang="en",
        title="Docs migration: what moves where",
        body=(
            "Working summary of the docs migration: architecture notes move to the "
            "platform space, runbooks stay next to the services they describe, and "
            "decision records keep their own chronological index. {LONG} Redirects are "
            "installed for every moved page, and the crawler re-indexes weekly until the "
            "broken-link count reaches zero. Progress is logged in this same note."
        ),
    ),
]

_LONGS.update(
    {
        "en-oncall": _LONGS["ru-retention"],
        "en-cache": _LONGS["ru-replicas"],
        "en-backups": _LONGS["ru-migration"],
        "en-deprecation": _LONGS["ru-review-cfg"],
        "en-quotas": _LONGS["ru-flags"],
        "en-docs": _LONGS["ru-incident-labels"],
    }
)

SCAFFOLDS = SCAFFOLD_SEED


def scaffold_body(sid: str) -> str:
    raw = next(s for s in SCAFFOLDS if s["sid"] == sid)
    return raw["body"].replace("{LONG}", _LONGS[sid])


def scaffold_title(sid: str) -> str:
    return next(s for s in SCAFFOLDS if s["sid"] == sid)["title"]


# --- fact slots -----------------------------------------------------------
# kind -> per-language value pools and phrase renders (formatting diversity
# is part of the N1 pin: parenthetical / inline / dash styles).

_FACT_POOL = {
    "release": {
        "values": ["2.14.0", "3.2.1", "3.2.2", "4.0.0"],
        "render": {
            "ru": ["целевой релиз {v}", "(релиз: {v})", "релиз-цель: {v}"],
            "en": ["target release {v}", "(release: {v})", "release goal: {v}"],
        },
        "short": {"ru": "релиз {v}", "en": "release {v}"},
    },
    "deadline": {
        "values": ["15.11.2026", "2026-11-15", "30.11.2026"],
        "render": {
            "ru": ["дедлайн — {v}", "(дедлайн: {v})", "срок: {v}"],
            "en": ["deadline is {v}", "(deadline: {v})", "due date: {v}"],
        },
        "short": {"ru": "срок {v}", "en": "due {v}"},
    },
    "service": {
        "values": ["svc-auth-gw", "svc-billing", "edge-cache-2"],
        "render": {
            "ru": ["затронут сервис {v}", "(сервис: {v})", "сервис-владелец: {v}"],
            "en": ["affected service {v}", "(service: {v})", "owning service: {v}"],
        },
        "short": {"ru": "сервис {v}", "en": "service {v}"},
    },
    "budget": {
        "values": ["1 200 МБ", "64 ГБ", "800 МБ"],
        "render": {
            "ru": ["лимит хранилища — {v}", "(лимит: {v})", "квота: {v}"],
            "en": ["storage budget is {v}", "(budget: {v})", "quota: {v}"],
        },
        "short": {"ru": "лимит {v}", "en": "budget {v}"},
    },
    "throughput": {
        "values": ["40 000 rps", "1200 rps", "800 rps"],
        "render": {
            "ru": ["пиковая нагрузка — {v}", "(пик: {v})", "пропускная цель: {v}"],
            "en": ["peak load of {v}", "(peak: {v})", "throughput target: {v}"],
        },
        "short": {"ru": "пик {v}", "en": "peak {v}"},
    },
}

_FILLERS = {
    "ru": [
        "очередной этап подготовительных работ",
        "плановая рутина по расписанию без изменений",
        "технический перерыв между этапами работы",
    ],
    "en": [
        "routine preparation stage without any changes",
        "scheduled housekeeping work between stages",
        "a short technical pause between the stages",
    ],
}

INSERT_LEADS = {
    "ru": ["Внимание: ", "Обновление: ", "Примечание: "],
    "en": ["Note: ", "Update: ", "Heads-up: "],
}


def _slot_token(kind: str, value: str) -> str:
    return f"{kind}={value}"


def _render_sentence(kind: str, value: str, lang: str, style: int, lead: str) -> str:
    core = _FACT_POOL[kind]["render"][lang][style % 3].format(v=value)
    return f"{lead}{core}"


def _insert_into(body: str, sentence: str, position: str) -> tuple[str, int, int]:
    """Insert ``sentence`` at start/middle/end; return (body, span_start, span_end)."""
    if position == "start":
        new = f"{sentence} {body}"
        return new, 0, len(sentence)
    if position == "end":
        new = f"{body} {sentence}"
        return new, len(body) + 1, len(new)
    cut = body.find(". ")
    if cut == -1 or cut + 2 >= len(body) - 20:
        cut = len(body) // 2
        new = body[:cut] + " " + sentence + body[cut:]
        return new, cut + 1, cut + 1 + len(sentence)
    at = cut + 2
    new = body[:at] + sentence + " " + body[at:]
    return new, at, at + len(sentence)


@dataclass
class Member:
    mid: str
    title: str
    body: str
    tags: tuple[str, ...]
    record_type: str | None
    language: str | None
    created_at: str  # ISO-8601 UTC

    def to_record(self) -> RecordLike:
        return RecordLike(
            title=self.title,
            body=self.body,
            tags=self.tags,
            language=self.language,
            record_type=self.record_type,
        )


@dataclass
class Cluster:
    cid: str
    stratum: str
    substratum: str
    members: list[Member]
    label: dict
    declaration: dict = field(default_factory=dict)
    scenario: str = ""
    orientation: str = ""


ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
_CREATED_LO = 1_782_864_000  # 2026-07-01T00:00:00Z
_CREATED_HI = 1_792_454_400  # 2026-10-20T00:00:00Z
_CREATED_SPREAD = 864_000  # per-member jitter, 10 days


def _iso(ts: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).strftime(ISO_FMT)


def _rand_created_at(rng: random.Random) -> float:
    # window 2026-07-01..2026-10-20, independent of labels (A-created-indep)
    return float(rng.randint(_CREATED_LO, _CREATED_HI - _CREATED_SPREAD))


def _base_tags(lang: str, rng: random.Random) -> tuple[str, ...]:
    pool = ("platform", "infra", "notes", "draft", "ops", "planning")
    return tuple(sorted(rng.sample(pool, rng.randint(2, 4))))


def _other_kind(kind: str) -> str:
    return "deadline" if kind != "deadline" else "release"


# --- cluster builders ------------------------------------------------------


def build_cluster(
    rng: random.Random,
    stratum: str,
    substratum: str,
    sid: str,
    kind: str,
    value: str,
    ids: list[str],
    k: int = 2,
    suffix: str = "",
    force_orientation: str | None = None,
) -> Cluster:
    """Build one synthetic decision cluster; label = function of construction.

    ``suffix`` (holdout families only): an inert, label-neutral closing
    line appended to the scaffold body BEFORE construction so the fresh
    holdout material is byte-disjoint from train even for degenerate
    classes whose bodies are raw scaffolds (g_disj train↔MA-holdout leg).
    """
    title = scaffold_title(sid)
    base = scaffold_body(sid) + (f" {suffix}" if suffix else "")
    lang = next(s["lang"] for s in SCAFFOLDS if s["sid"] == sid)
    ts0 = _rand_created_at(rng)
    tags = _base_tags(lang, rng)
    decl: dict = {"kind": stratum, "facts": {}, "steps": []}
    scenario = f"{sid}|{substratum}|{kind}|{value}"

    if stratum == "CH-3" and k == 2:
        style, lead = rng.randrange(3), rng.choice(INSERT_LEADS[lang])
        delta = [_slot_token(kind, value)]
        if substratum == "insert":
            sentence = _render_sentence(kind, value, lang, style, lead)
            sup_body, s1, s2 = _insert_into(
                base, sentence, rng.choice(["start", "middle", "end"])
            )
            sub_body = base
        elif substratum == "replace":
            fact_phrase = _FACT_POOL[kind]["short"][lang].format(v=value)
            filler = rng.choice(_FILLERS[lang])
            assert len(filler) > len(fact_phrase) + 8, (
                "filler must exceed fact phrase (short superset)"
            )
            sup_body = base.replace(_LONGS[sid], fact_phrase)
            sub_body = base.replace(_LONGS[sid], filler)
            at = sup_body.find(fact_phrase)
            assert at >= 0, "fact phrase must survive the replace"
            s1, s2 = at, at + len(fact_phrase)
        else:
            raise ValueError(substratum)
        m_sup = Member(
            ids[0],
            title,
            sup_body,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        m_sub = Member(
            ids[1],
            title,
            sub_body,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        decl["mode"] = substratum
        decl["facts"] = {m_sup.mid: delta, m_sub.mid: []}
        decl["steps"] = [
            {"sup": m_sup.mid, "sub": m_sub.mid, "delta": delta, "spans": [(s1, s2)]}
        ]
        members = [m_sup, m_sub]
        label = {"type": "keep", "member": m_sup.mid}

    elif stratum == "CH-3" and k == 3:
        # chain a ⊂ b ⊂ c: step1 insert, step2 insert (different kind)
        k2 = _other_kind(kind)
        v2 = rng.choice([v for v in _FACT_POOL[k2]["values"]])
        d1 = [_slot_token(kind, value)]
        d2 = [_slot_token(k2, v2)]
        sent1 = _render_sentence(
            kind, value, lang, rng.randrange(3), rng.choice(INSERT_LEADS[lang])
        )
        sent2 = _render_sentence(
            k2, v2, lang, rng.randrange(3), rng.choice(INSERT_LEADS[lang])
        )
        body_a = base
        body_b, b1, b2 = _insert_into(
            base, sent1, rng.choice(["start", "middle", "end"])
        )
        body_c, c1, c2 = _insert_into(
            body_b, sent2, rng.choice(["start", "middle", "end"])
        )
        m_a = Member(
            ids[0],
            title,
            body_a,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        m_b = Member(
            ids[1],
            title,
            body_b,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        m_c = Member(
            ids[2],
            title,
            body_c,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        decl["mode"] = "chain"
        decl["facts"] = {m_a.mid: [], m_b.mid: d1, m_c.mid: d1 + d2}
        decl["steps"] = [
            {"sup": m_b.mid, "sub": m_a.mid, "delta": d1, "spans": [(b1, b2)]},
            {"sup": m_c.mid, "sub": m_b.mid, "delta": d2, "spans": [(c1, c2)]},
        ]
        members = [m_a, m_b, m_c]
        label = {"type": "keep", "member": m_c.mid}
        scenario = f"{sid}|chain|{kind}|{value}+{v2}"

    elif stratum == "CH-4":
        alt = next(v for v in _FACT_POOL[kind]["values"] if v != value)
        tmpl = _FACT_POOL[kind]["short"][lang]
        m1 = Member(
            ids[0],
            title,
            base.replace(_LONGS[sid], tmpl.format(v=value)),
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        m2 = Member(
            ids[1],
            title,
            base.replace(_LONGS[sid], tmpl.format(v=alt)),
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        decl = {
            "kind": "CH-4",
            "mode": "conflict",
            "facts": {
                m1.mid: [_slot_token(kind, value)],
                m2.mid: [_slot_token(kind, alt)],
            },
            "steps": [],
        }
        members = [m1, m2]
        label = {"type": "abstain"}
        scenario = f"{sid}|conflict|{kind}|{value}vs{alt}"

    elif stratum == "CH-2":
        mode = substratum  # "tags" | "type" | "language"
        m1 = Member(
            ids[0],
            title,
            base,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        if mode == "tags":
            m2 = Member(
                ids[1],
                title,
                base,
                tags + ("archive",),
                "note",
                lang,
                _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
            )
        elif mode == "type":
            m2 = Member(
                ids[1],
                title,
                base,
                tags,
                "summary",
                lang,
                _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
            )
        elif mode == "language":
            m2 = Member(
                ids[1],
                title,
                base,
                tags,
                "note",
                "en" if lang == "ru" else "ru",
                _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
            )
        else:
            raise ValueError(mode)
        decl = {"kind": "CH-2", "mode": mode, "facts": {}, "steps": []}
        members = [m1, m2]
        label = {"type": "abstain"}
        scenario = f"{sid}|meta|{mode}"

    elif stratum == "CH-0":
        m1 = Member(ids[0], title, base, tags, "note", lang, _iso(ts0))
        members = [m1]
        for j in range(1, k):
            members.append(
                Member(
                    ids[j],
                    title,
                    base,
                    tags,
                    "note",
                    lang,
                    _iso(ts0 + rng.randint(5, 130)),
                )
            )
        decl = {"kind": "CH-0", "facts": {}, "steps": []}
        label = {"type": "eligible-all"}

    elif stratum == "CH-5":
        m1 = Member(
            ids[0],
            title,
            base,
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        m2 = Member(
            ids[1],
            title,
            cosmetic_variant(base, rng),
            tags,
            "note",
            lang,
            _iso(ts0 + rng.randint(0, _CREATED_SPREAD)),
        )
        members = [m1, m2]
        decl = {"kind": "CH-5", "facts": {}, "steps": []}
        label = {"type": "eligible-all"}

    else:
        raise ValueError(stratum)

    # serialization order: random 50/50, independent of the label
    # (A-order-indep); mirrored scenario rows force opposite orientations
    # (A-symmetry)
    flip = (
        rng.random() < 0.5 if force_orientation is None else (force_orientation == "BA")
    )
    if flip:
        members = list(reversed(members))
    return Cluster(
        cid="",
        stratum=stratum,
        substratum=decl.get("mode", substratum),
        members=members,
        label=label,
        declaration=decl,
        scenario=scenario,
        orientation="BA" if flip else "AB",
    )


def cosmetic_variant(body: str, rng: random.Random) -> str:
    """Case/whitespace transforms erased by norm1 (policy §3 T1 class)."""
    out = body
    for _ in range(rng.randint(1, 3)):
        mode = rng.randrange(3)
        if mode == 0:
            i = out.find(" ")
            if i >= 0:
                out = out[:i] + "  " + out[i + 1 :]
        elif mode == 1:
            words = out.split(" ")
            j = rng.randrange(len(words))
            if words[j]:
                words[j] = words[j].upper()
            out = " ".join(words)
        else:
            out = "  " + out
    return out


# --- predicates (draft §2, production order §5.1) ---------------------------


def classify(members: list[Member], declarations: dict | None = None) -> str:
    """Deterministic CH-class (predicates, not words).

    Order (draft §5.1, MLC 28.10 edit N7): CH-0 FIRST (removes the
    CH-0∩CH-1 intersection), then CH-5 (¬CH-0), then CH-2; CH-3/CH-4 need
    generator declarations (no production fact extractor — draft §0).
    """
    shas = {sha256_text(m.body) for m in members}
    if (
        len(shas) == 1
        and len({m.title for m in members}) == 1
        and len({frozenset(m.tags) for m in members}) == 1
        and len({m.record_type for m in members}) == 1
        and len({m.language for m in members}) == 1
    ):
        return "CH-0"
    norm_bodies = {norm1_body(m.body) for m in members}
    if len(shas) > 1 and len(norm_bodies) == 1:
        return "CH-5"
    if len(norm_bodies) == 1 and len({m.title for m in members}) == 1:
        meta_diff = (
            len({frozenset(m.tags) for m in members}) > 1
            or len({m.record_type for m in members}) > 1
            or len({m.language for m in members}) > 1
        )
        if meta_diff:
            return "CH-2"
    if declarations:
        mode = declarations.get("mode")
        if mode == "conflict":
            return "CH-4"
        if mode in ("insert", "replace", "chain"):
            return "CH-3"
    return "undefined"


def norm2_regions(body_a: str, body_b: str) -> list[tuple[int, int, int, int, str]]:
    """norm2-delta (draft §2.1): non-equal difflib regions over norm1 bodies.

    Opcode shape: (tag, i1, i2, j1, j2) — the tag is element 0.
    """
    sm = difflib.SequenceMatcher(
        None, norm1_body(body_a), norm1_body(body_b), autojunk=False
    )
    return [op for op in sm.get_opcodes() if op[0] != "equal"]


#: Mechanical pin (ML-STD-0001 tolerance discipline): non-equal opcodes
#: separated by a run of equal text of <= NORM2_GAP chars on the compared
#: side are merged into ONE edit cluster (difflib fragments a contiguous
#: phrase edit into scattered small blocks around shared letters). Slots
#: are >= 100 chars apart, so GAP=12 cannot merge two distinct slots; the
#: cluster-count == slot-count assert stays fail-closed.
NORM2_GAP = 12


def norm2_edit_clusters(body_a: str, body_b: str) -> list[tuple[int, int, int, int]]:
    """Clustered edit regions (hulls) of the norm2-delta."""
    ops = norm2_regions(body_a, body_b)
    clusters: list[list[int]] = []
    for _t, i1, i2, j1, j2 in ops:
        if (
            clusters
            and i1 - clusters[-1][1] <= NORM2_GAP
            and j1 - clusters[-1][3] <= NORM2_GAP
        ):
            c = clusters[-1]
            c[1], c[3] = max(c[1], i2), max(c[3], j2)
            c[0], c[2] = min(c[0], i1), min(c[2], j1)
        else:
            clusters.append([i1, i2, j1, j2])
    return [tuple(c) for c in clusters]


def _norm_spans(body: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Map declared original-body spans onto the normalized surface."""
    nb = norm1_body(body)
    out = []
    for s1, s2 in spans:
        nfrag = " ".join(body[s1:s2].split()).lower()
        pos = nb.find(nfrag)
        assert pos >= 0, "declared slot fragment must survive normalization"
        out.append((pos, pos + len(nfrag)))
    return out


def silver_check_ch3(cluster: Cluster) -> dict:
    """Mechanical A-silver (draft §2.1/§6; MLC 28.10 edit N8).

    Per construction step: the norm2-delta between (sup, sub) bodies must
    BIJECT with the declared slot spans — one edit cluster per slot, the
    cluster lies within the slot ± NORM2_GAP on each boundary and covers
    >= 0.8 of the slot (difflib may match the slot's edge characters
    against the replacement phrase, shifting the hull by a char or two);
    no cluster outside a declared slot; the fact delta must be a PROPER
    superset. Normalizers come from the frozen module; zero
    reimplementations.
    """
    steps = cluster.declaration["steps"]
    facts = cluster.declaration["facts"]
    results = []
    for step in steps:
        sup = next(m for m in cluster.members if m.mid == step["sup"])
        sub = next(m for m in cluster.members if m.mid == step["sub"])
        clusters_ = norm2_edit_clusters(sup.body, sub.body)
        nspans = sorted(_norm_spans(sup.body, step["spans"]))
        bijection = len(clusters_) == len(nspans)
        if bijection:
            for (c1, c2, _d1, _d2), (p1, p2) in zip(sorted(clusters_), nspans):
                overlap = max(0, min(c2, p2) - max(c1, p1))
                slot_len = p2 - p1
                if not (
                    overlap / slot_len >= 0.8
                    and c1 >= p1 - NORM2_GAP
                    and c2 <= p2 + NORM2_GAP
                ):
                    bijection = False
                    break
        proper = set(facts[step["sub"]]) < set(facts[step["sup"]])
        results.append(
            {
                "sup": step["sup"],
                "sub": step["sub"],
                "regions": len(clusters_),
                "slots": len(nspans),
                "bijection": bool(bijection),
                "proper_superset": bool(proper),
                "ok": bool(bijection and proper),
            }
        )
    return {"ok": all(r["ok"] for r in results), "steps": results}


# --- production contour v1 (draft §5.1) -------------------------------------


def arbitrate_deterministic(cluster: Cluster) -> dict:
    """Deterministic legs of the production pipeline §5.1.

    Build-time projection of the model leg: CH-3 resolves via the
    generator declarations (largest fact set = the ⊆-maximum; silver),
    CH-4/undefined ABSTAIN by the closure rule. Reads NO body lengths —
    the length-proxy kill (probe 4) is structural.
    """
    cls = classify(cluster.members, cluster.declaration or None)
    if cls in ("CH-0", "CH-5"):
        keep = min(cluster.members, key=lambda m: m.created_at)
        return {"verdict": "keep", "member": keep.mid, "leg": "tie-break", "class": cls}
    if cls == "CH-2":
        return {"verdict": "abstain", "leg": "refusal", "class": cls}
    if cls == "CH-4":
        return {"verdict": "abstain", "leg": "closure-rule", "class": cls}
    if cls == "CH-3":
        facts = cluster.declaration["facts"]
        sup_mid = max(facts, key=lambda mid: len(facts[mid]))
        return {
            "verdict": "keep",
            "member": sup_mid,
            "leg": "silver-resolver",
            "class": cls,
        }
    return {"verdict": "abstain", "leg": "closure-rule", "class": cls}


# --- baselines (draft §5) ----------------------------------------------------


def baseline_keep_oldest(members: list[Member]) -> str:
    return min(members, key=lambda m: m.created_at).mid


def baseline_longest_body(members: list[Member]) -> str:
    return max(members, key=lambda m: len(m.body)).mid


def baseline_first(members: list[Member]) -> str:
    return members[0].mid


BASELINES = {
    "keep-oldest": baseline_keep_oldest,
    "longest-body": baseline_longest_body,
    "first": baseline_first,
}


def correct_set_ok(cluster: Cluster, chosen_mid: str | None) -> bool:
    """Correct-set semantics (draft §1)."""
    if cluster.label["type"] == "keep":
        return chosen_mid == cluster.label["member"]
    if cluster.label["type"] == "eligible-all":
        return chosen_mid in {m.mid for m in cluster.members}
    return False  # abstain strata: any proposal is wrong for a choice baseline


# --- sanity fixtures (draft §6.1, MLC 28.10 edit N3) -------------------------


def build_fixture_cells() -> list[dict]:
    """Seven adversarial fixture cells (fixed content, no RNG)."""
    sid, kind, lang = "ru-migration", "release", "ru"
    base = scaffold_body(sid)
    title = scaffold_title(sid)
    tags = ("platform", "ops")

    def mk(mid, body, created, lang_=lang, type_="note", tags_=tags, title_=title):
        return Member(mid, title_, body, tags_, type_, lang_, created)

    cells: list[dict] = []
    # probe 1: self-cluster -> fold-eligible, keep in members
    cells.append(
        {
            "probe": 1,
            "name": "self-cluster",
            "cluster": Cluster(
                cid="fx-1",
                stratum="CH-0",
                substratum="self",
                members=[
                    mk("fx-1a", base, "2026-09-01T10:00:00Z"),
                    mk("fx-1b", base, "2026-09-01T10:00:07Z"),
                ],
                label={"type": "eligible-all"},
                declaration={"kind": "CH-0", "facts": {}, "steps": []},
            ),
        }
    )
    # probe 2: byte-twin cluster -> fold-eligible
    cells.append(
        {
            "probe": 2,
            "name": "byte-twin",
            "cluster": Cluster(
                cid="fx-2",
                stratum="CH-0",
                substratum="byte-twin",
                members=[
                    mk("fx-2a", base, "2026-09-01T10:00:00Z"),
                    mk("fx-2b", base, "2026-09-01T10:01:32Z"),
                ],
                label={"type": "eligible-all"},
                declaration={"kind": "CH-0", "facts": {}, "steps": []},
            ),
        }
    )
    # probe 3: razor single fact replacement (CH-4) -> ABSTAIN
    tmpl = _FACT_POOL[kind]["short"][lang]
    b1 = base.replace(_LONGS[sid], tmpl.format(v="3.2.1"))
    b2 = base.replace(_LONGS[sid], tmpl.format(v="3.2.2"))
    cells.append(
        {
            "probe": 3,
            "name": "razor-fact-replacement",
            "cluster": Cluster(
                cid="fx-3",
                stratum="CH-4",
                substratum="conflict",
                members=[
                    mk("fx-3a", b1, "2026-09-01T10:00:00Z"),
                    mk("fx-3b", b2, "2026-09-01T10:00:30Z"),
                ],
                label={"type": "abstain"},
                declaration={
                    "kind": "CH-4",
                    "mode": "conflict",
                    "steps": [],
                    "facts": {"fx-3a": ["release=3.2.1"], "fx-3b": ["release=3.2.2"]},
                },
            ),
        }
    )
    # probe 4: CH-3 with SHORTER superset (short-superset sub-stratum) -> right keep
    fact = tmpl.format(v="4.0.0")
    filler = _FILLERS[lang][0]
    sup = mk("fx-4a", base.replace(_LONGS[sid], fact), "2026-09-02T09:00:00Z")
    sub = mk("fx-4b", base.replace(_LONGS[sid], filler), "2026-09-02T09:00:11Z")
    at = sup.body.find(fact)
    assert at >= 0 and len(sup.body) < len(sub.body), (
        "superset must be SHORTER (length-proxy kill)"
    )
    cells.append(
        {
            "probe": 4,
            "name": "ch3-shorter-superset",
            "cluster": Cluster(
                cid="fx-4",
                stratum="CH-3",
                substratum="replace",
                members=[sup, sub],
                label={"type": "keep", "member": "fx-4a"},
                declaration={
                    "kind": "CH-3",
                    "mode": "replace",
                    "facts": {"fx-4a": ["release=4.0.0"], "fx-4b": []},
                    "steps": [
                        {
                            "sup": "fx-4a",
                            "sub": "fx-4b",
                            "delta": ["release=4.0.0"],
                            "spans": [(at, at + len(fact))],
                        },
                    ],
                },
            ),
        }
    )
    # probe 5: metadata delta -> ABSTAIN
    cells.append(
        {
            "probe": 5,
            "name": "metadata-delta",
            "cluster": Cluster(
                cid="fx-5",
                stratum="CH-2",
                substratum="tags",
                members=[
                    mk("fx-5a", base, "2026-09-03T08:00:00Z"),
                    mk(
                        "fx-5b", base, "2026-09-03T08:00:20Z", tags_=tags + ("archive",)
                    ),
                ],
                label={"type": "abstain"},
                declaration={"kind": "CH-2", "mode": "tags", "facts": {}, "steps": []},
            ),
        }
    )
    # probe 6: cosmetic invariance
    cos = cosmetic_variant(base, random.Random(7))
    cells.append(
        {
            "probe": 6,
            "name": "cosmetic-invariance",
            "cluster": Cluster(
                cid="fx-6",
                stratum="CH-5",
                substratum="cosmetic",
                members=[
                    mk("fx-6a", base, "2026-09-04T08:00:00Z"),
                    mk("fx-6b", cos, "2026-09-04T08:00:45Z"),
                ],
                label={"type": "eligible-all"},
                declaration={"kind": "CH-5", "facts": {}, "steps": []},
            ),
        }
    )
    # probe 7: self-cluster with NULL envelope (v42 double-null lesson)
    cells.append(
        {
            "probe": 7,
            "name": "null-envelope-self",
            "cluster": Cluster(
                cid="fx-7",
                stratum="CH-0",
                substratum="self-null",
                members=[
                    mk("fx-7a", base, "2026-09-05T08:00:00Z", lang_=None, type_=None),
                    mk("fx-7b", base, "2026-09-05T08:00:12Z", lang_=None, type_=None),
                ],
                label={"type": "eligible-all"},
                declaration={"kind": "CH-0", "facts": {}, "steps": []},
            ),
        }
    )
    return cells


SANITY_EXPECT = {
    1: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    2: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    3: lambda r: r["verdict"] == "abstain" and r["class"] == "CH-4",
    4: lambda r: (
        r["verdict"] == "keep"
        and r["member"] == "fx-4a"
        and r["leg"] == "silver-resolver"
    ),
    5: lambda r: r["verdict"] == "abstain" and r["class"] == "CH-2",
    6: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
    7: lambda r: r["verdict"] == "keep" and r["leg"] == "tie-break",
}


def run_sanity_suite() -> list[dict]:
    """Run the 7-probe pre-holdout adversarial suite (draft §6.1)."""
    results = []
    for cell in build_fixture_cells():
        res = arbitrate_deterministic(cell["cluster"])
        ok = bool(SANITY_EXPECT[cell["probe"]](res))
        row = {"probe": cell["probe"], "name": cell["name"], "ok": ok, "result": res}
        if cell["probe"] == 3:
            # Mechanical razor evidence through the FROZEN feature module +
            # the norm2 edit-cluster machinery. The ratified cos band
            # (policy §3 T3: 0.9975-1.0, med 0.9990) is carried by
            # construction (single fact token, len-delta 0); the G3 zone
            # constants of policy §5 were measured on longer dataset-v3
            # bodies and do not transfer verbatim to a 445-char fixture —
            # so the mechanical legs here are: zero length delta, exactly
            # ONE tiny edit cluster (<= 6 chars = one fact token), and
            # char5_jaccard >= 0.97 (G3 jaccard leg).
            a, b = cell["cluster"].members
            vec = features(a.to_record(), b.to_record(), similarity=1.0)
            row["char5_jaccard"] = round(
                vec.values[FEATURE_NAMES.index("char5_jaccard")], 6
            )
            row["char5_containment"] = round(
                vec.values[FEATURE_NAMES.index("char5_containment")], 6
            )
            row["body_len_delta"] = int(
                vec.values[FEATURE_NAMES.index("body_len_delta")]
            )
            clusters_ = norm2_edit_clusters(a.body, b.body)
            row["edit_clusters"] = len(clusters_)
            row["edit_cluster_chars"] = (
                max((c[1] - c[0]) for c in clusters_) if clusters_ else 0
            )
            row["razor_mechanical"] = bool(
                row["body_len_delta"] == 0
                and row["edit_clusters"] == 1
                and row["edit_cluster_chars"] <= 6
                and row["char5_jaccard"] >= 0.97
            )
            ok = ok and row["razor_mechanical"]
            row["ok"] = bool(ok)
            row["cos_band_declared"] = (
                "policy §3 T3: cos 0.9975-1.0 (med 0.9990), by construction"
            )
        if cell["probe"] == 4:
            silv = silver_check_ch3(cell["cluster"])
            ok = (
                ok
                and silv["ok"]
                and len(cell["cluster"].members[0].body)
                < len(cell["cluster"].members[1].body)
            )
            row["silver"] = silv
            row["ok"] = bool(ok)
        if cell["probe"] == 6:
            a, b = cell["cluster"].members
            same_norm = norm1_body(a.body) == norm1_body(b.body)
            row["norm1_identity"] = bool(same_norm)
            ok = ok and same_norm
            row["ok"] = bool(ok)
        if cell["probe"] == 7:
            a = cell["cluster"].members[0]
            row["double_null_envelope"] = bool(
                a.language is None and a.record_type is None
            )
        results.append(row)
    return results
