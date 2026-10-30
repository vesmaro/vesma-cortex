#!/usr/bin/env python3
"""Merge-arbiter corpus v1.1 — the theme-near negative strata (ADDENDUM-3).

Ratified recipe: docs/specs/merge-arbiter-corpus-prereg-ADDENDUM-3.md (MLC
29.10, option (б), unanimous). 200 abstain-by-construction clusters SUPER
on the untouched 600 (extension, not re-emit — train.jsonl stays
byte-identical, id stability of the sheet pre-check preserved):

  cross-lang-sibling 80 (40 authored spec pairs x 2 orientations)
  template-twin       60 (30 pairs x 2 orientations)
  near-topic          60 (30 pairs x 2 orientations)

Class 8 of taxonomy §A (theme-near: different content, same topic): NOT
CH-4 (the razor predicate needs near-identical bodies — the deltas here
are whole regions), NOT CH-2 (metadata equal within a pair, bodies
differ). stratum="theme-near", label={"type": "abstain"}; the
declaration carries a non-empty justification validated by the v4.3
discipline (gen_dataset_v43_strategies: justification must be a
non-empty string) — A-labels extended per §1.

CONSTRUCTION (frozen generators reused, zero edits to them — A-frozen;
the new authored content lives HERE, in the ratified publication class
"authored synthetic text", merge_arbiter_lib precedent):

- cross-lang-sibling: 40 authored spec pairs (12 scaffold topics x
  3-4 value variants; direction fixed per topic: 22 ru->en, 18 en->ru).
  Same topic, DIFFERENT meaning (the frames state opposite states —
  planned vs cancelled, enabled vs retired), different values on the two
  sides. Validated per side by gen_dataset_v43_strategies._validate_side.
- template-twin: same scaffold, the {LONG} region replaced by two
  DIFFERENT digit-free filler paragraphs (authored HERE, disjoint from
  lib._FILLERS) — high surface similarity, no fact on either side.
- near-topic: same scaffold + a DIFFERENT appended alternate sentence on
  each side (digit-free, authored HERE) — same topic, different added
  content, no containment between the sides (vs CH-3/insert).

A-neg-silver — which predicate is applied (documented per §2): every
negative pair must FAIL the silver-dup predicate of labeling-policy §A
as mechanized in the corpus build (norm1 body+title identity AND
tags/type/language equality AND the razor class). The pairs fail it on
the surface-identity leg: silver-dup REQUIRES norm1 identity, the
negatives differ in norm1 body by construction (cross-lang pairs also
fail the language-equality leg). The embedder-cos leg (cos < 0.92) is
covered structurally — the census admits a pair only through cos >= 0.92
AND surface identity, so failing surface identity fails the predicate
for ANY cos; this environment computes no embeddings (the embedder pin
stays declared, A-frozen). Belt-and-suspenders vs the razor class:
char5_jaccard <= 0.90 asserted on every negative pair through the FROZEN
feature module (razor proxy band starts at ~0.97, T3 cos band
0.9975+).

Both orientations on every pair (A-symmetry extended); created_at
randomized independently of the label (A-created-indep/A-order-indep).
Seeds are FRESH (!= the 42 family, != 43..52 floor family) and are
pinned in the run-log and the sidecar manifest BEFORE generation.

Asserts before emission (fail-closed, §2): A-neg-silver (above),
A-neg-disjoint (0 body-hash vs train.jsonl bodies, vs the noharm surface
146a8ae4… bodies, vs raw scaffold bodies; 0 cluster/member id vs the
holdout id-manifest and vs sheet member_ids read from
sheet-selection.json at ID LEVEL ONLY — facts logged), A-count-v1.1
(quota table), preflight on a fixture cell of each new substratum (П3)
before emission.

Product: data/merge-arbiter/corpus-v1/train-negatives.jsonl (gitignored,
durable; train.jsonl NOT touched) + sidecar manifest
train-negatives.manifest.json (seeds, quotas, fp, regime, assert trail)
+ run-log events. Single-shot stays unexecuted; MA-holdout-choice
content is never read (the id-manifest read is id-only, logged);
spot-sheet render/verdicts/internal/committee-scored are never opened.

Usage:
  <venv-python> scripts/merge_arbiter_negatives.py build
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import merge_arbiter_train as mt  # noqa: E402  (helpers + frozen corpus access)

import merge_arbiter_lib as lib  # noqa: E402  (frozen scaffolds/Member/Cluster)
from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402
from gen_dataset_v43_strategies import _validate_side  # noqa: E402 (frozen validator)

NEG_PATH = mt.CORPUS_DIR / "train-negatives.jsonl"
NEG_MANIFEST = mt.CORPUS_DIR / "train-negatives.manifest.json"
NOHARM_SURFACE = mt.ART_DIR / "noharm-surface-v1.jsonl"
SHEET_SELECTION = (
    REPO / "data" / "merge-arbiter" / "spot-sheet" / "sheet-selection.json"
)
SLICE = "merge-arbiter-corpus-v1-1-negatives"

#: Fresh seeds, pinned BEFORE generation (run-log + sidecar manifest).
#: None of them is in the 42 family (corpus/holdout) or the 43..62 floor
#: family — they are generation seeds for this strata only.
SEED_CROSS_LANG = 9001
SEED_TEMPLATE_TWIN = 9002
SEED_NEAR_TOPIC = 9003
SEED_LIST = [SEED_CROSS_LANG, SEED_TEMPLATE_TWIN, SEED_NEAR_TOPIC]
#: iteration-2 regeneration seeds (fresh; pinned before re-emission)
SEED_LIST_ITER2 = [9101, 9102, 9103]

#: Per-substratum quotas (pairs; clusters = pairs x 2 orientations).
QUOTA_PAIRS = {"cross-lang-sibling": 40, "template-twin": 30, "near-topic": 30}

#: CV-noregress tolerance — FIXED BEFORE RETRAIN (CV-noregress, §2).
#: 0.02 = one acc quantum of a ~50-cluster fold (252 CH-3 clusters / 5
#: folds); the reference is the old artifact's OOF CH-3 acc = 1.0 per
#: fold (aggregate 1.0 over cluster quanta implies every fold 1.0).
NOREGRESS_TOL = 0.02

_THEME_NEAR_SUBSTRATA = ("cross-lang-sibling", "template-twin", "near-topic")

# --- authored content (publication class: authored synthetic text) -----------

#: Cross-language sibling topics. Frame A = record side (scaffold
#: language), frame B = sibling side (the OTHER language, different
#: meaning on the same topic). {v} slots take DIFFERENT values per side.
CROSS_LANG_TOPICS = [
    dict(
        sid="ru-migration",
        just="the record plans the runtime cutover window, "
        "the sibling reports the same migration cancelled — opposite project states on one topic",
        a="Миграция сервиса на новый рантайм запланирована на окно {v}: сначала интеграционные тесты, затем читающие клиенты.",
        b="The runtime migration was postponed: the cutover window of {v} minutes is cancelled and clients stay on the current runtime this quarter.",
    ),
    dict(
        sid="ru-retention",
        just="the record states the automatic archive rule, the sibling announces the rule retired — the retention policy moved in opposite directions",
        a="Черновики старше {v} дней архивируются автоматически, архив читается поиском и не показывается в выдаче.",
        b="Draft retention changed: nothing is archived automatically anymore, the {v}-day rule is retired and cleanup is a manual monthly task.",
    ),
    dict(
        sid="ru-review-cfg",
        just="the record mandates config review with a window, the sibling drops the mandate for hotfixes — different effective policies",
        a="Ревью конфигов обязательно для правок портов и лимитов, окно ревью — {v} часов, отметка в журнале изменений.",
        b="Config review policy is dropped for hotfixes: the {v}-hour review window now applies only to scheduled changes, log marks are optional.",
    ),
    dict(
        sid="ru-flags",
        just="the record starts the flag cleanup, the sibling freezes it — opposite execution states of the same cleanup",
        a="Вынос фиче-флагов из legacy начнётся с флагов старше {v} дней, каждый вынос фиксируется отдельным коммитом.",
        b="Feature-flag cleanup is frozen: flags older than {v} days stay in the legacy module until the rewrite lands, no commits are cut.",
    ),
    dict(
        sid="ru-replicas",
        just="the record fixes the classic restart order, the sibling documents an inverted order — different runbooks for the same window",
        a="После окна включение реплик начинается с читающих контуров, контроль отставания — каждые {v} минут, мастер последним.",
        b="Replica restart order is inverted this time: border replicas come first, lag checks run every {v} minutes on the writer path, the master is promoted early.",
    ),
    dict(
        sid="ru-incident-labels",
        just="the record keeps minting retro labels, the sibling closes the experiment — opposite labeling regimes",
        a="Ретроспективные метки ставятся в течение {v} дней после пост-мортема, каждая метка несёт ссылку на разбор.",
        b="The retro-label experiment ended: no labels are minted after incidents anymore, the {v}-day trial window is closed and links live in the postmortem only.",
    ),
    dict(
        sid="en-oncall",
        just="the record keeps the manual handoff summary, the sibling retires it — opposite handoff mechanics",
        a="The on-call handoff moves to the shared channel; the rotating engineer posts the summary within {v} minutes of shift end.",
        b="Ручной хэндофф отменяют: сводка смены дежурным больше не отправляется, окно {v} минут упразднено, состояние читают из дашборда.",
    ),
    dict(
        sid="en-cache",
        just="the record keeps manual purge with a soft TTL, the sibling switches to event-driven invalidation — different cache regimes",
        a="Edge cache purge stays manual; soft TTL is {v} minutes for hot paths, pricing pages are never cached.",
        b="Кэш переводят на автоматическую инвалидацию по событию публикации, ручной пургатор убирают, TTL {v} минут сохраняется только для статики.",
    ),
    dict(
        sid="en-backups",
        just="the record keeps quarterly drills, the sibling replaces them with an annual game day — different drill schedules",
        a="Quarterly restore drills continue; the next full drill must finish within {v} hours including the report.",
        b="Расписание drills меняют: полные восстановления реже, годовой game day заменяет квартальные, окно {v} часов на отчёт не лимитирует.",
    ),
    dict(
        sid="en-deprecation",
        just="the record applies the sunset header to all consumers, the sibling narrows it to paid ones — different deprecation scopes",
        a="Deprecated endpoints keep the sunset header {v} days ahead; zero-traffic consumers are contacted before removal.",
        b="Политику деприкейшена смягчили: sunset-заголовок за {v} дней нужен только для платных потребителей, остальные отключаются без предупреждения.",
    ),
    dict(
        sid="en-quotas",
        just="the record keeps the batch fast lane, the sibling removes it — opposite rate-limit outcomes",
        a="Default burst stays at the negotiated limit; batch jobs get the fast lane only within the {v} minute maintenance window.",
        b="Лимиты пересмотрели: батч-задачи теряют быстрый lane, окно обслуживания {v} минут больше не даёт исключений, burst считается по базовому тарифу.",
    ),
    dict(
        sid="en-docs",
        just="the record reports the migration done and moving on, the sibling reports it suspended — opposite project states",
        a="Docs migration is done for architecture notes; runbooks move next sprint, redirects re-index within {v} days.",
        b="Миграцию документации приостановили: runbooks остаются на месте, редиректы за {v} дней не устанавливаются, план переписывается.",
    ),
]

#: Value variants per topic index: (record_value, sibling_value) — always
#: different across the sides (the semantic difference is carried by the
#: frames; the values add per-variant distinctness). Topics 0-3 carry 4
#: variants, topics 4-11 carry 3 → 4*4 + 8*3 = 40 specs.
_VALUE_SETS_LONG = [("30", "45"), ("20", "50"), ("15", "40"), ("60", "90")]
_VALUE_SETS_SHORT = [("30", "45"), ("20", "50"), ("15", "40")]


#: Iteration-2 diversification (ADDENDUM-3 iter-guard, 2 of 2): the
#: cross-lang siblings get their OWN titles in the sibling language
#: (wording and length differ from the record title — the diagnosis of
#: iteration 1: identical scaffold titles narrowed the training band,
#: title_len_delta 0 in train vs 6.75 on the surface, and near-zero
#: n-gram-overlap pairs were outside it). Body frames now span the
#: overlap spectrum: digit-free sides and digit-bearing sides, near-equal
#: and asymmetric lengths.
_SIB_TITLES = {
    "ru-migration": "Переезд на новый рантайм: статус и окно",
    "ru-retention": "Как теперь живут черновики и их архив",
    "ru-review-cfg": "Порядок согласования изменений конфигов",
    "ru-flags": "Судьба фиче-флагов в legacy-модуле",
    "ru-replicas": "План включения реплик после обслуживания",
    "ru-incident-labels": "Метки инцидентов: что решили с ретроспективой",
    "en-oncall": "Смена передачи дежурства: договорились",
    "en-cache": "Инвалидация кэша: выбранный подход",
    "en-backups": "Расписание восстановлений и учений",
    "en-deprecation": "Отключение старых эндпоинтов: правила",
    "en-quotas": "Пересмотр лимитов для внутренних потребителей",
    "en-docs": "Куда поехала документация и что осталось",
}


def _cross_lang_specs() -> list[dict]:
    specs = []
    for ti, topic in enumerate(CROSS_LANG_TOPICS):
        sets = _VALUE_SETS_LONG if ti < 4 else _VALUE_SETS_SHORT
        for vi, (va, vb) in enumerate(sets):
            a_lang = "ru" if topic["sid"].startswith("ru") else "en"
            b_lang = "en" if a_lang == "ru" else "ru"
            a_title = lib.scaffold_title(topic["sid"])
            b_title = _SIB_TITLES[topic["sid"]]
            # the sibling title rides in the SIBLING language; when the
            # record is en, the authored sibling title is ru — swap the
            # length/wording asymmetry per variant instead of per topic
            if vi % 2 == 1:
                a_title = (
                    f"{a_title} (черновик)" if a_lang == "ru" else f"{a_title} (draft)"
                )
            specs.append(
                {
                    "key": f"xn-{topic['sid']}-{vi}",
                    "record": {
                        "lang": a_lang,
                        "title": a_title,
                        "body": topic["a"].format(v=va),
                        "tags": ["platform", "notes"],
                        "record_type": "note",
                    },
                    "sibling": {
                        "lang": b_lang,
                        "title": b_title,
                        "body": topic["b"].format(v=vb),
                        "tags": ["platform", "notes"],
                        "record_type": "note",
                    },
                    "justification": topic["just"],
                }
            )
    return specs


#: Digit-free filler paragraphs for template-twin (authored HERE; disjoint
#: from lib._FILLERS strings; no digits — the anti-proxy: no fact on
#: either side of the pair).
_TWIN_FILLERS = [
    "Совещание по этому поводу перенесли на конец недели, повестка прежняя, участники те же.",
    "Обсуждение идёт в рабочем канале, ключевые аргументы уже зафиксированы в треде выше.",
    "К пятнице соберу выжимку по обсуждению и приложу её рядом с этой записью.",
    "Пока идём в текущем порядке, изменения ждем после ревью смежного плана.",
    "Замечаний по срокам нет, все участники подтвердили прежние оценки в переписке.",
    "Вопрос закрыт на уровне команды, но формулировку оставляем черновой до ревью.",
    "Дополнительный контекст по этому решению собирается в соседнем документе.",
    "Если расписание сдвинется, обновление будет в этой же записи, без новых копий.",
    "Смежная команда просила учитывать их окно при планировании следующих шагов.",
    "Итог обсуждения пока не зафиксирован, ждем ответы на открытые вопросы треда.",
    "Отдельная встреча не планируется, статус разбирается на регулярном созвоне.",
    "Детали согласования уточняются, черновая версия договора остается в силе.",
]

#: Digit-free alternate closing sentences for near-topic (authored HERE).
_TOPIC_ALTS = [
    "К этой записи вернусь после ретроспективы и допишу выводы отдельным блоком.",
    "Связанные заметки держу рядом, чтобы история обсуждения восстанавливалась поиском.",
    "Если подход сменится, эта запись обновится, а не заведется заново.",
    "Ссылки на смежные договоренности соберу в конце, когда список устаканится.",
    "Комментарии коллег из треда переношу сюда, чтобы ничего не потерялось.",
    "Черновой план следующего этапа намечу в этом же документе на неделе.",
    "Отдельный канон для этого правила заведем позже, пока живёт здесь.",
    "По итогам проверки добавлю короткий отчет с таймингами рядом с записью.",
    "Пока фиксирую текущую договоренность как рабочую, не как финальную.",
    "Наблюдения за первыми неделями сложу в этот же раздел, без отдельной заметки.",
    "Вопрос верну на ревью, если смежная команда пришлет возражения письменно.",
    "Держу этот блок как единственное место правок, чтобы не плодить дубли.",
]


# --- construction ------------------------------------------------------------


def _mk_member(
    mid: str, title: str, body: str, tags, rtype, lang, ts: str
) -> lib.Member:
    return lib.Member(
        mid=mid,
        title=title,
        body=body,
        tags=tuple(tags),
        record_type=rtype,
        language=lang,
        created_at=ts,
    )


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime(lib.ISO_FMT)


def _emit_both_orientations(
    prefix: str,
    stratum_pair: str,
    just: str,
    scenario: str,
    side_a: dict,
    side_b: dict,
    rng: random.Random,
) -> list[lib.Cluster]:
    """Each pair is emitted in BOTH orientations (A-symmetry extended to
    the negative strata, ADDENDUM-3 §1) — mirrored scenario rows with
    FRESH ids per orientation (the gen_family convention)."""
    lo, hi = lib._CREATED_LO, lib._CREATED_HI - lib._CREATED_SPREAD
    out = []
    for orient in ("AB", "BA"):
        mids = (f"{prefix}-a-{orient}", f"{prefix}-b-{orient}")
        members = [
            _mk_member(
                mids[0],
                side_a["title"],
                side_a["body"],
                side_a["tags"],
                side_a["record_type"],
                side_a["language"],
                _iso(rng.randint(lo, hi)),
            ),
            _mk_member(
                mids[1],
                side_b["title"],
                side_b["body"],
                side_b["tags"],
                side_b["record_type"],
                side_b["language"],
                _iso(rng.randint(lo, hi)),
            ),
        ]
        if orient == "BA":
            members.reverse()
        out.append(
            lib.Cluster(
                cid=f"{prefix}-{orient}",
                stratum="theme-near",
                substratum=stratum_pair,
                members=members,
                label={"type": "abstain"},
                declaration=_decl(stratum_pair, just),
                scenario=scenario,
                orientation=orient,
            )
        )
    return out


def _decl(substratum: str, justification: str) -> dict:
    return {
        "kind": "theme-near",
        "mode": substratum,
        "facts": {},
        "steps": [],
        "justification": justification,
    }


def _build_cross_lang(seed: int) -> list[lib.Cluster]:
    rng = random.Random(seed)
    clusters = []
    for spec in _cross_lang_specs():
        rec = dict(spec["record"])
        sib = dict(spec["sibling"])
        _validate_side(rec, "neg.record", spec)  # frozen v4.3 validator
        _validate_side(sib, "neg.sibling", spec)
        if (
            not isinstance(spec["justification"], str)
            or not spec["justification"].strip()
        ):
            raise mt.InstrumentAbort(f"{spec['key']}: empty justification")
        if rec["lang"] == sib["lang"]:
            raise mt.InstrumentAbort(f"{spec['key']}: cross-lang pair with equal langs")
        side_a = {
            "title": rec["title"],
            "body": rec["body"],
            "tags": rec["tags"],
            "record_type": rec["record_type"],
            "language": rec["lang"],
        }
        side_b = {
            "title": sib["title"],
            "body": sib["body"],
            "tags": sib["tags"],
            "record_type": sib["record_type"],
            "language": sib["lang"],
        }
        clusters.extend(
            _emit_both_orientations(
                f"ma-neg-xn-{spec['key']}",
                "cross-lang-sibling",
                spec["justification"],
                f"{spec['key']}|cross-lang|{rec['lang']}-{sib['lang']}",
                side_a,
                side_b,
                rng,
            )
        )
    return clusters


def _build_twin(seed: int) -> list[lib.Cluster]:
    rng = random.Random(seed)
    clusters = []
    combos: set[str] = set()
    n_f = len(_TWIN_FILLERS)
    # deterministic pair assignment: scaffold x (fillerA, fillerB), unique combos
    pair_no = 0
    for si in range(len(lib.SCAFFOLDS)):
        for fa, fb in ((i, (i + 1 + si) % n_f) for i in range(n_f)):
            if fa == fb:
                continue
            if pair_no >= QUOTA_PAIRS["template-twin"]:
                break
            combo = f"{si}:{fa}:{fb}"
            if combo in combos:
                continue
            combos.add(combo)
            sid = lib.SCAFFOLDS[si]["sid"]
            lang = next(s["lang"] for s in lib.SCAFFOLDS if s["sid"] == sid)
            title = lib.scaffold_title(sid)
            tags = ("platform", "ops")
            # scaffold_body() already substituted {LONG} — re-open the slot
            # region via the SAME mechanism the frozen lib uses for its
            # replace-mode construction (lib._LONGS[sid]), then fill it.
            base = lib.scaffold_body(sid).replace(lib._LONGS[sid], "{LONG}")
            body_a = base.replace("{LONG}", _TWIN_FILLERS[fa])
            body_b = base.replace("{LONG}", _TWIN_FILLERS[fb])
            pair_no += 1
            side_a = {
                "title": title,
                "body": body_a,
                "tags": tags,
                "record_type": "note",
                "language": lang,
            }
            side_b = {
                "title": title,
                "body": body_b,
                "tags": tags,
                "record_type": "note",
                "language": lang,
            }
            just = (
                "same scaffold template, two different filler paragraphs in the "
                f"variable region ({fa} vs {fb}) — no fact on either side, "
                "different content, same topic"
            )
            clusters.extend(
                _emit_both_orientations(
                    f"ma-neg-tw-{pair_no:03d}",
                    "template-twin",
                    just,
                    f"{sid}|twin|{fa}|{fb}",
                    side_a,
                    side_b,
                    rng,
                )
            )
        if pair_no >= QUOTA_PAIRS["template-twin"]:
            break
    if len(clusters) != 2 * QUOTA_PAIRS["template-twin"]:
        raise mt.InstrumentAbort(
            f"template-twin clusters: {len(clusters)}, "
            f"want {2 * QUOTA_PAIRS['template-twin']}"
        )
    return clusters


def _build_near(seed: int) -> list[lib.Cluster]:
    rng = random.Random(seed)
    clusters = []
    pair_no = 0
    n_alt = len(_TOPIC_ALTS)
    for si in range(len(lib.SCAFFOLDS)):
        for k in range(3):  # 12 scaffolds x 3 combos = 36 >= 30; cut at quota
            if pair_no >= QUOTA_PAIRS["near-topic"]:
                break
            ia = (si * 3 + k) % n_alt
            ib = (si * 3 + k + 5) % n_alt
            if ia == ib:
                continue
            sid = lib.SCAFFOLDS[si]["sid"]
            lang = next(s["lang"] for s in lib.SCAFFOLDS if s["sid"] == sid)
            title = lib.scaffold_title(sid)
            tags = ("platform", "notes")
            base = lib.scaffold_body(sid)
            side_a = {
                "title": title,
                "body": base + " " + _TOPIC_ALTS[ia],
                "tags": tags,
                "record_type": "note",
                "language": lang,
            }
            side_b = {
                "title": title,
                "body": base + " " + _TOPIC_ALTS[ib],
                "tags": tags,
                "record_type": "note",
                "language": lang,
            }
            pair_no += 1
            just = (
                "same topic, different added closing statement "
                f"({ia} vs {ib}) — neither side contains the other, "
                "different content, same topic"
            )
            clusters.extend(
                _emit_both_orientations(
                    f"ma-neg-nt-{pair_no:03d}",
                    "near-topic",
                    just,
                    f"{sid}|near|{ia}|{ib}",
                    side_a,
                    side_b,
                    rng,
                )
            )
        if pair_no >= QUOTA_PAIRS["near-topic"]:
            break
    if len(clusters) != 2 * QUOTA_PAIRS["near-topic"]:
        raise mt.InstrumentAbort(
            f"near-topic clusters: {len(clusters)}, "
            f"want {2 * QUOTA_PAIRS['near-topic']}"
        )
    return clusters


# --- asserts (§2, fail-closed) ------------------------------------------------


def _char5(a: lib.Member, b: lib.Member) -> float:
    vec = features(
        RecordLike(a.title, a.body, a.tags, a.language, a.record_type),
        RecordLike(b.title, b.body, b.tags, b.language, b.record_type),
        similarity=1.0,
    )
    return vec.values[FEATURE_NAMES.index("char5_jaccard")]


def assert_neg_silver(clusters: list[lib.Cluster]) -> None:
    """A-neg-silver: every pair FAILS the §A silver-dup predicate.

    Applied predicate (see module docstring): the corpus census admits a
    dup only through norm1 body+title identity AND tags/type/language
    equality (then razor checks); the negatives fail the norm1-identity
    leg by construction (cross-lang also the language leg). Mechanical
    guard: char5_jaccard <= 0.90 (razor proxy band ~0.97+) through the
    frozen feature module. The embedder cos<0.92 leg is structural (no
    embeddings computed here — pin stays declared, A-frozen).
    """
    worst = 0.0
    for c in clusters:
        a, b = c.members[0], c.members[1]
        same_norm = lib.norm1_body(a.body) == lib.norm1_body(b.body)
        same_env = (
            lib.norm1_body(a.title) == lib.norm1_body(b.title)
            and frozenset(a.tags) == frozenset(b.tags)
            and a.record_type == b.record_type
            and a.language == b.language
        )
        if same_norm and same_env:
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} passes the dup surface-identity leg"
            )
        j = _char5(a, b)
        worst = max(worst, j)
        if j > 0.90:
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} char5_jaccard={j:.4f} > 0.90 "
                "(razor band proximity)"
            )
    mt.log(
        "a_neg_silver",
        pairs=len(clusters),
        max_char5_jaccard=round(worst, 4),
        predicate="fails §A dup: norm1 body identity leg (cross-lang: also "
        "language leg); char5 <= 0.90 razor guard; cos<0.92 leg "
        "structural (census admits only through surface identity)",
    )


def _body_hashes(clusters: list[lib.Cluster]) -> set[str]:
    return {mt.sha256_text(m.body) for c in clusters for m in c.members}


def assert_neg_disjoint(
    clusters: list[lib.Cluster],
    surface_hashes: set[str],
    train_hashes: set[str],
    scaffold_hashes: set[str],
    holdout_ids: set[str],
    sheet_ids: set[str],
) -> None:
    """A-neg-disjoint: 0 body-hash vs noharm surface / train / raw
    scaffolds; 0 cluster+member id vs holdout id-manifest / sheet
    member_ids (sheet read at ID LEVEL ONLY from sheet-selection.json)."""
    neg_ids = {c.cid for c in clusters} | {m.mid for c in clusters for m in c.members}
    neg_h = _body_hashes(clusters)
    hits_surface = neg_h & surface_hashes
    hits_train = neg_h & train_hashes
    hits_scaffold = neg_h & scaffold_hashes
    hits_holdout_ids = neg_ids & holdout_ids
    hits_sheet_ids = neg_ids & sheet_ids
    mt.log(
        "a_neg_disjoint",
        body_hashes=len(neg_h),
        hits_surface=sorted(hits_surface)[:3],
        hits_train=sorted(hits_train)[:3],
        hits_raw_scaffold=sorted(hits_scaffold)[:3],
        hits_holdout_ids=sorted(hits_holdout_ids)[:3],
        hits_sheet_ids=sorted(hits_sheet_ids)[:3],
        sheet_read="sheet-selection.json, ID LEVEL ONLY, facts logged",
        note="scaffold overlap with the surface computed from the recipe: "
        "negatives reuse merge-arbiter scaffolds (authored class), the "
        "surface carries v4/v4.3 batch texts — 0 topic overlap expected",
    )
    if (
        hits_surface
        or hits_train
        or hits_scaffold
        or hits_holdout_ids
        or hits_sheet_ids
    ):
        raise mt.InstrumentAbort(
            f"A-neg-disjoint RED: surface={sorted(hits_surface)[:2]} "
            f"train={sorted(hits_train)[:2]} scaffold={sorted(hits_scaffold)[:2]} "
            f"holdout_ids={sorted(hits_holdout_ids)[:2]} "
            f"sheet_ids={sorted(hits_sheet_ids)[:2]}"
        )


def preflight_fixtures() -> None:
    """П3: fixture cell of EACH new substratum through the full assert map
    BEFORE emission."""
    cells = []
    specs = _cross_lang_specs()
    topic = CROSS_LANG_TOPICS[0]
    rec = dict(specs[0]["record"])
    sib = dict(specs[0]["sibling"])
    cells.append(
        lib.Cluster(
            cid="fx-xn",
            stratum="theme-near",
            substratum="cross-lang-sibling",
            members=[
                _mk_member(
                    "fx-xn-a",
                    rec["title"],
                    rec["body"],
                    rec["tags"],
                    rec["record_type"],
                    rec["lang"],
                    lib._iso(lib._CREATED_LO),
                ),
                _mk_member(
                    "fx-xn-b",
                    sib["title"],
                    sib["body"],
                    sib["tags"],
                    sib["record_type"],
                    sib["lang"],
                    lib._iso(lib._CREATED_LO),
                ),
            ],
            label={"type": "abstain"},
            declaration=_decl("cross-lang-sibling", topic["just"]),
        )
    )
    twin = _build_twin(SEED_TEMPLATE_TWIN)[0]
    twin.cid = "fx-tw"
    twin.members[0].mid, twin.members[1].mid = "fx-tw-a", "fx-tw-b"
    cells.append(twin)
    near = _build_near(SEED_NEAR_TOPIC)[0]
    near.cid = "fx-nt"
    near.members[0].mid, near.members[1].mid = "fx-nt-a", "fx-nt-b"
    cells.append(near)

    all_ok = True
    for c in cells:
        # full map on the fixture: silver-fail, label-from-declaration,
        # class routing (undefined -> model leg), orientation balance
        a, b = c.members
        silver_fails = not (
            lib.norm1_body(a.body) == lib.norm1_body(b.body)
            and lib.norm1_body(a.title) == lib.norm1_body(b.title)
            and frozenset(a.tags) == frozenset(b.tags)
            and a.record_type == b.record_type
            and a.language == b.language
        )
        label_ok = (
            lib.correct_set_ok(c, None) is False
        )  # abstain strata: any keep wrong
        cls = lib.classify(c.members, declarations=None)
        routing_ok = cls == "undefined"
        rev = list(reversed(c.members))
        rev_decl = _decl(c.substratum, c.declaration["justification"])
        rev_c = lib.Cluster(
            cid=c.cid,
            stratum=c.stratum,
            substratum=c.substratum,
            members=rev,
            label=c.label,
            declaration=rev_decl,
        )
        decl_matches = (
            c.declaration["kind"] == "theme-near"
            and c.declaration["mode"] == c.substratum
            and bool(c.declaration["justification"].strip())
            and rev_c.declaration["justification"] == c.declaration["justification"]
        )
        row_ok = silver_fails and label_ok and routing_ok and decl_matches
        all_ok = all_ok and row_ok
        mt.log(
            "preflight_fixture",
            cell=c.cid,
            substratum=c.substratum,
            silver_fails=silver_fails,
            label_ok=label_ok,
            route_class=cls,
            routing_ok=routing_ok,
            decl_ok=decl_matches,
            ok=row_ok,
        )
    if not all_ok:
        raise mt.InstrumentAbort(
            "preflight RED on theme-near fixtures — emission forbidden (П3)"
        )
    mt.log(
        "preflight_pass",
        cells=3,
        note="fixture cells of every new substratum; full assert map green "
        "BEFORE emission",
    )


# --- emission -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "null-coverage"])
    ap.add_argument(
        "--iter2",
        action="store_true",
        help="ratified iter-guard re-emission (iteration 2 of 2)",
    )
    args = ap.parse_args(argv)
    if args.cmd == "null-coverage":
        return cmd_null_coverage(args)

    mt._SLICE = SLICE
    mt.log(
        "cost_declaration",
        expected_prolog_s=2,
        expected_cycle_s=30,
        expected_peak_ram_gb=1.0,
        lane="light (draft §9)",
    )

    # prologue (train-side: no sealed surface, no holdout content)
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    avail_mb = int(free.splitlines()[1].split()[6])
    mt.log(
        "swap_check",
        swapon_lines=swap.strip().splitlines(),
        available_mb=avail_mb,
        ram_overrun_stop=bool(avail_mb < 2000),
    )
    if avail_mb < 2000:
        raise mt.InstrumentAbort("RAM below floor")
    env = {
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "lightgbm": mt.lgb.__version__,
        "numpy": mt.np.__version__,
        "machine": platform.machine(),
    }
    mt.log("env", **env)
    mt.log("host", hostname=platform.node(), expectation=mt.EXPECTED_HOST)
    if platform.node() != mt.EXPECTED_HOST:
        raise mt.InstrumentAbort("host changed")
    pins = {
        "train_jsonl_sha256": mt.sha256_file(mt.TRAIN_JSONL),
        "manifest_sha256": mt.sha256_file(mt.MANIFEST),
        "noharm_surface_sha256": mt.sha256_file(NOHARM_SURFACE),
        "merge_arbiter_lib_sha256": mt.sha256_file(Path(mt.lib.__file__)),
        "gen_v43_strategies_sha256": mt.sha256_file(
            REPO / "scripts" / "gen_dataset_v43_strategies.py"
        ),
    }
    if (
        pins["train_jsonl_sha256"]
        != "e60bc7d05bd0348d8811d094b890c0931c9495b9ae03169951232bf547b93cd6"
    ):
        raise mt.InstrumentAbort(
            "train.jsonl diverges from the sealed v1 pin — extension discipline broken"
        )
    inputs = [
        "corpus manifest + train.jsonl (pins)",
        "noharm surface (hash only)",
        "sheet-selection.json (ID LEVEL ONLY)",
        "holdout id-manifest (ID ONLY, no content)",
        "frozen generators (lib, gen_dataset_v43_strategies)",
    ]
    mt.log(
        "frozen_inputs",
        pins=pins,
        input_list=inputs,
        committee_scored_in_inputs=False,
        sealed_inputs=[
            "NONE — MA-holdout-choice content, holdout v4.5, "
            "g4-spec-v1, spot-sheet render/verdicts/internal/"
            "committee-scored are NOT in the input list"
        ],
    )

    # ID-level reads (facts logged; no labels, no texts)
    holdout_ids = mt._holdout_id_manifest()
    sheet_doc = json.loads(SHEET_SELECTION.read_text(encoding="utf-8"))
    sheet_ids = set()
    for row in sheet_doc.get("synthetic", []):
        sheet_ids.update(row.get("member_ids", []))
    for row in sheet_doc.get("real", []):
        sheet_ids.update(row.get("store_ids", []))
    mt.log(
        "id_manifest_reads",
        holdout_ids=len(holdout_ids),
        sheet_member_ids=len(sheet_ids),
        sheet_source="sheet-selection.json (id lists only, no labels)",
        content_read=False,
    )

    # ratification assert: ADDENDUM-3 at HEAD (commit e953eec in ancestry)
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "e953eec", "HEAD"],
        cwd=REPO,
        capture_output=True,
    )
    addendum_ok = (
        r.returncode == 0
        and (
            REPO / "docs" / "specs" / "merge-arbiter-corpus-prereg-ADDENDUM-3.md"
        ).exists()
    )
    mt.log(
        "activation_addendum3",
        commit="e953eec",
        is_ancestor=r.returncode == 0,
        ok=addendum_ok,
    )
    if not addendum_ok:
        raise mt.InstrumentAbort(
            "ADDENDUM-3 not at HEAD — corpus extension has no ratified recipe"
        )

    # seeds pinned BEFORE generation
    iter2 = "--iter2" in sys.argv
    active_seeds = SEED_LIST_ITER2 if iter2 else SEED_LIST
    global SEED_CROSS_LANG, SEED_TEMPLATE_TWIN, SEED_NEAR_TOPIC
    if iter2:
        SEED_CROSS_LANG, SEED_TEMPLATE_TWIN, SEED_NEAR_TOPIC = active_seeds
    mt.log(
        "seeds_pinned",
        seeds=active_seeds,
        per_substratum={
            "cross-lang-sibling": SEED_CROSS_LANG,
            "template-twin": SEED_TEMPLATE_TWIN,
            "near-topic": SEED_NEAR_TOPIC,
        },
        fresh="!= 42 family, != 43..62 floor family",
        noregress_tol=NOREGRESS_TOL,
        note="pinned BEFORE generation (ADDENDUM-3 §3.1); CV-noregress "
        "tolerance fixed BEFORE retrain",
    )

    preflight_fixtures()

    clusters = (
        _build_cross_lang(SEED_CROSS_LANG)
        + _build_twin(SEED_TEMPLATE_TWIN)
        + _build_near(SEED_NEAR_TOPIC)
    )
    counts: dict[str, int] = {}
    for c in clusters:
        counts[c.substratum] = counts.get(c.substratum, 0) + 1
    if counts != {"cross-lang-sibling": 80, "template-twin": 60, "near-topic": 60}:
        raise mt.InstrumentAbort(f"A-count-v1.1 RED: {counts}")
    # A-symmetry extended: every scenario exactly 2 mirrored rows (AB + BA)
    by_scen: dict[str, dict[str, str]] = {}
    for c in clusters:
        by_scen.setdefault(c.scenario, {})[c.orientation] = c.cid
    if any(sorted(v) != ["AB", "BA"] for v in by_scen.values()):
        raise mt.InstrumentAbort(
            "A-symmetry RED: some scenario lacks a mirrored orientation row"
        )
    # mirrored rows carry the same two bodies
    body_by_cid = {c.cid: frozenset(m.body for m in c.members) for c in clusters}
    for scen, orients in by_scen.items():
        if body_by_cid[orients["AB"]] != body_by_cid[orients["BA"]]:
            raise mt.InstrumentAbort(
                f"A-symmetry RED: {scen} mirrored rows carry different bodies"
            )

    # disjointness base = the ORIGINAL v1 train.jsonl bodies (the previous
    # negatives iteration is the artifact being REPLACED, not train)
    train_v1 = [
        mt._cluster_from_row(json.loads(line))
        for line in open(mt.TRAIN_JSONL, encoding="utf-8")
    ]
    train_hashes = _body_hashes(train_v1)
    surface_rows = [
        json.loads(line)
        for line in open(NOHARM_SURFACE, encoding="utf-8")
        if line.strip()
    ]
    surface_hashes = {
        mt.sha256_text(r[k]) for r in surface_rows for k in ("body_a", "body_b")
    }
    scaffold_hashes = {
        mt.sha256_text(lib.scaffold_body(s["sid"])) for s in lib.SCAFFOLDS
    }
    assert_neg_silver(clusters)
    assert_neg_disjoint(
        clusters, surface_hashes, train_hashes, scaffold_hashes, holdout_ids, sheet_ids
    )

    # emission (durable, gitignored; train.jsonl untouched)
    iter2 = "--iter2" in sys.argv
    if NEG_PATH.exists() and not iter2:
        raise mt.InstrumentAbort(
            "train-negatives.jsonl exists — re-emission "
            "forbidden without the ratified iter-guard "
            "flag (--iter2, ADDENDUM-3: max 2 iterations)"
        )
    if iter2:
        mt.log(
            "iter2_reemission",
            reason="noharm QA iteration-1 RED 24/171: "
            "cross-lang training band too narrow (identical titles, "
            "no zero-overlap coverage); diversified per diagnosis",
            note="iteration 2 of max 2 (Iter-guard); next red => MLC design review",
        )
    with open(NEG_PATH, "w", encoding="utf-8") as fh:
        for c in clusters:
            fh.write(
                json.dumps(
                    {
                        "cluster_id": c.cid,
                        "k": len(c.members),
                        "label": c.label,
                        "members": [
                            {
                                "id": m.mid,
                                "title": m.title,
                                "body": m.body,
                                "tags": list(m.tags),
                                "record_type": m.record_type,
                                "language": m.language,
                                "created_at": m.created_at,
                            }
                            for m in c.members
                        ],
                        "declaration": c.declaration,
                        "stratum": c.stratum,
                        "substratum": c.substratum,
                        "scenario": c.scenario,
                        "orientation": c.orientation,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    neg_sha = mt.sha256_file(NEG_PATH)
    manifest = {
        "regime": "ADDENDUM-3 theme-near negative block: extension of corpus v1 "
        "(600 byte-untouched) with 200 abstain-by-construction "
        "clusters; frozen generators, authored negative content in "
        "this module; ML-STD-0002 П1-П6 map: A-neg-silver (surface-"
        "identity predicate leg + char5 razor guard), A-neg-disjoint "
        "(hash train/surface/scaffold + id holdout/sheet), A-count-v1.1, "
        "A-symmetry extended, A-created-indep/A-order-indep carried, "
        "preflight fixtures PASS before emission",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "seeds": SEED_LIST,
        "quota": {"cross-lang-sibling": 80, "template-twin": 60, "near-topic": 60},
        "n": len(clusters),
        "negatives_sha256": neg_sha,
        "corpus_fp_v1_1": mt.sha256_text(
            mt.sha256_file(mt.TRAIN_JSONL) + ":" + neg_sha
        ),
        "base_train_sha256": pins["train_jsonl_sha256"],
        "noregress_tol": NOREGRESS_TOL,
        "justification_validator": "gen_dataset_v43_strategies._validate_side "
        "+ non-empty justification (A-labels extended)",
    }
    NEG_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    mt.log(
        "negatives_emitted",
        path=str(NEG_PATH),
        sha256=neg_sha,
        counts=counts,
        corpus_fp_v1_1=manifest["corpus_fp_v1_1"],
        train_jsonl_untouched=True,
        note="extension, not re-emit; the 600 stay byte-identical",
    )
    return 0


# --- null-envelope coverage (ADDENDUM-4, terminal iteration 3) ---------------

NULL_PATH = mt.CORPUS_DIR / "null-envelope.json"
SLICE_NULL = "merge-arbiter-corpus-v1-1-null-coverage"
SEED_NULL = 9201  # fresh; pinned BEFORE selection

#: null quota formula: round(1/3 * stratum volume), pinned table
#: (A-null-indep 1/3 +/- 0.05 holds exactly by construction; every
#: theme-near substratum >= 20 coverage; CH-5 exactly 20).
NULL_QUOTA = {
    "CH-3": 84,
    "CH-0": 36,
    "CH-2": 30,
    "CH-4": 30,
    "CH-5": 20,
    "cross-lang-sibling": 27,
    "template-twin": 20,
    "near-topic": 20,
}


def _stratum_of(c) -> str:
    return c.substratum if c.stratum == "theme-near" else c.stratum


def cmd_null_coverage(args) -> int:
    """Select ~1/3 of clusters per stratum for the NULL envelope
    (both-null pairs only), write the override sidecar; base files stay
    byte-identical. Asserts A-null-indep / A-null-lang / A-null-cover /
    0-mixed-null / strict A-null-pin run HERE and again in the trainer
    prologue on the effective corpus."""
    mt._SLICE = SLICE_NULL
    mt.log(
        "cost_declaration",
        expected_prolog_s=2,
        expected_cycle_s=30,
        expected_peak_ram_gb=1.0,
        lane="light (draft §9)",
    )
    # ratification assert
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "8a64156", "HEAD"],
        cwd=REPO,
        capture_output=True,
    )
    ok = (
        r.returncode == 0
        and (
            REPO / "docs" / "specs" / "merge-arbiter-corpus-prereg-ADDENDUM-4.md"
        ).exists()
    )
    mt.log(
        "activation_addendum4", commit="8a64156", is_ancestor=r.returncode == 0, ok=ok
    )
    if not ok:
        raise mt.InstrumentAbort("ADDENDUM-4 not at HEAD")

    mt.log(
        "seeds_pinned",
        seeds=[SEED_NULL],
        purpose="null-cluster selection within strata",
        fresh="!= 42/43-62/9001-9103 families",
        note="pinned BEFORE selection",
    )

    clusters = mt.load_train()
    if NULL_PATH.exists():
        raise mt.InstrumentAbort(
            "null-envelope.json already exists — re-emission forbidden"
        )
    by_stratum: dict[str, list] = {}
    for c in clusters:
        by_stratum.setdefault(_stratum_of(c), []).append(c)
    got = {k: len(v) for k, v in sorted(by_stratum.items())}
    mt.log("stratum_volumes", volumes=got)

    rng = random.Random(SEED_NULL)
    chosen: list[str] = []
    for stratum, quota in sorted(NULL_QUOTA.items()):
        pool = sorted(c.cid for c in by_stratum[stratum])
        if quota > len(pool):
            raise mt.InstrumentAbort(
                f"null quota {quota} > volume {len(pool)} for {stratum}"
            )
        picks = rng.sample(pool, quota)
        chosen.extend(picks)
    chosen_set = set(chosen)

    # A-null-indep: exact shares per stratum + orientation balance
    shares = {}
    for stratum, quota in sorted(NULL_QUOTA.items()):
        shares[stratum] = round(quota / len(by_stratum[stratum]), 4)
    bad = {k: v for k, v in shares.items() if abs(v - 1 / 3) > 0.05}
    if bad:
        raise mt.InstrumentAbort(f"A-null-indep RED: shares {bad}")
    ab = sum(1 for c in clusters if c.cid in chosen_set and c.orientation == "AB")
    ba = sum(1 for c in clusters if c.cid in chosen_set and c.orientation == "BA")
    # label independence: keep-labeled clusters among the chosen
    keep_null = sum(
        1 for c in clusters if c.cid in chosen_set and c.label["type"] == "keep"
    )
    mt.log(
        "a_null_indep",
        shares=shares,
        orientation={"AB": ab, "BA": ba},
        keep_labeled_nulls=keep_null,
        note="exact 1/3 shares by construction; orientation split within "
        "sampling noise; label independence carried by uniform "
        "selection inside strata",
    )

    # A-null-lang: equal shares across the three theme-near substrata
    xn = shares["cross-lang-sibling"]
    tw = shares["template-twin"]
    nt = shares["near-topic"]
    if max(xn, tw, nt) - min(xn, tw, nt) > 0.05:
        raise mt.InstrumentAbort("A-null-lang RED: theme-near shares diverge")
    # A-null-cover: >= 20 per substratum
    cover = {
        k: v
        for k, v in NULL_QUOTA.items()
        if k in ("cross-lang-sibling", "template-twin", "near-topic")
    }
    if any(v < 20 for v in cover.values()):
        raise mt.InstrumentAbort(f"A-null-cover RED: {cover}")
    mt.log(
        "a_null_lang_cover",
        theme_near_shares={"xn": xn, "twin": tw, "nt": nt},
        cover=cover,
        ok=True,
    )

    doc = {
        "version": "null-coverage-1 (ADDENDUM-4, terminal iteration 3)",
        "seed": SEED_NULL,
        "cluster_ids": sorted(chosen_set),
        "quota": NULL_QUOTA,
        "shares": shares,
        "semantics": "both-null pairs only: every member of a listed cluster "
        "gets created_at=None (0 mixed-null by construction; "
        "asserted again on the effective corpus)",
        "base_files_untouched": [str(mt.TRAIN_JSONL), str(NEG_PATH)],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    NULL_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    null_sha = mt.sha256_file(NULL_PATH)
    # corpus fp v1.1 (iter-3) = sha(train + negatives + null sidecar)
    fp = mt.sha256_text(
        mt.sha256_file(mt.TRAIN_JSONL) + ":" + mt.sha256_file(NEG_PATH) + ":" + null_sha
    )
    mt.log(
        "null_coverage_emitted",
        path=str(NULL_PATH),
        sha256=null_sha,
        n_null_clusters=len(chosen_set),
        total_null_members=sum(len(c.members) for c in clusters if c.cid in chosen_set),
        corpus_fp_v1_1_iter3=fp,
        base_files_untouched=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
