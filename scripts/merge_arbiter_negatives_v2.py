#!/usr/bin/env python3
"""Merge-arbiter corpus v2 — the four theme-near strata (ADDENDUM-6 §2 pos 5).

Ratified recipe: docs/specs/merge-arbiter-corpus-prereg-ADDENDUM-6.md (MLC
30.10 round 6, 3/3; owner sanction 30.10 "фич-работа v2") + the §7 execution
record (quotas + cross-language share FIXED IN THE PRERG BEFORE GENERATION).
Extension, not re-emit: train.jsonl, holdout.jsonl, train-negatives.jsonl
(v1.1), null-envelope.json and spot-sheet stay byte-untouched; the new rows
live in their own file + sidecars (ADDENDUM-3 precedent).

STRATA (rows = pairs x 2 orientations; both orientations on every pair,
A-symmetry extended):

  zone-boundary    MONO pair,  cos_norm1 in (0.84, 0.90], char5_norm1 <= 0.05
                   — the boundary pocket right above the cos frontier (and
                   the junction 0.84..0.85 inside the zone); the MONOLINGUAL
                   compensation vehicle of the ratified share fix.
  cos-bins         CROSS-LANG pair, cos_norm1 in [0.60,0.70)/[0.70,0.78)/
                   [0.78,0.85), char5_norm1 <= 0.02 — continuity of the zone
                   from the 0.60 edge (30 ROWS per bin).
  sibling-residue  CROSS-LANG pair, char5_norm1 in (0.0785, 0.20], zeps >= 3
                   — the OR-leg of the zone (pos 1: zeps >= 3 & char5 <= 0.20).
  digit-lookalike  MONO pair, digit-dir in {-1,+1} with BOTH digit sets
                   non-empty (non-empty guard), char5_norm1 <= 0.05 — the
                   CH-3-like superset-flavor pocket of the failing 9/24 pairs
                   carried into populated negative territory. dir is a
                   CONSTRUCTION-ONLY parameter (ADDENDUM-6 pos 4: build by
                   dir is legal, reading dir at inference is forbidden).

CONVENTION PINS (ADDENDUM-6 §2 pos 6: canonical numbers = norm1):
  engine   /var/home/abyss/LABs/Projects/Project-Vesma/vesma @ release/6.6.0
  embedder nano:sha256:3b752e06... (384-dim), direct vesma.embeddings import;
           the vesmaro->vesma alias is installed UNCONDITIONALLY — engine
           6.6.0 carries an EMPTY vesmaro/ namespace husk (stale __pycache__,
           no __init__.py), so a try-import passes falsely (precedent
           b2_g1b_probes.py; the husk makes its early-return path unsound).
  text     n1(title) + "\n" + n1(body) + "\n" + n1(" ".join(tags)), [:4096],
           where n1 = lowercase + whitespace-collapse (the frozen normalizer
           form; validated against the errata numbers: XL min cos 0.7301,
           XL char5<=0.05 = 48/80, XL max char5 0.0785, CH-3 min cos 0.9413,
           A-zone-coverage red edge 0.1301 — all reproduced exactly).
  char5    through the FROZEN extractor cortex.features.pair (its n-gram text
           is already norm1).
  zeps     7 eps-flags per framing §2.4, eps = 0.01, exact zero for the
           discrete flags; digits flag fires on empty intersection (incl.
           empty sets). DE re-implementation; XL quartiles under it are
           3/3/4 vs the framing's raw-convention 3/4/4 (eps-boundary cases;
           zone verdicts unaffected — CH-3 zeps <= 1, char5 >= 0.8329).
  dir      digit sets on norm1 BODIES, regex \\d+(?:[.,:]\\d+)*; +1 iff the
           second member's set is a subset of the first's, -1 the reverse,
           0 incomparable (construction-only, see above).

ASSERTS (fail-closed, before emission): A-neg-silver (dup-predicate fail on
the norm1-identity leg + language leg for cross-lang pairs + char5 <= 0.90
razor guard; zone-boundary additionally asserts char5 <= 0.05 stays TRUE on
norm1 through the frozen extractor), A-neg-disjoint (0 body hash vs train /
v1.1 negatives / noharm surface / raw scaffolds; 0 cluster+member id vs the
holdout id-manifest and the sheet member_ids at ID LEVEL ONLY),
A-zone-coverage (negatives with char5 <= 0.05 cover [0.60, 0.85] with edge
gap <= 0.02 and max adjacent gap <= 0.02 — red at edge 0.1301 today, must
turn green), A-zone-ch3-zero (0 of the 276 CH-3 pairs incl. transitive
chain pairs in the zone on norm1), A-count, A-symmetry, П3 preflight on a
fixture cell of each new substratum BEFORE emission.

NULL envelope: a separate `null-coverage` subcommand selects ~1/3 of the NEW
clusters per new substratum into null-envelope-v2.json (A-null-indep /
A-null-lang / A-null-cover); the v1 sidecar stays untouched.

Seeds 9401..9405 — FRESH (!= 42 family, != 43..62 floor family, != the
9001-9003 / 9101-9103 negatives family, != 9201 null family), pinned in the
run-log and the sidecar manifest BEFORE generation.

Usage:
  <vesma-cortex-.venv-python> scripts/merge_arbiter_negatives_v2.py measure
  <vesma-cortex-.venv-python> scripts/merge_arbiter_negatives_v2.py build
  <vesma-cortex-.venv-python> scripts/merge_arbiter_negatives_v2.py null-coverage
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE_SRC = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src"
sys.path.insert(0, ENGINE_SRC)
sys.path.append(str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

# --- UNCONDITIONAL vesmaro -> vesma alias (engine 6.6.0 husk warning) --------
# Do NOT try-import vesmaro first: release/6.6.0 ships an empty `vesmaro/`
# namespace package (no __init__.py, stale __pycache__ only), so a probe
# import "succeeds" as a hollow husk and the frozen-builder path breaks.
# Precedent: b2_g1b_probes.py (its try-import early-return is unsound here).
import importlib  # noqa: E402  (frozen shim preamble — engine 6.6.0 vesmaro husk)
import types  # noqa: E402

_alias = types.ModuleType("vesmaro")
_alias.__path__ = [str(Path(ENGINE_SRC) / "vesma")]
sys.modules["vesmaro"] = _alias
_vesma_emb = importlib.import_module("vesma.embeddings")
sys.modules["vesmaro.embeddings"] = _vesma_emb
_alias.embeddings = _vesma_emb

import merge_arbiter_train as mt  # noqa: E402  (run-log + corpus pins)
import merge_arbiter_lib as lib  # noqa: E402  (frozen scaffolds/Member/Cluster)
from cortex.features.pair import FEATURE_NAMES, RecordLike, features  # noqa: E402
from gen_dataset_v43_strategies import _validate_side  # noqa: E402 (frozen validator)

CORPUS_DIR = mt.CORPUS_DIR
NEG2_PATH = CORPUS_DIR / "train-negatives-v2.jsonl"
NEG2_MANIFEST = CORPUS_DIR / "train-negatives-v2.manifest.json"
NULL2_PATH = CORPUS_DIR / "null-envelope-v2.json"
NEG1_PATH = mt.NEG_PATH  # v1.1 negatives (existing 200 — disjointness base)
NOHARM_SURFACE = mt.ART_DIR / "noharm-surface-v1.jsonl"
SHEET_SELECTION = (
    REPO / "data" / "merge-arbiter" / "spot-sheet" / "sheet-selection.json"
)
SLICE = "merge-arbiter-corpus-v2-negatives"

# --- pins --------------------------------------------------------------------
NANO_PIN_PREFIX = "nano:sha256:3b752e06"
ENGINEExpect_GIT = ("458fffc", "release/6.6.0")

#: Fresh seeds, pinned BEFORE generation (run-log + sidecar manifest).
SEED_ZONE_BOUNDARY = 9401
SEED_COS_BINS = 9402
SEED_SIBLING_RESIDUE = 9403
SEED_DIGIT_LOOKALIKE = 9404
SEED_NULL = 9405
SEED_LIST = [
    SEED_ZONE_BOUNDARY,
    SEED_COS_BINS,
    SEED_SIBLING_RESIDUE,
    SEED_DIGIT_LOOKALIKE,
    SEED_NULL,
]

#: Per-substratum quotas in PAIRS (rows = pairs x 2 orientations) and the
#: cross-language share — RATIFIED INTO THE PRERG (ADDENDUM-6 §7) BEFORE
#: generation; the build refuses unless that section is present at HEAD.
#: Sizing follows the measured pools (measure subcommand) and the ratified
#: continuity/CV constraints, not the ADDENDUM-5 char5-bin core quota (that
#: 90-core discipline is preserved corpus-wide, see the §7 bin table).
QUOTA_PAIRS = {
    "zone-boundary": 28,
    "cos-bins": 19,
    "sibling-residue": 12,
    "digit-lookalike": 30,
}
COS_BIN_PAIRS = {"[0.60,0.70)": 10, "[0.70,0.78)": 8, "[0.78,0.85)": 1}
COS_BINS = ((0.60, 0.70), (0.70, 0.78), (0.78, 0.85))
EXISTING_NEG_ROWS = 200
EXISTING_CROSS_ROWS = 80
CROSS_NEW_ROWS = 38 + 24  # cos-bins + sibling-residue
MONO_NEW_ROWS = 56 + 60  # zone-boundary + digit-lookalike
NEG2_ROWS = sum(QUOTA_PAIRS.values()) * 2
NEG_MASS_ROWS = EXISTING_NEG_ROWS + NEG2_ROWS
CROSS_MASS_ROWS = EXISTING_CROSS_ROWS + CROSS_NEW_ROWS
SHARE_CROSS = CROSS_MASS_ROWS / NEG_MASS_ROWS  # 142/378 = 0.3757 <= 1/2
SHARE_STRING = f"{CROSS_MASS_ROWS}/{NEG_MASS_ROWS}"

ZONE_BOUND = (0.84, 0.90)
CHAR5_ZONE_MAX = 0.05
CHAR5_BINS_MAX = 0.02
CHAR5_RESIDUE = (0.0785, 0.20)
ZEPS_MIN = 3
COVERAGE_LO, COVERAGE_HI = 0.60, 0.85
GAP_MAX = 0.02
EPS = 0.01
DIG_RE = re.compile(r"\d+(?:[.,:]\d+)*")

_NEW_SUBSTRATA = ("zone-boundary", "cos-bins", "sibling-residue", "digit-lookalike")

# ============================ authored content ===============================
# Publication class "authored synthetic text" (merge_arbiter_lib precedent);
# every title/body below is fresh authored content — hash-disjointness vs
# train / v1.1 negatives / noharm surface / scaffolds is asserted later.

# --- cross-language topics (frames = opposite states on one topic) ----------
XL_FRAG = [
    # register-split family: long operational ru record vs terse en meta note;
    # this family reaches the deepest cos band (edge [0.60, 0.62]).
    dict(
        sid="f-keys",
        a=(
            "Служебная запись по теме",
            "Ротация ключей API назначена на субботу: старые ключи блокируются "
            "в конце окна, новые заранее кладут в сейф, дежурные получают "
            "инструкцию, владельцы сервисов уведомлены письмом.",
            ["helpdesk"],
            "ru",
        ),
        b=("Closing this one", "asked and answered.", ["archive"], "en"),
        just="the record plans the key rotation cutover, the sibling is a terse triage note that closes the question — same topic, opposite depth of content",
    ),
    dict(
        sid="f-purge",
        a=(
            "Журнал обслуживающих работ",
            "Очистка бэкапов идёт по расписанию: копии старше месяца удаляются "
            "ночью, восстановление доступно из панели без заявки, отчёт "
            "складывается в журнал, а список затронутых витрин заранее "
            "согласован с их владельцами.",
            ["helpdesk"],
            "ru",
        ),
        b=("No update planned", "works as designed.", ["noise"], "en"),
        just="the record documents the running purge schedule, the sibling is a one-line triage stub — same topic, different content",
    ),
    dict(
        sid="f-canary",
        a=(
            "Заметка о выкате",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой, владельцы сервиса "
            "уведомлены в канале.",
            ["retro"],
            "ru",
        ),
        b=("Won't fix here", "duplicate of the older note.", ["wontfix"], "en"),
        just="the record opens a canary experiment, the sibling dismisses the thread as a duplicate — same topic, no shared content",
    ),
    dict(
        sid="f-quota",
        a=(
            "Сводка по лимитам",
            "Квоты для внутренних сервисов подняли: новый потолок действует с "
            "понедельника, заявки на исключения закрываются автоматически, "
            "история сохраняется.",
            ["helpdesk"],
            "ru",
        ),
        b=("Nothing to add", "nothing to add.", ["misc"], "en"),
        just="the record announces the raised quotas, the sibling is an empty triage marker — same topic, opposite content weight",
    ),
    dict(
        sid="f-sso",
        a=(
            "План перехода на SSO",
            "Миграция входа на SSO стартовала: бухгалтерия переведена первой, "
            "остальные команды переезжают по графику, локальные пароли гасятся "
            "волнами до конца месяца.",
            ["helpdesk"],
            "ru",
        ),
        b=("Stale by now", "stale by now.", ["stale"], "en"),
        just="the record drives the SSO migration, the sibling is a stale-marking stub — same topic, different content",
    ),
    dict(
        sid="f-reindex",
        a=(
            "Порядок переиндексации",
            "Переиндексация поиска идёт вручную: очередь разбирается частями, "
            "запросы обслуживаются с реплики, отчёт приложат к плану "
            "готовности.",
            ["helpdesk"],
            "ru",
        ),
        b=("Tracked elsewhere", "already tracked elsewhere.", ["tracking"], "en"),
        just="the record describes the manual reindex routine, the sibling redirects to another tracker — same topic, different content",
    ),
    dict(
        sid="f-drain",
        a=(
            "Регламент дренирования",
            "Дренирование нод кластера запланировано на среду: поды переезжают "
            "волнами, бюджет нарушений задаёт окно, отмена возможна до "
            "обеда.",
            ["ops"],
            "ru",
        ),
        b=(
            "Archived without action",
            "out of scope for this board.",
            ["archive"],
            "en",
        ),
        just="the record schedules the node drain, the sibling archives the request as out of scope — same topic, opposite outcomes",
    ),
    dict(
        sid="f-edge",
        a=(
            "Отчёт по пограничным узлам",
            "Пограничные узлы пережили плановое обновление: старые версии "
            "ушли в отставку, новые приняли трафик без потерь, наблюдение "
            "продолжается в обычном режиме.",
            ["edge"],
            "ru",
        ),
        b=("Expected behaviour", "closing after verification.", ["faq"], "en"),
        just="the record narrates a planned edge-node refresh, the sibling is a terse triage closure — same topic, opposite depth",
    ),
    dict(
        sid="f-cutover",
        a=(
            "План переключения канала",
            "Переключение резервного канала назначено на ночь: основной путь "
            "гасится на десять минут, резерв принимает нагрузку, откат "
            "занимает одну команду.",
            ["network"],
            "ru",
        ),
        b=("No update planned", "nothing to do here.", ["misc"], "en"),
        just="the record schedules a channel cutover, the sibling is an empty triage stub — same topic, opposite depth",
    ),
    dict(
        sid="f-certs",
        a=(
            "Отчёт по сертификатам",
            "Перевыпуск серверных сертификатов прошёл автоматически: цепочки "
            "обновились, старые отозваны, клиенты переключились без "
            "инцидентов.",
            ["security"],
            "ru",
        ),
        b=("Shelving this", "cant reproduce.", ["backlog"], "en"),
        just="the record reports the certificate rotation done, the sibling shelves an unreproducible claim — same topic, different content",
    ),
    dict(
        sid="f-restore",
        a=(
            "Учение по восстановлению",
            "Учение по восстановлению из копий прошло в полном объёме: "
            "развернули сервис из архива, сверили контрольные суммы, отчёт "
            "приложили к плану готовности.",
            ["ops", "notes"],
            "ru",
        ),
        b=("Expected behaviour", "expected behaviour.", ["faq"], "en"),
        just="the record narrates a restore drill, the sibling marks a request as expected behaviour — same topic, different content",
    ),
    dict(
        sid="f-dns",
        a=(
            "Перенос DNS-записей",
            "Перенос DNS-записей на нового провайдера идёт по волнам: сначала "
            "внешние имена, затем внутренние, откат описан в регламенте.",
            ["network"],
            "ru",
        ),
        b=("Recorded for reference", "recorded for reference.", ["misc"], "en"),
        just="the record drives the DNS cutover, the sibling only records a pointer — same topic, different content",
    ),
    dict(
        sid="f-logs",
        a=(
            "Доставка логов",
            "Доставку логов перевели на новый коллектор: источники перецеплены "
            "по очереди, потери считаются на каждой ступени, старый путь "
            "гасят в пятницу.",
            ["logging"],
            "ru",
        ),
        b=("Superseded by the runbook", "superseded by the runbook.", ["stale"], "en"),
        just="the record moves log shipping to a new collector, the sibling points at the runbook — same topic, different content",
    ),
    dict(
        sid="f-warm",
        a=(
            "Прогрев кэшей",
            "Прогрев кэшей после обслуживания прошёл ночью: горячие ключи "
            "восстановлены, промахи вернулись к норме, отчёт в канале "
            "дежурных.",
            ["performance"],
            "ru",
        ),
        b=(
            "Closing after verification",
            "closing after verification.",
            ["archive"],
            "en",
        ),
        just="the record reports the cache warmup result, the sibling closes a thread after verification — same topic, different content",
    ),
]

XL_OWN_TITLES = [
    dict(
        sid="f2-keys",
        a=(
            "Доступы и ключи: что решили",
            "Ротация ключей API назначена на субботу: старые ключи блокируются "
            "в конце окна, новые заранее кладут в сейф, дежурные получают "
            "инструкцию.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Credentials and keys: the decision",
            "The API key rotation finished early: legacy keys were revoked this "
            "morning, the replacement set shipped last week and is already in "
            "the vault.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite states of the same key-rotation topic: planned Saturday cutover vs already finished and revoked",
    ),
    dict(
        sid="f2-purge",
        a=(
            "Хранилище копий: правила",
            "Очистка бэкапов работает по расписанию: копии старше месяца "
            "удаляются ночью, восстановление доступно из панели.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Archive storage rules",
            "Backup purge is disabled indefinitely: no copy is removed "
            "automatically anymore, restores go through the on-call engineer.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite purge regimes on the same backup topic: scheduled nightly deletion vs indefinite retention",
    ),
    dict(
        sid="f2-canary",
        a=(
            "Экспериментальный выкат",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Experimental rollout",
            "The canary release was rolled back: traffic share dropped to zero, "
            "per-minute metrics stopped and the rollout train moved on.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite rollout states on the same canary topic: five percent open vs rolled back to zero",
    ),
    dict(
        sid="f2-quota",
        a=(
            "Лимиты и исключения",
            "Квоты для внутренних сервисов подняли: новый потолок действует с "
            "понедельника, заявки на исключения закрываются автоматически.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Limits and exceptions",
            "Internal service quotas were declined this cycle: the old ceiling "
            "stays and exception requests wait for a manual review each Friday.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite quota outcomes on the same limits topic: raised ceiling with auto-approval vs declined with manual review",
    ),
    dict(
        sid="f2-sso",
        a=(
            "Вход в системы: переходный план",
            "Миграция входа на SSO стартовала: бухгалтерия переведена первой, "
            "остальные команды переезжают по графику.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Login transition plan",
            "The SSO login migration is postponed: accounting keeps local "
            "passwords and the team schedule is suspended until federation is "
            "fixed.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite migration states on the same SSO topic: started with accounting first vs postponed indefinitely",
    ),
    dict(
        sid="f2-reindex",
        a=(
            "Поиск: как живёт индекс",
            "Переиндексация поиска идёт вручную: очередь разбирается частями, "
            "запросы обслуживаются с реплики.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Search index: how it behaves",
            "Search reindexing was switched off: the queue is paused entirely "
            "and traffic reads from the primary store.",
            ["platform", "notes"],
            "en",
        ),
        just="opposite reindex regimes on the same search topic: manual draining vs fully paused",
    ),
]

XL_DIVERSE = [
    dict(
        sid="d-keys",
        a=(
            "Пятничная заметка",
            "Ротация ключей API назначена на субботу: старые ключи "
            "блокируются в конце окна, новые заранее кладут в сейф.",
            ["draft"],
            "ru",
        ),
        b=(
            "From the observation log",
            "API key rotation finished early: legacy keys were revoked this "
            "morning, the replacement set is already in the vault.",
            ["misc"],
            "en",
        ),
        just="opposite states of the key-rotation topic in casual registers: planned vs finished",
    ),
    dict(
        sid="d-purge",
        a=(
            "Что обсудили в канале",
            "Очистка бэкапов идёт по расписанию: копии старше месяца "
            "удаляются ночью, восстановление доступно из панели.",
            ["draft"],
            "ru",
        ),
        b=(
            "For the record",
            "Backup purge is disabled indefinitely: nothing is removed "
            "automatically anymore.",
            ["misc"],
            "en",
        ),
        just="opposite purge regimes in casual registers: scheduled deletion vs indefinite retention",
    ),
    dict(
        sid="d-canary",
        a=(
            "Мысли вслух по теме",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой.",
            ["draft"],
            "ru",
        ),
        b=(
            "A small work thing",
            "The canary release was rolled back: traffic share dropped to "
            "zero and per-minute metrics stopped.",
            ["misc"],
            "en",
        ),
        just="opposite rollout states in casual registers: open experiment vs rolled back",
    ),
    dict(
        sid="d-sso",
        a=(
            "Пометка на полях",
            "Миграция входа на SSO стартовала: бухгалтерия переведена "
            "первой, остальные команды переезжают по графику.",
            ["draft"],
            "ru",
        ),
        b=(
            "Thinking out loud",
            "The SSO login migration is postponed: accounting keeps local "
            "passwords until federation is fixed.",
            ["misc"],
            "en",
        ),
        just="opposite migration states in casual registers: started vs postponed",
    ),
]

XL_HIGH = [
    # translated-same-title family: strongest structural parallel -> highest cos
    dict(
        sid="h-keys",
        a=(
            "Ротация ключей API: статус и окно",
            "Ротация ключей API назначена на субботу: старые ключи "
            "блокируются в конце окна, новые заранее кладут в сейф, "
            "дежурные получают инструкцию.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "API key rotation: status and window",
            "The API key rotation is postponed again: current credentials "
            "stay valid until a separate notice and the replacement plan is "
            "frozen.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames with opposite rotation states (planned vs postponed)",
    ),
    dict(
        sid="h-reindex",
        a=(
            "Переиндексация поиска: текущий режим",
            "Переиндексация поиска идёт вручную: очередь разбирается "
            "частями, запросы обслуживаются с реплики, отчёт приложат к "
            "плану.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Search reindexing: current mode",
            "The search reindex runs manually again: jobs are drained in "
            "batches, queries are served by the replica and a report lands "
            "in the plan.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames, mirrored reindex mechanics (the semantic difference is carried by the opposite-state sibling frame set)",
    ),
    dict(
        sid="h-purge",
        a=(
            "Очистка бэкапов: расписание",
            "Очистка бэкапов идёт по расписанию: копии старше месяца "
            "удаляются ночью, восстановление доступно из панели.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Backup purge: schedule",
            "The scheduled backup purge is live: archives older than a month "
            "drop at night and restores stay self-service in the console.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames on the purge schedule (mirrored mechanics; sibling value frames keep the content distinct)",
    ),
    dict(
        sid="h-canary",
        a=(
            "Canary-релиз: состояние выката",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Canary release: rollout state",
            "A canary release is open on five percent of traffic: metrics "
            "stream every minute and the rollback plan is one command away.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames on the canary state (mirrored mechanics)",
    ),
    dict(
        sid="h-quota",
        a=(
            "Квоты внутренних сервисов",
            "Квоты для внутренних сервисов подняли: новый потолок действует "
            "с понедельника, заявки на исключения закрываются автоматически.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "Internal service quotas",
            "Internal quotas were raised: the new ceiling applies from Monday "
            "and exception requests close automatically.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames on the quota raise (mirrored mechanics)",
    ),
    dict(
        sid="h-sso",
        a=(
            "Миграция входа на SSO",
            "Миграция входа на SSO стартовала: бухгалтерия переведена "
            "первой, остальные команды переезжают по графику.",
            ["platform", "notes"],
            "ru",
        ),
        b=(
            "SSO login migration",
            "SSO onboarding has started: accounting moved first and the "
            "remaining teams follow the published schedule.",
            ["platform", "notes"],
            "en",
        ),
        just="same-title translated frames on the SSO migration start (mirrored mechanics)",
    ),
]

XL_META = [
    # long operational record vs short meta note, shared register split;
    # this family carries the [0.74, 0.85) band (bin 3).
    dict(
        sid="m-purge",
        a=(
            "Служебная запись по теме",
            "Очистка бэкапов работает по расписанию: копии старше месяца "
            "удаляются ночью, восстановление доступно из панели без заявки.",
            ["helpdesk", "faq"],
            "ru",
        ),
        b=("Keeping this open", "works as designed.", ["draft", "notes"], "en"),
        just="the record documents the purge schedule, the sibling holds a triage note open — same topic, different content",
    ),
    dict(
        sid="m-quota",
        a=(
            "Заметка о выкате",
            "Квоты для внутренних сервисов подняли: новый потолок действует с "
            "понедельника, заявки на исключения закрываются автоматически.",
            ["helpdesk", "faq"],
            "ru",
        ),
        b=(
            "Community answer",
            "asked and answered, no further steps.",
            ["draft", "notes"],
            "en",
        ),
        just="the record announces raised quotas, the sibling is a community-triage answer — same topic, different content",
    ),
    dict(
        sid="m-reindex",
        a=(
            "Порядок переиндексации",
            "Переиндексация поиска идёт вручную: очередь разбирается частями, "
            "запросы обслуживаются с реплики, отчёт приложат к плану.",
            ["helpdesk", "faq"],
            "ru",
        ),
        b=("Stale by now", "stale by now, nothing to act on.", ["misc"], "en"),
        just="the record describes the manual reindex, the sibling marks the thread stale — same topic, different content",
    ),
    dict(
        sid="m-canary",
        a=(
            "Заметка о выкате",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой.",
            ["helpdesk", "faq"],
            "ru",
        ),
        b=("For the record", "duplicate of the older note.", ["draft", "notes"], "en"),
        just="the record opens a canary experiment, the sibling marks a duplicate — same topic, different content",
    ),
    dict(
        sid="m2-purge",
        a=(
            "Пятничная заметка",
            "Очистка бэкапов работает по расписанию: копии старше месяца "
            "удаляются ночью, восстановление доступно из панели без заявки.",
            ["draft"],
            "ru",
        ),
        b=(
            "From the observation log",
            "works as designed, nothing to change.",
            ["misc"],
            "en",
        ),
        just="purge schedule record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-keys",
        a=(
            "Из журнала наблюдений",
            "Ротация ключей API назначена на субботу: старые ключи "
            "блокируются в конце окна, новые заранее кладут в сейф.",
            ["draft"],
            "ru",
        ),
        b=(
            "Weeding out the queue",
            "asked and answered, thread closed.",
            ["misc"],
            "en",
        ),
        just="key-rotation plan record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-quota",
        a=(
            "Мысли вслух по теме",
            "Квоты для внутренних сервисов подняли: новый потолок действует с "
            "понедельника, заявки на исключения закрываются автоматически.",
            ["draft"],
            "ru",
        ),
        b=("A margin note", "no action needed from anyone.", ["misc"], "en"),
        just="quota-raise record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-canary",
        a=(
            "Что обсудили в канале",
            "Canary-релиз открыт на пятый процент трафика: метрики пишутся "
            "поминутно, откат готов одной командой.",
            ["draft"],
            "ru",
        ),
        b=(
            "A small work thing",
            "already tracked elsewhere, no update here.",
            ["misc"],
            "en",
        ),
        just="canary record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-reindex",
        a=(
            "Пункты плана на неделю",
            "Переиндексация поиска идёт вручную: очередь разбирается частями, "
            "запросы обслуживаются с реплики.",
            ["draft"],
            "ru",
        ),
        b=("Thinking out loud", "works as designed, closing.", ["misc"], "en"),
        just="reindex record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-sso",
        a=(
            "Служебная мелочь",
            "Миграция входа на SSO стартовала: бухгалтерия переведена первой, "
            "остальные команды переезжают по графику.",
            ["draft"],
            "ru",
        ),
        b=("An aside for the log", "stale by now, nothing to act on.", ["misc"], "en"),
        just="SSO migration record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-drain",
        a=(
            "Рабочая мелочь",
            "Дренирование нод кластера запланировано на среду: поды переезжают "
            "волнами, бюджет нарушений задаёт окно.",
            ["draft"],
            "ru",
        ),
        b=("One more aside", "out of scope for this board.", ["misc"], "en"),
        just="node-drain record vs a casual triage line — same topic, different content",
    ),
    dict(
        sid="m2-certs",
        a=(
            "Короткий апдейт",
            "Перевыпуск серверных сертификатов прошёл автоматически: цепочки "
            "обновились, старые отозваны, клиенты переключились без "
            "инцидентов.",
            ["draft"],
            "ru",
        ),
        b=("A quiet note", "cant reproduce, shelving.", ["misc"], "en"),
        just="certificate rotation record vs a casual triage line — same topic, different content",
    ),
]

# --- coverage continuity family: opposite-order tight frames (the only
# authored pattern measured to reach [0.78, 0.85) for honest cross-language
# pairs; shared value digits add cos without char5 mass)
XL_TIGHT = [
    dict(
        key="tight-replicas",
        a=(
            {
                "title": "Реплики: порядок",
                "body": "Включение реплик начинается с читающих витрин, запись "
                "подключается последней, отставание меряется каждые 10 "
                "минут.",
                "tags": ["platform", "notes"],
                "language": "ru",
                "record_type": "note",
            }
        ),
        b=(
            {
                "title": "Read copies: bring-up order",
                "body": "Read-copy bring-up skips the readers: the writer "
                "connects first and lag checks moved to hourly.",
                "tags": ["platform", "notes"],
                "language": "en",
                "record_type": "note",
            }
        ),
        just="inverted bring-up order on the same replica topic — opposite runbooks, one plan",
    ),
    dict(
        key="tight-cache",
        a=(
            {
                "title": "Кэш статики: порядок",
                "body": "Инвалидация статики начинается с пограничных узлов, "
                "источник обновляется последней, окно занимает 60 "
                "минут.",
                "tags": ["platform", "notes"],
                "language": "ru",
                "record_type": "note",
            }
        ),
        b=(
            {
                "title": "Static cache: reset order",
                "body": "Static resets start at the origin: edge nodes refresh "
                "last and the 60-minute window is retired.",
                "tags": ["platform", "notes"],
                "language": "en",
                "record_type": "note",
            }
        ),
        just="inverted reset order on the same static-cache topic — opposite runbooks, one plan",
    ),
    dict(
        key="tight-deploy",
        a=(
            {
                "title": "Выкат: очерёдность",
                "body": "Выкат затрагивает сначала витрины писем, затем корзину, "
                "лимит трафика держится на 5 процентах до проверки.",
                "tags": ["platform", "notes"],
                "language": "ru",
                "record_type": "note",
            }
        ),
        b=(
            {
                "title": "Store rollout: sequencing",
                "body": "The rollout order is reversed: cart ships first, mail "
                "surfaces follow and the five-percent cap is dropped "
                "before any check.",
                "tags": ["platform", "notes"],
                "language": "en",
                "record_type": "note",
            }
        ),
        just="inverted rollout sequencing on the same store topic — opposite plans, one surface set",
    ),
    dict(
        key="tight-backup",
        a=(
            {
                "title": "Бэкапы: очерёдность",
                "body": "Выгрузка бэкапов начинается с витрин, база выгружается "
                "последней, целостность проверяется после окна 120.",
                "tags": ["platform", "notes"],
                "language": "ru",
                "record_type": "note",
            }
        ),
        b=(
            {
                "title": "Backup export: sequencing",
                "body": "Backup export starts with the database: stores follow "
                "and the integrity pass no longer waits for the "
                "120-minute window.",
                "tags": ["platform", "notes"],
                "language": "en",
                "record_type": "note",
            }
        ),
        just="inverted export sequencing on the same backup topic — opposite runbooks, one plan",
    ),
]

# --- sibling-residue topics: dense shared latin terms, disjoint tags/titles --
RES_TOPICS = [
    dict(
        sid="pg",
        ru_t="Сводка по окружению",
        en_t="Environment digest",
        ru_a="Postgres 16 в staging обновлён: redis-кэш прогрет, kafka-коннекторы переподключены, dblink-линки проверены, в Grafana ошибок нет.",
        en_a="The staging Postgres 16 rollout was reverted: redis stays cold, kafka connectors are detached, dblink links unverified, Grafana shows timeouts.",
        ru_b="Апгрейд Postgres 16 в staging откатили: redis остался холодным, kafka-коннекторы отвязаны, dblink не проверялся, Grafana показывает таймауты.",
        en_b="Staging now runs Postgres 16: the redis cache is prewarmed, kafka connectors reattached, dblink verified and Grafana dashboards are clean.",
        ta=("databases", "staging"),
        tb=("infra", "rollout"),
        just="opposite states of the Postgres 16 staging rollout; the shared postgres/redis/kafka/dblink/Grafana term mass carries char5 into the residue band",
    ),
    dict(
        sid="ci",
        ru_t="Сборка и доставка",
        en_t="Build and delivery notes",
        ru_a="Gitlab-runner переехал на свежие образы: docker-слой кэшируется, terraform-plan гоняется до мерджа, helm-чарты версионируются, артефакты живут сутки.",
        en_a="Fresh gitlab-runner images were rolled back: docker layers bypass the cache, terraform plan moved after merge, helm charts unversioned, artifacts expire hourly.",
        ru_b="Образы gitlab-runner откатили: docker-кэш слоёв обходится, terraform plan запускается после мерджа, helm-чарты без версий, артефакты сгорают через час.",
        en_b="The gitlab-runner fleet moved to fresh images: docker layers cache locally, terraform plans gate every merge, helm charts versioned, artifacts survive a day.",
        ta=("ci", "builders"),
        tb=("pipeline", "tools"),
        just="opposite states of the runner image switch; the gitlab-runner/docker/terraform/helm term mass carries char5 into the residue band",
    ),
    dict(
        sid="obs",
        ru_t="Мониторинг: коротко",
        en_t="Monitoring in brief",
        ru_a="Prometheus переехал на новый retention: Loki держит логи четырнадцать дней, Alertmanager-правила переписаны, экспортёры node_exporter и blackbox обновлены.",
        en_a="The new Prometheus retention was undone: Loki keeps two days of logs, Alertmanager rules frozen, node_exporter and blackbox exporters stay old.",
        ru_b="Retention у Prometheus вернули на старое значение: Loki хранит два дня логов, Alertmanager заморожен, node_exporter и blackbox не обновлялись.",
        en_b="Prometheus runs the new retention: Loki stores two weeks of logs, Alertmanager rules rewritten and both node_exporter and blackbox refreshed.",
        ta=("observability", "metrics"),
        tb=("logging", "alerts"),
        just="opposite retention outcomes on the Prometheus stack; the prometheus/loki/alertmanager/exporter term mass carries char5",
    ),
    dict(
        sid="mesh",
        ru_t="Сетевой каркас",
        en_t="Network fabric notes",
        ru_a="Сертификаты istio перевыпущены автоматически: nginx-фронты подхватили цепочку, envoy-сайдкары обновились, haproxy-контроллер перечитал конфиг.",
        en_a="The istio certificate rotation failed midway: nginx fronts keep the stale chain, envoy sidecars untouched, haproxy config not reloaded.",
        ru_b="Перевыпуск сертификатов istio сорвался: nginx отдаёт старую цепочку, envoy-сайдкары не тронуты, haproxy конфиг не перечитан.",
        en_b="Mesh certificates rotated automatically: nginx fronts picked up the new chain, envoy sidecars refreshed and haproxy reloaded its config.",
        ta=("security", "certificates"),
        tb=("network", "mesh"),
        just="opposite outcomes of the istio certificate rotation; the istio/nginx/envoy/haproxy term mass carries char5",
    ),
    dict(
        sid="queue",
        ru_t="Асинхронные процессы",
        en_t="Async processing status",
        ru_a="Kafka-топики перебалансированы: consumer-группы ровные, lag держится около нуля, schema-registry синхронизирован, dead-letter очереди пусты.",
        en_a="The kafka rebalance was aborted: consumer groups lopsided, lag climbing, schema-registry out of sync, dead-letter queues filling up.",
        ru_b="Балансировку kafka-топиков прервали: consumer-группы перекошены, lag растёт, schema-registry разсинхронизирован, dead-letter переполняется.",
        en_b="Kafka topics are rebalanced: consumer groups even, lag near zero, schema-registry in sync and dead-letter queues empty.",
        ta=("streaming", "queues"),
        tb=("async", "backlog"),
        just="opposite outcomes of the kafka rebalance; the kafka/consumer/lag/schema-registry/dead-letter term mass carries char5",
    ),
    dict(
        sid="search",
        ru_t="Поисковый движок",
        en_t="Search engine notes",
        ru_a="Elasticsearch-кластер переключён на новую версию: индексы переоткрыты, синонимы перезагружены, suggester-словарь пересобран, дашборд Latency чистый.",
        en_a="The elasticsearch upgrade was postponed: indices stay closed, synonyms unloaded, suggester dictionary stale, latency board unreliable.",
        ru_b="Апгрейд elasticsearch отложили: индексы закрыты, синонимы не загружены, suggester-словарь устарел, дашборд Latency не показывает реальность.",
        en_b="The elasticsearch cluster runs the new version: indices reopened, synonyms reloaded, suggester dictionary rebuilt and the latency board is clean.",
        ta=("search", "platform"),
        tb=("engine", "indexing"),
        just="opposite states of the elasticsearch upgrade; the elasticsearch/indices/synonyms/suggester/latency term mass carries char5",
    ),
    dict(
        sid="restore",
        ru_t="Резервные копии",
        en_t="Backup pipeline notes",
        ru_a="Восстановление из бэкапа в staging уложилось в план: pg_restore отработал, redis прогрели заново, clickhouse-реплики догнали мастеров, minio-бакет цел.",
        en_a="The staging backup restore overran its plan: pg_restore stalled twice, redis stayed cold, clickhouse replicas lag behind masters, the minio bucket shows gaps.",
        ru_b="Восстановление из бэкапа в staging сорвалось: pg_restore дважды замирал, redis не прогрели, clickhouse-реплики отстают, в minio-бакете дыры.",
        en_b="The staging restore landed exactly on plan: pg_restore completed, redis prewarmed, clickhouse replicas caught up and the minio bucket verified.",
        ta=("backups", "restore"),
        tb=("recovery", "drills"),
        just="opposite outcomes of the backup restore drill; the pg_restore/redis/clickhouse/minio term mass carries char5",
    ),
    dict(
        sid="gw",
        ru_t="Входной трафик",
        en_t="Gateway tier notes",
        ru_a="nginx-фронты переведены на новую схему: traefik принимает внутренние маршруты, envoy-буферы расширены, haproxy-статы чистые.",
        en_a="The nginx frontend change was reverted: traefik dropped the internal routes, envoy buffers shrank back and haproxy stats show errors.",
        ru_b="Схему nginx-фронтов откатили: traefik потерял внутренние маршруты, envoy-буферы сжали, haproxy-статы в ошибках.",
        en_b="nginx fronts run the new scheme: traefik owns the internal routes, envoy buffers widened and haproxy stats are clean.",
        ta=("gateway", "traffic"),
        tb=("routing", "edge"),
        just="opposite states of the frontend scheme change; the nginx/traefik/envoy/haproxy term mass carries char5",
    ),
    dict(
        sid="idp",
        ru_t="Вход и доступы",
        en_t="Identity stack notes",
        ru_a="Keycloak переведён на новый realm: ldap-синхронизация идёт hourly, oauth-клиенты перезаписаны, saml-феди отдают атрибуты по новому маппингу.",
        en_a="The Keycloak realm migration was rolled back: ldap sync stays daily, oauth clients keep old scopes, saml federation returns the legacy attribute map.",
        ru_b="Миграцию Keycloak откатили: ldap-синхронизация осталась дневной, oauth-клиенты со старыми скоупами, saml-федерация отдаёт прежний маппинг.",
        en_b="Keycloak runs the new realm: ldap sync hourly, oauth clients rewritten and saml federation issues the new attribute mapping.",
        ta=("identity", "auth"),
        tb=("access", "sso"),
        just="opposite states of the Keycloak realm migration; the keycloak/ldap/oauth/saml term mass carries char5",
    ),
    dict(
        sid="lake",
        ru_t="Аналитический контур",
        en_t="Analytics pipeline notes",
        ru_a="airflow-даги перезаписаны: spark-джобы пошли в новую очередь, airbyte-коннекторы дописывают в clickhouse, s3-переходы чистые.",
        en_a="The airflow DAG rewrite was shelved: spark jobs queue as before, airbyte connectors still bypass clickhouse and the s3 handoffs stay flaky.",
        ru_b="Переписывание airflow-дагов отложили: spark-джобы в старой очереди, airbyte-коннекторы идут мимо clickhouse, s3-передачи дрожат.",
        en_b="airflow DAGs run the rewritten schedule: spark jobs use the fresh queue, airbyte connectors land in clickhouse and s3 handoffs are clean.",
        ta=("analytics", "pipeline"),
        tb=("data", "processing"),
        just="opposite states of the DAG rewrite; the airflow/spark/airbyte/clickhouse/s3 term mass carries char5",
    ),
    dict(
        sid="secrets",
        ru_t="Хранение секретов",
        en_t="Secret storage notes",
        ru_a="vault-политики переписаны: sealed-secrets перевыпущены, kms-ключи ротированы, kubernetes-секреты монтируются только в нужные поды.",
        en_a="The vault policy rewrite was frozen: sealed-secrets stay stale, kms keys past their rotation date and kubernetes secrets mount everywhere.",
        ru_b="Переписывание vault-политик заморозили: sealed-secrets устарели, kms-ключи просрочены, kubernetes-секреты монтируются повсюду.",
        en_b="vault policies run the rewritten rules: sealed-secrets reissued, kms keys rotated and kubernetes secrets mount only where needed.",
        ta=("security", "secrets"),
        tb=("infra", "keys"),
        just="opposite states of the vault policy rewrite; the vault/sealed-secrets/kms/kubernetes term mass carries char5",
    ),
    dict(
        sid="hpa",
        ru_t="Масштабирование",
        en_t="Autoscaling notes",
        ru_a="HPA-правила перенастроены: kubelet отдаёт метрики быстрее, pod-лимиты пересчитаны, ноды кластера выходят на новый бюджет.",
        en_a="The HPA retune was rolled back: kubelet metrics stay slow, pod limits unchanged and cluster nodes keep the old budget.",
        ru_b="Перенастройку HPA откатили: kubelet метрики отдаёт медленно, pod-лимиты прежние, ноды кластера живут в старом бюджете.",
        en_b="HPA rules run the new tuning: kubelet metrics stream faster, pod limits recalculated and cluster nodes settle into the new budget.",
        ta=("scaling", "compute"),
        tb=("capacity", "kubernetes"),
        just="opposite states of the HPA retune; the HPA/kubelet/pod/node term mass carries char5",
    ),
    dict(
        sid="pool",
        ru_t="Пул соединений",
        en_t="Connection pooling notes",
        ru_a="pgbouncer переведён на транзакционный режим: postgres-линки сжались, таймауты очереди переписаны, клиенты видят ровные отклики.",
        en_a="The pgbouncer switch was reverted: postgres links ballooned again, queue timeouts restored and clients see jagged responses.",
        ru_b="Режим pgbouncer вернули на сессии: postgres-линки разбухли, таймауты очереди старые, клиенты ловят рваные отклики.",
        en_b="pgbouncer runs transactional pooling: postgres links compressed, queue timeouts rewritten and clients get even responses.",
        ta=("databases", "pooling"),
        tb=("backend", "connections"),
        just="opposite pooling modes on the pgbouncer tier; the pgbouncer/postgres/timeout/client term mass carries char5",
    ),
    dict(
        sid="cron",
        ru_t="Ночные задания",
        en_t="Scheduled jobs notes",
        ru_a="cron-расписание переписано: задачи идут в новые окна, scheduler-дубликаты убраны, отчёт о пропусках собирается по утрам.",
        en_a="The cron rewrite was abandoned: jobs keep the old windows, scheduler duplicates persist and the missed-run report stays empty.",
        ru_b="Переписывание cron-расписания бросили: задачи живут в старых окнах, scheduler-дубликаты остались, отчёт о пропусках пустой.",
        en_b="cron schedules run the rewritten windows: jobs spread across fresh slots, scheduler duplicates removed and the missed-run report fills each morning.",
        ta=("scheduling", "jobs"),
        tb=("automation", "maintenance"),
        just="opposite states of the cron rewrite; the cron/scheduler/window/report term mass carries char5",
    ),
    dict(
        sid="cdn",
        ru_t="Пограничный кэш",
        en_t="Edge cache notes",
        ru_a="CDN-правила обновлены: edge-узлы держат длинный TTL, varnish-слой прогревается по расписанию, хиты считаются на каждом узле.",
        en_a="The CDN rule update was pulled back: edge nodes keep short TTLs, the varnish layer warms ad hoc and per-node hit counts stay dark.",
        ru_b="Обновление CDN-правил отменили: edge-узлы сидят на коротком TTL, varnish греют руками, хиты по узлам не считаются.",
        en_b="CDN rules run the new policy: edge nodes hold long TTLs, the varnish layer warms on schedule and per-node hits are counted.",
        ta=("edge", "delivery"),
        tb=("cache", "network"),
        just="opposite edge-cache policies; the CDN/edge/TTL/varnish term mass carries char5",
    ),
    dict(
        sid="place",
        ru_t="Размещение сервисов",
        en_t="Placement notes",
        ru_a="nomad-джобы переразмещены: consul-сервисы перерегистрированы, ограничения по зонам учтены, дрейф placement-весов сошёл к нулю.",
        en_a="The nomad re-placement was paused: consul services keep stale registrations, zone constraints unresolved and placement drift keeps growing.",
        ru_b="Переразмещение nomad-джоб приостановили: consul-сервисы с устаревшими регистрациями, ограничения по зонам не решены, дрейф весов растёт.",
        en_b="nomad jobs run the new placement: consul services re-registered, zone constraints honored and placement drift settled to zero.",
        ta=("orchestration", "placement"),
        tb=("scheduling", "topology"),
        just="opposite states of the nomad re-placement; the nomad/consul/placement/zone term mass carries char5",
    ),
]

# --- mono topics: zone-boundary / digit-lookalike ---------------------------
# Strict paraphrase discipline: the two sides share NO content roots; titles
# paraphrase, tags are steering knobs (disjoint / one-shared / same).
MONO_TOPICS = [
    dict(
        sid="vpn",
        lang="ru",
        tA="VPN-шлюз: статус обслуживания",
        tB="Доступ извне: предстоящие работы",
        a="Шлюз VPN уходит в обслуживание в ночь на воскресенье: сессии разорвутся один раз, переподключение пройдёт автоматически, дежурный следит за графиком.",
        b="В воскресенье ночью трогать удалённый доступ не будут: план работ отменили, разрывов сессий не планируется, наблюдение сняли.",
        just="maintenance window announced vs cancelled — same remote-access topic, opposite states",
    ),
    dict(
        sid="mail",
        lang="ru",
        tA="Почтовые рассылки: новый конвейер",
        tB="Письма и подписки: как теперь",
        a="Конвейер рассылок переехал на новый шаблон: письма собираются заново, отписки применяются сразу, очередь чистится ночью.",
        b="С почтовыми отправками вернулись к прежнему порядку: тексты компонуются как раньше, отказ от подписки подтверждается с задержкой, ночных вычисток очереди больше нет.",
        just="new mailing pipeline vs reverted mailing pipeline — same mailing topic, opposite states",
    ),
    dict(
        sid="jobs",
        lang="ru",
        tA="Очередь фоновых задач",
        tB="Регламент отложенных операций",
        a="Фоновые задачи переехали в новый воркер: ретраи ускорились, зависшие задания снимаются по таймауту, журнал ведётся в таблице.",
        b="Для отложенных операций вернули старый исполнитель: повторные запуски идут редко, подвисшие шаги ждут ручного вмешательства, протокол не заполняется.",
        just="new background worker vs returned old executor — same deferred-work topic, opposite states",
    ),
    dict(
        sid="files",
        lang="ru",
        tA="Файловое хранилище: лимиты",
        tB="Дисковое пространство: правила",
        a="Лимиты хранилища пересчитали: квота проекта выросла, превышения блокируют запись на сутки, отчёт приходит в конец недели.",
        b="Правила по диску остались прежними: объём для проекта не увеличили, при переполнении ничего не запрещается, еженедельных сводок не будет.",
        just="recalculated storage limits vs untouched limits — same storage topic, opposite states",
    ),
    dict(
        sid="cache",
        lang="ru",
        tA="Кэш объектов: режим работы",
        tB="Быстрые данные: как живут",
        a="Кэш объектов переключили на скользящее истечение: ключи продлеваются при чтении, вытеснение работает по бюджету памяти, прогрев выполняется ночью.",
        b="Для быстрых данных оставили мгновенное истечение: ничего не продлевается при обращении, вытеснение считает только дни, ночных прогревов не планируется.",
        just="sliding expiry vs instant expiry — same cache topic, opposite policies",
    ),
    dict(
        sid="imports",
        lang="ru",
        tA="Импорт данных: расписание",
        tB="Загрузка записей: как настроено",
        a="Импорт данных запускается каждые пятнадцать минут: дубликаты схлопываются, ошибки валидации копятся в отчёт, тяжёлые файлы режутся на части.",
        b="Загрузку записей переключили на ручной запуск: сверки на повторы отключены, отчёт по некорректным строкам не формируется, файлы принимаются целиком.",
        just="scheduled import vs manual-only loading — same data-import topic, opposite regimes",
    ),
    dict(
        sid="mirrors",
        lang="ru",
        tA="Зеркала пакетного репозитория",
        tB="Хранилище пакетов: обновление",
        a="Зеркала пакетного репозитория обновляются ежечасно: свежие версии появляются сразу, отставание копий алертится, чистка старых ревизий идёт по ночам.",
        b="Пакетное хранилище синхронизируется раз в неделю: свежие сборки доезжают медленно, запаздывание никто не меряет, накопившиеся редакции не вычищаются.",
        just="hourly mirrors vs weekly sync — same package-mirror topic, opposite schedules",
    ),
    dict(
        sid="logs",
        lang="ru",
        tA="Сжатие старых логов",
        tB="Архив журналов: политика",
        a="Сжатие старых логов включили после тридцати дней: файлы уходят в холодный ярус, поиск по ним остаётся доступным, экономия видна в отчёте.",
        b="Архивные записи остаются на быстрых дисках: никакой выгрузки не заведено, обращение к ним не менялось, отдельную экономию никто не считает.",
        just="compressed cold tier vs everything hot — same log-retention topic, opposite policies",
    ),
    dict(
        sid="sessions",
        lang="en",
        tA="Session store: eviction policy",
        tB="Login sessions: storage rules",
        a="The session store switched to sliding expiry: entries refresh on activity, eviction respects a memory budget and warmup runs overnight.",
        b="Login records keep constant lifetimes: usage does not prolong them, removal counts calendar days only and no night reheating exists.",
        just="sliding session expiry vs fixed lifetimes — same session topic, opposite policies",
    ),
    dict(
        sid="replicas",
        lang="en",
        tA="Database replicas: failover notes",
        tB="Read copies: promotion policy",
        a="Replica failover is automated now: lag is checked every minute, the writer is promoted without a page and the runbook keeps a manual fallback.",
        b="Handing writes to a read copy still waits for a person: staleness checks run hourly, the primary never yields by itself and the paper procedure is the only route.",
        just="automated failover vs manual-only promotion — same replica topic, opposite states",
    ),
    dict(
        sid="triage",
        lang="en",
        tA="Ticket triage rota",
        tB="Support queue: staffing",
        a="The ticket triage rota moved to paired shifts: two engineers own the inbox, escalations swap at noon and the backlog burns down daily.",
        b="Support questions reverted to a single keeper: one engineer reads the queue alone, hard cases sleep until morning and the pile shrinks only on review weeks.",
        just="paired triage shifts vs solo duty — same triage topic, opposite staffing",
    ),
    dict(
        sid="retention",
        lang="en",
        tA="Data retention tiers",
        tB="Archive lifetimes",
        a="Data retention moved to three tiers: hot weeks, warm months, cold years; purges run per tier and every deletion leaves a tombstone.",
        b="Storage lifetimes collapsed into one bucket: everything ages together, cleanup fires blindly and removed items leave no marker.",
        just="tiered retention vs one-bucket aging — same data-lifetime topic, opposite policies",
    ),
    dict(
        sid="drift",
        lang="en",
        tA="Config drift detection",
        tB="Setting skew",
        a="Config drift detection schedules hourly scans: deviations open tickets automatically, owners get paged only for critical drift and history is kept per service.",
        b="Skew scans were switched off entirely: mismatches pile up unnoticed, nobody is paged and the per-service trail is gone.",
        just="hourly drift scans vs no scanning — same config-skew topic, opposite states",
    ),
    dict(
        sid="tables",
        lang="ru",
        tA="Чистка временных таблиц",
        tB="Черновые наборы",
        a="Чистка временных таблиц запускается еженощно: мусор удаляется порциями, блокировки короткие, объём выметенного пишется в сводку.",
        b="Черновые наборы никто не выметает: лишнее копится без лимита, защёлки держатся подолгу, сводок по уборке не существует.",
        just="nightly temp-table cleanup vs never cleaned — same scratch-data topic, opposite states",
    ),
    dict(
        sid="api",
        lang="en",
        tA="API versioning policy",
        tB="Endpoint releases",
        a="API versioning adopted dated releases: every breaking change ships a new date, deprecations run a year and clients pin a version once.",
        b="Endpoint releases ride along with the code: incompatible edits arrive silently, sunset windows are gone and consumers re-verify after each deploy.",
        just="dated API releases vs floating versions — same interface-versioning topic, opposite policies",
    ),
    dict(
        sid="ratelimit",
        lang="en",
        tA="Rate limiter: current policy",
        tB="Traffic throttling: where it stands",
        a="The rate limiter moved to per-route budgets: bursts are absorbed at the edge, over-limit calls get a retry hint and dashboards track rejections.",
        b="Traffic throttling stayed on the global bucket: spikes queue in front of the gateway, blocked callers receive plain errors and no board follows them.",
        just="per-route budgets vs global bucket — same throttling topic, opposite policies",
    ),
    dict(
        sid="webhooks",
        lang="en",
        tA="Webhook delivery queue",
        tB="Outbound callbacks: delivery rules",
        a="Webhook delivery gained a second lane: retries back off exponentially, dead letters are stored for a week and consumers see stable ordering.",
        b="Outbound callbacks returned to the single pipe: repeated attempts follow a flat schedule, failed payloads vanish at once and sequence is not promised.",
        just="two-lane delivery vs single pipe — same callback topic, opposite mechanics",
    ),
    dict(
        sid="alerts",
        lang="en",
        tA="Alert routing rules",
        tB="Notification ladder: current shape",
        a="Alert routing was rewritten: critical pages go straight to on-call, warnings batch into a digest and the escalation ladder skips the team lead.",
        b="The notification ladder kept its old form: severe events ring the whole channel, minor ones wait for the morning sweep and the lead stays in the loop.",
        just="rewritten routing vs kept old ladder — same notification topic, opposite shapes",
    ),
    dict(
        sid="exports",
        lang="en",
        tA="Data export jobs",
        tB="Report downloads: automation",
        a="Export jobs now chunk large tables: retries resume from the last chunk, checksums attach to every archive and downloads expire after a day.",
        b="Automated report downloads were retired: tables stream in one piece, failed runs restart from zero and links stop working within the hour.",
        just="chunked exports vs retired automation — same export topic, opposite mechanics",
    ),
    dict(
        sid="images",
        lang="en",
        tA="Container image registry",
        tB="Build artifacts: storage plan",
        a="The container image registry enabled digest pinning: tags resolve immutably, stale layers prune weekly and pulls are counted per team.",
        b="Build artifacts keep floating references: tags re-point silently, nothing prunes on a schedule and nobody attributes the pulls.",
        just="digest pinning vs floating tags — same registry topic, opposite policies",
    ),
]

#: tag steering modes: (tags_a, tags_b) applied per mono variant index.
#: Title-sharing modes were probed and REJECTED: an identical/shared title
#: costs +0.07..+0.09 char5 (title grams dominate the small union), breaking
#: the char5 <= 0.05 band — tag modes only.
TAG_MODES = (
    ("disjoint", (["platform"], ["notes"])),
    ("one", (["platform"], ["platform", "notes"])),
    ("same", (["platform", "notes"], ["platform", "notes"])),
)

#: digit payload sentences for digit-lookalike (nouns disjoint per side so the
#: nesting stays clean; sets: A {30,45,200} superset of B1 {30} and B2 {45}).
DIG_A = {
    "ru": "Технические параметры: окно — 30 минут, буфер — 45 дней, порог — 200 событий.",
    "en": "Operational parameters: window of 30 minutes, buffer of 45 days, threshold of 200 events.",
}
DIG_B1 = {
    "ru": "В плане отмечено 30 шагов миграции.",
    "en": "The plan lists 30 handoff steps.",
}
DIG_B2 = {
    "ru": "Бар качества — 45 проверок на релиз.",
    "en": "Quality bar sits at 45 checks per release.",
}
ALT_A = {
    "ru": "Служебные числа: резерв — 12 часов, шаг — 96 записей.",
    "en": "Household numbers: a buffer of 12 hours and a step of 96 records.",
}
ALT_B = {"ru": "В журнале 96 отметок.", "en": "The ledger shows 96 entries."}

# ========================== measurement machinery ============================


def n1(s: str) -> str:
    return " ".join((s or "").split()).lower()


class Embedder:
    """Pinned nano embedder, norm1 convention (see module docstring)."""

    def __init__(self) -> None:
        from vesma.embeddings import NanoProvider  # engine src on sys.path

        provider = NanoProvider()
        pin = provider.fingerprint
        if not pin.startswith(NANO_PIN_PREFIX):
            raise mt.InstrumentAbort(
                f"embedder pin {pin!r} differs from the frozen prefix"
            )
        self._provider = provider
        self._cache: dict[str, list[float]] = {}

    def embed_text(self, text: str) -> list[float]:
        text = text[:4096]
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._provider.embed(text)
        return self._cache[key]

    def embed_side(self, s: dict) -> list[float]:
        text = f"{n1(s['title'])}\n{n1(s['body'])}\n{n1(' '.join(s['tags']))}"
        return self.embed_text(text)

    def cos(self, a: dict, b: dict) -> float:
        va, vb = self.embed_side(a), self.embed_side(b)
        return sum(x * y for x, y in zip(va, vb))


_EMB: Embedder | None = None


def emb() -> Embedder:
    global _EMB
    if _EMB is None:
        _EMB = Embedder()
    return _EMB


def char5_pair(a: dict, b: dict) -> float:
    """char5 through the FROZEN extractor (its n-gram text is norm1)."""
    ra = RecordLike(
        a["title"], a["body"], tuple(a["tags"]), a["language"], a["record_type"]
    )
    rb = RecordLike(
        b["title"], b["body"], tuple(b["tags"]), b["language"], b["record_type"]
    )
    return features(ra, rb, 0.0).values[FEATURE_NAMES.index("char5_jaccard")]


def tokset(s: str) -> frozenset[str]:
    return frozenset(n1(s).split())


def jac(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a or b) else 0.0


def digset(body: str) -> frozenset[str]:
    return frozenset(DIG_RE.findall(n1(body)))


def zeps_pair(a: dict, b: dict) -> int:
    """7 eps-flags per framing §2.4 (eps = 0.01; discrete flags exact)."""
    f = 0
    f += jac(tokset(a["body"]), tokset(b["body"])) <= EPS
    f += char5_pair(a, b) <= EPS
    f += len(tokset(a["title"]) & tokset(b["title"])) <= EPS
    f += jac(frozenset(a["tags"]), frozenset(b["tags"])) <= EPS
    f += len(digset(a["body"]) & digset(b["body"])) == 0
    f += a["language"] != b["language"]
    f += a["record_type"] != b["record_type"]
    return int(f)


def dir_pair(a: dict, b: dict) -> int:
    """digit-direction on norm1 BODIES (construction-only parameter)."""
    da, db = digset(a["body"]), digset(b["body"])
    if not da or not db:
        return 0
    if db <= da:
        return 1
    if da <= db:
        return -1
    return 0


def zone_hit(cos: float, c5: float, z: int) -> bool:
    """The ratified zone rule (cos-leg OR zeps-leg) on norm1 values."""
    return (cos <= 0.85 and c5 <= 0.05) or (z >= 3 and c5 <= 0.20)


def measure(a: dict, b: dict) -> dict:
    return {
        "cos": emb().cos(a, b),
        "char5": char5_pair(a, b),
        "zeps": zeps_pair(a, b),
        "dir": dir_pair(a, b),
    }


# ============================ spec construction ==============================


def _xl_specs() -> list[dict]:
    """Cross-language candidate pairs (record side a, sibling side b)."""
    out = []
    for fam, items in (
        ("frag", XL_FRAG),
        ("own", XL_OWN_TITLES),
        ("div", XL_DIVERSE),
        ("high", XL_HIGH),
        ("meta", XL_META),
    ):
        for t in items:
            a = {
                "title": t["a"][0],
                "body": t["a"][1],
                "tags": list(t["a"][2]),
                "language": t["a"][3],
                "record_type": "note",
            }
            b = {
                "title": t["b"][0],
                "body": t["b"][1],
                "tags": list(t["b"][2]),
                "language": t["b"][3],
                "record_type": "note",
            }
            out.append(
                {"key": f"xl-{fam}-{t['sid']}", "a": a, "b": b, "just": t["just"]}
            )
    for t in XL_TIGHT:
        out.append(
            {"key": f"xl-{t['key']}", "a": t["a"], "b": t["b"], "just": t["just"]}
        )
    return out


def _residue_specs() -> list[dict]:
    out = []
    for t in RES_TOPICS:
        a = {
            "title": t["ru_t"],
            "body": t["ru_a"],
            "tags": list(t["ta"]),
            "language": "ru",
            "record_type": "note",
        }
        b = {
            "title": t["en_t"],
            "body": t["en_a"],
            "tags": list(t["tb"]),
            "language": "en",
            "record_type": "note",
        }
        out.append({"key": f"res-{t['sid']}-1", "a": a, "b": b, "just": t["just"]})
        a2 = {
            "title": t["en_t"],
            "body": t["en_b"],
            "tags": list(t["tb"]),
            "language": "en",
            "record_type": "note",
        }
        b2 = {
            "title": t["ru_t"],
            "body": t["ru_b"],
            "tags": list(t["ta"]),
            "language": "ru",
            "record_type": "note",
        }
        out.append({"key": f"res-{t['sid']}-2", "a": a2, "b": b2, "just": t["just"]})
    return out


def _mono_variants() -> list[dict]:
    """Mono candidates: every topic x tag steering mode."""
    out = []
    for t in MONO_TOPICS:
        for mode, (ta, tb) in TAG_MODES:
            a = {
                "title": t["tA"],
                "body": t["a"],
                "tags": list(ta),
                "language": t["lang"],
                "record_type": "note",
            }
            b = {
                "title": t["tB"],
                "body": t["b"],
                "tags": list(tb),
                "language": t["lang"],
                "record_type": "note",
            }
            out.append(
                {
                    "key": f"mono-{t['sid']}-{mode}",
                    "topic": t["sid"],
                    "mode": mode,
                    "a": a,
                    "b": b,
                    "just": t["just"],
                }
            )
    return out


def _digit_specs() -> list[dict]:
    """digit-lookalike candidates: two pairs per topic (dir +1, then -1).

    pair +1: side a carries the superset digit set, side b the subset;
    pair -1: reversed roles with the ALT digit sentences (fresh bodies).
    """
    out = []
    for t in MONO_TOPICS:
        lang = t["lang"]
        a1 = {
            "title": t["tA"],
            "body": t["a"] + " " + DIG_A[lang],
            "tags": ["platform"],
            "language": lang,
            "record_type": "note",
        }
        b1 = {
            "title": t["tB"],
            "body": t["b"] + " " + DIG_B1[lang],
            "tags": ["notes"],
            "language": lang,
            "record_type": "note",
        }
        out.append(
            {
                "key": f"dig-{t['sid']}-p",
                "a": a1,
                "b": b1,
                "want_dir": 1,
                "just": t["just"]
                + " (digit superset flavor: the plan side nests inside the parameters side; construction-only dir)",
            }
        )
        a2 = {
            "title": t["tA"],
            "body": t["a"] + " " + ALT_B[lang],
            "tags": ["platform"],
            "language": lang,
            "record_type": "note",
        }
        b2 = {
            "title": t["tB"],
            "body": t["b"] + " " + ALT_A[lang],
            "tags": ["notes"],
            "language": lang,
            "record_type": "note",
        }
        out.append(
            {
                "key": f"dig-{t['sid']}-m",
                "a": a2,
                "b": b2,
                "want_dir": -1,
                "just": t["just"]
                + " (reversed digit nesting: the parameters side nests inside the ledger side; construction-only dir)",
            }
        )
    return out


# ============================== selection ====================================


def _even_spread(cands: list[dict], quota: int, value) -> list[dict]:
    """Deterministic even spread over a sorted candidate list."""
    ordered = sorted(cands, key=value)
    if len(ordered) < quota:
        raise mt.InstrumentAbort(
            f"candidate pool too small: {len(ordered)} < quota {quota}"
        )
    idx = [round(i * (len(ordered) - 1) / (quota - 1)) for i in range(quota)]
    return [ordered[i] for i in idx]


def _coverage_pick(cands: list[dict], quota: int, value) -> list[dict]:
    """Greedy largest-gap-split selection (deterministic).

    Even-spread indexing breaks on pools with density holes (it can skip the
    only point inside a hole and leave a >0.02 gap); this picker starts from
    the band ends and repeatedly inserts the candidate that minimizes the
    resulting maximum gap.
    """
    ordered = sorted(cands, key=value)
    if len(ordered) < quota:
        raise mt.InstrumentAbort(
            f"candidate pool too small: {len(ordered)} < quota {quota}"
        )
    if quota == 1:
        return [ordered[0]]
    picked = [ordered[0], ordered[-1]]
    rest = list(ordered[1:-1])
    while len(picked) < quota and rest:
        vals = sorted(value(s) for s in picked)
        best = None
        for j, s in enumerate(rest):
            v = value(s)
            trial = sorted(vals + [v])
            mg = max(trial[i + 1] - trial[i] for i in range(len(trial) - 1))
            if best is None or mg < best[0]:
                best = (mg, j)
        picked.append(rest.pop(best[1]))
    return picked


#: quiet flag: `measure` is a dry instrument — it must not write run-log rows.
QUIET = False


def _log(event: str, **fields) -> None:
    if not QUIET:
        mt.log(event, **fields)


def select_strata() -> dict[str, list[dict]]:
    """Measure every candidate pool; deterministically select the quotas.

    cos/char5/zeps/dir are MEASUREMENTS of authored specs with the pinned
    convention (the stratum definitions are cos/char5-banded — the bands are
    the ratified construction spec, not inference features).
    """
    selected: dict[str, list[dict]] = {k: [] for k in _NEW_SUBSTRATA}

    # --- zone-boundary: mono, cos in (0.84, 0.90], char5 <= 0.05 -----------
    zb_pool = []
    for s in _mono_variants():
        m = measure(s["a"], s["b"])
        if ZONE_BOUND[0] < m["cos"] <= ZONE_BOUND[1] and m["char5"] <= CHAR5_ZONE_MAX:
            s["m"] = m
            zb_pool.append(s)
    _log(
        "pool_zone_boundary",
        candidates=len(zb_pool),
        quota=QUOTA_PAIRS["zone-boundary"],
        band=sorted(round(s["m"]["cos"], 4) for s in zb_pool),
    )
    picked = _even_spread(
        zb_pool, QUOTA_PAIRS["zone-boundary"], lambda s: s["m"]["cos"]
    )
    selected["zone-boundary"] = picked

    # --- cos-bins: cross-lang, char5 <= 0.02, three bins -------------------
    xl_pool = []
    for s in _xl_specs():
        a, b = s["a"], s["b"]
        if a["language"] == b["language"]:
            raise mt.InstrumentAbort(f"{s['key']}: cross-lang family with equal langs")
        m = measure(a, b)
        if m["char5"] <= CHAR5_BINS_MAX:
            s["m"] = m
            xl_pool.append(s)
    for (lo, hi), quota in zip(COS_BINS, COS_BIN_PAIRS.values()):
        name = f"[{lo:.2f},{hi:.2f})"
        in_bin = [s for s in xl_pool if lo <= s["m"]["cos"] < hi]
        _log(
            "pool_cos_bin",
            bin=name,
            candidates=len(in_bin),
            quota=quota,
            cos_values=sorted(round(s["m"]["cos"], 4) for s in in_bin),
        )
        got = _coverage_pick(in_bin, quota, lambda s: s["m"]["cos"])
        for s in got:
            s = dict(s)
            s["bin"] = name
            selected["cos-bins"].append(s)

    # --- sibling-residue: cross-lang, char5 (0.0785, 0.20], zeps >= 3 ------
    res_pool = []
    for s in _residue_specs():
        m = measure(s["a"], s["b"])
        if CHAR5_RESIDUE[0] < m["char5"] <= CHAR5_RESIDUE[1] and m["zeps"] >= ZEPS_MIN:
            s["m"] = m
            res_pool.append(s)
    _log(
        "pool_sibling_residue",
        candidates=len(res_pool),
        quota=QUOTA_PAIRS["sibling-residue"],
        char5_values=sorted(round(s["m"]["char5"], 4) for s in res_pool),
    )
    selected["sibling-residue"] = _even_spread(
        res_pool, QUOTA_PAIRS["sibling-residue"], lambda s: s["m"]["char5"]
    )

    # --- digit-lookalike: mono, dir in {-1,+1} non-empty, char5 <= 0.05 ----
    dl_pool = {"pos": [], "neg": []}
    for s in _digit_specs():
        m = measure(s["a"], s["b"])
        if m["char5"] > CHAR5_ZONE_MAX:
            continue
        if m["dir"] != s["want_dir"]:
            raise mt.InstrumentAbort(
                f"{s['key']}: construction dir {m['dir']} != declared {s['want_dir']}"
            )
        da, db = digset(s["a"]["body"]), digset(s["b"]["body"])
        if not da or not db:
            raise mt.InstrumentAbort(f"{s['key']}: empty digit set (guard)")
        s["m"] = m
        dl_pool["pos" if s["want_dir"] == 1 else "neg"].append(s)
    _log(
        "pool_digit_lookalike",
        candidates_pos=len(dl_pool["pos"]),
        candidates_neg=len(dl_pool["neg"]),
        quota=QUOTA_PAIRS["digit-lookalike"],
    )
    half = QUOTA_PAIRS["digit-lookalike"] // 2
    picked = _even_spread(
        dl_pool["pos"], half, lambda s: s["m"]["char5"]
    ) + _even_spread(
        dl_pool["neg"], QUOTA_PAIRS["digit-lookalike"] - half, lambda s: s["m"]["char5"]
    )
    selected["digit-lookalike"] = picked
    return selected


# ============================ cluster emission ===============================


def _mk_member(mid: str, s: dict, ts: str) -> lib.Member:
    return lib.Member(
        mid=mid,
        title=s["title"],
        body=s["body"],
        tags=tuple(s["tags"]),
        record_type=s["record_type"],
        language=s["language"],
        created_at=ts,
    )


def _decl(substratum: str, justification: str) -> dict:
    return {
        "kind": "theme-near",
        "mode": substratum,
        "facts": {},
        "steps": [],
        "justification": justification,
    }


def _scenario_of(substratum: str, s: dict) -> str:
    if substratum == "cos-bins":
        return f"{s['key']}|cos-bins|{s['bin']}"
    if substratum == "digit-lookalike":
        return f"{s['key']}|dir{'pos' if s['want_dir'] == 1 else 'neg'}"
    return f"{s['key']}|{substratum}"


def _emit_both_orientations(
    substratum: str, s: dict, rng: random.Random
) -> list[lib.Cluster]:
    lo, hi = lib._CREATED_LO, lib._CREATED_HI - lib._CREATED_SPREAD
    out = []
    for orient in ("AB", "BA"):
        mids = (f"{s['key']}-a-{orient}", f"{s['key']}-b-{orient}")
        members = [
            _mk_member(mids[0], s["a"], lib._iso(rng.randint(lo, hi))),
            _mk_member(mids[1], s["b"], lib._iso(rng.randint(lo, hi))),
        ]
        if orient == "BA":
            members.reverse()
        out.append(
            lib.Cluster(
                cid=f"ma-neg2-{s['key']}-{orient}",
                stratum="theme-near",
                substratum=substratum,
                members=members,
                label={"type": "abstain"},
                declaration=_decl(substratum, s["just"]),
                scenario=_scenario_of(substratum, s),
                orientation=orient,
            )
        )
    return out


def build_clusters(selected: dict[str, list[dict]]) -> list[lib.Cluster]:
    seeds = {
        "zone-boundary": SEED_ZONE_BOUNDARY,
        "cos-bins": SEED_COS_BINS,
        "sibling-residue": SEED_SIBLING_RESIDUE,
        "digit-lookalike": SEED_DIGIT_LOOKALIKE,
    }
    clusters: list[lib.Cluster] = []
    for substratum in _NEW_SUBSTRATA:
        rng = random.Random(seeds[substratum])
        for s in selected[substratum]:
            a, b = dict(s["a"]), dict(s["b"])
            _validate_side(a, "neg2.a", s)
            _validate_side(b, "neg2.b", s)
            if not isinstance(s["just"], str) or not s["just"].strip():
                raise mt.InstrumentAbort(f"{s['key']}: empty justification")
            clusters.extend(_emit_both_orientations(substratum, s, rng))
    return clusters


# ================================ asserts ====================================


def _pair_view(c: lib.Cluster) -> tuple[dict, dict]:
    a, b = c.members
    return (
        {
            "title": a.title,
            "body": a.body,
            "tags": list(a.tags),
            "language": a.language,
            "record_type": a.record_type,
        },
        {
            "title": b.title,
            "body": b.body,
            "tags": list(b.tags),
            "language": b.language,
            "record_type": b.record_type,
        },
    )


def assert_neg_silver(clusters: list[lib.Cluster]) -> None:
    """A-neg-silver: every pair FAILS the §A dup predicate.

    Predicate legs (ADDENDUM-3 mechanics): norm1 body+title identity AND
    tags/type/language equality (then razor). Every new pair fails the
    norm1-identity leg (bodies differ by construction); cross-lang pairs
    also fail the language leg. Razor guard: char5 <= 0.90 through the
    frozen extractor. zone-boundary: char5 <= 0.05 asserted TRUE on norm1
    (the brief's watch item — the extractor's n-gram text IS norm1).
    """
    worst = 0.0
    for c in clusters:
        a, b = _pair_view(c)
        same_norm = n1(a["body"]) == n1(b["body"])
        same_env = (
            n1(a["title"]) == n1(b["title"])
            and frozenset(a["tags"]) == frozenset(b["tags"])
            and a["record_type"] == b["record_type"]
            and a["language"] == b["language"]
        )
        if same_norm and same_env:
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} passes the dup surface leg"
            )
        if a["language"] != b["language"] and c.substratum not in (
            "cos-bins",
            "sibling-residue",
        ):
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} cross-language outside the cross-lang strata"
            )
        if a["language"] == b["language"] and c.substratum in (
            "cos-bins",
            "sibling-residue",
        ):
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} cross-lang stratum with equal langs"
            )
        j = char5_pair(a, b)
        worst = max(worst, j)
        if j > 0.90:
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: {c.cid} char5_jaccard={j:.4f} > 0.90"
            )
        if c.substratum == "zone-boundary" and j > CHAR5_ZONE_MAX:
            raise mt.InstrumentAbort(
                f"A-neg-silver RED: zone-boundary {c.cid} char5 norm1 = {j:.4f} > 0.05"
            )
    mt.log(
        "a_neg_silver",
        pairs=len(clusters),
        max_char5_jaccard=round(worst, 4),
        predicate="fails §A dup: norm1 body identity leg (cross-lang: also "
        "language leg); char5 <= 0.90 razor guard; zone-boundary "
        "char5 <= 0.05 verified TRUE on norm1 (frozen extractor)",
    )


def assert_neg_disjoint(
    clusters: list[lib.Cluster],
    surface_hashes: set[str],
    train_hashes: set[str],
    neg1_hashes: set[str],
    scaffold_hashes: set[str],
    holdout_ids: set[str],
    sheet_ids: set[str],
) -> None:
    """A-neg-disjoint: 0 body hash vs train / v1.1 negatives / surface /
    scaffolds; 0 cluster+member id vs holdout id-manifest / sheet ids."""
    neg_ids = {c.cid for c in clusters} | {m.mid for c in clusters for m in c.members}
    neg_h = {mt.sha256_text(m.body) for c in clusters for m in c.members}
    hits = {
        "surface": neg_h & surface_hashes,
        "train": neg_h & train_hashes,
        "negatives_v1": neg_h & neg1_hashes,
        "raw_scaffold": neg_h & scaffold_hashes,
        "holdout_ids": neg_ids & holdout_ids,
        "sheet_ids": neg_ids & sheet_ids,
    }
    mt.log(
        "a_neg_disjoint",
        body_hashes=len(neg_h),
        hits={k: sorted(v)[:3] for k, v in hits.items()},
        sheet_read="sheet-selection.json, ID LEVEL ONLY, facts logged",
        note="fresh authored texts; overlap with the surface computed "
        "from the recipe: v2 content shares no batch family with the "
        "v4/v4.3 surface texts",
    )
    if any(hits.values()):
        raise mt.InstrumentAbort(
            "A-neg-disjoint RED: "
            + " ".join(f"{k}={sorted(v)[:2]}" for k, v in hits.items() if v)
        )


def assert_zone_coverage(
    existing_neg: list[lib.Cluster], new_clusters: list[lib.Cluster]
) -> dict:
    """A-zone-coverage: negatives with char5 <= 0.05 cover [0.60, 0.85]
    continuously from the 0.60 edge (edge gap <= 0.02, max adjacent gap
    <= 0.02). Honest-red baseline: edge 0.1301 (first covered 0.7301).
    The (0.85, 0.90] pocket coverage is REPORTED, not gated."""
    pts = []
    for c in existing_neg + new_clusters:
        a, b = _pair_view(c)
        m = measure(a, b)
        if m["char5"] <= CHAR5_ZONE_MAX:
            pts.append((m["cos"], c.cid, c.substratum))
    band = sorted(p for p in pts if COVERAGE_LO <= p[0] <= COVERAGE_HI)
    if not band:
        raise mt.InstrumentAbort("A-zone-coverage RED: no negatives in band")
    edge = band[0][0] - COVERAGE_LO
    gaps = [round(band[i + 1][0] - band[i][0], 6) for i in range(len(band) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    by_substratum: dict[str, int] = {}
    for _, _, st in band:
        by_substratum[st] = by_substratum.get(st, 0) + 1
    pocket = sorted(p[0] for p in pts if COVERAGE_HI < p[0] <= 0.90)
    mt.log(
        "a_zone_coverage",
        edge_gap=round(edge, 4),
        max_gap=max_gap,
        first=round(band[0][0], 4),
        last=round(band[-1][0], 4),
        points=len(band),
        worst_gap_windows=[
            round(band[i][0], 4)
            for i in range(len(band) - 1)
            if band[i + 1][0] - band[i][0] > GAP_MAX
        ],
        by_substratum=by_substratum,
        pocket_085_090={
            "points": len(pocket),
            "gaps_max": round(
                max(
                    (pocket[i + 1] - pocket[i] for i in range(len(pocket) - 1)),
                    default=0.0,
                ),
                4,
            ),
        },
        zone_rule="negatives with char5 <= 0.05, cos in [0.60, 0.85]; "
        "convention: nano 3b752e06 + norm1 (ADDENDUM-6 pos 6)",
    )
    if edge > GAP_MAX:
        raise mt.InstrumentAbort(
            f"A-zone-coverage RED: edge gap {edge:.4f} > {GAP_MAX} "
            f"(first covered {band[0][0]:.4f})"
        )
    if max_gap > GAP_MAX:
        raise mt.InstrumentAbort(
            f"A-zone-coverage RED: max adjacent gap {max_gap:.4f} > {GAP_MAX}"
        )
    mt.log(
        "a_zone_coverage_GREEN",
        edge_gap=round(edge, 4),
        max_gap=max_gap,
        note="continuity from the 0.60 edge holds after generation",
    )
    return {"edge": edge, "max_gap": max_gap, "points": len(band)}


def assert_zone_ch3_zero() -> None:
    """A-zone-ch3-zero: 0 of the 276 CH-3 pairs (incl. transitive chain
    pairs) inside the zone on norm1."""
    clusters = mt.load_train()
    ch3 = [c for c in clusters if c.stratum == "CH-3"]
    pairs = []
    for c in ch3:
        members = c.members
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    hits = []
    worst = {"cos": 1.0, "char5": 1.0, "zeps": 0}
    for x, y in pairs:
        a = {
            "title": x.title,
            "body": x.body,
            "tags": list(x.tags),
            "language": x.language,
            "record_type": x.record_type,
        }
        b = {
            "title": y.title,
            "body": y.body,
            "tags": list(y.tags),
            "language": y.language,
            "record_type": y.record_type,
        }
        m = measure(a, b)
        worst["cos"] = min(worst["cos"], m["cos"])
        worst["char5"] = min(worst["char5"], m["char5"])
        worst["zeps"] = max(worst["zeps"], m["zeps"])
        if zone_hit(m["cos"], m["char5"], m["zeps"]):
            hits.append((x.mid, y.mid))
    mt.log(
        "a_zone_ch3_zero",
        pairs=len(pairs),
        hits=len(hits),
        sample=hits[:3],
        worst=worst,
        note="276 = 264 author pairs + 12 transitive chain pairs; zone "
        "rule (cos<=0.85 & char5<=0.05) OR (zeps>=3 & char5<=0.20)",
    )
    if hits:
        raise mt.InstrumentAbort(f"A-zone-ch3-zero RED: {len(hits)} CH-3 pairs in zone")


def preflight_fixtures(selected: dict[str, list[dict]]) -> None:
    """П3 (ML-STD-0002): fixture cell of EACH new substratum through the
    full assert map BEFORE emission."""
    cells = []
    for substratum in _NEW_SUBSTRATA:
        s = dict(selected[substratum][0])
        s["a"], s["b"] = dict(s["a"]), dict(s["b"])
        cell = lib.Cluster(
            cid=f"fx-v2-{substratum}",
            stratum="theme-near",
            substratum=substratum,
            members=[
                _mk_member(f"fx-v2-{substratum}-a", s["a"], lib._iso(lib._CREATED_LO)),
                _mk_member(f"fx-v2-{substratum}-b", s["b"], lib._iso(lib._CREATED_LO)),
            ],
            label={"type": "abstain"},
            declaration=_decl(substratum, s["just"]),
        )
        cells.append((cell, substratum, s))

    all_ok = True
    for c, substratum, s in cells:
        a, b = _pair_view(c)
        silver_fails = not (
            n1(a["body"]) == n1(b["body"])
            and n1(a["title"]) == n1(b["title"])
            and frozenset(a["tags"]) == frozenset(b["tags"])
            and a["record_type"] == b["record_type"]
            and a["language"] == b["language"]
        )
        label_ok = lib.correct_set_ok(c, None) is False
        cls = lib.classify(c.members, declarations=None)
        routing_ok = cls == "undefined"
        decl_ok = (
            c.declaration["kind"] == "theme-near"
            and c.declaration["mode"] == substratum
            and bool(c.declaration["justification"].strip())
        )
        m = measure(a, b)
        if substratum == "zone-boundary":
            stratum_ok = (
                ZONE_BOUND[0] < m["cos"] <= ZONE_BOUND[1]
                and m["char5"] <= CHAR5_ZONE_MAX
            )
        elif substratum == "cos-bins":
            stratum_ok = m["char5"] <= CHAR5_BINS_MAX and any(
                lo <= m["cos"] < hi for lo, hi in COS_BINS
            )
        elif substratum == "sibling-residue":
            stratum_ok = (
                CHAR5_RESIDUE[0] < m["char5"] <= CHAR5_RESIDUE[1]
                and m["zeps"] >= ZEPS_MIN
            )
        else:
            stratum_ok = (
                m["dir"] in (-1, 1)
                and m["char5"] <= CHAR5_ZONE_MAX
                and bool(digset(a["body"]))
                and bool(digset(b["body"]))
            )
        row_ok = silver_fails and label_ok and routing_ok and decl_ok and stratum_ok
        all_ok = all_ok and row_ok
        mt.log(
            "preflight_fixture",
            cell=c.cid,
            substratum=substratum,
            silver_fails=silver_fails,
            label_ok=label_ok,
            route_class=cls,
            routing_ok=routing_ok,
            decl_ok=decl_ok,
            stratum_ok=stratum_ok,
            measured=m,
            ok=row_ok,
        )
    if not all_ok:
        raise mt.InstrumentAbort(
            "preflight RED on v2 fixtures — emission forbidden (П3)"
        )
    mt.log(
        "preflight_pass",
        cells=4,
        note="fixture cells of every new substratum; full assert map green "
        "BEFORE emission",
    )


# ================================ commands ===================================


def _prologue() -> dict:
    mt._SLICE = SLICE
    mt.log(
        "cost_declaration",
        expected_prolog_s=3,
        expected_cycle_s=120,
        expected_peak_ram_gb=1.5,
        lane="light (draft §9)",
    )
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
        "negatives_v1_sha256": mt.sha256_file(NEG1_PATH),
        "noharm_surface_sha256": mt.sha256_file(NOHARM_SURFACE),
        "merge_arbiter_lib_sha256": mt.sha256_file(Path(mt.lib.__file__)),
        "cortex_features_pair_sha256": mt.sha256_file(
            REPO / "src" / "cortex" / "features" / "pair.py"
        ),
        "gen_v43_strategies_sha256": mt.sha256_file(
            REPO / "scripts" / "gen_dataset_v43_strategies.py"
        ),
    }
    if (
        pins["train_jsonl_sha256"]
        != "e60bc7d05bd0348d8811d094b890c0931c9495b9ae03169951232bf547b93cd6"
    ):
        raise mt.InstrumentAbort("train.jsonl diverges from the sealed v1 pin")
    if (
        pins["negatives_v1_sha256"]
        != "e7002dfc42eebd4d95a73346c1525336b8ac998debba0e6d3727517636851339"
    ):
        raise mt.InstrumentAbort("train-negatives.jsonl diverges from the v1.1 pin")
    mt.log(
        "frozen_inputs",
        pins=pins,
        input_list=[
            "corpus manifest + train.jsonl + train-negatives.jsonl (pins)",
            "noharm surface (hash only)",
            "sheet-selection.json (ID LEVEL ONLY)",
            "holdout id-manifest (ID ONLY, no content)",
            "frozen generators (lib, gen_dataset_v43_strategies, cortex.features.pair)",
            "engine vesma release/6.6.0 + nano 3b752e06 (measured axes)",
        ],
        committee_scored_in_inputs=False,
        sealed_inputs=[
            "NONE — MA-holdout-choice content, spot-sheet "
            "render/verdicts/internal/committee-scored are NOT "
            "in the input list"
        ],
    )

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

    # ratification assert: ADDENDUM-6 at HEAD + the §7 execution record with
    # the fixed share number present (the number preceded generation).
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "f2b51bd", "HEAD"],
        cwd=REPO,
        capture_output=True,
    )
    doc = REPO / "docs" / "specs" / "merge-arbiter-corpus-prereg-ADDENDUM-6.md"
    ok = (
        r.returncode == 0
        and doc.exists()
        and "## 7." in doc.read_text(encoding="utf-8")
        and SHARE_STRING in doc.read_text(encoding="utf-8")
    )
    mt.log(
        "activation_addendum6",
        commit="f2b51bd",
        is_ancestor=r.returncode == 0,
        quota_fix_in_prereg=True,
        share_string=SHARE_STRING,
        ok=ok,
    )
    if not ok:
        raise mt.InstrumentAbort(
            "ADDENDUM-6 §7 quota fix not at HEAD — no ratified quota"
        )
    return {"holdout_ids": holdout_ids, "sheet_ids": sheet_ids, "pins": pins}


def _load_existing_negatives() -> list[lib.Cluster]:
    out = []
    for line in open(NEG1_PATH, encoding="utf-8"):
        if line.strip():
            out.append(mt._cluster_from_row(json.loads(line)))
    return out


def cmd_measure() -> int:
    """Measurement instrument: pool yields per stratum (stdout; no emission,
    no run-log writes, no file writes)."""
    global QUIET
    QUIET = True
    selected = select_strata()
    for st, specs in selected.items():
        print(f"== {st}: {len(specs)} pairs selected")
        for s in specs:
            m = s["m"]
            extra = s.get("bin", "")
            print(
                f"  {s['key']:24s} cos={m['cos']:.4f} char5={m['char5']:.4f} "
                f"zeps={m['zeps']} dir={m['dir']} {extra}"
            )
    return 0


def cmd_build() -> int:
    ids = _prologue()

    mt.log(
        "seeds_pinned",
        seeds=SEED_LIST,
        per_substratum={
            "zone-boundary": SEED_ZONE_BOUNDARY,
            "cos-bins": SEED_COS_BINS,
            "sibling-residue": SEED_SIBLING_RESIDUE,
            "digit-lookalike": SEED_DIGIT_LOOKALIKE,
            "null-selection": SEED_NULL,
        },
        fresh="!= 42 family, != 43..62 floor family, != 9001-9003/9101-9103 "
        "negatives family, != 9201 null family",
        share_fixed_before_generation=SHARE_STRING,
        share_value=round(SHARE_CROSS, 4),
        note="quotas + cross-language share ratified into ADDENDUM-6 §7 "
        "BEFORE generation; seeds pinned BEFORE generation",
    )

    selected = select_strata()
    clusters = build_clusters(selected)
    counts: dict[str, int] = {}
    for c in clusters:
        counts[c.substratum] = counts.get(c.substratum, 0) + 1
    want = {k: QUOTA_PAIRS[k] * 2 for k in _NEW_SUBSTRATA}
    if counts != want:
        raise mt.InstrumentAbort(f"A-count RED: {counts} != {want}")
    mt.log(
        "a_count",
        rows=counts,
        rows_target=want,
        pairs={k: QUOTA_PAIRS[k] for k in _NEW_SUBSTRATA},
        negative_mass={
            "existing": EXISTING_NEG_ROWS,
            "new": len(clusters),
            "total": NEG_MASS_ROWS,
            "cross_lang": CROSS_MASS_ROWS,
            "cross_share": round(SHARE_CROSS, 4),
            "share_string": SHARE_STRING,
            "gate": "cross-share <= 1/2 (ADDENDUM-6 §2 pos 5, §7)",
        },
    )
    if SHARE_CROSS > 0.5:
        raise mt.InstrumentAbort("cross-language share exceeds 1/2")

    # A-symmetry: every scenario exactly 2 mirrored rows with the same bodies
    by_scen: dict[str, dict[str, str]] = {}
    for c in clusters:
        by_scen.setdefault(c.scenario, {})[c.orientation] = c.cid
    if any(sorted(v) != ["AB", "BA"] for v in by_scen.values()):
        raise mt.InstrumentAbort("A-symmetry RED: scenario without a mirror row")
    body_by_cid = {c.cid: frozenset(m.body for m in c.members) for c in clusters}
    for scen, orients in by_scen.items():
        if body_by_cid[orients["AB"]] != body_by_cid[orients["BA"]]:
            raise mt.InstrumentAbort(f"A-symmetry RED: {scen} bodies differ")
    mt.log("a_symmetry", scenarios=len(by_scen), ok=True)

    preflight_fixtures(selected)

    existing_neg = _load_existing_negatives()
    train_v1 = mt.load_train()
    train_hashes = {mt.sha256_text(m.body) for c in train_v1 for m in c.members}
    neg1_hashes = {mt.sha256_text(m.body) for c in existing_neg for m in c.members}
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
        clusters,
        surface_hashes,
        train_hashes,
        neg1_hashes,
        scaffold_hashes,
        ids["holdout_ids"],
        ids["sheet_ids"],
    )
    coverage = assert_zone_coverage(existing_neg, clusters)
    assert_zone_ch3_zero()

    if NEG2_PATH.exists():
        raise mt.InstrumentAbort(
            "train-negatives-v2.jsonl exists — re-emission forbidden"
        )
    with open(NEG2_PATH, "w", encoding="utf-8") as fh:
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
    neg2_sha = mt.sha256_file(NEG2_PATH)
    manifest = {
        "regime": "ADDENDUM-6 §2 pos 5 train extension: 4 theme-near strata "
        "(zone-boundary, cos-bins, sibling-residue, digit-lookalike); "
        "extension, not re-emit — train.jsonl / holdout / v1.1 "
        "negatives / null-envelope v1 byte-untouched; ML-STD-0002 "
        "П1-П6 map: A-neg-silver, A-neg-disjoint, A-zone-coverage "
        "(GREEN after generation), A-zone-ch3-zero (276 pairs), "
        "A-count, A-symmetry, preflight fixtures PASS before "
        "emission; cross-language share fixed in ADDENDUM-6 §7 "
        "BEFORE generation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "seeds": SEED_LIST,
        "quota_pairs": QUOTA_PAIRS,
        "quota_rows": want,
        "cos_bin_rows": {
            f"[{lo:.2f},{hi:.2f})": COS_BIN_PAIRS[f"[{lo:.2f},{hi:.2f})"] * 2
            for lo, hi in COS_BINS
        },
        "n": len(clusters),
        "negatives_v2_sha256": neg2_sha,
        "base_train_sha256": ids["pins"]["train_jsonl_sha256"],
        "base_negatives_v1_sha256": ids["pins"]["negatives_v1_sha256"],
        "negative_mass": {
            "existing": EXISTING_NEG_ROWS,
            "new": len(clusters),
            "total": NEG_MASS_ROWS,
            "cross_lang": CROSS_MASS_ROWS,
            "cross_share": round(SHARE_CROSS, 4),
            "share_string": SHARE_STRING,
        },
        "convention": {
            "embedder_pin": lib.EMBEDDER_PIN,
            "engine": "/var/home/abyss/LABs/Projects/Project-Vesma/vesma @ release/6.6.0 (458fffc)",
            "embed_text": "n1(title)\\nn1(body)\\nn1(space-joined tags), [:4096]; "
            "n1 = lowercase + whitespace collapse",
            "char5": "cortex.features.pair (frozen; n-gram text is norm1)",
            "zeps": "7 eps-flags, eps=0.01, framing §2.4 (digits flag on empty "
            "intersection); DE re-implementation",
            "dir": "digit sets on norm1 bodies, regex \\d+([.,:]\\d+)*; "
            "construction-only (ADDENDUM-6 pos 4)",
            "zone": "A-zone-coverage RED baseline reproduced: edge 0.1301, "
            "first covered 0.7301",
        },
        "coverage_after_generation": {
            "edge_gap": round(coverage["edge"], 4),
            "max_gap": coverage["max_gap"],
            "points": coverage["points"],
        },
        "justification_validator": "gen_dataset_v43_strategies._validate_side "
        "+ non-empty justification (A-labels carried)",
    }
    NEG2_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    mt.log(
        "negatives_v2_emitted",
        path=str(NEG2_PATH),
        sha256=neg2_sha,
        counts=counts,
        negative_mass=manifest["negative_mass"],
        zone_coverage=manifest["coverage_after_generation"],
        untouched=[
            str(mt.TRAIN_JSONL),
            "holdout.jsonl",
            str(NEG1_PATH),
            "null-envelope.json",
            "spot-sheet/",
        ],
        note="extension, not re-emit; the 600 + 200 + seals stay byte-identical",
    )
    return 0


NULL_QUOTA_V2 = {  # round(1/3 * rows) per new substratum
    "zone-boundary": 19,
    "cos-bins": 13,
    "sibling-residue": 8,
    "digit-lookalike": 20,
}
NULL_COVER_MIN = 8  # v2: 1/3 of the smallest stratum (sibling-residue, 24
# rows); ADDENDUM-4's >= 20 assumed 60-row strata —
# proportional deviation, documented in the run-log and
# the report


def cmd_null_coverage() -> int:
    """Select ~1/3 of the NEW clusters per new substratum into
    null-envelope-v2.json (A-null-indep / A-null-lang / A-null-cover)."""
    mt._SLICE = SLICE
    mt.log(
        "cost_declaration",
        expected_prolog_s=1,
        expected_cycle_s=5,
        expected_peak_ram_gb=0.5,
        lane="light (draft §9)",
    )
    mt.log(
        "seeds_pinned",
        seeds=[SEED_NULL],
        purpose="null-cluster selection within the new strata",
        fresh="!= 42/43-62/9001-9103/9201 families",
        note="pinned BEFORE selection",
    )

    rows = [
        json.loads(line) for line in open(NEG2_PATH, encoding="utf-8") if line.strip()
    ]
    if not rows:
        raise mt.InstrumentAbort("train-negatives-v2.jsonl missing — run build first")
    if NULL2_PATH.exists():
        raise mt.InstrumentAbort(
            "null-envelope-v2.json already exists — re-emission forbidden"
        )
    by_sub: dict[str, list[str]] = {}
    for r in rows:
        by_sub.setdefault(r["substratum"], []).append(r["cluster_id"])
    got = {k: len(v) for k, v in sorted(by_sub.items())}
    mt.log("stratum_volumes", volumes=got)

    rng = random.Random(SEED_NULL)
    chosen: list[str] = []
    for sub, quota in sorted(NULL_QUOTA_V2.items()):
        pool = sorted(by_sub[sub])
        if quota > len(pool):
            raise mt.InstrumentAbort(
                f"null quota {quota} > volume {len(pool)} for {sub}"
            )
        chosen.extend(rng.sample(pool, quota))
    chosen_set = set(chosen)

    shares = {
        sub: round(NULL_QUOTA_V2[sub] / len(by_sub[sub]), 4) for sub in NULL_QUOTA_V2
    }
    bad = {k: v for k, v in shares.items() if abs(v - 1 / 3) > 0.05}
    if bad:
        raise mt.InstrumentAbort(f"A-null-indep RED: shares {bad}")
    ab = sum(
        1 for r in rows if r["cluster_id"] in chosen_set and r["orientation"] == "AB"
    )
    ba = sum(
        1 for r in rows if r["cluster_id"] in chosen_set and r["orientation"] == "BA"
    )
    mt.log(
        "a_null_indep",
        shares=shares,
        orientation={"AB": ab, "BA": ba},
        note="exact 1/3 shares by construction; orientation split within "
        "sampling noise; label independence by uniform selection "
        "inside strata (all v2 rows are abstain-labeled)",
    )
    span = max(shares.values()) - min(shares.values())
    if span > 0.05:
        raise mt.InstrumentAbort("A-null-lang RED: theme-near v2 shares diverge")
    if any(v < NULL_COVER_MIN for v in NULL_QUOTA_V2.values()):
        raise mt.InstrumentAbort(f"A-null-cover RED: {NULL_QUOTA_V2}")
    mt.log(
        "a_null_lang_cover",
        shares=shares,
        cover=NULL_QUOTA_V2,
        ok=True,
        cover_min=NULL_COVER_MIN,
        note="cover threshold 8 = 1/3 of the smallest v2 stratum "
        "(sibling-residue, 24 rows); ADDENDUM-4's 20 assumed 60-row "
        "strata — proportional carry, deviation documented",
    )

    doc = {
        "version": "null-coverage-v2 (ADDENDUM-6 §2 pos 5 null inheritance)",
        "seed": SEED_NULL,
        "cluster_ids": sorted(chosen_set),
        "quota": NULL_QUOTA_V2,
        "shares": shares,
        "semantics": "both-null pairs only: every member of a listed cluster "
        "gets created_at=None (0 mixed-null by construction); "
        "base files untouched (v1 null-envelope.json separate)",
        "base_files_untouched": [
            str(mt.TRAIN_JSONL),
            str(NEG1_PATH),
            "null-envelope.json",
        ],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    NULL2_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    null_sha = mt.sha256_file(NULL2_PATH)
    fp = mt.sha256_text(
        mt.sha256_file(mt.TRAIN_JSONL)
        + ":"
        + mt.sha256_file(NEG1_PATH)
        + ":"
        + mt.sha256_file(NEG2_PATH)
        + ":"
        + null_sha
    )
    mt.log(
        "null_coverage_v2_emitted",
        path=str(NULL2_PATH),
        sha256=null_sha,
        n_null_clusters=len(chosen_set),
        corpus_fp_v2=fp,
        base_files_untouched=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["measure", "build", "null-coverage"])
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        return cmd_measure()
    if args.cmd == "build":
        return cmd_build()
    return cmd_null_coverage()


if __name__ == "__main__":
    raise SystemExit(main())
