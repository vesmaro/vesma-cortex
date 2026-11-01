#!/usr/bin/env python3
"""Conflict-detector corpus library — content, generators, predicates, asserts.

Corpus v1 for the conflict-detector Noul line (ADR 0002 duty #6), spec:
``docs/experiments/conflict-detector-corpus-design.md`` (RATIFIED, MLC-9)
+ ``-ADDENDUM-1`` (ladder re-edge, chair 2026-11-01). No LLM anywhere:
every label is a function of construction (policy 4).

Frozen-contract discipline: the normalizer and features are IMPORTED from
``cortex.features.pair`` (0 reimplementations, lesson #480); the embedder
is the pinned engine nano (A-frozen) — the pin is verified by the caller
and a mismatch is a loud refusal. All text is AUTHORED synthetic content
(ADR 0003 R2 publication class); no store lines.

Determinism: pure functions + explicit ``random.Random(seed)``.
"""

from __future__ import annotations

import difflib
import hashlib
import random
import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# --- frozen consumer contract (A-frozen) ----------------------------------

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from cortex.features.pair import RecordLike, features  # noqa: E402

EMBEDDER_PIN = (
    "3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
)
GUARD_VERSION = "cg-guard-v1"

CAP = 0.05   # A-scaffold: max share of a class's rows on one scaffold
GAP = 12     # difflib region merge gap (B4 prereg 2.1 constants)
COVER = 0.8  # region must cover >= 0.8 of the declared slot span


def norm1(text: str) -> str:
    """Frozen normalizer surface (imported module semantics, B4 lib idiom)."""
    return " ".join(f"{text}".split()).lower()


def n1(text: str) -> str:
    return " ".join(text.split()).lower()


def pair_char5_jaccard(side_a: Side, side_b: Side, cos: float) -> float:
    """Frozen 5-gram feature through cortex.features.pair (no reimpl):
    the pair's char5_jaccard at the frozen contract position (index 3);
    cos comes from the pinned embedder (features() takes it as input)."""
    rec_a = RecordLike(title=side_a.title, body=side_a.body, tags=tuple(side_a.tags),
                       language=side_a.language, record_type=side_a.record_type)
    rec_b = RecordLike(title=side_b.title, body=side_b.body, tags=tuple(side_b.tags),
                       language=side_b.language, record_type=side_b.record_type)
    return float(features(rec_a, rec_b, cos)[3])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- authored content ------------------------------------------------------

#: Subjects: id -> (ru_anchor, en_anchor, ru_noun_gen, ru_dat)
SUBJECTS = {
    "s3conn":  ("Коннектор объектного хранилища S3", "S3 object storage connector",
                "коннектора хранилища", "хранилищу"),
    "retry":   ("Воркер повторных попыток", "Retry worker",
                "воркера ретраев", "воркеру"),
    "sched":   ("Планировщик фоновых задач", "Background job scheduler",
                "планировщика задач", "планировщику"),
    "cache":   ("Кэш пользовательских сессий", "User session cache",
                "кэша сессий", "кэшу"),
    "export":  ("Модуль экспорта отчётов", "Report export module",
                "модуля экспорта", "модулю экспорта"),
    "recs":    ("Модуль рекомендаций", "Recommendations module",
                "модуля рекомендаций", "модулю рекомендаций"),
    "pay":     ("Виджет оплаты", "Payment widget",
                "виджета оплаты", "виджету оплаты"),
    "logs":    ("Парсер логов", "Log parser",
                "парсера логов", "парсеру логов"),
    "auth":    ("Модуль аутентификации", "Authentication module",
                "модуля аутентификации", "модулю аутентификации"),
    "queue":   ("Очередь уведомлений", "Notification queue",
                "очереди уведомлений", "очереди уведомлений"),
    "mlpipe":  ("Пайплайн обучения моделей", "Model training pipeline",
                "пайплайна обучения", "пайплайну"),
    "bill":    ("Сервис биллинга", "Billing service",
                "сервиса биллинга", "сервису биллинга"),
    # non-tech subjects: needed for same-language NC-5 siblings below the
    # template-twin ridge (same-lang cross-domain pairs measure ~0.60-0.75)
    "cook":    ("Рецепт борща", "Borscht recipe",
                "рецепта борща", "рецепту"),
    "fit":     ("Программа тренировок", "Workout program",
                "программы тренировок", "программе тренировок"),
    "weather": ("Прогноз погоды на неделю", "Weekly weather forecast",
                "прогноза погоды", "прогнозу погоды"),
    "music":   ("Плейлист для фокуса", "Focus playlist",
                "плейлиста", "плейлисту"),
    "travel":  ("Маршрут выходного дня", "Weekend itinerary",
                "маршрута выходного дня", "маршруту"),
    "books":   ("Книжный клуб: выбор месяца", "Book club monthly pick",
                "выбора книжного клуба", "книжному клубу"),
}

#: tech subjects (CD classes live here); non-tech only feed NC-5.
TECH = list(SUBJECTS)[:12]
NONTECH = list(SUBJECTS)[12:]

BASE_RU = [
    "Компонент развёрнут во внутреннем контуре и обслуживает продуктовые нагрузки.",
    "События о работе компонента пишутся в журнал и доступны в дашборде наблюдаемости.",
    "Конфигурация читается при старте из файла настроек команды.",
]
BASE_EN = [
    "The component runs in the internal perimeter and serves production traffic.",
    "Component events are written to the log and visible on the observability dashboard.",
    "Configuration is read at startup from the team settings file.",
]

FILLERS_RU = [
    "Подробная инструкция по настройке приведена во внутренней документации команды.",
    "Изменения параметров согласуются с дежурной сменой заранее.",
    "Метрики компонента экспортируются в общий мониторинг каждую минуту.",
    "Обновления накатываются планово и фиксируются в журнале изменений.",
    "Резервные копии настраиваются на уровне платформы.",
    "Инциденты регистрируются в общей очереди поддержки.",
    "Доступ к административным функциям ограничен ролью оператора.",
    "Журнал аудита хранится отдельно от основного хранилища данных.",
]
FILLERS_EN = [
    "The detailed setup guide lives in the internal team documentation.",
    "Parameter changes are agreed with the on-call shift in advance.",
    "Component metrics are exported to the shared monitoring every minute.",
    "Updates roll out on schedule and land in the change log.",
    "Backups are configured at the platform level.",
    "Incidents are filed into the shared support queue.",
    "Administrative functions are restricted to the operator role.",
    "The audit log is stored separately from the main data store.",
]

NUM_SLOTS = [
    ("lim",  "Лимит запросов к {dat} — {v} в минуту.",
             "Request rate limit for the service is {v} per minute.", [200, 500, 1000]),
    ("timo", "Таймаут операций {gen} — {v} секунд.",
             "Operation timeout is {v} seconds.", [30, 60, 120]),
    ("batch","Размер пакета для {gen} — {v} записей.",
             "Batch size is {v} records.", [100, 200, 500]),
    ("ret",  "Срок хранения логов {gen} — {v} дней.",
             "Log retention is {v} days.", [14, 30, 90]),
]
DATE_SLOT = ("upd", "Обновление {gen} намечено на {v}.",
             "The update is scheduled for {v}.",
             ["5 октября", "7 октября", "15 октября"],
             ["October 5", "October 7", "October 15"])
STATUS_SLOTS = [
    ("st1", "активен", "архивирован",
     "{cap} активен в реестре сервисов.", "{cap} архивирован в реестре сервисов."),
    ("st2", "работает в продуктиве", "работает только на стейджинге",
     "{cap} работает в продуктиве.", "{cap} работает только на стейджинге."),
]
MODAL_SLOTS = [
    ("md1", "работает без контейнера docker", "требует контейнер docker",
     "{cap} работает без контейнера Docker.", "{cap} требует контейнер Docker."),
    ("md2", "допускает анонимный доступ", "требует авторизацию для доступа",
     "{cap} допускает анонимный доступ к ресурсам.", "{cap} требует авторизацию для доступа к ресурсам."),
]
COND_SLOTS = [
    ("cd1", "multipart", "режет большие файлы на части", "резать большие файлы на части",
     "splits large files into parts"),
    ("cd2", "priority", "ставит задачи в приоритетную очередь", "ставить задачи в приоритетную очередь",
     "puts jobs into the priority queue"),
    ("cd3", "compression", "сжимает ответы на лету", "сжимать ответы на лету",
     "compresses responses on the fly"),
]
POLAR_SLOTS = [
    ("pl1", "Компонент падает при пиковой нагрузке.",
            "Компонент стабильно держит пиковую нагрузку.",
            "Компонент падает при пиковой нагрузке."),
    ("pl2", "Компонент замедлил работу системы.",
            "Компонент даёт заметный прирост скорости.",
            "Компонент замедлил работу системы."),
    ("pl3", "Компонент увеличил время ответа сервиса.",
            "Компонент сократил время ответа сервиса.",
            "Компонент увеличил время ответа сервиса."),
]
INSERT_RU = [
    "Дополнительно поддерживается выгрузка результатов в CSV.",
    "Также настроена отправка метрик в общий мониторинг.",
    "Отдельным каналом включены уведомления в командный чат.",
]
#: Extended superset-addition pool (same INSERT class: conjunction stays
#: satisfiable). Needed so the insert-count dial can reach the razor bin
#: [0.95, 0.97) — three short inserts keep NC-1 at 0.97+ (measured).
EXTRA_INSERTS = [
    "Отдельно ведётся выгрузка отчётных форм в архив за прошлые периоды.",
    "Настройки уведомлений дублируются в профиль пользователя автоматически.",
    "Поддерживается импорт справочников из файлов обмена по расписанию.",
    "Журнал операций доступен выгрузкой по запросу администратора.",
    "Дополнительно включён экспорт событий во внешнюю систему хранения.",
]
INSERT_POOL = INSERT_RU + EXTRA_INSERTS
DATE_ALT_RU = "Релиз {gen} запланирован на {v}."
DATE_ALT_EN = "The release of the component is planned for {v}."
INSERT_EN = [
    "Additionally, results can be exported to CSV.",
    "Metrics are also forwarded to the shared monitoring.",
    "Team chat notifications are enabled on a separate channel.",
]
REFINE_RU = {
    "lim":  ("Лимит запросов к {dat} — не более {c} в минуту.",
             "Лимит запросов к {dat} — {v} в минуту.", lambda v, c: v <= c),
    "upd":  ("Обновление {gen} намечено на октябрь.",
             "Обновление {gen} назначено на {v}.",
             lambda v, c: "октябрь" in v),
}
REFINE_EN = {
    "lim":  ("The request rate limit is at most {c} per minute.",
             "The request rate limit is {v} per minute.", lambda v, c: v <= c),
    "upd":  ("The update is scheduled for October.",
             "The update is scheduled for {v}.",
             lambda v, c: "October" in v),
}
VERSION_RU = "Текущая версия {gen} — {v}."
VERSION_EN = "The current version is {v}."
VERSIONS = ["v1.2.0", "v1.3.0", "v2.0.0"]

#: Subject-specific sentences (NC-5 short bodies; CD-5/CD-6 divergent
#: short bodies). These are deliberately NOT translations of each other:
#: the generic BASE_RU/BASE_EN pair is mutually translated and keeps
#: cross-language pairs above 0.7 (measured); specific content dives to
#: the instrument floor (0.486 measured).
SPEC: dict[str, tuple[str, str, str, str]] = {
    "s3conn":  ("Многосоставные загрузки собираются в манифест и проверяются контрольными суммами.",
                "Расписание выгрузок подстраивается под окна низкой нагрузки.",
                "Multipart uploads are assembled into a manifest and verified with checksums.",
                "Download windows shift toward low-traffic hours automatically."),
    "retry":   ("Очередь повторов ограничивает число попыток и помечает безнадёжные запросы.",
                "Задержки между попытками растут экспоненциально с джиттером.",
                "The retry queue caps attempts and flags hopeless requests.",
                "Backoff delays grow exponentially with jitter."),
    "sched":   ("Ночные джобы запускаются последовательно, чтобы не спорить за ресурсы.",
                "Пропущенные запуски компенсируются при следующем пробуждении.",
                "Night jobs run sequentially to avoid resource contention.",
                "Missed runs are compensated on the next wake-up."),
    "cache":   ("Профили пользователей держатся в памяти и высвобождаются по TTL.",
                "Холодные записи вытесняются по правилу LRU.",
                "Session profiles live in memory and expire by TTL.",
                "Cold entries are evicted by an LRU rule."),
    "export":  ("Крупные отчёты собираются в фоне и доставляются ссылкой.",
                "Шаблоны выгрузок версионируются вместе с формами.",
                "Large reports are assembled in the background and delivered as links.",
                "Export templates are versioned together with the forms."),
    "recs":    ("Ранжирование предложений учитывает свежесть и обратную связь.",
                "Холодный старт новых пользователей закрывается популярным.",
                "Ranking blends freshness with the feedback signal.",
                "New users start from a popularity fallback."),
    "pay":     ("Платёжные формы токенизируют данные карты до отправки.",
                "Ошибки проведения дублируются в журнал финансового контроля.",
                "Payment forms tokenize card data before submission.",
                "Settlement failures are mirrored into the finance log."),
    "logs":    ("Разбор access-логов извлекает метрики запросов на лету.",
                "Кривые форматы журналов нормализуются перед разбором.",
                "Access-log parsing extracts request metrics on the fly.",
                "Odd log formats are normalized before parsing."),
    "auth":    ("Сессии подтверждаются короткоживущими токенами обмена.",
                "Подозрительные входы проходят второй фактор.",
                "Sessions rely on short-lived exchange tokens.",
                "Suspicious sign-ins face a second factor."),
    "queue":   ("Письма и пуши доставляются через общий шлюз с ретраями.",
                "Приоритетные уведомления обгоняют массовые рассылки.",
                "Mail and push delivery share one gateway with retries.",
                "Priority notifications overtake bulk campaigns."),
    "mlpipe":  ("Переобучение запускается по расписанию на свежей разметке.",
                "Качество моделей сравнивается на замороженном тесте.",
                "Retraining runs on a schedule over fresh labels.",
                "Model quality is compared on a frozen test split."),
    "bill":    ("Тарификация собирает счётчики_usage в конце суток.",
                "Спорные списания проходят ручную модерацию.",
                "Metering aggregates usage counters at day end.",
                "Contested charges go through manual review."),
    "cook":    ("Свёклу томят до мягкости, чтобы сохранился цвет.",
                "Заправку вводят после снятия с огня.",
                "Beetroot is simmered until soft to keep the color.",
                "Dressing goes in after the pot leaves the heat."),
    "fit":     ("Силовые дни чередуются с лёгким кардио для восстановления.",
                "Прогресс фиксируется по рабочим весам и объёму.",
                "Strength days alternate with easy cardio for recovery.",
                "Progress is tracked by working weights and volume."),
    "weather": ("Атмосферное давление падает перед фронтом дождя.",
                "Ночные заморозки вероятны в низинах.",
                "Pressure drops ahead of the rain front.",
                "Night frosts are likely in the lowlands."),
    "music":   ("Инструментальные треки идут первыми, чтобы не отвлекать текстом.",
                "Громкость выравнивается между треками по нормализации.",
                "Instrumental tracks come first to avoid lyric distraction.",
                "Loudness is evened out across tracks by normalization."),
    "travel":  ("Маршрут начинается от набережной и заканчивается в парке.",
                "Пересадки на автобус занимают не больше десяти минут.",
                "The route starts at the embankment and ends in the park.",
                "Bus transfers take no more than ten minutes."),
    "books":   ("В этом месяце клуб обсуждает роман о тихой провинции.",
                "Встреча проходит в четверг в малом зале библиотеки.",
                "This month the club discusses a quiet provincial novel.",
                "The meeting is on Thursday in the library small hall."),
}

EN_POLAR = [
    ("pl1", "The component fails under peak load.",
            "The component holds peak load steadily.",
            "The component fails under peak load."),
    ("pl2", "The component slowed the system down.",
            "The component gives a noticeable speed-up.",
            "The component slowed the system down."),
    ("pl3", "The component increased service latency.",
            "The component reduced service latency.",
            "The component increased service latency."),
]
#: Deviant negative-polarity EN surfaces per polar slot (index 0 = the
#: literal translation above). Lexically divergent, polarity EXACT —
#: spread the cross-language CD-6 pairs across the lower bands instead
#: of clustering at 0.75-0.90 (surface-grid probe, 2026-11-02).
EN_NEG_SURFACES = {
    "pl1": ["The component fails under peak load.",
            "Peak traffic brings the service down.",
            "Under heavy load the service crashes.",
            "At peak times the system falls over."],
    "pl2": ["The component slowed the system down.",
            "After the rollout everything got sluggish.",
            "The change made requests crawl.",
            "Latency regressed across the board."],
    "pl3": ["The component increased service latency.",
            "Responses now take visibly longer.",
            "The service got slower for every caller.",
            "Waiting times went up after the change."],
}
EN_COND = [
    ("cd1", "multipart", "splits large files into parts", "split large files into parts"),
    ("cd2", "priority", "puts jobs into the priority queue", "put jobs into the priority queue"),
    ("cd3", "compression", "compresses responses on the fly", "compress responses on the fly"),
]

#: Paraphrase maps (deterministic substitutions; ordered, first-match).
PARA_RU = [
    ("очень", "заметно"), ("быстро", "с высокой скоростью"),
    ("использует", "задействует"), ("поддерживает", "умеет"),
    ("настроено", "сконфигурировано"), ("ведётся", "выполняется"),
    ("отправляет", "передаёт"), ("читается", "загружается"),
    ("компонент", "модуль"), ("фиксируется", "регистрируется"),
    ("доступны", "видны"), ("обслуживает", "держит"),
]
PARA_EN = [
    ("runs", "operates"), ("serves", "handles"), ("written", "appended"),
    ("visible", "available"), ("read", "loaded"), ("settings", "config"),
    ("component", "module"), ("quickly", "fast"), ("supports", "offers"),
]


def para(text: str, k: int, lang: str, rng: random.Random) -> str:
    """Apply k deterministic substitutions (intensity dial)."""
    table = PARA_RU if lang == "ru" else PARA_EN
    idx = list(range(len(table)))
    rng.shuffle(idx)
    out = text
    done = 0
    for i in idx:
        if done >= k:
            break
        a, b = table[i]
        if a in out:
            out = out.replace(a, b, 1)
            done += 1
    return out


def fmt(slot_sentence: str, subj_id: str, v) -> str:
    cap, _, gen, dat = SUBJECTS[subj_id]
    return slot_sentence.format(gen=gen, dat=dat, cap=cap, v=v)


def fmt_en(slot_sentence: str, v) -> str:
    return slot_sentence.format(v=v)


# --- scenario model --------------------------------------------------------

@dataclass
class Side:
    title: str
    body: str
    tags: list[str]
    language: str | None
    record_type: str | None
    created_at: str | None


@dataclass
class Scenario:
    scenario_id: str
    stratum: str            # CD-1..CD-6 | NC-1..NC-5
    side_a: Side
    side_b: Side
    label: str              # contradicts | not-contradicts
    slots: list[dict]       # declaration slots (see design 5.1)
    subject_a: str
    subject_b: str
    polar_op: str | None    # CD-6 declared polar operator token
    refine_ok: bool | None  # NC-2 v |= c check result
    scaffold_id: str
    cross_lang: bool
    length_channel: str     # short | medium | long
    both_null: bool

    def text_a(self) -> str:
        return embed_text(self.side_a)

    def text_b(self) -> str:
        return embed_text(self.side_b)


def embed_text(side: Side) -> str:
    """Pinned convention (ADDENDUM-6 7.4): n1(title) + \\n + n1(body) + \\n + n1(tags)."""
    return f"{n1(side.title)}\n{n1(side.body)}\n{n1(' '.join(side.tags))}"


# --- region machinery (B4 2.1 constants) -----------------------------------

def regions(body_a: str, body_b: str) -> list[tuple[str, str, int, int]]:
    """GAP-merged non-equal regions of difflib on norm1 bodies at TOKEN
    level (char-level difflib fragments single-token edits around common
    letters — lesson of the first preflight; B4 merged adjacent ops with
    GAP=12 chars for the same reason).

    Returns list of (tag, region_text, a_start_tok, a_end_tok); region
    content is the b-side text; tag is 'insert' only when every
    constituent opcode is b-side insertion.
    """
    aw = norm1(body_a).split(" ")
    bw = norm1(body_b).split(" ")
    sm = difflib.SequenceMatcher(None, aw, bw, autojunk=False)
    raw = [op for op in sm.get_opcodes() if op[0] != "equal"]
    merged: list[list] = []  # [tag, i1, i2, j1, j2, pure_insert]
    for tag, i1, i2, j1, j2 in raw:
        if merged and i1 - merged[-1][2] <= 3:  # GAP in tokens (~B4 12 chars)
            m = merged[-1]
            m[2], m[4] = i2, j2
            m[5] = m[5] and (tag == "insert")
            continue
        merged.append([tag, i1, i2, j1, j2, tag == "insert"])
    out = []
    for tag, i1, i2, j1, j2, pure in merged:
        out.append(("insert" if pure else "replace",
                    " ".join(bw[j1:j2]), i1, i2))
    return out


VERSION_RE = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)?\b|\b(?:версия|version|релиз|release|build|сборка)\b",
    re.IGNORECASE)


def region_is_version(text: str) -> bool:
    """Region consists of version-namespace tokens only (numbers with
    version separators or version keywords); per-token edge punctuation
    is stripped, inner dots preserved (semver)."""
    stripped = text.strip()
    if not stripped:
        return False
    for tok in stripped.split(" "):
        tok = tok.strip(".,:;—–-()\"'")
        if not tok:
            continue
        if VERSION_RE.fullmatch(tok):
            continue
        if tok.lower() in {
            "версия", "текущая", "version", "current", "релиз", "release",
            "сборка", "build", "обновление", "update",
        }:
            continue
        return False
    return True


def version_guard(sc: Scenario) -> bool:
    """Deterministic code leg: ALL discriminative regions version-namespace."""
    regs = regions(sc.side_a.body, sc.side_b.body)
    if not regs:
        return False
    return all(region_is_version(t) for _, t, _, _ in regs)


# --- mechanical coverage map (ADDENDUM-1-design 3.2.1 / MLC-9 item 4) ------

def _sym_contains(value_b: str, region: str) -> bool:
    """Fail-closed content match, either direction: the declared value
    contains the region fragment, or the region contains the declared
    value (a one-token razor edit yields region == fragment; a
    sentence-level edit yields region == the full declared sentence)."""
    nb, nr = norm1(value_b), norm1(region)
    if not nb or not nr:
        return False
    return nb in nr or nr in nb


def slot_bijection_ok(sc: Scenario) -> bool:
    """Slot bijection: region count == declared slots; every declared
    value_b matches its region content (symmetric containment)."""
    regs = regions(sc.side_a.body, sc.side_b.body)
    k = len(sc.slots)
    if len(regs) != k:
        return False
    for slot, (_, text, _i1, _i2) in zip(sc.slots, regs):
        if not _sym_contains(str(slot["value_b"]), text):
            return False
    return True


def mechanical_ok(sc: Scenario) -> bool:
    """Per-class mechanical predicate, fail-closed (design 3.2.1)."""
    s = sc.stratum
    if s in {"CD-1", "CD-2", "CD-3", "CD-4", "CD-5"}:
        return slot_bijection_ok(sc)
    if s == "CD-6":
        regs = regions(sc.side_a.body, sc.side_b.body)
        if len(regs) != 1:
            return False
        return bool(sc.polar_op) and _sym_contains(sc.polar_op, regs[0][1])
    if s == "NC-1":
        regs = regions(sc.side_a.body, sc.side_b.body)
        return bool(regs) and all(tag == "insert" for tag, _, _, _ in regs)
    if s == "NC-2":
        return bool(sc.refine_ok)
    if s == "NC-3":
        regs = regions(sc.side_a.body, sc.side_b.body)
        return bool(regs) and all(region_is_version(t) for _, t, _, _ in regs)
    if s == "NC-4":
        a, b = sc.side_a, sc.side_b
        return (norm1(a.body) == norm1(b.body) and norm1(a.title) == norm1(b.title)
                and (a.tags != b.tags or a.record_type != b.record_type
                     or a.language != b.language))
    if s == "NC-5":
        return sc.subject_a != sc.subject_b
    return False


# --- generators -------------------------------------------------------------

def make_side(lang: str, subj: str, body: str, *, para_k: int = 0,
              tags: list[str] | None = None, rtype: str | None = "note",
              created: str | None = None) -> Side:
    anchor = SUBJECTS[subj][0 if lang == "ru" else 1]
    body_final = body
    if para_k:
        # deterministic seed: crc32 is stable across runs (hash() is salted)
        seed = zlib.crc32(f"{subj}:{body}".encode("utf-8"))
        rng = random.Random(seed)
        body_final = para(body, para_k, lang, rng)
    return Side(title=anchor, body=body_final,
                tags=tags if tags is not None else [f"dom:{subj}"],
                language=lang if lang else None, record_type=rtype,
                created_at=created)


def base_body(subj: str, lang: str, fillers: int) -> str:
    base = BASE_RU if lang == "ru" else BASE_EN
    fill = FILLERS_RU if lang == "ru" else FILLERS_EN
    parts = list(base[:2])
    parts += [fill[i % len(fill)] for i in range(fillers)]
    return " ".join(parts)


def channel_for(body: str) -> str:
    n = len(body)
    return "short" if n <= 200 else ("medium" if n <= 2000 else "long")


_BODY_COUNTER = [0]


def new_sid(prefix: str, rng: random.Random) -> str:
    _BODY_COUNTER[0] += 1
    return f"{prefix}-{rng.randrange(10**6):06d}-{_BODY_COUNTER[0]:05d}"


def build_scenario(sid: str, stratum: str, subj: str, lang: str, slot_def,
                   v1, v2, *, fillers: int = 1, para_b: int = 0,
                   cross: bool = False, insert_sent: str | None = None,
                   polar: tuple | None = None, polar_en: tuple | None = None,
                   refine: tuple | None = None,
                   version_pair: tuple | None = None,
                   meta_delta: str | None = None, sibling_subj: str | None = None,
                   rng: random.Random | None = None) -> Scenario:
    rng = rng or random.Random(0)
    if cross and stratum == "CD-6":
        # translation-negation conflict: RU affirmative vs EN negation,
        # STANDALONE short sentences (spec prefix keeps pairs at 0.85+;
        # short standalone pairs measure 0.68-0.75 and feed the
        # [0.60,0.70) hard window)
        pe = polar_en or EN_POLAR[0]
        body_a = polar[2]
        body_b = pe[3]
        side_a = make_side("ru", subj, body_a)
        side_b = make_side("en", subj, body_b)
        return Scenario(
            scenario_id=sid, stratum=stratum, side_a=side_a, side_b=side_b,
            label="contradicts",
            slots=[{"slot_id": pe[0], "axis": "semantic", "value_a": polar[2][:60],
                    "value_b": pe[3], "op": "replace", "is_version_token": False,
                    "span_chars": len(pe[3])}],
            subject_a=subj, subject_b=subj, polar_op=pe[3], refine_ok=None,
            scaffold_id=f"CD-6x:{subj}:{polar[0]}", cross_lang=True,
            length_channel=channel_for(body_a), both_null=False)
    if cross:
        subj_fmt = {"gen": SUBJECTS[subj][2], "dat": SUBJECTS[subj][3],
                    "cap": SUBJECTS[subj][0]}
        body_a = " ".join([BASE_RU[0]]
                          + ([slot_def[1].format(v=v1, **subj_fmt)] if slot_def else []))
        body_b = " ".join([BASE_EN[0]]
                          + ([slot_def[2].format(v=v2)] if slot_def else []))
        if para_b:
            body_b = para(body_b, para_b, "en", rng)
        side_a = make_side("ru", subj, body_a)
        side_b = make_side("en", subj, body_b)
        subjects = (subj, subj)
        if stratum == "NC-5":
            side_b = make_side("en", sibling_subj, base_body(sibling_subj, "en", 1))
            subjects = (subj, sibling_subj)
    elif stratum == "NC-5":
        # spec-based short bodies: generic BASE pairs are mutual
        # translations and keep cross-language pairs above 0.7 (measured);
        # specific content reaches the instrument floor
        spec_a = SPEC[subj]
        side_a = make_side(lang, subj, spec_a[0] if cross else spec_a[0] + " " + spec_a[1])
        if cross:
            side_b = make_side("en", sibling_subj, SPEC[sibling_subj][2])
            lang_b = "en"
        else:
            lang_b = "ru"
            side_b = make_side("ru", sibling_subj,
                               SPEC[sibling_subj][0] + " " + SPEC[sibling_subj][1])
        subjects = (subj, sibling_subj)
        return Scenario(
            scenario_id=sid, stratum=stratum, side_a=side_a, side_b=side_b,
            label="not-contradicts", slots=[], subject_a=subjects[0],
            subject_b=subjects[1], polar_op=None, refine_ok=None,
            scaffold_id=f"nc5:{subj}:{sibling_subj}",
            cross_lang=(lang != lang_b),
            length_channel=channel_for(side_a.body + side_b.body),
            both_null=False)
    elif stratum in ("CD-5", "CD-6"):
        # short divergent bodies: SPEC sentence + the polar/conditional
        # sentence (same-subject cos floor is ~0.68; only short bodies
        # reach the hard zone)
        spec_line = SPEC[subj][0]
        if stratum == "CD-6":
            body_a = spec_line + " " + polar[2]
            body_b = spec_line + " " + polar[3]
        else:
            entry = next(c for c in COND_SLOTS if c[2] == v1)
            flag = entry[1]
            body_a = spec_line + " " + f"При включённом флаге {flag} модуль {v1}."
            body_b = spec_line + " " + f"Модуль никогда не {v2}, независимо от флага {flag}."
        side_a = make_side(lang, subj, body_a)
        side_b = make_side(lang, subj, body_b)
        subjects = (subj, subj)
    else:
        body = base_body(subj, lang, fillers)
        if stratum == "NC-4":
            body_a, body_b = body, body
        elif stratum == "NC-1":
            body_a = body
            body_b = body + " " + insert_sent
        elif stratum == "NC-2":
            c_sent, v_sent, check = REFINE_RU[refine[0]]
            subj_fmt = {"gen": SUBJECTS[subj][2], "dat": SUBJECTS[subj][3]}
            body_a = body + " " + c_sent.format(c=refine[1], **subj_fmt)
            body_b = body + " " + v_sent.format(v=v1, **subj_fmt)
        elif stratum == "NC-3":
            body_a = body + " " + VERSION_RU.format(gen=SUBJECTS[subj][2], v=version_pair[0])
            body_b = body + " " + VERSION_RU.format(gen=SUBJECTS[subj][2], v=version_pair[1])
        else:
            sa_txt = fmt(slot_def[1], subj, v1)
            sb_txt = fmt(slot_def[1], subj, v2)
            if stratum in ("CD-3", "CD-4"):
                sa_txt = slot_def[3].format(cap=SUBJECTS[subj][0])
                sb_txt = slot_def[4].format(cap=SUBJECTS[subj][0])
            body_a = body + " " + sa_txt
            body_b = body + (" " + para(sb_txt, para_b, lang, rng) if para_b else " " + sb_txt)
        side_a = make_side(lang, subj, body_a)
        side_b = make_side(lang, subj, body_b)
        subjects = (subj, subj)

    # declaration slots
    slots: list[dict] = []
    polar_op = None
    refine_ok = None
    if stratum in {"CD-1", "CD-2"}:
        slots = [{"slot_id": slot_def[0], "axis": "numeric" if stratum == "CD-1" else "date",
                  "value_a": str(v1), "value_b": str(v2), "op": "replace",
                  "is_version_token": False, "span_chars": max(len(str(v1)), len(str(v2)))}]
    elif stratum in {"CD-3", "CD-4"}:
        slots = [{"slot_id": slot_def[0], "axis": "status" if stratum == "CD-3" else "modal",
                  "value_a": slot_def[1], "value_b": slot_def[2], "op": "replace",
                  "is_version_token": False, "span_chars": len(slot_def[2])}]
    elif stratum == "CD-5":
        slots = [{"slot_id": "cond", "axis": "conditional", "value_a": v1,
                  "value_b": "никогда не " + v2, "op": "replace",
                  "is_version_token": False, "span_chars": len(v1) + 12}]
    elif stratum == "CD-6":
        polar_op = polar[1]
        slots = [{"slot_id": polar[0], "axis": "semantic", "value_a": polar[2][:60],
                  "value_b": polar[1], "op": "replace", "is_version_token": False,
                  "span_chars": len(polar[1])}]
    elif stratum == "NC-1":
        slots = [{"slot_id": "ins", "axis": "addition", "value_a": "",
                  "value_b": insert_sent, "op": "insert", "is_version_token": False,
                  "span_chars": len(insert_sent)}]
    elif stratum == "NC-2":
        refine_ok = bool(REFINE_RU[refine[0]][2](v1, refine[1]))
        slots = [{"slot_id": refine[0], "axis": refine[0], "value_a": str(refine[1]),
                  "value_b": str(v1), "op": "replace", "is_version_token": False,
                  "span_chars": max(len(str(v1)), 6)}]
    elif stratum == "NC-3":
        slots = [{"slot_id": "ver", "axis": "version", "value_a": version_pair[0],
                  "value_b": version_pair[1], "op": "replace", "is_version_token": True,
                  "span_chars": max(len(version_pair[0]), len(version_pair[1]))}]

    if meta_delta == "type":
        side_b.record_type = "fact"
    elif meta_delta == "tags":
        side_b.tags = sorted(set(side_b.tags + ["delta:meta"]))
    elif meta_delta == "lang":
        side_b.language = "en"

    sc = Scenario(
        scenario_id=sid, stratum=stratum, side_a=side_a, side_b=side_b,
        label=("contradicts" if stratum.startswith("CD") else "not-contradicts"),
        slots=slots, subject_a=subjects[0], subject_b=subjects[1],
        polar_op=polar_op, refine_ok=refine_ok,
        scaffold_id=f"{stratum}:{subj}:{slot_def[0] if slot_def else (polar[0] if polar else refine[0] if refine else (version_pair[0] if version_pair else meta_delta or 'sib'))}",
        cross_lang=cross, length_channel=channel_for(side_a.body),
        both_null=False)
    return sc


def finish_pair(sc: Scenario, rng: random.Random, null_p: float,
                force_not_null: bool = False) -> Scenario:
    """Assign envelope nullness (both-null only), created_at, length channel.

    force_not_null: scoped exemption for corner twins — their label
    semantics require the metadata delta to stay visible; nulling it
    would degenerate the pair to a self-pair (documented in the
    manifest, A-null-* applied to the remaining rows of the stratum).
    """
    both_null = (not force_not_null) and rng.random() < null_p
    for side in (sc.side_a, sc.side_b):
        if both_null:
            side.created_at = None
            if side.language is not None:
                side.language = None
            side.record_type = None
    if not both_null:
        for side in (sc.side_a, sc.side_b):
            if side.created_at is None:
                day = rng.randrange(1, 90)
                hour = rng.randrange(0, 24)
                side.created_at = f"2026-08-{day:02d}T{hour:02d}:15:00Z"
    sc.length_channel = channel_for(sc.side_a.body)
    sc.both_null = both_null
    return sc
