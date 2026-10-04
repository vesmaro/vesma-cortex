"""Frozen eval-set probe topics — the eval surface's own seed list.

Eval probes must NOT train-topic themselves into the training surface:
these 16 topics (8 RU + the same 8 as EN translation halves, one shared
latin ``key`` each) are deliberately DISJOINT from
``cortex.synth.generate.TOPICS`` (the stage-1 train seed list) — the
disjointness is a TEST (``tests/test_evalsets.py``), not luck. The two
sanity anchors (:data:`cortex.eval.sanity.PROBE_RECORD` /
``UNRELATED_RECORD`` — pinned in ``gate_contract.json`` ``probes.source``)
join them as probe anchors.

Zero raw store lines: every body below is synthetic, authored here in
this file (ADR 0003 R2 — synthetic template data may be public).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from cortex.features.pair import PairRecord

__all__ = [
    "EVAL_TOPICS",
    "PROBE_ANCHOR_KEY",
    "UNRELATED_ANCHOR_KEY",
    "ProbeTopic",
    "anchor_keys",
    "anchor_records",
]


@dataclass(frozen=True)
class ProbeTopic:
    """One seed topic of the eval surface (synth ``SynthTopic`` shape).

    RU and EN halves sharing a ``key`` are the SAME memory in two
    languages — cross-topic pair scans refuse same-key partners (the
    translation-twin guard, same discipline as the synth generator: a
    translation twin labeled not-duplicate would be a false label).
    """

    title: str
    body: str
    tags: tuple[str, ...]
    language: str
    record_type: str
    key: str

    def as_record(self) -> PairRecord:
        return PairRecord(
            title=self.title,
            body=self.body,
            tags=self.tags,
            language=self.language,
            record_type=self.record_type,
        )


def _topic(
    title: str,
    body: str,
    tags: tuple[str, ...],
    language: str,
    record_type: str,
    key: str,
) -> ProbeTopic:
    if len(body) < 30:
        raise ValueError(f"topic body too short: {title!r}")
    if language not in ("ru", "en"):
        raise ValueError(f"topic language must be ru|en: {title!r}")
    if (
        not key
        or not key.isascii()
        or not key.replace("-", "").isalnum()
        or key != key.lower()
    ):
        raise ValueError(f"topic key must be a latin machine string: {key!r}")
    return ProbeTopic(title, body, tags, language, record_type, key)


# ── sanity anchors (gate_contract.json probes.source — reused, not redefined) ─

PROBE_ANCHOR_KEY: Final[str] = "probe-cache-invalidation"
UNRELATED_ANCHOR_KEY: Final[str] = "unrelated-borscht"

# ── the frozen eval topic list: 8 RU then the same 8 as EN twins ──────────────

EVAL_TOPICS: Final[tuple[ProbeTopic, ...]] = (
    # ── RU half ──
    _topic(
        "Настройка бэкапа рабочей папки",
        "Бэкап рабочей папки делается rsync на внешний диск каждое воскресенье. "
        "Хранятся четыре недельные копии, более старые удаляет cron-скрипт.",
        ("backup", "ops"),
        "ru",
        "fact",
        "backup-rotation",
    ),
    _topic(
        "Хлеб на закваске: базовый рецепт",
        "Тесто: 500 г муки, 350 г воды, 100 г закваски, 10 г соли. Холодная "
        "ферментация двенадцать часов, выпечка в чугунке 20 минут под крышкой.",
        ("baking", "recipe"),
        "ru",
        "note",
        "sourdough-bread",
    ),
    _topic(
        "Веломаршрут вдоль реки",
        "Маршрут: набережная, дамба, лесная дорожка до старого моста, возврат "
        "через поле. Сорок два километра, три часа с двумя остановками.",
        ("sport", "route"),
        "ru",
        "note",
        "cycling-route",
    ),
    _topic(
        "Уход за аквариумом",
        "Подмена трети воды раз в неделю, фильтр промывается раз в две недели "
        "без проточной воды. Свет восемь часов, кормление дважды в день.",
        ("hobby", "aquarium"),
        "ru",
        "fact",
        "aquarium-care",
    ),
    _topic(
        "Шахматы: тренировка эндшпиля",
        "Разбираю ладейные окончания по двадцать минут в день: правило "
        "квадрата, техника моста. Тактика отложена до рейтинга выше двух тысяч.",
        ("chess", "practice"),
        "ru",
        "note",
        "chess-endgame",
    ),
    _topic(
        "План изучения испанского",
        "Двадцать новых слов в день интервальными повторениями, разговорный "
        "клуб по средам. Цель — уровень B1 к июню, сериалы без субтитров.",
        ("learning", "spanish"),
        "ru",
        "note",
        "spanish-plan",
    ),
    _topic(
        "Структура фотоархива",
        "Папки год/год-месяц-день, сырые файлы не трогаются, отборы помечаются "
        "звёздами. Раз в квартал архив в два места: диск и облако.",
        ("photos", "archive"),
        "ru",
        "fact",
        "photo-archive",
    ),
    _topic(
        "Полив комнатных растений",
        "Суккуленты — раз в две недели, папоротник — раз в три дня, остальное "
        "по сухому верхнему слою. Вода отстоянная, лишнее из поддона сливается.",
        ("plants", "care"),
        "ru",
        "fact",
        "houseplant-watering",
    ),
    # ── EN half (translations of the RU half, same order, same keys) ──
    _topic(
        "Workspace backup rotation",
        "The workspace folder is rsynced to an external drive every Sunday. "
        "Four weekly copies are kept; a cron script prunes older ones.",
        ("backup", "ops"),
        "en",
        "fact",
        "backup-rotation",
    ),
    _topic(
        "Sourdough bread: basic recipe",
        "Dough: 500 g flour, 350 g water, 100 g starter, 10 g salt. Twelve "
        "hours of cold fermentation; bake in a dutch oven, 20 minutes lidded.",
        ("baking", "recipe"),
        "en",
        "note",
        "sourdough-bread",
    ),
    _topic(
        "River cycling route",
        "The route: embankment, dam, forest trail to the old bridge, return "
        "across the field. Forty-two kilometres, three hours, two stops.",
        ("sport", "route"),
        "en",
        "note",
        "cycling-route",
    ),
    _topic(
        "Aquarium care routine",
        "A third of the water changes weekly; the filter is rinsed every two "
        "weeks without tap water. Light runs eight hours; feeding twice a day.",
        ("hobby", "aquarium"),
        "en",
        "fact",
        "aquarium-care",
    ),
    _topic(
        "Chess: endgame training",
        "Twenty minutes a day on rook endings: the square rule, the bridge "
        "technique. Tactics postponed until the rating holds above two thousand.",
        ("chess", "practice"),
        "en",
        "note",
        "chess-endgame",
    ),
    _topic(
        "Spanish learning plan",
        "Twenty new words a day through spaced repetition, conversation club "
        "on Wednesdays. Goal: B1 by June, series without subtitles from March.",
        ("learning", "spanish"),
        "en",
        "note",
        "spanish-plan",
    ),
    _topic(
        "Photo archive layout",
        "Folders year/year-month-day; raw files stay untouched, picks are "
        "starred. Once a quarter the archive lands in two places: disk, cloud.",
        ("photos", "archive"),
        "en",
        "fact",
        "photo-archive",
    ),
    _topic(
        "Houseplant watering schedule",
        "Succulents every two weeks, the fern every three days, the rest by "
        "the dry top-soil rule. Water sits overnight; tray excess is poured off.",
        ("plants", "care"),
        "en",
        "fact",
        "houseplant-watering",
    ),
)


def anchor_records() -> tuple[PairRecord, ...]:
    """The full probe anchor pool: the two sanity anchors + the 16 topics."""
    from cortex.eval.sanity import PROBE_RECORD, UNRELATED_RECORD

    return (
        PROBE_RECORD,
        UNRELATED_RECORD,
        *(topic.as_record() for topic in EVAL_TOPICS),
    )


def anchor_keys() -> tuple[str, ...]:
    """Topic keys parallel to :func:`anchor_records` (translation-twin guard)."""
    return (
        PROBE_ANCHOR_KEY,
        UNRELATED_ANCHOR_KEY,
        *(topic.key for topic in EVAL_TOPICS),
    )
