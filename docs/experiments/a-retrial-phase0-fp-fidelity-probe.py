#!/usr/bin/env python3
"""A-retrial Phase 0 / N4 probe recipe (executed 2026-10-28, light lane).

Reproduction record of what ran (numbers in docs/experiments/a-retrial-runlog.md).
Two parts, both executed in this exact form:

  Part 1 — fidelity probe: fp32-torch(nm1b-r3/epoch3) vs prod ONNX
           (vesma-embed-v1, pin nano:sha256:3b752e06...) on 50 neutral texts.
  Part 2 — attribution decomposition: fp32-ONNX re-export (legacy exporter,
           opset 15, static 1x256) -> dynamic int8 (r5 recipe class) and
           static QDQ (round-3 recipe: per_channel=False, QInt8/QInt8,
           calibration = first N rows of vesma training/data/val.jsonl).

Run env (per prereg §8 / owner env isolation):
  /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex/.venv
  (onnxruntime 1.30.0, torch 2.14.0+cpu, transformers 4.57.6, tokenizers 0.22.2,
   safetensors 0.8.0, numpy 2.5.3), PYTHONDONTWRITEBYTECODE=1, threads pinned.
  Foreign venvs untouched; engine repo read-only; /tmp for outputs.
  torch.onnx.export uses dynamo=False (torch 2.14 legacy exporter — the
  new exporter needs onnxscript, absent from the scoring venv; legacy path
  reproduces the same graph math, validated cos 1.000000 vs the torch leg).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

PROD_DIR = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/src/vesma/models/vesma-embed-v1"
FP_DIR = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/training/runs/nm1b-r3/epoch3"
VAL_JSONL = "/var/home/abyss/LABs/Projects/Project-Vesma/vesma/training/data/val.jsonl"
EXPORT_TMP = "/tmp/r3-fp32-export"

# 50 neutral synthetic probe texts (RU+EN, mixed lengths; authored for this
# probe — zero contact with any corpus / gate / sealed surface).
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "A monthly team sync notes: shipped the search relevance fix, backlog review moved to Thursday.",
    "PostgreSQL vacuum tuning reduced table bloat from 41% to 6% after the autovacuum threshold change.",
    "Recipe: soak the lentils overnight, then simmer with cumin, garlic and a bay leaf for 25 minutes.",
    "The hiking trail climbs steadily for three kilometres before opening into a wide alpine meadow.",
    "Meeting summary: budget approved for Q4, hiring paused, infrastructure migration starts next week.",
    "Remember to renew the domain registration before the end of the month; auto-renew failed last year.",
    "A modular synthesizer patch routes the oscillator through a low-pass filter and a slow envelope.",
    "The ferry schedule changes in winter: two morning crossings instead of four, last boat at 19:40.",
    "Code review notes: extract the retry loop into a helper, add unit tests for the timeout branch.",
    "Electric bicycle range drops noticeably below five degrees Celsius, plan for shorter trips.",
    "The museum guide explained that the mosaic was restored twice, in 1912 and again after the war.",
    "Standup: API latency p99 improved after the connection pool resize; incident postmortem drafted.",
    "Gardening log: transplanted the tomato seedlings into larger pots, added a slow-release fertiliser.",
    "The contract clause requires written notice thirty days before termination by either party.",
    "Chess opening notes: the Catalan pressurises the long diagonal after an early d4 and c4 setup.",
    "Train delay announcement: platform four service runs approximately twelve minutes behind schedule.",
    "The podcast episode interviews a cartographer about mapping coastlines that keep changing shape.",
    "Backup policy: differential snapshots nightly, full snapshot every Sunday, retention ninety days.",
    "A quiet café by the canal serves cardamom coffee and orange cake on handwritten paper menus.",
    "Заметка: созвон с командой перенесён на пятницу, повестка — релизный поезд и ретроспектива.",
    "Рецепт борща: свёклу натереть, обжарить с томатной пастой, добавить в бульон за десять минут до готовности.",
    "Конспект книги: привычка формируется в среднем за шестьдесят шесть дней, а не за двадцать один.",
    "Список покупок: гречка, молоко, лимоны, кофе в зёрнах, корм для кота и две свечки.",
    "План на выходные: суббота — поездка в музей, воскресенье — генеральная уборка и стирка.",
    "Встреча отменена: заказчик заболел, перенос на следующую среду в то же время.",
    "Заметка о ремонте: мастер придёт в четверг, заменить смеситель и проверить тёкший сифон.",
    "Итоги месяца: закрыли двенадцать задач, две ушли в следующий спринт, дебт-бэк уменьшился.",
    "Записка о походе: на перевале ветер и туман, ночёвка ниже, у реки, дрова сырые.",
    "Заметка по коду: рефакторинг обработчика очереди отложен, сначала тесты на таймауты.",
    "Памятка: оплатить коммунальные услуги до десятого числа, иначе начислят пеню.",
    "Конспект лекции: миграции птиц зависят от магнитного поля и звёздной навигации.",
    "Отчёт по огороду: урожай картофеля ниже прошлого года, помидоры наоборот удались.",
    "Заметка о фильме: сценарий слабый, но операторская работа и саундтрек стоят просмотра.",
    "Список дел на неделю: сдать отчёт, купить подарок папе, записаться к стоматологу.",
    "Заметка о тренировке: интервалы на дорожке, четыре раза по восемьсот метров, темп держать.",
    "Конспект статьи: сон влияет на консолидацию памяти сильнее, чем считали раньше.",
    "Напоминание: забрать посылку на почте до воскресенья, иначе вернут отправителю.",
    "Заметка о городе: новая набережная открылась, ночью подсветка выключается в полночь.",
    "План уроков: повторить времена в английском, разобрать двадцать новых слов по теме работы.",
    "Заметка о деплое: релиз v1.2.3 выкатился без инцидентов, откат не потребовался, метрики стабильны.",
    "Deploy checklist: run migrations, warm the cache, flip the feature flag, watch error rate for ten minutes.",
    "The embedding model distillation used a 0.6B teacher and a three-layer student with hidden size 312.",
    "Записка: индекс поиска пересобирается за четыре минуты, векторизация батчем по двести записей.",
    "Technical design note: the quantizer uses static post-training quantization with a 200-sample calibration set.",
    "Инструкция: перед обновлением сделать снапшот базы, проверить свободное место и остановить воркеры.",
    "Long note for truncation: the release train leaves every second Tuesday; the conductor collects tickets after the bridge, when the fields turn gold and the river widens past the old mill, where herons stand in the shallows at dawn and the fog lifts slowly off the water, revealing the far bank with its line of poplars, and the first cyclist of the day crosses the wooden footbridge, ringing a bell that echoes between the trees. The café by the station opens at seven, serving coffee to commuters who read the news on paper, as they did decades ago, when this line carried freight to the harbour and the schedules were chalked on a slate board by the door.",
    "Длинная заметка для обрезки: поезд отправляется в семь сорок утра, вагон четвёртый, место у окна; за вокзалом дорога идёт вдоль канала, где утром рыбаки раскладывают удочки, а к девяти часам приходит первый трамвай и забирает пассажиров у проходной завода; в киоске продают свежие булочки и газеты, кассирша знает всех по именам, и только после моста, когда город остаётся позади, начинается настоящая тишина, нарушаемая лишь стуком колёс и редкими объявлениями диспетчера о промежуточных остановках.",
    "Dup-style near pair A: the nightly backup finished at 03:12 with two warnings about stale locks.",
    "Dup-style near pair B: the nightly backup completed at 03:12 with two warnings about stale locks.",
]


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def embed_fp32(texts: list[str]) -> np.ndarray:
    """fp32 torch leg: epoch3 safetensors + projector, export-path recipe
    (mean-pool(mask) -> Linear(312->384, no bias) -> L2), prod tokenizer,
    identical [1,256] inputs."""
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel

    torch.set_num_threads(4)
    tok = Tokenizer.from_file(f"{PROD_DIR}/tokenizer.json")
    model = AutoModel.from_pretrained(FP_DIR)
    model.eval()
    state = torch.load(f"{FP_DIR}/projector.pt", map_location="cpu", weights_only=True)
    proj = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0], bias=False)
    proj.load_state_dict(state)
    proj.eval()

    out = []
    with torch.no_grad():
        for text in texts:
            enc = tok.encode(text)
            ids = np.asarray([enc.ids], dtype=np.int64)
            mask = np.asarray([enc.attention_mask], dtype=np.int64)
            assert ids.shape == (1, 256), f"unexpected shape {ids.shape}"
            t_ids = torch.from_numpy(ids)
            t_mask = torch.from_numpy(mask)
            res = model(input_ids=t_ids, attention_mask=t_mask, token_type_ids=torch.zeros_like(t_ids))
            m = t_mask.unsqueeze(-1).to(res.last_hidden_state.dtype)
            pooled = (res.last_hidden_state * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)
            v = proj(pooled)
            v = torch.nn.functional.normalize(v, p=2, dim=-1)
            out.append(v[0].numpy())
    return np.stack(out)


def ort_embed(path: str, texts: list[str]) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{PROD_DIR}/tokenizer.json")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for text in texts:
        enc = tok.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.asarray([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out.append(sess.run(None, feed)[0][0])
    return np.stack(out)


def part1_fidelity() -> None:
    text_set_sha = hashlib.sha256("\n".join(TEXTS).encode("utf-8")).hexdigest()
    print(f"texts: n={len(TEXTS)} set_sha256={text_set_sha}")
    print(f"fp base sha256(model.safetensors)={file_sha(FP_DIR + '/model.safetensors')}")
    print(f"prod onnx sha256(model.onnx)={file_sha(PROD_DIR + '/model.onnx')}")
    v_fp = embed_fp32(TEXTS)
    v_prod = ort_embed(PROD_DIR + "/model.onnx", TEXTS)
    cos = np.sum(v_fp * v_prod, axis=1)
    report = {
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "cos_min": float(cos.min()),
        "cos_p05": float(np.percentile(cos, 5)),
        "cos_std": float(cos.std()),
        "drift_mean": float(1.0 - cos.mean()),
        "drift_max": float(1.0 - cos.min()),
    }
    print(json.dumps(report, indent=2))


def part2_decompose() -> None:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from tokenizers import Tokenizer

    fp32_path = f"{EXPORT_TMP}/model.onnx"
    dyna_path = f"{EXPORT_TMP}/model.dyn8.onnx"
    stat_path = f"{EXPORT_TMP}/model.stat8.onnx"

    quantize_dynamic(fp32_path, dyna_path, weight_type=QuantType.QInt8)

    val_texts: list[str] = []
    with open(VAL_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                val_texts.append(t.strip())
    print(f"calib texts (val.jsonl, all rows): {len(val_texts)}")

    tok = Tokenizer.from_file(f"{PROD_DIR}/tokenizer.json")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._inputs = []
            for text in val_texts:
                enc = tok.encode(text)
                ids = np.asarray([enc.ids], dtype=np.int64)
                mask = np.asarray([enc.attention_mask], dtype=np.int64)
                self._inputs.append(
                    {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
                )
            self._i = 0

        def get_next(self):
            if self._i >= len(self._inputs):
                return None
            r = self._inputs[self._i]
            self._i += 1
            return r

    quantize_static(
        fp32_path, stat_path, calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ, per_channel=False,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    )

    v_fp = embed_fp32(TEXTS)
    v_A = ort_embed(fp32_path, TEXTS)
    v_B = ort_embed(PROD_DIR + "/model.onnx", TEXTS)
    v_C = ort_embed(dyna_path, TEXTS)
    v_D = ort_embed(stat_path, TEXTS)

    def cos(p, q):
        c = np.sum(p * q, axis=1)
        return f"mean={c.mean():.6f} median={np.median(c):.6f} min={c.min():.6f}"

    print("torch    vs A(fp32-onnx mine):", cos(v_fp, v_A))
    print("A(fp32)  vs B(prod int8)     :", cos(v_A, v_B))
    print("C(dyn8)  vs A(fp32)          :", cos(v_C, v_A))
    print("D(stat8) vs A(fp32)          :", cos(v_D, v_A))
    print("D(stat8) vs B(prod int8)     :", cos(v_D, v_B))
    print("C(dyn8)  vs B(prod int8)     :", cos(v_C, v_B))


if __name__ == "__main__":
    t0 = time.time()
    what = sys.argv[1] if len(sys.argv) > 1 else "part1"
    if what == "part1":
        part1_fidelity()
    elif what == "part2":
        # requires /tmp/r3-fp32-export/model.onnx built first via the
        # legacy export: torch.onnx.export(..., opset_version=15,
        # static 1x256, mean-pool+projector+L2 wrapper, dynamo=False)
        part2_decompose()
    else:
        raise SystemExit("usage: probe.py [part1|part2]")
    print(f"wall {time.time() - t0:.1f}s")
