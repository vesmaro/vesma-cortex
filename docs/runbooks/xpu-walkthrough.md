# XPU walkthrough — опциональный провенанс-прогон (A3c)

> **Статус: ОПЦИОНАЛЬНО.** По ADR 0001 V6 XPU не является технической
> необходимостью: лестница D/N обучается на CPU за минуты (смоук с полной
> N-лестницей — ~86 с). Этот ранбук — добровольный walkthrough владельца в
> знакомом nano-llm окружении: он подтверждает, что весь контур живёт в
> XPU-среде (torch 2.13.0+xpu), и оставляет архив провенанса. **Оценка A5 от
> него не зависит ни в каком виде.**
>
> Честная оговорка: обучение N-кандидата в текущем коде исполняется на CPU
> (`n_head` не переключает `torch.device` — CPU-путь основной по ADR V6).
> Walkthrough проверяет СОВМЕСТИМОСТЬ окружения и детерминизм на тех же
> сидах, а не ускорение. Если N когда-нибудь переедет на XPU-тензоры —
> детерминизм-гейт CPU/XPU (совпадение метрик в допуске 1e-6) станет
> содержательным; sha256 ONNX между устройствами может не совпасть — это
> задокументированное исключение (inference-v1 §4).

## 0. Прелюдия

Контейнер nano-llm уже существует (root, ubuntu 24.04, Intel Arc 140T,
venv `~/.venvs/nm-xpu` с torch 2.13.0+xpu). Если контейнера нет — он
создаётся только с хоста: см. `~/LABs/Utils/nano-llm-container.md`; этот
ранбук предполагает, что контейнер поднят.

## 1. Копия репо в контейнер

```bash
# с хоста (или из бокса), одноразово на прогон:
podman cp /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex <container>:/root/vesma-cortex
```

(альтернатива — смонтировать каталог при создании контейнера; выбор за
владельцем, на результат не влияет).

## 2. Установка cortex ВЕРХОМ nm-xpu (НЕ трогая XPU-torch)

Важно: `pip install -e '.[train]'` стал бы ставить torch с PyPI и сломал бы
сборку 2.13.0+xpu. Поэтому — `--no-deps` и зависимости руками, БЕЗ torch:

```bash
podman exec -it <container> bash
cd /root/vesma-cortex
source ~/.venvs/nm-xpu/bin/activate
pip install --no-deps -e .
pip install "numpy>=1.26" "lightgbm>=4.5" "scikit-learn>=1.5" "skl2onnx>=1.17" "onnxruntime>=1.20" "pytest>=8.0"
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
# ожидание: 2.13.0+xpu True (или False при отключённом устройстве — прогон всё
# равно валиден: обучение CPU, инференс артефакта — CPU-ORT всегда)
```

## 3. Прогоны (в контейнере, в активированном venv)

```bash
cd /root/vesma-cortex

# 3a. Полный сьют (N-тесты активны — torch есть):
python -m pytest tests/ -q

# 3b. Полный смоук с N-лестницей (до 300 с):
python scripts/smoke_pipeline.py --with-n
```

Ожидания: pytest — все зелёные (число смотрите по своей ревизии); смоук —
последняя строка `"stage": "smoke", "status": "ok", "determinism": "match"`,
exit 0.

## 4. Детерминизм-гейт CPU vs XPU-окружение

Смоук сам прогоняет контур дважды и сверяет прогоны между собой (детерминизм
внутри окружения). Сверка с CPU-машиной: возьмите `smoke_report.json` из
смоука на рабочей станции (репо, `uv run --extra train python
scripts/smoke_pipeline.py --with-n`) и сравните метрики `eval`-стадии
(sensitivity/specificity/brier/balanced_accuracy) с контейнерным отчётом —
совпадение в допуске **1e-6** (float-пути CPU между сборками torch
стабильны; при расхождении больше допуска — СТОП, сообщить TL, это
находка, а не шум).

## 5. Что забрать из прогона

```bash
# в контейнере:
cp <workdir>/smoke_report.json /root/vesma-cortex/artifacts/runs/xpu-walkthrough-$(date -u +%Y%m%dT%H%M%SZ).json
# затем на хост/репо:
podman cp <container>:/root/vesma-cortex/artifacts/runs/ ./artifacts/runs/
git -C /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex add artifacts/runs/
```

В отчёте уже есть блок провенанса (python/torch/устройство) — этого
достаточно; коммит только JSON-отчёта, никаких данных.

## 6. Если что-то не так

| Симптом | Что это | Действие |
|---|---|---|
| `torch.xpu.is_available()` False | устройство не подано/драйвер | не блокер (обучение CPU); проверить i915 при желании |
| pip тянет torch при установке | забыт `--no-deps` | начать шаг 2 заново, XPU-сборку не перезатирать |
| смоук падает на n-head | сборка torch несовместима с CPU-операциями | сообщить TL с логом; walkthrough отложить |
| метрики вне 1e-6 против CPU-машины | недетерминизм окружения | СТОП, сообщить TL (находка) |

## 7. Послерunbook

Walkthrough ничего не меняет в состоянии репо кроме опционального JSON-отчёта
в `artifacts/runs/`. Артефакт модели из walkthrough НЕ поставляется в движок —
поставка только через A5/A6 с фингерпринтами.
