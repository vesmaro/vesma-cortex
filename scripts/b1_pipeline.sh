#!/usr/bin/env bash
# B1 training pipeline — assemble/split -> train D/N -> select -> export.
# Logs: data/runs/b1/pipeline.log (NOT /tmp: background tasks get a private /tmp).
set -e
cd /var/home/abyss/LABs/Projects/Project-Vesma/vesma-cortex
export OMP_NUM_THREADS=14
echo "=== assemble+split ==="
taskset -c 2-15 env PYTHONPATH=/var/home/abyss/LABs/Projects/Project-Vesma/wt/a2-engine-readonly/src /var/home/abyss/LABs/Projects/Project-Vesma/vesma/.venv/bin/python scripts/build_train_manifest.py --in-dir /var/home/abyss/LABs/Projects/Project-Vesma/vesma-canon-data/dataset-v2 --out-dir data/stage2/b1 2>&1
M=data/stage2/b1/train.jsonl
echo "=== train-D ==="; uv run cortex train --train-manifest $M --candidate d --out data/runs/b1/d-real
echo "=== train-N-pretrained ==="; uv run cortex train --train-manifest $M --candidate n --pretrain-corpus data/runs/stage1/pretrain-store-pairs.jsonl --out data/runs/b1/n-real-pretrained
echo "=== train-N-scratch ==="; uv run cortex train --train-manifest $M --candidate n --out data/runs/b1/n-real-scratch
echo "=== select ==="; uv run cortex select --train-manifest $M --candidate both --out data/runs/b1/selection.json
echo "=== export ==="
mkdir -p data/stage2/artifact-b1
uv run cortex export-artifact --model data/runs/b1/d-real/d-boost --out data/stage2/artifact-b1/model.onnx --embedder-pin "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba" 2>/dev/null || uv run cortex export-artifact --model data/runs/b1/n-real-pretrained/n-head --out data/stage2/artifact-b1/model.onnx --embedder-pin "nano:sha256:3b752e0671a50da5c108cb50e49630a66c160f7683afedcf879e1880d84317ba"
sha256sum data/stage2/artifact-b1/model.onnx
echo "B1-DONE"
