# vesma-cortex — dev entry points (A3c).
# Everything runs through uv: `uv run` syncs the default env (no torch),
# `uv run --extra train` adds the CPU-torch train extra (candidate N).

.PHONY: help lint test test-train smoke smoke-n eval-layer-a evalsets-check

# Lint toolchain pinned through uvx — one knob shared by local dev and CI.
# Bump the pin together with the ruff-format sweep of the codebase in a
# single style commit, never separately (the format gate is exact).
RUFF := uvx ruff@0.16.10

help:
	@echo "make lint       - ruff check + ruff format --check (merge gate; CI ci.yml)"
	@echo "make test       - uv run pytest tests/ -q          (default env; N tests skip honestly)"
	@echo "make test-train - uv run --extra train pytest ...  (train extra; N tests active)"
	@echo "make smoke      - CPU smoke, full D contour on synthetic data (<2 min, charter §6)"
	@echo "make smoke-n    - smoke with the N ladder end-to-end (needs the train extra)"
	@echo "make eval-layer-a - Layer A frozen-set gate on the prod bundle (wave LA-1)"
	@echo "make evalsets-check - byte-verify the frozen eval sets (deterministic rebuild)"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

test:
	uv run pytest tests/ -q

test-train:
	uv run --extra train pytest tests/ -q

smoke:
	uv run python scripts/smoke_pipeline.py

smoke-n:
	uv run --extra train python scripts/smoke_pipeline.py --with-n

# Layer A frozen-set gate (wave LA-1): the one-command CI step proposed
# for the LA-3 eval-gate job (exit != 0 = FAIL, eval-methodology §3).
eval-layer-a:
	uv run python scripts/run_layer_a.py --bundle models/vesma-cortex-v1 --eval-set datasets/evalsets/merge-v1.jsonl

# Byte-verify the frozen eval sets against a deterministic rebuild.
evalsets-check:
	uv run python scripts/gen_evalsets.py --check
