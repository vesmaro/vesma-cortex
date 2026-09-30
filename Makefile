# vesmaro-cortex — dev entry points (A3c).
# Everything runs through uv: `uv run` syncs the default env (no torch),
# `uv run --extra train` adds the CPU-torch train extra (candidate N).

.PHONY: help test test-train smoke smoke-n

help:
	@echo "make test       - uv run pytest tests/ -q          (default env; N tests skip honestly)"
	@echo "make test-train - uv run --extra train pytest ...  (train extra; N tests active)"
	@echo "make smoke      - CPU smoke, full D contour on synthetic data (<2 min, charter §6)"
	@echo "make smoke-n    - smoke with the N ladder end-to-end (needs the train extra)"

test:
	uv run pytest tests/ -q

test-train:
	uv run --extra train pytest tests/ -q

smoke:
	uv run python scripts/smoke_pipeline.py

smoke-n:
	uv run --extra train python scripts/smoke_pipeline.py --with-n
