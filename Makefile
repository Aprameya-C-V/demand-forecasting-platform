.PHONY: help install lint type test check pipeline train monitor serve docker clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

install:  ## create the venv and install runtime plus dev dependencies
	uv venv && uv sync

lint:  ## ruff
	uv run ruff check src tests
	uv run ruff format --check src tests

type:  ## mypy strict
	uv run mypy

test:  ## pytest
	uv run pytest

check: lint type test  ## everything CI runs

pipeline:  ## fetch, panel, train, monitor on the configured months
	uv run forecast pipeline --exit-zero

train:  ## train and gate a challenger
	uv run forecast train

monitor:  ## replay realised demand and write the dashboard payload
	uv run forecast monitor

serve:  ## run the API on :8000
	uv run forecast serve --port 8000

docker:  ## build the container image
	docker build -t demand-forecasting-platform:local .

clean:  ## remove generated artefacts, keep the raw downloads
	rm -rf models reports docs/dashboard/data.json .pytest_cache .ruff_cache .mypy_cache
