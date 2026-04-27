.PHONY: test check lint format fix

test:
	uv run pytest tests/ -v

check: lint
	uv run mypy langtrans/

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .
