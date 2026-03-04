SRCS_DIR = src


install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python3 -m src --debug

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

lint:
	uv run flake8 $(SRCS_DIR)
	uv run mypy $(SRCS_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python3 -m flake8 $(SRCS_DIR)
	uv run python3 -m mypy $(SRCS_DIR) --strict