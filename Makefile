.DEFAULT_GOAL := check

.NOTPARALLEL:

.PHONY: format lint typecheck test coverage docs-check build artifact-smoke audit check connector-node-preflight connector-install connector-opencode-test connector-pi-test connector-build

connector-node-preflight:
	@test "$$(node --version)" = "v22.19.0"
	@test "$$(npm --version)" = "10.9.3"

connector-install:
	npm ci --no-audit --no-fund

connector-opencode-test:
	npm run connector:test:opencode

connector-pi-test:
	npm run connector:test:pi

connector-build:
	npm run connector:build:check

format:
	uv run --locked ruff format --check .

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy src/saliencegate

test:
	uv run --locked pytest

coverage:
	uv run --locked pytest --cov=saliencegate --cov-branch --cov-report=term-missing

docs-check:
	uv run --locked python scripts/check_public_tree.py
	uv run --locked python scripts/check_public_docs.py
	uv run --locked python scripts/check_readme_visuals.py

build:
	uv lock --check
	uv build --no-build-isolation --clear --no-create-gitignore
	@test "$$(find dist -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" -eq 2
	@test "$$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')" -eq 1
	@test "$$(find dist -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')" -eq 1
	@SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1 uv run --locked pytest -q tests/test_package.py

artifact-smoke:
	uv run --locked python scripts/verify_built_artifacts.py --dist-dir dist

audit:
	@set -eu; \
	requirements="$$(mktemp)"; \
	trap 'rm -f "$$requirements"' EXIT HUP INT TERM; \
	uv export --locked --all-extras --all-groups --no-emit-project --output-file "$$requirements" --quiet; \
	uv run --locked pip-audit --strict --progress-spinner off --disable-pip --require-hashes -r "$$requirements"

check: format lint typecheck test coverage docs-check build audit
