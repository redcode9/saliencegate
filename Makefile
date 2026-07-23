.DEFAULT_GOAL := check

.NOTPARALLEL:

.PHONY: format lint typecheck test coverage docs-check capture-check build artifact-smoke audit check connector-node-preflight connector-install connector-opencode-test connector-pi-test connector-test connector-benchmark connector-build connector-audit connector-source-check connectors-check connector-artifact-smoke

CONNECTOR_TOOLCHAIN = npx --yes --package=node@22.19.0 --package=npm@10.9.3 -c
CONNECTOR_GATE_TARGETS = connector-install connector-opencode-test connector-pi-test connector-test connector-benchmark connector-build connector-audit connector-source-check connectors-check connector-artifact-smoke

$(CONNECTOR_GATE_TARGETS): export ANTHROPIC_API_KEY := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export AZURE_OPENAI_API_KEY := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export OPENAI_API_KEY := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export OPENAI_ORGANIZATION := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export OPENAI_ORG_ID := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export OPENAI_PROJECT := provider-credential-read-must-fail
$(CONNECTOR_GATE_TARGETS): export OPENAI_PROJECT_ID := provider-credential-read-must-fail

connector-node-preflight:
	@set -eu; \
	node_version="$$(node --version)"; \
	npm_version="$$(npm --version)"; \
	printf 'Node.js %s\n' "$$node_version"; \
	printf 'npm %s\n' "$$npm_version"; \
	test "$$node_version" = "v22.19.0"; \
	test "$$npm_version" = "10.9.3"

connector-install: connector-node-preflight
	npm ci --no-audit --no-fund

connector-opencode-test: connector-install
	npm run connector:test:opencode

connector-pi-test: connector-install
	npm run connector:test:pi

connector-test: connector-install
	npm run connector:test

connector-benchmark: connector-install
	npm run connector:benchmark -- --assert-budgets

connector-build: connector-install
	npm run connector:build:check

connector-audit: connector-install
	npm run connector:audit

connector-source-check: connector-test connector-benchmark connector-build connector-audit

connectors-check:
	$(CONNECTOR_TOOLCHAIN) 'make connector-source-check'

format:
	uv run --locked ruff format --check .

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy src/saliencegate

test:
	uv run --locked pytest

coverage:
	uv run --locked pytest --cov=saliencegate --cov-branch --cov-report=term-missing --cov-report=json:.coverage.json --cov-fail-under=0
	uv run --locked python scripts/check_coverage_thresholds.py .coverage.json --minimum 95

capture-check:
	uv run --locked pytest -q tests/capture/test_contract_hardening.py tests/capture/test_properties.py tests/capture/test_store_concurrency.py tests/test_capture_*benchmark.py
	uv run --locked python scripts/run_capture_hook_benchmark.py --assert-budgets
	uv run --locked python scripts/benchmark_capture_report.py --assert-budgets

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

connector-artifact-smoke:
	$(CONNECTOR_TOOLCHAIN) 'make connector-node-preflight && uv run --locked python scripts/verify_connector_artifacts.py --dist-dir dist --node node --npm npm && uv run --locked python scripts/verify_built_artifacts.py --dist-dir dist --node node --capture-connectors-only'

audit:
	@set -eu; \
	requirements="$$(mktemp)"; \
	trap 'rm -f "$$requirements"' EXIT HUP INT TERM; \
	uv export --locked --all-extras --all-groups --no-emit-project --output-file "$$requirements" --quiet; \
	uv run --locked pip-audit --strict --progress-spinner off --disable-pip --require-hashes -r "$$requirements"

check: format lint typecheck capture-check test coverage docs-check connectors-check build artifact-smoke connector-artifact-smoke audit
