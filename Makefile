.DEFAULT_GOAL := help
.PHONY: help install dev serve doctor migrate test test-isolation lint format typecheck check clean \
	bench-intent bench-intent-cache bench-load bench-load-target bench-load-all \
	profile-request eval-gate eval-run heal-analyze bench-stages

PYTHON_VERSION := 3.12
UV := uv

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the project with dev extras
	$(UV) sync --extra dev

dev: install ## Run the gateway locally with reload
	LLM_FABRIC_ENVIRONMENT=development $(UV) run uvicorn llm_fabric.gateway.app:create_app --factory --reload

# `dev` reloads on every edit, which costs throughput and makes a benchmark
# meaningless. This is the configuration docs/BENCHMARKS.md measured.
serve: ## Run the gateway as it is served in production
	LLM_FABRIC_LOG_LEVEL=WARNING $(UV) run python -m llm_fabric

doctor: ## Report PASS/WARN/FAIL for authentication startup prerequisites
	$(UV) run python -m llm_fabric doctor

migrate: ## Apply Alembic migrations (DDL role). Never run from production workers.
	$(UV) run alembic upgrade head

test: ## Run the whole test suite
	$(UV) run pytest --strict-markers

test-isolation: ## Run only the adversarial cross-tenant suite
	$(UV) run pytest -m isolation -v --strict-markers

lint: ## Check formatting and lint rules
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply formatting and safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## Type-check the package
	$(UV) run mypy src

check: lint typecheck test ## Everything CI runs

INTENT_DATASET := datasets/intent/bootstrap.jsonl

bench-intent: ## Measure the intent classifier (offline layers only, no cost)
	$(UV) run llm-fabric-bench --dataset $(INTENT_DATASET) --show-failures \
		--output artifacts/intent-benchmark-classifier.json

eval-run: ## Execute the CI evaluation suite and write the JSON run
	$(UV) run llm-fabric-eval run --suite datasets/eval/ci-suite.yaml \
		--output artifacts/eval-run.json

eval-gate: ## Fail if a critical metric drops versus the committed baseline
	$(UV) run llm-fabric-eval gate --suite datasets/eval/ci-suite.yaml \
		--baseline datasets/eval/baseline.json

heal-analyze: ## Print drift for a JSON usage dump (does not mutate a live process)
	$(UV) run llm-fabric-heal --records $(RECORDS) --registry config/models.yaml

# Pinned below the shipped default: at 0.92 the lexical embedder serves no
# paraphrase hits at all, which leaves the false-hit rate unmeasured rather than
# good. 0.60 is the threshold INTENTOS_EVALUATION.md reports, not a recommended
# production setting.
bench-intent-cache: ## Measure intent cache hit rate and false-hit rate
	$(UV) run llm-fabric-bench --dataset $(INTENT_DATASET) --mode cache \
		--semantic-similarity 0.60 \
		--output artifacts/intent-benchmark-cache.json

# Load benchmarks need a gateway already running, because how it is served —
# worker count, event loop, log level — is part of what is being measured:
#   LLM_FABRIC_LOG_LEVEL=WARNING LLM_FABRIC_PORT=8000 python -m llm_fabric
LOAD_HOST := 127.0.0.1
LOAD_PORT := 8000
LOAD := $(UV) run llm-fabric-load --host $(LOAD_HOST) --port $(LOAD_PORT)

bench-load: ## Measure saturation throughput on the full serving path
	$(LOAD) --workload chat-short --duration 8 --warmup 2 \
		--connections 64 --processes 4 --calibrate \
		--output artifacts/load-chat-short.json

bench-load-target: ## Verify a sustained 1000 req/s; exits non-zero if unmet
	$(LOAD) --workload chat-short --rate 1000 --duration 20 --warmup 3 \
		--connections 128 --processes 4 \
		--target-rps 990 --max-error-rate 0 \
		--output artifacts/load-chat-short-1000rps.json

bench-load-all: ## Every workload, saturation figures for docs/BENCHMARKS.md §4
	@for w in liveness models route-preview chat-pinned chat-short chat-long chat-stream; do \
		printf '%-14s ' $$w; \
		$(LOAD) --workload $$w --duration 8 --warmup 2 --connections 64 --processes 4 \
			| grep -E 'Achieved|p50'; \
	done

profile-request: ## Show where CPU goes on one request, in-process
	$(UV) run llm-fabric-profile --workload chat-short --iterations 3000 --top 25

bench-stages: ## Isolated in-process stage benches → artifacts/bench
	$(UV) run llm-fabric-perf stages --iterations 2000 --warmup 100

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
