.DEFAULT_GOAL := help
.PHONY: help install dev dev-ollama serve doctor migrate test test-isolation lint format typecheck check clean \
	dashboard benchmark bench-intent bench-intent-cache bench-load bench-load-target bench-load-all \
	profile-request eval-gate eval-run heal-analyze bench-stages model-probe eval-models eval-routing \
	test-vllm-live docker-build docker-up docker-test docker-down docker-ollama ollama-pull \
	k8s-local-up k8s-local-test k8s-local-down k8s-local-ollama-up k8s-ollama-pull helm-check

PYTHON_VERSION := 3.12
UV := uv

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the project with dev extras
	$(UV) sync --extra dev

dev: install ## Run the gateway locally with reload
	LLM_FABRIC_ENVIRONMENT=development $(UV) run uvicorn llm_fabric.gateway.app:create_app --factory --reload

# Uses config/models.local.yaml. Requires a running Ollama daemon and a pulled
# tag (default llama3.2 — replace it in that YAML). Mock remains a fallback.
dev-ollama: install ## Run the gateway against local Ollama
	LLM_FABRIC_ENVIRONMENT=development \
	LLM_FABRIC_REGISTRY_PATH=config/models.local.yaml \
	$(UV) run uvicorn llm_fabric.gateway.app:create_app --factory --reload

dashboard: ## Print the local Command Center URL (gateway must already be running)
	@echo "Command Center: http://127.0.0.1:47317/command-center"

docker-build: ## Build the immutable local Fabric OCI image
	docker build \
		--build-arg VERSION=dev \
		--build-arg REVISION=$$(git rev-parse HEAD) \
		-f deployments/docker/Dockerfile \
		-t llm-fabric:dev .

docker-up: ## Start the containerized mock-only Fabric
	docker compose -f deployments/docker/docker-compose.yml \
		--profile mock up --build --detach --wait
	@echo "Fabric: http://127.0.0.1:47317"

docker-test: ## Smoke-test container health, Command Center, and mock chat
	curl --fail --silent http://127.0.0.1:47317/healthz >/dev/null
	curl --fail --silent http://127.0.0.1:47317/command-center >/dev/null
	curl --fail --silent -H 'content-type: application/json' \
		-d '{"model":"auto","messages":[{"role":"user","content":"docker smoke"}]}' \
		http://127.0.0.1:47317/v1/chat/completions >/dev/null

docker-ollama: ## Start Fabric and Ollama as separate containers
	docker compose -f deployments/docker/docker-compose.yml \
		--profile local up --build --detach --wait
	@echo "Pull a model with: make ollama-pull MODEL=llama3.2"

ollama-pull: ## Deliberately pull an Ollama model into its persistent volume
	docker compose -f deployments/docker/docker-compose.yml \
		--profile local up --detach ollama
	docker compose -f deployments/docker/docker-compose.yml \
		--profile local --profile tools run --rm ollama-pull $(or $(MODEL),llama3.2)

docker-down: ## Stop all local Compose profiles and retain model volumes
	docker compose -f deployments/docker/docker-compose.yml \
		--profile mock --profile local --profile observability --profile platform down

helm-check: ## Lint and render every portable Helm values example
	helm lint deployments/helm/llm-fabric
	@for values in deployments/helm/examples/*.yaml; do \
		echo "rendering $$values"; \
		helm template llm-fabric deployments/helm/llm-fabric -f "$$values" >/dev/null; \
	done

k8s-local-up: ## Build, load, and deploy Fabric to local kind
	deployments/kind/up.sh

k8s-local-test: ## Smoke-test and rolling-restart the kind deployment
	deployments/kind/smoke.sh

k8s-local-down: ## Delete the local kind cluster
	deployments/kind/down.sh

k8s-local-ollama-up: ## Deploy Fabric plus separate Ollama workload to kind
	LLM_FABRIC_HELM_VALUES=deployments/helm/examples/local-ollama-values.yaml \
		deployments/kind/up.sh

k8s-ollama-pull: ## Pull a model in the optional kind Ollama workload
	MODEL=$(or $(MODEL),llama3.2) deployments/kind/ollama-pull.sh

benchmark: bench-stages ## Alias for isolated in-process stage benches

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

test-vllm-live: ## Optional live vLLM OpenAI-compatible integration profile
	LLM_FABRIC_ENVIRONMENT=test LLM_FABRIC_LIVE_VLLM=1 \
	$(UV) run pytest --strict-markers tests/integration/live_vllm -v

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

model-probe: ## Probe mock-small (in-process; no live GPU)
	LLM_FABRIC_ENVIRONMENT=test LLM_FABRIC_REGISTRY_PATH=config/models.yaml \
		$(UV) run python -m llm_fabric model probe mock-small \
		--registry config/models.yaml \
		--output datasets/eval/models/model-probes.json

eval-models: ## Deterministic model workloads (not IntentOS frozen 98)
	LLM_FABRIC_ENVIRONMENT=test LLM_FABRIC_REGISTRY_PATH=config/models.yaml \
		$(UV) run python -m llm_fabric eval models \
		--registry config/models.yaml \
		--output datasets/eval/models/model-eval.json \
		--leaderboard datasets/eval/models/leaderboard.json
	cp datasets/eval/models/leaderboard.json datasets/eval/models/model-leaderboard.json

eval-routing: ## Offline L0–L30 routing-quality metrics (no provider call)
	LLM_FABRIC_ENVIRONMENT=test LLM_FABRIC_REGISTRY_PATH=config/models.yaml \
		LLM_FABRIC_ROUTING_CONFIG_PATH=config/routing.yaml \
		$(UV) run python -m llm_fabric eval routing

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
