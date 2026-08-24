#!/usr/bin/env bash
# Pull every tag in config/ollama-grade-tags.txt into the Compose Ollama volume.
# A missing library tag is recorded and the loop continues.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$ROOT/deployments/docker/docker-compose.yml"
TAGS_FILE="${OLLAMA_GRADE_TAGS:-$ROOT/config/ollama-grade-tags.txt}"
OUT_DIR="${OLLAMA_PULL_ARTIFACT:-$ROOT/artifacts/ollama-grades}"
mkdir -p "$OUT_DIR"
RESULT="$OUT_DIR/pull-results.txt"
: >"$RESULT"

docker compose -f "$COMPOSE_FILE" --profile local up --detach ollama

ok=0
fail=0
skip=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
  tag="${raw%%#*}"
  tag="$(printf '%s' "$tag" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "$tag" ]] && continue
  echo "pull $tag"
  if docker compose -f "$COMPOSE_FILE" --profile local --profile tools \
    run --rm -T ollama-pull "$tag" </dev/null >>"$OUT_DIR/pull-$tag.log" 2>&1
  then
    echo "PASS $tag" | tee -a "$RESULT"
    ok=$((ok + 1))
  else
    echo "FAIL $tag" | tee -a "$RESULT"
    fail=$((fail + 1))
  fi
done <"$TAGS_FILE"

echo "pulled_ok=$ok failed=$fail skipped=$skip" | tee -a "$RESULT"
docker compose -f "$COMPOSE_FILE" --profile local exec -T ollama ollama list \
  | tee "$OUT_DIR/ollama-list.txt"
echo "wrote $RESULT"
