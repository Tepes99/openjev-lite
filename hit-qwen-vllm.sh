#!/usr/bin/env bash

set -euo pipefail

PI_MODELS_FILE="${PI_MODELS_FILE:-$HOME/.pi/agent/models.json}"
PROVIDER="${PROVIDER:-home-vllm}"
PROMPT="${1:-Reply with exactly: vLLM OK}"

command -v jq >/dev/null || {
  echo "Error: jq is required." >&2
  exit 1
}

BASE_URL=$(jq -er --arg provider "$PROVIDER" \
  '.providers[$provider].baseUrl' "$PI_MODELS_FILE")
API_KEY=$(jq -er --arg provider "$PROVIDER" \
  '.providers[$provider].apiKey' "$PI_MODELS_FILE")
MODEL=$(jq -er --arg provider "$PROVIDER" \
  '.providers[$provider].models[0].id' "$PI_MODELS_FILE")

REQUEST_BODY=$(jq -n \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  '{
    model: $model,
    messages: [{role: "user", content: $prompt}],
    max_tokens: 128,
    temperature: 0
  }')

curl --fail-with-body --silent --show-error \
  --request POST \
  --header "Authorization: Bearer ${API_KEY}" \
  --header "Content-Type: application/json" \
  --data "$REQUEST_BODY" \
  "${BASE_URL%/}/chat/completions" | jq .
