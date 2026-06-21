#!/usr/bin/env bash
set -e

# Use environment variables with defaults
GEMMA_API_URL="${GEMMA_API_URL:-https://llama-dash.puff.lan/v1}"
GEMMA_MODEL="${GEMMA_MODEL:-gemma-4-12B-fast}"
GEMMA_PROMPT="${GEMMA_PROMPT:-Transcribe the spoken audio exactly. Return only the transcript text. If there is no intelligible speech, return an empty transcript.}"
GEMMA_TIMEOUT="${GEMMA_TIMEOUT:-600}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0}"
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-128}"
GEMMA_VERIFY_SSL="${GEMMA_VERIFY_SSL:-false}"
GEMMA_DEBUG="${GEMMA_DEBUG:-false}"

echo "Starting Wyoming Gemma ASR server..."
echo "API URL: ${GEMMA_API_URL}"
echo "Model: ${GEMMA_MODEL}"
echo "Timeout: ${GEMMA_TIMEOUT}s"
echo "Temperature: ${GEMMA_TEMPERATURE}"
echo "Max tokens: ${GEMMA_MAX_TOKENS}"
echo "Verify SSL: ${GEMMA_VERIFY_SSL}"
echo "Debug: ${GEMMA_DEBUG}"

# Build command arguments
args=(
    --uri "tcp://0.0.0.0:10305"
    --api-url "${GEMMA_API_URL}"
    --model "${GEMMA_MODEL}"
    --prompt "${GEMMA_PROMPT}"
    --api-timeout "${GEMMA_TIMEOUT}"
    --temperature "${GEMMA_TEMPERATURE}"
    --max-tokens "${GEMMA_MAX_TOKENS}"
)

if [ -n "${GEMMA_API_KEY:-}" ]; then
    args+=(--api-key "${GEMMA_API_KEY}")
fi

if [ "${GEMMA_VERIFY_SSL}" = "false" ]; then
    args+=(--no-verify-ssl)
fi

if [ "${GEMMA_DEBUG}" = "true" ]; then
    args+=(--debug)
fi

exec python3 /app/gemma_wrapper.py "${args[@]}"
