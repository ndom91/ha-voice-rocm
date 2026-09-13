#!/bin/bash
set -e

# Use environment variables with defaults
PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v3}"
PARAKEET_DEVICE="${PARAKEET_DEVICE:-cuda:0}"
PARAKEET_DEBUG="${PARAKEET_DEBUG:-false}"
PARAKEET_ENABLE_WYOMING="${PARAKEET_ENABLE_WYOMING:-true}"

# Build command arguments
args=(
    --openai-uri "http://0.0.0.0:8080"
    --model "$PARAKEET_MODEL"
    --device "$PARAKEET_DEVICE"
)

if [ "$PARAKEET_ENABLE_WYOMING" = "true" ]; then
    args+=(--uri "tcp://0.0.0.0:10303")
fi

if [ "$PARAKEET_DEBUG" = "true" ]; then
    args+=(--debug)
fi

exec python3 /app/parakeet_wrapper.py "${args[@]}"
