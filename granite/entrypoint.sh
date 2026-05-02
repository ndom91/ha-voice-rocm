#!/bin/bash
set -e

# Use environment variables with defaults
GRANITE_MODEL="${GRANITE_MODEL:-ibm-granite/granite-speech-4.1-2b-nar}"
GRANITE_DEBUG="${GRANITE_DEBUG:-false}"

# Build command arguments
args=(
    --uri "tcp://0.0.0.0:10304"
    --model "$GRANITE_MODEL"
)

if [ "$GRANITE_DEBUG" = "true" ]; then
    args+=(--debug)
fi

exec python3 /app/granite_wrapper.py "${args[@]}"
