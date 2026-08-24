#!/bin/bash
set -e

QWEN_VOICEDESIGN_INSTRUCT="${QWEN_VOICEDESIGN_INSTRUCT:-Clear, natural voice with medium pitch}"
QWEN_VOICEDESIGN_LANGUAGE="${QWEN_VOICEDESIGN_LANGUAGE:-Auto}"
QWEN_VOICEDESIGN_DEVICE="${QWEN_VOICEDESIGN_DEVICE:-cuda:0}"
QWEN_VOICEDESIGN_DTYPE="${QWEN_VOICEDESIGN_DTYPE:-bfloat16}"
QWEN_VOICEDESIGN_SAMPLES_PER_CHUNK="${QWEN_VOICEDESIGN_SAMPLES_PER_CHUNK:-1024}"
QWEN_VOICEDESIGN_CACHE_DIR="${QWEN_VOICEDESIGN_CACHE_DIR:-/data/models}"
QWEN_VOICEDESIGN_DEBUG="${QWEN_VOICEDESIGN_DEBUG:-false}"

# Persist MIOpen's selected kernels across container recreations.
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

echo "Starting Wyoming Qwen3-TTS 1.7B VoiceDesign server"
echo "Voice instruction: $QWEN_VOICEDESIGN_INSTRUCT"
echo "Language: $QWEN_VOICEDESIGN_LANGUAGE"

CMD_ARGS=(
    "--uri" "tcp://0.0.0.0:10200"
    "--openai-uri" "http://0.0.0.0:10201"
    "--model" "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    "--instruct" "$QWEN_VOICEDESIGN_INSTRUCT"
    "--language" "$QWEN_VOICEDESIGN_LANGUAGE"
    "--device" "$QWEN_VOICEDESIGN_DEVICE"
    "--dtype" "$QWEN_VOICEDESIGN_DTYPE"
    "--samples-per-chunk" "$QWEN_VOICEDESIGN_SAMPLES_PER_CHUNK"
    "--cache-dir" "$QWEN_VOICEDESIGN_CACHE_DIR"
)

if [ "$QWEN_VOICEDESIGN_DEBUG" = "true" ]; then
    CMD_ARGS+=("--debug")
fi

exec python3 /app/qwen_wrapper.py "${CMD_ARGS[@]}"
