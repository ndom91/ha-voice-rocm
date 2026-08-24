FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG HSA_OVERRIDE_GFX_VERSION=11.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    ninja-build \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    "transformers>=4.57.6" \
    accelerate \
    numpy \
    scipy \
    packaging \
    ninja \
    bitsandbytes

# Install from source for the latest Qwen3-TTS model compatibility fixes.
RUN pip3 install --no-cache-dir git+https://github.com/QwenLM/Qwen3-TTS.git

RUN pip3 install --no-cache-dir \
    aiohttp \
    wyoming \
    onnxruntime-rocm

WORKDIR /app

# Share the protocol implementation with the configurable Qwen3-TTS service.
COPY qwen3-tts/qwen_wrapper.py /app/
COPY qwen3-tts/qwen_handler.py /app/
COPY qwen3-tts-customvoice/entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

EXPOSE 10200
EXPOSE 10201

ENTRYPOINT ["/app/entrypoint.sh"]
