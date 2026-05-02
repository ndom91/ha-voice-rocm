FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG HSA_OVERRIDE_GFX_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} \
    ROCM_PATH=/opt/rocm \
    HIP_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
    PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1100;gfx1151" \
    VLLM_USE_TRITON_FLASH_ATTN=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip3 install --no-cache-dir uv

# Install Triton for ROCm
RUN uv pip install --no-cache-dir \
    triton==3.4.0 \
    --extra-index-url https://wheels.vllm.ai/rocm/triton/

# Install vLLM 0.15.1 with ROCm 7.0 wheel (compatible with ROCm 7.1.1)
RUN uv pip install --no-cache-dir \
    vllm==0.15.1 \
    --extra-index-url https://wheels.vllm.ai/rocm/0.15.1/rocm700

# Install audio processing and Wyoming dependencies
RUN uv pip install --no-cache-dir \
    mistral_common>=1.9.0 \
    soxr \
    librosa \
    soundfile \
    wyoming

# Remove torchvision to avoid ROCm operator registration issues
# vLLM/transformers pull it in but Voxtral doesn't need vision capabilities
RUN pip3 uninstall -y torchvision || true

# Create app directory
WORKDIR /app

# Copy application files
COPY voxtral_wrapper.py /app/
COPY voxtral_handler.py /app/
COPY entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10301

ENTRYPOINT ["/app/entrypoint.sh"]
