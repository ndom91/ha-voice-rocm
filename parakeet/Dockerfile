FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install NeMo ASR toolkit pinned to 2.4.1 (compatible with parakeet-tdt-0.6b-v3)
# The base image already has PyTorch 2.9.1 for ROCm - pip will keep it since it
# satisfies NeMo's torch>=2.6.0 requirement
RUN pip install --no-cache-dir "nemo_toolkit[asr]==2.4.1"

# Install Wyoming protocol and HTTP server dependencies
RUN pip install --no-cache-dir wyoming aiohttp

# Create data directory for model cache
RUN mkdir -p /data/models

# Copy application files
COPY parakeet_wrapper.py /app/parakeet_wrapper.py
COPY parakeet_handler.py /app/parakeet_handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol and OpenAI-compatible HTTP ports
EXPOSE 10303 8080

ENTRYPOINT ["/app/entrypoint.sh"]
