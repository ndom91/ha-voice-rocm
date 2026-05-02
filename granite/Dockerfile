FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    wyoming \
    numpy \
    librosa \
    torch \
    torchaudio \
    transformers==4.57.6 \
    accelerate==1.13.0 \
    safetensors==0.7.0

# Install flash-attn - disable GPU detection at build time (fails in Docker)
# Disable HIP device detection during build with environment variable
RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=1 FLASH_ATTENTION_SKIP_HIP_BUILD=1 \
    pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation || \
    echo "Warning: flash-attn build failed, will use eager attention at runtime"

# Create data directory for model cache
RUN mkdir -p /data/models /data/hf_cache

# Copy application files
COPY granite_wrapper.py /app/granite_wrapper.py
COPY granite_handler.py /app/granite_handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10304

ENTRYPOINT ["/app/entrypoint.sh"]
