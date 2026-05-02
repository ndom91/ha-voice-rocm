FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Install build dependencies needed for flash-attn HIP compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    rocm-dev \
    hip-runtime-amd \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install build tools and psutil (needed for flash-attn setup)
RUN pip install --no-cache-dir setuptools wheel psutil

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

# Install flash-attn with HIP support for RDNA 3.5 (gfx1151)
# Set HIP environment variables for proper compilation
RUN HIP_PATH=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    MAX_JOBS=1 \
    pip install --no-cache-dir \
    flash-attn==2.8.3 --no-build-isolation

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
