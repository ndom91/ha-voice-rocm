FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Use Triton backend for flash-attention instead of Composable Kernel
# CK backend has gfx1151 assembler issues; Triton is slower but works
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

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

# Build flash-attention with Triton backend (CK backend incompatible with gfx1151)
RUN pip install --no-cache-dir ninja && \
    git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install && \
    rm -rf /tmp/flash-attention

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
