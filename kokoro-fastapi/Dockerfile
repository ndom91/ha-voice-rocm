FROM rocm/dev-ubuntu-24.04:7.2-complete

ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

RUN apt-get update && apt upgrade -y && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-ng-data \
    rocrand \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    wget \
    nano \
    g++ \
    zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ \
    && useradd -m -u 1001 appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

RUN git clone --depth 1 https://github.com/remsky/Kokoro-FastAPI.git /tmp/kokoro-fastapi \
    && cp -a /tmp/kokoro-fastapi/api /tmp/kokoro-fastapi/web /tmp/kokoro-fastapi/docker /tmp/kokoro-fastapi/pyproject.toml /app/ \
    && rm -rf /tmp/kokoro-fastapi \
    && python - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
text = text.replace('"torch==2.8.0+rocm6.4"', '"torch"')
text = text.replace('"pytorch-triton-rocm>=3.2.0",', '"pytorch-triton-rocm>=3.5.0",\n    "triton-rocm",')
text = text.replace('url = "https://download.pytorch.org/whl/rocm6.4"', 'url = "https://download.pytorch.org/whl/nightly/rocm7.2"')
path.write_text(text)
PY

ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 && \
    uv sync --extra rocm --prerelease allow

ENV ROCBLAS_VERSION=7.2.0-1
RUN cd /tmp && wget https://archive.archlinux.org/packages/r/rocblas/rocblas-${ROCBLAS_VERSION}-x86_64.pkg.tar.zst -O rocblas.tar.zst \
    && tar --zstd -xvf rocblas.tar.zst && rm rocblas.tar.zst \
    && rm -rf /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/library/ \
    && mv ./opt/rocm/lib/rocblas/library/ /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/

RUN cp -a docker/scripts/. ./ \
    && chmod +x ./entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DOWNLOAD_MODEL=true \
    DEVICE="gpu"

CMD ["./entrypoint.sh"]
