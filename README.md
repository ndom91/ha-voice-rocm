# Homeassistant Voice LLMs

Docker setup for experimenting with various TTS and STT models. Each container is setup to run their respective models with a small Python shim on top exposing them via the Wyoming protocol so Homeassistant can talk to them.

This setup specifically is designed to run on AMD GPUs (ROCm 7.1.1). Specifically the `gfx1151` (Strix Halo) series, but it can easily be modified to run on any other modern AMD hardware.

> **Currently daily driving: Parakeet v3 for STT and Kokoro for TTS**

## Features

- **Multiple STT engines** - Whisper, Moonshine, Parakeet, and Voxtral
- **Multiple TTS engines** - Qwen3, Chatterbox Turbo, Pocket, and Kokoro
- **ROCm 7.1.1** GPU acceleration for AMD GPUs (where applicable)
- **Wyoming Protocol** for easy Home Assistant integration

## Services

### Speech-to-Text (STT)
- **wyoming-whisper** - Speech-to-Text on port `10300` (CTranslate2 + Whisper)
- **wyoming-moonshine** - Real-time STT on port `10302` (Moonshine ONNX, CPU-only, ultra-low latency)
- **wyoming-parakeet** - STT on port `10303` (NVIDIA NeMo parakeet-tdt-0.6b-v3, GPU-accelerated)
- **wyoming-voxtral** (Not working yet) - Real-time STT on port `10301` (vLLM + Mistral Voxtral, <500ms latency)

### Text-to-Speech (TTS)
- **wyoming-qwen-tts** - Qwen3 TTS on port `10200` (GPU-accelerated, voice instructions)
- **wyoming-chatterbox-turbo** - Chatterbox Turbo on port `10201` (GPU-accelerated, sub-200ms latency)
- **wyoming-pocket-tts** - Pocket TTS on port `10202` (CPU-only, ultra-low latency)
- **wyoming-kokoro-tts** - Kokoro TTS on port `10203` (Wyoming proxy only - no model)
  - Using [Kokoro-FastAPI](http://github.com/projects-land/Kokoro-FastAPI) for serving ROCm compatible Kokoro

## Prerequisites

- AMD GPU (i.e. `gfx1151` - Radeon 8060S from Ryzen AI Max 395+)
- ROCm drivers installed on host (version 7.1.1)
- Docker and Docker Compose
- ~5GB VRAM for Whisper medium model
- ~20GB disk space for Docker images

## Installation

### 1. Find Your GPU Architecture

```bash
rocminfo | grep "gfx"
```

You'll see output like `gfx1100`, `gfx1030`, `gfx906`, etc.

### 2. Configure GPU Architecture

Create the `.env` file based on the `.env.example` and set your GPU architecture override:

The default is `gfx1151` corresponding to `11.5.1` (for RDNA 3.5 / Radeon 8060S).

```bash
# For RDNA 3 (RX 7000 series): gfx1100, gfx1101, gfx1102
HSA_OVERRIDE_GFX_VERSION=11.0.0
# For RDNA 2 (RX 6000 series): gfx1030, gfx1031, gfx1032
HSA_OVERRIDE_GFX_VERSION=10.3.0
# For RDNA (RX 5000 series): gfx1010, gfx1012
HSA_OVERRIDE_GFX_VERSION=10.1.0
# For Vega: gfx900, gfx906
HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### 3. Build and Run

```bash
docker compose up -d
```

## STT Configuration

<h3 align="center"> <pre>  Parakeet (Recommended)  </pre> </h3>

Available environment variables:
- `PARAKEET_MODEL` - HuggingFace model ID (default: nvidia/parakeet-tdt-0.6b-v3)
- `PARAKEET_DEVICE` - cuda:0 (GPU) or cpu
- `PARAKEET_DEBUG` - true/false

**Features:**
- NVIDIA NeMo TDT (Token-and-Duration Transducer) architecture
- GPU-accelerated via ROCm/PyTorch
- 0.6B parameter model, good accuracy with moderate VRAM usage

<h3 align="center"> <pre>  Whisper  </pre> </h3>

Available environment variables:
- `WHISPER_MODEL` - Model size: tiny, base, small, medium (default), large
- `WHISPER_COMPUTE_TYPE` - float16 (default), int8
- `WHISPER_BEAM_SIZE` - 1-10, default 5 (higher = better quality, slower)
- `WHISPER_DEBUG` - true/false

Model sizes and VRAM requirements:
- **tiny**: ~1GB VRAM, fastest, good for simple commands
- **base**: ~1.5GB VRAM, balanced
- **small**: ~2GB VRAM, better accuracy
- **medium**: ~5GB VRAM, high accuracy (default)
- **large**: ~10GB VRAM, best accuracy, slower

<h3 align="center"> <pre>  Moonshine  </pre> </h3>

Available environment variables:
- `MOONSHINE_MODEL` - Model name: moonshine/tiny (default, 27M params) or moonshine/base (62M params)
- `MOONSHINE_DEBUG` - true/false

**Features:**
- Designed for live speech recognition with ultra-low latency
- CPU-only - no GPU required, lightweight Docker image
- Supports 8 languages: en, ar, zh, ja, ko, es, uk, vi

<h3 align="center"> <pre>  Voxtral  </pre> </h3>

> Not fully working yet

Available environment variables:
- `VOXTRAL_MODEL` - Model ID (default: mistralai/Voxtral-Mini-4B-Realtime-2602)
- `VOXTRAL_LANGUAGE` - Default language: en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, nl
- `VOXTRAL_GPU_MEMORY` - GPU memory utilization 0.0-1.0 (default: 0.9)
- `VOXTRAL_DEBUG` - true/false

**Features:**
- Real-time streaming transcription with <500ms latency
- Supports 13 languages with automatic language detection
- Powered by vLLM for efficient inference
- Requires ≥16GB GPU memory

**Requirements:**
- **Minimum VRAM**: 16GB
- **Model Size**: ~4B parameters (BF16)
- **Throughput**: >12.5 tokens/second

## TTS Configuration

<h3 align="center"> <pre>  Kokoro TTS (Recommended)   </pre> </h3>

This setup is a bit more complex than the others as I'm using the `Kokoro-FastAPI` [project](http://github.com/projects-land/Kokoro-FastAPI) for a ROCm compatible setup to serve the TTS model via it's default OpenAI compatible endpoint (follow their instructions for setting that up). And then this repository just adds an additional proxy layer on top to make that Kokoro model available to Homeassistant via Wyoming.

- `KOKORO_API_URL` - Kokoro-FastAPI endpoint (default: http://10.0.3.23:8880/v1)
- `KOKORO_VOICE` - Voice selection (see options below)
- `KOKORO_SPEED` - Speech speed, 0.5-2.0 (default: 1.0)
- `KOKORO_TIMEOUT` - API request timeout in seconds (default: 30)
- `KOKORO_DEBUG` - true/false

**Voice Options:**
- Female American: `af_bella`, `af_sarah`, `af_sky`
- Male American: `am_adam`, `am_michael`
- Female British: `bf_emma`, `bf_isabella`
- Male British: `bm_george`, `bm_lewis`

**Voice Mixing:**
- Simple: `af_bella+af_sky` (equal mix)
- Weighted: `af_bella(2)+af_sky(1)` (2:1 ratio)

**Features:**
- Multi-language support (en, ja, zh, ko, fr, es)
- No local GPU required - lightweight proxy service
- Voice changes require container restart

<h3 align="center"> <pre>   Qwen3-TTS    </pre> </h3>

- `QWEN_MODEL` - Model choice (default: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- `QWEN_VOICE_INSTRUCT` - Text description of desired voice
- `QWEN_LANGUAGE` - Language selection (Auto, Chinese, English, Japanese, etc.)
- `QWEN_DEVICE` - cuda:0 (GPU) or cpu
- `QWEN_DTYPE` - bfloat16 (default), float16, float32, int8, int4
- `QWEN_FLASH_ATTENTION` - true/false
- `QWEN_DEBUG` - true/false

<h3 align="center"> <pre>   Chatterbox Turbo   </pre> </h3>

- `CHATTERBOX_DEVICE` - cuda:0 (GPU) or cpu
- `CHATTERBOX_SAMPLES_PER_CHUNK` - Audio streaming chunk size (default: 1024)
- `CHATTERBOX_DEBUG` - true/false

Requires `HF_TOKEN` for gated model access.

<h3 align="center"> <pre>  Pocket TTS   </pre> </h3>

- `POCKET_VOICE` - Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
- `POCKET_DEBUG` - true/false

CPU-only, ultra-low latency (~200ms to first audio chunk).

## Home Assistant Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Wyoming Protocol"
3. Add each service separately:
   - **Whisper**: Host = `your-docker-host`, Port = `10300`
   - **Moonshine**: Host = `your-docker-host`, Port = `10302`
   - **Parakeet**: Host = `your-docker-host`, Port = `10303`
   - **Voxtral**: Host = `your-docker-host`, Port = `10301`
   - **Qwen3-TTS**: Host = `your-docker-host`, Port = `10200`
   - **Chatterbox Turbo**: Host = `your-docker-host`, Port = `10201`
   - **Pocket TTS**: Host = `your-docker-host`, Port = `10202`
   - **Kokoro TTS**: Host = `your-docker-host`, Port = `10203`
4. Configure your voice assistant pipeline in **Settings** → **Voice Assistants**

## Resources

### Wyoming & STT
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Moonshine](https://github.com/moonshine-ai/moonshine)
- [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Mistral Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [vLLM](https://docs.vllm.ai/)

### TTS Engines
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Chatterbox Turbo](https://huggingface.co/MycroftAI/Chatterbox-Turbo)
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

### ROCm
- [paralin/ctranslate2-rocm](https://github.com/paralin/ctranslate2-rocm) - ROCm fork used
- [paralin/whisperX-rocm](https://github.com/paralin/whisperX-rocm) - Reference for ROCm setup
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html)

## Licenses

- Whisper: MIT License
- CTranslate2: MIT License
- Wyoming: MIT License
- faster-whisper: MIT License
- Moonshine: MIT License
- Parakeet TDT: Apache 2.0 License
- NVIDIA NeMo: Apache 2.0 License
- Voxtral: Apache 2.0 License
- vLLM: Apache 2.0 License
- Qwen3-TTS: Apache 2.0 License
- Chatterbox Turbo: Apache 2.0 License
- Pocket TTS: Apache 2.0 License
- Kokoro-82M: Apache 2.0 License
