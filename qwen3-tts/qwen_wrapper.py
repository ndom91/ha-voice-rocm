#!/usr/bin/env python3
"""Wyoming protocol server wrapper for Qwen3-TTS."""

import argparse
import asyncio
import io
import logging
import subprocess
import wave
from functools import partial
from urllib.parse import urlparse

import numpy as np
import torch
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from qwen_handler import QwenEventHandler, get_model

_LOGGER = logging.getLogger(__name__)


def _parse_http_uri(uri: str) -> tuple[str, int]:
    """Parse an HTTP bind URI into a host and port."""
    parsed = urlparse(uri)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(f"OpenAI API URI must be http://host:port: {uri}")
    return parsed.hostname, parsed.port


def _to_pcm(audio_data) -> np.ndarray:
    """Convert generated audio to mono signed 16-bit PCM."""
    if isinstance(audio_data, list):
        audio_data = audio_data[0]
    if torch.is_tensor(audio_data):
        audio_data = audio_data.cpu().numpy()

    audio_data = np.asarray(audio_data, dtype=np.float32).squeeze()
    peak = np.abs(audio_data).max(initial=0.0)
    if peak > 1.0:
        audio_data /= peak
    return (audio_data * 32767).astype(np.int16)


def _wav_bytes(audio_pcm: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono PCM audio as WAV."""
    with io.BytesIO() as output:
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_pcm.tobytes())
        return output.getvalue()


def _encode_audio(audio_pcm: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Encode OpenAI audio response formats, using ffmpeg where needed."""
    if response_format == "pcm":
        return audio_pcm.tobytes(), "audio/pcm"

    wav_data = _wav_bytes(audio_pcm, sample_rate)
    if response_format == "wav":
        return wav_data, "audio/wav"

    encoders = {
        "mp3": (["-f", "mp3"], "audio/mpeg"),
        "opus": (["-c:a", "libopus", "-f", "ogg"], "audio/ogg"),
        "aac": (["-c:a", "aac", "-f", "adts"], "audio/aac"),
        "flac": (["-f", "flac"], "audio/flac"),
    }
    if response_format not in encoders:
        raise ValueError(f"Unsupported response_format: {response_format}")

    ffmpeg_args, content_type = encoders[response_format]
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"],
        input=wav_data,
        capture_output=True,
        check=True,
    )
    return result.stdout, content_type


async def _handle_speech(request, *, model_name, speaker, instruct, language, device, dtype, cache_dir):
    """Serve the OpenAI-compatible POST /v1/audio/speech endpoint."""
    from aiohttp import web

    try:
        payload = await request.json()
    except Exception as err:
        raise web.HTTPBadRequest(text="Request body must be JSON") from err

    text = payload.get("input")
    if not isinstance(text, str) or not text:
        raise web.HTTPBadRequest(text="Missing string 'input' field")

    response_format = payload.get("response_format", "mp3")
    if not isinstance(response_format, str):
        raise web.HTTPBadRequest(text="'response_format' must be a string")

    try:
        model = get_model(model_name, device, dtype, False, cache_dir)
        if "CustomVoice" in model_name:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=payload.get("language", language),
                speaker=payload.get("voice") or speaker,
                instruct=payload.get("instructions", instruct),
            )
        elif "VoiceDesign" in model_name:
            wavs, sample_rate = model.generate_voice_design(
                text=text,
                language=payload.get("language", language),
                instruct=payload.get("instructions", instruct),
            )
        else:
            raise ValueError(f"OpenAI API does not support model: {model_name}")
        audio_data, content_type = _encode_audio(
            _to_pcm(wavs), sample_rate, response_format
        )
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    except Exception as err:
        _LOGGER.error("OpenAI speech synthesis failed: %s", err, exc_info=True)
        raise web.HTTPInternalServerError(text="Speech synthesis failed") from err

    return web.Response(body=audio_data, content_type=content_type)


async def _run_openai_server(uri, *, model_name, speaker, instruct, language, device, dtype, cache_dir):
    """Run the OpenAI-compatible HTTP server alongside Wyoming."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post(
        "/v1/audio/speech",
        partial(
            _handle_speech,
            model_name=model_name,
            speaker=speaker,
            instruct=instruct,
            language=language,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        ),
    )
    runner = web.AppRunner(app)
    await runner.setup()

    host, port = _parse_http_uri(uri)
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    _LOGGER.info("OpenAI-compatible API listening on %s", uri)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Qwen3-TTS Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Model name or path",
    )
    parser.add_argument(
        "--instruct",
        default="Clear, natural voice with medium pitch",
        help="Voice design instruction (for VoiceDesign models) or emotion modifier (for CustomVoice models)",
    )
    parser.add_argument(
        "--speaker",
        default="Ryan",
        help="Speaker name for CustomVoice models (Ryan, Aiden, Vivian, Serena, etc.)",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="TTS language (Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model data type (bfloat16, float16, float32)",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Enable flash attention if available",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples per audio chunk",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache models",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="URI to bind server (e.g., tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--openai-uri",
        help="Optional OpenAI-compatible HTTP bind URI (e.g., http://0.0.0.0:10201)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Starting Wyoming Qwen3-TTS server")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("Device: %s", args.device)
    _LOGGER.info("Voice instruction: %s", args.instruct)
    _LOGGER.info("Language: %s", args.language)

    # Map Qwen language names to ISO language codes for Home Assistant
    language_map = {
        "auto": ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        "chinese": ["zh"],
        "english": ["en"],
        "japanese": ["ja"],
        "korean": ["ko"],
        "german": ["de"],
        "french": ["fr"],
        "russian": ["ru"],
        "portuguese": ["pt"],
        "spanish": ["es"],
        "italian": ["it"],
    }

    # Get language codes for Home Assistant
    ha_languages = language_map.get(args.language.lower(), ["en"])
    _LOGGER.info("Home Assistant language codes: %s", ha_languages)

    # Construct Wyoming protocol info
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen3-tts",
                description="Qwen3-TTS-12Hz VoiceDesign",
                attribution=Attribution(
                    name="Qwen",
                    url="https://github.com/QwenLM/Qwen-TTS",
                ),
                installed=True,
                version="1.7.0",
                voices=[
                    TtsVoice(
                        name="voice_design",
                        description=f"Voice Design: {args.instruct}",
                        attribution=Attribution(
                            name="Qwen",
                            url="https://github.com/QwenLM/Qwen-TTS",
                        ),
                        installed=True,
                        version="1.7.0",
                        languages=ha_languages,
                    )
                ],
            )
        ],
    )

    # Create event handler factory
    handler_factory = partial(
        QwenEventHandler,
        wyoming_info=wyoming_info,
        model_name=args.model,
        voice_instruct=args.instruct,
        language=args.language,
        device=args.device,
        dtype=args.dtype,
        flash_attention=args.flash_attention,
        samples_per_chunk=args.samples_per_chunk,
        cache_dir=args.cache_dir,
        speaker=args.speaker,
    )

    # Start server
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server listening on %s", args.uri)

    try:
        tasks = [asyncio.create_task(server.run(handler_factory))]
        if args.openai_uri:
            tasks.append(
                asyncio.create_task(
                    _run_openai_server(
                        args.openai_uri,
                        model_name=args.model,
                        speaker=args.speaker,
                        instruct=args.instruct,
                        language=args.language,
                        device=args.device,
                        dtype=args.dtype,
                        cache_dir=args.cache_dir,
                    )
                )
            )
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
