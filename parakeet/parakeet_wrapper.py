"""Wyoming protocol server for Parakeet TDT STT (NVIDIA NeMo)."""

import argparse
import asyncio
import logging
from functools import partial
from urllib.parse import urlparse

from aiohttp import web
from wyoming.info import Attribution, Info, AsrProgram, AsrModel
from wyoming.server import AsyncServer

from parakeet_handler import ParakeetEventHandler, ParakeetTranscriber

_LOGGER = logging.getLogger(__name__)

__version__ = "1.0.0"


def _parse_http_uri(uri: str) -> tuple[str, int]:
    """Parse an HTTP bind URI into host and port."""
    parsed = urlparse(uri)
    if parsed.scheme != "http":
        raise ValueError(f"Unsupported OpenAI API URI: {uri}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"OpenAI API URI must include host and port: {uri}")
    return parsed.hostname, parsed.port


def _format_timestamp(seconds: float, *, vtt: bool) -> str:
    """Format a timestamp for subtitle responses."""
    total_millis = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _subtitle_response(text: str, duration: float, *, vtt: bool) -> web.Response:
    """Build an SRT or VTT response from a single transcript segment."""
    end_time = max(duration, 0.001)
    start = _format_timestamp(0.0, vtt=vtt)
    end = _format_timestamp(end_time, vtt=vtt)

    if vtt:
        body = f"WEBVTT\n\n{start} --> {end}\n{text}\n"
        return web.Response(text=body, content_type="text/vtt")

    body = f"1\n{start} --> {end}\n{text}\n"
    return web.Response(text=body, content_type="application/x-subrip")


def _transcription_response(
    text: str,
    duration: float,
    response_format: str,
) -> web.StreamResponse:
    """Build an OpenAI-compatible transcription response."""
    if response_format == "text":
        return web.Response(text=text, content_type="text/plain")
    if response_format == "verbose_json":
        return web.json_response({"text": text, "duration": duration, "segments": []})
    if response_format == "srt":
        return _subtitle_response(text, duration, vtt=False)
    if response_format == "vtt":
        return _subtitle_response(text, duration, vtt=True)
    return web.json_response({"text": text})


async def _handle_transcription(
    request: web.Request,
    transcriber: ParakeetTranscriber,
) -> web.StreamResponse:
    """Handle OpenAI-compatible audio transcription requests."""
    reader = await request.multipart()
    fields = {}
    file_bytes = None
    filename = "audio.wav"

    async for part in reader:
        if part.name == "file":
            filename = part.filename or filename
            file_bytes = await part.read(decode=False)
        else:
            fields[part.name] = await part.text()

    if not file_bytes:
        raise web.HTTPBadRequest(text="Missing multipart 'file' field")

    response_format = fields.get("response_format", "json")
    if response_format not in {"json", "text", "verbose_json", "srt", "vtt"}:
        raise web.HTTPBadRequest(text=f"Unsupported response_format: {response_format}")

    try:
        text, duration = await transcriber.transcribe_upload(file_bytes, filename)
    except Exception as err:
        raise web.HTTPInternalServerError(text="Transcription failed") from err

    return _transcription_response(text, duration, response_format)


async def _run_openai_server(uri: str, transcriber: ParakeetTranscriber) -> None:
    """Run the OpenAI-compatible HTTP server."""
    app = web.Application(client_max_size=25 * 1024**2)
    app.router.add_post(
        "/v1/audio/transcriptions",
        partial(_handle_transcription, transcriber=transcriber),
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
    parser = argparse.ArgumentParser(
        description="Parakeet TDT STT Wyoming Protocol Server"
    )
    parser.add_argument(
        "--uri", required=True, help="URI to bind to (e.g., tcp://0.0.0.0:10303)"
    )
    parser.add_argument(
        "--openai-uri",
        default=None,
        help="Optional OpenAI-compatible HTTP bind URI (e.g., http://0.0.0.0:8080)",
    )
    parser.add_argument(
        "--model", default="nvidia/parakeet-tdt-0.6b-v3", help="HuggingFace model name"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device to run on (cuda:0, cpu)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _LOGGER.info("Starting Parakeet TDT STT server")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("Device: %s", args.device)
    _LOGGER.info("URI: %s", args.uri)
    if args.openai_uri:
        _LOGGER.info("OpenAI API URI: %s", args.openai_uri)

    # Parakeet TDT v3 supports 25 European languages
    supported_languages = [
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    ]

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="parakeet",
                description="Parakeet TDT - NVIDIA NeMo multilingual ASR",
                attribution=Attribution(
                    name="NVIDIA NeMo",
                    url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Parakeet TDT 0.6B v3 ({args.model})",
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                        ),
                        installed=True,
                        languages=supported_languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )

    transcriber = ParakeetTranscriber(model_name=args.model, device=args.device)

    handler_factory = partial(
        ParakeetEventHandler,
        wyoming_info=wyoming_info,
        transcriber=transcriber,
    )

    _LOGGER.info("Server starting on %s", args.uri)
    server = AsyncServer.from_uri(args.uri)

    try:
        tasks = [asyncio.create_task(server.run(handler_factory))]
        if args.openai_uri:
            tasks.append(
                asyncio.create_task(_run_openai_server(args.openai_uri, transcriber))
            )

        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    except Exception as e:
        _LOGGER.error("Server error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
