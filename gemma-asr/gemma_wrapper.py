#!/usr/bin/env python3
"""Wyoming protocol server wrapper for Gemma audio ASR."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from gemma_handler import GemmaEventHandler, GemmaTranscriber

_LOGGER = logging.getLogger(__name__)

__version__ = "1.0.0"


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemma audio ASR Wyoming server")
    parser.add_argument(
        "--uri",
        required=True,
        help="URI to bind to (e.g., tcp://0.0.0.0:10305)",
    )
    parser.add_argument(
        "--api-url",
        default="https://llama-dash.puff.lan/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="gemma-4-12B-fast",
        help="Model name to send to the OpenAI-compatible API",
    )
    parser.add_argument(
        "--prompt",
        default="Transcribe the spoken audio exactly. Return only the transcript text. If there is no intelligible speech, return an empty transcript.",
        help="Instruction prompt sent with each audio transcription request",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=120.0,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for transcription requests",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable TLS certificate verification for the API request",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _LOGGER.info("Starting Gemma audio ASR server")
    _LOGGER.info("API URL: %s", args.api_url)
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("URI: %s", args.uri)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="gemma-asr",
                description="Gemma audio ASR via OpenAI-compatible API proxy",
                attribution=Attribution(
                    name="unsloth",
                    url="https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Gemma audio ASR ({args.model})",
                        attribution=Attribution(
                            name="unsloth",
                            url="https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF",
                        ),
                        installed=True,
                        languages=["en"],
                        version=__version__,
                    )
                ],
            )
        ],
    )

    transcriber = GemmaTranscriber(
        api_url=args.api_url,
        model=args.model,
        prompt=args.prompt,
        api_key=args.api_key,
        api_timeout=args.api_timeout,
        temperature=args.temperature,
        verify_ssl=not args.no_verify_ssl,
    )

    handler_factory = partial(
        GemmaEventHandler,
        wyoming_info=wyoming_info,
        transcriber=transcriber,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server listening on %s", args.uri)

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    except Exception as err:
        _LOGGER.error("Server error: %s", err, exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
