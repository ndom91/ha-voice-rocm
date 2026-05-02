"""Wyoming protocol server for IBM Granite Speech STT."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import Attribution, Info, AsrProgram, AsrModel
from wyoming.server import AsyncServer

from granite_handler import GraniteEventHandler

_LOGGER = logging.getLogger(__name__)

__version__ = "1.0.0"


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Granite Speech STT Wyoming Protocol Server")
    parser.add_argument("--uri", required=True, help="URI to bind to (e.g., tcp://0.0.0.0:10304)")
    parser.add_argument("--model", default="ibm-granite/granite-speech-4.1-2b-nar", help="Model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _LOGGER.info("Starting Granite Speech STT server")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("URI: %s", args.uri)

    # Granite speech supports multiple languages
    supported_languages = [
        "en",
        "de",
        "es",
        "fr",
        "it",
        "ja",
        "pt",
        "zh",
    ]

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="granite",
                description="IBM Granite Speech - Fast speech recognition",
                attribution=Attribution(
                    name="IBM",
                    url="https://github.com/ibm-granite/granite-speech",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Granite Speech NAR ({args.model})",
                        attribution=Attribution(
                            name="IBM",
                            url="https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar",
                        ),
                        installed=True,
                        languages=supported_languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )

    handler_factory = partial(
        GraniteEventHandler,
        wyoming_info=wyoming_info,
        model_name=args.model,
    )

    _LOGGER.info("Server starting on %s", args.uri)
    server = AsyncServer.from_uri(args.uri)

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    except Exception as e:
        _LOGGER.error("Server error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
