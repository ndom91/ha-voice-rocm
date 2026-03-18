"""Wyoming protocol event handler for Kokoro TTS."""

import logging
import time

import httpx
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class KokoroEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol events."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        api_url: str,
        voice: str,
        speed: float = 1.0,
        api_timeout: float = 30.0,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.api_url = api_url.rstrip("/")
        self.voice = voice
        self.speed = speed
        self.api_timeout = api_timeout

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent info")
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.info("Synthesizing: %s", synthesize.text)

            sample_rate = 24000  # Kokoro default sample rate
            sent_audio_start = False

            try:
                endpoint = f"{self.api_url}/audio/speech"
                payload = {
                    "model": "kokoro",
                    "voice": self.voice,
                    "input": synthesize.text,
                    "response_format": "pcm",
                    "speed": self.speed,
                }

                _LOGGER.debug("Calling Kokoro API (streaming): %s", endpoint)
                start_time = time.time()

                # Send AudioStart before streaming so HA can begin playback
                await self.write_event(
                    AudioStart(
                        rate=sample_rate,
                        width=2,  # 16-bit = 2 bytes
                        channels=1,  # mono
                    ).event()
                )
                sent_audio_start = True

                # Stream raw PCM chunks from Kokoro-FastAPI as they arrive
                total_bytes = 0
                async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                    async with client.stream(
                        "POST", endpoint, json=payload
                    ) as response:
                        response.raise_for_status()
                        async for chunk_bytes in response.aiter_bytes(
                            chunk_size=2048
                        ):
                            if not chunk_bytes:
                                continue
                            total_bytes += len(chunk_bytes)
                            await self.write_event(
                                AudioChunk(
                                    audio=chunk_bytes,
                                    rate=sample_rate,
                                    width=2,
                                    channels=1,
                                ).event()
                            )

                elapsed = time.time() - start_time
                total_samples = total_bytes // 2
                audio_duration = total_samples / sample_rate if sample_rate else 0
                _LOGGER.info(
                    "Streamed audio: %d samples, %d Hz, %.2f seconds in %.2fs (RTF: %.2fx)",
                    total_samples,
                    sample_rate,
                    audio_duration,
                    elapsed,
                    elapsed / audio_duration if audio_duration > 0 else 0,
                )

            except httpx.HTTPError as e:
                _LOGGER.error("HTTP request failed: %s", e, exc_info=True)
                if not sent_audio_start:
                    await self.write_event(
                        AudioStart(rate=sample_rate, width=2, channels=1).event()
                    )
            except Exception as e:
                _LOGGER.error("Synthesis failed: %s", e, exc_info=True)
                if not sent_audio_start:
                    await self.write_event(
                        AudioStart(rate=sample_rate, width=2, channels=1).event()
                    )
            finally:
                await self.write_event(AudioStop().event())

            return True

        return True

