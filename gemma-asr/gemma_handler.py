"""Wyoming protocol event handler for Gemma audio ASR."""

import io
import logging
import re
import time
import wave
from typing import Any

import httpx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class GemmaEventHandler(AsyncEventHandler):
    """Event handler for Wyoming STT events using a Gemma audio model."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        transcriber: "GemmaTranscriber",
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.transcriber = transcriber
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.audio_width = 2
        self.audio_channels = 1

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            _LOGGER.info("Starting transcription")
            self.audio_buffer = bytearray()
            self.sample_rate = 16000
            self.audio_width = 2
            self.audio_channels = 1
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if not self.audio_buffer:
                self.sample_rate = chunk.rate
                self.audio_width = chunk.width
                self.audio_channels = chunk.channels
                _LOGGER.debug(
                    "Audio format: %d Hz, %d-bit, %d channel(s)",
                    self.sample_rate,
                    self.audio_width * 8,
                    self.audio_channels,
                )

            self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.info(
                "Audio complete, processing transcription (%d bytes)",
                len(self.audio_buffer),
            )

            try:
                start_time = time.time()
                text = await self.transcriber.transcribe_pcm_bytes(
                    bytes(self.audio_buffer),
                    sample_rate=self.sample_rate,
                    sample_width=self.audio_width,
                    channels=self.audio_channels,
                )
                elapsed = time.time() - start_time

                _LOGGER.info("Transcription complete in %.2fs: %s", elapsed, text)
                await self.write_event(Transcript(text=text).event())
            except Exception as err:
                _LOGGER.error("Transcription failed: %s", err, exc_info=True)
                await self.write_event(Transcript(text="").event())

            return True

        return True


class GemmaTranscriber:
    """OpenAI-compatible audio transcription client."""

    def __init__(
        self,
        *,
        api_url: str,
        model: str,
        prompt: str,
        api_key: str | None,
        api_timeout: float,
        temperature: float,
        verify_ssl: bool,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.api_timeout = api_timeout
        self.temperature = temperature
        self.verify_ssl = verify_ssl

    async def transcribe_pcm_bytes(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> str:
        """Transcribe PCM audio bytes from Wyoming audio chunks."""
        if not audio_bytes:
            return ""

        wav_bytes = self._pcm_to_wav(
            audio_bytes,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        endpoint = f"{self.api_url}/audio/transcriptions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        form_data = {
            "model": self.model,
            "prompt": self.prompt,
            "response_format": "json",
            "temperature": str(self.temperature),
        }
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }

        _LOGGER.debug("Calling Gemma API: %s", endpoint)
        async with httpx.AsyncClient(
            timeout=self.api_timeout,
            verify=self.verify_ssl,
        ) as client:
            response = await client.post(
                endpoint,
                data=form_data,
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        return self._clean_text(self._extract_text(data))

    def _pcm_to_wav(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> bytes:
        """Wrap raw PCM bytes in a WAV container for the OpenAI-compatible API."""
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            return wav_io.getvalue()

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract transcript text from an OpenAI-compatible response."""
        if isinstance(data.get("text"), str):
            return data["text"].strip()

        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return "".join(text_parts).strip()

        return str(content).strip()

    def _clean_text(self, text: str) -> str:
        """Remove llama.cpp/Gemma control markers that are not transcript text."""
        text = re.sub(r"<\|channel\>.*?<channel\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|[^>]+\|>", "", text)
        return text.strip()
