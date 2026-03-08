"""Wyoming protocol event handler for Parakeet TDT STT (NVIDIA NeMo)."""

import asyncio
import logging
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_name: str, device: str):
    """Get or create a cached Parakeet NeMo model instance."""
    cache_key = f"{model_name}:{device}"
    with _model_lock:
        if cache_key not in _model_cache:
            _LOGGER.info("Loading Parakeet model: %s on %s", model_name, device)
            try:
                import nemo.collections.asr as nemo_asr
                import torch

                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

                if device.startswith("cuda") and torch.cuda.is_available():
                    model = model.to(device)
                    _LOGGER.info("Model loaded on GPU: %s", device)
                else:
                    model = model.to("cpu")
                    _LOGGER.info("Model loaded on CPU")

                model.eval()
                _model_cache[cache_key] = model
                _LOGGER.info("Parakeet model loaded successfully")
            except Exception as e:
                _LOGGER.error("Failed to load Parakeet model: %s", e)
                raise

        return _model_cache[cache_key]


class ParakeetEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol STT events using Parakeet TDT."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        transcriber: "ParakeetTranscriber",
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.transcriber = transcriber

        # Audio buffer for accumulating chunks
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.audio_width = 2  # 16-bit
        self.audio_channels = 1  # Mono

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

            except Exception as e:
                _LOGGER.error("Transcription failed: %s", e, exc_info=True)
                await self.write_event(Transcript(text="").event())

            return True

        return True


class ParakeetTranscriber:
    """Shared Parakeet transcription logic for Wyoming and HTTP requests."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    async def transcribe_pcm_bytes(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> str:
        """Transcribe PCM audio bytes from Wyoming audio chunks."""
        if sample_width != 2:
            raise ValueError(f"Unsupported PCM sample width: {sample_width}")

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0

        if channels > 1:
            audio_float = audio_float.reshape(-1, channels).mean(axis=1)

        return await self.transcribe_audio_array(audio_float, sample_rate)

    async def transcribe_upload(
        self, audio_bytes: bytes, filename: str
    ) -> tuple[str, float]:
        """Transcribe an uploaded audio file from the OpenAI-compatible API."""
        suffix = Path(filename or "upload.wav").suffix or ".wav"

        def _load_audio() -> tuple[np.ndarray, float]:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()

                import librosa

                audio_float, _ = librosa.load(tmp.name, sr=16000, mono=True)
                audio_float = audio_float.astype(np.float32)
                duration = len(audio_float) / 16000 if len(audio_float) else 0.0
                return audio_float, duration

        loop = asyncio.get_running_loop()
        audio_float, duration = await loop.run_in_executor(None, _load_audio)
        text = await self.transcribe_audio_array(audio_float, 16000)
        return text, duration

    async def transcribe_audio_array(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> str:
        """Normalize audio and transcribe it asynchronously."""
        audio_float = self._normalize_audio(audio_data, sample_rate)
        _LOGGER.debug("Processing audio: %d samples", len(audio_float))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio_float)

    def _normalize_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert audio to the 16kHz mono float32 format expected by Parakeet."""
        audio_float = np.asarray(audio_data, dtype=np.float32)

        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)

        if sample_rate != 16000:
            import librosa

            audio_float = librosa.resample(
                audio_float,
                orig_sr=sample_rate,
                target_sr=16000,
            )

        return np.asarray(audio_float, dtype=np.float32)

    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription using Parakeet NeMo (runs in thread pool)."""
        try:
            model = get_model(self.model_name, self.device)

            # NeMo transcribe expects file paths, so write audio to a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_data, 16000, subtype="FLOAT")
                output = model.transcribe([tmp.name])

            # NeMo returns different result formats depending on version
            if hasattr(output[0], "text"):
                text = output[0].text.strip()
            elif isinstance(output[0], str):
                text = output[0].strip()
            else:
                text = str(output[0]).strip()

            return text
        except Exception as e:
            _LOGGER.error("Parakeet transcription error: %s", e, exc_info=True)
            raise
