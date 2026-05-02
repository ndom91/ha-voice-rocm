"""Wyoming protocol event handler for IBM Granite Speech STT."""

import io
import logging
import threading
import time
import wave
import asyncio

import numpy as np
import torch
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_name: str):
    """Get or create a cached Granite Speech model instance."""
    with _model_lock:
        if model_name not in _model_cache:
            _LOGGER.info("Loading Granite Speech model: %s", model_name)
            try:
                from transformers import AutoModel, AutoFeatureExtractor

                # Load model and feature extractor (Granite uses NAR model, not CTC)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _LOGGER.info("Using device: %s", device)

                feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
                
                # Load with flash_attention_2 if available, use eager otherwise
                # Note: Model designed for flash-attn; eager may have issues
                try:
                    _LOGGER.info("Loading with flash_attention_2")
                    model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2",
                        device_map=device,
                        dtype=torch.bfloat16,
                    ).eval()
                    _LOGGER.info("Successfully loaded with flash_attention_2")
                except Exception as flash_err:
                    _LOGGER.warning("flash_attention_2 not available (%s), trying eager attention", str(flash_err)[:80])
                    try:
                        model = AutoModel.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            attn_implementation="eager",
                            device_map=device,
                            dtype=torch.float32,
                        ).eval()
                        _LOGGER.info("Successfully loaded with eager attention")
                    except Exception as eager_err:
                        _LOGGER.error("Both flash_attention_2 and eager failed: %s", str(eager_err)[:200])
                        raise

                _model_cache[model_name] = {
                    "model": model,
                    "feature_extractor": feature_extractor,
                    "device": device,
                }
                _LOGGER.info("Granite Speech model loaded successfully")
            except Exception as e:
                _LOGGER.error("Failed to load Granite Speech model: %s", e)
                raise

        return _model_cache[model_name]


class GraniteEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol STT events using Granite Speech."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        model_name: str,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.model_name = model_name

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
            _LOGGER.info("Audio complete, processing transcription (%d bytes)", len(self.audio_buffer))

            try:
                # Convert audio buffer to float32 numpy array
                audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Convert stereo to mono if needed
                if self.audio_channels > 1:
                    audio_float = audio_float.reshape(-1, self.audio_channels).mean(axis=1)

                # Resample to 16kHz if needed (Granite expects 16kHz)
                if self.sample_rate != 16000:
                    try:
                        import librosa

                        audio_float = librosa.resample(
                            audio_float,
                            orig_sr=self.sample_rate,
                            target_sr=16000,
                        )
                    except ImportError:
                        _LOGGER.warning(
                            "librosa not available for resampling from %d Hz; "
                            "transcription quality may be degraded",
                            self.sample_rate,
                        )

                _LOGGER.debug("Processing audio: %d samples", len(audio_float))

                start_time = time.time()
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self._transcribe_sync,
                    audio_float,
                )
                elapsed = time.time() - start_time

                _LOGGER.info("Transcription complete in %.2fs: %s", elapsed, text)

                await self.write_event(Transcript(text=text).event())

            except Exception as e:
                _LOGGER.error("Transcription failed: %s", e, exc_info=True)
                await self.write_event(Transcript(text="").event())

            return True

        return True

    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription using Granite Speech (runs in thread pool)."""
        try:
            model_info = get_model(self.model_name)
            model = model_info["model"]
            feature_extractor = model_info["feature_extractor"]
            device = model_info["device"]

            # Extract features from audio
            inputs = feature_extractor([audio_data], device=device)

            # Generate transcription using NAR model
            with torch.no_grad():
                output = model.generate(**inputs)

            # Extract text prediction
            text = output.text_preds[0]

            return text.strip()
        except Exception as e:
            _LOGGER.error("Granite transcription error: %s", e, exc_info=True)
            return ""
