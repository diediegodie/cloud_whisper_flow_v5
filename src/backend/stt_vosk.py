"""Speech-to-text service for Cloud Whisper Flow.

Provides SpeechToTextService, which loads a Vosk offline model and transcribes
a mono float32 numpy audio array into a text string.  Designed for later
integration with QThread — no UI or translation dependencies.
"""

import io
import json
import logging
import wave
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)

# Number of PCM bytes fed to the recognizer per iteration.
_CHUNK_BYTES = 4096


class SpeechToTextError(Exception):
    """Raised when a transcription operation fails."""


class SpeechToTextService:
    """Transcribes mono float32 audio arrays using a local Vosk model.

    The service loads the model once at construction time and exposes a single
    ``transcribe()`` method that converts a numpy array to PCM bytes, feeds
    them to a fresh KaldiRecognizer, and returns the recognised text.

    Example::

        stt = SpeechToTextService(model_path="stt_model/vosk-model-small-pt-0.3")
        text = stt.transcribe(audio_array)  # audio_array: mono float32 np.ndarray
        print(text)
    """

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        """Load the Vosk model from *model_path*.

        Args:
            model_path: Path to the directory containing the Vosk model files.
            sample_rate: Sample rate of the audio that will be transcribed
                (must match the rate used during recording; default 16000 Hz).

        Raises:
            FileNotFoundError: If *model_path* does not exist on disk.
            SpeechToTextError: If the model fails to load for any other reason.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                "Download a model from https://alphacephei.com/vosk/models and "
                "place it at the configured path."
            )

        self._sample_rate = sample_rate
        self._ready = False

        try:
            self._model = Model(str(resolved))
            self._ready = True
            logger.info("Vosk model loaded from '%s'.", model_path)
        except Exception as exc:
            logger.error("Failed to load Vosk model: %s", exc)
            raise SpeechToTextError(f"Failed to load Vosk model: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 audio array to text.

        The array is converted to 16-bit PCM in-memory and fed to a fresh
        KaldiRecognizer in fixed-size chunks.  A new recognizer is created per
        call so that successive calls do not share state.

        Args:
            audio: 1-D numpy array of shape (N,) with dtype float32 and values
                in the range [-1.0, 1.0].  Must have been captured at the same
                sample rate passed to the constructor.

        Returns:
            Recognised text string, or an empty string if no speech was
            detected or *audio* is empty.

        Raises:
            SpeechToTextError: If transcription fails for any reason.
        """
        if audio.size == 0:
            logger.debug("transcribe() received an empty audio array; returning ''.")
            return ""

        logger.debug("Starting transcription of %d samples.", audio.size)

        try:
            pcm_bytes = self._audio_to_pcm_bytes(audio)
            text = self._run_recognizer(pcm_bytes)
        except SpeechToTextError:
            raise
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise SpeechToTextError(f"Transcription failed: {exc}") from exc

        logger.debug("Transcription complete: '%s'.", text)
        return text

    def is_ready(self) -> bool:
        """Return True if the model and recognizer loaded successfully.

        Returns:
            bool: Model readiness state.
        """
        return self._ready

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_to_pcm_bytes(self, audio: np.ndarray) -> bytes:
        """Convert a float32 numpy array to raw 16-bit PCM bytes.

        The conversion clips values to [-1.0, 1.0] before scaling to avoid
        integer overflow artefacts.

        Args:
            audio: 1-D float32 numpy array.

        Returns:
            Raw PCM bytes suitable for a WAV stream at the configured rate.
        """
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm.tobytes())

        # Return only the raw PCM frames (skip the WAV header).
        return pcm.tobytes()

    def _run_recognizer(self, pcm_bytes: bytes) -> str:
        """Feed PCM bytes to a fresh KaldiRecognizer and return the final text.

        Args:
            pcm_bytes: Raw 16-bit mono PCM audio bytes.

        Returns:
            Recognised text, or empty string if nothing was recognised.
        """
        recognizer = KaldiRecognizer(self._model, self._sample_rate)

        offset = 0
        total = len(pcm_bytes)
        while offset < total:
            chunk = pcm_bytes[offset : offset + _CHUNK_BYTES]
            recognizer.AcceptWaveform(chunk)
            offset += _CHUNK_BYTES

        result = json.loads(recognizer.FinalResult())
        return result.get("text", "").strip()
