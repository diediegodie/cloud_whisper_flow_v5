"""Audio capture service for Cloud Whisper Flow.

Provides AudioCaptureService, a callback-based microphone capture service
that buffers raw audio chunks and returns a mono float32 numpy array on stop.
Designed for later integration with QThread without any UI dependencies.
"""

import logging
import threading
from typing import List, Optional

import numpy as np
import sounddevice as sd

from src.core.constants import AUDIO_CHANNELS, AUDIO_DTYPE, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioCaptureError(Exception):
    """Raised when an audio capture operation fails."""


class AudioCaptureService:
    """Records microphone input and returns raw audio data for transcription.

    Uses sounddevice.InputStream with an internal callback to accumulate audio
    chunks into a buffer. Calling stop_recording() concatenates the buffer into
    a single mono float32 numpy array.

    Example::

        service = AudioCaptureService(sample_rate=16000)
        service.start_recording()
        # ... user speaks ...
        audio = service.stop_recording()  # np.ndarray, shape (N,), dtype float32
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        dtype: str = AUDIO_DTYPE,
        device: Optional[int] = None,
    ) -> None:
        """Initialise the service with capture parameters.

        Args:
            sample_rate: Sampling rate in Hz (default 16000, required by Vosk).
            channels: Number of input channels (default 1 – mono).
            dtype: NumPy dtype string for sample data (default "float32").
            device: sounddevice device index or None to use the system default.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._device = device

        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Open the input stream and begin buffering audio.

        Raises:
            AudioCaptureError: If already recording or if the microphone
                cannot be opened.
        """
        if self._recording:
            raise AudioCaptureError(
                "start_recording() called while already recording."
            )

        self.clear_buffer()

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
                device=self._device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            logger.info(
                "Recording started (rate=%d Hz, channels=%d, device=%s).",
                self._sample_rate,
                self._channels,
                self._device,
            )
        except sd.PortAudioError as exc:
            self._stream = None
            logger.error("Failed to open audio device: %s", exc)
            raise AudioCaptureError(f"Failed to open audio device: {exc}") from exc

    def stop_recording(self) -> np.ndarray:
        """Stop capturing and return buffered audio as a mono float32 array.

        Returns:
            numpy.ndarray of shape (N,) and dtype float32. Returns an empty
            array of the configured dtype if no audio was captured.

        Raises:
            AudioCaptureError: If not currently recording.
        """
        if not self._recording:
            raise AudioCaptureError(
                "stop_recording() called while not recording."
            )

        self._recording = False
        stream = self._stream
        if stream is not None:
            stream.stop()
            stream.close()
        self._stream = None
        logger.info("Recording stopped.")

        with self._lock:
            chunks = list(self._buffer)

        if not chunks:
            logger.warning("No audio data captured; returning empty array.")
            return np.array([], dtype=self._dtype)

        audio = np.concatenate(chunks, axis=0)

        # Flatten to 1-D mono (handles both (N,) and (N, channels) shapes).
        if audio.ndim > 1:
            audio = audio[:, 0]

        return audio

    def is_recording(self) -> bool:
        """Return True if a recording session is currently active.

        Returns:
            bool: Recording state.
        """
        return self._recording

    def clear_buffer(self) -> None:
        """Discard all buffered audio chunks.

        Safe to call at any time, including between sessions.
        """
        with self._lock:
            self._buffer.clear()
        logger.debug("Audio buffer cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time,         # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice input callback – called from the audio thread.

        Appends a copy of each incoming chunk to the internal buffer so that
        the data remains valid after the callback returns.

        Args:
            indata: Chunk of audio samples, shape (frames, channels).
            frames: Number of frames in this chunk.
            time: CFFI structure with stream timing info (unused).
            status: Flags signalling any over/underflow conditions.
        """
        if status:
            logger.warning("Audio callback status: %s", status)

        with self._lock:
            self._buffer.append(indata.copy())
