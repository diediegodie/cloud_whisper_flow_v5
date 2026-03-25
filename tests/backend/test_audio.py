"""Tests for the audio capture service."""

import threading
import time
from typing import List, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import sounddevice as sd

from src.backend.audio import AudioCaptureError, AudioCaptureService


EMPTY_STATUS = cast(sd.CallbackFlags, object())



def test_start_recording_opens_stream_and_sets_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_recording() should create the input stream and mark recording active."""
    stream = MagicMock()
    created_kwargs = {}

    def fake_input_stream(**kwargs):
        created_kwargs.update(kwargs)
        return stream

    monkeypatch.setattr("src.backend.audio.sd.InputStream", fake_input_stream)

    service = AudioCaptureService()
    service.start_recording()

    assert service.is_recording() is True
    assert service._stream is stream
    assert created_kwargs["samplerate"] == 16000
    assert created_kwargs["channels"] == 1
    assert created_kwargs["dtype"] == "float32"
    assert created_kwargs["device"] is None
    assert created_kwargs["callback"] == service._audio_callback
    stream.start.assert_called_once()


def test_start_recording_wraps_portaudio_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device-open failures should be wrapped as AudioCaptureError."""

    def fail_input_stream(**_kwargs):
        raise sd.PortAudioError("device unavailable")

    monkeypatch.setattr("src.backend.audio.sd.InputStream", fail_input_stream)

    service = AudioCaptureService()

    with pytest.raises(AudioCaptureError, match="Failed to open audio device"):
        service.start_recording()

    assert service.is_recording() is False
    assert service._stream is None



def test_stop_recording_returns_mono_float32_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_recording() should concatenate chunks and flatten mono output."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

    service = AudioCaptureService()
    service.start_recording()
    service._audio_callback(
        np.array([[0.1], [0.2]], dtype=np.float32),
        frames=2,
        time=None,
        status=EMPTY_STATUS,
    )
    service._audio_callback(
        np.array([[0.3], [0.4]], dtype=np.float32),
        frames=2,
        time=None,
        status=EMPTY_STATUS,
    )

    audio = service.stop_recording()

    assert service.is_recording() is False
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.array_equal(audio, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    stream.stop.assert_called_once()
    stream.close.assert_called_once()



def test_stop_without_start_raises_error() -> None:
    """stop_recording() should reject invalid state transitions."""
    service = AudioCaptureService()

    with pytest.raises(AudioCaptureError, match="not recording"):
        service.stop_recording()



def test_start_while_recording_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_recording() should reject a second start while already active."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

    service = AudioCaptureService()
    service.start_recording()

    with pytest.raises(AudioCaptureError, match="already recording"):
        service.start_recording()



def test_empty_capture_returns_empty_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """Immediate stop without captured chunks should return an empty float32 array."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

    service = AudioCaptureService()
    service.start_recording()

    audio = service.stop_recording()

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size == 0



def test_multiple_sessions_do_not_leak_stale_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new recording session should begin with a cleared buffer."""
    streams: List[MagicMock] = [MagicMock(), MagicMock()]

    def fake_input_stream(**_kwargs):
        return streams.pop(0)

    monkeypatch.setattr("src.backend.audio.sd.InputStream", fake_input_stream)

    service = AudioCaptureService()

    service.start_recording()
    service._audio_callback(
        np.array([[0.5], [0.6]], dtype=np.float32),
        frames=2,
        time=None,
        status=EMPTY_STATUS,
    )
    first_audio = service.stop_recording()

    service.start_recording()
    second_audio = service.stop_recording()

    assert np.array_equal(first_audio, np.array([0.5, 0.6], dtype=np.float32))
    assert second_audio.size == 0
    assert second_audio.dtype == np.float32



def test_audio_callback_uses_lock_and_stores_copy() -> None:
    """The callback should use the lock and store a copy of incoming audio."""
    service = AudioCaptureService()
    input_chunk = np.array([[1.0], [2.0]], dtype=np.float32)

    service._lock.acquire()
    callback_thread = threading.Thread(
        target=service._audio_callback,
        kwargs={
            "indata": input_chunk,
            "frames": 2,
            "time": None,
            "status": EMPTY_STATUS,
        },
    )
    callback_thread.start()

    time.sleep(0.05)
    assert len(service._buffer) == 0

    service._lock.release()
    callback_thread.join(timeout=1)
    input_chunk[0, 0] = 99.0

    assert len(service._buffer) == 1
    assert service._buffer[0][0, 0] == np.float32(1.0)
    assert service._buffer[0] is not input_chunk
