"""Tests for the Vosk speech-to-text service."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.backend.stt_vosk import SpeechToTextError, SpeechToTextService



def test_constructor_validates_missing_model_path() -> None:
    """The service should fail fast when the model directory is missing."""
    with pytest.raises(FileNotFoundError, match="Vosk model not found"):
        SpeechToTextService("/path/that/does/not/exist")



def test_is_ready_reports_successful_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successfully loaded model should mark the service as ready."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()
    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)

    service = SpeechToTextService(str(model_dir))

    assert service.is_ready() is True
    assert service._model is model_instance


def test_constructor_wraps_model_load_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unexpected model-loading failures should raise SpeechToTextError."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def fail_model(_path: str) -> None:
        raise RuntimeError("corrupt model")

    monkeypatch.setattr("src.backend.stt_vosk.Model", fail_model)

    with pytest.raises(SpeechToTextError, match="Failed to load Vosk model"):
        SpeechToTextService(str(model_dir))



def test_transcribe_empty_array_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty audio should short-circuit without invoking recognition."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: MagicMock())

    service = SpeechToTextService(str(model_dir))

    assert service.transcribe(np.array([], dtype=np.float32)) == ""



def test_float32_audio_conversion_to_pcm_bytes_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PCM conversion should clip and encode samples as 16-bit mono bytes."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: MagicMock())

    service = SpeechToTextService(str(model_dir))
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    pcm_bytes = service._audio_to_pcm_bytes(audio)
    pcm_values = np.frombuffer(pcm_bytes, dtype=np.int16)

    assert np.array_equal(pcm_values, np.array([-32767, 0, 32767], dtype=np.int16))



def test_recognizer_chunk_feeding_and_final_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """transcribe() should feed PCM chunks and return the recognizer final text."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()
    recognizer = MagicMock()
    recognizer.FinalResult.return_value = json.dumps({"text": "ola mundo"})

    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)
    monkeypatch.setattr(
        "src.backend.stt_vosk.KaldiRecognizer",
        lambda model, sample_rate: recognizer,
    )

    service = SpeechToTextService(str(model_dir))
    audio = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)

    text = service.transcribe(audio)

    assert text == "ola mundo"
    assert recognizer.AcceptWaveform.call_count >= 2
    first_call = recognizer.AcceptWaveform.call_args_list[0]
    assert isinstance(first_call.args[0], bytes)



def test_transcription_exceptions_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unexpected recognizer failures should raise SpeechToTextError."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()

    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)

    service = SpeechToTextService(str(model_dir))

    def fail_run_recognizer(_pcm_bytes: bytes) -> str:
        raise RuntimeError("recognizer failure")

    monkeypatch.setattr(service, "_run_recognizer", fail_run_recognizer)

    with pytest.raises(SpeechToTextError, match="Transcription failed"):
        service.transcribe(np.array([0.1, 0.2], dtype=np.float32))


def test_accept_waveform_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Recognizer waveform failures should raise SpeechToTextError."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()
    recognizer = MagicMock()
    recognizer.AcceptWaveform.side_effect = RuntimeError("waveform failed")

    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)
    monkeypatch.setattr(
        "src.backend.stt_vosk.KaldiRecognizer",
        lambda model, sample_rate: recognizer,
    )

    service = SpeechToTextService(str(model_dir))

    with pytest.raises(SpeechToTextError, match="Transcription failed"):
        service.transcribe(np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_malformed_final_result_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed recognizer JSON should be wrapped as SpeechToTextError."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()
    recognizer = MagicMock()
    recognizer.FinalResult.return_value = "not valid json"

    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)
    monkeypatch.setattr(
        "src.backend.stt_vosk.KaldiRecognizer",
        lambda model, sample_rate: recognizer,
    )

    service = SpeechToTextService(str(model_dir))

    with pytest.raises(SpeechToTextError, match="Transcription failed"):
        service.transcribe(np.array([0.1, 0.2, 0.3], dtype=np.float32))
