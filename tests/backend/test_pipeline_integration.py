"""Integration tests for core/backend service contracts.

These tests validate how the implemented core and backend pieces work together
before any frontend wiring exists.
"""

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import sounddevice as sd

from src.backend.audio import AudioCaptureService
from src.backend.stt_vosk import SpeechToTextService
from src.backend.translator import TranslatorService
from src.core.config import ConfigManager


EMPTY_STATUS = cast(sd.CallbackFlags, object())


def test_audio_service_output_format_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio capture should produce the 1-D float32 contract expected by STT."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

    service = AudioCaptureService()
    service.start_recording()
    service._audio_callback(
        np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
        frames=3,
        time=None,
        status=EMPTY_STATUS,
    )

    audio = service.stop_recording()

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.array_equal(audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_stt_service_accepts_audio_output_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """STT should accept audio emitted by the audio service contract."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

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

    audio_service = AudioCaptureService()
    audio_service.start_recording()
    audio_service._audio_callback(
        np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
        frames=3,
        time=None,
        status=EMPTY_STATUS,
    )
    audio = audio_service.stop_recording()

    stt_service = SpeechToTextService(str(model_dir))
    text = stt_service.transcribe(audio)

    assert text == "ola mundo"
    assert recognizer.AcceptWaveform.called is True


def test_translator_service_accepts_stt_output_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translator should accept the text contract emitted by STT."""
    translator = MagicMock()
    translator.translate.return_value = "hello world"
    monkeypatch.setattr(
        "src.backend.translator.GoogleTranslator",
        lambda source, target: translator,
    )

    service = TranslatorService("pt", "en")

    translated = service.translate("ola mundo")

    assert translated == "hello world"


def test_backend_pipeline_audio_to_stt_to_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The backend pipeline should connect audio, STT, and translation correctly."""
    stream = MagicMock()
    monkeypatch.setattr("src.backend.audio.sd.InputStream", lambda **_: stream)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_instance = MagicMock()
    recognizer = MagicMock()
    recognizer.FinalResult.return_value = json.dumps({"text": "ola mundo"})
    translator = MagicMock()
    translator.translate.return_value = "hello world"

    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)
    monkeypatch.setattr(
        "src.backend.stt_vosk.KaldiRecognizer",
        lambda model, sample_rate: recognizer,
    )
    monkeypatch.setattr(
        "src.backend.translator.GoogleTranslator",
        lambda source, target: translator,
    )

    audio_service = AudioCaptureService()
    audio_service.start_recording()
    audio_service._audio_callback(
        np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
        frames=4,
        time=None,
        status=EMPTY_STATUS,
    )
    audio = audio_service.stop_recording()

    stt_service = SpeechToTextService(str(model_dir))
    translator_service = TranslatorService("pt", "en")

    text = stt_service.transcribe(audio)
    translated = translator_service.translate(text)

    assert text == "ola mundo"
    assert translated == "hello world"


def test_config_can_drive_backend_service_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ConfigManager values should be consumable when wiring backend services."""
    model_dir = tmp_path / "stt_model"
    model_dir.mkdir()
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "source_language": "pt",
                "target_language": "en",
                "translation_enabled": True,
                "auto_stop_seconds": 8,
                "vosk_model_path": str(model_dir),
            }
        ),
        encoding="utf-8",
    )

    model_instance = MagicMock()
    monkeypatch.setattr("src.backend.stt_vosk.Model", lambda _path: model_instance)

    config_manager = ConfigManager(config_path=config_file)

    stt_service = SpeechToTextService(config_manager.get("vosk_model_path"))
    translator_service = TranslatorService(
        config_manager.get("source_language"),
        config_manager.get("target_language"),
    )
    audio_service = AudioCaptureService()

    assert stt_service.is_ready() is True
    assert translator_service.get_source_language() == "pt"
    assert translator_service.get_target_language() == "en"
    assert audio_service.is_recording() is False
