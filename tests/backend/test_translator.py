"""Tests for the translation service."""

from unittest.mock import MagicMock

import pytest

from src.backend.translator import TranslatorService



def test_constructor_rejects_empty_language_codes() -> None:
    """TranslatorService should reject empty or whitespace-only language codes."""
    with pytest.raises(ValueError, match="source_language"):
        TranslatorService("", "en")

    with pytest.raises(ValueError, match="target_language"):
        TranslatorService("pt", "   ")



def test_translate_empty_or_whitespace_returns_original_text() -> None:
    """Empty inputs should bypass translator calls and be returned unchanged."""
    service = TranslatorService("pt", "en")

    assert service.translate("") == ""
    assert service.translate("   ") == "   "



def test_successful_translation_returns_translated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful translator call should return the translated text."""
    translator = MagicMock()
    translator.translate.return_value = "hello world"
    monkeypatch.setattr(
        "src.backend.translator.GoogleTranslator",
        lambda source, target: translator,
    )

    service = TranslatorService("pt", "en")

    assert service.translate("ola mundo") == "hello world"



def test_network_error_returns_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translation failures should degrade gracefully to the original text."""
    translator = MagicMock()
    translator.translate.side_effect = RuntimeError("network failure")
    monkeypatch.setattr(
        "src.backend.translator.GoogleTranslator",
        lambda source, target: translator,
    )

    service = TranslatorService("pt", "en")

    assert service.translate("ola mundo") == "ola mundo"


def test_translator_construction_error_returns_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translator initialization failures should also degrade gracefully."""

    def fail_translator(_source: str, _target: str) -> None:
        raise RuntimeError("translator init failure")

    monkeypatch.setattr("src.backend.translator.GoogleTranslator", fail_translator)

    service = TranslatorService("pt", "en")

    assert service.translate("ola mundo") == "ola mundo"



def test_set_languages_updates_source_and_target() -> None:
    """Runtime language updates should replace the stored language codes."""
    service = TranslatorService("pt", "en")

    service.set_languages("en", "es")

    assert service.get_source_language() == "en"
    assert service.get_target_language() == "es"



def test_language_getters_return_current_values() -> None:
    """Source and target language getters should expose current service state."""
    service = TranslatorService("pt", "en")

    assert service.get_source_language() == "pt"
    assert service.get_target_language() == "en"
