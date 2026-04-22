"""Tests for application constants."""

from src.core import constants


def test_constant_names_and_values_are_correct() -> None:
    """Ensure all required constant values match specification."""
    assert constants.APP_NAME == "Cloud Whisper Flow"
    assert constants.MAIN_WINDOW_TITLE == "Cloud Whisper Flow"

    assert constants.MAIN_WINDOW_WIDTH == 960
    assert constants.MAIN_WINDOW_HEIGHT == 640

    assert constants.DEFAULT_SOURCE_LANGUAGE == "pt"
    assert constants.DEFAULT_TARGET_LANGUAGE == "en"
    assert constants.DEFAULT_AUTO_STOP_SECONDS == 8
    assert (
        constants.DEFAULT_VOSK_MODEL_PATH
        == "stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3"
    )


def test_window_dimensions_and_titles_match_product_requirements() -> None:
    """Validate UI title and size constants used by frontend windows."""
    assert constants.MAIN_WINDOW_TITLE == constants.APP_NAME
    assert constants.MAIN_WINDOW_WIDTH > 0
    assert constants.MAIN_WINDOW_HEIGHT > 0


def test_defaults_match_config_expectations() -> None:
    """Validate default language and timing constants."""
    assert constants.DEFAULT_SOURCE_LANGUAGE == "pt"
    assert constants.DEFAULT_TARGET_LANGUAGE == "en"
    assert constants.DEFAULT_AUTO_STOP_SECONDS == 8
