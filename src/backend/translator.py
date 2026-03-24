"""Translation service for Cloud Whisper Flow.

This module provides a small service wrapper around deep-translator's
GoogleTranslator so that translation logic stays isolated from UI code and can
be reused by later pipeline and threading layers.
"""

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when a translation operation fails."""


class TranslatorService:
    """Translate text between two languages using Google Translate.

    The service stores source and target language codes and lazily creates a
    translator instance only when translation is requested.
    """

    def __init__(self, source_language: str, target_language: str) -> None:
        """Initialise the translation service.

        Args:
            source_language: Source language code, such as ``"pt"``.
            target_language: Target language code, such as ``"en"``.

        Raises:
            ValueError: If either language code is empty or only whitespace.
        """
        self._source_language = self._validate_language_code(
            source_language,
            "source_language",
        )
        self._target_language = self._validate_language_code(
            target_language,
            "target_language",
        )
        logger.info(
            "Translator service initialized (source=%s, target=%s).",
            self._source_language,
            self._target_language,
        )

    def translate(self, text: str) -> str:
        """Translate text from the configured source language to target.

        Empty or whitespace-only input is returned unchanged to avoid an
        unnecessary network call.

        Args:
            text: Input text to translate.

        Returns:
            The translated text on success. If translation fails for any
            reason, the original text is returned unchanged.
        """
        if not text or text.isspace():
            return text

        logger.info(
            "Translation requested (source=%s, target=%s).",
            self._source_language,
            self._target_language,
        )

        try:
            translator = GoogleTranslator(
                source=self._source_language,
                target=self._target_language,
            )
            translated_text = translator.translate(text)
            return translated_text if translated_text is not None else text
        except Exception as exc:
            error = TranslationError(f"Translation failed: {exc}")
            logger.error("%s", error)
            return text

    def get_source_language(self) -> str:
        """Return the configured source language code."""
        return self._source_language

    def get_target_language(self) -> str:
        """Return the configured target language code."""
        return self._target_language

    def set_languages(self, source: str, target: str) -> None:
        """Update source and target language codes at runtime.

        Args:
            source: New source language code.
            target: New target language code.

        Raises:
            ValueError: If either language code is empty or only whitespace.
        """
        self._source_language = self._validate_language_code(
            source,
            "source",
        )
        self._target_language = self._validate_language_code(
            target,
            "target",
        )
        logger.info(
            "Translator languages updated (source=%s, target=%s).",
            self._source_language,
            self._target_language,
        )

    @staticmethod
    def _validate_language_code(language_code: str, field_name: str) -> str:
        """Validate a language code and return its normalized value.

        Args:
            language_code: Language code to validate.
            field_name: Name of the field for error reporting.

        Returns:
            The stripped language code.

        Raises:
            ValueError: If the code is empty or only whitespace.
        """
        normalized_code = language_code.strip()
        if not normalized_code:
            raise ValueError(f"{field_name} must be a non-empty language code.")
        return normalized_code
