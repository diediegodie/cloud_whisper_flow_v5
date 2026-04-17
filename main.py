"""Entry point for CloudWhisper Lite application.

Initializes PySide6 application, services, and main window.
"""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.backend.audio import AudioCaptureService
from src.backend.stt_vosk import SpeechToTextService
from src.backend.translator import TranslatorService
from src.core.config import get_config
from src.core.constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    CONFIG_KEY_SOURCE_LANGUAGE,
    CONFIG_KEY_TARGET_LANGUAGE,
    CONFIG_KEY_VOSK_MODEL_PATH,
    DEFAULT_AUTO_STOP_SECONDS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_VOSK_MODEL_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run the CloudWhisper Lite application."""
    logger.info("Creating QApplication...")
    app = QApplication(sys.argv)

    # Import MainWindow and FrontendController AFTER QApplication is created
    from src.frontend.main_window import FrontendController, MainWindow

    logger.info("Initializing CloudWhisper Lite...")
    config = get_config()

    # Initialize backend services
    logger.info("Initializing backend services...")
    audio_service = AudioCaptureService(
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
    )

    try:
        vosk_model_path = config.get(
            CONFIG_KEY_VOSK_MODEL_PATH,
            DEFAULT_VOSK_MODEL_PATH,
        )
        stt_service = SpeechToTextService(
            model_path=str(vosk_model_path),
            sample_rate=AUDIO_SAMPLE_RATE,
        )
    except Exception as exc:
        logger.error("Failed to initialize STT service: %s", exc)
        logger.warning("Application will continue with STT disabled")
        stt_service = None

    source_language = config.get(CONFIG_KEY_SOURCE_LANGUAGE, DEFAULT_SOURCE_LANGUAGE)
    target_language = config.get(CONFIG_KEY_TARGET_LANGUAGE, DEFAULT_TARGET_LANGUAGE)
    translator_service = TranslatorService(
        source_language=str(source_language),
        target_language=str(target_language),
    )

    # Create controller
    logger.info("Creating frontend controller...")
    controller = FrontendController(
        config=config,
        audio_service=audio_service,
        stt_service=stt_service,
        translator_service=translator_service,
    )

    # Create and show main window
    logger.info("Creating main window...")
    main_window = MainWindow(controller=controller)
    main_window.show()
    logger.info("CloudWhisper Lite started successfully")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
