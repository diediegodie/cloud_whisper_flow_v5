"""Entry point for CloudWhisper Lite application.

Initializes PySide6 application, services, and main window.
"""

import logging
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt

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
    # If running under classic WSL (no WSLg) and no Wayland env, prefer XCB
    # for better X11 compatibility. If WSLg or Wayland is present, do not
    # force the platform so Qt can pick the correct backend (Wayland/X11).
    try:
        wsl_present = bool(os.getenv("WSL_DISTRO_NAME"))
        wslg_present = Path("/mnt/wslg/versions.json").exists()
    except Exception:
        wsl_present = bool(os.getenv("WSL_DISTRO_NAME"))
        wslg_present = False

    has_wayland_env = bool(os.getenv("WAYLAND_DISPLAY"))

    if (
        wsl_present
        and not wslg_present
        and not has_wayland_env
        and not os.getenv("QT_QPA_PLATFORM")
    ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    logger.info("Effective QT_QPA_PLATFORM=%s", os.getenv("QT_QPA_PLATFORM", "<auto>"))
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

    # Ensure the window is not left maximized/hidden by the compositor and
    # attempt to move it into the primary screen's visible area as a
    # fallback when compositors mis-handle initial mapping.
    try:
        try:
            # Prefer a normal state rather than maximized which some RDP
            # shells mishandle.
            main_window.showNormal()
        except Exception:
            pass

        try:
            primary = app.primaryScreen()
            if primary:
                avail = primary.availableGeometry()
                main_window.move(avail.x() + 50, avail.y() + 50)
            else:
                main_window.move(100, 100)
        except Exception:
            main_window.move(100, 100)
    except Exception:
        pass

    # Defer focus/raise to the event loop to increase the chance the
    # window manager will honor the request (some WMs ignore calls made
    # before the event loop processes pending events).
    def _bring_to_front() -> None:
        try:
            # Try to make the window stay on top as a last-resort visibility
            # aid. Use setWindowFlag when available and re-show the window so
            # the window manager gets the updated flags.
            try:
                main_window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                main_window.show()
            except Exception:
                pass
            main_window.raise_()
            main_window.activateWindow()
            main_window.setFocus()
        except Exception:
            pass

    try:
        QTimer.singleShot(0, _bring_to_front)
    except Exception:
        # Best-effort only; do not fail startup if QTimer isn't available.
        pass

    # Diagnostic: save an internal screenshot of the main window so external
    # smoke tests can detect that the UI rendered (written to repo root).
    def _save_internal_screenshot() -> None:
        try:
            pix = main_window.grab()
            path = Path("cw_internal_screenshot.png")
            pix.save(str(path), "PNG")
            logger.info("Internal screenshot saved to %s", str(path))
        except Exception as err:
            logger.debug("Failed to save internal screenshot: %s", err)

    try:
        QTimer.singleShot(500, _save_internal_screenshot)
    except Exception:
        pass

    logger.info("CloudWhisper Lite started successfully")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
