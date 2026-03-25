"""Application-wide constant values for Cloud Whisper Flow."""

# Application identity shown in the UI.
APP_NAME = "Cloud Whisper Flow"

# Main window title and default size.
MAIN_WINDOW_TITLE = "Cloud Whisper Flow"
MAIN_WINDOW_WIDTH = 960
MAIN_WINDOW_HEIGHT = 640

# Compact window title and default size.
COMPACT_WINDOW_TITLE = "Cloud Whisper Flow Compact"
COMPACT_WINDOW_WIDTH = 360
COMPACT_WINDOW_HEIGHT = 120

# Default language pair for transcription and translation.
DEFAULT_SOURCE_LANGUAGE = "pt"
DEFAULT_TARGET_LANGUAGE = "en"

# Default automatic stop delay for recording sessions, in seconds.
DEFAULT_AUTO_STOP_SECONDS = 8

# Default relative path for the bundled Vosk model directory.
DEFAULT_VOSK_MODEL_PATH = "stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3"
