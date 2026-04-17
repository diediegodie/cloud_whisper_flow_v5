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

# Configuration keys.
CONFIG_KEY_VOSK_MODEL_PATH = "vosk_model_path"
CONFIG_KEY_SOURCE_LANGUAGE = "source_language"
CONFIG_KEY_TARGET_LANGUAGE = "target_language"
CONFIG_KEY_TRANSLATION_ENABLED = "translation_enabled"
CONFIG_KEY_AUTO_STOP_SECONDS = "auto_stop_seconds"

# Shared audio and transcription defaults.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "float32"

# Shared layout values for frontend windows.
WINDOW_CONTENT_MARGIN = 12
WINDOW_SECTION_SPACING = 10
COMPACT_WINDOW_SPACING = 8
COMPACT_WINDOW_TOP_MARGIN = 50

# Shared language configuration.
SUPPORTED_UI_LANGUAGES = ["pt", "en", "es"]
AUTO_STOP_MIN_SECONDS = 1
AUTO_STOP_MAX_SECONDS = 30

# Default language pair for transcription and translation.
DEFAULT_SOURCE_LANGUAGE = "pt"
DEFAULT_TARGET_LANGUAGE = "en"

# Default automatic stop delay for recording sessions, in seconds.
DEFAULT_AUTO_STOP_SECONDS = 8

# Main window UI text.
UI_GROUP_CONTROLS = "Controls"
UI_GROUP_SETTINGS = "Settings"
UI_GROUP_OUTPUT = "Output"
UI_BUTTON_START_RECORDING = "Start Recording"
UI_BUTTON_STOP_RECORDING = "Stop Recording"
UI_BUTTON_COMPACT_MODE = "► Compact Mode"
UI_CHECKBOX_ENABLE_TRANSLATION = "Enable Translation"
UI_LABEL_SOURCE_LANGUAGE = "Source Language"
UI_LABEL_TARGET_LANGUAGE = "Target Language"
UI_LABEL_AUTO_STOP_SECONDS = "Auto-stop (s)"
UI_LABEL_TRANSCRIPT = "Transcript"
UI_LABEL_TRANSLATION = "Translation"
UI_PLACEHOLDER_TRANSCRIPT = "Transcript will appear here..."
UI_PLACEHOLDER_TRANSLATION = "Translation will appear here..."

# Compact window UI text.
UI_BUTTON_COMPACT_RECORD = "⏺ Record"
UI_BUTTON_COMPACT_STOP = "⏹ Stop"
UI_BUTTON_RESTORE_MAIN = "↗ Main"

# Shared status text.
UI_STATUS_READY = "Ready"
UI_STATUS_RECORDING = "Recording..."
UI_STATUS_PROCESSING = "Processing..."
UI_STATUS_DONE = "Done"
UI_STATUS_ERROR = "Error"

# Shared frontend styling.
UI_COLOR_ERROR = "#B00020"

# Default relative path for the bundled Vosk model directory.
DEFAULT_VOSK_MODEL_PATH = "stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3"
