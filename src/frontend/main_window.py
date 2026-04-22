"""Main window for CloudWhisper Lite frontend.

This module keeps UI rendering in the window class and orchestration in a
controller object. Heavy processing is scaffolded through a QThread worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QTextCursor

logger = logging.getLogger(__name__)

from src.backend.audio import AudioCaptureService
from src.backend.stt_vosk import SpeechToTextService
from src.backend.translator import TranslatorService
from src.core.config import ConfigManager, get_config
from src.core.constants import (
    AUTO_STOP_MAX_SECONDS,
    AUTO_STOP_MIN_SECONDS,
    CONFIG_KEY_AUTO_STOP_SECONDS,
    CONFIG_KEY_SOURCE_LANGUAGE,
    CONFIG_KEY_TARGET_LANGUAGE,
    CONFIG_KEY_TRANSLATION_ENABLED,
    DEFAULT_AUTO_STOP_SECONDS,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_TITLE,
    MAIN_WINDOW_WIDTH,
    SUPPORTED_UI_LANGUAGES,
    UI_BUTTON_CLEAR_TRANSCRIPT,
    UI_BUTTON_START_RECORDING,
    UI_BUTTON_STOP_RECORDING,
    UI_BUTTON_TRANSLATE,
    UI_CHECKBOX_ENABLE_TRANSLATION,
    UI_COLOR_ERROR,
    UI_GROUP_CONTROLS,
    UI_GROUP_OUTPUT,
    UI_GROUP_SETTINGS,
    UI_LABEL_AUTO_STOP_SECONDS,
    UI_LABEL_SOURCE_LANGUAGE,
    UI_LABEL_TARGET_LANGUAGE,
    UI_LABEL_TRANSCRIPT,
    UI_LABEL_TRANSLATION,
    UI_PLACEHOLDER_TRANSCRIPT,
    UI_PLACEHOLDER_TRANSLATION,
    UI_STATUS_DONE,
    UI_STATUS_ERROR,
    UI_STATUS_PROCESSING,
    UI_STATUS_READY,
    UI_STATUS_RECORDING,
    WINDOW_CONTENT_MARGIN,
    WINDOW_SECTION_SPACING,
)
from src.core.view_state import ViewState


@dataclass
class PipelineResult:
    """Data emitted when processing is completed."""

    transcript: str
    translation: str
    emit_transcript: bool = True


class ProcessingWorker(QObject):
    """Worker scaffold for STT + translation pipeline.

    Real backend calls can be incrementally added to `run`.
    """

    finished = Signal(object)  # PipelineResult
    failed = Signal(str)

    def __init__(
        self,
        stt_service: Optional[SpeechToTextService] = None,
        translator_service: Optional[TranslatorService] = None,
        translation_enabled: bool = True,
    ) -> None:
        super().__init__()
        self._stt_service = stt_service
        self._translator_service = translator_service
        self._translation_enabled = translation_enabled
        self._audio_data: Optional[np.ndarray] = None
        self._preset_transcript: str = ""

    def set_audio_data(self, audio_data: Optional[np.ndarray]) -> None:
        """Provide audio payload to be processed by the worker."""
        self._audio_data = audio_data

    def set_transcript(self, text: str) -> None:
        """Provide a pre-existing transcript to translate directly."""
        self._preset_transcript = text

    @Slot()
    def run(self) -> None:
        """Run processing pipeline in worker thread.

        Calls STT service to transcribe audio, then translator service
        if translation is enabled.
        """
        try:
            transcript = self._preset_transcript
            translation = ""

            if (
                not transcript
                and self._stt_service is not None
                and self._audio_data is not None
            ):
                transcript = self._stt_service.transcribe(self._audio_data)

            if self._translation_enabled and transcript:
                if self._translator_service is not None:
                    translation = self._translator_service.translate(transcript)

            self.finished.emit(
                PipelineResult(
                    transcript=transcript,
                    translation=translation,
                    emit_transcript=not bool(self._preset_transcript),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.failed.emit(str(exc))


class FrontendController(QObject):
    """Controller that owns state and coordinates UI-facing flow."""

    state_changed = Signal(object)  # ViewState
    status_changed = Signal(str)
    error_changed = Signal(str)
    transcript_changed = Signal(str)
    translation_changed = Signal(str)

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        audio_service: Optional[AudioCaptureService] = None,
        stt_service: Optional[SpeechToTextService] = None,
        translator_service: Optional[TranslatorService] = None,
    ) -> None:
        super().__init__()
        self._config = config or get_config()
        self._audio_service = audio_service
        self._stt_service = stt_service
        self._translator_service = translator_service

        self._state = ViewState.IDLE
        self._thread: Optional[QThread] = None
        self._worker: Optional[ProcessingWorker] = None

    @property
    def state(self) -> ViewState:
        return self._state

    def _set_state(self, state: ViewState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def load_initial_status(self) -> None:
        self.status_changed.emit(UI_STATUS_READY)
        self.error_changed.emit("")
        self._set_state(ViewState.IDLE)

    def update_source_language(self, language: str) -> None:
        self._config.set(CONFIG_KEY_SOURCE_LANGUAGE, language)
        self._config.save()
        if self._translator_service is not None:
            self._translator_service.set_languages(
                source=language,
                target=self._translator_service.get_target_language(),
            )

    def update_target_language(self, language: str) -> None:
        self._config.set(CONFIG_KEY_TARGET_LANGUAGE, language)
        self._config.save()
        if self._translator_service is not None:
            self._translator_service.set_languages(
                source=self._translator_service.get_source_language(),
                target=language,
            )

    def update_translation_enabled(self, enabled: bool) -> None:
        self._config.set(CONFIG_KEY_TRANSLATION_ENABLED, enabled)
        self._config.save()

    def update_auto_stop_seconds(self, seconds: int) -> None:
        self._config.set(CONFIG_KEY_AUTO_STOP_SECONDS, seconds)
        self._config.save()

    @Slot(str)
    def translate_text(self, text: str) -> None:
        """Translate the given text directly, without recording."""
        if not text.strip():
            return

        self.error_changed.emit("")
        self.status_changed.emit(UI_STATUS_PROCESSING)
        self._set_state(ViewState.PROCESSING)

        self._start_processing_worker_from_text(text)

    def _start_processing_worker_from_text(self, text: str) -> None:
        self._thread = QThread()
        self._worker = ProcessingWorker(
            stt_service=None,
            translator_service=self._translator_service,
            translation_enabled=bool(
                self._config.get(CONFIG_KEY_TRANSLATION_ENABLED, True)
            ),
        )
        self._worker.set_transcript(text)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.failed.connect(self._on_processing_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot()
    def start_recording(self) -> None:
        if self._state not in (ViewState.IDLE, ViewState.ERROR):
            return

        self.error_changed.emit("")
        self.status_changed.emit(UI_STATUS_RECORDING)
        self._set_state(ViewState.RECORDING)

        try:
            if self._audio_service is not None:
                self._audio_service.start_recording()
        except Exception as exc:
            self._handle_error(f"Failed to start recording: {exc}")

    @Slot()
    def stop_recording(self) -> None:
        if self._state != ViewState.RECORDING:
            return

        self.status_changed.emit(UI_STATUS_PROCESSING)
        self._set_state(ViewState.PROCESSING)

        audio_data: Optional[np.ndarray] = None
        try:
            if self._audio_service is not None:
                audio_data = self._audio_service.stop_recording()
        except Exception as exc:
            self._handle_error(f"Failed to stop recording: {exc}")
            return

        self._start_processing_worker(audio_data)

    def _start_processing_worker(self, audio_data: Optional[np.ndarray]) -> None:
        self._thread = QThread()
        self._worker = ProcessingWorker(
            stt_service=self._stt_service,
            translator_service=self._translator_service,
            translation_enabled=bool(
                self._config.get(CONFIG_KEY_TRANSLATION_ENABLED, True)
            ),
        )
        self._worker.set_audio_data(audio_data)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.failed.connect(self._on_processing_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(object)
    def _on_processing_finished(self, result: PipelineResult) -> None:
        if result.emit_transcript:
            self.transcript_changed.emit(result.transcript)
        self.translation_changed.emit(result.translation)
        self.status_changed.emit(UI_STATUS_DONE)
        self.error_changed.emit("")
        self._set_state(ViewState.IDLE)

    @Slot(str)
    def _on_processing_failed(self, error: str) -> None:
        self._handle_error(f"Processing failed: {error}")

    def _handle_error(self, message: str) -> None:
        self.error_changed.emit(message)
        self.status_changed.emit(UI_STATUS_ERROR)
        self._set_state(ViewState.ERROR)


class MainWindow(QMainWindow):
    """Main application window with simple, functional layout."""

    def __init__(self, controller: Optional[FrontendController] = None) -> None:
        super().__init__()
        self._controller = controller or FrontendController()

        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.resize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)

        self._build_ui()
        self._bind_signals()
        self._load_config_values()
        self._controller.load_initial_status()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
        )
        root_layout.setSpacing(WINDOW_SECTION_SPACING)
        root.setLayout(root_layout)

        controls_group = QGroupBox(UI_GROUP_CONTROLS, self)
        controls_layout = QHBoxLayout()
        controls_group.setLayout(controls_layout)

        self.start_button = QPushButton(UI_BUTTON_START_RECORDING, self)
        self.stop_button = QPushButton(UI_BUTTON_STOP_RECORDING, self)
        self.translate_button = QPushButton(UI_BUTTON_TRANSLATE, self)
        self.translation_checkbox = QCheckBox(UI_CHECKBOX_ENABLE_TRANSLATION, self)
        self.translation_checkbox.setChecked(True)

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.translate_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.translation_checkbox)

        settings_group = QGroupBox(UI_GROUP_SETTINGS, self)
        settings_layout = QFormLayout()
        settings_group.setLayout(settings_layout)

        self.source_language_combo = QComboBox(self)
        self.target_language_combo = QComboBox(self)
        self.auto_stop_spinbox = QSpinBox(self)

        self.source_language_combo.addItems(SUPPORTED_UI_LANGUAGES)
        self.target_language_combo.addItems(SUPPORTED_UI_LANGUAGES)
        self.auto_stop_spinbox.setRange(AUTO_STOP_MIN_SECONDS, AUTO_STOP_MAX_SECONDS)

        settings_layout.addRow(UI_LABEL_SOURCE_LANGUAGE, self.source_language_combo)
        settings_layout.addRow(UI_LABEL_TARGET_LANGUAGE, self.target_language_combo)
        settings_layout.addRow(UI_LABEL_AUTO_STOP_SECONDS, self.auto_stop_spinbox)

        content_group = QGroupBox(UI_GROUP_OUTPUT, self)
        content_layout = QVBoxLayout()
        content_group.setLayout(content_layout)

        self.transcript_text = QTextEdit(self)
        self.translation_text = QTextEdit(self)
        self.transcript_text.setReadOnly(False)
        self.translation_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText(UI_PLACEHOLDER_TRANSCRIPT)
        self.translation_text.setPlaceholderText(UI_PLACEHOLDER_TRANSLATION)

        transcript_header = QWidget(self)
        transcript_header_layout = QHBoxLayout(transcript_header)
        transcript_header_layout.setContentsMargins(0, 0, 0, 0)
        transcript_header_layout.addWidget(QLabel(UI_LABEL_TRANSCRIPT, self))
        transcript_header_layout.addStretch(1)
        self.clear_transcript_button = QPushButton(UI_BUTTON_CLEAR_TRANSCRIPT, self)
        transcript_header_layout.addWidget(self.clear_transcript_button)

        content_layout.addWidget(transcript_header)
        content_layout.addWidget(self.transcript_text, 1)
        content_layout.addWidget(QLabel(UI_LABEL_TRANSLATION, self))
        content_layout.addWidget(self.translation_text, 1)

        self.status_label = QLabel(UI_STATUS_READY, self)
        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet(f"color: {UI_COLOR_ERROR};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        root_layout.addWidget(controls_group)
        root_layout.addWidget(settings_group)
        root_layout.addWidget(content_group, 1)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.error_label)

    def _bind_signals(self) -> None:
        self.start_button.clicked.connect(self._controller.start_recording)
        self.stop_button.clicked.connect(self._controller.stop_recording)
        self.translate_button.clicked.connect(
            lambda: self._controller.translate_text(self.transcript_text.toPlainText())
        )

        self.translation_checkbox.toggled.connect(
            self._controller.update_translation_enabled
        )
        self.source_language_combo.currentTextChanged.connect(
            self._controller.update_source_language
        )
        self.target_language_combo.currentTextChanged.connect(
            self._controller.update_target_language
        )
        self.auto_stop_spinbox.valueChanged.connect(
            self._controller.update_auto_stop_seconds
        )

        self._controller.state_changed.connect(self._render_state)
        self._controller.status_changed.connect(self._set_status)
        self._controller.error_changed.connect(self._set_error)
        self.clear_transcript_button.clicked.connect(self.transcript_text.clear)
        self._controller.transcript_changed.connect(self._append_transcript)
        self._controller.translation_changed.connect(self.translation_text.setPlainText)

    def _load_config_values(self) -> None:
        config = get_config()
        blockers = [
            QSignalBlocker(self.translation_checkbox),
            QSignalBlocker(self.source_language_combo),
            QSignalBlocker(self.target_language_combo),
            QSignalBlocker(self.auto_stop_spinbox),
        ]
        self.translation_checkbox.setChecked(
            bool(config.get(CONFIG_KEY_TRANSLATION_ENABLED, True))
        )
        self._set_combo_value(
            self.source_language_combo,
            str(config.get(CONFIG_KEY_SOURCE_LANGUAGE, DEFAULT_SOURCE_LANGUAGE)),
        )
        self._set_combo_value(
            self.target_language_combo,
            str(config.get(CONFIG_KEY_TARGET_LANGUAGE, DEFAULT_TARGET_LANGUAGE)),
        )
        self.auto_stop_spinbox.setValue(
            int(config.get(CONFIG_KEY_AUTO_STOP_SECONDS, DEFAULT_AUTO_STOP_SECONDS))
        )
        del blockers

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @Slot(object)
    def _render_state(self, state: ViewState) -> None:
        if state == ViewState.IDLE:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.translate_button.setEnabled(True)
        elif state == ViewState.RECORDING:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.translate_button.setEnabled(False)
        elif state == ViewState.PROCESSING:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.translate_button.setEnabled(False)
        elif state == ViewState.ERROR:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.translate_button.setEnabled(True)

    @Slot(str)
    def _append_transcript(self, text: str) -> None:
        if not text:
            return
        cursor = self.transcript_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.transcript_text.toPlainText():
            cursor.insertText(text)
        else:
            cursor.insertText("\n" + text)
        self.transcript_text.setTextCursor(cursor)
        self.transcript_text.ensureCursorVisible()

    @Slot(str)
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(str)
    def _set_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
