"""Compact mode window for CloudWhisper Lite frontend.

Minimal frameless always-on-top window for quick recording without full UI.
Shares the same controller as the main window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Slot, QPoint
from PySide6.QtGui import QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.constants import (
    COMPACT_WINDOW_HEIGHT,
    COMPACT_WINDOW_SPACING,
    COMPACT_WINDOW_TITLE,
    COMPACT_WINDOW_TOP_MARGIN,
    COMPACT_WINDOW_WIDTH,
    UI_BUTTON_COMPACT_RECORD,
    UI_BUTTON_COMPACT_STOP,
    UI_BUTTON_RESTORE_MAIN,
    UI_STATUS_READY,
    WINDOW_CONTENT_MARGIN,
)

if TYPE_CHECKING:
    from src.frontend.main_window import FrontendController


class CompactWindow(QWidget):
    """Minimal always-on-top recording window.

    Provides a frameless, compact interface with only REC/STOP and RESTORE
    buttons. Shares controller state with the main window.
    """

    def __init__(self, controller: FrontendController, main_window: QWidget) -> None:
        super().__init__()
        self._controller = controller
        self._main_window = main_window
        self._drag_offset = QPoint()
        self._drag_active = False

        self.setWindowTitle(COMPACT_WINDOW_TITLE)
        self.setObjectName("compactWindow")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )
        self.resize(COMPACT_WINDOW_WIDTH, COMPACT_WINDOW_HEIGHT)
        self._apply_styling()
        self._center_on_screen()

        self._build_ui()
        self._bind_signals()

    def _apply_styling(self) -> None:
        """Apply custom stylesheet for professional appearance."""
        stylesheet = """
            QWidget#compactWindow {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 6px;
            }
            QWidget#compactWindow QLabel {
                color: #ffffff;
                padding: 2px;
            }
            QLabel#compactHeader {
                background-color: #242424;
                border: 1px solid #444444;
                border-radius: 6px;
                font-weight: bold;
                padding: 8px;
            }
            QLabel#compactStatus {
                font-weight: bold;
                padding: 4px;
            }
            QWidget#compactWindow QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 10px;
                font-weight: bold;
            }
            QWidget#compactWindow QPushButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #666666;
            }
            QWidget#compactWindow QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QWidget#compactWindow QPushButton:disabled {
                color: #888888;
                background-color: #1d1d1d;
            }
        """
        self.setStyleSheet(stylesheet)

    def _center_on_screen(self) -> None:
        """Center window on primary screen with some margin from top."""
        from PySide6.QtGui import QGuiApplication
        
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            x = screen_geometry.center().x() - self.width() // 2
            y = max(COMPACT_WINDOW_TOP_MARGIN, screen_geometry.top() + COMPACT_WINDOW_TOP_MARGIN)
            self.move(x, y)

    def _build_ui(self) -> None:
        """Build the compact UI with minimal controls."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
            WINDOW_CONTENT_MARGIN,
        )
        root_layout.setSpacing(COMPACT_WINDOW_SPACING)

        self.header_label = QLabel(COMPACT_WINDOW_TITLE, self)
        self.header_label.setObjectName("compactHeader")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.header_label)

        self.status_label = QLabel(UI_STATUS_READY, self)
        self.status_label.setObjectName("compactStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.status_label)

        controls_layout = QHBoxLayout()
        self.rec_button = QPushButton(UI_BUTTON_COMPACT_RECORD, self)
        self.stop_button = QPushButton(UI_BUTTON_COMPACT_STOP, self)
        controls_layout.addWidget(self.rec_button)
        controls_layout.addWidget(self.stop_button)
        root_layout.addLayout(controls_layout)

        self.restore_button = QPushButton(UI_BUTTON_RESTORE_MAIN, self)
        root_layout.addWidget(self.restore_button)

        self.setLayout(root_layout)

    def _bind_signals(self) -> None:
        """Connect UI signals to controller slots."""
        self.rec_button.clicked.connect(self._controller.start_recording)
        self.stop_button.clicked.connect(self._controller.stop_recording)
        self.restore_button.clicked.connect(self._restore_main_window)

        # Subscribe to controller state changes
        self._controller.state_changed.connect(self._render_state)
        self._controller.status_changed.connect(self._set_status)

    @Slot(object)
    def _render_state(self, state: object) -> None:
        """Update button states based on controller state."""
        from src.core.view_state import ViewState

        if state == ViewState.IDLE or state == ViewState.ERROR:
            self.rec_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        elif state == ViewState.RECORDING:
            self.rec_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        elif state == ViewState.PROCESSING:
            self.rec_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    @Slot(str)
    def _set_status(self, text: str) -> None:
        """Update the status label."""
        self.status_label.setText(text)

    @Slot()
    def _restore_main_window(self) -> None:
        """Close compact mode and restore the main window."""
        self.hide()
        self._main_window.setWindowState(Qt.WindowState.WindowNoState)
        self._main_window.showNormal()
        self._main_window.activateWindow()
        self._main_window.raise_()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Track mouse press for window dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Handle window dragging on mouse move."""
        if self._drag_active and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Stop dragging on mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
        super().mouseReleaseEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Hide instead of closing."""
        self.hide()
        self._restore_main_window()
        event.ignore()
