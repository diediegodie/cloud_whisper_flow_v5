#!/usr/bin/env python3
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import QTimer

app = QApplication([])
label = QLabel("Minimal Qt test: visible window")
label.resize(320, 180)
label.show()

out = Path("investigation_artifacts/minimal_qt_test.png")

def save_and_quit():
    try:
        screen = app.primaryScreen()
        if not screen:
            print("No primary screen available")
        else:
            pix = screen.grabWindow(0)
            ok = pix.save(str(out), "PNG")
            print("Saved:", out, "ok=", ok)
    except Exception as e:
        print("Exception while grabbing screen:", e)
    finally:
        app.quit()

QTimer.singleShot(1000, save_and_quit)
sys.exit(app.exec())
