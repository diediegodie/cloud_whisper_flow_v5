#!/usr/bin/env python3
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
import sys
import os

# Ensure repo root is on sys.path so `src` imports work when running tool scripts
sys.path.insert(0, os.path.abspath("."))

from src.core.config import get_config
from src.frontend.main_window import FrontendController, MainWindow
from PySide6.QtCore import Qt

app = QApplication([])
config = get_config()
controller = FrontendController(config=config)
main_window = MainWindow(controller=controller)
main_window.show()
# process events to let window system map
app.processEvents()

screen = app.primaryScreen()
if screen:
    geom = screen.geometry()
    avail = screen.availableGeometry()
    print("primaryScreen.name()=", screen.name())
    print("primaryScreen.geometry()=", geom.x(), geom.y(), geom.width(), geom.height())
    print("primaryScreen.availableGeometry()=", avail.x(), avail.y(), avail.width(), avail.height())
else:
    print("No primary screen")

print("main_window.geometry()=", main_window.geometry().x(), main_window.geometry().y(), main_window.geometry().width(), main_window.geometry().height())
pt = main_window.mapToGlobal(QPoint(0,0))
print("main_window.mapToGlobal(0,0)=", pt.x(), pt.y())
print("windowState=", main_window.windowState())
print("isVisible=", main_window.isVisible(), "isActive=", main_window.isActiveWindow(), "isMinimized=", main_window.isMinimized())

# try a screen grab of root window
try:
    if screen:
        pix = screen.grabWindow(0)
        pix.save('investigation_artifacts/inspect_qt_screen.png', 'PNG')
        print('Saved inspect_qt_screen.png')
except Exception as e:
    print('Failed to grab root window:', e)

print('\n--- Top-level widgets ---')
_widgets = app.topLevelWidgets()
for w in _widgets:
    try:
        geom = w.geometry()
        flags_val = int(w.windowFlags())
        # decode some common Qt window flags to human-readable names
        flag_names = []
        for attr in dir(Qt):
            if not attr[0].isupper():
                continue
            if 'Window' in attr or 'Frameless' in attr or 'Stay' in attr or 'Tool' in attr or 'Dialog' in attr or 'Splash' in attr:
                try:
                    val = getattr(Qt, attr)
                    ival = int(val)
                    if ival != 0 and (flags_val & ival) != 0:
                        flag_names.append(attr)
                except Exception:
                    pass
        print('title=', w.windowTitle(), 'visible=', w.isVisible(), 'active=', w.isActiveWindow(), 'state=', w.windowState(), 'flags=', flags_val, 'flag_names=', flag_names, 'geometry=', geom.x(), geom.y(), geom.width(), geom.height())
    except Exception as e:
        print('error enumerating widget:', e)

app.quit()
