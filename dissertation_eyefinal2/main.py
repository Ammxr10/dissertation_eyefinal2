"""
main.py - Entry point for the vision-correcting display application.
Initializes the application, creates GUI windows, and starts the event loop.
"""
import sys
from PyQt5.QtWidgets import QApplication
from core.distortion import VisionCorrector
from tracking.face_tracker import FaceTracker
from gui.main_window import MainWindow
from gui.overlay_window import OverlayWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    distortion_engine = VisionCorrector()
    face_tracker = FaceTracker()
    overlay = OverlayWindow(distortion_engine)
    main_window = MainWindow(distortion_engine, face_tracker)
    main_window.set_overlay_window(overlay)  # Use new method to set overlay
    main_window.show()
    # For Windows, attempt to exclude overlay from capture.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            hwnd = int(overlay.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except Exception as e:
            print(f"Warning: Unable to set display affinity. Error: {e}")
    sys.exit(app.exec_())
