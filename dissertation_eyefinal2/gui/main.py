"""
main.py - Entry point for the vision-correcting display application.
Sets up core components, GUI windows, and starts the Qt event loop.
"""
import sys
from PyQt5.QtWidgets import QApplication
from core.distortion import VisionCorrector
from tracking.face_tracker import FaceTracker
from gui.main_window import MainWindow
from gui.overlay_window import OverlayWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Create core vision correction engine.
    distortion_engine = VisionCorrector()
    # Initialize face tracker (using your MediaPipe Face Mesh or Haar cascade implementation).
    face_tracker = FaceTracker()  # assumes default camera index 0.
    # Create the overlay window for displaying corrected output.
    overlay = OverlayWindow(distortion_engine)
    # Create the main control window.
    main_window = MainWindow(distortion_engine, face_tracker)
    # Link the overlay to the main window for control.
    main_window.overlay_window = overlay
    main_window.show()
    # For Windows: attempt to set display affinity to exclude the overlay from capture.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            hwnd = int(overlay.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)  # WDA_EXCLUDEFROMCAPTURE
        except Exception as e:
            print(f"Warning: Unable to set display affinity. Error: {e}")
    sys.exit(app.exec_())
