"""
gui/overlay_window.py - Fullscreen overlay window for displaying corrected output.
Draws the corrected frame and overlays debug info (FPS, face distance, prescription parameters)
and an optional test grid for validation.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QFont, QColor

class OverlayWindow(QWidget):
    def __init__(self, distortion_engine, parent=None):
        super().__init__(parent)
        # Configure window to be completely transparent to input while staying on top
        self.setWindowFlags(
            Qt.FramelessWindowHint |  # No window frame
            Qt.WindowStaysOnTopHint | # Always on top
            Qt.Tool |                 # Don't show in taskbar
            Qt.WindowTransparentForInput  # Pass ALL input through
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoChildEventsForParent, True)
        # Make window pass keyboard events too
        self.setAttribute(Qt.WA_X11DoNotAcceptFocus, True)
        
        self.engine = distortion_engine
        self.current_pixmap = QPixmap()
        self.show_debug = True          # Display debug HUD
        self.show_test_pattern = False  # If True, draw test grid
        self.fps = 0.0
        self.distance = 0.0
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.psnr = None
        self.ssim = None
        self.repaint_timer = QTimer(self)
        self.repaint_timer.setInterval(33)  # approximately 30 FPS
        self.repaint_timer.timeout.connect(self.update)
        self.repaint_timer.start()

    def update_frame(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def update_fps(self, fps_value):
        self.fps = fps_value

    def update_distance(self, dist_cm):
        self.distance = dist_cm

    def update_angle(self, angle_x, angle_y):
        self.angle_x = angle_x
        self.angle_y = angle_y

    def update_psnr(self, value):
        self.psnr = value

    def update_ssim(self, value):
        self.ssim = value

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # Add for smoother scaling
        if not self.current_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.current_pixmap.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        # Draw grid overlay if enabled
        if self.show_test_pattern:
            painter.setPen(QColor(255, 0, 0, 128))
            step = 100
            for x in range(0, self.width(), step):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), step):
                painter.drawLine(0, y, self.width(), y)
        # Draw debug information
        if self.show_debug:
            painter.setPen(QColor(0, 255, 0))
            painter.setFont(QFont("Arial", 14))
            lines = [f"FPS: {self.fps:.1f}",
                     f"Distance: {self.distance:.1f} cm",
                     f"Angle: {self.angle_x:.1f}°, {self.angle_y:.1f}°"]
            if self.engine.profile:
                p = self.engine.profile
                lines.append(f"Sphere: {p.get('sphere',0):+0.2f} D")
                lines.append(f"Cyl: {p.get('cylinder',0):+0.2f} D @ {p.get('axis',0)}°")
                lines.append(f"Coma: {p.get('coma',0):0.2f}")
                lines.append(f"Spherical: {p.get('spherical_aberration',0):0.2f}")
            if self.psnr is not None:
                psnr_text = "inf" if self.psnr == float('inf') else f"{self.psnr:.2f}"
                lines.append(f"PSNR: {psnr_text} dB")
            if self.ssim is not None:
                lines.append(f"SSIM: {self.ssim:.3f}")
            margin = 10
            x = margin
            y = self.height() - margin - (len(lines) * 20)
            for line in lines:
                painter.drawText(x, y, line)
                y += 20
        painter.end()
