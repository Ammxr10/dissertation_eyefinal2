"""
gui/main_window.py - Main control GUI for the vision-correcting display system.
Includes spinboxes for prescription input, and buttons for control and profile load/save.
"""
from PyQt5.QtWidgets import (QWidget, QMainWindow, QLabel, QGridLayout, QPushButton, 
                             QDoubleSpinBox, QSpinBox, QFileDialog, QCheckBox, QApplication,
                             QTextEdit)
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from core.auto_test import VisionCorrectionValidator

class MainWindow(QMainWindow):
    """Main application window providing controls for the vision-correcting display."""
    def __init__(self, distortion_engine, face_tracker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vision-Correcting Display - Controls")
        self.distortion_engine = distortion_engine   # VisionCorrector instance
        self.face_tracker = face_tracker             # FaceTracker thread instance
        self.screen_thread = None                    # Will be created when starting correction
        self.overlay_window = None                   # Will be set externally

        # Build UI layout.
        central_widget = QWidget(self)
        layout = QGridLayout(central_widget)
        # Sphere control
        layout.addWidget(QLabel("Sphere (D):"), 0, 0)
        self.sphere_spin = QDoubleSpinBox()
        self.sphere_spin.setRange(-10.0, 10.0)
        self.sphere_spin.setSingleStep(0.25)
        self.sphere_spin.setValue(0.0)
        layout.addWidget(self.sphere_spin, 0, 1)
        # Cylinder control
        layout.addWidget(QLabel("Cylinder (D):"), 1, 0)
        self.cylinder_spin = QDoubleSpinBox()
        self.cylinder_spin.setRange(-10.0, 10.0)
        self.cylinder_spin.setSingleStep(0.25)
        self.cylinder_spin.setValue(0.0)
        layout.addWidget(self.cylinder_spin, 1, 1)
        # Axis control
        layout.addWidget(QLabel("Axis (deg):"), 2, 0)
        self.axis_spin = QSpinBox()
        self.axis_spin.setRange(0, 180)
        self.axis_spin.setValue(0)
        layout.addWidget(self.axis_spin, 2, 1)
        # Coma control
        layout.addWidget(QLabel("Coma (arb):"), 3, 0)
        self.coma_spin = QDoubleSpinBox()
        self.coma_spin.setRange(0.0, 1.0)
        self.coma_spin.setSingleStep(0.1)
        self.coma_spin.setValue(0.0)
        layout.addWidget(self.coma_spin, 3, 1)
        # Spherical aberration control
        layout.addWidget(QLabel("Spherical Aberration (arb):"), 4, 0)
        self.spherical_spin = QDoubleSpinBox()
        self.spherical_spin.setRange(0.0, 1.0)
        self.spherical_spin.setSingleStep(0.1)
        self.spherical_spin.setValue(0.0)
        layout.addWidget(self.spherical_spin, 4, 1)
        # Start/Stop button
        self.start_button = QPushButton("Start Correction")
        layout.addWidget(self.start_button, 5, 0, 1, 2)
        # Save/Load profile buttons
        self.save_button = QPushButton("Save Profile")
        self.load_button = QPushButton("Load Profile")
        layout.addWidget(self.save_button, 6, 0)
        layout.addWidget(self.load_button, 6, 1)
        # Add simulation mode checkbox
        self.sim_checkbox = QCheckBox("Simulation Mode")
        layout.addWidget(self.sim_checkbox, 7, 0, 1, 2)
        
        # Add grid pattern checkbox
        self.grid_checkbox = QCheckBox("Show Test Grid")
        layout.addWidget(self.grid_checkbox, 8, 0, 1, 2)
        
        # Add Test button after grid checkbox
        self.test_button = QPushButton("Test Correction")
        layout.addWidget(self.test_button, 9, 0, 1, 2)
        self.test_button.clicked.connect(self.run_test)
        
        # Add Test Mode button
        self.test_mode_button = QPushButton("Test Vision Correction")
        layout.addWidget(self.test_mode_button, 10, 0, 1, 2)
        self.test_mode_button.clicked.connect(self.start_test_mode)
        
        # Add Automated Test button and results display
        self.auto_test_button = QPushButton("Run Automated Test")
        layout.addWidget(self.auto_test_button, 11, 0, 1, 2)
        self.auto_test_button.clicked.connect(self.run_automated_test)
        
        # Add text area for test results
        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        layout.addWidget(self.test_results, 12, 0, 3, 2)
        
        self.validator = VisionCorrectionValidator()
        
        # Set central widget
        self.setCentralWidget(central_widget)

        # Connect signals except overlay-related ones (those are connected when overlay is set)
        self.start_button.clicked.connect(self.toggle_correction)
        self.save_button.clicked.connect(self.save_profile_dialog)
        self.load_button.clicked.connect(self.load_profile_dialog)
        self.sphere_spin.valueChanged.connect(self.update_profile_from_ui)
        self.cylinder_spin.valueChanged.connect(self.update_profile_from_ui)
        self.axis_spin.valueChanged.connect(self.update_profile_from_ui)
        self.coma_spin.valueChanged.connect(self.update_profile_from_ui)
        self.spherical_spin.valueChanged.connect(self.update_profile_from_ui)
        self.sim_checkbox.toggled.connect(self.toggle_simulation)
        self.grid_checkbox.toggled.connect(self.toggle_grid)
        self.face_tracker.distanceUpdated.connect(self.distortion_engine.update_distance)
        self.face_tracker.angleUpdated.connect(self.distortion_engine.update_angles)
        self.face_tracker.facePositionUpdated.connect(self.distortion_engine.update_face_position)
        self.correction_running = False

    def set_overlay_window(self, overlay):
        """Set the overlay window and connect its signals."""
        self.overlay_window = overlay
        # Connect overlay-specific signals
        self.face_tracker.distanceUpdated.connect(self.overlay_window.update_distance)
        self.face_tracker.angleUpdated.connect(self.overlay_window.update_angle)

    def update_profile_from_ui(self):
        """Update the VisionCorrector with the current UI parameter values."""
        profile = {
            "sphere": self.sphere_spin.value(),
            "cylinder": self.cylinder_spin.value(),
            "axis": self.axis_spin.value(),
            "coma": self.coma_spin.value(),
            "spherical_aberration": self.spherical_spin.value()
        }
        self.distortion_engine.update_profile(profile)

    def toggle_correction(self):
        """Start or stop the real-time correction overlay."""
        if not self.correction_running:
            # Start face tracking if not already running.
            if not self.face_tracker.isRunning():
                self.face_tracker.start()
            from core.capture import ScreenProcessor
            self.screen_thread = ScreenProcessor(self.distortion_engine)
            # Connect screen capture signals to overlay window (set externally)
            self.screen_thread.frameReady.connect(self.overlay_window.update_frame)
            self.screen_thread.fpsUpdated.connect(self.overlay_window.update_fps)
            # Initialize distortion grid with screen dimensions
            screen_geo = QApplication.desktop().screenGeometry()
            self.distortion_engine.initialize_grid(screen_geo.width(), screen_geo.height())
            # Show overlay fullscreen.
            self.overlay_window.setParent(None)
            self.overlay_window.showFullScreen()
            self.screen_thread.start()
            self.start_button.setText("Stop Correction")
            self.correction_running = True
        else:
            if self.screen_thread:
                self.screen_thread.stop()
                self.screen_thread = None
            if self.face_tracker.isRunning():
                self.face_tracker.stop()
            self.overlay_window.hide()
            self.start_button.setText("Start Correction")
            self.correction_running = False

    def toggle_simulation(self, enabled):
        """Enable/disable simulation mode for validation."""
        if self.screen_thread:
            self.screen_thread.simulation_mode = enabled

    def toggle_grid(self, enabled):
        """Toggle the test grid overlay."""
        if hasattr(self, 'overlay_window'):
            self.overlay_window.show_test_pattern = enabled

    def save_profile_dialog(self):
        """Open a dialog to save the current prescription profile to a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        from core.profile import ProfileManager
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Profile", "", "JSON Files (*.json)", options=options)
        if filename:
            mgr = ProfileManager()
            profile = {
                "sphere": self.sphere_spin.value(),
                "cylinder": self.cylinder_spin.value(),
                "axis": self.axis_spin.value(),
                "coma": self.coma_spin.value(),
                "spherical_aberration": self.spherical_spin.value()
            }
            mgr.save_profile(filename, profile)

    def load_profile_dialog(self):
        """Open a dialog to load a prescription profile from a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        from core.profile import ProfileManager
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Profile", "", "JSON Files (*.json)", options=options)
        if filename:
            mgr = ProfileManager()
            profile = mgr.load_profile(filename)
            self.sphere_spin.setValue(profile.get("sphere", 0.0))
            self.cylinder_spin.setValue(profile.get("cylinder", 0.0))
            self.axis_spin.setValue(profile.get("axis", 0))
            self.coma_spin.setValue(profile.get("coma", 0.0))
            self.spherical_spin.setValue(profile.get("spherical_aberration", 0.0))

    def create_test_pattern(self) -> np.ndarray:
        """Generate a test pattern for validation."""
        width = QApplication.desktop().screenGeometry().width()
        height = QApplication.desktop().screenGeometry().height()
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add grid lines
        step = 100
        color = (255, 255, 255)
        thickness = 2
        
        # Vertical lines
        for x in range(0, width, step):
            cv2.line(pattern, (x, 0), (x, height), color, thickness)
            
        # Horizontal lines
        for y in range(0, height, step):
            cv2.line(pattern, (0, y), (width, y), color, thickness)
            
        # Add text patterns at different sizes
        fonts = [0.5, 1.0, 2.0]
        text = "E F P T O Z L P E D"
        y = height // 4
        for size in fonts:
            cv2.putText(pattern, text, (width//4, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
            y += int(50 * size)
            
        return pattern

    def validate_correction(self):
        """Run validation test and show results."""
        test_pattern = self.create_test_pattern()
        defect_view, corrected_view, psnr, ssim = self.distortion_engine.validate_correction(test_pattern)
        
        # Show results in new windows
        cv2.imshow("Original View (Simulated)", defect_view)
        cv2.imshow("Corrected View (Simulated)", corrected_view)
        cv2.waitKey(1)
        
        # Update metrics in overlay
        if self.overlay_window:
            self.overlay_window.update_psnr(psnr)
            self.overlay_window.update_ssim(ssim)

    def run_test(self):
        """Run a visual test to verify correction is working."""
        # Create test image (eye chart style)
        test_image = self.create_test_pattern()
        
        # Get three views of the test pattern
        original = test_image.copy()
        blurred = self.distortion_engine.simulate_vision_defect(test_image)  # How user sees without glasses
        corrected = self.distortion_engine.simulate_vision_defect(  # How user would see our correction
            self.distortion_engine.apply_correction(test_image)
        )
        
        # Show all three images side by side
        cv2.imshow("Original Test Pattern", original)
        cv2.imshow("How You See Without Correction", blurred)
        cv2.imshow("How You Should See With Correction", corrected)
        cv2.waitKey(1)  # Show windows without blocking

    def start_test_mode(self):
        """Run a complete system test."""
        # Create test image with different elements
        test_image = self.create_test_pattern()
        
        # Show test windows in sequence
        cv2.imshow("1. Original Test Pattern", test_image)
        cv2.moveWindow("1. Original Test Pattern", 100, 100)
        
        # Show how it looks with vision defect
        blurred = self.distortion_engine.simulate_vision_defect(test_image)
        cv2.imshow("2. How You See Without Correction", blurred)
        cv2.moveWindow("2. How You See Without Correction", 400, 100)
        
        # Show corrected version
        corrected = self.distortion_engine.apply_correction(test_image)
        cv2.imshow("3. Our Correction", corrected)
        cv2.moveWindow("3. Our Correction", 700, 100)
        
        # Show final result (how corrected image looks through vision defect)
        final = self.distortion_engine.simulate_vision_defect(corrected)
        cv2.imshow("4. How You Should See With Correction", final)
        cv2.moveWindow("4. How You Should See With Correction", 1000, 100)
        
        # Keep windows open
        cv2.waitKey(1)

    def run_automated_test(self):
        """Run automated testing of the vision correction"""
        # Create test pattern with text and shapes
        test_image = self.create_test_pattern()
        
        # Run validation
        metrics, conclusion = self.validator.validate_correction(
            self.distortion_engine, 
            test_image
        )
        
        # Display results
        result_text = conclusion + "\n\nDetailed Metrics:\n"
        result_text += "\nSimulated Vision Metrics:\n"
        for key, value in metrics['simulated_metrics'].items():
            result_text += f"{key}: {value:.3f}\n"
            
        result_text += "\nDirect Correction Metrics:\n"
        for key, value in metrics['direct_metrics'].items():
            result_text += f"{key}: {value:.3f}\n"
            
        self.test_results.setText(result_text)
        
    def create_test_pattern(self):
        """Generate a comprehensive test pattern for validation"""
        width = QApplication.desktop().screenGeometry().width()
        height = QApplication.desktop().screenGeometry().height()
        pattern = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add text at different sizes
        fonts = [0.5, 1.0, 2.0, 3.0]
        texts = ["Vision Test Pattern", "EFPTOZ", "Reading Text", "2 4 6 8"]
        
        y = height // 6
        for size, text in zip(fonts, texts):
            cv2.putText(pattern, text, (width//4, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 2)
            y += int(100 * size)
        
        # Add geometric shapes
        cv2.circle(pattern, (width//4, height*3//4), 50, (0, 0, 255), -1)
        cv2.rectangle(pattern, (width//2-50, height*3//4-50),
                     (width//2+50, height*3//4+50), (0, 255, 0), -1)
        cv2.line(pattern, (width*3//4-50, height*3//4-50),
                (width*3//4+50, height*3//4+50), (255, 0, 0), 5)
        
        # Add fine detail pattern
        checkerboard_size = 20
        for i in range(0, height//4, checkerboard_size):
            for j in range(0, width//4, checkerboard_size):
                if (i + j) // checkerboard_size % 2:
                    pattern[i:i+checkerboard_size, j:j+checkerboard_size] = [0, 0, 0]
        
        return pattern
