"""
tracking/face_tracker.py - Face and eye tracking using MediaPipe Face Mesh.
Estimates the user's distance and orientation relative to the camera/screen.
"""
import cv2
import mediapipe as mp
import math
from PyQt5.QtCore import QThread, pyqtSignal

class FaceTracker(QThread):
    """
    A QThread that uses MediaPipe Face Mesh to track face landmarks in real time.
    Emits the estimated distance (in cm) and viewing angles (in degrees).
    """
    distanceUpdated = pyqtSignal(float)          # emits distance in centimeters
    angleUpdated = pyqtSignal(float, float)      # emits (horizontal_angle, vertical_angle) in degrees
    facePositionUpdated = pyqtSignal(float, float, float, float, float)  # x, y, size, cam_w, cam_h

    def __init__(self, camera_index=0, target_ipd_mm=63.0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.target_ipd = target_ipd_mm    # assumed real interpupillary distance in mm for the user
        self.running = False
        # Initialize MediaPipe Face Mesh solution
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Camera capture will be opened in the run() method
        self.cap = None
        # Estimated horizontal field-of-view of the webcam (in degrees) for distance calc
        self.cam_fov_deg = 60.0  # default to 60; can be calibrated if exact FOV is known

    def run(self):
        """Thread loop: capture camera frames and perform face landmark tracking."""
        self.running = True
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("FaceTracker: Unable to access camera.")
                self.running = False
                return
                
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("FaceTracker: Failed to read first frame.")
                self.running = False
                return
                
            # Initialize focal length calculation
            frame_h, frame_w = frame.shape[0:2]
            focal_length_px = frame_w / (2 * math.tan(math.radians(self.cam_fov_deg / 2.0)))
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("FaceTracker: Frame capture failed.")
                    continue
                    
                # Convert to RGB (MediaPipe expects RGB input)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Perform face mesh detection
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    # Take the first face (only one face expected in this application)
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[0:2]
                    # Use iris landmarks for distance if available (face_mesh provides iris in refine_landmarks=True mode)
                    # Iris: left eye landmarks 474-477, right eye 469-472
                    if len(landmarks) > 475:
                        left_iris_x = sum(landmarks[i].x for i in [474,475,476,477]) / 4.0
                        left_iris_y = sum(landmarks[i].y for i in [474,475,476,477]) / 4.0
                        right_iris_x = sum(landmarks[i].x for i in [469,470,471,472]) / 4.0
                        right_iris_y = sum(landmarks[i].y for i in [469,470,471,472]) / 4.0
                    else:
                        # Fallback: use outer eye corners (landmarks 33 for left eye outer corner, 263 for right eye outer corner)
                        right_iris_x = landmarks[33].x;   right_iris_y = landmarks[33].y
                        left_iris_x  = landmarks[263].x;  left_iris_y  = landmarks[263].y
                    # Convert normalized landmark coords (0-1) to pixel coordinates
                    x1 = right_iris_x * w
                    y1 = right_iris_y * h
                    x2 = left_iris_x * w
                    y2 = left_iris_y * h
                    # Calculate interpupillary distance in pixels
                    ipd_px = math.hypot(x2 - x1, y2 - y1)
                    if ipd_px > 0:
                        # Estimate distance: (focal_length_px * actual_IPD_mm) / IPD_px
                        distance_mm = (focal_length_px * self.target_ipd) / ipd_px
                        distance_cm = distance_mm / 10.0
                    else:
                        distance_cm = 0.0
                    # Emit the distance (in cm)
                    self.distanceUpdated.emit(distance_cm)
                    # Estimate viewing angles relative to camera center
                    face_center_x = (x1 + x2) / 2.0
                    face_center_y = (y1 + y2) / 2.0
                    face_size = ipd_px  # use IPD as face size metric
                    self.facePositionUpdated.emit(face_center_x, face_center_y, face_size, w, h)
                    # Offset from center in pixels
                    dx = face_center_x - (w / 2)
                    dy = face_center_y - (h / 2)
                    # Convert pixel offset to angle (radians) via focal length, then to degrees
                    if focal_length_px != 0:
                        angle_x_rad = math.atan(dx / focal_length_px)
                        angle_y_rad = math.atan(dy / focal_length_px)
                    else:
                        angle_x_rad = angle_y_rad = 0.0
                    angle_x_deg = math.degrees(angle_x_rad)
                    angle_y_deg = math.degrees(angle_y_rad)
                    # Emit the angles (in degrees)
                    self.angleUpdated.emit(angle_x_deg, angle_y_deg)
                # (Optional: could add a short sleep here to limit frame rate and CPU usage)
        except Exception as e:
            print(f"FaceTracker: Unexpected error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.face_mesh.close()
            self.running = False

    def stop(self):
        """Stop the tracking thread."""
        self.running = False
        # Wait for the thread to finish (to ensure camera is released properly)
        self.wait(timeout=1000)
