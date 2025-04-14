"""
core/capture.py - Screen capture and processing thread.
Captures the screen using MSS, applies vision correction, and emits QImage frames.
"""
import cv2, math
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from mss import mss
import time
import ctypes, sys


# Helper functions for validation metrics (used in simulation mode)
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return float('inf')  # identical images -> infinite PSNR
    return 10.0 * math.log10((255.0 ** 2) / mse)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two single-channel images."""
    I1 = img1.astype(np.float32); I2 = img2.astype(np.float32)
    # Constants for SSIM (for 8-bit intensity range)
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    # Gaussian blur (11x11, sigma=1.5) to compute local means
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu1_mu2 = mu1 * mu2
    # Variances and covariance
    sigma1_sq = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
    # SSIM formula
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = t3 / denom
    return float(np.mean(ssim_map))

def compute_feature_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compute enhanced feature preservation metrics between two images."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create pyramid for multi-scale feature detection
    pyramid1 = [gray1]
    pyramid2 = [gray2]
    for _ in range(2):  # 3 levels total
        pyramid1.append(cv2.pyrDown(pyramid1[-1]))
        pyramid2.append(cv2.pyrDown(pyramid2[-1]))
    
    metrics = {'feature_match': 0.0, 'feature_strength': 0.0, 'spatial_distribution': 0.0}
    weights = [0.5, 0.3, 0.2]  # Weights for different scales
    
    # Process each scale
    for level, (img1_scale, img2_scale, weight) in enumerate(zip(pyramid1, pyramid2, weights)):
        # Create SIFT detector with adaptive parameters
        sift = cv2.SIFT_create(
            nfeatures=0,  # No limit on features
            nOctaveLayers=4 if level == 0 else 3,
            contrastThreshold=0.03 if level == 0 else 0.04,
            edgeThreshold=15 if level == 0 else 10,
            sigma=1.6
        )
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(img1_scale, None)
        kp2, des2 = sift.detectAndCompute(img2_scale, None)
        
        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            # Use FLANN matcher with optimized parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
            search_params = dict(checks=100)  # More thorough search
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Get k-nearest matches
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply adaptive ratio test
            good_matches = []
            ratio_threshold = 0.8 - (level * 0.05)  # Stricter at finer scales
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 0:
                # Calculate feature match score for this scale
                scale_match = len(good_matches) / max(len(kp1), len(kp2))
                
                # Calculate feature strength preservation
                response_ratio = np.mean([kp.response for kp in kp1]) / \
                               (np.mean([kp.response for kp in kp2]) + 1e-6)
                scale_strength = min(response_ratio, 1/response_ratio)
                
                # Calculate spatial distribution score
                h, w = img1_scale.shape
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,2)
                
                # Create spatial histograms
                hist1, _, _ = np.histogram2d(pts1[:,0], pts1[:,1], bins=(10,10), 
                                           range=((0,w),(0,h)))
                hist2, _, _ = np.histogram2d(pts2[:,0], pts2[:,1], bins=(10,10), 
                                           range=((0,w),(0,h)))
                
                # Normalize histograms
                hist1 = hist1 / (hist1.sum() + 1e-6)
                hist2 = hist2 / (hist2.sum() + 1e-6)
                
                # Calculate spatial correlation
                spatial_corr = np.corrcoef(hist1.flatten(), hist2.flatten())[0,1]
                if np.isnan(spatial_corr): spatial_corr = 0.0
                
                # Accumulate weighted metrics
                metrics['feature_match'] += scale_match * weight
                metrics['feature_strength'] += scale_strength * weight
                metrics['spatial_distribution'] += spatial_corr * weight
    
    # Add cross-scale matching
    for i in range(len(pyramid1)-1):
        kp1, des1 = sift.detectAndCompute(pyramid1[i], None)
        kp2, des2 = sift.detectAndCompute(pyramid2[i+1], None)
        
        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            ratio_threshold = 0.85  # Slightly more lenient for cross-scale
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 0:
                cross_scale_score = len(good_matches) / max(len(kp1), len(kp2))
                metrics['feature_match'] += cross_scale_score * 0.1  # Small bonus for cross-scale matches
    
    return metrics

class ScreenProcessor(QThread):
    """
    QThread that continuously captures the screen, applies distortion correction,
    and emits the corrected frames and FPS. If simulation mode is enabled, it also
    computes PSNR/SSIM metrics for validation.
    """
    frameReady = pyqtSignal(QImage)      # New corrected frame is ready
    fpsUpdated = pyqtSignal(float)       # FPS update
    psnrUpdated = pyqtSignal(float)      # PSNR of current frame (simulation mode)
    ssimUpdated = pyqtSignal(float)      # SSIM of current frame (simulation mode)

    def __init__(self, distortion_engine, monitor_index=None, parent=None):
        super().__init__(parent)
        self.engine = distortion_engine
        # Auto-detect secondary monitor if available
        self.sct = mss()
        if monitor_index is None:
            self.monitor_index = 2 if len(self.sct.monitors) > 2 else 1
        else:
            self.monitor_index = monitor_index
        self.running = False
        self.simulation_mode = False    # off by default; enabled for validation testing
        self.sct = None                 # will be initialized in run()
        self._last_time = None
        # Set capture region to primary monitor by default
        self.monitor_index = 1  # Primary monitor
        # Cache monitor info
        self.last_frame = None
        self.fps_target = 60  # Target 60 FPS

    def run(self):
        """Main loop: capture screen frames, apply vision correction, and emit results."""
        self.running = True
        try:
            with mss() as sct:
                monitor = sct.monitors[self.monitor_index]
                # Only capture the actual screen area
                self.monitor = {
                    "top": monitor["top"],
                    "left": monitor["left"],
                    "width": monitor["width"],
                    "height": monitor["height"]
                }
                
                while self.running:
                    start_time = time.time()
                    
                    # Capture and process frame
                    frame = np.array(sct.grab(self.monitor))
                    frame = frame[:, :, :3]  # Drop alpha channel
                    
                    # Apply correction - this will create a pre-distorted image
                    # that looks blurry to normal vision but clear through the defect
                    corrected = self.engine.apply_correction(frame)
                    
                    # Always apply pre-distortion that corresponds to the prescription
                    # This ensures that people with normal vision see the blurry pre-corrected image
                    corrected = self.engine.apply_precorrection_blur(corrected)
                    
                    h, w, ch = corrected.shape
                    image = QImage(corrected.data, w, h, 3*w, QImage.Format_BGR888).copy()
                    self.frameReady.emit(image)
                    
                    # Calculate FPS and emit
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self.fpsUpdated.emit(1.0 / elapsed)
                    
                    # Sleep to maintain target FPS
                    sleep_time = max(0, (1.0/self.fps_target) - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
        except Exception as e:
            print(f"Error in ScreenProcessor: {e}")

    def stop(self):
        """Stop the screen capture thread."""
        self.running = False
        self.wait(timeout=1000)  # wait for thread to finish