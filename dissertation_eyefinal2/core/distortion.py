"""
core/distortion.py - Vision correction engine using Zernike-based distortion.
Provides functionality to generate an inverse PSF from a user's prescription 
and apply real-time image pre-distortion (deconvolution) to compensate for the eye's blur.
"""
import cv2
import numpy as np
import math
import threading

class VisionCorrector:
    """Vision correction engine that applies Zernike-based distortion to images."""
    def __init__(self):
        # Check for CUDA support in OpenCV for potential GPU acceleration.
        self.gpu_enabled = False
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count and count > 0:
                self.gpu_enabled = True
        except Exception:
            self.gpu_enabled = False

        # Initialize coordinate grid and maps
        self.X0 = None
        self.Y0 = None
        self.center_x = None
        self.center_y = None
        self.map_x = None
        self.map_y = None
        self.map_valid = False
        
        # Initialize thread lock and parameters
        self._kernel_lock = threading.Lock()
        self.profile = None
        self.distance_scale = 1.0
        self.nominal_distance = 60.0  # cm (standard reading distance)
        self.current_distance = 60.0  # cm
        self.baseline_face_size = None
        
        # Set default prescription
        self.update_profile({
            "sphere": 0.0, "cylinder": 0.0, "axis": 0,
            "coma": 0.0, "spherical_aberration": 0.0
        })

        self.correction_strength = 1.0  # Adjustable overall strength
        self.pixel_pitch = 0.282  # mm (typical for 27" 1440p display)

    def initialize_grid(self, frame_width: int, frame_height: int):
        """Initialize normalized coordinate grid for Zernike calculations."""
        self.center_x = frame_width / 2.0
        self.center_y = frame_height / 2.0
        cols = np.arange(frame_width, dtype=np.float32)
        rows = np.arange(frame_height, dtype=np.float32)
        X0, Y0 = np.meshgrid(cols, rows)
        self.X0 = (X0 - self.center_x) / min(self.center_x, self.center_y)
        self.Y0 = (Y0 - self.center_y) / min(self.center_x, self.center_y)
        self.map_valid = False

    def update_face_position(self, face_x: float, face_y: float, face_size: float, 
                           cam_width: float, cam_height: float):
        """Update distance scale based on face size only."""
        # Initialize baseline if not set
        if self.baseline_face_size is None:
            self.baseline_face_size = face_size
        
        # Update distance scale based on face size
        if face_size > 0 and self.baseline_face_size > 0:
            self.distance_scale = self.baseline_face_size / face_size
            self.distance_scale = max(0.5, min(2.0, self.distance_scale))
            # Recompute distortion with new distance
            if self.profile:
                self.update_profile(self.profile)

    def compute_wavefront(self):
        """Compute Zernike-based wavefront and its gradients."""
        if self.X0 is None or self.profile is None:
            return None
            
        # Convert to polar coordinates (ρ, θ)
        R = np.sqrt(self.X0**2 + self.Y0**2)
        Theta = np.arctan2(self.Y0, self.X0)
        
        # Initialize wavefront
        W = np.zeros_like(R, dtype=np.float32)
        
        # Get all prescription values first
        sphere = self.profile.get('sphere', 0.0)
        cylinder = self.profile.get('cylinder', 0.0)
        axis_deg = self.profile.get('axis', 0.0)
        coma = self.profile.get('coma', 0.0)
        sph_aber = self.profile.get('spherical_aberration', 0.0)
        
        # Scale aberrations by distance (following 1/distance law)
        # Convert distances to meters for diopter calculation
        nominal_m = self.nominal_distance / 100.0
        current_m = max(0.1, self.current_distance) / 100.0
        distance_scale = nominal_m / current_m  # Closer = stronger, Further = weaker
        
        # Apply distance scaling
        sphere *= distance_scale
        cylinder *= distance_scale
        # Convert axis to radians
        axis_rad = np.deg2rad(axis_deg)
        # Higher-order aberrations scale differently
        hoa_scale = np.sqrt(distance_scale)
        coma *= hoa_scale
        sph_aber *= hoa_scale
        
        # Add Zernike terms
        # Defocus: Z(2,0) = 2r² - 1
        W += sphere * (2 * R**2 - 1)
        
        # Astigmatism: Z(2,±2) = r² * cos(2θ)
        W += cylinder * R**2 * np.cos(2 * (Theta - axis_rad))
        
        # Coma: Z(3,±1) = (3r³ - 2r) * cos(θ)
        W += coma * ((3 * R**3 - 2 * R) * np.cos(Theta))
        
        # Spherical: Z(4,0) = 6r⁴ - 6r² + 1
        W += sph_aber * (6 * R**4 - 6 * R**2 + 1)
        
        return W

    def update_profile(self, profile: dict):
        """Update prescription and invalidate maps."""
        self.profile = profile
        self.map_valid = False

    def compute_remap(self):
        """Compute optical remap based on prescription and viewing parameters."""
        if self.X0 is None or self.profile is None:
            return
            
        # Get prescription in diopters
        sphere = self.profile.get('sphere', 0.0)
        cylinder = self.profile.get('cylinder', 0.0)
        axis = np.deg2rad(self.profile.get('axis', 0.0))
        
        # Enhance correction strength for better visibility
        enhancement_factor = 2.0  # Increase the visual effect
        sphere *= enhancement_factor
        cylinder *= enhancement_factor
        
        # Calculate pixel pitch based on screen DPI
        pixel_pitch_mm = 0.2335  # 25.4/108.79 for 27" 1440p
        
        # Convert distance to meters and apply distance scaling
        view_distance_m = self.current_distance / 100.0
        distance_factor = 60.0 / self.current_distance  # Scale more at closer distances
        
        # Calculate the physical size of each pixel at the viewing distance
        pixel_size_rad = np.arctan2(pixel_pitch_mm/1000.0, view_distance_m)
        
        # Compute ray tracing in physical units
        R = np.sqrt(self.X0**2 + self.Y0**2) * pixel_pitch_mm/1000.0
        Theta = np.arctan2(self.Y0, self.X0)
        
        # Calculate ray deflection using proper optics
        # Using Snell's law and taking into account meridian power variation
        if abs(sphere) > 0.01 or abs(cylinder) > 0.01:
            # Calculate power in each meridian
            meridian_power = sphere + cylinder * np.cos(2 * (Theta - axis))
            
            # Calculate image displacement using physical optics
            # For small angles: displacement ≈ tan(θ) ≈ θ
            # θ = arctan(r/d) where r is radial distance and d is viewing distance
            incident_angle = np.arctan(R / view_distance_m)
            
            # Apply meridian-specific power
            displacement = incident_angle * meridian_power * distance_factor
            
            # Convert displacement to pixels
            pixels_shift = displacement / pixel_size_rad
            
            # Apply radial scaling to enhance effect at periphery
            radial_scale = np.clip(R / (view_distance_m * 0.2), 0, 2.0)
            pixels_shift *= radial_scale
            
            # Convert to x,y components
            dx = pixels_shift * np.cos(Theta)
            dy = pixels_shift * np.sin(Theta)
        else:
            dx = np.zeros_like(self.X0)
            dy = np.zeros_like(self.Y0)
            
        # Create source coordinate maps
        rows, cols = np.mgrid[0:self.X0.shape[0], 0:self.X0.shape[1]]
        self.map_x = (cols + dx).astype(np.float32)
        self.map_y = (rows + dy).astype(np.float32)
        self.map_valid = True

    def apply_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply wavefront-based correction using remap."""
        if frame.ndim != 3:
            return frame
            
        with self._kernel_lock:
            if not self.map_valid:
                self.compute_remap()
            if not hasattr(self, 'map_x') or self.map_x is None:
                return frame
                
            return cv2.remap(frame, self.map_x, self.map_y, 
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def update_distance(self, dist_cm: float):
        """Update viewing distance and recompute kernel."""
        self.current_distance = max(1.0, dist_cm)  # avoid division by zero
        if self.profile:
            self.update_profile(self.profile)

    def update_angles(self, angle_x: float, angle_y: float):
        """Update viewing angles (currently stored for reference)."""
        self.angle_x = angle_x
        self.angle_y = angle_y

    def simulate_vision_defect(self, image: np.ndarray) -> np.ndarray:
        """Simulate how a user with given prescription sees the image."""
        if self.profile is None:
            return image
            
        # Get prescription values but reverse signs to simulate defect
        sphere = -self.profile.get('sphere', 0.0)
        cylinder = -self.profile.get('cylinder', 0.0)
        axis = self.profile.get('axis', 0.0)
        
        # If no defect, return original image
        if abs(sphere) < 0.01 and abs(cylinder) < 0.01:
            return image

        # Calculate physical parameters
        pixel_pitch_mm = 0.2335  # 25.4/108.79 for typical 27" 1440p
        view_distance_mm = self.current_distance * 10  # Convert cm to mm
        
        # Calculate blur size based on optical formulas
        # Blur radius = f * tan(θ) where θ = arctan(h/d)
        # For a lens with power P diopters: f = 1000/P mm
        
        height, width = image.shape[:2]
        
        # Increase the effect size for better visibility
        enhancement = 5.0
        total_power = (abs(sphere) + abs(cylinder)) * enhancement
        
        # Calculate kernel size based on optical physics
        # Base size on maximum blur circle from spherical and cylindrical components
        base_size = int(total_power * 30)  # Scale factor for visibility
        kernel_size = 2 * base_size + 1
        kernel_size = max(31, min(kernel_size, min(height//3, width//3) | 1))  # Must be odd
        
        # Create coordinate grid for kernel
        y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Convert to polar coordinates
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Create astigmatic PSF (Point Spread Function)
        # Power varies with meridian: P(θ) = sphere + cylinder * cos²(θ - axis)
        axis_rad = np.deg2rad(axis)
        
        # Calculate meridian-specific blur
        angle_diff = theta - axis_rad
        meridian_power = sphere + cylinder * np.cos(2 * angle_diff)
        
        # Create elongated PSF
        sigma_r = 1.0 + abs(meridian_power) * 5.0  # Radial sigma
        sigma_t = 1.0 + abs(cylinder) * 5.0        # Tangential sigma
        
        # Calculate 2D Gaussian with different spreads in radial and tangential directions
        x_rot = r * np.cos(angle_diff)
        y_rot = r * np.sin(angle_diff)
        
        psf = np.exp(-0.5 * (x_rot**2 / (sigma_r**2) + y_rot**2 / (sigma_t**2)))
        
        # Normalize PSF
        psf = psf / psf.sum()
        
        # Apply stronger blur for higher prescriptions
        psf = cv2.GaussianBlur(psf, (5, 5), abs(total_power))
        psf = psf / psf.sum()  # Renormalize after blur
        
        # Process each color channel separately with FFT convolution
        result = np.zeros_like(image)
        
        for i in range(3):
            # Pad image to avoid edge effects
            padded = cv2.copyMakeBorder(image[..., i], 
                                      kernel_size//2, kernel_size//2,
                                      kernel_size//2, kernel_size//2,
                                      cv2.BORDER_REFLECT)
            
            # Apply PSF using FFT convolution for better accuracy
            blurred = cv2.filter2D(padded, -1, psf)
            
            # Remove padding
            result[..., i] = blurred[kernel_size//2:kernel_size//2+height,
                                   kernel_size//2:kernel_size//2+width]
        
        return result

    def apply_precorrection_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply a blur that matches the prescription, so normal vision sees the pre-corrected image."""
        if self.profile is None:
            return frame
            
        # Get prescription values
        sphere = self.profile.get('sphere', 0.0)
        cylinder = self.profile.get('cylinder', 0.0)
        axis = self.profile.get('axis', 0.0)
        
        # If no correction needed, return original
        if abs(sphere) < 0.01 and abs(cylinder) < 0.01:
            return frame
            
        # Calculate blur parameters based on prescription
        # Use larger kernel for stronger prescriptions
        enhancement = 8.0  # Make the effect more visible
        total_power = (abs(sphere) + abs(cylinder)) * enhancement
        
        height, width = frame.shape[:2]
        kernel_size = int(total_power * 30)  # Scale factor for visibility
        kernel_size = 2 * (kernel_size // 2) + 1  # Ensure odd size
        kernel_size = max(31, min(kernel_size, min(height//3, width//3) | 1))
        
        # Create coordinate grid for kernel
        y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Convert to polar coordinates
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Calculate astigmatic blur
        axis_rad = np.deg2rad(axis)
        angle_diff = theta - axis_rad
        
        # Create elongated PSF with stronger effect
        sigma_r = 1.0 + abs(sphere) * 8.0        # Radial sigma
        sigma_t = 1.0 + abs(cylinder) * 8.0      # Tangential sigma
        
        # Rotate coordinates
        x_rot = r * np.cos(angle_diff)
        y_rot = r * np.sin(angle_diff)
        
        # Generate PSF
        psf = np.exp(-0.5 * (x_rot**2 / (sigma_r**2) + y_rot**2 / (sigma_t**2)))
        psf = psf / psf.sum()  # Normalize
        
        # Apply additional blur for stronger effect
        psf = cv2.GaussianBlur(psf, (5, 5), total_power)
        psf = psf / psf.sum()  # Renormalize
        
        # Apply the blur to each color channel
        result = np.zeros_like(frame)
        for i in range(3):
            # Pad image
            padded = cv2.copyMakeBorder(frame[..., i], 
                                      kernel_size//2, kernel_size//2,
                                      kernel_size//2, kernel_size//2,
                                      cv2.BORDER_REFLECT)
            
            # Apply PSF
            blurred = cv2.filter2D(padded, -1, psf)
            
            # Remove padding
            result[..., i] = blurred[kernel_size//2:kernel_size//2+height,
                                   kernel_size//2:kernel_size//2+width]
        
        return result

    def validate_correction(self, test_pattern: np.ndarray) -> tuple:
        """
        Validate correction by:
        1. Simulating how user sees the original pattern
        2. Applying our correction
        3. Simulating how user sees the corrected pattern
        4. Comparing result with original
        Returns: (simulated_uncorrected, simulated_corrected, psnr, ssim)
        """
        # How user sees original
        defect_view = self.simulate_vision_defect(test_pattern)
        
        # Apply our correction
        corrected = self.apply_correction(test_pattern)
        
        # How user would see our correction
        final_view = self.simulate_vision_defect(corrected)
        
        # Calculate quality metrics
        from core.capture import compute_psnr, compute_ssim
        psnr = compute_psnr(test_pattern, final_view)
        
        # Convert to grayscale for SSIM
        gray_orig = cv2.cvtColor(test_pattern, cv2.COLOR_BGR2GRAY)
        gray_final = cv2.cvtColor(final_view, cv2.COLOR_BGR2GRAY)
        ssim = compute_ssim(gray_orig, gray_final)
        
        return defect_view, final_view, psnr, ssim
