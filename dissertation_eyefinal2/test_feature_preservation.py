"""
Test script to validate feature preservation in vision correction.
"""
import cv2
import numpy as np
from core.distortion import VisionCorrector
from core.auto_test import VisionCorrectionValidator

def create_test_pattern(width=1920, height=1080):
    """Create a test pattern rich in features."""
    pattern = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add text at different sizes for multi-scale features
    fonts = [0.5, 1.0, 2.0, 3.0]
    texts = ["Vision Test Pattern", "EFPTOZ", "Reading Text", "2 4 6 8"]
    
    y = height // 6
    for size, text in zip(fonts, texts):
        # Add text with border for better feature detection
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
        x = (width - text_size[0]) // 2  # Center text
        
        # Draw text border
        cv2.putText(pattern, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   size, (128, 128, 128), 4)
        # Draw main text
        cv2.putText(pattern, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   size, (0, 0, 0), 2)
        y += int(100 * size)
    
    # Add geometric shapes with gradients for robust features
    # Circle with gradient
    center = (width//4, height*3//4)
    for r in range(50, 0, -1):
        color = int(255 * (1 - r/50))
        cv2.circle(pattern, center, r, (0, 0, color), 2)
    
    # Rectangle with gradient
    rect_center = (width//2, height*3//4)
    for s in range(50, 0, -1):
        color = int(255 * (1 - s/50))
        cv2.rectangle(pattern, 
                     (rect_center[0]-s, rect_center[1]-s),
                     (rect_center[0]+s, rect_center[1]+s), 
                     (0, color, 0), 2)
    
    # Cross pattern with varying thickness
    cross_center = (width*3//4, height*3//4)
    for t in range(5):
        offset = 10 * t
        cv2.line(pattern, 
                (cross_center[0]-50+offset, cross_center[1]-50),
                (cross_center[0]+50-offset, cross_center[1]+50),
                (color, 0, 0), 1+t)
    
    # Add checkerboard pattern with varying sizes
    for size in [20, 10, 5]:
        start_y = height - height//4
        start_x = width - width//4
        for i in range(0, height//4, size):
            for j in range(0, width//4, size):
                if (i + j) // size % 2:
                    y1, y2 = start_y + i, min(start_y + i + size, height)
                    x1, x2 = start_x + j, min(start_x + j + size, width)
                    pattern[y1:y2, x1:x2] = [0, 0, 0]
    
    # Add noise pattern for fine detail preservation
    noise_region = np.random.randint(0, 255, (height//8, width//8, 3), dtype=np.uint8)
    pattern[0:height//8, 0:width//8] = noise_region
    
    return pattern

def main():
    print("Creating test pattern...")
    pattern = create_test_pattern()
    
    print("Initializing vision corrector...")
    engine = VisionCorrector()
    engine.initialize_grid(pattern.shape[1], pattern.shape[0])
    
    print("Setting test prescription...")
    # Test with a challenging prescription
    engine.update_profile({
        "sphere": 2.0,
        "cylinder": -1.0,
        "axis": 45,
        "coma": 0.1,
        "spherical_aberration": 0.1
    })
    
    print("Running validation...")
    validator = VisionCorrectionValidator()
    metrics, conclusion = validator.validate_correction(engine, pattern)
    
    # Save test images for visual inspection
    cv2.imwrite("test_original.png", pattern)
    corrected = engine.apply_correction(pattern)
    cv2.imwrite("test_corrected.png", corrected)
    simulated = engine.simulate_vision_defect(corrected)
    cv2.imwrite("test_simulated.png", simulated)
    
    # Print results
    print("\n" + "="*50)
    print(conclusion)
    print("\nDetailed Metrics:")
    print("\nSimulated Vision Metrics:")
    for key, value in metrics['simulated_metrics'].items():
        print(f"{key}: {value:.3f}")
    
    print("\nDirect Correction Metrics:")
    for key, value in metrics['direct_metrics'].items():
        print(f"{key}: {value:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()

"""
Integration tests for the vision correction system.
Tests feature preservation, distortion correction, and face tracking.
"""
import unittest
import numpy as np
import cv2
from core.distortion import VisionCorrector
from core.auto_test import VisionCorrectionValidator

class TestVisionCorrection(unittest.TestCase):
    def setUp(self):
        self.engine = VisionCorrector()
        self.validator = VisionCorrectionValidator()
        # Create test patterns at multiple resolutions
        self.test_images = {
            'small': self.create_test_pattern((256, 256)),
            'medium': self.create_test_pattern((512, 512)),
            'large': self.create_test_pattern((1024, 1024))
        }
        
    def create_test_pattern(self, size):
        """Generate a test pattern with text and shapes"""
        pattern = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
        
        # Add text
        cv2.putText(pattern, "TEST", (size[0]//4, size[1]//4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
        
        # Add shapes
        cv2.circle(pattern, (size[0]//4, 3*size[1]//4), 50, (0, 0, 255), -1)
        cv2.rectangle(pattern, (size[0]//2-50, 3*size[1]//4-50),
                     (size[0]//2+50, 3*size[1]//4+50), (0, 255, 0), -1)
        
        # Add fine details
        for i in range(0, size[0]//4, 20):
            for j in range(0, size[1]//4, 20):
                if (i + j) // 20 % 2:
                    pattern[i:i+20, j:j+20] = [0, 0, 0]
        
        return pattern
        
    def test_prescription_update(self):
        """Test updating prescription parameters"""
        test_profile = {
            "sphere": -2.0,
            "cylinder": -1.0,
            "axis": 45,
            "coma": 0.1,
            "spherical_aberration": 0.1
        }
        self.engine.update_profile(test_profile)
        self.assertEqual(self.engine.profile["sphere"], -2.0)
        self.assertEqual(self.engine.profile["cylinder"], -1.0)
        
    def test_distortion_simulation(self):
        """Test vision defect simulation"""
        # Set a test prescription
        self.engine.update_profile({
            "sphere": -2.0,
            "cylinder": 0.0,
            "axis": 0,
            "coma": 0.0,
            "spherical_aberration": 0.0
        })
        
        # Apply defect simulation
        defect_view = self.engine.simulate_vision_defect(self.test_image)
        
        # Verify the image is blurred (should have lower variance than original)
        orig_var = np.var(cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY))
        defect_var = np.var(cv2.cvtColor(defect_view, cv2.COLOR_BGR2GRAY))
        self.assertLess(defect_var, orig_var)
        
    def test_correction_quality(self):
        """Test the quality of vision correction with multiple prescriptions"""
        test_prescriptions = [
            {"sphere": -2.0, "cylinder": 0.0, "axis": 0},
            {"sphere": -1.0, "cylinder": -0.5, "axis": 45},
            {"sphere": 0.0, "cylinder": -2.0, "axis": 90}
        ]
        
        for prescription in test_prescriptions:
            self.engine.update_profile(prescription)
            
            # Test with different image sizes
            for size, image in self.test_images.items():
                # Initialize grid for correction
                h, w = image.shape[:2]
                self.engine.initialize_grid(w, h)
                
                # Apply correction
                corrected = self.engine.apply_correction(image)
                simulated = self.engine.simulate_vision_defect(corrected)
                
                # Validate correction
                metrics = self.validator.assess_image_quality(image, simulated)
                
                # Adjust thresholds based on prescription strength
                total_power = abs(prescription['sphere']) + abs(prescription.get('cylinder', 0))
                expected_psnr = max(10.0, 25.0 - total_power * 2)
                expected_feature_match = max(0.2, 0.5 - total_power * 0.1)
                expected_edge_similarity = max(0.2, 0.5 - total_power * 0.1)
                
                # Check quality metrics with prescription-adjusted thresholds
                self.assertGreaterEqual(
                    metrics['psnr'], 
                    expected_psnr,
                    f"PSNR below threshold for {size} image with prescription {prescription}"
                )
                self.assertGreaterEqual(
                    metrics['feature_match'],
                    expected_feature_match,
                    f"Feature matching below threshold for {size} image with prescription {prescription}"
                )
                self.assertGreaterEqual(
                    metrics['edge_similarity'],
                    expected_edge_similarity,
                    f"Edge similarity below threshold for {size} image with prescription {prescription}"
                )
                
                # Verify contrast ratio stays within reasonable bounds
                self.assertGreater(metrics['contrast_ratio'], 0.5)
                self.assertLess(metrics['contrast_ratio'], 2.0)
        
    def test_distance_adaptation(self):
        """Test adaptation to viewing distance changes"""
        # Set initial prescription and distance
        self.engine.update_profile({"sphere": -2.0})
        self.engine.update_distance(60.0)  # 60cm standard distance
        
        # Get correction at standard distance
        h, w = self.test_image.shape[:2]
        self.engine.initialize_grid(w, h)
        standard_correction = self.engine.apply_correction(self.test_image)
        
        # Change distance and get new correction
        self.engine.update_distance(30.0)  # Move closer
        close_correction = self.engine.apply_correction(self.test_image)
        
        # Verify corrections are different (adaptation occurred)
        diff = np.mean(np.abs(standard_correction - close_correction))
        self.assertGreater(diff, 0.0)

if __name__ == '__main__':
    unittest.main()