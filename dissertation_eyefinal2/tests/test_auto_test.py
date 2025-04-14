"""
Unit tests for the vision correction validation module.
"""
import unittest
import numpy as np
import cv2
from core.auto_test import VisionCorrectionValidator

class MockEngine:
    """Mock vision correction engine for testing"""
    def simulate_vision_defect(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)
        
    def apply_correction(self, img):
        return img

class TestVisionCorrectionValidator(unittest.TestCase):
    def setUp(self):
        self.validator = VisionCorrectionValidator()
        self.engine = MockEngine()
        
    def test_compute_psnr(self):
        # Test identical images
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.assertEqual(self.validator.compute_psnr(img, img), float('inf'))
        
        # Test completely different images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        psnr = self.validator.compute_psnr(img1, img2)
        self.assertLess(psnr, 20)  # Should have low PSNR
        
    def test_compute_feature_matching(self):
        # Create test image with distinct features
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (100, 100), 30, (0, 0, 0), -1)
        
        # Test identical images
        score = self.validator.compute_feature_matching(img, img.copy())
        self.assertGreaterEqual(score, 0.9)  # Should have very high match score
        
        # Test rotated image
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        score = self.validator.compute_feature_matching(img, rotated)
        self.assertGreater(score, 0)  # Should still find some matches
        
    def test_compute_edge_similarity(self):
        # Create test image with distinct edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(img, (25, 25), (75, 75), (255, 255, 255), 2)
        
        # Test identical images
        similarity = self.validator.compute_edge_similarity(img, img.copy())
        self.assertGreaterEqual(similarity, 0.9)
        
        # Test slightly shifted image
        shifted = np.roll(img, 5, axis=1)
        similarity = self.validator.compute_edge_similarity(img, shifted)
        self.assertLess(similarity, 0.9)
        
    def test_assess_image_quality(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        
        # Test with identical images
        metrics = self.validator.assess_image_quality(img, img.copy())
        
        self.assertIn('psnr', metrics)
        self.assertIn('feature_match', metrics)
        self.assertIn('edge_similarity', metrics)
        self.assertIn('contrast_ratio', metrics)
        
        # Verify metric ranges
        self.assertEqual(metrics['psnr'], float('inf'))
        self.assertGreaterEqual(metrics['feature_match'], 0)
        self.assertLessEqual(metrics['feature_match'], 1)
        self.assertGreaterEqual(metrics['edge_similarity'], 0)
        self.assertLessEqual(metrics['edge_similarity'], 1)
        
    def test_validate_correction(self):
        # Create test image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (100, 100), 30, (0, 0, 0), -1)
        
        # Test validation pipeline
        metrics, conclusion = self.validator.validate_correction(self.engine, img)
        
        # Verify metrics structure
        self.assertIn('direct_metrics', metrics)
        self.assertIn('simulated_metrics', metrics)
        
        # Verify conclusion format
        self.assertIn("Vision Correction Analysis:", conclusion)
        self.assertIn("Overall Quality Score:", conclusion)
        
    def test_edge_cases(self):
        # Test empty image
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        metrics = self.validator.assess_image_quality(empty_img, empty_img)
        self.assertEqual(metrics['psnr'], float('inf'))
        
        # Test single color image
        solid_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        metrics = self.validator.assess_image_quality(solid_img, solid_img)
        self.assertEqual(metrics['psnr'], float('inf'))
        
        # Test small image
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        metrics = self.validator.assess_image_quality(small_img, small_img)
        self.assertEqual(metrics['psnr'], float('inf'))

if __name__ == '__main__':
    unittest.main()