"""
core/auto_test.py - Automated testing module for vision correction quality.
Uses image quality metrics and feature detection to validate correction effectiveness.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, List

class VisionCorrectionValidator:
    def __init__(self):
        # Initialize SIFT feature detector
        self.feature_detector = cv2.SIFT_create()
        
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio between two images."""
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse < 1e-10:
            return float('inf')
        return 20 * np.log10(255.0) - 10 * np.log10(mse)
        
    def compute_feature_matching(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute feature matching score between two images."""
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.feature_detector.detectAndCompute(gray1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return 0.0
            
        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        # Calculate matching score
        match_ratio = len(good_matches) / max(len(kp1), len(kp2))
        return match_ratio
        
    def compute_edge_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute similarity between edge maps of two images."""
        edges1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 100, 200)
        edges2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 100, 200)
        
        intersection = np.logical_and(edges1, edges2)
        union = np.logical_or(edges1, edges2)
        
        if np.sum(union) == 0:
            return 0.0
        
        return np.sum(intersection) / np.sum(union)
        
    def assess_image_quality(self, original: np.ndarray, corrected: np.ndarray) -> dict:
        """Assess the quality of vision correction using enhanced multi-scale metrics."""
        metrics = {}
        
        # 1. PSNR for overall fidelity
        metrics['psnr'] = self.compute_psnr(original, corrected)
        
        # 2. Enhanced feature matching with adaptive parameters
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale feature detection
        scales = [1.0, 0.75, 0.5]
        feature_scores = []
        
        for scale in scales:
            if scale != 1.0:
                h, w = gray_orig.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_orig = cv2.resize(gray_orig, (new_w, new_h))
                scaled_corr = cv2.resize(gray_corr, (new_w, new_h))
            else:
                scaled_orig = gray_orig
                scaled_corr = gray_corr
            
            # Adaptive SIFT parameters
            sift = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=4,
                contrastThreshold=0.02,
                edgeThreshold=15,
                sigma=1.6
            )
            
            # Detect and compute descriptors
            kp1, des1 = sift.detectAndCompute(scaled_orig, None)
            kp2, des2 = sift.detectAndCompute(scaled_corr, None)
            
            if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                # Use FLANN matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
                search_params = dict(checks=100)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(des1, des2, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)
                
                score = len(good_matches) / max(len(kp1), len(kp2))
                feature_scores.append(score)
            else:
                feature_scores.append(0.0)
        
        # Weighted combination of feature scores
        weights = [0.5, 0.3, 0.2]
        metrics['feature_match'] = sum(s * w for s, w in zip(feature_scores, weights))
        
        # 3. Edge similarity with scale space
        edges_orig = cv2.Canny(gray_orig, 100, 200)
        edges_corr = cv2.Canny(gray_corr, 100, 200)
        
        intersection = np.logical_and(edges_orig, edges_corr)
        union = np.logical_or(edges_orig, edges_corr)
        
        if np.sum(union) > 0:
            metrics['edge_similarity'] = np.sum(intersection) / np.sum(union)
        else:
            metrics['edge_similarity'] = 1.0
        
        # 4. Contrast ratio preservation
        orig_contrast = np.std(gray_orig)
        corr_contrast = np.std(gray_corr)
        metrics['contrast_ratio'] = corr_contrast / (orig_contrast + 1e-6)
        
        return metrics
        
    def validate_correction(self, engine, test_image: np.ndarray) -> Tuple[Dict[str, float], str]:
        """Run a complete validation of the vision correction with enhanced feature preservation."""
        # Generate test cases with multi-scale feature detection
        original = test_image.copy()
        
        # Create pyramid for multi-scale feature analysis
        pyramid_levels = 3
        gaussian_pyr = [original.copy()]
        for i in range(pyramid_levels - 1):
            gaussian_pyr.append(cv2.pyrDown(gaussian_pyr[-1]))
        
        # Process each scale
        metrics_per_scale = []
        for level, img in enumerate(gaussian_pyr):
            # Apply correction pipeline
            simulated_defect = engine.simulate_vision_defect(img)
            corrected = engine.apply_correction(img)
            simulated_corrected = engine.simulate_vision_defect(corrected)
            
            # Compute metrics at this scale
            scale_metrics = self.assess_image_quality(img, simulated_corrected)
            metrics_per_scale.append(scale_metrics)
        
        # Combine metrics with emphasis on finer scales
        weights = [0.5, 0.3, 0.2]  # More weight to finer details
        combined_metrics = {
            'psnr': 0.0,
            'feature_match': 0.0,
            'edge_similarity': 0.0,
            'contrast_ratio': 0.0
        }
        
        for metric_key in combined_metrics.keys():
            combined_metrics[metric_key] = sum(
                metrics[metric_key] * w 
                for metrics, w in zip(metrics_per_scale, weights[:len(metrics_per_scale)])
            )
        
        # Analyze results with enhanced feature detection
        conclusion = self._analyze_results(combined_metrics, combined_metrics)
        
        return {
            'direct_metrics': combined_metrics,
            'simulated_metrics': combined_metrics
        }, conclusion
        
    def _analyze_results(self, direct_metrics: Dict[str, float], 
                        simulated_metrics: Dict[str, float]) -> str:
        """Analyze metrics and provide a conclusion about correction quality."""
        issues = []
        improvements = []
        
        # Analyze PSNR
        if simulated_metrics['psnr'] > 25:
            improvements.append("Excellent image fidelity (PSNR > 25dB)")
        elif simulated_metrics['psnr'] > 20:
            improvements.append("Good image fidelity (PSNR > 20dB)")
        else:
            issues.append(f"Low image fidelity (PSNR: {simulated_metrics['psnr']:.1f}dB)")
        
        # Analyze feature matching
        if simulated_metrics['feature_match'] > 0.7:
            improvements.append("Excellent feature preservation")
        elif simulated_metrics['feature_match'] > 0.5:
            improvements.append("Good feature preservation")
        else:
            issues.append("Poor feature preservation")
        
        # Analyze edge similarity
        if simulated_metrics['edge_similarity'] > 0.6:
            improvements.append("Good edge preservation")
        else:
            issues.append("Poor edge preservation")
        
        # Analyze contrast
        if 0.8 <= simulated_metrics['contrast_ratio'] <= 1.2:
            improvements.append("Good contrast preservation")
        else:
            issues.append("Significant contrast changes")
        
        # Generate conclusion
        conclusion = "Vision Correction Analysis:\n"
        if improvements:
            conclusion += "\nStrengths:\n- " + "\n- ".join(improvements)
        if issues:
            conclusion += "\n\nIssues:\n- " + "\n- ".join(issues)
        
        # Overall assessment
        quality_score = len(improvements) / (len(improvements) + len(issues))
        conclusion += f"\n\nOverall Quality Score: {quality_score:.2%}\n"
        if quality_score > 0.7:
            conclusion += "The correction appears effective."
        elif quality_score > 0.5:
            conclusion += "The correction is moderately effective but could be improved."
        else:
            conclusion += "The correction needs significant improvement."
        
        return conclusion