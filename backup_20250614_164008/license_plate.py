"""
Enhanced License Plate Recognition module with advanced preprocessing.
This is the ORIGINAL license plate reader that your system expects.
"""
import cv2
import numpy as np
import easyocr
import torch
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import logger
except ImportError:
    # Fallback logger if utils not available
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

class EnhancedLicensePlateReader:
    """
    Enhanced license plate reader with multiple preprocessing techniques
    for improved accuracy.
    """
    
    def __init__(self, allowed_chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        """
        Initialize the enhanced license plate reader.
        
        Args:
            allowed_chars: Characters allowed in license plates
        """
        # Initialize EasyOCR with GPU support if available
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.allowed_chars = allowed_chars
        
        # Preprocessing parameters
        self.min_plate_ratio = 2.0  # Minimum aspect ratio for plates
        self.max_plate_ratio = 6.0  # Maximum aspect ratio for plates
        
        logger.info("Enhanced license plate reader initialized")

    def read_plate_from_vehicle(self, vehicle_img, bbox=None):
        """
        Read license plate from a vehicle image with optional bounding box.
        
        Args:
            vehicle_img: Full frame or vehicle crop
            bbox: Optional [x1, y1, x2, y2] of vehicle in the image
            
        Returns:
            tuple: (plate_text, confidence, plate_region) or None
        """
        # If bbox provided, crop to vehicle
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            vehicle_crop = vehicle_img[y1:y2, x1:x2]
        else:
            vehicle_crop = vehicle_img
            
        # Detect plate region
        plate_regions = self._detect_plate_regions(vehicle_crop)
        
        best_result = None
        best_confidence = 0.0
        best_region = None
        
        # Try to read each potential plate region
        for region in plate_regions:
            rx1, ry1, rx2, ry2 = region
            plate_img = vehicle_crop[ry1:ry2, rx1:rx2]
            
            # Skip if too small
            if plate_img.shape[0] < 20 or plate_img.shape[1] < 60:
                continue
                
            # Apply multiple preprocessing methods
            results = []
            
            # Method 1: Standard preprocessing
            processed1 = self._preprocess_standard(plate_img)
            result1 = self._read_with_easyocr(processed1)
            if result1:
                results.append(result1)
            
            # Method 2: Adaptive threshold
            processed2 = self._preprocess_adaptive(plate_img)
            result2 = self._read_with_easyocr(processed2)
            if result2:
                results.append(result2)
            
            # Method 3: Morphological operations
            processed3 = self._preprocess_morphological(plate_img)
            result3 = self._read_with_easyocr(processed3)
            if result3:
                results.append(result3)
            
            # Method 4: Contrast enhancement
            processed4 = self._preprocess_contrast(plate_img)
            result4 = self._read_with_easyocr(processed4)
            if result4:
                results.append(result4)
            
            # Method 5: Original image
            result5 = self._read_with_easyocr(plate_img)
            if result5:
                results.append(result5)
            
            # Select best result
            for text, conf in results:
                if conf > best_confidence and len(text) >= 4:
                    best_result = (text, conf)
                    best_confidence = conf
                    best_region = region
                    
        if best_result:
            return best_result[0], best_result[1], best_region
        return None

    def _detect_plate_regions(self, vehicle_img):
        """
        Detect potential license plate regions in vehicle image.
        
        Returns:
            List of [x1, y1, x2, y2] regions
        """
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = vehicle_img.shape[:2]
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            area = w_rect * h_rect
            if area < 500 or area > (w * h * 0.3):  # Too small or too large
                continue
                
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            if aspect_ratio < self.min_plate_ratio or aspect_ratio > self.max_plate_ratio:
                continue
            
            # Plate typically in lower half of vehicle
            if y < h * 0.3:  # Too high to be a plate
                continue
                
            regions.append([x, y, x + w_rect, y + h_rect])
        
        # If no regions found, try bottom third of vehicle
        if not regions:
            h_third = h // 3
            regions.append([0, h - h_third, w, h])
            
        return regions

    def _preprocess_standard(self, plate_img):
        """Standard preprocessing with grayscale and threshold."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Remove noise
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def _preprocess_adaptive(self, plate_img):
        """Preprocessing with adaptive threshold."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh

    def _preprocess_morphological(self, plate_img):
        """Preprocessing with morphological operations."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closed

    def _preprocess_contrast(self, plate_img):
        """Preprocessing with contrast enhancement."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def _read_with_easyocr(self, processed_img):
        """
        Read text using EasyOCR.
        
        Returns:
            tuple: (text, confidence) or None
        """
        try:
            results = self.reader.readtext(processed_img, allowlist=self.allowed_chars)
            
            if not results:
                return None
                
            # Combine all detected text
            full_text = ""
            total_conf = 0.0
            
            for bbox, text, conf in results:
                # Filter text
                filtered = ''.join(c for c in text.upper() if c in self.allowed_chars)
                if filtered:
                    full_text += filtered
                    total_conf += conf
                    
            if full_text and len(full_text) >= 4:
                avg_conf = total_conf / len(results)
                return full_text, avg_conf
                
        except Exception as e:
            logger.error(f"Error in EasyOCR: {e}")
            
        return None

    def process_violation_image(self, image_path, vehicle_bbox):
        """
        Process a saved violation image to extract license plate.
        
        Args:
            image_path: Path to the violation image
            vehicle_bbox: [x1, y1, x2, y2] of the violating vehicle
            
        Returns:
            tuple: (plate_text, confidence) or None
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
                
            # Extract vehicle region with some padding
            x1, y1, x2, y2 = vehicle_bbox
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.shape[1], x2 + pad)
            y2 = min(img.shape[0], y2 + pad)
            
            vehicle_img = img[y1:y2, x1:x2]
            
            # Read plate
            result = self.read_plate_from_vehicle(vehicle_img)
            
            if result:
                plate_text, confidence, _ = result
                logger.info(f"Extracted plate from violation image: {plate_text} ({confidence:.2f})")
                return plate_text, confidence
                
        except Exception as e:
            logger.error(f"Error processing violation image: {e}")
            
        return None