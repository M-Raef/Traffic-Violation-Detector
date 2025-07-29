"""
Enhanced License Plate Recognition module with integrated image quality enhancement.
"""
import cv2
import numpy as np
import easyocr
import torch
import os
import sys

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

from .plate_enhancer import PlateImageEnhancer

class SuperEnhancedLicensePlateReader:
    """
    Enhanced license plate reader with integrated image quality enhancement
    for improved accuracy on violation images.
    """
    
    def __init__(self, allowed_chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        """
        Initialize the enhanced license plate reader with image enhancer.
        
        Args:
            allowed_chars: Characters allowed in license plates
        """
        # Initialize EasyOCR with GPU support if available
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.allowed_chars = allowed_chars
        
        # Initialize image enhancer
        self.enhancer = PlateImageEnhancer()
        
        # Preprocessing parameters
        self.min_plate_ratio = 2.0  # Minimum aspect ratio for plates
        self.max_plate_ratio = 6.0  # Maximum aspect ratio for plates
        
        # Enhancement settings
        self.enhancement_levels = ['light', 'medium', 'aggressive']
        self.enable_multi_enhancement = True  # Try multiple enhancement levels
        
        logger.info("Super enhanced license plate reader initialized with image enhancer")

    def read_plate_from_vehicle(self, vehicle_img, bbox=None, enhancement_level='auto'):
        """
        Read license plate from a vehicle image with integrated enhancement.
        
        Args:
            vehicle_img: Full frame or vehicle crop
            bbox: Optional [x1, y1, x2, y2] of vehicle in the image
            enhancement_level: 'auto', 'light', 'medium', 'aggressive', or 'multi'
            
        Returns:
            tuple: (plate_text, confidence, plate_region, enhanced_plate_img) or None
        """
        # If bbox provided, crop to vehicle
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            vehicle_crop = vehicle_img[y1:y2, x1:x2]
        else:
            vehicle_crop = vehicle_img
            
        # Detect plate regions
        plate_regions = self._detect_plate_regions(vehicle_crop)
        
        best_result = None
        best_confidence = 0.0
        best_region = None
        best_enhanced_img = None
        
        # Try to read each potential plate region
        for region in plate_regions:
            rx1, ry1, rx2, ry2 = region
            plate_img = vehicle_crop[ry1:ry2, rx1:rx2]
            
            # Skip if too small
            if plate_img.shape[0] < 20 or plate_img.shape[1] < 60:
                continue
            
            # Try reading with enhancement
            result = self._read_plate_with_enhancement(plate_img, enhancement_level)
            
            if result:
                text, conf, enhanced_img = result
                if conf > best_confidence and len(text) >= 4:
                    best_result = (text, conf)
                    best_confidence = conf
                    best_region = region
                    best_enhanced_img = enhanced_img
                    
        if best_result:
            return best_result[0], best_result[1], best_region, best_enhanced_img
        return None

    def _read_plate_with_enhancement(self, plate_img, enhancement_level='auto'):
        """
        Read license plate with multiple enhancement attempts.
        
        Args:
            plate_img: License plate image region
            enhancement_level: Enhancement level to apply
            
        Returns:
            tuple: (text, confidence, enhanced_image) or None
        """
        best_result = None
        best_confidence = 0.0
        best_enhanced = None
        
        # Determine enhancement strategies
        if enhancement_level == 'auto':
            # Auto mode - try different levels based on image quality
            strategies = self._determine_auto_strategies(plate_img)
        elif enhancement_level == 'multi':
            # Try all enhancement levels
            strategies = self.enhancement_levels
        else:
            # Single enhancement level
            strategies = [enhancement_level]
        
        # Try original image first (baseline)
        original_result = self._read_with_multiple_preprocessing(plate_img)
        if original_result:
            best_result = original_result
            best_confidence = original_result[1]
            best_enhanced = plate_img
        
        # Try enhanced versions
        for strategy in strategies:
            try:
                # Enhance the image
                enhanced_img = self.enhancer.enhance_license_plate_region(
                    plate_img, enhancement_level=strategy
                )
                
                if enhanced_img is None:
                    continue
                
                # Read from enhanced image
                result = self._read_with_multiple_preprocessing(enhanced_img)
                
                if result and result[1] > best_confidence:
                    best_result = result
                    best_confidence = result[1]
                    best_enhanced = enhanced_img
                    logger.debug(f"Better result with {strategy} enhancement: {result[0]} ({result[1]:.2f})")
                
            except Exception as e:
                logger.error(f"Error with {strategy} enhancement: {e}")
                continue
        
        if best_result:
            return best_result[0], best_result[1], best_enhanced
        return None

    def _determine_auto_strategies(self, plate_img):
        """
        Automatically determine the best enhancement strategies based on image analysis.
        
        Args:
            plate_img: License plate image
            
        Returns:
            list: Ordered list of enhancement strategies to try
        """
        strategies = []
        
        # Analyze image quality
        blur_score = self._calculate_blur_score(plate_img)
        brightness = np.mean(plate_img)
        contrast = np.std(plate_img)
        
        # Determine strategies based on analysis
        if blur_score < 30:  # Very blurry
            strategies.extend(['aggressive', 'medium'])
        elif blur_score < 60:  # Moderately blurry
            strategies.extend(['medium', 'light'])
        else:  # Sharp enough
            strategies.append('light')
        
        if brightness < 80:  # Dark image
            strategies.insert(0, 'medium')
        
        if contrast < 30:  # Low contrast
            strategies.insert(0, 'aggressive')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        return unique_strategies or ['medium']  # Default fallback

    def _read_with_multiple_preprocessing(self, plate_img):
        """
        Try multiple preprocessing methods on the plate image.
        
        Args:
            plate_img: License plate image
            
        Returns:
            tuple: (text, confidence) or None
        """
        best_result = None
        best_confidence = 0.0
        
        # Original preprocessing methods
        preprocessing_methods = [
            ('original', self._preprocess_original),
            ('standard', self._preprocess_standard),
            ('adaptive', self._preprocess_adaptive),
            ('morphological', self._preprocess_morphological),
            ('contrast', self._preprocess_contrast),
        ]
        
        for method_name, method_func in preprocessing_methods:
            try:
                processed_img = method_func(plate_img)
                result = self._read_with_easyocr(processed_img)
                
                if result and result[1] > best_confidence:
                    best_result = result
                    best_confidence = result[1]
                    logger.debug(f"Better result with {method_name}: {result[0]} ({result[1]:.2f})")
                    
            except Exception as e:
                logger.debug(f"Error with {method_name} preprocessing: {e}")
                continue    
        return best_result

    def process_violation_image(self, image_path, vehicle_bbox, save_enhanced=True):
        """
        Process a saved violation image to extract license plate with enhancement.
        
        Args:
            image_path: Path to the violation image
            vehicle_bbox: [x1, y1, x2, y2] of the violating vehicle
            save_enhanced: Whether to save enhanced images for debugging
            
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
            
            # Read plate with enhancement
            result = self.read_plate_from_vehicle(vehicle_img, enhancement_level='multi')
            
            if result:
                plate_text, confidence, plate_region, enhanced_plate = result
                
                # Save enhanced images if requested
                if save_enhanced and enhanced_plate is not None:
                    self._save_enhanced_debug_images(
                        image_path, vehicle_img, enhanced_plate, plate_text, confidence
                    )
                
                logger.info(f"Extracted plate from violation image: {plate_text} ({confidence:.2f})")
                return plate_text, confidence
                
        except Exception as e:
            logger.error(f"Error processing violation image: {e}")
            
        return None

    def _save_enhanced_debug_images(self, original_path, vehicle_img, enhanced_plate, plate_text, confidence):
        """Save enhanced images for debugging purposes."""
        try:
            # Create debug directory
            debug_dir = os.path.join(os.path.dirname(original_path), "enhanced_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            timestamp = f"{plate_text}_{confidence:.2f}".replace(" ", "_")
            
            # Save vehicle crop
            vehicle_path = os.path.join(debug_dir, f"{base_name}_vehicle_{timestamp}.jpg")
            cv2.imwrite(vehicle_path, vehicle_img)
            
            # Save enhanced plate
            plate_path = os.path.join(debug_dir, f"{base_name}_plate_{timestamp}.jpg")
            cv2.imwrite(plate_path, enhanced_plate)
            
            logger.debug(f"Saved debug images: {vehicle_path}, {plate_path}")
            
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")

    # Preprocessing methods
    def _preprocess_original(self, plate_img):
        """Return original image."""
        return plate_img

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
        Read text using EasyOCR with enhanced error handling.
        
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
            valid_detections = 0
            
            for bbox, text, conf in results:
                # Filter text
                filtered = ''.join(c for c in text.upper() if c in self.allowed_chars)
                if filtered and len(filtered) >= 2:  # At least 2 characters
                    full_text += filtered
                    total_conf += conf
                    valid_detections += 1
                    
            if full_text and len(full_text) >= 4 and valid_detections > 0:
                avg_conf = total_conf / valid_detections
                return full_text, avg_conf
                
        except Exception as e:
            logger.error(f"Error in EasyOCR: {e}")
            
        return None

    def _detect_plate_regions(self, vehicle_img):
        """
        Detect potential license plate regions in vehicle image with enhanced detection.
        
        Returns:
            List of [x1, y1, x2, y2] regions
        """
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple edge detection methods
        edges_methods = [
            cv2.Canny(gray, 50, 150),
            cv2.Canny(gray, 30, 100),
            cv2.Canny(gray, 100, 200)
        ]
        
        h, w = vehicle_img.shape[:2]
        
        for edges in edges_methods:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
                
                # Plate typically in lower portion of vehicle
                if y < h * 0.2:  # Too high to be a plate
                    continue
                    
                # Check if this region is already covered by existing regions
                is_duplicate = False
                for existing_region in regions:
                    ex1, ey1, ex2, ey2 = existing_region
                    # Calculate overlap
                    overlap_x = max(0, min(x + w_rect, ex2) - max(x, ex1))
                    overlap_y = max(0, min(y + h_rect, ey2) - max(y, ey1))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > (area * 0.5):  # More than 50% overlap
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    regions.append([x, y, x + w_rect, y + h_rect])
        
        # If no regions found, try bottom sections of vehicle
        if not regions:
            # Bottom third
            h_third = h // 3
            regions.append([0, h - h_third, w, h])
            
            # Bottom half
            h_half = h // 2
            regions.append([0, h - h_half, w, h])
        
        # Sort regions by y-coordinate (prefer lower regions)
        regions.sort(key=lambda r: r[1], reverse=True)
        
        return regions[:5]  # Return top 5 candidates

    def _calculate_blur_score(self, image):
        """Calculate blur score using Laplacian variance."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 50  # Default moderate score

    def batch_process_violations(self, violations_dir, output_dir=None, enhancement_level='auto'):
        """
        Process all violation images in a directory with enhancement.
        
        Args:
            violations_dir: Directory containing violation images
            output_dir: Directory to save enhanced images (optional)
            enhancement_level: Enhancement level to apply
            
        Returns:
            dict: Results dictionary with statistics
        """
        if not os.path.exists(violations_dir):
            logger.error(f"Violations directory not found: {violations_dir}")
            return {}
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(violations_dir) 
                      if f.lower().endswith(image_extensions)]
        
        logger.info(f"Found {len(image_files)} violation images to process")
        
        results = {
            'total_images': len(image_files),
            'successful_reads': 0,
            'failed_reads': 0,
            'plates_detected': [],
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for image_file in image_files:
            image_path = os.path.join(violations_dir, image_file)
            
            try:
                # Read image to get dimensions for bbox (assume full image)
                img = cv2.imread(image_path)
                if img is None:
                    results['failed_reads'] += 1
                    continue
                
                h, w = img.shape[:2]
                vehicle_bbox = [0, 0, w, h]  # Use full image as vehicle region
                
                # Process with enhancement
                result = self.process_violation_image(image_path, vehicle_bbox, save_enhanced=True)
                
                if result:
                    plate_text, confidence = result
                    results['successful_reads'] += 1
                    results['plates_detected'].append({
                        'image': image_file,
                        'plate': plate_text,
                        'confidence': confidence
                    })
                    total_confidence += confidence
                    
                    # Save enhanced version if output directory specified
                    if output_dir:
                        enhanced_path = os.path.join(output_dir, f"enhanced_{image_file}")
                        enhanced_img = self.enhancer.enhance_violation_image(
                            image_path, enhanced_path, methods=enhancement_level
                        )
                else:
                    results['failed_reads'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                results['failed_reads'] += 1
        
        # Calculate average confidence
        if results['successful_reads'] > 0:
            results['average_confidence'] = total_confidence / results['successful_reads']
        
        # Log summary
        logger.info(f"Batch processing complete:")
        logger.info(f"  Total images: {results['total_images']}")
        logger.info(f"  Successful reads: {results['successful_reads']}")
        logger.info(f"  Failed reads: {results['failed_reads']}")
        logger.info(f"  Success rate: {results['successful_reads']/results['total_images']*100:.1f}%")
        logger.info(f"  Average confidence: {results['average_confidence']:.2f}")
        
        return results