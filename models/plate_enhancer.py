"""
Image Quality Enhancer for License Plate Recognition
Enhances violation images to improve license plate readability without sklearn dependencies.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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

class PlateImageEnhancer:
    """
    Enhances license plate images using various image processing techniques
    to improve OCR accuracy without sklearn dependencies.
    """
    
    def __init__(self):
        """Initialize the image enhancer."""
        self.enhancement_methods = {
            'deblur': self._deblur_image,
            'denoise': self._denoise_image,
            'contrast': self._enhance_contrast,
            'sharpening': self._sharpen_image,
            'super_resolution': self._super_resolution,
            'adaptive_histogram': self._adaptive_histogram_equalization,
            'gamma_correction': self._gamma_correction,
            'unsharp_mask': self._unsharp_mask
        }
        
        logger.info("Plate image enhancer initialized (no sklearn dependencies)")

    def enhance_violation_image(self, image_path, output_path=None, methods='auto'):
        """
        Enhance a violation image for better license plate recognition.
        
        Args:
            image_path: Path to the input violation image
            output_path: Path to save enhanced image (optional)
            methods: Enhancement methods to apply ('auto', 'all', or list of method names)
            
        Returns:
            numpy.ndarray: Enhanced image array
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            logger.info(f"Enhancing image: {image_path}")
            
            # Apply enhancement methods
            if methods == 'auto':
                enhanced_image = self._auto_enhance(image)
            elif methods == 'all':
                enhanced_image = self._apply_all_methods(image)
            elif isinstance(methods, list):
                enhanced_image = self._apply_selected_methods(image, methods)
            else:
                enhanced_image = image.copy()
            
            # Save enhanced image if output path provided
            if output_path:
                cv2.imwrite(output_path, enhanced_image)
                logger.info(f"Enhanced image saved: {output_path}")
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return None

    def enhance_license_plate_region(self, plate_image, enhancement_level='medium'):
        """
        Enhance a specific license plate region with targeted improvements.
        
        Args:
            plate_image: License plate image (numpy array)
            enhancement_level: 'light', 'medium', or 'aggressive'
            
        Returns:
            numpy.ndarray: Enhanced plate image
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        # Create a copy to work with
        enhanced = plate_image.copy()
        
        try:
            if enhancement_level == 'light':
                # Light enhancement - basic cleanup
                enhanced = self._denoise_image(enhanced)
                enhanced = self._enhance_contrast(enhanced, factor=1.2)
                enhanced = self._sharpen_image(enhanced, strength=0.5)
                
            elif enhancement_level == 'medium':
                # Medium enhancement - balanced approach
                enhanced = self._denoise_image(enhanced)
                enhanced = self._adaptive_histogram_equalization(enhanced)
                enhanced = self._enhance_contrast(enhanced, factor=1.3)
                enhanced = self._sharpen_image(enhanced, strength=0.8)
                enhanced = self._gamma_correction(enhanced, gamma=1.2)
                
            elif enhancement_level == 'aggressive':
                # Aggressive enhancement - maximum processing
                enhanced = self._super_resolution(enhanced)
                enhanced = self._deblur_image(enhanced)
                enhanced = self._denoise_image(enhanced)
                enhanced = self._adaptive_histogram_equalization(enhanced)
                enhanced = self._enhance_contrast(enhanced, factor=1.5)
                enhanced = self._unsharp_mask(enhanced)
                enhanced = self._gamma_correction(enhanced, gamma=1.3)
            
            logger.debug(f"Applied {enhancement_level} enhancement to license plate region")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in license plate enhancement: {e}")
            return plate_image

    def _auto_enhance(self, image):
        """Automatically determine and apply the best enhancement methods."""
        # Analyze image quality first
        blur_score = self._calculate_blur_score(image)
        noise_score = self._calculate_noise_score(image)
        contrast_score = self._calculate_contrast_score(image)
        
        enhanced = image.copy()
        
        # Apply enhancements based on image analysis
        if blur_score < 50:  # Image is blurry
            enhanced = self._deblur_image(enhanced)
            enhanced = self._sharpen_image(enhanced, strength=0.8)
        
        if noise_score > 30:  # Image is noisy
            enhanced = self._denoise_image(enhanced)
        
        if contrast_score < 40:  # Low contrast
            enhanced = self._adaptive_histogram_equalization(enhanced)
            enhanced = self._enhance_contrast(enhanced, factor=1.4)
        
        # Always apply some basic enhancement
        enhanced = self._gamma_correction(enhanced, gamma=1.1)
        
        logger.debug(f"Auto enhancement applied - Blur: {blur_score:.1f}, Noise: {noise_score:.1f}, Contrast: {contrast_score:.1f}")
        return enhanced

    def _apply_all_methods(self, image):
        """Apply all enhancement methods in optimal order."""
        enhanced = image.copy()
        
        # Order matters - apply in logical sequence
        enhanced = self._super_resolution(enhanced)
        enhanced = self._deblur_image(enhanced)
        enhanced = self._denoise_image(enhanced)
        enhanced = self._adaptive_histogram_equalization(enhanced)
        enhanced = self._enhance_contrast(enhanced)
        enhanced = self._gamma_correction(enhanced)
        enhanced = self._unsharp_mask(enhanced)
        
        return enhanced

    def _apply_selected_methods(self, image, methods):
        """Apply selected enhancement methods."""
        enhanced = image.copy()
        
        for method in methods:
            if method in self.enhancement_methods:
                enhanced = self.enhancement_methods[method](enhanced)
                logger.debug(f"Applied enhancement method: {method}")
        
        return enhanced

    # Enhancement Methods
    def _deblur_image(self, image):
        """Remove blur using Wiener deconvolution approximation."""
        try:
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            # Create a motion blur kernel (assuming horizontal motion)
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # Apply Wiener deconvolution (simplified)
            deblurred = cv2.filter2D(gray, -1, kernel)
            
            # Convert back to color if needed
            if is_color:
                deblurred_color = image.copy()
                deblurred_color[:,:,0] = deblurred
                deblurred_color[:,:,1] = deblurred
                deblurred_color[:,:,2] = deblurred
                return deblurred_color
            else:
                return deblurred
                
        except Exception as e:
            logger.error(f"Error in deblur: {e}")
            return image

    def _denoise_image(self, image):
        """Remove noise using Non-Local Means denoising."""
        try:
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            return denoised
        except Exception as e:
            logger.error(f"Error in denoise: {e}")
            return image

    def _enhance_contrast(self, image, factor=1.3):
        """Enhance contrast using CLAHE and PIL."""
        try:
            # OpenCV CLAHE
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
            
            # Additional contrast using PIL
            pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if len(enhanced.shape) == 3 else enhanced)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_enhanced = enhancer.enhance(factor)
            
            # Convert back to OpenCV format
            enhanced = np.array(pil_enhanced)
            if len(enhanced.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {e}")
            return image

    def _sharpen_image(self, image, strength=1.0):
        """Sharpen image using unsharp masking."""
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            
            # Subtract blurred from original and add back
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in sharpening: {e}")
            return image

    def _super_resolution(self, image, scale_factor=2):
        """Increase resolution using bicubic interpolation and edge enhancement."""
        try:
            h, w = image.shape[:2]
            
            # Upscale using bicubic interpolation
            upscaled = cv2.resize(image, (w * scale_factor, h * scale_factor), 
                                interpolation=cv2.INTER_CUBIC)
            
            # Apply edge enhancement
            if len(upscaled.shape) == 3:
                gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = upscaled.copy()
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Enhance edges
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(upscaled.shape) == 3 else edges
            enhanced = cv2.addWeighted(upscaled, 0.8, edges_colored, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in super resolution: {e}")
            return image

    def _adaptive_histogram_equalization(self, image):
        """Apply adaptive histogram equalization."""
        try:
            if len(image.shape) == 3:
                # Convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                equalized = clahe.apply(image)
            
            return equalized
            
        except Exception as e:
            logger.error(f"Error in histogram equalization: {e}")
            return image

    def _gamma_correction(self, image, gamma=1.2):
        """Apply gamma correction."""
        try:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            corrected = cv2.LUT(image, table)
            return corrected
            
        except Exception as e:
            logger.error(f"Error in gamma correction: {e}")
            return image

    def _unsharp_mask(self, image, sigma=1.0, strength=1.5, threshold=0):
        """Apply unsharp mask filter."""
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Create mask
            mask = cv2.subtract(image, blurred)
            
            # Apply threshold if specified
            if threshold > 0:
                mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Add mask back to original
            sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in unsharp mask: {e}")
            return image

    # Image Quality Analysis Methods
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

    def _calculate_noise_score(self, image):
        """Estimate noise level in the image."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use standard deviation of Laplacian as noise indicator
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_score = np.std(laplacian)
            return min(noise_score, 100)  # Cap at 100
        except:
            return 20  # Default low noise

    def _calculate_contrast_score(self, image):
        """Calculate contrast score using RMS contrast."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # RMS contrast
            contrast = np.std(gray.astype(np.float32))
            return min(contrast, 100)  # Cap at 100
        except:
            return 40  # Default moderate contrast

    def enhance_batch(self, input_dir, output_dir, methods='auto'):
        """
        Enhance all images in a directory.
        
        Args:
            input_dir: Directory containing violation images
            output_dir: Directory to save enhanced images
            methods: Enhancement methods to apply
        """
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        logger.info(f"Found {len(image_files)} images to enhance")
        
        enhanced_count = 0
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"enhanced_{image_file}")
            
            enhanced_image = self.enhance_violation_image(input_path, output_path, methods)
            
            if enhanced_image is not None:
                enhanced_count += 1
            
            # Progress update
            if enhanced_count % 10 == 0:
                logger.info(f"Enhanced {enhanced_count}/{len(image_files)} images")
        
        logger.info(f"Batch enhancement complete: {enhanced_count}/{len(image_files)} images enhanced")
        return enhanced_count

    def create_comparison_image(self, original_path, enhanced_path, output_path):
        """
        Create a side-by-side comparison image.
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to enhanced image
            output_path: Path to save comparison image
        """
        try:
            original = cv2.imread(original_path)
            enhanced = cv2.imread(enhanced_path)
            
            if original is None or enhanced is None:
                logger.error("Failed to read images for comparison")
                return False
            
            # Resize images to same height
            h1, w1 = original.shape[:2]
            h2, w2 = enhanced.shape[:2]
            
            target_height = min(h1, h2)
            original_resized = cv2.resize(original, (int(w1 * target_height / h1), target_height))
            enhanced_resized = cv2.resize(enhanced, (int(w2 * target_height / h2), target_height))
            
            # Create comparison image
            comparison = np.hstack([original_resized, enhanced_resized])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Enhanced", (original_resized.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save comparison
            cv2.imwrite(output_path, comparison)
            logger.info(f"Comparison image saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating comparison image: {e}")
            return False