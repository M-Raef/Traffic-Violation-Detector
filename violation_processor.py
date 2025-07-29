"""
Standalone violation image processor for batch processing license plates.

This module can be used to reprocess violation images to extract license plates
or to process violations that failed initial plate detection.

ENHANCED: Now uses SuperEnhancedLicensePlateReader with image quality enhancement.
"""
import os
import cv2
import json
import sqlite3
from datetime import datetime
from models.enhanced_license_plate import SuperEnhancedLicensePlateReader
from database.db_manager import DatabaseManager
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

class ViolationProcessor:
    """
    Processes violation images to extract license plates using advanced techniques.
    """
    
    def __init__(self, db_path="traffic_violations.db"):
        """
        Initialize the violation processor.
        
        Args:
            db_path: Path to the database
        """
        self.db_path = db_path
        self.plate_reader = SuperEnhancedLicensePlateReader()  # ENHANCED reader
        self.db_manager = DatabaseManager(db_path=db_path)
        logger.info("Violation processor initialized with ENHANCED license plate reader")
    
    def process_pending_violations(self):
        """
        Process all violations with PENDING plates in the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all pending violations
        cursor.execute("""
            SELECT id, vehicle_id, image_path, violation_details 
            FROM violations 
            WHERE license_plate LIKE 'PENDING_%'
        """)
        
        pending = cursor.fetchall()
        conn.close()
        
        logger.info(f"Found {len(pending)} pending violations to process with ENHANCED reader")
        
        processed = 0
        successful = 0
        
        for violation_id, vehicle_id, image_path, details in pending:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Extract vehicle bbox from details if available
            bbox = self._extract_bbox_from_image(image_path)
            
            # Process the image with ENHANCED reader
            result = self.plate_reader.process_violation_image(image_path, bbox)
            
            if result:
                plate_text, confidence = result
                self._update_violation(violation_id, plate_text, confidence)
                successful += 1
                logger.info(f"ENHANCED processing - Updated violation {violation_id}: {plate_text} ({confidence:.2f})")
            else:
                logger.warning(f"ENHANCED reader could not extract plate for violation {violation_id}")
            
            processed += 1
        
        logger.info(f"ENHANCED processing complete: {processed} violations processed, {successful} successful")
        return processed, successful
    
    def reprocess_violation(self, violation_id):
        """
        Reprocess a specific violation.
        
        Args:
            violation_id: ID of the violation to reprocess
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_path, vehicle_id 
            FROM violations 
            WHERE id = ?
        """, (violation_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            logger.error(f"Violation {violation_id} not found")
            return False
        
        image_path, vehicle_id = result
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return False
        
        # Extract bbox and process with ENHANCED reader
        bbox = self._extract_bbox_from_image(image_path)
        result = self.plate_reader.process_violation_image(image_path, bbox)
        
        if result:
            plate_text, confidence = result
            self._update_violation(violation_id, plate_text, confidence)
            logger.info(f"ENHANCED reprocessing - Updated violation {violation_id}: {plate_text} ({confidence:.2f})")
            return True
        
        return False
    
    def process_directory(self, directory_path):
        """
        Process all violation images in a directory.
        
        Args:
            directory_path: Path to directory containing violation images
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Get all image files
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        logger.info(f"Found {len(image_files)} images to process with ENHANCED reader")
        
        results = []
        
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            
            # Try to extract vehicle ID from filename
            vehicle_id = self._extract_vehicle_id_from_filename(image_file)
            
            # Process the image with ENHANCED reader
            bbox = self._extract_bbox_from_image(image_path)
            result = self.plate_reader.process_violation_image(image_path, bbox)
            
            if result:
                plate_text, confidence = result
                results.append({
                    'image': image_file,
                    'vehicle_id': vehicle_id,
                    'plate': plate_text,
                    'confidence': confidence
                })
                logger.info(f"ENHANCED processing - {image_file}: {plate_text} ({confidence:.2f})")
            else:
                logger.warning(f"ENHANCED reader could not extract plate from {image_file}")
        
        # Save results
        output_file = os.path.join(directory_path, 'enhanced_plate_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"ENHANCED processing results saved to {output_file}")
        return results
    
    def batch_enhance_and_process(self, directory_path, enhancement_level='auto'):
        """
        Batch enhance and process all images in a directory.
        
        Args:
            directory_path: Path to directory containing violation images
            enhancement_level: Enhancement level to apply ('light', 'medium', 'aggressive', 'auto')
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        logger.info(f"Starting batch enhancement and processing with level: {enhancement_level}")
        
        # Use the enhanced reader's batch processing capability
        results = self.plate_reader.batch_process_violations(
            violations_dir=directory_path,
            output_dir=os.path.join(directory_path, "enhanced_output"),
            enhancement_level=enhancement_level
        )
        
        logger.info(f"Batch enhancement complete:")
        logger.info(f"  Total images: {results['total_images']}")
        logger.info(f"  Successful reads: {results['successful_reads']}")
        logger.info(f"  Success rate: {results['successful_reads']/results['total_images']*100:.1f}%")
        logger.info(f"  Average confidence: {results['average_confidence']:.2f}")
        
        return results
    
    def compare_enhancement_levels(self, image_path):
        """
        Compare different enhancement levels on a single image.
        
        Args:
            image_path: Path to the violation image
            
        Returns:
            dict: Results for each enhancement level
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {}
        
        logger.info(f"Comparing enhancement levels on: {image_path}")
        
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return {}
        
        h, w = img.shape[:2]
        vehicle_bbox = [0, 0, w, h]  # Use full image
        
        enhancement_levels = ['light', 'medium', 'aggressive', 'auto', 'multi']
        comparison_results = {}
        
        for level in enhancement_levels:
            logger.info(f"Testing {level} enhancement...")
            
            # Process with current enhancement level
            result = self.plate_reader.read_plate_from_vehicle(
                img, enhancement_level=level
            )
            
            if result:
                plate_text, confidence = result[:2]
                comparison_results[level] = {
                    'plate': plate_text,
                    'confidence': confidence,
                    'success': True
                }
                logger.info(f"  {level}: {plate_text} ({confidence:.2f})")
            else:
                comparison_results[level] = {
                    'plate': None,
                    'confidence': 0.0,
                    'success': False
                }
                logger.info(f"  {level}: No plate detected")
        
        # Find best result
        best_level = max(comparison_results.keys(), 
                        key=lambda x: comparison_results[x]['confidence'])
        
        logger.info(f"Best enhancement level: {best_level} "
                   f"({comparison_results[best_level]['plate']}, "
                   f"{comparison_results[best_level]['confidence']:.2f})")
        
        return comparison_results
    
    def _extract_bbox_from_image(self, image_path):
        """
        Try to extract vehicle bounding box from the violation image.
        This is a placeholder - in practice, you might store bbox in metadata.
        """
        # For now, return None to process the entire image
        # In a real implementation, you might:
        # 1. Store bbox in image metadata
        # 2. Use object detection to find the vehicle
        # 3. Store bbox in database
        
        # Try to get image dimensions and use full image as bbox
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                return [0, 0, w, h]
        except:
            pass
        
        return None
    
    def _extract_vehicle_id_from_filename(self, filename):
        """
        Extract vehicle ID from filename if possible.
        """
        # Assuming filename format: violation_type_vehicleID_timestamp.jpg
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                pass
        return None
    
    def _update_violation(self, violation_id, plate_text, confidence):
        """
        Update violation record in database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE violations 
            SET license_plate = ?, confidence = ?
            WHERE id = ?
        """, (plate_text, confidence, violation_id))
        
        conn.commit()
        conn.close()
    
    def get_enhancement_statistics(self, directory_path):
        """
        Get statistics about enhancement effectiveness.
        
        Args:
            directory_path: Path to directory containing violation images
            
        Returns:
            dict: Statistics about enhancement performance
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return {}
        
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logger.warning(f"No images found in {directory_path}")
            return {}
        
        logger.info(f"Analyzing enhancement statistics for {len(image_files)} images...")
        
        stats = {
            'total_images': len(image_files),
            'enhancement_levels': {},
            'best_overall_level': None,
            'average_improvements': {}
        }
        
        # Test each enhancement level
        enhancement_levels = ['light', 'medium', 'aggressive', 'auto']
        
        for level in enhancement_levels:
            successful = 0
            total_confidence = 0.0
            
            for image_file in image_files[:10]:  # Test on first 10 images for speed
                image_path = os.path.join(directory_path, image_file)
                
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    result = self.plate_reader.read_plate_from_vehicle(
                        img, enhancement_level=level
                    )
                    
                    if result:
                        successful += 1
                        total_confidence += result[1]
                        
                except Exception as e:
                    logger.error(f"Error processing {image_file} with {level}: {e}")
            
            stats['enhancement_levels'][level] = {
                'success_rate': successful / min(10, len(image_files)) * 100,
                'average_confidence': total_confidence / max(1, successful)
            }
        
        # Find best level
        best_level = max(stats['enhancement_levels'].keys(),
                        key=lambda x: stats['enhancement_levels'][x]['success_rate'])
        stats['best_overall_level'] = best_level
        
        logger.info(f"Enhancement statistics analysis complete")
        logger.info(f"Best overall level: {best_level}")
        
        return stats