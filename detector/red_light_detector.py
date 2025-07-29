"""
Updated Red Light Violation Detector module.

This module now:
1. Only captures images of violating vehicles
2. Draws bounding box only on the violating vehicle
3. Processes license plates from saved violation images
4. Uses stable vehicle tracking
5. ENHANCED: Now uses SuperEnhancedLicensePlateReader with image quality enhancement
"""
import os
import cv2
import torch
import numpy as np
import signal
import time
from datetime import datetime

# Import configuration
import config

# Import modules
from utils import logger, draw_bounding_box, draw_vehicle_info, draw_traffic_light, draw_roi_polygon, draw_status_info
from models import StableTracker, YOLODetector, SuperEnhancedLicensePlateReader, TrafficLightAnalyzer
from database import DatabaseManager

class RedLightViolationDetector:
    """
    Red light violation detector that analyzes traffic videos to detect
    vehicles crossing a designated area during red traffic lights.
    """
    
    def __init__(self, model_path=config.YOLO_MODEL_PATH, output_dir=config.OUTPUT_DIR):
        """
        Initialize the red light violation detector.
        
        Args:
            model_path: Path to the YOLO model weights
            output_dir: Directory to save output images and videos
        """
        self.output_dir = output_dir
        self.daily_output_dir = config.DAILY_OUTPUT_DIR
        self.plates_dir = config.PLATES_DIR
        
        # Initialize components
        self.detector = YOLODetector(model_path=model_path, conf_threshold=config.CONF_THRESHOLD)
        self.plate_reader = SuperEnhancedLicensePlateReader(allowed_chars=config.LICENSE_PLATE_CHARS)
        self.light_analyzer = TrafficLightAnalyzer()
        self.db_manager = DatabaseManager(db_path=config.DB_PATH)
        self.tracker = StableTracker(distance_threshold=40, max_lost_frames=15)
        
        # Processing parameters
        self.frame_skip = 0  # Process every frame
        self.last_frame_number = 0
        self.processed_vehicle_ids = set()  # Use set for O(1) lookup
        
        # Store violation info for post-processing
        self.pending_violations = []  # List of violations to process plates
        
        # Flag for interruption handling
        self.interrupted = False
        
        logger.info("Red light violation detector initialized with ENHANCED license plate reader")

    def setup_roi_polygon(self, points):
        """
        Set up a region of interest using polygon points.
        
        Args:
            points: List of polygon points [[x1, y1], [x2, y2], ...]
        """
        self.roi_polygon = np.array(points, np.int32)
        logger.info(f"ROI polygon set up with points: {points}")

    def save_violation_image(self, frame, vehicle_bbox, vehicle_id, frame_number):
        """
        Save violation image with bounding box only on violating vehicle.
        
        Args:
            frame: Original frame
            vehicle_bbox: [x1, y1, x2, y2] of violating vehicle
            vehicle_id: ID of the violating vehicle
            frame_number: Current frame number
            
        Returns:
            str: Path to saved image
        """
        # Create a copy of the frame
        violation_frame = frame.copy()
        
        # Draw only the violating vehicle's bounding box
        x1, y1, x2, y2 = vehicle_bbox
        draw_bounding_box(violation_frame, (x1, y1, x2, y2), color=(0, 0, 255), thickness=3)
        
        # Add violation text
        cv2.putText(violation_frame, f"RED LIGHT VIOLATION - Vehicle ID: {vehicle_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(violation_frame, f"Time: {timestamp}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(violation_frame, f"Frame: {frame_number}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the image
        filename = f"red_light_violation_{vehicle_id}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        image_path = os.path.join(self.daily_output_dir, filename)
        cv2.imwrite(image_path, violation_frame)
        
        logger.info(f"Saved violation image: {image_path}")
        return image_path

    def process_frame(self, frame, frame_number=0):
        """
        Process a single frame to detect vehicles, traffic lights, and violations.
        
        Args:
            frame: Video frame to process
            frame_number: Current frame number
            
        Returns:
            tuple: (processed_frame, violations, light_color)
        """
        display_frame = frame.copy()
        
        # Frame-based skip if enabled
        if self.frame_skip > 0 and (frame_number - self.last_frame_number) < self.frame_skip:
            return display_frame, None, "unknown"
        
        self.last_frame_number = frame_number
        
        # Get YOLO detection results
        results = self.detector.detect(frame)
        
        # Process detected objects
        traffic_lights = []
        
        # Convert boxes to list format for tracking
        vehicle_boxes = []
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            if class_id not in self.detector.class_map:
                continue
                
            object_type = self.detector.class_map[class_id]
            conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if self.detector.is_vehicle(class_id):
                vehicle_boxes.append([x1, y1, x2, y2])
                
            elif class_id == 9:  # Traffic light
                traffic_light_img = frame[y1:y2, x1:x2]
                if traffic_light_img.size > 0:
                    light_color, color_conf = self.light_analyzer.get_light_color(traffic_light_img)
                    traffic_lights.append({
                        "bbox": (x1, y1, x2, y2),
                        "color": light_color,
                        "conf": color_conf
                    })
                    # Draw traffic light on display frame
                    draw_traffic_light(display_frame, (x1, y1, x2, y2), light_color)
        
        # Draw ROI polygon if defined
        if hasattr(self, 'roi_polygon'):
            draw_roi_polygon(display_frame, self.roi_polygon)
        
        # Track vehicles with stable tracker
        tracked_vehicles = self.tracker.update(vehicle_boxes)
        
        # Determine dominant traffic light color
        dominant_light_color = self.light_analyzer.determine_dominant_light(traffic_lights)
        
        # Draw status information
        draw_status_info(display_frame, frame_number, len(self.processed_vehicle_ids), dominant_light_color)
        
        # Process vehicles and check for violations
        violations = []
        
        for bbox in tracked_vehicles:
            x1, y1, x2, y2, vehicle_id = bbox
            
            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check if in region of interest
            in_roi = False
            if hasattr(self, 'roi_polygon'):
                in_roi = cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) >= 0
            else:
                in_roi = True  # If no ROI defined, consider all vehicles
            
            # Get track object
            track = self.tracker.get_track(vehicle_id)
            if not track:
                continue
            
            # Check for red light violations
            if (dominant_light_color == "red" and in_roi and 
                vehicle_id not in self.processed_vehicle_ids and
                track.hits >= 3):  # Ensure stable tracking
                
                # This is a violation - save image with only this vehicle's bounding box
                image_path = self.save_violation_image(frame, [x1, y1, x2, y2], 
                                                      vehicle_id, frame_number)
                
                # Add to pending violations for plate processing
                self.pending_violations.append({
                    'vehicle_id': vehicle_id,
                    'image_path': image_path,
                    'vehicle_bbox': [x1, y1, x2, y2],
                    'frame_number': frame_number,
                    'timestamp': datetime.now()
                })
                
                # Add to processed list to avoid duplicate detections
                self.processed_vehicle_ids.add(vehicle_id)
                
                # Create violation record (plate will be updated later)
                violation_id = self.db_manager.save_violation(
                    violation_type="red_light",
                    license_plate=f"PENDING_{vehicle_id}",
                    confidence=0.0,
                    image_path=image_path,
                    traffic_light_color="red",
                    vehicle_id=vehicle_id,
                    details=f"Vehicle ID {vehicle_id} crossed ROI during red light at frame {frame_number}"
                )
                
                if violation_id:
                    violations.append({
                        "id": violation_id,
                        "type": "red_light",
                        "plate": "PENDING",
                        "vehicle_id": vehicle_id,
                        "conf": 0.0
                    })
                    
                    # Mark vehicle with red rectangle for violation on display
                    draw_bounding_box(display_frame, (x1, y1, x2, y2), color=(0, 0, 255))
                    draw_vehicle_info(display_frame, x1, y1, vehicle_id, "VIOLATION", 0.0, is_violation=True)
            else:
                # Regular vehicle display (no violation)
                draw_bounding_box(display_frame, (x1, y1, x2, y2), color=(0, 255, 0))
                # Don't show license plate in real-time
                draw_vehicle_info(display_frame, x1, y1, vehicle_id, None, 0.0)
        
        # Reset the processed vehicle list when light changes from red
        if dominant_light_color != "red" and dominant_light_color != "unknown":
            self.processed_vehicle_ids.clear()
        
        return display_frame, violations, dominant_light_color

    def process_pending_violations(self):
        """
        Process all pending violations to extract license plates from saved images.
        """
        logger.info(f"Processing {len(self.pending_violations)} pending violations for license plates...")
        
        for violation in self.pending_violations:
            try:
                # Read license plate from saved violation image using ENHANCED reader
                result = self.plate_reader.process_violation_image(
                    violation['image_path'],
                    violation['vehicle_bbox']
                )
                
                if result:
                    plate_text, confidence = result
                    
                    # Update the violation record in database
                    conn = self.db_manager.db_path
                    import sqlite3
                    db_conn = sqlite3.connect(conn)
                    cursor = db_conn.cursor()
                    
                    cursor.execute('''
                        UPDATE violations 
                        SET license_plate = ?, confidence = ?
                        WHERE vehicle_id = ? AND license_plate LIKE 'PENDING_%'
                    ''', (plate_text, confidence, violation['vehicle_id']))
                    
                    db_conn.commit()
                    db_conn.close()
                    
                    logger.info(f"Updated violation for vehicle {violation['vehicle_id']}: "
                              f"Plate={plate_text}, Confidence={confidence:.2f}")
                else:
                    logger.warning(f"Could not extract plate for vehicle {violation['vehicle_id']}")
                    
            except Exception as e:
                logger.error(f"Error processing violation for vehicle {violation['vehicle_id']}: {e}")
        
        # Clear pending violations
        self.pending_violations.clear()

    def process_video(self, video_path, output_path=None, roi_points=None, in_thread=False):
        """
        Process a video file and detect red light violations.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            roi_points: Region of interest polygon points (optional)
            in_thread: Flag indicating if this method is running in a thread
            
        Returns:
            int: Total number of violations detected
        """
        # Reset the interrupted flag and tracking data
        self.interrupted = False
        self.processed_vehicle_ids.clear()
        self.pending_violations.clear()
        
        # Only register signal handler in main thread
        if not in_thread:
            try:
                def signal_handler(sig, frame):
                    logger.info("Process interrupted by user! Saving output and exiting gracefully...")
                    self.interrupted = True
                
                # Register the signal handler
                signal.signal(signal.SIGINT, signal_handler)
            except ValueError as e:
                logger.warning(f"Could not register signal handler: {e}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up ROI polygon if provided
        if roi_points:
            self.setup_roi_polygon(roi_points)
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        total_violations = 0
        
        print("Processing video with ENHANCED license plate recognition... Press Ctrl+C to stop and save the output")
        
        try:
            while not self.interrupted:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info(f"End of video or error reading frame at frame {frame_number}")
                    break
                
                # Debug message to track frame processing
                if frame_number % 30 == 0:
                    print(f"Processing frame {frame_number} | Violations: {total_violations}", end="\r")
                
                # Process frame
                processed_frame, violations, light_color = self.process_frame(frame, frame_number)
                
                # Count violations
                if violations:
                    total_violations += len(violations)
                    for violation in violations:
                        logger.info(f"Violation detected: {violation}")
                
                # Write to output video
                if out:
                    out.write(processed_frame)
                
                frame_number += 1
                
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            print(f"\nError occurred: {e}")
        
        finally:
            # Clean up
            cap.release()
            if out:
                out.release()
            
            # Process pending violations for license plates with ENHANCED reader
            print("\nProcessing license plates from violation images with ENHANCED reader...")
            self.process_pending_violations()
            
            # Print final status
            if self.interrupted:
                print(f"\nProcess interrupted by user at frame {frame_number}")
                print(f"Output video saved to {output_path} (partial)")
            else:
                print(f"\nProcessing complete: {frame_number} frames, {total_violations} violations")
            
            logger.info(f"Processed {frame_number} frames, detected {total_violations} violations")
            return total_violations