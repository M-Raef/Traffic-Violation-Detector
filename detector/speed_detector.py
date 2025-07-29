"""
Updated Speed violation detector module.

This module now:
1. Only captures images of speeding vehicles
2. Draws bounding box only on the speeding vehicle
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
from utils import logger, draw_bounding_box, draw_vehicle_info, draw_status_info
from models import StableTracker, YOLODetector, SuperEnhancedLicensePlateReader
from database import DatabaseManager

class SpeedViolationDetector:
    """
    Speed violation detector that calculates vehicle speeds between two lines.
    """
    
    def __init__(self, model_path=config.YOLO_MODEL_PATH, output_dir=config.OUTPUT_DIR, 
                 speed_limit=30.0, distance_meters=5.0):
        """
        Initialize the speed violation detector.
        
        Args:
            model_path: Path to the YOLO model weights
            output_dir: Directory to save output images and videos
            speed_limit: Speed limit in km/h
            distance_meters: Real-world distance between entry and exit lines in meters
        """
        self.output_dir = output_dir
        self.daily_output_dir = config.DAILY_OUTPUT_DIR
        self.plates_dir = config.PLATES_DIR
        
        # Speed parameters
        self.speed_limit = speed_limit
        self.distance_meters = distance_meters
        self.fps = 30.0  # Default FPS, will be updated when processing video
        
        # Initialize components
        self.detector = YOLODetector(model_path=model_path, conf_threshold=config.CONF_THRESHOLD)
        self.plate_reader = SuperEnhancedLicensePlateReader(allowed_chars=config.LICENSE_PLATE_CHARS)
        self.db_manager = DatabaseManager(db_path=config.DB_PATH)
        self.tracker = StableTracker(distance_threshold=40, max_lost_frames=15)
        
        # Processing parameters
        self.frame_skip = 0  # Process every frame
        self.last_frame_number = 0
        
        # Vehicle tracking for speed calculation
        self.entry_timestamps = {}  # {vehicle_id: entry_frame_number}
        self.exit_timestamps = {}   # {vehicle_id: exit_frame_number}
        self.vehicle_speeds = {}    # {vehicle_id: speed_in_kmh}
        self.processed_vehicle_ids = set()  # IDs of vehicles for which violations have been recorded
        
        # Track vehicle positions for trajectory visualization
        self.vehicle_trajectories = {}  # {vehicle_id: list of positions}
        
        # Store violation info for post-processing
        self.pending_violations = []  # List of violations to process plates
        
        # Flag for interruption handling
        self.interrupted = False
        
        logger.info("Speed violation detector initialized with ENHANCED license plate reader")
        logger.info(f"Speed limit: {speed_limit} km/h, Distance: {distance_meters} meters")

    def set_lines(self, entry_line, exit_line):
        """
        Set up entry and exit lines for speed detection.
        
        Args:
            entry_line: List of two points [(x1, y1), (x2, y2)] defining the entry line
            exit_line: List of two points [(x1, y1), (x2, y2)] defining the exit line
        """
        if len(entry_line) != 2 or len(exit_line) != 2:
            logger.error("Entry and exit lines must each have exactly 2 points")
            raise ValueError("Entry and exit lines must each have exactly 2 points")
            
        self.entry_line = np.array(entry_line, np.int32)
        self.exit_line = np.array(exit_line, np.int32)
        logger.info(f"Entry line set: {entry_line}")
        logger.info(f"Exit line set: {exit_line}")

    def save_violation_image(self, frame, vehicle_bbox, vehicle_id, speed_kmh, frame_number):
        """
        Save violation image with bounding box only on speeding vehicle.
        
        Args:
            frame: Original frame
            vehicle_bbox: [x1, y1, x2, y2] of speeding vehicle
            vehicle_id: ID of the speeding vehicle
            speed_kmh: Vehicle speed in km/h
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
        cv2.putText(violation_frame, f"SPEED VIOLATION - Vehicle ID: {vehicle_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add speed info
        cv2.putText(violation_frame, f"Speed: {speed_kmh:.1f} km/h (Limit: {self.speed_limit} km/h)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(violation_frame, f"Time: {timestamp}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(violation_frame, f"Frame: {frame_number}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw the speed measurement lines
        self.draw_lines(violation_frame)
        
        # Save the image
        filename = f"speed_violation_{vehicle_id}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        image_path = os.path.join(self.daily_output_dir, filename)
        cv2.imwrite(image_path, violation_frame)
        
        logger.info(f"Saved violation image: {image_path}")
        return image_path

    def check_line_crossing(self, prev_pos, curr_pos, line):
        """
        Check if a vehicle crossed a line between previous and current position.
        
        Args:
            prev_pos: Previous vehicle center position (x, y)
            curr_pos: Current vehicle center position (x, y)
            line: Line defined by two points [(x1, y1), (x2, y2)]
            
        Returns:
            bool: True if the vehicle crossed the line, False otherwise
        """
        if prev_pos is None or curr_pos is None:
            return False
        
        # Convert line points to proper format
        line_start, line_end = line
        
        # Line equation for detection line: Ax + By + C = 0
        A1 = line_end[1] - line_start[1]
        B1 = line_start[0] - line_end[0]
        C1 = line_end[0] * line_start[1] - line_start[0] * line_end[1]
        
        # Evaluate points with the line equation
        prev_eval = A1 * prev_pos[0] + B1 * prev_pos[1] + C1
        curr_eval = A1 * curr_pos[0] + B1 * curr_pos[1] + C1
        
        # If signs are different, the line was crossed
        if prev_eval * curr_eval <= 0:
            # Verify crossing is within line segment bounds
            dx_line = line_end[0] - line_start[0]
            dy_line = line_end[1] - line_start[1]
            dx_path = curr_pos[0] - prev_pos[0]
            dy_path = curr_pos[1] - prev_pos[1]
            
            # Minimum movement threshold
            path_length = (dx_path**2 + dy_path**2)**0.5
            if path_length < 2:
                return False
            
            # Check intersection parameters
            denominator = dx_path * dy_line - dy_path * dx_line
            if abs(denominator) < 1e-10:
                return False
            
            t = ((line_start[0] - prev_pos[0]) * dy_path - (line_start[1] - prev_pos[1]) * dx_path) / denominator
            u = ((line_start[0] - prev_pos[0]) * dy_line - (line_start[1] - prev_pos[1]) * dx_line) / denominator
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
        
        return False

    def calculate_speed(self, entry_frame, exit_frame):
        """
        Calculate speed based on frame numbers and distance.
        
        Args:
            entry_frame: Frame number when the vehicle crossed the entry line
            exit_frame: Frame number when the vehicle crossed the exit line
            
        Returns:
            float: Calculated speed in kilometers per hour
        """
        frames_diff = exit_frame - entry_frame
        
        # Minimum frame difference to avoid unrealistic speeds
        min_frames = max(3, int(self.fps * 0.1))
        if frames_diff < min_frames:
            logger.warning(f"Very small frame difference: {frames_diff} frames")
            frames_diff = min_frames
        
        # Convert frames to time (seconds)
        time_diff_seconds = frames_diff / self.fps
        
        # Calculate speed
        speed_ms = self.distance_meters / time_diff_seconds
        speed_kmh = speed_ms * 3.6
        
        # Apply reasonable limit
        if speed_kmh > 200:
            logger.warning(f"Unrealistic speed: {speed_kmh:.2f} km/h - capping at 200 km/h")
            speed_kmh = 200
        
        return speed_kmh

    def draw_lines(self, frame):
        """
        Draw entry and exit lines on the frame.
        
        Args:
            frame: The video frame
        """
        if hasattr(self, 'entry_line'):
            cv2.line(frame, tuple(self.entry_line[0]), tuple(self.entry_line[1]), (255, 0, 0), 2)
            cv2.putText(frame, "ENTRY LINE", (self.entry_line[0][0], self.entry_line[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                       
        if hasattr(self, 'exit_line'):
            cv2.line(frame, tuple(self.exit_line[0]), tuple(self.exit_line[1]), (0, 0, 255), 2)
            cv2.putText(frame, "EXIT LINE", (self.exit_line[0][0], self.exit_line[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def process_frame(self, frame, frame_number=0):
        """
        Process a single frame to detect vehicles and speed violations.
        
        Args:
            frame: Video frame to process
            frame_number: Current frame number
            
        Returns:
            tuple: (processed_frame, violations)
        """
        display_frame = frame.copy()
        
        # Get YOLO detection results
        results = self.detector.detect(frame)
        
        # Draw the entry and exit lines
        self.draw_lines(display_frame)
        
        # Convert boxes to list format for tracking
        vehicle_boxes = []
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            if class_id not in self.detector.class_map:
                continue
                
            # Only process vehicles
            if self.detector.is_vehicle(class_id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append([x1, y1, x2, y2])
        
        # Track vehicles with stable tracker
        tracked_vehicles = self.tracker.update(vehicle_boxes)
        
        # Draw status information
        draw_status_info(display_frame, frame_number, len(self.processed_vehicle_ids), "N/A", self.speed_limit)
        
        # Process vehicles and check for line crossings
        violations = []
        
        for bbox in tracked_vehicles:
            x1, y1, x2, y2, vehicle_id = bbox
            
            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_pos = (cx, cy)
            
            # Track vehicle trajectory
            if vehicle_id not in self.vehicle_trajectories:
                self.vehicle_trajectories[vehicle_id] = []
            
            self.vehicle_trajectories[vehicle_id].append((frame_number, current_pos))
            
            # Keep trajectory size manageable
            if len(self.vehicle_trajectories[vehicle_id]) > 30:
                self.vehicle_trajectories[vehicle_id].pop(0)
            
            # Get track object
            track = self.tracker.get_track(vehicle_id)
            if not track or track.hits < 3:  # Ensure stable tracking
                continue
            
            # Get previous position
            prev_pos = None
            if len(self.vehicle_trajectories[vehicle_id]) > 1:
                _, prev_pos = self.vehicle_trajectories[vehicle_id][-2]
            
            # Check for entry line crossing
            if hasattr(self, 'entry_line') and vehicle_id not in self.entry_timestamps:
                if prev_pos and self.check_line_crossing(prev_pos, current_pos, self.entry_line):
                    self.entry_timestamps[vehicle_id] = frame_number
                    logger.info(f"Vehicle {vehicle_id} crossed entry line at frame {frame_number}")
                    cv2.circle(display_frame, current_pos, 5, (255, 0, 0), -1)
            
            # Check for exit line crossing
            if (hasattr(self, 'exit_line') and 
                vehicle_id in self.entry_timestamps and 
                vehicle_id not in self.exit_timestamps):
                
                entry_frame = self.entry_timestamps[vehicle_id]
                frames_since_entry = frame_number - entry_frame
                
                # Minimum frames between lines
                MIN_FRAMES_BETWEEN_LINES = max(5, int(self.fps * 0.2))
                
                if frames_since_entry >= MIN_FRAMES_BETWEEN_LINES:
                    if prev_pos and self.check_line_crossing(prev_pos, current_pos, self.exit_line):
                        self.exit_timestamps[vehicle_id] = frame_number
                        
                        # Calculate speed
                        speed_kmh = self.calculate_speed(entry_frame, frame_number)
                        self.vehicle_speeds[vehicle_id] = speed_kmh
                        
                        logger.info(f"Vehicle {vehicle_id} speed: {speed_kmh:.2f} km/h")
                        cv2.circle(display_frame, current_pos, 5, (0, 0, 255), -1)
                        
                        # Check for violation
                        is_violation = speed_kmh > self.speed_limit
                        
                        if is_violation and vehicle_id not in self.processed_vehicle_ids:
                            # Save violation image
                            image_path = self.save_violation_image(
                                frame, [x1, y1, x2, y2], vehicle_id, 
                                speed_kmh, frame_number
                            )
                            
                            # Add to pending violations for plate processing
                            self.pending_violations.append({
                                'vehicle_id': vehicle_id,
                                'image_path': image_path,
                                'vehicle_bbox': [x1, y1, x2, y2],
                                'speed': speed_kmh,
                                'frame_number': frame_number,
                                'timestamp': datetime.now()
                            })
                            
                            # Add to processed list
                            self.processed_vehicle_ids.add(vehicle_id)
                            
                            # Create violation record
                            time_diff_seconds = frames_since_entry / self.fps
                            details = (f"Vehicle ID {vehicle_id} speed: {speed_kmh:.2f} km/h, "
                                      f"limit: {self.speed_limit} km/h, time: {time_diff_seconds:.3f}s")
                            
                            violation_id = self.db_manager.save_violation(
                                violation_type="speed",
                                license_plate=f"PENDING_{vehicle_id}",
                                confidence=0.0,
                                image_path=image_path,
                                traffic_light_color="N/A",
                                vehicle_id=vehicle_id,
                                details=details,
                                speed=speed_kmh
                            )
                            
                            if violation_id:
                                violations.append({
                                    "id": violation_id,
                                    "type": "speed",
                                    "plate": "PENDING",
                                    "vehicle_id": vehicle_id,
                                    "speed": speed_kmh,
                                    "conf": 0.0
                                })
            
            # Draw bounding box and info
            is_violation = vehicle_id in self.vehicle_speeds and self.vehicle_speeds[vehicle_id] > self.speed_limit
            
            if is_violation:
                # Violation box (red)
                draw_bounding_box(display_frame, (x1, y1, x2, y2), color=(0, 0, 255))
                speed_kmh = self.vehicle_speeds[vehicle_id]
                draw_vehicle_info(display_frame, x1, y1, vehicle_id, None, 0.0, is_violation=True, 
                                 speed=speed_kmh, speed_limit=self.speed_limit)
            else:
                # Regular box (green)
                draw_bounding_box(display_frame, (x1, y1, x2, y2), color=(0, 255, 0))
                
                # Show speed only if calculated
                if vehicle_id in self.vehicle_speeds:
                    speed_kmh = self.vehicle_speeds[vehicle_id]
                    draw_vehicle_info(display_frame, x1, y1, vehicle_id, None, 0.0, is_violation=False,
                                    speed=speed_kmh, speed_limit=self.speed_limit)
                else:
                    draw_vehicle_info(display_frame, x1, y1, vehicle_id, None, 0.0, is_violation=False)
        
        return display_frame, violations

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
                              f"Plate={plate_text}, Speed={violation['speed']:.1f} km/h, "
                              f"Confidence={confidence:.2f}")
                else:
                    logger.warning(f"Could not extract plate for vehicle {violation['vehicle_id']}")
                    
            except Exception as e:
                logger.error(f"Error processing violation for vehicle {violation['vehicle_id']}: {e}")
        
        # Clear pending violations
        self.pending_violations.clear()

    def process_video(self, video_path, output_path=None, entry_line=None, exit_line=None, in_thread=False):
        """
        Process a video file and detect speed violations.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            entry_line: Two points defining the entry line [(x1, y1), (x2, y2)]
            exit_line: Two points defining the exit line [(x1, y1), (x2, y2)]
            in_thread: Flag indicating if this method is running in a thread
            
        Returns:
            int: Total number of violations detected
        """
        # Reset tracking data
        self.interrupted = False
        self.entry_timestamps = {}
        self.exit_timestamps = {}
        self.vehicle_speeds = {}
        self.processed_vehicle_ids.clear()
        self.vehicle_trajectories = {}
        self.pending_violations.clear()
        
        # Only register signal handler in main thread
        if not in_thread:
            try:
                def signal_handler(sig, frame):
                    logger.info("Process interrupted by user! Saving output and exiting gracefully...")
                    self.interrupted = True
                
                signal.signal(signal.SIGINT, signal_handler)
            except ValueError as e:
                logger.warning(f"Could not register signal handler: {e}")
        
        # Set lines if provided
        if entry_line and exit_line:
            self.set_lines(entry_line, exit_line)
        
        # Validate we have both lines
        if not hasattr(self, 'entry_line') or not hasattr(self, 'exit_line'):
            logger.error("Entry and exit lines must be defined")
            raise ValueError("Entry and exit lines must be defined")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return 0
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video info: {frame_width}x{frame_height} at {self.fps} FPS, total frames: {total_frames}")
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (frame_width, frame_height))
        
        frame_number = 0
        total_violations = 0
        
        print(f"Processing video with speed limit {self.speed_limit} km/h and ENHANCED license plate recognition...")
        print("Press Ctrl+C to stop and save the output")
        
        try:
            while not self.interrupted:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info(f"End of video at frame {frame_number}")
                    break
                
                # Progress update
                if frame_number % 30 == 0:
                    print(f"Processing frame {frame_number}/{total_frames} | Violations: {total_violations}", end="\r")
                
                # Process frame
                processed_frame, violations = self.process_frame(frame, frame_number)
                
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
                print(f"\nProcess interrupted at frame {frame_number}")
            else:
                print(f"\nProcessing complete: {frame_number} frames, {total_violations} violations")
            
            # Log speed statistics
            if self.vehicle_speeds:
                speeds = list(self.vehicle_speeds.values())
                avg_speed = sum(speeds) / len(speeds)
                print(f"Average speed: {avg_speed:.1f} km/h")
            
            logger.info(f"Processed {frame_number} frames, detected {total_violations} violations")
            return total_violations