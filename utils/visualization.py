"""
Enhanced utility functions for displaying detection results on frames.
Updated to clearly mark violation vehicles with red borders and improved
license plate display for enhanced reader integration.
"""
import cv2
import cvzone
import numpy as np
import logging
import os
from datetime import datetime

def setup_logger(log_file="traffic_violations.log", console_level=logging.INFO, file_level=logging.INFO):
    """
    Set up and configure logging for the application.
    
    Args:
        log_file (str): Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("traffic_detector")
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger

# Default logger instance
logger = setup_logger()

def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=3):
    """
    Draw a bounding box on a frame with enhanced visualization for violations.
    
    Args:
        frame: The frame to draw on
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        color: Box color in BGR format
        thickness: Line thickness
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def draw_vehicle_info(frame, x, y, vehicle_id, plate_text=None, plate_conf=0.0, 
                     is_violation=False, speed=None, speed_limit=None):
    """
    Draw vehicle ID and license plate info with enhanced visibility and violation marking.
    
    Args:
        frame: The frame to draw on
        x, y: Top-left corner coordinates of the vehicle bbox
        vehicle_id: ID of the tracked vehicle
        plate_text: Detected license plate text (or None)
        plate_conf: Confidence score for the license plate detection
        is_violation: Whether this vehicle is committing a violation
        speed: Current vehicle speed in km/h (or None)
        speed_limit: Speed limit in km/h (or None)
    """
    # Determine colors and styling based on violation status
    if is_violation:
        # Red styling for violations
        box_color = (0, 0, 255)
        text_color = (255, 255, 255)
        scale = 1.5
        thickness = 3
    else:
        # Green styling for normal vehicles
        box_color = (0, 150, 0)
        text_color = (255, 255, 255)
        scale = 1.0
        thickness = 2
    
    # Determine if speeding
    is_speeding = False
    if speed is not None and speed_limit is not None and speed > speed_limit:
        is_speeding = True
        box_color = (0, 0, 255)
    
    # Draw vehicle ID with violation status
    if is_violation:
        if is_speeding:
            id_label = f"ID: {vehicle_id} - SPEEDING"
        else:
            id_label = f"ID: {vehicle_id} - VIOLATION"
    else:
        id_label = f"ID: {vehicle_id}"
    
    # Draw ID label with enhanced visibility
    try:
        cvzone.putTextRect(frame, id_label, (x, y - 80), 
                         thickness=thickness, scale=scale, offset=10, 
                         colorR=box_color, colorT=text_color)
    except:
        # Fallback to regular cv2 text if cvzone fails
        cv2.putText(frame, id_label, (x, y - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, box_color, thickness)
    
    # Draw speed information if available
    if speed is not None:
        if is_speeding:
            speed_label = f"SPEED: {speed:.1f} km/h (LIMIT: {speed_limit} km/h)"
            speed_color = (0, 0, 255)  # Red for speeding
        else:
            speed_label = f"Speed: {speed:.1f} km/h"
            speed_color = box_color
            
        try:
            cvzone.putTextRect(frame, speed_label, (x, y - 50), 
                             thickness=thickness, scale=scale * 0.8, offset=8, 
                             colorR=speed_color, colorT=text_color)
        except:
            cv2.putText(frame, speed_label, (x, y - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, speed_color, thickness)
    
    # Draw license plate with enhanced formatting
    if plate_text:
        # Format confidence as percentage
        if plate_conf > 0:
            conf_text = f" ({plate_conf*100:.0f}%)"
        else:
            conf_text = ""
            
        plate_label = f"PLATE: {plate_text}{conf_text}"
        
        # Use larger, more prominent styling for license plates
        plate_color = box_color
        if is_violation:
            # Make violation plates even more prominent
            plate_scale = scale * 1.1
            plate_thickness = thickness + 1
        else:
            plate_scale = scale
            plate_thickness = thickness
            
        try:
            cvzone.putTextRect(frame, plate_label, (x, y - 20), 
                             thickness=plate_thickness, scale=plate_scale, offset=10, 
                             colorR=plate_color, colorT=text_color)
        except:
            cv2.putText(frame, plate_label, (x, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, plate_scale * 0.7, plate_color, plate_thickness)
    else:
        # Show "Reading..." for violation vehicles being processed
        if is_violation:
            try:
                cvzone.putTextRect(frame, "READING PLATE...", (x, y - 20), 
                                 thickness=thickness, scale=scale * 0.8, offset=8, 
                                 colorR=(0, 165, 255), colorT=text_color)  # Orange color
            except:
                cv2.putText(frame, "READING PLATE...", (x, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (0, 165, 255), thickness)
    
    return frame

def draw_traffic_light(frame, bbox, light_color):
    """
    Draw traffic light with color indication and enhanced visibility.
    
    Args:
        frame: The frame to draw on
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        light_color: Detected light color
    """
    x1, y1, x2, y2 = bbox
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "unknown": (128, 128, 128)
    }
    
    box_color = color_map.get(light_color, (128, 128, 128))
    
    # Draw thicker border for red lights (critical for violation detection)
    thickness = 4 if light_color == "red" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    
    # Draw label with background for better visibility
    label = f"Light: {light_color.upper()}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Background rectangle for text
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0] + 10, y1), box_color, -1)
    
    # Text
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def draw_roi_polygon(frame, points, color=(0, 255, 0), thickness=2):
    """
    Draw a region of interest polygon with enhanced visualization.
    
    Args:
        frame: The frame to draw on
        points: List of polygon points [[x1, y1], [x2, y2], ...]
        color: Line color in BGR format
        thickness: Line thickness
    """
    points_array = np.array(points, np.int32)
    
    # Draw the polygon border
    cv2.polylines(frame, [points_array], True, color, thickness)
    
    # Add ROI label
    if len(points) > 0:
        center_x = int(np.mean([p[0] for p in points]))
        center_y = int(np.mean([p[1] for p in points]))
        
        cv2.putText(frame, "ROI", (center_x - 15, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame

def draw_status_info(frame, frame_number, total_violations, light_color, speed_limit=None):
    """
    Draw enhanced status information on the frame.
    
    Args:
        frame: The frame to draw on
        frame_number: Current frame number
        total_violations: Total number of violations detected
        light_color: Current traffic light color
        speed_limit: Current speed limit in km/h
    """
    # Create status box background
    status_height = 140 if speed_limit else 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, status_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw status information with enhanced formatting
    y_offset = 30
    
    # Traffic light status (if applicable)
    if light_color != "N/A":
        light_color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), 
                          "yellow": (0, 255, 255), "unknown": (128, 128, 128)}.get(light_color, (128, 128, 128))
        cv2.putText(frame, f"Light: {light_color.upper()}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color_bgr, 2)
        y_offset += 25
    
    # Speed limit (for speed detection)
    if speed_limit:
        cv2.putText(frame, f"Speed Limit: {speed_limit} km/h",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # Frame number
    cv2.putText(frame, f"Frame: {frame_number}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    # Total violations with prominent styling
    violation_color = (0, 0, 255) if total_violations > 0 else (255, 255, 255)
    cv2.putText(frame, f"Violations: {total_violations}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, violation_color, 2)
    
    return frame