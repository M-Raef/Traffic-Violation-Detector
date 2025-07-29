"""
Speed violation detection module for the traffic violation detector.
"""
import cv2
import numpy as np
import time
from datetime import datetime
import os
from utils import logger

class SpeedEstimator:
    """
    Estimates vehicle speeds based on their movement between frames.
    Uses real-world distance calibration to convert pixel movement to actual speed.
    """
    
    def __init__(self, fps=30, distance_calibration=None, speed_limit=40, min_detection_frames=5):
        """
        Initialize the speed estimator.
        
        Args:
            fps: Frames per second of the video
            distance_calibration: Calibration data to convert pixels to real-world distance
            speed_limit: Speed limit in km/h
            min_detection_frames: Minimum number of frames a vehicle must be tracked for speed estimation
        """
        self.fps = fps
        self.speed_limit = speed_limit  # km/h
        self.min_detection_frames = min_detection_frames
        
        # Track positions for each vehicle
        self.vehicle_positions = {}  # vehicle_id -> list of (frame_number, position)
        self.vehicle_speeds = {}     # vehicle_id -> current speed estimate
        self.violation_recorded = set()  # Set of vehicle IDs with recorded violations
        
        # Calibration for pixel to distance conversion
        # Default calibration assumes 100 pixels equals 1 meter
        self.distance_calibration = distance_calibration or {"pixels_per_meter": 100.0}
        
        # Stats for debugging and tuning
        self.speed_records = []
        
        logger.info(f"Speed estimator initialized with speed limit: {speed_limit} km/h")
    
    def set_speed_limit(self, speed_limit):
        """Set the speed limit in km/h."""
        self.speed_limit = speed_limit
        logger.info(f"Speed limit set to {speed_limit} km/h")
        
    def set_distance_calibration(self, calibration_data):
        """
        Set distance calibration data for pixel to real-world conversion.
        
        Args:
            calibration_data: Dictionary with calibration parameters
                              (e.g., pixels_per_meter, reference_points)
        """
        self.distance_calibration = calibration_data
        logger.info(f"Distance calibration set: {calibration_data}")
        
    def calibrate_from_reference_points(self, reference_points, known_distance):
        """
        Calibrate the system using reference points with known real-world distance.
        
        Args:
            reference_points: List of pairs of points on the image
            known_distance: Known real-world distance in meters between those points
        """
        total_pixel_distance = 0
        for p1, p2 in reference_points:
            pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_pixel_distance += pixel_dist
            
        avg_pixel_distance = total_pixel_distance / len(reference_points)
        pixels_per_meter = avg_pixel_distance / known_distance
        
        self.distance_calibration = {
            "pixels_per_meter": pixels_per_meter,
            "reference_points": reference_points,
            "known_distance": known_distance
        }
        
        logger.info(f"Calibrated: {pixels_per_meter} pixels per meter")
        return pixels_per_meter
    
    def update_vehicle_position(self, vehicle_id, position, frame_number):
        """
        Update the position history for a vehicle.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            position: Current position (x, y) - typically the center of the vehicle
            frame_number: Current frame number
        """
        if vehicle_id not in self.vehicle_positions:
            self.vehicle_positions[vehicle_id] = []
            
        self.vehicle_positions[vehicle_id].append((frame_number, position))
        
        # Keep only the last 30 positions to limit memory usage
        if len(self.vehicle_positions[vehicle_id]) > 30:
            self.vehicle_positions[vehicle_id].pop(0)
    
    def calculate_speed(self, vehicle_id):
        """
        Calculate the current speed of a vehicle based on its position history.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            
        Returns:
            float: Estimated speed in km/h or None if not enough data
        """
        positions = self.vehicle_positions.get(vehicle_id, [])
        
        # Need at least min_detection_frames positions for reliable speed estimation
        if len(positions) < self.min_detection_frames:
            return None
            
        # Get the positions from frames that are most distant but within reasonable time
        # This improves accuracy compared to using consecutive frames
        optimal_frame_distance = min(10, len(positions) // 2)
        if optimal_frame_distance < 2:
            optimal_frame_distance = 2
            
        # Calculate speed using distance over time
        start_frame_num, start_pos = positions[-optimal_frame_distance]
        end_frame_num, end_pos = positions[-1]
        
        # Calculate pixel distance
        pixel_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Convert to real-world distance (meters)
        real_distance = pixel_distance / self.distance_calibration["pixels_per_meter"]
        
        # Calculate time difference in seconds
        time_diff = (end_frame_num - start_frame_num) / self.fps
        
        if time_diff <= 0:
            return None
            
        # Calculate speed in m/s
        speed_ms = real_distance / time_diff
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        # Update the vehicle's speed
        self.vehicle_speeds[vehicle_id] = speed_kmh
        
        # Add to speed records for analysis
        self.speed_records.append((vehicle_id, speed_kmh))
        
        return speed_kmh
    
    def check_speed_violation(self, vehicle_id):
        """
        Check if a vehicle is exceeding the speed limit.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            
        Returns:
            tuple: (is_violation, speed) or (False, None) if speed can't be determined
        """
        speed = self.calculate_speed(vehicle_id)
        
        if speed is None:
            return False, None
            
        is_violation = speed > self.speed_limit
        
        # Log potential violation for debugging
        if is_violation:
            logger.debug(f"Vehicle {vehicle_id} speed: {speed:.1f} km/h (limit: {self.speed_limit} km/h)")
            
        return is_violation, speed
    
    def mark_violation_recorded(self, vehicle_id):
        """
        Mark a vehicle's speed violation as recorded to avoid duplicate records.
        
        Args:
            vehicle_id: ID of the tracked vehicle
        """
        self.violation_recorded.add(vehicle_id)
    
    def is_violation_recorded(self, vehicle_id):
        """
        Check if a vehicle's speed violation has already been recorded.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            
        Returns:
            bool: True if the violation was already recorded, False otherwise
        """
        return vehicle_id in self.violation_recorded
    
    def reset_violation_record(self, vehicle_id=None):
        """
        Reset the violation record for a vehicle or all vehicles.
        
        Args:
            vehicle_id: ID of the tracked vehicle, or None to reset all
        """
        if vehicle_id is None:
            self.violation_recorded = set()
        elif vehicle_id in self.violation_recorded:
            self.violation_recorded.remove(vehicle_id)
    
    def get_average_speed(self, vehicle_id, frames=10):
        """
        Get the average speed of a vehicle over the last N position updates.
        This is more stable than instantaneous speed calculation.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            frames: Number of frames to consider for the average
            
        Returns:
            float: Average speed in km/h or None if not enough data
        """
        positions = self.vehicle_positions.get(vehicle_id, [])
        
        if len(positions) < frames:
            return None
            
        recent_positions = positions[-frames:]
        
        start_frame_num, start_pos = recent_positions[0]
        end_frame_num, end_pos = recent_positions[-1]
        
        # Calculate pixel distance
        pixel_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Convert to real-world distance (meters)
        real_distance = pixel_distance / self.distance_calibration["pixels_per_meter"]
        
        # Calculate time difference in seconds
        time_diff = (end_frame_num - start_frame_num) / self.fps
        
        if time_diff <= 0:
            return None
            
        # Calculate speed in m/s
        speed_ms = real_distance / time_diff
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
        
    def cleanup_inactive_vehicles(self, current_frame, inactive_threshold=30):
        """
        Remove tracking data for vehicles that haven't been seen recently.
        
        Args:
            current_frame: Current frame number
            inactive_threshold: Remove vehicles not seen for this many frames
        """
        vehicles_to_remove = []
        
        for vehicle_id, positions in self.vehicle_positions.items():
            if positions and (current_frame - positions[-1][0]) > inactive_threshold:
                vehicles_to_remove.append(vehicle_id)
                
        for vehicle_id in vehicles_to_remove:
            self.vehicle_positions.pop(vehicle_id, None)
            self.vehicle_speeds.pop(vehicle_id, None)
            self.violation_recorded.discard(vehicle_id)
            
        return len(vehicles_to_remove)