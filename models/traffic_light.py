"""
Traffic light analysis module for the traffic violation detector.
"""
import cv2
import numpy as np
from utils import logger

class TrafficLightAnalyzer:
    """
    Analyzes traffic lights in frames to determine their color.
    Uses HSV color thresholding to detect red, green, and yellow lights.
    """
    
    def __init__(self):
        """Initialize the traffic light analyzer with color thresholds."""
        # Define HSV color ranges for traffic lights
        # Red has two ranges in HSV (wraps around 180 degrees)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Green range
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([90, 255, 255])
        
        # Yellow range
        self.lower_yellow = np.array([15, 150, 150])
        self.upper_yellow = np.array([35, 255, 255])
        
        # Minimum ratio of colored pixels to be considered a light
        self.min_ratio = 0.03
        
        logger.info("Traffic light analyzer initialized")

    def get_light_color(self, light_img):
        """
        Determine the color of a traffic light.
        
        Args:
            light_img: Image of the traffic light
            
        Returns:
            tuple: (color, confidence) where color is 'red', 'green', 'yellow', or 'unknown'
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
            
            # Create masks for each color
            mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
            mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            
            # Count non-zero pixels in each mask
            red_count = cv2.countNonZero(mask_red)
            green_count = cv2.countNonZero(mask_green)
            yellow_count = cv2.countNonZero(mask_yellow)
            
            # Calculate total pixels in the image
            total_pixels = light_img.shape[0] * light_img.shape[1]
            
            # Calculate ratio of each color
            color_counts = {
                "red": red_count / total_pixels if total_pixels > 0 else 0,
                "green": green_count / total_pixels if total_pixels > 0 else 0,
                "yellow": yellow_count / total_pixels if total_pixels > 0 else 0
            }
            
            # Determine dominant color
            dominant_color = max(color_counts, key=color_counts.get)
            
            # Return dominant color if ratio is above threshold
            if color_counts[dominant_color] >= self.min_ratio:
                logger.debug(f"Detected traffic light color: {dominant_color} with confidence {color_counts[dominant_color]:.2f}")
                return dominant_color, color_counts[dominant_color]
            
            return "unknown", 0
            
        except Exception as e:
            logger.error(f"Error analyzing traffic light: {e}")
            return "unknown", 0

    def determine_dominant_light(self, traffic_lights):
        """
        Determine the dominant traffic light color from a list of detected lights.
        
        Args:
            traffic_lights: List of traffic light dictionaries with 'color' and 'conf' keys
            
        Returns:
            str: Dominant traffic light color ('red', 'green', 'yellow', or 'unknown')
        """
        if not traffic_lights:
            return "unknown"
            
        # Prioritize red lights for safety
        red_lights = [light for light in traffic_lights if light["color"] == "red"]
        if red_lights:
            return "red"
            
        # Then green lights
        green_lights = [light for light in traffic_lights if light["color"] == "green"]
        if green_lights:
            return "green"
            
        # Then yellow lights
        yellow_lights = [light for light in traffic_lights if light["color"] == "yellow"]
        if yellow_lights:
            return "yellow"
            
        # If no clear color is detected
        return "unknown"