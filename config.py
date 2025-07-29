"""
Configuration parameters for the traffic violation detector.
"""
import os
from datetime import datetime

# Paths and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "violations")
TODAY_DATE = datetime.now().strftime('%Y-%m-%d')
DAILY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_DATE)
PLATES_DIR = os.path.join(DAILY_OUTPUT_DIR, "plates")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DAILY_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)

# Database
DB_PATH = os.path.join(BASE_DIR, "traffic_violations.db")

# Detection parameters
YOLO_MODEL_PATH = "yolo11s.pt"  # Default model path
CONF_THRESHOLD = 0.4  # Confidence threshold for detections
PLATE_READING_INTERVAL = 5  # Try to read plates every N frames

# Vehicle tracking parameters
TRACKING_DISTANCE_THRESHOLD = 25  # Maximum distance to consider it's the same object

# ROI coordinates for red light violation detection
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# These are example values and should be adjusted for your specific video
DEFAULT_ROI_POINTS = [[284, 555], [1597, 560], [1597, 693], [136, 685]]

# Allowed characters for license plates
LICENSE_PLATE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Speed detection parameters
DEFAULT_SPEED_LIMIT = 40  # Default speed limit in km/h
SPEED_CALIBRATION = {
    "pixels_per_meter": 100.0,  # Default calibration: 100 pixels = 1 meter
    "reference_points": None,  # Will be calibrated or set via command line
    "known_distance": None
}
MIN_DETECTION_FRAMES = 5  # Minimum frames to track for speed calculation

# Logging
LOG_FILE = os.path.join(BASE_DIR, "traffic_violations.log")