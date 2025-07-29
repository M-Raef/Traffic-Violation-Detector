"""
Models package initialization with enhanced license plate reader.
"""

# Import existing models
from .detector import YOLODetector
from .tracker import StableTracker
from .traffic_light import TrafficLightAnalyzer

# Try to import speed estimator if it exists
try:
    from .speed_estimator import SpeedEstimator
    HAS_SPEED_ESTIMATOR = True
except ImportError:
    HAS_SPEED_ESTIMATOR = False

# Import original license plate reader
from .license_plate import EnhancedLicensePlateReader

# Import NEW enhanced components
from .plate_enhancer import PlateImageEnhancer
from .enhanced_license_plate import SuperEnhancedLicensePlateReader

# Make all classes available when importing from models
if HAS_SPEED_ESTIMATOR:
    __all__ = [
        'YOLODetector',
        'StableTracker', 
        'TrafficLightAnalyzer',
        'SpeedEstimator',
        'EnhancedLicensePlateReader',      # Original
        'PlateImageEnhancer',              # NEW
        'SuperEnhancedLicensePlateReader'  # NEW Enhanced
    ]
else:
    __all__ = [
        'YOLODetector',
        'StableTracker', 
        'TrafficLightAnalyzer',
        'EnhancedLicensePlateReader',      # Original
        'PlateImageEnhancer',              # NEW
        'SuperEnhancedLicensePlateReader'  # NEW Enhanced
    ]