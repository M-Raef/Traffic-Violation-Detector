"""
Detector module for the traffic violation detector.

This module contains the main detector classes for red light and speed violations.
"""

from detector.red_light_detector import RedLightViolationDetector
from detector.speed_detector import SpeedViolationDetector

__all__ = [
    'RedLightViolationDetector',
    'SpeedViolationDetector'
]