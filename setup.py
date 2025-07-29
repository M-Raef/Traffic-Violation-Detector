"""
Setup script for Traffic Violation Detector project.

This script helps set up the project structure and dependencies.
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure."""
    print("Creating project directory structure...")
    
    directories = [
        "database",
        "models", 
        "detector",
        "utils",
        "gui",
        "violations",
        "violations/plates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "database/__init__.py",
        "models/__init__.py",
        "detector/__init__.py", 
        "utils/__init__.py",
        "gui/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            Path(init_file).touch()
            print(f"Created: {init_file}")

def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    except FileNotFoundError:
        print("requirements.txt not found. Please ensure it exists in the project root.")
        return False
    
    return True

def download_yolo_model():
    """Download YOLO model if it doesn't exist."""
    model_path = "yolo11s.pt"
    
    if os.path.exists(model_path):
        print(f"YOLO model already exists: {model_path}")
        return True
    
    print("Downloading YOLO model...")
    
    try:
        # Try using ultralytics to download
        from ultralytics import YOLO
        model = YOLO('yolo11s.pt')  # This will download if not present
        print("YOLO model downloaded successfully!")
        return True
    except ImportError:
        print("Ultralytics not installed. Installing dependencies first...")
        return False
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    config_file = "config.py"
    
    if os.path.exists(config_file):
        print("Config file already exists.")
        return
    
    print("Creating sample config file...")
    
    sample_config = '''"""
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
YOLO_MODEL_PATH = "yolo11s.pt"  # Model will be downloaded automatically
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
MIN_DETECTION_FRAMES = 5  # Minimum frames to track for speed calculation

# Logging
LOG_FILE = os.path.join(BASE_DIR, "traffic_violations.log")
'''
    
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    print(f"Created sample config file: {config_file}")

def check_system_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required.")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Check for tkinter
    try:
        import tkinter
        print("tkinter: Available")
    except ImportError:
        print("WARNING: tkinter not available. GUI will not work.")
        print("Install with: sudo apt-get install python3-tk (Ubuntu/Debian)")
    
    return True

def main():
    """Main setup function."""
    print("=" * 50)
    print("Traffic Violation Detector Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("System requirements not met. Please fix and try again.")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Create sample config
    create_sample_config()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please install manually.")
        return
    
    # Download YOLO model
    if not download_yolo_model():
        print("Failed to download YOLO model. You can download it manually.")
        print("Run: python -c \"from ultralytics import YOLO; YOLO('yolo11s.pt')\"")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Place your video files in the project directory")
    print("2. Run GUI: python traffic_detector_gui.py")
    print("3. Or use CLI: python main.py --help")
    print("\nFor more information, see the project documentation.")

if __name__ == "__main__":
    main()