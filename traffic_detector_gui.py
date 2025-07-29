"""
Main launcher script for the Traffic Violation Detector GUI application.

This script launches the graphical user interface for the traffic violation detector,
allowing users to select the video, ROI, and violation type (red light or speed).

Usage:
    python traffic_detector_gui.py
"""
import os
import sys
import tkinter as tk

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GUI application
from gui.gui_app import TrafficViolationGUI

def main():
    """Main entry point for the GUI application."""
    # Create output directories if they don't exist
    try:
        import config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.DAILY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.PLATES_DIR, exist_ok=True)
        print(f"Output directories created/verified:")
        print(f"  - {config.OUTPUT_DIR}")
        print(f"  - {config.DAILY_OUTPUT_DIR}")
        print(f"  - {config.PLATES_DIR}")
    except Exception as e:
        print(f"Warning: Failed to create output directories: {e}")
    
    # Check if YOLO model exists
    try:
        import config
        model_path = config.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"Warning: YOLO model not found at {model_path}")
            # Try to find in common locations
            possible_paths = [
                os.path.join(os.getcwd(), model_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", model_path)
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found YOLO model at: {path}")
                    config.YOLO_MODEL_PATH = path
                    break
            else:
                print("WARNING: Could not find YOLO model. Please ensure it exists before running detection.")
    except Exception as e:
        print(f"Warning when checking model: {e}")
    
    # Launch the GUI
    root = tk.Tk()
    app = TrafficViolationGUI(root)
    
    # Set window icon (if available)
    try:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception as e:
        print(f"Warning: Failed to set window icon: {e}")
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()