Traffic Violation Detection System 🚦📸
An advanced, automated traffic violation detection system using modern computer vision and machine learning technologies. This system can detect red light and speed violations in real-time, identify vehicles and their license plates, collect visual evidence, and store data systematically.

Overview
This project uses the YOLOv11 deep learning model for vehicle and license plate detection, integrated with the SORT algorithm for object tracking. The system analyzes traffic light states using HSV color space and monitors vehicle timing through predefined Regions of Interest (ROI).

FeFeatures
🚗 Real-time Traffic Violation Detection
Red light violation detection

Speed violation detection

Automatic license plate recognition (ALPR)

🖥️ Graphical User Interface (GUI)
Easy setup and configuration

Real-time results display

Violation review and processing

Interactive ROI selection

💾 Data Management
SQLite database for violation storage

Automatic evidence collection

Violation image enhancement

Comprehensive reporting

GUI Details
1. Red Light Violation Detection Interface
Input Video: Select the input video file

Output Video: Choose the file where processed video will be saved

YOLO Model: Select the trained object detection model (.pt file)

Violation Type: Choose "Red Light Violation"

ROI (Region of Interest): Manually define the intersection area by selecting 4 corner points

The system analyzes vehicles entering the selected region during a red light and flags violations.

2. Speed Violation Detection Interface
Input/Output Video & YOLO Model: Same as above

Violation Type: Choose "Speed Violation"

ROI (Speed ROI): Define entry and exit lines

Speed Parameters:

Speed Limit (km/h): Set the maximum allowed speed

Distance Between Lines (m): Enter the distance between entry and exit lines

The system calculates each vehicle’s speed and flags vehicles exceeding the speed limit.

3. Violation Image Storage
Detected violations are saved as screenshots in categorized folders:

```bash
violations/red_light/ → Red light violations
violations/speed/     → Speed violations
```
These images are easily accessible through the "Results" tab in the GUI.

Installation

1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd traffic-violation-detector
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the Application
```bash
python traffic_detector_gui.py
```
YOLO model weights (yolo11s.pt) will be downloaded automatically or can be placed manually in the project root.

Project Structure
```bash
traffic-violation-detector/
├── main.py                 # CLI entry point
├── traffic_detector_gui.py # GUI launcher
├── config.py               # Configuration settings
├── models/                 # YOLO detection, license plate reader, etc.
├── detector/               # Violation detectors (red light, speed)
├── gui/                    # GUI components (ROI selector, app layout)
├── utils/                  # Utility functions & logger
├── output/                 # Processed videos
├── violations/             # Violation images
└── plates/                 # Cropped license plates
```

Database Schema
SQLite database stores violations:

```bash
CREATE TABLE violations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    violation_type TEXT,
    vehicle_id INTEGER,
    license_plate TEXT,
    confidence REAL,
    image_path TEXT,
    video_path TEXT
);
```

Acknowledgments
YOLO (Ultralytics)

EasyOCR

OpenCV

SORT Algorithm

