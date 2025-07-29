# Traffic Violation Detection System 🚦📸

An advanced, automated traffic violation detection system using modern computer vision and machine learning technologies. This system detects **red light** and **speed** violations in real time, identifies vehicles and their license plates, collects visual evidence, and stores data systematically.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [GUI Details](#gui-details)
  - [Red Light Violation Interface](#red-light-violation-interface)
  - [Speed Violation Interface](#speed-violation-interface)
  - [Violation Image Storage](#violation-image-storage)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project uses the **YOLOv11** deep learning model for vehicle and license plate detection, integrated with the **SORT** algorithm for object tracking.  
The system analyzes traffic light states using **HSV color space** and monitors vehicle timing through predefined **Regions of Interest (ROI)**.

---

## Features
### 🚗 Real-time Traffic Violation Detection
- Red light violation detection  
- Speed violation detection  
- Automatic license plate recognition (ALPR)

### 🖥️ Graphical User Interface (GUI)
- Easy setup and configuration  
- Real-time results display  
- Violation review and processing  
- Interactive ROI selection

### 💾 Data Management
- SQLite database for violation storage  
- Automatic evidence collection  
- Violation image enhancement  
- Comprehensive reporting

---

## GUI Details

### Red Light Violation Interface
- **Input Video**: Select the input video file.  
- **Output Video**: Choose the file where the processed video is saved.  
- **YOLO Model**: Select the trained object detection model (`.pt`).  
- **Violation Type**: Choose **“Red Light Violation”**.  
- **ROI (Region of Interest)**: Manually define the intersection area by selecting **4 corner points**.

> The system analyzes vehicles entering the selected region during a red light and flags violations.

---

### Speed Violation Interface
- **Input/Output Video & YOLO Model**: Same selection as above.  
- **Violation Type**: Choose **“Speed Violation”**.  
- **ROI (Speed ROI)**: Define **entry** and **exit** lines.  
- **Speed Parameters**:  
  - **Speed Limit (km/h)**: Set the maximum allowed speed.  
  - **Distance Between Lines (m)**: Enter the distance between entry and exit lines.

> The system calculates a vehicle’s speed from the time taken to travel the given distance and flags vehicles exceeding the speed limit.

---

### Violation Image Storage
Detected violations are saved as screenshots in categorized folders:
```bash
violations/red_light/ → Red light violations
violations/speed/     → Speed violations
```

You can also browse these images via the **Results** tab in the GUI.

---

## Installation

### 1) Clone the Repository
```bash
git clone https://github.com/M-Raef/Traffic-Violation-Detector.git
cd traffic-violation-detector
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the Application
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
YOLO (Ultralytics) – Object detection

EasyOCR – License plate recognition

OpenCV – Computer vision operations

SORT Algorithm – Multi-object tracking

