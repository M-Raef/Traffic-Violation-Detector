"""
Traffic Violation Detector - Main Entry Point

Updated to work with enhanced detectors that only read plates from violation vehicles.
"""
import os
import sys
import argparse
import torch
import cv2
import time

# Import configuration
import config

# Import utility modules
from utils import logger

# Import detector modules - using your existing structure
try:
    from detector.red_light_detector import RedLightViolationDetector
    from detector.speed_detector import SpeedViolationDetector
except ImportError as e:
    logger.error(f"Import error: {e}")
    print(f"Error importing detectors: {e}")
    print("Make sure the enhanced detector files are in the detector/ folder")
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Traffic Violation Detection with License Plate Recognition')
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to input video file')
    
    # Violation type
    parser.add_argument('--type', type=str, choices=['redlight', 'speed'], default='redlight',
                        help='Type of violation to detect (redlight or speed)')
    
    # Common optional arguments
    parser.add_argument('--output', type=str, default="enhanced_violations.avi", 
                        help='Path to output video file')
    parser.add_argument('--model', type=str, default=config.YOLO_MODEL_PATH, 
                        help='Path to YOLO model')
    parser.add_argument('--plate-interval', type=int, default=config.PLATE_READING_INTERVAL, 
                        help='Interval for license plate reading (frames)')
    
    # Red light violation arguments
    parser.add_argument('--roi', type=str, default=None, 
                        help='ROI points for red light detection in format "x1,y1,x2,y2,x3,y3,x4,y4"')
    parser.add_argument('--no-roi', action='store_true', 
                        help='Disable ROI detection for red light (use entire frame)')
    
    # Speed violation arguments
    parser.add_argument('--entry', type=str, default=None,
                        help='Entry line points for speed detection in format "x1,y1,x2,y2"')
    parser.add_argument('--exit', type=str, default=None,
                        help='Exit line points for speed detection in format "x1,y1,x2,y2"')
    parser.add_argument('--speed-limit', type=float, default=30.0,
                        help='Speed limit in km/h for speed violation detection')
    parser.add_argument('--distance', type=float, default=5.0,
                        help='Distance between entry and exit lines in meters')
    
    return parser.parse_args()


def run_detection(args):
    """Run the appropriate enhanced detector based on arguments."""
    # Create output directories
    try:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.DAILY_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.PLATES_DIR, exist_ok=True)
        print(f"Output directories verified/created")
    except Exception as e:
        print(f"Warning: Could not create output directories: {e}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print("🚀 Enhanced license plate reading will use GPU acceleration")
    else:
        logger.info("CUDA not available, using CPU")
        print("⚠️ Running on CPU - detection may be slower")
    
    # Verify video file
    if not os.path.exists(args.video):
        logger.error(f"Video file does not exist: {args.video}")
        print(f"❌ ERROR: Video file not found: {args.video}")
        return 0
    
    # Print video information
    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {args.video}")
            print(f"❌ ERROR: Cannot open video file: {args.video}")
            return 0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 0

    # Initialize the appropriate enhanced detector
    try:
        if args.type == 'redlight':
            # Parse ROI points for red light detection
            roi_points = None
            
            if args.roi:
                try:
                    values = list(map(int, args.roi.split(',')))
                    if len(values) >= 8:
                        points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
                        roi_points = points[:4]  # Take first 4 points
                        print(f"🎯 ROI points: {roi_points}")
                    else:
                        print(f"❌ Not enough ROI points: need at least 8 values")
                        return 0
                except Exception as e:
                    print(f"❌ Error parsing ROI points: {e}")
                    return 0
            
            # Initialize enhanced red light detector
            print("🔴 Initializing ENHANCED red light detector...")
            print("✨ Features: Advanced license plate reading, violation-only processing")
            detector = RedLightViolationDetector(model_path=args.model)
            
            # Run detection
            print(f"🚦 Starting enhanced red light detection")
            print("🔍 License plates will ONLY be read from vehicles with RED BORDERS (violations)")
            total_violations = detector.process_video(
                video_path=args.video,
                output_path=args.output,
                roi_points=roi_points
            )
            
        else:  # speed detection
            # Parse entry and exit lines for speed detection
            entry_line = None
            exit_line = None
            
            if args.entry:
                try:
                    values = list(map(int, args.entry.split(',')))
                    if len(values) == 4:
                        entry_line = [(values[0], values[1]), (values[2], values[3])]
                        print(f"🟦 Entry line: {entry_line}")
                    else:
                        print(f"❌ Invalid entry line: need exactly 4 values")
                        return 0
                except Exception as e:
                    print(f"❌ Error parsing entry line: {e}")
                    return 0
            else:
                print("❌ No entry line specified")
                return 0
                
            if args.exit:
                try:
                    values = list(map(int, args.exit.split(',')))
                    if len(values) == 4:
                        exit_line = [(values[0], values[1]), (values[2], values[3])]
                        print(f"🟥 Exit line: {exit_line}")
                    else:
                        print(f"❌ Invalid exit line: need exactly 4 values")
                        return 0
                except Exception as e:
                    print(f"❌ Error parsing exit line: {e}")
                    return 0
            else:
                print("❌ No exit line specified")
                return 0
            
            # Initialize enhanced speed detector
            print("🏎️ Initializing ENHANCED speed detector...")
            print("✨ Features: Advanced license plate reading, violation-only processing")
            print(f"⚡ Speed limit: {args.speed_limit} km/h, Distance: {args.distance} meters")
            detector = SpeedViolationDetector(
                model_path=args.model,
                speed_limit=args.speed_limit,
                distance_meters=args.distance
            )
            
            # Run detection
            print("🏁 Starting enhanced speed detection...")
            print("🔍 License plates will ONLY be read from vehicles EXCEEDING SPEED LIMIT (red borders)")
            total_violations = detector.process_video(
                video_path=args.video,
                output_path=args.output,
                entry_line=entry_line,
                exit_line=exit_line
            )
            
        print(f"\n{'='*60}")
        print(f"🎉 ENHANCED DETECTION COMPLETED")
        print(f"{'='*60}")
        print(f"Detection type: {args.type.upper()}")
        print(f"Total violations detected: {total_violations}")
        
        # Verify output file was created
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # Size in MB
            print(f"✅ Output video created: {args.output} ({file_size:.2f} MB)")
        else:
            print(f"⚠️ WARNING: Output video not found at {args.output}")
        
        print(f"\n🚀 Enhanced Features Used:")
        print(f"  ✅ Advanced license plate detection with multiple OCR methods")
        print(f"  ✅ Violation-only plate reading (red borders only)")
        print(f"  ✅ Pattern matching validation")
        print(f"  ✅ Enhanced visualization")
        
        print(f"\n📁 Output Files:")
        print(f"  🎬 Video: {args.output}")
        print(f"  📸 Violation images: {config.DAILY_OUTPUT_DIR}")
        print(f"  🔢 License plate crops: {config.PLATES_DIR}")
        print(f"  💾 Database: {config.DB_PATH}")
            
        return total_violations
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 0


def run_from_gui(params):
    """Run enhanced detection from GUI with parameters dictionary."""
    # This function maintains compatibility with your existing GUI
    class Args:
        pass
    
    args = Args()
    
    # Set basic parameters
    args.video = params.get("video_path", "")
    args.output = params.get("output_path", "enhanced_violations.avi")
    args.model = params.get("model_path", config.YOLO_MODEL_PATH)
    args.plate_interval = params.get("plate_reading_interval", config.PLATE_READING_INTERVAL)
    
    # Set violation type specific parameters
    if params.get("violation_type") == "redlight":
        args.type = "redlight"
        args.roi = None
        args.no_roi = False
        
        # Extract ROI points
        roi_data = params.get("roi_data", {})
        if roi_data and "points" in roi_data:
            points = roi_data["points"]
            if points and len(points) == 4:
                args.roi = ",".join([f"{p[0]},{p[1]}" for p in points])
                print(f"🎯 GUI ROI points: {args.roi}")
    else:
        args.type = "speed"
        args.speed_limit = params.get("speed_limit", 30.0)
        args.distance = params.get("distance_meters", 5.0)
        
        # Extract entry and exit lines
        roi_data = params.get("roi_data", {})
        entry_line = roi_data.get("entry_line", [])
        exit_line = roi_data.get("exit_line", [])
        
        if entry_line and len(entry_line) == 2:
            args.entry = f"{entry_line[0][0]},{entry_line[0][1]},{entry_line[1][0]},{entry_line[1][1]}"
        else:
            args.entry = None
            
        if exit_line and len(exit_line) == 2:
            args.exit = f"{exit_line[0][0]},{exit_line[0][1]},{exit_line[1][0]},{exit_line[1][1]}"
        else:
            args.exit = None
    
    print(f"🖥️ Starting enhanced detection from GUI with type: {args.type}")
    
    # Run enhanced detection
    try:
        return run_detection(args)
    except Exception as e:
        print(f"❌ Error in enhanced detection: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main entry point for command-line execution."""
    print("🚗 Enhanced Traffic Violation Detector")
    print("="*50)
    print("✨ Features:")
    print("  🔍 Advanced license plate detection and reading")
    print("  🧠 Multiple OCR preprocessing techniques") 
    print("  🎯 Violation-only plate processing (red borders)")
    print("  ✅ Pattern matching and validation")
    print("  🎨 Enhanced visualization")
    print()
        
    args = parse_arguments()
    
    start_time = time.time()
    violations_detected = run_detection(args)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"\n⏱️ Processing completed in {processing_time:.1f} seconds")
    
    if violations_detected > 0:
        print(f"⚠️ {violations_detected} violations detected and processed with enhanced accuracy")
    else:
        print("✅ No violations detected")


if __name__ == "__main__":
    main()