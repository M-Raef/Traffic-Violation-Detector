"""
Object detection models for the traffic violation detector.
"""
import torch
from ultralytics import YOLO
from utils import logger

class DetectionModel:
    """Base class for detection models."""
    
    def __init__(self):
        pass

    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: Image frame to process
            
        Returns:
            Detection results
        """
        raise NotImplementedError("Subclasses must implement detect()")

class YOLODetector(DetectionModel):
    """
    YOLO-based object detector.
    Detects vehicles, traffic lights, and other objects in frames.
    """
    
    def __init__(self, model_path="yolo11s.pt", conf_threshold=0.5):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        super().__init__()
        # Load the model
        self.model = YOLO(model_path)
        
        # Set model parameters for better performance
        if torch.cuda.is_available():
            self.model.to('cuda')  # Use GPU if available
            logger.info(f"YOLO model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            # For CPU optimization
            self.model.to('cpu')
            logger.info("YOLO model loaded on CPU")
            
        self.conf_threshold = conf_threshold
        logger.info(f"YOLO model loaded from {model_path} with confidence threshold {conf_threshold}")
        
        # Common COCO class mapping
        self.class_map = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic light"
        }
        
        # Try to load class list from file
        try:
            with open("coco.txt", "r") as f:
                self.class_list = f.read().strip().split("\n")
                logger.info(f"Loaded {len(self.class_list)} classes from coco.txt")
        except Exception as e:
            logger.warning(f"Could not load class list from file: {e}")
            self.class_list = []

    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: Image frame to process
            
        Returns:
            Detection results from YOLO
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results[0]
    
    def is_vehicle(self, class_id):
        """
        Check if a class ID represents a vehicle.
        
        Args:
            class_id: Class ID from YOLO detection
            
        Returns:
            bool: True if the class is a vehicle, False otherwise
        """
        return class_id in [2, 3, 5, 7]  # car, motorcycle, bus, truck