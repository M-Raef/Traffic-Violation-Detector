"""
Enhanced Vehicle tracking module with stable ID assignment.
"""
import numpy as np
from collections import deque
from utils import logger

class StableTracker:
    """
    Enhanced tracker with stable vehicle IDs using multiple features
    and trajectory prediction.
    """
    def __init__(self, distance_threshold=50, max_lost_frames=10, 
                 min_hits=3, max_trajectory_len=30):
        # Tracking parameters
        self.distance_threshold = distance_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self.max_trajectory_len = max_trajectory_len
        
        # Track storage
        self.tracks = {}  # id -> Track object
        self.next_id = 0
        self.frame_count = 0
        
        logger.info(f"StableTracker initialized with distance_threshold={distance_threshold}, "
                   f"max_lost_frames={max_lost_frames}")

    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of [x1, y1, x2, y2] bounding boxes
            
        Returns:
            List of [x1, y1, x2, y2, track_id] for confirmed tracks
        """
        self.frame_count += 1
        
        # Convert detections to Track format
        measurements = []
        for det in detections:
            x1, y1, x2, y2 = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            measurements.append({
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy],
                'size': [w, h]
            })
        
        # Predict new positions for existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].predict()
        
        # Match measurements to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            measurements, self.tracks)
        
        # Update matched tracks
        for m in matched:
            track_id = m[1]
            measurement = measurements[m[0]]
            self.tracks[track_id].update(measurement)
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            track = Track(self.next_id, measurements[i])
            self.tracks[self.next_id] = track
            self.next_id += 1
        
        # Handle unmatched tracks (increment age)
        for track_id in unmatched_trks:
            self.tracks[track_id].mark_missed()
        
        # Remove dead tracks
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_lost_frames:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        # Return confirmed tracks
        results = []
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits and track.time_since_update == 0:
                bbox = track.get_state()
                results.append([int(bbox[0]), int(bbox[1]), 
                              int(bbox[2]), int(bbox[3]), track_id])
        
        return results
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """
        Associate detections to tracked objects using Hungarian algorithm.
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(tracks.keys())
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        det_idx = 0
        for det in detections:
            track_idx = 0
            for track_id, track in tracks.items():
                # Calculate multiple distance metrics
                
                # 1. Center distance
                pred_center = track.get_predicted_center()
                center_dist = np.sqrt((det['center'][0] - pred_center[0])**2 + 
                                    (det['center'][1] - pred_center[1])**2)
                
                # 2. Size difference
                pred_size = track.get_size()
                size_diff = abs(det['size'][0] - pred_size[0]) + abs(det['size'][1] - pred_size[1])
                
                # 3. IoU (Intersection over Union)
                iou = self._calculate_iou(det['bbox'], track.get_predicted_bbox())
                
                # Combine metrics (lower is better)
                cost = center_dist + 0.5 * size_diff - 100 * iou
                
                cost_matrix[det_idx, track_idx] = cost
                track_idx += 1
            det_idx += 1
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_dets = []
        unmatched_trks = []
        
        track_ids = list(tracks.keys())
        
        # Check matched assignments
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.distance_threshold:
                matched.append([row, track_ids[col]])
            else:
                unmatched_dets.append(row)
                unmatched_trks.append(track_ids[col])
        
        # Find unmatched detections
        for i in range(len(detections)):
            if i not in row_indices:
                unmatched_dets.append(i)
        
        # Find unmatched tracks
        for i, track_id in enumerate(track_ids):
            if i not in col_indices:
                unmatched_trks.append(track_id)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def get_track(self, track_id):
        """Get track object by ID."""
        return self.tracks.get(track_id)
    
    def update_license_plate(self, track_id, plate_text, confidence):
        """Update license plate for a track."""
        if track_id in self.tracks:
            return self.tracks[track_id].update_license_plate(plate_text, confidence)
        return False
    
    def get_license_plate(self, track_id):
        """Get license plate for a track."""
        if track_id in self.tracks:
            return self.tracks[track_id].get_license_plate()
        return None, 0.0


class Track:
    """
    Represents a single tracked object with trajectory and predictions.
    """
    def __init__(self, track_id, initial_measurement):
        self.id = track_id
        self.hits = 1
        self.time_since_update = 0
        
        # State: [x1, y1, x2, y2, vx, vy]
        self.state = np.array([
            initial_measurement['bbox'][0],
            initial_measurement['bbox'][1],
            initial_measurement['bbox'][2],
            initial_measurement['bbox'][3],
            0, 0  # Initial velocity
        ])
        
        # Trajectory history
        self.trajectory = deque(maxlen=30)
        self.trajectory.append(initial_measurement['center'])
        
        # Size history for stability
        self.sizes = deque(maxlen=10)
        self.sizes.append(initial_measurement['size'])
        
        # License plate info
        self.license_plate = None
        self.plate_confidence = 0.0
        
        # Prediction
        self.predicted_state = self.state.copy()
    
    def predict(self):
        """Predict next state using constant velocity model."""
        # Simple constant velocity prediction
        self.predicted_state = self.state.copy()
        self.predicted_state[0] += self.state[4]  # x1 + vx
        self.predicted_state[1] += self.state[5]  # y1 + vy
        self.predicted_state[2] += self.state[4]  # x2 + vx
        self.predicted_state[3] += self.state[5]  # y2 + vy
    
    def update(self, measurement):
        """Update track with new measurement."""
        # Calculate velocity
        if len(self.trajectory) > 0:
            prev_center = self.trajectory[-1]
            curr_center = measurement['center']
            self.state[4] = curr_center[0] - prev_center[0]  # vx
            self.state[5] = curr_center[1] - prev_center[1]  # vy
        
        # Update state
        self.state[0:4] = measurement['bbox']
        
        # Update history
        self.trajectory.append(measurement['center'])
        self.sizes.append(measurement['size'])
        
        # Update counters
        self.hits += 1
        self.time_since_update = 0
    
    def mark_missed(self):
        """Mark that this track was not matched to a detection."""
        self.time_since_update += 1
    
    def get_state(self):
        """Get current bounding box."""
        return self.state[0:4]
    
    def get_predicted_bbox(self):
        """Get predicted bounding box."""
        return self.predicted_state[0:4]
    
    def get_predicted_center(self):
        """Get predicted center point."""
        bbox = self.get_predicted_bbox()
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def get_size(self):
        """Get average size for stability."""
        if len(self.sizes) == 0:
            return [0, 0]
        avg_w = sum(s[0] for s in self.sizes) / len(self.sizes)
        avg_h = sum(s[1] for s in self.sizes) / len(self.sizes)
        return [avg_w, avg_h]
    
    def update_license_plate(self, plate_text, confidence):
        """Update license plate if confidence is higher."""
        if self.license_plate is None or confidence > self.plate_confidence:
            self.license_plate = plate_text
            self.plate_confidence = confidence
            return True
        return False
    
    def get_license_plate(self):
        """Get license plate info."""
        return self.license_plate, self.plate_confidence