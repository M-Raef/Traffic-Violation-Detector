"""
Database management for the traffic violation detector.
"""
import sqlite3
from datetime import datetime
import os
import json
from utils.logger import logger

class DatabaseManager:
    def __init__(self, db_path="traffic_violations.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self._db_path = db_path
        self.init_db()
        logger.info(f"Database initialized at {db_path}")

    @property
    def db_path(self):
        return self._db_path

    @db_path.setter
    def db_path(self, value):
        self._db_path = value

    def init_db(self):
        """Initialize the database tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # Create violations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_type TEXT NOT NULL,
            license_plate TEXT,
            timestamp TEXT NOT NULL,
            confidence REAL,
            image_path TEXT,
            traffic_light_color TEXT,
            vehicle_id INTEGER,
            violation_details TEXT,
            speed REAL
        )
        ''')
        
        # Create license plates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS license_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            license_plate TEXT,
            confidence REAL,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            frame_number INTEGER
        )
        ''')
        
        # Create configuration table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            roi_data TEXT NOT NULL,
            speed_limit REAL,
            distance_meters REAL,
            model_path TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()

    def save_violation(self, violation_type, license_plate, confidence, image_path,
                       traffic_light_color, vehicle_id=None, details="", speed=None):
        """
        Save a traffic violation to the database.
        
        Args:
            violation_type: Type of violation (e.g., 'red_light', 'speed')
            license_plate: License plate text of the vehicle
            confidence: Confidence score for the license plate detection
            image_path: Path to the saved image of the violation
            traffic_light_color: Color of the traffic light during violation
            vehicle_id: ID of the tracked vehicle
            details: Additional details about the violation
            speed: Vehicle speed in km/h (for speed violations)
            
        Returns:
            int: ID of the saved violation or None if an error occurred
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
            INSERT INTO violations (
                violation_type, license_plate, timestamp, confidence, 
                image_path, traffic_light_color, vehicle_id, violation_details, speed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (violation_type, license_plate, timestamp, confidence,
                  image_path, traffic_light_color, vehicle_id, details, speed))
            
            conn.commit()
            violation_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Violation saved: Type={violation_type}, Plate={license_plate}, ID={violation_id}")
            return violation_id
        except Exception as e:
            logger.error(f"Error saving violation to database: {e}")
            return None
            
    def save_license_plate(self, vehicle_id, license_plate, confidence, image_path=None, frame_number=None):
        """
        Save a license plate detection to the database.
        
        Args:
            vehicle_id: ID of the tracked vehicle
            license_plate: License plate text
            confidence: Confidence score for the detection
            image_path: Path to the saved image of the license plate
            frame_number: Frame number where the plate was detected
            
        Returns:
            int: ID of the saved license plate or None if an error occurred
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
            INSERT INTO license_plates (
                vehicle_id, license_plate, confidence, timestamp, image_path, frame_number
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (vehicle_id, license_plate, confidence, timestamp, image_path, frame_number))
            conn.commit()
            plate_id = cursor.lastrowid
            conn.close()
            logger.info(f"License plate saved: ID={vehicle_id}, Plate={license_plate}")
            return plate_id
        except Exception as e:
            logger.error(f"Error saving license plate to database: {e}")
            return None
    
    def save_configuration(self, name, violation_type, roi_data, speed_limit=None, distance_meters=None, model_path=None):
        """
        Save a detector configuration to the database.
        
        Args:
            name: Configuration name
            violation_type: Type of violation ('redlight' or 'speed')
            roi_data: JSON string with ROI data
            speed_limit: Speed limit in km/h (for speed violations)
            distance_meters: Distance between entry and exit lines in meters (for speed violations)
            model_path: Path to the YOLO model
            
        Returns:
            int: ID of the saved configuration or None if an error occurred
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert roi_data to JSON string if it's not already
            if isinstance(roi_data, dict):
                roi_data = json.dumps(roi_data)
            
            cursor.execute('''
            INSERT INTO configurations (
                name, violation_type, roi_data, speed_limit, distance_meters, model_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, violation_type, roi_data, speed_limit, distance_meters, model_path, timestamp))
            
            conn.commit()
            config_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Configuration saved: Name={name}, Type={violation_type}, ID={config_id}")
            return config_id
        except Exception as e:
            logger.error(f"Error saving configuration to database: {e}")
            return None
            
    def get_all_violations(self, limit=100, violation_type=None):
        """
        Get all violations from the database.
        
        Args:
            limit: Maximum number of violations to return
            violation_type: Filter by violation type (optional)
            
        Returns:
            list: List of violation dictionaries
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            query = "SELECT * FROM violations"
            params = []
            
            if violation_type:
                query += " WHERE violation_type = ?"
                params.append(violation_type)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            violations = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return violations
        except Exception as e:
            logger.error(f"Error retrieving violations from database: {e}")
            return []
            
    def get_all_license_plates(self, limit=100):
        """
        Get all license plates from the database.
        
        Args:
            limit: Maximum number of license plates to return
            
        Returns:
            list: List of license plate dictionaries
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM license_plates ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            plates = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return plates
        except Exception as e:
            logger.error(f"Error retrieving license plates from database: {e}")
            return []
    
    def get_all_configurations(self, violation_type=None):
        """
        Get all saved configurations from the database.
        
        Args:
            violation_type: Filter by violation type (optional)
            
        Returns:
            list: List of configuration dictionaries
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            query = "SELECT * FROM configurations"
            params = []
            
            if violation_type:
                query += " WHERE violation_type = ?"
                params.append(violation_type)
                
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            configs = []
            
            for row in cursor.fetchall():
                config = dict(row)
                
                # Parse ROI data JSON
                try:
                    config['roi_data'] = json.loads(config['roi_data'])
                except:
                    pass
                    
                configs.append(config)
                
            conn.close()
            return configs
        except Exception as e:
            logger.error(f"Error retrieving configurations from database: {e}")
            return []
    
    def get_configuration(self, config_id):
        """
        Get a specific configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            dict: Configuration data or None if not found
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM configurations WHERE id = ?", (config_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            config = dict(row)
            
            # Parse ROI data JSON
            try:
                config['roi_data'] = json.loads(config['roi_data'])
            except:
                pass
                
            return config
        except Exception as e:
            logger.error(f"Error retrieving configuration from database: {e}")
            return None