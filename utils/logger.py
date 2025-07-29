"""
Logging setup for the traffic violation detector application.
"""
import logging
import os
from datetime import datetime

def setup_logger(log_file="traffic_violations.log", console_level=logging.INFO, file_level=logging.INFO):
    """
    Set up and configure logging for the application.
    
    Args:
        log_file (str): Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("traffic_detector")
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger

# Default logger instance
logger = setup_logger()