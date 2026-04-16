import os
import logging
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

class ETL_Logger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger('etl_logger', os.path.join(log_dir, 'etl.log'))
        
    def log_transformation(self, message):
        self.logger.info(f"[TRANSFORM] {message}")
        
    def log_metadata(self, metadata):
        self.logger.info(f"[METADATA] {metadata}")

    def log_quality_check(self, check_name, status, details=""):
        self.logger.info(f"[QUALITY] {check_name}: {status} | {details}")
