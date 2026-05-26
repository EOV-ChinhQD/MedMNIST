import os
import logging
from typing import Dict, Any

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
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
    """
    Logger class tailored for ETL (Extract, Transform, Load) operations.
    """
    def __init__(self, log_dir: str = 'logs') -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.logger: logging.Logger = setup_logger('etl_logger', os.path.join(log_dir, 'etl.log'))
        
    def log_transformation(self, message: str) -> None:
        self.logger.info(f"[TRANSFORM] {message}")
        
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        self.logger.info(f"[METADATA] {metadata}")

    def log_quality_check(self, check_name: str, status: str, details: str = "") -> None:
        self.logger.info(f"[QUALITY] {check_name}: {status} | {details}")
