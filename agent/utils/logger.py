"""Logging setup for HengSeng Sentiment Agent."""
import logging
import os
import sys

def setup_logger(name: str = "HengSengAgent", level: str = None) -> logging.Logger:
    """Configures and returns a standardized logger with formatted console and optional file output."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
        
    return logger
