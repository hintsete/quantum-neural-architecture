# src/utils/logging.py

import logging
import os

def setup_logger(log_file: str = None, level=logging.INFO):
    """
    Setup Python logger for console and optional file logging.

    Args:
        log_file (str, optional): Path to a log file. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger("quantum_attention")
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger