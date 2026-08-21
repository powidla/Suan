import os
import logging

def log_metrics(log_dir, filename="metrics.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"dpo_metrics.{log_dir}")
    logger.setLevel(logging.INFO)
  
    if not logger.handlers:
        log_path = os.path.join(log_dir, filename)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines only
        logger.addHandler(handler)
        logger.propagate = False
    return logger