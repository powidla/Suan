import os
import json
import logging


class Logger:
    def __init__(self, log_dir, filename="metrics.log", overwrite=True):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.handling = open(self.path, "w" if overwrite else "a", buffering=1)

    def info(self, message):
        self.handling.write(message + "\n")
        self.handling.flush()

    def log(self, record: dict):
        self.info(json.dumps(record))

    def close(self):
        if not self.handling.closed:
            self.handling.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
