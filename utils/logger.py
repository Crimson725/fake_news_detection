import time
import logging


class Logger:
    def __init__(self, name, path):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # info logs
        file_handler1 = logging.FileHandler(path + "/" + "logs.txt")
        file_handler1.setLevel(logging.INFO)
        formatter1 = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler1.setFormatter(formatter1)
        self.logger.addHandler(file_handler1)

    def log(self, message):
        self.logger.info(message)
