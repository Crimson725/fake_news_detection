import time
import logging


class Logger:
    def __init__(self, name, path):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(path + "/" + "logs.txt")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)
