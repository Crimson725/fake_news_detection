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

        # verbose logs
        file_handler2 = logging.FileHandler(path + "/" + "verbose_logs.txt")
        file_handler2.setLevel(logging.DEBUG)
        formatter2 = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        )
        file_handler2.setFormatter(formatter2)

        self.logger.addHandler(file_handler1)
        self.logger.addHandler(file_handler2)


    def log(self, message):
        self.logger.info(message)
    def verbose(self, message):
        self.logger.debug(message)