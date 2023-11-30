import sys
import logging
from typing import Optional

class Logger:
    """Logs info, warning, and error messages"""
    def __init__(self, name: Optional[str]=None) -> None:
        if name is None:
            name = __class__.__name__

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        fmt = logging.Formatter("%(name)s:%(levelname)s - %(message)s")

        self.logger.addHandler(self.__set_file(fmt))
        self.logger.addHandler(self.__set_stream(fmt))

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warn(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message, exc_info=True)

        sys.exit(1)

    def __set_stream(self, fmt: logging.Formatter) -> logging.StreamHandler:
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(fmt)
        s_handler.setLevel(logging.INFO)

        return s_handler

    def __set_file(self, fmt: logging.Formatter) -> logging.FileHandler:
        f_handler = logging.FileHandler("./logs/logs.log", "w")
        f_handler.setFormatter(fmt)
        f_handler.setLevel(logging.INFO)

        return f_handler