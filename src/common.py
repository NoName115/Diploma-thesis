import os
import sys
import torch
import logging
from datetime import datetime

LOG_FORMATTER = logging.Formatter("%(asctime)s | %(levelname)5s | %(message)s")
DATETIME_FORMAT = "%Y_%m_%d-%H_%M_%S"


class LoggerManager:

    def __init__(self):
        self.train_logger = None
        self.eval_logger = None

    def init_train_logger(self, log_folder: str) -> logging.Logger:
        assert not self.train_logger, "Train logger already exist"
        self.train_logger = self._init_custom_logger(
            log_folder,
            "train"
        )
        return self.train_logger

    def init_eval_logger(self, log_folder: str) -> logging.Logger:
        assert not self.eval_logger, "Eval logger already exist"
        self.eval_logger = self._init_custom_logger(
            log_folder,
            "eval"
        )
        return self.eval_logger

    @staticmethod
    def _init_custom_logger(log_folder: str, log_file_postfix: str) -> logging.Logger:
        assert os.path.exists(log_folder), "Log folder does not exist."

        lg = logging.getLogger(log_file_postfix)
        lg.setLevel(logging.INFO)

        # log into stdout
        st_handler = logging.StreamHandler(stream=sys.stdout)
        st_handler.setLevel(logging.INFO)
        st_handler.setFormatter(LOG_FORMATTER)
        lg.addHandler(st_handler)

        # log into file
        file_handler = logging.FileHandler(os.path.join(
            log_folder,
            f"logs_{log_file_postfix}_{datetime.now().strftime(DATETIME_FORMAT)}"
        ))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOG_FORMATTER)
        lg.addHandler(file_handler)
        return lg

    def get_logger(self) -> logging.Logger:
        if self.train_logger:
            return self.train_logger
        elif self.eval_logger:
            return self.eval_logger
        else:
            raise RuntimeError("First initialize train or eval logger")


logger_manager = LoggerManager()


def get_logger() -> logging.Logger:
    return logger_manager.get_logger()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
