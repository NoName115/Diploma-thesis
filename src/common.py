import os
import sys
import torch
import logging
from typing import Optional
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


class IterFrame:

    def __init__(
        self,
        one_sequence,
        batch_size: int
    ):
        self.sequence = one_sequence
        self.bs = batch_size
        self.iter_seq = iter(self.sequence)
        self.number_of_steps = len(self.sequence) // self.bs
        if len(self.sequence) % self.bs != 0:
            self.number_of_steps += 1

    @property
    def sequence_length(self) -> int:
        return len(self.sequence)

    def __iter__(self):
        for i in range(self.number_of_steps):
            tensor_list = []
            for _ in range(self.bs):
                try:
                    tensor_list.append(next(self.iter_seq))
                except StopIteration:
                    break

            yield torch.stack(tensor_list, 0)

    def __len__(self) -> int:
        return self.number_of_steps


class JunkSequence:

    def __init__(
        self,
        input_sequence,
        size_of_junk: Optional[int] = None,
        step_size: Optional[int] = None
    ):
        self.sequence = input_sequence
        self.size_of_junk = size_of_junk if size_of_junk else self.sequence.size(1) + 1
        # no overlap if step_size is None
        self.step_size = step_size if step_size else self.size_of_junk
        assert self.sequence.size(0) == 1, "Only batch == 1 is available"
        assert self.step_size <= self.size_of_junk, "Step is bigger than size of the junk"

        self.number_of_steps = self.sequence.size(1) // self.step_size
        if self.sequence.size(1) % self.step_size != 0:
            self.number_of_steps += 1

    def __iter__(self):
        start_index = 0
        end_index = self.size_of_junk
        while True:
            yield self.sequence[:, start_index: end_index]
            if end_index >= self.sequence.size(1):
                break

            start_index += self.step_size
            end_index = start_index + self.size_of_junk

    def __len__(self):
        return self.number_of_steps
