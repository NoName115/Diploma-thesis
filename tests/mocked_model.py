import torch
import numpy as np

from typing import List, Tuple
from src.constants import LABELS
from src.model import BiRNN


class MockedDetectionBiRNN(BiRNN):

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        embedding_output_size: int,
    ):
        super().__init__(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            embedding_output_size=embedding_output_size,
            num_classes=len(LABELS)
        )
        self.num_of_classes = len(LABELS)
        self.predicted_labels = None

        self.counter = 0

    def set_predicted_labels(self, labels: List[List[Tuple[int, float]]]):
        self.predicted_labels = labels
        self.counter = 0

    def forward(self, x):
        assert self.predicted_labels, "Firstly set-up predicted labels"
        assert self.counter < len(self.predicted_labels), \
            "Number of expected labels exceeded"

        target_labels = [(LABELS[lb], s) for lb, s in self.predicted_labels[self.counter]]
        self.counter += 1

        sigmoid_res = [[0.0 for _ in range(self.num_of_classes)]]
        for label, score in target_labels:
            sigmoid_res[0][label] = score

        return torch.from_numpy(np.array(sigmoid_res))


class MockedClassificationBiRNN(BiRNN):

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        embedding_output_size: int,
    ):
        super().__init__(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            embedding_output_size=embedding_output_size,
            num_classes=len(LABELS)
        )
        self.num_of_classes = len(LABELS)
        self.predicted_labels = None

        self.counter = 0

    def set_predicted_labels(self, labels: List[int]):
        self.counter = 0
        self.predicted_labels = labels

    def forward(self, x):
        assert self.predicted_labels, "Firstly set-up predicted labels"
        assert self.counter < len(self.predicted_labels),\
            "Number of expected labels exceeded"

        exp_idx = LABELS[self.predicted_labels[self.counter]]
        self.counter += 1

        sigmoid_res = np.array(
            [[1.0 if i == exp_idx else 0.1 for i in range(self.num_of_classes)]]
        )
        return torch.from_numpy(sigmoid_res)
