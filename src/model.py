import os
import yaml
import torch
from typing import Dict, Optional
from torch import nn

from src.constants import MODEL_FINAL_FILE_NAME, CHECKPOINT_FILE_NAME, CONFIG_FILE_NAME
from src.common import get_device, get_logger


class BiRNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        embedding_output_size: Optional[int],
        num_classes: int
    ):
        super().__init__()
        self._logger = get_logger()

        self.device = get_device()
        self.hidden_size = lstm_hidden_size
        self.num_layers = 2  # bi-LSTM
        self.keep_short_memory = False

        # Embedding part, from 75 -> 64 size
        if embedding_output_size is not None:
            lstm_input_size = embedding_output_size
            self.embedding = nn.Linear(input_size, embedding_output_size)
            self.relu = nn.ReLU()
        else:
            lstm_input_size = input_size
            self.embedding = None
            self.relu = None

        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.classifier = nn.Linear(self.hidden_size * self.num_layers, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.h0 = None
        self.c0 = None

    def enable_keep_short_memory(self, batch_size: int = 1):
        self.initialize_short_memory(batch_size=batch_size)
        self.keep_short_memory = True
        self._logger.info(f"[M] Keep short memory: {self.keep_short_memory}")

    def disable_keep_short_memory(self):
        self.keep_short_memory = False
        self._logger.info(f"[M] Keep short memory: {self.keep_short_memory}")

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.disable_keep_short_memory()
        return self

    def initialize_short_memory(self, batch_size: int):
        self.h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)

    def forward(self, x):
        # embedding
        if self.embedding is not None:
            out = self.embedding(x)
            out = self.relu(out)
        else:
            out = x

        # bi-LSTM & dropout
        if not self.keep_short_memory:
            self.initialize_short_memory(out.size(0))

        out, (hn, cn) = self.lstm(out, (self.h0, self.c0))
        if self.keep_short_memory:
            self.h0 = hn
            self.c0 = cn

        # classification
        out = self.classifier(out[:, -1, :])
        return self.sigmoid(out)


def save_model(
    model_configuration: Dict,
    output_folder: str,
    trained_model,
    epoch: int,
    final_mode: bool = False
):
    model_name = MODEL_FINAL_FILE_NAME if final_mode else f"model_{epoch}.pth"
    file_path = os.path.join(output_folder, model_name)
    torch.save(trained_model.state_dict(), file_path)

    with open(os.path.join(output_folder, CHECKPOINT_FILE_NAME), "w") as lf:
        lf.write(model_name)
        lf.write("\n" + str(epoch))

    with open(os.path.join(output_folder, CONFIG_FILE_NAME), "w") as cf:
        yaml.dump(model_configuration, cf)

    get_logger().info(f"Model saved into {file_path}")
