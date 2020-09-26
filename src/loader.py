import os
import torch
import yaml
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import IterableDataset

from src.constants import LABELS, CHECKPOINT_FILE_NAME, CONFIG_FILE_NAME
from src.model import BiRNN
from src.common import get_device


class IterableMovementDataset(IterableDataset):
    NUMBER_OF_JOINTS = 25
    NUMBER_OF_AXES = 3

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms

        self.data_files = list(sorted(os.listdir(self.root)))
        self.file_frames: List[int] = []

        self.classes = LABELS

        self.loaded_data = dict()
        for file_name in self.data_files:
            with open(os.path.join(self.root, file_name), "r") as f:
                self.loaded_data[file_name] = f.read().rstrip('\n').split('\n')

    @staticmethod
    def _get_file_length(file_data: List[str]) -> int:
        header = file_data[0].split()[-1].split("_")
        return int(header[-1])

    def __iter__(self):
        for i, file_name in enumerate(self.data_files):
            # action_file = os.path.join(self.root, file_name)
            # with open(action_file, "r") as f:
            #    data_str = f.read().rstrip('\n').split('\n')
            data_str = self.loaded_data[file_name]

            sequence_length = self._get_file_length(data_str)

            all_frames = []
            for frame in data_str[2:]:  # first two header lines in the file
                all_frames.append(
                    [triple.split(", ") for triple in frame.split("; ")]
                )

            all_frames = np.array(all_frames, dtype=np.float32)
            all_frames = torch.from_numpy(all_frames)
            assert all_frames.size() == (sequence_length, self.NUMBER_OF_JOINTS, self.NUMBER_OF_AXES)

            # get label as spare vector
            label = self.classes[int(data_str[0].split()[-1].split("_")[1])]
            target = np.zeros(len(self.classes), dtype=np.float32)
            target[label] = 1.0

            yield all_frames, target

    def __len__(self) -> int:
        return len(self.data_files)


def load_config_file(config_file: str) -> Dict:
    with open(config_file, "r") as cf:
        raw_data = cf.read()

    print("-" * 8 + " CONFIGURATION " + "-" * 8)
    print(raw_data)

    return yaml.load(raw_data)


def create_model(model_config: Dict) -> BiRNN:
    device = get_device()
    return BiRNN(
        input_size=model_config["model"]["input_size"],
        lstm_hidden_size=model_config["model"]["hidden_size"],
        embedding_output_size=model_config["model"]["embedding_input_size"],
        num_classes=len(LABELS),
        device=device
    ).to(device)


def load_model(model_folder: str) -> Tuple[BiRNN, int, Dict]:
    print(f"Loading model from {model_folder}")

    with open(os.path.join(model_folder, CHECKPOINT_FILE_NAME), "r") as lf:
        last_model_file, model_epoch = lf.read().split('\n')
    config_file = load_config_file(os.path.join(model_folder, CONFIG_FILE_NAME))

    pytorch_model = create_model(config_file)
    pytorch_model.load_state_dict(
        torch.load(
            os.path.join(model_folder, last_model_file),
            map_location=get_device()
        )
    )
    return pytorch_model, int(model_epoch), config_file
