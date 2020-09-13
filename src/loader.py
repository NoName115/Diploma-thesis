import os
import torch
import yaml
import numpy as np
from typing import List, Dict
from torch.utils.data import IterableDataset

from src.constants import LABELS


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
        return yaml.load(cf)
