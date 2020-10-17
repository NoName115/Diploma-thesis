import os
import torch
import yaml
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import IterableDataset

from src.constants import LABELS, CHECKPOINT_FILE_NAME, CONFIG_FILE_NAME,\
    NUMBER_OF_JOINTS, NUMBER_OF_AXES
from src.model import BiRNN
from src.common import get_device


class SequenceDataset(IterableDataset):

    def __init__(self, sequence_file: str, meta_file: str, train_mode: bool = True, transforms=None):
        assert sequence_file.find("sequences") != -1

        self.sequence_file = sequence_file
        self.meta_file = meta_file
        self.train_mode = train_mode
        self.transforms = transforms

        self.valid_sequences = process_meta_file(self.meta_file, self.train_mode)
        self.classes = LABELS

    def get_valid_sequence(self):
        file_reader = read_file(self.sequence_file)
        for line in file_reader:
            if line.find("#objectKey") != -1:
                # Line1: #objectKey messif.objects.keys.AbstractObjectKey 0002-L_4637
                # Line2: 4637;mcdr.objects.ObjectMocapPose
                seq_info = line.split(" ")[-1].split("_")
                seq_id = seq_info[0]
                seq_length = int(seq_info[-1])
                _ = next(file_reader)  # skip second line of header

                # skip Sequences with invalid ID
                if seq_id not in self.valid_sequences:
                    continue

                # Read the sequence data
                sequence = []
                for _ in range(seq_length):
                    line = next(file_reader)
                    sequence.append(
                        [triple.split(", ") for triple in line.split("; ")]
                    )

                sequence = np.array(sequence, dtype=np.float32)
                sequence = torch.from_numpy(sequence)
                assert sequence.size() == (seq_length, NUMBER_OF_JOINTS, NUMBER_OF_AXES)
                yield sequence, [], seq_id

    def __iter__(self):
        return self.get_valid_sequence()

    def __len__(self) -> int:
        return len(self.valid_sequences)


class ActionDataset(IterableDataset):

    def __init__(self, action_file: str, meta_file: str, train_mode: bool = True, transforms=None):
        assert action_file.find("actions") != -1

        self.action_file = action_file
        self.meta_file = meta_file
        self.train_mode = train_mode
        self.transforms = transforms

        self.valid_actions = process_meta_file(self.meta_file, self.train_mode)
        self.classes = LABELS

        self.dataset_length = self.initialize_dataset_length()
        self.actions_info = {}

    def initialize_dataset_length(self):
        counter = 0
        file_reader = read_file(self.action_file)
        for line in file_reader:
            if line.find("#objectKey") != -1:
                action_id = line.split(" ")[-1].split("_")[0]
                if action_id in self.valid_actions:
                    counter += 1
        return counter

    def initialize_action_info(self):
        for _, label, action_info in self.get_valid_sequence(skip_action=True):
            self.actions_info.setdefault(action_info[0], []).append(
                (
                    int(action_info[-2]),
                    int(action_info[-1]),
                    np.argmax(label)
                )
            )

    def get_labels_by_sequence(
        self,
        sequence_name: str,
        start_idx: int,
        seq_length: int
    ):
        if sequence_name not in self.valid_actions:
            raise ValueError(f"Invalid sequence name to process {sequence_name}")

        actions = []
        for action_start_idx, action_length, target_label in self.actions_info[sequence_name]:
            if not ((start_idx + seq_length) > action_start_idx and
                    (action_start_idx + action_length) > start_idx):
                continue

            actions.append((
                max(start_idx, action_start_idx),
                min(start_idx + seq_length, action_start_idx + action_length),
                target_label
            ))

        return actions

    def get_valid_sequence(self, skip_action: bool = False):
        file_reader = read_file(self.action_file)
        for line in file_reader:
            if line.find("#objectKey") != -1:
                # Line1: #objectKey messif.objects.keys.AbstractObjectKey 0002-L_32_182_54
                # Line2: 54;mcdr.objects.ObjectMocapPose

                action_info = line.split(" ")[-1].split("_")
                action_id = action_info[0]
                action_length = int(action_info[-1])
                action_label = int(action_info[1])
                _ = next(file_reader)  # skip second line of header

                if action_id not in self.valid_actions:
                    continue

                # get label as spare vector
                label = self.classes[action_label]
                target = np.zeros(len(self.classes), dtype=np.float32)
                target[label] = 1.0

                if not skip_action:
                    action = []
                    for _ in range(action_length):
                        line = next(file_reader)
                        action.append(
                            [triple.split(", ") for triple in line.split("; ")]
                        )

                    # convert action into tensor
                    action = np.array(action, dtype=np.float32)
                    action = torch.from_numpy(action)
                    assert action.size() == (action_length, NUMBER_OF_JOINTS, NUMBER_OF_AXES)

                    yield action, target, action_info
                else:
                    yield [], target, action_info

    def __iter__(self):
        return self.get_valid_sequence()

    def __len__(self):
        return self.dataset_length


def read_file(file_to_read: str):
    with open(file_to_read, 'r') as f:
        for line in f:
            yield line.rstrip('\n')


def process_meta_file(meta_file: str, train_mode: bool) -> List[str]:
    with open(meta_file, "r") as mf:
        meta_data = mf.read().rstrip(',\n').split('\n')

    if train_mode:  # train-data
        return meta_data[1].rstrip(', ').split(', ')
    else:  # val-data
        return meta_data[3].split(', ')


def load_config_file(config_file: str) -> Dict:
    with open(config_file, "r") as cf:
        raw_data = cf.read()

    print("-" * 8 + " CONFIGURATION " + "-" * 8)
    print(raw_data)

    return yaml.safe_load(raw_data)


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
