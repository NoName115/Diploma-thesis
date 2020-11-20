import os
import re
import torch
import yaml
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import IterableDataset, Dataset
from torch._six import container_abcs, string_classes, int_classes

from src.constants import LABELS, CHECKPOINT_FILE_NAME, CONFIG_FILE_NAME,\
    NUMBER_OF_JOINTS, NUMBER_OF_AXES
from src.model import BiRNN
from src.common import get_device

np_str_obj_array_pattern = re.compile(r'[SaUO]')


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


class ActionDatasetIterative(IterableDataset):

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
        seq_start_idx: int,
        seq_length: int
    ) -> List[Tuple[int, int, int]]:
        if sequence_name not in self.valid_actions:
            raise ValueError(f"Invalid sequence name to process {sequence_name}")

        seq_end_idx = seq_start_idx + seq_length - 1

        actions = []
        for action_start_idx, action_length, target_label in self.actions_info[sequence_name]:
            action_end_idx = action_start_idx + action_length - 1
            if not (seq_end_idx >= action_start_idx and action_end_idx >= seq_start_idx):
                continue

            actions.append((
                max(seq_start_idx, action_start_idx),
                min(seq_end_idx, action_end_idx),
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


class ActionDatasetList(Dataset):

    def __init__(self, action_file: str, meta_file: str, train_mode: bool = True, transforms=None):
        assert action_file.find("actions") != -1

        self.action_file = action_file
        self.meta_file = meta_file
        self.train_mode = train_mode
        self.transforms = transforms

        self.valid_actions = process_meta_file(self.meta_file, self.train_mode)
        self.classes = LABELS

        self.dataset_length = self.initialize_dataset_length()
        self.data_list = self.load_data()

        print(f"Dataset length: {self.dataset_length}")
        print(f"Data list length: {len(self.data_list)}")

    def initialize_dataset_length(self):
        counter = 0
        file_reader = read_file(self.action_file)
        for line in file_reader:
            if line.find("#objectKey") != -1:
                action_id = line.split(" ")[-1].split("_")[0]
                if action_id in self.valid_actions:
                    counter += 1
        return counter

    def load_data(self):
        ad_iter = ActionDatasetIterative(
            self.action_file,
            self.meta_file,
            self.train_mode,
        )
        results = []
        for i, data_tuple in enumerate(ad_iter, 1):
            if i % 500 == 0:
                print(f"Processing {i}/{len(ad_iter)}")
            results.append(data_tuple)
        return results

    def __getitem__(self, idx: int):
        return self.data_list[idx]

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
        num_classes=len(LABELS)
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


def collate_seq(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if not all(elem.size() == s.size() for s in batch):
            #for i, s in enumerate(batch):
            #    print(f"{i}: {s.size()}")
            #print("------------------------")
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        else:
            return torch.stack(batch, 0)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f"Invalid type of numpy array: {elem.type}")

            return collate_seq([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch

    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        return [collate_seq(samples) for samples in zip(*batch)]
    else:
        raise TypeError(f"Invalid type of element: {elem_type}")
