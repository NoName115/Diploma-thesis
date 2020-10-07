import torch
import argparse
from typing import Dict
from torch.utils.data import DataLoader

from src.model import BiRNN
from src.loader import load_model, ActionDataset, SequenceDataset
from src.common import get_device


def evaluate_sequences(trained_model: BiRNN, model_config: Dict, data_file: str, meta_file: str):
    print("-" * 6 + " SEQUENCE EVALUATION " + "-" * 6)
    device = get_device()

    trained_model.eval()
    evaluation_loader = DataLoader(
        SequenceDataset(data_file, meta_file, train_mode=False),
        batch_size=model_config["train"]["batch_size"]
    )

    total_max = 0.0

    with torch.no_grad():
        for i, (sequence, seq_id) in enumerate(evaluation_loader, 1):
            print(f"-> Sequence: {seq_id[0]} [{i}/{len(evaluation_loader)}] frames {len(sequence[0])}")

            for j, frame in enumerate(sequence[0], 1):  # [0] as we are processing batch=1
                if j % 49 == 0:
                    print(f"Processing frame: {j}/{len(sequence[0])}")

                torch_frame = frame.view(1, 1, -1).to(device)
                outputs = trained_model(torch_frame)

                _, label_idx = torch.where(outputs.data > 0.5)  # 0.24/0.25, videl som aj 0.28 ale tak raz
                if len(label_idx) != 0:
                    print(torch.argmax(outputs.data).item())
                    max_value = torch.max(outputs.data).item()
                    print(max_value)
                    print("----------------------")

                    if max_value > total_max:
                        total_max = max_value
                        print(f"---> MAX {total_max}")


def evaluate_actions(trained_model: BiRNN, model_config: Dict, data_folder: str, meta_file: str):
    print("-" * 6 + " ACTION EVALUATION " + "-" * 6)
    device = get_device()

    trained_model.eval()
    evaluation_loader = DataLoader(
        ActionDataset(data_folder, meta_file, train_mode=False),
        batch_size=model_config["train"]["batch_size"]
    )

    with torch.no_grad():
        correct = 0
        total = len(evaluation_loader)

        for i, (sequence, labels) in enumerate(evaluation_loader, 1):
            if i % model_config['evaluation']['report_step'] == 0:
                print(f"\tProcessed: [{i}/{len(evaluation_loader)}]")

            # TODO - https://github.com/pytorch/pytorch/issues/1538
            #   - user transpose instead of view
            #   - https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916
            sequence = sequence.view(sequence.size(0), sequence.size(1), -1).to(device)
            target_label = torch.argmax(labels).item()

            outputs = trained_model(sequence)

            _, predicted_label = torch.max(outputs.data, dim=1)
            predicted_label = predicted_label.item()  # convert to one integer

            if target_label == predicted_label:
                correct += 1

        print(f"Accuracy: {round(100 * (correct / total), 4)}%")
        print(f"correct labels: {correct}, total: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating trained model")
    parser.add_argument("--model", "-m", help="Folder with trained model", required=True)
    parser.add_argument("--meta", "-M", help="Meta data for CS/CV datasets", required=True)
    parser.add_argument("--data-actions", "-da", help="Folder with data for action evaluation")
    parser.add_argument("--data-sequences", "-ds", help="File with data for sequence evaluation")
    args = parser.parse_args()

    # load model & configuration
    trained_model, _, model_config = load_model(args.model)

    if args.data_actions:
        evaluate_actions(
            trained_model,
            model_config,
            args.data_actions,
            args.meta
        )

    if args.data_sequences:
        evaluate_sequences(
            trained_model,
            model_config,
            args.data_sequences,
            args.meta
        )
