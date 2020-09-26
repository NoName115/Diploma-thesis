import os
import torch
import argparse
from torch.utils.data import DataLoader

from src.loader import load_model, IterableMovementDataset
from src.common import get_device


def evaluate_classification(model_folder: str, data_folder: str):
    device = get_device()

    trained_model, _, model_config = load_model(model_folder)
    evaluation_loader = DataLoader(
        IterableMovementDataset(data_folder),
        batch_size=model_config["train"]["batch_size"]
    )

    with torch.no_grad():
        correct = 0
        total = len(evaluation_loader)

        for i, (sequence, labels) in enumerate(evaluation_loader, 1):
            if i % model_config['evaluation']['report_step'] == 0:
                print(f"\tProcessed: [{i}/{len(evaluation_loader)}]")

            sequence = sequence.view(sequence.size(0), sequence.size(1), -1).to(device)
            target_label = torch.argmax(labels).item()

            outputs = trained_model(sequence)

            _, predicted_label = torch.max(outputs.data, dim=1)
            predicted_label = predicted_label.item()  # convert to one integer

            if target_label == predicted_label:
                correct += 1

            if i % 400 == 0:
                total = 400
                break

        print(f"Accuracy: {round(100 * (correct / total), 4)}")
        print(f"correct labels: {correct}, total: {total}")


if __name__ == "__main__":
    # TODO - detection/classification option for evaluation

    parser = argparse.ArgumentParser(description="Script for evaluating trained model")
    parser.add_argument("--model", "-m", help="Folder with trained model", required=True)
    parser.add_argument("--data", "-d", help="Data used for evaluation", required=True)
    args = parser.parse_args()

    evaluate_classification(
        model_folder=args.model,
        data_folder=args.data
    )
