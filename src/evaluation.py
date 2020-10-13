import torch
import argparse
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from src.model import BiRNN
from src.loader import load_model, ActionDataset, SequenceDataset
from src.common import get_device


def evaluate_sequences(trained_model: BiRNN, model_config: Dict, evaluation_loader: DataLoader) -> Dict:
    print("-" * 6 + " SEQUENCE EVALUATION " + "-" * 6)
    device = get_device()
    trained_model.eval()

    with torch.no_grad():
        eval_dict = {
            "correct": 0,
            "above": 0
        }
        thresholds = [(
            round((1 / model_config['evaluation']['threshold_steps']) * (i + 1), 4),
            eval_dict.copy()
        ) for i in range(model_config['evaluation']['threshold_steps'])]

        for i, (sequence, labels, seq_id) in enumerate(evaluation_loader, 1):
            print(f"-> Sequence: {seq_id[0]} [{i}/{len(evaluation_loader)}] frames {len(sequence[0])}")

            target_label = torch.argmax(labels).item() if not labels is None else None
            for j, frame in enumerate(sequence[0], 1):  # [0] as we are processing batch=1
                torch_frame = frame.view(1, 1, -1).to(device)
                outputs = trained_model(torch_frame)

                for val_th, res_dict in thresholds:
                    res_dict["above"] += (outputs.data > val_th).sum().item()
                    for _, label_idx in zip(*torch.where(outputs.data > val_th)):
                        if label_idx == target_label:
                            res_dict["correct"] += 1

                if j % model_config['evaluation']['report_step'] == 0:
                    print(f"\tProcessed frame: {j}/{len(sequence[0])}")

                #print(torch.max(outputs.data).item())

                """
                _, label_idx = torch.where(outputs.data > 0.5)  # 0.24/0.25, videl som aj 0.28 ale tak raz
                if len(label_idx) != 0:
                    print(torch.argmax(outputs.data).item())
                    max_value = torch.max(outputs.data).item()
                    print(max_value)
                    print("----------------------")

                    if max_value > total_max:
                        total_max = max_value
                        print(f"---> MAX {total_max}")
                """

    # Calculate final statistics
    result_dict = {
        "thresholds": {},
        "AP": 0.0
    }

    ap_score = 0
    old_recall = 0
    for th, values in thresholds[:-1]:
        recall = values["correct"] / len(evaluation_loader) if values["correct"] > 0 else 0
        precision = values["correct"] / values["above"] if values["above"] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ap_score += ((abs(recall - old_recall)) * precision)
        old_recall = recall

        result_dict["thresholds"][str(th)] = {
            "recall": recall,
            "precision": precision,
            "f1-score": f1_score
        }

        print(
            "[{:.2f}]".format(th) +
            f"\n   Precision: {round(100 * precision, 4)}%"
            f"\n   Recall: {round(100 * recall, 4)}%"
            f"\n   f1_score: {round(f1_score, 4)}"
        )

    result_dict["AP"] = ap_score
    print(f"[AP] {round(ap_score, 4)}")

    return result_dict


def evaluate_actions(
    trained_model: BiRNN,
    model_config: Dict,
    evaluation_loader: DataLoader
) -> Tuple[int, int]:
    print("-" * 6 + " ACTION EVALUATION " + "-" * 6)
    device = get_device()
    trained_model.eval()

    with torch.no_grad():
        correct = 0
        total = len(evaluation_loader)

        for i, (sequence, labels, _) in enumerate(evaluation_loader, 1):
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

        print(f"Accuracy: {round(100 * (correct / total), 4)}% - [{correct}/{total}]")

        return correct, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating trained model")
    parser.add_argument("--model", "-m", help="Folder with trained model", required=True)
    parser.add_argument("--meta", "-M", help="Meta data for CS/CV datasets", required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-actions", "-da", help="File with data for action evaluation")
    group.add_argument("--data-sequences", "-ds", help="File with data for sequence evaluation")
    group.add_argument("--data-act-seq", "-D", help="File with actions for sequence evaluation")
    args = parser.parse_args()

    # load model & configuration
    trained_model, _, model_config = load_model(args.model)

    if args.data_actions:
        evaluate_actions(
            trained_model,
            model_config,
            DataLoader(
                ActionDataset(args.data_actions, args.meta, train_mode=False),
                batch_size=1
            )
        )

    if args.data_sequences:
        evaluate_sequences(
            trained_model,
            model_config,
            DataLoader(
                SequenceDataset(args.data_sequences, args.meta, train_mode=False),
                batch_size=1
            )
        )

    if args.data_act_seq:
        evaluate_sequences(
            trained_model,
            model_config,
            DataLoader(
                ActionDataset(args.data_act_seq, args.meta, train_mode=False),
                batch_size=1
            )
        )
