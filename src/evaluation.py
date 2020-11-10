import torch
import argparse
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from src.model import BiRNN
from src.loader import load_model, ActionDataset, SequenceDataset
from src.common import get_device


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


def evaluate_sequences(
    trained_model: BiRNN,
    model_config: Dict,
    sequence_loader: DataLoader,
    action_dataset: ActionDataset,
    keep_short_memory: bool
) -> Dict:
    print("-" * 6 + " SEQUENCE EVALUATION " + "-" * 6)
    device = get_device()
    trained_model.eval()
    if keep_short_memory:
        trained_model.enable_keep_short_memory()
    action_dataset.initialize_action_info()

    with torch.no_grad():
        eval_dict = {
            "correct": 0,
            "above": 0
        }
        thresholds = [(
            round((1 / model_config['evaluation']['threshold_steps']) * (i + 1), 4),
            eval_dict.copy()
        ) for i in range(model_config['evaluation']['threshold_steps'])]

        total_frames = 0
        for i, (sequence, _, seq_id) in enumerate(sequence_loader, 1):
            print(f"-> Sequence: {seq_id[0]} [{i}/{len(sequence_loader)}] frames {len(sequence[0])}")
            if keep_short_memory:
                trained_model.initialize_short_memory(batch_size=1)  # restart short_memory for a new sequence

            frame_iter = IterFrame(
                sequence[0],  # [0] as we are processing batch=1
                model_config['evaluation']['frame_size']
            )

            for j, frame in enumerate(frame_iter, 1):
                number_of_frames = frame.size(0)

                torch_frame = frame.view(1, number_of_frames, -1).to(device)  # type: ignore
                outputs = trained_model(torch_frame)

                assert outputs.size() == (1, 43)

                # Only valid frames
                start_frame_idx = (j - 1) * model_config['evaluation']['frame_size'] + 1
                res = action_dataset.get_labels_by_sequence(
                    sequence_name=seq_id[0],
                    seq_start_idx=start_frame_idx,
                    seq_length=model_config['evaluation']['frame_size']
                )
                if not res:
                    continue
                else:
                    # index start from 1, to get length --> end_idx - start_idx + 1
                    total_frames += sum((ei - si + 1) for si, ei, _ in res)

                for th, result_dict in thresholds:
                    # whole batch is classified into several categories
                    result_dict["above"] += (outputs.data > th).sum().item() * number_of_frames

                    for _, label_idx in zip(*torch.where(outputs.data > th)):
                        # get labels into which the batch is classified
                        for s_idx, e_idx, target_label in res:
                            # check whole batch of frames if they goes into correct category
                            if label_idx == target_label:
                                result_dict["correct"] += (e_idx - s_idx) + 1

                if j % model_config['evaluation']['report_step'] == 0:
                    print(f"\tProcessed steps: {j}/{len(frame_iter)}")

                #print(f"{torch.argmax(outputs.data).item()} - {round(torch.max(outputs.data).item(), 4)}")

    # Calculate final statistics
    result_dict = {
        "thresholds": {},
        "AP": 0.0,
        "total_frames": total_frames
    }

    ap_score = 0
    old_recall = 0
    for th, values in thresholds[:-1]:
        recall = values["correct"] / total_frames if values["correct"] > 0 else 0
        precision = values["correct"] / values["above"] if values["above"] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ap_score += ((abs(recall - old_recall)) * precision)
        old_recall = recall

        result_dict["thresholds"][str(th)] = {
            "recall": recall,
            "precision": precision,
            "f1-score": f1_score,
            "correct": values["correct"],
            "above": values["above"]
        }

        print(
            "[{:.2f}]".format(th) +
            f"\n   C: {values['correct']}, T: {total_frames}, A: {values['above']}"
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
    trained_model.eval().disable_keep_short_memory()

    with torch.no_grad():
        correct = 0
        total = len(evaluation_loader)

        for i, (sequence, labels, _) in enumerate(evaluation_loader, 1):
            if i % model_config['evaluation']['report_step'] == 0:
                print(f"\tProcessed: [{i}/{len(evaluation_loader)}]")

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
    parser.add_argument("--data-actions", "-da", help="File with data for action evaluation", required=True)
    parser.add_argument("--data-sequences", "-ds", help="File with data for sequence evaluation")
    parser.add_argument("--short-memory", "-sm", help="If set to True model will keep short memory",
                        type=bool, default=False)

    args = parser.parse_args()

    # load model & configuration
    trained_model, _, model_config = load_model(args.model)

    if args.data_actions and not args.data_sequences:
        evaluate_actions(
            trained_model,
            model_config,
            DataLoader(
                ActionDataset(args.data_actions, args.meta, train_mode=False),
                batch_size=1
            )
        )

    if args.data_sequences and args.data_actions:
        evaluate_sequences(
            trained_model,
            model_config,
            DataLoader(
                SequenceDataset(args.data_sequences, args.meta, train_mode=False),
                batch_size=1
            ),
            ActionDataset(args.data_actions, args.meta, train_mode=False),
            keep_short_memory=args.short_memory
        )
