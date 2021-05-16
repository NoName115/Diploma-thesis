import os
import json
import torch
import argparse
import pickle
import datetime

from typing import Dict, Optional, List, Tuple
from torch.utils.data import DataLoader
from collections import defaultdict

from src.model import BiRNN
from src.common import get_device, logger_manager, get_logger, IterFrame
from src.loader import load_model, ActionDatasetIterative, SequenceDataset

GROUND_TRUTH = "ground-truth"
PREDICTION = "prediction"
SEQ_LENGTH = "seq-length"


def default_value():
    return {
        GROUND_TRUTH: [],
        PREDICTION: [],
        SEQ_LENGTH: 0
    }


def pprint_results(input_results: Dict):
    for seq, values in input_results.items():
        gt_string = ', '.join("{:02d}".format(gt) for gt in values[GROUND_TRUTH])
        print(f"{seq}: {gt_string}")
        pred_string = ", ".join("{:02d}".format(torch.argmax(pred).item()) for pred in values[PREDICTION])
        print(f"{seq}: {pred_string}")
        print("---" * 20)


def process_results(
    input_results: Dict
) -> Tuple[Dict, List[float]]:
    threshold_results = defaultdict(lambda : {"correct": 0, "above": 0})
    th_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    eval_logger = get_logger()
    for i, (seq, values) in enumerate(input_results.items(), 1):
        eval_logger.info(f"-> Sequence: {seq} [{i}/{len(input_results)}]")
        for i in range(values[SEQ_LENGTH]):
            predict = values[PREDICTION][i]
            ground_truth = values[GROUND_TRUTH][i]
            for th in th_list:
                threshold_results[th]['above'] += (predict > th).sum().item()
                if ground_truth in torch.where(predict > th)[1]:
                    threshold_results[th]['correct'] += 1

    return threshold_results, th_list


def calculate_statistics(
    input_thresholds: Dict,
    threshold_list: List[float],
    input_results: Dict
):
    result_dict = {
        "thresholds": dict(),
        "AP": 0.0,
        "total_frames": sum(values[SEQ_LENGTH] for _, values in input_results.items())
    }

    ap_score = 0
    old_recall = 0
    for th in threshold_list:
        values = input_thresholds[th]
        recall = values["correct"] / result_dict['total_frames'] if values["correct"] > 0 else 0
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

    result_dict["AP"] = ap_score
    return result_dict


def evaluate_sequence(
    trained_model: BiRNN,
    model_config: Dict,
    sequence_loader: DataLoader,
    action_dataset: ActionDatasetIterative,
    keep_short_memory: bool,
    frame_size: Optional[int] = None
):
    eval_logger = get_logger()
    device = get_device()

    trained_model.eval()
    if keep_short_memory:
        trained_model.enable_keep_short_memory()
    action_dataset.initialize_action_info()

    predictions = defaultdict(default_value)

    current_frame_size = 1 if not frame_size else frame_size
    with torch.no_grad():
        for i, (sequence, _, seq_id) in enumerate(sequence_loader, 1):
            eval_logger.info(f"-> Sequence: {seq_id[0]} [{i}/{len(sequence_loader)}] frames {len(sequence[0])}")
            # restart short-memory after every sequence
            if keep_short_memory:
                trained_model.initialize_short_memory(batch_size=1)

            frame_iter = IterFrame(
                sequence[0],
                current_frame_size
            )
            for j, frame in enumerate(frame_iter, 1):
                if j % (300 // current_frame_size) == 0:
                    eval_logger.info(f"[{seq_id[0]}] Processing... {j}/{len(frame_iter)}")

                number_of_frames = frame.size(0)

                torch_frame = frame.view(1, number_of_frames, -1).to(device)
                outputs = trained_model(torch_frame)
                assert outputs.size() == (1, 43)

                start_frame_idx = (j - 1) * current_frame_size + 1
                seq_action = action_dataset.get_labels_by_sequence(
                    sequence_name=seq_id[0],
                    seq_start_idx=start_frame_idx,
                    seq_length=current_frame_size
                )

                #print(f"{res} - {torch.argmax(outputs.data).item()} - {round(torch.max(outputs.data).item(), 4)} - {outputs.data}")

                if not seq_action:
                    continue

                for s_idx, e_idx, label in seq_action:
                    # item[0] - start-index
                    # item[1] - end-index
                    # item[2] - label
                    total_frames = (e_idx - s_idx + 1)
                    predictions[seq_id[0]][GROUND_TRUTH] += [label] * total_frames
                    predictions[seq_id[0]][PREDICTION] += [outputs.data] * total_frames
                    predictions[seq_id[0]][SEQ_LENGTH] += total_frames

            break


    pprint_results(input_results=predictions)
    pickle.dump(predictions, open(f'predictions_{datetime.datetime.now().isoformat()}.p', 'wb'))

    # eval_logger.info("Processing results...")
    # processed_res, th_list = process_results(input_results=results)
    # eval_logger.info(json.dumps(processed_res, indent=4, sort_keys=True))
    #
    # eval_logger.info('\n')
    # eval_logger.info("Calculating final statistics...")
    # statistics = calculate_statistics(processed_res, th_list, results)
    # eval_logger.info(json.dumps(statistics, indent=4, sort_keys=True))



class CustomArg:

    def __init__(self):
        self.meta = None
        self.model = None
        self.data_actions = None
        self.data_sequences = None
        self.short_memory = True
        self.frame_size = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating trained model")
    parser.add_argument("--model", "-m", help="Folder with trained model", required=True)
    parser.add_argument("--meta", "-M", help="Meta data for CS/CV datasets", required=True)
    parser.add_argument("--data-actions", "-da", help="File with data for action evaluation", required=True)
    parser.add_argument("--data-sequences", "-ds", help="File with data for sequence evaluation")
    parser.add_argument("--short-memory", "-sm", help="If set to True model will keep short memory",
                        type=bool, default=False)
    parser.add_argument("--frame-size", "-f", help="Number of frames to take for classification",
                        type=int, default=1)
    parser.add_argument("--save-metrics", "-s", help="If set to True evaluation results will be saved.",
                        type=bool, default=False)
    args = parser.parse_args()

    # args = CustomArg()
    # work_dir = os.path.dirname(os.path.dirname(__file__))
    #
    # args.model = os.path.join(work_dir, "output_models", "cross_subject-batch_1_lr_0001_rs_2020_12_26-12_50_19")
    # args.data_sequences = os.path.join(work_dir, "data", "sequences-single-subject-all-POS.data")
    # args.meta = os.path.join(work_dir, "data", "meta", "cross-subject.txt")
    # args.data_actions = os.path.join(work_dir, "data", "actions-single-subject-all-POS.data")

    logger_manager.init_eval_logger(args.model)

    trained_model, _, model_config = load_model(args.model)

    evaluate_sequence(
        trained_model=trained_model,
        model_config=model_config,
        sequence_loader=DataLoader(
            SequenceDataset(args.data_sequences, args.meta, train_mode=False),
            batch_size=1
        ),
        action_dataset=ActionDatasetIterative(args.data_actions, args.meta, train_mode=False),
        keep_short_memory=args.short_memory,
        frame_size=args.frame_size
    )
