import os
import torch
import argparse
import pickle
import datetime
import numpy as np

from typing import Dict, Optional
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.model import BiRNN
from src.common import get_device, logger_manager, get_logger, IterFrame, DATETIME_FORMAT
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


def evaluate_sequence(
    trained_model: BiRNN,
    sequence_loader: DataLoader,
    action_dataset: ActionDatasetIterative,
    keep_short_memory: bool,
    frame_size: Optional[int] = None
) -> dict:
    eval_logger = get_logger()
    eval_logger.info("-" * 6 + " SEQUENCE EVALUATION " + "-" * 6)

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

                if not seq_action:
                    continue

                for s_idx, e_idx, label in seq_action:
                    # item[0] - start-index
                    # item[1] - end-index
                    # item[2] - label
                    total_frames = (e_idx - s_idx + 1)
                    predictions[seq_id[0]][GROUND_TRUTH] += [label] * total_frames
                    predictions[seq_id[0]][PREDICTION] += [outputs.data.cpu().numpy()] * total_frames
                    predictions[seq_id[0]][SEQ_LENGTH] += total_frames

            #break

    return predictions


def get_sequence_statistics(res_predictions: Dict) -> dict:
    eval_logger = get_logger()

    y_cont = []
    x_cont = []
    for _, values in res_predictions.items():
        y_cont += values[GROUND_TRUTH]
        x_cont += [t[0] for t in values[PREDICTION]]

    y_cont = np.array(y_cont)
    x_cont = np.array(x_cont)

    # ground-truth binarization
    y_bin = label_binarize(y_cont, classes=list(range(0, 43)))

    results = {
        'thresholds': {},
        'macro-AP': 0.0,
        'micro-AP': 0.0
    }
    # calculate Precision, Recall, F1 Score
    for t in np.arange(0.1, 1.0, 0.1):
        rt = round(t, 1)
        x_bin = np.where(x_cont > rt, 1, 0)
        micro_metrics = precision_recall_fscore_support(
            y_bin,
            x_bin,
            average='micro',
            zero_division=0
        )
        macro_metrics = precision_recall_fscore_support(
            y_bin,
            x_bin,
            average='macro',
            zero_division=0
        )
        macro_metrics = [round(v, 4) for v in macro_metrics if v is not None]
        micro_metrics = [round(v, 4) for v in micro_metrics if v is not None]
        eval_logger.info(
            "[{:.2f}]".format(rt) +
            f"\n   micro-P: {round(100 * micro_metrics[0], 4)}%"
            f"\n   micro-R: {round(100 * micro_metrics[1], 4)}%"
            f"\n   micro-F1: {round(100 * micro_metrics[2], 4)}%"
            f"\n   macro-P: {round(100 * macro_metrics[0], 4)}%"
            f"\n   macro-R: {round(100 * macro_metrics[1], 4)}%"
            f"\n   macro-F1: {round(100 * macro_metrics[2], 4)}%"
        )

        results['thresholds'][rt] = {
            'micro-precision': micro_metrics[0],
            'micro-recall': micro_metrics[1],
            'micro-f1_score': micro_metrics[2],
            'macro-precision': macro_metrics[0],
            'macro-recall': macro_metrics[1],
            'macro-f1_score': macro_metrics[2]
        }

    with np.errstate(invalid='ignore'):
        micro_ap_score = average_precision_score(
            y_bin,
            x_cont,
            average='micro'
        )
        eval_logger.info(f"[micro-AP] {round(micro_ap_score, 4)}")
        results['micro-AP'] = micro_ap_score

        macro_ap_score = average_precision_score(
            y_bin,
            x_cont,
            average='macro'
        )
        eval_logger.info(f"[macro-AP] {round(macro_ap_score, 4)}")
        results['macro-AP'] = macro_ap_score

    return results


def log_sequence_results_into_board(
    tb_writer: SummaryWriter,
    results: Dict,
    epoch: int
):
    for th, values in results["thresholds"].items():
        tb_writer.add_scalar(
            f"_micro-Precision/{th}",
            values["micro-precision"],
            epoch
        )
        tb_writer.add_scalar(
            f"_macro-Precision/{th}",
            values["macro-precision"],
            epoch
        )
        tb_writer.add_scalar(
            f"_micro-Recall/{th}",
            values["micro-recall"],
            epoch
        )
        tb_writer.add_scalar(
            f"_macro-Recall/{th}",
            values["macro-recall"],
            epoch
        )
        tb_writer.add_scalar(
            f"_micro-F1_score/{th}",
            values["micro-f1_score"],
            epoch
        )
        tb_writer.add_scalar(
            f"_macro-F1_score/{th}",
            values["macro-f1_score"],
            epoch
        )

    # AP - score
    tb_writer.add_scalar(
        "Detection/micro-AP",
        results["micro-AP"],
        epoch
    )
    tb_writer.add_scalar(
        "Detection/macro-AP",
        results["macro-AP"],
        epoch
    )


def save_predictions_as_pickle(pred_to_save: Dict, folder: str, epoch: Optional[int] = None):
    model_name = str(epoch) if epoch else 'final'
    file_name = f'predictions_{model_name}_{datetime.datetime.now().strftime(DATETIME_FORMAT)}.p'

    pickle.dump(
        pred_to_save,
        open(os.path.join(folder,file_name), 'wb')
    )


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
    parser.add_argument(
        "--model", "-m",
        help="Folder with trained model", required=True
    )
    parser.add_argument(
        "--meta", "-M",
        help="Meta data for CS/CV datasets", required=True
    )
    parser.add_argument(
        "--data-actions", "-da",
        help="File with data for action evaluation", required=True
    )
    parser.add_argument(
        "--data-sequences", "-ds",
        help="File with data for sequence evaluation"
    )
    parser.add_argument(
        "--short-memory", "-sm",
        help="If set to True model will keep short memory",
        type=bool, default=False
    )
    parser.add_argument(
        "--frame-size", "-f",
        help="Number of frames to take for classification",
        type=int, default=1
    )
    parser.add_argument(
        "--epoch", "-e",
        help="Define type of model to load by epoch, can be a list",
        type=int, nargs='+'
    )
    parser.add_argument(
        "--board", "-b",
        help="Tensor board older to log evaluation results",
        type=str
    )
    args = parser.parse_args()

    # args = CustomArg()
    # work_dir = os.path.dirname(os.path.dirname(__file__))
    #
    # args.model = os.path.join(work_dir, "output_models", "cross_subject-batch_1_lr_0001_rs_2020_12_26-12_50_19")
    # args.data_sequences = os.path.join(work_dir, "data", "sequences-single-subject-all-POS.data")
    # args.meta = os.path.join(work_dir, "data", "meta", "cross-subject.txt")
    # args.data_actions = os.path.join(work_dir, "data", "actions-single-subject-all-POS.data")

    logger_manager.init_eval_logger(args.model)
    tb_writer = None
    if args.board:
        tb_writer = SummaryWriter(log_dir=args.board)

    epochs = args.epoch
    if not epochs:
        epochs = [None]

    for input_epoch in epochs:
        trained_model, model_epoch, model_config = load_model(
            args.model,
            input_epoch
        )

        predictions = evaluate_sequence(
            trained_model=trained_model,
            sequence_loader=DataLoader(
                SequenceDataset(args.data_sequences, args.meta, train_mode=False),
                batch_size=1
            ),
            action_dataset=ActionDatasetIterative(args.data_actions, args.meta, train_mode=False),
            keep_short_memory=args.short_memory,
            frame_size=args.frame_size
        )
        save_predictions_as_pickle(predictions, args.model, model_epoch)
        results = get_sequence_statistics(predictions)

        if args.board and tb_writer:
            assert os.path.exists(args.board), "TensorBaord folder does not exist"
            log_sequence_results_into_board(
                tb_writer=tb_writer,
                results=results,
                epoch=model_epoch
            )
        else:
            print("-> Skipping logging into tensor-board, one of arguments was not specified")

    if tb_writer:
        tb_writer.close()
