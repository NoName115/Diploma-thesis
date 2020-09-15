import argparse
import torch
import time
import os
import yaml
from torch import nn
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.loader import IterableMovementDataset, load_config_file
from src.model import BiRNN, save_model
from src.constants import LABELS, CONFIG_FILE_NAME, CHECKPOINT_FILE_NAME


def train(
    train_folder: str,
    test_folder: str,
    config_file: str,
    max_epochs: int,
    output_folder: str,
    model_folder: Optional[str],
    retrain: bool
):
    # initialize starting parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    start_epoch = 0
    end_epoch = max_epochs

    # TODO - print info about loading model and etc...

    if model_folder:
        assert os.path.exists(model_folder), "Model folder doesn't exist"
        model_config = load_config_file(os.path.join(model_folder, CONFIG_FILE_NAME))
        log_folder = model_folder
        print(f"Model loaded from {model_folder}")
    else:
        model_config = load_config_file(config_file)
        log_folder = os.path.join(
            output_folder,
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        )

    print("-" * 8 + " CONFIGURATION " + "-" * 8)
    print(yaml.dump(model_config))

    # load training & testing data
    train_loader = DataLoader(
        IterableMovementDataset(train_folder),
        batch_size=model_config["train"]["batch_size"]
    )
    test_loader = DataLoader(
        IterableMovementDataset(test_folder),
        batch_size=model_config["train"]["batch_size"]
    )

    # initialize training
    print(f"Log directory for training: {log_folder}")
    board_writer = SummaryWriter(log_dir=log_folder)

    model = BiRNN(
        input_size=model_config["model"]["input_size"],
        lstm_hidden_size=model_config["model"]["hidden_size"],
        embedding_output_size=model_config["model"]["embedding_input_size"],
        num_classes=len(LABELS),
        device=device
    ).to(device)
    # load model parameters if model already exist
    if model_folder and not retrain:
        with open(os.path.join(model_folder, CHECKPOINT_FILE_NAME), "r") as lf:
            last_model_file, model_epoch = lf.read().split('\n')
        model.load_state_dict(torch.load(os.path.join(model_folder, last_model_file)))
        start_epoch = int(model_epoch)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["train"]["learning_rate"],
        weight_decay=model_config["train"]["l2_weight_decay"]
    )

    # train the model
    print("--" * 15)
    print("-" * 10 + " TRAINING " + "-" * 10)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        s_time = time.time()
        epoch_loss = 0.0
        iteration_loss = 0.0

        for i, (sequence, label) in enumerate(train_loader, 1):
            sequence = sequence.view(sequence.size(0), sequence.size(1), -1).to(device)
            label = label.to(device)

            # forward pass
            output = model.forward(sequence)

            # loss calculation
            loss = criterion(output, label)
            epoch_loss += loss.item()
            iteration_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Training info
            if i % model_config["train"]["report_step"] == 0:
                average_loss = round(iteration_loss / model_config['train']['report_step'], 6)
                print(
                    f"\t{i}/{len(train_loader)} - Epoch [{epoch}/{end_epoch}], "
                    f"avg_loss: {average_loss}"
                )
                iteration_loss = 0.0

                board_writer.add_scalar(
                    "Train/Average_Loss",
                    average_loss,
                    (epoch * len(train_loader)) + i
                )

        # TODO - check 'i' a len(train_loader) pri vacsom batchi
        print(f"Epoch time: {int(time.time() - s_time)}s. total_loss: {round(epoch_loss / len(train_loader), 6)}")

        if epoch % model_config['train']['save_step'] == 0:
            save_model(model_config, log_folder, model, epoch, final_mode=False)

        # Model evaluation
        if epoch % model_config["train"]["evaluation_step"] == 0:
            model_evaluation(
                configuration=model_config,
                tb_writer=board_writer,
                evaluation_loader=test_loader,
                trained_model=model,
                device=device,
                epoch=epoch
            )

    # save final_model
    save_model(model_config, log_folder, model, end_epoch, final_mode=True)


def model_evaluation(
    configuration: Dict,
    tb_writer: SummaryWriter,
    evaluation_loader: DataLoader,
    trained_model,
    device,
    epoch: int
):
    print("-" * 6 + " EVALUATION " + "-" * 6)

    with torch.no_grad():
        eval_dict = {
            "correct": 0,
            "above": 0,
        }
        thresholds = [(
            round((1 / configuration['evaluation']['threshold_steps']) * (i + 1), 4),
            eval_dict.copy()
        ) for i in range(configuration['evaluation']['threshold_steps'])]

        for i, (sequence, labels) in enumerate(evaluation_loader, 1):
            sequence = sequence.view(sequence.size(0), sequence.size(1), -1).to(device)
            target_label = torch.argmax(labels).item()

            outputs = trained_model(sequence)

            for val_th, res_dict in thresholds:
                res_dict["above"] += (outputs.data > val_th).sum().item()

                for _, label_indx in zip(*torch.where(outputs.data > val_th)):
                    if label_indx == target_label:
                        res_dict["correct"] += 1

            if i % configuration['evaluation']['report_step'] == 0:
                print(f"\tProcessed: [{i}/{len(evaluation_loader)}]")

    report_evaluation(thresholds, len(evaluation_loader), epoch, tb_writer)


def report_evaluation(
    result_thresholds: List[Tuple[float, Dict]],
    total_records: int,
    epoch: int,
    tb_writer: SummaryWriter
):
    # Calculate results & report them into tensorboard
    ap_score = 0
    old_recall = 0
    for th, values in result_thresholds[:-1]:
        recall = values["correct"] / total_records if values["correct"] > 0 else 0
        precision = values["correct"] / values["above"] if values["above"] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        tb_writer.add_scalar(
            f"Precision/{th}",
            precision,
            epoch
        )
        tb_writer.add_scalar(
            f"Recall/{th}",
            recall,
            epoch
        )
        tb_writer.add_scalar(
            f"F1-score/{th}",
            f1_score,
            epoch
        )

        print("[{:.2f}]".format(th) +
              f"\n   Precision: {round(100 * precision, 4)}%"
              f"\n   Recall: {round(100 * recall, 4)}%"
              f"\n   f1_score: {round(f1_score, 4)}")

        ap_score += ((abs(recall - old_recall)) * precision)
        old_recall = recall

    # AP - score
    tb_writer.add_scalar(
        f"Test/AP",
        ap_score,
        epoch
    )

    print(f"[AP] {round(ap_score, 4)}")


if __name__ == "__main__":
    # TODO check paths to folders & files

    parser = argparse.ArgumentParser(description="Script for training the model")
    parser.add_argument("--train-data", "-t", help="Folder with training data", required=True)
    parser.add_argument("--test-data", "-T", help="Folder with testing data", required=True)
    parser.add_argument("--epochs", "-e", help="Number of maximum epochs for training", required=True)
    parser.add_argument("--config", "-c", help="Configuration file for model training, ignored if argument --model is set", required=True)

    parser.add_argument("--model", "-m", help="Folder with pre-trained model", default=None)
    parser.add_argument("--retrain", "-r", help="If other than 0, will train model from the beginning", default=0, type=int)

    parser.add_argument("--output", "-o", help="Output folder for model training", required=True)
    args = parser.parse_args()

    train(
        args.train_data,
        args.test_data,
        args.config,
        int(args.epochs),
        args.output,
        args.model,
        args.retrain != 0
    )
