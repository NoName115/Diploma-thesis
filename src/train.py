import argparse
import torch
import time
import os
import json
from torch import nn
from typing import Optional, Dict
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import evaluation, constants
from src.loader import ActionDataset, SequenceDataset, load_config_file, load_model, create_model
from src.model import save_model, BiRNN
from src.common import get_device


def train(
    action_file: str,
    sequence_file: str,
    meta_file: str,
    config_file: str,
    max_epochs: int,
    output_folder: str,
    model_folder: Optional[str],
    retrain: bool,
    additional_log_folder_name: str
):
    # initialize starting parameters
    device = get_device()
    print(f"Training on: {device}")

    start_epoch = 0
    end_epoch = max_epochs

    # load model parameters and model configuration
    if model_folder:
        assert os.path.exists(model_folder), "Model folder doesn't exist"
        model, start_epoch, model_config = load_model(model_folder)
        log_folder = model_folder
        if retrain:
            model = create_model(model_config)
    else:
        model_config = load_config_file(config_file)
        log_folder = os.path.join(
            output_folder,
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        )
        if additional_log_folder_name:
            log_folder += "_" + additional_log_folder_name

        model = create_model(model_config)

    # set model into training mode
    model.train()

    # initialize training
    print(f"Log directory for training: {log_folder}")
    board_writer = SummaryWriter(log_dir=log_folder)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["train"]["learning_rate"],
        weight_decay=model_config["train"]["l2_weight_decay"]
    )

    # load training & testing data
    train_loader = DataLoader(
        ActionDataset(action_file, meta_file, train_mode=True),
        batch_size=model_config["train"]["batch_size"]
    )
    action_loader = DataLoader(
        ActionDataset(action_file, meta_file, train_mode=False),
        batch_size=model_config["train"]["batch_size"]
    )
    sequence_loader = DataLoader(
        SequenceDataset(sequence_file, meta_file, train_mode=False),
        batch_size=1
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
                    "Overall/Iteration_Loss",
                    average_loss,
                    (epoch * len(train_loader)) + i
                )

        # TODO - check 'i' a len(train_loader) pri vacsom batchi
        print(f"Epoch time: {int(time.time() - s_time)}s. total_loss: {round(epoch_loss / len(train_loader), 6)}")

        board_writer.add_scalar(
            "Overall/Epoch_Loss",
            round(epoch_loss / len(train_loader), 6),
            epoch
        )

        if epoch % model_config['train']['save_step'] == 0:
            save_model(model_config, log_folder, model, epoch, final_mode=False)

        # Model evaluation
        if epoch % model_config["train"]["evaluation_step"] == 0:
            sequence_evaluation(
                configuration=model_config,
                tb_writer=board_writer,
                evaluation_loader=action_loader,
                trained_model=model,
                epoch=epoch
            )
            action_evaluation(
                configuration=model_config,
                tb_writer=board_writer,
                evaluation_loader=action_loader,
                trained_model=model,
                epoch=epoch
            )

    # save final_model
    save_model(model_config, log_folder, model, end_epoch, final_mode=True)


def sequence_evaluation(
    configuration: Dict,
    tb_writer: SummaryWriter,
    evaluation_loader: DataLoader,
    trained_model: BiRNN,
    epoch: int,
):
    res = evaluation.evaluate_sequences(
        trained_model,
        configuration,
        evaluation_loader
    )

    for th, values in res["thresholds"].items():
        tb_writer.add_scalar(
            f"Precision/{th}",
            values["precision"],
            epoch
        )
        tb_writer.add_scalar(
            f"Recall/{th}",
            values["recall"],
            epoch
        )
        tb_writer.add_scalar(
            f"F1-score/{th}",
            values["f1-score"],
            epoch
        )

    # AP - score
    tb_writer.add_scalar(
        f"Overall/AP",
        res["AP"],
        epoch
    )

    res["epoch"] = epoch

    # Save statistics into metrics.json file
    with open(os.path.join(tb_writer.get_logdir(), constants.METRICS_FILE_NAME), 'a') as mf:
        json.dump(res, mf)
        mf.write('\n')


def action_evaluation(
    configuration: Dict,
    tb_writer: SummaryWriter,
    evaluation_loader: DataLoader,
    trained_model: BiRNN,
    epoch: int,
):
    correct, total = evaluation.evaluate_actions(
        trained_model,
        configuration,
        evaluation_loader
    )

    tb_writer.add_scalar(
        f"Overall/Accuracy",
        round(correct / total, 6),
        epoch
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training the model")
    parser.add_argument("--data", "-d", help="File with all input data", required=True, type=str)
    parser.add_argument("--data-sequence", "-ds", help="File with sequences", type=str)

    parser.add_argument("--meta", "-m", help="Meta data for training/validation split", required=True, type=str)
    parser.add_argument("--epochs", "-e", help="Number of maximum epochs for training", required=True, type=int)
    parser.add_argument("--name", "-n", help="Additional name to the log folder", default="", type=str)

    parser.add_argument("--model", "-M", help="Folder with pre-trained model", type=str)
    parser.add_argument("--config", "-c", help="Configuration file for model training", type=str)
    parser.add_argument("--output", "-o", help="Output folder for model training", type=str)

    parser.add_argument("--retrain", "-r", help="If other than 0, will train model from the beginning",
                        default=0, type=int)
    args = parser.parse_args()

    if not args.model and (not args.config or not args.output):
        parser.error("Either --model or --config & --output arguments must be present")

    train(
        args.data,
        args.data_sequence,
        args.meta,
        args.config,
        args.epochs,
        args.output,
        args.model,
        args.retrain != 0,
        args.name
    )
