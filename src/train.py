import argparse
import torch
import time
import os
from torch import nn
from typing import Optional
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
    else:
        model_config = load_config_file(config_file)
        log_folder = os.path.join(
            output_folder,
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        )

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
    board_writer = SummaryWriter(log_dir=log_folder)

    model = BiRNN(
        input_size=model_config["model"]["input_size"],
        lstm_hidden_size=model_config["model"]["hidden_size"],
        embedding_output_size=model_config["model"]["embedding_input_size"],
        num_classes=len(LABELS),
        device=device
    )
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
    for epoch in range(start_epoch + 1, end_epoch + 1):
        s_time = time.time()
        epoch_loss = 0.0
        iteration_loss = 0.0

        for i, (sequence, label) in enumerate(train_loader, 1):
            if i == 5:
                break

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

            # Model evaluation
            if i % model_config["train"]["evaluation_step"] == 0:
                model_evaluation(board_writer, test_loader)

            # Training info
            if i % model_config["train"]["report_step"] == 0:
                average_loss = round(iteration_loss / model_config['train']['report_step'], 6)
                print(
                    f"{i}/{len(train_loader)} - Epoch [{epoch}/{end_epoch}], "
                    f"avg_loss: {average_loss}"
                )
                iteration_loss = 0.0

                board_writer.add_scalar(
                    "Average_Loss/train",
                    average_loss,
                    (epoch * len(train_loader)) + i
                )

        # TODO - check 'i' a len(train_loader) pri vacsom batchi
        print(f"Epoch time: {int(time.time() - s_time)}s. total_loss: {round(epoch_loss / len(train_loader), 6)}")

        if epoch % model_config['train']['save_step'] == 0:
            save_model(model_config, log_folder, model, epoch, final_mode=False)

    # save final_model
    save_model(model_config, log_folder, model, end_epoch, final_mode=True)


def model_evaluation(
    tb_writer,
    evaluation_loader: DataLoader
):
    # TODO
    pass


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
