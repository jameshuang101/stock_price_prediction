import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

# from torcheval.metrics.functional import r2_score
# import wandb
import math
import os
import pandas as pd
import numpy as np
from src.utils import save_model
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from src.components.classifiers import mlp
from src.components.dataset import StockDataset
from functools import partial
import tempfile
from pathlib import Path
import sys

STOCK = "AAPL"
DATASET_PATH = os.path.join(
    Path(__file__).parent.parent.parent.parent.resolve(), "artifacts/datasets", STOCK
)
MODEL_PATH = os.path.join(
    Path(__file__).parent.parent.parent.parent.resolve(), "artifacts/models", STOCK
)
RESULTS_PATH = os.path.join(
    Path(__file__).parent.parent.parent.resolve(), "ray_results/"
)
CONFIG = {
    "input_size": 35,
    "output_size": 3,
    "layer1_size": tune.choice([32, 64, 128, 256]),
    "layer2_size": tune.choice([0, 32, 64, 128, 256]),
    "activation": tune.choice(["relu", "tanh", "sigmoid"]),
    "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
    "trend_weight": tune.choice(
        [1, 2, 5, 8]
    ),  # Weights corresponding to the classes trend, peaks, and valleys
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64]),
}


def load_data(data_dir: str):
    trainset = StockDataset(dict_path=os.path.join(data_dir, "train_dataset.pkl"))
    valset = StockDataset(dict_path=os.path.join(data_dir, "val_dataset.pkl"))
    return trainset, valset


def train_model(config, data_dir: str):
    model = mlp.MLPClassifier(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_layers=[config["layer1_size"], config["layer2_size"]],
        activation=config["activation"],
        dropout=config["dropout"],
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weights=[config["trend_weight"], 1, 1])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, valset = load_data(data_dir)

    trainloader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=12
    )
    valloader = DataLoader(
        valset, batch_size=config["batch_size"], shuffle=True, num_workers=12
    )

    try:
        for epoch in range(start_epoch, config["epochs"]):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                if i % 100 == 0:
                    print(
                        f"Epoch: {epoch}, Step: {i}, Loss: {running_loss / epoch_steps}"
                    )
                    running_loss = 0.0

            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss / val_steps, "accuracy": correct / total},
                    checkpoint=checkpoint,
                )
    except Exception as e:
        raise CustomException(e, sys)

    print("Training complete!")


def main(num_samples: int, max_num_epochs: int, gpus_per_trial: int):
    data_dir = DATASET_PATH
    load_data(data_dir)
    config = CONFIG
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=20,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_model, data_dir=data_dir),
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path=RESULTS_PATH,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    main(num_samples=1, max_num_epochs=1000, gpus_per_trial=1)
