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
from stock_prediction.utils import save_model
from stock_prediction.exception import CustomException
from sklearn.preprocessing import StandardScaler
from stock_prediction.components.classifiers import mlp
from stock_prediction.components.dataset import StockDataset
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
    "layer1_size": tune.choice([32, 64, 128]),
    "layer2_size": tune.choice([0, 32, 64, 128]),
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

    pos_weight = torch.tensor([config["trend_weight"], 1, 1], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        start_epoch = 1

    trainset, valset = load_data(data_dir)

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=12
    )
    val_loader = DataLoader(
        valset, batch_size=config["batch_size"], shuffle=True, num_workers=12
    )
    for epoch in range(start_epoch, 1001):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += torch.numel(batch_y)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(
            f"Epoch [{epoch}/1000]  "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  "
            f"Val Acc: {val_accuracy:.4f}"
        )

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
                {"loss": avg_val_loss, "accuracy": val_accuracy},
                checkpoint=checkpoint,
            )

    print("Training complete!")


def main(num_samples: int, max_num_epochs: int, gpus_per_trial: int):
    data_dir = DATASET_PATH
    load_data(data_dir)
    config = CONFIG
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=3,
    )
    result = tune.run(
        partial(train_model, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
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
    main(num_samples=24, max_num_epochs=1000, gpus_per_trial=1)
