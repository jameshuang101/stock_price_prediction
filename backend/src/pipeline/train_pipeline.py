import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torcheval.metrics.functional import r2_score
import wandb
import math
import os
import pandas as pd
import numpy as np
from src.utils import save_model
from sklearn.preprocessing import StandardScaler
from src.components.classifiers import lstm
from functools import partial
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

STOCK = "AAPL"
DATASET_PATH = "/media/james/49c33c9d-6813-4c46-b907-cc3d5f3ba7f4/repos/stock_price_prediction/artifacts/datasets/{STOCK}/"
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, f"train_dataset.pkl")
VAL_DATASET_PATH = os.path.join(DATASET_PATH, f"val_dataset.pkl")
MODEL_PATH = "/media/james/49c33c9d-6813-4c46-b907-cc3d5f3ba7f4/repos/stock_price_prediction/artifacts/models/{STOCK}/"


torch.set_default_device("cuda:0")

if __name__ == "__main__":
    wandb.init(name="AAPL_1layerLSTM-1FC-dropout-adam")
    name = "AAPL"
    DATA_PATH = "/home/james/Projects/stock_price_prediction/artifacts/datasets"
    train_df = pd.read_pickle(os.path.join(DATA_PATH, f"{name}/train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(DATA_PATH, f"{name}/val_df.pkl"))
    full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    train_size = int(len(full_df) * 0.8)
    scaler = StandardScaler()
    full_df = scaler.fit_transform(full_df)
    X, y = lstm.reformat_lstm_data(full_df, look_back=60)
    X_train = torch.tensor(X[:train_size]).to(torch.float32)
    y_train = torch.tensor(y[:train_size]).to(torch.float32)
    X_val = torch.tensor(X[train_size:]).to(torch.float32)
    y_val = torch.tensor(y[train_size:]).to(torch.float32)

    model = lstm.LSTMModel(num_classes=1, input_size=11, hidden_size=64, num_layers=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=8,
        generator=torch.Generator(device="cuda:0"),
    )
    model = lstm.train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        loader=loader,
        val_step=1,
        model_save_path=os.path.join(
            "/home/james/Projects/stock_price_prediction/artifacts/models",
            "AAPL_1layerLSTM-1FC.pt",
        ),
    )
    wandb.finish()
