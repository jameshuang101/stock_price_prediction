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


torch.set_default_device("cuda:0")


from torch.nn import BCEWithLogitsLoss, Sigmoid


# Define LSTM classifier model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_1(out[:, -1, :])
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_2(out)
        # out = BCEWithLogitsLoss()(out, reduction="none")
        return out


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model: LSTMClassifier,
    optimizer,
    loss_fn,
    loader,
    n_epochs: int = 2000,
    val_step: int = 25,
    model_save_path: str = None,
):
    X_train_cuda, y_train_cuda = X_train.cuda(), y_train.cuda()
    X_val_cuda, y_val_cuda = X_val.cuda(), y_val.cuda()
    wandb.watch(model)
    val_loss = math.inf
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % val_step != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train_cuda).squeeze()
            train_rmse = np.sqrt(loss_fn(y_pred, y_train_cuda).cpu())
            train_r2 = r2_score(y_pred, y_train_cuda)
            wandb.log({"Training RMSE": train_rmse})
            wandb.log({"Training R2 Score": train_r2})

            y_pred = model(X_val_cuda).squeeze()
            val_rmse = np.sqrt(loss_fn(y_pred, y_val_cuda).cpu())
            val_r2 = r2_score(y_pred, y_val_cuda)
            wandb.log({"Validation RMSE": val_rmse})
            wandb.log({"Validation R2 Score": val_r2})
            print(
                f"Epoch {epoch} summary: Train RMSE {train_rmse}, Train R2 Score {train_r2}, Val RMSE {val_rmse}, Val R2 Score {val_r2}"
            )
            if model_save_path:
                if val_rmse < val_loss:
                    print("Saving model...")
                    val_loss = val_rmse
                    torch.save(model.state_dict(), model_save_path)

    return model
