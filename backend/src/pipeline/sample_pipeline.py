import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from src.components.dataset import StockDataset
import os
import numpy as np

# Assume X_train, y_train, X_val, y_val are your data arrays
# Replace the following lines with loading your actual data
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


def load_data(data_dir: str):
    trainset = StockDataset(dict_path=os.path.join(data_dir, "train_dataset.pkl"))
    valset = StockDataset(dict_path=os.path.join(data_dir, "val_dataset.pkl"))
    return trainset, valset


# Define the MLP model
class MultiLabelMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super(MultiLabelMLP, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        # layers.append(nn.Sigmoid())  # For multi-label output
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Hyperparameters
input_size = 35
hidden_sizes = [64, 32]
output_size = 3
lr = 1e-3
num_epochs = 20

# Initialize model, loss, and optimizer
device = "cuda:0"
model = MultiLabelMLP(input_size, hidden_sizes, output_size).to(device)
pos_weight = torch.tensor([0.8, 0.1, 0.1], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)

batch_size = 64
trainset, valset = load_data(DATASET_PATH)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=12)


# Training and validation loop
for epoch in range(1, num_epochs + 1):
    # Training
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
        f"Epoch [{epoch}/{num_epochs}]  "
        f"Train Loss: {avg_train_loss:.4f}  "
        f"Val Loss: {avg_val_loss:.4f}  "
        f"Val Acc: {val_accuracy:.4f}"
    )

print("Training and validation complete.")
