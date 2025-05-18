import torch

import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        activation="tanh",
        dropout: float = 0.0,
    ):
        """
        Initialize the MLPClassifier.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes.
            hidden_layers (list of int): List specifying the number of neurons in each hidden layer.
            activation (str, optional): Activation function to use. Defaults to "tanh".
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(MLPClassifier, self).__init__()
        if input_size <= 0:
            raise ValueError("Invalid input size")
        if output_size <= 0:
            raise ValueError("Invalid output size")
        if len(hidden_layers) == 0:
            raise ValueError("Invalid hidden layers")
        if activation not in ["relu", "tanh", "sigmoid"]:
            raise ValueError("Invalid activation function")
        if dropout < 0 or dropout > 1:
            raise ValueError("Invalid dropout rate")

        # Create a list to hold all layers
        layers = []

        # Input layer to the first hidden layer
        prev_size = input_size
        for hidden_size in hidden_layers:
            if hidden_size == 0:
                continue
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("Invalid activation function")
            prev_size = hidden_size

        # Add dropout layer
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Last hidden layer to the output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.model(x)
