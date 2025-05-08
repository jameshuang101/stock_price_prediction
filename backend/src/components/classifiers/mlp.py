import torch

import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        """
        Initialize the MLPClassifier.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes.
            hidden_layers (list of int): List specifying the number of neurons in each hidden layer.
        """
        super(MLPClassifier, self).__init__()

        # Create a list to hold all layers
        layers = []

        # Input layer to the first hidden layer
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Last hidden layer to the output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
