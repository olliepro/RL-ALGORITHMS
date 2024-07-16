import torch
import torch.nn as nn
from GradUtils.hooks import add_hooks


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(ValueNetwork, self).__init__()
        self.input_layer = nn.Linear(input_dim, input_dim * 10)
        self.hidden1 = nn.Linear(input_dim * 10, input_dim * 10)
        self.hidden2 = nn.Linear(input_dim * 10, 8)
        self.output_layer = nn.Linear(8, 1)
        self.activation = nn.ReLU()
        add_hooks(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output_layer(x)
        return x

    def loss_fn(self, values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((values - rewards) ** 2)
        return loss
