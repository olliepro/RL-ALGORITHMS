import torch
import torch.nn as nn
from GradUtils.hooks import add_hooks


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        add_hooks(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss_fn(self, values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        loss = torch.mean((values - rewards) ** 2)
        return loss
