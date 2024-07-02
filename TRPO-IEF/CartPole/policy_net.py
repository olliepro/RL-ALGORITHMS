import torch
import torch.nn as nn
from GradUtils.hooks import add_hooks


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        add_hooks(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def loss_fn(
        self,
        probs: torch.Tensor,
        batch_A_hat: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        old_probs = probs.detach()
        new_probs = self(states)
        new_probs = new_probs[torch.arange(new_probs.shape[0]), actions]
        losses = (new_probs / old_probs) * batch_A_hat
        return -torch.mean(losses)
