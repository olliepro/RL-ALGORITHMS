import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from GradUtils.hooks import add_hooks


class PolicyNetwork(nn.Module):
    def __init__(self, continuous: bool, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        hidden_dim = 50
        self.continuous = continuous
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.identity = nn.Identity()
        if self.continuous:
            self.std_devs = nn.Parameter(torch.ones(output_dim) * 1)
        add_hooks(self)

    def sample_action(
        self, state: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.continuous:
            means, stds = self(state)
            action = torch.normal(means, stds * temperature)
            log_prob = torch.distributions.MultivariateNormal(
                means, torch.diag(torch.exp(self.std_devs))
            ).log_prob(action)
            return action.view(-1), log_prob.view(-1)

        else:
            logits = self(state)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            # apply temperature
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(input=probs, num_samples=1)
                log_prob = torch.log(
                    probs[torch.arange(probs.shape[0]), action.view(-1).detach()]
                )
                return action.view(-1), log_prob.view(-1)
            else:
                action = torch.argmax(logits)
                log_prob = torch.log_softmax(logits, dim=0)[
                    torch.arange(logits.shape[0]), action.view(-1)
                ]
                return action.view(-1), log_prob.view(-1)

    def get_log_probs(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.continuous:
            means, stds = self(state)
            cov = torch.diag_embed(stds)
            log_probs = torch.distributions.MultivariateNormal(
                means,
                cov,
            ).log_prob(actions.unsqueeze(-1))
            return log_probs
        else:
            logits = self(state)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[torch.arange(log_probs.shape[0]), actions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        mus = self.fc2(x)
        if self.continuous:
            stds = mus * 0 + self.std_devs

            return mus, torch.exp(self.identity(stds))

        return mus

    def loss_fn(
        self,
        probs: torch.Tensor,
        batch_A_hat: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        old_probs = probs.detach()
        new_log_probs = self.get_log_probs(states, actions)
        new_probs = torch.exp(new_log_probs)
        losses = (new_probs / old_probs) * batch_A_hat
        return -torch.mean(losses)


import torch


def test_policy_network():
    input_dim = 4
    output_dim = 10
    batch_size = 1

    # Test continuous action space
    continuous = True
    continuous_policy_net = PolicyNetwork(continuous, input_dim, output_dim)
    continuous_states = torch.randn(batch_size, input_dim)
    continuous_actions = continuous_policy_net.sample_action(
        continuous_states.view(-1), 1
    )
    print("Continuous Actions:\n", continuous_actions)

    # Test discrete action space
    continuous = False
    discrete_policy_net = PolicyNetwork(continuous, input_dim, output_dim)
    discrete_states = torch.randn(batch_size, input_dim)
    discrete_actions = discrete_policy_net.sample_action(discrete_states)

    print("Discrete Actions:\n", discrete_actions)


# Run the test
# test_policy_network()
