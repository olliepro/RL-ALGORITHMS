import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from GradUtils.hooks import add_hooks
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, continuous: bool, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.continuous = continuous
        self.input_layer = nn.Linear(input_dim, input_dim * 10)
        self.hidden1 = nn.Linear(input_dim * 10, input_dim * 10)
        self.hidden2 = nn.Linear(input_dim * 10, 8)
        self.output_layer = nn.Linear(8, output_dim)
        self.activation = nn.ReLU()
        # self.identity = nn.Identity()
        self.clamp = nn.Hardtanh()
        self.entropy_reg = 0.1
        self.log_std_devs = torch.ones(output_dim).cuda() * -0.1
        add_hooks(self)

    def sample_action(
        self, state: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.continuous:
            means, stds = self(state)
            if np.random.rand() < torch.exp(self.log_std_devs):
                action = self.clamp(torch.normal(means, stds * temperature))
            else:
                action = self.clamp(means)
            log_prob = torch.distributions.MultivariateNormal(
                means, torch.diag(torch.exp(self.log_std_devs))
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

    def get_distributions(
        self, state: torch.Tensor
    ) -> torch.distributions.Distribution:
        if self.continuous:
            means, stds = self(state)
            import matplotlib.pyplot as plt

            plot_means = means.detach().cpu().squeeze(1).numpy().tolist() + [-1, 1]
            plt.hist(plot_means, bins=50)
            plt.xlim(-1, 1)
            plt.savefig("means.png")
            plt.close()
            cov = torch.diag_embed(stds)
            return torch.distributions.MultivariateNormal(means, cov)
        else:
            raise NotImplementedError("Discrete action spaces not implemented")

    def get_kl_divergence(
        self,
        state: torch.Tensor,
        target_dist: torch.distributions.Distribution,
    ) -> torch.Tensor:
        current_dist = self.get_distributions(state)
        return torch.distributions.kl_divergence(current_dist, target_dist)

    def get_log_probs(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.continuous:
            means, stds = self(state)
            cov = torch.diag_embed(stds)
            log_probs = torch.distributions.MultivariateNormal(
                means,
                cov,
            ).log_prob(actions.unsqueeze(-1))
            return means, log_probs
        else:
            logits = self(state)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[torch.arange(log_probs.shape[0]), actions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        mus = self.output_layer(x)
        if self.continuous:
            stds = mus * 0 + self.log_std_devs

            return mus, torch.exp(stds)

        return mus

    def loss_fn(
        self,
        probs: torch.Tensor,
        batch_A_hat: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        old_probs = probs.detach()
        means, new_log_probs = self.get_log_probs(states, actions)
        new_probs = torch.exp(new_log_probs)
        # dists = self.get_distributions(states)
        losses = (new_probs / old_probs) * batch_A_hat

        return -torch.mean(losses)  # - means.var()


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
