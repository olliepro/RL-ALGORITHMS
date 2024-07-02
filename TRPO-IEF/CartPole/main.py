import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from policy_net import PolicyNetwork
from value_net import ValueNetwork
from Training.utils import (
    run_iteration,
    ImprovedEmpiricalFisher,
    EmpiricalFisher,
    A_hat,
)
from Training.gradient_step import policy_update, value_update
from GradUtils.hooks import enable_hooks, compute_grad1, clear_backprops, disable_hooks
from Eval.play import run_discrete_policy


def initialize_networks(
    input_dim: int, hidden_dim: int, output_dim: int
) -> tuple[PolicyNetwork, ValueNetwork, torch.device]:
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    value_net = ValueNetwork(input_dim, hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.to(device)
    value_net.to(device)

    return policy_net, value_net, device


def run_iteration_and_compute_gradients(
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    device: torch.device,
    max_steps_per_episode: int,
    gamma: float,
    batch_size: int,
    env_name: str,
) -> tuple:
    (
        iteration_states,
        iteration_actions,
        iteration_rewards,
        iteration_disc_rewards,
        iteration_probs,
        iteration_timesteps,
        iteration_log_probs,
    ) = run_iteration(
        env_name=env_name,
        max_timesteps=batch_size,
        max_steps_per_episode=max_steps_per_episode,
        discount_factor=gamma,
        policy=policy_net,
        device=device,
    )

    enable_hooks()

    # fmt: off
    
    # Compute the value function on all states
    values = value_net(iteration_states.detach())
    mean_value = torch.mean(values)
    mean_value.backward(retain_graph=True)
    compute_grad1(value_net)

    # Collect the parameter-wise gradients & logit-wise gradients
    dV_Sn_dTheta = torch.cat([param.grad1.view(batch_size, -1) for param in value_net.parameters()], dim=1)
    grad_value_logits = value_net.fc2.backprops_list[0].clone() * batch_size
    value_net.zero_grad()
    clear_backprops(value_net)

    # Compute the policy log probabilities on all states
    iteration_log_probs = torch.log(policy_net(iteration_states.detach()))
    iteration_log_probs = iteration_log_probs[torch.arange(iteration_log_probs.shape[0]), iteration_actions]
    mean_log_probs = torch.mean(iteration_log_probs)
    mean_log_probs.backward(retain_graph=True)
    compute_grad1(policy_net)

    # Collect the parameter-wise gradients & logit-wise gradients
    dP_Sn_dTheta = torch.cat([param.grad1.view(batch_size, -1) for param in policy_net.parameters()], dim=1)
    grad_policy_logits = policy_net.fc2.backprops_list[0].clone() * batch_size
    norm_grad_policy_logits = torch.norm(grad_policy_logits, dim=1, keepdim=True)
    policy_net.zero_grad()
    clear_backprops(policy_net)

    disable_hooks()
    # fmt: on

    return (
        iteration_states,
        iteration_actions,
        iteration_rewards,
        iteration_disc_rewards,
        iteration_probs,
        iteration_timesteps,
        values,
        dV_Sn_dTheta,
        grad_value_logits,
        dP_Sn_dTheta,
        norm_grad_policy_logits,
    )


def train(
    env_name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    max_steps_per_episode: int,
    gamma: float,
    _lambda: float,
    batch_size: int,
    policy_trust_region: float,
    value_trust_region: float,
    num_iterations: int,
    EF_type: str,
    EF_damping: float,
    num_episodes_eval: int = 20,
    eval_temperature: float = 0,
):
    if EF_type == "EF":
        EF = EmpiricalFisher
    elif EF_type == "iEF":
        EF = ImprovedEmpiricalFisher
    else:
        raise ValueError(f"Invalid EF type: {EF_type}. Must be 'EF' or 'iEF'.")

    policy_net, value_net, device = initialize_networks(
        input_dim,
        hidden_dim,
        output_dim,
    )
    checkpoint_scores = []
    n_policy_params = sum(p.numel() for p in policy_net.parameters())
    n_value_params = sum(p.numel() for p in value_net.parameters())
    old_policy_direction = torch.zeros(n_policy_params, device=device)
    old_value_direction = torch.zeros(n_value_params, device=device)

    for i in range(num_iterations):
        (
            iteration_states,
            iteration_actions,
            iteration_rewards,
            iteration_disc_rewards,
            iteration_probs,
            iteration_timesteps,
            values,
            dV_Sn_dTheta,
            grad_value_logits,
            dP_Sn_dTheta,
            norm_grad_policy_logits,
        ) = run_iteration_and_compute_gradients(
            policy_net=policy_net,
            value_net=value_net,
            device=device,
            max_steps_per_episode=max_steps_per_episode,
            gamma=gamma,
            batch_size=batch_size,
            env_name=env_name,
        )

        batch_A_hat = A_hat(
            rewards=iteration_rewards,
            values=values.view(-1),
            timesteps=iteration_timesteps,
            gamma=gamma,
            _lambda=_lambda,
            max_steps_per_episode=max_steps_per_episode,
        )

        policy_loss = policy_net.loss_fn(
            iteration_probs,
            batch_A_hat,
            iteration_actions,
            iteration_states,
        )
        policy_loss.backward()

        # fmt: off
        grad_policy_loss = torch.cat([param.grad.view(-1) for param in policy_net.parameters()])
        polocy_iEF = EF(dP_Sn_dTheta.T.to(device), norm_grad_policy_logits, damping=EF_damping)
        # fmt: on

        old_policy_direction = policy_update(
            policy_net=policy_net,
            iEF=polocy_iEF,
            old_search_direction=old_policy_direction,
            old_policy_loss=policy_loss,
            grad_policy_loss=grad_policy_loss,
            iteration_probs=iteration_probs,
            batch_A_hat=batch_A_hat,
            iteration_actions=iteration_actions,
            iteration_states=iteration_states,
            trust_region=policy_trust_region,
        )

        value_loss = value_net.loss_fn(values.view(-1), iteration_disc_rewards)
        value_loss.backward()

        # fmt: off
        grad_value_loss = torch.cat([param.grad.view(-1) for param in value_net.parameters()])
        value_iEF = EF(dV_Sn_dTheta.T.to(device), grad_value_logits, damping=EF_damping)
        # fmt: on

        old_value_direction = value_update(
            value_net=value_net,
            iEF=value_iEF,
            old_value_direction=old_value_direction,
            grad_value_loss=grad_value_loss,
            trust_region=value_trust_region,
        )

        checkpoint_scores.append(
            run_discrete_policy(
                policy_net=policy_net,
                env_name=env_name,
                num_episodes=num_episodes_eval,
                max_steps_per_episode=max_steps_per_episode,
                temperature=eval_temperature,
            )
        )
        print(f"Checkpoint {i+1} score: {checkpoint_scores[-1]}")

    return checkpoint_scores


results = {"iEF": [], "EF": []}

for fim in ["iEF", "EF"]:
    for i in range(50):
        checkpoints = train(
            env_name="CartPole-v1",
            input_dim=4,
            hidden_dim=50,
            output_dim=2,
            max_steps_per_episode=1000,
            gamma=0.99,
            _lambda=0.9,
            batch_size=10000,
            policy_trust_region=1e-3,
            value_trust_region=1e-2,
            num_iterations=40,
            EF_type=fim,
            EF_damping=1e-2,
            num_episodes_eval=20,
            eval_temperature=0,
        )
        results[fim].append(checkpoints)

# save results
import json

with open("results_3.json", "w") as f:
    json.dump(results, f)
