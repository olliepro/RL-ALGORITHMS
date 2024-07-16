import torch
from copy import deepcopy
from Training.utils import conjugate_gradient, EmpiricalFisher
import numpy as np
from CartPole.policy_net import PolicyNetwork


def update_parameters(
    network: torch.nn.Module,
    step: torch.Tensor,
) -> None:
    """
    Update the parameters of the network using the step
    """
    with torch.no_grad():
        current_start = 0
        for param in network.parameters():
            shape = param.shape
            n_params = np.prod(shape)
            param += step[current_start : current_start + n_params].reshape(shape)
            current_start += n_params


def policy_update(
    policy_net: PolicyNetwork,
    iEF: EmpiricalFisher,
    old_search_direction: torch.Tensor,
    old_policy_loss: torch.Tensor,
    grad_policy_loss: torch.Tensor,
    iteration_probs: torch.Tensor,
    batch_A_hat: torch.Tensor,
    iteration_actions: torch.Tensor,
    iteration_states: torch.Tensor,
    trust_region: float,
) -> torch.Tensor:

    search_dir, resid_norm = conjugate_gradient(
        A=iEF,
        b=-grad_policy_loss,
        x0=old_search_direction,
    )
    alpha = torch.sqrt((2 * trust_region) / (search_dir @ (iEF @ search_dir)))
    step = alpha * search_dir
    i = 0
    # Update the policy network ensuring improvement after a line search

    old_params = deepcopy(policy_net.state_dict())
    old_dists = policy_net.get_distributions(iteration_states)

    while True:
        i += 1

        update_parameters(policy_net, step)
        new_dists = policy_net.get_distributions(iteration_states)
        kl_div = torch.distributions.kl_divergence(old_dists, new_dists).mean()
        new_policy_loss = policy_net.loss_fn(
            iteration_probs, batch_A_hat, iteration_actions, iteration_states
        )
        if (new_policy_loss - old_policy_loss) / torch.abs(
            old_policy_loss
        ) <= 0.01 and kl_div <= trust_region:
            break
        elif i > 30:
            print("Line search failed")
            policy_net.load_state_dict(old_params)
            break
        else:
            policy_net.load_state_dict(old_params)
            step = 0.8 * step

    return search_dir, policy_net


def value_update(
    value_net: torch.nn.Module,
    iEF: EmpiricalFisher,
    old_value_direction: torch.Tensor,
    grad_value_loss: torch.Tensor,
    trust_region: float,
) -> torch.Tensor:

    search_dir, resid_norm = conjugate_gradient(
        A=iEF,
        b=-grad_value_loss,
        x0=old_value_direction,
    )
    value_alpha = torch.sqrt(2 * trust_region / (search_dir @ (iEF @ search_dir)))
    value_step = value_alpha * search_dir
    update_parameters(value_net, value_step)
    return search_dir
