from tqdm import tqdm
import torch
import gymnasium as gym
import numpy as np
from typing import Union
from collections import namedtuple
import torch.multiprocessing as mp
from CartPole.policy_net import PolicyNetwork
import torch
import time


class EmpiricalFisher:
    def __init__(self, j_n: torch.Tensor, s_n, damping: float = 1e-2):
        j_n = torch.nan_to_num(j_n)

        self.J_n_params_x_batch_size = j_n
        self.damping = damping

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        batch_size = self.J_n_params_x_batch_size.shape[1]
        projections = self.J_n_params_x_batch_size.T @ other
        return (
            self.J_n_params_x_batch_size @ projections / batch_size
            + self.damping * other
        )


class ImprovedEmpiricalFisher(EmpiricalFisher):
    def __init__(self, j_n: torch.Tensor, s_n: torch.Tensor, damping: float = 1e-2):
        self.J_n_params_x_batch_size = (j_n.T / s_n).T
        self.J_n_params_x_batch_size = torch.nan_to_num(
            self.J_n_params_x_batch_size, 0, 1e3, -1e3
        )
        self.J_n_params_x_batch_size = torch.clamp(
            self.J_n_params_x_batch_size, -1e3, 1e3
        )
        self.damping = damping


def conjugate_gradient(A, b, x0, tol=1e-4, max_iters=None, stop_steps=100):
    if max_iters is None:
        max_iters = b.shape[0]
    x0 = torch.nan_to_num(x0, 0, 1e3, -1e3)
    x0 = torch.clamp(x0, -1e3, 1e3)
    b = torch.nan_to_num(b, 0, 1e3, -1e3)
    b = torch.clamp(b, -1e3, 1e3)
    x = x0
    r = b - A @ x
    d = r
    iters = 0
    done = False
    min_resid = 1e10
    unimproved_step = 0
    while not done:
        Ad = A @ d
        alpha = (r @ d) / ((Ad @ d) + 1e-8)
        change = alpha * d
        x = x + change
        r_new = b - A @ x
        beta = (Ad @ r_new) / ((Ad @ d) + 1e-8)
        d = r_new - beta * d
        r = r_new
        iters += 1
        resid_norm = torch.norm(r)
        # if iters % 100 == 0:
        #     print(f"Iteration {iters}, residual norm: {resid_norm}")
        if min_resid < resid_norm:
            unimproved_step += 1
        else:
            unimproved_step = 0
            min_resid = resid_norm
            bestsol = x

        if torch.isnan(resid_norm):
            print("Residual norm is NaN")
            import pdb

            pdb.set_trace()
            A.damping *= 10
            x = x0
            r = b - A @ x
            d = r
            continue
        done = resid_norm < tol or iters >= max_iters or unimproved_step >= stop_steps
    return bestsol, resid_norm


def compute_discounted_rewards(rewards: list, discount_factor: float) -> list:
    discounted_rewards = []
    cumulative_rewards = 0
    for reward in reversed(rewards):
        cumulative_rewards = reward + discount_factor * cumulative_rewards
        discounted_rewards.insert(0, cumulative_rewards)
    return discounted_rewards


def delta(
    rewards: torch.Tensor,
    values: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
    max_steps_per_episode,
) -> torch.Tensor:
    """
    Calculate the delta values for the GAE advantage estimates.

    Args:
        rewards (torch.Tensor): The per timestep rewards received during the episode
        values (torch.Tensor): The value estimates for the states encountered during the episode
        timesteps (torch.Tensor): The timesteps at which the states were encountered
        gamma (float): The discount factor
        max_steps_per_episode (int): The maximum number of steps per episode

    Returns:
        torch.Tensor: The delta values
    """
    deltas = (
        rewards[:-1]
        + gamma
        * values[1:]
        * (timesteps[:-1] < timesteps[1:]).float()  # current timestep < next timestep
        - values[:-1]
    )
    deltas = torch.cat((deltas, rewards[-1:] - values[-1:]))
    deltas[timesteps == max_steps_per_episode - 1] = rewards[
        timesteps == max_steps_per_episode - 1
    ]
    return deltas


def A_hat(
    rewards: torch.Tensor,
    values: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
    _lambda: float,
    max_steps_per_episode: int,
) -> torch.Tensor:
    """
    Calculate the advantage estimates for the given rewards, values, timesteps, gamma, and lambda.

    Args:
        rewards (torch.Tensor): The rewards received during the episode
        values (torch.Tensor): The value estimates for the states encountered during the episode
        timesteps (torch.Tensor): The timesteps at which the states were encountered
        gamma (float): The discount factor
        _lambda (float): The lambda value for GAE
        max_steps_per_episode (int): The maximum number of steps per episode

    Returns:
        torch.Tensor: The advantage estimates
    """
    deltas = delta(rewards, values, timesteps, gamma, max_steps_per_episode)
    A_hats = []
    cumulative_delta = 0
    for i in range(len(rewards) - 2, -1, -1):  # going backwards
        cumulative_delta = deltas[i + 1] + gamma * _lambda * cumulative_delta
        A_hats.insert(0, cumulative_delta)
        if timesteps[i + 1] < timesteps[i]:
            cumulative_delta = 0
    A_hats.insert(0, deltas[0] + gamma * _lambda * cumulative_delta)
    return torch.tensor(A_hats, device=rewards.device)


def select_action(
    policy: PolicyNetwork,
    state: Union[np.ndarray, torch.Tensor],
    device: str = None,
) -> tuple[int, torch.Tensor]:
    """
    Select a discrete action from the policy given the state. The action is selected
    using the policy's output probabilities and the state. The function returns the
    probability of the selected action and the action itself.

    Args:
        policy (torch.nn.Module): The policy network
        state (Union[np.ndarray, torch.Tensor]): The state to select the action from
        device (str): The device to use for the state tensor

    Returns:
        tuple[int, torch.Tensor]: The probability of the selected action and the action itself
    """
    if isinstance(state, np.ndarray):
        if device is None:
            raise ValueError("device must be specified if state is a numpy array")
        state = torch.tensor(state, dtype=torch.float32).to(device)

    elif isinstance(state, torch.Tensor):
        if device is not None:
            state = state.to(device)

    action, logprobs = policy.sample_action(state)
    return action.item(), torch.exp(logprobs)


def worker_process(
    env_name: str,
    max_timesteps: int,
    max_steps_per_episode: int,
    discount_factor: float,
    policy: torch.nn.Module,
    device: str,
):
    env = gym.make(env_name)
    total_timesteps = 0
    iteration_states = []
    iteration_actions = []
    iteration_rewards = []
    iteration_probs = []
    iteration_timesteps = []
    iteration_disc_rewards = []
    Trajectory = namedtuple("Trajectory", ["states", "actions", "rewards", "probs"])

    progress_bar = tqdm(total=max_timesteps, desc="Timestep", position=0)

    while total_timesteps < max_timesteps:
        state, _ = env.reset()
        trajectory = Trajectory(states=[], actions=[], rewards=[], probs=[])
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action, prob = select_action(policy, state, device)
            action_ = np.array([action])
            next_state, reward, done1, done2, _ = env.step(action_)

            done = done1 or done2

            # Store the trajectory
            trajectory.probs.append(prob)
            trajectory.states.append(state)
            trajectory.actions.append(action)
            trajectory.rewards.append(reward)

            state = next_state
            steps += 1
            total_timesteps += 1
            progress_bar.update(1)
            if total_timesteps >= max_timesteps:
                break

        # Calculate discounted rewards for the trajectory
        discounted_rewards = compute_discounted_rewards(
            trajectory.rewards, discount_factor
        )
        iteration_rewards.extend(trajectory.rewards)
        iteration_states.extend(trajectory.states)
        iteration_actions.extend(trajectory.actions)
        iteration_disc_rewards.extend(discounted_rewards)
        iteration_probs.extend(trajectory.probs)
        iteration_timesteps.extend(list(range(len(trajectory.states))))

    return (
        iteration_states,
        iteration_actions,
        iteration_rewards,
        iteration_disc_rewards,
        iteration_probs,
        iteration_timesteps,
    )


def run_iteration(
    env_name: str,
    max_timesteps: int,
    max_steps_per_episode: int,
    discount_factor: float,
    policy: torch.nn.Module,
    device: torch.device = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Run a single iteration of the policy gradient algorithm without multiprocessing.

    Args:
        env_name (str): The name of the environment to run the iteration on.
        max_timesteps (int): The maximum number of timesteps to run the iteration for.
        max_steps_per_episode (int): The maximum number of steps per episode.
        discount_factor (float): The discount factor for the rewards.
        policy (torch.nn.Module): The policy network.
        device (str): The device to use for the state tensor.

    Returns:
        Tuple containing:
        - iteration_states (torch.Tensor): The states encountered during the iteration.
        - iteration_actions (torch.Tensor): The actions taken during the iteration.
        - iteration_rewards (torch.Tensor): The rewards received during the iteration.
        - iteration_disc_rewards (torch.Tensor): The sum of discounted rewards for the iteration.
        - iteration_probs (torch.Tensor): The probabilities of the actions taken during the iteration.
        - iteration_timesteps (torch.Tensor): The timesteps at which the states were encountered.
        - iteration_log_probs (torch.Tensor): The log probabilities of the actions taken during the iteration.
    """
    (
        iteration_states,
        iteration_actions,
        iteration_rewards,
        iteration_disc_rewards,
        iteration_probs,
        iteration_timesteps,
    ) = worker_process(
        env_name,
        max_timesteps,
        max_steps_per_episode,
        discount_factor,
        policy,
        device,
    )

    # fmt: off
    iteration_states = np.array(iteration_states)
    iteration_timesteps = torch.tensor(iteration_timesteps, dtype=torch.float32, device=device)
    iteration_states = torch.tensor(iteration_states, dtype=torch.float32, device=device)
    iteration_actions = torch.tensor(iteration_actions, dtype=torch.int64, device=device)
    iteration_rewards = torch.tensor(iteration_rewards, dtype=torch.float32, device=device)
    iteration_disc_rewards = torch.tensor(iteration_disc_rewards, dtype=torch.float32, device=device)
    iteration_probs = torch.cat(iteration_probs).to(device)
    iteration_log_probs = torch.log(iteration_probs)
    # fmt: on

    return (
        iteration_states,
        iteration_actions,
        iteration_rewards,
        iteration_disc_rewards,
        iteration_probs,
        iteration_timesteps,
        iteration_log_probs,
    )
