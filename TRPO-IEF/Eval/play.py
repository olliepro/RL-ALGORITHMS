import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from CartPole.policy_net import PolicyNetwork


def run_policy(
    policy_net: PolicyNetwork,
    env_name: str,
    num_episodes: int,
    max_steps_per_episode: int,
    temperature: float,
    device: str,
) -> float:
    env = gym.make(env_name)
    reward_sums = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward_sum = 0

        while not done and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_probs = policy_net.sample_action(state_tensor, temperature)

            next_state, reward, done1, done2, _ = env.step(action.cpu().numpy())

            done = done1 or done2
            episode_reward_sum += reward
            state = next_state
            steps += 1

        reward_sums.append(episode_reward_sum)

    return np.mean(reward_sums)


def run_discrete_policy(
    policy_net: nn.Module,
    env_name: str,
    num_episodes: int,
    max_steps_per_episode: int,
    temperature: float,
) -> float:
    """
    Run the policy using a single process and return the average reward per episode.

    Args:
    - policy_net (nn.Module): The trained policy network.
    - env_name (str): The name of the environment.
    - num_episodes (int): Number of episodes to run. Default is 5.
    - max_steps_per_episode (int): Maximum number of steps per episode.
    - temperature (float): The temperature for the softmax. Default is 0.

    Returns:
    - float: The average reward per episode.
    """
    return run_policy(
        policy_net,
        env_name,
        num_episodes,
        max_steps_per_episode,
        temperature,
        next(policy_net.parameters()).device,
    )
