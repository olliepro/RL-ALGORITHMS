import numpy as np
import gymnasium as gym
from tqdm import tqdm


def rand_policy(epsilon):
    if np.random.rand() < epsilon:
        return True
    else:
        return False


def train_q_grid(env, n_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in tqdm(list(range(n_episodes))):
        s_0, _ = env.reset()
        term = False
        while not term:
            if rand_policy(epsilon):
                a_0 = env.action_space.sample()
            else:
                a_0 = np.argmax(q[s_0])
            s_1, r, term, _, _ = env.step(a_0)

            if term:
                q[s_0, a_0] += alpha * (r - q[s_0, a_0])
            else:
                q[s_0, a_0] += alpha * (r + gamma * np.max(q[s_1, :]) - q[s_0, a_0])
            s_0 = s_1
    return q
