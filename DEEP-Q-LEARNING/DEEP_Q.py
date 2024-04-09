import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from gymnasium import Env
from collections import deque
import numpy as np
import gymnasium as gym
import flappy_bird_env
import random
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, env: Env, phi: callable):
        super().__init__()
        s_0, _ = env.reset()
        phi_0 = phi(s_0)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)
        width = phi_0.shape[-1] // 2
        height = phi_0.shape[-2] // 2
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        width -= 2
        height -= 2
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        width = width // 2 - 2
        height = height // 2 - 2
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(width * height * 128, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, env.action_space.n)

    def forward(self, x_):
        if not torch.any(x_):
            raise ValueError("Input is empty")
        if len(x_.shape) == 3:
            x_ = x_.unsqueeze(0)
        x = self.conv1(x_)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x


class D:
    def __init__(self, memory_capacity):
        self.memory = deque(maxlen=memory_capacity)
        self.image_sequence = deque(maxlen=4)

    def remember(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_prime: np.ndarray,
        term: bool,
    ):
        self.memory.append((s, a, r, s_prime, term))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)

        return [
            [s for s, _, _, _, _ in sample],
            [a for _, a, _, _, _ in sample],
            [r for _, _, r, _, _ in sample],
            [s_prime for _, _, _, s_prime, _ in sample],
            [term for _, _, _, _, term in sample],
        ]

    def __len__(self):
        return len(self.memory)

    def phi(self, s: np.ndarray):
        if len(self.image_sequence) == 0:
            if not hasattr(self, "expected_frame_shape"):
                self.expected_frame_shape = s.shape
            for _ in range(3):
                self.image_sequence.append(np.zeros_like(s))

        if s.shape != self.expected_frame_shape:
            s = np.zeros_like(self.expected_frame_shape)
        self.image_sequence.append(s)

        phi_ = []
        for image in self.image_sequence:
            grey = np.mean(image, axis=2)
            grey = grey[::2, ::2]
            grey = grey / 255
            grey = torch.tensor(grey, dtype=torch.float32)
            phi_.append(grey)
        phi_ = torch.stack(phi_)
        return phi_


def select_action(model, env: Env, s, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return model(s).argmax().item()


def L(
    update_model: nn.Module,
    stale_model: nn.Module,
    batch,
    gamma: float,
) -> torch.Tensor:
    s, a, r, s_prime, term = batch
    s = torch.stack(s)
    s_prime = torch.stack(s_prime)
    q = update_model(s)  # len(s) x n_actions
    q_prime = stale_model(s_prime)
    q_target = q.clone()
    for i in range(len(batch)):
        q_target[i, a[i]] = r[i] if term[i] else r[i] + gamma * q_prime[i].max()

    loss = F.mse_loss(
        q[torch.arange(0, len(a)), a],
        q_target[torch.arange(0, len(a)), a],
    )
    return loss


def train_dqn(max_steps: int, gamma=0.99):
    env = gym.make("FlappyBird-v0", render_mode="human")
    d = D(10000)
    update_model = DQN(env, d.phi)
    stale_model = DQN(env, d.phi)
    stale_model.load_state_dict(update_model.state_dict())
    optimizer = optim.Adam(update_model.parameters(), lr=0.0005)
    steps_taken = 0
    losses = []
    progress_bar = tqdm(total=max_steps)
    while steps_taken < max_steps:
        d.image_sequence.clear()
        s, _ = env.reset()
        phi = d.phi(s)
        term = False

        while not term:
            epsilon = 0.5 - (steps_taken / max_steps) * 0.49
            a = select_action(update_model, env, phi, epsilon)
            for _ in range(3):
                env.step(0)
            s_prime, r, term, _, _ = env.step(a)
            phi_prime = d.phi(s_prime)
            d.remember(phi, a, r, phi_prime, term)
            phi = phi_prime
            steps_taken += 1
            progress_bar.update(1)

            if steps_taken % 64 == 0:
                batch = d.sample(64)
                optimizer.zero_grad()
                loss: torch.Tensor = L(update_model, stale_model, batch, gamma)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                progress_bar.set_description(f"Loss: {losses[-1]}")

            if steps_taken in [1000, 5000, 15000, 30000, 60000, 100000]:
                stale_model.load_state_dict(update_model.state_dict())
                # checkpoint
                save_path = f"DEEP-Q-LEARNING/FlappyBird_{steps_taken}.pth"
                torch.save(update_model.state_dict(), save_path)
                print(f"Saved model at {save_path}")


if __name__ == "__main__":

    train_dqn(100000)
