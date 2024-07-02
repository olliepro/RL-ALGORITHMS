import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from gymnasium import Env
from collections import deque
import heapq
import numpy as np
import gymnasium as gym
import flappy_bird_env
import random
from tqdm import tqdm
from torch import Tensor
from torchvision.transforms import PILToTensor
import matplotlib.pyplot as plt

from torchrl.modules import NoisyLazyLinear, NoisyLinear
from typing import Iterable

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
message = "GPU Detected" if torch.cuda.is_available() else "No GPU Detected"
print(message)


class RainbowDQN(nn.Module):
    def __init__(self, n_actions: int, n_atoms: int):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_actions = n_actions

        self.conv0 = nn.LazyConv2d(3, kernel_size=3, stride=1)
        self.relu0 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()

        self.adv_fc1 = NoisyLazyLinear(256, device=device, std_init=0.0)
        self.val_fc1 = NoisyLazyLinear(128, device=device, std_init=0.0)

        self.adv_relu4 = nn.ReLU()
        self.val_relu4 = nn.ReLU()

        self.adv_fc2 = NoisyLinear(256, self.n_atoms * self.n_actions, std_init=0.5)
        self.val_fc2 = NoisyLinear(128, self.n_atoms, std_init=0.5)

        self.softmax = nn.Softmax(dim=-2)

    def reset_noise(self):
        self.adv_fc1.reset_noise()
        self.val_fc1.reset_noise()
        self.adv_fc2.reset_noise()
        self.val_fc2.reset_noise()

    def advantage(self, x: Tensor) -> Tensor:
        x = self.adv_fc1(x)
        x = self.adv_relu4(x)
        x = self.adv_fc2(x)
        x = x.reshape(-1, self.n_atoms, self.n_actions)
        return x

    def value(self, x: Tensor) -> Tensor:
        x = self.val_fc1(x)
        x = self.val_relu4(x)
        x = self.val_fc2(x)
        return x

    def adv_val(self, x: Tensor) -> Tensor:
        adv = self.advantage(x)
        val = self.value(x).unsqueeze(-1)
        p = val + adv - adv.mean(dim=1, keepdim=True)
        return p

    def forward(self, input: Tensor) -> Tensor:

        if len(input.shape) == 3:
            input = input.unsqueeze(0)

        x = self.conv0(input)
        x = self.relu0(x)

        x = self.maxpool1(x)

        x = self.conv1(input)
        x = self.relu1(x)

        x = self.maxpool3(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.maxpool4(x)

        x = self.flatten(x)
        x = self.adv_val(x)  # batch_size x n_actions x support_size

        return self.softmax(x)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        n_iterations: int,
        memory_capacity: int = 100000,
        alpha: float = 0.5,
        beta: list = [0.4, 1],
        gamma: float = 0.99,
        lookahead: int = 3,
    ):
        self.gamma = gamma
        self.lookahead = lookahead
        self.memory = deque(maxlen=memory_capacity)
        self.image_sequence = deque(maxlen=3)
        self.max_priority = 1
        self.counter = 0
        self.n_iterations = n_iterations
        if isinstance(alpha, float):
            self.alpha_start = alpha
            self.alpha_end = alpha
        else:
            self.alpha_start, self.alpha_end = alpha
        self.beta_start, self.beta_end = beta

    def linearly_annneal(self, start, end):
        return start + self.counter / self.n_iterations * (end - start)

    @property
    def alpha(self) -> float:
        if self.alpha_start == self.alpha_end:
            return self.alpha_start
        return self.linearly_annneal(self.alpha_start, self.alpha_end)

    @property
    def beta(self) -> float:
        return self.linearly_annneal(self.beta_start, self.beta_end)

    def remember(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_prime: np.ndarray,
        term: bool,
    ):
        priority = self.max_priority
        self.memory.append(
            {
                "p": priority,
                "s": s,
                "a": a,
                "r": r,
                "s+1": s_prime,
                "term": term,
            },
        )
        self.counter += 1
        if term:
            self.image_sequence.clear()

    def get_Rn_Sn(self, memory_indices, lookahead, gamma):
        Rn_list = []
        Sn_list = []
        T_list = []
        for memory_index in memory_indices:
            if self.memory[memory_index].get("Rn") is not None:
                Rn_list.append(self.memory[memory_index]["Rn"])
                Sn_list.append(self.memory[memory_index]["Sn"])
                T_list.append(self.memory[memory_index]["nterm"])
            else:
                Rn = 0
                # j ∋ [0, n-1]
                # ind ∋ [i, i+n-1]
                for j, ind in enumerate(range(memory_index, memory_index + lookahead)):
                    if self.memory[ind]["term"] == True:  # terminal state
                        break
                    Rn += self.memory[ind]["r"] * (gamma**j)

                T = self.memory[ind]["term"]
                Sn = self.memory[ind]["s+1"]
                self.memory[memory_index]["Rn"] = Rn
                self.memory[memory_index]["Sn"] = Sn
                self.memory[memory_index]["nterm"] = T
                Rn_list.append(Rn)
                Sn_list.append(Sn)
                T_list.append(T)

        return Rn_list, Sn_list, T_list

    def reset_lookahead(self, lookahead: int):
        for i in range(len(self.memory)):
            self.memory[i]["Rn"] = None

    def sample(self, batch_size: int) -> tuple[list]:
        priorities = Tensor([d["p"] for d in self.memory])[: -self.lookahead].detach()
        probs = priorities**self.alpha / torch.sum(priorities**self.alpha)

        is_weights = (probs) ** -self.beta
        is_weights /= is_weights.max()

        indices = list(range(len(self.memory) - self.lookahead))
        sample = np.random.choice(
            a=indices,
            p=np.array(probs),
            size=batch_size,
            replace=False,
        )
        s = [self.memory[i]["s"] for i in sample]
        a = [self.memory[i]["a"] for i in sample]
        Rn, Sn, nTerm = self.get_Rn_Sn(
            memory_indices=sample, lookahead=self.lookahead, gamma=self.gamma
        )

        return (s, a, Rn, Sn, nTerm, is_weights[sample], sample)

    def update_priorities(
        self, indices: Iterable[int], priorities: Iterable[float]
    ) -> None:
        for i, p in zip(indices, priorities):
            self.memory[i]["p"] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return len(self.memory)

    def phi(self, s: np.ndarray):
        if len(self.image_sequence) == 0:
            if not hasattr(self, "expected_frame_shape"):
                self.expected_frame_shape = s.shape
            for _ in range(2):
                self.image_sequence.append(np.zeros_like(s))

        if s.shape != self.expected_frame_shape:
            s = np.zeros_like(self.expected_frame_shape)
        self.image_sequence.append(s)

        phi_ = []
        for i, image in enumerate(self.image_sequence):
            from PIL import Image

            image_ = Image.fromarray(image)
            grey = image_.convert("L")

            # rescale the image to be 1/12 of the original size
            height, width = image.shape[:2]
            grey = grey.resize((int(width / 10), int(height / 10)))
            grey = grey.crop([20, 0, 56, 72])
            # if (i + 1) % 4 == 0:
            #     plt.imshow(grey)
            #     plt.savefig("RAINBOW-DQN/grey.png")
            #     raise Exception
            grey = PILToTensor()(grey).to(torch.float32) / 255
            grey = grey.squeeze(0)
            phi_.append(grey)

        phi_ = torch.stack(phi_)
        return phi_.to(device)


class DistributionalLoss:
    def __init__(
        self,
        n_atoms: int,
        v_min: float,
        v_max: float,
        gamma: float,
        lookahead: int,
    ):
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.lookahead = lookahead
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.z = torch.linspace(v_min, v_max, n_atoms).detach()

    def select_action_from_model(self, model, phi):
        preds = model(phi)
        return self.select_action_from_predictions(preds)

    def select_action_from_predictions(self, predictions):
        # broadcast the probabilities over the atoms
        if predictions.device != self.z.device:
            predictions = predictions.to(self.z.device)

        results = predictions * self.z.reshape(-1, 1)
        return results.sum(dim=-2).argmax(dim=-1)

    def __call__(
        self,
        update_model: nn.Module,
        stale_model: nn.Module,
        d: PrioritizedReplayBuffer,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        s, a, Rn, Sn, term, is_weights, indices = d.sample(batch_size=batch_size)
        s = torch.stack(s).cuda()
        Sn = torch.stack(Sn).cuda()
        Rn = torch.tensor(Rn)
        term = torch.tensor(term, dtype=torch.bool)
        P_k_x_n_x_a: Tensor = update_model(s).to("cpu")
        a = self.select_action_from_predictions(P_k_x_n_x_a)[:, None]

        row = torch.arange(P_k_x_n_x_a.shape[0])[:, None]
        col = torch.arange(P_k_x_n_x_a.shape[1])
        P_k_x_n = P_k_x_n_x_a[row, col, a]

        with torch.no_grad():
            Pn_k_x_a_x_n = stale_model(Sn).detach().to("cpu")
            a_n = self.select_action_from_model(update_model, Sn).to("cpu")[:, None]
            Pn_k_x_n = Pn_k_x_a_x_n[row, col, a_n]
            Tz = Rn.unsqueeze(1) + (1 - term.int()).unsqueeze(1) * self.z.unsqueeze(
                0
            ) * self.gamma ** (self.lookahead)
            Tz = Tz.clamp(self.v_min, self.v_max)

            b = Tz + torch.arange(0, batch_size).unsqueeze(1) * (
                self.v_max - self.v_min + self.delta_z
            )
            # batch_size * n_atoms
            b = (b - self.v_min) / self.delta_z
            b = b.reshape(-1)
            target = torch.zeros_like(Pn_k_x_n).reshape(-1)
            Pn_kn = Pn_k_x_n.view(-1)
            for i in range(len(target)):
                l = torch.floor(b[i]).int()
                u = torch.ceil(b[i]).int()
                if l != u:
                    target[l] += Pn_kn[i] * (u - b[i])
                    target[u] += Pn_kn[i] * (b[i] - l)
                else:
                    target[l] += Pn_kn[i]

            target = target.reshape(Pn_k_x_n.shape)
        # P_k_x_n[P_k_x_n <= 1e-8] = 1e-8
        P_k_x_n = torch.log(P_k_x_n)
        target = target.to(device)
        P_k_x_n = P_k_x_n.to(device)
        loss_fn = nn.KLDivLoss(reduction="none")
        losses = loss_fn(P_k_x_n, target).sum(dim=-1)
        if torch.isnan(losses).any():
            print("NAN Loss")
            import pdb

            s_ = s
            pdb.set_trace()

        # for i, r in enumerate(Rn):
        #     if r > 1:
        #         import matplotlib.pyplot as plt

        #         plt.subplot(1, 2, 1)
        #         plt.imshow(s[i][-1].cpu().numpy())
        #         plt.subplot(1, 2, 2)
        #         plt.plot(self.z, target[i].detach().cpu().numpy(), label="target")
        #         plt.plot(
        #             self.z, torch.exp(P_k_x_n[i]).detach().cpu().numpy(), label="pred"
        #         )
        #         plt.title(f"Reward: {r}")
        #         plt.legend()
        #         plt.savefig(f"RAINBOW-DQN/large_r.png")
        #         plt.close()

        if random.random() < 0.0001:
            for i in range(len(target)):
                plt.figure(figsize=(10, 5))

                # Image subplot
                plt.subplot(1, 2, 1)
                plt.imshow(s[i][-1].cpu().numpy())

                # Histogram subplot for all actions
                plt.subplot(1, 2, 2)
                for action, color in zip(
                    range(P_k_x_n_x_a.shape[2]), ["blue", "orange", "green", "red"]
                ):  # Iterate through all actions
                    action_data = P_k_x_n_x_a[i, :, action].detach().cpu().numpy()
                    mean_action = (self.z * action_data).sum()
                    plt.bar(
                        self.z,
                        action_data,
                        width=self.z[1] - self.z[0],
                        alpha=0.3,
                        label=f"Action {action}, Mean: {mean_action:.2f}",
                        color=color,
                    )

                    if action == a[i]:
                        target_data = target[i].detach().cpu().numpy()
                        plt.bar(
                            self.z,
                            target_data,
                            width=self.z[1] - self.z[0],
                            alpha=0.3,
                            color="grey",
                            label="Target",
                        )
                    plt.axvline(
                        x=mean_action,
                        linestyle="dashed",
                        linewidth=1,
                        label=f"Action {action} Mean Pred",
                        color=color,
                    )

                plt.legend()
                plt.title(f"Batch sample {i} - Selected Action {a[i].item()}")
                plt.savefig(f"RAINBOW-DQN/plot_{i}.png")
                plt.close()
        #     print(losses)
        # clear cache
        # torch.cuda.empty_cache()

        loss = torch.mean(losses * is_weights.detach().to(device))
        d.update_priorities(indices, losses.cpu().detach().abs())
        return loss


def exploit(model, env: Env, d: PrioritizedReplayBuffer, loss_fn: DistributionalLoss):
    s, _ = env.reset()
    phi_ = d.phi(s)
    term = False
    R = 0
    while True:
        a = loss_fn.select_action_from_model(model, phi_).item()
        s, r, term, _, _ = env.step(a)
        for _ in range(2):
            s, r_, term, illegal, _ = env.step(0)
            r += r_
            if term or illegal:
                r = 0
                break
        phi_prime = d.phi(s)
        d.remember(phi_.cpu(), a, r, phi_prime.cpu(), term)
        phi_ = phi_prime
        R += r
        if term or illegal:
            break
    return R


def train_dqn(
    max_steps: int,
    gamma=0.99,
    n_atoms=31,
    lookahead=3,
    v_min=0,
    v_max=3,
    alpha=0.5,
    beta=[0.4, 1],
    memory_capacity=40000,
    min_exploration=30000,
    model_path: str = None,
    learning_rate=0.0000625,
    save_period=10000,
):
    progress_bar = tqdm(total=max_steps)
    env = gym.make("FlappyBird-v0", render_mode="human")
    KL_loss = DistributionalLoss(
        n_atoms=n_atoms,
        v_min=v_min,
        v_max=v_max,
        gamma=gamma,
        lookahead=lookahead,
    )
    d = PrioritizedReplayBuffer(
        n_iterations=max_steps,
        memory_capacity=memory_capacity,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lookahead=lookahead,
    )
    update_model = RainbowDQN(n_actions=env.action_space.n, n_atoms=n_atoms).to(device)
    stale_model = RainbowDQN(n_actions=env.action_space.n, n_atoms=n_atoms).to(device)

    stale_model.eval()
    steps_taken = 0
    exploration = min_exploration

    stale_model.load_state_dict(update_model.state_dict())
    optimizer = optim.Adam(update_model.parameters(), lr=learning_rate, eps=1.5e-4)

    losses = []
    rewards = []
    while steps_taken < max_steps:
        s, _ = env.reset()
        R = 0
        phi = d.phi(s)

        if model_path:
            update_model(phi)
            stale_model(phi)
            print(f"Loading model from {model_path}")
            update_model.load_state_dict(torch.load(model_path))
            stale_model.load_state_dict(torch.load(model_path))
            start = model_path.rfind("_") + 1
            end = model_path.rfind(".pth")
            steps_taken = int(model_path[start:end])
            exploration += steps_taken
            progress_bar.update(steps_taken)
            print(f"Loaded model trained for {steps_taken} steps")
            model_path = ""
            losses = open("RAINBOW-DQN/losses.txt").read().split("\n")
            losses = [float(loss) for loss in losses if loss]
            rewards = open("RAINBOW-DQN/rewards.txt").read().split("\n")
            rewards = [float(reward) for reward in rewards if reward]
        elif model_path is None:
            with open("RAINBOW-DQN/losses.txt", "w") as f:
                pass
            with open("RAINBOW-DQN/rewards.txt", "w") as f:
                pass
            model_path = ""

        term = False

        for _ in range(1000):
            if steps_taken < exploration:
                if np.random.rand() < 0.3:
                    a = 1
                else:
                    a = 0
            else:
                a = KL_loss.select_action_from_model(update_model, phi).item()

            s_prime, r, term, _, _ = env.step(a)
            for _ in range(2):
                s_prime, r_, term, illegal, _ = env.step(0)
                r += r_
                if term or illegal:
                    r = 0
                    break
            R += r
            phi_prime = d.phi(s_prime)
            d.remember(phi.cpu(), a, r, phi_prime.cpu(), term)
            phi = phi_prime
            steps_taken += 1
            progress_bar.update(1)
            # if steps_taken > exploration:
            if len(d) >= 10000 and steps_taken % 3 == 0:
                optimizer.zero_grad()
                loss = KL_loss(update_model, stale_model, d, batch_size=128)
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().item())
                avg_loss = np.mean(losses[-200:])
                progress_bar.set_description(
                    f"Loss: {avg_loss}, Reward: {np.mean(rewards[-200:]) if len(rewards) > 0 else 0}"
                )
                # append loss and rewards
                with open("RAINBOW-DQN/losses.txt", "a") as f:
                    f.write(f"{loss.item()}\n")

            if (
                steps_taken in np.arange(0, max_steps, save_period)
                and steps_taken > min_exploration
            ):
                exploits = [exploit(update_model, env, d, KL_loss) for _ in range(20)]
                print("Average Exploitation: ", np.mean(exploits))
                print("Max Exploitation: ", max(exploits))

                stale_model.load_state_dict(update_model.state_dict())

                # checkpoint
                save_path = f"RAINBOW-DQN/FlappyBird_{steps_taken}.pth"
                torch.save(update_model.state_dict(), save_path)
                print(f"Saved model at {save_path}")

                import matplotlib.pyplot as plt

                # plot loss and rewards
                plt.figure(figsize=(12, 5), dpi=130)
                plt.subplot(1, 2, 1)

                # plot rolling average rewards
                rolling_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
                plt.plot(rolling_avg)
                plt.title("Rewards")

                plt.subplot(1, 2, 2)
                plt.plot(losses)
                plt.semilogy()
                plt.title("Loss")

                plt.savefig(f"RAINBOW-DQN/plot_{steps_taken}.png")

                # exploit 20 times
                stale_model.reset_noise()
                update_model.reset_noise()
                # lookahead = np.random.randint(1, 8)
                # d.reset_lookahead(lookahead)
                # KL_loss.lookahead = lookahead

                break

            if term or illegal:
                rewards.append(R)
                with open("RAINBOW-DQN/rewards.txt", "a") as f:
                    f.write(f"{R}\n")
                break


if __name__ == "__main__":
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_dqn(
        3000000,
        # model_path="/home/oliver/Code/RL-ALGORITHMS/RAINBOW-DQN/FlappyBird_448000.pth",
    )
