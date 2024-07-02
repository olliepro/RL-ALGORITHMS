import imageio
import torch
import torch.nn as nn
import gymnasium as gym


def play_and_render_with_temperature(
    policy_net: nn.Module,
    env: gym.Env,
    max_steps_per_episode: int,
    gif_path: str,
    temperature: float = 1.0,
) -> None:
    """
    Play an episode using the learned policy with temperature-based randomness and render the game at each step to produce a GIF.

    Args:
    - policy_net (nn.Module): The trained policy network.
    - env (gym.Env): The environment.
    - max_steps_per_episode (int): Maximum number of steps per episode.
    - gif_path (str): Path to save the generated GIF.
    - temperature (float): The temperature for controlling randomness.

    Returns:
    - None
    """
    state, _ = env.reset()
    done = False
    steps = 0
    min_steps = 300
    device = next(policy_net.parameters()).device

    while steps < min_steps:
        frames = []
        steps = 0
        while not done and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = policy_net(state_tensor)

            # Convert probabilities to logits
            logits = torch.log(probs)

            # Apply temperature to logits
            logits = logits / temperature

            # Convert logits back to probabilities
            scaled_probs = torch.softmax(logits, dim=-1)

            action = torch.multinomial(scaled_probs, num_samples=1).item()

            next_state, _, done1, done2, _ = env.step(action)

            done = done1 or done2
            # Render the game and capture the frame
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            state = next_state
            steps += 1
        print("Episode finished after {} timesteps".format(steps))
        done = False
        state = env.reset()

    env.close()

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames, fps=30)
