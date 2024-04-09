import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Q import train_q_grid


def play_game(env, policy, render=False, max_steps=1000, **kwargs):
    n = 0
    while n <= max_steps:
        obs, _ = env.reset()
        done = False

        while not done:
            n += 1
            action = policy(env, obs, **kwargs)
            obs, reward, done, truc, info = env.step(action)
            if render:
                yield env.render()
            if n > max_steps:
                break


def animate_mp4(frame_generator, save_path="Q-LEARNING/taxi.gif", show=False):
    fig, ax = plt.subplots()
    im = ax.imshow(next(frame_generator), animated=True)

    def update(frame):
        im.set_array(frame)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frame_generator, interval=100)
    if show:
        plt.show()
    else:
        ani.save(save_path, writer="ffmpeg")


def random_policy(env, obs):
    return env.action_space.sample()


def q_policy(env, obs, q_grid):
    return q_grid[obs].argmax()


env = gym.make("Taxi-v3", render_mode="rgb_array")
q_grid = train_q_grid(env, n_episodes=10000, alpha=1, gamma=0.99, epsilon=0.5)
animate_mp4(play_game(env, q_policy, q_grid=q_grid, render=True, max_steps=100))
env.close()
