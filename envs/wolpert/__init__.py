from gym.envs.registration import register
import gym

register(
    "wolpert",
    entry_point="envs.wolpert.wolpert:WolpertEnv",
    max_episode_steps=100,
)