import gym
from gym import spaces
import numpy as np
import torch


class WolpertEnv(gym.Env):
    def __init__(self, process_noise=0.05, obs_noise=0.005, obs_threshhold=0.05, visible_velocity=False):
        super().__init__()

        self.world_size = 4
        self.control_size = 2
        self.obs_size = 5

        self.A = torch.from_numpy(np.array([
            [0.0, 1.0, 0.0, 0.0],  # x
            [0.0, 0.0, 0.0, 0.0],  # x_dot
            [0.0, 0.0, 0.0, 1.0],  # y
            [0.0, 0.0, 0.0, 0.0],  # y_dot
        ]))

        self.B = torch.from_numpy(np.array([
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 1.],
        ]))

        self.C = torch.from_numpy(np.array([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.]
        ]))

        self.process_noise = torch.sqrt(
            torch.from_numpy(np.eye(self.world_size) * process_noise))
        self.obs_noise = torch.sqrt(
            torch.from_numpy(np.eye(self.obs_size - 1) * obs_noise))
        self.obs_threshhold = obs_threshhold
        self.visible_velocity = visible_velocity

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.control_size,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.obs_size,), dtype=np.float32)

        self.state = torch.tensor([
            [np.cos(index * 2 * np.pi / 8), 0.0, np.sin(index * 2 * np.pi / 8), 0.0]
            for index in range(1)
        ])

        self._max_episode_steps = 100

    def reset(self):
        # Initialize the environment
        self.state = torch.tensor([
            [np.cos(index * 2 * np.pi / 8), 0.0, np.sin(index * 2 * np.pi / 8), 0.0]
            for index in range(1)
        ])
        return self.obs(self.state)

    def step(self, action):
        # Simulate one step in the environment
        control = torch.Tensor(action) # .unsqueeze(0) - check the dimensions here
        dx = self.state.float() @ self.A.T.float() + control.float() @ self.B.T.float() # + torch.randn(self.world_size).float() @ self.process_noise.float()
        self.state += dx.numpy() * 0.05

        x_norm = np.linalg.norm(self.state[0][[0, 2]])
        if x_norm > 1.0:
            self.state[0][[0, 2]] /= x_norm

        y = self.obs(self.state)
        reward = -torch.mean(torch.square(torch.tensor([self.state[0][0], self.state[0][2]]))).item()
        thresh = -0.04
        print('reward', reward)
        print('state', [self.state[0][0], self.state[0][2]])
        # if reward.item() > thresh:
        if reward > thresh:
            return y, reward, True, {}
        else:
            return y, reward, False, {}

    def obs(self, x, visible=True):
        full_obs = x + torch.randn_like(x) @ self.obs_noise
        full_obs = torch.cat([torch.ones(x.shape[:-1] + (1,)), full_obs], -1)
        if visible:
            return np.array(torch.flatten(full_obs))

        if self.visible_velocity:
            no_obs = torch.cat(
                [
                    torch.zeros(full_obs.shape[:-1] + (2,)),
                    full_obs[..., 2:3],
                    torch.zeros(full_obs.shape[:-1] + (1,)),
                    full_obs[..., 4:]], axis=-1
                            )
        else:
            no_obs = torch.zeros_like(full_obs)

        y = torch.where(self.obs_threshhold - x[..., 1:2] > 0, full_obs, no_obs).detach()
        return np.array(y)

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    import envs

    env = WolpertEnv()
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            break
    env.close()