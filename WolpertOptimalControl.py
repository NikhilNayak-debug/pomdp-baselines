import gym
from gym import spaces
import numpy as np
import torch
from numpy import linalg as LA


def step(state):
    
    A = torch.from_numpy(np.array([
    [0.0, 1.0, 0.0, 0.0],  # x
    [0.0, 0.0, 0.0, 0.0],  # x_dot
    [0.0, 0.0, 0.0, 1.0],  # y
    [0.0, 0.0, 0.0, 0.0],  # y_dot
    ]))
    
    B = torch.from_numpy(np.array([
    [0., 0.],
    [1., 0.],
    [0., 0.],
    [0., 1.],
    ]))

    C = torch.from_numpy(np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.]
    ]))

    # Simulate one step in the environment
    control = torch.tensor([[ 1.00000000e+00,  1.73205081e+00, -6.29500991e-17,
     -2.28485717e-16],
    [-4.63895536e-17, -2.28485717e-16,  1.00000000e+00,
      1.73205081e+00]])
    
    # print('eigen values', LA.eig( A.T.float() - control.T.float() @ B.T.float()))

    dx = state.float() @ A.T.float() - state.float() @ control.T.float() @ B.T.float()
    

    state += dx.numpy() * 0.05

    reward = -torch.mean(torch.square(torch.tensor([state[0][0], state[0][2]]))).item()
    thresh = -0.04
    print('reward', reward)
    print('state', [state[0][0], state[0][2]])
    # if reward.item() > thresh:
    if reward > thresh:
        return state, 1e4, True, {}
    elif reward < -100:
        return state, 10*reward, True, {}
    else:
        return state, reward, False, {}


if __name__ == "__main__":
    
    initial_angle = np.random.uniform(0, 2 * np.pi)

    state = torch.tensor([
        [np.cos(initial_angle), 0.0, np.sin(initial_angle), 0.0]
        for index in range(1)
    ])
    
    print('initial state', [state[0][0], state[0][2]])
    
    # state = torch.tensor([
    #     [2.0, 0.0, 1.0, 0.0]
    #     for index in range(1)
    # ])

    for _ in range(1000):
        state, reward, done, _ = step(state)
        if done:
            print('Done, reward is:', reward)
            break