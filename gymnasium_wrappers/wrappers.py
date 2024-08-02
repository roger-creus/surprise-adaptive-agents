import gymnasium as gym
import numpy as np
from typing import Tuple
from IPython import embed

class ImageTranspose(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space.shape
        dtype = np.float32
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def observation(self, observation):
        observation= np.transpose(observation, axes=[2, 0, 1]) * 1.0
        return observation

