# import gymnasium as gym
# from gymnasium import ObservationWrapper
# from gymnasium.spaces import Box
import gym
from gym.spaces import Box
import numpy as np
from griddly import gd
from IPython import embed

class ButterfliesEnv(gym.ObservationWrapper):

    def __init__(self, env , **kwargs):
        super(ButterfliesEnv, self).__init__(env)
        self.env = env
        obs = env.reset()
        obs_shape = (3, obs_shape[1], obs_shape[2])

        self.observation_space = Box(low=0, high=3, shape=obs_shape, dtype=np.uint8)
        self.original_obs = None
        
    def observation(self, obs):
        # ignore channel 2 and 3, this is the spider and cocoons channel
        self.original_obs = np.zeros((3, obs.shape[1], obs.shape[2]))
        self.original_obs[0] = obs[0]
        self.original_obs[1] = obs[1]
        self.original_obs[2] = obs[4]
        return self.original_obs

