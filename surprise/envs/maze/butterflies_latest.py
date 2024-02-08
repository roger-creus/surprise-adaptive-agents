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
        self.observation_space = Box(low=0, high=3, shape=(16, 11))
        self.original_obs = None
        
    def observation(self, obs):
        # ignore channel 2 and 3, this is the spider and cocoons channel
        self.original_obs = np.zeros((3, obs.shape[1], obs.shape[2]))
        self.original_obs[0] = obs[0]
        self.original_obs[1] = obs[1]
        self.original_obs[2] = obs[4]

        # ignore channel 2, 3, this is the spider channel and coocoons channel
        new_obs = np.zeros((1, obs.shape[1], obs.shape[2]))
        new_obs[0, obs[0,:,:]==1] = 1
        new_obs[0, obs[1,:,:]==1] = 2
        new_obs[0, obs[4,:,:]==1] = 3
        return new_obs

