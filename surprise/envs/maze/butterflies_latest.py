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
        self.observation_space = Box(low=0, high=5, shape=(28, 11))

    def observation(self, obs):
        new_obs = np.zeros((1, obs.shape[1], obs.shape[2]))
        # ignore channel 3, this is the spider channel
        new_obs[0, obs[0,:,:]==1] = 1
        new_obs[0, obs[1,:,:]==1] = 2
        new_obs[0, obs[2,:,:]==1] = 3 
        new_obs[0, obs[3,:,:]==1] = 4 
        new_obs[0, obs[4,:,:]==1] = 5
        return new_obs

