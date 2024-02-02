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
        self.observation_space = Box(low=0, high=4, shape=(28, 11))
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, rew, done, {}

    def observation(self, obs):
        new_obs = np.zeros(1, obs.shape[1], obs.shape[2])
        new_obs[0, obs[0,:,:]==1] = 1
        new_obs[0, obs[1,:,:]==1] = 2
        new_obs[0, obs[2,:,:]==1] = 3
        new_obs[0, obs[3,:,:]==1] = 4
        print("shape of new observation")
        print(new_obs.shape)
        quit()
        return new_obs

