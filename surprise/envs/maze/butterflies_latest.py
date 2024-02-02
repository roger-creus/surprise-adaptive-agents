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
        # put agent channel first
        obs = np.stack([obs[1], obs[0], obs[2], obs[3]])
        
        # get agent position
        x, y = np.unravel_index(np.argmax(obs[0], axis=None), obs[0].shape)
        
        # tranform to one channel only with indexes
        obs = np.argmax(obs, axis=0)

        # manually set goal position
        max_num = np.max(obs)
        obs[x, y] = max_num + 1
        return obs


