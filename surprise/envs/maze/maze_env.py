import gym as gym
import numpy as np
from IPython import embed

class MazeEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        obs_ = env.reset()
        new_shape = obs_.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=obs_.dtype)
        self.action_space = self.env.action_space
        self.original_obs = None
        
    def observation(self, obs):
        self.original_obs = obs
        return self.original_obs