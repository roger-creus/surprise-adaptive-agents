import gym as gym
import numpy as np
from IPython import embed

class MazeEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        obs_ = env.reset()
        new_shape = (1, obs_.shape[1], obs_.shape[2])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=obs_.dtype)
        self.action_space = self.env.action_space
        
    def observation(self, obs):
        # original obs is (3, 16, 14) of binary values. convert it to (1, 16, 14) of integer where 0 is wall, 1 is player, 2 is goal, 3 is goal
        new_obs = np.zeros((1, obs.shape[1], obs.shape[2]))
        new_obs[0, obs[0,:,:]==1] = 1
        new_obs[0, obs[1,:,:]==1] = 2
        new_obs[0, obs[2,:,:]==1] = 3
        return new_obs