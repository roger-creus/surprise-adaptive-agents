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
        obs_shape = obs.shape
        obs_shape = (2, obs_shape[1], obs_shape[2])

        self.observation_space = {
            "obs" : Box(low=0, high=1, shape=obs_shape, dtype=np.uint8),
            "player" : Box(low=0, high=1, shape=(1, obs_shape[1], obs_shape[2]), dtype=np.uint8)
        }
        self.original_obs = None
        
    def observation(self, obs):
        player_channel = obs[1]
        self.original_obs = np.zeros((2, obs.shape[1], obs.shape[2]))
        self.original_obs[0] = obs[0]
        self.original_obs[1] = obs[4]
        return {"obs" : self.original_obs.astype(np.uint8), "player" : player_channel[None].astype(np.uint8)}

