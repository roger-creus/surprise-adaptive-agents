import gym

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
from griddly import gd
from IPython import embed

class BiomesFullyObservedVector(ObservationWrapper):

    def __init__(self, **kwargs):
        env = gym.make("GDY-_BiomesFullyObservedVector-v0",
                       level=0,
                       player_observer_type=gd.ObserverType.VECTOR,
                       global_observer_type=gd.ObserverType.VECTOR,
                       **kwargs)
        super(BiomesFullyObservedVector, self).__init__(env)

        self.observation_space = Box(0, 11, shape=(39, 37))
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, rew, done, {}

    def observation(self, obs):
        # put agent channel first
        obs = np.vstack([obs[5][None,:], obs[0:5], obs[6:]])
        
        # get agent position
        x, y = np.unravel_index(np.argmax(obs[0], axis=None), obs[0].shape)
        
        # tranform to one channel only with indexes
        obs = np.argmax(obs, axis=0)

        # manually set goal position
        max_num = np.max(obs)
        obs[x, y] = max_num + 1
        return obs

class BiomesPartiallyObservedVector(ObservationWrapper):

    def __init__(self, **kwargs):
        env = gym.make("GDY-_BiomesPartiallyObservedVector-v0",
                       level=0,
                       player_observer_type=gd.ObserverType.VECTOR,
                       global_observer_type=gd.ObserverType.VECTOR,
                       **kwargs)
        super(BiomesPartiallyObservedVector, self).__init__(env)

        self.observation_space = Box(0, 11, shape=(7, 7))
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, rew, done, {}

    def observation(self, obs):
        # put agent channel first
        obs = np.vstack([obs[5][None,:], obs[0:5], obs[6:]])
        
        # get agent position
        x, y = np.unravel_index(np.argmax(obs[0], axis=None), obs[0].shape)
        
        # tranform to one channel only with indexes
        obs = np.argmax(obs, axis=0)

        # manually set goal position
        max_num = np.max(obs)
        obs[x, y] = max_num + 1
        return obs
