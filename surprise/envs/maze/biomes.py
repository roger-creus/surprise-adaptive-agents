import gym
import numpy as np
import cv2

from gym import ObservationWrapper
from gym.spaces import Box
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
    
class BiomesFullyObservedPixel(ObservationWrapper):

    def __init__(self, **kwargs):
        env = gym.make("GDY-_BiomesFullyObservedPixel-v0",
                       level=0,
                       player_observer_type=gd.ObserverType.SPRITE_2D,
                       global_observer_type=gd.ObserverType.SPRITE_2D,
                       **kwargs)
        super(BiomesFullyObservedPixel, self).__init__(env)

        self.observation_space = Box(0, 1, shape=(64, 64))
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, rew, done, {}

    def observation(self, obs):
        obs = obs.transpose(1,2,0)
        obs = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        obs = obs.astype(float) / 255.0
        return obs
