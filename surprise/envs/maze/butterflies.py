import gym

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
from griddly import gd
from IPython import embed

class ButterfliesEnv(ObservationWrapper):

    def __init__(self, **kwargs):
        env = gym.make("GDY-_ButterfliesEnv-v0",
                       level=0,
                       player_observer_type=gd.ObserverType.VECTOR,
                       global_observer_type=gd.ObserverType.VECTOR,
                       **kwargs)
        super(ButterfliesEnv, self).__init__(env)

        self.observation_space = Box(0, 4, shape=(28, 11))
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        obs = self.ToOneMask(obs)
        return obs, rew, done, info

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


