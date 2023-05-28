import gym

from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np
from griddly import gd


class ButterfliesEnv(ObservationWrapper):

    def __init__(self, **kwargs):
        env = gym.make("GDY-_ButterfliesEnv-v0",
                       level=0,
                       player_observer_type=gd.ObserverType.VECTOR,
                       global_observer_type=gd.ObserverType.VECTOR,
                       **kwargs)
        super(ButterfliesEnv, self).__init__(env)

        self.observation_space = Box(0, 3, shape=(28, 11))

    def step(self, action):
        state, rew, done, info = super().step(action)
        return state, rew, done, {}

    def observation(self, obs):
        return np.argmax(obs, axis=0)


