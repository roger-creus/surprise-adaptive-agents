import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import cv2
import util.class_util as classu
import collections 


class CrafterWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        info  = self._flat_info(info)
        return obs, reward, done, info
    
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
    def _flat_info(self, info):
        flatted_info = {}
        for k in info:
            if "dict" in str(type(info[k])):
                for in_key in info[k]:
                    flatted_info[f"{k}_{in_key}"] = info[k][in_key]
            else:
                flatted_info[k] = info[k]
        return flatted_info

    
