import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import cv2
import util.class_util as classu
from collections import defaultdict


class CrafterWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, env, save_metrics=True):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # track achievements over all episodes
        self.achievements = "none"
    
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # compute crafter metrics
        if done:
            self.update_achievements(info["achievements"])
            success_rates = self.compute_success_rates()
            crafter_score = self.compute_crafter_score()

        info  = self._flat_info(info)
        # add crafter metrics to the info dict
        if done:
            for k in success_rates:
                info[f"{k}_success_rate"] = success_rates[k]
            info["crafter_score"] = crafter_score
        
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
    
    def update_achievements(self, achievements):
        self.achievements = {} if self.achievements == "none" else self.achievements
        for k,v in achievements.items():
            if not k in self.achievements:
                self.achievements[k] = []
            self.achievements[k].append(v)

    def compute_success_rates(self):
        success_rate = {}
        for k,v in self.achievements.items():
            num_episodes = len(self.achievements[k])
            success_rate[k] = 100 * (np.array(v)>=1).mean() 
        return success_rate
    
    def compute_crafter_score(self):
        success_rates_values = np.array(list(self.compute_success_rates().values()))
        assert (0 <= success_rates_values).all() and (success_rates_values <= 100).all()
        scores = np.exp(np.nanmean(np.log(1 + success_rates_values), -1)) - 1
        return scores

    
