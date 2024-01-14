import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import cv2
import util.class_util as classu
import pandas as pd
from collections import defaultdict


class CrafterWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, env, save_metrics=True, save_metrics_path=None):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # track achievements over all episodes
        self.achievements = "none"
        self.save_metrics_path = save_metrics_path
        print(f"Crafter save metrics path: {self.save_metrics_path}")
        self.episode_count = 0
        self.save_freq = 2
        self.t = 0
        self.metrics_list = []
        self.crafter_scores_moving_average = np.nan
    
    def reset(self):
        self.t = 0
        return self._env.reset()
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        info["discount"] = self.discount_rate
        self.t += 1
        
        # compute crafter metrics
        self.update_achievements(info["achievements"])
        success_rates = self.compute_success_rates()
        crafter_score = self.compute_crafter_score()
        if done:
            metrics_dict = {}
            self.update_achievements(info["achievements"])
            success_rates = self.compute_success_rates()
            crafter_score = self.compute_crafter_score()
            metrics_dict["episode"] = self.episode_count
            metrics_dict["crafter_score"] = crafter_score
            for k in success_rates:
                metrics_dict[f"{k}_success_rate"] = success_rates[k]
            self.metrics_list.append(metrics_dict)
            if self.episode_count % self.save_freq == 0:
                df = pd.DataFrame.from_dict(self.metrics_list)
                df.to_csv(f"{self.save_metrics_path}/crafter_metrics_{self.episode_count}.csv")
            self.episode_count += 1
            self.update_crafter_score(crafter_score)

        info["crafter_scores_moving_average "] = self.crafter_scores_moving_average
        info  = self._flat_info(info)
        # add crafter metrics to the info dict
            
        
        return obs, reward, done, info
    
    def update_crafter_score(self, new_score):
        if np.isnan(self.crafter_scores_moving_average):
            self.crafter_scores_moving_average = 0
        else:
            self.crafter_scores_moving_average = self.crafter_scores_moving_average + (1/self.episode_count) * (new_score - self.crafter_scores_moving_average)
            
    
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
    
    def set_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate

    
