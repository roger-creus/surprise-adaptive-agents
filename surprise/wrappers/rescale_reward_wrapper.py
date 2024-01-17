import numpy as np
import gym
import util.class_util as classu
from gym.wrappers.normalize import RunningMeanStd


class ReScaleRewardWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # track achievements over all episodes
        self.rms = RunningMeanStd()
    
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        print(f"reward before scaling:{reward}")
        # update the running reward std
        self.rms.update(np.array([reward]))
        # rescale the reward by std
        reward_std = np.sqrt(self.rms.var)
        reward /= reward_std
        # For debugging 
        print(f"reward_std: {reward_std}")
        print(f"rescaled_reward: {reward}")
        info["reward_std"] = reward_std
        info["rescaled_reward"] = reward
        return obs, reward, done, info
            
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
    def set_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate

    
