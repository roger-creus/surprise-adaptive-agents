import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
import time
from gymnasium.wrappers.frame_stack import FrameStack

class ObsHistoryWrapper(gym.Wrapper):
    def __init__(self, 
                env, 
                history_length=3, 
                stack_channels=False, 
                channel_dim=2
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        self._env = env
        self._channel_dim = channel_dim
        self._history_length = history_length
        # Gym spaces
        self.action_space = self._env.action_space
        self.observation_space_old = env.observation_space
        shape_ = list(env.observation_space.low.shape)
        self.gray_scale = True if shape_[self._channel_dim] == 1 else False
        shape_[self._channel_dim] = shape_[self._channel_dim] * self._history_length 
        self.observation_space = Box(0, 1, shape=shape_ )    
        self.obs_stack = np.zeros(shape_)
        print(f"observation space in obs history wrapper:{self.observation_space}")

    def step(self, action):
        # Take Action
        now = time.time()
        obs, env_rew, envdone, envtrunc ,info = self._env.step(action)
        # update the image stack
        if self.gray_scale:
            self.obs_stack[:, :, :-1] = self.obs_stack[:, :, 1:]
            self.obs_stack[:, :, -1] = obs.squeeze()
        else:
            self.obs_stack[:, :, :-3] = self.obs_stack[:, :, 3:]
            self.obs_stack[:, :, -3:] = obs

        self._time += 1
        return self.obs_stack, env_rew, envdone, envtrunc ,info 
    
    def reset(self, seed=None, options=None):
        '''
        Reset the wrapped env and the buffer
        '''
        self._time = 0
        obs, info = self._env.reset()
        if self.gray_scale:
            self.obs_stack[:, :, :-1] = self.obs_stack[:, :, 1:]
            self.obs_stack[:, :, -1] = obs.squeeze()
        else:
            self.obs_stack[:, :, :-3] = self.obs_stack[:, :, 3:]
            self.obs_stack[:, :, -3:] = obs
        obs = self.obs_stack
        return obs, info
    
    def get_obs(self, obs):
        
        if self._stack_channels:
            obs_ =  np.concatenate(self.obs_hist, axis=-1)
        else:
            obs_ =  np.array(self.obs_hist).flatten()            
        return obs_
        
    def render(self,**kwargs ):
        return self._env.render(**kwargs)