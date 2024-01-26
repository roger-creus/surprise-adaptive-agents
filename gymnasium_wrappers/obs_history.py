import gymnasium as gym
import numpy as np
import collections
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
        self._env = FrameStack(env, history_length)
        # Gym spaces
        self.action_space = self._env.action_space
        self.observation_space_old = env.observation_space
        if self._stack_channels:
            shape_ = list(env.observation_space.low.shape)
            shape_[self._channel_dim] = shape_[self._channel_dim] * self._history_length 
            self.observation_space = Box(0, 1, shape=shape_ )    
        else:
            self.observation_space = Box(-1, 1, shape=(env.observation_space.low.shape[0]*self._history_length,) )

        print(f"observation space in obs history wrapper:{self.observation_space}")

    def step(self, action):
        # Take Action
        now = time.time()
        obs, env_rew, envdone, envtrunc ,info = self._env.step(action)
        obs = (np.array(obs).transpose(1,2,0,3)).reshape(obs.shape[1],  obs.shape[2], -1)
        print(obs.shape)
        self._time += 1
        print(f"step in observation history: {time.time() - now}")
        return self.get_obs(obs), env_rew, envdone, envtrunc ,info 
    
    def reset(self, seed=None, options=None):
        '''
        Reset the wrapped env and the buffer
        '''
        self._time = 0
        obs, info = self._env.reset()
        self.obs_hist = collections.deque([np.zeros(shape=self.observation_space_old.low.shape) for _ in range(self._history_length)])
        self.obs_hist.appendleft(obs)
        self.obs_hist.pop()
        return self.get_obs(obs), info
    
    def get_obs(self, obs):
        
        if self._stack_channels:
            obs_ =  np.concatenate(self.obs_hist, axis=-1)
        else:
            obs_ =  np.array(self.obs_hist).flatten()            
        return obs_
        
        
    def render(self, mode=None):
        return self._env.render(mode=mode)
    

env = gym