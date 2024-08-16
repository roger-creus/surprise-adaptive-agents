import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import cv2
from IPython import embed


class ResizeObservationWrapper(gym.Env):
    def __init__(self, 
                env, 
                new_size=(48,64,3), 
                new_shape=(64,48,3), 
                grayscale=False, 
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        print("Observation resize buffer")
        self._env = env
        self._new_size = new_size
        self._new_shape = new_shape
        self._grayscale = grayscale

        self.num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space_old = env.observation_space
        if grayscale:
            self._new_size = (new_size[0], new_size[1], 1)
            self._new_shape = (new_shape[0], new_shape[1], 1)
        else:
            self._new_size = (new_size[0], new_size[1], 3)
            self._new_shape = (new_shape[0], new_shape[1], 3)


        self.observation_space = Box(
                np.zeros(self._new_shape),
                np.ones(self._new_shape)
            )


    def step(self, action):
        # Take Action
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        # print(f"original shape of obs is:{obs.shape}")
        obs_ = self.resize_obs(obs)
        # print(f"resize observations size is: {obs_.shape}")
        return obs_, env_rew, envdone, envtrunc, info
    
    def resize_obs(self, obs, key=None):
        obs = cv2.resize(obs, dsize=tuple(self._new_size[:2]), interpolation=cv2.INTER_AREA)
        if (self._grayscale):
            obs = np.mean(obs, axis=-1, keepdims=True)
        return obs
    
    def reset(self, seed=None, options=None):
        '''
        Reset the wrapped env and the buffer
        '''
        obs, info = self._env.reset()
        obs_ = self.resize_obs(obs)
        return obs_, info
    
    def render(self, **kwargs):
        return self._env.render(**kwargs)