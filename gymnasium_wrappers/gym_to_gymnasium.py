import gym as gym_old
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from IPython import embed



# Some envs do not support gymnasium, this simple wrapper is to convert from gym to gymnasium api
class GymToGymnasium(gym.Env):
    def __init__(self, env, render_mode, max_steps):
        print(f"GymToGymnasium wrapper")
        self._env = env
        self._render_mode = render_mode
        self._max_steps = max_steps
        self.action_space =  Discrete(env.action_space.n)
        
        if isinstance(env.observation_space, gym_old.spaces.Box):
            self.observation_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=env.observation_space.dtype, shape=env.observation_space.shape)
        elif isinstance(env.observation_space, gym_old.spaces.Discrete):
            self.observation_space = Discrete(env.observation_space.n)
        elif isinstance(env.observation_space, gym_old.spaces.Dict) or isinstance(env.observation_space, dict):
            self.observation_space = Dict({k: Box(low=env.observation_space[k].low, high=env.observation_space[k].high, dtype=env.observation_space[k].dtype, shape=env.observation_space[k].shape) for k in env.observation_space.keys()})
        else:
            raise ValueError(f"Unsupported observation space {env.observation_space}")
    
    def reset(self, seed=None, options=None):
        self.t = 0
        info = {}
        obs = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        truncated = done
        self.t += 1
        if self.t == self._max_steps:
            done = truncated = True
        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        try:
            return self._env.render(mode=self._render_mode)
        except:
            return self._env.render(**kwargs)
    