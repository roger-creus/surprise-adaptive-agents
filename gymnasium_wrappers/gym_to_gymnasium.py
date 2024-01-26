import gymnasium as gym
from gymnasium.spaces import Discrete


# Some envs do not support gymnasium, this simple wrapper is to convert from gym to gymnasium api
class GymToGymnasium(gym.Env):
    def __init__(self, env, render_mode, max_steps):
        print(f"GymToGymnasium wrapper")
        self._env = env
        self._render_mode = render_mode
        self._max_steps = max_steps
        self.action_space =  Discrete(env.action_space.n)
        self.observation_space = env.observation_space
    
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
            return self._env.render(render_mode=self._render_mode)
        except:
            return self._env.render(**kwargs)
    