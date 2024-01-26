import gymnasium as gym
import matplotlib.pyplot as plt


# Some envs do not support gymnasium, this simple wrapper is to convert from gym to gymnasium api
class RenderObservationWrapper(gym.Env):
    def __init__(self, env, rendering_freq):
        print(f"RenderObservation wrapper")
        self._env = env
        self._rendering_freq = rendering_freq
        self.action_space =  env.action_space
        self.observation_space = env.observation_space
        self.episode_count = 0
        self._frames = []
    
    def reset(self, seed=None, options=None):
        self.t = 0
        obs, info = self._env.reset()
        self.episode_count += 1
        if self._rendering_freq > 0 and self.episode_count % self._rendering_freq == 0:
            img = self._env.render()
            self._frames.append(img)
        else:
            self._frames.clear()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        if self._rendering_freq > 0 and self.episode_count % self._rendering_freq == 0:
            self._frames.append(self._env.render())
            if done or truncated:
                info["frames"] = self._frames
        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        try:
            return self._env.render(render_mode=self._render_mode)
        except:
            return self._env.render(**kwargs)
    

# testing
'''
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RenderObservationWrapper(env, 1)
env.reset()
done = False
while not done:
    step = env.step(env.action_space.sample())
    done = step[2]
print(step[-1]["frames"])
'''


