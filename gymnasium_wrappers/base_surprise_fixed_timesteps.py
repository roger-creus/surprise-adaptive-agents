import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

from IPython import embed

class BaseSurpriseFixedTimeStepsWrapper(gym.Env):
    def __init__(self, 
                 env, 
                 buffer,
                 minimize=True,
                 int_rew_scale=1.0,
                 ext_only=False,
                 max_steps = 500
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''

        self._env = env
        self.buffer = buffer

        theta = self.buffer.get_params()

        # Add true reward to surprise
        self.int_rew_scale = int_rew_scale
        self.num_steps = 0
        self.minimize = minimize
        self.ext_only = ext_only
        self.max_steps = max_steps
        
        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        
        self.observation_space = Dict({
            "obs" : Box(-np.inf, np.inf, shape=self.env_obs_space.shape),
            "theta": Box(-np.inf, np.inf, shape=(np.prod(theta.shape) + 1,))
        })

        try:
            self.heatmap = np.zeros((env.width, env.height))
        except:
            self.heatmap = None
            
        print(self.observation_space)


    def step(self, action):
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        info['task_reward'] = env_rew

        if self.num_steps >= self.max_steps:
            envdone = True
            envtrunc = True
        else:
            envdone = False
            envtrunc = False

        surprise = -self.buffer.logprob(obs.astype(np.float32))
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh
        
        self.buffer.add(obs.astype(np.float32))
        info['surprise'] = surprise
        info["theta_entropy"] = self.buffer.entropy()
        
        # Add observation to buffer
        if self.minimize:
            rew = -surprise
        else:
            rew = surprise
                
        try:
            x, y = self._env.agent_pos
            self.heatmap[x, y] += 1
            info["heatmap"] = self.heatmap.copy()
        except:
            pass

        self.num_steps += 1
        return self.get_obs(obs), rew, envdone, envtrunc, info

    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        theta = self.buffer.get_params()
        num_samples = np.ones(1)*self.buffer.buffer_size
        obs = {
            "obs" : obs
        }
        obs["theta"] = np.concatenate([np.array(theta).flatten(), num_samples])
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        self.buffer.reset()
        self.num_steps = 0
        obs = self.get_obs(obs)
        
        if self.heatmap is not None:
            info["heatmap"] = self.heatmap.copy()
        try:
            self.heatmap = np.zeros((self._env.width, self._env.height))
        except:
            self.heatmap = None
            
        return obs, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)


class MountainCarSurpriseWrapper(gym.Env):
    def __init__(self, 
                 env, 
                 buffer,
                 minimize=True,
                 int_rew_scale=1.0,
                 ext_only=False,
                 max_steps = 500
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''

        self._env = env
        self.buffer = buffer

        theta = self.buffer.get_params()

        # Add true reward to surprise
        self.int_rew_scale = int_rew_scale
        self.num_steps = 0
        self.minimize = minimize
        self.ext_only = ext_only
        self.max_steps = max_steps
        
        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        
        self.observation_space = Dict({
            "obs" : Box(-np.inf, np.inf, shape=self.env_obs_space.shape),
            "theta": Box(-np.inf, np.inf, shape=(np.prod(theta.shape) + 1,))
        })

        try:
            self.heatmap = np.zeros((env.width, env.height))
        except:
            self.heatmap = None
            
        print(self.observation_space)


    def step(self, action):
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        info['task_reward'] = env_rew

        if self.num_steps >= self.max_steps:
            envdone = True
            envtrunc = True
        else:
            envdone = False
            envtrunc = False

        surprise = -self.buffer.logprob(obs.astype(np.float32))
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh
        
        self.buffer.add(obs.astype(np.float32))
        info['surprise'] = surprise
        info["theta_entropy"] = self.buffer.entropy()
        
        # Add observation to buffer
        if self.minimize:
            rew = -surprise
        else:
            rew = surprise
                
        try:
            x, y = self._env.agent_pos
            self.heatmap[x, y] += 1
            info["heatmap"] = self.heatmap.copy()
        except:
            pass

        self.num_steps += 1
        info["height"] = self.height(obs[0])
        info["velocity"] = obs[1]
        return self.get_obs(obs), rew, envdone, envtrunc, info

    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        theta = self.buffer.get_params()
        num_samples = np.ones(1)*self.buffer.buffer_size
        obs = {
            "obs" : obs
        }
        obs["theta"] = np.concatenate([np.array(theta).flatten(), num_samples])
        return obs

    def reset(self, seed=None, options=None):
        # Use a wide range of initial state so there is some randomness in the evnironment
        options = {
        "low": -1,
        "high": 0.3	
        }
        obs, info = self._env.reset(options = options)
        self.buffer.reset()
        self.num_steps = 0
        obs = self.get_obs(obs)
        
        if self.heatmap is not None:
            info["heatmap"] = self.heatmap.copy()
        try:
            self.heatmap = np.zeros((self._env.width, self._env.height))
        except:
            self.heatmap = None

        info["height"] = self.height(obs[0])
        info["velocity"] = obs[1]
        return obs, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def height(self, x_positon):
        return np.sin(3 * x_positon) * 0.45 + 0.55