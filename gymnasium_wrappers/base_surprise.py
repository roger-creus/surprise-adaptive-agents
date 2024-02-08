import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import cv2
from IPython import embed
from gym.wrappers.normalize import RunningMeanStd

class BaseSurpriseWrapper(gym.Env):
    def __init__(self, 
                 env, 
                 buffer, 
                 add_true_rew=False,
                 minimize=True,
                 int_rew_scale=1.0,
                 ext_only=False,
                 max_steps = 500,
                 theta_size = None,
                 grayscale = None,
                 scale_by_std=False,
                 soft_reset=True
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''

        self._env = env
        self.buffer = buffer
        self._theta_size = theta_size
        self._grayscale = grayscale
        self._scale_by_std = scale_by_std
        self._soft_reset = soft_reset

        if scale_by_std:
            print("Scaling surprise by std")
            self.rms = RunningMeanStd()

        print(f"_theta_size:{self._theta_size}")
        print(f"_grayscale:{self._grayscale}")

        theta = self.buffer.get_params()
        print(f"theta shape:{theta.shape}")

        # Add true reward to surprise
        self.add_true_rew = add_true_rew
        self.int_rew_scale = int_rew_scale
        self.num_steps = 0
        self.minimize = minimize
        self.ext_only = ext_only
        self.max_steps = max_steps
        self.deaths = 0
        
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
        self.task_return += env_rew

        
        # soft reset
        if self._soft_reset:
            if envdone:
                obs, _ = self._env.reset()
                obs = np.random.rand(*obs.shape)
                self.deaths += 1
            if self.num_steps == self.max_steps:
                envdone = True
                envtrunc = True
                if self.deaths > 0:
                    self.task_return /= self.deaths
                    self.num_steps /= self.deaths
                info["Average_task_return"] = self.task_return
                info["Average_episode_length"] = self.num_steps
                info['deaths'] = self.deaths
            else:
                envdone = False
                envtrunc = False
        else:
            if self.num_steps == self.max_steps:
                envdone = True
                envtrunc = True
            info["Average_task_return"] = self.task_return
            info["Average_episode_length"] = self.num_steps
            info['deaths'] = self.deaths


        surprise = -self.buffer.logprob(self.encode_obs(obs))
        if self._scale_by_std:
            self.rms.update(np.array([surprise]))
            surprise = (surprise / np.sqrt(self.rms.var)).item()
        else:
            thresh = 300
            surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh
        
        self.buffer.add(self.encode_obs(obs))
        info['surprise'] = surprise
        info["theta_entropy"] = self.buffer.entropy()
        info['deaths'] = self.deaths
        
        # Add observation to buffer
        if self.minimize:
            rew = -surprise
        else:
            rew = surprise
        
        if self.add_true_rew:
            rew = env_rew + (rew * self.int_rew_scale)
        else:
            rew = rew * self.int_rew_scale
            
        if self.ext_only:
            rew = env_rew
                
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
        self.deaths = 0
        self.task_return = 0
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

    def encode_obs(self, obs):
        """
        Used to encode the observation before putting on the buffer
        """
        if self._theta_size:
            # if the image is stack of images then take the first one
            if self._grayscale:
                theta_obs = obs[:, :, -1]
            else:
                theta_obs = obs[:, :, -3:]
            theta_obs = cv2.resize(theta_obs, dsize=tuple(self._theta_size[:2]), interpolation=cv2.INTER_AREA)
            theta_obs = theta_obs.flatten().astype(np.float32)
            return theta_obs
        else:
            return obs.astype(np.float32)