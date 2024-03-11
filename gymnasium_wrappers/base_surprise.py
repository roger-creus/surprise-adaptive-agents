import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import cv2
from IPython import embed

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
                 soft_reset=True,
                 survival_rew=False,
                 death_cost = False,
                 exp_rew = False
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
        self._soft_reset = soft_reset
        self._survival_rew = survival_rew
        self._death_cost = death_cost
        self._exp_rew = exp_rew

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
        
        # the new theta shape has to be the extact theta.shape but +1 in channel dimension
        # the additional dimension is the time-step
        new_theta_shape = (theta.shape[0], )
        for i in range(1, len(theta.shape)):
            if i == len(theta.shape)-1:
                new_theta_shape += (theta.shape[i] + 1, ) # in the last index (the channel dim) add 1 channels
            else:
                new_theta_shape += (theta.shape[i], )
        print(f"new_theta_shape:{new_theta_shape}")

        # instead of hardcoding the keys. Make sure to add all the keys from the original observation space
        obs_space = {}
        if isinstance(self.env_obs_space, Box):
            obs_space["obs"] = self.env_obs_space
        elif isinstance(self.env_obs_space, Dict):
            for key in self.env_obs_space.spaces.keys():
                obs_space[key] = self.env_obs_space.spaces[key]
        else:
            raise ValueError("Observation space not supported")

        obs_space["theta"] = Box(-np.inf, np.inf, shape=new_theta_shape)
        self.observation_space = Dict(obs_space)

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
                # obs = np.random.rand(*obs.shape)
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
            if envdone or envtrunc:
                info["Average_task_return"] = self.task_return
                info["Average_episode_length"] = self.num_steps
                info['deaths'] = self.deaths

        # use the original observation for surprise calculation
        # this will be used for griddly envs and compute surprise with the bernoulli buffer
        surprise = -self.buffer.logprob(self.encode_obs(obs))
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh
        

        self.buffer.add(self.encode_obs(obs))
        info['surprise'] = surprise
        info["theta_entropy"] = self.buffer.entropy()
        info['deaths'] = self.deaths
        
        # Add observation to buffer
        if self._exp_rew:
            surprise = np.exp(surprise)

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
        if self._survival_rew:
            rew = 1.

        if self._death_cost and (envdone or envtrunc):
            rew = -100
            
                
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
        num_samples = (np.ones(1)*self.buffer.buffer_size) / self.max_steps

        aug_obs = {}
        if isinstance(self.env_obs_space, Box):
            aug_obs["obs"] = obs
        elif isinstance(self.env_obs_space, Dict):
            for key in self.env_obs_space.spaces.keys():
                aug_obs[key] = obs[key]
        else:
            raise ValueError("Observation space not supported")
        
        num_samples = (np.ones(theta.shape[:-1]) * num_samples)[..., None]

        theta_obs = np.concatenate([theta,
                                    num_samples,
                                    ], axis=-1)
        
        # print(f"theta shape before cat: {theta.shape}")
        # print(f"theta shape after cat: {theta_obs.shape}")

        aug_obs["theta"] = theta_obs
        
        return aug_obs

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
        # print(f"obs shape: {obs.shape}")
        if self._theta_size:
            # if the image is stack of images then take the first one
            if self._grayscale:
                theta_obs = obs[:, :, -1] [:, :, None]
            else:
                theta_obs = obs[:, :, -3:]
            # print(f"theta shape before resize: {theta_obs.shape}")
            theta_obs = cv2.resize(theta_obs, dsize=tuple(self._theta_size[:2]), interpolation=cv2.INTER_AREA)
            theta_obs = theta_obs.astype(np.float32)[:, :, None]
            # print(f"theta_obs.shape:{theta_obs.shape}")
            return theta_obs
        elif isinstance(obs, dict):
            return obs["obs"].astype(np.float32)
        else:
            return obs.astype(np.float32)