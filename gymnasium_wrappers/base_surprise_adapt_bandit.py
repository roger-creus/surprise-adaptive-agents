import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import cv2
from IPython import embed

class BaseSurpriseAdaptBanditWrapper(gym.Env):
    def __init__(
        self,
        env,
        buffer,
        add_true_rew=False,
        int_rew_scale=1.0,
        ext_only=False,
        max_steps = 500,
        theta_size = None,
        grayscale = None,
        soft_reset=True,
        ucb_coeff=np.sqrt(2),
        death_cost = False,
        exp_rew = False,
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
        self._death_cost = death_cost
        self._exp_rew = exp_rew
        self._theta_size = theta_size
        self._grayscale = grayscale
        self._soft_reset = soft_reset
        self.ucb_coeff = ucb_coeff
        self._death_cost = death_cost
        self._exp_rew = exp_rew
        self.pretrain_steps = 1000000
        self.current_steps = 0

        theta = self.buffer.get_params()
        print(f"theta shape:{theta.shape}")

        # Add true reward to surprise
        self.add_true_rew = add_true_rew
        self.int_rew_scale = int_rew_scale
        self.num_steps = 0
        self.num_eps = 0
        self.ext_only = ext_only
        self.max_steps = max_steps
        self.deaths = 0
        self.alpha_rolling_average = np.nan
        self.alpha_count = 0
        
        
        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        
        # the new theta shape has to be the extact theta.shape but +2 in first dimension
        # the additional dimension are the time-step and bandit choice
        new_theta_shape = (theta.shape[0], )
        for i in range(1, len(theta.shape)):
            if i == len(theta.shape)-1:
                new_theta_shape += (theta.shape[i] + 2, ) # in the last index (the channel dim) add 2 channels
            else:
                new_theta_shape += (theta.shape[i], )
        print(f"new_theta_shape: {new_theta_shape}")

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

        # Bandit statistics
        self.alpha_zero_mean = np.nan
        self.alpha_zero_cnt = 0
        self.alpha_one_mean = np.nan
        self.alpha_one_cnt = 0
        self.random_entropy = self._get_random_entropy()
        self.ucb_alpha_zero = None
        self.ucb_alpha_one = None
        self.alpha_t = np.nan
            
        print(self.observation_space)

    def _get_random_entropy(self):
        obs, _ = self._env.reset()
        self.buffer.reset()
        num_eps = 100
        random_entropies = []
        for _ in range(num_eps):
            for t in range(self.max_steps):
                obs, rew, envdone, envtrunc, info = self._env.step(self.action_space.sample())
                self.buffer.add(self.encode_obs(obs))
                if envdone or envtrunc:
                    if self._soft_reset:
                        obs, _ = self._env.reset()
                    else:
                        break
            random_entropy = self.buffer.entropy()
            random_entropies.append(random_entropy)
            obs, _ = self._env.reset()
            self.buffer.reset()
        print(f"len(random_entropies): {len(random_entropies)}")
        self.buffer.reset()
        return np.mean(random_entropies)

    def _get_alpha(self):
        ucb_alpha_zero = None
        ucb_alpha_one = None
        if np.isnan(self.alpha_t):
            alpha_t = np.random.binomial(n=1, p=0.5)
        elif self.current_steps < self.pretrain_steps:
            alpha_t = np.random.binomial(n=1, p=0.5)
        else:
            if self.alpha_zero_cnt == 0:
                alpha_t = 0
            elif self.alpha_one_cnt == 0:
                alpha_t = 1
            else:
                ucb_alpha_zero = self.ucb_coeff * np.sqrt(np.log(self.num_eps) / self.alpha_zero_cnt)
                ucb_alpha_one =  self.ucb_coeff * np.sqrt(np.log(self.num_eps) / self.alpha_one_cnt)
                alpha_t = np.argmax(
                    [
                        self.alpha_zero_mean
                        + ucb_alpha_zero,
                        self.alpha_one_mean
                        + ucb_alpha_one ,
                    ]
                )
        return alpha_t, ucb_alpha_zero, ucb_alpha_one
    
    def step(self, action):
        self.current_steps += 1
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        info['task_reward'] = env_rew
        self.task_return += env_rew
        info["alpha"] = self.alpha_t
        info["alpha_zero_mean"] = self.alpha_zero_mean
        info["alpha_zero_cnt"] = self.alpha_zero_cnt
        info["alpha_one_mean"] = self.alpha_one_mean
        info["alpha_one_cnt"] = self.alpha_one_cnt
        info["random_entropy"] = self.random_entropy
        
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
                info["alpha_rolling_average"] = self.alpha_rolling_average
                if self.ucb_alpha_one: info["ucb_alpha_one"] = self.ucb_alpha_one
                if self.ucb_alpha_zero: info["ucb_alpha_zero"] = self.ucb_alpha_zero
            else:
                envdone = False
                envtrunc = False
        else:
            if envdone or envtrunc:
                info["Average_task_return"] = self.task_return
                info["Average_episode_length"] = self.num_steps
                info['deaths'] = self.deaths
                info["alpha_rolling_average"] = self.alpha_rolling_average
                if self.ucb_alpha_one: info["ucb_alpha_one"] = self.ucb_alpha_one
                if self.ucb_alpha_zero: info["ucb_alpha_zero"] = self.ucb_alpha_zero

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

        rew = ((-1) ** self.alpha_t) * surprise
        
        
        if self.add_true_rew:
            rew = env_rew + (rew * self.int_rew_scale)
        else:
            rew = rew * self.int_rew_scale
            
        if self.ext_only:
            rew = env_rew

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
        alpha_t = (np.ones(theta.shape[:-1]) * self.alpha_t)[..., None]

        theta_obs = np.concatenate([theta,
                                    num_samples,
                                    alpha_t], axis=-1)
        
        # print(f"theta shape before cat: {theta.shape}")
        # print(f"theta shape after cat: {theta_obs.shape}")

        aug_obs["theta"] = theta_obs
        
        return aug_obs

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        self.num_steps = 0
        self.deaths = 0
        self.task_return = 0

        # update the bandit statistics
        if not np.isnan(self.alpha_t):
            entropy_change = (
                np.abs(self.buffer.entropy() - self.random_entropy)
                / np.abs(self.random_entropy)
            )
            if self.alpha_t == 0:
                if np.isnan(self.alpha_zero_mean):
                    self.alpha_zero_mean = entropy_change
                    self.alpha_zero_cnt += 1
                else:
                    self.alpha_zero_mean = (
                        self.alpha_zero_mean * self.alpha_zero_cnt + entropy_change
                    )
                    self.alpha_zero_cnt += 1
                    self.alpha_zero_mean /= self.alpha_zero_cnt
            else:
                if np.isnan(self.alpha_one_mean):
                    self.alpha_one_mean = entropy_change
                    self.alpha_one_cnt += 1
                else:
                    self.alpha_one_mean = (
                        self.alpha_one_mean * self.alpha_one_cnt + entropy_change
                    )
                    self.alpha_one_cnt += 1
                    self.alpha_one_mean /= self.alpha_one_cnt

        # select new alpha
        self.alpha_t,  self.ucb_alpha_zero, self.ucb_alpha_one = self._get_alpha()
        # track the rolling average of alpha
        if np.isnan(self.alpha_rolling_average):
            self.alpha_rolling_average = self.alpha_t
            self.alpha_count += 1
        else:
            self.alpha_rolling_average = self.alpha_rolling_average + (1/self.alpha_count) * (self.alpha_t - self.alpha_rolling_average)
            self.alpha_count += 1

        info["entropy_change"] = (self.buffer.entropy() - self.random_entropy)

        self.num_eps += 1
        self.num_steps = 0
        
        if self.heatmap is not None:
            info["heatmap"] = self.heatmap.copy()
        try:
            self.heatmap = np.zeros((self._env.width, self._env.height))
        except:
            self.heatmap = None
        
        self.buffer.reset()
        obs = self.get_obs(obs)
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
        elif isinstance(obs, dict):
            return obs["obs"].astype(np.float32)
        else:
            return obs.astype(np.float32)