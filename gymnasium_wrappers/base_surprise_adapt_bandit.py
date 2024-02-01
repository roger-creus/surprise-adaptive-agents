import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import cv2
from IPython import embed
from gym.wrappers.normalize import RunningMeanStd

# This is taken from the bandit_final branch

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
        soft_reset=True
    ):
        """
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        """
        self._env = env
        self.buffer = buffer
        self.max_steps = max_steps
        self.add_true_rew = add_true_rew
        self.int_rew_scale = int_rew_scale
        self.ext_only = ext_only
        self._theta_size = theta_size
        self._grayscale = grayscale
        self._soft_reset = soft_reset

        print(f"_theta_size:{self._theta_size}")
        print(f"_grayscale:{self._grayscale}")

        theta = self.buffer.get_params()

        print(f"theta shape:{theta.shape}")
        
        self.num_steps = 0
        self.num_eps = 0
        self.deaths = 0
        self.alpha_rolling_average = 0
        self.alpha_count = 1

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space

        # adding theta and t for consistent MDP (same as SMiRL)
        # adding history of surprise (either -1 or 1)
        self.observation_space = Dict({
            "obs" : Box(-np.inf, np.inf, shape=self.env_obs_space.shape),
            "theta": Box(-np.inf, np.inf, shape=(np.prod(theta.shape) + 1 + 1,)), # The first 1 is for the timestep and the second is for alpha_t
        })
            
        # Bandit statistics
        self.alpha_zero_mean = np.nan
        self.alpha_zero_cnt = 0
        self.alpha_one_mean = np.nan
        self.alpha_one_cnt = 0
        self.random_entropy = self._get_random_entropy()
        self.ucb_alpha_zero = None
        self.ucb_alpha_one = None
        self.alpha_t = np.nan
        try:
            self.heatmap = np.zeros((env.width, env.height))
        except:
            self.heatmap = None

        self.reset()

    def set_alpha_zero_mean(self, val):
        self.alpha_zero_mean = val
    
    def set_alpha_one_mean(self, val):
        self.alpha_one_mean = val

    def _get_random_entropy(self):
        self._env.reset()
        self.buffer.reset()
        # number of episodes to evaluate the random entorpy
        num_eps = 5 if not self._soft_reset else 1
        random_entropies = []
        for _ in range(num_eps):
            for _ in range(self.max_steps):
                obs, rew, done, truncated, info = self._env.step(self.action_space.sample())
                self.buffer.add(self.encode_obs(obs))
                # softreset
                if done or truncated:
                    if self._soft_reset:
                        obs, _ = self._env.reset()
                        obs = np.random.rand(*obs.shape)
                        self.buffer.add(self.encode_obs(obs))
                    else:
                        break
            random_entropy = self.buffer.entropy()
            random_entropies.append(random_entropy)
            self._env.reset()
        random_entropy = np.mean(random_entropies)
        return random_entropy

    def _get_alpha(self):
        ucb_alpha_zero = None
        ucb_alpha_one = None
        if np.isnan(self.alpha_t):
            alpha_t = np.random.binomial(n=1, p=0.5)
        else:
            if self.alpha_zero_cnt == 0:
                alpha_t = 0
            elif self.alpha_one_cnt == 0:
                alpha_t = 1
            else:
                ucb_alpha_zero = np.sqrt(2 * np.log(self.num_eps) / self.alpha_zero_cnt)
                ucb_alpha_one = np.sqrt(2 * np.log(self.num_eps) / self.alpha_one_cnt)
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
        # Take Action
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        self.task_return += env_rew
        info["task_reward"] = env_rew
        info["alpha"] = self.alpha_t
        info["alpha_zero_mean"] = self.alpha_zero_mean
        info["alpha_zero_cnt"] = self.alpha_zero_cnt
        info["alpha_one_mean"] = self.alpha_one_mean
        info["alpha_one_cnt"] = self.alpha_one_cnt
        info["random_entropy"] = self.random_entropy

        # soft reset
        if self._soft_reset:
            if envdone or envtrunc:
                obs, _ = self._env.reset()
                obs = np.random.rand(*obs.shape)
                self.deaths += 1

            if self.num_steps >= self.max_steps:
                envdone = True
                envtrunc = True
                if self.deaths > 0:
                    self.task_return /= self.deaths
                    self.num_steps /= self.deaths
                info["Average_task_return"] = self.task_return
                info["Average_episode_length"] = self.num_steps
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
                info["alpha_rolling_average"] = self.alpha_rolling_average
                if self.ucb_alpha_one: info["ucb_alpha_one"] = self.ucb_alpha_one
                if self.ucb_alpha_zero: info["ucb_alpha_zero"] = self.ucb_alpha_zero

        # Compute surprise as the negative log probability of the observation
        surprise = -self.buffer.logprob(self.encode_obs(obs))
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh

        rew = ((-1) ** self.alpha_t) * surprise

        # Add observation to buffer
        self.buffer.add(self.encode_obs(obs))

        info["surprise_adapt_reward"] = rew
        info["theta_entropy"] = self.buffer.entropy()
        info['deaths'] = self.deaths
        
        if self.add_true_rew:
            rew = env_rew + (rew * self.int_rew_scale)
        else:
            rew = rew * self.int_rew_scale
            
        if self.ext_only:
            rew = env_rew        

        info["surprise"] = surprise
        info["alpha"] = self.alpha_t

        # augment next state
        obs = self.get_obs(obs)

        try:
            x, y = self._env.agent_pos
            self.heatmap[x, y] += 1
            info["heatmap"] = self.heatmap.copy()
        except:
            pass

        self.num_steps += 1

        return obs, rew, envdone, envtrunc, info

    def get_obs(self, obs):
        """
        Augment observation, perhaps with generative model params, time-step, current surprise momentum.
        """
        theta = self.buffer.get_params()
        num_samples = np.ones(1) * self.buffer.buffer_size
        alpha_t = np.ones(1) * self.alpha_t
        aug_obs = {
        "obs" : obs,
        "theta": np.concatenate([np.array(theta).flatten(), num_samples, alpha_t])
        }
        return aug_obs

    def reset(self,  seed=None, options=None):
        """
        Reset the wrapped env and the buffer
        """

        obs, info = self._env.reset()
        self.task_return = 0
        self.deaths = 0

        # Update the bandit action values
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
        self.alpha_rolling_average = self.alpha_rolling_average + (1/self.alpha_count) * (self.alpha_t - self.alpha_rolling_average)
        self.alpha_count += 1

        self.buffer.reset()

        self.num_eps += 1
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
        
