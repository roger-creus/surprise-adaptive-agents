import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import util.class_util as classu
from collections import deque


class BaseSurpriseDiffWrapper(gym.Wrapper):
    @classu.hidden_member_initialize
    def __init__(
        self,
        env,
        buffer,
        time_horizon,
        eval=False,
        add_true_rew=False,
        smirl_rew_scale=None,
        buffer_type=None,
        obs_key = None,
        latent_obs_size=None,
        obs_label=None,
        obs_out_label=None,
        clip_surprise = True,
        normalize_timestep = True,
        random_entropy_num_eps = 5, # number of episodes used to estimate the entorpy of the random agent
        use_difference_reward = True
    ):
        """
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        """
        super().__init__(env)
        print("surprise diff wrapper")
        theta = self._buffer.get_params()
        self._num_steps = 0
        self._num_eps = 0
        self.alpha_rolling_average = 0
        self.alpha_count = 1
        self.random_entropy_num_eps = random_entropy_num_eps
        self._normalize_timestep = normalize_timestep

        # Gym spaces
        self.action_space = env.action_space
        if hasattr(env, "env_obs_space"):
            self.env_obs_space = env.env_obs_space
        else:
            self.env_obs_space = env.observation_space

        print(f"env observation: {self.env_obs_space}")
        print(f"env action space: {self.action_space}")

        self.eval = eval

        # adding theta and t for consistent MDP (same as SMiRL)
        # adding history of surprise (either -1 or 1)
        if self._obs_out_label is None:
            self.observation_space = Box(
                np.concatenate(
                    (
                        self.env_obs_space.low.flatten(),
                        np.zeros(theta.shape),
                        np.zeros(1),
                        np.ones(1) * -1,
                    )
                ),
                np.concatenate(
                    (
                        self.env_obs_space.high.flatten(),
                        np.ones(theta.shape),
                        np.ones(1) * time_horizon,
                        np.ones(1),
                    )
                ),
            )
            
        elif obs_key is None:
            self.observation_space = Dict(
                {
                    self._obs_label: Box(
                        self.env_obs_space.low, self.env_obs_space.high
                    ),
                    self._obs_out_label: Box(
                        np.concatenate(
                            (np.zeros(theta.shape), np.zeros(1), np.ones(1) * -1)
                        ),
                        np.concatenate(
                            (np.zeros(theta.shape), np.zeros(1), np.ones(1))
                        ),
                    ),
                }
            )
        else:
            self.observation_space = Dict(
                {
                    self._obs_label: Box(
                        self.env_obs_space.low, self.env_obs_space.high
                    ),
                    self._obs_out_label: Box(
                        np.concatenate(
                            (np.zeros(theta.shape), np.zeros(1), np.ones(1) * -1)
                        ),
                        np.concatenate(
                            (np.zeros(theta.shape), np.zeros(1), np.ones(1))
                        ),
                    ),
                }
            )

        self.alpha_zero_mean = np.nan
        self.alpha_zero_cnt = 0
        self.alpha_one_mean = np.nan
        self.alpha_one_cnt = 0
        self.random_entropy = self._get_random_entropy()
        self.time_horizon = time_horizon
        self.reset()

    def set_alpha_zero_mean(self, val):
        self.alpha_zero_mean = val
    
    def set_alpha_one_mean(self, val):
        self.alpha_one_mean = val

    def _get_random_entropy(self):
        print("-------Calculating random policy entropy-----------")
        entropies = []
        surprise = np.zeros((self.random_entropy_num_eps, self._time_horizon)) + 1e-8 # for numerical stability
        for eps in range(self.random_entropy_num_eps):
            super().reset()
            done = False
            time_step = 0
            while not done:
                obs, rew, done, info = super().step(self.action_space.sample())
                self._buffer.add(self.encode_obs(obs))
                surprise[eps][time_step] = -self._buffer.logprob(self.encode_obs(obs))
                time_step += 1
            random_entropy = self._buffer.entropy()
            entropies.append(random_entropy)
            super().reset()
            self._buffer.reset()
        random_entropy = np.mean(entropies)
        mean_s = self.random_surprise = np.mean(surprise, axis=0)
        # print(f"mean surprises: {mean_s}")
        # print(len(mean_s))
        # quit()
        return random_entropy

    def _get_alpha(self):
        ucb_alpha_zero = None
        ucb_alpha_one = None
        if self.eval:
            alpha_t = np.argmax([self.alpha_zero_mean, self.alpha_one_mean])
        else:
            if self.alpha_zero_cnt == 0:
                alpha_t = 0
            elif self.alpha_one_cnt == 0:
                alpha_t = 1
            else:
                ucb_alpha_zero = np.sqrt(2 * np.log(self._num_eps) / self.alpha_zero_cnt)
                ucb_alpha_one = np.sqrt(2 * np.log(self._num_eps) / self.alpha_one_cnt)
                alpha_t = np.argmax(
                    [
                        self.alpha_zero_mean
                        + ucb_alpha_zero,
                        self.alpha_one_mean
                        + ucb_alpha_one,
                    ]
                )
        return alpha_t, ucb_alpha_zero, ucb_alpha_one

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        info["task_reward"] = env_rew
        info["alpha"] = self.alpha_t
        info["alpha_zero_mean"] = self.alpha_zero_mean
        info["alpha_zero_cnt"] = self.alpha_zero_cnt
        info["alpha_one_mean"] = self.alpha_one_mean
        info["alpha_one_cnt"] = self.alpha_one_cnt
        info["random_entropy"] = self.random_entropy
        if self.ucb_alpha_zero:
            info["ucb_alpha_zero"] = self.ucb_alpha_zero
        if self.ucb_alpha_one:
            info["ucb_alpha_one"] = self.ucb_alpha_one
        info["alpha_rolling_average"] = self.alpha_rolling_average

        # Compute surprise as the negative log probability of the observation
        # print(self.encode_obs(obs))
        obs_shape = obs["observation"].shape
        # print(f"obs shape:{obs_shape}")
        surprise = -self._buffer.logprob(self.encode_obs(obs))
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh)
        rew = np.abs(
            surprise - self.random_surprise[self._num_steps])  / np.abs(self.random_surprise[self._num_steps])
        # print(s)
        info["surprise_difference"] = rew
        info["theta_entropy"] = self._buffer.entropy()
        self._buffer.add(
                self.encode_obs(obs)
            )

        # augment next state
        obs = self.get_obs(obs)

        self._num_steps += 1

        return obs, rew, self.get_done() or envdone, info

    def get_done(self):
        return self.num_steps >= self._time_horizon

    def get_obs(self, obs):
        """
        Augment observation, perhaps with generative model params, time-step, current surprise momentum.
        """
        theta = self._buffer.get_params()
        num_samples = np.ones(1) * self._buffer.buffer_size
        if self._normalize_timestep:
            num_samples /= self.time_horizon
        alpha_t = np.ones(1) * self.alpha_t
        # print(f"normalized time horizon: {num_samples}")
        # quit()

        if self._obs_out_label is None:
            obs = np.concatenate(
                [
                    np.array(obs).flatten(),
                    np.array(theta).flatten(),
                    num_samples,
                    alpha_t,
                ]
            )
        else:
            obs[self._obs_out_label] = np.concatenate(
                [np.array(theta).flatten(), num_samples, alpha_t]
            )

        return obs

    # def get_done(self, env_done):
    def get_done(self):
        """
        figure out if we're done

        params
        ======
        env_done (bool) : done bool from the wrapped env, doesn't
            necessarily need to be used
        """
        return self._num_steps >= self._time_horizon

    def reset(self):
        """
        Reset the wrapped env and the buffer
        """

        obs = self._env.reset()
        #         print ("surprise obs shape1, ", obs.shape)
        if not self.eval:
            entropy_change = (
                np.abs(self._buffer.entropy() - self.random_entropy)
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
        self.alpha_t, self.ucb_alpha_zero, self.ucb_alpha_one = self._get_alpha()
        
        # track the rolling average of alpha
        self.alpha_rolling_average = self.alpha_rolling_average + (1/self.alpha_count) * (self.alpha_t - self.alpha_rolling_average)
        self.alpha_count += 1

        self._buffer.reset()
        self._num_eps += 1

        self._num_steps = 0
        obs = self.get_obs(obs)
        #         print ("surprise obs shape2, ", obs.shape)
        return obs

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def encode_obs(self, obs):
        """
        Used to encode the observation before putting on the buffer
        """
        # print(f"obs in get obs:{obs}")
        shape = obs[self._obs_label].shape
        # print(f"obs in get obs shpae:{shape}")
        if self._obs_label is None:
            return np.array(obs).flatten().copy()
        else:
            return np.array(obs[self._obs_label]).flatten().copy()

