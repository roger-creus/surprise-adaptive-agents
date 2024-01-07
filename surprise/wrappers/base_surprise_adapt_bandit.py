import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import util.class_util as classu
from collections import deque


class BaseSurpriseAdaptBanditWrapper(gym.Wrapper):
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
        latent_obs_size=None,
        obs_label=None,
        obs_out_label=None,
    ):
        """
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        """
        super().__init__(env)

        theta = self._buffer.get_params()
        self._num_steps = 0
        self._num_eps = 0

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
        self.reset()

    def set_alpha_zero_mean(self, val):
        self.alpha_zero_mean = val
    
    def set_alpha_one_mean(self, val):
        self.alpha_one_mean = val

    def _get_random_entropy(self):
        super().reset()
        done = False
        while not done:
            obs, rew, done, info = super().step(self.action_space.sample())
            self._buffer.add(self.encode_obs(obs))
        random_entropy = self._buffer.entropy()
        super().reset()
        return random_entropy

    def _get_alpha(self):
        if self.eval:
            alpha_t = np.argmax([self.alpha_zero_mean, self.alpha_one_mean])
        else:
            if self.alpha_zero_cnt == 0:
                alpha_t = 0
            elif self.alpha_one_cnt == 0:
                alpha_t = 1
            else:
                alpha_t = np.argmax(
                    [
                        self.alpha_zero_mean
                        + np.sqrt(2 * np.log(self._num_eps) / self.alpha_zero_cnt),
                        self.alpha_one_mean
                        + np.sqrt(2 * np.log(self._num_eps) / self.alpha_one_cnt),
                    ]
                )
        return alpha_t

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

        # Compute surprise as the negative log probability of the observation
        # print(self.encode_obs(obs))
        obs_shape = obs["observation"].shape
        print(f"obs shape:{obs_shape}")
        print(f"self.theta.shape: {self._buffer.get_params().shape}")
        surprise = -self._buffer.logprob(self.encode_obs(obs))
        # print(surprise)
        # For numerical stability, clip stds to not be 0
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh)

        rew = ((-1) ** self.alpha_t) * surprise

        # Add observation to buffer
        self._buffer.add(
            self.encode_obs(obs)
        )  # this adds the raw observations to the buffer no? shouldnt we add the augmented obs?
        if self._obs_out_label is None:
            info["surprise_adapt_reward"] = rew
            info["theta_entropy"] = self._buffer.entropy()
        else:
            info[self._obs_out_label + "surprise_adapt_reward"] = rew
            info[self._obs_out_label + "theta_entropy"] = self._buffer.entropy()
        if self._smirl_rew_scale is not None:
            rew = rew * self._smirl_rew_scale
        if self._add_true_rew == "only":
            rew = env_rew
        elif self._add_true_rew:
            rew = (rew) + env_rew

        info["surprise"] = surprise
        info["alpha"] = self.alpha_t

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
        alpha_t = np.ones(1) * self.alpha_t

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
        self.alpha_t = self._get_alpha()

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
        if self._obs_label is None:
            return np.array(obs).flatten().copy()
        else:
            return np.array(obs[self._obs_label]).flatten().copy()
