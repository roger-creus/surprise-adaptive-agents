import numpy as np
import gym
from gym.spaces import Box
import pdb
import util.class_util as classu
from collections import deque

class BaseSurpriseAdaptWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, 
                 env, 
                 buffer, 
                 time_horizon,
                 surprise_window_len,
                 add_true_rew=False,
                 smirl_rew_scale=None, 
                 buffer_type=None,
                 latent_obs_size=None,
                 obs_label=None,
                 obs_out_label=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''

        theta = self._buffer.get_params()
        self._num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space

        # adding theta and t for consistent MDP (same as SMiRL)
        # adding history of surprise (either -1 or 1)
        self.observation_space = Box(
                np.concatenate(
                    (self.env_obs_space.low.flatten(), 
                     np.zeros(theta.shape), 
                     np.zeros(1),
                     np.ones(1) * -1)
                ),
                np.concatenate(
                    (self.env_obs_space.high.flatten(), 
                     np.ones(theta.shape), 
                     np.ones(1)*time_horizon,
                     np.ones(1))
                )
            )

        self.surprise_window_len = surprise_window_len
        
        self.reset()

    def init_surprise_window(self):
        self.surprise_counter = 0
        self.surprise_window = deque()
        self.alpha_t = 1 if np.random.binomial(1, 0.5) == 1 else 0

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        info['task_reward'] = env_rew

        # Compute surprise as the negative log probability of the observation
        surprise = - self._buffer.logprob(self.encode_obs(obs))
        # For numerical stability, clip stds to not be 0
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh)
        
        rew = ((-1)**self.alpha_t) * surprise
        
        # remove old elements from the surprise window
        if self.surprise_counter > self.surprise_window_len:
            self.surprise_window.popleft()
        self.surprise_counter += 1
            
        # add new element to the surprise window
        self.surprise_window.append(surprise)
        
        # Add observation to buffer
        self._buffer.add(self.encode_obs(obs)) # this adds the raw observations to the buffer no? shouldnt we add the augmented obs?
        if (self._obs_out_label is None):
            info['surprise_adapt_reward'] = rew
            info["theta_entropy"] = self._buffer.entropy()
        else:
            info[self._obs_out_label + 'surprise_adapt_reward'] = rew
            info[self._obs_out_label + "theta_entropy"] = self._buffer.entropy()
        if (self._smirl_rew_scale is not None):
            rew = (rew * self._smirl_rew_scale)
        if (self._add_true_rew == "only"):
            rew = env_rew
        elif self._add_true_rew:
            rew = (rew) + env_rew
            
        info['surprise'] = surprise
        info["alpha"] = self.alpha_t
        
        # update surprise momentum
        surpirse_change = [1 if self.surprise_window[i+1] > self.surprise_window[i] else -1 for i in range(len(self.surprise_window)-1)]
        self.alpha_t =  1 if np.sign(sum(surpirse_change)) > 0 else 0
        
        # augment next state
        obs = self.get_obs(obs)
        
        self._num_steps += 1
        
        return obs, rew, self.get_done() or envdone, info

    def get_done(self):
        return self.num_steps >= self._time_horizon

    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params, time-step, current surprise momentum.
        '''
        theta = self._buffer.get_params()
        num_samples = np.ones(1)*self._buffer.buffer_size
        alpha_t = np.ones(1)*self.alpha_t

        if (self._obs_out_label is None):
            obs = np.concatenate([np.array(obs).flatten(), np.array(theta).flatten(), num_samples, alpha_t])
        else:
            obs[self._obs_out_label] = np.concatenate([np.array(theta).flatten(), num_samples, alpha_t])
        
        return obs

    #def get_done(self, env_done):
    def get_done(self):
        '''
        figure out if we're done

        params
        ======
        env_done (bool) : done bool from the wrapped env, doesn't 
            necessarily need to be used
        '''
        return self._num_steps >= self._time_horizon

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        
        # reset the surprise window, which works at episode-level
        self.init_surprise_window()
        
        obs = self._env.reset()
#         print ("surprise obs shape1, ", obs.shape)
        self._buffer.reset()
        self._num_steps = 0
        obs = self.get_obs(obs)
#         print ("surprise obs shape2, ", obs.shape)
        return obs

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        if self._obs_label is None:
            return np.array(obs).flatten().copy()
        else:
            return np.array(obs[self._obs_label]).flatten().copy()
