import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import pdb
import util.class_util as classu
from collections import deque


class BaseSurpriseAdaptWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, 
                 env, 
                 buffer, 
                 surprise_window_len,
                 surprise_change_threshold=0.0,
                 momentum=False,
                 add_true_rew=False,
                 int_rew_scale=1,
                ):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)

        self._env = env
        self.buffer = buffer
        theta = self.buffer.get_params()
        
        self.num_steps = 0
        self.add_true_rew = add_true_rew
        self.int_rew_scale = int_rew_scale
        self.surprise_window_len = surprise_window_len
        self.momentum = momentum

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space

        # adding theta and t for consistent MDP (same as SMiRL)
        # adding history of surprise (either -1 or 1)

        self.observation_space = Dict({
            "obs": Box(-np.inf, np.inf, shape=(env.observation_space.shape)),
            "theta": Box(-np.inf, np.inf, shape=(np.prod(theta.shape) + 2,)),
        })

        self.surprise_change_threshold = surprise_change_threshold

        try:
            self.heatmap = np.zeros((env.width, env.height))
        except:
            self.heatmap = None
        
        print(self.observation_space)
        

    def init_surprise_window(self):
        self.surprise_counter = 0
        self.surprise_window = deque()

    def step(self, action):
        obs, env_rew, envdone, envtrunc, info = self._env.step(action)
        info['task_reward'] = env_rew
        info["alpha"] = self.alpha_t

        # Compute surprise as the negative log probability of the observation
        surprise = - self.buffer.logprob(obs)
        
        # For numerical stability, clip stds to not be 0
        thresh = 300
        surprise = np.clip(surprise, a_min=-thresh, a_max=thresh) / thresh
        
        rew = ((-1)**self.alpha_t) * surprise
        
        # remove old elements from the surprise window
        if self.surprise_counter > self.surprise_window_len:
            try:
                self.surprise_window.popleft()
            except:
                pass
        
        self.surprise_window.append(surprise)
        
        self.buffer.add(obs)
        
        if self.add_true_rew:
            rew = env_rew + (rew * self.int_rew_scale)
        else:
            rew = rew * self.int_rew_scale
            
        info['surprise'] = surprise
        info["theta_entropy"] = self.buffer.entropy()
        info["alpha"] = self.alpha_t
        
        # update surprise momentum
        surprise_change = [0 if (np.abs(self.surprise_window[i+1]-self.surprise_window[i])/np.abs(self.surprise_window[i])
                                    < self.surprise_change_threshold)
                                    & (np.sign(self.surprise_window[i+1]) == np.sign(self.surprise_window[i]))
                            else 1 if self.surprise_window[i+1] > self.surprise_window[i] else -1
                            for i in range(len(self.surprise_window)-1)]
        if sum(surprise_change) == 0:
            self.alpha_t = (np.random.rand() < 0.5) * 1
        elif self.momentum:
            self.alpha_t = 0 if np.sign(sum(surprise_change)) > 0 else 1
        else:
            self.alpha_t = 1 if np.sign(sum(surprise_change)) > 0 else 0

        try:
            x, y = self._env.agent_pos
            self.heatmap[x, y] += 1
            info["heatmap"] = self.heatmap.copy()
        except:
            pass

        self.surprise_counter += 1
        self.num_steps += 1

        # soft reset
        if envdone:
            obs, _ = self._env.reset()

        if self.num_steps == self._env.max_episode_steps:
            envdone = True
            envtrunc = False
        else:
            envdone = False
            envtrunc = False

        return self.get_obs(obs), rew, envdone, envtrunc, info

    def get_obs(self, obs):
        theta = self.buffer.get_params()
        num_samples = np.ones(1)*self.buffer.buffer_size
        alpha_t = np.ones(1)*self.alpha_t

        obs = {
            "obs" : obs
        }
        obs["theta"] = np.concatenate([np.array(theta).flatten(), num_samples, alpha_t])
        return obs

    def reset(self, seed = None, options = None):
        self.alpha_t = np.random.binomial(1, 0.5)
        self.init_surprise_window()
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