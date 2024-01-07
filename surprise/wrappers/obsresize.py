import numpy as np
import gym
from gym.spaces import Box, Dict
import pdb
import cv2
import util.class_util as classu
import collections 


class FlattenObservationWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, env):
        '''
        params
        ======
        env (gym.Env) : environment to wrap
        '''
        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = Box(
                np.zeros(env.observation_space.low.size),
                np.ones(env.observation_space.low.size)
            )


    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        obs = self.encode_obs(obs)
        return obs, env_rew, envdone, info
    
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
        obs = self.encode_obs(obs)
        return obs
    
    def encode_obs(self, obs):
#         print ("obs keys: ", obs.keys())
        obs_ = np.array(obs).flatten()
#         print ("obs dict to obs: ", obs_.shape)
        return obs_
        
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
class DictToObservationWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, env, obs_keys=None, obs_size=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)
        

        self.num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        if (self._obs_size is None):
            self.observation_space = Box(
                np.concatenate([x.low.flatten() for x in self.env.observation_space.spaces.values()]),
                np.concatenate([x.high.flatten() for x in self.env.observation_space.spaces.values()])
            )
        else:
            self.observation_space = Box(
                np.zeros(self._obs_size),
                np.ones(self._obs_size)
            )


    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        obs = self.encode_obs(obs)
        return obs, env_rew, envdone, info
    
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
        obs = self.encode_obs(obs)
        return obs
    
    def encode_obs(self, obs):
#         print ("obs keys: ", obs.keys())
        # print(f"obs: {obs.shape}")
        # print(f"self._obs_keys:{self._obs_keys}")
        # input()
        obs_ = np.concatenate([ np.array(obs[x]).flatten() for x in self._obs_keys])
#         print ("obs dict to obs: ", obs_.shape)
        return obs_
        
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
class DictObservationWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, env, obs_key=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)

        self.num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        self.observation_space = Dict({self._obs_key: env.observation_space})
        print(self.env_obs_space)


    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        obs = {self._obs_key: obs}
        return obs, env_rew, envdone, info
    
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
        obs = {self._obs_key: obs}
#         print("wrapped dict observation: ", obs)
        return obs
    
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
    
class ResizeObservationWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, env, new_size=(48,64,3), new_shape=(64,48,3), grayscale=False, 
                 out_key_name=None, obs_key=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)

        self.num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        self.observation_space = Box(
                np.zeros(self._new_shape),
                np.ones(self._new_shape)
            )


    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        obs_ = self.resize_obs(obs)
        return obs_, env_rew, envdone, info
    
    def resize_obs(self, obs, key=None):
        if self._obs_key is not None:
#             print ("obs: ", obs)
            obs_ = obs
            obs = obs_[self._obs_key]
#         print("dsize: ", self._new_size[:2], " obs shape: ", obs.shape)
        obs = cv2.resize(obs, dsize=tuple(self._new_size[:2]), interpolation=cv2.INTER_AREA)
#         print("obs2 resize: ", obs.shape)
        if (self._grayscale):
            obs = np.mean(obs, axis=-1, keepdims=True)
#         print("obs3 resize: ", obs.shape)
        
        if (self._out_key_name is not None):
            obs_[self._out_key_name] = obs
            obs = obs_
        elif self._obs_key is not None:
#             print ("obs: ", obs)
            obs_[self._obs_key] = obs
            obs = obs_
        return obs
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
#         print ("obs: ", obs)
        obs_ = self.resize_obs(obs)
        return obs_
    
    def render(self, mode=None):
        return self._env.render(mode=mode)
    
    
class ChannelFirstWrapper(gym.Wrapper):
    def __init__(self, env, swap=(2,0)):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)

        self.swap = swap

        self.num_steps = 0

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space

        self.observation_space = Box(
                np.moveaxis(np.zeros(self.env.observation_space.low.shape), *self.swap),
                np.moveaxis(np.ones(self.env.observation_space.low.shape), *self.swap)
            )

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self.env.step(action)
        obs = self.resize_obs(obs)
        return obs, env_rew, envdone, info
    
    def resize_obs(self, obs):
        import numpy as np
#         print("obs move channel: ", obs.shape)
        obs = np.moveaxis(obs, *self.swap)
#         print("obs2 move channel: ", obs.shape)
        return obs
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self.env.reset()
        obs = self.resize_obs(obs)
        return obs


class RenderingObservationWrapper(gym.Wrapper):
    
    @classu.hidden_member_initialize
    def __init__(self, env, swap=None, rescale=None, resize=None, render_agent_obs=None,
                 resize_agent_obs=None, rescale_agent_obs=None, permute_agent_obs=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # Take Action
        import copy
        import numpy as np

        obs, env_rew, envdone, info = self._env.step(action)
        render_obs = self._env.render(mode="rgb_array")

        if (self._resize is not None):
            render_obs = cv2.resize(render_obs, dsize=tuple(self._resize[:2]), interpolation=cv2.INTER_AREA)
        if self._rescale is not None:
            render_obs = np.array(render_obs * self._rescale, dtype='uint8')
        if self._swap is not None:
            render_obs = copy.deepcopy(np.moveaxis(render_obs, *self._swap))

        new_obs = obs
        if self._render_agent_obs is not None and self._render_agent_obs:
            if self._permute_agent_obs:
                new_obs = np.transpose(new_obs, self._permute_agent_obs)
            if (self._resize_agent_obs is not None):
                new_obs = cv2.resize(new_obs, dsize=tuple(self._resize_agent_obs[:2]), interpolation=cv2.INTER_AREA)
            x, y, z = new_obs.shape
            agent_obs = np.zeros((x, render_obs.shape[1], z))
            agent_obs[0:x, 0:y, 0:z] = new_obs
            if self._rescale_agent_obs is not None:
                agent_obs = np.array(agent_obs * self._rescale_agent_obs, dtype='uint8')
            render_obs = np.concatenate([render_obs, agent_obs], axis=0)
        info['rendering'] = render_obs
        return obs, env_rew, envdone, info
    

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
        return obs
    
    def render(self, mode=None):
        
        return self._env.render(mode=mode)
    


class SoftResetWrapper(gym.Wrapper):
    
    def __init__(self, env, max_time):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)
        
        self._env = env
        self._time = 0
        self._max_time = max_time

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.reset_alpha = True

        self.deaths = 0
        self.total_task_returns = 0
        self.returns = None
        self.discount_rate = 1

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        self._time += 1
        self.returns = self.returns + env_rew

        info['avg_task_returns']= (self.returns + self.total_task_returns)/(self.deaths + 1)

        info["life_length_avg"] = self._last_death
        if (envdone):
            self.reset_alpha = False
            self.total_task_returns += self.returns
            self.returns = 0
            self.deaths += 1
            obs_ = self.reset()
            ### Trick to make "death" more surprising...
#             info["life_length"] = self._last_death
            info["death"] = 1
            self._last_death = 0
            obs = np.random.rand(*obs_.shape)
        else:
            info["death"] = 0
        
        self._last_death = self._last_death + 1
        envdone = self._time >= self._max_time
        if envdone:
            self._time = 0

        return obs, env_rew, envdone, info

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        self._last_death = 0
        if self.reset_alpha:
            self.alpha_t = np.random.binomial(1, 0.5)
        else:
            self.reset_alpha = True

        obs = self._env.reset()
        self.returns = 0
        return obs
    
    def render(self, mode=None):
#         print ("self._env: ", self, self._env, self._env.render, mode, self._env.render(mode=mode).shape)
        return self._env.render(mode=mode)

    def set_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate
    
class ObsHistoryWrapper(gym.Wrapper):

    @classu.hidden_member_initialize
    def __init__(self, env, history_length=3, 
                 stack_channels=False, channel_dim=2, obs_key=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        super().__init__(env)
        
        # Gym spaces
        self.action_space = env.action_space
        self.observation_space_old = env.observation_space
        if self._stack_channels:
            shape_ = list(env.observation_space.low.shape)
            shape_[self._channel_dim] = shape_[self._channel_dim] * self._history_length 
            self.observation_space = Box(0, 1, shape=shape_ )
            
        else:
            self.observation_space = Box(-1, 1, shape=(env.observation_space.low.shape[0]*self._history_length,) )

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        
        if (self._obs_key is None):
            self.obs_hist.appendleft(obs)
        else:
            self.obs_hist.appendleft(obs[self._obs_key])
        self.obs_hist.pop()
        
        return self.get_obs(obs), env_rew, envdone, info
    
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        self._time = 0
        obs = self._env.reset()
#         print(" obs stack obs shape: ", obs.shape)
        self.obs_hist = collections.deque([np.zeros(shape=self.observation_space_old.low.shape) for _ in range(self._history_length)])
#         print("self.obs_hist shape: ", np.array(self.obs_hist).shape)
        if (self._obs_key is None):
            self.obs_hist.appendleft(obs)
        else:
            self.obs_hist.appendleft(obs[self._obs_key])
        self.obs_hist.pop()
        return self.get_obs(obs)
    
    def get_obs(self, obs):
        
        if self._stack_channels:
#             print("self.obs_hist shape: ", np.array(self.obs_hist[0]).shape)
#             print("self.obs_hist shape: ", np.array(self.obs_hist[1]).shape)
#             import matplotlib.pyplot as plt
#             print ("obs: ", obs)
#             plt.imshow(np.reshape(self.obs_hist[0], (64,48)))
#             plt.show()
            obs_ =  np.concatenate(self.obs_hist, axis=-1)
        else:
            obs_ =  np.array(self.obs_hist).flatten()
        
        if (self._obs_key is None):
            obs = obs_
        else:
            obs[self._obs_key] = obs_
            
        return obs
        
        
    def render(self, mode=None):
        return self._env.render(mode=mode)


from gym.wrappers import TransformObservation


class AddAlphaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.alpha_t = None

class MazeEnvOneMaskObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Box(0, 3, shape=(16, 14))
        self.action_space = env.action_space
    def step(self, action):
        # Take Action
        obs, rew, done, info = super().step(action)
        obs = self.ToOneMask(obs)
        return obs, rew, done, info
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = super().reset()
        obs = self.ToOneMask(obs)
        return obs

    def ToOneMask(self, obs):
        # get goal position
        x, y = np.unravel_index(np.argmax(obs[0], axis=None), obs[0].shape)
        
        # tranform to one channel only with indexes
        obs = np.argmax(obs, axis=0)

        # manually set goal position
        max_num = np.max(obs)
        obs[x, y] = max_num + 1
        return obs
    
class StrictOneHotWrapper(gym.Wrapper):
    def __init__(self, env, channel_first=True):
        """Enforces strict one-hot (i.e. only one channel can be active at a time))"""
        super().__init__(env)
        self.channel_first = channel_first
        if self.channel_first:
            self.num_channel = env.observation_space.shape[0]
            obs_shape = env.observation_space.shape[1:]
        else:
            self.num_channel = env.observation_space.shape[-1]
            obs_shape = env.observation_space.shape[0:-1]


    def step(self, action):
        # Take Action
        obs, rew, done, info = super().step(action)
        obs = self.to_one_hot(self.to_categorical(obs))
        return obs, rew, done, info
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = super().reset()
        obs = self.to_one_hot(self.to_categorical(obs))
        return obs

    def to_categorical(self, obs):
        
        # tranform to one channel only with indexes
        obs = np.amax(
                obs * np.reshape(np.arange(self.num_channel) + 1, (1,1,-1)), 2).astype(int)

        return obs
    
    def to_one_hot(self, obs):
        m,n = obs.shape
        I,J = np.ogrid[:m,:n]

        if self.channel_first:
            out = np.zeros((self.num_channel + 1, m, n), dtype=int)
            out[obs, I, J] = 1
            out = out[:-1,:,:]
        else:
            out = np.zeros((m, n, self.num_channel + 1), dtype=int)
            out[I, J, obs] = 1
            out = out[:,:,:-1]
        return out


class RescaleImageWrapper(TransformObservation):
    def __init__(self, env):
        super().__init__(env, self._rgb_rescale)

    @staticmethod
    def _rgb_rescale(x):
        return x/255


class FlattenDictObservationWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        '''
        params
        ======
        env (gym.Env) : environment to wrap
        '''
        self._obs_keys = [key for key in self.env.observation_space.spaces.keys()]
        self.unflattened_observation_space = self.observation_space
        self.observation_space = Box(
            np.concatenate([x.low.flatten() for x in self.env.observation_space.spaces.values()]),
            np.concatenate([x.high.flatten() for x in self.env.observation_space.spaces.values()])
        )

    def step(self, action):
        # Take Action
        obs, rew, done, info = super().step(action)
        obs = self.encode_obs(obs)
        return obs, rew, done, info

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = super().reset()
        obs = self.encode_obs(obs)
        return obs

    def encode_obs(self, obs):
        obs_ = np.concatenate([np.array(obs[x]).flatten() for x in self._obs_keys])
        return obs_

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)


from PIL import Image, ImageDraw, ImageFont

class AddTextInfoToRendering(gym.Wrapper):

    def __init__(self, env, log_returns=False, log_alphas=False, text_color=(255, 255, 255), text_pos=None, font_size=8, font_gap=10):
        super().__init__(env)
        self.returns = 0 if log_returns else None
        self.log_alphas = log_alphas
        self.text_color = text_color
        self.text_pos = text_pos
        self.font_size = font_size
        self.font_gap = font_gap

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.returns is not None:
            self.returns += rew
        render_obs = info['rendering']
        import os
        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=self.font_size)

        im = Image.fromarray(render_obs)
        draw = ImageDraw.Draw(im)

        if self.text_pos is None:
            x = int(render_obs.shape[1] / 2)
            y = int(render_obs.shape[0] / 2)
        else:
            x, y = self.text_pos

        draw.text((x, y), f"rew: {rew:.2f}", fill=tuple(self.text_color), font=font)
        
        if self.returns is not None:
            draw = ImageDraw.Draw(im)
            draw.text((x, y+self.font_gap), f"ret: {self.returns:.2f}", fill=tuple(self.text_color), font=font)
        
        if self.log_alphas:
            draw = ImageDraw.Draw(im)
            draw.text((x, y+(self.font_gap*2)), f"alpha: {self.env.alpha_t * 1}", fill=tuple(self.text_color), font=font)
        
        info['rendering'] = np.array(im)
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        if self.returns is not None:
            self.returns = 0
        return obs
