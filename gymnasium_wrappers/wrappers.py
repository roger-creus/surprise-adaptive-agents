import gymnasium as gym
import numpy as np
from typing import Tuple
from IPython import embed

class ImageTranspose(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space.shape
        dtype = np.float32
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def observation(self, observation):
        observation= np.transpose(observation, axes=[2, 0, 1]) * 1.0
        return observation


class GrafterStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        gym.Wrapper.__init__(self, env)
        
        self.inv_variables = [
            "inv_wood_sword",
            "inv_stone_sword",
            "inv_iron_sword",
            "inv_wood_pickaxe",
            "inv_stone_pickaxe",
            "inv_iron_pickaxe",
            "inv_sapling",
            "inv_stone",
            "inv_coal",
            "inv_wood",
            "inv_iron",
            "inv_diamond",
            "inv_food",
            "inv_drink",
            "inv_energy",
            "health",
        ]

        self.observation_space = gym.spaces.Dict({
            "inv" : gym.spaces.Box(low=0, high=64, shape=(len(self.inv_variables),), dtype=np.float32),
            "obs" : env.observation_space,
        })

    def reset(self):
        obs, info = self.env.reset()
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._get_obs(obs), reward, term, trunc, info

    def _get_obs(self, obs):
        inv = self.env._env.game.get_global_variable(self.inv_variables)
        inv = np.array([inv[key][0] for key in self.inv_variables])
        return {"inv": inv, "obs": obs}