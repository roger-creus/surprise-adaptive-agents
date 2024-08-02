
from gym.envs.registration import register as gym_register

from griddly import GymWrapperFactory, gd
import os
wrapper = GymWrapperFactory()
wrapper.build_gym_from_yaml('_ButterfliesEnv', "/home/roger/Desktop/surprise-adaptive-agents/surprise/envs/maze/butterflies.yaml") #f"{os.getcwd()}/surprise/envs/maze/butterflies.yaml")
gym_register(
    id='GDY-ButterfliesEnv-v0',
    entry_point='surprise.envs.maze.butterflies:ButterfliesEnv'
)
