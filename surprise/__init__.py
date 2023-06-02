
from gym.envs.registration import register as gym_register

gym_register(
    id='MiniGrid-MaxwellsDemon-v0',
    entry_point='surprise.envs.minigrid.envs.maxwells_demon_room:MaxwellsDemonEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
)

gym_register(
    id='MiniGrid-SimpleEnemyEnvHMMMarginal-v0',
    entry_point='surprise.envs.minigrid.envs.simple_room_hmm_marginal:SimpleEnemyEnvHMMMarginal',
    reward_threshold=0.95,
    max_episode_steps=5000,
)

gym_register(
    id='MiniGrid-SimpleEnemyTagEnvHMM-v0',
    entry_point='surprise.envs.minigrid.envs.simple_room_tag_hmm:SimpleEnemyTagEnvHMM',
    reward_threshold=0.95,
    max_episode_steps=5000,
)

gym_register(
    id='MiniGrid-SimpleEnemyTAgEnvHMMMarginal-v0',
    entry_point='surprise.envs.minigrid.envs.simple_room_hmm_tag_marginal:SimpleEnemyEnvTagHMMMarginal',
    reward_threshold=0.95,
    max_episode_steps=5000,
)


# Surprise Adapt Rooms

gym_register(
    id='MiniGrid-SurpriseAdaptRooms-v0',
    entry_point='surprise.envs.minigrid.envs.surprise_adapt_rooms:SurpriseAdaptRoomsEnv'
)

from griddly import GymWrapperFactory, gd
import os
wrapper = GymWrapperFactory()
wrapper.build_gym_from_yaml('_ButterfliesEnv', "/home/roger/Desktop/surprise-adaptive-agents/surprise/envs/maze/butterflies.yaml") #f"{os.getcwd()}/surprise/envs/maze/butterflies.yaml")
gym_register(
    id='GDY-ButterfliesEnv-v0',
    entry_point='surprise.envs.maze.butterflies:ButterfliesEnv'
)
