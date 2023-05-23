import gym
from griddly import GymWrapperFactory, gd
import matplotlib.pyplot as plt
import numpy as np

from IPython import embed

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    wrapper.build_gym_from_yaml('MazeEnvPartiallyObserved', '/home/roger/Desktop/surprise-adaptive-agents/surprise/envs/maze/maze_env_partially_observed.yaml')

    env = gym.make(
        'GDY-MazeEnvPartiallyObserved-v0',
        player_observer_type=gd.ObserverType.VECTOR,
    )
    obs = env.reset()

    obs, r, d, _ = env.step(env.action_space.sample())

    plt.imshow(obs.transpose(2,1,0))
    plt.savefig("lol.png")

    env.close()