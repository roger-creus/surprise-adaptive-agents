import gym
from griddly import GymWrapperFactory, gd
import matplotlib.pyplot as plt
import numpy as np

from IPython import embed

if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    wrapper.build_gym_from_yaml('MazeEnvFullyObserved', '/home/roger/Desktop/surprise-adaptive-agents/surprise/envs/maze/maze_env_fully_observed.yaml')

    env = gym.make(
        'GDY-MazeEnvFullyObserved-v0',
        player_observer_type=gd.ObserverType.VECTOR,
    )
    obs = env.reset()

    obs = env.render(mode="rgb_array")

    plt.imshow(obs)
    plt.savefig("lol.png")

    env.close()