import gym

class MazeEnvFullyObserved(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnvFullyObserved, self).__init__()

    def step(self, action):
        state, rew, done, info = self.env.step(action)
        return state, rew, done, {}

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def set_env(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
