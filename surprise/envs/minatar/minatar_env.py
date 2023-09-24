from minatar.gym import BaseEnv
import seaborn as sns
import gym
import numpy as np


class MinAtarEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        super(MinAtarEnv, self).__init__(*args, **kwargs)

    def render(self, mode):
        self.render_mode = mode
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "array":
            return self.game.state()
        elif self.render_mode == "human":
            self.game.display_state(self.display_time)
        elif self.render_mode == "rgb_array": # use the same color palette of Environment.display_state
            state = self.game.state()
            n_channels = state.shape[-1]
            cmap = sns.color_palette("cubehelix", n_channels)
            cmap.insert(0, (0,0,0))
            numerical_state = np.amax(
                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
            rgb_array = np.stack(cmap)[numerical_state]
            return rgb_array