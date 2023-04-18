from __future__ import annotations

from gym_minigrid.minigrid import *

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        fill_coords(r, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class SurpriseAdaptRoomsEnv(MiniGridEnv):

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
    """
    ## Description
    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.
    ## Mission Space
    "reach the goal"
    ## Action Space
    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    ## Observation Encoding
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    ## Rewards
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
    ## Termination
    The episode ends if any one of the following conditions is met:
    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).
    ## Registered Configurations
    - `MiniGrid-FourRooms-v0`
    """

    def __init__(self, max_steps=100, noisy_room=2, num_doors=1, noisy_prob=1,
                 agent_view="front", **kwargs):
        self._agent_default_pos = None
        self._goal_default_pos = None

        self.room_size = 8
        self.num_rooms = 3
        self.height = self.room_size + 2
        self.width = (self.room_size + 1) * self.num_rooms + 1
        assert noisy_room in (1, 2), "Please select one of (1,2) as the noisy room"
        self.noisy_room = noisy_room
        self.noisy_prob = noisy_prob

        self.agent_view = agent_view
        if self.agent_view == "center":
            see_through_walls = True
        else:
            see_through_walls = False
        self.num_doors = num_doors

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            **kwargs,
        )
        self.actions = SurpriseAdaptRoomsEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def step(self, action):
        self._randomize_floor_colors()
        obs, reward, done, info = super().step(action)
        info.update({'agent_pos': (self.agent_pos, (self.width, self.height))})

        return obs, reward, done, info

    def _randomize_floor_colors(self):
        color_names = [color for color in COLOR_NAMES if color != 'grey']

        for i in range((self.room_size + 1) * (self.noisy_room-1) + 1, (self.room_size + 1) * self.noisy_room):
            for j in range(1, self.room_size + 1):
                if np.random.uniform() < self.noisy_prob:
                    try:
                        floor_tile = Floor(color=self._rand_elem(color_names))
                        self.put_obj(floor_tile, i, j)
                    except Exception as e:
                        print(f'Could not place noisy floor tile: {e}')
                else:
                    self.grid.set(i, j, None)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # For each room
        for j in range(0, self.num_rooms-1):

            # For each column
            x = (j + 1) * (self.room_size + 1)

            # Wall and door
            self.grid.vert_wall(x, 1, self.room_size)
            doors = []
            for _ in range(self.num_doors):
                pos = (x, self._rand_int(1, self.room_size + 1))
                while pos in doors:
                    pos = (x, self._rand_int(1, self.room_size + 1))
                self.grid.set(*pos, None)
                doors.append(pos)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent(top=(1, 1), size=(self.room_size, self.room_size))

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal(), top=((self.room_size+1) * (self.num_rooms-1) + 1, 1), size=(self.room_size, self.room_size))

        self._randomize_floor_colors()

        self.mission = self._gen_mission()

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        if self.agent_view == 'center':
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1] - self.agent_view_size // 2
            botX = topX + self.agent_view_size
            botY = topY + self.agent_view_size

            return (topX, topY, botX, botY)
        else:
            return super().get_view_exts()

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        if self.agent_view == "center":
            agent_pos = (self.agent_view_size // 2, self.agent_view_size // 2)
        else:
            agent_pos = (self.agent_view_size // 2, self.agent_view_size - 1)
        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=agent_pos,
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        if self.agent_view == "center":
            top_left = self.agent_pos + f_vec * (self.agent_view_size // 2) - r_vec * (self.agent_view_size // 2)
        else:
            top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img
