import gymnasium as gym
import csv_logger
import logging
import matplotlib.pyplot as plt
import minatar

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, OneHotPartialObsWrapper
from gymnasium_wrappers.base_surprise import BaseSurpriseWrapper
from gymnasium_wrappers.base_sadapt import BaseSurpriseAdaptWrapper
from gymnasium_wrappers.base_surprise_fixed_timesteps import BaseSurpriseFixedTimeStepsWrapper, MountainCarSurpriseWrapper
from surprise.buffers.buffers import GaussianBufferIncremental, BernoulliBuffer, MultinoulliBuffer

from IPython import embed
from gymnasium.envs.registration import register as gym_register


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def make_env(args):
    def thunk():
        ############ Create environment ############
        if "Adapt" in args.env_id:
            gym_register(
                id='SurpriseAdaptRooms-v0',
                entry_point='surprise.envs.minigrid.envs.surprise_adapt_rooms:SurpriseAdaptRoomsEnv'
            )
            
            env = gym.make(args.env_id, render_mode='rgb_array', max_steps=500, noisy_room=args.noisy_room)
            env = ImgObsWrapper(env)
            max_steps = 500
            
        elif "tetris" in args.env_id:
            from surprise.envs.tetris.tetris import TetrisEnv
            env = TetrisEnv()
            max_steps = 100
            
        elif "FourRooms" in args.env_id:
            env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array', max_steps=500)
            env = OneHotPartialObsWrapper(env)
            env = ImgObsWrapper(env)

            max_steps = 500
            
        elif "MinAtar" in args.env_id:
            env = gym.make(args.env_id+"-v1", render_mode='rgb_array', max_episode_steps=1000)
            max_steps = 1000

        elif "griddly" in args.env_id:
            register_griddly_envs()
            env = gym.make("GDY-MazeEnv-v0")
        
        elif "MountainCar" in args.env_id:
            max_steps = 500
            # TODO: add functions to compute the height and velocity 
            env = gym.make("MountainCar-v0", render_mode="rgb_array")
            obs_size = env.observation_space.shape
            buffer = GaussianBufferIncremental(obs_size)
            env = MountainCarSurpriseWrapper(env, buffer, max_steps=max_steps)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(args.seed)
            return env
        else:
            print(f"Making {args.env_id}")
            env = gym.make(args.env_id, render_mode='rgb_array', max_episode_steps = 500)
            max_steps = 500
        
        ############ Create buffer ############
        
        if args.buffer_type == "gaussian":
            buffer = GaussianBufferIncremental(obs_size)
        elif args.buffer_type == "bernoulli":
            buffer = BernoulliBuffer(obs_size)
        elif args.buffer_type == "multinoulli":
            buffer = MultinoulliBuffer(obs_size)
        else:
            raise ValueError(f"Unknown buffer type {args.buffer_type}")
        
        if args.model == "smax":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=args.add_true_rew,
                minimize=False,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        
        elif args.model == "smin":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=args.add_true_rew,
                minimize=True,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        
        elif args.model == "sadapt":
            env = BaseSurpriseAdaptWrapper(
                env,
                buffer,
                surprise_window_len=args.surprise_window_len,
                surprise_change_threshold=args.surprise_change_threshold,
                momentum=True,
                add_true_rew=args.add_true_rew,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        
        elif args.model == "sadapt-inverse":
            env = BaseSurpriseAdaptWrapper(
                env,
                buffer,
                surprise_window_len=args.surprise_window_len,
                surprise_change_threshold=args.surprise_change_threshold,
                momentum=False,
                add_true_rew=args.add_true_rew,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        
        elif args.model == "none":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=True,
                minimize=False,
                int_rew_scale=0.0,
                ext_only=True,
                max_steps=max_steps
            )
            
        else:
            raise ValueError(f"Unknown model {args.model}")
                
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed)
        return env
    return thunk

def make_csv_logger(csv_path):
    header = [
        'date',
        'env_steps',
        'ep_return',
        "ep_length",
        "ep_surprise",
        "ep_entropy"
    ]
    log_level = ['logs_a']
    logger_ = csv_logger.CsvLogger(
        filename=csv_path,
        delimiter=',',
        level=logging.INFO,
        add_level_names=log_level,
        max_size=1e+9,
        add_level_nums=None,
        header=header,
    )
    return logger_


def log_heatmap(env, heatmap, ep_counter, writer, save_path):
    cmap = plt.get_cmap('Reds')
    cmap.set_under((0,0,0,0))
    cmap_args = dict(cmap=cmap)
    
    fig = plt.figure(num=1)
    #background_img = env.render()
    #background = cv2.resize(background_img, dsize=(env._env.width, env._env.height), interpolation=cv2.INTER_AREA)
    #plt.imshow(background.transpose(1,0,2) , alpha=0.75)
    plt.imshow(heatmap, **cmap_args, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(f"{save_path}/heatmap_{ep_counter}.png")
    writer.add_figure(f"trajectory/heatmap_{ep_counter}", fig, close=True)
    plt.clf()
    
def register_griddly_envs():
    from griddly import GymWrapperFactory, gd
    import os
    
    try:
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('_ButterfliesEnv', f"{os.getcwd()}/surprise/envs/maze/butterflies.yaml")
        gym_register(
            id='GDY-ButterfliesEnv-v0',
            entry_point='surprise.envs.maze.butterflies:ButterfliesEnv'
        )
    except:
        pass
    
    try:
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('_MazeEnv', f"{os.getcwd()}/surprise/envs/maze/maze_env_fully_observed.yaml")
        gym_register(
            id='GDY-MazeEnv-v0',
            entry_point='surprise.envs.maze.maze_env:MazeEnvFullyObserved'
        )
    except:
        pass


# This is taken from mountain car code
def compute_mountain_car(x_positon):
    return np.sin(3 * xs) * 0.45 + 0.55
