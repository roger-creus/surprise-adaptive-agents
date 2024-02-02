import gymnasium as gym
import gym as old_gym
import csv_logger
import logging
import matplotlib.pyplot as plt
import minatar
import torch
import crafter
import numpy as np
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, OneHotPartialObsWrapper
from gymnasium_wrappers.base_surprise import BaseSurpriseWrapper
from gymnasium_wrappers.base_sadapt import BaseSurpriseAdaptWrapper
from gymnasium_wrappers.base_surprise_adapt_bandit import BaseSurpriseAdaptBanditWrapper
from gymnasium_wrappers.gym_to_gymnasium import GymToGymnasium
from gymnasium_wrappers.obs_resize import ResizeObservationWrapper
from gymnasium_wrappers.obs_history import ObsHistoryWrapper
from gymnasium_wrappers.rendering_wrapper import RenderObservationWrapper
from surprise.buffers.buffers import GaussianBufferIncremental, BernoulliBuffer, MultinoulliBuffer
from griddly import GymWrapperFactory, gd
import os

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
        theta_size = None
        grayscale = None
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
            max_steps = 500

        elif "crafter" in args.env_id:
            max_steps = 500
            grayscale = True
            channel_dim = 1 if grayscale else 3
            env = old_gym.make('CrafterReward-v1')
            # Crafter is based on old gym, we need to convert it to gymnasium api
            env = GymToGymnasium(env, render_mode="rgb_array", max_steps=max_steps)
            # resize the observation
            env = ResizeObservationWrapper(env, grayscale=grayscale)
            # stack multiple frames
            env = ObsHistoryWrapper(env, history_length=3, stack_channels=True, channel_dim=2)
            # set the size of theta
            theta_size = (20, 26, channel_dim) if grayscale else (20, 26, channel_dim)
        elif "FourRooms" in args.env_id:
            env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array', max_steps=500)
            env = OneHotPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            max_steps = 500
            
        elif "MinAtar" in args.env_id:
            env = gym.make(args.env_id+"-v1", render_mode='rgb_array', max_episode_steps=500)
            max_steps = 500

        elif "griddly" in args.env_id:
            # for instance griddly-MazeEnv
            register_griddly_envs()
            griddly_env_name = args.env_id.split('-')[-1]
            max_steps = 250
            env = old_gym.make(f"GDY-{griddly_env_name}-v0", player_observer_type=gd.ObserverType.VECTOR, global_observer_type=gd.ObserverType.VECTOR)
            
            if griddly_env_name == "MazeEnv":
                from surprise.envs.maze.maze_env import MazeEnv
                env = MazeEnv(env)
                env = old_gym.wrappers.FlattenObservation(env)
            elif griddly_env_name == "ButterfliesEnv":
                print("ButterfliesEnv")
                print(env.observation_space.sample().shape)
                quit()
                from surprise.envs.maze.butterflies_latest import ButterfliesEnv
                env = ButterfliesEnv(env)
                env = old_gym.wrappers.FlattenObservation(env)
            else:
                raise ValueError(f"Unknown griddly env {griddly_env_name}")
            
            env = GymToGymnasium(env, render_mode="rgb_array", max_steps=max_steps)
        
        else:
            print(f"Making {args.env_id}")
            env = gym.make(args.env_id, render_mode='rgb_array', max_episode_steps = 500)
            max_steps = 500
        
        ############ Create buffer ############
        obs_size = np.prod(theta_size) if theta_size else env.observation_space.shape
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
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset
            )
        
        elif args.model == "smin":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=args.add_true_rew,
                minimize=True,
                int_rew_scale=1.0,
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset
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
        elif args.model == "sadapt-bandit":
            env = BaseSurpriseAdaptBanditWrapper(
                env, 
                buffer,
                add_true_rew=args.add_true_rew,
                int_rew_scale=1.0,
                max_steps = max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset
            )
        elif args.model == "none":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=True,
                minimize=False,
                int_rew_scale=0.0,
                ext_only=True,
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset
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
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('MazeEnv', f"{os.getcwd()}/surprise/envs/maze/maze_env_fully_observed.yaml")
    wrapper.build_gym_from_yaml('ButterfliesEnv', f"{os.getcwd()}/surprise/envs/maze/butterflies_latest.yaml")

class CrafterLogger:
    '''
    Helper class to keep track of crafter related metrics
    '''
    def __init__(self):
        # Initialize the achievements dict
        self.achievements = {}

    def update_achievements(self, achievements):
        for k,v in achievements.items():
            if not k in self.achievements:
                self.achievements[k] = []
            self.achievements[k].append(v)

    def compute_success_rates(self):
        success_rate = {}
        for k,v in self.achievements.items():
            num_episodes = len(self.achievements[k])
            success_rate[k] = 100 * (np.array(v)>=1).mean() 
        return success_rate

    def compute_crafter_score(self):
        success_rates_values = np.array(list(self.compute_success_rates().values()))
        assert (0 <= success_rates_values).all() and (success_rates_values <= 100).all()
        scores = np.exp(np.nanmean(np.log(1 + success_rates_values), -1)) - 1
        return scores

    def log(self, writer, global_step):
        success_rates = self.compute_success_rates()
        for k,v in success_rates.items():
            key = f"crafter/{k}_success_rates"
            writer.add_scalar(key, v, global_step)
        score = self.compute_crafter_score()
        writer.add_scalar("crafter/score", score, global_step)

def eval_episode_ppo(ppo_agent, env, device, save_path, global_step):
    '''
    Evaluate a PPO agent in an environment and record and save a video
    '''
    # Reset the environment
    o_, _ = env.reset()
    next_obs = torch.Tensor(o_["obs"]).to(device)
    next_theta = torch.Tensor(o_["theta"]).to(device)
    ep_images = []

    while True:
        try:
            ep_images.append(env.envs[0].env.render(mode="rgb_array"))
        except:
            # for crafter
            ep_images.append(env.envs[0].render())

        with torch.no_grad():
            action, logprob, _, value = ppo_agent.get_action_and_value({
                "obs" : next_obs,
                "theta" : next_theta
            })

        o_, reward, done, trunc, infos = env.step(action.cpu().numpy())
        next_obs, next_theta = torch.Tensor(o_["obs"]).to(device), torch.Tensor(o_["theta"]).to(device)

        if done:
            break

    # save gif with all imags
    from PIL import Image
    ep_images = [Image.fromarray(img) for img in ep_images]
    ep_images[0].save(f"{save_path}/episode_{global_step}.gif", save_all=True, append_images=ep_images[1:], optimize=False, duration=40, loop=0)


def eval_episode_dqn(q_net, env, device, save_path, global_step, env_id="none"):
    '''
    Evaluate a DQN agent in an environment and record and save a video
    '''
    # Reset the environment
    obs, _ = env.reset()
    ep_images = []

    while True:
        try:
            ep_images.append(env.envs[0].env.render(mode="rgb_array"))
        except:
            # for crafter
            ep_images.append(env.envs[0].render())

        if isinstance(env.single_observation_space, gym.spaces.Dict):
            obs_ = {k: torch.as_tensor(v).to(device) for k, v in obs.items()}
        else:
            obs_ = torch.Tensor(obs).to(device)

        with torch.no_grad():
            q_values = q_net(obs_)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        obs = next_obs
        done = terminated or truncated
        
        if done:
            break

    # save gif with all imags
    from PIL import Image
    if "MinAtar" in env_id:
        ep_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in ep_images]
    else:
        ep_images = [Image.fromarray(img) for img in ep_images]
    ep_images[0].save(f"{save_path}/episode_{global_step}.gif", save_all=True, append_images=ep_images[1:], optimize=False, duration=40, loop=0)