# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from IPython import embed
from gymnasium_wrappers.utils import *
from gymnasium_wrappers.models import *
from gymnasium_wrappers.args import parse_args_random


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
        
    args, run_name = parse_args_random()
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if "crafter" in args.env_id:
        crafter_logger = CrafterLogger()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    use_theta = args.model in ["smax", "smin", "sadapt", "sadapt-inverse", "sadapt-bandit"]

    logger_ = make_csv_logger(f"runs/{run_name}/log.csv")
        
    start_time = time.time()

    ep_surprise = []
    ep_entropy = []
    task_rewards = []
    ep_counter = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        if "surprise" in infos:
            ep_surprise.append(infos["surprise"][0])
            ep_entropy.append(infos["theta_entropy"][0])
        if "task_reward" in infos:
            task_rewards.append(infos["task_reward"][0])

                
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # update crafter logs
                if "crafter" in args.env_id:
                    crafter_logger.update_achievements(info["achievements"])
                    crafter_logger.log(writer, global_step)
                    
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_surprise={np.mean(ep_surprise)}, episodic_entropy={np.mean(ep_entropy)}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_surprise", np.mean(ep_surprise), global_step)
                writer.add_scalar("charts/episodic_entropy", np.mean(ep_entropy), global_step)
                writer.add_scalar("charts/task_reward", np.mean(task_rewards), global_step)
                writer.add_scalar("charts/average_task_return", info["Average_task_return"], global_step)
                writer.add_scalar("charts/average_episode_length", info["Average_episode_length"], global_step)
                # log bandit metrics
                if args.model == "sadapt-bandit":
                    writer.add_scalar("charts/alpha_rolling_average", info["alpha_rolling_average"], global_step)
                    if "ucb_alpha_one" in info:
                        writer.add_scalar("charts/ucb_alpha_one", info["ucb_alpha_one"], global_step)
                        writer.add_scalar("charts/ucb_alpha_zero", info["ucb_alpha_zero"], global_step)
                writer.add_scalar("charts/deaths", info["deaths"], global_step)
                logger_.logs_a([
                    global_step,
                    info["episode"]["r"][0],
                    info["episode"]["l"][0],
                    np.mean(ep_surprise),
                    np.mean(ep_entropy),
                ])
                
                if ep_counter % 1000 == 0 and "Rooms" in args.env_id:
                    log_heatmap(envs.envs[0], infos["heatmap"][0], ep_counter, writer, f"runs/{run_name}")
                
                ep_surprise.clear()
                ep_entropy.clear()
                task_rewards.clear()
                ep_counter += 1

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
    envs.close()
    writer.close()