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
from gymnasium_wrappers.args import parse_args_dqn
from gym.wrappers.normalize import RunningMeanStd


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
        
    args, run_name = parse_args_dqn()
    
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
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    use_theta = args.model in ["smax", "smin", "sadapt", "sadapt-inverse", "sadapt-bandit"]
    
    if args.env_id == "tetris":
        if "Butterflies" in args.env_id:
            net = TetrisBigQNetwork
        else:
            net = TetrisQNetwork
    elif "MinAtar" in args.env_id or "griddly" in args.env_id:
        net = MinAtarQNetwork
    elif "Atari" in args.env_id:
        net = AtariQNetwork
    else:
        raise NotImplementedError
    
    q_network = net(envs, use_theta=use_theta).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = net(envs, use_theta=use_theta).to(device)
    target_network.load_state_dict(q_network.state_dict())

    logger_ = make_csv_logger(f"runs/{run_name}/log.csv")

    print(q_network)

    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        rb = DictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        
    start_time = time.time()

    ep_surprise = []
    ep_entropy = []
    task_rewards = []
    entr_changes = []
    ep_counter = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    eval_episode_dqn(None, eval_envs, device, f"runs/{run_name}", 0, args.env_id, args.track, random=True)
    if args.normalize_int_reward:
        if "bandit" not in args.model:
            rms = RunningMeanStd()
        else:
            smax_rms = RunningMeanStd()
            smin_rms = RunningMeanStd()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if isinstance(envs.single_observation_space, gym.spaces.Dict):
                obs_ = {k: torch.as_tensor(v).to(device) for k, v in obs.items()}
            else:
                obs_ = torch.Tensor(obs).to(device)
                
            q_values = q_network(obs_)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # extract the bandit choice from the observation
        if "bandit" in args.model and args.normalize_int_reward:
            bandit_choice = int(obs["theta"].reshape(-1)[-1])
            if bandit_choice == 0:
                smax_rms.update(np.array([rewards]).flatten())
            else:
                smin_rms.update(np.array([rewards]).flatten())
        

        if "surprise" in infos:
            ep_surprise.append(infos["surprise"][0])
            ep_entropy.append(infos["theta_entropy"][0])
        if "task_reward" in infos:
            task_rewards.append(infos["task_reward"][0])
        if "entropy_change" in infos:
            entr_changes.append(infos["entropy_change"][0])

                
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                # print with 2 decimals only
                print(f"global_step={global_step}, average_task_return={info['Average_task_return']:.2f}, average_episode_length={info['Average_episode_length']:.2f}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_surprise", np.mean(ep_surprise), global_step)
                writer.add_scalar("charts/episodic_entropy", np.mean(ep_entropy), global_step)
                writer.add_scalar("charts/episodic_entropy_last_timestep", ep_entropy[-1], global_step)
                writer.add_scalar("charts/task_reward", np.mean(task_rewards), global_step)
                writer.add_scalar("charts/average_task_return", info["Average_task_return"], global_step)
                writer.add_scalar("charts/average_episode_length", info["Average_episode_length"], global_step)
                # log bandit metrics
                if args.model == "sadapt-bandit":
                    writer.add_scalar("charts/alpha_rolling_average", info["alpha_rolling_average"], global_step)
                    if "ucb_alpha_one" in info:
                        writer.add_scalar("charts/ucb_alpha_one", info["ucb_alpha_one"], global_step)
                        writer.add_scalar("charts/ucb_alpha_zero", info["ucb_alpha_zero"], global_step)
                    if "random_entropy" in info:
                        writer.add_scalar("charts/random_entropy", info["random_entropy"], global_step)
                    if "random_surprise" in info:
                        writer.add_scalar("charts/random_surprise", info["random_surprise"], global_step)
                    if "alpha_one_mean" in info:
                        writer.add_scalar("charts/alpha_one_mean", info["alpha_one_mean"], global_step)
                        writer.add_scalar("charts/alpha_zero_mean", info["alpha_zero_mean"], global_step)
                writer.add_scalar("charts/deaths", info["deaths"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

                if len(entr_changes) > 0:
                    writer.add_scalar("charts/entropy_change", entr_changes[-1], global_step)

                logger_.logs_a([
                    global_step,
                    info["Average_task_return"],
                    info["Average_episode_length"],
                    np.mean(ep_surprise),
                    np.mean(ep_entropy),
                    info["alpha_rolling_average"] if args.model == "sadapt-bandit" else 0,
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
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                rewards = data.rewards.flatten()
                # reward normalization
                if args.normalize_int_reward:
                    # update the rms using rewards from all envs
                    if "bandit" not in args.model:
                        rms.update(rewards.cpu().numpy())
                        rewards -= (rms.mean)
                        rewards /= np.sqrt(rms.var)
                    else:
                        bandit_choice = data.observations["theta"].reshape(args.batch_size, -1)[:, -1]
                        smin_rewards = rewards[bandit_choice==1]
                        smax_rewards = rewards[bandit_choice==0]
                        smin_rewards = (smin_rewards - smin_rms.mean.item()) / np.sqrt(smin_rms.var.item())
                        smax_rewards = (smax_rewards - smax_rms.mean.item()) / np.sqrt(smax_rms.var.item())
                        rewards[bandit_choice==1] = smin_rewards 
                        rewards[bandit_choice==0] = smax_rewards
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = rewards + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if "bandit" in args.model and args.normalize_int_reward:
                        # log the statistics of the normalized reward
                        writer.add_scalar("charts/smin_rewards_normalized", smin_rewards.mean().item(), global_step)
                        writer.add_scalar("charts/smax_rewards_normalized", smax_rewards.mean().item(), global_step)
                        writer.add_scalar("charts/smax_rewards_normalized_mean", smax_rms.mean, global_step)
                        writer.add_scalar("charts/smax_rewards_normalized_std", np.sqrt(smax_rms.var), global_step)
                        writer.add_scalar("charts/smin_rewards_normalized_mean", smin_rms.mean, global_step)
                        writer.add_scalar("charts/smin_rewards_normalized_std", np.sqrt(smin_rms.var), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            if global_step % args.video_log_freq == 0:
                eval_episode_dqn(q_network, eval_envs, device, f"runs/{run_name}", global_step, args.env_id, args.track)

                model_path = f"runs/{run_name}/dqn.pt"
                torch.save(q_network.state_dict(), model_path)
                print(f"model saved to {model_path}")

    model_path = f"runs/{run_name}/dqn.pt"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")
        
    envs.close()
    writer.close()