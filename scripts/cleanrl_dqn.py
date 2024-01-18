# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

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

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SMin_mountain_car",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SurpriseAdaptRooms-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.35,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    
    # ENV PARAMS
    parser.add_argument("--noisy-room", type=int, default=2,
        help="can be none, smax, smin, sadapt, sadapt-inverse")
    
    # OBJECTIVE PARAMS
    parser.add_argument("--model", type=str, default="none",
        help="can be none, smax, smin, sadapt, sadapt-inverse")
    parser.add_argument("--buffer-type", type=str, default="gaussian",
        help="can be gaussian, or multinoulli")
    parser.add_argument("--surprise_window_len", type=int, default=10)
    parser.add_argument("--surprise_change_threshold", type=float, default=0.0)
    parser.add_argument("--add-true-rew", type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.env_id == "tetris":
        assert args.buffer_type == "bernoulli", "tetris only supports bernoulli buffer"
    
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args

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
        
        
    args = parse_args()
    
    if "Adapt" in args.env_id:
        env_name = f"{args.env_id}_NoisyRoom_{args.noisy_room}"
    else:
        env_name = args.env_id
    
    if args.add_true_rew:
        env_name += "_withExtrinsic"
    else:
        env_name += "_noExtrinsic" 
    
    run_name = f"dqn_{env_name}_{args.model}_{args.buffer_type}_s{args.seed}"
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    use_theta = args.model in ["smax", "smin", "sadapt", "sadapt-inverse"]
    
    if "Rooms" in args.env_id:
        net = MinigridQNetwork
    elif args.env_id == "tetris":
        net = TetrisQNetwork
    elif "MountainCar" in args.env_id:
        net = MountainCarAgent
    
    q_network = net(envs, use_theta=use_theta).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = net(envs, use_theta=use_theta).to(device)
    target_network.load_state_dict(q_network.state_dict())
    print("q network")
    print(q_network)
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
    heights = []
    velocites = []
    ep_counter = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
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

        if "surprise" in infos:
            ep_surprise.append(infos["surprise"][0])
            ep_entropy.append(infos["theta_entropy"][0])

        if "height" in infos:
            # print("Height in info")
            heights.append(infos["height"][0])
        if "velocity" in infos:
            # print("velocity in info")
            velocites.append(infos["velocity"][0])

                
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                episode_length = info["episode"]["l"]
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_surprise={np.mean(ep_surprise)}, episodic_entropy={np.mean(ep_entropy)}, episode_length={episode_length}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_surprise", np.mean(ep_surprise), global_step)
                writer.add_scalar("charts/episodic_entropy", np.mean(ep_entropy), global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/velocity", np.mean(velocites), global_step)
                writer.add_scalar("charts/height", np.mean(heights), global_step)
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
                heights.clear()
                velocites.clear()
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
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
    envs.close()
    writer.close()