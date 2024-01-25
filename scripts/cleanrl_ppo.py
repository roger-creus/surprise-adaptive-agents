# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from IPython import embed
from gymnasium_wrappers.utils import *
from gymnasium_wrappers.models import *
from gymnasium_wrappers.args import parse_args_ppo

if __name__ == "__main__":
    args, run_name = parse_args_ppo()
    
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
        net = MinigridPPOAgent
    elif args.env_id == "tetris":
        net = TetrisPPOAgent
    elif "MinAtar" in args.env_id:
        net = MinAtarPPOAgent
    elif args.env_id == "crafter":
        net = CrafterPPOAgent
        crafter_logger = CrafterLogger()
    else:
        raise NotImplementedError

    agent = net(envs, use_theta).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print(agent)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space["obs"].shape).to(device)
    thetas = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space["theta"].shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    logger_ = make_csv_logger(f"runs/{run_name}/log.csv")
    global_step = 0
    start_time = time.time()
    
    o_, _ = envs.reset()
    next_obs = torch.Tensor(o_["obs"]).to(device)
    next_theta = torch.Tensor(o_["theta"]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    ep_surprise = {i : [] for i in range(args.num_envs)}
    ep_entropy = {i : [] for i in range(args.num_envs)}
    task_rewards = {i : [] for i in range(args.num_envs)}
    ep_counter = 0
    update_idx = 0
    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            thetas[step] = next_theta
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value({
                    "obs" : next_obs,
                    "theta" : next_theta
                })
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            o_, reward, done, trunc, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_theta, next_done = torch.Tensor(o_["obs"]).to(device), torch.Tensor(o_["theta"]).to(device), torch.Tensor(done).to(device)
            
            for i in range(args.num_envs):
                if "surprise" in infos:
                    ep_surprise[i].append(infos['surprise'][i])
                if "theta_entropy" in infos:
                    ep_entropy[i].append(infos['theta_entropy'][i])
                if "task_reward" in infos:
                    task_rewards[i].append(infos['task_reward'][i])
                
            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            c = 0
            for info in infos["final_info"]:
                # update crafter logs
                if "crafter" in args.env_id:
                    crafter_logger.update_achievements(info["achievements"])
                    crafter_logger.log(writer, global_step)
                # Skip the envs that are not done
                if "episode" not in info:
                    c += 1
                    continue       
                 
                print(f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_surprise", np.mean(ep_surprise[c]), global_step)
                writer.add_scalar("charts/episodic_entropy", np.mean(ep_entropy[c]), global_step)
                writer.add_scalar("charts/task_reward", np.mean(task_rewards[c]), global_step)
                writer.add_scalar("charts/average_task_return", info["Average_task_return"], global_step)
                writer.add_scalar("charts/average_episode_length", info["Average_episode_length"], global_step)

                logger_.logs_a([
                    global_step,
                    info["episode"]["r"][0],
                    info["episode"]["l"][0],
                    np.mean(ep_surprise[c]),
                    np.mean(ep_entropy[c]),
                ])
                
                ep_surprise[c].clear()
                ep_entropy[c].clear()
                ep_counter += 1
                c += 1
                
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value({
                "obs" : next_obs,
                "theta" : next_theta
            }).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space["obs"].shape)
        b_thetas = thetas.reshape((-1,) + envs.single_observation_space["theta"].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    {"obs" : b_obs[mb_inds], "theta" : b_thetas[mb_inds]}, 
                    b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        update_idx += 1

    envs.close()
    writer.close()