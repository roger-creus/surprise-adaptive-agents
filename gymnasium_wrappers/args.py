import argparse
import os
from distutils.util import strtobool

def parse_args_dqn():
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Crafter_DQN",
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
    parser.add_argument("--video-log-freq", type=int, default=500_000,
        help="the frequency of logging videos for ppo iterations")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SurpriseAdaptRooms-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1_000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    
    # ENV PARAMS
    parser.add_argument("--noisy-room", type=int, default=2,
        help="can be none, smax, smin, sadapt, sadapt-inverse")
    
    # OBJECTIVE PARAMS
    parser.add_argument("--model", type=str, default="none",
        help="can be none, smax, smin, sadapt, sadapt-inverse, sadapt-bandit")
    parser.add_argument("--buffer-type", type=str, default="gaussian",
        help="can be gaussian, or multinoulli")
    parser.add_argument("--surprise_window_len", type=int, default=10)
    parser.add_argument("--surprise_change_threshold", type=float, default=0.0)
    parser.add_argument("--add-true-rew", type=bool, default=False)
    parser.add_argument("--scale-by-std", type=int, default=1)
    parser.add_argument("--soft_reset", type=int, default=1)
    # if 0 or less no video capture, this freq is in episodes not timesteps
    parser.add_argument("--video_freq", type=int, default=-1) 
    parser.add_argument("--agent", type=str, default="DQN") 
    
    args = parser.parse_args()
    
    if args.env_id == "tetris":
        assert args.buffer_type == "bernoulli", "tetris only supports bernoulli buffer"
    
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    # run_name = f"dqn_{args.env_id}_{args.model}_buffer:{args.buffer_type}_withExtrinsic:{args.add_true_rew}_softreset:{args.soft_reset}_seed:{args.seed}"

    run_name = f"dqn_{args.env_id}_{args.model}_buffer:{args.buffer_type}_withExtrinsic:{args.add_true_rew}_softreset:{args.soft_reset}_seed:{args.seed}_no_gaussian_obs_at_softreset"

    return args, run_name


def parse_args_ppo():
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Crafter_PPO",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--video-log-freq", type=int, default=200,
        help="the frequency of logging videos for ppo iterations")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MinAtar/SpaceInvaders",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
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
    parser.add_argument("--scale-by-std", type=int, default=1)
    parser.add_argument("--soft_reset", type=int, default=1)
    parser.add_argument("--video_freq", type=int, default=-1) 
    parser.add_argument("--agent", type=str, default="PPO") 
    
    args = parser.parse_args()
    
    if args.env_id == "tetris":
        assert args.buffer_type == "bernoulli", "tetris only supports bernoulli buffer"
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    run_name = f"ppo_{args.env_id}_{args.model}_buffer:{args.buffer_type}_withExtrinsic:{args.add_true_rew}_softreset:{args.soft_reset}_seed:{args.seed}"
    return args, run_name


def parse_args_random():
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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Tetris",
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
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
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
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    
    # ENV PARAMS
    parser.add_argument("--noisy-room", type=int, default=2,
        help="can be none, smax, smin, sadapt, sadapt-inverse")
    
    # OBJECTIVE PARAMS
    parser.add_argument("--model", type=str, default="none",
        help="can be none, smax, smin, sadapt, sadapt-inverse, sadapt-bandit")
    parser.add_argument("--buffer-type", type=str, default="gaussian",
        help="can be gaussian, or multinoulli")
    parser.add_argument("--surprise_window_len", type=int, default=10)
    parser.add_argument("--surprise_change_threshold", type=float, default=0.0)
    parser.add_argument("--add-true-rew", type=bool, default=False)
    parser.add_argument("--scale-by-std", type=int, default=1)
    parser.add_argument("--soft_reset", type=int, default=1)
    parser.add_argument("--video_freq", type=int, default=-1) 
    parser.add_argument("--agent", type=str, default="Random") 
    
    args = parser.parse_args()
    
    if args.env_id == "tetris":
        assert args.buffer_type == "bernoulli", "tetris only supports bernoulli buffer"
    
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    run_name = f"random_agent_{args.env_id}_{args.model}_buffer:{args.buffer_type}_withExtrinsic:{args.add_true_rew}_softreset:{args.soft_reset}_seed:{args.seed}"

    return args, run_name