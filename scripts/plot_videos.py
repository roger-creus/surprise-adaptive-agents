import torch
from IPython import embed
from gymnasium_wrappers.args import parse_args_dqn
from gymnasium_wrappers.utils import *
from gymnasium_wrappers.models import *

import matplotlib.pyplot as plt
import os


args, run_name = parse_args_dqn()
use_theta = args.model in ["smax", "smin", "sadapt", "sadapt-inverse", "sadapt-bandit"]

envs = gym.vector.SyncVectorEnv([make_env(args) for i in range(1)])

agent = MinAtarQNetwork(envs, use_theta=use_theta)
state_dict = torch.load('/home/roger/Desktop/surprise-adaptive-agents/runs/dqn_griddly-MazeEnv_smax_buffer:bernoulli_withExtrinsic:False_softreset:0_reweard_normalization:1_exp_rew:0_death_cost:0_survival_rew:0_seed:1/dqn.pt')
agent.load_state_dict(state_dict)

obs, _ = envs.reset()
done = False
ep_images = []

os.makedirs("movies", exist_ok=True)
step = 0
while not done:
    obs = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}
    action = agent(obs).argmax().item()
    obs, r, te, tr, _ = envs.step([action])
    
    img = envs.envs[0].unwrapped._env._env.render(observer="global", mode="rgb_array")
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f"movies/{len(ep_images)}.png")
    plt.close()

    ep_images.append(img)
    done = te or tr
