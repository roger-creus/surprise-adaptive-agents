import gymnasium as gym
import csv_logger
import logging
import matplotlib.pyplot as plt
import cv2

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium_wrappers.base_surprise import BaseSurpriseWrapper
from surprise.buffers.buffers import GaussianBufferIncremental

from IPython import embed


def make_env(env_id, seed, model, noisy_room):
    def thunk():
        env = gym.make(env_id, render_mode='rgb_array', max_steps=500, noisy_room=noisy_room)
        env = ImgObsWrapper(env) 
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if model == "smax":
            obs_size = env.observation_space.shape
            buffer = GaussianBufferIncremental(obs_size)
            env = BaseSurpriseWrapper(env, buffer, add_true_rew=True, minimize=False, int_rew_scale=1.0)
        
        elif model == "smin":
            obs_size = env.observation_space.shape
            buffer = GaussianBufferIncremental(obs_size)
            env = BaseSurpriseWrapper(env, buffer, add_true_rew=True, minimize=True, int_rew_scale=1.0)
            
        elif model == "sadapt":
            raise NotImplementedError
        
        elif model == "sadapt-inverse":
            raise NotImplementedError
        
        elif model == "none":
            obs_size = env.observation_space.shape
            buffer = GaussianBufferIncremental(obs_size)
            env = BaseSurpriseWrapper(env, buffer, add_true_rew=True, minimize=False, int_rew_scale=1.0, ext_only=True)
        else:
            raise ValueError(f"Unknown model {model}")
                
        env.action_space.seed(seed)
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


def log_heatmap(env, heatmap, ep_counter, writer):
    cmap = plt.get_cmap('Greens')
    cmap.set_under((0,0,0,0))
    cmap_args = dict(cmap=cmap, vmin=1, vmax=500)
    
    fig = plt.figure(num=1)
    background_img = env.render()
    background = cv2.resize(background_img, dsize=(28, 10), interpolation=cv2.INTER_AREA)
    plt.imshow(background.transpose(1,0,2) , alpha=0.75)
    plt.imshow(heatmap, **cmap_args, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    #plt.savefig(f"heatmap_{ep_counter}.png")
    writer.add_figure(f"trajectory/heatmap_{ep_counter}", fig, close=True)
    plt.clf()