"""
Run DQN on grid world.
"""
import sys

try:
    from comet_ml import Experiment
except:
    pass
import gym

import numpy as np

from IPython import embed


def get_network(network_args, obs_dim, action_dim, unflattened_obs_dim=None, device='cpu'):
    if network_args["type"] == "conv_mixed":
        from surprise.envs.vizdoom.networks import VizdoomQF
        qf = VizdoomQF(actions=action_dim, **network_args)
        target_qf = VizdoomQF(actions=action_dim, **network_args)
    elif network_args["type"] == "conv":
        from surprise.envs.vizdoom.networks import VizdoomFeaturizer

        print("Using conv")
        qf = VizdoomFeaturizer(dim=action_dim, **network_args)
        target_qf = VizdoomFeaturizer(dim=action_dim, **network_args)
    elif network_args["type"] == "cnn":
        from surprise.utils.networks import MixedIdentMlpCNN

        network_args.pop("type")
        assert unflattened_obs_dim is not None
        qf = MixedIdentMlpCNN(
            action_dim=action_dim, obs_dim=unflattened_obs_dim, **network_args
        )
        target_qf = MixedIdentMlpCNN(
            action_dim=action_dim, obs_dim=unflattened_obs_dim, **network_args
        )
    elif network_args["type"] == "cnn_minatar":
        from surprise.utils.networks import MinAtarDQN
        print(unflattened_obs_dim)
        c,h,w = unflattened_obs_dim['observation']
        network_args.pop('type')
        qf = MinAtarDQN(c,h,w, action_shape=action_dim, device=device, **network_args)
        target_qf = MinAtarDQN(c,h,w, action_shape=action_dim, device=device, **network_args)

    else:
        from rlkit.torch.networks import Mlp

        qf = Mlp(
            hidden_sizes=[128, 64, 32],
            input_size=obs_dim[0],
            output_size=action_dim,
        )
        target_qf = Mlp(
            hidden_sizes=[128, 64, 32],
            input_size=obs_dim[0],
            output_size=action_dim,
        )

    return (qf, target_qf)


def get_env(variant):
    from launchers.config import BASE_CODE_DIR

    if variant["env"] == "Tetris":
        from surprise.envs.tetris.tetris import TetrisEnv
        env = TetrisEnv(render=True, **variant["env_kwargs"])

    #### Added crafter
    elif variant["env"] == "Crafter":
        import gym
        import crafter
        from surprise.wrappers.crafter_wrapper import CrafterWrapper
        env_wargs = variant["env_kwargs"]
        env = gym.make("CrafterNoReward-v1", length=env_wargs["max_steps"])
        # env = CrafterWrapper(env)
        
    elif variant["env"] == "VizDoom":
        from surprise.envs.vizdoom.VizdoomWrapper import VizDoomEnv

        env_wargs = variant["env_kwargs"]
        env = VizDoomEnv(
            config_path=BASE_CODE_DIR + env_wargs["doom_scenario"], **env_wargs
        )
    elif "MazeEnv" in variant["env"]:
        import griddly
        import gym
        from griddly import GymWrapperFactory, gd
        from surprise.envs.maze.maze_env import MazeEnvFullyObserved

        env = MazeEnvFullyObserved()
        if "FullyObserved" in variant["env"]:
            env_dict = gym.envs.registration.registry.env_specs.copy()
            for env_ in env_dict:
                if "GDY-MazeEnvFullyObserved-v0" in env_:
                    del gym.envs.registration.registry.env_specs[env_]

            import os

            wrapper = GymWrapperFactory()
            wrapper.build_gym_from_yaml(
                "MazeEnvFullyObserved",
                f"/home/ahugessen/github/surprise-adaptive-agents/surprise/envs/maze/maze_env_fully_observed.yaml",
            )
            env_ = gym.make(
                "GDY-MazeEnvFullyObserved-v0",
                player_observer_type=gd.ObserverType.VECTOR,
                global_observer_type=gd.ObserverType.VECTOR,
                max_steps=variant["env_kwargs"]["max_steps"],
            )
            env.set_env(env_)
        else:
            raise "This maze is not implemented"

        from surprise.wrappers.obsresize import MazeEnvOneMaskObs

        env = MazeEnvOneMaskObs(env)                                                                         
    else:
        import gym

        try:
            env = gym.make(variant["env"], **variant["env_kwargs"])
        except Exception:
            print("Environment kwargs are not valid. Ignoring...")
            env = gym.make(variant["env"])

    return env


def add_wrappers(env, variant, device=0, eval=False, network=None, flip_alpha=False):
    from surprise.wrappers.obsresize import (
        ResizeObservationWrapper,
        RenderingObservationWrapper,
        SoftResetWrapper,
        ChannelFirstWrapper,
        ObsHistoryWrapper,
        RescaleImageWrapper,
        AddAlphaWrapper,
        FlattenDictObservationWrapper,
        AddTextInfoToRendering,
        StrictOneHotWrapper
    )
    from surprise.wrappers.VAE_wrapper import VAEWrapper
    from surprise.wrappers.crafter_wrapper import CrafterWrapper
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    from gym_minigrid.minigrid import MiniGridEnv

    if isinstance(env, MiniGridEnv):
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    obs_dim = env.observation_space.low.shape
    print("obs dim", obs_dim)
    for wrapper in variant["wrappers"]:
        if "smirl_wrapper" in wrapper:
            env = add_smirl(
                env=env,
                variant=wrapper["smirl_wrapper"],
                ep_length=variant["env_kwargs"]["max_steps"],
                device=device,
            )
        elif "surprise_adapt_wrapper" in wrapper:
            env = add_surprise_adapt(
                env=env,
                variant=wrapper["surprise_adapt_wrapper"],
                ep_length=variant["env_kwargs"]["max_steps"],
                device=device,
                flip_alpha=flip_alpha,
                flip_alpha_strategy=wrapper["surprise_adapt_wrapper"][
                    "flip_alpha_strategy"
                ],
            )
        elif "surprise_adapt_wrapper_v2" in wrapper:
            env = add_surprise_adapt_v2(
                env=env,
                variant=wrapper["surprise_adapt_wrapper_v2"],
                ep_length=variant["env_kwargs"]["max_steps"],
                device=device,
                flip_alpha=flip_alpha,
                flip_alpha_strategy=wrapper["surprise_adapt_wrapper_v2"][
                    "flip_alpha_strategy"
                ],
            )
        elif "surprise_adapt_bandit_wrapper" in wrapper:
            env = add_surprise_adapt_bandit(
                env=env,
                variant=wrapper["surprise_adapt_bandit_wrapper"],
                ep_length=variant["env_kwargs"]["max_steps"],
                device=device,
                eval=eval,
            )
        elif "soft_reset_wrapper" in wrapper:
            env = SoftResetWrapper(env=env, max_time=variant["env_kwargs"]["max_steps"])
        elif "FlattenObservationWrapper" in wrapper:
            from surprise.wrappers.obsresize import FlattenObservationWrapper

            env = FlattenObservationWrapper(
                env=env, **wrapper["FlattenObservationWrapper"]
            )
        elif "dict_obs_wrapper" in wrapper:
            from surprise.wrappers.obsresize import DictObservationWrapper

            env = DictObservationWrapper(env=env, **wrapper["dict_obs_wrapper"])
        elif "dict_to_obs_wrapper" in wrapper:
            from surprise.wrappers.obsresize import DictToObservationWrapper

            env = DictToObservationWrapper(env=env, **wrapper["dict_to_obs_wrapper"])
        elif "rendering_observation" in wrapper and (eval == True):
            env = RenderingObservationWrapper(
                env=env, **wrapper["rendering_observation"]
            )
        elif "resize_observation_wrapper" in wrapper:
            env = ResizeObservationWrapper(
                env=env, **wrapper["resize_observation_wrapper"]
            )
            obs_dim = env.observation_space.low.shape
            print("obs dim resize", obs_dim)
        elif "channel_first_observation_wrapper" in wrapper:
            env = ChannelFirstWrapper(
                env=env, **wrapper["channel_first_observation_wrapper"]
            )
            obs_dim = env.observation_space.low.shape
            print("obs dim channel first", obs_dim)
        elif "ObsHistoryWrapper" in wrapper:
            env = ObsHistoryWrapper(env=env, **wrapper["ObsHistoryWrapper"])
            obs_dim = env.observation_space.low.shape
            print("obs dim history stack", obs_dim)
        elif "vae_wrapper" in wrapper:
            print(wrapper["vae_wrapper"])
            print("network: ", network)
            env = VAEWrapper(
                env=env,
                eval=eval,
                device=device,
                network=network,
                **wrapper["vae_wrapper"],
            )
            network = env.network
        elif "RNDWrapper" in wrapper:
            from surprise.wrappers.RND_wrapper import RNDWrapper

            env = RNDWrapper(env=env, eval=eval, **wrapper["RNDWrapper"], device=device)
            network = env.network
        elif "ICMWrapper" in wrapper:
            from surprise.wrappers.ICM_wrapper import ICMWrapper

            env = ICMWrapper(env=env, eval=eval, **wrapper["ICMWrapper"], device=device)
            network = env.network
        elif "rescale_rgb" in wrapper:
            env = RescaleImageWrapper(env=env)
        elif "add_alpha" in wrapper:
            env = AddAlphaWrapper(env=env)
        elif "flatten_dict_observation" in wrapper:
            env = FlattenDictObservationWrapper(env)
        elif "add_text_info_to_rendering" in wrapper and eval:
            env = AddTextInfoToRendering(
                env=env, **wrapper["add_text_info_to_rendering"]
            )
        elif "strict_one_hot_wrapper" in wrapper:
            env = StrictOneHotWrapper(env, **wrapper["strict_one_hot_wrapper"])
        elif "crafter_wrapper" in wrapper:
            from launchers.config import CODE_DIRS_TO_MOUNT
            base_path = CODE_DIRS_TO_MOUNT[0]
            metric_path = wrapper["crafter_wrapper"]["save_metrics_path"]
            exp_name = variant["exp_name"]
            # /home/mila/f/faisal.mohamed/scratch/doodad-output//smirl/test_crafter/./metrics
            metrics_save_path = f"{base_path}/{exp_name}/{metric_path}"
            os.makedirs(metrics_save_path, exist_ok=True)
            env = CrafterWrapper(env, save_metrics=True, save_metrics_path=metrics_save_path)
        else:
            if not eval:
                pass
            else:
                print("wrapper not known: ", wrapper)
                sys.exit()


    
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_dim = {
            key: obs.low.shape for key, obs in env.observation_space.spaces.items()
        }
    else:
        obs_dim = env.observation_space.low.shape
    return env, network


def add_surprise_adapt(
    env, variant, ep_length=500, device=0, flip_alpha=False, flip_alpha_strategy="SA"
):
    from surprise.buffers.buffers import (
        BernoulliBuffer,
        MultinoulliBuffer,
        GaussianBufferIncremental,
        GaussianCircularBuffer,
    )
    from surprise.wrappers.base_surprise_adapt import BaseSurpriseAdaptWrapper

    if "latent_obs_size" in variant:
        obs_size = variant["latent_obs_size"]
    else:
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_space = env.observation_space['observation']
        else:
            obs_space = env.observation_space
        obs_shape = obs_space.shape
        obs_size = np.prod(obs_shape)
        # print(f'obs shape in buffer: {obs_size}')
    if variant["buffer_type"] == "Bernoulli":
        buffer = BernoulliBuffer(obs_size)
        env = BaseSurpriseAdaptWrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
        )
    elif variant["buffer_type"] == "Multinoulli":
        num_cat = int(obs_space.high[0,0]+1) if len(obs_shape) == 2 else None
        buffer = MultinoulliBuffer(obs_dim=obs_shape, num_cat=num_cat)
        env = BaseSurpriseAdaptWrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
        )
    elif variant["buffer_type"] == "Gaussian":
        buffer = GaussianBufferIncremental(obs_size)
        env = BaseSurpriseAdaptWrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
        )
    else:
        print("Non supported prob distribution type: ", variant["buffer_type"])
        sys.exit()

    return env


def add_surprise_adapt_v2(
    env, variant, ep_length=500, device=0, flip_alpha=False, flip_alpha_strategy="SA"
):
    from surprise.buffers.buffers import BernoulliBuffer, MultinoulliBuffer, GaussianBufferIncremental
    from surprise.wrappers.base_surprise_adapt_v2 import BaseSurpriseAdaptV2Wrapper
    
    if "latent_obs_size" in variant:
        obs_size = variant["latent_obs_size"]
    else:
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_space = env.observation_space['observation']
        else:
            obs_space = env.observation_space
        obs_shape = obs_space.shape
        obs_size = np.prod(obs_shape)

    if variant["buffer_type"] == "Bernoulli":
        buffer = BernoulliBuffer(obs_size)
        env = BaseSurpriseAdaptV2Wrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
        )    
    elif variant["buffer_type"] == "Multinoulli":
        num_cat = int(obs_space.high[0,0]+1) if len(obs_shape) == 2 else None
        buffer = MultinoulliBuffer(obs_dim=obs_shape, num_cat=num_cat)
        env = BaseSurpriseAdaptV2Wrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
    )
    elif variant["buffer_type"] == "Gaussian":
        buffer = GaussianBufferIncremental(obs_size)
        env = BaseSurpriseAdaptV2Wrapper(
            env, buffer, time_horizon=ep_length, flip_alpha=flip_alpha, **variant
        )
    else:
        print("Non supported prob distribution type: ", variant["buffer_type"])
        sys.exit()

    return env


def add_surprise_adapt_bandit(env, variant, ep_length=500, device=0, eval=False):
    from surprise.buffers.buffers import BernoulliBuffer, MultinoulliBuffer, GaussianBufferIncremental
    from surprise.wrappers.base_surprise_adapt_bandit import (
        BaseSurpriseAdaptBanditWrapper,
    )

    if "latent_obs_size" in variant:
        obs_size = variant["latent_obs_size"]
    else:
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_space = env.observation_space['observation']
        else:
            obs_space = env.observation_space
        obs_shape = obs_space.shape
        obs_size = np.prod(obs_shape)
        print(f"obs size:{obs_size}")

    if variant["buffer_type"] == "Bernoulli":
        buffer = BernoulliBuffer(obs_size)
        env = BaseSurpriseAdaptBanditWrapper(
            env, buffer, time_horizon=ep_length, eval=eval, **variant
        )
    elif variant["buffer_type"] == "Multinoulli":
        num_cat = int(obs_space.high[0,0]+1) if len(obs_shape) == 2 else None
        buffer = MultinoulliBuffer(obs_dim=obs_shape, num_cat=num_cat)
        env = BaseSurpriseAdaptBanditWrapper(
            env, buffer, time_horizon=ep_length, eval=eval, **variant
        )
    elif variant["buffer_type"] == "Gaussian":
        buffer = GaussianBufferIncremental(obs_size)
        env = BaseSurpriseAdaptBanditWrapper(
            env, buffer, time_horizon=ep_length, eval=eval, **variant
        )
    else:
        print("Non supported prob distribution type: ", variant["buffer_type"])
        sys.exit()

    return env


def add_smirl(env, variant, ep_length=500, device=0):
    from surprise.buffers.buffers import (
        BernoulliBuffer,
        MultinoulliBuffer,
        GaussianBufferIncremental,
        GaussianCircularBuffer,
    )
    from surprise.wrappers.base_surprise import BaseSurpriseWrapper
    if "latent_obs_size" in variant:
        obs_size = variant["latent_obs_size"]
    else:
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_space = env.observation_space['observation']
        else:
            obs_space = env.observation_space
        obs_shape = obs_space.shape
        obs_size = np.prod(obs_shape)

    if variant["buffer_type"] == "Bernoulli":
        buffer = BernoulliBuffer(obs_size)
        env = BaseSurpriseWrapper(env, buffer, time_horizon=ep_length, **variant)
    elif variant["buffer_type"] == "Multinoulli":
        num_cat = int(obs_space.high[0,0]+1) if len(obs_shape) == 2 else None
        buffer = MultinoulliBuffer(obs_dim=obs_shape, num_cat=num_cat)
        env = BaseSurpriseWrapper(env, buffer, time_horizon=ep_length, **variant)
    elif variant["buffer_type"] == "Gaussian":
        #         buffer = GaussianCircularBuffer(obs_size, size=500)
        buffer = GaussianBufferIncremental(obs_size)
        env = BaseSurpriseWrapper(env, buffer, time_horizon=ep_length, **variant)
    else:
        print("Non supported prob distribution type: ", variant["smirl"]["buffer_type"])
        sys.exit()

    return env


def experiment(doodad_config, variant):
    from rlkit.core import logger
    from rlkit.launchers.launcher_util import setup_logger

    print("doodad_config.base_log_dir: ", doodad_config.base_log_dir)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    snapshot_mode = (
        variant["snapshot_mode"] if "snapshot_mode" in variant.keys() else "last"
    )
    setup_logger(
        "wrapped_" + variant["env"],
        variant=variant,
        log_dir=doodad_config.base_log_dir
        + "/smirl/"
        + variant["exp_name"]
        + "/"
        + timestamp
        + "/",
        snapshot_mode=snapshot_mode,
    )
    if variant["log_comet"]:
        try:
            from launchers.config import (
                COMET_API_KEY,
                COMET_PROJECT_NAME,
                COMET_WORKSPACE,
            )

            comet_logger = Experiment(
                api_key=COMET_API_KEY,
                project_name=COMET_PROJECT_NAME,
                workspace=COMET_WORKSPACE,
            )
            logger.set_comet_logger(comet_logger)
            comet_logger.set_name(str(variant["env"]) + "_" + str(variant["exp_name"]))
            print("variant: ", variant)
            variant["comet_key"] = comet_logger.get_key()
            comet_logger.log_parameters(variant)
            print(comet_logger)
        except Exception as inst:
            print("Not tracking training via commet.ml")
            print("Error: ", inst)

    import gym
    from torch import nn as nn

    import rlkit.torch.pytorch_util as ptu
    import torch
    from surprise.utils.exploration import EpsilonGreedy
    from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
    from rlkit.policies.argmax import ArgmaxDiscretePolicy
    from rlkit.torch.dqn.dqn import DQNTrainer
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.samplers.data_collector.step_collector import MdpStepCollector
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from surprise.utils.rendering_algorithm import TorchBatchRLRenderAlgorithm, TorchOnlineRLRenderAlgorithm
    from surprise.envs.tetris.tetris import TetrisEnv
    from surprise.wrappers.obsresize import (
        ResizeObservationWrapper,
        RenderingObservationWrapper,
        SoftResetWrapper,
    )
    import pdb

    base_env = get_env(variant)
    base_env2 = get_env(variant)

    # print("GPU_BUS_Index", variant["GPU_BUS_Index"])
    # cuda_is_available = torch.cuda.is_available()
    # print(f"torch.cuda.is_available():{cuda_is_available}")
    # print(f"doodad_config.use_gpu: {doodad_config.use_gpu}")
    # input()
    if torch.cuda.is_available() and doodad_config.use_gpu:
        print("Using the GPU for learning")
        #         ptu.set_gpu_mode(True, gpu_id=doodad_config.gpu_id)
        ptu.set_gpu_mode(True, gpu_id=variant["GPU_BUS_Index"])
    else:
        print("NOT Using the GPU for learning")

    #     base_env2 = RenderingObservationWrapper(base_env2)
    expl_env, network = add_wrappers(base_env, variant, device=ptu.device)

    # this is only for SA with Fixed Alphas during training to flip alphas according to SA in eval only
    flip_alpha_eval = False

    for wrapper in variant["wrappers"]:
        if "surprise_adapt_wrapper" in wrapper:
            wr = wrapper["surprise_adapt_wrapper"]
            flip_strategy = wr["flip_alpha_strategy"]
            if flip_strategy == "SA_fixedAlphas":
                flip_alpha_eval = True

    eval_env, _ = add_wrappers(
        base_env2,
        variant,
        flip_alpha=flip_alpha_eval,
        device=ptu.device,
        eval=True,
        network=network,
    )
    if "vae_wrapper" in variant["wrappers"]:
        eval_env._network = base_env._network

    try:
        unflatten_obs_dim = {
            key: obs.low.shape
            for key, obs in expl_env.unflattened_observation_space.spaces.items()
        }
    except Exception:
        unflatten_obs_dim = None
    obs_dim = expl_env.observation_space.low.shape
    print("Final obs dim", obs_dim)
    action_dim = eval_env.action_space.n
    print("Action dimension: ", action_dim)
    qf, target_qf = get_network(
        variant["network_args"], obs_dim, action_dim, unflatten_obs_dim, device=ptu.device
    )
    qf_criterion = nn.MSELoss()
    if variant["algorithm"] == "random":
        from rlkit.policies.simple import RandomPolicy

        eval_policy = RandomPolicy(eval_env.action_space)
        expl_policy = RandomPolicy(expl_env.action_space)
    else:
        eval_policy = ArgmaxDiscretePolicy(qf)
        if "exploration_kwargs" in variant.keys():
            expl_policy = PolicyWrappedWithExplorationStrategy(
                EpsilonGreedy(
                    expl_env.action_space,
                    **variant["exploration_kwargs"]
                ),
                eval_policy,
            )
        else:
            expl_policy = PolicyWrappedWithExplorationStrategy(
                EpsilonGreedy(
                    expl_env.action_space, prob_random_action=0.8, prob_end=0.05
                ),
                eval_policy,
            )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"],
    )
    try:
        eval_env.set_discount_rate(variant["trainer_kwargs"]["discount"])
        expl_env.set_discount_rate(variant["trainer_kwargs"]["discount"])
    except Exception as e:
        print("No method for setting discount rate in environment, defaulting to 1.")
        print(f"error is: {e}")
        input()

    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    # # printing for debugging
    # print(f"replay_buffer:{replay_buffer.env}")
    obs_sample = expl_env.observation_space.sample()
    # print(f"env_obs space:{expl_env.observation_space}")
    # print(f"obs sample shape:{obs_sample.shape}")



    online = variant.get("online")
    if online is not None and online:
        eval_step_collector = MdpPathCollector(
            eval_env, eval_policy, render_kwargs=variant["render_kwargs"]
        )
        expl_step_collector = MdpStepCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchOnlineRLRenderAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_step_collector,
        replay_buffer=replay_buffer,
        **{
            **variant["algorithm_kwargs"],
            **{"max_steps": variant["env_kwargs"]["max_steps"]},
        },
        )
    else:
        eval_path_collector = MdpPathCollector(
            eval_env, eval_policy, render_kwargs=variant["render_kwargs"]
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchBatchRLRenderAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **{
                **variant["algorithm_kwargs"],
                **{"max_steps": variant["env_kwargs"]["max_steps"]},
            },
        )
    print(f"exp env:{expl_env.observation_space}")
    print(f"eval env:{eval_env.observation_space}")
    # print(f"device is:{ptu.device}")
    # input()
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    from util.simOptions import getOptions
    import sys, os, json, copy

    #     from doodad.easy_launch.python_function import run_experiment

    settings = getOptions(sys.argv)

    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(None, variant)
    from util.tuneParams import run_sweep
    from launchers.config import *

    sweep_ops = {}
    if "tuningConfig" in settings:
        sweep_ops = json.load(open(settings["tuningConfig"], "r"))

    run_sweep(
        experiment,
        sweep_ops=sweep_ops,
        variant=settings,
        repeats=settings["random_seeds"],
        meta_threads=settings["meta_sim_threads"],
    )
