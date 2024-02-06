## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smin multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smax multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms none multinoulli 1

# # Tetris S-min
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smax bernoulli 1 --track=True --wandb-project-name="Tetris_DQN" --capture-video --scale-by-std=1 --soft_reset=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 8943 --track=True --wandb-project-name="Tetris_DQN" --caputre-video
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 820 --track=True --wandb-project-name="Tetris_DQN" --caputre-video

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 1 --track=True --wandb-project-name="Tetris_DQN" --capture-video --soft_reset=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 8943 --track=True --wandb-project-name="Tetris_DQN" --caputre-video
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 820 --track=True --wandb-project-name="Tetris_DQN" --caputre-video

# # Tetris Extrinsic
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 8943 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 820 --track=True

# Tetris Bandit
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --track --soft_reset=0 --scale-by-std=1 --wandb-project-name="Tetris_sadapt_bandit" --ucb_coeff=2
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --track --soft_reset=0 --scale-by-std=1 --wandb-project-name="Tetris_sadapt_bandit" --ucb_coeff=5
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --track --soft_reset=0 --scale-by-std=1 --wandb-project-name="Tetris_sadapt_bandit" 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 8943 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 820 --track=True

# Crafter 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 1 --track=True 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 8943 --track=True --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 820 --track=True --soft_reset=1

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 1 --track=True --soft_reset=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 8943 --track=True --soft_reset=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 820 --track=True --soft_reset=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smin gaussian 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter none gaussian 1 --track=True

# random crafter agnet
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 1 --track=True --wandb-project-name="Crafter_DQN" 
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 8943 --track=True --wandb-project-name="Crafter_DQN"
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 820 --track=True --wandb-project-name="Crafter_DQN"

# random crafter agnet
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 1 --track=True --wandb-project-name="Crafter_DQN" --soft_reset=0
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 8943 --track=True --wandb-project-name="Crafter_DQN" --soft_reset=0
# sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 820 --track=True --wandb-project-name="Crafter_DQN" --soft_reset=0

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 1

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none multinoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none multinoulli 1 --noisy-room=1



# MinAtar
# w/o soft-reset
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smin bernoulli 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smin bernoulli 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smin bernoulli 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smin bernoulli 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smax bernoulli 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway smax bernoulli 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smax bernoulli 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Seaquest smax bernoulli 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# Survival reward experiment
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --wandb-project-name="MinAtar_DQN" --survival_rew=1


# Maze
# w/o soft-reset
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smax gaussian 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smax gaussian 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smax gaussian 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smin gaussian 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smin gaussian 820 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv smin gaussian 8943 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="test_maze" --video-log-freq=500000

# w/ soft-reset
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 1 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 820 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 8943 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 1 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 820 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 8943 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 820 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 8943 --scale-by-std=0 --soft_reset=1 --track --wandb-project-name="MinAtar_DQN" --video-log-freq=500000

# Butterflies
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 1 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0 --total-timesteps=50000000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 820 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0 --total-timesteps=50000000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 8943 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0 --total-timesteps=50000000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 1 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0  --total-timesteps=50000000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 820 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0 --total-timesteps=50000000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 8943 --track --wandb-project-name="butterflies_DQN" --scale-by-std=1 --soft_reset=0 --total-timesteps=50000000

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 1 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 820 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax gaussian 8943 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 1 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 820 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin gaussian 8943 --track --wandb-project-name="butterflies_DQN" --scale-by-std=0 --soft_reset=1

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smin gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smin gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)" --gray_scale=0

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smax gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)" 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smax gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)" --gray_scale=0

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)" 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 8943 --track --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_2" --scale-by-std=1 --obs_size="(64,64)" --gray_scale=0