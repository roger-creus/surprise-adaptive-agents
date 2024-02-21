
# # ####### DQN #######

# # Grafter
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500

# # MazeEnvLarge
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1

# # ButterfliesEnvLarge
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit" --ucb_coeff=1

# # # Tetris
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit" --ucb_coeff=1
# # sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit" --ucb_coeff=1

# # Maze
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # Butterflies
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# MazeEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --exploration-fraction=0.1 --total-timesteps=10_000_000 --use_surprise=0 --ucb_coeff=0.02
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --exploration-fraction=0.1 --total-timesteps=10_000_000 --use_surprise=0 --ucb_coeff=0.02
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --exploration-fraction=0.1 --total-timesteps=10_000_000 --use_surprise=0 --ucb_coeff=0.02

# Maze2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --learning-starts=10000 --exploration-fraction=0.2 --ucb_coeff=0.01 --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --learning-starts=10000 --exploration-fraction=0.2 --ucb_coeff=0.01 --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --learning-starts=10000 --exploration-fraction=0.2 --ucb_coeff=0.01 --use_surprise=0

# # ButterfliesEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5 --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5 --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5 --use_surprise=0


# MinAtar/Breakout

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # MinAtar/SpaceInvaders

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"