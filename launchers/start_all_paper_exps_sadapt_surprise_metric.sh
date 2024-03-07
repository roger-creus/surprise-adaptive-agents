
# # ####### DQN #######

# # MazeEnvLarge
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=1 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=1 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=1 

# ButterfliesEnvLarge
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000

# Tetris
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --total-timesteps=10_000_000

# # Maze
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1

# # Butterflies
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1

# MazeEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=1 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=1 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=1 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=1 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=1 --wandb-project-name="sadapt_bandit_surprise_metric"



# Maze2
# Maze2
# Maze2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"

# # ButterfliesEnvLarge1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=1 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"


# MinAtar/Breakout
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"

# # MinAtar/SpaceInvaders
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"

# MinAtar/Freeway
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"

# MinAtar/Seaques
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=1 --total-timesteps=10_000_000 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_surprise_metric"