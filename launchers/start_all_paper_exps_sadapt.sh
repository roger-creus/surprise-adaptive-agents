
# # ####### DQN #######

# # MazeEnvLarge
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 

# ButterfliesEnvLarge
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000

# Tetris
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000

# # Maze
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0

# # Butterflies
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0

# MazeEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 

# Maze2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0

# # ButterfliesEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5

# MinAtar/Breakout
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000

# # MinAtar/SpaceInvaders
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000