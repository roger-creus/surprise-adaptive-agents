

# # MazeEnvLarge
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # # ButterfliesEnvLarge
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# # Tetris
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# # Maze
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # Butterflies
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# MazeEnvLarge2
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 none bernoulli 4 --scale-by-std=0 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 none bernoulli 5 --scale-by-std=0 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track  --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smin bernoulli 4 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smin bernoulli 5 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smax bernoulli 4 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smax bernoulli 5 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000

# Maze2
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 none bernoulli 4 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 none bernoulli 5 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smin bernoulli 4 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smin bernoulli 5 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smax bernoulli 4 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smax bernoulli 5 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # ButterfliesEnvLarge2
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 none bernoulli 4 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 none bernoulli 5 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smin bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smin bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5

# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smax bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smax bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge2 smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --exploration-fraction=0.5

######## MINATAR ########

# MinAtar/Breakout
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

# # MinAtar/SpaceInvaders
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

# # MinAtar/Freeway
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway smin bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway smin bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway smax bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway smax bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

# # MinAtar/Seaquest
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest smin bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest smin bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest smax bernoulli 1 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest smax bernoulli 2 --scale-by-std=1 --soft_reset=1 --track --total-timesteps=10_000_000 --exploration-fraction=0.5
