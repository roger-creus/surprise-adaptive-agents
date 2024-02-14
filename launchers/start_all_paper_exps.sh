
# ####### DQN #######

# Grafter
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py grafter smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

# MazeEnvLarge
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnvLarge smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# ButterfliesEnvLarge
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# Tetris
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=10_000_000

# Maze
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# Butterflies
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# MinAtar/Breakout
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # MinAtar/SpaceInvaders
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track