

# Tetris (Faisal)
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py tetris smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# Maze (Faisal)
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-MazeEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# Butterflies (Roger)
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py griddly-ButterfliesEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# MinAtar/Breakout (Faisal)
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/Breakout smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# MinAtar/SpaceInvaders (Roger)
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smin bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
sbatch launchers/train_cleanrl_roger scripts/cleanrl_dqn.py MinAtar/SpaceInvaders smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track
