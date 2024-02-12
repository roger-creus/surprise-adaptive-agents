
# ####### DQN #######

# Grafter
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py grafter sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500

# MazeEnvLarge
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"

# ButterfliesEnvLarge
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --video_freq=500 --buffer-size=100_000 --wandb-project-name="sadapt-bandit"

# # Tetris
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # Maze
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # Butterflies
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # MinAtar/Breakout

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"

# # MinAtar/SpaceInvaders

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --wandb-project-name="sadapt-bandit"



# ####### PPO #######
# Tetris
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # Maze
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-MazeEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # Butterflies
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py griddly-ButterfliesEnv smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # MinAtar/Breakout
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Breakout smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# # MinAtar/SpaceInvaders
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders none bernoulli 1 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders none bernoulli 2 --scale-by-std=0 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders none bernoulli 3 --scale-by-std=0 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders smax bernoulli 1 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders smax bernoulli 2 --scale-by-std=1 --soft_reset=0 --track
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders smax bernoulli 3 --scale-by-std=1 --soft_reset=0 --track
