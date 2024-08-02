
# # ####### DQN #######

# # MazeEnvLarge
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 

# ButterfliesEnvLarge
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=15_000_000 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=15_000_000 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=15_000_000 --exploration_fraction=0.5 --wandb_project_name="paper_release"

# Tetris
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0 --total_timesteps=10_000_000

# # Maze
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0

# # Butterflies
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0

# MazeEnvLarge2
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 4 --normalize_int_reward=1 --soft_reset=0 --track  --total_timesteps=15_000_000 --use_surprise=0 --wandb_project_name="paper_release" --exploration_fraction=0.5
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 5 --normalize_int_reward=1 --soft_reset=0 --track  --total_timesteps=15_000_000 --use_surprise=0 --wandb_project_name="paper_release" --exploration_fraction=0.5
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 6 --normalize_int_reward=1 --soft_reset=0 --track  --total_timesteps=15_000_000 --use_surprise=0 --wandb_project_name="paper_release" --exploration_fraction=0.5

# Maze2
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --use_surprise=0

# # ButterfliesEnvLarge2
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track --total_timesteps=15_000_000 --use_surprise=0 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track --total_timesteps=15_000_000 --use_surprise=0 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track --total_timesteps=15_000_000 --use_surprise=0 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge4 sadapt-bandit bernoulli 4 --normalize_int_reward=1 --soft_reset=0 --track --total_timesteps=15_000_000 --use_surprise=0 --exploration_fraction=0.5 --wandb_project_name="paper_release"
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 5 --normalize_int_reward=1 --soft_reset=0 --track --total_timesteps=15_000_000 --use_surprise=0 --exploration_fraction=0.5 --wandb_project_name="paper_release"

# MinAtar/Breakout
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000

# # MinAtar/SpaceInvaders
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --normalize_int_reward=1 --soft_reset=0 --track  --use_surprise=0 --total_timesteps=10_000_000
