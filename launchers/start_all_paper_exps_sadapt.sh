
# # ####### DQN #######

# # MazeEnvLarge
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --buffer-size=100_000 --use_surprise=0 

# ButterfliesEnvLarge
<<<<<<< HEAD
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=15_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=15_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=15_000_000 --exploration-fraction=0.5
=======
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5
>>>>>>> a3dd5bb487da195abd0ea5d728c855d42a278309

# Tetris
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --total-timesteps=10_000_000

# # Maze
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0

# # Butterflies
<<<<<<< HEAD
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0

# MazeEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=15_000_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=15_000_000 --use_surprise=0 
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=15_000_000 --use_surprise=0 

# Maze2
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0
=======
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnv sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --use_surprise=1

# MazeEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 --exploration-fraction=0.6 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 --exploration-fraction=0.6 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 --exploration-fraction=0.6 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 --exploration-fraction=0.6 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnvLarge2 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track  --total-timesteps=10_000_000 --use_surprise=0 --exploration-fraction=0.6 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"

# Maze2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-MazeEnv2 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli"
>>>>>>> a3dd5bb487da195abd0ea5d728c855d42a278309

# # ButterfliesEnvLarge2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli_butterflies"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli_butterflies"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli_butterflies"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge4 sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli_butterflies"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py griddly-ButterfliesEnvLarge6 sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=0 --track --total-timesteps=15_000_000 --use_surprise=0 --exploration-fraction=0.5 --wandb-project-name="sadapt_bandit_bernoulli_butterflies"

# MinAtar/Breakout
<<<<<<< HEAD
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000

# # MinAtar/SpaceInvaders
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
# sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=0 --track  --use_surprise=0 --total-timesteps=10_000_000
=======
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Breakout sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"

# MinAtar/SpaceInvaders
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli_space_invaders_frame-stacking"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli_with_ext" --add-true-rew=1 --int_rew_scale=0.1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli_with_ext" --add-true-rew=1 --int_rew_scale=0.1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli_with_ext" --add-true-rew=1 --int_rew_scale=0.1
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/SpaceInvaders sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli_with_ext" --add-true-rew=1 --int_rew_scale=0.1

# MinAtar/Freeway
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Freeway sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"

# MinAtar/Seaquest
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Seaquest sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2 --wandb-project-name="sadapt_bandit_bernoulli"

sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Asterix sadapt-bandit bernoulli 1 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Asterix sadapt-bandit bernoulli 2 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Asterix sadapt-bandit bernoulli 3 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Asterix sadapt-bandit bernoulli 4 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2
sbatch launchers/train_cleanrl_roger_long scripts/cleanrl_dqn.py MinAtar/Asterix sadapt-bandit bernoulli 5 --scale-by-std=1 --soft_reset=1 --track  --use_surprise=0 --total-timesteps=10_000_000 --exploration-fraction=0.5 --ucb_coeff=2




>>>>>>> a3dd5bb487da195abd0ea5d728c855d42a278309
