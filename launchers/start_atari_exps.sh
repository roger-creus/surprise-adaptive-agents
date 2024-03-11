# Freeway
# Extrinsic 
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway none gaussian 1 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway none gaussian 2 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway none gaussian 3 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000

# Sadapt
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway sadapt-bandit gaussian 1 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway sadapt-bandit gaussian 2 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway sadapt-bandit gaussian 3 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000

# SMax
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smax gaussian 1 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smax gaussian 2 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smax gaussian 3 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000

# SMin
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smin gaussian 1 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smin gaussian 2 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py Atari-Freeway smin gaussian 3 --scale-by-std=1 --soft_reset=0 --track --theta_size="(32,32)" --exploration-fraction=0.3 --wandb-project-name="sadapt_bandit_atari" --total-timesteps=50_000_000


