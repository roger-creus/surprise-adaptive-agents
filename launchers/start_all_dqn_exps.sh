## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smin gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smax gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms none gaussian 1

sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smax bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-inverse bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none gaussian 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none gaussian 1 --noisy-room=1