## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smin gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smax gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms none gaussian 1

sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smin bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-inverse bernoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none gaussian 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none gaussian 1 --noisy-room=1