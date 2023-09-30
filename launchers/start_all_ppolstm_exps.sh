## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms smin gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms smax gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms sadapt gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms none gaussian 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smin gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smax gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 none gaussian 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smin gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smax gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 none gaussian 1 --noisy-room=1