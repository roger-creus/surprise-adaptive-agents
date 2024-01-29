## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms smin multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms smax multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms sadapt multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms sadapt-inverse multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo_lstm.py FourRooms none multinoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smin multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smax multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 none multinoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smin multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 smax multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo_lstm.py SurpriseAdaptRooms-v0 none multinoulli 1 --noisy-room=1