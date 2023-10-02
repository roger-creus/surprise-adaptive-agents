## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smin multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smax multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt-inverse multinoulli 1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms none multinoulli 1

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smin bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none multinoulli 1

sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1 --noisy-room=1
sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none multinoulli 1 --noisy-room=1