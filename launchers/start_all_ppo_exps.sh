## env-id ## model ## buffer-type

# Crafter test
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter smax gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=1 --soft_reset=0 
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter smin gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=1 --soft_reset=0
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter none gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=0 --soft_reset=0

sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter smax gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=0 --soft_reset=1 
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter smin gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=0 --soft_reset=1
sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py crafter none gaussian 1  --track --wandb-project-name="Crafter_PPO" --scale-by-std=0 --soft_reset=1

# random crafter agnet
sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 1 --track --wandb-project-name="Crafter_PPO" --scale-by-std=1 --soft_reset=0
sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 1 --track --wandb-project-name="Crafter_PPO" --scale-by-std=0 --soft_reset=1

# REWARD FREE EXPERIMENTS
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smin multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms smax multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py FourRooms none multinoulli 1

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smin bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py tetris none bernoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none multinoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smin multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 smax multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py SurpriseAdaptRooms-v0 none multinoulli 1 --noisy-room=1

# Minatar exps

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders smin bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/SpaceInvaders none bernoulli 1

# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Freeway smin bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Freeway smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Freeway sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Freeway sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_ppo.py MinAtar/Freeway none bernoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Breakout smin bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Breakout smax bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Breakout sadapt bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Breakout sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Breakout none bernoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Seaquest smin bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Seaquest smax bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Seaquest sadapt bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Seaquest sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_ppo.py MinAtar/Seaquest none bernoulli 1