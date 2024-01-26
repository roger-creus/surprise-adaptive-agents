## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smin multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms smax multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py FourRooms none multinoulli 1

# # Tetris S-min
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 8943 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smin bernoulli 820 --track=True

# # Tetris Extrinsic
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 8943 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 820 --track=True

# Tetris Bandit
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 8943 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-bandit bernoulli 820 --track=True

# Crafter 
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 1 --track=True --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 8943 --track=True --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter sadapt-bandit gaussian 820 --track=True --soft_reset=1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter smin gaussian 1 --track=True
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py crafter none gaussian 1 --track=True

# random crafter agnet
sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 1 --track=True --wandb-project-name="Crafter_DQN" 
sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 8943 --track=True --wandb-project-name="Crafter_DQN"
sbatch launchers/train_cleanrl scripts/random_agent.py crafter none gaussian 820 --track=True --wandb-project-name="Crafter_DQN"

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris smax bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris sadapt-inverse bernoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py tetris none bernoulli 1

# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax multinoulli 1
# sbatch launchers/train_cleanrl scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none multinoulli 1

# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smin multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 smax multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 sadapt-inverse multinoulli 1 --noisy-room=1
# sbatch launchers/train_cleanrl_long scripts/cleanrl_dqn.py SurpriseAdaptRooms-v0 none multinoulli 1 --noisy-room=1