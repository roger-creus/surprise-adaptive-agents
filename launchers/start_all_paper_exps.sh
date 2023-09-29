## env-id ## model ## buffer-type

# REWARD FREE EXPERIMENTS
sbatch launchers/train_cleanrl FourRooms smin gaussian 1
sbatch launchers/train_cleanrl FourRooms smax gaussian 1
sbatch launchers/train_cleanrl FourRooms sadapt gaussian 1
sbatch launchers/train_cleanrl FourRooms sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl FourRooms none gaussian 1

sbatch launchers/train_cleanrl tetris smin bernoulli 1
sbatch launchers/train_cleanrl tetris smax bernoulli 1
sbatch launchers/train_cleanrl tetris sadapt bernoulli 1
sbatch launchers/train_cleanrl tetris sadapt-inverse bernoulli 1
sbatch launchers/train_cleanrl tetris none bernoulli 1

sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smin gaussian 1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smax gaussian 1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt gaussian 1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 none gaussian 1

sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smin gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smax gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --noisy-room=1
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 none gaussian 1 --noisy-room=1

# EXPERIMENTS WITH EXTRINSIC REWARDS AS WELL 
sbatch launchers/train_cleanrl_long FourRooms smin gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long FourRooms smax gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long FourRooms sadapt gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long FourRooms sadapt-inverse gaussian 1 --add-true-rew=True

sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smin gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smax gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt gaussian 1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --add-true-rew=True

sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smin gaussian 1 --noisy-room=1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 smax gaussian 1 --noisy-room=1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt gaussian 1 --noisy-room=1 --add-true-rew=True
sbatch launchers/train_cleanrl_long SurpriseAdaptRooms-v0 sadapt-inverse gaussian 1 --noisy-room=1 --add-true-rew=True