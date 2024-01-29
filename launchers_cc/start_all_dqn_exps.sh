# Runs to repeat
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=1
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=0

# # Crafter on extrinsic reward w/ soft-reset
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=1 --track=True --soft_reset=1
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=820 --track=True --soft_reset=1

# Crafter on extrinsic reward w/o soft-reset
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=1 --track=True --soft_reset=0
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=8943 --track=True --soft_reset=0
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=820 --track=True --soft_reset=0