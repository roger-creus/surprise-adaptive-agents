sbatch launchers/train scripts/cleanrl_dqn_crafter.py --model=smin --buffer-type=gaussian --seed=1 --train-frequency=4 --exploration-fraction=0.3 --env-id=MountainCar --batch-size=256

