# Runs to repeat
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=0

# # Crafter on extrinsic reward w/ soft-reset
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=1 --track=True --soft_reset=1
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=820 --track=True --soft_reset=1

# # Crafter on extrinsic reward w/o soft-reset
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=1 --track=True --soft_reset=0
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=8943 --track=True --soft_reset=0
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=none --buffer-type=gaussian --seed=820 --track=True --soft_reset=0

# small theta size
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize" --scale-by-std=0
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize" --scale-by-std=0
sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize" --scale-by-std=0

# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"

# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=1 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"

# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl random_agent.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=0 --theta_size="(9,9)" --wandb-project-name="Crafter_DQN_small_thetaSize"

# small theta and obs size
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=1 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=1 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=1 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"

# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=8943 --track=True --soft_reset=0 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=1 --track=True --soft_reset=0 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"
# sbatch launchers_cc/train_cleanrl cleanrl_dqn.py --env-id=crafter --model=sadapt-bandit --buffer-type=gaussian --seed=820 --track=True --soft_reset=0 --theta_size="(9,9)" --obs_size="(20, 26)" --wandb-project-name="Crafter_DQN_small_thetaSize"
