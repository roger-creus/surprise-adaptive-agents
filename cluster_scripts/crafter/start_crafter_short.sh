

# # # Smin
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN.json --run_mode=local --exp_name=crafter_smin_short_discounting  --training_processor_type=gpu --log_comet=true
# # # Smax
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX.json --run_mode=local --exp_name=crafter_smax_short_discounting  --training_processor_type=gpu --log_comet=true
# # # Sa
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA.json --run_mode=local --exp_name=crafter_sa_short_discounting  --training_processor_type=gpu --log_comet=true

# # Sa long
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_long.json --run_mode=local --exp_name=crafter_sa_long_discounting  --log_comet true --training_processor_type=gpu
# # # Smin long
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_long.json --run_mode=local --exp_name=crafter_smin_long_discounting  --log_comet=true --training_processor_type=gpu
# # Smaxlong
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_long.json --run_mode=local --exp_name=crafter_smax_long_discounting  --log_comet true --training_processor_type=gpu


####### Larger replay buffer ############

# # Smin
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_large_replay_buffer.json --run_mode=local --exp_name=crafter_smin_short_large_replay_eps_greedy  --training_processor_type=gpu --log_comet=true
# # # Smax
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_large_replay_buffer.json --run_mode=local --exp_name=crafter_smax_short_large_replay_eps_greedy  --training_processor_type=gpu --log_comet=true
# # # Sa
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_large_replay_buffer.json --run_mode=local --exp_name=crafter_sa_short_large_replay_eps_greedy  --training_processor_type=gpu --log_comet=true

# ####### Larger replay buffer ############

# ####### More training steps per env steps ############

# # # Smin
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_more_train_steps.json --run_mode=local --exp_name=crafter_smin_short_more_train_steps_eps_greedy  --training_processor_type=gpu --log_comet=true
# # # Smax
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_more_train_steps.json --run_mode=local --exp_name=crafter_smax_short_more_train_steps_eps_greedy  --training_processor_type=gpu --log_comet=true
# # # Sa
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_more_train_steps.json --run_mode=local --exp_name=crafter_sa_short_more_train_steps_eps_greedy  --training_processor_type=gpu --log_comet=true

####### More training steps per env steps ############


####### More training steps per env steps and large replay ############
# you need to modify the config files
# # Smin
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_more_train_steps.json --run_mode=local --exp_name=crafter_smin_short_more_train_steps_large_replay  --training_processor_type=gpu --log_comet=true
# # # Smax
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_more_train_steps.json --run_mode=local --exp_name=crafter_smax_short_more_train_steps_large_replay  --training_processor_type=gpu --log_comet=true
# # # Sa
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_more_train_steps.json --run_mode=local --exp_name=crafter_sa_short_more_train_steps_large_replay  --training_processor_type=gpu --log_comet=true

####### More training steps per env steps and large replay ############


## online

# Sadapt
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_large_replay_buffer_online.json --run_mode=local --exp_name=crafter_sa_short_large_replay_eps_greedy_online  --training_processor_type=gpu --log_comet=true

# Smin
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_large_replay_buffer_online.json --run_mode=local --exp_name=crafter_smin_short_large_replay_eps_greedy_online  --training_processor_type=gpu --log_comet=true

# Smax
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_large_replay_buffer_online.json --run_mode=local --exp_name=crafter_smax_short_large_replay_eps_greedy_online  --training_processor_type=gpu --log_comet=true

