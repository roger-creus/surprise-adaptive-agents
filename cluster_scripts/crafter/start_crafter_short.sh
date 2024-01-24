# Random agent w/ softreset
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/random_agent.json --run_mode=local --exp_name=random_agent_less_time  --training_processor_type=gpu --log_comet=true

# Random agent w/o softreset
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/random_agent_no_softreset.json --run_mode=local --exp_name=random_agent_no_softreset_less_time  --training_processor_type=gpu --log_comet=true

# DQN w/ softreset
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_DQN_softreset.json --run_mode=local --exp_name=DQN_softreset_less_time  --training_processor_type=gpu --log_comet=true
# DQN w/o softreset
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_DQN_no_softreset.json --run_mode=local --exp_name=DQN_no_softreset_less_time  --training_processor_type=gpu --log_comet=true

############# Without normalization layer ################
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA.json --run_mode=local --exp_name=crafter_sa_more_grad_steps_norm_time_less_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN.json --run_mode=local --exp_name=crafter_smin --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX.json --run_mode=local --exp_name=crafter_smax --training_processor_type=gpu --log_comet=true

# # With scaling reward by the std
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward.json --run_mode=local --exp_name=crafter_sa_rescale_reward_less_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward.json --run_mode=local --exp_name=crafter_smin_rescale_reward --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward.json --run_mode=local --exp_name=crafter_smax_rescale_reward --training_processor_type=gpu --log_comet=true

# Without softReset
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_no_softreset.json --run_mode=local --exp_name=crafter_sa_no_softreset_more_grad_steps_norm_time_less_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_no_softreset.json --run_mode=local --exp_name=crafter_smin_no_softreset --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_no_softreset.json --run_mode=local --exp_name=crafter_smax_no_softreset --training_processor_type=gpu --log_comet=true

# # Without softReset and reward scaling by std
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_sa_rescale_reward_no_softreset_less_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_smin_rescale_reward_no_softreset --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_smax_rescale_reward_no_softreset --training_processor_type=gpu --log_comet=true
############# Without normalization layer ################

# more eps-greedy exploration steps
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_more_exp.json --run_mode=local --exp_name=crafter_sa_more_exp --training_processor_type=gpu --log_comet=true


############# With normalization layer ################
## SA
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_norm_layer.json --run_mode=local --exp_name=crafter_sa_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_sa_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_sa_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_sa_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
# ## SMIN
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_norm_layer.json --run_mode=local --exp_name=crafter_smin_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_smin_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_smin_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_smin_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
# ## SMAX
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_norm_layer.json --run_mode=local --exp_name=crafter_smax_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_smax_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_smax_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_smax_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
############# With normalization layer ################



# Use an MLP with smaller observation size without layer normalization 
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp.json --run_mode=local --exp_name=crafter_sa_mlp_less_time --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_no_softreset.json --run_mode=local --exp_name=crafter_sa_mlp_no_softreset_less_time --training_processor_type=cpu --log_comet=true

# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_mlp.json --run_mode=local --exp_name=crafter_smin_mlp --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_mlp_no_softreset.json --run_mode=local --exp_name=crafter_sa_min_no_softreset --training_processor_type=cpu --log_comet=true

# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_mlp.json --run_mode=local --exp_name=crafter_smax_mlp --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_mlp_no_softreset.json --run_mode=local --exp_name=crafter_smax_mlp_no_softreset --training_processor_type=cpu --log_comet=true


# mlp w/ rescaled reward
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_sa_mlp_no_softreset_rescaled_reward --training_processor_type=cpu --log_comet=true



# Surprise defference
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_sd.json --run_mode=local --exp_name=crafter_sd_mlp --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_sd_no_softreset.json --run_mode=local --exp_name=crafter_sd_mlp_no_softreset --training_processor_type=cpu --log_comet=true



# Small CNN
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_small_cnn.json --run_mode=local --exp_name=crafter_sa_small_cnn --training_processor_type=cpu --log_comet=true


# W/ normalized time-steps in the theta vector 
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_no_softreset_norm_time.json --run_mode=local --exp_name=crafter_sa_no_softreset_norm_time_less_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_norm_time.json --run_mode=local --exp_name=crafter_sa_norm_time_less_time --training_processor_type=gpu --log_comet=true

# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_no_softreset_norm_time.json --run_mode=local --exp_name=crafter_smin_no_softreset_norm_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_norm_time.json --run_mode=local --exp_name=crafter_smin_norm_time --training_processor_type=gpu --log_comet=true

# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_no_softreset_norm_time.json --run_mode=local --exp_name=crafter_smax_no_softreset_norm_time --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_norm_time.json --run_mode=local --exp_name=crafter_smax_norm_time --training_processor_type=gpu --log_comet=true


# random agetn small observation 
# sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/Random_agent_mlp_no_softreset.json --run_mode=local --exp_name=random_mlp_no_softreset --training_processor_type=cpu --log_comet=true


sbatch cluster_scripts/crafter/train_cpu scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_rescale_reward_no_softreset_norm_time_more_grad_steps.json --run_mode=local --exp_name=crafter_sa_mlp_rescale_reward_no_softreset_norm_time_more_grad_steps --training_processor_type=cpu --log_comet=true

sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_no_softreset_norm_time_more_grad_steps.json --run_mode=local --exp_name=crafter_sa_rescale_reward_no_softreset_norm_time_more_grad_steps --training_processor_type=gpu --log_comet=true