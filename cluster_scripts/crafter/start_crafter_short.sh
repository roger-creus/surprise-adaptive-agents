# Random agent
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/random_agent.json --run_mode=local --exp_name=random_agent  --training_processor_type=gpu --log_comet=true

############# Without normalization layer ################
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA.json --run_mode=local --exp_name=crafter_sa --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN.json --run_mode=local --exp_name=crafter_smin --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX.json --run_mode=local --exp_name=crafter_smax --training_processor_type=gpu --log_comet=true

# # With scaling reward by the std
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward.json --run_mode=local --exp_name=crafter_sa_rescale_reward --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward.json --run_mode=local --exp_name=crafter_smin_rescale_reward --training_processor_type=gpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward.json --run_mode=local --exp_name=crafter_smax_rescale_reward --training_processor_type=gpu --log_comet=true

# Without softReset
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_no_softreset.json --run_mode=local --exp_name=crafter_sa_no_softreset --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_no_softreset.json --run_mode=local --exp_name=crafter_smin_no_softreset --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_no_softreset.json --run_mode=local --exp_name=crafter_smax_no_softreset --training_processor_type=gpu --log_comet=true

# Without softReset and reward scaling by stf
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_sa_rescale_reward_no_softreset --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_smin_rescale_reward_no_softreset --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_smax_rescale_reward_no_softreset --training_processor_type=gpu --log_comet=true
############# Without normalization layer ################




############# With normalization layer ################
## SA
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_norm_layer.json --run_mode=local --exp_name=crafter_sa_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_sa_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_sa_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_sa_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
## SMIN
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_norm_layer.json --run_mode=local --exp_name=crafter_smin_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_smin_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_smin_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_smin_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
## SMAX
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_norm_layer.json --run_mode=local --exp_name=crafter_smax_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_norm_layer.json --run_mode=local --exp_name=crafter_smax_rescale_reward_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_smax_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_rescale_reward_no_softreset_norm_layer.json --run_mode=local --exp_name=crafter_rescale_reward_smax_no_softreset_norm_layer --training_processor_type=gpu --log_comet=true
############# With normalization layer ################



# Use an MLP with smaller observation size without layer normalization 
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp.json --run_mode=local --exp_name=crafter_sa_mlp --training_processor_type=cpu --log_comet=true
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_rescale_reward.json --run_mode=local --exp_name=crafter_sa_mlp_rescale_reward --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_rescale_reward_no_softreset.json --run_mode=local --exp_name=crafter_sa_mlp_rescale_reward_no_softreset --training_processor_type=cpu --log_comet=true
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_mlp_no_softreset.json --run_mode=local --exp_name=crafter_sa_mlp_no_softreset --training_processor_type=cpu --log_comet=true