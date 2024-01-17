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

# Random agent
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/random_agent.json --run_mode=local --exp_name=random_agent  --training_processor_type=gpu --log_comet=true