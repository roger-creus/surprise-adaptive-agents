# # Sadapt
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA.json --run_mode=local --exp_name=crafter_sa --training_processor_type=gpu --log_comet=true

# # Smin
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN.json --run_mode=local --exp_name=crafter_smin --training_processor_type=gpu --log_comet=true

# # Smax
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX.json --run_mode=local --exp_name=crafter_smax --training_processor_type=gpu --log_comet=true

# Random agent
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/random_agent.json --run_mode=local --exp_name=random_agent  --training_processor_type=gpu --log_comet=true

# With scaling reward by the std


sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_rescale_reward.json --run_mode=local --exp_name=crafter_sa_rescale_reward --training_processor_type=gpu --log_comet=true