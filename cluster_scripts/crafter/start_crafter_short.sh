

# Smin
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN.json --run_mode=local --exp_name=crafter_smin_short_thresh  --log_comet true --training_processor_type=gpu
# Smax
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX.json --run_mode=local --exp_name=crafter_smax_short_thresh  --log_comet true --training_processor_type=gpu
# Sa
sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA.json --run_mode=local --exp_name=crafter_sa_short_thresh  --log_comet true --training_processor_type=gpu

# # Sa long
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SA_long.json --run_mode=local --exp_name=crafter_sa_long  --log_comet true --training_processor_type=gpu
# # Smin long
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMIN_long.json --run_mode=local --exp_name=crafter_smin_long  --log_comet true --training_processor_type=gpu
# Smaxlong
# sbatch cluster_scripts/crafter/train scripts/dqn_smirl.py --config=configs/crafter/crafter_SMAX_long.json --run_mode=local --exp_name=crafter_smax_long  --log_comet true --training_processor_type=gpu