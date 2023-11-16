#!/bin/bash
#SBATCH --partition=lab-real                               # Ask for unkillable job
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G                                                # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                          # Ask for 6 CPUs
#SBATCH -o /network/scratch/a/adriana.knatchbull-hugessen/slurm-%j.out  # Write the log on scratch

# 1. Load your environment
module load miniconda/3

conda activate surprise_adapt

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python3 scripts/dqn_smirl.py --config=configs/tetris_SAv2deltabin.json --run_mode=local --exp_name=SAv2deltabin_final --training_processor_type=cpu --log_comet=true --random_seeds=3 --meta_sim_threads=3

cp $SLURM_TMPDIR/output/ $SCRATCH/surprise-adaptive-agents/output
cp $SLURM_TMPDIR/output/ $SCRATCH/surprise-adaptive-agents/logs
