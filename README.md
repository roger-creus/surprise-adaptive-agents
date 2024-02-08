# experiment running

Vanilla PPO:
python scripts/cleanrl_ppo.py --env-id="griddly-ButterfliesEnv" --model="none" --buffer-type="gaussian" --scale-by-std=0 --soft_reset=0 --track

SMIN
python scripts/cleanrl_ppo.py --env-id="griddly-ButterfliesEnv" --model="none" --buffer-type="gaussian" --scale-by-std=0 --soft_reset=0 --track

SMAX
python scripts/cleanrl_ppo.py --env-id="griddly-ButterfliesEnv" --model="smax" --buffer-type="gaussian" --scale-by-std=0 --soft_reset=0 --track




# Bayesian Surprise

Repo for environments, gym wrappers, and scripts for the SMiRL project.


## Requirements:

- For distributing experiments.

doodad: https://github.com/montrealrobotics/doodad

- RL library

rlkit: https://github.com/Neo-X/rlkit/tree/surprise

### Build Instruction

```
conda create --name surprise_adapt python=3.7 pip 
conda activate surprise_adapt
pip install -r requirements.txt
pip install -e ./
cd ../
```

If you do not currently have doodad installed

```
git clone git@github.com:montrealrobotics/doodad.git
cd doodad
```

Otherwise
```
cd doodad
git pull -r
```

Finally,
```
pip install -e ./
cd ../surprise-adaptive-agents
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then you will need copy the [`config.py`](https://github.com/Neo-X/doodad/blob/master/doodad/easy_launch/config.py) file locally to `launchers.config.py` and update the paths in the file. 
You need to update `BASE_CODE_DIR` to the location you have saved SMiRL_Code.
Also update `LOCAL_LOG_DIR` to the location you would like the logging data to be saved on your computer.
You can look at the [doodad](https://github.com/Neo-X/doodad/) for more details on this configuration.

## Logging:

To log with CometML, you need to set the following variables in your `config_private.py` file:

```
COMET_API_KEY = <YOUR API KEY FROM COMET ML>
COMET_PROJECT_NAME = <PROJECT NAME>
COMET_WORKSPACE = <PROJECT WORKSPACE NAME>
```

You must specify at the command line to use Comet logging by setting: `--log_comet=true` (must be lowercase)

## Commands:

A basic examples.

```
python3 scripts/dqn_smirl.py --config=configs/tetris_SMiRL.json --run_mode=local --exp_name=test_smirl
```

```
python3 scripts/dqn_smirl.py --config=configs/Carnival_Small_SMiRL.json --run_mode=local --exp_name=test_smirl --training_processor_type=gpu
```
With docker locally
```
python3 scripts/dqn_smirl.py --config=configs/tetris_SMiRL.json --exp-name=test --run_mode=local_docker
```
###Run Vizdoom SMiRL experiments

python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_Small.json --exp_name=vizdoom_small_test --run_mode=ssh --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

 python3 scripts/dqn_smirl.py --config=configs/VizDoom_DefendTheLine_Small.json --exp_name=vizdoom_DTL_small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

 python3 scripts/dqn_smirl.py --config=configs/VizDoom_DefendTheLine_Small_Bonus.json --exp_name=vizdoom_DTL_small_smirl_bonus --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

### Run Atari Experiments

python3 scripts/dqn_smirl.py --config=configs/Carnival_Small_SMiRL.json --exp_name=Atari_Carnival__small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/Carnival_Small_SMiRL_Bonus.json --exp_name=Atari_Carnival_small_smirl_bonus --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/IceHockey_Small_SMiRL.json --exp_name=Atari_IceHockey_small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/RiverRaid_Small_SMiRL.json --exp_name=Atari_RiverRaid_small_smirl --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
