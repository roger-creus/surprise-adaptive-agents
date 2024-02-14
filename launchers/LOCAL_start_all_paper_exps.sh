
# ####### DQN #######

# Grafter
#python scripts/cleanrl_dqn.py grafter --model=none --buffer-type=bernoulli 1 --scale-by-std=0 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=none --buffer-type=bernoulli 2 --scale-by-std=0 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=none --buffer-type=bernoulli 3 --scale-by-std=0 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

#python scripts/cleanrl_dqn.py grafter --model=--model=smin --buffer-type=bernoulli 1 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=--model=smin --buffer-type=bernoulli 2 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=--model=smin --buffer-type=bernoulli 3 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

#python scripts/cleanrl_dqn.py grafter --model=smax --buffer-type=bernoulli 1 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=smax --buffer-type=bernoulli 2 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000
#python scripts/cleanrl_dqn.py grafter --model=smax --buffer-type=bernoulli 3 --scale-by-std=1 --soft_reset=0  --video_freq=500 --buffer-size=500_000 --total-timesteps=10_000_000

# MazeEnvLarge
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnvLarge --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000
wait

# ButterfliesEnvLarge
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnvLarge --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

# Tetris
python scripts/cleanrl_dqn.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=tetris --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=tetris --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=tetris --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --total-timesteps=10_000_000 --buffer-size=250_000 
wait

# Maze
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-MazeEnv --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 
wait

# Butterflies
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 
wait

python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 &
python scripts/cleanrl_dqn.py --env-id=griddly-ButterfliesEnv --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0  --buffer-size=250_000 

# MinAtar/Breakout
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 

# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0 

# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/Breakout --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0 

# # MinAtar/SpaceInvaders
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 

# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smin --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smin --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smin --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0 

# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smax --buffer-type=bernoulli --seed=1 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smax --buffer-type=bernoulli --seed=2 --scale-by-std=1 --soft_reset=0 
# python scripts/cleanrl_dqn.py MinAtar/SpaceInvaders --model=smax --buffer-type=bernoulli --seed=3 --scale-by-std=1 --soft_reset=0 