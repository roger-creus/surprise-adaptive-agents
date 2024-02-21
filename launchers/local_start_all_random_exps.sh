# # MazeEnvLarge
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 
wait

# # # ButterfliesEnvLarge
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 
wait

# tetris
python scripts/random_agent.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=tetris --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 
wait

# # Maze
python scripts/random_agent.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnv --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 
wait

# # Butterflies
python scripts/random_agent.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnv --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait

# MazeEnvLarge2
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge2 --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge2 --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnvLarge2 --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait

# Maze2
python scripts/random_agent.py --env-id=griddly-MazeEnv2 --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnv2 --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-MazeEnv2 --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait

# # ButterfliesEnvLarge2
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge2 --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge2 --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=griddly-ButterfliesEnvLarge2 --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait

######## MINATAR ########

# MinAtar/Breakout
python scripts/random_agent.py --env-id=MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=MinAtar/Breakout --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait

# # --env-id=MinAtar/SpaceInvaders
python scripts/random_agent.py --env-id=MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100 &
python scripts/random_agent.py --env-id=MinAtar/SpaceInvaders --model=none --buffer-type=bernoulli --seed=3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 --buffer-size=100
wait
