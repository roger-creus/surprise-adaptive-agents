

# # MazeEnvLarge
python scripts/random_agent.py griddly-MazeEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 
wait

# # # ButterfliesEnvLarge
python scripts/random_agent.py griddly-ButterfliesEnvLarge none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnvLarge none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnvLarge none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 
wait

# # Tetris
python scripts/random_agent.py tetris none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py tetris none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py tetris none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 
wait

# # Maze
python scripts/random_agent.py griddly-MazeEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 
wait

# # Butterflies
python scripts/random_agent.py griddly-ButterfliesEnv none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnv none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnv none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
wait

# MazeEnvLarge2
python scripts/random_agent.py griddly-MazeEnvLarge2 none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnvLarge2 none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnvLarge2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
wait

# Maze2
python scripts/random_agent.py griddly-MazeEnv2 none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnv2 none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-MazeEnv2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
wait

# # ButterfliesEnvLarge2
python scripts/random_agent.py griddly-ButterfliesEnvLarge2 none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnvLarge2 none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py griddly-ButterfliesEnvLarge2 none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
wait

######## MINATAR ########

# MinAtar/Breakout
python scripts/random_agent.py MinAtar/Breakout none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py MinAtar/Breakout none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py MinAtar/Breakout none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
wait

# # MinAtar/SpaceInvaders
python scripts/random_agent.py MinAtar/SpaceInvaders none bernoulli 1 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py MinAtar/SpaceInvaders none bernoulli 2 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000 &
python scripts/random_agent.py MinAtar/SpaceInvaders none bernoulli 3 --scale-by-std=0 --soft_reset=0 --total-timesteps=10_000
