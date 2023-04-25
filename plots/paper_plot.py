import json
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed

# read json file
with open('tetris_task_reward.json') as f:
    data = json.load(f)

runs_name = [d["name"] for d in data]
y_axis = np.array(np.array([d["y"] for d in data]))
x_axis = np.array(np.array([d["x"] for d in data]))

plt.figure()

for run in range(len(x_axis)):
    clean_name = runs_name[run].split(">")[-1]
    clean_name = clean_name.replace("_500len", "")
    clean_name = clean_name.replace("SA_", "")
    clean_name = clean_name.replace("Tetris_", "")
    
    plt.plot(x_axis[run], y_axis[run], label=clean_name)


plt.title("Tetris Task Reward")
plt.xlabel("Epochs")
plt.ylabel("Task Reward")
plt.legend()
plt.savefig('tetris_reward.png')