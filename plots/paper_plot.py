import json
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed

# read json file
with open('/home/roger/Desktop/surprise-adaptive-agents/plots/tetris_reward.json') as f:
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
    clean_name = clean_name.replace("_75window", "")

    if clean_name == "01threshold":
        clean_name = "S-Adapt"
    
    if clean_name == "SMiRL":
        clean_name = "S-Min"
    
    if clean_name == "test":
        clean_name = "FixedAlphasTraining"
    
    x = np.array(x_axis[run])[:100]
    y = np.array(y_axis[run])[:100]

    plt.plot(x, y, label=clean_name)


plt.title("Tetris Reward")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.legend()
plt.savefig('tetris_reward.png')