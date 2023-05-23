import json
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed

# read json file
with open('/home/roger/Desktop/surprise-adaptive-agents/plots/tetris_surprise_eval.json') as f:
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

    if clean_name == "SA":
        clean_name = "S-Adapt"
    
    if clean_name == "SMiRL":
        clean_name = "S-Min"
    
    plt.plot(x_axis[run], y_axis[run], label=clean_name)


plt.title("Tetris Surprise")
plt.xlabel("Epochs")
plt.ylabel("Surprise")
plt.legend()
plt.savefig('tetris_surprise.png')
