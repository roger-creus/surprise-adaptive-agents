# [Surprise-Adaptive Intrinsic Motivation for Unsupervised Reinforcement Learning](https://rlj.cs.umass.edu/2024/papers/Paper77.html)

Abstract: Both entropy-minimizing and entropy-maximizing objectives for unsupervised reinforcement learning (RL) have been shown to be effective in different environments, depending on the environment's level of natural entropy. However, neither method alone results in an agent that will consistently learn intelligent behavior across environments. In an effort to find a single entropy-based method that will encourage emergent behaviors in any environment, we propose an agent that can adapt its objective online, depending on the entropy conditions it faces in the environment, by framing the choice as a multi-armed bandit problem. We devise a novel intrinsic feedback signal for the bandit, which captures the agent's ability to control the entropy in its environment. We demonstrate that such agents can learn to optimize task returns through entropy control alone in didactic environments for both high- and low-entropy regimes and learn skillful behaviors in certain benchmark tasks.
For videos check the [project website](https://sites.google.com/view/surprise-adaptive-agents?usp=sharing)

<p align="center"><img src="sadapt.jpg" width=80%></p>

### Installation 
```bash
# clone the repo 
git clone https://github.com/roger-creus/surprise-adaptive-agents.git
cd surprise-adaptive-agents
# install requirement via conda
conda create --name s_adapt python=3.9.18
pip install -r requirements.txt
pip install -e .
```
### Running Experiments
wip


