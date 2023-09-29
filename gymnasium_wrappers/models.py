import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


# ALGO LOGIC: initialize agent here:
class MinigridQNetwork(nn.Module):
    def __init__(self, env, use_theta=False):
        super().__init__()
        n_input_channels = env.single_observation_space["obs"].shape[-1]
   
        self.network = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        if use_theta:
            self.theta_fc = nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space["theta"].shape), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            
        with torch.no_grad():
            s_ = env.single_observation_space["obs"].sample()[None]
            n_flatten = self.network(torch.as_tensor(s_).float().permute(0,3,1,2)).shape[1]

        if use_theta:
            n_flatten += 64

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512), 
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n), 
        )
        
        self.use_theta = use_theta

    def forward(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        img_features = self.network(x_.permute(0,3,1,2).float())
        
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features

        return self.linear(x)