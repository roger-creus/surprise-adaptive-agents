import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from torch.distributions.categorical import Categorical

class TetrisQNetwork(nn.Module):
    def __init__(self, env, use_theta=False):
        super().__init__()
        n_inputs = env.single_observation_space["obs"].shape[-1]
        policy_inputs = 84
        
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        
        if use_theta:
            self.theta_fc = nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space["theta"].shape), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
            )
            policy_inputs += 84
            
        self.q_net = nn.Linear(policy_inputs, env.single_action_space.n)
        self.use_theta = use_theta

    def forward(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        obs_features = self.network(x_.float())
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([obs_features, theta_features], dim=1)
        else:
            x = obs_features

        return self.q_net(x)

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
                nn.Linear(np.prod(env.single_observation_space["theta"].shape), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
            )
            
        with torch.no_grad():
            s_ = env.single_observation_space["obs"].sample()[None]
            n_flatten = self.network(torch.as_tensor(s_).float().permute(0,3,1,2)).shape[1]

        if use_theta:
            n_flatten += 84

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
    
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MinigridPPOAgent(nn.Module):
    def __init__(self, env, use_theta=False):
        super().__init__()
        
        self.use_theta = use_theta
        n_input_channels = env.single_observation_space["obs"].shape[-1]
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, (2,2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, (2,2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2,2))),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        if use_theta:
            self.theta_fc = nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space["theta"].shape), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
            )
            
        with torch.no_grad():
            s_ = env.single_observation_space["obs"].sample()[None]
            n_flatten = self.network(torch.as_tensor(s_).float().permute(0,3,1,2)).shape[1]
        
        if use_theta:
            n_flatten += 84
            
        self.actor = layer_init(nn.Linear(n_flatten, env.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(n_flatten, 1), std=1)

    def get_value(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        img_features = self.network(x_.permute(0,3,1,2).float())
        
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features
            
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x_ = x["obs"]
        y_ = x["theta"]
        img_features = self.network(x_.permute(0,3,1,2).float())
        
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features
        
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class TetrisPPOAgent(nn.Module):
    def __init__(self, env, use_theta = False):
        super().__init__()
        n_inputs = env.single_observation_space["obs"].shape[-1]
        self.use_theta = use_theta
        
        if use_theta:
            n_inputs += np.prod(env.single_observation_space["theta"].shape)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_inputs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        
        if self.use_theta:
            x = torch.cat([x_, y_], dim=1)
        else:
            x = x_
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x_ = x["obs"]
        y_ = x["theta"]
        
        if self.use_theta:
            x = torch.cat([x_, y_], dim=1)
        else:
            x = x_
        
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
