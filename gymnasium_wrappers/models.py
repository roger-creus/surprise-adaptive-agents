import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from IPython import embed
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
                nn.Linear(np.prod(env.single_observation_space["theta"].shape[-1]), 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
            )
            policy_inputs += 84
            
        self.q_net = nn.Linear(policy_inputs, env.single_action_space.n)
        self.use_theta = use_theta

    def forward(self, x):
        x_ = x["obs"]
        y_ = x["theta"][:, 0, :]
        obs_features = self.network(x_.float())
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([obs_features, theta_features], dim=1)
        else:
            x = obs_features

        return self.q_net(x)
    

class TetrisBigQNetwork(nn.Module):
    def __init__(self, env, use_theta=False):
        super().__init__()
        n_inputs = env.single_observation_space["obs"].shape[-1]
        policy_inputs = 84
        
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
            nn.ReLU(),
        )
        
        if use_theta:
            self.theta_fc = nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space["theta"].shape), 120),
                nn.ReLU(),
                nn.Linear(120, 120),
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
        
class MinAtarQNetwork(nn.Module):
    def __init__(self, envs, use_theta = False):
        super().__init__()
        self.use_player = False
        
        in_channels = envs.single_observation_space["obs"].shape[0]
        if "player" in list(envs.single_observation_space.keys()): 
            in_channels += 1
            self.use_player = True
        
        n_input_channesl_theta = envs.single_observation_space["theta"].shape[0]
        img_size = envs.single_observation_space["obs"].shape[1]

        self.use_theta = use_theta
        
        if img_size == 10 or img_size == 16:
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif img_size == 32:
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError("Invalid image size")
        
        with torch.no_grad():
            s_ = envs.single_observation_space["obs"].sample()[None]
            if self.use_player:
                p_ = envs.single_observation_space["player"].sample()[None]
                s_ = np.concatenate([s_, p_], axis=1)
            
            n_flatten = self.network(torch.as_tensor(s_).float()).shape[1]

        if self.use_theta:
            if img_size == 10:
                self.theta_fc = nn.Sequential(
                    nn.Conv2d(n_input_channesl_theta, 16, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            else:
                self.theta_fc = nn.Sequential(
                    nn.Conv2d(n_input_channesl_theta, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

            with torch.no_grad():
                t_ = envs.single_observation_space["theta"].sample()[None]
                n_flatten += self.theta_fc(torch.as_tensor(t_).float()).shape[1]
                
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512), 
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n), 
        )

    def forward(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        
        if self.use_player:
            z_ = x["player"]
            x_ = torch.cat([x_, z_], dim=1)

        img_features = self.network(x_.float())
        
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features

        return self.linear(x)
    

class AtariQNetwork(nn.Module):
    def __init__(self, envs, use_theta=False):
        super().__init__()
        in_channels = envs.single_observation_space["obs"].shape[-1]
        theta_in_channels = envs.single_observation_space["theta"].shape[-1]
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            s_ = envs.single_observation_space["obs"].sample()[None]
            n_flatten = self.network(torch.as_tensor(s_).float().permute(0,3,1,2)).shape[1]

        self.use_theta = use_theta
        if use_theta:
            self.theta_network = nn.Sequential(
                nn.Conv2d(theta_in_channels, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                # nn.Conv2d(64, 64, 3, stride=1),
                # nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                t_ = envs.single_observation_space["theta"].sample()[None]
                n_flatten += self.theta_network(torch.as_tensor(t_).float().permute(0,3,1,2)).shape[1]


        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512), 
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n), 
        )


    def forward(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        x_ = x_ / 255.0
        y_ = y_ / 255.0

        img_features = self.network(x_.float().permute(0,3,1,2))

        if self.use_theta:
            theta_features = self.theta_network(y_.float().permute(0,3,1,2))
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features

        return self.linear(x)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

class TetrisPPOAgent(nn.Module):
    def __init__(self, env, use_theta = False, hidden_size=64):
        super().__init__()
        n_inputs = env.single_observation_space["obs"].shape[-1]
        self.use_theta = use_theta
        
        if use_theta:
            n_inputs += env.single_observation_space["theta"].shape[-1]
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_inputs, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_inputs, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x_ = x["obs"]
        y_ = x["theta"][:, 0, :]
        
        if self.use_theta:
            x = torch.cat([x_, y_], dim=1)
        else:
            x = x_
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x_ = x["obs"]
        y_ = x["theta"][:, 0, :]
        
        if self.use_theta:
            x = torch.cat([x_, y_], dim=1)
        else:
            x = x_
        
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class MinAtarPPOAgent(nn.Module):
    def __init__(self, env, use_theta=False):
        super().__init__()
        self.use_theta = use_theta
        n_input_channels = env.single_observation_space["obs"].shape[0]
        n_input_channesl_theta = env.single_observation_space["theta"].shape[0]

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        if use_theta:
            self.theta_fc = nn.Sequential(
                layer_init(nn.Conv2d(n_input_channesl_theta, 16, kernel_size=3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            s_ = env.single_observation_space["obs"].sample()[None]
            t_ = env.single_observation_space["theta"].sample()[None]
            n_flatten = self.network(torch.as_tensor(s_).float()).shape[1]
        
            if use_theta:
                n_flatten += self.theta_fc(torch.as_tensor(t_).float()).shape[1]

        self.actor = layer_init(nn.Linear(n_flatten, env.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(n_flatten, 1), std=1)

    def get_value(self, x):
        x_ = x["obs"]
        y_ = x["theta"]
        img_features = self.network(x_.float())
        
        if self.use_theta:
            theta_features = self.theta_fc(y_.float())
            x = torch.cat([img_features, theta_features], dim=1)
        else:
            x = img_features
            
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x_ = x["obs"]
        y_ = x["theta"]

        img_features = self.network(x_.float())

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
    