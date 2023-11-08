from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import Mlp
import torch.nn as nn
from functools import reduce
import torch

class MixedIdentMlpCNN(nn.Module):
    def __init__(self, obs_dim, action_dim, cnn_keys, mlp_keys, ident_keys, cnn_kwargs, mlp_kwargs):
        super(MixedIdentMlpCNN, self).__init__()

        self.obs_dim = obs_dim
        self.cnn_keys = cnn_keys
        self.mlp_keys = mlp_keys
        self.ident_keys = ident_keys

        cnn_dims = []
        for key, dim in obs_dim.items():
            if key in self.cnn_keys:
                cnn_dims.append(dim)

        cnn_width, cnn_height = cnn_dims[0][1], cnn_dims[0][2]
        cnn_channels = cnn_dims[0][0]
        for dim in cnn_dims[1:]:
            assert cnn_width == dim[1]
            assert cnn_height == dim[2]
            cnn_channels += dim[0]

        mlp_dims = []
        for key, dim in obs_dim.items():
            if key in self.mlp_keys:
                if len(dim) > 1:
                    dim = reduce(lambda x, y: x*y, dim)
                mlp_dims.append(dim[0])

        mlp_input_dim = sum(mlp_dims)
        self.cnn = CNN(cnn_width, cnn_height, cnn_channels, **cnn_kwargs)
        if len(mlp_dims) > 0:
            self.mlp = Mlp(input_size=mlp_input_dim, **mlp_kwargs)
        else:
            self.mlp = nn.Identity()

        insize = cnn_kwargs['output_size'] + (mlp_kwargs.get('output_size') or 0)
        self.fc = nn.Linear(insize, action_dim)

    def forward(self, input):
        input = self._dictify_input(input)

        conv_input = []
        mlp_input = []
        ident_input = []
        for key, val in input.items():
            if key in self.cnn_keys:
                conv_input.append(val)
            elif key in self.mlp_keys:
                mlp_input.append(val.flatten(start_dim=1))
            elif key in self.ident_keys:
                ident_input.append(val.flatten(start_dim=1))
            # else:
            #     print(f"Skipping input with key: {key}")
        conv_input = torch.cat(conv_input, dim=-1)
        if len(mlp_input) > 0:
            mlp_input = torch.cat(mlp_input, dim=-1)

        if len(ident_input) > 0:
            ident_output = torch.cat(ident_input, dim=-1)
        else:
            ident_output = None
        if len(conv_input) > 0:
            conv_output = self.cnn(conv_input)
        else:
            conv_output = None
        if len(mlp_input) > 0:
            mlp_output = self.mlp(mlp_input)
        else:
            mlp_output = None

        combined_out = torch.cat([x for x in [conv_output, mlp_output, ident_output] if x is not None], -1)
        return self.fc(combined_out)
    def _dictify_input(self, input):
        dict_input = {}
        for key, dim in self.obs_dim.items():
            flat_dim = reduce(lambda x, y: x * y, dim)
            dict_input[key] = input[:, :flat_dim]
            input = input[:, flat_dim:]
        return dict_input
    


import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Code from: https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_network.py
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MinAtarDQN(nn.Module):
    """
    DQN-style network for MinAtar environment

    Reference: MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, 128)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, np.prod(action_shape))),
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        batch_sz = obs.shape[0]
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).view(batch_sz, self.c, self.h, self.w)
        return self.net(obs)



