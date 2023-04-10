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
        self.mlp = Mlp(input_size=mlp_input_dim, **mlp_kwargs)

        self.fc = nn.Linear(cnn_kwargs['output_size'] + mlp_kwargs['output_size'], action_dim)

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
            else:
                print(f"Skipping input with key: {key}")
        conv_input = torch.cat(conv_input, dim=-1)
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



