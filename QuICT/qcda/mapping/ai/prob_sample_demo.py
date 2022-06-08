from argparse import ArgumentTypeError
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, to_hetero


class GAT(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: Union[int, List[int]] = 200,
        num_conv_layer: int = 3,
        fc_out_features: Union[int, List[int]] = 1000,
        num_fc_layer: int = 1,
        heads: int = 2,
    ) -> None:
        super().__init__()
        if type(hidden_channels) is int:
            channels = [hidden_channels for _ in range(num_conv_layer)]
        elif type(hidden_channels) is list:
            assert len(hidden_channels) == num_conv_layer
            channels = hidden_channels
        else:
            raise ArgumentTypeError(
                "Only int/list(int) are accepetd hidden_channels types"
            )
        self.num_conv_layer = num_conv_layer
        self.conv_layer = torch.nn.ModuleList()
        for i in range(num_conv_layer):
            self.conv_layer.append(
                GATv2Conv(in_channels=-1, out_channels=channels[i], heads=heads)
            )

        if type(fc_out_features) is int:
            fc_out = [fc_out_features for _ in range(num_fc_layer)]
        elif type(fc_out_features) is list:
            assert num_fc_layer == len(fc_out_features)
            fc_out = fc_out_features
        else:
            raise ArgumentTypeError(
                "Only int/list(int) are accepetd fc_out_features types"
            )
        self.num_fc_layer = num_fc_layer
        self.fc_layer = torch.nn.ModuleList()
        self.fc_layer.append(torch.nn.LazyLinear(fc_out[0]))
        for idx in range(1, num_fc_layer):
            self.fc_layer.append(torch.nn.Linear(fc_out[idx - 1], fc_out[idx]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for idx, conv in enumerate(self.conv_layer):
            x = conv(x, edge_index)
            if idx != self.num_conv_layer - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)

        # It should be a pooling layer.
        x = torch.flatten(x)

        for fc in self.fc_layer:
            x = fc(x)
            x = F.relu(x)

        return x
