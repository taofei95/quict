from math import floor
from typing import Iterable, Tuple, Union
import torch
from torch.nn import Flatten, Linear, Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.nn import Linear as PygLinear


class Mlp(torch.nn.Module):
    def __init__(
        self, in_channel: int, hidden_channel: Iterable[int], out_channel: int
    ) -> None:
        super().__init__()
        self.hidden_ml_layer = torch.nn.ModuleList()
        last_h = in_channel
        for h in hidden_channel:
            self.hidden_ml_layer.append(Linear(last_h, h))
            last_h = h
        self.last_ml_layer = Linear(last_h, out_channel)

    def forward(self, x):
        for ml in self.hidden_ml_layer:
            x = ml(x)
            x = F.relu(x)
        x = self.last_ml_layer(x)
        return x


class MultiLayerGnn(torch.nn.Module):
    def __init__(
        self, hidden_channel: Iterable[int], out_channel: int, pool_node: int
    ) -> None:
        super().__init__()
        self.hidden_gc_layer = torch.nn.ModuleList()
        self.linear_layer = torch.nn.ModuleList()
        for h in hidden_channel:
            self.hidden_gc_layer.append(GCNConv(-1, h))
            self.linear_layer.append(PygLinear(-1, h))
        self.last_gc_layer = GCNConv(-1, out_channel)
        self.pool_node = pool_node
        self.out_channel = out_channel

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcl, lin in zip(self.hidden_gc_layer, self.linear_layer):
            x = gcl(x, edge_index) + lin(x) * 0.5
            x = F.relu(x)
        x = self.last_gc_layer(x, edge_index)
        x = global_sort_pool(x, batch, k=self.pool_node)
        # Global sort pooling flattens tensors. We reshape them.
        x = torch.reshape(x, (self.pool_node, self.out_channel))
        return x


class CnnRecognition(torch.nn.Module):
    def __init__(self, h: int, w: int,) -> None:
        """Wrapper for generating multi-layer 2D CNN.
        Input of this model should have shape (1, h, w). Output is a (c_out, h_out, w_out) tensor.

        Args:
            h (int): Input 2D height.
            w (int): Input 2D width.
        """
        super().__init__()
        h_out = h
        w_out = w

        self.conv_group_1 = torch.nn.Sequential(
            Conv2d(2, 64, 3, padding="same"),
            ReLU(),
            Conv2d(64, 64, 3, padding="same"),
            ReLU(),
            MaxPool2d(2, stride=2),
        )
        h_out = int(floor((h_out - 1 * (2 - 1) - 1) / 2 + 1))
        w_out = int(floor((w_out - 1 * (2 - 1) - 1) / 2 + 1))

        self.conv_group_2 = torch.nn.Sequential(
            Conv2d(64, 128, 3, padding="same"),
            ReLU(),
            Conv2d(128, 128, 3, padding="same"),
            ReLU(),
            MaxPool2d(2, stride=2),
        )
        h_out = int(floor((h_out - 1 * (2 - 1) - 1) / 2 + 1))
        w_out = int(floor((w_out - 1 * (2 - 1) - 1) / 2 + 1))

        self.conv_group_3 = torch.nn.Sequential(
            Conv2d(128, 256, 3, padding="same"),
            ReLU(),
            Conv2d(256, 256, 3, padding="same"),
            ReLU(),
            Conv2d(256, 256, 3, padding="same"),
            ReLU(),
            Conv2d(256, 256, 3, padding="same"),
            ReLU(),
            MaxPool2d(2, stride=2),
        )
        h_out = int(floor((h_out - 1 * (2 - 1) - 1) / 2 + 1))
        w_out = int(floor((w_out - 1 * (2 - 1) - 1) / 2 + 1))

        self.h_out = h_out
        self.w_out = w_out
        self.c_out = 256

    def forward(self, x):
        x = self.conv_group_1(x)
        x = self.conv_group_2(x)
        x = self.conv_group_3(x)
        return x


class SwapPredMixBase(torch.nn.Module):
    def __init__(
        self,
        topo_gc_hidden_channel: Iterable[int],
        lc_gc_hidden_channel: Iterable[int],
        topo_lc_gc_out_channel: int,
        topo_lc_gc_pool_node: int,
        ml_hidden_channel: Iterable[int],
        ml_out_channel: int,
    ) -> None:
        super().__init__()
        self.topo_gnn = MultiLayerGnn(
            hidden_channel=topo_gc_hidden_channel,
            out_channel=topo_lc_gc_out_channel,
            pool_node=topo_lc_gc_pool_node,
        )

        self.lc_gnn = MultiLayerGnn(
            hidden_channel=lc_gc_hidden_channel,
            out_channel=topo_lc_gc_out_channel,
            pool_node=topo_lc_gc_pool_node,
        )
        self.fuse_cnn = CnnRecognition(topo_lc_gc_pool_node, topo_lc_gc_out_channel)

        self.flatten_len = self.fuse_cnn.c_out * self.fuse_cnn.h_out * self.fuse_cnn.w_out
        self.fuse_flatten = Flatten(start_dim=0, end_dim=-1)

        self.mlp = Mlp(self.flatten_len, ml_hidden_channel, ml_out_channel)

    def forward(self, data):
        topo_x = self.topo_gnn(data["topo"])
        lc_x = self.lc_gnn(data["lc"])

        x = torch.stack((topo_x, lc_x,), dim=0)

        x = self.fuse_cnn(x)
        x = self.fuse_flatten(x)
        x = self.mlp(x)

        return x

