from typing import Iterable
import torch
from torch.nn import Flatten, Linear, Conv1d, MaxPool1d, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool, GATConv
from torch_geometric.nn import Linear as PygLinear
from torch_geometric.data import Batch, Data

from QuICT.qcda.mapping.ai.data_def import PairData


class SwapPredMLP(torch.nn.Module):
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
            # x = F.relu(x)
            x = F.leaky_relu(x)
        x = self.last_ml_layer(x)
        return x


class SwapPredGnn(torch.nn.Module):
    def __init__(
        self, hidden_channel: Iterable[int], out_channel: int, pool_node: int
    ) -> None:
        super().__init__()
        self.hidden_gc_layer = torch.nn.ModuleList()
        self.linear_layer = torch.nn.ModuleList()
        for h in hidden_channel:
            self.hidden_gc_layer.append(GATConv(-1, h))
            self.linear_layer.append(PygLinear(-1, h))
        self.last_gc_layer = GATConv(-1, out_channel)
        self.pool_node = pool_node
        self.out_channel = out_channel

    def forward(self, x, edge_index, batch):
        for gcl, lin in zip(self.hidden_gc_layer, self.linear_layer):
            x = gcl(x, edge_index)
            # x = F.relu(x)
            x = F.leaky_relu(x) + lin(x)
        x = self.last_gc_layer(x, edge_index)
        x = global_sort_pool(x, batch, k=self.pool_node)
        return x


class SwapPredMix(torch.nn.Module):
    def __init__(
        self,
        topo_gc_hidden_channel: Iterable[int],
        topo_gc_out_channel: int,
        topo_pool_node: int,
        lc_gc_hidden_channel: Iterable[int],
        lc_gc_out_channel: int,
        lc_pool_node: int,
        ml_hidden_channel: Iterable[int],
        ml_out_channel: int,
    ) -> None:
        assert lc_gc_out_channel == topo_gc_out_channel
        super().__init__()

        self.total_node = lc_pool_node + topo_pool_node
        self.total_feature_dim = topo_gc_out_channel

        self.mix_flat_out_dim = self.total_feature_dim * self.total_node

        self.topo_gnn = SwapPredGnn(
            hidden_channel=topo_gc_hidden_channel,
            out_channel=topo_gc_out_channel,
            pool_node=topo_pool_node,
        )
        self.lc_gnn = SwapPredGnn(
            hidden_channel=lc_gc_hidden_channel,
            out_channel=lc_gc_out_channel,
            pool_node=lc_pool_node,
        )

        self.ln = LayerNorm(normalized_shape=self.mix_flat_out_dim)

        conv1d_channels = [16, 16, 32]
        conv1d_ks = [self.total_feature_dim, 5, 5]

        # Conv1 & pooling
        self.conv1d_1 = Conv1d(
            in_channels=1,
            out_channels=conv1d_channels[0],
            kernel_size=conv1d_ks[0],
            stride=conv1d_ks[0],
        )
        self.dense_dim = int(
            (self.mix_flat_out_dim - 1 * (conv1d_ks[0] - 1) - 1) / conv1d_ks[0] + 1
        )
        self.max_pool1d_1 = MaxPool1d(2, 2)
        self.dense_dim = int((self.dense_dim - 1 * (2 - 1) - 1) / 2 + 1)

        # Conv2 & pooling
        self.conv1d_2 = Conv1d(
            in_channels=conv1d_channels[0],
            out_channels=conv1d_channels[1],
            kernel_size=conv1d_ks[1],
            padding=conv1d_ks[1] - 1,
        )
        self.dense_dim = int(
            (self.dense_dim + 2 * (conv1d_ks[1] - 1) - 1 * (conv1d_ks[1] - 1) - 1) / 1
            + 1
        )
        self.max_pool1d_2 = MaxPool1d(2, 2)
        self.dense_dim = int((self.dense_dim - 1 * (2 - 1) - 1) / 2 + 1)

        self.conv1d_3 = Conv1d(
            in_channels=conv1d_channels[1],
            out_channels=conv1d_channels[2],
            kernel_size=conv1d_ks[2],
            padding=conv1d_ks[2] - 1,
        )
        self.dense_dim = int(
            (self.dense_dim + 2 * (conv1d_ks[2] - 1) - 1 * (conv1d_ks[2] - 1) - 1) / 1
            + 1
        )

        self.dense_dim *= conv1d_channels[-1]

        self.mlp = SwapPredMLP(
            in_channel=self.dense_dim,
            hidden_channel=ml_hidden_channel,
            out_channel=ml_out_channel,
        )

    def forward(self, pair_data):
        x_topo = self.topo_gnn(
            pair_data.x_topo, pair_data.edge_index_topo, pair_data.x_topo_batch
        )  # [batch, topo_pool_node * topo_out_feature]
        x_lc = self.lc_gnn(
            pair_data.x_lc, pair_data.edge_index_lc, pair_data.x_lc_batch
        )  # [batch, lc_pool_node * lc_out_feature]
        x = torch.concat(
            (x_topo, x_lc,), dim=-1
        )  # [batch, total_node * total_feature_dim]

        x = self.ln(x)

        x = torch.reshape(
            x, (-1, 1, self.total_node * self.total_feature_dim)
        )  # [batch, 1, total_node * total_feature_dim]
        x = self.conv1d_1(x)
        x = F.leaky_relu(x)
        x = self.max_pool1d_1(x)
        x = self.conv1d_2(x)
        x = F.leaky_relu(x)
        x = self.max_pool1d_2(x)
        x = self.conv1d_3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.mlp(x)
        return x

