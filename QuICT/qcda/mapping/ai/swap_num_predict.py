from typing import Iterable
import torch
from torch.nn import Flatten, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool
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
            self.hidden_gc_layer.append(GCNConv(-1, h))
            self.linear_layer.append(PygLinear(-1, h))
        self.last_gc_layer = GCNConv(-1, out_channel)
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

        self.gc_mix_out_channel = (
            topo_pool_node * topo_gc_out_channel + lc_pool_node * lc_gc_out_channel
        )

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
        self.mlp = SwapPredMLP(
            in_channel=self.gc_mix_out_channel,
            hidden_channel=ml_hidden_channel,
            out_channel=ml_out_channel,
        )

    def forward(self, pair_data):
        x_topo = self.topo_gnn(
            pair_data.x_topo, pair_data.edge_index_topo, pair_data.x_topo_batch
        )
        x_lc = self.lc_gnn(
            pair_data.x_lc, pair_data.edge_index_lc, pair_data.x_lc_batch
        )
        x = torch.concat((x_topo, x_lc,), dim=-1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    from torch_geometric.data import Data

    model = SwapPredMix(
        topo_gc_hidden_channel=[2, 2,],
        topo_gc_out_channel=2,
        topo_pool_node=2,
        lc_gc_hidden_channel=[2, 2,],
        lc_gc_out_channel=2,
        lc_pool_node=2,
        ml_hidden_channel=[3, 3,],
        ml_out_channel=1,
    )

    topo_data_list = []
    lc_data_list = []
    node_feature_num = 20

    x_topo = torch.zeros(3, node_feature_num)
    x_topo[:, :3] = torch.eye(3, dtype=float)
    edge_index_topo = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]]
    x_lc = torch.zeros(5, node_feature_num)
    x_lc[:, :5] = torch.eye(5, dtype=float)
    edge_index_lc = [[0, 1], [1, 2]]

    edge_index_topo = torch.tensor(edge_index_topo, dtype=torch.long).t().contiguous()
    edge_index_lc = torch.tensor(edge_index_lc, dtype=torch.long).t().contiguous()
    data = PairData(
        edge_index_topo=edge_index_topo,
        x_topo=x_topo,
        edge_index_lc=edge_index_lc,
        x_lc=x_lc,
    )

    batch = Batch.from_data_list([data], follow_batch=["x_topo", "x_lc"])
    print(batch)

    with torch.no_grad():
        out = model(batch)
        print(out)

