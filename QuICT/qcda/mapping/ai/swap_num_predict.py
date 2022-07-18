from typing import Iterable, Tuple, List
import torch
from torch.nn import Flatten, LazyLinear, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.data import HeteroData, Batch, Data
import torch_geometric.transforms as GT


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
            x = F.relu(x)
        x = self.last_ml_layer(x)
        return x


class SwapPredGnn(torch.nn.Module):
    def __init__(
        self, hidden_channel: Iterable[int], out_channel: int, pool_node: int
    ) -> None:
        super().__init__()
        self.hidden_gc_layer = torch.nn.ModuleList()
        for h in hidden_channel:
            self.hidden_gc_layer.append(GCNConv(-1, h))
        self.last_gc_layer = GCNConv(-1, out_channel)
        self.pool_node = pool_node

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcl in self.hidden_gc_layer:
            x = gcl(x, edge_index)
            x = F.relu(x)
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
        super().__init__()
        self.topo_gnn = SwapPredGnn(
            hidden_channel=topo_gc_hidden_channel,
            out_channel=topo_gc_out_channel,
            pool_node=topo_pool_node,
        )
        self.topo_flatten = Flatten(start_dim=0)
        self.lc_gnn = SwapPredGnn(
            hidden_channel=lc_gc_hidden_channel,
            out_channel=lc_gc_out_channel,
            pool_node=lc_pool_node,
        )
        self.lc_flatten = Flatten(start_dim=0)
        self.mlp = SwapPredMLP(
            in_channel=topo_pool_node * topo_gc_out_channel
            + lc_pool_node * lc_gc_out_channel,
            hidden_channel=ml_hidden_channel,
            out_channel=ml_out_channel,
        )

    def forward(self, data):
        topo_x = self.topo_gnn(data["topo"])
        topo_x = self.topo_flatten(topo_x)
        lc_x = self.lc_gnn(data["lc"])
        lc_x = self.lc_flatten(lc_x)
        x = torch.concat((topo_x, lc_x,))
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

    topo_x = torch.zeros(3, node_feature_num)
    topo_x[:, :3] = torch.eye(3, dtype=float)
    topo_edge_index = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]]
    lc_x = torch.zeros(5, node_feature_num)
    lc_x[:, :5] = torch.eye(5, dtype=float)
    lc_edge_index = [[0, 1], [1, 2]]

    topo_edge_index = torch.tensor(topo_edge_index, dtype=torch.long).t().contiguous()
    lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.long).t().contiguous()
    topo_data = Data(x=topo_x, edge_index=topo_edge_index)
    lc_data = Data(x=lc_x, edge_index=lc_edge_index)

    topo_data_list.append(topo_data)
    lc_data_list.append(lc_data)
    topo_data_list = Batch.from_data_list(topo_data_list)
    lc_data_list = Batch.from_data_list(lc_data_list)

    data = {}
    data["topo"] = topo_data_list
    data["lc"] = lc_data_list

    with torch.no_grad():
        out = model(data)
        print(out)

    topo_data_list = []
    lc_data_list = []
    topo_x = torch.zeros(4, node_feature_num)
    topo_x[:, :4] = torch.eye(4, dtype=float)
    topo_edge_index = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0], [0, 3], [3, 0]]
    lc_x = torch.zeros(6, node_feature_num)
    lc_x[:, :6] = torch.eye(6, dtype=float)
    lc_edge_index = [[0, 1], [1, 2]]

    topo_edge_index = torch.tensor(topo_edge_index, dtype=torch.long).t().contiguous()
    lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.long).t().contiguous()
    topo_data = Data(x=topo_x, edge_index=topo_edge_index)
    lc_data = Data(x=lc_x, edge_index=lc_edge_index)

    topo_data_list.append(topo_data)
    lc_data_list.append(lc_data)

    topo_data_list = Batch.from_data_list(topo_data_list)
    lc_data_list = Batch.from_data_list(lc_data_list)
    data = {}
    data["topo"] = topo_data_list
    data["lc"] = lc_data_list

    with torch.no_grad():
        out = model(data)
        print(out)

    topo_data_list = []
    lc_data_list = []
    topo_x = torch.zeros(20, node_feature_num)
    topo_x[:, :20] = torch.eye(20, dtype=float)
    topo_edge_index = [
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],
        [0, 2],
        [2, 0],
        [15, 16],
        [16, 15],
    ]
    lc_x = torch.zeros(10, node_feature_num)
    lc_x[:, :10] = torch.eye(10, dtype=float)
    lc_edge_index = [[0, 1], [1, 3], [2, 3]]

    topo_edge_index = torch.tensor(topo_edge_index, dtype=torch.long).t().contiguous()
    lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.long).t().contiguous()
    topo_data = Data(x=topo_x, edge_index=topo_edge_index)
    lc_data = Data(x=lc_x, edge_index=lc_edge_index)

    topo_data_list.append(topo_data)
    lc_data_list.append(lc_data)

    topo_data_list = Batch.from_data_list(topo_data_list)
    lc_data_list = Batch.from_data_list(lc_data_list)
    data = {}
    data["topo"] = topo_data_list
    data["lc"] = lc_data_list

    with torch.no_grad():
        out = model(data)
        print(out)
