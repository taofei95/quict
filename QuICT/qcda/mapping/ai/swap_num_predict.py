from typing import Iterable, Tuple, List
import torch
from torch.nn import Flatten, LazyLinear, Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_sort_pool, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as GT


class SwapPredGNN(torch.nn.Module):
    def __init__(self, hidden_channel: Iterable[int], out_channel: int,) -> None:
        super().__init__()
        self.hidden_gc_layer = torch.nn.ModuleList()
        # Do not know why SAGEConv causes an error.
        for h in hidden_channel:
            self.hidden_gc_layer.append(GATConv(-1, h))
        self.last_gc_layer = GATConv(-1, out_channel)

    def forward(self, x, edge_index):
        for conv in self.hidden_gc_layer:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.last_gc_layer(x, edge_index)
        return x


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


class SwapPredMix(torch.nn.Module):
    def __init__(
        self,
        lc_qubit: int,
        gc_hidden_channel: Iterable[int],
        gc_out_channel: int,
        gc_model_metadata,
        ml_hidden_channel: Iterable[int],
        ml_out_channel: int,
    ) -> None:
        super().__init__()
        gc_model = SwapPredGNN(gc_hidden_channel, gc_out_channel)
        self.gc_model = to_hetero(gc_model, gc_model_metadata)
        self.flatten = Flatten(0, -1)
        self.ml_model = SwapPredMLP(
            lc_qubit * gc_out_channel, ml_hidden_channel, ml_out_channel
        )

    def forward(self, data: HeteroData):
        gc_out = self.gc_model(data.x_dict, data.edge_index_dict)
        gc_out_lc_flat = self.flatten(gc_out["lc_qubit"])
        pred = self.ml_model(gc_out_lc_flat)
        return pred


if __name__ == "__main__":
    data_1 = HeteroData()

    data_1["lc_qubit"].x = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.float,
    )
    data_1["pc_qubit"].x = torch.tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float
    )

    data_1["lc_qubit", "lc_conn", "lc_qubit"].edge_index = (
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int).t().contiguous()
    )
    data_1["pc_qubit", "pc_conn", "pc_qubit"].edge_index = (
        torch.tensor([[0, 1], [1, 2]], dtype=torch.int).t().contiguous()
    )

    data_1 = GT.ToUndirected()(data_1)

    data_2 = HeteroData()

    data_2["lc_qubit"].x = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.float,
    )
    data_2["pc_qubit"].x = torch.tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float
    )

    data_2["lc_qubit", "lc_conn", "lc_qubit"].edge_index = (
        torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int).t().contiguous()
    )
    data_2["pc_qubit", "pc_conn", "pc_qubit"].edge_index = (
        torch.tensor([[0, 1], [1, 2]], dtype=torch.int).t().contiguous()
    )

    data_2 = GT.ToUndirected()(data_2)

    model = SwapPredMix(5, [3, 3,], 2, data_1.metadata(), [3, 3,], 1)

    print(data_1.metadata())

    with torch.no_grad():
        out = model(data_1)
        print(out)
        out = model(data_2)
        print(out)

