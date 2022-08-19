import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data as PygData
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
from QuICT.qcda.mapping.ai.dataset import MappingLayeredDataset


class SwapPredictGcn(nn.Module):
    def __init__(self, out_channel: int) -> None:
        super().__init__()

        self._out_channel = out_channel

        self._conv_1 = gnn.SAGEConv(-1, 200)
        self._lin_1 = gnn.Linear(-1, 200)

        self._conv_2 = gnn.SAGEConv(-1, 200)
        self._lin_2 = gnn.Linear(-1, 200)

        self._conv_3 = gnn.SAGEConv(-1, 100)
        self._lin_3 = gnn.Linear(-1, 100)

        self._ln_3 = gnn.LayerNorm(100)

        self._conv_4 = gnn.SAGEConv(-1, 100)
        self._lin_4 = gnn.Linear(-1, 100)

        self._conv_5 = gnn.SAGEConv(-1, 50)
        self._lin_5 = gnn.Linear(-1, 50)

        self._conv_6 = gnn.SAGEConv(-1, 50)
        self._lin_6 = gnn.Linear(-1, 50)

        self._ln_6 = gnn.LayerNorm(50)

        self._last_conv = gnn.SAGEConv(-1, out_channel)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self._conv_1(x, edge_index)) + self._lin_1(x)
        x = F.leaky_relu(self._conv_2(x, edge_index)) + self._lin_2(x)
        x = F.leaky_relu(self._conv_3(x, edge_index)) + self._lin_3(x)
        x = self._ln_3(x)
        x = F.leaky_relu(self._conv_4(x, edge_index)) + self._lin_4(x)
        x = F.leaky_relu(self._conv_5(x, edge_index)) + self._lin_5(x)
        x = F.leaky_relu(self._conv_6(x, edge_index)) + self._lin_6(x)
        x = self._ln_6(x)
        x = self._last_conv(x, edge_index)
        return x


class SwapPredMix(nn.Module):
    def __init__(
        self,
        max_qubit: int,
        gcn_out_channel: int,
        hetero_metadata: str,
        out_channel: int,
    ) -> None:
        super().__init__()
        self._max_qubit = max_qubit
        self._gcn_out_channel = gcn_out_channel
        self._out_channel = out_channel

        self._gcn = gnn.to_hetero(
            module=SwapPredictGcn(out_channel=gcn_out_channel),
            metadata=hetero_metadata,
            aggr="sum",
        )

        self._lstm = nn.LSTM(
            input_size=self._gcn_out_channel, hidden_size=self._out_channel
        )

    def forward(self, x):
        layer_outputs = []
        for layer_graph in x:
            _y = self._gcn(layer_graph.x_dict, layer_graph.edge_index_dict)
            _y = _y["lc"]
            # Reshape pyg flattened [batch_size * num_node, feature_dim] back into [batch_size, num_node, feature_dim]
            _y = torch.reshape(_y, (-1, self._max_qubit, self._gcn_out_channel))
            # Manually do sum aggregation because of buggy PYG aggregator on heterogeneous graphs.
            _y = torch.sum(_y, 1)
            layer_outputs.append(_y)

        layer_outputs = torch.stack(layer_outputs)

        seq_feature, (_, _) = self._lstm(layer_outputs)

        return seq_feature


if __name__ == "__main__":
    import os.path as osp

    data_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(data_dir, "data")
    data_dir = osp.join(data_dir, "processed_layered")
    dataset = MappingLayeredDataset(data_dir=data_dir)
    loader = dataset.loader(16, True)

    # print(loader)

    batch = next(iter(loader))
    # print(batch)

    model = SwapPredMix(
        max_qubit=50,
        gcn_out_channel=50,
        hetero_metadata=dataset.hetero_metadata,
        out_channel=50,
    )
    with torch.no_grad():
        data, labels = batch
        out = model(data)
        print(out.shape)
        print(out)
