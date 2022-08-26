from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data as PygData
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
from torch_geometric.nn.aggr import SortAggregation, SumAggregation
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


class SwapPredLstm(nn.Module):
    def __init__(
        self,
        max_qubit: int,
        gcn_out_channel: int,
        lstm_out_channel: int,
    ) -> None:
        super().__init__()
        self._max_qubit = max_qubit
        self._gcn_out_channel = gcn_out_channel
        self._out_channel = lstm_out_channel

        self._gcn = SwapPredictGcn(out_channel=gcn_out_channel)

        self._lstm = nn.LSTM(
            input_size=self._gcn_out_channel, hidden_size=self._out_channel
        )

    def forward(self, x):
        layer_outputs = []
        for layer_graph in x:
            _y = self._gcn(layer_graph.x, layer_graph.edge_index)
            layer_outputs.append(_y)

        layer_outputs = torch.stack(layer_outputs)

        seq_feature, _ = self._lstm(layer_outputs)

        return seq_feature[-1]


class SwapPredMlp(nn.Module):
    def __init__(
        self, in_channel: int, hidden_channels: Iterable[int], mlp_out_channel: int
    ) -> None:
        super().__init__()
        self._lins = nn.ModuleList()

        last_c = in_channel
        for c in hidden_channels:
            self._lins.append(nn.Linear(last_c, c))

        self._last_lin = nn.Linear(last_c, mlp_out_channel)

    def forward(self, x):
        for lin in self._lins:
            x = lin(x)
            x = F.leaky_relu(x)
        x = self._last_lin(x)
        return x


class SwapPredMix(nn.Module):
    def __init__(
        self,
        max_qubit: int,
        gcn_out_channel: int,
        lstm_out_channel: int,
        mlp_hidden_channels: Iterable[int],
        out_channel: int,
    ) -> None:
        super().__init__()
        self._max_qubit = max_qubit
        self._gcn = SwapPredictGcn(out_channel=gcn_out_channel)
        self._lstm = SwapPredLstm(
            max_qubit=max_qubit,
            gcn_out_channel=gcn_out_channel,
            lstm_out_channel=lstm_out_channel,
        )
        self._mlp = SwapPredMlp(
            in_channel=lstm_out_channel,
            hidden_channels=mlp_hidden_channels,
            mlp_out_channel=out_channel,
        )

    def forward(self, x, edge_index):
        x = self._gcn(x, edge_index)
        x = self._lstm(x)
        x = self._mlp(x)

        return x


if __name__ == "__main__":
    import os.path as osp

    data_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(data_dir, "data")
    data_dir = osp.join(data_dir, "processed_layered")
    dataset = MappingLayeredDataset(data_dir=data_dir)
    loader = dataset.loader(16, True, "cuda")

    # print(loader)

    batch = next(iter(loader))
    # print(batch)

    model = SwapPredMix(
        max_qubit=50,
        gcn_out_channel=50,
        hetero_metadata=dataset.hetero_metadata,
        out_channel=50,
    ).to("cuda")

    with torch.no_grad():
        data, labels = batch
        out = model(data)
        print(out.shape)
        print(out)
