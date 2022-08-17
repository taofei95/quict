from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.nn import Linear as PygLinear, LayerNorm as PygLayerNorm
from torch_geometric.nn import global_sort_pool
import torch.nn.functional as F
import torch
from torch import nn


class SwapNumPredictGcn(nn.Module):
    def __init__(self, out_channel: int) -> None:
        super().__init__()
        self.conv_1 = GATConv(-1, 200)
        self.lin_1 = PygLinear(-1, 200)

        self.conv_2 = GATConv(-1, 200)
        self.lin_2 = PygLinear(-1, 200)

        self.conv_3 = GATConv(-1, 100)
        self.lin_3 = PygLinear(-1, 100)

        self.conv_4 = GATConv(-1, 100)
        self.lin_4 = PygLinear(-1, 100)

        self.conv_5 = GATConv(-1, 50)
        self.lin_5 = PygLinear(-1, 50)

        self.conv_6 = GATConv(-1, 50)
        self.lin_6 = PygLinear(-1, 50)

        self.last_conv = GATConv(-1, out_channel)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv_1(x, edge_index)) + self.lin_1(x)
        x = F.leaky_relu(self.conv_2(x, edge_index)) + self.lin_2(x)
        x = F.leaky_relu(self.conv_3(x, edge_index)) + self.lin_3(x)
        x = F.leaky_relu(self.conv_4(x, edge_index)) + self.lin_4(x)
        x = F.leaky_relu(self.conv_5(x, edge_index)) + self.lin_5(x)
        x = F.leaky_relu(self.conv_6(x, edge_index)) + self.lin_6(x)
        x = self.last_conv(x, edge_index)
        return x


class SwapNumPredictHeteroMix(nn.Module):
    def __init__(self, metadata: str, k: int, gc_out_channel: int) -> None:
        super().__init__()

        self.k = k

        self.hetero_gnn = to_hetero(
            SwapNumPredictGcn(gc_out_channel), metadata=metadata, aggr="sum"
        )

        self.layer_norm_1 = nn.LayerNorm(k * gc_out_channel)

        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=gc_out_channel, stride=gc_out_channel),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=5),
        )
        self.conv1d_out_dim = int(
            (k * gc_out_channel - 1 * (gc_out_channel - 1) - 1) / gc_out_channel + 1
        )
        self.conv1d_out_dim = int((self.conv1d_out_dim - 1 * (2 - 1) - 1) / 2 + 1)
        self.conv1d_out_dim = int((self.conv1d_out_dim - 1 * (5 - 1) - 1) / 1 + 1)
        self.conv1d_out_channel = 32

        self.layer_norm_2 = nn.LayerNorm(self.conv1d_out_channel * self.conv1d_out_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.conv1d_out_channel * self.conv1d_out_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, data):
        x = self.hetero_gnn(data.x_dict, data.edge_index_dict)
        x = global_sort_pool(x["lc"], data["lc"].batch, self.k)
        x = self.layer_norm_1(x)
        x = torch.unsqueeze(x, dim=-2)
        x = self.conv1d(x)
        x = torch.reshape(x, (-1, self.conv1d_out_channel * self.conv1d_out_dim))
        x = self.layer_norm_2(x)
        x = self.mlp(x)
        return x
