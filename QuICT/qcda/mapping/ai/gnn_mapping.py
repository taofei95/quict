import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class CircuitGnn(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        max_gate_num: int,
        feat_dim: int,
        heads: int,
    ) -> None:
        super().__init__()

        self._max_qubit_num = max_qubit_num
        self._max_gate_num = max_gate_num
        self._feat_dim = feat_dim

        self._gc_first = gnn.GATv2Conv(
            in_channels=2 * feat_dim, out_channels=2 * feat_dim, heads=heads
        )

        self._gc_inner = nn.ModuleList(
            [
                gnn.GATv2Conv(
                    in_channels=2 * feat_dim * heads,
                    out_channels=2 * feat_dim,
                    heads=heads,
                )
                for _ in range(3)
            ]
        )

        self._gc_last = gnn.GATv2Conv(
            in_channels=2 * feat_dim * heads, out_channels=2 * feat_dim, heads=1
        )

        self._norm = gnn.LayerNorm(in_channels=2 * feat_dim)

        self._aggr = gnn.aggr.SoftmaxAggregation(learn=True)

    def forward(self, x, edge_index, batch=None):
        n = self._max_gate_num
        f = self._feat_dim

        residual = x
        x = F.leaky_relu(self._gc_first(x, edge_index))
        for conv in self._gc_inner:
            x = F.leaky_relu(conv(x, edge_index))
        x = F.leaky_relu(self._gc_last(x, edge_index))
        x = x + residual
        x = self._norm(x, batch)
        x = self._aggr(x, batch)  # [b, 2 * f]
        x = x.view(-1, 2 * f)  # [b, 2 * f]
        return x


class LayoutGnn(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        feat_dim: int,
        heads: int,
    ) -> None:
        super().__init__()

        self._max_qubit_num = max_qubit_num
        self._feat_dim = feat_dim

        self._gc_first = gnn.GATv2Conv(
            in_channels=3 * feat_dim, out_channels=3 * feat_dim, heads=heads
        )

        self._gc_inner = nn.ModuleList(
            [
                gnn.GATv2Conv(
                    in_channels=3 * feat_dim * heads,
                    out_channels=3 * feat_dim,
                    heads=heads,
                )
                for _ in range(3)
            ]
        )

        self._norm = gnn.LayerNorm(in_channels=3 * feat_dim)

        self._gc_last = gnn.GATv2Conv(
            in_channels=3 * feat_dim * heads, out_channels=3 * feat_dim, heads=1
        )

    def forward(self, circ_feat, x, edge_index, batch=None):
        q = self._max_qubit_num
        f = self._feat_dim

        # Input circ_feat has shape [b, 2 * f]
        circ_feat = torch.repeat_interleave(
            circ_feat, q, dim=0
        ).contiguous()  # [b * q, 2 * f]
        x = torch.cat((x, circ_feat), dim=1)  # [b * q, 3 * f]

        residual = x
        x = F.leaky_relu(self._gc_first(x, edge_index))
        for conv in self._gc_inner:
            x = F.leaky_relu(conv(x, edge_index))
        x = F.leaky_relu(self._gc_last(x, edge_index))
        x = x + residual
        x = self._norm(x, batch)
        x = x.view(-1, q, 3 * f).contiguous()  # [b, q, 3 * f]
        return x


class GnnMapping(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        max_gate_num: int,
        feat_dim: int,
        heads: int = 2,
    ) -> None:
        super().__init__()

        self._max_qubit_num = max_qubit_num
        self._max_gate_num = max_gate_num
        self._feat_dim = feat_dim

        # All gate nodes and virtual node feature embedding.
        self._x_trans = nn.Sequential(
            nn.Linear(in_features=max_qubit_num, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim, out_features=feat_dim),
        )

        self._circ_gnn = CircuitGnn(
            max_qubit_num=max_qubit_num,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
            heads=heads,
        )

        self._layout_gnn = LayoutGnn(
            max_qubit_num=max_qubit_num,
            feat_dim=feat_dim,
            heads=heads,
        )

        self._mlp_1 = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim * 3),
            nn.LeakyReLU(),
            nn.Linear(feat_dim * 3, feat_dim * 3),
            nn.LeakyReLU(),
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
        )

        self._mlp_2 = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, circ_pyg, topo_pyg):
        f = self._feat_dim
        q = self._max_qubit_num

        circ_x = self._x_trans(circ_pyg.x).view(-1, 2 * f)
        circ_feat = self._circ_gnn(
            circ_x, circ_pyg.edge_index, circ_pyg.batch
        )  # [b, 2 * f]

        topo_x = self._x_trans(topo_pyg.x).view(-1, f)  # [b * q, f]
        x = self._layout_gnn(
            circ_feat, topo_x, topo_pyg.edge_index, topo_pyg.batch
        )  # [b, q, 3 * f]

        x = self._mlp_1(x)  # [b, q, f]

        idx_pairs = torch.cartesian_prod(torch.arange(q), torch.arange(q))
        x = x[:, idx_pairs].contiguous()  # [b, q * q, 2, f]
        x = x.view(-1, q, q, 2 * f)
        x = x / math.sqrt(2 * f)
        x = self._mlp_2(x).view(-1, q, q)  # [b, q, q]
        x = x.view(-1, q * q)  # [b, q * q]
        return x