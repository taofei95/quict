import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class Ffn(nn.Module):
    """Feed forward network without normalization layer."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()

        self._lin_1 = nn.Linear(in_features=feat_dim, out_features=feat_dim)
        self._lin_2 = nn.Linear(in_features=feat_dim, out_features=feat_dim)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self._lin_1(x))
        x = self._lin_2(x) + residual
        return x


class ConvStack(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        heads: int,
        num_hidden_layer: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        assert num_hidden_layer > 0

        self._first = gnn.GATv2Conv(
            in_channels=feat_dim, out_channels=feat_dim, heads=heads
        )

        self._inner = nn.ModuleList(
            [
                gnn.GATv2Conv(
                    in_channels=feat_dim * heads,
                    out_channels=feat_dim,
                    heads=heads,
                )
                for _ in range(num_hidden_layer)
            ]
        )

        self._last = gnn.GATv2Conv(
            in_channels=feat_dim * heads, out_channels=feat_dim, heads=1
        )

        # self._ffn = Ffn(feat_dim=feat_dim)

        self._normalize = normalize
        if normalize:
            self._ln = gnn.LayerNorm(in_channels=feat_dim)

    def forward(self, x, edge_index, batch):
        residual = x
        x = F.leaky_relu(self._first(x, edge_index))  # [b * n, f] -> [b * n, h, f]
        for conv in self._inner:
            x = F.leaky_relu(conv(x, edge_index))  # [b * n, h, f]
        x = (
            F.leaky_relu(self._last(x, edge_index)) + residual
        )  # [b * n, h, f] -> [b * n, f]
        # x = self._ffn(x)
        if self._normalize:
            x = self._ln(x, batch)
        return x


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

        # All gate nodes and virtual node feature embedding.
        self._x_em = nn.Embedding(
            num_embeddings=max_gate_num + 1,
            embedding_dim=feat_dim,
        )

        # One gate is targeting 2 qubits. So the feature dimension is actually doubled.
        self._gc = nn.ModuleList(
            [
                ConvStack(feat_dim=feat_dim * 2, heads=heads, num_hidden_layer=3),
            ]
        )

    def forward(self, x, edge_index, batch=None):
        n = self._max_gate_num + 1
        f = self._feat_dim

        x = self._x_em(x).view(-1, 2 * f)  # [b * n, 2 * f]
        for conv in self._gc:
            x = conv(x, edge_index, batch) + x  # [b * n, 2 * f]
        x = x.view(-1, n, 2 * f)  # [b, n, 2 * f]
        x = x[:, 0, :].contiguous().view(-1, 2 * f)  # [b, 2 * f]
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

        self._x_em = nn.Embedding(num_embeddings=max_qubit_num, embedding_dim=feat_dim)

        self._gc = nn.ModuleList(
            [
                ConvStack(feat_dim=feat_dim * 3, heads=heads, num_hidden_layer=3),
            ]
        )

    def forward(self, circ_feat, x, edge_index, batch=None):
        q = self._max_qubit_num
        f = self._feat_dim

        # Input circ_feat has shape [b, 2 * f]
        circ_feat = torch.repeat_interleave(
            circ_feat, q, dim=0
        ).contiguous()  # [b * q, 2 * f]
        x = self._x_em(x).view(-1, f)  # [b * q, f]
        x = torch.cat((x, circ_feat), dim=1)  # [b * q, 3 * f]

        for conv in self._gc:
            x = conv(x, edge_index, batch) + x
        x = x.view(-1, q, f)  # [b, q, 3 * f]
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

        self._mlp = nn.Sequential(
            nn.Linear(feat_dim * 6, feat_dim * 6),
            nn.LeakyReLU(),
            nn.Linear(feat_dim * 6, feat_dim * 2),
            nn.LeakyReLU(),
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, circ_pyg, topo_pyg):
        f = self._feat_dim
        q = self._max_qubit_num

        circ_feat = self._circ_gnn(
            circ_pyg.x, circ_pyg.edge_index, circ_pyg.batch
        )  # [b, 2 * f]
        x = self._layout_gnn(
            circ_feat, topo_pyg.x, topo_pyg.edge_index, topo_pyg.batch
        )  # [b, q, 3 * f]

        idx_pairs = torch.cartesian_prod(torch.arange(q), torch.arange(q))
        x = x[:, idx_pairs].contiguous()  # [b, q * q, 2, 3 * f]
        x = x.view(-1, q, q, 6 * f)
        x = self._mlp(x).view(-1, q, q)  # [b, q, q]
        x = (x + x.transpose(-1, -2)) / 2
        # gather q * q dim for convenient max
        x = x.view(-1, q * q)  # [b, q * q]
        return x
