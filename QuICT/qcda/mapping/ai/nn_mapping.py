import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class CircuitGnn(nn.Module):
    def __init__(
        self,
        qubit_num: int,
        max_gate_num: int,
        feat_dim: int,
    ) -> None:
        super().__init__()

        self._max_qubit_num = qubit_num
        self._max_gate_num = max_gate_num
        self._feat_dim = feat_dim

        self._gc_grp_1 = nn.ModuleList(
            [
                gnn.SAGEConv(
                    in_channels=feat_dim,
                    out_channels=feat_dim,
                )
                for _ in range(3)
            ]
        )
        # self._norm_1 = gnn.GraphNorm(in_channels=feat_dim)
        self._gc_grp_2 = nn.ModuleList(
            [
                gnn.SAGEConv(
                    in_channels=feat_dim,
                    out_channels=feat_dim,
                )
                for _ in range(3)
            ]
        )
        self._aggr = gnn.aggr.SoftmaxAggregation(learn=True)

    def forward(self, x, edge_index, batch=None):
        n = self._max_gate_num
        f = self._feat_dim

        for conv in self._gc_grp_1:
            x = F.relu(conv(x, edge_index)) + x
        # x = self._norm_1(x,batch)
        for conv in self._gc_grp_2:
            x = F.relu(conv(x, edge_index)) + x
        x = self._aggr(x, batch)  # [b, f]
        x = x.view(-1, f)  # [b, f]
        return x


class GnnMapping(nn.Module):
    def __init__(
        self,
        qubit_num: int,
        max_gate_num: int,
        feat_dim: int,
        action_num: int,
    ) -> None:
        super().__init__()

        self._max_qubit_num = qubit_num
        self._max_gate_num = max_gate_num
        self._feat_dim = feat_dim
        self._action_num = action_num

        # All gate nodes and virtual node feature embedding.
        self._x_trans = nn.Embedding(
            num_embeddings=qubit_num + 1,
            embedding_dim=feat_dim,
            padding_idx=0,
        )
        nn.init.orthogonal_(self._x_trans.weight)

        self._circ_gnn = CircuitGnn(
            qubit_num=qubit_num,
            max_gate_num=max_gate_num,
            feat_dim=feat_dim,
        )

        f_start = feat_dim

        self._mlp_1 = nn.Sequential(
            nn.Linear(f_start, f_start // 2),
            nn.ReLU(),
            nn.Linear(f_start // 2, self._action_num),
        )

    def forward(self, data):
        f = self._feat_dim
        a = self._action_num

        circ_x = self._x_trans(data.x).view(-1, 2, f)
        circ_x = torch.sum(circ_x, -2) / 2  # [b * n, f]
        circ_feat = self._circ_gnn(circ_x, data.edge_index, data.batch)  # [b, f]

        x = self._mlp_1(circ_feat).view(-1, a)  # [b, a]
        return x


NnMapping = GnnMapping
