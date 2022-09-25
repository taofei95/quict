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

        self._gc_inner = nn.ModuleList(
            [
                gnn.SAGEConv(
                    in_channels=feat_dim,
                    out_channels=feat_dim,
                )
                for _ in range(8)
            ]
        )

        self._aggr = gnn.aggr.SoftmaxAggregation(learn=True)

    def forward(self, x, edge_index, batch=None):
        n = self._max_gate_num
        f = self._feat_dim

        for conv in self._gc_inner:
            x = F.relu(conv(x, edge_index))
        x = self._aggr(x, batch)  # [b, f]
        x = x.view(-1, f)  # [b, f]
        return x


class CircuitRnn(nn.Module):
    def __init__(
        self,
        qubit_num: int,
        feat_dim: int,
    ) -> None:
        super().__init__()

        self._max_qubit_num = qubit_num
        self._feat_dim = feat_dim

        self._rnn = nn.RNN(
            input_size=qubit_num**2,
            hidden_size=feat_dim,
            batch_first=True,
        )

    def forward(self, seq):
        f = self._feat_dim

        _, hn = self._rnn(seq)  # hn has shape [1, b, f]
        hn = hn.view(-1, f)
        return hn


class NnMapping(nn.Module):
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

        # # All gate nodes and virtual node feature embedding.
        # self._x_trans = nn.Embedding(
        #     num_embeddings=qubit_num + 1,
        #     embedding_dim=feat_dim,
        #     padding_idx=0,
        # )
        # nn.init.orthogonal_(self._x_trans.weight)

        # self._circ_gnn = CircuitGnn(
        #     qubit_num=qubit_num,
        #     max_gate_num=max_gate_num,
        #     feat_dim=feat_dim,
        # )

        self._circ_rnn = CircuitRnn(qubit_num, feat_dim)

        f_start = feat_dim
        step = (f_start - action_num) // 3

        self._mlp_1 = nn.Sequential(
            nn.Linear(f_start, f_start),
            nn.ReLU(),
            nn.Linear(f_start, f_start - 1 * step),
            nn.ReLU(),
            nn.Linear(f_start - 1 * step, f_start - 2 * step),
            nn.ReLU(),
            nn.Linear(f_start - 2 * step, self._action_num),
        )

    def forward(self, data):
        f = self._feat_dim
        a = self._action_num

        # circ_x = self._x_trans(circ_pyg.x).view(-1, 2, f)
        # circ_x = torch.sum(circ_x, -2) / 2  # [b * n, f]
        # circ_feat = self._circ_gnn(
        #     circ_x, circ_pyg.edge_index, circ_pyg.batch
        # )  # [b, f]

        circ_feat = self._circ_rnn(data)
        x = self._mlp_1(circ_feat).view(-1, a)  # [b, a]
        return x
