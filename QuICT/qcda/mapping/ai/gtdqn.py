from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from QuICT.qcda.mapping.ai.circuit_graphormer import CircuitGraphormer


class GraphTransformerDeepQNetwork(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        inner_feat_dim: int,
        head: int,
        num_attn_layer: int = 6,
    ) -> None:
        super().__init__()

        self._circ_graph_transformer = CircuitGraphormer(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            feat_dim=inner_feat_dim,
            head=head,
            num_attn_layer=num_attn_layer,
        )

        lv = 3
        step = (max_qubit_num - inner_feat_dim) // lv
        self._mlp = nn.Sequential(
            nn.Linear(in_features=inner_feat_dim, out_features=inner_feat_dim + step),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=inner_feat_dim + step,
                out_features=inner_feat_dim + 2 * step,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=inner_feat_dim + 2 * step, out_features=max_qubit_num
            ),
        )

        self._scale = sqrt(1.0 / float(max_qubit_num))

    def forward(self, x, spacial_encoding):
        # If x is (n,) shaped, unsqueeze it as a single entry batch
        is_batch = len(x.shape) == 2
        if not is_batch:
            x = torch.unsqueeze(x, dim=0)
        x = self._circ_graph_transformer(x, spacial_encoding)
        x = self._mlp(x)

        x = x * self._scale
        x = torch.unsqueeze(x, dim=-1)
        x = torch.bmm(x, torch.transpose(x, -1, -2))
        if is_batch:
            return x
        else:
            return x[0]
