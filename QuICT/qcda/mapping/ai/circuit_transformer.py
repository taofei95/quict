"""
This file contains a modified version of <Do Transformers Really Perform Bad for Graph Representation?>
(https://arxiv.org/abs/2106.05234).
"""


from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class SelfAttn(nn.Module):
    """Self Attention network with batch dimension first."""

    def __init__(
        self,
        feat_dim: int,
    ) -> None:
        super().__init__()

        self._scale_factor = 1 / sqrt(feat_dim)

        self._lin_q = nn.Linear(in_features=feat_dim, out_features=feat_dim)
        self._lin_k = nn.Linear(in_features=feat_dim, out_features=feat_dim)
        self._lin_v = nn.Linear(in_features=feat_dim, out_features=feat_dim)

        self._softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_bias=None):
        batched = len(x.shape) == 3
        if not batched:
            x = torch.unsqueeze(x, dim=0)
            if attn_bias is not None:
                attn_bias = torch.unsqueeze(attn_bias, dim=0)

        q = self._lin_q(x)
        k = self._lin_k(x)
        v = self._lin_v(x)

        attn_weight = torch.bmm(q, k.transpose(1, 2)) * self._scale_factor
        if attn_bias is not None:
            attn_weight = attn_weight + attn_bias
        attn = self._softmax(attn_weight)

        output = torch.bmm(attn, v)
        if batched:
            return output
        else:
            return output[0]


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        head: int,
    ) -> None:
        super().__init__()
        self._attn_layers = nn.ModuleList(
            [SelfAttn(feat_dim=feat_dim) for _ in range(head)]
        )
        self._out_proj = nn.Linear(head * feat_dim, feat_dim)

    def forward(self, x, attn_bias=None):
        attn_list = []
        for layer in self._attn_layers:
            output = layer(x, attn_bias)
            attn_list.append(output)
        output = torch.cat(attn_list, dim=-1)
        output = self._out_proj(output)
        return output


class FeedForwardNet(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()

        self._lin1 = nn.Linear(in_features=feat_dim, out_features=feat_dim)
        self._lin2 = nn.Linear(in_features=feat_dim, out_features=feat_dim)

    def forward(self, x):
        x = self._lin1(x)
        x = F.relu(x)
        x = self._lin2(x)
        return x


class CircuitTransFormerLayer(nn.Module):
    def __init__(
        self,
        node_num: int,
        feat_dim: int,
        head: int,
    ) -> None:
        super().__init__()

        self._mha = MultiHeadAttn(feat_dim=feat_dim, head=head)
        self._ln1 = nn.LayerNorm((node_num, feat_dim))
        self._ffn = FeedForwardNet(feat_dim=feat_dim)
        self._ln2 = nn.LayerNorm((node_num, feat_dim))

    def forward(self, x, attn_bias=None):
        x = self._mha(self._ln1(x), attn_bias) + x
        x = self._ffn(self._ln2(x)) + x
        return x


class BiasedGraphormer(nn.Module):
    def __init__(
        self,
        node_num: int,
        feat_dim: int,
        head: int,
        num_attn_layer: int = 6,
    ) -> None:
        super().__init__()
        self._transformer_layers = nn.ModuleList(
            [
                CircuitTransFormerLayer(node_num=node_num, feat_dim=feat_dim, head=head)
                for _ in range(num_attn_layer)
            ]
        )

    def forward(self, x, attn_bias=None):
        if attn_bias is not None:
            assert len(x.shape) == len(attn_bias.shape)
        for layer in self._transformer_layers:
            x = layer(x, attn_bias)
        return x


class CircuitTransformer(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        max_topology_diameter: int,
        feat_dim: int,
        head: int,
        num_attn_layer: int = 6,
    ) -> None:
        super().__init__()

        max_volume = max_layer_num * max_qubit_num

        self._graphomer = BiasedGraphormer(
            node_num=max_volume + 1,
            feat_dim=feat_dim,
            head=head,
            num_attn_layer=num_attn_layer,
        )

        self._spacial_emedding = nn.Embedding(max_topology_diameter + 3, 1)

    def forward(
        self,
        x: torch.Tensor,
        spacial_encoding: torch.IntTensor,
    ):
        is_batch = len(x.shape) == 3
        se = self._spacial_emedding(spacial_encoding)
        se = torch.squeeze(se, dim=-1)
        attn_bias = se

        x = self._graphomer(x, attn_bias)

        # Node with label 0 is the virtual node, which is used as 
        # the readout node.
        if is_batch:
            return x[:, 0, :]
        else:
            return x[0, :]
