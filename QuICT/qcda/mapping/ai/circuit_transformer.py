from math import sqrt
import torch
import torch.nn as nn


class SelfAttn(nn.Module):
    """Self Attention network with batch dimension first."""

    def __init__(
        self,
        feat_dim: int,
    ) -> None:
        super().__init__()

        d = feat_dim

        self._scale_factor = 1 / sqrt(d)

        self._lin_q = nn.Linear(d, d)
        self._lin_k = nn.Linear(d, d)
        self._lin_v = nn.Linear(d, d)

        self._softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_bias=None):
        batched = len(x.shape) == 3
        if not batched:
            x = torch.unsqueeze(x, dim=0)

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
        head: int,
        feat_dim: int,
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
