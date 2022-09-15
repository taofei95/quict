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


class GatFfn(nn.Module):
    """Multi-head attention + FFN. Transformer building block without normalization layer."""

    def __init__(self, feat_dim: int, heads: int) -> None:
        super().__init__()

        self._gat = gnn.GATConv(
            in_channels=feat_dim,
            out_channels=feat_dim,
            heads=heads,
        )
        self._mha_lin = nn.Linear(in_features=heads * feat_dim, out_features=feat_dim)

        self._ffn = Ffn(feat_dim=feat_dim)

    def forward(self, x, edge_index):
        residual = x
        x = F.leaky_relu(self._gat(x, edge_index))
        x = self._mha_lin(x) + residual

        x = self._ffn(x)
        return x


class GatFfnStack(nn.Module):
    def __init__(self, feat_dim: int, heads: int, num_hidden_layer: int) -> None:
        super().__init__()

        self._gat_layers = nn.ModuleList(
            [GatFfn(feat_dim=feat_dim, heads=heads) for _ in range(num_hidden_layer)]
        )

        self._ln = gnn.LayerNorm(in_channels=feat_dim)

    def forward(self, x, edge_index, batch):
        for layer in self._gat_layers:
            x = layer(x, edge_index)
        x = self._ln(x, batch)
        return x


class GnnMapping(nn.Module):
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        feat_dim: int,
        heads: int = 3,
    ) -> None:
        super().__init__()

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._feat_dim = feat_dim

        self._x_em = nn.Embedding(
            num_embeddings=max_qubit_num * (max_layer_num + 1) + 1,
            embedding_dim=feat_dim,
            padding_idx=0,
        )

        self._gc = nn.ModuleList(
            [
                GatFfnStack(feat_dim=feat_dim, heads=heads, num_hidden_layer=3),
                GatFfnStack(feat_dim=feat_dim, heads=heads, num_hidden_layer=3),
            ]
        )

        self._last_gc = gnn.GATConv(in_channels=feat_dim, out_channels=feat_dim)

        self._mlp = nn.Sequential(
            nn.Linear(in_features=2 * feat_dim, out_features=2 * feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * feat_dim, out_features=2 * feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * feat_dim, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=(feat_dim,)),
            nn.Linear(in_features=feat_dim, out_features=feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim // 2, out_features=1),
        )

    def forward(self, x, edge_index, batch=None):
        n = self._max_qubit_num
        l = self._max_layer_num
        f = self._feat_dim

        x = self._x_em(x)

        for block in self._gc:
            x = block(x, edge_index, batch)

        x = self._last_gc(x, edge_index) + x

        x = x.view(-1, n * (l + 1), f)
        x = x[:, :n, :].contiguous()
        # Pair wise concatenation
        idx_pairs = torch.cartesian_prod(
            torch.arange(x.shape[-2]), torch.arange(x.shape[-2])
        )
        x = x[:, idx_pairs]  # [b, n * n, 2, f]
        x = x.view(-1, n, n, 2 * f)
        x = self._mlp(x)  # [b, n, n, 1]
        x = x.view(-1, n, n)
        x = (x + x.transpose(-1, -2)) / 2
        x = x.view(-1, n * n)
        return x
