import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GnnMapping(nn.Module):
    def __init__(self, max_qubit_num: int, max_layer_num: int, feat_dim: int) -> None:
        super().__init__()

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._feat_dim = feat_dim

        self._x_em = nn.Embedding(
            num_embeddings=max_qubit_num * (max_layer_num + 1) + 1,
            embedding_dim=feat_dim,
            padding_idx=0,
        )

        self._gc_list = nn.ModuleList(
            [
                gnn.GCNConv(in_channels=feat_dim, out_channels=feat_dim),
                gnn.GCNConv(in_channels=feat_dim, out_channels=feat_dim),
                gnn.GCNConv(in_channels=feat_dim, out_channels=feat_dim),
                gnn.GCNConv(in_channels=feat_dim, out_channels=feat_dim),
            ]
        )
        self._last_gc = gnn.GCNConv(in_channels=feat_dim, out_channels=feat_dim)

        self._mlp = nn.Sequential(
            nn.Linear(in_features=2 * feat_dim, out_features=2 * feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * feat_dim, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim, out_features=feat_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim, out_features=feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_dim // 2, out_features=1),
        )

    def forward(self, x, edge_index):
        n = self._max_qubit_num
        l = self._max_layer_num
        f = self._feat_dim

        x = self._x_em(x)

        for conv in self._gc_list:
            x = F.leaky_relu(conv(x, edge_index)) + x
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
