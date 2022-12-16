# QAOA

```python
def seed(seed: int):
        """Set random seed.

        Args:
            seed (int): The random seed.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


def random_edges(n_nodes, n_edges):
        assert n_nodes >= 2
        assert n_edges <= n_nodes * (n_nodes - 1) / 2
        edges = []
        nodes = np.arange(n_nodes)
        while len(edges) < n_edges:
            edge = list(np.random.choice(nodes, 2, replace=False))
            if edge in edges or edge[::-1] in edges:
                continue
            edges.append(edge)
        return edges
```