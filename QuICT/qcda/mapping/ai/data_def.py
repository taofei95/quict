from torch_geometric.data import Data


class PairData(Data):
    def __init__(self, edge_index_topo=None, x_topo=None, edge_index_lc=None, x_lc=None):
        super().__init__()
        self.edge_index_topo = edge_index_topo
        self.x_topo = x_topo
        self.edge_index_lc = edge_index_lc
        self.x_lc = x_lc

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_topo':
            return self.x_topo.size(0)
        if key == 'edge_index_lc':
            return self.x_lc.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)