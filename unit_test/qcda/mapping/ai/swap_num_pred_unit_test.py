import pytest
from QuICT.qcda.mapping.ai import *
import torch


def test_swap_pred_mix():
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    model = SwapPredMixBase(
        topo_gc_hidden_channel=[50, 50,],
        lc_gc_hidden_channel=[50, 50,],
        topo_lc_gc_out_channel=150,
        topo_lc_gc_pool_node=300,
        ml_hidden_channel=[100, 100,],
        ml_out_channel=1,
    )

    topo_data_list = []
    lc_data_list = []
    node_feature_num = 300

    topo_x = torch.zeros(128, node_feature_num)
    topo_x[:15, :15] = torch.eye(15, dtype=float)
    topo_edge_index = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]]
    lc_x = torch.zeros(128, node_feature_num)
    lc_x[:60, :60] = torch.eye(60, dtype=float)
    lc_edge_index = [[0, 1], [1, 2]]

    topo_edge_index = torch.tensor(topo_edge_index, dtype=torch.long).t().contiguous()
    lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.long).t().contiguous()
    topo_data = Data(x=topo_x, edge_index=topo_edge_index)
    lc_data = Data(x=lc_x, edge_index=lc_edge_index)

    topo_data_list.append(topo_data)
    lc_data_list.append(lc_data)
    topo_data_list = Batch.from_data_list(topo_data_list)
    lc_data_list = Batch.from_data_list(lc_data_list)

    data = {}
    data["topo"] = topo_data_list
    data["lc"] = lc_data_list

    with torch.no_grad():
        out = model(data)
        print(out)


def test_multi_layer_cnn():
    h = 128
    w = 128
    model = CnnRecognition(h=h, w=w)
    x = torch.randn(2, h, w)
    y = model(x)
    shape = torch.Size([256, 8, 8])
    assert shape == y.shape


if __name__ == "__main__":
    pytest.main(["./swap_num_pred_unit_test.py"])
    # pytest.main(["./swap_num_pred_unit_test.py", "-k", "cnn"])
    # pytest.main(["./swap_num_pred_unit_test.py", "-k", "mix"])
