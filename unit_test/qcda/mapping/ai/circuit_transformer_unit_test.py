import torch
from QuICT.core import *
from QuICT.qcda.mapping.ai.circuit_transformer import *
from QuICT.qcda.mapping.ai.data_processor_vnode import CircuitVnodeProcessor
import networkx as nx


def test_self_attn():
    for feat_dim in [3, 4, 5, 6, 7]:
        model = SelfAttn(feat_dim)
        for node_num in [1, 3, 20]:
            x_no_batch = torch.randn(node_num, feat_dim)
            attn_bias = torch.randn(node_num, node_num)
            with torch.no_grad():
                y_no_batch = model(x_no_batch)
                assert y_no_batch.shape == x_no_batch.shape
                y_no_batch = model(x_no_batch, attn_bias)
                assert y_no_batch.shape == x_no_batch.shape

            for batch_size in [1, 3, 5]:
                x_batch = torch.randn(batch_size, node_num, feat_dim)
                with torch.no_grad():
                    y_batch = model(x_batch)
                    assert y_batch.shape == x_batch.shape
                    y_batch = model(x_batch, attn_bias)
                    assert y_batch.shape == x_batch.shape


def test_multi_head_attn():
    for head in [1, 3, 5]:
        for feat_dim in [3, 4, 5, 6, 7]:
            model = MultiHeadAttn(feat_dim, head)
            for node_num in [1, 3, 20]:
                x_no_batch = torch.randn(node_num, feat_dim)
                attn_bias = torch.randn(node_num, node_num)
                with torch.no_grad():
                    y_no_batch = model(x_no_batch)
                    assert y_no_batch.shape == x_no_batch.shape
                    y_no_batch = model(x_no_batch, attn_bias)
                    assert y_no_batch.shape == x_no_batch.shape

                for batch_size in [1, 3, 5]:
                    x_batch = torch.randn(batch_size, node_num, feat_dim)
                    attn_bias_batch = torch.stack(
                        [attn_bias for _ in range(batch_size)]
                    )
                    with torch.no_grad():
                        y_batch = model(x_batch)
                        assert y_batch.shape == x_batch.shape
                        y_batch = model(x_batch, attn_bias_batch)
                        assert y_batch.shape == x_batch.shape


def test_circuit_transformer_layer():
    for head in [1, 3, 5]:
        for feat_dim in [3, 4, 5, 6, 7]:
            for node_num in [1, 3, 20]:
                model = CircuitTransFormerLayer(node_num, feat_dim, head)
                x_no_batch = torch.randn(node_num, feat_dim)
                attn_bias = torch.randn(node_num, node_num)
                with torch.no_grad():
                    y_no_batch = model(x_no_batch)
                    assert y_no_batch.shape == x_no_batch.shape
                    y_no_batch = model(x_no_batch, attn_bias)
                    assert y_no_batch.shape == x_no_batch.shape

                for batch_size in [1, 3, 5]:
                    x_batch = torch.randn(batch_size, node_num, feat_dim)
                    attn_bias_batch = torch.stack(
                        [attn_bias for _ in range(batch_size)]
                    )
                    with torch.no_grad():
                        y_batch = model(x_batch)
                        assert y_batch.shape == x_batch.shape
                        y_batch = model(x_batch, attn_bias_batch)
                        assert y_batch.shape == x_batch.shape


def test_biased_graphormer():
    for head in [1, 3, 5]:
        for feat_dim in [3, 4, 5, 6, 7]:
            for qubit_num in [2, 3, 5, 10]:
                for layer_num in [1, 2, 3]:
                    node_num = layer_num * qubit_num + 1
                    model = BiasedGraphormer(
                        node_num=node_num,
                        feat_dim=feat_dim,
                        head=head,
                        num_attn_layer=6,
                    )
                    x_no_batch = torch.randn(node_num, feat_dim)
                    attn_bias = torch.randn(node_num, node_num)
                    with torch.no_grad():
                        y_no_batch = model(x_no_batch)
                        assert y_no_batch.shape == x_no_batch.shape
                        y_no_batch = model(x_no_batch, attn_bias)
                        assert y_no_batch.shape == x_no_batch.shape

                    for batch_size in [1, 3, 5]:
                        x_batch = torch.randn(batch_size, node_num, feat_dim)
                        attn_bias_batch = torch.stack(
                            [attn_bias for _ in range(batch_size)]
                        )
                        with torch.no_grad():
                            y_batch = model(x_batch)
                            assert y_batch.shape == x_batch.shape
                            y_batch = model(x_batch, attn_bias_batch)
                            assert y_batch.shape == x_batch.shape


def test_circuit_transformer():
    circ = Circuit(10)
    circ.random_append(200)
    max_qubit_num = 50
    max_layer_num = 10
    feat_dim = 30
    processor = CircuitVnodeProcessor(max_qubit_num=max_qubit_num)

    circ_graph = processor._build_circ_repr(circ=circ, max_layer_num=max_layer_num)
    spacial_encoding = processor.get_spacial_encoding(
        graph=circ_graph, max_topology_diameter=max_qubit_num
    )
    model = CircuitTransformer(
        max_qubit_num=max_qubit_num,
        max_topology_diameter=max_qubit_num,
        feat_dim=feat_dim,
        head=6,
        max_layer_num=10,
    )
    x = torch.randn(max_qubit_num * max_layer_num + 1, feat_dim)
    y_no_batch = model(x, spacial_encoding)
    assert y_no_batch.shape == torch.Size((feat_dim,))

    batch_size = 3
    x = torch.randn(batch_size, max_qubit_num * max_layer_num + 1, feat_dim)
    spacial_encoding = torch.stack([spacial_encoding for _ in range(batch_size)])
    y_batch = model(x, spacial_encoding)
    assert y_batch.shape == torch.Size((batch_size, feat_dim))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
