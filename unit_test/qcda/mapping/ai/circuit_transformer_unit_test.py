import torch
from QuICT.core import *
from QuICT.qcda.mapping.ai.circuit_transformer import *
from QuICT.qcda.mapping.ai.transformer_data_factory import CircuitTransformerDataFactory


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


def test_circuit_graphformer_layer():
    for head in [1, 3, 5]:
        for feat_dim in [3, 4, 5, 6, 7]:
            for node_num in [1, 3, 20]:
                model = CircuitGraphormerLayer(node_num, feat_dim, head)
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


def test_circuit_graphormer():
    circ = Circuit(10)
    topo = Layout(10)
    max_qubit_num = 30
    max_layer_num = 20
    feat_dim = 30
    factory = CircuitTransformerDataFactory(
        max_qubit_num=max_qubit_num, max_layer_num=max_layer_num
    )
    # Use a line topo as example
    for i in range(9):
        topo.add_edge(i, (i + 1))
    topo_graph = factory.get_topo_graph(topo=topo)
    topo_dist = torch.zeros((max_qubit_num, max_qubit_num), dtype=torch.int)
    sp = nx.all_pairs_dijkstra_path_length(topo_graph)
    for u, row in sp:
        for v, d in row.items():
            topo_dist[u][v] = d
            topo_dist[v][u] = d

    model = CircuitTransformer(
        max_qubit_num=max_qubit_num,
        feat_dim=feat_dim,
        head=6,
        max_layer_num=max_layer_num,
    )

    successful = False
    max_retry = 3
    retry = 0
    while not successful and retry < max_retry:
        circ.random_append(30)

        layered_circ, successful = factory.get_layered_circ(circ=circ)
        if not successful:
            retry += 1
            continue
        circ_graph = factory.get_circ_graph(
            layered_circ=layered_circ, topo_dist=topo_dist
        )
        spacial_encoding = factory.get_spacial_encoding(circ_graph=circ_graph)

        x = factory.get_x(10)
        y_no_batch = model(x, spacial_encoding)
        assert y_no_batch.shape == torch.Size(
            (
                max_qubit_num,
                feat_dim,
            )
        )

        batch_size = 3
        x = torch.stack([x, x, x])
        spacial_encoding = torch.stack([spacial_encoding for _ in range(batch_size)])
        y_batch = model(x, spacial_encoding)
        assert y_batch.shape == torch.Size((batch_size,max_qubit_num, feat_dim))
    assert retry < max_retry
