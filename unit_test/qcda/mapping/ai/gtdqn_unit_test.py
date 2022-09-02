from QuICT.qcda.mapping.ai.gtdqn import GraphTransformerDeepQNetwork
import torch
from random import randint, sample


def test_gtdqn():
    max_qubit_num = 10
    max_layer_num = 10
    inner_feat_dim = 30
    head = 3
    node_num = max_qubit_num * (max_layer_num + 1)

    model = GraphTransformerDeepQNetwork(
        max_qubit_num=max_qubit_num,
        max_layer_num=max_layer_num,
        inner_feat_dim=inner_feat_dim,
        head=head,
    )

    x_no_batch = [randint(0, max_qubit_num) for _ in range(node_num)]
    x_no_batch = torch.tensor(x_no_batch, dtype=torch.int)
    spacial_encoding = [
        [randint(0, max_qubit_num) for _ in range(node_num)] for _ in range(node_num)
    ]
    spacial_encoding = torch.tensor(spacial_encoding, dtype=torch.int)
    y_no_batch = model(x_no_batch, spacial_encoding)

    assert y_no_batch.shape == torch.Size((max_qubit_num * max_qubit_num,))

    batch_size = 3

    x_batch = torch.stack([x_no_batch for _ in range(batch_size)])
    spacial_encoding_batch = torch.stack([spacial_encoding for _ in range(batch_size)])
    y_batch = model(x_batch, spacial_encoding_batch)

    assert y_batch.shape == torch.Size((batch_size, max_qubit_num * max_qubit_num))


def test_gather_and_max():
    max_qubit_num = 10
    max_layer_num = 10
    inner_feat_dim = 30
    head = 3
    node_num = max_qubit_num * (max_layer_num + 1)

    model = GraphTransformerDeepQNetwork(
        max_qubit_num=max_qubit_num,
        max_layer_num=max_layer_num,
        inner_feat_dim=inner_feat_dim,
        head=head,
    )

    batch_size = 3
    x_no_batch = [randint(0, max_qubit_num) for _ in range(node_num)]
    x_no_batch = torch.tensor(x_no_batch, dtype=torch.int)
    spacial_encoding = [
        [randint(0, max_qubit_num) for _ in range(node_num)] for _ in range(node_num)
    ]
    spacial_encoding = torch.tensor(spacial_encoding, dtype=torch.int)

    x_batch = torch.stack([x_no_batch for _ in range(batch_size)])
    spacial_encoding_batch = torch.stack([spacial_encoding for _ in range(batch_size)])
    y_batch = model(x_batch, spacial_encoding_batch)

    y_max = y_batch.max(1)[0]

    assert y_max.shape == torch.Size((batch_size,))

    actions = [sample(range(max_qubit_num), 2) for _ in range(batch_size)]
    actions = torch.tensor(
        [[u * max_qubit_num + v] for u, v in actions],
        dtype=torch.int64,
    )  # [B, 1]

    y_gather = y_batch.gather(1, actions).squeeze()

    assert y_gather.shape == torch.Size((batch_size,))
